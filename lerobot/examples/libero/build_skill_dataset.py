"""
build_skill_dataset.py — libero_90 전체 데모를 VF cosine divergence로 스킬 분할 후 .npz 저장.

Pipeline:
  1. 모든 task의 모든 episode 순회
  2. VF analysis (skill_divider.py 로직 재사용)
  3. SG smooth + peak detection → skill boundaries
  4. actions / states를 boundary 단위로 잘라 .npz 저장
  5. 이미 처리된 episode는 skip (resume)

Output layout:
  output_dir/skills/ep{ep_id:05d}_skill{si:02d}.npz
    └── actions, states, episode_id, skill_index, frame_start, frame_end

Usage:
  python examples/libero/build_skill_dataset.py \
    --dataset_dir .../libero_90 \
    --policy_path .../checkpoints/080000/pretrained_model \
    --output_dir .../outputs/skill_dataset
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tyro

sys.path.insert(0, str(Path(__file__).parent))
from skill_divider import (
    _find_peaks_above_mean,
    _savgol_smooth,
    get_episode_timestamps,
    get_video_path,
    load_data,
    load_episodes_meta,
    load_policy,
    run_vf_analysis,
)
from SBD_visualize import SkillVisualizer


# ── Args ──────────────────────────────────────────────────────────────────────

@dataclass
class Args:
    dataset_dir: str = "/data2/dohyeon/SBD/libero_dataset/libero_90"
    output_dir: str = "/data2/dohyeon/SBD/outputs/skill_dataset"
    policy_path: str = ""
    device: str = "cuda"
    task_ids: list[int] | None = None
    """처리할 task index 목록. None이면 전체 task 처리."""
    # ── Diffusion scheduler ──────────────────────────────────────────────────
    noise_scheduler_type: str = "DDIM"
    num_inference_steps: int = 10
    eval_at_step: int = 7
    # ── VF analysis ──────────────────────────────────────────────────────────
    replan_interval: int = 3
    n_gmm_components: int = 5
    # ── Peak detection ────────────────────────────────────────────────────────
    smooth_window: int = 7
    savgol_polyorder: int = 4
    peak_nms: bool = True
    nms_dist: int | None = None
    """NMS 거리. None이면 replan_interval * 2 사용."""
    # ── Dataset filtering ─────────────────────────────────────────────────────
    min_skill_len: int = 2
    """스킬 세그먼트 최소 프레임 수 (미만이면 해당 세그먼트 제외)."""
    min_skills: int = 2
    """유효 스킬 수가 이 값 미만이면 episode 전체 skip."""
    # ── Misc ─────────────────────────────────────────────────────────────────
    seed: int | None = 42
    resume: bool = True
    """True면 이미 저장된 episode skip."""
    # ── WandB ────────────────────────────────────────────────────────────────
    wandb_project: str | None = None
    wandb_run_name: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_boundaries(replan_ts: list, div_cos: np.ndarray,
                        n_frames: int, args: Args) -> list[int]:
    nms_dist = (args.nms_dist if args.nms_dist is not None
                else args.replan_interval * 2) if args.peak_nms else 0
    sg_vals = _savgol_smooth(list(div_cos), args.smooth_window, polyorder=args.savgol_polyorder)
    peak_ts, _ = _find_peaks_above_mean(sg_vals, replan_ts,
                                         min_distance=nms_dist, margin=nms_dist)
    return sorted(set([0] + [int(p) for p in peak_ts] + [n_frames]))


def _save_skills(skills_dir: Path, ep_id: int, task_id: int,
                  gt_actions: np.ndarray, states: np.ndarray,
                  boundaries: list[int], min_skill_len: int, min_skills: int) -> list[str]:
    valid_segs = [
        (si, s, e)
        for si, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:]))
        if e - s >= min_skill_len
    ]
    if len(valid_segs) < min_skills:
        return []

    task_dir = skills_dir / f"task{task_id:02d}"
    task_dir.mkdir(exist_ok=True)

    saved = []
    for si, s, e in valid_segs:
        fname = task_dir / f"ep{ep_id:05d}_task{task_id:02d}_skill{si:02d}.npz"
        np.savez(
            str(fname),
            actions=gt_actions[s:e].astype(np.float32),
            states=states[s:e].astype(np.float32),
            episode_id=np.array(ep_id),
            task_id=np.array(task_id),
            skill_index=np.array(si),
            frame_start=np.array(s),
            frame_end=np.array(e),
        )
        saved.append(str(fname))
    return saved


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: Args) -> None:
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    skills_dir = output_dir / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        import random, torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    print("Loading metadata...")
    episodes_meta = load_episodes_meta(dataset_dir)
    tasks_meta = pd.read_parquet(dataset_dir / "meta" / "tasks.parquet").reset_index()

    video_cols = [c for c in episodes_meta.columns
                  if c.startswith("videos/") and c.endswith("/chunk_index")]
    camera_keys = [c.split("/")[1] for c in video_cols]
    print(f"Cameras: {camera_keys}")

    task_ids = args.task_ids if args.task_ids is not None else sorted(tasks_meta["task_index"].tolist())
    print(f"Tasks to process: {len(task_ids)}")

    # Pre-count total episodes for ETA
    all_episode_ids = []
    for task_id in task_ids:
        task_row = tasks_meta[tasks_meta["task_index"] == task_id]
        if task_row.empty:
            continue
        target_lang = task_row.iloc[0]["task"]
        ep_of_task = episodes_meta[episodes_meta["tasks"].apply(
            lambda t: target_lang in (
                [str(x) for x in t] if isinstance(t, (list, np.ndarray)) else [str(t)]
            )
        )]
        all_episode_ids.extend(ep_of_task["episode_index"].tolist())
    n_total_eps_global = len(all_episode_ids)
    print(f"Total episodes: {n_total_eps_global}")

    print(f"Loading policy from {args.policy_path} ...")
    t0 = time.time()
    policy, preprocessor = load_policy(
        args.policy_path, args.device, args.noise_scheduler_type, args.num_inference_steps
    )
    print(f"  [time] policy load: {time.time()-t0:.1f}s")

    viz = SkillVisualizer(output_dir)

    # ── WandB init ────────────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb_project:
        import wandb
        run_name = args.wandb_run_name or f"skill_dataset_{len(task_ids)}tasks"
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "dataset_dir": args.dataset_dir,
                "n_tasks": len(task_ids),
                "n_total_episodes": n_total_eps_global,
                "replan_interval": args.replan_interval,
                "eval_at_step": args.eval_at_step,
                "n_gmm_components": args.n_gmm_components,
                "smooth_window": args.smooth_window,
                "savgol_polyorder": args.savgol_polyorder,
                "nms_dist": args.nms_dist,
                "min_skill_len": args.min_skill_len,
                "min_skills": args.min_skills,
            },
        )

    n_total_eps = 0
    n_processed = 0  # resume 포함 전체 처리 완료 수
    n_saved = 0
    n_skipped_resume = 0
    n_skipped_filter = 0
    n_error = 0
    total_skills = 0
    t_start = time.time()

    for task_id in task_ids:
        task_row = tasks_meta[tasks_meta["task_index"] == task_id]
        if task_row.empty:
            print(f"  [warn] task_id {task_id} not found, skipping.")
            continue
        target_lang = task_row.iloc[0]["task"]
        ep_of_task = episodes_meta[episodes_meta["tasks"].apply(
            lambda t: target_lang in (
                [str(x) for x in t] if isinstance(t, (list, np.ndarray)) else [str(t)]
            )
        )]
        episode_ids = ep_of_task["episode_index"].tolist()
        n_total_eps += len(episode_ids)
        print(f"\nTask {task_id}: '{target_lang}' — {len(episode_ids)} episodes")

        ep_data = load_data(dataset_dir, episode_ids=episode_ids)

        for ep_id in episode_ids:
            # Resume
            task_dir = skills_dir / f"task{task_id:02d}"
            if args.resume and any(task_dir.glob(f"ep{ep_id:05d}_task{task_id:02d}_skill*.npz")):
                existing = list(task_dir.glob(f"ep{ep_id:05d}_task{task_id:02d}_skill*.npz"))
                n_skipped_resume += 1
                n_processed += 1
                total_skills += len(existing)
                print(f"  [skip] ep{ep_id:05d} already done ({len(existing)} skills)")
                continue

            if episodes_meta[episodes_meta["episode_index"] == ep_id].empty:
                continue
            ep_df = ep_data[ep_data["episode_index"] == ep_id].reset_index(drop=True)
            if len(ep_df) == 0:
                continue

            print(f"  ep{ep_id:05d} ...", end="", flush=True)
            t_ep = time.time()

            def _load_cam(cam_key):
                src = get_video_path(dataset_dir, ep_id, cam_key, episodes_meta).resolve()
                start_sec, end_sec = get_episode_timestamps(dataset_dir, ep_id, episodes_meta, cam_key)
                return cam_key, viz.load_episode_frames(src, start_sec, end_sec)

            with ThreadPoolExecutor(max_workers=len(camera_keys)) as pool:
                cam_frames = dict(pool.map(_load_cam, camera_keys))

            try:
                vf_replan_ts, _, _, div_cos, _, _ = run_vf_analysis(
                    policy, preprocessor, ep_df, cam_frames, camera_keys,
                    args.eval_at_step, args.replan_interval,
                    n_gmm_components=args.n_gmm_components,
                )
            except Exception as e:
                import traceback
                print(f" [ERROR] {e}")
                traceback.print_exc()
                n_error += 1
                n_processed += 1
                continue

            n_frames = len(ep_df)
            boundaries = _detect_boundaries(vf_replan_ts, div_cos, n_frames, args)

            gt_actions = np.stack(ep_df["action"].values[:n_frames])
            states_arr = np.stack(ep_df["observation.state"].values[:n_frames])

            saved = _save_skills(skills_dir, ep_id, task_id, gt_actions, states_arr,
                                  boundaries, args.min_skill_len, args.min_skills)

            elapsed = time.time() - t_ep
            n_processed += 1
            if saved:
                n_saved += 1
                total_skills += len(saved)
                print(f" {len(saved)} skills  [{elapsed:.1f}s]")
            else:
                n_skipped_filter += 1
                print(f" skipped (≤{args.min_skills - 1} valid skills)  [{elapsed:.1f}s]")

            if wandb_run is not None:
                import wandb
                elapsed_total = time.time() - t_start
                eps_per_sec = n_processed / elapsed_total if elapsed_total > 0 else 0
                remaining = n_total_eps_global - n_processed
                eta_sec = remaining / eps_per_sec if eps_per_sec > 0 else 0
                wandb_run.log({
                    "progress/episodes_done": n_processed,
                    "progress/episodes_total": n_total_eps_global,
                    "progress/episodes_pct": n_processed / n_total_eps_global * 100,
                    "progress/episodes_saved": n_saved,
                    "progress/episodes_skipped_filter": n_skipped_filter,
                    "progress/episodes_skipped_resume": n_skipped_resume,
                    "progress/episodes_error": n_error,
                    "progress/skills_total": total_skills,
                    "progress/eta_min": eta_sec / 60,
                    "progress/current_task_id": task_id,
                    "progress/ep_time_sec": elapsed,
                })

    print(f"\n{'=' * 60}")
    print(f"Done.")
    print(f"  total episodes  : {n_total_eps_global}")
    print(f"  saved           : {n_saved}")
    print(f"  skipped/resume  : {n_skipped_resume}")
    print(f"  skipped/filter  : {n_skipped_filter}")
    print(f"  errors          : {n_error}")
    print(f"  total skills    : {total_skills}")
    print(f"  skills dir      : {skills_dir}")

    if wandb_run is not None:
        wandb_run.summary.update({
            "final/episodes_saved": n_saved,
            "final/episodes_skipped_filter": n_skipped_filter,
            "final/episodes_skipped_resume": n_skipped_resume,
            "final/episodes_error": n_error,
            "final/skills_total": total_skills,
            "final/total_time_min": (time.time() - t_start) / 60,
        })
        wandb_run.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
