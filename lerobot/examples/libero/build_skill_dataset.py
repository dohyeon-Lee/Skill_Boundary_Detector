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

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import tyro

sys.path.insert(0, str(Path(__file__).parent))
from skill_divider import (
    _savgol_smooth,
    load_data,
    load_episodes_meta,
    load_policy,
    run_vf_analysis,
)


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
    boundary_threshold_mode: Literal["episode_mean", "global_mean"] = "episode_mean"
    """episode_mean(legacy) 또는 전체 episode의 단일 global_mean threshold."""
    global_threshold_path: str = ""
    """global_mean에서 사용하는 global_boundary_threshold.json 경로."""
    use_cached_curves: bool = False
    """True면 VF를 다시 계산하지 않고 curves/ep*.npz의 SG curve로 skill을 자른다 (global 2-pass용)."""
    # ── Dataset filtering ─────────────────────────────────────────────────────
    min_skill_len: int = 2
    """스킬 세그먼트 최소 프레임 수 (미만이면 해당 세그먼트 제외)."""
    min_skills: int = 2
    """유효 스킬 수가 이 값 미만이면 episode 전체 skip."""
    # ── Misc ─────────────────────────────────────────────────────────────────
    seed: int | None = 42
    resume: bool = True
    """True면 이미 저장된 episode skip."""
    write_done_markers: bool = False
    """True면 episode 처리 완료 시 .done 마커 기록 (필터/0-skill 포함). 재시도 검증용 (verify_skillset.py)."""
    # ── Boundary curves (eval HTML 오버레이용) ────────────────────────────────
    dump_curves: bool = True
    """True면 episode별 multimodality(VF cos-divergence) 곡선을 output_dir/curves/에 저장 (eval HTML 그래프용)."""
    curves_only: bool = False
    """True면 skill npz는 만들지 않고 곡선만 저장 (이미 빌드된 run 백필용). resume은 곡선 파일 존재 기준."""
    # ── WandB ────────────────────────────────────────────────────────────────
    wandb_project: str | None = None
    wandb_run_name: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _threshold_for_episode(sg_vals: np.ndarray, args: Args,
                           global_threshold: float | None) -> float:
    if args.boundary_threshold_mode == "global_mean" and global_threshold is not None:
        return float(global_threshold)
    return float(np.mean(sg_vals)) if len(sg_vals) else 0.0


def _detect_boundaries(replan_ts: list, div_cos: np.ndarray,
                        n_frames: int, args: Args,
                        global_threshold: float | None = None) -> list[int]:
    nms_dist = (args.nms_dist if args.nms_dist is not None
                else args.replan_interval * 2) if args.peak_nms else 0
    sg_vals = _savgol_smooth(list(div_cos), args.smooth_window, polyorder=args.savgol_polyorder)
    threshold = _threshold_for_episode(sg_vals, args, global_threshold)
    peak_ts, _ = _find_peaks_above_threshold(
        sg_vals, replan_ts, threshold=threshold, min_distance=nms_dist, margin=nms_dist
    )
    return sorted(set([0] + [int(p) for p in peak_ts] + [n_frames]))


def _save_boundary_curve(curves_dir: Path, ep_id: int, task_id: int, replan_ts,
                         div_cos: np.ndarray, boundaries: list[int], n_frames: int,
                         args: "Args", global_threshold: float | None = None) -> None:
    """Persist the per-episode multimodality (VF cos-divergence) curve so the eval
    HTML can overlay it. Stores raw + SG-smoothed values, the selected threshold, the
    detected peaks and the final boundaries — everything the plot needs, so eval
    needs neither scipy nor the detection params (mirrors _detect_boundaries)."""
    replan_ts = np.asarray(replan_ts, dtype=np.int64)
    div_cos = np.asarray(div_cos, dtype=np.float32)
    sg_vals = np.asarray(
        _savgol_smooth(list(div_cos), args.smooth_window, polyorder=args.savgol_polyorder),
        dtype=np.float32,
    )
    threshold = _threshold_for_episode(sg_vals, args, global_threshold)
    nms_dist = (args.nms_dist if args.nms_dist is not None
                else args.replan_interval * 2) if args.peak_nms else 0
    peak_ts, peak_vals = _find_peaks_above_threshold(
        sg_vals, replan_ts.tolist(), threshold=threshold,
        min_distance=nms_dist, margin=nms_dist,
    )
    curves_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(curves_dir / f"ep{ep_id:07d}.npz"),
        episode_id=np.array(ep_id), task_id=np.array(task_id),
        replan_ts=replan_ts, div_cos=div_cos, sg_vals=sg_vals,
        mean_val=np.array(threshold, dtype=np.float32),
        threshold_mode=np.array(
            "episode_mean_staging"
            if args.curves_only and args.boundary_threshold_mode == "global_mean"
            else args.boundary_threshold_mode
        ),
        peak_ts=np.asarray(peak_ts, dtype=np.int64),
        peak_vals=np.asarray(peak_vals, dtype=np.float32),
        boundaries=np.asarray(boundaries, dtype=np.int64),
        n_frames=np.array(n_frames),
    )


def _find_peaks_above_threshold(vals: np.ndarray, ts: list, *, threshold: float,
                                min_distance: int = 0, margin: int = 0) -> tuple[list, list]:
    """Same peak/NMS policy as skill_divider, with an explicit threshold."""
    from scipy.signal import find_peaks

    peak_idxs, _ = find_peaks(vals)
    above = [i for i in peak_idxs if vals[i] > threshold]

    if margin > 0 and len(ts) >= 2:
        t_min, t_max = ts[0], ts[-1]
        above = [i for i in above if ts[i] - t_min > margin and t_max - ts[i] > margin]

    if min_distance > 0 and len(above) > 1:
        # Greedy NMS by peak height (not scan order), matching _find_peaks_above_mean.
        by_height = sorted(above, key=lambda i: vals[i], reverse=True)
        kept, suppressed = [], set()
        for idx in by_height:
            if idx in suppressed:
                continue
            kept.append(idx)
            for other in by_height:
                if other != idx and abs(ts[other] - ts[idx]) <= min_distance:
                    suppressed.add(other)
        above = kept

    return [ts[i] for i in above], [float(vals[i]) for i in above]


def _load_global_threshold(path: str) -> float:
    if not path:
        raise ValueError("global_mean requires --global_threshold_path")
    threshold_path = Path(path)
    if not threshold_path.is_file():
        raise FileNotFoundError(
            f"global_mean threshold file is missing: {threshold_path}. "
            "Run the curves collection/global-threshold pass first."
        )
    with threshold_path.open() as f:
        payload = json.load(f)
    if payload.get("boundary_threshold_mode") != "global_mean":
        raise ValueError(f"Not a global_mean threshold file: {threshold_path}")
    value = float(payload["global_mean"])
    if not np.isfinite(value):
        raise ValueError(f"Invalid global mean in {threshold_path}: {value}")
    return value


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


def _done_marker(skills_dir: Path, ep_id: int, task_id: int) -> Path:
    return skills_dir / f"task{task_id:02d}" / f"ep{ep_id:05d}_task{task_id:02d}.done"


def _touch_done(skills_dir: Path, ep_id: int, task_id: int) -> None:
    """Mark an (episode, task) as processed, even when it produced 0 skills
    (filtered / empty). Lets verify_skillset.py tell finished-but-empty episodes
    apart from ones a dead GPU never reached (so retries terminate)."""
    m = _done_marker(skills_dir, ep_id, task_id)
    m.parent.mkdir(parents=True, exist_ok=True)
    m.touch()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: Args) -> None:
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    skills_dir = output_dir / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = output_dir / "curves"
    if args.dump_curves or args.curves_only or args.use_cached_curves:
        curves_dir.mkdir(parents=True, exist_ok=True)

    if args.curves_only and args.use_cached_curves:
        raise ValueError("--curves_only and --use_cached_curves are mutually exclusive")
    global_threshold = None
    if args.boundary_threshold_mode == "global_mean" and not args.curves_only:
        global_threshold = _load_global_threshold(args.global_threshold_path)
        print(f"Global boundary threshold: {global_threshold:.8f} ({args.global_threshold_path})")

    if args.seed is not None:
        import random, torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    print("Loading metadata...")
    episodes_meta = load_episodes_meta(dataset_dir)
    tasks_meta = pd.read_parquet(dataset_dir / "meta" / "tasks.parquet").reset_index()

    # Episode → task_index (the UNIQUE task key). The dataset's "tasks" field is the language
    # STRING, which is NOT unique across task_index — e.g. a merged FT+PT dataset can have two
    # task_index sharing one language. Selecting episodes by language would conflate them and
    # segment an episode under every task that shares its language (duplicate skills). Map each
    # episode to its task_index from the per-frame data instead (each episode has a single one).
    _ep_task: dict[int, int] = {}
    for _pq in sorted((dataset_dir / "data").glob("*/*.parquet")):
        _df = pd.read_parquet(_pq, columns=["episode_index", "task_index"]).drop_duplicates("episode_index")
        _ep_task.update({int(e): int(t) for e, t in zip(_df["episode_index"], _df["task_index"])})

    def _episodes_for_task(tid: int) -> list[int]:
        return sorted(ep for ep, ti in _ep_task.items() if ti == tid)

    task_ids = args.task_ids if args.task_ids is not None else sorted(tasks_meta["task_index"].tolist())
    print(f"Tasks to process: {len(task_ids)}")

    # Pre-count total episodes for ETA (by task_index — see _ep_task note above).
    all_episode_ids = []
    for task_id in task_ids:
        all_episode_ids.extend(_episodes_for_task(task_id))
    n_total_eps_global = len(all_episode_ids)
    print(f"Total episodes: {n_total_eps_global}")

    policy = preprocessor = None
    if args.use_cached_curves:
        print(f"Using cached curves from {curves_dir}; skipping DP/VF inference.")
    else:
        print(f"Loading policy from {args.policy_path} ...")
        t0 = time.time()
        policy, preprocessor = load_policy(
            args.policy_path, args.device, args.noise_scheduler_type, args.num_inference_steps
        )
        print(f"  [time] policy load: {time.time()-t0:.1f}s")

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
                "boundary_threshold_mode": args.boundary_threshold_mode,
                "global_threshold": global_threshold,
                "use_cached_curves": args.use_cached_curves,
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
        target_lang = task_row.iloc[0]["task"]          # for display / skill metadata
        episode_ids = _episodes_for_task(task_id)        # by task_index, NOT language (see note above)
        n_total_eps += len(episode_ids)
        print(f"\nTask {task_id}: '{target_lang}' — {len(episode_ids)} episodes")

        ep_data = load_data(dataset_dir, episode_ids=episode_ids)

        for ep_id in episode_ids:
            # Resume: an episode is done if it has a .done marker (covers filtered /
            # 0-skill episodes too) or any saved skill npz.
            task_dir = skills_dir / f"task{task_id:02d}"
            marker = _done_marker(skills_dir, ep_id, task_id)
            existing = list(task_dir.glob(f"ep{ep_id:05d}_task{task_id:02d}_skill*.npz"))
            curve_path = curves_dir / f"ep{ep_id:07d}.npz"
            if args.curves_only:
                # Backfill mode: resume keyed by the curve file only, so an already
                # fully-built run (all skills present) still regenerates its curves.
                if args.resume and curve_path.exists():
                    n_skipped_resume += 1
                    n_processed += 1
                    print(f"  [skip] ep{ep_id:05d} curve exists")
                    continue
            elif args.resume and (marker.exists() or existing):
                n_skipped_resume += 1
                n_processed += 1
                total_skills += len(existing)
                if args.write_done_markers and not marker.exists():
                    _touch_done(skills_dir, ep_id, task_id)
                print(f"  [skip] ep{ep_id:05d} already done ({len(existing)} skills)")
                continue

            if episodes_meta[episodes_meta["episode_index"] == ep_id].empty:
                if args.write_done_markers:
                    _touch_done(skills_dir, ep_id, task_id)
                continue
            ep_df = ep_data[ep_data["episode_index"] == ep_id].reset_index(drop=True)
            if len(ep_df) == 0:
                if args.write_done_markers:
                    _touch_done(skills_dir, ep_id, task_id)
                continue

            print(f"  ep{ep_id:05d} ...", end="", flush=True)
            t_ep = time.time()

            try:
                if args.use_cached_curves:
                    if not curve_path.is_file():
                        raise FileNotFoundError(
                            f"Cached curve missing for ep{ep_id:05d}: {curve_path}"
                        )
                    with np.load(curve_path) as curve:
                        vf_replan_ts = curve["replan_ts"].astype(np.int64).tolist()
                        div_cos = curve["div_cos"].astype(np.float32)
                else:
                    assert policy is not None and preprocessor is not None
                    vf_replan_ts, _, _, div_cos, _, _, _ = run_vf_analysis(
                        policy, preprocessor, ep_df,
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
            boundaries = _detect_boundaries(
                vf_replan_ts, div_cos, n_frames, args, global_threshold=global_threshold
            )

            if args.dump_curves or args.curves_only or args.use_cached_curves:
                _save_boundary_curve(curves_dir, ep_id, task_id, vf_replan_ts, div_cos,
                                     boundaries, n_frames, args, global_threshold=global_threshold)

            if args.curves_only:
                n_processed += 1
                print(f" curve saved  [{time.time() - t_ep:.1f}s]")
                continue

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

            # mark processed (errors above `continue` earlier → no marker → retried)
            if args.write_done_markers:
                _touch_done(skills_dir, ep_id, task_id)

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
