"""
Skill Boundary Detector — demonstration replay and skill boundary detection.

Core algorithm:
  1. Load demonstration episodes (observations + ground-truth actions).
  2. Run a pretrained Diffusion Policy on each episode's GT observations.
  3. Compute per-chunk MSE between predicted and GT actions.
  4. Smooth the MSE signal (moving average or Savitzky-Golay).
  5. Detect MSE peaks above the mean → skill boundaries.
  6. Delegate all plotting / video / wandb to SkillVisualizer (SBD_visualize.py).

Usage:
python examples/libero/replay_demo.py \
  --dataset_dir /scratch/mdorazi/Skill_Boundary_Detector/libero_dataset/libero_10 \
  --task_id 0 --n_episodes 1 \
  --output_dir /scratch/mdorazi/Skill_Boundary_Detector/outputs/replay \
  --policy_path /scratch/mdorazi/Skill_Boundary_Detector/outputs/dp_libero90_yonsei_pretrain/checkpoints/080000/pretrained_model \
  --replan_interval 5 --mse_smooth_method savgol --mse_smooth_window 5 \
  --wandb_project SBD_replay \
  --skip_viz
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio
import numpy as np
import pandas as pd
import tyro

# SBD_visualize.py lives in the same directory
sys.path.insert(0, str(Path(__file__).parent))
from SBD_visualize import SkillVisualizer


@dataclass
class Args:
    dataset_dir: str = "/data2/dohyeon/SBD/libero_dataset/libero_90"
    task_id: int | None = None
    """Dataset task_index (0-based). Replays first n_episodes episodes of this task."""
    episode_ids: str | None = None
    """JSON list of episode indices, e.g. '[0,1,2]'. Overrides task_id."""
    n_episodes: int = 3
    output_dir: str = "/data2/dohyeon/SBD/outputs/replay"
    wandb_project: str | None = None
    fps: int = 20
    policy_path: str | None = None
    """Path to pretrained_model dir (e.g. .../checkpoints/060000/pretrained_model)."""
    device: str = "cuda"
    replan_interval: int = 0
    """0 = use policy default (n_action_steps). N = replan every N steps."""
    seed: int | None = 42
    mse_window: int | None = 4
    """Trim last mse_window steps from each chunk when computing MSE."""
    mse_smooth_window: int = 1
    """Smoothing window size. 1 = no smoothing."""
    mse_smooth_method: str = "ma"
    """'ma' (moving average), 'savgol' (Savitzky-Golay), or 'both'."""
    savgol_polyorder: int = 4
    """Polynomial order for Savitzky-Golay filter."""
    peak_nms: bool = True
    """Suppress smaller SG peaks within replan_interval*2 distance from a larger peak."""
    save_skills: bool = True
    """Save detected skill segments (actions + states) as .npz files under output_dir/skills/."""
    skip_viz: bool = False
    """Skip all video extraction and plot generation. Only run inference + skill saving."""


# ── Signal processing ──────────────────────────────────────────────────────────

def _centered_moving_avg(vals: list, window: int) -> np.ndarray:
    if window <= 1:
        return np.array(vals, dtype=float)
    return np.convolve(vals, np.ones(window) / window, mode="same")


def _savgol_smooth(vals: list, window: int, polyorder: int = 3) -> np.ndarray:
    from scipy.signal import savgol_filter
    if window <= 1:
        return np.array(vals, dtype=float)
    win = window if window % 2 == 1 else window + 1   # window_length must be odd
    win = min(win, len(vals))
    if win % 2 == 0:
        win -= 1
    if win < polyorder + 2:
        return np.array(vals, dtype=float)
    return savgol_filter(vals, window_length=win, polyorder=polyorder)


def _compute_smoothed(vals: list, window: int, method: str, polyorder: int = 3) -> dict[str, np.ndarray]:
    result = {}
    if method in ("ma", "both"):
        result["ma"] = _centered_moving_avg(vals, window)
    if method in ("savgol", "both"):
        result["savgol"] = _savgol_smooth(vals, window, polyorder)
    return result


# ── MSE computation ────────────────────────────────────────────────────────────

def _compute_mse_per_chunk(
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    n_action_steps: int,
    pred_chunks: np.ndarray | None = None,
    mse_window: int = 4,
    replan_interval: int = 1,
) -> tuple[list, list, float]:
    """Return (replan_ts, mse_vals, bar_width).

    Each value in mse_vals is the mean squared error over the xyz dims
    for one action chunk, optionally trimming the last mse_window steps.
    """
    T = len(gt_actions)
    if pred_chunks is not None:
        replan_ts, mse_vals = [], []
        for ci in range(len(pred_chunks)):
            t = ci * replan_interval
            end = min(t + mse_window, T)
            mse = float(np.mean((pred_chunks[ci, : end - t, :3] - gt_actions[t:end, :3]) ** 2))
            replan_ts.append(t)
            mse_vals.append(mse)
        bar_width = replan_interval * 0.8
    else:
        replan_ts = list(range(0, T, n_action_steps))
        mse_vals = []
        for t in replan_ts:
            end = min(t + mse_window, T)
            mse = float(np.mean((pred_actions[t:end, :3] - gt_actions[t:end, :3]) ** 2))
            mse_vals.append(mse)
        bar_width = n_action_steps * 0.8
    return replan_ts, mse_vals, bar_width


# ── Skill boundary detection ───────────────────────────────────────────────────

def _find_peaks_above_mean(vals: np.ndarray, ts: list, min_distance: int = 0) -> tuple[list, list]:
    """Return (peak_ts, peak_vals) for local maxima of vals that exceed the mean.

    If min_distance > 0, apply greedy NMS: when two peaks are within
    min_distance frames of each other, keep only the taller one.
    """
    from scipy.signal import find_peaks
    mean_val = float(np.mean(vals))
    peak_idxs, _ = find_peaks(vals)
    above = [i for i in peak_idxs if vals[i] > mean_val]

    if min_distance > 0 and len(above) > 1:
        above_by_height = sorted(above, key=lambda i: vals[i], reverse=True)
        kept, suppressed = [], set()
        for i in above_by_height:
            if i in suppressed:
                continue
            kept.append(i)
            for j in above_by_height:
                if j != i and abs(ts[j] - ts[i]) <= min_distance:
                    suppressed.add(j)
        above = kept

    return [ts[i] for i in above], [float(vals[i]) for i in above]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(dataset_dir: Path) -> pd.DataFrame:
    files = sorted((dataset_dir / "data").rglob("file-*.parquet"))
    if not files:
        files = sorted((dataset_dir / "data").rglob("episode_*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_episodes_meta(dataset_dir: Path) -> pd.DataFrame:
    files = sorted((dataset_dir / "meta" / "episodes").rglob("file-*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def get_video_path(dataset_dir: Path, episode_index: int, camera_key: str, episodes_meta: pd.DataFrame) -> Path:
    row = episodes_meta[episodes_meta["episode_index"] == episode_index].iloc[0]
    chunk_idx = row[f"videos/{camera_key}/chunk_index"]
    file_idx = row[f"videos/{camera_key}/file_index"]
    return dataset_dir / "videos" / camera_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"


def get_episode_timestamps(dataset_dir: Path, episode_index: int, episodes_meta: pd.DataFrame, camera_key: str) -> tuple[float, float]:
    row = episodes_meta[episodes_meta["episode_index"] == episode_index].iloc[0]
    return float(row[f"videos/{camera_key}/from_timestamp"]), float(row[f"videos/{camera_key}/to_timestamp"])


# ── Policy loading and inference ───────────────────────────────────────────────

def load_policy(policy_path: str, device: str = "cuda"):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors
    policy = DiffusionPolicy.from_pretrained(policy_path)
    policy = policy.to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=policy_path)
    return policy, preprocessor, postprocessor


def _tensor_to_numpy(t) -> np.ndarray:
    arr = t.cpu().numpy() if hasattr(t, "cpu") else t["action"].cpu().numpy()
    return arr[0] if arr.ndim == 2 else arr


def run_policy_inference(
    policy, preprocessor, postprocessor,
    ep_df: pd.DataFrame, cam_clips: list[Path], camera_keys: list[str],
    replan_interval: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Feed GT observations into the policy and return predicted actions.

    Returns:
        pred_actions: (T, action_dim)
        pred_chunks:  (n_chunks, n_action_steps, action_dim) or None
    """
    import torch
    from lerobot.utils.constants import ACTION

    n_action_steps = policy.config.n_action_steps
    states = np.stack(ep_df["observation.state"].values)

    cam_frames: dict[str, np.ndarray] = {}
    for cam_key, clip_path in zip(camera_keys, cam_clips):
        reader = imageio.get_reader(str(clip_path))
        cam_frames[cam_key] = np.stack([f for f in reader])
        reader.close()

    # Clamp T to the shortest sequence among dataframe and all videos
    T = min(len(ep_df), *(len(v) for v in cam_frames.values()))

    states = states[:T]
    for k in cam_frames:
        cam_frames[k] = cam_frames[k][:T]

    eff_interval = replan_interval if replan_interval > 0 else n_action_steps
    capture_chunks = replan_interval > 0

    policy.reset()
    pred_actions = []
    pred_chunks = [] if capture_chunks else None

    for t in range(T):
        obs = {k: torch.from_numpy(cam_frames[k][t]).float().div(255.0).permute(2, 0, 1) for k in camera_keys}
        obs["observation.state"] = torch.from_numpy(states[t]).float()
        obs = preprocessor(obs)

        if capture_chunks and t % eff_interval == 0:
            policy._queues[ACTION].clear()

        with torch.inference_mode():
            raw_action = policy.select_action(obs)

        if capture_chunks and t % eff_interval == 0:
            remaining = list(policy._queues[ACTION])
            chunk = [_tensor_to_numpy(postprocessor(raw_action))]
            chunk += [_tensor_to_numpy(postprocessor(a)) for a in remaining]
            pred_chunks.append(np.stack(chunk))

        pred_actions.append(_tensor_to_numpy(postprocessor(raw_action)))

    return np.stack(pred_actions), (np.stack(pred_chunks) if pred_chunks is not None else None)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: Args) -> None:
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    viz = SkillVisualizer(output_dir, fps=args.fps)

    print("Loading metadata...")
    episodes_meta = load_episodes_meta(dataset_dir)
    tasks_meta = pd.read_parquet(dataset_dir / "meta" / "tasks.parquet").reset_index()

    # Determine episode IDs
    if args.episode_ids is not None:
        episode_ids = json.loads(args.episode_ids)
    elif args.task_id is not None:
        task_row = tasks_meta[tasks_meta["task_index"] == args.task_id]
        if task_row.empty:
            raise ValueError(f"task_id {args.task_id} not found")
        target_lang = task_row.iloc[0]["task"]
        ep_of_task = episodes_meta[episodes_meta["tasks"].apply(
            lambda t: target_lang in ([str(x) for x in t] if isinstance(t, (list, np.ndarray)) else [str(t)])
        )]
        episode_ids = ep_of_task["episode_index"].tolist()[:args.n_episodes]
        print(f"Task {args.task_id}: '{target_lang}' — {len(episode_ids)} episodes")
    else:
        raise ValueError("Provide --task_id or --episode_ids")

    video_cols = [c for c in episodes_meta.columns if c.startswith("videos/") and c.endswith("/chunk_index")]
    camera_keys = [c.split("/")[1] for c in video_cols]
    print(f"Cameras: {camera_keys}")

    print("Loading state data...")
    all_data = load_data(dataset_dir)

    if args.seed is not None:
        import random, torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"Seed fixed: {args.seed}")

    policy_bundle = None
    if args.policy_path:
        print(f"Loading policy from {args.policy_path} ...")
        policy_bundle = load_policy(args.policy_path, args.device)

    skills_dir = output_dir / "skills"
    if args.save_skills:
        skills_dir.mkdir(parents=True, exist_ok=True)

    # ── wandb progress run ─────────────────────────────────────────────────────
    progress_run = None
    if args.wandb_project:
        import wandb
        run_name = f"task{args.task_id}_progress" if args.task_id is not None else "replay_progress"
        progress_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "task_id": args.task_id,
                "n_episodes": len(episode_ids),
                "replan_interval": args.replan_interval,
                "mse_smooth_method": args.mse_smooth_method,
                "mse_smooth_window": args.mse_smooth_window,
            },
            reinit=True,
        )

    n_total = len(episode_ids)
    n_done = 0
    total_skills = 0

    results = []
    for ep_id in episode_ids:
        row = episodes_meta[episodes_meta["episode_index"] == ep_id]
        if row.empty:
            print(f"  Episode {ep_id} not found, skipping.")
            continue
        tasks_list = row.iloc[0]["tasks"]
        lang = str(tasks_list[0] if isinstance(tasks_list, (list, np.ndarray)) else tasks_list)
        ep_tag = f"ep{ep_id}: {lang[:80]}"

        # ── Resume: skip already-processed episodes ────────────────────────────
        if args.save_skills and any(skills_dir.glob(f"ep{ep_id:05d}_skill*.npz")):
            existing = list(skills_dir.glob(f"ep{ep_id:05d}_skill*.npz"))
            n_done += 1
            total_skills += len(existing)
            print(f"  [skip] Episode {ep_id} already done ({len(existing)} skills)")
            if progress_run is not None:
                import wandb
                progress_run.log({
                    "progress/episodes_done": n_done,
                    "progress/episodes_total": n_total,
                    "progress/episodes_pct": n_done / n_total * 100,
                    "progress/skills_found": total_skills,
                    "progress/skills_this_ep": len(existing),
                })
            continue

        print(f"\n  Episode {ep_id}: '{lang}'")

        # ── Extract clips ──────────────────────────────────────────────────────
        cam_clips = []
        _tmp_clips = []  # clips to delete after inference when skip_viz
        for cam_key in camera_keys:
            src_video = get_video_path(dataset_dir, ep_id, cam_key, episodes_meta).resolve()
            start_sec, end_sec = get_episode_timestamps(dataset_dir, ep_id, episodes_meta, cam_key)
            if args.skip_viz:
                import tempfile, os
                tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                clip_path = Path(tmp.name)
                tmp.close()
                _tmp_clips.append(clip_path)
            else:
                clip_path = output_dir / f"ep{ep_id:05d}_{cam_key.replace('.', '_')}.mp4"
            viz.extract_clip(src_video, clip_path, start_sec, end_sec)
            cam_clips.append(clip_path)

        combined_path = None
        if not args.skip_viz:
            if len(cam_clips) == 2:
                combined_path = output_dir / f"ep{ep_id:05d}_combined.mp4"
                viz.stack_clips_side_by_side(cam_clips[0], cam_clips[1], combined_path)

        ep_df = all_data[all_data["episode_index"] == ep_id].reset_index(drop=True)
        if not args.skip_viz:
            traj_path = viz.plot_eef_trajectory(ep_df, ep_id, title=ep_tag)
        else:
            traj_path = None

        # ── Policy inference ───────────────────────────────────────────────────
        pred_path = cum_path = mse_path = None
        mse_data_for_results = None
        sg_peak_ts: list = []

        if policy_bundle is not None:
            policy, preprocessor, postprocessor = policy_bundle
            n_act = policy.config.n_action_steps
            eff_replan = args.replan_interval if args.replan_interval > 0 else n_act

            print(f"    Running policy inference (replan_interval={args.replan_interval})...")
            gt_actions = np.stack(ep_df["action"].values)
            pred_actions, pred_chunks = run_policy_inference(
                policy, preprocessor, postprocessor, ep_df, cam_clips, camera_keys,
                replan_interval=args.replan_interval,
            )

            # ── Action comparison plots ────────────────────────────────────────
            if not args.skip_viz:
                if args.replan_interval > 0:
                    pred_path = viz.plot_action_compare_multichunk(gt_actions, pred_chunks, ep_id, ep_tag, args.replan_interval)
                    cum_path = viz.plot_cumulative_multichunk(gt_actions, pred_chunks, ep_id, ep_tag, args.replan_interval)
                else:
                    pred_path = viz.plot_action_comparison(gt_actions, pred_actions, ep_id, ep_tag)
                    cum_path = viz.plot_cumulative_trajectory(gt_actions, pred_actions, ep_id, ep_tag)

            # ── MSE signal ────────────────────────────────────────────────────
            replan_ts, mse_vals, bar_width = _compute_mse_per_chunk(
                gt_actions, pred_actions, n_act, pred_chunks, args.mse_window, eff_replan
            )

            # ── Smooth MSE ────────────────────────────────────────────────────
            smoothed_dict = _compute_smoothed(
                mse_vals, args.mse_smooth_window, args.mse_smooth_method, args.savgol_polyorder
            ) if args.mse_smooth_window > 1 else {}

            # ── Detect skill boundaries (SG peaks above mean) ─────────────────
            if "savgol" in smoothed_dict:
                nms_dist = eff_replan * 2 if args.peak_nms else 0
                sg_peak_ts, _ = _find_peaks_above_mean(
                    smoothed_dict["savgol"], replan_ts, min_distance=nms_dist
                )

            # ── Plot MSE ──────────────────────────────────────────────────────
            if not args.skip_viz:
                mse_path = viz.plot_mse(
                    replan_ts, mse_vals, bar_width, smoothed_dict, sg_peak_ts,
                    ep_id, ep_tag, args.mse_smooth_window,
                )
            mse_data_for_results = (
                replan_ts, mse_vals,
                {k: v.tolist() for k, v in smoothed_dict.items()},
                bar_width,
            )

            # ── Save skill segments ────────────────────────────────────────────
            if args.save_skills and sg_peak_ts:
                states_arr = np.stack(ep_df["observation.state"].values[:len(gt_actions)])
                cuts = sorted(set(sg_peak_ts))
                breakpoints = [0] + [min(c, len(gt_actions)) for c in cuts] + [len(gt_actions)]
                breakpoints = sorted(set(breakpoints))
                skill_files = []
                for si in range(len(breakpoints) - 1):
                    s, e = breakpoints[si], breakpoints[si + 1]
                    if e - s < 2:
                        continue
                    fname = skills_dir / f"ep{ep_id:05d}_skill{si:02d}.npz"
                    np.savez(
                        str(fname),
                        actions=gt_actions[s:e].astype(np.float32),
                        states=states_arr[s:e].astype(np.float32),
                        episode_id=np.array(ep_id),
                        skill_index=np.array(si),
                        frame_start=np.array(s),
                        frame_end=np.array(e),
                    )
                    skill_files.append(str(fname))
                print(f"    Skill files:      {len(skill_files)} saved to {skills_dir.name}/")

            print(f"    Skill boundaries: {sg_peak_ts}")
            if not args.skip_viz:
                if pred_path:
                    print(f"    Action compare:   {pred_path.name}")
                if mse_path:
                    print(f"    Replanning MSE:   {mse_path.name}")

        # ── Clean up temp clips (skip_viz mode) ───────────────────────────────
        for tmp_clip in _tmp_clips:
            try:
                tmp_clip.unlink()
            except Exception:
                pass

        # ── wandb progress logging ─────────────────────────────────────────────
        n_done += 1
        ep_skills = len(sg_peak_ts) + 1 if sg_peak_ts else 0
        total_skills += ep_skills
        if progress_run is not None:
            import wandb
            progress_run.log({
                "progress/episodes_done": n_done,
                "progress/episodes_total": n_total,
                "progress/episodes_pct": n_done / n_total * 100,
                "progress/skills_found": total_skills,
                "progress/skills_this_ep": ep_skills,
            })
            print(f"  [{n_done}/{n_total}] ep{ep_id} done — {ep_skills} skills (total: {total_skills})")

        results.append({
            "episode_id": ep_id,
            "language": lang,
            "video_paths": [str(p) for p in cam_clips],
            "combined_path": str(combined_path) if combined_path else None,
            "traj_path": str(traj_path),
            "pred_path": str(pred_path) if pred_path else None,
            "cum_path": str(cum_path) if cum_path else None,
            "mse_path": str(mse_path) if mse_path else None,
            "mse_data": mse_data_for_results,
            "sg_peak_ts": sg_peak_ts,
            "n_frames": len(ep_df),
            "slider_video": str(combined_path) if combined_path else (str(cam_clips[0]) if cam_clips else None),
        })

    # Save results JSON
    results_path = output_dir / "replay_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. Results saved to {results_path}")

    # Upload to wandb (media/plots — skip_viz면 아무것도 없음)
    if args.wandb_project and not args.skip_viz:
        try:
            run_name = f"task{args.task_id}_ep{args.n_episodes}"
            viz.log_to_wandb(
                results, args.wandb_project, run_name,
                replan_interval=args.replan_interval,
                mse_smooth_window=args.mse_smooth_window,
            )
        except Exception as e:
            print(f"wandb logging failed: {e}")

    if progress_run is not None:
        progress_run.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
