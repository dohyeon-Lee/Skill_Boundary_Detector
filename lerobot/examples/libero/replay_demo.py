"""
Extract demonstration videos from a LeRobot v3.0 dataset.

Reads frame timestamps from parquet and extracts the corresponding clip
from the pre-encoded MP4 videos (no simulation required).
Produces one video per camera per episode, and optionally logs to wandb.

Usage:
PYTHONPATH=/data2/dohyeon/SBD/lerobot/src \
/data2/dohyeon/SBD/.venv/bin/python \
/data2/dohyeon/SBD/lerobot/examples/libero/replay_demo.py \
--dataset_dir /data2/dohyeon/SBD/libero_dataset/libero_90 \
--task_id 2 --n_episodes 1 \
--output_dir /data2/dohyeon/SBD/outputs/replay \
--wandb_project SBD_replay



python examples/libero/replay_demo.py \
--dataset_dir /data2/dohyeon/SBD/libero_dataset/libero_10 \
--task_id 4 --n_episodes 1 \
--output_dir /data2/dohyeon/SBD/outputs/replay \
--wandb_project SBD_replay



python examples/libero/replay_demo.py \
  --dataset_dir /scratch/mdorazi/Skill_Boundary_Detector/libero_dataset/libero_10 \
  --n_episodes 20 \
  --output_dir /scratch/mdorazi/Skill_Boundary_Detector/outputs/replay \
  --wandb_project SBD_replay \
  --policy_path /scratch/mdorazi/Skill_Boundary_Detector/outputs/dp_libero90_yonsei_pretrain/checkpoints/080000/pretrained_model \
  --mse_window 4 \
  --savgol_polyorder 4 \
  --mse_smooth_window 5 \
  --replan_interval 5 \
  --mse_smooth_method savgol \
  --peak_nms \
  --task_id 0 

  --------------
  savgol
  --savgol_polyorder 4
  --seed None
"""

import json
from dataclasses import dataclass
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro
from PIL import Image, ImageDraw, ImageFont


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
    """Path to pretrained_model dir (e.g. outputs/.../checkpoints/060000/pretrained_model)"""
    device: str = "cuda"
    replan_interval: int = 0
    """Replanning interval in steps. 0 = use policy default (n_action_steps). 1 = every step. N = every N steps."""
    seed: int | None = 42
    """Random seed for reproducible diffusion inference. None = non-deterministic."""
    mse_window: int | None = 0
    """MSE caculation window size (chunk size - mse_window)"""
    mse_smooth_window: int = 1
    """Centered moving average window for MSE smoothing. 1 = no smoothing."""
    mse_smooth_method: str = "ma"
    """Smoothing method: 'ma' (moving average), 'savgol' (Savitzky-Golay), or 'both'."""
    savgol_polyorder: int = 3
    """Polynomial order for Savitzky-Golay filter."""
    peak_nms: bool = True
    """If True, suppress smaller SG peaks within replan_interval*2 distance."""

def extract_clip(src_video: Path, dst_video: Path, start_sec: float, end_sec: float, fps: int) -> None:
    reader = imageio.get_reader(str(src_video))
    src_fps = reader.get_meta_data().get("fps", fps)
    start_frame = int(start_sec * src_fps)
    end_frame = int(end_sec * src_fps)
    writer = imageio.get_writer(str(dst_video), fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1)
    ep_t = 0
    for i, frame in enumerate(reader):
        if i < start_frame:
            continue
        if i >= end_frame:
            break
        writer.append_data(_overlay_timestep(frame, ep_t))
        ep_t += 1
    writer.close()
    reader.close()


def _overlay_timestep(frame: np.ndarray, t: int, ep_start: int = 0) -> np.ndarray:
    """Burn frame index and episode-relative timestep into the top-left corner."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    text = f"t={t - ep_start:04d}  (frame {t})"
    # draw shadow then white text for readability on any background
    draw.text((9, 9), text, fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255))
    return np.array(img)


def stack_videos_side_by_side(left: Path, right: Path, dst: Path, fps: int, ep_start: int = 0) -> None:
    """Concatenate two videos side by side with timestep overlay."""
    reader_l = imageio.get_reader(str(left))
    reader_r = imageio.get_reader(str(right))
    writer = imageio.get_writer(str(dst), fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1)
    for i, (fl, fr) in enumerate(zip(reader_l, reader_r)):
        combined = np.concatenate([fl, fr], axis=1)
        combined = _overlay_timestep(combined, i, ep_start=0)
        writer.append_data(combined)
    writer.close()
    reader_l.close()
    reader_r.close()


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
    policy,
    preprocessor,
    postprocessor,
    ep_df: pd.DataFrame,
    cam_clips: list[Path],
    camera_keys: list[str],
    replan_interval: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run policy on dataset observations.

    Returns:
        pred_actions: (T, action_dim) — one action per timestep
        pred_chunks:  (T // replan_interval, n_action_steps, action_dim) or None if replan_interval=0
    """
    import torch
    from lerobot.utils.constants import ACTION

    n_action_steps = policy.config.n_action_steps
    states = np.stack(ep_df["observation.state"].values)

    cam_frames: dict[str, np.ndarray] = {}
    for cam_key, clip_path in zip(camera_keys, cam_clips):
        reader = imageio.get_reader(str(clip_path))
        frames = [f for f in reader]
        reader.close()
        cam_frames[cam_key] = np.stack(frames)  # (N, H, W, C) uint8

    T = min(len(ep_df), *(len(v) for v in cam_frames.values()))  # clamp to shortest
    states = states[:T]
    for cam_key in cam_frames:
        cam_frames[cam_key] = cam_frames[cam_key][:T]

    eff_interval = replan_interval if replan_interval > 0 else n_action_steps
    capture_chunks = replan_interval > 0

    policy.reset()
    pred_actions = []
    pred_chunks = [] if capture_chunks else None

    for t in range(T):
        obs = {}
        for cam_key in camera_keys:
            img = cam_frames[cam_key][t]
            obs[cam_key] = torch.from_numpy(img).float().div(255.0).permute(2, 0, 1)  # (C, H, W)
        obs["observation.state"] = torch.from_numpy(states[t]).float()
        obs = preprocessor(obs)

        if capture_chunks and t % eff_interval == 0:
            policy._queues[ACTION].clear()

        with torch.inference_mode():
            raw_action = policy.select_action(obs)  # (1, action_dim), still normalized

        if capture_chunks and t % eff_interval == 0:
            remaining_raw = list(policy._queues[ACTION])  # n_action_steps-1 items
            chunk_np = [_tensor_to_numpy(postprocessor(raw_action))]
            for raw_a in remaining_raw:
                chunk_np.append(_tensor_to_numpy(postprocessor(raw_a)))
            pred_chunks.append(np.stack(chunk_np))  # (n_action_steps, action_dim)

        action = postprocessor(raw_action)
        pred_actions.append(_tensor_to_numpy(action))

    chunks_arr = np.stack(pred_chunks) if pred_chunks is not None else None  # (T, n_action_steps, action_dim)
    return np.stack(pred_actions), chunks_arr  # (T, action_dim), optional


def plot_action_comparison(
    gt_actions: np.ndarray, pred_actions: np.ndarray, save_path: Path, title: str = ""
) -> None:
    """3 subplots (x, y, z) comparing GT vs predicted action over time."""
    T = len(gt_actions)
    t = np.arange(T)
    labels = ["x", "y", "z"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(t, gt_actions[:, i], label="GT", color="tab:blue", linewidth=1.2)
        ax.plot(t, pred_actions[:, i], label="Pred", color="tab:orange", linewidth=1.2, linestyle="--")
        ax.set_ylabel(f"action {label}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("frame")
    if title:
        axes[0].set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=120)
    plt.close(fig)


def plot_action_compare_multichunk(
    gt_actions: np.ndarray, pred_chunks: np.ndarray, save_path: Path, title: str = "",
    replan_interval: int = 1,
) -> None:
    """GT as one solid line; each predicted chunk as a short faint line segment."""
    n_chunks, n_steps, _ = pred_chunks.shape
    T_gt = len(gt_actions)
    labels = ["x", "y", "z"]
    t_all = np.arange(T_gt)
    replan_ts = [i * replan_interval for i in range(n_chunks)]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(t_all, gt_actions[:, i], label="GT", color="tab:blue", linewidth=1.5, zorder=5)
        for ci, t in enumerate(replan_ts):
            end = min(t + n_steps, T_gt)
            ax.plot(np.arange(t, end), pred_chunks[ci, : end - t, i],
                    color="tab:orange", alpha=0.4, linewidth=1.0)
        ax.set_ylabel(f"action {label}")
        ax.grid(True, alpha=0.3)
    axes[0].legend(["GT", "Pred chunks"], loc="upper right", fontsize=8)
    axes[-1].set_xlabel("frame")
    if title:
        axes[0].set_title(f"[Multi-chunk] {title}", fontsize=9)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=120)
    plt.close(fig)


def plot_cumulative_multichunk(
    gt_actions: np.ndarray, pred_chunks: np.ndarray, save_path: Path, title: str = "",
    replan_interval: int = 1,
) -> None:
    """Cumsum of GT + each predicted chunk starting from GT's cumulative position at t."""
    n_chunks, n_steps, _ = pred_chunks.shape
    T_gt = len(gt_actions)
    labels = ["x", "y", "z"]
    gt_cum = np.cumsum(gt_actions[:, :3], axis=0)
    replan_ts = [i * replan_interval for i in range(n_chunks)]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(np.arange(T_gt), gt_cum[:, i], label="GT (cumsum)", color="tab:blue", linewidth=1.5, zorder=5)
        for ci, t in enumerate(replan_ts):
            end = min(t + n_steps, T_gt)
            length = end - t
            offset = gt_cum[t - 1, i] if t > 0 else 0.0
            pred_cum = offset + np.cumsum(pred_chunks[ci, :length, i])
            ax.plot(np.arange(t, end), pred_cum, color="tab:orange", alpha=0.4, linewidth=1.0)
        ax.set_ylabel(f"Σ Δ{label}")
        ax.grid(True, alpha=0.3)
    axes[0].legend(["GT (cumsum)", "Pred chunks"], loc="upper right", fontsize=8)
    axes[-1].set_xlabel("frame")
    if title:
        axes[0].set_title(f"[Multi-chunk cumsum] {title}", fontsize=9)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=120)
    plt.close(fig)


def _centered_moving_avg(vals: list, window: int) -> np.ndarray:
    """Centered moving average: current value is at the center of the window."""
    if window <= 1:
        return np.array(vals, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(vals, kernel, mode="same")


def _savgol_smooth(vals: list, window: int, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay filter: fits local polynomial — preserves peaks better than MA."""
    from scipy.signal import savgol_filter
    if window <= 1:
        return np.array(vals, dtype=float)
    win = window if window % 2 == 1 else window + 1  # must be odd
    win = min(win, len(vals))
    if win % 2 == 0:
        win -= 1
    if win < polyorder + 2:
        return np.array(vals, dtype=float)
    return savgol_filter(vals, window_length=win, polyorder=polyorder)


def _compute_smoothed(vals: list, window: int, method: str, polyorder: int = 3) -> dict[str, np.ndarray]:
    """Return dict of {method_name: smoothed_array} based on method ('ma', 'savgol', 'both')."""
    result = {}
    if method in ("ma", "both"):
        result["ma"] = _centered_moving_avg(vals, window)
    if method in ("savgol", "both"):
        result["savgol"] = _savgol_smooth(vals, window, polyorder)
    return result


def _compute_mse_data(
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    n_action_steps: int,
    pred_chunks: np.ndarray | None = None,
    mse_window: int = 0,
    replan_interval: int = 1,
) -> tuple[list, list, float]:
    """Return (replan_ts, mse_vals, bar_width)."""
    T = len(gt_actions)
    if pred_chunks is not None:
        n_chunks = len(pred_chunks)
        replan_ts, mse_vals = [], []
        for ci in range(n_chunks):
            t = ci * replan_interval
            end = min(t + n_action_steps - mse_window, T)
            length = end - t
            mse = float(np.mean((pred_chunks[ci, :length, :3] - gt_actions[t:end, :3]) ** 2))
            mse_vals.append(mse)
            replan_ts.append(t)
        bar_width = replan_interval * 0.8
    else:
        replan_ts = list(range(0, T, n_action_steps))
        mse_vals = []
        for t in replan_ts:
            end = min(t + n_action_steps - mse_window, T)
            mse = float(np.mean((pred_actions[t:end, :3] - gt_actions[t:end, :3]) ** 2))
            mse_vals.append(mse)
        bar_width = n_action_steps * 0.8
    return replan_ts, mse_vals, bar_width


def _find_peaks_above_mean(vals: np.ndarray, ts: list, min_distance: int = 0) -> tuple[list, list]:
    """Return (peak_ts, peak_vals) for local maxima above mean.

    If min_distance > 0, suppress the smaller peak when two peaks are closer than min_distance.
    """
    from scipy.signal import find_peaks
    mean_val = float(np.mean(vals))
    peak_idxs, _ = find_peaks(vals)
    above = [i for i in peak_idxs if vals[i] > mean_val]

    if min_distance > 0 and len(above) > 1:
        # Greedy NMS: pick highest peak first, suppress neighbors within min_distance
        above_sorted = sorted(above, key=lambda i: vals[i], reverse=True)
        kept = []
        suppressed = set()
        for i in above_sorted:
            if i in suppressed:
                continue
            kept.append(i)
            for j in above_sorted:
                if j != i and abs(ts[j] - ts[i]) <= min_distance:
                    suppressed.add(j)
        above = kept

    return [ts[i] for i in above], [float(vals[i]) for i in above]


_SMOOTH_COLORS = {"ma": "tab:orange", "savgol": "tab:green"}
_SMOOTH_LABELS = {"ma": "Moving Avg", "savgol": "Savitzky-Golay"}
_SMOOTH_MEAN_LINESTYLES = {"ma": (0, (5, 3)), "savgol": (0, (2, 2))}  # MA: long dash, SG: short dash


def render_combined_frame(
    video_frame: np.ndarray,
    replan_ts: list, mse_vals: list, bar_width: float, T: int, t: int,
    title: str = "", smoothed_dict: dict | None = None, replan_interval: int = 1, peak_nms: bool = True,
) -> np.ndarray:
    """Render video frame (top) + MSE bar chart with vertical line (bottom) as one image."""
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.35)

    # Top: video frame
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(video_frame)
    ax_img.axis("off")
    ax_img.set_title(f"t={t:04d}  {title[:60]}", fontsize=9)

    # Bottom: MSE bar chart with vertical line
    ax_mse = fig.add_subplot(gs[1])
    ax_mse.bar(replan_ts, mse_vals, width=bar_width, align="center", alpha=0.4, color="tab:purple")
    if smoothed_dict:
        for method, vals in smoothed_dict.items():
            color = _SMOOTH_COLORS.get(method, "gray")
            mean_val = float(np.mean(vals))
            ax_mse.plot(replan_ts, vals, color=color,
                        linewidth=2, label=_SMOOTH_LABELS.get(method, method))
            ax_mse.axhline(mean_val, color=color, linewidth=1.5,
                           linestyle=_SMOOTH_MEAN_LINESTYLES.get(method, "--"),
                           label=f"{_SMOOTH_LABELS.get(method, method)} mean={mean_val:.4f}")
            if method == "savgol":
                nms_dist = replan_interval * 2 if peak_nms else 0
                pk_ts, pk_vals = _find_peaks_above_mean(vals, replan_ts, min_distance=nms_dist)
                if pk_ts:
                    ax_mse.scatter(pk_ts, pk_vals, color="red", zorder=5, s=40, label="SG peaks > mean")
        ax_mse.legend(fontsize=7, loc="upper right")
    ax_mse.axvline(x=t, color="red", linewidth=2)
    ax_mse.set_xticks(replan_ts)
    ax_mse.set_xticklabels([str(x) for x in replan_ts], rotation=45, ha="right", fontsize=7)
    ax_mse.set_xlim(-bar_width, T)
    ax_mse.set_xlabel("frame")
    ax_mse.set_ylabel("MSE  xyz")
    ax_mse.grid(True, alpha=0.3, axis="y")

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return img.copy()


def plot_replanning_mse(
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    n_action_steps: int,
    save_path: Path,
    title: str = "",
    pred_chunks: np.ndarray | None = None,
    mse_window: int = 0,
    mse_smooth_window: int = 1,
    replan_interval: int = 1,
    smooth_method: str = "ma",
    savgol_polyorder: int = 3,
    peak_nms: bool = True,
) -> tuple[list, list, dict, float]:
    """Plot MSE graph and return (replan_ts, mse_vals, smoothed_dict, bar_width) for later use."""
    replan_ts, mse_vals, bar_width = _compute_mse_data(
        gt_actions, pred_actions, n_action_steps, pred_chunks, mse_window, replan_interval
    )
    smoothed_dict = _compute_smoothed(mse_vals, mse_smooth_window, smooth_method, savgol_polyorder) if mse_smooth_window > 1 else {}
    sg_peak_ts: list = []
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(replan_ts, mse_vals, width=bar_width, align="center", alpha=0.4, color="tab:purple", label="raw")
    for method, vals in smoothed_dict.items():
        color = _SMOOTH_COLORS.get(method, "gray")
        mean_val = float(np.mean(vals))
        ax.plot(replan_ts, vals, color=color,
                linewidth=2, label=f"{_SMOOTH_LABELS.get(method, method)} (w={mse_smooth_window})")
        ax.axhline(mean_val, color=color, linewidth=1.5,
                   linestyle=_SMOOTH_MEAN_LINESTYLES.get(method, "--"),
                   label=f"{_SMOOTH_LABELS.get(method, method)} mean={mean_val:.4f}")
        if method == "savgol":
            nms_dist = replan_interval * 2 if peak_nms else 0
            pk_ts, pk_vals = _find_peaks_above_mean(vals, replan_ts, min_distance=nms_dist)
            sg_peak_ts = pk_ts
            if pk_ts:
                ax.scatter(pk_ts, pk_vals, color="red", zorder=5, s=40, label="SG peaks > mean")
    if smoothed_dict:
        ax.legend(fontsize=8)
    ax.set_xticks(replan_ts)
    ax.set_xticklabels([str(t) for t in replan_ts], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("frame")
    ax.set_ylabel("MSE  xyz")
    ax.grid(True, alpha=0.3, axis="y")
    if title:
        ax.set_title(f"[Replanning MSE] {title}", fontsize=9)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=120)
    plt.close(fig)
    return replan_ts, mse_vals, smoothed_dict, bar_width, sg_peak_ts


def cut_skill_videos(
    src_video: Path, boundaries: list[int], output_dir: Path, ep_id: int, fps: int
) -> list[Path]:
    """Cut src_video at frame boundaries and return list of skill video paths."""
    reader = imageio.get_reader(str(src_video))
    frames = [f for f in reader]
    reader.close()
    skill_paths = []
    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        skill_path = output_dir / f"ep{ep_id:05d}_skill{i + 1}.mp4"
        writer = imageio.get_writer(str(skill_path), fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1)
        for frame in frames[start:min(end, len(frames))]:
            writer.append_data(frame)
        writer.close()
        skill_paths.append(skill_path)
    return skill_paths


def plot_cumulative_trajectory(
    gt_actions: np.ndarray, pred_actions: np.ndarray, save_path: Path, title: str = ""
) -> None:
    """Cumsum of delta x/y/z: shows estimated EEF path from GT vs Pred."""
    labels = ["x", "y", "z"]
    t = np.arange(len(gt_actions))
    gt_cum = np.cumsum(gt_actions[:, :3], axis=0)
    pred_cum = np.cumsum(pred_actions[:, :3], axis=0)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(t, gt_cum[:, i], label="GT (cumsum)", color="tab:blue", linewidth=1.2)
        ax.plot(t, pred_cum[:, i], label="Pred (cumsum)", color="tab:orange", linewidth=1.2, linestyle="--")
        ax.set_ylabel(f"Σ Δ{label}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("frame")
    if title:
        axes[0].set_title(f"[Cumulative] {title}", fontsize=9)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=120)
    plt.close(fig)


def plot_eef_trajectory(ep_df: pd.DataFrame, save_path: Path, title: str = "") -> None:
    """Plot EEF x,y,z position over time and save as PNG."""
    states = np.stack(ep_df["observation.state"].values)  # (T, 8)
    t = ep_df["timestamp"].values
    xyz = states[:, :3]
    labels = ["x", "y", "z"]
    colors = ["tab:red", "tab:green", "tab:blue"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for ax, label, color in zip(axes, labels, colors):
        ax.plot(t, xyz[:, labels.index(label)], color=color)
        ax.set_ylabel(f"{label} (m)")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    if title:
        axes[0].set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=120)
    plt.close(fig)


def load_data(dataset_dir: Path) -> pd.DataFrame:
    files = sorted((dataset_dir / "data").rglob("file-*.parquet"))
    if not files:
        files = sorted((dataset_dir / "data").rglob("episode_*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_episodes_meta(dataset_dir: Path) -> pd.DataFrame:
    files = sorted((dataset_dir / "meta" / "episodes").rglob("file-*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def get_video_path(dataset_dir: Path, episode_index: int, camera_key: str, episodes_meta: pd.DataFrame) -> Path:
    """Resolve video file path for a given episode and camera."""
    row = episodes_meta[episodes_meta["episode_index"] == episode_index].iloc[0]
    chunk_idx = row[f"videos/{camera_key}/chunk_index"]
    file_idx = row[f"videos/{camera_key}/file_index"]
    return dataset_dir / "videos" / camera_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"


def get_episode_timestamps(dataset_dir: Path, episode_index: int, episodes_meta: pd.DataFrame, camera_key: str) -> tuple[float, float]:
    row = episodes_meta[episodes_meta["episode_index"] == episode_index].iloc[0]
    start = float(row[f"videos/{camera_key}/from_timestamp"])
    end = float(row[f"videos/{camera_key}/to_timestamp"])
    return start, end


def main(args: Args) -> None:
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
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

    # Detect camera keys from episodes_meta columns
    video_cols = [c for c in episodes_meta.columns if c.startswith("videos/") and c.endswith("/chunk_index")]
    camera_keys = [c.split("/")[1] for c in video_cols]
    print(f"Cameras: {camera_keys}")

    # Load state data for trajectory plots
    print("Loading state data...")
    all_data = load_data(dataset_dir)

    # Fix random seed for reproducibility
    if args.seed is not None:
        import random, torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"Seed fixed: {args.seed}")

    # Load policy if provided
    policy_bundle = None
    if args.policy_path:
        print(f"Loading policy from {args.policy_path} ...")
        policy_bundle = load_policy(args.policy_path, args.device)

    results = []
    for ep_id in episode_ids:
        row = episodes_meta[episodes_meta["episode_index"] == ep_id]
        if row.empty:
            print(f"  Episode {ep_id} not found, skipping.")
            continue
        tasks_list = row.iloc[0]["tasks"]
        lang = str(tasks_list[0] if isinstance(tasks_list, (list, np.ndarray)) else tasks_list)

        print(f"  Episode {ep_id}: '{lang}'")

        cam_clips = []
        for cam_key in camera_keys:
            src_video = get_video_path(dataset_dir, ep_id, cam_key, episodes_meta)
            # Follow symlink if needed
            src_video = src_video.resolve()
            start_sec, end_sec = get_episode_timestamps(dataset_dir, ep_id, episodes_meta, cam_key)

            clip_path = output_dir / f"ep{ep_id:05d}_{cam_key.replace('.', '_')}.mp4"
            extract_clip(src_video, clip_path, start_sec, end_sec, args.fps)
            cam_clips.append(clip_path)
            print(f"    {cam_key}: {clip_path.name}")

        # Side-by-side if two cameras
        combined_path = None
        if len(cam_clips) == 2:
            combined_path = output_dir / f"ep{ep_id:05d}_combined.mp4"
            stack_videos_side_by_side(cam_clips[0], cam_clips[1], combined_path, args.fps)
            print(f"    Combined: {combined_path.name}")

        # EEF trajectory plot
        ep_df = all_data[all_data["episode_index"] == ep_id].reset_index(drop=True)
        traj_path = output_dir / f"ep{ep_id:05d}_eef_traj.png"
        plot_eef_trajectory(ep_df, traj_path, title=f"ep{ep_id}: {lang[:80]}")
        print(f"    EEF traj: {traj_path.name}")

        # Policy inference
        pred_path = cum_path = mse_path = None
        if policy_bundle is not None:
            policy, preprocessor, postprocessor = policy_bundle
            print(f"    Running policy inference (replan_interval={args.replan_interval})...")
            gt_actions = np.stack(ep_df["action"].values)  # (T, action_dim)
            pred_actions, pred_chunks = run_policy_inference(
                policy, preprocessor, postprocessor, ep_df, cam_clips, camera_keys,
                replan_interval=args.replan_interval,
            )
            n_act = policy.config.n_action_steps
            ep_tag = f"ep{ep_id}: {lang[:80]}"

            if args.replan_interval > 0:
                pred_path = output_dir / f"ep{ep_id:05d}_action_compare.png"
                plot_action_compare_multichunk(gt_actions, pred_chunks, pred_path, title=ep_tag, replan_interval=args.replan_interval)
                cum_path = output_dir / f"ep{ep_id:05d}_action_cumsum.png"
                plot_cumulative_multichunk(gt_actions, pred_chunks, cum_path, title=ep_tag, replan_interval=args.replan_interval)
            else:
                pred_path = output_dir / f"ep{ep_id:05d}_action_compare.png"
                plot_action_comparison(gt_actions, pred_actions, pred_path, title=ep_tag)
                cum_path = output_dir / f"ep{ep_id:05d}_action_cumsum.png"
                plot_cumulative_trajectory(gt_actions, pred_actions, cum_path, title=ep_tag)

            mse_path = output_dir / f"ep{ep_id:05d}_replanning_mse.png"
            eff_replan = args.replan_interval if args.replan_interval > 0 else n_act
            mse_data = plot_replanning_mse(gt_actions, pred_actions, n_act, mse_path, title=ep_tag, pred_chunks=pred_chunks, mse_window=args.mse_window, mse_smooth_window=args.mse_smooth_window, replan_interval=eff_replan, smooth_method=args.mse_smooth_method, savgol_polyorder=args.savgol_polyorder, peak_nms=args.peak_nms)
            np.save(str(output_dir / f"ep{ep_id:05d}_pred_actions.npy"), pred_actions)
            print(f"    Action compare:   {pred_path.name}")
            print(f"    Action cumsum:    {cum_path.name}")
            print(f"    Replanning MSE:   {mse_path.name}")

        results.append({
            "episode_id": ep_id,
            "language": lang,
            "video_paths": [str(p) for p in cam_clips],
            "combined_path": str(combined_path) if combined_path else None,
            "traj_path": str(traj_path),
            "pred_path": str(pred_path) if pred_path else None,
            "cum_path": str(cum_path) if pred_path else None,
            "mse_path": str(mse_path) if pred_path else None,
            "mse_data": (mse_data[0], mse_data[1], {k: v.tolist() for k, v in mse_data[2].items()}, mse_data[3]) if pred_path and mse_data else None,
            "sg_peak_ts": mse_data[4] if pred_path and mse_data else [],
            "n_frames": len(ep_df),
            "slider_video": str(combined_path) if combined_path else (str(cam_clips[0]) if cam_clips else None),
        })

    # Save results
    results_path = output_dir / "replay_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. Results saved to {results_path}")

    # Log to wandb
    if args.wandb_project:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=f"task_id_{args.task_id}_ep_{args.n_episodes}")
            wandb.define_metric("timestep")
            wandb.define_metric("slider/*", step_metric="timestep")
            for r in results:
                label = f"ep{r['episode_id']:05d}: {r['language'][:60]}"
                log_dict = {}
                vp = r.get("combined_path") or (r["video_paths"][0] if r["video_paths"] else None)
                if vp and Path(vp).exists():
                    log_dict[f"replay/{label}"] = wandb.Video(vp, fps=args.fps)
                tp = r.get("traj_path")
                if tp and Path(tp).exists():
                    log_dict[f"eef_traj/{label}"] = wandb.Image(tp)
                pp = r.get("pred_path")
                if pp and Path(pp).exists():
                    log_dict[f"action_compare/{label}"] = wandb.Image(pp)
                cp = r.get("cum_path")
                if cp and Path(cp).exists():
                    log_dict[f"action_cumsum/{label}"] = wandb.Image(cp)
                mp = r.get("mse_path")
                if mp and Path(mp).exists():
                    log_dict[f"replanning_mse/{label}"] = wandb.Image(mp)
                if log_dict:
                    wandb.log(log_dict)

                # Interactive step slider: combined image (video + MSE) per timestep
                mse_data = r.get("mse_data")
                sv = r.get("slider_video")
                if mse_data is not None and sv and Path(sv).exists():
                    replan_ts, mse_vals, smoothed_dict_raw, bar_width = mse_data
                    smoothed_dict = {k: np.array(v) for k, v in smoothed_dict_raw.items()}
                    T_ep = max(replan_ts) + 1 if replan_ts else 1
                    print(f"    Uploading step slider frames for {label} ...")
                    reader = imageio.get_reader(str(sv))
                    for t, frame in enumerate(reader):
                        combined_img = render_combined_frame(
                            frame, replan_ts, mse_vals, bar_width, T_ep, t,
                            title=label, smoothed_dict=smoothed_dict,
                            replan_interval=args.replan_interval,
                            peak_nms=args.peak_nms,
                        )
                        wandb.log({
                            "timestep": t,
                            f"slider/{label}": wandb.Image(combined_img),
                        })
                    reader.close()

                # Skill segmentation: cut video at SG peak boundaries
                sg_peak_ts = r.get("sg_peak_ts") or []
                n_frames = r.get("n_frames")
                if sv and Path(sv).exists() and n_frames:
                    boundaries = sorted(set([0] + [int(p) for p in sg_peak_ts] + [n_frames]))
                    if len(boundaries) >= 2:
                        skill_paths = cut_skill_videos(
                            Path(sv), boundaries, output_dir, r["episode_id"], args.fps
                        )
                        skill_log = {}
                        for i, sp in enumerate(skill_paths):
                            if sp.exists():
                                skill_log[f"skills/{label}/skill_{i + 1}"] = wandb.Video(str(sp), fps=args.fps)
                        if skill_log:
                            wandb.log(skill_log)
                            print(f"    Uploaded {len(skill_log)} skill video(s) for {label}")

            wandb.finish()
            print("Logged to wandb.")
        except Exception as e:
            print(f"wandb logging failed: {e}")


if __name__ == "__main__":
    main(tyro.cli(Args))
