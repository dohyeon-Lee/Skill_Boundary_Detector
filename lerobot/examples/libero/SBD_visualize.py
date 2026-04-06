"""
SkillVisualizer — visualization and wandb logging for Skill Boundary Detector.

All matplotlib plots, video extraction/stacking/cutting, and wandb upload live here.
The core algorithm (MSE computation, smoothing, peak detection) lives in replay_demo.py.
"""

from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

_SMOOTH_COLORS = {"ma": "tab:orange", "savgol": "tab:green"}
_SMOOTH_LABELS = {"ma": "Moving Avg", "savgol": "Savitzky-Golay"}
_SMOOTH_MEAN_LINESTYLES = {"ma": (0, (5, 3)), "savgol": (0, (2, 2))}  # long-dash vs short-dash


class SkillVisualizer:
    def __init__(self, output_dir: Path, fps: int = 20):
        self.output_dir = output_dir
        self.fps = fps

    # ── Video helpers ──────────────────────────────────────────────────────────

    def _overlay_timestep(self, frame: np.ndarray, t: int, ep_start: int = 0) -> np.ndarray:
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        text = f"t={t - ep_start:04d}  (frame {t})"
        draw.text((9, 9), text, fill=(0, 0, 0))
        draw.text((8, 8), text, fill=(255, 255, 255))
        return np.array(img)

    def extract_clip(self, src_video: Path, dst_video: Path, start_sec: float, end_sec: float) -> None:
        reader = imageio.get_reader(str(src_video))
        src_fps = reader.get_meta_data().get("fps", self.fps)
        start_frame, end_frame = int(start_sec * src_fps), int(end_sec * src_fps)
        writer = imageio.get_writer(str(dst_video), fps=self.fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1)
        ep_t = 0
        for i, frame in enumerate(reader):
            if i < start_frame:
                continue
            if i >= end_frame:
                break
            writer.append_data(self._overlay_timestep(frame, ep_t))
            ep_t += 1
        writer.close()
        reader.close()

    def stack_clips_side_by_side(self, left: Path, right: Path, dst: Path) -> None:
        reader_l = imageio.get_reader(str(left))
        reader_r = imageio.get_reader(str(right))
        writer = imageio.get_writer(str(dst), fps=self.fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1)
        for i, (fl, fr) in enumerate(zip(reader_l, reader_r)):
            combined = np.concatenate([fl, fr], axis=1)
            writer.append_data(self._overlay_timestep(combined, i))
        writer.close()
        reader_l.close()
        reader_r.close()

    def cut_skill_videos(self, src_video: Path, boundaries: list[int], ep_id: int) -> list[Path]:
        """Cut src_video at frame boundaries; return list of per-skill video paths."""
        reader = imageio.get_reader(str(src_video))
        frames = [f for f in reader]
        reader.close()
        skill_paths = []
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            skill_path = self.output_dir / f"ep{ep_id:05d}_skill{i + 1}.mp4"
            writer = imageio.get_writer(str(skill_path), fps=self.fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1)
            for frame in frames[start:min(end, len(frames))]:
                writer.append_data(frame)
            writer.close()
            skill_paths.append(skill_path)
        return skill_paths

    # ── Trajectory / action plots ──────────────────────────────────────────────

    def plot_eef_trajectory(self, ep_df: pd.DataFrame, ep_id: int, title: str = "") -> Path:
        states = np.stack(ep_df["observation.state"].values)
        t = ep_df["timestamp"].values
        xyz, labels, colors = states[:, :3], ["x", "y", "z"], ["tab:red", "tab:green", "tab:blue"]

        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        for ax, label, color in zip(axes, labels, colors):
            ax.plot(t, xyz[:, labels.index(label)], color=color)
            ax.set_ylabel(f"{label} (m)")
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("time (s)")
        if title:
            axes[0].set_title(title, fontsize=9)
        fig.tight_layout()
        save_path = self.output_dir / f"ep{ep_id:05d}_eef_traj.png"
        fig.savefig(str(save_path), dpi=120)
        plt.close(fig)
        return save_path

    def plot_action_comparison(self, gt: np.ndarray, pred: np.ndarray, ep_id: int, title: str = "") -> Path:
        T, labels = len(gt), ["x", "y", "z"]
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        for i, (ax, label) in enumerate(zip(axes, labels)):
            ax.plot(np.arange(T), gt[:, i], label="GT", color="tab:blue", linewidth=1.2)
            ax.plot(np.arange(T), pred[:, i], label="Pred", color="tab:orange", linewidth=1.2, linestyle="--")
            ax.set_ylabel(f"action {label}")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("frame")
        if title:
            axes[0].set_title(title, fontsize=9)
        fig.tight_layout()
        save_path = self.output_dir / f"ep{ep_id:05d}_action_compare.png"
        fig.savefig(str(save_path), dpi=120)
        plt.close(fig)
        return save_path

    def plot_action_compare_multichunk(self, gt: np.ndarray, chunks: np.ndarray, ep_id: int, title: str = "", replan_interval: int = 1) -> Path:
        n_chunks, n_steps, _ = chunks.shape
        T_gt, labels = len(gt), ["x", "y", "z"]
        replan_ts = [i * replan_interval for i in range(n_chunks)]

        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        for i, (ax, label) in enumerate(zip(axes, labels)):
            ax.plot(np.arange(T_gt), gt[:, i], label="GT", color="tab:blue", linewidth=1.5, zorder=5)
            for ci, t in enumerate(replan_ts):
                end = min(t + n_steps, T_gt)
                ax.plot(np.arange(t, end), chunks[ci, : end - t, i], color="tab:orange", alpha=0.4, linewidth=1.0)
            ax.set_ylabel(f"action {label}")
            ax.grid(True, alpha=0.3)
        axes[0].legend(["GT", "Pred chunks"], loc="upper right", fontsize=8)
        axes[-1].set_xlabel("frame")
        if title:
            axes[0].set_title(f"[Multi-chunk] {title}", fontsize=9)
        fig.tight_layout()
        save_path = self.output_dir / f"ep{ep_id:05d}_action_compare.png"
        fig.savefig(str(save_path), dpi=120)
        plt.close(fig)
        return save_path

    def plot_cumulative_trajectory(self, gt: np.ndarray, pred: np.ndarray, ep_id: int, title: str = "") -> Path:
        labels = ["x", "y", "z"]
        t = np.arange(len(gt))
        gt_cum, pred_cum = np.cumsum(gt[:, :3], axis=0), np.cumsum(pred[:, :3], axis=0)

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
        save_path = self.output_dir / f"ep{ep_id:05d}_action_cumsum.png"
        fig.savefig(str(save_path), dpi=120)
        plt.close(fig)
        return save_path

    def plot_cumulative_multichunk(self, gt: np.ndarray, chunks: np.ndarray, ep_id: int, title: str = "", replan_interval: int = 1) -> Path:
        n_chunks, n_steps, _ = chunks.shape
        T_gt, labels = len(gt), ["x", "y", "z"]
        gt_cum = np.cumsum(gt[:, :3], axis=0)
        replan_ts = [i * replan_interval for i in range(n_chunks)]

        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        for i, (ax, label) in enumerate(zip(axes, labels)):
            ax.plot(np.arange(T_gt), gt_cum[:, i], label="GT (cumsum)", color="tab:blue", linewidth=1.5, zorder=5)
            for ci, t in enumerate(replan_ts):
                end = min(t + n_steps, T_gt)
                length = end - t
                offset = gt_cum[t - 1, i] if t > 0 else 0.0
                pred_cum = offset + np.cumsum(chunks[ci, :length, i])
                ax.plot(np.arange(t, end), pred_cum, color="tab:orange", alpha=0.4, linewidth=1.0)
            ax.set_ylabel(f"Σ Δ{label}")
            ax.grid(True, alpha=0.3)
        axes[0].legend(["GT (cumsum)", "Pred chunks"], loc="upper right", fontsize=8)
        axes[-1].set_xlabel("frame")
        if title:
            axes[0].set_title(f"[Multi-chunk cumsum] {title}", fontsize=9)
        fig.tight_layout()
        save_path = self.output_dir / f"ep{ep_id:05d}_action_cumsum.png"
        fig.savefig(str(save_path), dpi=120)
        plt.close(fig)
        return save_path

    # ── MSE plot ───────────────────────────────────────────────────────────────

    def _draw_mse_axes(
        self, ax, replan_ts: list, mse_vals: list, bar_width: float,
        smoothed_dict: dict, sg_peak_ts: list, mse_smooth_window: int,
    ) -> None:
        """Shared MSE bar+smooth+peaks drawing used by both static plot and slider frames."""
        ax.bar(replan_ts, mse_vals, width=bar_width, align="center", alpha=0.4, color="tab:purple", label="raw")
        sg_vals = smoothed_dict.get("savgol")
        for method, vals in smoothed_dict.items():
            color = _SMOOTH_COLORS.get(method, "gray")
            mean_val = float(np.mean(vals))
            ax.plot(replan_ts, vals, color=color, linewidth=2,
                    label=f"{_SMOOTH_LABELS.get(method, method)} (w={mse_smooth_window})")
            ax.axhline(mean_val, color=color, linewidth=1.5,
                       linestyle=_SMOOTH_MEAN_LINESTYLES.get(method, "--"),
                       label=f"{_SMOOTH_LABELS.get(method, method)} mean={mean_val:.4f}")
        if sg_peak_ts and sg_vals is not None:
            ts_to_idx = {ts: i for i, ts in enumerate(replan_ts)}
            pk_ts_valid = [ts for ts in sg_peak_ts if ts in ts_to_idx]
            pk_vals = [float(sg_vals[ts_to_idx[ts]]) for ts in pk_ts_valid]
            if pk_ts_valid:
                ax.scatter(pk_ts_valid, pk_vals, color="red", zorder=5, s=40, label="SG peaks > mean")
        if smoothed_dict or sg_peak_ts:
            ax.legend(fontsize=8, loc="upper right")
        ax.set_xlabel("frame")
        ax.set_ylabel("MSE  xyz")
        ax.grid(True, alpha=0.3, axis="y")

    def plot_mse(
        self,
        replan_ts: list, mse_vals: list, bar_width: float,
        smoothed_dict: dict, sg_peak_ts: list,
        ep_id: int, title: str = "", mse_smooth_window: int = 1,
    ) -> Path:
        fig, ax = plt.subplots(figsize=(10, 4))
        self._draw_mse_axes(ax, replan_ts, mse_vals, bar_width, smoothed_dict, sg_peak_ts, mse_smooth_window)
        ax.set_xticks(replan_ts)
        ax.set_xticklabels([str(t) for t in replan_ts], rotation=45, ha="right", fontsize=7)
        if title:
            ax.set_title(f"[Replanning MSE] {title}", fontsize=9)
        fig.tight_layout()
        save_path = self.output_dir / f"ep{ep_id:05d}_replanning_mse.png"
        fig.savefig(str(save_path), dpi=120)
        plt.close(fig)
        return save_path

    def render_combined_frame(
        self,
        video_frame: np.ndarray,
        replan_ts: list, mse_vals: list, bar_width: float,
        T: int, t: int,
        title: str = "", smoothed_dict: dict | None = None,
        sg_peak_ts: list | None = None, mse_smooth_window: int = 1,
    ) -> np.ndarray:
        """Render video frame (top) + MSE chart with current-timestep marker (bottom)."""
        fig = plt.figure(figsize=(10, 7))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.35)

        ax_img = fig.add_subplot(gs[0])
        ax_img.imshow(video_frame)
        ax_img.axis("off")
        ax_img.set_title(f"t={t:04d}  {title[:60]}", fontsize=9)

        ax_mse = fig.add_subplot(gs[1])
        self._draw_mse_axes(
            ax_mse, replan_ts, mse_vals, bar_width,
            smoothed_dict or {}, sg_peak_ts or [], mse_smooth_window,
        )
        ax_mse.axvline(x=t, color="red", linewidth=2)
        ax_mse.set_xticks(replan_ts)
        ax_mse.set_xticklabels([str(x) for x in replan_ts], rotation=45, ha="right", fontsize=7)
        ax_mse.set_xlim(-bar_width, T)

        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plt.close(fig)
        return img.copy()

    # ── WandB logging ──────────────────────────────────────────────────────────

    def log_to_wandb(
        self, results: list[dict], wandb_project: str, run_name: str,
        replan_interval: int, mse_smooth_window: int = 1,
    ) -> None:
        import wandb
        wandb.init(project=wandb_project, name=run_name)
        wandb.define_metric("timestep")
        wandb.define_metric("slider/*", step_metric="timestep")

        for r in results:
            label = f"ep{r['episode_id']:05d}: {r['language'][:60]}"
            log_dict = {}

            # Static images and replay video
            vp = r.get("combined_path") or (r["video_paths"][0] if r["video_paths"] else None)
            if vp and Path(vp).exists():
                log_dict[f"replay/{label}"] = wandb.Video(vp, fps=self.fps)
            for path_key, wandb_prefix in [
                ("traj_path", "eef_traj"),
                ("pred_path", "action_compare"),
                ("cum_path", "action_cumsum"),
                ("mse_path", "replanning_mse"),
            ]:
                p = r.get(path_key)
                if p and Path(p).exists():
                    log_dict[f"{wandb_prefix}/{label}"] = wandb.Image(p)
            if log_dict:
                wandb.log(log_dict)

            # Per-timestep slider (video frame + MSE chart)
            mse_data = r.get("mse_data")
            sv = r.get("slider_video")
            sg_peak_ts = r.get("sg_peak_ts") or []
            if mse_data is not None and sv and Path(sv).exists():
                replan_ts, mse_vals, smoothed_dict_raw, bar_width = mse_data
                smoothed_dict = {k: np.array(v) for k, v in smoothed_dict_raw.items()}
                T_ep = max(replan_ts) + 1 if replan_ts else 1
                print(f"    Uploading step slider for {label} ...")
                reader = imageio.get_reader(str(sv))
                for t, frame in enumerate(reader):
                    combined_img = self.render_combined_frame(
                        frame, replan_ts, mse_vals, bar_width, T_ep, t,
                        title=label, smoothed_dict=smoothed_dict,
                        sg_peak_ts=sg_peak_ts, mse_smooth_window=mse_smooth_window,
                    )
                    wandb.log({"timestep": t, f"slider/{label}": wandb.Image(combined_img)})
                reader.close()

            # Skill segmentation: cut at SG peak boundaries and upload
            n_frames = r.get("n_frames")
            if sv and Path(sv).exists() and n_frames:
                boundaries = sorted(set([0] + [int(p) for p in sg_peak_ts] + [n_frames]))
                if len(boundaries) >= 2:
                    skill_paths = self.cut_skill_videos(Path(sv), boundaries, r["episode_id"])
                    skill_log = {
                        f"skills/{label}/skill_{i + 1}": wandb.Video(str(sp), fps=self.fps)
                        for i, sp in enumerate(skill_paths) if sp.exists()
                    }
                    if skill_log:
                        wandb.log(skill_log)
                        print(f"    Uploaded {len(skill_log)} skill video(s) for {label}")

        wandb.finish()
        print("Logged to wandb.")
