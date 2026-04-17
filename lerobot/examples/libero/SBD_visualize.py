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

    def _overlay_timestep(self, frame: np.ndarray, t: int, ep_start: int = 0,
                          ep_id: int | None = None, skill_idx: int | None = None) -> np.ndarray:
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        parts = []
        if ep_id is not None:
            parts.append(f"ep{ep_id:05d}")
        if skill_idx is not None:
            parts.append(f"skill {skill_idx}")
        parts.append(f"t={t - ep_start:04d}  (frame {t})")
        text = " | ".join(parts)
        draw.text((9, 9), text, fill=(0, 0, 0))
        draw.text((8, 8), text, fill=(255, 255, 255))
        return np.array(img)

    def load_episode_frames(self, src_video: Path, start_sec: float, end_sec: float) -> np.ndarray:
        """소스 영상에서 에피소드 구간만 잘라 (T, H, W, C) uint8 배열로 반환.
        imageio_ffmpeg 번들 바이너리로 fast seek — start_sec 이전 프레임 디코딩 없음.
        """
        import imageio_ffmpeg

        duration = end_sec - start_sec
        gen = imageio_ffmpeg.read_frames(
            str(src_video),
            input_params=["-ss", str(start_sec)],
            output_params=["-t", str(duration), "-pix_fmt", "rgb24"],
        )
        meta = next(gen)
        w, h = meta["size"]
        frame_size = h * w * 3

        raw = b"".join(gen)
        n_frames = len(raw) // frame_size
        return np.frombuffer(raw, dtype=np.uint8).reshape(n_frames, h, w, 3).copy()

    def write_single_video(self, frames: np.ndarray, dst: Path, ep_id: int | None = None) -> None:
        """단일 카메라 프레임을 mp4로 저장. frames: (T, H, W, C)"""
        writer = imageio.get_writer(str(dst), fps=self.fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1)
        for i, frame in enumerate(frames):
            writer.append_data(self._overlay_timestep(frame, i, ep_id=ep_id))
        writer.close()

    def write_combined_video(self, frames_l: np.ndarray, frames_r: np.ndarray, dst: Path) -> None:
        """두 카메라 프레임을 좌우로 합쳐 mp4로 저장. frames: (T, H, W, C)"""
        writer = imageio.get_writer(str(dst), fps=self.fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1)
        for i, (fl, fr) in enumerate(zip(frames_l, frames_r)):
            writer.append_data(self._overlay_timestep(np.concatenate([fl, fr], axis=1), i))
        writer.close()

    def cut_skill_videos(self, frames: np.ndarray, boundaries: list[int], ep_id: int) -> list[Path]:
        """numpy 프레임 배열을 boundary에서 잘라 스킬별 mp4 저장. frames: (T, H, W, C)"""
        skill_paths = []
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            skill_path = self.output_dir / f"ep{ep_id:05d}_skill{i + 1}.mp4"
            writer = imageio.get_writer(str(skill_path), fps=self.fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1)
            for t, frame in enumerate(frames[start:min(end, len(frames))]):
                writer.append_data(self._overlay_timestep(frame, start + t, ep_id=ep_id, skill_idx=i + 1))
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
        smoothed_dict: dict, div_cos_peak_ts: list, mse_smooth_window: int,
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
        if div_cos_peak_ts and sg_vals is not None:
            ts_to_idx = {ts: i for i, ts in enumerate(replan_ts)}
            pk_ts_valid = [ts for ts in div_cos_peak_ts if ts in ts_to_idx]
            pk_vals = [float(sg_vals[ts_to_idx[ts]]) for ts in pk_ts_valid]
            if pk_ts_valid:
                ax.scatter(pk_ts_valid, pk_vals, color="red", zorder=5, s=40, label="SG peaks > mean")
        if smoothed_dict or div_cos_peak_ts:
            ax.legend(fontsize=8, loc="upper right")
        ax.set_xlabel("frame")
        ax.set_ylabel("MSE  xyz")
        ax.grid(True, alpha=0.3, axis="y")

    def plot_mse(
        self,
        replan_ts: list, mse_vals: list, bar_width: float,
        smoothed_dict: dict, div_cos_peak_ts: list,
        ep_id: int, title: str = "", mse_smooth_window: int = 1,
    ) -> Path:
        fig, ax = plt.subplots(figsize=(10, 4))
        self._draw_mse_axes(ax, replan_ts, mse_vals, bar_width, smoothed_dict, div_cos_peak_ts, mse_smooth_window)
        ax.set_xticks(replan_ts)
        ax.set_xticklabels([str(t) for t in replan_ts], rotation=45, ha="right", fontsize=7)
        if title:
            ax.set_title(f"[Replanning MSE] {title}", fontsize=9)
        fig.tight_layout()
        save_path = self.output_dir / f"ep{ep_id:05d}_replanning_mse.png"
        fig.savefig(str(save_path), dpi=120)
        plt.close(fig)
        return save_path

    def _draw_cos_div_axes(
        self, ax, vf_replan_ts: list, div_cos_vals: list,
        div_cos_peak_ts: list, smooth_window: int, polyorder: int = 3,
    ) -> None:
        """Cos divergence bar+SG smooth+peaks for slider frames."""
        from scipy.signal import savgol_filter
        vals = np.array(div_cos_vals, dtype=float)
        bar_width = (vf_replan_ts[1] - vf_replan_ts[0]) * 0.8 if len(vf_replan_ts) > 1 else 4
        ax.bar(vf_replan_ts, vals, width=bar_width, align="center", alpha=0.4, color="tab:red", label="raw")
        # Same window logic as _savgol_smooth in replay_demo.py
        if smooth_window > 1:
            win = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
            win = min(win, len(vals))
            if win % 2 == 0:
                win -= 1
            if smooth_window > 1 and win >= polyorder + 2:
                sg_vals = savgol_filter(vals, window_length=win, polyorder=polyorder)
            else:
                sg_vals = vals
        else:
            sg_vals = vals
        mean_val = float(np.mean(sg_vals))
        ax.plot(vf_replan_ts, sg_vals, color="tab:orange", linewidth=2, label=f"SG (w={smooth_window})")
        ax.axhline(mean_val, color="tab:orange", linewidth=1.5, linestyle="--", label=f"mean={mean_val:.4f}")
        if div_cos_peak_ts:
            ts_to_idx = {ts: i for i, ts in enumerate(vf_replan_ts)}
            pk_ts_valid = [ts for ts in div_cos_peak_ts if ts in ts_to_idx]
            pk_vals = [float(sg_vals[ts_to_idx[ts]]) for ts in pk_ts_valid]
            if pk_ts_valid:
                ax.scatter(pk_ts_valid, pk_vals, color="red", zorder=5, s=40, label="peaks > mean")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlabel("frame")
        ax.set_ylabel("cos divergence")
        ax.grid(True, alpha=0.3, axis="y")

    def render_combined_frame(
        self,
        video_frame: np.ndarray,
        replan_ts: list, mse_vals: list, bar_width: float,
        T: int, t: int,
        title: str = "", smoothed_dict: dict | None = None,
        div_cos_peak_ts: list | None = None, mse_smooth_window: int = 1,
        cos_div_data: tuple | None = None, savgol_polyorder: int = 3,
    ) -> np.ndarray:
        """Render video frame (top) + cos div chart (bottom) with current-timestep marker."""
        fig = plt.figure(figsize=(10, 7))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.35)

        ax_img = fig.add_subplot(gs[0])
        ax_img.imshow(video_frame)
        ax_img.axis("off")
        ax_img.set_title(f"t={t:04d}  {title[:60]}", fontsize=9)

        ax_bot = fig.add_subplot(gs[1])
        if cos_div_data is not None:
            vf_replan_ts, div_cos_vals = cos_div_data
            self._draw_cos_div_axes(ax_bot, vf_replan_ts, div_cos_vals, div_cos_peak_ts or [], mse_smooth_window, polyorder=savgol_polyorder)
            ax_bot.axvline(x=t, color="blue", linewidth=2)
            ax_bot.set_xticks(vf_replan_ts)
            ax_bot.set_xticklabels([str(x) for x in vf_replan_ts], rotation=45, ha="right", fontsize=7)
            ax_bot.set_xlim(-4, T)
        else:
            self._draw_mse_axes(ax_bot, replan_ts, mse_vals, bar_width,
                                smoothed_dict or {}, div_cos_peak_ts or [], mse_smooth_window)
            ax_bot.axvline(x=t, color="red", linewidth=2)
            ax_bot.set_xticks(replan_ts)
            ax_bot.set_xticklabels([str(x) for x in replan_ts], rotation=45, ha="right", fontsize=7)
            ax_bot.set_xlim(-bar_width, T)

        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plt.close(fig)
        return img.copy()

    # ── WandB logging ──────────────────────────────────────────────────────────

    def log_to_wandb(
        self, results: list[dict], wandb_project: str, run_name: str,
        replan_interval: int, mse_smooth_window: int = 1, savgol_polyorder: int = 3,
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
                ("vf_plot_path", "vf_error"),
                ("vf_div_plot_path", "vf_divergence"),
                ("vf_l2_div_plot_path", "vf_l2_divergence"),
            ]:
                p = r.get(path_key)
                if p and Path(p).exists():
                    log_dict[f"{wandb_prefix}/{label}"] = wandb.Image(p)

            # boundary_criteria: replanning_mse + cos divergence + L2 divergence stacked vertically
            mse_p = r.get("mse_path")
            div_p = r.get("vf_div_plot_path")
            l2_p = r.get("vf_l2_div_plot_path")
            imgs_to_stack = [
                np.array(Image.open(p).convert("RGB"))
                for p in [mse_p, div_p, l2_p]
                if p and Path(p).exists()
            ]
            if len(imgs_to_stack) >= 2:
                def _pad_width(img, target_w):
                    if img.shape[1] < target_w:
                        pad = np.full((img.shape[0], target_w - img.shape[1], 3), 255, dtype=np.uint8)
                        img = np.concatenate([img, pad], axis=1)
                    return img
                w = max(img.shape[1] for img in imgs_to_stack)
                imgs_to_stack = [_pad_width(img, w) for img in imgs_to_stack]
                combined = np.concatenate(imgs_to_stack, axis=0)
                log_dict[f"boundary_criteria/{label}"] = wandb.Image(combined)

            if log_dict:
                wandb.log(log_dict)

            # GMM 3D interactive HTML — inline panel + Artifact download link
            gmm_p = r.get("gmm_3d_path")
            if gmm_p and Path(gmm_p).exists():
                html_content = Path(gmm_p).read_text()
                wandb.log({f"gmm_3d/{label}": wandb.Html(html_content)})
                artifact = wandb.Artifact(
                    name=f"gmm_3d_ep{r['episode_id']:05d}",
                    type="html",
                    description=f"GMM 3D interactive plot for {label}",
                )
                artifact.add_file(gmm_p)
                wandb.log_artifact(artifact)
                print(f"    Uploaded GMM 3D: inline + artifact ep{r['episode_id']:05d}")

            # Per-timestep slider (video frame + cos div chart)
            mse_data = r.get("mse_data")
            sv = r.get("slider_video")
            div_cos_peak_ts = r.get("div_cos_peak_ts") or []
            cos_div_data = r.get("cos_div_data")  # (vf_replan_ts, div_cos_vals) or None
            # Run slider if cos_div_data is available (preferred); fallback to mse_data
            if sv and Path(sv).exists() and (cos_div_data is not None or mse_data is not None):
                if mse_data is not None:
                    replan_ts, mse_vals, smoothed_dict_raw, bar_width = mse_data
                    smoothed_dict = {k: np.array(v) for k, v in smoothed_dict_raw.items()}
                    T_ep = max(replan_ts) + 1 if replan_ts else 1
                else:
                    replan_ts, mse_vals, smoothed_dict, bar_width = [], [], {}, 1
                    T_ep = r.get("n_frames") or 1
                print(f"    Uploading step slider for {label} ...")
                reader = imageio.get_reader(str(sv))
                for t, frame in enumerate(reader):
                    combined_img = self.render_combined_frame(
                        frame, replan_ts, mse_vals, bar_width, T_ep, t,
                        title=label, smoothed_dict=smoothed_dict,
                        div_cos_peak_ts=div_cos_peak_ts, mse_smooth_window=mse_smooth_window,
                        cos_div_data=cos_div_data, savgol_polyorder=savgol_polyorder,
                    )
                    wandb.log({"timestep": t, f"slider/{label}": wandb.Image(combined_img)})
                reader.close()

            # Skill segmentation: cut at SG peak boundaries and upload
            n_frames = r.get("n_frames")
            if sv and Path(sv).exists() and n_frames:
                boundaries = sorted(set([0] + [int(p) for p in div_cos_peak_ts] + [n_frames]))
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
