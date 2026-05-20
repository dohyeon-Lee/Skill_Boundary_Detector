"""
Skill Divider — skill boundary detection via VF cosine divergence.

Pipeline:
  1. Load demonstration episodes for a given task.
  2. Run VF divergence analysis using spherical action probes at a fixed denoising step.
  3. Fit GMM on probe VF outputs → compute pairwise cosine divergence per replanning step.
  4. Apply Savitzky-Golay smoothing → detect peaks above mean → skill boundaries.
  5. Output: cos divergence plot, full demo video, per-skill videos, GMM 3D HTML.

Usage:
  python examples/libero/skill_divider.py \
    --dataset_dir /scratch/.../libero_dataset/libero_90 \
    --task_id 0 --n_episodes 5 \
    --policy_path .../checkpoints/080000/pretrained_model \
    --output_dir .../outputs/skill_divider/task0 \
    --replan_interval 5 \
    --eval_at_step 8 \
    --wandb_project SBD_skill
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro

sys.path.insert(0, str(Path(__file__).parent))
from SBD_visualize import SkillVisualizer


# ── Args ──────────────────────────────────────────────────────────────────────

@dataclass
class Args:
    dataset_dir: str = "/data2/dohyeon/SBD/libero_dataset/libero_90"
    task_id: int = 0
    """Task index (0-based)."""
    n_episodes: int = 5
    """Number of episodes to process for the task."""
    output_dir: str = "/data2/dohyeon/SBD/outputs/skill_divider"
    policy_path: str = ""
    """Path to pretrained_model dir (e.g. .../checkpoints/080000/pretrained_model)."""
    device: str = "cuda"
    replan_interval: int = 5
    """Replanning interval in frames."""
    seed: int | None = 42
    # ── Diffusion scheduler ──────────────────────────────────────────────────
    noise_scheduler_type: str = "DDIM"
    num_inference_steps: int = 10
    eval_at_step: int = 8
    """Denoising step index (0-based) at which to query the VF error."""
    # ── GMM / divergence ─────────────────────────────────────────────────────
    n_gmm_components: int = 7
    """Number of GMM clusters for VF divergence computation."""
    # ── SG filter / peak detection ───────────────────────────────────────────
    smooth_window: int = 6
    """Savitzky-Golay window length for cos divergence smoothing."""
    savgol_polyorder: int = 4
    """Polynomial order for Savitzky-Golay filter."""
    peak_nms: bool = True
    """Non-max suppression: suppress smaller peaks within replan_interval*2 frames."""
    nms_dist: int | None = None
    """NMS distance in frames. Overrides the default (replan_interval*2) when set."""
    # ── Output / wandb ───────────────────────────────────────────────────────
    plot_gmm_3d: bool = False
    """Save interactive 3D GMM HTML per episode (slower)."""
    plot_pred_mse: bool = False
    """Overlay pred-vs-demo action xyz MSE (SG-filtered) on the boundary criteria plot. Adds full denoising per replan step (global_cond already computed, no extra CNN call)."""
    mse_window: int = 4
    """Number of action steps used to compute pred-demo MSE (first N steps of each chunk)."""
    fps: int = 20
    dino_feature_dir: str = ""
    """Per-episode DINO feature dir (e.g. .../libero_90_DINO/dinov3_vits16_pg8). Required when policy uses DINO features."""
    wandb_project: str | None = None
    wandb_run_name: str | None = None


# ── Spherical probe generation ────────────────────────────────────────────────

def _normalize(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return None, n
    return v / n, n


def _make_local_basis(pole_vec):
    """Returns (e1, e2, e3) orthonormal basis where e3 = normalized pole direction."""
    e3, _ = _normalize(pole_vec)
    ref = np.array([1.0, 0.0, 0.0]) if abs(e3[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = ref - np.dot(ref, e3) * e3
    e1, _ = _normalize(e1)
    e2 = np.cross(e3, e1)
    e2, _ = _normalize(e2)
    return e1, e2, e3


def _rodrigues_rotation(u, v, eps=1e-8):
    """Rotation matrix that rotates unit vector u → v."""
    cross = np.cross(u, v)
    dot = np.clip(np.dot(u, v), -1.0, 1.0)
    s = np.linalg.norm(cross)
    if s < eps and dot > 0:
        return np.eye(3)
    if s < eps and dot < 0:
        ref = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = ref - np.dot(ref, u) * u
        axis, _ = _normalize(axis)
        return -np.eye(3) + 2.0 * np.outer(axis, axis)
    K = np.array([[0.0, -cross[2], cross[1]],
                  [cross[2], 0.0, -cross[0]],
                  [-cross[1], cross[0], 0.0]])
    return np.eye(3) + K + K @ K * ((1.0 - dot) / (s ** 2))


def _generate_spherical_samples(gt_delta_xyz, polar_degs=(30, 60), azimuth_step_deg=30):
    """GT chunk + spherical probe chunks around the GT endpoint direction.
    Returns list of (H, 3) arrays — index 0 is GT, rest are probes.
    """
    gt_path = np.cumsum(gt_delta_xyz, axis=0)
    endpoint = gt_path[-1]
    u, _ = _normalize(endpoint, eps=0.0)
    if u is None:
        return [gt_delta_xyz.copy()]

    e1, e2, e3 = _make_local_basis(u)
    result = [gt_delta_xyz.copy()]

    for theta_deg in polar_degs:
        theta = np.deg2rad(theta_deg)
        for phi_deg in range(0, 360, azimuth_step_deg):
            phi = np.deg2rad(phi_deg)
            d_local = np.array([np.sin(theta) * np.cos(phi),
                                 np.sin(theta) * np.sin(phi),
                                 np.cos(theta)])
            d_world, _ = _normalize(d_local[0] * e1 + d_local[1] * e2 + d_local[2] * e3)
            R = _rodrigues_rotation(u, d_world)
            new_path = (R @ gt_path.T).T
            new_delta = np.zeros_like(new_path)
            new_delta[0] = new_path[0]
            new_delta[1:] = new_path[1:] - new_path[:-1]
            result.append(new_delta)

    return result


def _make_probe_chunks(gt_chunk_np, polar_degs=(30, 60), azimuth_step_deg=30):
    """Returns (N_samples, H, action_dim) — index 0 is GT, rest are probes."""
    xyz = gt_chunk_np[:, :3]
    rest = gt_chunk_np[:, 3:]
    xyz_samples = _generate_spherical_samples(xyz, polar_degs, azimuth_step_deg)
    return np.stack([np.concatenate([s, rest], axis=1) for s in xyz_samples], axis=0)


# ── VF query ──────────────────────────────────────────────────────────────────

def _query_vf_error(policy, global_cond, chunks_batch, eval_at_step: int):
    """One deterministic DDIM step at eval_at_step → (B, action_dim) total displacement."""
    import torch
    with torch.no_grad():
        scheduler = policy.diffusion.noise_scheduler
        t = scheduler.timesteps[eval_at_step]
        B = chunks_batch.shape[0]
        t_batch = torch.full((B,), t, dtype=torch.long, device=chunks_batch.device)
        gc = global_cond.expand(B, -1)
        eps_pred = policy.diffusion.unet(chunks_batch, t_batch, global_cond=gc)
        denoised = scheduler.step(eps_pred, t, chunks_batch).prev_sample
        return denoised.sum(dim=1).cpu().numpy()


# ── GMM divergence ────────────────────────────────────────────────────────────

def compute_vf_divergence(vf_batch, n_components=3):
    """Fit GMM on xyz VF outputs and compute pairwise cosine divergence over cluster means.

    Returns (div_cos, div_l2, means_full).
    """
    from sklearn.mixture import GaussianMixture

    n_components = min(n_components, len(vf_batch))
    data = vf_batch[:, :3].astype(np.float64)  # raw xyz, same as replay_demo.py

    fitted = False
    for reg_covar in [1e-6, 1e-4, 1e-2]:
        try:
            gmm = GaussianMixture(n_components=n_components, random_state=0,
                                  max_iter=200, reg_covar=reg_covar)
            gmm.fit(data)
            fitted = True
            break
        except (ValueError, np.linalg.LinAlgError):
            continue
    if not fitted:
        try:
            gmm = GaussianMixture(n_components=min(2, n_components), random_state=0,
                                  max_iter=200, reg_covar=1e-1)
            gmm.fit(data)
            n_components = gmm.n_components
        except Exception:
            return 0.0, 0.0, np.zeros((n_components, vf_batch.shape[1]))

    labels = gmm.predict(data)
    means_full = np.zeros((n_components, vf_batch.shape[1]))
    for k in range(n_components):
        mask = labels == k
        means_full[k] = vf_batch[mask].mean(axis=0) if mask.any() else 0.0
    means_xyz = means_full[:, :3]

    if n_components < 2:
        return 0.0, 0.0, means_full

    cos_sq, l2_sq, count = 0.0, 0.0, 0
    for i in range(n_components):
        for j in range(n_components):
            if i != j:
                g_i, g_j = means_xyz[i], means_xyz[j]
                denom = np.linalg.norm(g_i) * np.linalg.norm(g_j) + 1e-8
                cos_sq += (1.0 - np.dot(g_i, g_j) / denom) ** 2
                l2_sq += np.linalg.norm(g_i - g_j) ** 2
                count += 1

    return float(np.sqrt(cos_sq / count)), float(np.sqrt(l2_sq / count)), means_full


# ── VF analysis (per episode) ─────────────────────────────────────────────────

def run_vf_analysis(
    policy, preprocessor, ep_df, cam_frames: dict, camera_keys,
    eval_at_step: int, replan_interval: int,
    polar_degs=(30, 60), azimuth_step_deg=30,
    n_gmm_components=3,
    compute_pred_mse: bool = False,
    mse_window: int = 4,
    dino_tokens: np.ndarray | None = None,
) -> tuple:
    """Returns (replan_ts, vf_values, gt_orig, divergences).

    vf_values:   (N_replan, N_samples, action_dim) — index 0 is GT, rest probes.
    gt_orig:     (N_replan, action_dim) — original GT total displacement.
    divergences: (N_replan,) — GMM divergence scalar per replan step.

    cam_frames: {camera_key: (T, H, W, C) uint8 numpy array} — 호출 전에 미리 로드.
    select_action 대신 obs queue를 직접 관리해 replan 시점에만
    _prepare_global_conditioning (CNN encoder)을 호출한다.
    UNet diffusion은 전혀 실행하지 않으므로 대폭 빠르다.
    """
    import torch
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

    policy.diffusion.noise_scheduler.set_timesteps(policy.diffusion.num_inference_steps)

    n_obs_steps = policy.config.n_obs_steps
    horizon = policy.config.horizon
    eff_interval = replan_interval if replan_interval > 0 else policy.config.n_action_steps
    device = next(policy.parameters()).device
    image_keys = list(policy.config.image_features.keys())
    _use_dino = (dino_tokens is not None) and policy.config.use_dino_features
    _dino_key = policy.config.dino_token_key  # "observation.dino.tokens"

    states = np.stack(ep_df["observation.state"].values)

    T = min(len(ep_df), *(len(v) for v in cam_frames.values())) if cam_frames else len(ep_df)
    if _use_dino:
        T = min(T, len(dino_tokens))
    states = states[:T]
    gt_actions = np.stack(ep_df["action"].values[:T])
    for k in cam_frames:
        cam_frames[k] = cam_frames[k][:T]

    replan_ts, vf_values, gt_orig, div_cos_list, div_l2_list, gmm_means_list = [], [], [], [], [], []
    pred_mse_list = [] if compute_pred_mse else None
    action_dim = gt_actions.shape[1]

    def _build_obs_batch(t: int) -> dict:
        """replan step t에서 쓸 (1, n_obs_steps, ...) 배치 구성.
        populate_queues와 동일하게: 히스토리 부족 시 첫 프레임으로 패딩.
        """
        indices = list(range(max(0, t - n_obs_steps + 1), t + 1))
        pad = n_obs_steps - len(indices)
        indices = [indices[0]] * pad + indices  # 앞쪽 패딩

        obs_states = []
        obs_imgs = [] if not _use_dino else None
        for fi in indices:
            obs = {OBS_STATE: torch.from_numpy(states[fi]).float()}
            if not _use_dino:
                obs.update({k: torch.from_numpy(cam_frames[k][fi]).float().div(255.0).permute(2, 0, 1)
                             for k in camera_keys})
            obs = preprocessor(obs)
            obs_states.append(obs[OBS_STATE])
            if not _use_dino:
                obs_imgs.append(torch.stack([obs[k] for k in image_keys], dim=-4))

        batch = {OBS_STATE: torch.stack(obs_states, dim=1).to(device)}  # (1, n_obs, state_dim)
        if _use_dino:
            # (n_obs, n_tokens, feat_dim) → (1, n_obs, 1, n_tokens, feat_dim)
            tok = torch.from_numpy(
                np.stack([dino_tokens[i] for i in indices]).astype(np.float32)
            ).unsqueeze(0).unsqueeze(2).to(device)
            batch[_dino_key] = tok
        else:
            batch[OBS_IMAGES] = torch.stack(obs_imgs, dim=1).to(device)  # (1, n_obs, n_cam, C, H, W)
        return batch

    # replan step에서만 순회 (T회 → T/eff_interval회)
    for t in range(0, T, eff_interval):
        with torch.no_grad():
            batch = _build_obs_batch(t)
            global_cond = policy.diffusion._prepare_global_conditioning(batch)

            end = min(t + horizon, T)
            chunk = gt_actions[t:end]
            if len(chunk) < horizon:
                chunk = np.concatenate([chunk, np.tile(chunk[-1:], (horizon - len(chunk), 1))], axis=0)

            # Build GT + probe batch
            chunks_np = _make_probe_chunks(chunk, polar_degs, azimuth_step_deg)  # (N, H, action_dim)
            chunks_t = torch.from_numpy(chunks_np).float().to(device)            # (N, H, action_dim)
            vf_batch = _query_vf_error(policy, global_cond, chunks_t, eval_at_step)  # (N, action_dim)
            div_cos, div_l2, means = compute_vf_divergence(vf_batch, n_components=n_gmm_components)
            replan_ts.append(t)
            vf_values.append(vf_batch)
            gt_orig.append(chunk.sum(axis=0))
            div_cos_list.append(div_cos)
            div_l2_list.append(div_l2)
            gmm_means_list.append(means)  # (n_components, action_dim)

            # Full denoising for pred-demo MSE (global_cond already computed — no extra CNN call)
            if compute_pred_mse:
                noise = torch.randn(1, horizon, action_dim, device=device)
                x = noise
                scheduler = policy.diffusion.noise_scheduler
                for ts in scheduler.timesteps:
                    t_batch = torch.full((1,), ts.item(), dtype=torch.long, device=device)
                    eps = policy.diffusion.unet(x, t_batch, global_cond=global_cond)
                    x = scheduler.step(eps, ts, x).prev_sample
                pred_chunk = x[0].cpu().numpy()  # (H, action_dim)
                win = min(mse_window, end - t)
                mse = float(np.mean((pred_chunk[:win, :3] - gt_actions[t:t + win, :3]) ** 2))
                pred_mse_list.append(mse)

    return (replan_ts, np.stack(vf_values), np.stack(gt_orig),
            np.array(div_cos_list), np.array(div_l2_list), np.stack(gmm_means_list),
            np.array(pred_mse_list) if pred_mse_list is not None else None)
    # shapes: (N,), (N, S, D), (N, D), (N,), (N,), (N, K, D), (N,) or None


# ── Signal processing ─────────────────────────────────────────────────────────

def _savgol_smooth(vals: list, window: int, polyorder: int = 3) -> np.ndarray:
    from scipy.signal import savgol_filter
    if window <= 1:
        return np.array(vals, dtype=float)
    win = window if window % 2 == 1 else window + 1
    win = min(win, len(vals))
    if win % 2 == 0:
        win -= 1
    if win < polyorder + 2:
        return np.array(vals, dtype=float)
    return savgol_filter(vals, window_length=win, polyorder=polyorder)


def _find_peaks_above_mean(vals: np.ndarray, ts: list, min_distance: int = 0,
                           margin: int = 0) -> tuple[list, list]:
    """Local maxima above mean, with optional greedy NMS and edge margin.

    margin: ts 기준 앞뒤 margin 이내의 피크는 제외 (너무 짧은 스킬 방지).
    """
    from scipy.signal import find_peaks
    mean_val = float(np.mean(vals))
    peak_idxs, _ = find_peaks(vals)
    above = [i for i in peak_idxs if vals[i] > mean_val]

    if margin > 0 and len(ts) >= 2:
        t_min, t_max = ts[0], ts[-1]
        above = [i for i in above if ts[i] - t_min > margin and t_max - ts[i] > margin]

    if min_distance > 0 and len(above) > 1:
        by_height = sorted(above, key=lambda i: vals[i], reverse=True)
        kept, suppressed = [], set()
        for i in by_height:
            if i in suppressed:
                continue
            kept.append(i)
            for j in by_height:
                if j != i and abs(ts[j] - ts[i]) <= min_distance:
                    suppressed.add(j)
        above = kept

    return [ts[i] for i in above], [float(vals[i]) for i in above]


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_cos_divergence(replan_ts, div_cos, ep_id, ep_tag, eval_at_step,
                        output_dir, smooth_window=5, peak_nms_dist=0,
                        edge_margin=0, polyorder=3,
                        pred_mse_vals: np.ndarray | None = None) -> tuple[Path, list]:
    """Bar chart of cos divergence with SG smooth, mean line, and skill boundary markers.

    If pred_mse_vals is given (xyz MSE between full-denoised predicted action and GT action),
    a second subplot is added below with SG-filtered MSE and boundary markers.
    """
    vals = np.array(div_cos, dtype=float)
    bar_width = (replan_ts[1] - replan_ts[0]) * 0.8 if len(replan_ts) > 1 else 4
    sg_vals = _savgol_smooth(vals.tolist(), smooth_window, polyorder=polyorder)
    mean_val = float(np.mean(sg_vals))
    peak_ts, peak_vals = _find_peaks_above_mean(sg_vals, replan_ts, min_distance=peak_nms_dist,
                                                margin=edge_margin)

    n_rows = 2 if pred_mse_vals is not None else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows), sharex=True)
    ax = axes[0] if n_rows == 2 else axes

    ax.bar(replan_ts, vals, width=bar_width, align="center", alpha=0.4, color="tab:red", label="raw")
    ax.plot(replan_ts, sg_vals, color="tab:orange", linewidth=2, label=f"SG (w={smooth_window})")
    ax.axhline(mean_val, color="tab:orange", linewidth=1.5, linestyle="--",
               label=f"SG mean={mean_val:.4f}")
    if peak_ts:
        ax.scatter(peak_ts, peak_vals, color="red", zorder=5, s=60, label="skill boundary")
    ax.set_ylabel("cos divergence")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title(f"[Cos Divergence] ep{ep_id} | step={eval_at_step} | {ep_tag[:60]}", fontsize=9)

    if pred_mse_vals is not None:
        ax2 = axes[1]
        sg_mse = _savgol_smooth(pred_mse_vals.tolist(), smooth_window, polyorder=polyorder)
        ax2.bar(replan_ts, pred_mse_vals, width=bar_width, align="center", alpha=0.35, color="tab:blue", label="raw MSE")
        ax2.plot(replan_ts, sg_mse, color="tab:blue", linewidth=2, label=f"SG (w={smooth_window})")
        ax2.set_xlabel("frame")
        ax2.set_ylabel("pred-demo xyz MSE")
        ax2.legend(fontsize=8, loc="upper right")
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.set_title("Predicted vs Demo Action MSE", fontsize=9)
        ax2.set_xticks(replan_ts)
        ax2.set_xticklabels([str(t) for t in replan_ts], rotation=45, ha="right", fontsize=7)
    else:
        ax.set_xlabel("frame")
        ax.set_xticks(replan_ts)
        ax.set_xticklabels([str(t) for t in replan_ts], rotation=45, ha="right", fontsize=7)

    fig.tight_layout()
    path = Path(output_dir) / f"ep{ep_id:05d}_cos_divergence_step{eval_at_step}.png"
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
    return path, peak_ts


def plot_gmm_3d_interactive(replan_ts, vf_values, gmm_means,
                             ep_id, ep_tag, eval_at_step, output_dir) -> Path:
    """Interactive 3D HTML: slider over replanning steps showing GMM cluster vectors."""
    import plotly.graph_objects as go

    vf = np.array(vf_values)   # (N, S, D)
    gm = np.array(gmm_means)   # (N, K, D)
    K = gm.shape[1]
    COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231",
              "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]

    frames = []
    for idx, t in enumerate(replan_ts):
        pts = vf[idx]
        data = [
            go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                         mode="markers", marker=dict(size=4, color="lightblue", opacity=0.6),
                         name="probes"),
            go.Scatter3d(x=[pts[0, 0]], y=[pts[0, 1]], z=[pts[0, 2]],
                         mode="markers", marker=dict(size=7, color="navy", symbol="diamond"),
                         name="GT"),
        ]
        for k in range(K):
            mx, my, mz = gm[idx, k, 0], gm[idx, k, 1], gm[idx, k, 2]
            c = COLORS[k % len(COLORS)]
            data += [
                go.Scatter3d(x=[0, mx], y=[0, my], z=[0, mz],
                             mode="lines+markers",
                             line=dict(color=c, width=6),
                             marker=dict(size=[2, 10], color=c, symbol=["circle", "diamond"]),
                             name=f"cluster {k}"),
                go.Cone(x=[mx], y=[my], z=[mz],
                        u=[mx * 0.15], v=[my * 0.15], w=[mz * 0.15],
                        colorscale=[[0, c], [1, c]], showscale=False,
                        sizemode="absolute", sizeref=0.05,
                        name=f"cluster {k} tip"),
            ]
        frames.append(go.Frame(data=data, name=str(t)))

    all_xyz = vf[:, :, :3].reshape(-1, 3)
    pad = max(float(np.abs(all_xyz).max()) * 0.1, 0.01)
    ax_min, ax_max = float(all_xyz.min()) - pad, float(all_xyz.max()) + pad

    fig = go.Figure(
        data=list(frames[0].data) if frames else [],
        frames=frames,
        layout=go.Layout(
            title=f"GMM 3D | ep{ep_id} | step={eval_at_step} | {ep_tag[:50]}",
            scene=dict(
                xaxis=dict(title="Δx", range=[ax_min, ax_max]),
                yaxis=dict(title="Δy", range=[ax_min, ax_max]),
                zaxis=dict(title="Δz", range=[ax_min, ax_max]),
                aspectmode="cube",
            ),
            updatemenus=[dict(
                type="buttons", showactive=False, y=0, x=0.5, xanchor="center",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, {"frame": {"duration": 400, "redraw": True},
                                      "fromcurrent": True}]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
                ],
            )],
            sliders=[dict(
                active=0,
                steps=[dict(method="animate",
                            args=[[str(t)], {"frame": {"duration": 0, "redraw": True},
                                             "mode": "immediate"}],
                            label=str(t)) for t in replan_ts],
                x=0.05, y=0, len=0.9,
                currentvalue=dict(prefix="frame: ", visible=True, xanchor="center"),
            )],
        ),
    )
    path = Path(output_dir) / f"ep{ep_id:05d}_gmm_3d_step{eval_at_step}.html"
    fig.write_html(str(path))
    return path


# ── DINO episode loader ───────────────────────────────────────────────────────

def load_dino_episode(dino_feature_dir: Path, image_key: str, episode_id: int) -> np.ndarray:
    """Load per-episode DINO tokens. Returns (T, n_tokens, feat_dim) float16."""
    npz_path = dino_feature_dir / image_key.replace(".", "_") / f"episode_{episode_id:07d}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"DINO episode file not found: {npz_path}")
    return np.load(str(npz_path))["features"]  # (T, 65, 384) float16


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(dataset_dir: Path, episode_ids: list[int] | None = None) -> pd.DataFrame:
    """Load step-level data, optionally filtering to specific episode_ids using pyarrow pushdown."""
    import pyarrow.parquet as pq

    files = sorted((dataset_dir / "data").rglob("file-*.parquet"))
    if not files:
        files = sorted((dataset_dir / "data").rglob("episode_*.parquet"))

    filters = [("episode_index", "in", set(episode_ids))] if episode_ids is not None else None
    dfs = []
    for f in files:
        table = pq.read_table(f, filters=filters)
        if len(table) > 0:
            dfs.append(table.to_pandas())
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_episodes_meta(dataset_dir: Path) -> pd.DataFrame:
    files = sorted((dataset_dir / "meta" / "episodes").rglob("file-*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def get_video_path(dataset_dir: Path, episode_index: int,
                   camera_key: str, episodes_meta: pd.DataFrame) -> Path:
    row = episodes_meta[episodes_meta["episode_index"] == episode_index].iloc[0]
    chunk_idx = row[f"videos/{camera_key}/chunk_index"]
    file_idx = row[f"videos/{camera_key}/file_index"]
    return dataset_dir / "videos" / camera_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"


def get_episode_timestamps(dataset_dir: Path, episode_index: int,
                            episodes_meta: pd.DataFrame, camera_key: str) -> tuple[float, float]:
    row = episodes_meta[episodes_meta["episode_index"] == episode_index].iloc[0]
    return (float(row[f"videos/{camera_key}/from_timestamp"]),
            float(row[f"videos/{camera_key}/to_timestamp"]))


# ── Policy loading ────────────────────────────────────────────────────────────

def load_policy(policy_path: str, device: str,
                noise_scheduler_type: str, num_inference_steps: int):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy, _make_noise_scheduler
    from lerobot.policies.factory import make_pre_post_processors

    policy = DiffusionPolicy.from_pretrained(policy_path)
    policy = policy.to(device).eval()

    cfg = policy.config
    policy.diffusion.noise_scheduler = _make_noise_scheduler(
        noise_scheduler_type,
        num_train_timesteps=cfg.num_train_timesteps,
        beta_start=cfg.beta_start, beta_end=cfg.beta_end,
        beta_schedule=cfg.beta_schedule,
        clip_sample=cfg.clip_sample, clip_sample_range=cfg.clip_sample_range,
        prediction_type=cfg.prediction_type,
    )
    policy.diffusion.num_inference_steps = num_inference_steps
    print(f"  [scheduler] {noise_scheduler_type}, steps={num_inference_steps}")

    preprocessor, _ = make_pre_post_processors(cfg, pretrained_path=policy_path)
    return policy, preprocessor


# ── wandb logging ─────────────────────────────────────────────────────────────

def log_to_wandb(results: list[dict], args: Args) -> None:
    import wandb

    run_name = args.wandb_run_name or f"task{args.task_id}_{args.n_episodes}eps_{args.n_gmm_components}gmm"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "task_id": args.task_id, "n_episodes": args.n_episodes,
            "replan_interval": args.replan_interval,
            "eval_at_step": args.eval_at_step,
            "n_gmm_components": args.n_gmm_components,
            "smooth_window": args.smooth_window,
            "savgol_polyorder": args.savgol_polyorder,
        },
    )
    for r in results:
        label = f"ep{r['episode_id']:05d}: {r['language'][:60]}"
        log_dict = {}

        # Full demo video
        vp = r.get("combined_path") or (r["cam_clips"][0] if r["cam_clips"] else None)
        if vp and Path(vp).exists():
            log_dict[f"video/{label}"] = wandb.Video(vp, format="mp4")

        # Cos divergence static plot
        if r.get("cos_div_plot") and Path(r["cos_div_plot"]).exists():
            log_dict[f"boundary_criteria/{label}"] = wandb.Image(r["cos_div_plot"])

        # Per-skill videos
        for i, sp in enumerate(r.get("skill_video_paths") or []):
            if Path(sp).exists():
                log_dict[f"skills/{label}/skill_{i + 1}"] = wandb.Video(sp, format="mp4")

        print(f"  [wandb] logging keys: {list(log_dict.keys())}")
        if log_dict:
            wandb.log(log_dict)

        # GMM 3D HTML artifact
        if r.get("gmm_3d_path") and Path(r["gmm_3d_path"]).exists():
            artifact = wandb.Artifact(f"gmm_3d_ep{r['episode_id']:05d}", type="html")
            artifact.add_file(r["gmm_3d_path"])
            wandb.log_artifact(artifact)
            wandb.log({f"gmm_3d/{label}": wandb.Html(Path(r["gmm_3d_path"]).read_text())})

    wandb.finish()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: Args) -> None:
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    viz = SkillVisualizer(output_dir, fps=args.fps)

    if args.seed is not None:
        import random, torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    print("Loading metadata...")
    episodes_meta = load_episodes_meta(dataset_dir)
    tasks_meta = pd.read_parquet(dataset_dir / "meta" / "tasks.parquet").reset_index()

    task_row = tasks_meta[tasks_meta["task_index"] == args.task_id]
    if task_row.empty:
        raise ValueError(f"task_id {args.task_id} not found")
    target_lang = task_row.iloc[0]["task"]
    ep_of_task = episodes_meta[episodes_meta["tasks"].apply(
        lambda t: target_lang in (
            [str(x) for x in t] if isinstance(t, (list, np.ndarray)) else [str(t)]
        )
    )]
    episode_ids = ep_of_task["episode_index"].tolist()[:args.n_episodes]
    print(f"Task {args.task_id}: '{target_lang}' — {len(episode_ids)} episodes")

    video_cols = [c for c in episodes_meta.columns
                  if c.startswith("videos/") and c.endswith("/chunk_index")]
    camera_keys = [c.split("/")[1] for c in video_cols]
    print(f"Cameras: {camera_keys}")

    import time
    print("Loading state data...")
    all_data = load_data(dataset_dir, episode_ids=episode_ids)

    print(f"Loading policy from {args.policy_path} ...")
    t0 = time.time()
    policy, preprocessor = load_policy(
        args.policy_path, args.device, args.noise_scheduler_type, args.num_inference_steps
    )
    print(f"  [time] policy load: {time.time()-t0:.1f}s")

    use_dino = policy.config.use_dino_features
    dino_feature_dir = Path(args.dino_feature_dir) if args.dino_feature_dir else None
    if use_dino and dino_feature_dir is None:
        raise ValueError("--dino_feature_dir is required when policy uses DINO features.")
    dino_image_key = policy.config.dino_image_keys[0] if use_dino else None

    nms_dist = (args.nms_dist if args.nms_dist is not None else args.replan_interval * 2) if args.peak_nms else 0
    results = []

    for ep_id in episode_ids:
        row = episodes_meta[episodes_meta["episode_index"] == ep_id]
        if row.empty:
            print(f"  Episode {ep_id} not found, skipping.")
            continue
        tasks_list = row.iloc[0]["tasks"]
        lang = str(tasks_list[0] if isinstance(tasks_list, (list, np.ndarray)) else tasks_list)
        ep_tag = f"ep{ep_id}: {lang[:80]}"
        ep_df = all_data[all_data["episode_index"] == ep_id].reset_index(drop=True)

        print(f"\n  Episode {ep_id}: '{lang}'")
        t_ep = time.time()

        # ── 프레임 로드 (카메라별 병렬 디코딩) ────────────────────────────────────
        from concurrent.futures import ThreadPoolExecutor

        def _load_cam(cam_key):
            src = get_video_path(dataset_dir, ep_id, cam_key, episodes_meta).resolve()
            start_sec, end_sec = get_episode_timestamps(dataset_dir, ep_id, episodes_meta, cam_key)
            return cam_key, viz.load_episode_frames(src, start_sec, end_sec)

        t1 = time.time()
        with ThreadPoolExecutor(max_workers=len(camera_keys)) as pool:
            cam_frames = dict(pool.map(_load_cam, camera_keys))
        print(f"    [time] frame load: {time.time()-t1:.1f}s")

        # ── main cam 영상만 저장 (wrist cam 제외) ─────────────────────────────
        t1b = time.time()
        main_cam_key = camera_keys[0]
        combined_path = output_dir / f"ep{ep_id:05d}_main.mp4"
        viz.write_single_video(cam_frames[main_cam_key], combined_path, ep_id=ep_id)
        print(f"    [time] video write: {time.time()-t1b:.1f}s")

        # ── DINO token 로드 ────────────────────────────────────────────────────
        ep_dino_tokens = None
        if use_dino:
            ep_dino_tokens = load_dino_episode(dino_feature_dir, dino_image_key, ep_id)

        # ── VF divergence analysis ─────────────────────────────────────────────
        t2 = time.time()
        try:
            print(f"    Running VF analysis (eval_at_step={args.eval_at_step}) ...")
            vf_replan_ts, vf_values, gt_orig, div_cos, div_l2, gmm_means, pred_mse = run_vf_analysis(
                policy, preprocessor, ep_df, cam_frames, camera_keys,
                args.eval_at_step, args.replan_interval,
                n_gmm_components=args.n_gmm_components,
                compute_pred_mse=args.plot_pred_mse,
                mse_window=args.mse_window,
                dino_tokens=ep_dino_tokens,
            )
        except Exception as e:
            import traceback
            print(f"    [ERROR] VF analysis failed: {e}")
            traceback.print_exc()
            results.append({
                "episode_id": ep_id, "language": lang,
                "cam_clips": [],
                "combined_path": str(combined_path) if combined_path else None,
                "cos_div_plot": None, "skill_video_paths": [],
                "gmm_3d_path": None, "cos_div_data": None,
                "div_cos_peak_ts": [], "n_frames": len(ep_df),
            })
            continue
        print(f"    [time] VF analysis: {time.time()-t2:.1f}s")

        # ── Cos divergence plot + peak detection ──────────────────────────────
        cos_div_plot, div_cos_peak_ts = plot_cos_divergence(
            vf_replan_ts, div_cos, ep_id, ep_tag, args.eval_at_step,
            output_dir, smooth_window=args.smooth_window,
            peak_nms_dist=nms_dist, edge_margin=nms_dist, polyorder=args.savgol_polyorder,
            pred_mse_vals=pred_mse,
        )
        print(f"    Skill boundaries: {div_cos_peak_ts}")
        print(f"    Cos div plot:     {cos_div_plot.name}")

        # ── Cut skill videos (main cam numpy에서 직접) ───────────────────────
        t3 = time.time()
        n_frames = len(ep_df)
        boundaries = sorted(set([0] + [int(p) for p in div_cos_peak_ts] + [n_frames]))
        skill_video_paths = viz.cut_skill_videos(cam_frames[main_cam_key], boundaries, ep_id)
        print(f"    Skill videos:     {len(skill_video_paths)} segments")
        print(f"    [time] cut skills: {time.time()-t3:.1f}s")
        print(f"    [time] episode total: {time.time()-t_ep:.1f}s")

        # ── Optional GMM 3D HTML ───────────────────────────────────────────────
        gmm_3d_path = None
        if args.plot_gmm_3d:
            print(f"    Building GMM 3D ...")
            gmm_3d_path = plot_gmm_3d_interactive(
                vf_replan_ts, vf_values, gmm_means,
                ep_id, ep_tag, args.eval_at_step, output_dir,
            )
            print(f"    GMM 3D:           {gmm_3d_path.name}")

        results.append({
            "episode_id": ep_id,
            "language": lang,
            "cam_clips": [],
            "combined_path": str(combined_path) if combined_path else None,
            "cos_div_plot": str(cos_div_plot),
            "skill_video_paths": [str(p) for p in skill_video_paths],
            "gmm_3d_path": str(gmm_3d_path) if gmm_3d_path else None,
            "cos_div_data": (vf_replan_ts, div_cos.tolist()),
            "div_cos_peak_ts": div_cos_peak_ts,
            "n_frames": n_frames,
        })

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. Results saved to {results_path}")

    if args.wandb_project:
        try:
            log_to_wandb(results, args)
        except Exception as e:
            print(f"wandb logging failed: {e}")


if __name__ == "__main__":
    main(tyro.cli(Args))
