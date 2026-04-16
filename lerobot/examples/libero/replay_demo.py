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

    python examples/libero/replay_demo.py \
    --dataset_dir /scratch/mdorazi/Skill_Boundary_Detector/libero_dataset/libero_spatial_object \
    --task_id  \
    --n_episodes 10 \
    --policy_path /scratch/mdorazi/Skill_Boundary_Detector/outputs/dp_libero_spatial_object/checkpoints/080000/pretrained_model \
    --replan_interval 5 \
    --mse_smooth_method savgol \
    --mse_smooth_window 5 \
    --output_dir /scratch/mdorazi/Skill_Boundary_Detector/outputs/replay_libero_spatial_object/task_test \
    --wandb_project SBD_skill_SO
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
    wandb_run_name: str | None = None
    """Custom wandb run name. Auto-generated if None."""
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
    mse_outlier_ratio: float = 10.0
    """If max(mse) > this × median(mse), the episode is dominated by a single outlier and is excluded from saving."""
    noise_scheduler_type: str | None = None
    """Override noise scheduler: 'DDPM' or 'DDIM'. None = use saved config default."""
    num_inference_steps: int | None = None
    """Override denoising steps at inference. None = use saved config default (100)."""
    eval_at_step: int | None = None
    """If set, compute VF error at this denoising step index (0-based) and log to wandb.
    Requires --noise_scheduler_type DDIM --num_inference_steps N (e.g. eval_at_step=8, N=10)."""


# ── VF error analysis ─────────────────────────────────────────────────────────

def _query_vf_error(policy, global_cond, chunks_batch, eval_at_step: int):
    """Feed action chunks into UNet, apply one deterministic denoising step, return x^{k-1}.

    Args:
        chunks_batch: (B, H, action_dim) tensor — GT chunk + probe chunks stacked.
        global_cond:  (1, cond_dim) tensor — broadcast to batch size B.
    Returns:
        (B, action_dim) numpy array — last action step of x^{k-1} for each chunk.
    """
    import torch
    with torch.no_grad():
        scheduler = policy.diffusion.noise_scheduler
        t = scheduler.timesteps[eval_at_step]
        B = chunks_batch.shape[0]
        t_batch = torch.full((B,), t, dtype=torch.long, device=chunks_batch.device)
        gc = global_cond.expand(B, -1)
        eps_pred = policy.diffusion.unet(chunks_batch, t_batch, global_cond=gc)
        # One deterministic denoising step: x^{k-1} = α(x^k - γ·ε_θ)
        # scheduler.step handles α/γ coefficients; DDIM is σ=0 (no noise added)
        denoised = scheduler.step(eps_pred, t, chunks_batch).prev_sample
        return denoised.sum(dim=1).cpu().numpy()  # (B, action_dim) — total displacement


def run_vf_analysis(
    policy, preprocessor, ep_df, cam_clips, camera_keys,
    eval_at_step: int, replan_interval: int,
    polar_degs=(30, 60), azimuth_step_deg=30,
    n_gmm_components=3,
) -> tuple:
    """Returns (replan_ts, vf_values, gt_orig, divergences).

    vf_values:   (N_replan, N_samples, action_dim) — index 0 is GT, rest probes.
    gt_orig:     (N_replan, action_dim) — original GT total displacement.
    divergences: (N_replan,) — GMM divergence scalar per replan step.

    select_action을 그대로 호출해 queue/image stacking을 동일하게 유지하고,
    _prepare_global_conditioning을 hook으로 가로채 global_cond를 캡처한다.
    """
    import torch
    from lerobot.utils.constants import OBS_STATE

    policy.diffusion.noise_scheduler.set_timesteps(policy.diffusion.num_inference_steps)

    n_action_steps = policy.config.n_action_steps
    horizon = policy.config.horizon
    eff_interval = replan_interval if replan_interval > 0 else n_action_steps
    device = next(policy.parameters()).device

    states = np.stack(ep_df["observation.state"].values)
    cam_frames = {}
    for cam_key, clip_path in zip(camera_keys, cam_clips):
        reader = imageio.get_reader(str(clip_path))
        cam_frames[cam_key] = np.stack([f for f in reader])
        reader.close()

    T = min(len(ep_df), *(len(v) for v in cam_frames.values()))
    states = states[:T]
    gt_actions = np.stack(ep_df["action"].values[:T])
    for k in cam_frames:
        cam_frames[k] = cam_frames[k][:T]

    replan_ts, vf_values, gt_orig, divergences = [], [], [], []
    policy.reset()

    # Hook: select_action 내부의 _prepare_global_conditioning 결과를 캡처
    _captured = {}
    _orig_pgc = policy.diffusion._prepare_global_conditioning

    def _hooking_pgc(batch):
        gc = _orig_pgc(batch)
        _captured["global_cond"] = gc.detach()
        return gc

    policy.diffusion._prepare_global_conditioning = _hooking_pgc

    try:
        for t in range(T):
            obs = {k: torch.from_numpy(cam_frames[k][t]).float().div(255.0).permute(2, 0, 1)
                   for k in camera_keys}
            obs[OBS_STATE] = torch.from_numpy(states[t]).float()
            obs = preprocessor(obs)

            with torch.no_grad():
                policy.select_action(obs)  # queue 유지 + global_cond 캡처

            # replan 시점에만 VF error 계산
            if t % eff_interval == 0 and "global_cond" in _captured:
                global_cond = _captured["global_cond"]

                end = min(t + horizon, T)
                chunk = gt_actions[t:end]
                if len(chunk) < horizon:
                    chunk = np.concatenate([chunk, np.tile(chunk[-1:], (horizon - len(chunk), 1))], axis=0)

                # Build GT + probe batch
                chunks_np = _make_probe_chunks(chunk, polar_degs, azimuth_step_deg)  # (N, H, action_dim)
                chunks_t = torch.from_numpy(chunks_np).float().to(device)  # (N, H, action_dim)
                vf_batch = _query_vf_error(policy, global_cond, chunks_t, eval_at_step)  # (N, action_dim)
                div, _ = compute_vf_divergence(vf_batch, n_components=n_gmm_components)
                replan_ts.append(t)
                vf_values.append(vf_batch)
                gt_orig.append(chunk.sum(axis=0))  # original GT total displacement (action_dim,)
                divergences.append(div)
    finally:
        policy.diffusion._prepare_global_conditioning = _orig_pgc

    return replan_ts, np.stack(vf_values), np.stack(gt_orig), np.array(divergences)
    # shapes: (N,), (N, S, D), (N, D), (N,)



# ── Spherical probe sampling ───────────────────────────────────────────────────

def _normalize(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return None, n
    return v / n, n


def _make_local_basis(pole_vec):
    """pole_vec: (3,) nonzero. Returns (e1, e2, e3) orthonormal basis where e3=pole direction."""
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
    """Generate GT + spherical probe trajectories around the GT endpoint direction.

    Args:
        gt_delta_xyz: (H, 3) delta EEF xyz chunk.
        polar_degs: polar angles (degrees from GT direction) to sample.
        azimuth_step_deg: azimuth step size in degrees.
    Returns:
        list of (H, 3) delta arrays — index 0 is GT, rest are probes.
    """
    gt_path = np.cumsum(gt_delta_xyz, axis=0)
    endpoint = gt_path[-1]
    u, _ = _normalize(endpoint)
    if u is None:
        return [gt_delta_xyz.copy()]  # endpoint near zero → GT only

    e1, e2, e3 = _make_local_basis(endpoint)
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
    """Build a batch of action chunks: GT + spherical probes (xyz perturbed, rest unchanged).

    Args:
        gt_chunk_np: (H, action_dim) GT action chunk.
    Returns:
        (N_samples, H, action_dim) — index 0 is GT, rest are probes.
    """
    xyz = gt_chunk_np[:, :3]
    rest = gt_chunk_np[:, 3:]

    xyz_samples = _generate_spherical_samples(xyz, polar_degs, azimuth_step_deg)
    chunks = []
    for xyz_s in xyz_samples:
        chunks.append(np.concatenate([xyz_s, rest], axis=1))

    return np.stack(chunks, axis=0)  # (N_samples, H, action_dim)


# ── GMM divergence ─────────────────────────────────────────────────────────────

def compute_vf_divergence(vf_batch, n_components=3):
    """Fit GMM to VF vectors and compute mean pairwise divergence of cluster means.

    Divergence formula (cosine-based, as in the paper):
        D = 1/(n(n-1)) * Σ_{i≠j} (1 - S_c(μ_i, μ_j))
        S_c(a, b) = (a·b) / (||a|| ||b||)

    Args:
        vf_batch: (N_samples, action_dim) — denoised vectors at one replan step.
        n_components: number of GMM clusters N.
    Returns:
        divergence: scalar
        means: (n_components, action_dim) cluster means
    """
    from sklearn.mixture import GaussianMixture

    n_components = min(n_components, len(vf_batch))
    gmm = GaussianMixture(n_components=n_components, random_state=0, max_iter=200)
    gmm.fit(vf_batch)
    means = gmm.means_  # (n_components, action_dim)

    if n_components < 2:
        return 0.0, means

    total, count = 0.0, 0
    for i in range(n_components):
        for j in range(n_components):
            if i != j:
                g_i, g_j = means[i], means[j]
                denom = np.linalg.norm(g_i) * np.linalg.norm(g_j) + 1e-8
                cos_sim = np.dot(g_i, g_j) / denom
                total += 1.0 - cos_sim
                count += 1

    return total / count, means  # scalar, (n_components, action_dim)



def plot_vf_error(replan_ts, vf_values, gt_orig, divergences, ep_id, ep_tag, eval_at_step, output_dir) -> Path:
    """Plot xyz denoised trajectories + GMM divergence over replan timesteps.

    Args:
        vf_values:   (N_replan, N_samples, action_dim) — index 0 is denoised GT, rest probes.
        gt_orig:     (N_replan, action_dim) — original GT total displacement.
        divergences: (N_replan,) — GMM divergence scalar per replan step.
    """
    import matplotlib.pyplot as plt
    vf = np.array(vf_values)        # (N_replan, N_samples, action_dim)
    gt_denoised = vf[:, 0, :]       # (N_replan, action_dim)
    probe_vf = vf[:, 1:, :]         # (N_replan, N_probes, action_dim)
    gt_orig = np.array(gt_orig)     # (N_replan, action_dim)
    divs = np.array(divergences)    # (N_replan,)

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for i, (ax, label) in enumerate(zip(axes[:3], ["x", "y", "z"])):
        for p in range(probe_vf.shape[1]):
            ax.plot(replan_ts, probe_vf[:, p, i], color="steelblue", alpha=0.2, linewidth=0.8)
        ax.plot(replan_ts, gt_denoised[:, i], color="tab:blue", linewidth=2.0,
                marker="o", markersize=3, label="GT denoised")
        ax.plot(replan_ts, gt_orig[:, i], color="tab:orange", linewidth=1.5,
                linestyle="--", marker="x", markersize=4, label="GT orig")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylabel(f"Δ{label}")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)

    # divergence subplot
    ax_div = axes[3]
    ax_div.plot(replan_ts, divs, color="tab:red", linewidth=2.0, marker="o", markersize=4)
    ax_div.set_ylabel("divergence")
    ax_div.set_xlabel("Frame")
    ax_div.grid(True, alpha=0.3)

    axes[0].set_title(f"VF denoised + GMM divergence | ep{ep_id} | step={eval_at_step} | {ep_tag[:60]}")
    fig.tight_layout()
    path = Path(output_dir) / f"ep{ep_id:05d}_vf_step{eval_at_step}.png"
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
    return path


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

def load_policy(
    policy_path: str,
    device: str = "cuda",
    noise_scheduler_type: str | None = None,
    num_inference_steps: int | None = None,
):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy, _make_noise_scheduler
    from lerobot.policies.factory import make_pre_post_processors
    policy = DiffusionPolicy.from_pretrained(policy_path)
    policy = policy.to(device).eval()

    # Override scheduler type / inference steps if requested
    if noise_scheduler_type is not None or num_inference_steps is not None:
        cfg = policy.config
        sched_type = noise_scheduler_type if noise_scheduler_type is not None else cfg.noise_scheduler_type
        policy.diffusion.noise_scheduler = _make_noise_scheduler(
            sched_type,
            num_train_timesteps=cfg.num_train_timesteps,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            beta_schedule=cfg.beta_schedule,
            clip_sample=cfg.clip_sample,
            clip_sample_range=cfg.clip_sample_range,
            prediction_type=cfg.prediction_type,
        )
        policy.diffusion.num_inference_steps = (
            num_inference_steps if num_inference_steps is not None else cfg.num_train_timesteps
        )
        print(
            f"  [scheduler] type={sched_type}, "
            f"num_inference_steps={policy.diffusion.num_inference_steps}"
        )

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
        if args.wandb_run_name:
            run_name = args.wandb_run_name
        elif args.task_id is not None:
            run_name = f"task{args.task_id}_progress"
        else:
            run_name = "replay_progress"
        progress_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "task_id": args.task_id,
                "n_episodes": len(episode_ids),
                "replan_interval": args.replan_interval,
                "mse_smooth_method": args.mse_smooth_method,
                "mse_smooth_window": args.mse_smooth_window,
                "noise_scheduler_type": args.noise_scheduler_type,
                "num_inference_steps": args.num_inference_steps,
                "eval_at_step": args.eval_at_step,
            },
            reinit=True,
        )

    n_total = len(episode_ids)
    n_done = 0
    total_skills = 0
    n_skipped_outlier = 0
    n_skipped_few_skills = 0

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
            # Outlier check: skip if max raw MSE >> Savitzky-Golay mean
            mse_arr = np.array(mse_vals)
            mse_max = float(np.max(mse_arr))
            sg_mean = float(np.mean(smoothed_dict["savgol"])) if "savgol" in smoothed_dict else float(np.mean(mse_arr))
            if sg_mean > 0 and mse_max > args.mse_outlier_ratio * sg_mean:
                n_skipped_outlier += 1
                print(
                    f"    Skipping save: MSE outlier detected "
                    f"(max={mse_max:.4f} > {args.mse_outlier_ratio}× sg_mean={sg_mean:.4f}) "
                    f"— demo excluded from VAE training."
                )
            elif args.save_skills and sg_peak_ts:
                states_arr = np.stack(ep_df["observation.state"].values[:len(gt_actions)])
                cuts = sorted(set(sg_peak_ts))
                breakpoints = [0] + [min(c, len(gt_actions)) for c in cuts] + [len(gt_actions)]
                breakpoints = sorted(set(breakpoints))

                # Collect valid segments first; skip this demo if ≤ 1 skill
                valid_segs = [
                    (si, s, e)
                    for si, (s, e) in enumerate(zip(breakpoints, breakpoints[1:]))
                    if e - s >= 2
                ]
                if len(valid_segs) <= 1:
                    n_skipped_few_skills += 1
                    print(f"    Skipping save: only {len(valid_segs)} valid skill segment(s) — demo excluded from VAE training.")
                else:
                    skill_files = []
                    for si, s, e in valid_segs:
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

        # ── VF error analysis ─────────────────────────────────────────────────
        vf_plot_path = None
        if args.eval_at_step is not None and policy_bundle is not None:
            print(f"    Running VF error analysis at step {args.eval_at_step} ...")
            policy, preprocessor, postprocessor = policy_bundle
            vf_replan_ts, vf_values, gt_orig, divergences = run_vf_analysis(
                policy, preprocessor, ep_df, cam_clips, camera_keys,
                args.eval_at_step, args.replan_interval,
            )
            vf_plot_path = plot_vf_error(
                vf_replan_ts, vf_values, gt_orig, divergences,
                ep_id, ep_tag, args.eval_at_step, output_dir
            )
            print(f"    VF error plot:    {vf_plot_path.name}")

        # ── Clean up temp clips (skip_viz mode) ───────────────────────────────
        for tmp_clip in _tmp_clips:
            try:
                tmp_clip.unlink()
            except Exception:
                pass

        # ── wandb: VF error graph ─────────────────────────────────────────────
        if progress_run is not None and args.eval_at_step is not None:
            import wandb
            if vf_plot_path and vf_plot_path.exists():
                progress_run.log({"vf_error": wandb.Image(str(vf_plot_path))})
            # log divergence as wandb line chart
            div_table = wandb.Table(
                data=[[int(ts), float(d)] for ts, d in zip(vf_replan_ts, divergences)],
                columns=["frame", "divergence"],
            )
            progress_run.log({"vf_divergence": wandb.plot.line(
                div_table, "frame", "divergence", title=f"GMM Divergence ep{ep_id}"
            )})

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
                "progress/skipped_outlier": n_skipped_outlier,
                "progress/skipped_few_skills": n_skipped_few_skills,
                "progress/skipped_total": n_skipped_outlier + n_skipped_few_skills,
            })
            print(f"  [{n_done}/{n_total}] ep{ep_id} done — {ep_skills} skills (total: {total_skills}) | skipped: outlier={n_skipped_outlier}, few_skills={n_skipped_few_skills}")

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
    print(f"[Summary] episodes={n_total} | saved={n_done - n_skipped_outlier - n_skipped_few_skills} | skipped_outlier={n_skipped_outlier} | skipped_few_skills={n_skipped_few_skills} | skipped_total={n_skipped_outlier + n_skipped_few_skills}")

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
