"""
eval_latent_space.py

Compare two VAE latent spaces (fixed-length vs variable-length skill segmentation)
by KMeans clustering on latent vectors → actual EEF trajectory comparison.

Pipeline:
  Load NPZ → KMeans on latents → for each cluster, load actual EEF state sequences
  from dataset parquet (observation.states.ee_state) → compute 6 metrics

6 Metrics (within-cluster vs between-cluster):
  direction    : cosine similarity of start→end displacement vector (6D)
  shape        : mean pairwise L2 of resampled relative EEF trajectory
  final_pose   : Euclidean distance of absolute final EEF position (xyz)
  delta_pose   : MSE of total displacement vector (6D)
  dtw          : DTW distance of relative EEF trajectory (normalized by length)
  wasserstein  : mean per-dim Wasserstein distance of per-step EEF deltas

Separation score:
  direction    : intra_mean - inter_mean  (higher → better)
  others       : inter_mean / intra_mean  (higher → better)

Also kept:
  reconstruction eval : decoded action (cumsum) vs original action sequence
  dataset stats       : skill count, average skill length

Usage:
  python examples/libero/eval_latent_space.py \\
    --npz_A  .../spline_vae_latents_epoch10000.npz \\
    --vae_A  .../spline_vae_epoch10000.pt \\
    --npz_B  .../spline_vae_latents_epoch10000.npz \\
    --vae_B  .../spline_vae_epoch10000.pt \\
    --dataset_path .../libero_dataset/libero_90_skillvla \\
    --output_dir .../outputs/latent_eval
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.cluster import KMeans
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spline_vae import GRIPPER_DIM, SplineVAE, spline_decode, spline_encode  # noqa: E402


# ── VAE loading ───────────────────────────────────────────────────────────────

def load_vae(ckpt_path: str) -> SplineVAE:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"]
    cfg_d = dataclasses.asdict(cfg)
    vae_keys = {"action_dim", "state_dim", "n_control", "spline_degree",
                "hidden_dim", "latent_dim", "num_layers", "dropout",
                "max_length", "action_min", "action_max"}
    vae = SplineVAE(**{k: v for k, v in cfg_d.items() if k in vae_keys})
    vae.load_state_dict(ckpt["model_state"])
    vae.eval()
    return vae


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_skill_states(
    dataset_path: str,
    episode_ids : np.ndarray,
    frame_starts: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    """Return {(episode_id, frame_start): observation.state} for reconstruction eval."""
    needed: dict[int, set[int]] = {}
    for ep, fs in zip(episode_ids.tolist(), frame_starts.tolist()):
        needed.setdefault(int(ep), set()).add(int(fs))

    parquets  = sorted((Path(dataset_path) / "data").rglob("*.parquet"))
    state_map : dict[tuple[int, int], np.ndarray] = {}

    for pf in tqdm(parquets, desc="Loading states"):
        df = pd.read_parquet(pf, columns=["episode_index", "frame_index", "observation.state"])
        eps_here = set(df["episode_index"].astype(int).unique()) & set(needed.keys())
        if not eps_here:
            continue
        sub = df[df["episode_index"].isin(eps_here)]
        for ep_id, fr_id, state in zip(
            sub["episode_index"].astype(int),
            sub["frame_index"].astype(int),
            sub["observation.state"],
        ):
            if fr_id in needed.get(ep_id, set()):
                state_map[(ep_id, fr_id)] = np.asarray(state, dtype=np.float32)

    return state_map


def _load_seq_from_parquet(
    dataset_path: str,
    needed      : dict[int, dict[int, int]],
    column      : str,
    desc        : str,
) -> dict[tuple[int, int], np.ndarray]:
    """Generic loader: returns {(ep_id, frame_start): (T, D)} for a given column."""
    parquets = sorted((Path(dataset_path) / "data").rglob("*.parquet"))
    result: dict[tuple[int, int], np.ndarray] = {}

    for pf in tqdm(parquets, desc=desc):
        df = pd.read_parquet(pf, columns=["episode_index", "frame_index", column])
        eps_here = set(df["episode_index"].astype(int).unique()) & set(needed.keys())
        if not eps_here:
            continue
        sub = df[df["episode_index"].isin(eps_here)].sort_values(
            ["episode_index", "frame_index"]
        )
        for ep_id, ep_df in sub.groupby(sub["episode_index"].astype(int)):
            for fs, fe in needed.get(ep_id, {}).items():
                rows = ep_df[
                    (ep_df["frame_index"] >= fs) & (ep_df["frame_index"] < fe)
                ]
                if len(rows) == 0:
                    continue
                result[(ep_id, fs)] = np.stack(rows[column].values).astype(np.float32)

    return result


def load_action_sequences(
    dataset_path: str,
    episode_ids : np.ndarray,
    frame_starts: np.ndarray,
    frame_ends  : np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    needed: dict[int, dict[int, int]] = {}
    for ep, fs, fe in zip(episode_ids.tolist(), frame_starts.tolist(), frame_ends.tolist()):
        needed.setdefault(int(ep), {})[int(fs)] = int(fe)
    return _load_seq_from_parquet(dataset_path, needed, "action", "Loading actions")


def load_ee_state_sequences(
    dataset_path: str,
    episode_ids : np.ndarray,
    frame_starts: np.ndarray,
    frame_ends  : np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    """Return {(episode_id, frame_start): ee_state_seq} where ee_state_seq is (T, 6)."""
    needed: dict[int, dict[int, int]] = {}
    for ep, fs, fe in zip(episode_ids.tolist(), frame_starts.tolist(), frame_ends.tolist()):
        needed.setdefault(int(ep), {})[int(fs)] = int(fe)
    return _load_seq_from_parquet(
        dataset_path, needed, "observation.states.ee_state", "Loading EEF sequences"
    )


# ── Reconstruction evaluation (decoded action vs original action) ──────────────

def process_raw_traj(raw_traj: np.ndarray, n_resample: int) -> np.ndarray:
    """cumsum(pos+ori) + resample on a raw delta action sequence."""
    gripper_idx = raw_traj.shape[1] - 1
    cum = raw_traj.copy()
    cum[:, :gripper_idx] = np.cumsum(raw_traj[:, :gripper_idx], axis=0)
    T = raw_traj.shape[0]
    if T == n_resample:
        return cum.astype(np.float32)
    t_orig = np.linspace(0.0, 1.0, T)
    t_new  = np.linspace(0.0, 1.0, n_resample)
    return np.stack(
        [np.interp(t_new, t_orig, cum[:, d]) for d in range(cum.shape[1])], axis=-1
    ).astype(np.float32)


def decode_and_process(
    vae       : SplineVAE,
    z         : np.ndarray,
    start_state: np.ndarray,
    n_resample: int = 100,
) -> np.ndarray:
    """Decode z → resampled cumulative trajectory (n_resample, action_dim)."""
    with torch.no_grad():
        z_t = torch.from_numpy(z).unsqueeze(0).float()
        s_t = torch.from_numpy(start_state).unsqueeze(0).float()
        ctrl_pts_norm, len_norm = vae.decode(z_t, s_t)

    ctrl_np = ctrl_pts_norm[0].cpu().numpy()
    len_np  = float(len_norm[0, 0].cpu().numpy())

    lo = vae.action_min.cpu().numpy()
    hi = vae.action_max.cpu().numpy()
    gripper_idx = vae.action_dim + GRIPPER_DIM

    ctrl_np[:, gripper_idx] = np.where(
        1.0 / (1.0 + np.exp(-ctrl_np[:, gripper_idx])) > 0.5, 1.0, -1.0
    )
    ctrl_np = (ctrl_np + 1) / 2 * (hi - lo + 1e-8) + lo

    T = max(2, round(len_np * vae.max_length))
    raw_traj = spline_decode(ctrl_np, T, vae.spline_degree)

    cum_traj = raw_traj.copy()
    cum_traj[:, :gripper_idx] = np.cumsum(raw_traj[:, :gripper_idx], axis=0)

    if T == n_resample:
        return cum_traj.astype(np.float32)
    t_orig = np.linspace(0.0, 1.0, T)
    t_new  = np.linspace(0.0, 1.0, n_resample)
    return np.stack(
        [np.interp(t_new, t_orig, cum_traj[:, d]) for d in range(cum_traj.shape[1])],
        axis=-1,
    ).astype(np.float32)


def evaluate_reconstruction(
    npz_path   : str,
    vae_ckpt   : str,
    action_map : dict[tuple[int, int], np.ndarray],
    state_map  : dict[tuple[int, int], np.ndarray],
    n_eval     : int = 500,
    n_resample : int = 100,
    seed       : int = 42,
) -> dict:
    """Compare decoded trajectory vs original action sequence for n_eval random skills."""
    rng  = np.random.default_rng(seed)
    data = np.load(npz_path)
    latents      = data["latents"]
    episode_ids  = data["episode_id"]
    frame_starts = data["frame_start"]

    vae = load_vae(vae_ckpt)

    available = [
        i for i in range(len(latents))
        if (int(episode_ids[i]), int(frame_starts[i])) in action_map
        and (int(episode_ids[i]), int(frame_starts[i])) in state_map
    ]
    sampled = rng.choice(available, min(n_eval, len(available)), replace=False)
    print(f"  Reconstruction eval on {len(sampled)} skills ...")

    mses, l2s, dir_sims = [], [], []
    mse_per_dim_list: list[np.ndarray] = []
    l2_per_dim_list:  list[np.ndarray] = []

    for idx in tqdm(sampled, desc="  Recon eval"):
        key         = (int(episode_ids[idx]), int(frame_starts[idx]))
        start_state = state_map[key]
        orig_raw    = action_map[key]

        orig_traj = process_raw_traj(orig_raw, n_resample)
        dec_traj  = decode_and_process(vae, latents[idx], start_state, n_resample)

        diff = orig_traj - dec_traj

        mses.append(float(np.mean(diff ** 2)))
        l2s.append(float(np.mean(np.linalg.norm(diff, axis=-1))))
        mse_per_dim_list.append(np.mean(diff ** 2, axis=0))
        l2_per_dim_list.append(np.mean(np.abs(diff), axis=0))

        ep_o = orig_traj[-1, :6]
        ep_d = dec_traj[-1, :6]
        cos  = np.dot(ep_o, ep_d) / (np.linalg.norm(ep_o) * np.linalg.norm(ep_d) + 1e-8)
        dir_sims.append(float(cos))

    return {
        "recon_mse":           float(np.mean(mses)),
        "recon_l2_per_step":   float(np.mean(l2s)),
        "recon_direction_cos": float(np.mean(dir_sims)),
        "recon_mse_per_dim":   np.mean(mse_per_dim_list, axis=0).tolist(),
        "recon_l2_per_dim":    np.mean(l2_per_dim_list,  axis=0).tolist(),
        "n_eval":              len(sampled),
    }


def evaluate_reconstruction_ctrl(
    npz_path   : str,
    vae_ckpt   : str,
    action_map : dict[tuple[int, int], np.ndarray],
    state_map  : dict[tuple[int, int], np.ndarray],
    n_eval     : int = 500,
    seed       : int = 42,
) -> dict:
    """Compare decoded vs original using (1) control points and (2) DTW on raw deltas."""
    rng  = np.random.default_rng(seed)
    data = np.load(npz_path)
    latents      = data["latents"]
    episode_ids  = data["episode_id"]
    frame_starts = data["frame_start"]

    vae = load_vae(vae_ckpt)
    lo  = vae.action_min.cpu().numpy()
    hi  = vae.action_max.cpu().numpy()

    available = [
        i for i in range(len(latents))
        if (int(episode_ids[i]), int(frame_starts[i])) in action_map
        and (int(episode_ids[i]), int(frame_starts[i])) in state_map
    ]
    sampled = rng.choice(available, min(n_eval, len(available)), replace=False)
    print(f"  Control-point + DTW eval on {len(sampled)} skills ...")

    ctrl_l2_per_dim_list: list[np.ndarray] = []
    dtw_list: list[float] = []

    for idx in tqdm(sampled, desc="  Ctrl/DTW eval"):
        key         = (int(episode_ids[idx]), int(frame_starts[idx]))
        start_state = state_map[key]
        orig_raw    = action_map[key]                   # (T_orig, action_dim)

        # ── Control points ────────────────────────────────────────────────────
        # Original: bspline-fit raw action → control points → normalize
        orig_ctrl, _ = spline_encode(orig_raw, vae.n_control, vae.spline_degree)
        orig_ctrl_norm = 2 * (orig_ctrl - lo) / (hi - lo + 1e-8) - 1   # (n_ctrl, D)

        # Decoded: VAE decode → already normalized control points
        with torch.no_grad():
            z_t = torch.from_numpy(latents[idx]).unsqueeze(0).float()
            s_t = torch.from_numpy(start_state).unsqueeze(0).float()
            dec_ctrl_norm, _ = vae.decode(z_t, s_t)
        dec_ctrl_norm = dec_ctrl_norm[0].cpu().numpy()                   # (n_ctrl, D)

        # Per-dim L2 (gripper excluded: last dim)
        gripper_idx = vae.action_dim + GRIPPER_DIM
        diff_ctrl   = orig_ctrl_norm[:, :gripper_idx] - dec_ctrl_norm[:, :gripper_idx]
        ctrl_l2_per_dim_list.append(np.mean(np.abs(diff_ctrl), axis=0))  # (D-1,)

        # ── DTW on raw delta actions (gripper excluded) ───────────────────────
        orig_no_grip = orig_raw[:, :gripper_idx].astype(np.float32)
        dec_raw = spline_decode(
            (dec_ctrl_norm[:, :gripper_idx] + 1) / 2 * (hi[:gripper_idx] - lo[:gripper_idx] + 1e-8) + lo[:gripper_idx],
            max(2, orig_raw.shape[0]),      # decode to same length as original for fair DTW
            vae.spline_degree,
        )
        dtw_list.append(_dtw_normalized(orig_no_grip, dec_raw))

    action_dim_labels = ["dx", "dy", "dz", "droll", "dpitch", "dyaw"]
    n_dims = ctrl_l2_per_dim_list[0].shape[0]

    return {
        "ctrl_l2_per_dim":  np.mean(ctrl_l2_per_dim_list, axis=0).tolist(),
        "ctrl_l2":          float(np.mean([v.mean() for v in ctrl_l2_per_dim_list])),
        "dtw_mean":         float(np.mean(dtw_list)),
        "n_eval":           len(sampled),
        "dim_labels":       action_dim_labels[:n_dims],
    }


# ── EEF trajectory metrics ────────────────────────────────────────────────────

def _resample(seq: np.ndarray, n: int) -> np.ndarray:
    """Resample a (T, D) sequence to (n, D) using linear interpolation."""
    T = len(seq)
    if T == n:
        return seq
    t_orig = np.linspace(0.0, 1.0, T)
    t_new  = np.linspace(0.0, 1.0, n)
    return np.stack(
        [np.interp(t_new, t_orig, seq[:, d]) for d in range(seq.shape[1])], axis=-1
    ).astype(np.float32)


def _dtw_normalized(a: np.ndarray, b: np.ndarray) -> float:
    """DTW distance normalized by average sequence length."""
    na, nb = len(a), len(b)
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)  # (na, nb)
    prev = np.full(nb + 1, np.inf)
    prev[0] = 0.0
    for i in range(na):
        curr = np.full(nb + 1, np.inf)
        for j in range(nb):
            curr[j + 1] = cost[i, j] + min(prev[j + 1], curr[j], prev[j])
        prev = curr
    return float(prev[nb]) / ((na + nb) / 2)


def _wasserstein_mean(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-dimension Wasserstein-1 distance."""
    from scipy.stats import wasserstein_distance
    return float(np.mean([
        wasserstein_distance(a[:, d], b[:, d]) for d in range(a.shape[1])
    ]))


def pairwise_ee_metrics(
    ee_seqs   : list[np.ndarray],  # each (T_i, 6) absolute EEF state
    n_resample: int,
) -> dict | None:
    """Compute pairwise metrics between actual EEF state sequences."""
    pairs = list(combinations(range(len(ee_seqs)), 2))
    if not pairs:
        return None

    dir_sims, final_dists, delta_mses, dtw_dists, was_dists = [], [], [], [], []
    shape_rel_all, shape_abs_all   = [], []   # aggregate L2
    shape_rel_dim_all, shape_abs_dim_all = [], []   # per-dim MAE

    for i, j in pairs:
        si, sj = ee_seqs[i][:, :3], ee_seqs[j][:, :3]   # xyz only

        # Relative trajectories (subtract initial position)
        rel_i = (si - si[0]).astype(np.float32)
        rel_j = (sj - sj[0]).astype(np.float32)

        # 1. Direction: cos-sim of start→end displacement (6D)
        disp_i = rel_i[-1]
        disp_j = rel_j[-1]
        dir_sims.append(float(
            np.dot(disp_i, disp_j) / (np.linalg.norm(disp_i) * np.linalg.norm(disp_j) + 1e-8)
        ))

        # 2a. Shape L2 relative (skill's own movement, initial subtracted)
        res_rel_i = _resample(rel_i, n_resample)
        res_rel_j = _resample(rel_j, n_resample)
        diff_rel  = res_rel_i - res_rel_j
        shape_rel_all.append(float(np.mean(np.linalg.norm(diff_rel, axis=-1))))
        shape_rel_dim_all.append(np.mean(np.abs(diff_rel), axis=0))   # (6,)

        # 2b. Shape L2 absolute (where in space the robot is)
        res_abs_i = _resample(si, n_resample)
        res_abs_j = _resample(sj, n_resample)
        diff_abs  = res_abs_i - res_abs_j
        shape_abs_all.append(float(np.mean(np.linalg.norm(diff_abs, axis=-1))))
        shape_abs_dim_all.append(np.mean(np.abs(diff_abs), axis=0))   # (6,)

        # 3. Final pose dist: absolute final xyz position
        final_dists.append(float(np.linalg.norm(si[-1] - sj[-1])))

        # 4. Delta pose MSE: total displacement (6D)
        delta_mses.append(float(np.mean((disp_i - disp_j) ** 2)))

        # 5. DTW: on relative EEF trajectory (no resampling needed)
        dtw_dists.append(_dtw_normalized(rel_i, rel_j))

        # 6. Wasserstein: on per-step EEF deltas
        d_i = np.diff(si, axis=0)
        d_j = np.diff(sj, axis=0)
        was_dists.append(_wasserstein_mean(d_i, d_j))

    return {
        "direction_cos_sim":      float(np.mean(dir_sims)),
        "shape_l2_rel":           float(np.mean(shape_rel_all)),
        "shape_l2_abs":           float(np.mean(shape_abs_all)),
        "shape_l2_rel_per_dim":   np.mean(shape_rel_dim_all, axis=0).tolist(),
        "shape_l2_abs_per_dim":   np.mean(shape_abs_dim_all, axis=0).tolist(),
        "final_pose_dist":        float(np.mean(final_dists)),
        "delta_pose_mse":         float(np.mean(delta_mses)),
        "dtw":                    float(np.mean(dtw_dists)),
        "wasserstein":            float(np.mean(was_dists)),
    }


# ── EEF-based clustering evaluation ──────────────────────────────────────────

def evaluate_ee(
    npz_path            : str,
    ee_map              : dict[tuple[int, int], np.ndarray],
    n_clusters          : int = 20,
    n_sample_per_cluster: int = 10,
    n_resample          : int = 80,
    seed                : int = 42,
) -> dict:
    rng          = np.random.default_rng(seed)
    data         = np.load(npz_path)
    latents      = data["latents"]
    episode_ids  = data["episode_id"]
    frame_starts = data["frame_start"]

    print(f"  Clustering {len(latents)} latents → {n_clusters} clusters ...")
    km     = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10, verbose=0)
    labels = km.fit_predict(latents)

    # Group by cluster
    cluster_seqs: dict[int, list[np.ndarray]] = {c: [] for c in range(n_clusters)}
    for c in tqdm(range(n_clusters), desc="  Sampling EEF seqs"):
        indices = np.where(labels == c)[0]
        if len(indices) == 0:
            continue
        sampled = rng.choice(indices, min(n_sample_per_cluster, len(indices)), replace=False)
        for idx in sampled:
            key = (int(episode_ids[idx]), int(frame_starts[idx]))
            if key in ee_map:
                cluster_seqs[c].append(ee_map[key])

    valid_clusters = [c for c, seqs in cluster_seqs.items() if len(seqs) >= 2]
    print(f"  Valid clusters (≥2 EEF seqs): {len(valid_clusters)}/{n_clusters}")

    # Intra-cluster metrics
    intra_list = []
    for c in valid_clusters:
        m = pairwise_ee_metrics(cluster_seqs[c], n_resample)
        if m:
            intra_list.append(m)

    intra_avg = {}
    for k in intra_list[0]:
        if k.endswith("_per_dim"):
            intra_avg[k] = np.mean([m[k] for m in intra_list], axis=0).tolist()
        else:
            intra_avg[k] = float(np.mean([m[k] for m in intra_list]))

    # Inter-cluster metrics: medoid per cluster (closest latent to centroid)
    print("  Finding medoids for inter-cluster comparison ...")
    centroids  = km.cluster_centers_
    inter_seqs = []
    for c in valid_clusters:
        indices    = np.where(labels == c)[0]
        dists      = np.linalg.norm(latents[indices] - centroids[c], axis=1)
        medoid_idx = indices[np.argmin(dists)]
        key        = (int(episode_ids[medoid_idx]), int(frame_starts[medoid_idx]))
        if key in ee_map:
            inter_seqs.append(ee_map[key])

    inter_avg = pairwise_ee_metrics(inter_seqs, n_resample)

    # Separation scores (scalars only)
    eps = 1e-8
    separation = {
        "direction_separation":    intra_avg["direction_cos_sim"] - inter_avg["direction_cos_sim"],
        "shape_l2_rel_separation": inter_avg["shape_l2_rel"]      / (intra_avg["shape_l2_rel"]      + eps),
        "shape_l2_abs_separation": inter_avg["shape_l2_abs"]      / (intra_avg["shape_l2_abs"]      + eps),
        "final_pose_separation":   inter_avg["final_pose_dist"]   / (intra_avg["final_pose_dist"]   + eps),
        "delta_pose_separation":   inter_avg["delta_pose_mse"]    / (intra_avg["delta_pose_mse"]    + eps),
        "dtw_separation":          inter_avg["dtw"]               / (intra_avg["dtw"]               + eps),
        "wasserstein_separation":  inter_avg["wasserstein"]       / (intra_avg["wasserstein"]       + eps),
    }

    return {
        "n_latents":      int(len(latents)),
        "n_clusters":     n_clusters,
        "valid_clusters": len(valid_clusters),
        "intra":          intra_avg,
        "inter":          inter_avg,
        "separation":     separation,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

_DIM_LABELS = ["pos_x", "pos_y", "pos_z", "ori_roll", "ori_pitch", "ori_yaw", "gripper"]
_COLORS     = ["#4C72B0", "#DD8452"]


def _bar_chart(ax, labels, vals_A, vals_B, name_A, name_B, fmt=".3f"):
    n     = len(labels)
    x     = np.arange(n)
    width = 0.35
    bA = ax.bar(x - width / 2, vals_A, width, label=name_A, color=_COLORS[0], alpha=0.85)
    bB = ax.bar(x + width / 2, vals_B, width, label=name_B, color=_COLORS[1], alpha=0.85)
    for bar in [*bA, *bB]:
        h = bar.get_height()
        ax.annotate(
            f"{h:{fmt}}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=7,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def plot_ee_comparison(
    results   : dict,
    name_A    : str,
    name_B    : str,
    n_clusters: int,
    out_dir   : Path,
) -> list:
    """Save three bar-chart PNGs (intra / inter / separation) for EEF metrics."""
    groups = {
        "intra": {
            "Direction cos-sim (↑)":  "direction_cos_sim",
            "Shape L2 rel (↓)":       "shape_l2_rel",
            "Shape L2 abs (↓)":       "shape_l2_abs",
            "Final pose dist (↓)":    "final_pose_dist",
            "Delta pose MSE (↓)":     "delta_pose_mse",
            "DTW (↓)":                "dtw",
            "Wasserstein (↓)":        "wasserstein",
        },
        "inter": {
            "Direction cos-sim (↓)":  "direction_cos_sim",
            "Shape L2 rel (↑)":       "shape_l2_rel",
            "Shape L2 abs (↑)":       "shape_l2_abs",
            "Final pose dist (↑)":    "final_pose_dist",
            "Delta pose MSE (↑)":     "delta_pose_mse",
            "DTW (↑)":                "dtw",
            "Wasserstein (↑)":        "wasserstein",
        },
        "separation": {
            "Direction sep (↑)":      "direction_separation",
            "Shape rel sep (↑)":      "shape_l2_rel_separation",
            "Shape abs sep (↑)":      "shape_l2_abs_separation",
            "Final pose sep (↑)":     "final_pose_separation",
            "Delta pose sep (↑)":     "delta_pose_separation",
            "DTW sep (↑)":            "dtw_separation",
            "Wasserstein sep (↑)":    "wasserstein_separation",
        },
    }
    titles = {
        "intra":      "EEF — Within-cluster similarity (actual trajectories)",
        "inter":      "EEF — Between-cluster distance (medoids)",
        "separation": "EEF — Separation scores",
    }
    saved = []
    for group_key, metric_map in groups.items():
        labels = list(metric_map.keys())
        keys   = list(metric_map.values())
        vals_A = [results[name_A][group_key][k] for k in keys]
        vals_B = [results[name_B][group_key][k] for k in keys]

        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2.0), 5))
        _bar_chart(ax, labels, vals_A, vals_B, name_A, name_B)
        ax.set_title(f"{titles[group_key]}  (K={n_clusters})", fontsize=12)
        fig.tight_layout()

        out_png = out_dir / f"latent_eval_k{n_clusters}_{group_key}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        saved.append(out_png)
        print(f"Saved → {out_png}")

    return saved


_EE_DIM_LABELS = ["ee_x", "ee_y", "ee_z"]


def plot_shape_l2_perdim_comparison(
    results   : dict,
    name_A    : str,
    name_B    : str,
    n_clusters: int,
    out_dir   : Path,
) -> list:
    """Two PNGs: per-dim shape L2 (rel and abs) for intra and inter, A vs B."""
    saved = []
    for variant in ("rel", "abs"):
        key_intra = f"shape_l2_{variant}_per_dim"
        key_inter = f"shape_l2_{variant}_per_dim"

        intra_A = np.array(results[name_A]["intra"][key_intra])
        intra_B = np.array(results[name_B]["intra"][key_intra])
        inter_A = np.array(results[name_A]["inter"][key_intra])
        inter_B = np.array(results[name_B]["inter"][key_intra])

        n_dims = len(intra_A)
        labels = _EE_DIM_LABELS[:n_dims]
        title_sfx = "relative (initial subtracted)" if variant == "rel" else "absolute"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, n_dims * 1.8), 9))
        _bar_chart(ax1, labels, intra_A, intra_B, name_A, name_B, fmt=".4f")
        ax1.set_title(f"Shape L2 {title_sfx} — Within-cluster (↓ better)  K={n_clusters}", fontsize=11)
        ax1.set_ylabel("MAE per step")
        _bar_chart(ax2, labels, inter_A, inter_B, name_A, name_B, fmt=".4f")
        ax2.set_title(f"Shape L2 {title_sfx} — Between-cluster medoids (↑ better)  K={n_clusters}", fontsize=11)
        ax2.set_ylabel("MAE per step")

        fig.tight_layout()
        out_png = out_dir / f"latent_eval_k{n_clusters}_shape_l2_{variant}_perdim.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        saved.append(out_png)
        print(f"Saved → {out_png}")

    return saved


def plot_recon_perdim_comparison(
    recon_results: dict,
    name_A       : str,
    name_B       : str,
    out_dir      : Path,
) -> Path:
    """2-subplot bar chart: L2 per step and MSE per action dimension."""
    l2_A  = np.array(recon_results[name_A]["recon_l2_per_dim"])
    l2_B  = np.array(recon_results[name_B]["recon_l2_per_dim"])
    mse_A = np.array(recon_results[name_A]["recon_mse_per_dim"])
    mse_B = np.array(recon_results[name_B]["recon_mse_per_dim"])

    n_dims = len(l2_A)
    labels = _DIM_LABELS[:n_dims]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, n_dims * 1.6), 9))
    _bar_chart(ax1, labels, l2_A, l2_B, name_A, name_B, fmt=".4f")
    ax1.set_title("Reconstruction L2 per step per dimension (↓ better)", fontsize=11)
    ax1.set_ylabel("MAE")
    _bar_chart(ax2, labels, mse_A, mse_B, name_A, name_B, fmt=".4f")
    ax2.set_title("Reconstruction MSE per dimension (↓ better)", fontsize=11)
    ax2.set_ylabel("MSE")

    fig.tight_layout()
    out_png = out_dir / "latent_eval_recon_perdim.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_png}")
    return out_png


def plot_recon_ctrl_perdim_comparison(
    ctrl_results: dict,
    name_A      : str,
    name_B      : str,
    out_dir     : Path,
) -> Path:
    """Bar chart: control-point L2 per action dimension, A vs B."""
    vals_A = np.array(ctrl_results[name_A]["ctrl_l2_per_dim"])
    vals_B = np.array(ctrl_results[name_B]["ctrl_l2_per_dim"])
    labels = ctrl_results[name_A]["dim_labels"][: len(vals_A)]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.6), 5))
    _bar_chart(ax, labels, vals_A, vals_B, name_A, name_B, fmt=".4f")
    ax.set_title("Reconstruction — Control-point L2 per dimension (normalized, ↓ better)", fontsize=11)
    ax.set_ylabel("MAE (normalized space)")
    ax.set_ylim(0, 0.2)
    fig.tight_layout()

    out_png = out_dir / "latent_eval_recon_ctrl_perdim.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_png}")
    return out_png


def plot_recon_dtw_comparison(
    ctrl_results: dict,
    name_A      : str,
    name_B      : str,
    out_dir     : Path,
) -> Path:
    """Bar chart: DTW reconstruction error, A vs B."""
    vals = [ctrl_results[name_A]["dtw_mean"], ctrl_results[name_B]["dtw_mean"]]

    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.bar([name_A, name_B], vals, color=_COLORS, alpha=0.85, width=0.4)
    for bar, v in zip(bars, vals):
        ax.annotate(
            f"{v:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_title("Reconstruction — DTW distance (raw deltas, ↓ better)", fontsize=11)
    ax.set_ylabel("DTW (normalized by length)")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    out_png = out_dir / "latent_eval_recon_dtw.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_png}")
    return out_png


def plot_dataset_stats(
    stats  : dict,
    name_A : str,
    name_B : str,
    out_dir: Path,
) -> Path:
    """Bar chart comparing skill count and average skill length."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
    for ax, key, title, fmt in [
        (ax1, "n_skills",         "Total skill count",         "{:.0f}"),
        (ax2, "avg_skill_length", "Avg skill length (frames)", "{:.1f}"),
    ]:
        vals = [stats[name_A][key], stats[name_B][key]]
        bars = ax.bar([name_A, name_B], vals, color=_COLORS, alpha=0.85, width=0.4)
        for bar, v in zip(bars, vals):
            ax.annotate(
                fmt.format(v),
                xy=(bar.get_x() + bar.get_width() / 2, v),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=10,
            )
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(0, max(vals) * 1.2)

    fig.tight_layout()
    out_png = out_dir / "latent_eval_dataset_stats.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_png}")
    return out_png


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_A",        required=True)
    parser.add_argument("--vae_A",        required=True)
    parser.add_argument("--name_A",       default="fixed50")
    parser.add_argument("--npz_B",        required=True)
    parser.add_argument("--vae_B",        required=True)
    parser.add_argument("--name_B",       default="variable")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--n_clusters",   type=int, default=20)
    parser.add_argument("--n_sample",     type=int, default=10,
                        help="Skills sampled per cluster for intra metrics")
    parser.add_argument("--n_resample",   type=int, default=80,
                        help="Steps for shape L2 resampling")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--n_recon_eval", type=int, default=500,
                        help="Skills sampled for reconstruction error eval")
    parser.add_argument("--wandb_enable",   action="store_true")
    parser.add_argument("--wandb_project",  default="VAE_eval")
    parser.add_argument("--wandb_job_name", default="latent_eval")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.wandb_enable:
        wandb.init(project=args.wandb_project, name=args.wandb_job_name, config=vars(args))

    data_A = np.load(args.npz_A)
    data_B = np.load(args.npz_B)
    all_ep = np.concatenate([data_A["episode_id"],  data_B["episode_id"]])
    all_fs = np.concatenate([data_A["frame_start"], data_B["frame_start"]])
    all_fe = np.concatenate([data_A["frame_end"],   data_B["frame_end"]])

    # Dataset-level stats
    dataset_stats = {}
    for name, data in [(args.name_A, data_A), (args.name_B, data_B)]:
        lengths = data["frame_end"] - data["frame_start"]
        dataset_stats[name] = {
            "n_skills":         int(len(data["latents"])),
            "avg_skill_length": float(np.mean(lengths)),
        }

    # Load EEF sequences (for clustering eval)
    print("Loading EEF state sequences from dataset ...")
    ee_map = load_ee_state_sequences(args.dataset_path, all_ep, all_fs, all_fe)
    print(f"  Loaded {len(ee_map)} skill EEF sequences")

    # Load action sequences + start states (for reconstruction eval)
    print("Loading action sequences from dataset ...")
    action_map = load_action_sequences(args.dataset_path, all_ep, all_fs, all_fe)
    print(f"  Loaded {len(action_map)} skill action sequences")

    print("Loading observation states from dataset ...")
    state_map = load_skill_states(args.dataset_path, all_ep, all_fs)
    print(f"  Loaded {len(state_map)} unique (episode, frame) states")

    results = {}
    recon_results = {}
    ctrl_results  = {}
    for name, npz, vae_ckpt in [
        (args.name_A, args.npz_A, args.vae_A),
        (args.name_B, args.npz_B, args.vae_B),
    ]:
        print(f"\n=== Evaluating: {name} ===")
        results[name] = evaluate_ee(
            npz_path             = npz,
            ee_map               = ee_map,
            n_clusters           = args.n_clusters,
            n_sample_per_cluster = args.n_sample,
            n_resample           = args.n_resample,
            seed                 = args.seed,
        )
        print(f"\n--- Reconstruction eval: {name} ---")
        recon_results[name] = evaluate_reconstruction(
            npz_path   = npz,
            vae_ckpt   = vae_ckpt,
            action_map = action_map,
            state_map  = state_map,
            n_eval     = args.n_recon_eval,
            n_resample = args.n_resample,
            seed       = args.seed,
        )
        r = recon_results[name]
        print(f"  recon_mse={r['recon_mse']:.4f}  "
              f"recon_l2={r['recon_l2_per_step']:.4f}  "
              f"recon_dir_cos={r['recon_direction_cos']:.4f}")

        print(f"\n--- Control-point + DTW eval: {name} ---")
        ctrl_results[name] = evaluate_reconstruction_ctrl(
            npz_path   = npz,
            vae_ckpt   = vae_ckpt,
            action_map = action_map,
            state_map  = state_map,
            n_eval     = args.n_recon_eval,
            seed       = args.seed,
        )
        c = ctrl_results[name]
        print(f"  ctrl_l2={c['ctrl_l2']:.4f}  dtw_mean={c['dtw_mean']:.4f}")

    # Print comparison table
    print("\n" + "=" * 70)
    print(f"{'Metric':<30} {args.name_A:>16} {args.name_B:>16}")
    print("=" * 70)
    table_rows = [
        ("intra", "direction_cos_sim",     "Intra direction (↑)"),
        ("intra", "shape_l2_rel",          "Intra shape L2 rel (↓)"),
        ("intra", "shape_l2_abs",          "Intra shape L2 abs (↓)"),
        ("intra", "final_pose_dist",       "Intra final pose (↓)"),
        ("intra", "delta_pose_mse",        "Intra delta pose (↓)"),
        ("intra", "dtw",                   "Intra DTW (↓)"),
        ("intra", "wasserstein",           "Intra Wasserstein (↓)"),
        ("inter", "direction_cos_sim",     "Inter direction (↓)"),
        ("inter", "shape_l2_rel",          "Inter shape L2 rel (↑)"),
        ("inter", "shape_l2_abs",          "Inter shape L2 abs (↑)"),
        ("inter", "final_pose_dist",       "Inter final pose (↑)"),
        ("inter", "delta_pose_mse",        "Inter delta pose (↑)"),
        ("inter", "dtw",                   "Inter DTW (↑)"),
        ("inter", "wasserstein",           "Inter Wasserstein (↑)"),
        ("separation", "direction_separation",    "Sep direction (↑)"),
        ("separation", "shape_l2_rel_separation", "Sep shape rel (↑)"),
        ("separation", "shape_l2_abs_separation", "Sep shape abs (↑)"),
        ("separation", "final_pose_separation",   "Sep final pose (↑)"),
        ("separation", "delta_pose_separation",   "Sep delta pose (↑)"),
        ("separation", "dtw_separation",          "Sep DTW (↑)"),
        ("separation", "wasserstein_separation",  "Sep Wasserstein (↑)"),
    ]
    for group, key, label in table_rows:
        va = results[args.name_A][group][key]
        vb = results[args.name_B][group][key]
        print(f"  {label:<28} {va:>16.4f} {vb:>16.4f}")
    print("=" * 70)

    # Save JSON
    out_path = out_dir / f"latent_eval_k{args.n_clusters}.json"
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Save charts
    ee_pngs         = plot_ee_comparison(results, args.name_A, args.name_B, args.n_clusters, out_dir)
    perdim_pngs     = plot_shape_l2_perdim_comparison(results, args.name_A, args.name_B, args.n_clusters, out_dir)
    recon_png       = plot_recon_perdim_comparison(recon_results, args.name_A, args.name_B, out_dir)
    ctrl_perdim_png = plot_recon_ctrl_perdim_comparison(ctrl_results, args.name_A, args.name_B, out_dir)
    dtw_png         = plot_recon_dtw_comparison(ctrl_results, args.name_A, args.name_B, out_dir)
    stats_png       = plot_dataset_stats(dataset_stats, args.name_A, args.name_B, out_dir)

    # wandb logging
    if args.wandb_enable:
        log_dict = {}
        for png in ee_pngs:
            tag = png.stem.split("_", 3)[-1]       # intra / inter / separation
            log_dict[f"k{args.n_clusters}/chart/{tag}"] = wandb.Image(str(png))
        for png in perdim_pngs:
            # stem: latent_eval_k20_shape_l2_rel_perdim → tag: shape_l2_rel_perdim
            tag = png.stem.split(f"k{args.n_clusters}_", 1)[-1]
            log_dict[f"k{args.n_clusters}/chart/{tag}"] = wandb.Image(str(png))
        log_dict["recon/chart_perdim"]      = wandb.Image(str(recon_png))
        log_dict["recon/chart_ctrl_perdim"] = wandb.Image(str(ctrl_perdim_png))
        log_dict["recon/chart_dtw"]         = wandb.Image(str(dtw_png))
        log_dict["dataset/chart_stats"]     = wandb.Image(str(stats_png))
        wandb.log(log_dict)


if __name__ == "__main__":
    main()
