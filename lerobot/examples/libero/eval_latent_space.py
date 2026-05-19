"""
eval_latent_space.py  (VQAE edition)

Compare two VQAE latent spaces (e.g. fixed-length vs variable-length skill segmentation)
using natural token clusters — no KMeans.

Pipeline:
  Load NPZ (tokens + metadata)
  → group skills by token index (natural clusters from codebook)
  → for each token group, compute pairwise EEF trajectory metrics (intra)
  → pick one representative per token, compute pairwise metrics (inter)
  → compare A vs B

3 evaluation blocks per system:
  1. Token cluster quality  : intra (same token) vs inter (different token) EEF metrics
  2. Reconstruction quality : decode(token, own_initial_state) vs GT action sequence
  3. Codebook utilization   : # tokens used, skills-per-token distribution

6 EEF metrics (intra / inter):
  direction    : cosine similarity of start→end displacement (xyz)
  shape_rel    : mean pairwise L2 of relative (start-subtracted) EEF trajectory
  shape_abs    : mean pairwise L2 of absolute EEF trajectory
  final_pose   : Euclidean distance of final EEF position (xyz)
  delta_pose   : MSE of total displacement vector (xyz)
  dtw          : DTW distance of relative EEF trajectory (normalized by length)
  wasserstein  : mean per-dim Wasserstein distance of per-step EEF deltas

Separation score:
  direction    : intra_mean - inter_mean  (higher → better)
  others       : inter_mean / intra_mean  (higher → better)

Usage:
  python examples/libero/eval_latent_space.py \\
    --npz_A  .../spline_vqae_latents_epoch5000.npz \\
    --vae_A  .../spline_vqae_epoch5000.pt \\
    --npz_B  .../spline_vqae_latents_epoch5000.npz \\
    --vae_B  .../spline_vqae_epoch5000.pt \\
    --dataset_path .../libero_dataset/libero_90 \\
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from FSQ import GRIPPER_DIM, SplineFSQAE, spline_decode, spline_encode  # noqa: E402


# ── VQAE loading ──────────────────────────────────────────────────────────────

def load_vae(ckpt_path: str) -> SplineFSQAE:
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg   = ckpt["cfg"]
    cfg_d = dataclasses.asdict(cfg)
    keys  = {"action_dim", "state_dim", "n_control", "spline_degree",
             "hidden_dim", "latent_dim", "num_embeddings", "num_layers",
             "dropout", "commitment_cost", "max_length", "action_min", "action_max"}
    model = SplineFSQAE(**{k: v for k, v in cfg_d.items() if k in keys})
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


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
    needed: dict[int, dict[int, int]] = {}
    for ep, fs, fe in zip(episode_ids.tolist(), frame_starts.tolist(), frame_ends.tolist()):
        needed.setdefault(int(ep), {})[int(fs)] = int(fe)
    return _load_seq_from_parquet(
        dataset_path, needed, "observation.states.ee_state", "Loading EEF sequences"
    )


# ── EEF trajectory metrics ────────────────────────────────────────────────────

def _resample(seq: np.ndarray, n: int) -> np.ndarray:
    T = len(seq)
    if T == n:
        return seq
    t_orig = np.linspace(0.0, 1.0, T)
    t_new  = np.linspace(0.0, 1.0, n)
    return np.stack(
        [np.interp(t_new, t_orig, seq[:, d]) for d in range(seq.shape[1])], axis=-1
    ).astype(np.float32)


def _dtw_normalized(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    prev = np.full(nb + 1, np.inf)
    prev[0] = 0.0
    for i in range(na):
        curr = np.full(nb + 1, np.inf)
        for j in range(nb):
            curr[j + 1] = cost[i, j] + min(prev[j + 1], curr[j], prev[j])
        prev = curr
    return float(prev[nb]) / ((na + nb) / 2)


def _wasserstein_mean(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import wasserstein_distance
    return float(np.mean([
        wasserstein_distance(a[:, d], b[:, d]) for d in range(a.shape[1])
    ]))


def pairwise_ee_metrics(
    ee_seqs   : list[np.ndarray],
    n_resample: int,
) -> dict | None:
    """Pairwise EEF metrics between GT EEF state sequences (each (T_i, 3) xyz)."""
    pairs = list(combinations(range(len(ee_seqs)), 2))
    if not pairs:
        return None

    dir_sims, final_dists, delta_mses, dtw_dists, was_dists = [], [], [], [], []
    shape_rel_all, shape_abs_all = [], []
    shape_rel_dim_all, shape_abs_dim_all = [], []

    for i, j in pairs:
        si = ee_seqs[i][:, :3].astype(np.float32)
        sj = ee_seqs[j][:, :3].astype(np.float32)

        # Relative (start-subtracted)
        rel_i = si - si[0]
        rel_j = sj - sj[0]

        # 1. Direction: cos-sim of total displacement (xyz)
        disp_i, disp_j = rel_i[-1], rel_j[-1]
        dir_sims.append(float(
            np.dot(disp_i, disp_j) / (np.linalg.norm(disp_i) * np.linalg.norm(disp_j) + 1e-8)
        ))

        # 2a. Shape L2 relative
        rri = _resample(rel_i, n_resample)
        rrj = _resample(rel_j, n_resample)
        diff_rel = rri - rrj
        shape_rel_all.append(float(np.mean(np.linalg.norm(diff_rel, axis=-1))))
        shape_rel_dim_all.append(np.mean(np.abs(diff_rel), axis=0))

        # 2b. Shape L2 absolute
        rai = _resample(si, n_resample)
        raj = _resample(sj, n_resample)
        diff_abs = rai - raj
        shape_abs_all.append(float(np.mean(np.linalg.norm(diff_abs, axis=-1))))
        shape_abs_dim_all.append(np.mean(np.abs(diff_abs), axis=0))

        # 3. Final pose
        final_dists.append(float(np.linalg.norm(si[-1] - sj[-1])))

        # 4. Delta pose MSE
        delta_mses.append(float(np.mean((disp_i - disp_j) ** 2)))

        # 5. DTW on relative trajectory
        dtw_dists.append(_dtw_normalized(rel_i, rel_j))

        # 6. Wasserstein on per-step EEF deltas
        was_dists.append(_wasserstein_mean(np.diff(si, axis=0), np.diff(sj, axis=0)))

    return {
        "direction_cos_sim":    float(np.mean(dir_sims)),
        "shape_l2_rel":         float(np.mean(shape_rel_all)),
        "shape_l2_abs":         float(np.mean(shape_abs_all)),
        "shape_l2_rel_per_dim": np.mean(shape_rel_dim_all, axis=0).tolist(),
        "shape_l2_abs_per_dim": np.mean(shape_abs_dim_all, axis=0).tolist(),
        "final_pose_dist":      float(np.mean(final_dists)),
        "delta_pose_mse":       float(np.mean(delta_mses)),
        "dtw":                  float(np.mean(dtw_dists)),
        "wasserstein":          float(np.mean(was_dists)),
    }


# ── Token cluster evaluation ──────────────────────────────────────────────────

def evaluate_token_clusters(
    npz_path            : str,
    ee_map              : dict[tuple[int, int], np.ndarray],
    vae_num_embeddings  : int,
    n_sample_per_token  : int = 10,
    n_resample          : int = 80,
    seed                : int = 42,
) -> dict:
    """Group skills by token index (no KMeans), compute intra/inter EEF metrics."""
    rng  = np.random.default_rng(seed)
    data = np.load(npz_path)
    tokens       = data["tokens"].astype(np.int32)     # (N,) — natural clusters
    episode_ids  = data["episode_id"]
    frame_starts = data["frame_start"]

    unique_tokens = np.unique(tokens)
    n_used = len(unique_tokens)
    print(f"  Codebook utilization: {n_used} / {vae_num_embeddings} tokens used "
          f"({100 * n_used / vae_num_embeddings:.1f}%)")

    # Build per-token EEF sequences (sampled)
    token_seqs: dict[int, list[np.ndarray]] = {}
    for tok in tqdm(unique_tokens, desc="  Grouping by token"):
        indices = np.where(tokens == tok)[0]
        sample  = rng.choice(indices, min(n_sample_per_token, len(indices)), replace=False)
        seqs = [ee_map[(int(episode_ids[i]), int(frame_starts[i]))]
                for i in sample
                if (int(episode_ids[i]), int(frame_starts[i])) in ee_map]
        if seqs:
            token_seqs[tok] = seqs

    valid_tokens = [t for t, seqs in token_seqs.items() if len(seqs) >= 2]
    print(f"  Valid tokens (≥2 EEF seqs): {len(valid_tokens)} / {n_used}")

    # ── Intra-token: pairwise GT trajectory similarity ────────────────────────
    intra_list = []
    for tok in tqdm(valid_tokens, desc="  Intra metrics"):
        m = pairwise_ee_metrics(token_seqs[tok], n_resample)
        if m:
            intra_list.append(m)

    intra_avg: dict = {}
    for k in intra_list[0]:
        vals = [m[k] for m in intra_list]
        intra_avg[k] = np.mean(vals, axis=0).tolist() if k.endswith("_per_dim") else float(np.mean(vals))

    # ── Inter-token: one representative skill per token ───────────────────────
    inter_seqs = []
    for tok in valid_tokens:
        indices = np.where(tokens == tok)[0]
        rep_idx = int(rng.choice(indices))
        key = (int(episode_ids[rep_idx]), int(frame_starts[rep_idx]))
        if key in ee_map:
            inter_seqs.append(ee_map[key])

    inter_avg = pairwise_ee_metrics(inter_seqs, n_resample)

    # ── Separation scores ─────────────────────────────────────────────────────
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

    # ── Codebook utilization stats ────────────────────────────────────────────
    counts = np.array([int(np.sum(tokens == t)) for t in unique_tokens])
    utilization = {
        "n_tokens_used":           int(n_used),
        "n_tokens_total":          int(vae_num_embeddings),
        "utilization_pct":         float(100 * n_used / vae_num_embeddings),
        "skills_per_token_mean":   float(np.mean(counts)),
        "skills_per_token_std":    float(np.std(counts)),
        "skills_per_token_min":    int(np.min(counts)),
        "skills_per_token_max":    int(np.max(counts)),
        "skills_per_token_median": float(np.median(counts)),
    }

    return {
        "n_skills":     int(len(tokens)),
        "valid_tokens": int(len(valid_tokens)),
        "intra":        intra_avg,
        "inter":        inter_avg,
        "separation":   separation,
        "utilization":  utilization,
        "token_counts": counts.tolist(),   # skills per used token (for histogram)
    }


# ── Reconstruction evaluation ─────────────────────────────────────────────────

def process_raw_traj(raw_traj: np.ndarray, n_resample: int) -> np.ndarray:
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


def decode_token_and_process(
    vae        : SplineFSQAE,
    token      : int,
    start_state: np.ndarray,
    n_resample : int = 100,
) -> np.ndarray:
    """decode(token, own_initial_state) → resampled cumulative trajectory."""
    with torch.no_grad():
        idx_t = torch.tensor([token], dtype=torch.long)
        z_q   = vae.quantizer.embedding(idx_t)          # (1, D)
        s_t   = torch.from_numpy(start_state).unsqueeze(0).float()
        ctrl_pts_norm, len_norm = vae.decode(z_q, s_t)

    ctrl_np = ctrl_pts_norm[0].cpu().numpy()
    len_np  = float(len_norm[0, 0].cpu().numpy())
    lo = vae.action_min.cpu().numpy()
    hi = vae.action_max.cpu().numpy()
    gripper_idx = (vae.action_dim + GRIPPER_DIM) % vae.action_dim

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
    """decode(token, own_initial_state) vs GT action sequence."""
    rng  = np.random.default_rng(seed)
    data = np.load(npz_path)
    tokens       = data["tokens"].astype(np.int32)
    episode_ids  = data["episode_id"]
    frame_starts = data["frame_start"]

    vae = load_vae(vae_ckpt)

    available = [
        i for i in range(len(tokens))
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
        dec_traj  = decode_token_and_process(vae, int(tokens[idx]), start_state, n_resample)

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
        "n_eval":              int(len(sampled)),
    }


def evaluate_reconstruction_ctrl(
    npz_path   : str,
    vae_ckpt   : str,
    action_map : dict[tuple[int, int], np.ndarray],
    state_map  : dict[tuple[int, int], np.ndarray],
    n_eval     : int = 500,
    seed       : int = 42,
) -> dict:
    """Control-point L2 and DTW on raw deltas: decoded vs original."""
    rng  = np.random.default_rng(seed)
    data = np.load(npz_path)
    tokens       = data["tokens"].astype(np.int32)
    episode_ids  = data["episode_id"]
    frame_starts = data["frame_start"]

    vae = load_vae(vae_ckpt)
    lo  = vae.action_min.cpu().numpy()
    hi  = vae.action_max.cpu().numpy()

    available = [
        i for i in range(len(tokens))
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
        orig_raw    = action_map[key]

        orig_ctrl, _ = spline_encode(orig_raw, vae.n_control, vae.spline_degree)
        orig_ctrl_norm = 2 * (orig_ctrl - lo) / (hi - lo + 1e-8) - 1

        with torch.no_grad():
            idx_t = torch.tensor([int(tokens[idx])], dtype=torch.long)
            z_q   = vae.quantizer.embedding(idx_t)
            s_t   = torch.from_numpy(start_state).unsqueeze(0).float()
            dec_ctrl_norm, _ = vae.decode(z_q, s_t)
        dec_ctrl_norm = dec_ctrl_norm[0].cpu().numpy()

        gripper_idx = (vae.action_dim + GRIPPER_DIM) % vae.action_dim
        diff_ctrl   = orig_ctrl_norm[:, :gripper_idx] - dec_ctrl_norm[:, :gripper_idx]
        ctrl_l2_per_dim_list.append(np.mean(np.abs(diff_ctrl), axis=0))

        orig_no_grip = orig_raw[:, :gripper_idx].astype(np.float32)
        dec_raw = spline_decode(
            (dec_ctrl_norm[:, :gripper_idx] + 1) / 2
            * (hi[:gripper_idx] - lo[:gripper_idx] + 1e-8) + lo[:gripper_idx],
            max(2, orig_raw.shape[0]),
            vae.spline_degree,
        )
        dtw_list.append(_dtw_normalized(orig_no_grip, dec_raw))

    action_dim_labels = ["dx", "dy", "dz", "droll", "dpitch", "dyaw"]
    n_dims = ctrl_l2_per_dim_list[0].shape[0]

    return {
        "ctrl_l2_per_dim": np.mean(ctrl_l2_per_dim_list, axis=0).tolist(),
        "ctrl_l2":         float(np.mean([v.mean() for v in ctrl_l2_per_dim_list])),
        "dtw_mean":        float(np.mean(dtw_list)),
        "n_eval":          int(len(sampled)),
        "dim_labels":      action_dim_labels[:n_dims],
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

_DIM_LABELS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
_EE_DIM_LABELS = ["ee_x", "ee_y", "ee_z"]
_COLORS = ["#4C72B0", "#DD8452"]


def _bar_chart(ax, labels, vals_A, vals_B, name_A, name_B, fmt=".3f"):
    x     = np.arange(len(labels))
    width = 0.35
    bA = ax.bar(x - width / 2, vals_A, width, label=name_A, color=_COLORS[0], alpha=0.85)
    bB = ax.bar(x + width / 2, vals_B, width, label=name_B, color=_COLORS[1], alpha=0.85)
    for bar in [*bA, *bB]:
        h = bar.get_height()
        ax.annotate(f"{h:{fmt}}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def plot_token_cluster_comparison(
    results: dict, name_A: str, name_B: str, out_dir: Path,
) -> list:
    groups = {
        "intra": {
            "Direction (↑)":    "direction_cos_sim",
            "Shape rel (↓)":    "shape_l2_rel",
            "Shape abs (↓)":    "shape_l2_abs",
            "Final pose (↓)":   "final_pose_dist",
            "Delta pose (↓)":   "delta_pose_mse",
            "DTW (↓)":          "dtw",
            "Wasserstein (↓)":  "wasserstein",
        },
        "inter": {
            "Direction (↓)":    "direction_cos_sim",
            "Shape rel (↑)":    "shape_l2_rel",
            "Shape abs (↑)":    "shape_l2_abs",
            "Final pose (↑)":   "final_pose_dist",
            "Delta pose (↑)":   "delta_pose_mse",
            "DTW (↑)":          "dtw",
            "Wasserstein (↑)":  "wasserstein",
        },
        "separation": {
            "Direction (↑)":    "direction_separation",
            "Shape rel (↑)":    "shape_l2_rel_separation",
            "Shape abs (↑)":    "shape_l2_abs_separation",
            "Final pose (↑)":   "final_pose_separation",
            "Delta pose (↑)":   "delta_pose_separation",
            "DTW (↑)":          "dtw_separation",
            "Wasserstein (↑)":  "wasserstein_separation",
        },
    }
    titles = {
        "intra":      "Within-token EEF similarity — GT trajectories (same token)",
        "inter":      "Between-token EEF distance — GT trajectories (different tokens)",
        "separation": "Separation scores (inter/intra ratio, higher → more distinct tokens)",
    }
    saved = []
    for gk, metric_map in groups.items():
        labels = list(metric_map.keys())
        keys   = list(metric_map.values())
        vals_A = [results[name_A][gk][k] for k in keys]
        vals_B = [results[name_B][gk][k] for k in keys]
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 2.0), 5))
        _bar_chart(ax, labels, vals_A, vals_B, name_A, name_B)
        ax.set_title(titles[gk], fontsize=11)
        fig.tight_layout()
        out_png = out_dir / f"token_cluster_{gk}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        saved.append(out_png)
        print(f"Saved → {out_png}")
    return saved


def plot_codebook_utilization(
    results: dict, name_A: str, name_B: str, out_dir: Path,
) -> list:
    """Codebook utilization bars + skills-per-token histogram."""
    saved = []

    # 1. Summary bar chart
    u_A = results[name_A]["utilization"]
    u_B = results[name_B]["utilization"]
    metrics = ["utilization_pct", "skills_per_token_mean", "skills_per_token_std",
               "skills_per_token_median"]
    labels  = ["Utilization %", "Skills/token (mean)", "Skills/token (std)", "Skills/token (median)"]
    vA = [u_A[m] for m in metrics]
    vB = [u_B[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    _bar_chart(ax, labels, vA, vB, name_A, name_B, fmt=".1f")
    ax.set_title("Codebook utilization summary", fontsize=11)
    fig.tight_layout()
    p = out_dir / "codebook_utilization_summary.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    saved.append(p); print(f"Saved → {p}")

    # 2. Skills-per-token histogram (one panel per system)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, name in zip(axes, [name_A, name_B]):
        counts = results[name]["token_counts"]
        ax.hist(counts, bins=30, color=_COLORS[0] if name == name_A else _COLORS[1],
                alpha=0.75, edgecolor="white")
        ax.set_title(f"{name} — skills per token (n={u_A['n_tokens_used'] if name == name_A else u_B['n_tokens_used']} tokens used)", fontsize=10)
        ax.set_xlabel("Skills per token")
        ax.set_ylabel("# tokens")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.suptitle("Skills-per-token distribution", fontsize=12)
    fig.tight_layout()
    p = out_dir / "codebook_utilization_histogram.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    saved.append(p); print(f"Saved → {p}")

    return saved


def plot_recon_perdim(recon: dict, name_A: str, name_B: str, out_dir: Path) -> Path:
    l2_A  = np.array(recon[name_A]["recon_l2_per_dim"])
    l2_B  = np.array(recon[name_B]["recon_l2_per_dim"])
    mse_A = np.array(recon[name_A]["recon_mse_per_dim"])
    mse_B = np.array(recon[name_B]["recon_mse_per_dim"])
    n_dims = len(l2_A)
    labels = _DIM_LABELS[:n_dims]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, n_dims * 1.6), 9))
    _bar_chart(ax1, labels, l2_A, l2_B, name_A, name_B, fmt=".4f")
    ax1.set_title("Reconstruction L2 per step per action dim (↓ better)", fontsize=11)
    ax1.set_ylabel("MAE")
    _bar_chart(ax2, labels, mse_A, mse_B, name_A, name_B, fmt=".4f")
    ax2.set_title("Reconstruction MSE per action dim (↓ better)", fontsize=11)
    ax2.set_ylabel("MSE")
    fig.tight_layout()
    p = out_dir / "recon_perdim.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved → {p}")
    return p


def plot_recon_ctrl_perdim(ctrl: dict, name_A: str, name_B: str, out_dir: Path) -> Path:
    vA = np.array(ctrl[name_A]["ctrl_l2_per_dim"])
    vB = np.array(ctrl[name_B]["ctrl_l2_per_dim"])
    labels = ctrl[name_A]["dim_labels"][:len(vA)]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.6), 5))
    _bar_chart(ax, labels, vA, vB, name_A, name_B, fmt=".4f")
    ax.set_title("Control-point L2 per action dim (normalized, ↓ better)", fontsize=11)
    ax.set_ylabel("MAE (normalized)")
    fig.tight_layout()
    p = out_dir / "recon_ctrl_perdim.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved → {p}")
    return p


def plot_dataset_stats(stats: dict, name_A: str, name_B: str, out_dir: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
    for ax, key, title, fmt in [
        (ax1, "n_skills",         "Total skill count",         "{:.0f}"),
        (ax2, "avg_skill_length", "Avg skill length (frames)", "{:.1f}"),
    ]:
        vals = [stats[name_A][key], stats[name_B][key]]
        bars = ax.bar([name_A, name_B], vals, color=_COLORS, alpha=0.85, width=0.4)
        for bar, v in zip(bars, vals):
            ax.annotate(fmt.format(v),
                        xy=(bar.get_x() + bar.get_width() / 2, v),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(0, max(vals) * 1.2)
    fig.tight_layout()
    p = out_dir / "dataset_stats.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved → {p}")
    return p


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
    parser.add_argument("--n_sample",     type=int, default=10,
                        help="Skills sampled per token for intra metrics")
    parser.add_argument("--n_resample",   type=int, default=80,
                        help="Steps for shape L2 resampling")
    parser.add_argument("--n_recon_eval", type=int, default=500,
                        help="Skills sampled for reconstruction eval")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--wandb_enable",   action="store_true")
    parser.add_argument("--wandb_project",  default="VAE_eval")
    parser.add_argument("--wandb_job_name", default="vqae_latent_eval")
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
    dataset_stats: dict = {}
    for name, data in [(args.name_A, data_A), (args.name_B, data_B)]:
        lengths = data["frame_end"] - data["frame_start"]
        dataset_stats[name] = {
            "n_skills":         int(len(data["tokens"])),
            "avg_skill_length": float(np.mean(lengths)),
        }

    print("Loading EEF state sequences ...")
    ee_map = load_ee_state_sequences(args.dataset_path, all_ep, all_fs, all_fe)
    print(f"  Loaded {len(ee_map)} EEF sequences")

    print("Loading action sequences ...")
    action_map = load_action_sequences(args.dataset_path, all_ep, all_fs, all_fe)
    print(f"  Loaded {len(action_map)} action sequences")

    print("Loading initial states ...")
    state_map = load_skill_states(args.dataset_path, all_ep, all_fs)
    print(f"  Loaded {len(state_map)} initial states")

    token_results: dict = {}
    recon_results: dict = {}
    ctrl_results:  dict = {}

    for name, npz, vae_ckpt in [
        (args.name_A, args.npz_A, args.vae_A),
        (args.name_B, args.npz_B, args.vae_B),
    ]:
        print(f"\n=== {name} ===")
        vae = load_vae(vae_ckpt)

        print(f"--- Token cluster eval: {name} ---")
        token_results[name] = evaluate_token_clusters(
            npz_path           = npz,
            ee_map             = ee_map,
            vae_num_embeddings = vae.num_embeddings,
            n_sample_per_token = args.n_sample,
            n_resample         = args.n_resample,
            seed               = args.seed,
        )
        u = token_results[name]["utilization"]
        print(f"  utilization={u['utilization_pct']:.1f}%  "
              f"skills/token: mean={u['skills_per_token_mean']:.1f} "
              f"std={u['skills_per_token_std']:.1f}")

        print(f"--- Reconstruction eval: {name} ---")
        recon_results[name] = evaluate_reconstruction(
            npz_path=npz, vae_ckpt=vae_ckpt,
            action_map=action_map, state_map=state_map,
            n_eval=args.n_recon_eval, n_resample=args.n_resample, seed=args.seed,
        )
        r = recon_results[name]
        print(f"  mse={r['recon_mse']:.4f}  l2={r['recon_l2_per_step']:.4f}  "
              f"dir_cos={r['recon_direction_cos']:.4f}")

        print(f"--- Control-point + DTW eval: {name} ---")
        ctrl_results[name] = evaluate_reconstruction_ctrl(
            npz_path=npz, vae_ckpt=vae_ckpt,
            action_map=action_map, state_map=state_map,
            n_eval=args.n_recon_eval, seed=args.seed,
        )
        c = ctrl_results[name]
        print(f"  ctrl_l2={c['ctrl_l2']:.4f}  dtw={c['dtw_mean']:.4f}")

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Metric':<32} {args.name_A:>16} {args.name_B:>16}")
    print("=" * 70)
    rows = [
        ("intra", "direction_cos_sim",     "Intra direction (↑)"),
        ("intra", "shape_l2_rel",          "Intra shape L2 rel (↓)"),
        ("intra", "shape_l2_abs",          "Intra shape L2 abs (↓)"),
        ("intra", "dtw",                   "Intra DTW (↓)"),
        ("intra", "wasserstein",           "Intra Wasserstein (↓)"),
        ("inter", "direction_cos_sim",     "Inter direction (↓)"),
        ("inter", "shape_l2_rel",          "Inter shape L2 rel (↑)"),
        ("inter", "shape_l2_abs",          "Inter shape L2 abs (↑)"),
        ("inter", "dtw",                   "Inter DTW (↑)"),
        ("separation", "direction_separation",    "Sep direction (↑)"),
        ("separation", "shape_l2_rel_separation", "Sep shape rel (↑)"),
        ("separation", "dtw_separation",          "Sep DTW (↑)"),
        ("separation", "wasserstein_separation",  "Sep Wasserstein (↑)"),
    ]
    for group, key, label in rows:
        va = token_results[args.name_A][group][key]
        vb = token_results[args.name_B][group][key]
        print(f"  {label:<30} {va:>16.4f} {vb:>16.4f}")
    print("=" * 70)

    # Save JSON
    out_json = out_dir / "latent_eval.json"
    with open(out_json, "w") as f:
        json.dump({
            "args": vars(args),
            "token_clusters": token_results,
            "reconstruction": recon_results,
        }, f, indent=2)
    print(f"\nSaved → {out_json}")

    # Plots
    cluster_pngs = plot_token_cluster_comparison(token_results, args.name_A, args.name_B, out_dir)
    util_pngs    = plot_codebook_utilization(token_results, args.name_A, args.name_B, out_dir)
    recon_png    = plot_recon_perdim(recon_results, args.name_A, args.name_B, out_dir)
    ctrl_png     = plot_recon_ctrl_perdim(ctrl_results, args.name_A, args.name_B, out_dir)
    stats_png    = plot_dataset_stats(dataset_stats, args.name_A, args.name_B, out_dir)

    if args.wandb_enable:
        log_dict: dict = {}
        for png in cluster_pngs:
            log_dict[f"cluster/{png.stem}"] = wandb.Image(str(png))
        for png in util_pngs:
            log_dict[f"utilization/{png.stem}"] = wandb.Image(str(png))
        log_dict["recon/perdim"]      = wandb.Image(str(recon_png))
        log_dict["recon/ctrl_perdim"] = wandb.Image(str(ctrl_png))
        log_dict["dataset/stats"]     = wandb.Image(str(stats_png))

        flat: dict = {}
        for name in [args.name_A, args.name_B]:
            u  = token_results[name]["utilization"]
            r  = recon_results[name]
            flat[f"{name}/utilization_pct"]       = u["utilization_pct"]
            flat[f"{name}/skills_per_token_mean"] = u["skills_per_token_mean"]
            flat[f"{name}/intra_direction"]        = token_results[name]["intra"]["direction_cos_sim"]
            flat[f"{name}/intra_shape_rel"]        = token_results[name]["intra"]["shape_l2_rel"]
            flat[f"{name}/intra_dtw"]              = token_results[name]["intra"]["dtw"]
            flat[f"{name}/sep_direction"]          = token_results[name]["separation"]["direction_separation"]
            flat[f"{name}/sep_shape_rel"]          = token_results[name]["separation"]["shape_l2_rel_separation"]
            flat[f"{name}/recon_mse"]              = r["recon_mse"]
            flat[f"{name}/recon_l2"]               = r["recon_l2_per_step"]
        log_dict.update(flat)
        wandb.log(log_dict)


if __name__ == "__main__":
    main()
