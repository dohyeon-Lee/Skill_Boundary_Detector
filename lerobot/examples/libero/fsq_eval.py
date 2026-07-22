"""Unified FSQ evaluation: encoder + decoder in one pass.

Produces:
  - wandb scalars
      encoder : codebook/utilization_pct, skills_per_entry_{mean,std,max}
      decoder : decoder/chunk_action_mse_mean, recon_mse_{xyz,rpy,gripper},
                termination_err_mean (|timing| steps), early_rate, late_rate,
                progress_err_mean
  - one interactive HTML: left FSQ lattice cube, right per-entry bar charts
      (skill count, recon error, termination error, progress error); clicking an
      entry shows N sample skills, each with start/end frames and stacked
      GT-vs-pred plots for per-dim reconstruction, termination, and progress.

The skill_latents.npz used here is (re)generated from the evaluated checkpoint
and written next to the HTML (FSQ_eval/outputs).

Usage (both cameras are decoded live from ``--dataset_dir``):
    python examples/libero/fsq_eval.py \
      --model_path   .../FSQ.pt \
      --skills_dir   .../skillset/skills \
      --dataset_dir  .../libero_90 \
      --output_dir   FSQ_eval/outputs/<run>/<epoch>
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from codebook_visualizer import (  # noqa: E402
    _clip_frame_or_blank,
    _episode_row,
    _load_episodes_meta,
    _read_episode_clip,
    _resolve_image_key,
    _video_path,
)
from FSQ import (  # noqa: E402
    encoder_start_eef_pose,
    load_fsq_model as load_original_fsq_model,
    spline_encode,
)
from train_FSQ import attach_episode_offsets, load_skill_files  # noqa: E402


def load_model(model_path: str, device: str, model_family: str, dino_model_path: str | None):
    if model_family == "fsq":
        model, cfg = load_original_fsq_model(
            model_path, device, dino_model_path=dino_model_path or None
        )
    elif model_family == "fsq_new":
        fsq_new_src = Path(__file__).parent / "configs" / "train_skills" / "FSQ_new" / "src"
        sys.path.insert(0, str(fsq_new_src))
        from FSQ_new import load_fsq_model as load_fsq_new_model  # noqa: PLC0415

        model, cfg = load_fsq_new_model(
            model_path, device, dino_model_path=dino_model_path or None
        )
    else:
        raise ValueError(f"model_family must be fsq|fsq_new, got {model_family!r}.")
    print(
        f"[EVAL] Loaded {model_family} model fsq_levels={cfg.fsq_levels} "
        f"codebook={model.fsq.codebook_size} K={cfg.chunk_size}"
    )
    return model, cfg


# ── latent saving ──────────────────────────────────────────────────────────────

def save_latents(path: Path, latents, tokens, metadata):
    save: dict = {"latents": latents, "tokens": tokens}
    for key in ("episode_id", "task_id", "skill_index", "frame_start", "frame_end", "length"):
        save[key] = np.array([m[key] for m in metadata])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **save)
    print(f"[fsq_eval] saved latents → {path}")


def select_far_active_skill_latents(
    latents: np.ndarray,
    tokens: np.ndarray,
    selected_ids: list[int],
    fsq_levels: list[int],
    seed: int,
    far_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick a random code from the farthest active-code fraction for each skill."""
    if not 0.0 < far_fraction <= 1.0:
        raise ValueError(f"random_far_fraction must be in (0,1], got {far_fraction}.")
    active_tokens = np.unique(tokens.astype(np.int64))
    if len(active_tokens) < 2:
        raise ValueError("Random far-skill comparison needs at least two active FSQ codes.")

    representatives = {
        int(token): latents[int(np.flatnonzero(tokens == token)[0])].astype(np.float32)
        for token in active_tokens
    }
    levels = np.asarray(fsq_levels, dtype=np.float32)
    half = (levels - 1.0) / 2.0
    offset = np.where(levels % 2 == 0, 0.5, 0.0).astype(np.float32)

    def normalized(value: np.ndarray) -> np.ndarray:
        return (value + offset) / half

    rep_coords = {token: normalized(value) for token, value in representatives.items()}
    rng = np.random.default_rng(seed)
    chosen_latents, chosen_tokens, chosen_distances = [], [], []
    for original_id in selected_ids:
        current_token = int(tokens[original_id])
        current_coord = normalized(latents[original_id])
        candidates = [int(token) for token in active_tokens if int(token) != current_token]
        ranked = sorted(
            candidates,
            key=lambda token: float(np.linalg.norm(rep_coords[token] - current_coord)),
            reverse=True,
        )
        pool_size = max(1, int(np.ceil(len(ranked) * far_fraction)))
        far_pool = ranked[:pool_size]
        chosen = int(far_pool[int(rng.integers(0, len(far_pool)))])
        chosen_latents.append(representatives[chosen])
        chosen_tokens.append(chosen)
        chosen_distances.append(float(np.linalg.norm(rep_coords[chosen] - current_coord)))
    return (
        np.stack(chosen_latents).astype(np.float32),
        np.asarray(chosen_tokens, dtype=np.int32),
        np.asarray(chosen_distances, dtype=np.float32),
    )


# ── decode + metrics ───────────────────────────────────────────────────────────

def _dim_groups(action_dim: int) -> dict[str, list[int]]:
    """xyz / rpy / gripper index groups (gripper = last dim)."""
    g = {"xyz": [0, 1, 2], "rpy": [3, 4, 5], "gripper": [action_dim - 1]}
    return {k: [i for i in v if i < action_dim] for k, v in g.items()}


def skill_metrics(delta, progress, term_prob, gt_actions, T, end_threshold, groups):
    """Per-skill metrics. Reconstruction MSE is computed over the whole chunk."""
    K, A = delta.shape[1], delta.shape[2]
    T_future = len(gt_actions)
    t_idx = np.arange(T)[:, None] + np.arange(K)[None, :]      # (T, K)
    valid = t_idx < T_future
    gt_exp = gt_actions[np.minimum(t_idx, T_future - 1)]       # (T, K, A)
    err = ((delta - gt_exp) ** 2) * valid[..., None]           # (T, K, A)
    n_valid = int(valid.sum())

    def group_mse(idxs: list[int]) -> float:
        if n_valid == 0 or not idxs:
            return 0.0
        return float(err[..., idxs].sum() / (n_valid * len(idxs)))

    chunk_mse = float(err.sum() / (n_valid * A)) if n_valid > 0 else 0.0

    # termination timing: first step crossing threshold, else argmax
    gt_end = T - 1
    hits = np.flatnonzero(term_prob[:T] >= end_threshold)
    pred_end = int(hits[0]) if len(hits) else int(np.argmax(term_prob[:T]))
    timing = pred_end - gt_end

    # progress regression error
    gt_prog = np.arange(T, dtype=np.float32) / max(T - 1, 1)
    prog_err = float(np.mean(np.abs(progress[:T] - gt_prog)))

    return {
        "chunk_mse":  chunk_mse,
        "mse_xyz":    group_mse(groups["xyz"]),
        "mse_rpy":    group_mse(groups["rpy"]),
        "mse_grip":   group_mse(groups["gripper"]),
        "timing":     timing,
        "timing_abs": abs(timing),
        "prog_err":   prog_err,
        "pred_end":   pred_end,
        "length":     T,
    }


# ── batched inference (fp16 clips → per-batch float32, no_grad) ─────────────────

@torch.no_grad()
def batched_encode(model, segments, lengths, device, batch_size):
    """Encode all skills (action-only) in length-bucketed batches. Returns latents (N,D), tokens (N,)."""
    N = len(segments)
    A = segments[0].shape[-1]
    nctrl, deg = model.n_control, model.spline_degree
    amin = model.encoder.encoder_min.cpu().numpy()
    amax = model.encoder.encoder_max.cpu().numpy()
    optimal = model.encoder.encoder_input_mode == "optimal"
    if optimal:
        smin = model.encoder.encoder_start_min.cpu().numpy()
        smax = model.encoder.encoder_start_max.cpu().numpy()

    ctrl_norm = []
    start_norm = []
    for seg in segments:
        cp, _ = spline_encode(
            seg.astype(np.float32),
            nctrl,
            deg,
            input_mode=model.encoder.encoder_input_mode,
        )
        cp = (cp - amin) / (amax - amin + 1e-8) * 2.0 - 1.0
        ctrl_norm.append(cp.astype(np.float32))
        if optimal:
            pose = encoder_start_eef_pose(seg)
            start_norm.append(((pose - smin) / (smax - smin + 1e-8) * 2.0 - 1.0).astype(np.float32))

    latents = np.zeros((N, int(model.fsq.latent_dim)), np.float32)
    tokens = np.zeros(N, np.int32)
    order = sorted(range(N), key=lambda i: lengths[i])
    for s in tqdm(range(0, N, batch_size), desc="Encoding (batched)"):
        idxs = order[s:s + batch_size]
        B = len(idxs)
        ctrl = torch.zeros(B, nctrl, A)
        lens = torch.zeros(B, dtype=torch.long)
        start = torch.zeros(B, len(start_norm[0])) if optimal else None
        for b, i in enumerate(idxs):
            ctrl[b] = torch.from_numpy(ctrl_norm[i])
            lens[b] = lengths[i]
            if start is not None:
                start[b] = torch.from_numpy(start_norm[i])
        z_q, idx = model.encode(ctrl.to(device), lens.to(device), None if start is None else start.to(device))
        z_q = z_q.cpu().numpy()
        idx = idx.cpu().numpy()
        for b, i in enumerate(idxs):
            latents[i] = z_q[b]
            tokens[i] = idx[b]
    return latents, tokens


@torch.no_grad()
def _batched_decode_impl(
    model,
    latents,
    states,
    metadata,
    raw_dataset,
    lengths,
    device,
    batch_size,
    broadcast_compare_scale=None,
    random_skill_latents=None,
):
    """Decode all valid frames in bounded frame microbatches.

    Returns per-skill lists sliced to T: deltas[i] (T,K,A), progresses[i] (T,),
    term_probs[i] (T,). GT progress is fed as the motion input (matches training).
    Only the current third-person and wrist frames are decoded for each item.

    ``batch_size`` is a frame count here. The v2 decoder is a 300M-class Gemma
    expert, so padding 64 whole trajectories to max(T) would create thousands of
    expert samples and OOM despite most of them being padding.
    """
    N = len(latents)
    K, A = model.chunk_size, model.action_dim
    deltas = [np.empty((lengths[i], K, A), np.float32) for i in range(N)]
    progs = [np.empty(lengths[i], np.float32) for i in range(N)]
    terms = [np.empty(lengths[i], np.float32) for i in range(N)]
    compare_deltas = (
        None
        if broadcast_compare_scale is None
        else [np.empty((lengths[i], K, A), np.float32) for i in range(N)]
    )
    random_deltas = (
        None
        if random_skill_latents is None
        else [np.empty((lengths[i], K, A), np.float32) for i in range(N)]
    )
    refs = [(i, t) for i, length in enumerate(lengths) for t in range(length)]
    for start in tqdm(range(0, len(refs), batch_size), desc="Decoding (frame microbatches)"):
        part = refs[start : start + batch_size]
        z = torch.from_numpy(np.stack([latents[i] for i, _ in part]).astype(np.float32)).to(device)
        random_z = (
            None
            if random_skill_latents is None
            else torch.from_numpy(
                np.stack([random_skill_latents[i] for i, _ in part]).astype(np.float32)
            ).to(device)
        )
        st = torch.from_numpy(np.stack([states[i][t] for i, t in part]).astype(np.float32)).to(device)
        frames = [
            raw_dataset[
                int(metadata[i]["dataset_from_index"]) + int(metadata[i]["frame_start"]) + int(t)
            ]
            for i, t in part
        ]
        dec = torch.stack([frame["observation.images.image"] for frame in frames]).to(device)
        dec_w = torch.stack([frame["observation.images.wrist_image"] for frame in frames]).to(device)
        # Both action variants start from exactly the same flow noise, so their
        # difference isolates broadcast strength rather than sampling variance.
        noise = model.action_expert.sample_noise(
            (len(part), model.chunk_size, model.cfg.max_action_dim), z.device
        )
        delta, prog, term_logits = model.decode(
            z, st[:, None], dec[:, None], dec_w[:, None], noise=noise
        )
        delta = delta[:, 0].cpu().numpy()
        prog = prog[:, 0].cpu().numpy()
        term = torch.sigmoid(term_logits[:, 0]).cpu().numpy()
        compare = None
        if broadcast_compare_scale is not None:
            compare = model.sample_action_chunks(
                z,
                st[:, None],
                noise=noise,
                broadcast_scale=float(broadcast_compare_scale),
            )[:, 0].cpu().numpy()
        random_action = None
        if random_z is not None:
            random_action = model.sample_action_chunks(
                random_z,
                st[:, None],
                noise=noise,
                broadcast_scale=1.0,
            )[:, 0].cpu().numpy()
        for row, (i, t) in enumerate(part):
            deltas[i][t] = delta[row]
            progs[i][t] = prog[row]
            terms[i][t] = term[row]
            if compare_deltas is not None:
                compare_deltas[i][t] = compare[row]
            if random_deltas is not None:
                random_deltas[i][t] = random_action[row]
    return deltas, progs, terms, compare_deltas, random_deltas


def batched_decode(model, latents, states, metadata, raw_dataset, lengths, device, batch_size):
    """Historical decode API: full-strength actions plus one progress/termination pass."""
    deltas, progs, terms, _, _ = _batched_decode_impl(
        model, latents, states, metadata, raw_dataset, lengths, device, batch_size
    )
    return deltas, progs, terms


def batched_decode_with_broadcast_compare(
    model,
    latents,
    states,
    metadata,
    raw_dataset,
    lengths,
    device,
    batch_size,
    broadcast_compare_scale,
):
    """Decode 100% and scaled broadcast actions with shared noise and shared aux outputs."""
    deltas, progs, terms, compare, _ = _batched_decode_impl(
        model,
        latents,
        states,
        metadata,
        raw_dataset,
        lengths,
        device,
        batch_size,
        broadcast_compare_scale=float(broadcast_compare_scale),
    )
    return deltas, progs, terms, compare


@torch.no_grad()
def batched_decode_fsq_new_abc(
    model,
    latents,
    states,
    metadata,
    raw_dataset,
    lengths,
    device,
    batch_size,
    random_skill_latents=None,
):
    """Decode FSQ_new A/B/C actions with shared noise and one terminator pass."""
    count = len(latents)
    chunk_size, action_dim = model.chunk_size, model.action_dim
    action_modes = ["A", "B", "C"]
    if random_skill_latents is not None:
        action_modes.append("C + far skill")
    actions = {
        mode: [np.empty((lengths[i], chunk_size, action_dim), np.float32) for i in range(count)]
        for mode in action_modes
    }
    progresses = [np.empty(lengths[i], np.float32) for i in range(count)]
    term_probs = [np.empty(lengths[i], np.float32) for i in range(count)]
    refs = [(i, step) for i, length in enumerate(lengths) for step in range(length)]

    for start in tqdm(
        range(0, len(refs), batch_size), desc="Decoding FSQ_new A/B/C (frame microbatches)"
    ):
        part = refs[start : start + batch_size]
        z_q = torch.from_numpy(
            np.stack([latents[i] for i, _ in part]).astype(np.float32)
        ).to(device)
        random_z_q = (
            None
            if random_skill_latents is None
            else torch.from_numpy(
                np.stack([random_skill_latents[i] for i, _ in part]).astype(np.float32)
            ).to(device)
        )
        state = torch.from_numpy(
            np.stack([states[i][step] for i, step in part]).astype(np.float32)
        ).to(device)
        current_frames = [
            raw_dataset[
                int(metadata[i]["dataset_from_index"])
                + int(metadata[i]["frame_start"])
                + int(step)
            ]
            for i, step in part
        ]
        goal_frames = [
            raw_dataset[
                int(metadata[i]["dataset_from_index"])
                + max(0, int(metadata[i]["frame_end"]) - 1)
            ]
            for i, _ in part
        ]
        third = torch.stack(
            [frame["observation.images.image"] for frame in current_frames]
        ).to(device)
        wrist = torch.stack(
            [frame["observation.images.wrist_image"] for frame in current_frames]
        ).to(device)
        goal_third = torch.stack(
            [frame["observation.images.image"] for frame in goal_frames]
        ).to(device)
        goal_wrist = torch.stack(
            [frame["observation.images.wrist_image"] for frame in goal_frames]
        ).to(device)

        current_features = model.terminator.image_features(third, wrist)
        goal_features = model.terminator.image_features(goal_third, goal_wrist)
        image_context = model.image_context(*current_features)
        goal_context = model.goal_context(*goal_features)
        noise = model.action_expert.sample_noise(
            (len(part), model.chunk_size, model.cfg.max_action_dim), z_q.device
        )
        shared = {"noise": noise, "num_steps": 10}
        predicted = {
            "A": model.sample_action_chunks(
                z_q,
                state[:, None],
                skill_scale=model.cfg.a_skill_scale,
                **shared,
            ),
            "B": model.sample_action_chunks(
                z_q,
                state[:, None],
                skill_scale=model.cfg.b_skill_scale,
                image_context=image_context,
                image_scale=model.cfg.b_image_scale,
                **shared,
            ),
            "C": model.sample_action_chunks(
                z_q,
                state[:, None],
                skill_scale=model.cfg.c_skill_scale,
                image_context=image_context,
                goal_context=goal_context,
                image_scale=model.cfg.c_image_scale,
                goal_scale=model.cfg.c_goal_scale,
                **shared,
            ),
        }
        if random_z_q is not None:
            predicted["C + far skill"] = model.sample_action_chunks(
                random_z_q,
                state[:, None],
                skill_scale=model.cfg.c_skill_scale,
                image_context=image_context,
                goal_context=goal_context,
                image_scale=model.cfg.c_image_scale,
                goal_scale=model.cfg.c_goal_scale,
                **shared,
            )
        z_norm = model.fsq.normalized(z_q)
        progress, term_logits = model.terminator(
            z_norm,
            state,
            third,
            wrist,
            image_features=current_features,
        )
        progress = progress.cpu().numpy()
        term = torch.sigmoid(term_logits).cpu().numpy()
        predicted_np = {
            mode: value[:, 0].cpu().numpy() for mode, value in predicted.items()
        }
        for row, (skill_index, step) in enumerate(part):
            for mode in action_modes:
                actions[mode][skill_index][step] = predicted_np[mode][row]
            progresses[skill_index][step] = progress[row]
            term_probs[skill_index][step] = term[row]

    return actions, progresses, term_probs


# ── per-sample composite plot (start/end frame + recon/termination/progress) ────

def make_sample_plot(start_img, end_img, delta, progress, term_prob, gt_actions, T,
                     dim_labels, n_action_steps, end_threshold,
                     delta_compare=None, compare_label="broadcast 50%",
                     action_variants=None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    A = delta.shape[2]
    K = delta.shape[1]
    n_rows = A + 2  # per-dim recon + termination + progress
    n_img_rows = 1     # row0: start / end frames
    img_h = 4.0     # inches per image row (≈ half figure width → fills the row)
    row_h = 1.0
    fig = plt.figure(figsize=(8.5, img_h * n_img_rows + row_h * n_rows))
    gs = GridSpec(n_img_rows + n_rows, 2, figure=fig,
                  height_ratios=[img_h] * n_img_rows + [row_h] * n_rows, wspace=0.03)

    # row 0: start / end frames (large, filling the row width)
    ax_s = fig.add_subplot(gs[0, 0]); ax_e = fig.add_subplot(gs[0, 1])
    ax_s.imshow(start_img); ax_s.set_title("start", fontsize=10); ax_s.axis("off")
    ax_e.imshow(end_img);   ax_e.set_title("end",   fontsize=10); ax_e.axis("off")

    base = n_img_rows
    t_full = np.arange(T)
    chunk_starts = np.arange(0, T, max(1, n_action_steps))
    if action_variants is None:
        action_variants = [
            (
                "pred 100%" if delta_compare is not None else "pred chunk",
                delta,
                "#B71C1C",
                "--",
            )
        ]
        if delta_compare is not None:
            action_variants.append((compare_label, delta_compare, "#00897B", ":"))

    # per-dim reconstruction (GT vs predicted chunks)
    for d in range(A):
        ax = fig.add_subplot(gs[base + d, :])
        ax.plot(t_full, gt_actions[:T, d], color="#0D47A1", linewidth=1.5, label="GT", zorder=3)
        for j, start in enumerate(chunk_starts):
            x = start + np.arange(K)
            m = x < len(gt_actions)
            if not np.any(m):
                continue
            for variant_index, (variant_label, variant_delta, color, linestyle) in enumerate(
                action_variants
            ):
                ax.plot(
                    x[m],
                    variant_delta[start, : int(m.sum()), d],
                    color=color,
                    linewidth=1.4 if variant_index == 0 else 1.3,
                    alpha=0.92,
                    linestyle=linestyle,
                    label=variant_label if j == 0 else None,
                    zorder=4 + variant_index,
                )
        ax.set_ylabel(dim_labels[d], fontsize=8, rotation=0, labelpad=26)
        ax.tick_params(labelsize=7); ax.grid(True, color="#eee", linewidth=0.6)
        ax.set_xticks([])
        if d == 0:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.8)

    # termination GT vs pred
    ax_t = fig.add_subplot(gs[base + A, :])
    gt_term = np.zeros(T); gt_term[T - 1] = 1.0
    ax_t.plot(t_full, gt_term, color="#0D47A1", linewidth=1.5, label="GT end")
    ax_t.plot(t_full, term_prob[:T], color="#B71C1C", linewidth=1.4, linestyle="--", label="pred prob")
    ax_t.axhline(end_threshold, color="#888", linewidth=0.8, linestyle=":")
    ax_t.set_ylabel("term", fontsize=8, rotation=0, labelpad=26)
    ax_t.set_ylim(-0.05, 1.05); ax_t.tick_params(labelsize=7)
    ax_t.grid(True, color="#eee", linewidth=0.6); ax_t.set_xticks([])
    ax_t.legend(fontsize=7, loc="upper left", framealpha=0.8)

    # progress GT vs pred
    ax_p = fig.add_subplot(gs[base + 1 + A, :])
    gt_prog = np.arange(T, dtype=np.float32) / max(T - 1, 1)
    ax_p.plot(t_full, gt_prog, color="#0D47A1", linewidth=1.5, label="GT")
    ax_p.plot(t_full, progress[:T], color="#B71C1C", linewidth=1.4, linestyle="--", label="pred")
    ax_p.set_ylabel("prog", fontsize=8, rotation=0, labelpad=26)
    ax_p.set_ylim(-0.05, 1.05); ax_p.tick_params(labelsize=7)
    ax_p.grid(True, color="#eee", linewidth=0.6)
    ax_p.legend(fontsize=7, loc="upper left", framealpha=0.8)

    fig.tight_layout(h_pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg", dpi=110, bbox_inches="tight", pil_kwargs={"quality": 88})
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ── HTML assembly ──────────────────────────────────────────────────────────────

_CSS = """
  body {font-family:sans-serif;background:#f5f5f5;margin:0;padding:12px;font-size:13px;}
  h1 {margin:0 0 4px;font-size:19px;color:#222;word-break:break-all;}
  h2 {margin:0 0 4px;font-size:16px;}
  .summary {color:#555;margin-bottom:10px;font-size:12px;}
  #top {display:flex;gap:12px;align-items:flex-start;}
  #cube-box {background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px;}
  #charts {flex:1;min-width:360px;display:flex;flex-direction:column;gap:6px;}
  .chart-box {background:#fff;border:1px solid #ddd;border-radius:6px;padding:4px 6px;overflow-x:auto;}
  .chart-box h4 {margin:0 0 3px;font-size:11px;color:#333;}
  canvas {display:block;cursor:pointer;}
  #panel {margin-top:12px;background:#fff;border:1px solid #ddd;border-radius:6px;padding:10px;display:none;}
  table {border-collapse:collapse;font-size:12px;margin-bottom:10px;}
  th,td {border:1px solid #e0e0e0;padding:4px 8px;text-align:right;}
  th {background:#f0f0f0;text-align:center;}
  td:first-child {text-align:left;}
  #samples {display:flex;flex-direction:row;flex-wrap:nowrap;gap:10px;overflow-x:auto;padding-bottom:8px;}
  .sample {flex:0 0 auto;}
  .sample img {display:block;width:300px;height:auto;border:1px solid #ddd;border-radius:4px;background:#fff;}
  .sample .cap {font-size:11px;color:#555;margin-bottom:3px;}
"""

# JS uses normal braces (no str.format) to keep it readable; data injected as JSON.
_JS = r"""
const N = COUNTS.length;
const LEVELS = FSQ_LEVELS;

// ── FSQ lattice cube ──────────────────────────────────────────────
const cubeCanvas = document.getElementById('cube');
const cubeCtx = cubeCanvas.getContext('2d');
cubeCanvas.width = 560; cubeCanvas.height = 560;
// Dimension-aware lattice: 1D line, 2D flat grid (e.g. [8,8]), 3D iso cube (e.g. [5,5,5] / [8,6,5]).
const NDIM = LEVELS.length;
const MAXL = Math.max.apply(null, LEVELS);
function idxToCoord(idx){const c=[];let r=idx;for(let d=0;d<NDIM;d++){c.push(r%LEVELS[d]);r=Math.floor(r/LEVELS[d]);}return c;}
// Center each axis, scale ALL axes by the same (MAXL-1) so non-cubic lattices keep true proportions.
function normCoord(c){const s=Math.max(1,MAXL-1)/2;return LEVELS.map((L,d)=>((c[d]||0)-(L-1)/2)/s);}
function projectCoord(c){
  const n=normCoord(c);
  const ox=cubeCanvas.width*0.5,oy=cubeCanvas.height*0.52;
  if(NDIM<=2){const xn=n[0]||0,yn=(n.length>1?n[1]:0);return [ox+xn*200, oy-yn*200, 0];}
  const xn=n[0]||0,yn=n[1]||0,zn=n[2]||0;
  const yaw=-0.63,pitch=0.46,cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
  const xr=cy*xn-sy*yn,yr=sy*xn+cy*yn,zr=zn,scale=150;
  return [ox+xr*scale, oy+yr*scale*sp-zr*scale*cp, yr*cp+zr*sp];
}
function drawCube(sel){
  const ctx=cubeCtx; ctx.clearRect(0,0,cubeCanvas.width,cubeCanvas.height);
  const maxC=Math.max(...COUNTS,1);
  ctx.strokeStyle='rgba(120,120,120,0.45)'; ctx.lineWidth=1;
  const line=(a,b)=>{const pa=projectCoord(a),pb=projectCoord(b);ctx.beginPath();ctx.moveTo(pa[0],pa[1]);ctx.lineTo(pb[0],pb[1]);ctx.stroke();};
  for(let i=0;i<N;i++){const c=idxToCoord(i);for(let d=0;d<NDIM;d++){if(c[d]+1<LEVELS[d]){const c2=c.slice();c2[d]+=1;line(c,c2);}}}
  const pts=[];
  for(let i=0;i<N;i++){const p=projectCoord(idxToCoord(i));pts.push({i,p,depth:p[2]});}
  pts.sort((a,b)=>a.depth-b.depth);
  pts.forEach(pt=>{
    const c=COUNTS[pt.i],active=c>0;
    const r=pt.i===sel?9:(active?4+7*Math.sqrt(c/maxC):2.6);
    ctx.beginPath();ctx.arc(pt.p[0],pt.p[1],r,0,Math.PI*2);
    ctx.fillStyle=pt.i===sel?'#f44336':(active?'rgba(25,118,210,0.78)':'rgba(180,180,180,0.4)');
    ctx.fill();
  });
  ctx.fillStyle='#777';ctx.font='10px sans-serif';
  ctx.fillText('levels = '+LEVELS.join(' x '),10,cubeCanvas.height-8);
}
cubeCanvas.addEventListener('click',e=>{
  const rect=cubeCanvas.getBoundingClientRect();
  const mx=(e.clientX-rect.left)*cubeCanvas.width/rect.width;
  const my=(e.clientY-rect.top)*cubeCanvas.height/rect.height;
  let best=-1,bd=1e9;
  for(let i=0;i<N;i++){const p=projectCoord(idxToCoord(i));const d=(p[0]-mx)**2+(p[1]-my)**2;if(d<bd){bd=d;best=i;}}
  if(best>=0&&bd<625&&COUNTS[best]>0)selectEntry(best);
});

// ── per-entry bar charts ──────────────────────────────────────────
const CHARTS=[
  {id:'c_count', vals:COUNTS,                title:'Skill count',        color:'#455A64', fmt:v=>v.toFixed(0)},
  {id:'c_recon', vals:VAL('chunk_mse'),      title:'Recon chunk MSE',    color:'#1976D2', fmt:v=>v.toExponential(1)},
  {id:'c_term',  vals:VAL('timing_abs'),     title:'Termination |err| (steps)', color:'#7B1FA2', fmt:v=>v.toFixed(1)},
  {id:'c_prog',  vals:VAL('prog_err'),       title:'Progress |err|',     color:'#2E7D32', fmt:v=>v.toFixed(3)},
];
function VAL(key){return Array.from({length:N},(_,i)=>ENTRY[i]?ENTRY[i][key]:null);}
function drawChart(ch,sel){
  const canvas=document.getElementById(ch.id),ctx=canvas.getContext('2d');
  const BW=Math.max(2,Math.min(16,Math.floor(1300/N))),PL=46,PR=10,PT=8,PB=16;
  canvas.width=PL+N*BW+PR; canvas.height=110;
  const vals=ch.vals,maxV=Math.max(...vals.filter(v=>v!=null),1e-12),H=canvas.height-PT-PB;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.strokeStyle='#bbb';ctx.beginPath();ctx.moveTo(PL,PT);ctx.lineTo(PL,PT+H);ctx.stroke();
  ctx.fillStyle='#888';ctx.font='9px sans-serif';ctx.textAlign='right';
  [0,maxV/2,maxV].forEach(v=>{const y=PT+H-(v/maxV)*H;ctx.fillText(ch.fmt(v),PL-3,y+3);});
  vals.forEach((v,i)=>{if(v==null)return;const x=PL+i*BW,h=(v/maxV)*H;ctx.fillStyle=(i===SEL)?'#f44336':ch.color;ctx.fillRect(x,PT+H-h,BW-1,h);});
}
function drawAllCharts(){CHARTS.forEach(ch=>drawChart(ch,SEL));}
CHARTS.forEach(ch=>{
  const canvas=document.getElementById(ch.id);
  canvas.addEventListener('click',e=>{
    const rect=canvas.getBoundingClientRect();const BW=Math.max(2,Math.min(16,Math.floor(1300/N)));
    const i=Math.floor((e.clientX-rect.left)*canvas.width/rect.width-46)/BW|0;
    if(i>=0&&i<N&&ENTRY[i])selectEntry(i);
  });
});

let SEL=-1;
function selectEntry(i){SEL=i;drawCube(SEL);drawAllCharts();showPanel(i);}
function showPanel(tok){
  const d=ENTRY[tok];if(!d)return;
  document.getElementById('panel-title').textContent='Entry '+tok+' — '+COUNTS[tok]+' skills';
  const baseLabel=d.base_action_label||'base';
  const rows=[
    ['# skills',COUNTS[tok]],
    ['Recon '+baseLabel+' chunk MSE',d.chunk_mse.toExponential(3)],
    ['Recon '+baseLabel+' MSE xyz',d.mse_xyz.toExponential(3)],
    ['Recon '+baseLabel+' MSE rpy',d.mse_rpy.toExponential(3)],
    ['Recon '+baseLabel+' MSE gripper',d.mse_grip.toExponential(3)],
    ['Termination |err| (steps)',d.timing_abs.toFixed(2)],
    ['Termination err (signed)',d.timing.toFixed(2)],
    ['Progress |err|',d.prog_err.toFixed(4)],
    ['Mean skill length',d.length.toFixed(1)],
  ];
  if(d.variant_metrics){
    const variantRows=[];
    Object.entries(d.variant_metrics).forEach(([label,m])=>{
      variantRows.push(
        ['Recon '+label+' chunk MSE',m.chunk_mse.toExponential(3)],
        ['Recon '+label+' MSE xyz',m.mse_xyz.toExponential(3)],
        ['Recon '+label+' MSE rpy',m.mse_rpy.toExponential(3)],
        ['Recon '+label+' MSE gripper',m.mse_grip.toExponential(3)]
      );
    });
    rows.splice(5,0,...variantRows);
  }
  document.getElementById('panel-table').innerHTML='<tr><th>Metric</th><th>Value</th></tr>'+
    rows.map(r=>'<tr><td>'+r[0]+'</td><td>'+r[1]+'</td></tr>').join('');
  const imgs=SAMPLES[tok]||[];
  document.getElementById('samples').innerHTML=imgs.length
    ? imgs.map((b,i)=>'<div class="sample"><div class="cap">sample '+i+'</div><img src="data:image/jpeg;base64,'+b+'"></div>').join('')
    : '<div class="cap">No sample plots rendered for this entry (increase max_plot_entries).</div>';
  document.getElementById('panel').style.display='block';
  document.getElementById('panel').scrollIntoView({behavior:'smooth',block:'nearest'});
}
drawCube(-1);drawAllCharts();
"""


def build_html(title, summary, fsq_levels, counts, entry_data, samples, codebook_size) -> str:
    data_js = (
        "const SUMMARY=" + json.dumps(summary) + ";\n"
        "const FSQ_LEVELS=" + json.dumps(list(fsq_levels)) + ";\n"
        "const COUNTS=" + json.dumps(counts) + ";\n"
        "const ENTRY=" + json.dumps([entry_data.get(i) for i in range(codebook_size)]) + ";\n"
        "const SAMPLES=" + json.dumps([samples.get(i) for i in range(codebook_size)]) + ";\n"
    )
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        "<title>" + title + "</title><style>" + _CSS + "</style></head><body>"
        "<h1>" + title + "</h1>"
        "<div class='summary'>FSQ Unified Evaluation &nbsp;|&nbsp; " + summary + "</div>"
        "<div id='top'>"
        "  <div id='cube-box'><canvas id='cube'></canvas></div>"
        "  <div id='charts'>"
        "    <div class='chart-box'><h4>Skill count</h4><canvas id='c_count'></canvas></div>"
        "    <div class='chart-box'><h4>Reconstruction chunk MSE</h4><canvas id='c_recon'></canvas></div>"
        "    <div class='chart-box'><h4>Termination |error| (steps)</h4><canvas id='c_term'></canvas></div>"
        "    <div class='chart-box'><h4>Progress |error|</h4><canvas id='c_prog'></canvas></div>"
        "  </div>"
        "</div>"
        "<div id='panel'><h4 id='panel-title'></h4><table id='panel-table'></table>"
        "<div id='samples'></div></div>"
        "<script>" + data_js + _JS + "</script>"
        "</body></html>"
    )


# ── frame loading for sampled skills ───────────────────────────────────────────

def load_sample_frames(metadata, sample_ids, dataset_dir: Path, image_key: str, thumb: int):
    """Return {skill_idx: (start_rgb, end_rgb)} for the requested sample skill indices."""
    episodes_meta = _load_episodes_meta(dataset_dir)
    key = _resolve_image_key(episodes_meta, image_key)
    blank = np.full((thumb, thumb, 3), 80, np.uint8)

    ep_to_ids: dict[int, list[int]] = defaultdict(list)
    for i in sample_ids:
        ep_to_ids[int(metadata[i]["episode_id"])].append(i)

    frames: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for ep_id, ids in tqdm(sorted(ep_to_ids.items()), desc="Loading frames"):
        try:
            row = _episode_row(episodes_meta, ep_id)
            from_ts = float(row[f"videos/{key}/from_timestamp"])
            to_ts = float(row[f"videos/{key}/to_timestamp"])
            clip = _read_episode_clip(_video_path(dataset_dir, episodes_meta, ep_id, key),
                                      from_ts, to_ts, int(row["length"]))
            for i in ids:
                fs, fe = int(metadata[i]["frame_start"]), int(metadata[i]["frame_end"])
                frames[i] = (_clip_frame_or_blank(clip, fs, thumb),
                             _clip_frame_or_blank(clip, max(0, fe - 1), thumb))
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] ep{ep_id}: {exc}")
            for i in ids:
                frames[i] = (blank, blank)
    return frames


# ── wandb ───────────────────────────────────────────────────────────────────────

def log_wandb(
    args,
    enc_stats,
    dec_means,
    decoded_skill_count,
    html_path,
    action_variant_means=None,
):
    import wandb
    wandb.init(project=args.wandb_project, name=args.wandb_run_name or Path(args.model_path).parent.name,
               config=vars(args), resume="allow")
    decoder_prefix = "decoder" if args.decoder_scope == "all" else "decoder_sample"
    values = {
        "codebook/utilization_pct":       enc_stats["utilization_pct"],
        "codebook/skills_per_entry_mean": enc_stats["skills_per_entry_mean"],
        "codebook/skills_per_entry_std":  enc_stats["skills_per_entry_std"],
        "codebook/skills_per_entry_max":  enc_stats["skills_per_entry_max"],
        "decoder/evaluated_skills":       decoded_skill_count,
    }
    values.update({
        f"{decoder_prefix}/chunk_action_mse_mean": dec_means["chunk_mse"],
        f"{decoder_prefix}/recon_mse_xyz":         dec_means["mse_xyz"],
        f"{decoder_prefix}/recon_mse_rpy":         dec_means["mse_rpy"],
        f"{decoder_prefix}/recon_mse_gripper":     dec_means["mse_grip"],
        f"{decoder_prefix}/termination_err_mean":  dec_means["timing_abs"],
        f"{decoder_prefix}/early_rate":            dec_means["early_rate"],
        f"{decoder_prefix}/late_rate":             dec_means["late_rate"],
        f"{decoder_prefix}/progress_err_mean":     dec_means["prog_err"],
    })
    for label, means in (action_variant_means or {}).items():
        metric_label = label.replace("%", "pct")
        tag = "".join(
            char.lower() if char.isalnum() else "_" for char in metric_label
        ).strip("_")
        values.update({
            f"{decoder_prefix}/{tag}/chunk_action_mse_mean": means["chunk_mse"],
            f"{decoder_prefix}/{tag}/recon_mse_xyz": means["mse_xyz"],
            f"{decoder_prefix}/{tag}/recon_mse_rpy": means["mse_rpy"],
            f"{decoder_prefix}/{tag}/recon_mse_gripper": means["mse_grip"],
        })
    wandb.log(values)
    wandb.finish()
    print(f"[wandb] logged scalars to project '{args.wandb_project}' (HTML saved locally: {html_path})")


# ── main ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_family", choices=["fsq", "fsq_new"], default="fsq")
    p.add_argument("--model_path", required=True)
    p.add_argument("--dino_model_path", default="")
    p.add_argument("--skills_dir", required=True)
    p.add_argument("--dataset_dir", required=True,
                   help="LeRobot dataset dir (videos + meta); both cameras are read live")
    p.add_argument("--output_dir", required=True, help="where the HTML is written (FSQ_eval/outputs/<run>/<epoch>)")
    p.add_argument("--latents_path", default="",
                   help="where to save/load skill_latents.npz (default: <output_dir>/skill_latents.npz). "
                        "Point this at the model checkpoint folder to keep latents next to FSQ.pt.")
    p.add_argument("--n_action_steps", type=int, default=5, help="chunk plot stride")
    p.add_argument("--max_plot_samples", type=int, default=5, help="sample skills per entry")
    p.add_argument("--max_plot_entries", type=int, default=0, help="0 = render all active entries")
    p.add_argument("--decoder_scope", choices=["samples", "all"], default="samples",
                   help="samples: decode only rendered entry samples (fast); all: decode every skill (full metrics).")
    p.add_argument("--image_key", default="observation.images.image",
                   help="Dataset camera key used for the start/end HTML thumbnails.")
    p.add_argument("--thumb_size", type=int, default=160)
    p.add_argument("--batch_size", type=int, default=64,
                   help="trajectory batch for encoding; frame microbatch for v2 expert/terminator inference")
    p.add_argument("--seed", type=int, default=42, help="seed for random sample selection per codebook entry")
    p.add_argument("--end_threshold", type=float, default=0.5)
    p.add_argument(
        "--broadcast_compare_scale",
        type=float,
        default=0.5,
        help="For broadcast checkpoints, overlay this action scale using identical flow noise; <0 disables.",
    )
    p.add_argument(
        "--random_far_skill",
        action="store_true",
        help="Overlay a different active skill sampled from the farthest codebook candidates.",
    )
    p.add_argument(
        "--random_far_fraction",
        type=float,
        default=0.1,
        help="Randomly sample within this farthest fraction of active non-current codes.",
    )
    p.add_argument("--force_encode", action="store_true",
                   help="Re-encode latents even if skill_latents.npz already exists in output_dir.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--wandb_project", default="VAE_eval")
    p.add_argument("--wandb_run_name", default="")
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(
        args.model_path, device, args.model_family, args.dino_model_path
    )
    levels = list(cfg.fsq_levels)
    codebook_size = int(model.fsq.codebook_size)
    broadcast_compare_scale = None
    if (
        args.model_family == "fsq"
        and cfg.state_cond_mode == "broadcast"
        and args.broadcast_compare_scale >= 0.0
    ):
        if not np.isfinite(args.broadcast_compare_scale):
            raise ValueError("broadcast_compare_scale must be finite or negative to disable it.")
        if args.broadcast_compare_scale != 1.0:
            broadcast_compare_scale = float(args.broadcast_compare_scale)
    elif (
        args.model_family == "fsq"
        and cfg.state_cond_mode != "broadcast"
        and args.broadcast_compare_scale >= 0.0
    ):
        print(
            f"[fsq_eval] state_cond_mode={cfg.state_cond_mode}: "
            "broadcast comparison is not applicable and will be skipped."
        )

    print("[fsq_eval] loading skills and live third+wrist frame reader ...")
    segments, dec_states, dec_targets, metadata = load_skill_files(Path(args.skills_dir))
    attach_episode_offsets(args.dataset_dir, metadata)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415
    raw_dataset = LeRobotDataset(
        repo_id=f"local/{Path(args.dataset_dir).name}",
        root=args.dataset_dir,
        video_keys_to_load=["observation.images.image", "observation.images.wrist_image"],
    )

    # ── encoder: reuse latents if already present for this run, else encode ──
    latents_path = Path(args.latents_path) if args.latents_path else out_dir / "skill_latents.npz"
    latents = tokens = None
    if latents_path.exists() and not args.force_encode:
        d = np.load(str(latents_path))
        # only trust the cache if it matches the current skill set
        if len(d["tokens"]) == len(metadata) and np.array_equal(
            d["frame_start"], np.array([m["frame_start"] for m in metadata])
        ):
            latents = d["latents"].astype(np.float32)
            tokens = d["tokens"].astype(np.int32)
            print(f"[fsq_eval] reusing existing latents → {latents_path} (--force_encode to redo)")
        else:
            print(f"[fsq_eval] existing latents at {latents_path} do not match skills; re-encoding")
    lengths = [int(m["length"]) for m in metadata]
    if latents is None:
        latents, tokens = batched_encode(model, segments, lengths, device, args.batch_size)
        save_latents(latents_path, latents, tokens, metadata)

    counts = [0] * codebook_size
    for t in tokens:
        counts[int(t)] += 1
    active = [c for c in counts if c > 0]
    enc_stats = {
        "utilization_pct":       100.0 * len(active) / codebook_size,
        "skills_per_entry_mean": float(np.mean(active)) if active else 0.0,
        "skills_per_entry_std":  float(np.std(active)) if active else 0.0,
        "skills_per_entry_max":  int(max(counts)) if counts else 0,
    }
    print(f"[fsq_eval] codebook {len(active)}/{codebook_size} used "
          f"({enc_stats['utilization_pct']:.1f}%)  mean={enc_stats['skills_per_entry_mean']:.1f}")

    # ── choose entries/samples before decoding ────────────────────────────────
    # Tokenization is action-only and cheap, so it always covers every skill. This
    # keeps utilization and the "top entries" selection exact. Live image decode
    # and the 300M decoder then run only on those visualized samples by default.
    by_entry: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(tokens):
        by_entry[int(t)].append(i)
    active_tokens = sorted(by_entry)
    if args.max_plot_entries > 0:
        plot_tokens = set(sorted(active_tokens, key=lambda t: (-counts[t], t))[: args.max_plot_entries])
    else:
        plot_tokens = set(active_tokens)
    rng = np.random.default_rng(args.seed)
    sample_ids: list[int] = []
    sample_by_tok: dict[int, list[int]] = {}
    for tok in sorted(plot_tokens):
        pool = by_entry[tok]
        n = min(args.max_plot_samples, len(pool))
        chosen = [pool[j] for j in rng.choice(len(pool), size=n, replace=False)] if n > 0 else []
        sample_by_tok[tok] = chosen
        sample_ids.extend(chosen)

    decode_ids = list(range(len(metadata))) if args.decoder_scope == "all" else sample_ids
    if not decode_ids:
        raise ValueError("decoder_scope=samples requires --max_plot_samples > 0 and at least one active entry")

    decode_random_latents = None
    random_skill_tokens_by_id: dict[int, int] = {}
    if args.random_far_skill:
        decode_random_latents, random_tokens, random_distances = select_far_active_skill_latents(
            latents,
            tokens,
            decode_ids,
            levels,
            args.seed + 10_003,
            args.random_far_fraction,
        )
        random_skill_tokens_by_id = {
            original_id: int(random_tokens[index])
            for index, original_id in enumerate(decode_ids)
        }
        print(
            f"[fsq_eval] far-skill comparison: top {args.random_far_fraction:.0%} active-code pool, "
            f"mean normalized distance={float(np.mean(random_distances)):.3f}"
        )

    # ── decoder: live-frame inference only for the requested scope ────────────
    action_dim = dec_targets[0].shape[-1]
    groups = _dim_groups(action_dim)
    dim_labels = [f"d{i}" for i in range(action_dim - 1)] + ["grip"]
    decode_latents = latents[decode_ids]
    decode_states = [dec_states[i] for i in decode_ids]
    decode_metadata = [metadata[i] for i in decode_ids]
    decode_lengths = [lengths[i] for i in decode_ids]
    decode_targets = [dec_targets[i] for i in decode_ids]
    print(f"[fsq_eval] decoder_scope={args.decoder_scope}: decoding {len(decode_ids)}/{len(metadata)} skills")
    decoded_variant_deltas: dict[str, list[np.ndarray]] = {}
    base_action_label = "pred chunk"
    if args.model_family == "fsq_new":
        abc_actions, decoded_progress, decoded_term = batched_decode_fsq_new_abc(
            model, decode_latents, decode_states, decode_metadata, raw_dataset,
            decode_lengths, device, args.batch_size,
            random_skill_latents=decode_random_latents,
        )
        decoded_delta = abc_actions["A"]
        decoded_variant_deltas = {"B": abc_actions["B"], "C": abc_actions["C"]}
        if decode_random_latents is not None:
            decoded_variant_deltas["C + far skill"] = abc_actions["C + far skill"]
        base_action_label = "A"
    else:
        decoded_delta, decoded_progress, decoded_term, compare_delta, random_delta = (
            _batched_decode_impl(
            model, decode_latents, decode_states, decode_metadata, raw_dataset,
                decode_lengths, device, args.batch_size,
                broadcast_compare_scale=broadcast_compare_scale,
                random_skill_latents=decode_random_latents,
            )
        )
        if compare_delta is not None:
            compare_label = f"broadcast {broadcast_compare_scale * 100:g}%"
            decoded_variant_deltas[compare_label] = compare_delta
            base_action_label = "broadcast 100%"
        elif random_delta is not None:
            base_action_label = "current skill"
        if random_delta is not None:
            decoded_variant_deltas["far active skill"] = random_delta
    decoded: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        original_id: (decoded_delta[j], decoded_progress[j], decoded_term[j])
        for j, original_id in enumerate(decode_ids)
    }
    decoded_variants = {
        label: {
            original_id: values[j] for j, original_id in enumerate(decode_ids)
        }
        for label, values in decoded_variant_deltas.items()
    }
    per_skill = {
        original_id: skill_metrics(
            decoded_delta[j], decoded_progress[j], decoded_term[j], decode_targets[j],
            decode_lengths[j], args.end_threshold, groups,
        )
        for j, original_id in enumerate(decode_ids)
    }
    keys = ["chunk_mse", "mse_xyz", "mse_rpy", "mse_grip", "timing_abs", "prog_err"]
    dec_means = {k: float(np.mean([s[k] for s in per_skill.values()])) for k in keys}
    dec_means["early_rate"] = float(np.mean([s["timing"] < 0 for s in per_skill.values()]))
    dec_means["late_rate"]  = float(np.mean([s["timing"] > 0 for s in per_skill.values()]))
    action_keys = ["chunk_mse", "mse_xyz", "mse_rpy", "mse_grip"]
    per_skill_variants = {
        label: {
            original_id: skill_metrics(
                values[j], decoded_progress[j], decoded_term[j], decode_targets[j],
                decode_lengths[j], args.end_threshold, groups,
            )
            for j, original_id in enumerate(decode_ids)
        }
        for label, values in decoded_variant_deltas.items()
    }
    dec_means_variants = {
        label: {
            key: float(np.mean([skill[key] for skill in metrics.values()]))
            for key in action_keys
        }
        for label, metrics in per_skill_variants.items()
    }
    base_metric_name = f"{base_action_label} chunk_mse" if decoded_variant_deltas else "chunk_mse"
    print(f"[fsq_eval] [{args.decoder_scope}, n={len(decode_ids)}] "
          f"{base_metric_name}={dec_means['chunk_mse']:.4e}  term|err|={dec_means['timing_abs']:.2f}  "
          f"early={dec_means['early_rate']:.1%} late={dec_means['late_rate']:.1%}  "
          f"prog_err={dec_means['prog_err']:.4f}")
    for label, means in dec_means_variants.items():
        print(f"[fsq_eval] action variant {label}: chunk_mse={means['chunk_mse']:.4e}")

    # Per-entry values are exact in decoder_scope=all; in samples mode they are
    # the rendered random samples only (entries outside the top set have no bar).
    entry_data: dict[int, dict] = {}
    for tok, ids in by_entry.items():
        metric_ids = [i for i in ids if i in per_skill]
        if not metric_ids:
            continue
        entry_data[tok] = {
            "chunk_mse":  float(np.mean([per_skill[i]["chunk_mse"] for i in metric_ids])),
            "mse_xyz":    float(np.mean([per_skill[i]["mse_xyz"] for i in metric_ids])),
            "mse_rpy":    float(np.mean([per_skill[i]["mse_rpy"] for i in metric_ids])),
            "mse_grip":   float(np.mean([per_skill[i]["mse_grip"] for i in metric_ids])),
            "timing_abs": float(np.mean([per_skill[i]["timing_abs"] for i in metric_ids])),
            "timing":     float(np.mean([per_skill[i]["timing"] for i in metric_ids])),
            "prog_err":   float(np.mean([per_skill[i]["prog_err"] for i in metric_ids])),
            "length":     float(np.mean([lengths[i] for i in ids])),
        }
        entry_data[tok]["base_action_label"] = base_action_label
        entry_data[tok]["variant_metrics"] = {
            label: {
                "chunk_mse": float(np.mean([
                    metrics[i]["chunk_mse"] for i in metric_ids
                ])),
                "mse_xyz": float(np.mean([metrics[i]["mse_xyz"] for i in metric_ids])),
                "mse_rpy": float(np.mean([metrics[i]["mse_rpy"] for i in metric_ids])),
                "mse_grip": float(np.mean([metrics[i]["mse_grip"] for i in metric_ids])),
            }
            for label, metrics in per_skill_variants.items()
        }

    frames = ({} if args.max_plot_samples <= 0 else
              load_sample_frames(metadata, sample_ids, Path(args.dataset_dir), args.image_key, args.thumb_size))

    samples: dict[int, list[str]] = {}
    for tok in tqdm(sample_by_tok, desc="Rendering samples"):
        imgs = []
        for i in sample_by_tok[tok]:
            T = lengths[i]
            blank = np.full((args.thumb_size,) * 2 + (3,), 80, np.uint8)
            s_img, e_img = frames.get(i, (blank, blank))
            delta, progress, term_prob = decoded[i]
            action_variants = None
            if decoded_variants:
                palette = [
                    ("#B71C1C", "--"),
                    ("#00897B", ":"),
                    ("#7B1FA2", "-."),
                    ("#EF6C00", "-"),
                ]
                variant_items = []
                for index, (label, values) in enumerate(decoded_variants.items()):
                    display_label = label
                    if "far" in label.lower() and i in random_skill_tokens_by_id:
                        display_label = f"{label} #{random_skill_tokens_by_id[i]}"
                    variant_items.append(
                        (display_label, values[i], *palette[(index + 1) % len(palette)])
                    )
                action_variants = [
                    (base_action_label, delta, *palette[0]),
                    *variant_items,
                ]
            imgs.append(make_sample_plot(s_img, e_img, delta, progress, term_prob,
                                         dec_targets[i], T, dim_labels, args.n_action_steps,
                                         args.end_threshold, action_variants=action_variants))
        samples[tok] = imgs

    summary = (
        f"codebook {len(active)}/{codebook_size} ({enc_stats['utilization_pct']:.1f}%) | "
        f"skills/entry mean={enc_stats['skills_per_entry_mean']:.1f} max={enc_stats['skills_per_entry_max']} | "
        f"decoder={args.decoder_scope} n={len(decode_ids)}/{len(metadata)} | "
        f"chunk MSE={dec_means['chunk_mse']:.3e} (xyz={dec_means['mse_xyz']:.2e} rpy={dec_means['mse_rpy']:.2e} "
        f"grip={dec_means['mse_grip']:.2e}) | term|err|={dec_means['timing_abs']:.2f} "
        f"early={dec_means['early_rate']:.0%} late={dec_means['late_rate']:.0%} | progress|err|={dec_means['prog_err']:.3f}"
    )
    if args.model_family == "fsq_new":
        summary = summary.replace("chunk MSE=", "A chunk MSE=", 1)
    for label, means in dec_means_variants.items():
        summary += f" | {label} chunk MSE={means['chunk_mse']:.3e}"
    title = f"{Path(args.model_path).parent.name} ({Path(args.model_path).stem})"
    html = build_html(title, summary, levels, counts, entry_data, samples, codebook_size)
    html_path = out_dir / "fsq_eval.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"[fsq_eval] HTML → {html_path}")

    if not args.no_wandb:
        log_wandb(
            args,
            enc_stats,
            dec_means,
            len(decode_ids),
            html_path,
            action_variant_means={
                **(
                    {"A": {key: dec_means[key] for key in action_keys}}
                    if args.model_family == "fsq_new"
                    else {}
                ),
                **dec_means_variants,
            },
        )


if __name__ == "__main__":
    main()
