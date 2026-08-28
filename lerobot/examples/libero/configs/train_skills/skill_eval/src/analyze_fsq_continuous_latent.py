#!/usr/bin/env python3
"""Visualize pre-round FSQ latents and frozen-decoder assignment quality."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from analyze_fsq_rate_distortion import decoder_prototypes, normalized_targets
from visualize_fsq_spline_io import (
    cfg_value,
    load_bundle,
    load_fsq_model,
    load_stats,
    resolve_checkpoint,
    scalar_text,
    spline_encode,
)


MODEL_GROUPS = {
    "pair_route": (
        ("none_route_OFF", "zero_recon_only__pairOFF_routeOFF_loss"),
        ("none_route_ON", "zero_recon_only__pairOFF_routeON_loss"),
    ),
    "terminator_effect3": (
        ("recon_only", "norm_action01_recon_only__jsON_routeON_loss"),
        ("termRES_base", "norm_action01_recon_termRES__jsON_routeON_loss__layer2_base"),
        ("termRES_01", "norm_action01_recon_termRES__jsON_routeON_loss__term01"),
        (
            "termRES_unfreeze_01",
            "norm_action01_recon_termRES__jsON_routeON_loss__vis_unfreeze_term01",
        ),
    ),
    "norm_action_01": (
        ("contrastive", "norm_action01_recon_only__contrastiveON_routeON_loss"),
        ("js", "norm_action01_recon_only__jsON_routeON_loss"),
        ("pair_OFF", "norm_action01_recon_only__pairOFF_routeON_loss"),
    ),
}
PROFILE_INFO = {
    "pair_route": {
        "parent": "2layer_full",
        "figure_title": "FSQ333 continuous encoder space · pair OFF · route OFF vs ON",
        "lead": "zero-grounded · pair loss OFF · 2-layer · 전체 11,221 skills. route loss만 OFF/ON으로 다르다.",
    },
    "terminator_effect3": {
        "parent": "terminator_effect3",
        "figure_title": "FSQ333 continuous encoder space · normalized action · terminator effect",
        "lead": "normalized action · gripper weight 0.1 · JS + route ON · 전체 11,221 skills. reconstruction-only와 세 terminator 조건을 비교한다.",
    },
    "norm_action_01": {
        "parent": "norm_action_01",
        "figure_title": "FSQ333 continuous encoder space · normalized action · pair objective",
        "lead": "normalized action · gripper weight 0.1 · reconstruction-only · route ON · 전체 11,221 skills. contrastive, JS, pair OFF를 비교한다.",
    },
}
DEFAULT_EPOCHS = (50, 100, 250, 500, 1000, 1500, 2000)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[7]
    eval_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skill-bundle",
        type=Path,
        default=root
        / "dataset_filtered/FSQ_dataset/libero_90_full_full/FSQ_inputs"
        / "seg_libero_90_full_full_state_obs20_ck100000_std_episodemean_100p"
        / "skillset/skills_bundle.npz",
    )
    parser.add_argument("--fsq-root", type=Path, default=root / "outputs_filtered/FSQ")
    parser.add_argument("--profile", choices=sorted(MODEL_GROUPS), default="pair_route")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, nargs="+", default=list(DEFAULT_EPOCHS))
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.output_dir is None:
        parent = PROFILE_INFO[args.profile]["parent"]
        args.output_dir = (
            eval_root / f"outputs/fsq_gt_replay/{parent}/continuous_latent_analysis"
        )
    return args


def prepare_encoder_inputs(
    bundle: dict[str, np.ndarray], *, mode: str, n_control: int, degree: int
) -> tuple[np.ndarray, np.ndarray]:
    lengths = bundle["states_len"].astype(np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths[:-1])))
    controls = np.empty((len(lengths), n_control, 8), dtype=np.float32)
    states_cat = bundle["states_cat"]
    for index, (offset, length) in enumerate(zip(offsets, lengths, strict=True)):
        states = states_cat[int(offset) : int(offset + length)].astype(np.float32, copy=False)
        controls[index], _ = spline_encode(
            states, n_control, degree, input_mode=mode
        )
        if (index + 1) % 2000 == 0:
            print(f"[{mode}] encoder inputs {index + 1:,}/{len(lengths):,}", flush=True)
    return controls, lengths


def prepare_action_inputs(
    bundle: dict[str, np.ndarray], model: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Build the exact padded normalized-action tensor consumed in training."""
    lengths = bundle["actions_len"].astype(np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths[:-1])))
    normalized_cat = model._prepare_actions_numpy(bundle["actions_cat"]).numpy()
    actions = np.zeros(
        (len(lengths), int(lengths.max()), normalized_cat.shape[-1]), dtype=np.float32
    )
    for index, (offset, length) in enumerate(zip(offsets, lengths, strict=True)):
        actions[index, : int(length)] = normalized_cat[
            int(offset) : int(offset + length)
        ]
    return actions, lengths


@torch.inference_mode()
def encode_all(
    model: Any,
    controls: np.ndarray,
    lengths: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_all = np.empty((len(controls), 3), dtype=np.float32)
    bounded_all = np.empty((len(controls), 3), dtype=np.float32)
    code_all = np.empty(len(controls), dtype=np.int32)
    # Length bucketing avoids evaluating every action-sequence batch at the
    # dataset-wide maximum length while restoring original order on output.
    order = np.argsort(lengths, kind="stable")
    for start in range(0, len(order), batch_size):
        stop = min(start + batch_size, len(order))
        indices = order[start:stop]
        lens_np = lengths[indices]
        steps = int(lens_np.max())
        ctrl = torch.from_numpy(controls[indices, :steps])
        lens = torch.from_numpy(lens_np)
        z_e = model.encoder.encode_continuous(ctrl, lens, normalized=False)
        bounded = model.fsq.bound(z_e)
        _, code = model.fsq(z_e)
        raw_all[indices] = z_e.cpu().numpy().astype(np.float32)
        bounded_all[indices] = bounded.cpu().numpy().astype(np.float32)
        code_all[indices] = code.cpu().numpy().astype(np.int32)
        if stop % 2048 < batch_size or stop == len(controls):
            print(f"encoded {stop:,}/{len(controls):,}", flush=True)
    return raw_all, bounded_all, code_all


def oracle_assignments(
    targets: np.ndarray, prototypes: np.ndarray, *, batch_size: int = 512
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    best_codes = np.empty(len(targets), dtype=np.int32)
    best_mse = np.empty(len(targets), dtype=np.float32)
    all_code_mse = np.empty((len(targets), len(prototypes)), dtype=np.float32)
    for start in range(0, len(targets), batch_size):
        stop = min(start + batch_size, len(targets))
        error = np.mean(
            np.square(targets[start:stop, None] - prototypes[None]), axis=(2, 3)
        )
        all_code_mse[start:stop] = error
        best_codes[start:stop] = error.argmin(axis=1)
        best_mse[start:stop] = error.min(axis=1)
    return best_codes, best_mse, all_code_mse


@torch.inference_mode()
def action_decoder_prototypes(
    model: Any, n_codes: int, steps: int
) -> np.ndarray:
    if getattr(model.reconstructor, "state_dim", 0) != 0:
        raise ValueError("Action decoder oracle does not support start-state conditioning.")
    codes = torch.arange(n_codes, dtype=torch.long)
    z_norm = model.fsq.code_to_normalized(codes)
    predicted, _ = model.reconstructor(z_norm, int(steps), start_state=None)
    return predicted.cpu().numpy().astype(np.float32)


def action_oracle_assignments(
    targets: np.ndarray,
    lengths: np.ndarray,
    prototypes: np.ndarray,
    *,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples, max_steps, action_dim = targets.shape
    n_codes = len(prototypes)
    best_codes = np.empty(n_samples, dtype=np.int32)
    best_mse = np.empty(n_samples, dtype=np.float32)
    all_code_mse = np.empty((n_samples, n_codes), dtype=np.float32)
    timeline = np.arange(max_steps)
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        mask = timeline[None] < lengths[start:stop, None]
        squared = np.square(
            targets[start:stop, None] - prototypes[None]
        )
        error = (squared * mask[:, None, :, None]).sum(axis=(2, 3))
        error /= lengths[start:stop, None] * action_dim
        all_code_mse[start:stop] = error
        best_codes[start:stop] = error.argmin(axis=1)
        best_mse[start:stop] = error.min(axis=1)
    return best_codes, best_mse, all_code_mse


def action_fixed_assignment_centroid_mse(
    targets: np.ndarray, lengths: np.ndarray, assignments: np.ndarray, n_codes: int
) -> float:
    max_steps = targets.shape[1]
    timeline = np.arange(max_steps)
    squared_error_sum = 0.0
    valid_elements = float(lengths.sum() * targets.shape[-1])
    for code in range(n_codes):
        members = np.flatnonzero(assignments == code)
        if len(members) == 0:
            continue
        mask = timeline[None] < lengths[members, None]
        count = mask.sum(axis=0)
        centroid = np.divide(
            (targets[members] * mask[..., None]).sum(axis=0),
            count[:, None],
            out=np.zeros_like(targets[0]),
            where=count[:, None] > 0,
        )
        squared_error_sum += float(
            (np.square(targets[members] - centroid) * mask[..., None]).sum()
        )
    return squared_error_sum / valid_elements


def code_coordinates(model: Any, n_codes: int) -> np.ndarray:
    with torch.inference_mode():
        codes = torch.arange(n_codes)
        return model.fsq.code_to_normalized(codes).cpu().numpy().astype(np.float32)


def equal_window_display(points: np.ndarray) -> np.ndarray:
    """Stretch the two clipped outer FSQ cells for an equal-window display.

    The real bounded coordinate occupies [-1,-.5], [-.5,.5], and [.5,1].
    For visualization only, map those intervals to three equal-width windows
    [-1.5,-.5], [-.5,.5], and [.5,1.5]. Metrics always use the real points.
    """
    shown = points.copy()
    lower = points < -0.5
    upper = points > 0.5
    shown[lower] = 2.0 * points[lower] + 0.5
    shown[upper] = 2.0 * points[upper] - 0.5
    return shown


def summarize_model(
    *,
    label: str,
    run_dir: Path,
    checkpoint_name: str,
    bundle: dict[str, np.ndarray],
    controls_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]],
    targets_cache: dict[tuple[Any, ...], np.ndarray],
    batch_size: int,
) -> dict[str, Any]:
    checkpoint = resolve_checkpoint(run_dir, checkpoint_name)
    print(f"[{label}] loading {checkpoint}", flush=True)
    model, cfg = load_fsq_model(checkpoint, device="cpu")
    model.eval()
    encoder_arch = str(cfg_value(cfg, "encoder_arch"))
    action_sequence = encoder_arch == "action_seq"
    if action_sequence:
        cache_key = (
            "action_seq",
            np.asarray(cfg_value(cfg, "action_q01")).tobytes(),
            np.asarray(cfg_value(cfg, "action_q99")).tobytes(),
            float(cfg_value(cfg, "action_gripper_weight")),
            str(cfg_value(cfg, "autoencoder_mode")),
        )
        if cache_key not in controls_cache:
            controls_cache[cache_key] = prepare_action_inputs(bundle, model)
    else:
        stats = load_stats(run_dir)
        input_mode = str(cfg_value(cfg, "encoder_input_mode"))
        n_control = int(cfg_value(cfg, "n_control"))
        degree = int(cfg_value(cfg, "spline_degree"))
        cache_key = ("spline", input_mode, n_control, degree)
        if cache_key not in controls_cache:
            controls_cache[cache_key] = prepare_encoder_inputs(
                bundle, mode=input_mode, n_control=n_control, degree=degree
            )
    controls, lengths = controls_cache[cache_key]
    raw, bounded, assigned = encode_all(
        model, controls, lengths, batch_size=batch_size
    )
    n_codes = int(model.fsq.codebook_size)
    counts = np.bincount(assigned, minlength=n_codes)

    if action_sequence:
        targets = controls
        prototypes = action_decoder_prototypes(
            model, n_codes, int(lengths.max())
        )
        oracle_code, oracle_mse, all_code_mse = action_oracle_assignments(
            targets, lengths, prototypes
        )
        assigned_mse = all_code_mse[np.arange(len(targets)), assigned]
        weights = lengths.astype(np.float64)
        model_mse = float(np.average(assigned_mse, weights=weights))
        oracle_value = float(np.average(oracle_mse, weights=weights))
        centroid_mse = action_fixed_assignment_centroid_mse(
            targets, lengths, assigned, n_codes
        )
        reconstruction_space = "normalized_action"
    else:
        output_mode = scalar_text(stats["reconstructor_output_mode"])
        target_key = (
            output_mode,
            n_control,
            degree,
            stats["reconstructor_min"].tobytes(),
            stats["reconstructor_max"].tobytes(),
        )
        if target_key not in targets_cache:
            targets_cache[target_key] = normalized_targets(
                bundle,
                mode=output_mode,
                minimum=stats["reconstructor_min"],
                maximum=stats["reconstructor_max"],
                n_control=n_control,
                degree=degree,
            )
        targets = targets_cache[target_key]
        prototypes = decoder_prototypes(model, n_codes)
        oracle_code, oracle_mse, all_code_mse = oracle_assignments(
            targets, prototypes
        )
        assigned_mse = all_code_mse[np.arange(len(targets)), assigned]
        centroids = np.full_like(prototypes, np.nan)
        for code in np.flatnonzero(counts):
            centroids[code] = targets[assigned == code].mean(axis=0)
        centroid_prediction = centroids[assigned]
        centroid_mse = float(np.mean(np.square(targets - centroid_prediction)))
        model_mse = float(assigned_mse.mean())
        oracle_value = float(oracle_mse.mean())
        reconstruction_space = "normalized_spline"

    gain = assigned_mse - oracle_mse
    decoder_gap = model_mse - centroid_mse
    mismatch = assigned != oracle_code
    min_margin = np.min(0.5 - np.abs(bounded - np.round(bounded)), axis=1)
    probability = counts[counts > 0].astype(np.float64) / len(assigned)

    result = {
        "label": label,
        "run_dir": str(run_dir),
        "checkpoint": checkpoint.name,
        "reconstruction_space": reconstruction_space,
        "raw": raw,
        "bounded": bounded,
        "assigned": assigned,
        "oracle_code": oracle_code,
        "gain": gain,
        "margin": min_margin.astype(np.float32),
        "code_xyz": code_coordinates(model, n_codes),
        "task_id": bundle["meta_task_id"].astype(np.int32),
        "skill_index": bundle["meta_skill_index"].astype(np.int32),
        "metrics": {
            "n": int(len(assigned)),
            "active_codes": int(np.count_nonzero(counts)),
            "effective_codes": float(np.exp(-(probability * np.log(probability)).sum())),
            "top_code_pct": float(100 * counts.max() / len(assigned)),
            "assigned_decoder_mse": model_mse,
            "fixed_assignment_centroid_mse": centroid_mse,
            "decoder_gap": decoder_gap,
            "decoder_oracle_mse": oracle_value,
            "oracle_gain": model_mse - oracle_value,
            "oracle_relative_gain_pct": float(100 * (model_mse - oracle_value) / model_mse),
            "assigned_oracle_agreement_pct": float(100 * np.mean(~mismatch)),
            "mismatch_pct": float(100 * np.mean(mismatch)),
            "median_boundary_margin": float(np.median(min_margin)),
            "near_boundary_005_pct": float(100 * np.mean(min_margin < 0.05)),
        },
        "counts": counts.astype(int).tolist(),
    }
    print(f"[{label}] {json.dumps(result['metrics'], indent=2)}", flush=True)
    return result


def add_grid(axis: Any, centers: np.ndarray) -> None:
    del centers  # The clean window view intentionally omits 27 center markers.
    low, high = -1.5, 1.5
    outer = (low, high)
    boundaries = (-0.5, 0.5)

    # Twelve outer edges.
    for first in outer:
        for second in outer:
            axis.plot(
                [low, high], [first, first], [second, second],
                color="#334155", alpha=0.72, linewidth=1.15,
            )
            axis.plot(
                [first, first], [low, high], [second, second],
                color="#334155", alpha=0.72, linewidth=1.15,
            )
            axis.plot(
                [first, first], [second, second], [low, high],
                color="#334155", alpha=0.72, linewidth=1.15,
            )

    # Only three visible faces carry subdivision lines. This communicates the
    # 3×3×3 equal windows without drawing a dense 48-edge lattice through data.
    for boundary in boundaries:
        style = {"color": "#64748b", "alpha": 0.52, "linewidth": 0.85}
        # Bottom face z=low.
        axis.plot([boundary, boundary], [low, high], [low, low], **style)
        axis.plot([low, high], [boundary, boundary], [low, low], **style)
        # Back face y=high.
        axis.plot([boundary, boundary], [high, high], [low, high], **style)
        axis.plot([low, high], [high, high], [boundary, boundary], **style)
        # Left face x=low.
        axis.plot([low, low], [boundary, boundary], [low, high], **style)
        axis.plot([low, low], [low, high], [boundary, boundary], **style)

    axis.grid(False)
    axis.xaxis.pane.set_alpha(0.0)
    axis.yaxis.pane.set_alpha(0.0)
    axis.zaxis.pane.set_alpha(0.0)
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_zlim(low, high)
    axis.set_box_aspect((1, 1, 1))
    ticks = (-1, 0, 1)
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.set_zticks(ticks)


def make_static_figure(
    results: list[dict[str, Any]], output: Path, *, epoch: int, title: str
) -> None:
    fig = plt.figure(figsize=(15, 6.3 * len(results)), constrained_layout=True)
    palette = plt.get_cmap("turbo", 27)
    for row, result in enumerate(results):
        raw = result["raw"]
        bounded = equal_window_display(result["bounded"])
        colors = palette(result["assigned"] / 26.0)

        raw_axis = fig.add_subplot(len(results), 2, row * 2 + 1, projection="3d")
        limits = np.quantile(raw, [0.005, 0.995], axis=0)
        visible = np.all((raw >= limits[0]) & (raw <= limits[1]), axis=1)
        raw_axis.scatter(
            raw[visible, 0], raw[visible, 1], raw[visible, 2],
            s=7, c=colors[visible], alpha=0.28, linewidths=0, rasterized=True,
        )
        raw_axis.set_title(f"{result['label']} · raw encoder output $z_e$")
        raw_axis.set_xlabel("z₁")
        raw_axis.set_ylabel("z₂")
        raw_axis.set_zlabel("z₃")
        raw_axis.view_init(elev=24, azim=-56)

        bound_axis = fig.add_subplot(len(results), 2, row * 2 + 2, projection="3d")
        bound_axis.scatter(
            bounded[:, 0], bounded[:, 1], bounded[:, 2],
            s=8, c=colors, alpha=0.32, linewidths=0, rasterized=True,
        )
        add_grid(bound_axis, result["code_xyz"])
        bound_axis.set_title(
            f"{result['label']} · pre-round latent · equal-window display"
        )
        bound_axis.set_xlabel("display w₁")
        bound_axis.set_ylabel("display w₂")
        bound_axis.set_zlabel("display w₃")
        bound_axis.view_init(elev=24, azim=-56)
    fig.suptitle(
        f"{title} · epoch {epoch}",
        fontsize=18,
    )
    fig.savefig(output, dpi=190)
    plt.close(fig)


def make_oracle_figure(
    results: list[dict[str, Any]], output: Path, *, epoch: int
) -> None:
    columns = min(2, len(results))
    rows = (len(results) + columns - 1) // columns
    fig = plt.figure(figsize=(15, 6.5 * rows), constrained_layout=True)
    for column, result in enumerate(results):
        axis = fig.add_subplot(rows, columns, column + 1, projection="3d")
        bounded = equal_window_display(result["bounded"])
        gain = np.maximum(result["gain"], 0)
        mismatch = result["assigned"] != result["oracle_code"]
        scale = np.quantile(gain[mismatch], 0.98) if np.any(mismatch) else 1.0
        color_value = np.clip(gain / max(float(scale), 1e-8), 0, 1)
        axis.scatter(
            bounded[~mismatch, 0], bounded[~mismatch, 1], bounded[~mismatch, 2],
            s=6, c="#64748b", alpha=0.10, linewidths=0, rasterized=True,
        )
        points = axis.scatter(
            bounded[mismatch, 0], bounded[mismatch, 1], bounded[mismatch, 2],
            s=10, c=color_value[mismatch], cmap="magma", vmin=0, vmax=1,
            alpha=0.48, linewidths=0, rasterized=True,
        )
        add_grid(axis, result["code_xyz"])
        metrics = result["metrics"]
        axis.set_title(
            f"{result['label']} · decoder-oracle mismatch {metrics['mismatch_pct']:.1f}%\n"
            f"MSE {metrics['assigned_decoder_mse']:.5f} → {metrics['decoder_oracle_mse']:.5f} "
            f"(−{metrics['oracle_relative_gain_pct']:.1f}%)"
        )
        axis.set_xlabel("display w₁")
        axis.set_ylabel("display w₂")
        axis.set_zlabel("display w₃")
        axis.view_init(elev=24, azim=-56)
        colorbar = fig.colorbar(points, ax=axis, shrink=0.62, pad=0.04)
        colorbar.set_label("current − best frozen-decoder MSE (scaled)")
    fig.suptitle(
        f"Frozen-decoder oracle mismatch · epoch {epoch}",
        fontsize=17,
    )
    fig.savefig(output, dpi=190)
    plt.close(fig)


def make_confusion_figure(
    results: list[dict[str, Any]], output: Path, *, epoch: int
) -> None:
    columns = min(2, len(results))
    rows = (len(results) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows, columns, figsize=(15, 5.8 * rows), constrained_layout=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for axis, result in zip(flat_axes, results, strict=False):
        matrix = np.zeros((27, 27), dtype=np.int64)
        np.add.at(matrix, (result["assigned"], result["oracle_code"]), 1)
        row_sum = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_sum, out=np.zeros_like(matrix, dtype=float), where=row_sum > 0)
        image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        axis.plot([-0.5, 26.5], [-0.5, 26.5], color="#ef4444", linewidth=1.2, alpha=0.8)
        axis.set_title(
            f"{result['label']} · agreement "
            f"{result['metrics']['assigned_oracle_agreement_pct']:.1f}%"
        )
        axis.set_xlabel("best code under frozen decoder")
        axis.set_ylabel("encoder-assigned code")
        axis.set_xticks(range(0, 27, 3))
        axis.set_yticks(range(0, 27, 3))
        fig.colorbar(image, ax=axis, shrink=0.82, label="row-normalized fraction")
    for axis in flat_axes[len(results) :]:
        axis.set_visible(False)
    fig.suptitle(
        f"Encoder assignment vs frozen-decoder oracle · epoch {epoch}", fontsize=17
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def json_ready(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {
            "raw", "bounded", "assigned", "oracle_code", "gain", "margin",
            "code_xyz", "task_id", "skill_index",
        }
    }


def metric_table(results: list[dict[str, Any]]) -> str:
    rows = []
    for result in results:
        m = result["metrics"]
        rows.append(
            "<tr>"
            f"<th>{html.escape(result['label'])}</th>"
            f"<td>{m['active_codes']}/27</td>"
            f"<td>{m['effective_codes']:.2f}</td>"
            f"<td>{m['assigned_decoder_mse']:.6f}</td>"
            f"<td>{m['fixed_assignment_centroid_mse']:.6f}</td>"
            f"<td>{m['decoder_gap']:.6f}</td>"
            f"<td>{m['decoder_oracle_mse']:.6f}</td>"
            f"<td>{m['oracle_relative_gain_pct']:.1f}%</td>"
            f"<td>{m['assigned_oracle_agreement_pct']:.1f}%</td>"
            f"<td>{m['near_boundary_005_pct']:.1f}%</td>"
            "</tr>"
        )
    return "".join(rows)


def write_report(
    epoch_results: dict[int, list[dict[str, Any]]],
    output_dir: Path,
    *,
    lead: str,
    missing_by_epoch: dict[int, list[str]],
) -> None:
    epochs = sorted(epoch_results)
    complete_epochs = [
        epoch for epoch in epochs if not missing_by_epoch.get(epoch)
    ]
    default_epoch = max(complete_epochs) if complete_epochs else max(epochs)
    browser_data = {
        str(epoch): {
            "directory": f"epoch_{epoch:04d}",
            "table": metric_table(results),
            "checkpoints": " · ".join(
                result["label"] + "=" + result["checkpoint"] for result in results
            ),
            "missing": missing_by_epoch.get(epoch, []),
        }
        for epoch, results in epoch_results.items()
    }
    buttons = "".join(
        f'<button class="epoch-button{" active" if epoch == default_epoch else ""}" '
        f'data-epoch="{epoch}" onclick="showEpoch({epoch})">epoch {epoch}</button>'
        for epoch in epochs
    )
    report = output_dir / "index.html"
    report.write_text(
        f"""<!doctype html><html lang=ko><head><meta charset=utf-8>
<meta name=viewport content=\"width=device-width,initial-scale=1\"><title>FSQ continuous latent</title>
<style>
body{{margin:0;background:#eef2f7;color:#172033;font:15px/1.62 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:34px}}h1{{font-size:34px;margin:.1em 0}}h2{{margin-top:34px}}.lead{{font-size:18px;color:#42526b;max-width:1100px}}.card{{background:white;border:1px solid #dbe3ef;border-radius:16px;padding:22px;margin:18px 0;box-shadow:0 8px 30px #24416a12}}.callout{{border-left:5px solid #2563eb;background:#eff6ff;padding:15px 18px;border-radius:10px;margin:15px 0}}.warn{{border-color:#d97706;background:#fff7ed}}img{{display:block;width:100%;height:auto;border-radius:10px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:10px 12px;border-bottom:1px solid #e5eaf2;text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{background:#e8eef7;padding:2px 5px;border-radius:5px}}small{{color:#667085}}.epoch-nav{{position:sticky;top:0;z-index:5;display:flex;gap:9px;flex-wrap:wrap;padding:13px;margin:20px 0;background:#eef2f7e8;backdrop-filter:blur(8px);border:1px solid #d7e0ed;border-radius:14px}}.epoch-button{{border:1px solid #b8c5d8;background:white;color:#334155;padding:9px 15px;border-radius:999px;font-weight:700;cursor:pointer}}.epoch-button:hover{{border-color:#2563eb;color:#1d4ed8}}.epoch-button.active{{background:#2563eb;border-color:#2563eb;color:white}}#selected-epoch{{color:#2563eb}}
</style></head><body><main>
<h1>FSQ333 continuous encoder space</h1>
<p class=lead>{html.escape(lead)} 현재 <b id=selected-epoch>epoch {default_epoch}</b>.</p>
<nav class=epoch-nav>{buttons}</nav>
<div class=callout><b>좌표를 읽는 법:</b> raw <code>z_e</code>는 encoder linear head의 출력이다. 실제 round는 여기에 바로 적용되지 않고, <code>u = tanh(z_e)</code>로 [−1,1]에 bound한 뒤 적용된다. 실제 decision interval은 [−1,−0.5], [−0.5,0.5], [0.5,1]이다. 오른쪽 그림은 세 window를 같은 크기로 비교하기 위해서만 바깥 두 interval을 2배 늘려 [−1.5,−0.5], [−0.5,0.5], [0.5,1.5]로 표시한 <b>equal-window display</b>다. assignment, boundary margin 및 모든 metric은 변환 전 실제 <code>u</code>로 계산했다.</div>
<section class=card><img id=latent-image alt=\"continuous FSQ latent comparison\"></section>
<h2>Frozen-decoder oracle test</h2>
<div class=callout><b>K-means를 쓰지 않은 직접 검증:</b> 학습된 decoder를 고정한 뒤 각 target에 27개 code를 모두 넣는다. encoder가 고른 code보다 더 낮은 reconstruction MSE를 내는 code가 있으면, decoder가 이미 제공하는 선택지 중에서도 encoder가 최선의 code를 고르지 못한 것이다. 회색은 encoder code가 decoder-optimal이고, 유색 점은 불일치이며 밝을수록 바꿨을 때의 MSE 감소가 크다.</div>
<section class=card><img id=oracle-image alt=\"decoder oracle mismatch in continuous latent\"></section>
<section class=card><img id=confusion-image alt=\"encoder and decoder oracle confusion matrix\"></section>
<h2>정량 결과</h2><section class=card><div class=scroll><table><thead><tr><th>model</th><th>active</th><th>effective</th><th>actual decoder MSE</th><th>fixed-assignment centroid</th><th>decoder gap</th><th>best-of-27 decoder MSE</th><th>oracle improvement</th><th>assignment agreement</th><th>boundary &lt;.05</th></tr></thead><tbody id=metric-body></tbody></table></div></section>
<div class=callout><b>“decoder가 현재 code space에 최적화됐다”의 기준:</b> actual decoder MSE와 fixed-assignment centroid MSE가 가까울수록, decoder 출력은 현재 encoder가 각 code에 넣어준 target들의 MSE-optimal 평균에 가깝다. 이 차이가 작은데 best-of-27 decoder 개선이 크다면, decoder fitting보다 encoder-to-code routing이 병목이라는 증거다.</div>
<div class=callout warn><b>해석 한계:</b> continuous point cloud가 겹쳐 보이는 것만으로 encoder가 나쁘다고 단정하면 안 된다. FSQ는 cell 내부 위치가 아니라 round 결과만 decoder에 전달한다. 그래서 결론은 반드시 decoder-gap 및 frozen-decoder oracle 수치와 함께 내려야 한다. 또한 oracle은 reconstruction target 기준이며 semantic/downstream category 정답은 아니다.</div>
<p><small id=checkpoint-line></small></p>
<p><small id=missing-line></small></p>
<script>
const epochData = {json.dumps(browser_data, ensure_ascii=False)};
function showEpoch(epoch) {{
  const item = epochData[String(epoch)];
  document.getElementById('selected-epoch').textContent = `epoch ${{epoch}}`;
  document.getElementById('latent-image').src = `${{item.directory}}/continuous_latent_3d.png`;
  document.getElementById('oracle-image').src = `${{item.directory}}/decoder_oracle_latent_3d.png`;
  document.getElementById('confusion-image').src = `${{item.directory}}/encoder_vs_decoder_oracle.png`;
  document.getElementById('metric-body').innerHTML = item.table;
  document.getElementById('checkpoint-line').textContent = `Checkpoints: ${{item.checkpoints}}`;
  document.getElementById('missing-line').textContent = item.missing.length
    ? `Unavailable at this epoch: ${{item.missing.join(', ')}}`
    : '';
  document.querySelectorAll('.epoch-button').forEach(button =>
    button.classList.toggle('active', Number(button.dataset.epoch) === Number(epoch)));
}}
showEpoch({default_epoch});
</script>
</main></body></html>""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(args.skill_bundle.resolve())
    controls_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = {}
    targets_cache: dict[tuple[Any, ...], np.ndarray] = {}
    epoch_results: dict[int, list[dict[str, Any]]] = {}
    missing_by_epoch: dict[int, list[str]] = {}
    for epoch in sorted(set(args.epochs)):
        checkpoint = f"FSQ_epoch{epoch:04d}.pt"
        results = []
        missing = []
        for label, directory in MODEL_GROUPS[args.profile]:
            run_dir = args.fsq_root / directory
            if not (run_dir / checkpoint).is_file():
                print(f"[{label}] missing {checkpoint}; skipped", flush=True)
                missing.append(label)
                continue
            results.append(
                summarize_model(
                    label=label,
                    run_dir=run_dir,
                    checkpoint_name=checkpoint,
                    bundle=bundle,
                    controls_cache=controls_cache,
                    targets_cache=targets_cache,
                    batch_size=args.batch_size,
                )
            )
        if not results:
            print(f"[epoch {epoch}] no checkpoints; skipped", flush=True)
            continue
        epoch_results[epoch] = results
        missing_by_epoch[epoch] = missing
        epoch_dir = args.output_dir / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        make_static_figure(
            results,
            epoch_dir / "continuous_latent_3d.png",
            epoch=epoch,
            title=PROFILE_INFO[args.profile]["figure_title"],
        )
        make_oracle_figure(
            results, epoch_dir / "decoder_oracle_latent_3d.png", epoch=epoch
        )
        make_confusion_figure(
            results, epoch_dir / "encoder_vs_decoder_oracle.png", epoch=epoch
        )
        np.savez_compressed(
            epoch_dir / "continuous_latent_data.npz",
            **{
                f"{result['label']}_{key}": result[key]
                for result in results
                for key in (
                    "raw", "bounded", "assigned", "oracle_code", "gain", "margin",
                    "code_xyz", "task_id", "skill_index",
                )
            },
        )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(
            {
                str(epoch): [json_ready(result) for result in results]
                for epoch, results in epoch_results.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(
        epoch_results,
        args.output_dir,
        lead=PROFILE_INFO[args.profile]["lead"],
        missing_by_epoch=missing_by_epoch,
    )
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()
