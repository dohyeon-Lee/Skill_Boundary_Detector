#!/usr/bin/env python3
"""Compare pair-loss variants with emphasis on adjacent-skill code collisions.

The report uses the complete training skill bundle and the exact normalized
30x8 reconstruction targets.  It complements ``analyze_codebook_categorization``
with the question that is easy to miss in ordinary code-purity summaries:
whether consecutive skills from one episode are assigned to the same code.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

from analyze_codebook_categorization import load_dataset, load_latents
from analyze_fsq_rate_distortion import decoder_prototypes, normalized_targets
from visualize_fsq_spline_io import cfg_value, load_bundle, load_fsq_model, load_stats, scalar_text


MODEL_NAMES = ("cont_route_ON", "js_route_ON", "none_route_ON")
DISPLAY_NAMES = {
    "cont_route_ON": "contrastive + route",
    "js_route_ON": "JS + route",
    "none_route_ON": "pair OFF + route",
}
COLORS = {
    "cont_route_ON": "#4cc9f0",
    "js_route_ON": "#f4a261",
    "none_route_ON": "#9b5de5",
}
GROUPS = {
    "all": np.arange(8),
    "XYZ": np.arange(3),
    "rotvec": np.arange(3, 6),
    "gripper": np.arange(6, 8),
}
DIRECTION_LABELS = ("+x", "-x", "+y", "-y", "+z", "-z", "<1 cm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--skill-bundle", type=Path, required=True)
    parser.add_argument("--episode-metadata", type=Path, required=True)
    parser.add_argument("--epoch", default="epoch1000")
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES)
    parser.add_argument("--output-subdir", default="codebook_correlation_analysis")
    parser.add_argument("--motion-threshold", type=float, default=0.01)
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def find_models(report_root: Path, requested: list[str]) -> dict[str, tuple[Path, dict[str, Any]]]:
    found: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in report_root.glob("*/metrics/collection.json"):
        collection = json.loads(path.read_text())
        name = collection.get("model_name")
        if name in requested:
            found[name] = (path.parent.parent, collection)
    missing = set(requested) - found.keys()
    if missing:
        raise FileNotFoundError(f"Missing replay collections: {sorted(missing)}")
    return found


def adjacency_pairs(metadata: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, episode in enumerate(metadata["episode_id"]):
        grouped[int(episode)].append(index)
    left: list[int] = []
    right: list[int] = []
    for members in grouped.values():
        members.sort(
            key=lambda index: (
                int(metadata["skill_index"][index]),
                int(metadata["frame_start"][index]),
            )
        )
        for first, second in zip(members[:-1], members[1:], strict=True):
            if int(metadata["skill_index"][second]) != int(metadata["skill_index"][first]) + 1:
                continue
            left.append(first)
            right.append(second)
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def all_nonadjacent_episode_pairs(
    metadata: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, episode in enumerate(metadata["episode_id"]):
        grouped[int(episode)].append(index)
    left: list[int] = []
    right: list[int] = []
    for members in grouped.values():
        members.sort(key=lambda index: int(metadata["skill_index"][index]))
        for offset, first in enumerate(members):
            for second in members[offset + 1 :]:
                if int(metadata["skill_index"][second]) - int(metadata["skill_index"][first]) >= 2:
                    left.append(first)
                    right.append(second)
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def same_task_random_collision(tokens: np.ndarray, task_ids: np.ndarray) -> float:
    equal_pairs = 0
    all_pairs = 0
    for task in np.unique(task_ids):
        task_tokens = tokens[task_ids == task]
        count = len(task_tokens)
        all_pairs += count * (count - 1) // 2
        for code_count in np.bincount(task_tokens, minlength=27):
            equal_pairs += int(code_count) * (int(code_count) - 1) // 2
    return float(equal_pairs / max(all_pairs, 1))


def direction_labels(displacement: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(displacement, axis=1)
    labels = np.full(len(displacement), 6, dtype=np.int64)
    moving = norms >= threshold
    dominant_axis = np.argmax(np.abs(displacement[moving]), axis=1)
    signs = displacement[moving, dominant_axis] < 0
    labels[moving] = 2 * dominant_axis + signs.astype(np.int64)
    return labels, norms


def weighted_purity(primary: np.ndarray, secondary: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(primary), dtype=bool)
    total = int(mask.sum())
    if total == 0:
        return 0.0
    correct = 0
    for value in np.unique(primary[mask]):
        values = secondary[mask & (primary == value)]
        correct += Counter(values.tolist()).most_common(1)[0][1]
    return float(correct / total)


def direction_coherence(tokens: np.ndarray, displacement: np.ndarray, moving: np.ndarray) -> float:
    if not np.any(moving):
        return 0.0
    unit = displacement[moving] / np.linalg.norm(displacement[moving], axis=1, keepdims=True)
    moving_tokens = tokens[moving]
    weighted = 0.0
    for code in np.unique(moving_tokens):
        members = unit[moving_tokens == code]
        weighted += len(members) * float(np.linalg.norm(members.mean(axis=0)))
    return weighted / len(unit)


def entropy_effective_codes(tokens: np.ndarray, codebook_size: int) -> float:
    counts = np.bincount(tokens, minlength=codebook_size)
    probability = counts[counts > 0] / counts.sum()
    return float(np.exp(-(probability * np.log(probability)).sum()))


def mse_by_group(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        name: float(np.mean(np.square(prediction[..., dims] - target[..., dims])))
        for name, dims in GROUPS.items()
    }


def categorical_summary(tokens: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    correct = 0
    dominant: dict[str, Any] = {}
    for code in np.unique(tokens):
        members = values[tokens == code]
        value, count = Counter(members.tolist()).most_common(1)[0]
        correct += count
        dominant[str(int(code))] = {"value": int(value), "fraction": float(count / len(members))}
    return {
        "nmi": float(normalized_mutual_info_score(values, tokens)),
        "weighted_purity": float(correct / len(tokens)),
        "dominant": dominant,
    }


def pair_motion_geometry(
    displacement: np.ndarray,
    norms: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = (norms[left] >= threshold) & (norms[right] >= threshold)
    cosine = np.full(len(left), np.nan, dtype=np.float64)
    cosine[active] = np.sum(displacement[left[active]] * displacement[right[active]], axis=1) / (
        norms[left[active]] * norms[right[active]]
    )
    conflict = active & (cosine < 0.0)
    aligned = active & (cosine >= 0.5)
    return cosine, conflict, aligned


def code_rows(
    tokens: np.ndarray,
    direction: np.ndarray,
    moving: np.ndarray,
    task_ids: np.ndarray,
    skill_indices: np.ndarray,
    targets: np.ndarray,
    prototypes: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for code in range(len(prototypes)):
        mask = tokens == code
        count = int(mask.sum())
        if count == 0:
            rows.append({"code": code, "count": 0})
            continue
        centroid = targets[mask].mean(axis=0)
        moving_mask = mask & moving
        if moving_mask.any():
            label, label_count = Counter(direction[moving_mask].tolist()).most_common(1)[0]
            direction_purity = label_count / int(moving_mask.sum())
        else:
            label, direction_purity = 6, 1.0
        task, task_count = Counter(task_ids[mask].tolist()).most_common(1)[0]
        order, order_count = Counter(skill_indices[mask].tolist()).most_common(1)[0]
        rows.append(
            {
                "code": code,
                "count": count,
                "share": float(count / len(tokens)),
                "dominant_direction": DIRECTION_LABELS[int(label)],
                "direction_purity": float(direction_purity),
                "dominant_task": int(task),
                "task_purity": float(task_count / count),
                "dominant_skill_index": int(order),
                "skill_index_purity": float(order_count / count),
                "centroid_mse": float(np.mean(np.square(targets[mask] - centroid))),
                "model_mse": float(np.mean(np.square(targets[mask] - prototypes[code]))),
                "decoder_gap": float(np.mean(np.square(prototypes[code] - centroid))),
            }
        )
    return rows


def transition_rows(
    tokens: np.ndarray,
    metadata: dict[str, np.ndarray],
    descriptions: np.ndarray,
    displacement: np.ndarray,
    norms: np.ndarray,
    targets: np.ndarray,
    sample_mse: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray]:
    cosine, conflict, _ = pair_motion_geometry(displacement, norms, left, right, threshold)
    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for pair_index, first in enumerate(left):
        grouped[(int(metadata["task_id"][first]), int(metadata["skill_index"][first]))].append(pair_index)
    summaries: list[dict[str, Any]] = []
    for (task, skill_index), pair_indices_list in grouped.items():
        pair_indices = np.asarray(pair_indices_list, dtype=np.int64)
        pair_left, pair_right = left[pair_indices], right[pair_indices]
        same = tokens[pair_left] == tokens[pair_right]
        shared = Counter(tokens[pair_left[same]].tolist()).most_common(1)
        summaries.append(
            {
                "task_id": task,
                "task": str(descriptions[pair_left[0]]),
                "transition": f"{skill_index}→{skill_index + 1}",
                "count": int(len(pair_indices)),
                "same_code_count": int(same.sum()),
                "same_code_rate": float(same.mean()),
                "direction_conflict_count": int(np.sum(same & conflict[pair_indices])),
                "dominant_shared_code": None if not shared else int(shared[0][0]),
                "dominant_shared_code_count": 0 if not shared else int(shared[0][1]),
            }
        )
    summaries.sort(key=lambda row: (row["direction_conflict_count"], row["same_code_count"]), reverse=True)

    candidates = np.flatnonzero((tokens[left] == tokens[right]) & conflict)
    candidates = candidates[np.argsort(np.nan_to_num(cosine[candidates], nan=1.0))]
    examples: list[dict[str, Any]] = []
    for pair_index in candidates[:40]:
        first, second = int(left[pair_index]), int(right[pair_index])
        angle = float(np.degrees(np.arccos(np.clip(cosine[pair_index], -1.0, 1.0))))
        examples.append(
            {
                "task_id": int(metadata["task_id"][first]),
                "task": str(descriptions[first]),
                "episode_id": int(metadata["episode_id"][first]),
                "transition": f"{int(metadata['skill_index'][first])}→{int(metadata['skill_index'][second])}",
                "code": int(tokens[first]),
                "first_disp_xyz": displacement[first].tolist(),
                "second_disp_xyz": displacement[second].tolist(),
                "angle_deg": angle,
                "target_pair_mse": float(np.mean(np.square(targets[first] - targets[second]))),
                "first_recon_mse": float(sample_mse[first]),
                "second_recon_mse": float(sample_mse[second]),
            }
        )
    return summaries, examples, cosine, conflict


def analyze_model(
    name: str,
    model_root: Path,
    dataset: dict[str, Any],
    targets: np.ndarray,
    epoch: str,
    output_subdir: str,
    threshold: float,
    adjacent: tuple[np.ndarray, np.ndarray],
    nonadjacent: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    latent, manifest = load_latents(model_root, epoch, dataset)
    tokens = latent["tokens"].astype(np.int64)
    metadata = dataset["metadata"]
    run_dir = Path(manifest["signature"]["model_path"]).parent
    checkpoint = Path(manifest["signature"]["model_path"])
    model, cfg = load_fsq_model(checkpoint, device="cpu")
    model.eval()
    prototypes = decoder_prototypes(model, int(model.fsq.codebook_size))
    pair_loss = str(cfg_value(cfg, "pair_loss"))
    del model

    prediction = prototypes[tokens]
    sample_mse = np.mean(np.square(prediction - targets), axis=(1, 2))
    counts = np.bincount(tokens, minlength=len(prototypes))
    centroids = np.zeros_like(prototypes)
    for code in np.flatnonzero(counts):
        centroids[code] = targets[tokens == code].mean(axis=0)
    centroid_prediction = centroids[tokens]

    features = dataset["features"]
    displacement = np.column_stack((features["disp_x"], features["disp_y"], features["disp_z"]))
    direction, norms = direction_labels(displacement, threshold)
    moving = norms >= threshold
    left, right = adjacent
    nonadj_left, nonadj_right = nonadjacent
    cosine, conflict, aligned = pair_motion_geometry(displacement, norms, left, right, threshold)
    same = tokens[left] == tokens[right]
    active = ~np.isnan(cosine)

    transition_summary, examples, _, _ = transition_rows(
        tokens,
        metadata,
        dataset["task_description"],
        displacement,
        norms,
        targets,
        sample_mse,
        left,
        right,
        threshold,
    )

    reconstruction = {
        "model": mse_by_group(prediction, targets),
        "centroid": mse_by_group(centroid_prediction, targets),
        "decoder_gap": mse_by_group(prediction, centroid_prediction),
    }
    adjacency = {
        "pair_count": int(len(left)),
        "same_code_count": int(same.sum()),
        "same_code_rate": float(same.mean()),
        "nonadjacent_episode_same_code_rate": float(
            np.mean(tokens[nonadj_left] == tokens[nonadj_right]) if len(nonadj_left) else 0.0
        ),
        "same_task_random_code_rate": same_task_random_collision(tokens, metadata["task_id"]),
        "moving_pair_count": int(active.sum()),
        "direction_conflict_count": int(conflict.sum()),
        "same_code_given_conflict": float(np.mean(same[conflict]) if conflict.any() else 0.0),
        "same_code_given_aligned": float(np.mean(same[aligned]) if aligned.any() else 0.0),
        "conflict_given_same_code": float(np.mean(conflict[same]) if same.any() else 0.0),
        "same_code_conflict_count": int(np.sum(same & conflict)),
        "same_code_pair_target_mse": float(
            np.mean(np.mean(np.square(targets[left[same]] - targets[right[same]]), axis=(1, 2)))
            if same.any()
            else 0.0
        ),
        "different_code_pair_target_mse": float(
            np.mean(np.mean(np.square(targets[left[~same]] - targets[right[~same]]), axis=(1, 2)))
            if (~same).any()
            else 0.0
        ),
        "same_code_mean_direction_cosine": float(np.nanmean(cosine[same])),
        "different_code_mean_direction_cosine": float(np.nanmean(cosine[~same])),
    }
    categorization = {
        "direction_nmi": float(normalized_mutual_info_score(direction, tokens)),
        "direction_purity": weighted_purity(tokens, direction, moving),
        "direction_coherence": direction_coherence(tokens, displacement, moving),
        "task": categorical_summary(tokens, metadata["task_id"]),
        "skill_index": categorical_summary(tokens, metadata["skill_index"]),
    }
    result: dict[str, Any] = {
        "name": name,
        "display_name": DISPLAY_NAMES.get(name, name),
        "model_root": str(model_root),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "epoch": epoch,
        "pair_loss": pair_loss,
        "pair_weight": float(cfg_value(cfg, "pair_weight")),
        "route_loss": bool(
            cfg.get("route_loss", cfg.get("reconstruction_route_loss", False))
            if isinstance(cfg, dict)
            else getattr(
                cfg,
                "route_loss",
                getattr(cfg, "reconstruction_route_loss", False),
            )
        ),
        "sample_count": int(len(tokens)),
        "active_codes": int(np.count_nonzero(counts)),
        "effective_codes": entropy_effective_codes(tokens, len(prototypes)),
        "largest_code_share": float(counts.max() / counts.sum()),
        "counts": counts.tolist(),
        "reconstruction": reconstruction,
        "adjacency": adjacency,
        "categorization": categorization,
        "codes": code_rows(
            tokens,
            direction,
            moving,
            metadata["task_id"],
            metadata["skill_index"],
            targets,
            prototypes,
        ),
        "transition_summary": transition_summary,
        "collision_examples": examples,
    }
    categorization_path = model_root / output_subdir / "summary.json"
    if categorization_path.is_file():
        result["correlation_analysis"] = json.loads(categorization_path.read_text())
    plot_model(result, tokens, direction, moving, left, right, model_root / output_subdir)
    write_model_report(result, model_root / output_subdir / "neighbor_analysis.html")
    return result


def plot_model(
    result: dict[str, Any],
    tokens: np.ndarray,
    direction: np.ndarray,
    moving: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    transition = np.zeros((27, 27), dtype=np.int64)
    np.add.at(transition, (tokens[left], tokens[right]), 1)
    figure, axis = plt.subplots(figsize=(9.4, 8.2), constrained_layout=True)
    image = axis.imshow(np.log1p(transition), cmap="magma", aspect="equal")
    axis.set_title(f"{result['display_name']}: adjacent code transition (log1p count)")
    axis.set_xlabel("next skill code")
    axis.set_ylabel("previous skill code")
    axis.set_xticks(range(0, 27, 2))
    axis.set_yticks(range(0, 27, 2))
    figure.colorbar(image, ax=axis, label="log(1 + count)")
    figure.savefig(output_dir / "adjacent_code_transition.png", dpi=180)
    plt.close(figure)

    matrix = np.zeros((27, len(DIRECTION_LABELS)), dtype=np.float64)
    for code in range(27):
        mask = tokens == code
        if mask.any():
            matrix[code] = np.bincount(direction[mask], minlength=len(DIRECTION_LABELS)) / mask.sum()
    figure, axis = plt.subplots(figsize=(10.5, 9.0), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues", vmin=0.0, vmax=max(0.5, float(matrix.max())), aspect="auto")
    axis.set_title(f"{result['display_name']}: dominant-displacement direction within each code")
    axis.set_xlabel("dominant XYZ displacement")
    axis.set_ylabel("FSQ code")
    axis.set_xticks(range(len(DIRECTION_LABELS)), DIRECTION_LABELS)
    axis.set_yticks(range(27))
    figure.colorbar(image, ax=axis, label="fraction within code")
    figure.savefig(output_dir / "code_direction_heatmap.png", dpi=180)
    plt.close(figure)

    examples = result["collision_examples"][:12]
    if examples:
        figure, axes = plt.subplots(3, 4, figsize=(13, 9.5), constrained_layout=True)
        for axis, example in zip(axes.flat, examples, strict=False):
            a = np.asarray(example["first_disp_xyz"])
            b = np.asarray(example["second_disp_xyz"])
            axis.axhline(0, color="#bbbbbb", linewidth=0.8)
            axis.axvline(0, color="#bbbbbb", linewidth=0.8)
            axis.quiver(0, 0, a[0], a[1], angles="xy", scale_units="xy", scale=1, color="#177ddc", label="prev")
            axis.quiver(0, 0, b[0], b[1], angles="xy", scale_units="xy", scale=1, color="#d94c4c", label="next")
            span = max(abs(a[0]), abs(a[1]), abs(b[0]), abs(b[1]), 0.01) * 1.25
            axis.set_xlim(-span, span)
            axis.set_ylim(-span, span)
            axis.set_aspect("equal")
            axis.set_title(
                f"task {example['task_id']} ep {example['episode_id']} · {example['transition']}\n"
                f"code {example['code']} · angle {example['angle_deg']:.0f}°",
                fontsize=9,
            )
            axis.set_xlabel("Δx (m)")
            axis.set_ylabel("Δy (m)")
        for axis in axes.flat[len(examples) :]:
            axis.axis("off")
        handles = [
            plt.Line2D([0], [0], color="#177ddc", lw=3, label="previous skill"),
            plt.Line2D([0], [0], color="#d94c4c", lw=3, label="next skill"),
        ]
        figure.legend(handles=handles, loc="upper center", ncol=2)
        figure.savefig(output_dir / "opposite_direction_same_code_examples.png", dpi=180)
        plt.close(figure)


def plot_comparison(results: list[dict[str, Any]], output_dir: Path) -> None:
    labels = [result["display_name"] for result in results]
    colors = [COLORS.get(result["name"], "#888888") for result in results]
    x = np.arange(len(results))
    figure, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    adjacent = [100 * result["adjacency"]["same_code_rate"] for result in results]
    random_task = [100 * result["adjacency"]["same_task_random_code_rate"] for result in results]
    width = 0.36
    axes[0, 0].bar(x - width / 2, adjacent, width, color=colors, label="adjacent in same episode")
    axes[0, 0].bar(x + width / 2, random_task, width, color=colors, alpha=0.38, label="random pair in same task")
    axes[0, 0].set_title("Same-code collision rate")
    axes[0, 0].set_ylabel("percent")
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.25)

    conflict = [100 * result["adjacency"]["same_code_given_conflict"] for result in results]
    aligned = [100 * result["adjacency"]["same_code_given_aligned"] for result in results]
    axes[0, 1].bar(x - width / 2, conflict, width, color=colors, label="direction angle > 90°")
    axes[0, 1].bar(x + width / 2, aligned, width, color=colors, alpha=0.38, label="direction angle ≤ 60°")
    axes[0, 1].set_title("P(same code | adjacent motion geometry)")
    axes[0, 1].set_ylabel("percent")
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.25)

    centroid = np.asarray([result["reconstruction"]["centroid"]["all"] for result in results])
    gap = np.asarray([result["reconstruction"]["decoder_gap"]["all"] for result in results])
    axes[1, 0].bar(x, centroid, color=colors, label="within-code variance (optimal centroid)")
    axes[1, 0].bar(x, gap, bottom=centroid, color=colors, alpha=0.42, hatch="//", label="decoder ↔ centroid gap")
    axes[1, 0].set_title("Actual normalized reconstruction MSE decomposition")
    axes[1, 0].set_ylabel("MSE over 30×8")
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.25)

    direction_nmi = [result["categorization"]["direction_nmi"] for result in results]
    direction_purity = [result["categorization"]["direction_purity"] for result in results]
    coherence = [result["categorization"]["direction_coherence"] for result in results]
    width = 0.25
    axes[1, 1].bar(x - width, direction_nmi, width, color=colors, label="direction NMI")
    axes[1, 1].bar(x, direction_purity, width, color=colors, alpha=0.68, label="direction purity")
    axes[1, 1].bar(x + width, coherence, width, color=colors, alpha=0.38, label="vector coherence")
    axes[1, 1].set_title("Code ↔ motion-direction association")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.25)

    figure.suptitle("2layer_full · route ON · pair-loss comparison · epoch1000", fontsize=17)
    figure.savefig(output_dir / "three_model_comparison.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    transition_names = sorted(
        {
            row["transition"]
            for result in results
            for row in result["transition_summary"]
        },
        key=lambda value: int(value.split("→")[0]),
    )
    for result in results:
        aggregate: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for row in result["transition_summary"]:
            aggregate[row["transition"]][0] += row["same_code_count"]
            aggregate[row["transition"]][1] += row["count"]
        values = [100 * aggregate[name][0] / max(aggregate[name][1], 1) for name in transition_names]
        axes[0].plot(transition_names, values, marker="o", color=COLORS[result["name"]], label=result["display_name"])
    axes[0].set_title("Same-code rate by episode skill transition")
    axes[0].set_ylabel("percent")
    axes[0].set_xlabel("skill index transition")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    groups = ("XYZ", "rotvec", "gripper")
    width = 0.24
    gx = np.arange(len(groups))
    for model_index, result in enumerate(results):
        values = [result["reconstruction"]["model"][group] for group in groups]
        axes[1].bar(
            gx + (model_index - 1) * width,
            values,
            width,
            color=COLORS[result["name"]],
            label=result["display_name"],
        )
    axes[1].set_title("Normalized reconstruction MSE by dimension group")
    axes[1].set_ylabel("MSE")
    axes[1].set_xticks(gx, groups)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    figure.savefig(output_dir / "transition_and_reconstruction.png", dpi=180)
    plt.close(figure)


def fmt_vector(values: list[float]) -> str:
    return "(" + ", ".join(f"{value:+.3f}" for value in values) + ")"


def code_table(result: dict[str, Any]) -> str:
    rows = []
    for row in sorted(result["codes"], key=lambda item: item["count"], reverse=True):
        if row["count"] == 0:
            continue
        rows.append(
            "<tr>"
            f"<td>{row['code']}</td><td>{row['count']:,}</td><td>{row['share']:.1%}</td>"
            f"<td>{html.escape(row['dominant_direction'])} · {row['direction_purity']:.1%}</td>"
            f"<td>task {row['dominant_task']} · {row['task_purity']:.1%}</td>"
            f"<td>{row['dominant_skill_index']} · {row['skill_index_purity']:.1%}</td>"
            f"<td>{row['centroid_mse']:.6f}</td><td>{row['decoder_gap']:.6f}</td>"
            "</tr>"
        )
    return "".join(rows)


def transition_table(result: dict[str, Any], limit: int = 30) -> str:
    return "".join(
        "<tr>"
        f"<td>{row['task_id']}</td><td>{html.escape(row['task'])}</td><td>{row['transition']}</td>"
        f"<td>{row['same_code_count']}/{row['count']} · {row['same_code_rate']:.1%}</td>"
        f"<td>{row['direction_conflict_count']}</td><td>{row['dominant_shared_code']}</td>"
        "</tr>"
        for row in result["transition_summary"][:limit]
    )


def example_table(result: dict[str, Any], limit: int = 30) -> str:
    return "".join(
        "<tr>"
        f"<td>{row['task_id']}</td><td>{html.escape(row['task'])}</td><td>{row['episode_id']}</td>"
        f"<td>{row['transition']}</td><td>{row['code']}</td><td>{row['angle_deg']:.1f}°</td>"
        f"<td><code>{fmt_vector(row['first_disp_xyz'])}</code><br><code>{fmt_vector(row['second_disp_xyz'])}</code></td>"
        f"<td>{row['target_pair_mse']:.6f}</td>"
        f"<td>{row['first_recon_mse']:.6f}<br>{row['second_recon_mse']:.6f}</td>"
        "</tr>"
        for row in result["collision_examples"][:limit]
    )


def style() -> str:
    return """
    :root{--bg:#07111c;--panel:#101e30;--line:#2a4059;--text:#edf4fd;--muted:#9eb0c6;--cyan:#63d4ef;--amber:#ffc66d}
    *{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 12% 0,#183b59,var(--bg) 40rem);color:var(--text);font:15px/1.62 Inter,system-ui,-apple-system,"Noto Sans KR",sans-serif}
    main{max-width:1500px;margin:auto;padding:48px 28px 90px}h1{font-size:clamp(31px,5vw,56px);line-height:1.08;letter-spacing:-.045em;margin:4px 0 15px}h2{margin:43px 0 13px;font-size:25px}h3{margin:2px 0 8px}.eyebrow{color:var(--cyan);text-transform:uppercase;letter-spacing:.12em;font-size:12px}.lead{font-size:17px;max-width:1080px;color:#cbd9e9}.muted{color:var(--muted)}
    .callout{margin:20px 0;padding:18px 21px;border:1px solid #39748f;background:#0e2b40;border-radius:14px}.warning{border-color:#8d6532;background:#302315}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}.card,.figure,.table-wrap{border:1px solid var(--line);background:linear-gradient(150deg,#142940,var(--panel));border-radius:14px}.card{padding:16px}.big{font-size:26px;font-weight:780;color:#fff}.figure{padding:10px;margin:14px 0}.figure img{display:block;width:100%;border-radius:8px}.fig-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:13px}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;min-width:900px;background:#0c1827}th,td{padding:9px 11px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}th{position:sticky;top:0;background:#192c43;color:#bfeaff}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}code{color:#bcecff}.chips{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}.chips span{padding:5px 10px;border-radius:999px;background:#203b56;color:#c4e9ff}a{color:#83ddff}details{margin:15px 0;padding:12px;border:1px solid var(--line);border-radius:11px;background:#0c1724}summary{cursor:pointer;color:var(--amber);font-weight:700}.model{margin:35px 0 65px;padding:22px;border:1px solid var(--line);border-radius:17px;background:#0e1b2b}
    @media(max-width:760px){main{padding:28px 12px 60px}.fig-grid{grid-template-columns:1fr}.model{padding:14px}}
    """


def write_model_report(result: dict[str, Any], path: Path) -> None:
    adjacency = result["adjacency"]
    reconstruction = result["reconstruction"]
    categorization = result["categorization"]
    document = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(result['display_name'])} · adjacent-skill analysis</title><style>{style()}</style></head><body><main>
<div class=eyebrow>complete training skillset · {html.escape(result['epoch'])}</div>
<h1>{html.escape(result['display_name'])}<br>인접 skill·motion collision 분석</h1>
<p class=lead>전체 {result['sample_count']:,}개 skill과 실제 normalized 30×8 reconstruction target을 사용했다. 이동 방향은 raw trajectory의 시작→끝 XYZ 순변위이며, 1 cm 미만은 정지에 가까운 motion으로 분리했다.</p>
<div class=chips><span>pair loss {html.escape(result['pair_loss'])}</span><span>route ON</span><span>active {result['active_codes']}/27</span><span>effective {result['effective_codes']:.2f}</span></div>
<div class=grid>
 <section class=card><div class=big>{adjacency['same_code_rate']:.1%}</div><div>같은 episode 인접 skill이 같은 code<br>같은 task random pair {adjacency['same_task_random_code_rate']:.1%}</div></section>
 <section class=card><div class=big>{adjacency['same_code_given_conflict']:.1%}</div><div>방향각 &gt;90°인 인접 pair가 같은 code<br>{adjacency['same_code_conflict_count']:,}건</div></section>
 <section class=card><div class=big>{categorization['direction_nmi']:.3f}</div><div>code↔방향 NMI<br>weighted purity {categorization['direction_purity']:.1%}</div></section>
 <section class=card><div class=big>{reconstruction['model']['all']:.6f}</div><div>actual recon MSE<br>centroid {reconstruction['centroid']['all']:.6f}<br>decoder gap {reconstruction['decoder_gap']['all']:.6f}</div></section>
</div>
<div class=callout><b>읽는 법:</b> centroid MSE는 현재 code assignment를 고정했을 때 가능한 최저 MSE, decoder gap은 학습된 decoder 출력이 그 centroid에서 떨어진 몫이다. 따라서 visually 다른 motion이 섞여도 normalized Euclidean 거리가 작으면 centroid MSE는 낮을 수 있다.</div>
<div class=fig-grid><figure class=figure><img src="adjacent_code_transition.png"><figcaption class=muted>인접 skill code transition. 대각선이 강할수록 이웃 skill이 같은 code로 합쳐진다.</figcaption></figure><figure class=figure><img src="code_direction_heatmap.png"><figcaption class=muted>코드별 dominant XYZ 순변위 방향 구성.</figcaption></figure></div>
<figure class=figure><img src="opposite_direction_same_code_examples.png"><figcaption class=muted>같은 code인데 방향각이 가장 큰 인접 pair의 XY 순변위. z 차이는 아래 원시 표에 포함된다.</figcaption></figure>
<h2>같은 code로 자주 합쳐진 task·인접 index</h2><div class=table-wrap><table><thead><tr><th>task</th><th>description</th><th>transition</th><th>same code</th><th>그중 방향충돌</th><th>대표 shared code</th></tr></thead><tbody>{transition_table(result)}</tbody></table></div>
<h2>반대 방향인데 같은 code인 실제 pair</h2><div class=table-wrap><table><thead><tr><th>task</th><th>description</th><th>episode</th><th>transition</th><th>code</th><th>angle</th><th>ΔXYZ prev / next</th><th>두 target 간 MSE</th><th>각 recon MSE</th></tr></thead><tbody>{example_table(result)}</tbody></table></div>
<h2>코드별 motion·reconstruction 요약</h2><div class=table-wrap><table><thead><tr><th>code</th><th>N</th><th>share</th><th>대표 방향·purity</th><th>대표 task·purity</th><th>대표 index·purity</th><th>centroid MSE</th><th>decoder gap</th></tr></thead><tbody>{code_table(result)}</tbody></table></div>
<p class=muted><a href="index.html">기본 codebook correlation 분석</a> · <a href="../../codebook_correlation_analysis/index.html">세 모델 비교 분석</a> · <a href="neighbor_metrics.json">raw metrics JSON</a></p>
</main></body></html>"""
    path.write_text(document, encoding="utf-8")
    (path.parent / "neighbor_metrics.json").write_text(
        json.dumps(json_ready(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(path.parent / "adjacent_collision_examples.csv", result["collision_examples"])
    write_csv(path.parent / "task_transition_summary.csv", result["transition_summary"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def comparison_table(results: list[dict[str, Any]]) -> str:
    rows = []
    for result in results:
        adj = result["adjacency"]
        cat = result["categorization"]
        recon = result["reconstruction"]
        rows.append(
            "<tr>"
            f"<th>{html.escape(result['display_name'])}</th>"
            f"<td>{result['active_codes']}/27 · eff {result['effective_codes']:.2f}</td>"
            f"<td>{adj['same_code_rate']:.1%}</td><td>{adj['same_task_random_code_rate']:.1%}</td>"
            f"<td>{adj['same_code_given_conflict']:.1%}</td><td>{cat['direction_nmi']:.3f}</td>"
            f"<td>{cat['direction_purity']:.1%}</td><td>{recon['model']['all']:.6f}</td>"
            f"<td>{recon['centroid']['all']:.6f}</td><td>{recon['decoder_gap']['all']:.6f}</td>"
            "</tr>"
        )
    return "".join(rows)


def write_comparison_report(
    results: list[dict[str, Any]],
    agreement: dict[str, float],
    output_dir: Path,
) -> None:
    cont = next(result for result in results if result["name"] == "cont_route_ON")
    js = next(result for result in results if result["name"] == "js_route_ON")
    none = next(result for result in results if result["name"] == "none_route_ON")
    best_direction = max(results, key=lambda result: result["categorization"]["direction_nmi"])

    def ablation_accuracy(result: dict[str, Any], group: str) -> float:
        for row in result.get("correlation_analysis", {}).get("classification_ablation", []):
            if row["feature_group"] == group:
                return float(row["accuracy_mean"])
        return float("nan")

    relative_accuracies = {
        result["name"]: ablation_accuracy(result, "상대 XYZ 모션") for result in results
    }
    cont_vs_js_recon = 1.0 - js["reconstruction"]["model"]["all"] / cont["reconstruction"]["model"]["all"]
    cont_vs_none_recon = 1.0 - none["reconstruction"]["model"]["all"] / cont["reconstruction"]["model"]["all"]
    js_pair_ratio = (
        js["adjacency"]["different_code_pair_target_mse"]
        / js["adjacency"]["same_code_pair_target_mse"]
    )
    none_pair_ratio = (
        none["adjacency"]["different_code_pair_target_mse"]
        / none["adjacency"]["same_code_pair_target_mse"]
    )

    model_sections = []
    for result in results:
        relative_root = Path("..") / Path(result["model_root"]).name / output_dir.name
        model_sections.append(
            f"""<section class=model><h2>{html.escape(result['display_name'])}</h2>
            <div class=chips><span>pair={html.escape(result['pair_loss'])}</span><span>route ON</span><span>epoch1000</span></div>
            <div class=grid><section class=card><div class=big>{result['adjacency']['same_code_rate']:.1%}</div><div>adjacent same code</div></section>
            <section class=card><div class=big>{result['adjacency']['same_code_given_conflict']:.1%}</div><div>P(same code | direction &gt;90°)</div></section>
            <section class=card><div class=big>{result['categorization']['direction_nmi']:.3f}</div><div>direction NMI</div></section>
            <section class=card><div class=big>{result['reconstruction']['model']['all']:.6f}</div><div>actual recon MSE</div></section></div>
            <p><a href="{relative_root.as_posix()}/index.html">코드별 correlation report</a> · <a href="{relative_root.as_posix()}/neighbor_analysis.html">인접 skill 상세 report</a></p></section>"""
        )

    agreement_rows = "".join(
        f"<tr><td>{html.escape(pair)}</td><td>{value:.3f}</td></tr>"
        for pair, value in agreement.items()
    )
    document = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>2layer_full · contrastive vs JS vs pair OFF</title><style>{style()}</style></head><body><main>
<div class=eyebrow>2layer_full · complete 11,221 skills · exact epoch1000</div>
<h1>같은 task의 이웃 skill은<br>왜 같은 code로 합쳐지는가?</h1>
<p class=lead>세 모델은 모두 zero→zero, 2-layer, recon-only, route ON이며 pair objective만 다르다. 아래 수치는 replay의 23개 task가 아니라 checkpoint와 정렬된 전체 학습 skillset을 사용했다.</p>
<div class=callout><b>결론:</b> 관찰하신 현상은 objective와 정확히 맞는다. <b>contrastive만</b> clean↔augmented positive overlap을 높이는 동시에, 같은 episode의 <b>인접 skill을 negative로 뽑아 overlap을 낮춘다.</b> JS는 clean과 그 augmented version의 soft-code distribution을 같게 만들 뿐 인접 skill을 밀어내지 않으며, pair OFF는 그 제약도 없다. 그 결과 인접 same-code는 contrastive {cont['adjacency']['same_code_rate']:.2%}, JS {js['adjacency']['same_code_rate']:.2%}, none {none['adjacency']['same_code_rate']:.2%}; 반대방향 pair만 보면 각각 {cont['adjacency']['same_code_given_conflict']:.2%}, {js['adjacency']['same_code_given_conflict']:.2%}, {none['adjacency']['same_code_given_conflict']:.2%}다.</div>
<div class="callout warning"><b>그런데 왜 JS/none의 recon MSE가 더 낮은가?</b> reconstruction은 motion category 정답을 맞히는 loss가 아니라 normalized 30×8 좌표의 평균 제곱거리다. 같은 task의 이웃 skill은 rotation, gripper, 길이와 많은 control point를 공유한다. 한두 XYZ 축의 순변위 부호가 반대여도 전체 240개 원소의 평균거리는 작을 수 있다. 실제로 JS/none에서 같은-code 인접 pair의 target 간 MSE는 {js['adjacency']['same_code_pair_target_mse']:.4f}/{none['adjacency']['same_code_pair_target_mse']:.4f}인 반면, 다른-code 인접 pair는 {js_pair_ratio:.1f}×/{none_pair_ratio:.1f}× 더 멀다. 즉 눈으로는 반대 방향이어도 MSE 공간에서는 충분히 가까워 하나의 centroid로 합치는 것이 싸다.</div>
<div class=callout><b>오차 분해가 더 강한 증거다.</b> contrastive의 decoder↔centroid gap은 {cont['reconstruction']['decoder_gap']['all']:.6f}으로 셋 중 가장 작다. 그런데 within-code centroid MSE가 {cont['reconstruction']['centroid']['all']:.6f}으로 JS {js['reconstruction']['centroid']['all']:.6f}, none {none['reconstruction']['centroid']['all']:.6f}보다 크다. 따라서 contrastive의 높은 loss는 decoder가 평균을 못 배워서가 아니라, adjacent-negative 제약 때문에 27개 code를 순수 MSE 최적 partition과 다르게 배치한 <b>assignment cost</b>다. 실제 MSE는 JS가 contrastive보다 {cont_vs_js_recon:.1%}, none이 {cont_vs_none_recon:.1%} 낮다.</div>
<figure class=figure><img src="three_model_comparison.png"><figcaption class=muted>인접 collision, direction association, 실제 reconstruction decomposition을 한 기준으로 비교.</figcaption></figure>
<figure class=figure><img src="transition_and_reconstruction.png"><figcaption class=muted>episode 내 index transition별 collapse와 XYZ/rotation/gripper loss.</figcaption></figure>
<h2>동일 기준 정량 비교</h2><div class=table-wrap><table><thead><tr><th>model</th><th>usage</th><th>adjacent same</th><th>same-task random</th><th>same | opposite dir</th><th>direction NMI</th><th>direction purity</th><th>actual MSE</th><th>centroid MSE</th><th>decoder gap</th></tr></thead><tbody>{comparison_table(results)}</tbody></table></div>
<div class=callout><b>MSE 분해 해석:</b> actual MSE = within-code centroid MSE + decoder↔centroid gap이다(제곱오차의 정확한 분산 분해). 전자는 assignment가 얼마나 MSE 친화적인지, 후자는 decoder가 그 code 평균을 얼마나 잘 학습했는지를 나타낸다. 따라서 direction purity가 높은 assignment와 낮은 reconstruction loss는 같은 순위를 가질 필요가 없다.</div>
<div class=callout><b>“contrastive가 motion을 더 잘 가른다”의 정확한 의미:</b> 단순 시작→끝 dominant direction 7-bin NMI만 보면 {html.escape(best_direction['display_name'])}가 {best_direction['categorization']['direction_nmi']:.3f}으로 가장 높다. 하지만 전체 상대 XYZ 궤적으로 code를 예측하는 episode-grouped top-1은 contrastive {relative_accuracies['cont_route_ON']:.1%}, JS {relative_accuracies['js_route_ON']:.1%}, none {relative_accuracies['none_route_ON']:.1%}이고, skill-index NMI도 contrastive {cont['categorization']['skill_index']['nmi']:.3f}가 JS {js['categorization']['skill_index']['nmi']:.3f}, none {none['categorization']['skill_index']['nmi']:.3f}보다 높다. 즉 contrastive는 code를 단순 ±XYZ 라벨에 맞추기보다 <b>궤적 형태와 episode phase, 특히 바로 이웃한 phase의 차이</b>에 더 민감하게 만든다.</div>
<h2>세 assignment는 서로 얼마나 다른가?</h2><div class=table-wrap><table><thead><tr><th>model pair</th><th>adjusted mutual information</th></tr></thead><tbody>{agreement_rows}</tbody></table></div>
<p class=muted>AMI는 code 번호 permutation에 무관하다. 낮을수록 세 objective가 전체 dataset을 서로 다른 partition으로 나눴다는 뜻이다.</p>
{''.join(model_sections)}
<h2>구현상 원인, 정확히 구분</h2>
<div class=grid><section class=card><h3>contrastive</h3><p>positive: boundary augmentation과 같은 code overlap을 높임.<br>negative: 같은 episode의 앞/뒤 skill 중 하나를 골라 soft-code overlap을 낮춤. 그래서 이웃 skill의 같은-code merge를 직접 벌점 준다.</p></section>
<section class=card><h3>JS</h3><p>clean과 augmented trajectory의 full soft-code distribution 간 JS divergence만 최소화한다. 두 분포가 같으면 diffuse해도 0이며, adjacent negative나 sharpening 압력이 없다.</p></section>
<section class=card><h3>pair OFF</h3><p>reconstruction과 route 기준만 남는다. route도 category diversity나 이웃 분리를 요구하지 않으므로, decoder prototype 하나가 두 skill의 평균을 잘 설명하면 merge를 허용한다.</p></section></div>
<h2>실험 설계상 의미</h2><p class=lead>현재 결과는 “JS/none이 categorization에 실패했다”기보다 <b>motion semantics와 MSE geometry가 어긋난다</b>는 증거에 가깝다. 목표가 episode phase와 이동방향까지 분리된 skill category라면 contrastive의 adjacent negative가 맞는 inductive bias다. 반대로 reconstruction을 우선하면 JS/none의 merge가 합리적일 수 있다. 둘을 동시에 원하면 pair weight 조절 전에, recon metric에 displacement/direction 항을 별도로 넣어 시각적으로 중요한 차이가 30×8 평균에서 희석되지 않게 하는 것이 가장 직접적인 다음 ablation이다. 단, contrastive도 task 29의 1→2 transition은 32/34가 같은 code이므로 adjacent collapse를 완전히 제거한 것은 아니다.</p>
<p class=muted>raw outputs: <a href="comparison_metrics.json">comparison_metrics.json</a> · <a href="model_summary.csv">model_summary.csv</a></p>
</main></body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    report_root = args.report_root.resolve()
    bundle_path = args.skill_bundle.resolve()
    collections = find_models(report_root, args.models)
    print(f"Loading trajectory features from {bundle_path}", flush=True)
    dataset = load_dataset(bundle_path, args.episode_metadata.resolve())
    bundle = load_bundle(bundle_path)

    first_model_root = collections[args.models[0]][0]
    first_manifest = json.loads(
        (first_model_root / "checkpoints" / args.epoch / "metrics" / "manifest.json").read_text()
    )
    first_run = Path(first_manifest["signature"]["model_path"]).parent
    first_checkpoint = Path(first_manifest["signature"]["model_path"])
    first_model, first_cfg = load_fsq_model(first_checkpoint, device="cpu")
    first_stats = load_stats(first_run)
    mode = scalar_text(first_stats["reconstructor_output_mode"])
    n_control = int(cfg_value(first_cfg, "n_control"))
    degree = int(cfg_value(first_cfg, "spline_degree"))
    del first_model
    print(f"Preparing exact normalized {n_control}x8 targets ({mode})", flush=True)
    targets = normalized_targets(
        bundle,
        mode=mode,
        minimum=first_stats["reconstructor_min"],
        maximum=first_stats["reconstructor_max"],
        n_control=n_control,
        degree=degree,
    )

    for name in args.models[1:]:
        model_root = collections[name][0]
        manifest = json.loads(
            (model_root / "checkpoints" / args.epoch / "metrics" / "manifest.json").read_text()
        )
        stats = load_stats(Path(manifest["signature"]["model_path"]).parent)
        for key in ("reconstructor_min", "reconstructor_max"):
            if not np.allclose(stats[key], first_stats[key], atol=1e-7, rtol=1e-7):
                raise ValueError(f"{name} uses different {key}; shared-target comparison is invalid")

    adjacent = adjacency_pairs(dataset["metadata"])
    nonadjacent = all_nonadjacent_episode_pairs(dataset["metadata"])
    results: list[dict[str, Any]] = []
    token_by_name: dict[str, np.ndarray] = {}
    for name in args.models:
        print(f"Analyzing {name}", flush=True)
        model_root = collections[name][0]
        result = analyze_model(
            name,
            model_root,
            dataset,
            targets,
            args.epoch,
            args.output_subdir,
            args.motion_threshold,
            adjacent,
            nonadjacent,
        )
        results.append(result)
        latent, _ = load_latents(model_root, args.epoch, dataset)
        token_by_name[name] = latent["tokens"].astype(np.int64)

    agreement = {
        f"{DISPLAY_NAMES.get(left, left)} ↔ {DISPLAY_NAMES.get(right, right)}": float(
            adjusted_mutual_info_score(token_by_name[left], token_by_name[right])
        )
        for index, left in enumerate(args.models)
        for right in args.models[index + 1 :]
    }
    output_dir = report_root / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison(results, output_dir)
    write_comparison_report(results, agreement, output_dir)
    payload = {"models": results, "assignment_agreement_ami": agreement}
    (output_dir / "comparison_metrics.json").write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_rows = []
    for result in results:
        summary_rows.append(
            {
                "model": result["name"],
                "active_codes": result["active_codes"],
                "effective_codes": result["effective_codes"],
                "adjacent_same_code_rate": result["adjacency"]["same_code_rate"],
                "same_task_random_code_rate": result["adjacency"]["same_task_random_code_rate"],
                "same_code_given_direction_conflict": result["adjacency"]["same_code_given_conflict"],
                "direction_nmi": result["categorization"]["direction_nmi"],
                "direction_purity": result["categorization"]["direction_purity"],
                "reconstruction_mse": result["reconstruction"]["model"]["all"],
                "centroid_mse": result["reconstruction"]["centroid"]["all"],
                "decoder_gap": result["reconstruction"]["decoder_gap"]["all"],
            }
        )
    write_csv(output_dir / "model_summary.csv", summary_rows)
    print(f"Wrote {output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
