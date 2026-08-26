#!/usr/bin/env python3
"""Analyze the proprioceptive categorization criteria of every FSQ code.

This joins the complete training latent artifact to the complete skill bundle,
then writes a Korean HTML report below each selected model report directory.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    normalized_mutual_info_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier

from analyze_code_motion_correlation import trajectory_features


DEFAULT_MODELS = (
    "recon w0.1 contrastive",
    "term+recon w0.1 contrastive dino",
)

FEATURE_LABEL_KO = {
    "start_x": "시작 EE x",
    "start_y": "시작 EE y",
    "start_z": "시작 EE z",
    "mean_x": "평균 EE x",
    "mean_y": "평균 EE y",
    "mean_z": "평균 EE z",
    "end_x": "종료 EE x",
    "end_y": "종료 EE y",
    "end_z": "종료 EE z",
    "disp_x": "x 순변위",
    "disp_y": "y 순변위",
    "disp_z": "z 순변위",
    "rel25_x": "25% 시점 상대 x",
    "rel25_y": "25% 시점 상대 y",
    "rel25_z": "25% 시점 상대 z",
    "rel50_x": "50% 시점 상대 x",
    "rel50_y": "50% 시점 상대 y",
    "rel50_z": "50% 시점 상대 z",
    "rel75_x": "75% 시점 상대 x",
    "rel75_y": "75% 시점 상대 y",
    "rel75_z": "75% 시점 상대 z",
    "net_xyz": "시작-끝 직선거리",
    "path_xyz": "위치 경로 길이",
    "straightness": "경로 직선성",
    "rot_net_angle": "시작-끝 실제 회전각",
    "rot_path_angle": "누적 회전 경로",
    "frames": "skill 프레임 길이",
    "skill_index": "episode 내 skill index",
    "skill_order": "episode 내 정규화 순서",
    "grip_start": "시작 gripper 값",
    "grip_end": "종료 gripper 값",
    "grip_delta": "gripper 변화량",
    "grip_range": "gripper 값 범위",
    "grip_path": "누적 gripper 변화",
    "rv_start_x": "시작 raw rotation x",
    "rv_start_y": "시작 raw rotation y",
    "rv_start_z": "시작 raw rotation z",
    "rv_mean_x": "평균 raw rotation x",
    "rv_mean_y": "평균 raw rotation y",
    "rv_mean_z": "평균 raw rotation z",
    "rv_end_x": "종료 raw rotation x",
    "rv_end_y": "종료 raw rotation y",
    "rv_end_z": "종료 raw rotation z",
    "rot_rel_x": "상대 회전벡터 x",
    "rot_rel_y": "상대 회전벡터 y",
    "rot_rel_z": "상대 회전벡터 z",
}

INTERPRET_FEATURES = (
    "start_x",
    "start_y",
    "start_z",
    "mean_x",
    "mean_y",
    "mean_z",
    "disp_x",
    "disp_y",
    "disp_z",
    "rel50_x",
    "rel50_y",
    "rel50_z",
    "net_xyz",
    "path_xyz",
    "straightness",
    "rot_net_angle",
    "rot_path_angle",
    "frames",
    "skill_index",
    "skill_order",
    "grip_delta",
    "grip_range",
    "grip_path",
)

HEATMAP_FEATURES = (
    "start_x",
    "start_y",
    "start_z",
    "mean_x",
    "mean_y",
    "mean_z",
    "disp_x",
    "disp_y",
    "disp_z",
    "path_xyz",
    "straightness",
    "rot_net_angle",
    "rot_path_angle",
    "frames",
    "skill_index",
    "grip_range",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--skill-bundle", type=Path, required=True)
    parser.add_argument("--episode-metadata", type=Path, required=True)
    parser.add_argument("--epoch", default="epoch2000")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--output-subdir", default="code_categorization_analysis")
    parser.add_argument("--cv-folds", type=int, default=3)
    return parser.parse_args()


def load_collections(report_root: Path, model_names: set[str]) -> dict[str, tuple[Path, dict]]:
    collections: dict[str, tuple[Path, dict]] = {}
    for path in report_root.glob("*/metrics/collection.json"):
        document = json.loads(path.read_text())
        if document.get("model_name") in model_names:
            collections[document["model_name"]] = (path.parent.parent, document)
    missing = model_names - collections.keys()
    if missing:
        raise FileNotFoundError(f"Missing model report folders for: {sorted(missing)}")
    return collections


def load_episode_metadata(path: Path) -> dict[int, str]:
    with np.load(path, allow_pickle=True) as data:
        return {
            int(episode_id): str(scene_file)
            for episode_id, scene_file in zip(
                data["episode_index"], data["scene_file"], strict=True
            )
        }


def task_description(scene_file: str) -> str:
    stem = re.sub(r"_demo\.hdf5$", "", scene_file)
    return re.sub(r"^[A-Z_]+_SCENE\d+_", "", stem).replace("_", " ")


def load_dataset(bundle_path: Path, episode_metadata_path: Path) -> dict[str, Any]:
    episode_to_scene = load_episode_metadata(episode_metadata_path)
    with np.load(bundle_path, allow_pickle=False) as bundle:
        lengths = bundle["states_len"].astype(np.int64)
        starts = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths[:-1])))
        states = bundle["states_cat"]
        metadata = {
            "episode_id": bundle["meta_episode_id"].astype(np.int64),
            "task_id": bundle["meta_task_id"].astype(np.int64),
            "skill_index": bundle["meta_skill_index"].astype(np.int64),
            "frame_start": bundle["meta_frame_start"].astype(np.int64),
            "frame_end": bundle["meta_frame_end"].astype(np.int64),
            "length": bundle["meta_length"].astype(np.int64),
        }
        feature_rows = [
            trajectory_features(states[int(start) : int(start + length)])
            for start, length in zip(starts, lengths, strict=True)
        ]

    feature_names = list(feature_rows[0])
    features = {
        name: np.asarray([row[name] for row in feature_rows], dtype=np.float64)
        for name in feature_names
    }
    features["skill_index"] = metadata["skill_index"].astype(np.float64)
    scenes = np.asarray([
        episode_to_scene[int(episode_id)] for episode_id in metadata["episode_id"]
    ])
    scene_families = np.asarray([scene.split("_SCENE", maxsplit=1)[0] for scene in scenes])
    descriptions = np.asarray([task_description(scene) for scene in scenes])
    return {
        "features": features,
        "metadata": metadata,
        "scene_file": scenes,
        "scene_family": scene_families,
        "task_description": descriptions,
    }


def load_latents(model_root: Path, epoch: str, dataset: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict]:
    manifest_path = model_root / "checkpoints" / epoch / "metrics" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    latent_path = Path(manifest["signature"]["latents_path"])
    with np.load(latent_path, allow_pickle=False) as data:
        latent = {key: data[key].copy() for key in data.files}

    metadata = dataset["metadata"]
    checks = {
        "episode_id": "episode_id",
        "task_id": "task_id",
        "skill_index": "skill_index",
        "frame_start": "frame_start",
        "frame_end": "frame_end",
        "length": "length",
    }
    for latent_key, metadata_key in checks.items():
        if not np.array_equal(latent[latent_key], metadata[metadata_key]):
            raise ValueError(f"Latent/bundle alignment mismatch for {latent_key}")
    expected_counts = np.asarray(manifest["train_codebook_counts"], dtype=np.int64)
    actual_counts = np.bincount(latent["tokens"], minlength=len(expected_counts))
    if not np.array_equal(expected_counts, actual_counts):
        raise ValueError("Latent token histogram does not match replay manifest")
    return latent, manifest


def numeric_matrix(dataset: dict[str, Any], latent: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    features = dict(dataset["features"])
    features["skill_order"] = latent["skill_order"].astype(np.float64)
    names = [name for name, values in features.items() if np.std(values) > 1e-10]
    matrix = np.column_stack([features[name] for name in names])
    return names, matrix


def feature_groups(names: list[str]) -> dict[str, list[int]]:
    absolute_xyz = [
        index
        for index, name in enumerate(names)
        if name.startswith(("start_", "mean_", "end_", "p25_", "p50_", "p75_"))
        and not name.startswith(("rv_", "grip_"))
    ]
    start_xyz = [names.index(name) for name in ("start_x", "start_y", "start_z")]
    relative_xyz = [
        index
        for index, name in enumerate(names)
        if name.startswith(("disp_", "rel25_", "rel50_", "rel75_", "range_", "std_"))
        and not name.startswith(("rv_", "grip_"))
        or name in {"net_xyz", "path_xyz", "straightness"}
    ]
    rotation = [
        index
        for index, name in enumerate(names)
        if name.startswith(("rv_", "rot_"))
    ]
    time_gripper = [
        index
        for index, name in enumerate(names)
        if name in {"frames", "skill_index", "skill_order"} or name.startswith("grip_")
    ]
    return {
        "시작 XYZ만": start_xyz,
        "절대 XYZ 궤적": absolute_xyz,
        "상대 XYZ 모션": relative_xyz,
        "회전 특징": rotation,
        "시간·순서·gripper": time_gripper,
        "전체 proprio 특징": list(range(len(names))),
    }


def evaluate_feature_groups(
    matrix: np.ndarray,
    tokens: np.ndarray,
    episode_ids: np.ndarray,
    groups: dict[str, list[int]],
    folds: int,
) -> list[dict]:
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    splits = list(splitter.split(matrix, tokens, groups=episode_ids))
    results = []
    all_codes = np.unique(tokens)
    for group_name, columns in groups.items():
        fold_metrics = []
        for fold, (train_index, test_index) in enumerate(splits):
            classifier = ExtraTreesClassifier(
                n_estimators=120,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                random_state=100 + fold,
                n_jobs=-1,
            )
            classifier.fit(matrix[train_index][:, columns], tokens[train_index])
            prediction = classifier.predict(matrix[test_index][:, columns])
            probabilities = classifier.predict_proba(matrix[test_index][:, columns])
            top_count = min(3, probabilities.shape[1])
            top_indices = np.argpartition(probabilities, -top_count, axis=1)[:, -top_count:]
            top_classes = classifier.classes_[top_indices]
            top3 = np.mean(np.any(top_classes == tokens[test_index, None], axis=1))
            fold_metrics.append(
                {
                    "accuracy": accuracy_score(tokens[test_index], prediction),
                    "balanced_accuracy": balanced_accuracy_score(tokens[test_index], prediction),
                    "macro_f1": f1_score(
                        tokens[test_index], prediction, labels=all_codes, average="macro", zero_division=0
                    ),
                    "top3_accuracy": top3,
                }
            )
        result = {"feature_group": group_name, "feature_count": len(columns)}
        for metric in fold_metrics[0]:
            values = np.asarray([item[metric] for item in fold_metrics])
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = float(values.std())
        results.append(result)
    return results


def fit_global_importance(
    names: list[str], matrix: np.ndarray, tokens: np.ndarray
) -> list[dict]:
    classifier = ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=7,
        n_jobs=-1,
    )
    classifier.fit(matrix, tokens)
    return sorted(
        (
            {"feature": name, "importance": float(importance)}
            for name, importance in zip(names, classifier.feature_importances_, strict=True)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )


def eta_squared(values: np.ndarray, categories: np.ndarray) -> float:
    total = np.sum((values - values.mean()) ** 2)
    if total <= 0:
        return 0.0
    between = sum(
        len(group) * (group.mean() - values.mean()) ** 2
        for level in np.unique(categories)
        if len(group := values[categories == level])
    )
    return float(between / total)


def analyze_axes(
    names: list[str], matrix: np.ndarray, latents: np.ndarray, dataset: dict[str, Any]
) -> list[dict]:
    rows = []
    for axis in range(latents.shape[1]):
        coordinate = latents[:, axis]
        for column, feature in enumerate(names):
            correlation = spearmanr(coordinate, matrix[:, column]).statistic
            level_means = {
                str(int(level)): float(matrix[coordinate == level, column].mean())
                for level in np.unique(coordinate)
            }
            rows.append(
                {
                    "axis": axis,
                    "feature": feature,
                    "eta_squared": eta_squared(matrix[:, column], coordinate),
                    "spearman_r": float(0.0 if np.isnan(correlation) else correlation),
                    "level_means": level_means,
                }
            )
        for category_name, values in (
            ("scene_family", dataset["scene_family"]),
            ("scene_file", dataset["scene_file"]),
            ("task_id", dataset["metadata"]["task_id"]),
            ("skill_index", dataset["metadata"]["skill_index"]),
        ):
            rows.append(
                {
                    "axis": axis,
                    "feature": f"categorical:{category_name}",
                    "normalized_mutual_info": float(
                        normalized_mutual_info_score(values, coordinate)
                    ),
                }
            )
    return rows


def code_metadata_associations(tokens: np.ndarray, dataset: dict[str, Any]) -> dict[str, float]:
    return {
        "scene_family_nmi": float(normalized_mutual_info_score(dataset["scene_family"], tokens)),
        "scene_file_nmi": float(normalized_mutual_info_score(dataset["scene_file"], tokens)),
        "task_id_nmi": float(normalized_mutual_info_score(dataset["metadata"]["task_id"], tokens)),
        "skill_index_nmi": float(
            normalized_mutual_info_score(dataset["metadata"]["skill_index"], tokens)
        ),
    }


def strongest_features_for_code(
    code_mask: np.ndarray,
    names: list[str],
    matrix: np.ndarray,
) -> list[dict]:
    available = [feature for feature in INTERPRET_FEATURES if feature in names]
    rows = []
    for feature in available:
        column = names.index(feature)
        values = matrix[:, column]
        inside = values[code_mask]
        outside = values[~code_mask]
        auc = roc_auc_score(code_mask.astype(np.int8), values)
        rows.append(
            {
                "feature": feature,
                "auc_separation": float(max(auc, 1.0 - auc)),
                "code_mean": float(inside.mean()),
                "rest_mean": float(outside.mean()),
                "global_z": float((inside.mean() - values.mean()) / values.std()),
            }
        )
    return sorted(rows, key=lambda item: item["auc_separation"], reverse=True)[:5]


def best_tree_rule(
    target_mask: np.ndarray,
    names: list[str],
    matrix: np.ndarray,
) -> dict | None:
    target_count = int(target_mask.sum())
    if target_count < 10:
        return None
    min_leaf = max(5, min(50, target_count // 5))
    classifier = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=min_leaf,
        class_weight="balanced",
        random_state=13,
    )
    classifier.fit(matrix, target_mask.astype(np.int8))
    leaves = classifier.apply(matrix)
    candidates = []
    for leaf in np.unique(leaves):
        leaf_mask = leaves == leaf
        true_positive = int(np.sum(leaf_mask & target_mask))
        if true_positive == 0:
            continue
        precision = true_positive / int(leaf_mask.sum())
        recall = true_positive / target_count
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        candidates.append((f1, precision, recall, int(leaf), np.flatnonzero(leaf_mask)[0]))
    if not candidates:
        return None
    f1, precision, recall, leaf, sample_index = max(candidates)
    path_nodes = classifier.decision_path(matrix[[sample_index]]).indices
    rules = []
    for node in path_nodes:
        feature_index = classifier.tree_.feature[node]
        if feature_index < 0:
            continue
        threshold = float(classifier.tree_.threshold[node])
        operator = "<=" if matrix[sample_index, feature_index] <= threshold else ">"
        rules.append(
            {"feature": names[feature_index], "operator": operator, "threshold": threshold}
        )
    return {
        "rules": rules,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "leaf_support": int(np.sum(leaves == leaf)),
    }


def reliability_label(count: int) -> str:
    if count >= 100:
        return "높음"
    if count >= 30:
        return "보통"
    if count >= 10:
        return "낮음"
    return "표본 부족"


def dominant(values: np.ndarray, mask: np.ndarray) -> dict:
    counts = Counter(values[mask].tolist())
    value, count = counts.most_common(1)[0]
    return {
        "value": int(value) if isinstance(value, (np.integer,)) else str(value),
        "count": int(count),
        "purity": float(count / mask.sum()),
        "top3": [
            {
                "value": int(item) if isinstance(item, (np.integer,)) else str(item),
                "count": int(item_count),
                "fraction": float(item_count / mask.sum()),
            }
            for item, item_count in counts.most_common(3)
        ],
    }


def analyze_codes(
    tokens: np.ndarray,
    coordinates: np.ndarray,
    names: list[str],
    matrix: np.ndarray,
    dataset: dict[str, Any],
) -> list[dict]:
    code_rows = []
    for code in sorted(np.unique(tokens)):
        mask = tokens == code
        count = int(mask.sum())
        code_rows.append(
            {
                "code": int(code),
                "coordinate": [int(value) for value in coordinates[mask][0]],
                "count": count,
                "fraction": float(count / len(tokens)),
                "reliability": reliability_label(count),
                "dominant_scene_family": dominant(dataset["scene_family"], mask),
                "dominant_scene": dominant(dataset["scene_file"], mask),
                "dominant_task": dominant(dataset["task_description"], mask),
                "dominant_task_id": dominant(dataset["metadata"]["task_id"], mask),
                "dominant_skill_index": dominant(dataset["metadata"]["skill_index"], mask),
                "strongest_features": strongest_features_for_code(mask, names, matrix),
                "tree_rule": best_tree_rule(mask, names, matrix),
            }
        )
    return code_rows


def draw_code_counts(code_rows: list[dict], output_path: Path) -> None:
    codes = [row["code"] for row in code_rows]
    counts = [row["count"] for row in code_rows]
    figure, axis = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    bars = axis.bar(codes, counts, color="#4f8ef7")
    axis.set_xticks(codes)
    axis.set_xlabel("FSQ code")
    axis.set_ylabel("full-training skill count")
    axis.set_title("Complete codebook usage")
    for bar, count in zip(bars, counts, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, count, str(count), ha="center", va="bottom", fontsize=7, rotation=90)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def draw_code_feature_heatmap(
    code_rows: list[dict],
    tokens: np.ndarray,
    names: list[str],
    matrix: np.ndarray,
    output_path: Path,
) -> None:
    features = [feature for feature in HEATMAP_FEATURES if feature in names]
    global_mean = matrix.mean(axis=0)
    global_std = matrix.std(axis=0)
    heatmap = []
    labels = []
    for row in code_rows:
        mask = tokens == row["code"]
        heatmap.append([
            (matrix[mask, names.index(feature)].mean() - global_mean[names.index(feature)])
            / global_std[names.index(feature)]
            for feature in features
        ])
        labels.append(f"{row['code']:02d} {tuple(row['coordinate'])} n={row['count']}")
    heatmap_array = np.asarray(heatmap)
    figure, axis = plt.subplots(figsize=(14, 10.5), constrained_layout=True)
    image = axis.imshow(np.clip(heatmap_array, -2.5, 2.5), cmap="coolwarm", vmin=-2.5, vmax=2.5, aspect="auto")
    axis.set_xticks(range(len(features)), features, rotation=55, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title("Per-code feature mean (global z-score, clipped to ±2.5)")
    figure.colorbar(image, ax=axis, label="z-score")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def draw_axis_heatmap(axis_rows: list[dict], output_path: Path) -> None:
    numeric_rows = [row for row in axis_rows if "eta_squared" in row]
    features = sorted(
        {row["feature"] for row in numeric_rows},
        key=lambda feature: max(
            row["eta_squared"] for row in numeric_rows if row["feature"] == feature
        ),
        reverse=True,
    )[:24]
    matrix = np.asarray([
        [
            next(
                row["eta_squared"]
                for row in numeric_rows
                if row["feature"] == feature and row["axis"] == axis
            )
            for axis in range(3)
        ]
        for feature in features
    ])
    figure, axis = plt.subplots(figsize=(7.5, 9.5), constrained_layout=True)
    image = axis.imshow(matrix, cmap="magma", vmin=0.0, vmax=max(0.35, float(matrix.max())), aspect="auto")
    axis.set_xticks(range(3), ["FSQ axis 0", "FSQ axis 1", "FSQ axis 2"])
    axis.set_yticks(range(len(features)), features)
    axis.set_title("Feature variance explained by each ternary FSQ axis (η²)")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", color="white" if value > matrix.max() * 0.45 else "black", fontsize=8)
    figure.colorbar(image, ax=axis, label="eta squared")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def draw_metadata_heatmaps(
    code_rows: list[dict],
    tokens: np.ndarray,
    dataset: dict[str, Any],
    output_path: Path,
) -> None:
    codes = [row["code"] for row in code_rows]
    families = ["KITCHEN", "LIVING_ROOM", "STUDY"]
    family_matrix = np.asarray([
        [np.mean(dataset["scene_family"][tokens == code] == family) for family in families]
        for code in codes
    ])
    skill_groups = ["0", "1", "2", "3+"]
    indices = dataset["metadata"]["skill_index"]
    skill_matrix = np.asarray([
        [
            np.mean(indices[tokens == code] == 0),
            np.mean(indices[tokens == code] == 1),
            np.mean(indices[tokens == code] == 2),
            np.mean(indices[tokens == code] >= 3),
        ]
        for code in codes
    ])
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 10), sharey=True, constrained_layout=True)
    for axis, values, columns, title in (
        (axes[0], family_matrix, families, "Scene-family composition"),
        (axes[1], skill_matrix, skill_groups, "Skill-index composition"),
    ):
        image = axis.imshow(values, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
        axis.set_xticks(range(len(columns)), columns, rotation=30, ha="right")
        axis.set_yticks(range(len(codes)), [f"code {code}" for code in codes])
        axis.set_title(title)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                axis.text(column, row, f"{value:.0%}", ha="center", va="center", color="white" if value > 0.55 else "black", fontsize=7)
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def draw_start_pose_scatter(
    tokens: np.ndarray, names: list[str], matrix: np.ndarray, output_path: Path
) -> None:
    x = matrix[:, names.index("start_x")]
    z = matrix[:, names.index("start_z")]
    figure, axis = plt.subplots(figsize=(10.5, 7.2), constrained_layout=True)
    scatter = axis.scatter(x, z, c=tokens, cmap="turbo", vmin=-0.5, vmax=26.5, s=7, alpha=0.42, linewidths=0)
    colorbar = figure.colorbar(scatter, ax=axis, ticks=np.arange(27))
    colorbar.set_label("FSQ code")
    axis.set_xlabel("start EE x (m)")
    axis.set_ylabel("start EE z (m)")
    axis.set_title("All training skills: absolute start pose colored by code")
    axis.grid(alpha=0.2)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    flattened = []
    for row in rows:
        flattened.append({
            key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        })
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)


def format_value(feature: str, value: float) -> str:
    if feature in {"frames", "skill_index"}:
        return f"{value:.1f}"
    if feature == "skill_order":
        return f"{value:.2f}"
    if feature.startswith("grip_"):
        return f"{value:.3f}"
    if "angle" in feature or feature.startswith("rot_") or feature.startswith("rv_"):
        return f"{value:.3f} rad"
    return f"{value:.3f} m"


def feature_direction_ko(item: dict) -> str:
    feature = item["feature"]
    higher = item["code_mean"] > item["rest_mean"]
    if feature.startswith("disp_") or feature.startswith("rel"):
        tendency = "더 + 방향" if higher else "더 - 방향"
    elif feature in {"frames"}:
        tendency = "더 김" if higher else "더 짧음"
    elif feature in {"skill_index", "skill_order"}:
        tendency = "episode 후반" if higher else "episode 초반"
    elif feature in {"net_xyz", "path_xyz", "rot_net_angle", "rot_path_angle", "grip_range", "grip_path"}:
        tendency = "더 큼" if higher else "더 작음"
    else:
        tendency = "전체보다 높음" if higher else "전체보다 낮음"
    return (
        f"{FEATURE_LABEL_KO.get(feature, feature)} {tendency} "
        f"({format_value(feature, item['code_mean'])} vs 나머지 {format_value(feature, item['rest_mean'])}, "
        f"AUC {item['auc_separation']:.2f})"
    )


def tree_rule_ko(tree_rule: dict | None) -> str:
    if tree_rule is None:
        return "표본이 너무 적어 규칙을 만들지 않음"
    conditions = [
        f"{FEATURE_LABEL_KO.get(rule['feature'], rule['feature'])} {rule['operator']} {rule['threshold']:.3f}"
        for rule in tree_rule["rules"]
    ]
    return (
        " AND ".join(conditions)
        + f" (precision {tree_rule['precision']:.1%}, recall {tree_rule['recall']:.1%}; 탐색적 in-sample 규칙)"
    )


def write_html_report(
    output_path: Path,
    model_name: str,
    manifest: dict,
    summary: dict,
    code_rows: list[dict],
    ablation_rows: list[dict],
    importance_rows: list[dict],
    axis_rows: list[dict],
) -> None:
    associations = summary["categorical_associations"]
    ablation_lookup = {row["feature_group"]: row for row in ablation_rows}
    start_only = ablation_lookup["시작 XYZ만"]
    absolute = ablation_lookup["절대 XYZ 궤적"]
    relative = ablation_lookup["상대 XYZ 모션"]
    all_features = ablation_lookup["전체 proprio 특징"]
    top_importance = importance_rows[:10]

    axis_cards = []
    for axis in range(3):
        numeric = sorted(
            (row for row in axis_rows if row.get("axis") == axis and "eta_squared" in row),
            key=lambda row: row["eta_squared"],
            reverse=True,
        )[:5]
        feature_lines = "".join(
            f"<li><code>{html.escape(row['feature'])}</code> · η² {row['eta_squared']:.3f} · Spearman r {row['spearman_r']:+.3f}</li>"
            for row in numeric
        )
        axis_cards.append(
            f"<section class=\"card\"><h3>FSQ axis {axis}</h3><ol>{feature_lines}</ol></section>"
        )

    ablation_table = "".join(
        f"<tr><td>{html.escape(row['feature_group'])}</td><td>{row['feature_count']}</td>"
        f"<td>{row['accuracy_mean']:.1%} ± {row['accuracy_std']:.1%}</td>"
        f"<td>{row['balanced_accuracy_mean']:.1%}</td><td>{row['macro_f1_mean']:.1%}</td>"
        f"<td>{row['top3_accuracy_mean']:.1%}</td></tr>"
        for row in ablation_rows
    )
    importance_table = "".join(
        f"<tr><td>{rank}</td><td><code>{html.escape(row['feature'])}</code><br><span class=ko>{html.escape(FEATURE_LABEL_KO.get(row['feature'], row['feature']))}</span></td><td>{row['importance']:.4f}</td></tr>"
        for rank, row in enumerate(top_importance, start=1)
    )

    code_table_rows = []
    for row in code_rows:
        feature_list = "".join(f"<li>{html.escape(feature_direction_ko(item))}</li>" for item in row["strongest_features"])
        task = row["dominant_task"]
        family = row["dominant_scene_family"]
        skill_index = row["dominant_skill_index"]
        reliability_class = {
            "높음": "good", "보통": "mid", "낮음": "warn", "표본 부족": "bad"
        }[row["reliability"]]
        code_table_rows.append(
            f"""
            <tr data-code="{row['code']}" data-count="{row['count']}" data-text="{html.escape(task['value'].lower())}">
              <td><strong>code {row['code']}</strong><br><code>{tuple(row['coordinate'])}</code></td>
              <td>{row['count']:,}<br><span class="pill {reliability_class}">{row['reliability']}</span></td>
              <td><details open><summary>상위 수치 기준</summary><ol>{feature_list}</ol><div class=rule><b>얕은 트리 규칙:</b> {html.escape(tree_rule_ko(row['tree_rule']))}</div></details></td>
              <td><b>{html.escape(family['value'])}</b> {family['purity']:.1%}<br><span class=muted>{html.escape(task['value'])} · {task['purity']:.1%}</span></td>
              <td>index {skill_index['value']} · {skill_index['purity']:.1%}</td>
            </tr>
            """
        )

    conclusion_relation = "절대 위치 쪽이 더 강함" if absolute["accuracy_mean"] > relative["accuracy_mean"] else "상대 모션 쪽이 더 강함"
    document = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{html.escape(model_name)} · 전체 코드 분석</title>
  <style>
    :root {{ color-scheme:dark; --bg:#09101d; --panel:#101a2c; --line:#293752; --text:#e9eff9; --muted:#9aa9c2; --cyan:#63d4e9; }}
    * {{ box-sizing:border-box }} body {{ margin:0; background:radial-gradient(circle at top,#162944,var(--bg) 42rem); color:var(--text); font:14px/1.55 system-ui,-apple-system,"Noto Sans KR",sans-serif }}
    main {{ width:min(1480px,calc(100% - 30px)); margin:auto; padding:34px 0 80px }} h1 {{ margin:0; font-size:clamp(27px,4vw,43px); letter-spacing:-.04em }} h2 {{ margin:38px 0 13px; font-size:22px }} h3 {{ margin:0 0 8px }}
    .sub,.muted,.ko {{ color:var(--muted) }} .notice {{ border:1px solid #2b6482; background:#10283c; padding:17px 19px; border-radius:14px; font-size:16px }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:13px; margin:13px 0 }} .card,.figure,.table-wrap {{ border:1px solid var(--line); background:rgba(16,26,44,.94); border-radius:13px }} .card {{ padding:16px }} .big {{ font-size:25px; font-weight:780 }}
    .figures {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(480px,1fr)); gap:14px }} .figure {{ padding:11px; overflow:auto }} .figure img {{ display:block; width:100%; min-width:460px; border-radius:7px }} .figure.wide {{ grid-column:1/-1 }}
    .caption {{ color:var(--muted); margin:8px 5px 2px }} .table-wrap {{ overflow:auto }} table {{ width:100%; border-collapse:collapse; min-width:900px }} th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); vertical-align:top; text-align:right }} th {{ position:sticky; top:0; background:#142039; z-index:1; font-size:12px; text-transform:uppercase; color:#cbd6e8 }} th:first-child,td:first-child,th:nth-child(3),td:nth-child(3),th:nth-child(4),td:nth-child(4) {{ text-align:left }} tbody tr:hover {{ background:#17243d }}
    code {{ color:#b8dcff }} ol {{ margin:7px 0 2px; padding-left:20px }} .rule {{ color:#c9d5e8; background:#0b1425; border-radius:7px; padding:8px; margin-top:8px }} summary {{ color:var(--cyan); cursor:pointer }}
    .pill {{ display:inline-block; padding:1px 7px; border-radius:99px; font-size:11px; margin-top:4px }} .good {{ background:#164e3b;color:#8ff0c1 }} .mid {{ background:#233f6b;color:#b6d5ff }} .warn {{ background:#5b4218;color:#ffd590 }} .bad {{ background:#5a2430;color:#ffb2bd }}
    .controls {{ display:flex; flex-wrap:wrap; gap:9px; margin:0 0 12px }} input {{ border:1px solid var(--line); background:#0f192b; color:var(--text); border-radius:8px; padding:9px 11px }} input[type=search] {{ min-width:260px }}
    @media(max-width:700px) {{ main {{ width:calc(100% - 18px);padding-top:20px }} .figures {{ grid-template-columns:1fr }} }}
  </style>
</head>
<body><main>
  <h1>{html.escape(model_name)}<br>전체 FSQ 코드 categorization 분석</h1>
  <p class=sub>{html.escape(manifest['run_name'])} · {html.escape(manifest['epoch_tag'])}</p>
  <div class=notice><b>분석 범위:</b> YAML replay task_ids가 아니라 checkpoint의 전체 학습 skill <b>{summary['sample_count']:,}개</b>를 사용했다. 27개 코드를 모두 포함한다. 수치적 연관은 코드가 무엇을 보며 갈렸는지에 대한 단서이지 인과 증명은 아니다.</div>

  <h2>핵심 결론</h2>
  <div class=grid>
    <section class=card><div class=big>{conclusion_relation}</div><div>절대 XYZ 궤적 top-1 {absolute['accuracy_mean']:.1%}<br>상대 XYZ 모션 top-1 {relative['accuracy_mean']:.1%}<br>시작 XYZ만 {start_only['accuracy_mean']:.1%}</div></section>
    <section class=card><div class=big>전체 특징 {all_features['accuracy_mean']:.1%}</div><div>top-3 {all_features['top3_accuracy_mean']:.1%}<br>macro-F1 {all_features['macro_f1_mean']:.1%}<br>episode-grouped {summary['cv_folds']}-fold CV</div></section>
    <section class=card><div class=big>task 연관 NMI {associations['task_id_nmi']:.3f}</div><div>scene file {associations['scene_file_nmi']:.3f}<br>scene family {associations['scene_family_nmi']:.3f}<br>skill index {associations['skill_index_nmi']:.3f}</div></section>
    <section class=card><div class=big>{summary['used_codes']}/27 codes</div><div>effective codes {manifest['train_codebook_effective']:.2f}<br>최대 code {summary['largest_code']} · {summary['largest_count']:,}개</div></section>
  </div>
  <p class=notice>해석: 이 모델의 코드 분류는 한 가지 의미축이 아니라 <b>절대 pose, 상대 이동, 회전, skill 순서</b>의 조합이다. 아래 FSQ-axis 표는 세 ternary 축이 어느 특징과 가장 강하게 연결되는지, 코드별 표는 각 조합이 어떤 데이터 영역을 차지하는지 보여준다.</p>

  <h2>특징군별 코드 예측력</h2>
  <div class=table-wrap><table><thead><tr><th>특징군</th><th>차원</th><th>top-1 accuracy</th><th>balanced accuracy</th><th>macro-F1</th><th>top-3</th></tr></thead><tbody>{ablation_table}</tbody></table></div>

  <h2>FSQ 3개 축의 역할</h2><div class=grid>{''.join(axis_cards)}</div>
  <div class=figures><section class=figure><a href="fsq_axis_feature_heatmap.png"><img src="fsq_axis_feature_heatmap.png"></a><p class=caption>η²는 각 ternary 축(-1/0/1)이 해당 특징 분산을 얼마나 설명하는지 나타낸다.</p></section>
  <section class=figure><div class=table-wrap><table><thead><tr><th>rank</th><th>전체 code 예측 중요 특징</th><th>importance</th></tr></thead><tbody>{importance_table}</tbody></table></div><p class=caption>전체 학습 데이터에 fit한 ExtraTrees의 탐색적 feature importance. 상관된 특징끼리는 중요도가 나뉠 수 있다.</p></section></div>

  <h2>전체 코드 시각화</h2><div class=figures>
    <section class="figure wide"><a href="code_feature_heatmap.png"><img src="code_feature_heatmap.png"></a><p class=caption>각 코드 평균이 전체 평균에서 얼마나 벗어나는지 z-score로 표시했다.</p></section>
    <section class=figure><a href="code_counts.png"><img src="code_counts.png"></a><p class=caption>전체 학습 skill 기준 code count.</p></section>
    <section class=figure><a href="code_metadata_heatmaps.png"><img src="code_metadata_heatmaps.png"></a><p class=caption>코드별 scene family와 skill index 구성.</p></section>
    <section class="figure wide"><a href="start_pose_by_code.png"><img src="start_pose_by_code.png"></a><p class=caption>절대 시작 x-z 위치와 코드. 같은 위치에서 여러 코드가 겹치면 후속 모션/회전/순서가 추가 분류축이라는 뜻이다.</p></section>
  </div>

  <h2>코드별 추정 categorization 기준</h2>
  <p class=sub>상위 수치 기준은 code-vs-rest 단일 특징 AUC 순서다. 얕은 트리 규칙은 최대 깊이 3의 탐색적 in-sample 규칙이며, 실제 neural encoder의 정확한 decision boundary로 해석하면 안 된다.</p>
  <div class=controls><input id=codeSearch type=search placeholder="code 또는 task 검색"><label>최소 count <input id=minCount type=number value=0 min=0 step=10></label></div>
  <div class=table-wrap><table id=codeTable><thead><tr><th>code / coord</th><th>count / 신뢰도</th><th>추정 수치 기준</th><th>dominant scene/task</th><th>dominant order</th></tr></thead><tbody>{''.join(code_table_rows)}</tbody></table></div>
  <p class=sub>원본 수치: <a href="code_summary.csv">code_summary.csv</a> · <a href="classification_ablation.csv">classification_ablation.csv</a> · <a href="global_feature_importance.csv">global_feature_importance.csv</a> · <a href="fsq_axis_associations.csv">fsq_axis_associations.csv</a> · <a href="summary.json">summary.json</a></p>
</main>
<script>
  const search=document.getElementById('codeSearch'), minCount=document.getElementById('minCount');
  const rows=[...document.querySelectorAll('#codeTable tbody tr')];
  function filter(){{const q=search.value.trim().toLowerCase(),n=Number(minCount.value||0);for(const row of rows){{const hit=!q||row.dataset.code.includes(q)||row.dataset.text.includes(q);row.hidden=!(hit&&Number(row.dataset.count)>=n)}}}}
  search.addEventListener('input',filter);minCount.addEventListener('input',filter);
</script></body></html>"""
    output_path.write_text(document)


def analyze_model(
    model_name: str,
    model_root: Path,
    dataset: dict[str, Any],
    epoch: str,
    output_subdir: str,
    cv_folds: int,
) -> dict:
    latent, manifest = load_latents(model_root, epoch, dataset)
    dataset["features"]["skill_index"] = latent["skill_index"].astype(np.float64)
    names, matrix = numeric_matrix(dataset, latent)
    tokens = latent["tokens"].astype(np.int64)
    coordinates = latent["latents"].astype(np.int64)
    groups = feature_groups(names)

    ablation = evaluate_feature_groups(
        matrix,
        tokens,
        latent["episode_id"],
        groups,
        cv_folds,
    )
    importance = fit_global_importance(names, matrix, tokens)
    axes = analyze_axes(names, matrix, coordinates, dataset)
    codes = analyze_codes(tokens, coordinates, names, matrix, dataset)
    associations = code_metadata_associations(tokens, dataset)
    largest = max(codes, key=lambda row: row["count"])
    summary = {
        "model_name": model_name,
        "run_name": manifest["run_name"],
        "epoch": epoch,
        "scope": "complete training skillset",
        "sample_count": int(len(tokens)),
        "used_codes": int(len(codes)),
        "largest_code": largest["code"],
        "largest_count": largest["count"],
        "cv_folds": cv_folds,
        "categorical_associations": associations,
    }

    output_dir = model_root / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    draw_code_counts(codes, output_dir / "code_counts.png")
    draw_code_feature_heatmap(codes, tokens, names, matrix, output_dir / "code_feature_heatmap.png")
    draw_axis_heatmap(axes, output_dir / "fsq_axis_feature_heatmap.png")
    draw_metadata_heatmaps(codes, tokens, dataset, output_dir / "code_metadata_heatmaps.png")
    draw_start_pose_scatter(tokens, names, matrix, output_dir / "start_pose_by_code.png")

    write_csv(output_dir / "code_summary.csv", codes)
    write_csv(output_dir / "classification_ablation.csv", ablation)
    write_csv(output_dir / "global_feature_importance.csv", importance)
    write_csv(output_dir / "fsq_axis_associations.csv", axes)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                **summary,
                "classification_ablation": ablation,
                "top_global_features": importance[:20],
                "codes": codes,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    write_html_report(
        output_dir / "index.html",
        model_name,
        manifest,
        summary,
        codes,
        ablation,
        importance,
        axes,
    )
    print(f"[{model_name}] wrote {output_dir / 'index.html'}")
    return {**summary, "output_dir": str(output_dir)}


def main() -> None:
    args = parse_args()
    collections = load_collections(args.report_root, set(args.models))
    print(f"Loading and featurizing {args.skill_bundle} ...")
    dataset = load_dataset(args.skill_bundle, args.episode_metadata)
    results = []
    for model_name in args.models:
        model_root, _ = collections[model_name]
        print(f"Analyzing {model_name} ...")
        results.append(
            analyze_model(
                model_name,
                model_root,
                dataset,
                args.epoch,
                args.output_subdir,
                args.cv_folds,
            )
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
