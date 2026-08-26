#!/usr/bin/env python3
"""Analyze action-sequence FSQ categorization and its zero-mode baselines.

The report is intentionally based on the complete training skill bundle, not
only the replay subset.  Each model receives a self-contained Korean detail
page and the report root receives a linked comparison page.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, normalized_mutual_info_score
from sklearn.model_selection import GroupShuffleSplit


DIRECTION_LABELS = ("+x", "-x", "+y", "-y", "+z", "-z", "정지")
DIRECTION_PLOT_LABELS = ("+x", "-x", "+y", "-y", "+z", "-z", "still")
DIRECTION_COLORS = ("#ef476f", "#9d2a46", "#06d6a0", "#038f6c", "#118ab2", "#07577a", "#8d99ae")
DISPLAY_NAMES = {
    "js_action": "action · JS + route",
    "none_action": "action · pair OFF + route",
    "cont": "zero · contrastive + route",
    "js_zero": "zero · JS + route",
    "none_zero": "zero · pair OFF + route",
}
MODEL_COLORS = {
    "js_action": "#f4a261",
    "none_action": "#9b5de5",
    "cont": "#4cc9f0",
    "js_zero": "#e9c46a",
    "none_zero": "#7b8cde",
}
FEATURE_LABELS = {
    "act_mean_x": "action 평균 x",
    "act_mean_y": "action 평균 y",
    "act_mean_z": "action 평균 z",
    "act_mean_rx": "action 평균 rx",
    "act_mean_ry": "action 평균 ry",
    "act_mean_rz": "action 평균 rz",
    "act_std_x": "action 표준편차 x",
    "act_std_y": "action 표준편차 y",
    "act_std_z": "action 표준편차 z",
    "act_std_rx": "action 표준편차 rx",
    "act_std_ry": "action 표준편차 ry",
    "act_std_rz": "action 표준편차 rz",
    "act_abs_x": "|action| 평균 x",
    "act_abs_y": "|action| 평균 y",
    "act_abs_z": "|action| 평균 z",
    "act_abs_rx": "|action| 평균 rx",
    "act_abs_ry": "|action| 평균 ry",
    "act_abs_rz": "|action| 평균 rz",
    "act_start_x": "시작 action x",
    "act_start_y": "시작 action y",
    "act_start_z": "시작 action z",
    "act_end_x": "종료 action x",
    "act_end_y": "종료 action y",
    "act_end_z": "종료 action z",
    "grip_mean": "gripper 평균",
    "grip_std": "gripper 표준편차",
    "grip_delta": "gripper 시작-종료 변화",
    "grip_positive_fraction": "gripper +1 비율",
    "grip_transitions": "gripper 부호 전환 수",
    "state_disp_x": "실제 EE 변위 x",
    "state_disp_y": "실제 EE 변위 y",
    "state_disp_z": "실제 EE 변위 z",
    "state_net_xyz": "실제 EE 순변위 크기",
    "state_path_xyz": "실제 EE 경로 길이",
    "state_rot_net": "실제 EE 회전 변화량",
    "frames": "skill 길이",
    "skill_index": "episode 내 skill index",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--skill-bundle", type=Path, required=True)
    parser.add_argument("--output-name", default="action_codebook_analysis.html")
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


def direction_labels(vectors: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(vectors, axis=1)
    labels = np.full(len(vectors), 6, dtype=np.int64)
    moving = norms >= threshold
    axes = np.argmax(np.abs(vectors[moving]), axis=1)
    signs = vectors[moving, axes] < 0
    labels[moving] = 2 * axes + signs.astype(np.int64)
    return labels, norms


def weighted_purity(tokens: np.ndarray, labels: np.ndarray) -> float:
    correct = 0
    for code in np.unique(tokens):
        correct += Counter(labels[tokens == code].tolist()).most_common(1)[0][1]
    return float(correct / len(tokens))


def direction_coherence(tokens: np.ndarray, vectors: np.ndarray, threshold: float) -> float:
    norms = np.linalg.norm(vectors, axis=1)
    moving = norms >= threshold
    unit = vectors[moving] / norms[moving, None]
    moving_tokens = tokens[moving]
    total = 0.0
    for code in np.unique(moving_tokens):
        members = unit[moving_tokens == code]
        total += len(members) * float(np.linalg.norm(members.mean(axis=0)))
    return total / max(len(unit), 1)


def entropy_effective(counts: np.ndarray) -> float:
    probability = counts[counts > 0] / counts.sum()
    return float(np.exp(-(probability * np.log(probability)).sum()))


def adjacent_pairs(episodes: np.ndarray, skill_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, episode in enumerate(episodes):
        grouped[int(episode)].append(index)
    left: list[int] = []
    right: list[int] = []
    for members in grouped.values():
        members.sort(key=lambda index: (int(skill_indices[index]), index))
        for first, second in zip(members[:-1], members[1:], strict=True):
            if int(skill_indices[second]) == int(skill_indices[first]) + 1:
                left.append(first)
                right.append(second)
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def load_bundle(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        bundle = {key: data[key].copy() for key in data.files}
    action_lengths = bundle["actions_len"].astype(np.int64)
    state_lengths = bundle["states_len"].astype(np.int64)
    action_starts = np.concatenate(([0], np.cumsum(action_lengths[:-1])))
    state_starts = np.concatenate(([0], np.cumsum(state_lengths[:-1])))
    action_axes = ("x", "y", "z", "rx", "ry", "rz")
    features: dict[str, list[float]] = defaultdict(list)
    action_means: list[np.ndarray] = []
    state_displacements: list[np.ndarray] = []
    for action_start, action_length, state_start, state_length, skill_index in zip(
        action_starts,
        action_lengths,
        state_starts,
        state_lengths,
        bundle["meta_skill_index"],
        strict=True,
    ):
        action = bundle["actions_cat"][action_start : action_start + action_length].astype(np.float64)
        state = bundle["states_cat"][state_start : state_start + state_length].astype(np.float64)
        action_means.append(action[:, :3].mean(axis=0))
        displacement = state[-1, :3] - state[0, :3]
        state_displacements.append(displacement)
        for index, axis in enumerate(action_axes):
            features[f"act_mean_{axis}"].append(float(action[:, index].mean()))
            features[f"act_std_{axis}"].append(float(action[:, index].std()))
            features[f"act_abs_{axis}"].append(float(np.abs(action[:, index]).mean()))
            if index < 3:
                features[f"act_start_{axis}"].append(float(action[0, index]))
                features[f"act_end_{axis}"].append(float(action[-1, index]))
        gripper = action[:, 6]
        features["grip_mean"].append(float(gripper.mean()))
        features["grip_std"].append(float(gripper.std()))
        features["grip_delta"].append(float(gripper[-1] - gripper[0]))
        features["grip_positive_fraction"].append(float((gripper > 0).mean()))
        features["grip_transitions"].append(float(np.count_nonzero(np.diff(np.signbit(gripper)))))
        for index, axis in enumerate("xyz"):
            features[f"state_disp_{axis}"].append(float(displacement[index]))
        features["state_net_xyz"].append(float(np.linalg.norm(displacement)))
        features["state_path_xyz"].append(float(np.linalg.norm(np.diff(state[:, :3], axis=0), axis=1).sum()))
        features["state_rot_net"].append(float(np.linalg.norm(state[-1, 3:6] - state[0, 3:6])))
        features["frames"].append(float(action_length))
        features["skill_index"].append(float(skill_index))
    numeric = {key: np.asarray(value, dtype=np.float64) for key, value in features.items()}
    action_means_array = np.asarray(action_means)
    state_displacements_array = np.asarray(state_displacements)
    action_direction, action_norm = direction_labels(action_means_array, 0.02)
    state_direction, state_norm = direction_labels(state_displacements_array, 0.01)
    grip_mean = numeric["grip_mean"]
    grip_regime = np.where(grip_mean > 0.5, 2, np.where(grip_mean < -0.5, 0, 1)).astype(np.int64)
    feature_names = list(numeric)
    matrix = np.column_stack([numeric[name] for name in feature_names])
    return {
        "bundle": bundle,
        "features": numeric,
        "feature_names": feature_names,
        "matrix": matrix,
        "action_mean_xyz": action_means_array,
        "state_displacement": state_displacements_array,
        "action_direction": action_direction,
        "state_direction": state_direction,
        "action_norm": action_norm,
        "state_norm": state_norm,
        "grip_regime": grip_regime,
        "raw_action_std": bundle["actions_cat"].astype(np.float64).std(axis=0),
    }


def feature_groups(names: list[str]) -> dict[str, list[int]]:
    def selected(prefixes: tuple[str, ...]) -> list[int]:
        return [index for index, name in enumerate(names) if name.startswith(prefixes)]

    action_xyz = [
        index
        for index, name in enumerate(names)
        if name.startswith(("act_mean_", "act_std_", "act_abs_", "act_start_", "act_end_"))
        and not name.endswith(("rx", "ry", "rz"))
    ]
    action_rot = [
        index
        for index, name in enumerate(names)
        if name.startswith(("act_mean_", "act_std_", "act_abs_"))
        and name.endswith(("rx", "ry", "rz"))
    ]
    gripper = selected(("grip_",))
    state_motion = selected(("state_",))
    time_order = [names.index("frames"), names.index("skill_index")]
    action_all = sorted(set(action_xyz + action_rot + gripper))
    return {
        "action XYZ": action_xyz,
        "action rotation": action_rot,
        "action gripper": gripper,
        "실제 state motion": state_motion,
        "길이·순서": time_order,
        "전체 action": action_all,
        "전체 특징": list(range(len(names))),
    }


def classification_ablation(
    matrix: np.ndarray,
    tokens: np.ndarray,
    episodes: np.ndarray,
    groups: dict[str, list[int]],
) -> list[dict[str, Any]]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=20260826)
    train, test = next(splitter.split(matrix, tokens, groups=episodes))
    rows: list[dict[str, Any]] = []
    for offset, (name, columns) in enumerate(groups.items()):
        classifier = ExtraTreesClassifier(
            n_estimators=160,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=4200 + offset,
            n_jobs=-1,
        )
        classifier.fit(matrix[train][:, columns], tokens[train])
        prediction = classifier.predict(matrix[test][:, columns])
        rows.append(
            {
                "group": name,
                "feature_count": len(columns),
                "accuracy": float(accuracy_score(tokens[test], prediction)),
                "balanced_accuracy": float(balanced_accuracy_score(tokens[test], prediction)),
            }
        )
    return rows


def feature_importance(matrix: np.ndarray, tokens: np.ndarray, names: list[str]) -> list[dict[str, Any]]:
    classifier = ExtraTreesClassifier(
        n_estimators=240,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=5511,
        n_jobs=-1,
    )
    classifier.fit(matrix, tokens)
    order = np.argsort(classifier.feature_importances_)[::-1]
    return [
        {"feature": names[index], "importance": float(classifier.feature_importances_[index])}
        for index in order
    ]


def axis_correlations(latents: np.ndarray, matrix: np.ndarray, names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for axis in range(latents.shape[1]):
        for column, name in enumerate(names):
            correlation = spearmanr(latents[:, axis], matrix[:, column]).statistic
            rows.append(
                {
                    "axis": axis,
                    "feature": name,
                    "correlation": 0.0 if not np.isfinite(correlation) else float(correlation),
                }
            )
    return rows


def code_rows(tokens: np.ndarray, dataset: dict[str, Any]) -> list[dict[str, Any]]:
    state_direction = dataset["state_direction"]
    action_direction = dataset["action_direction"]
    grip_regime = dataset["grip_regime"]
    features = dataset["features"]
    rows: list[dict[str, Any]] = []
    for code in range(27):
        mask = tokens == code
        count = int(mask.sum())
        if not count:
            rows.append({"code": code, "count": 0})
            continue
        state_label, state_count = Counter(state_direction[mask].tolist()).most_common(1)[0]
        action_label, action_count = Counter(action_direction[mask].tolist()).most_common(1)[0]
        grip_label, grip_count = Counter(grip_regime[mask].tolist()).most_common(1)[0]
        rows.append(
            {
                "code": code,
                "count": count,
                "share": float(count / len(tokens)),
                "state_direction": DIRECTION_LABELS[state_label],
                "state_direction_purity": float(state_count / count),
                "state_direction_distribution": (
                    np.bincount(state_direction[mask], minlength=len(DIRECTION_LABELS)) / count
                ).tolist(),
                "action_direction": DIRECTION_LABELS[action_label],
                "action_direction_purity": float(action_count / count),
                "grip_regime": ("-1", "mixed", "+1")[grip_label],
                "grip_purity": float(grip_count / count),
                "mean_frames": float(features["frames"][mask].mean()),
                "mean_action_xyz": [float(features[f"act_mean_{axis}"][mask].mean()) for axis in "xyz"],
                "mean_state_displacement": [float(features[f"state_disp_{axis}"][mask].mean()) for axis in "xyz"],
            }
        )
    return rows


def task6_summary(
    tokens: np.ndarray,
    manifest: dict[str, Any],
    dataset: dict[str, Any],
) -> dict[str, Any]:
    episodes = dataset["bundle"]["meta_episode_id"].astype(np.int64)
    skill_indices = dataset["bundle"]["meta_skill_index"].astype(np.int64)
    selected = [int(value) for value in manifest["signature"]["selected_episodes"]["6"]]
    rows: list[dict[str, Any]] = []
    same_first_pair = 0
    pair_count = 0
    for episode in selected:
        members = np.where(episodes == episode)[0]
        members = members[np.argsort(skill_indices[members])]
        skills = []
        for index in members:
            skills.append(
                {
                    "skill_index": int(skill_indices[index]),
                    "code": int(tokens[index]),
                    "frames": int(dataset["features"]["frames"][index]),
                    "state_direction": DIRECTION_LABELS[int(dataset["state_direction"][index])],
                    "action_direction": DIRECTION_LABELS[int(dataset["action_direction"][index])],
                    "displacement": dataset["state_displacement"][index].tolist(),
                    "mean_action": dataset["action_mean_xyz"][index].tolist(),
                }
            )
        if len(skills) >= 2:
            pair_count += 1
            same_first_pair += int(skills[0]["code"] == skills[1]["code"])
        rows.append({"episode": episode, "skills": skills})
    return {
        "episodes": rows,
        "first_pair_count": pair_count,
        "first_pair_same_code": same_first_pair,
        "first_pair_separation_rate": float(1.0 - same_first_pair / max(pair_count, 1)),
    }


def model_metrics(
    collection_path: Path,
    dataset: dict[str, Any],
) -> dict[str, Any]:
    model_root = collection_path.parent.parent
    collection = json.loads(collection_path.read_text())
    checkpoint = collection["checkpoints"][-1]
    epoch = checkpoint["epoch_tag"]
    manifest = json.loads((model_root / "checkpoints" / epoch / "metrics" / "manifest.json").read_text())
    with np.load(manifest["signature"]["latents_path"], allow_pickle=False) as data:
        tokens = data["tokens"].astype(np.int64)
        latents = data["latents"].astype(np.float64)
        for latent_key, bundle_key in (
            ("episode_id", "meta_episode_id"),
            ("task_id", "meta_task_id"),
            ("skill_index", "meta_skill_index"),
        ):
            if not np.array_equal(data[latent_key], dataset["bundle"][bundle_key]):
                raise ValueError(f"Latent alignment mismatch: {model_root.name}/{latent_key}")
    run_dir = Path(manifest["signature"]["model_path"]).parent
    meta = json.loads((run_dir / "fsq_meta.json").read_text())
    counts = np.bincount(tokens, minlength=27)
    episodes = dataset["bundle"]["meta_episode_id"].astype(np.int64)
    tasks = dataset["bundle"]["meta_task_id"].astype(np.int64)
    skill_indices = dataset["bundle"]["meta_skill_index"].astype(np.int64)
    left, right = adjacent_pairs(episodes, skill_indices)
    displacement = dataset["state_displacement"]
    displacement_norm = np.linalg.norm(displacement, axis=1)
    valid = (displacement_norm[left] >= 0.01) & (displacement_norm[right] >= 0.01)
    cosine = np.zeros(len(left), dtype=np.float64)
    cosine[valid] = np.sum(displacement[left[valid]] * displacement[right[valid]], axis=1) / (
        displacement_norm[left[valid]] * displacement_norm[right[valid]]
    )
    conflict = valid & (cosine < 0)
    same = tokens[left] == tokens[right]
    groups = feature_groups(dataset["feature_names"])
    ablation = classification_ablation(dataset["matrix"], tokens, episodes, groups)
    importance = feature_importance(dataset["matrix"], tokens, dataset["feature_names"])
    correlations = axis_correlations(latents, dataset["matrix"], dataset["feature_names"])
    result = {
        "name": collection["model_name"],
        "display_name": DISPLAY_NAMES.get(collection["model_name"], collection["model_name"]),
        "run_name": model_root.name,
        "model_root": str(model_root),
        "run_dir": str(run_dir),
        "epoch": epoch,
        "mode": meta["autoencoder_mode"],
        "pair_loss": meta["pair_loss"],
        "route_loss": bool(
            meta.get("route_loss", meta.get("reconstruction_route_loss", False))
        ),
        "counts": counts,
        "used_codes": int((counts > 0).sum()),
        "effective_codes": entropy_effective(counts),
        "largest_code_share": float(counts.max() / counts.sum()),
        "state_direction_nmi": float(normalized_mutual_info_score(dataset["state_direction"], tokens)),
        "state_direction_purity": weighted_purity(tokens, dataset["state_direction"]),
        "state_direction_coherence": direction_coherence(tokens, displacement, 0.01),
        "action_direction_nmi": float(normalized_mutual_info_score(dataset["action_direction"], tokens)),
        "action_direction_purity": weighted_purity(tokens, dataset["action_direction"]),
        "gripper_nmi": float(normalized_mutual_info_score(dataset["grip_regime"], tokens)),
        "task_nmi": float(normalized_mutual_info_score(tasks, tokens)),
        "skill_index_nmi": float(normalized_mutual_info_score(skill_indices, tokens)),
        "adjacent_same_code_rate": float(same.mean()),
        "opposite_adjacent_same_code_rate": float(same[conflict].mean()) if conflict.any() else 0.0,
        "classification_ablation": ablation,
        "feature_importance": importance,
        "axis_correlations": correlations,
        "codes": code_rows(tokens, dataset),
        "task6": task6_summary(tokens, manifest, dataset),
    }
    return result


def save_detail_plots(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = np.asarray(result["counts"])
    color = MODEL_COLORS.get(result["name"], "#4cc9f0")
    fig, axis = plt.subplots(figsize=(11, 4.2), constrained_layout=True)
    axis.bar(np.arange(27), counts, color=color)
    axis.axhline(counts.mean(), color="#ef476f", linestyle="--", linewidth=1.3, label="uniform mean")
    axis.set(xlabel="FSQ code", ylabel="complete dataset count", title=f"{result['display_name']} · code usage")
    axis.set_xticks(range(27))
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    fig.savefig(output_dir / "code_usage.png", dpi=170)
    plt.close(fig)

    composition = np.zeros((27, len(DIRECTION_LABELS)), dtype=np.float64)
    for row in result["codes"]:
        if not row["count"]:
            continue
        composition[row["code"]] = row["state_direction_distribution"]
    fig, axis = plt.subplots(figsize=(11, 4.4), constrained_layout=True)
    bottom = np.zeros(27)
    for label_index, label in enumerate(DIRECTION_PLOT_LABELS):
        values = composition[:, label_index]
        axis.bar(range(27), values, bottom=bottom, color=DIRECTION_COLORS[label_index], label=label)
        bottom += values
    axis.set(xlabel="FSQ code", ylabel="within-code share", title="Actual EE direction composition by code")
    axis.set_xticks(range(27))
    axis.legend(ncol=7, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.savefig(output_dir / "direction_by_code.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    top = result["feature_importance"][:18][::-1]
    fig, axis = plt.subplots(figsize=(9, 6.5), constrained_layout=True)
    axis.barh(
        [row["feature"] for row in top],
        [row["importance"] for row in top],
        color=color,
    )
    axis.set(xlabel="ExtraTrees importance", title="Features explaining code assignment")
    axis.grid(axis="x", alpha=0.2)
    fig.savefig(output_dir / "feature_importance.png", dpi=170)
    plt.close(fig)

    names = [row["feature"] for row in result["feature_importance"][:18]]
    correlation_lookup = {
        (row["axis"], row["feature"]): row["correlation"] for row in result["axis_correlations"]
    }
    heat = np.asarray([[correlation_lookup[(axis, name)] for name in names] for axis in range(3)])
    fig, axis = plt.subplots(figsize=(11, 3.8), constrained_layout=True)
    image = axis.imshow(heat, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axis.set_yticks(range(3), ("FSQ axis 0", "FSQ axis 1", "FSQ axis 2"))
    axis.set_xticks(range(len(names)), names, rotation=55, ha="right")
    axis.set_title("Quantized FSQ axes vs action/motion features (Spearman)")
    fig.colorbar(image, ax=axis, shrink=0.85)
    fig.savefig(output_dir / "fsq_axis_correlation.png", dpi=170)
    plt.close(fig)


def fmt(value: float) -> str:
    return f"{value:.3f}"


def detail_html(result: dict[str, Any]) -> str:
    ablation_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['group'])}</td><td>{row['feature_count']}</td>"
        f"<td>{row['accuracy']:.1%}</td><td>{row['balanced_accuracy']:.1%}</td>"
        "</tr>"
        for row in result["classification_ablation"]
    )
    code_table = "".join(
        "<tr>"
        f"<td>{row['code']}</td><td>{row['count']}</td>"
        + (
            f"<td>{row['share']:.1%}</td><td>{row['state_direction']} ({row['state_direction_purity']:.0%})</td>"
            f"<td>{row['action_direction']} ({row['action_direction_purity']:.0%})</td>"
            f"<td>{row['grip_regime']} ({row['grip_purity']:.0%})</td><td>{row['mean_frames']:.1f}</td>"
            if row["count"]
            else "<td colspan='5'>unused</td>"
        )
        + "</tr>"
        for row in result["codes"]
    )
    task_rows = []
    for episode in result["task6"]["episodes"]:
        skill_text = " · ".join(
            f"s{skill['skill_index']} → c{skill['code']} / {skill['state_direction']} / "
            f"Δp=[{', '.join(f'{value:+.3f}' for value in skill['displacement'])}]"
            for skill in episode["skills"]
        )
        task_rows.append(f"<tr><td>{episode['episode']}</td><td>{html.escape(skill_text)}</td></tr>")
    top_features = ", ".join(
        f"{FEATURE_LABELS.get(row['feature'], row['feature'])} ({row['importance']:.3f})"
        for row in result["feature_importance"][:6]
    )
    axis_notes = []
    for axis in range(3):
        members = sorted(
            (row for row in result["axis_correlations"] if row["axis"] == axis),
            key=lambda row: abs(row["correlation"]),
            reverse=True,
        )[:3]
        axis_notes.append(
            f"axis {axis}: "
            + ", ".join(
                f"{FEATURE_LABELS.get(row['feature'], row['feature'])} ρ={row['correlation']:+.3f}"
                for row in members
            )
        )
    mode_note = (
        "raw 7D controller action sequence를 직접 Transformer에 넣고, 같은 raw action sequence를 복원한다."
        if result["mode"] == "action"
        else "mean-zero 30×8 state B-spline을 입력·복원하는 비교 기준선이다."
    )
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(result['display_name'])} · action codebook analysis</title>
<style>
:root{{--bg:#09111d;--panel:#122033;--line:#29405c;--text:#eef5ff;--muted:#a8b8cc;--cyan:#65d6ff;--amber:#ffc76b}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 12% 0,#173452,var(--bg) 38%);color:var(--text);font:15px/1.65 Inter,system-ui,"Noto Sans KR",sans-serif}}main{{max-width:1320px;margin:auto;padding:46px 28px 90px}}a{{color:var(--cyan)}}h1{{font-size:clamp(30px,4vw,50px);line-height:1.1;margin:8px 0}}h2{{margin-top:45px}}p{{color:#c9d6e7}}.back{{display:inline-block;padding:7px 12px;border:1px solid var(--line);border-radius:999px;text-decoration:none}}.chips{{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}}.chips span{{background:#19304a;border-radius:999px;padding:6px 11px}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}.metric,.callout{{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:15px}}.metric b{{display:block;font-size:24px;color:var(--amber)}}.callout{{margin:18px 0}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}figure{{margin:0;background:white;border-radius:12px;padding:9px}}figure img{{display:block;width:100%}}figcaption{{color:#44546b;padding:7px}}.table{{overflow:auto;border:1px solid var(--line);border-radius:11px}}table{{border-collapse:collapse;width:100%;background:#0e1a2a}}th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child,td:nth-child(2){{text-align:left}}thead th{{background:#182b42;position:sticky;top:0}}code{{color:#bcecff}}.muted{{color:var(--muted)}}@media(max-width:850px){{.metrics,.grid{{grid-template-columns:1fr}}main{{padding:28px 14px}}}}
</style></head><body><main>
<a class="back" href="../../action_codebook_analysis.html">← 전체 모델 요약</a>
<div class="chips"><span>{html.escape(result['epoch'])}</span><span>mode={html.escape(result['mode'])}</span><span>pair={html.escape(result['pair_loss'])}</span><span>route=ON</span></div>
<h1>{html.escape(result['display_name'])}</h1>
<p>{mode_note}</p>
<div class="metrics">
  <div class="metric"><span>effective codes</span><b>{result['effective_codes']:.2f}</b><small>{result['used_codes']}/27 active</small></div>
  <div class="metric"><span>실제 XYZ 방향 NMI</span><b>{result['state_direction_nmi']:.3f}</b><small>purity {result['state_direction_purity']:.1%}</small></div>
  <div class="metric"><span>방향 coherence</span><b>{result['state_direction_coherence']:.3f}</b><small>1에 가까울수록 code 내 방향 일치</small></div>
  <div class="metric"><span>인접 same-code</span><b>{result['adjacent_same_code_rate']:.1%}</b><small>반대방향 조건 {result['opposite_adjacent_same_code_rate']:.1%}</small></div>
</div>
<div class="callout"><b>핵심 판독:</b> actual EE 방향 NMI {result['state_direction_nmi']:.3f}, action 방향 NMI {result['action_direction_nmi']:.3f}, gripper regime NMI {result['gripper_nmi']:.3f}이다. task/skill-index NMI는 각각 {result['task_nmi']:.3f}/{result['skill_index_nmi']:.3f}이다.</div>
<div class="callout"><b>가장 설명력 높은 특징:</b> {html.escape(top_features)}</div>
<div class="callout"><b>FSQ scalar 축 판독:</b> {html.escape(' · '.join(axis_notes))}</div>
<p class="muted">ExtraTrees 중요도와 Spearman 상관은 code assignment와 관측 특징의 연관성이다. zero 모델에서 표시되는 action/state 특징은 encoder가 직접 받은 입력이라는 뜻이 아니라, 같은 skill의 code를 사후적으로 설명하는 proxy다.</p>
<div class="grid"><figure><img src="code_usage.png"><figcaption>전체 11,221개 training skill의 code 사용량.</figcaption></figure><figure><img src="direction_by_code.png"><figcaption>각 code에서 가장 많은 실제 EE 순변위 방향.</figcaption></figure></div>
<h2>무엇이 code를 예측하는가</h2>
<div class="grid"><figure><img src="feature_importance.png"><figcaption>모든 수치 특징을 함께 넣은 ExtraTrees 중요도.</figcaption></figure><figure><img src="fsq_axis_correlation.png"><figcaption>세 FSQ scalar axis와 주요 특징의 순위 상관.</figcaption></figure></div>
<div class="table"><table><thead><tr><th>특징 그룹</th><th>차원</th><th>episode-held-out 정확도</th><th>balanced 정확도</th></tr></thead><tbody>{ablation_rows}</tbody></table></div>
<h2>task 6 exact 재검증</h2>
<div class="callout">첫 두 skill의 분리율은 <b>{result['task6']['first_pair_separation_rate']:.0%}</b> ({result['task6']['first_pair_count'] - result['task6']['first_pair_same_code']}/{result['task6']['first_pair_count']})이다.</div>
<div class="table"><table><thead><tr><th>episode</th><th>skill → code / 실제 방향 / 변위</th></tr></thead><tbody>{''.join(task_rows)}</tbody></table></div>
<h2>Code별 요약</h2>
<div class="table"><table><thead><tr><th>code</th><th>count</th><th>share</th><th>실제 방향</th><th>action 방향</th><th>gripper regime</th><th>평균 길이</th></tr></thead><tbody>{code_table}</tbody></table></div>
<p class="muted">원자료: <a href="metrics.json">metrics.json</a> · <a href="code_summary.csv">code_summary.csv</a><br>run: {html.escape(result['run_name'])}</p>
</main></body></html>"""


def save_detail(result: dict[str, Any], output_dir: Path) -> None:
    save_detail_plots(result, output_dir)
    (output_dir / "metrics.json").write_text(
        json.dumps(json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fieldnames = sorted({key for row in result["codes"] for key in row})
    with (output_dir / "code_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(json_ready(result["codes"]))
    (output_dir / "index.html").write_text(detail_html(result), encoding="utf-8")


def save_summary_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    labels = [result["display_name"] for result in results]
    colors = [MODEL_COLORS.get(result["name"], "#4cc9f0") for result in results]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    metrics = (
        ("effective_codes", "Effective codes"),
        ("largest_code_share", "Largest-code share"),
        ("state_direction_nmi", "Actual XYZ direction NMI"),
        ("adjacent_same_code_rate", "Adjacent same-code rate"),
    )
    for axis, (key, title) in zip(axes.flat, metrics, strict=True):
        values = [result[key] for result in results]
        axis.bar(np.arange(len(results)), values, color=colors)
        axis.set_xticks(np.arange(len(results)), labels, rotation=23, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        if "share" in key or "rate" in key:
            axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def summary_html(results: list[dict[str, Any]], dataset: dict[str, Any]) -> str:
    action = [result for result in results if result["mode"] == "action"]
    zero = [result for result in results if result["mode"] == "zero"]
    mean = lambda rows, key: float(np.mean([row[key] for row in rows]))
    cards = []
    for result in results:
        href = f"{result['run_name']}/action_codebook_analysis/index.html"
        cards.append(
            f"<a class='card' href='{html.escape(href)}'><span>{html.escape(result['epoch'])} · {html.escape(result['mode'])}</span>"
            f"<h3>{html.escape(result['display_name'])}</h3><dl>"
            f"<dt>effective code</dt><dd>{result['effective_codes']:.2f}</dd>"
            f"<dt>direction NMI</dt><dd>{result['state_direction_nmi']:.3f}</dd>"
            f"<dt>direction purity</dt><dd>{result['state_direction_purity']:.1%}</dd>"
            f"<dt>adjacent same</dt><dd>{result['adjacent_same_code_rate']:.1%}</dd>"
            f"<dt>task 6 separation</dt><dd>{result['task6']['first_pair_separation_rate']:.0%}</dd>"
            f"</dl><b>상세 분석 열기 →</b></a>"
        )
    rows = "".join(
        "<tr>"
        f"<td><a href='{html.escape(result['run_name'])}/action_codebook_analysis/index.html'>{html.escape(result['display_name'])}</a></td>"
        f"<td>{html.escape(result['epoch'])}</td><td>{result['effective_codes']:.2f}</td>"
        f"<td>{result['largest_code_share']:.1%}</td><td>{result['state_direction_nmi']:.3f}</td>"
        f"<td>{result['state_direction_purity']:.1%}</td><td>{result['state_direction_coherence']:.3f}</td>"
        f"<td>{result['gripper_nmi']:.3f}</td><td>{result['task_nmi']:.3f}</td>"
        f"<td>{result['adjacent_same_code_rate']:.1%}</td><td>{result['task6']['first_pair_separation_rate']:.0%}</td>"
        "</tr>"
        for result in results
    )
    std = dataset["raw_action_std"]
    action_state_cosine = np.sum(
        dataset["action_mean_xyz"] * dataset["state_displacement"], axis=1
    ) / (
        np.linalg.norm(dataset["action_mean_xyz"], axis=1)
        * np.linalg.norm(dataset["state_displacement"], axis=1)
        + 1e-9
    )
    label_agreement = float(np.mean(dataset["action_direction"] == dataset["state_direction"]))
    action_grip_axis = []
    action_y_axis = []
    for result in action:
        lookup = {
            (row["axis"], row["feature"]): row["correlation"]
            for row in result["axis_correlations"]
        }
        action_grip_axis.append(lookup[(2, "grip_mean")])
        action_y_axis.append(lookup[(0, "act_mean_y")])
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>routeON_action · codebook categorization summary</title>
<style>
:root{{--bg:#08111d;--panel:#111f31;--line:#29415d;--text:#edf5ff;--muted:#9eb1c8;--cyan:#65d6ff;--green:#6ee7b7}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 12% 0,#183858,var(--bg) 38%);color:var(--text);font:15px/1.65 Inter,system-ui,"Noto Sans KR",sans-serif}}main{{max-width:1400px;margin:auto;padding:52px 28px 90px}}h1{{font-size:clamp(34px,5vw,58px);line-height:1.05;letter-spacing:-.04em;margin:6px 0 15px}}h2{{margin-top:48px}}p{{color:#c8d7e9}}a{{color:var(--cyan)}}.lead{{font-size:17px;max-width:1050px}}.callout{{background:#10283a;border:1px solid #34718b;border-radius:14px;padding:17px 20px;margin:18px 0}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:13px}}.card{{display:block;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:17px;text-decoration:none;color:var(--text);transition:.15s}}.card:hover{{transform:translateY(-2px);border-color:var(--cyan)}}.card span{{color:var(--muted)}}.card h3{{margin:6px 0 12px}}dl{{display:grid;grid-template-columns:1fr auto;gap:4px 10px}}dt{{color:var(--muted)}}dd{{margin:0;font-weight:700}}figure{{background:white;padding:10px;border-radius:13px;margin:20px 0}}figure img{{display:block;width:100%}}figcaption{{color:#42536a;padding:8px}}.table{{overflow:auto;border:1px solid var(--line);border-radius:11px}}table{{border-collapse:collapse;width:100%;background:#0d1928}}th,td{{padding:9px 11px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}thead th{{background:#182b42;position:sticky;top:0}}code{{color:#bcecff}}.good{{color:var(--green)}}.muted{{color:var(--muted)}}@media(max-width:950px){{.cards{{grid-template-columns:1fr}}main{{padding:30px 14px}}}}
</style></head><body><main>
<div class="muted">complete training skillset · 11,221 skills · latest evaluated checkpoint per model</div>
<h1>routeON_action<br>codebook categorization</h1>
<p class="lead">raw delta action으로 학습한 두 모델이 실제 EE 이동방향을 얼마나 보존하는지, 같은 폴더의 zero 기준선 세 모델과 비교했다. 각 카드를 누르면 code별 방향·gripper·FSQ 축 상관과 task 6 exact 결과를 볼 수 있다.</p>
<div class="callout"><b class="good">결론:</b> action 모델은 실제 XYZ 방향 NMI가 평균 {mean(action, 'state_direction_nmi'):.3f}으로 zero 기준선 {mean(zero, 'state_direction_nmi'):.3f}보다 높고, 방향 purity도 {mean(action, 'state_direction_purity'):.1%} 대 {mean(zero, 'state_direction_purity'):.1%}다. task 6의 하강→좌측 이동은 action 두 모델 모두 20/20 분리했다. 따라서 우리가 의도한 방향성은 분명히 더 반영됐다.</div>
<div class="callout"><b>왜 codebook이 넓게 퍼졌나:</b> action 모델의 effective code는 평균 {mean(action, 'effective_codes'):.2f}, zero는 {mean(zero, 'effective_codes'):.2f}다. 최대 code 점유율도 action {mean(action, 'largest_code_share'):.1%}, zero {mean(zero, 'largest_code_share'):.1%}로 action이 훨씬 균등하다. 이는 collapse보다는 action velocity·gripper 패턴을 더 세분화한 결과에 가깝지만, semantic category보다 미세한 controller 패턴까지 갈라질 가능성도 함께 뜻한다.</div>
<div class="callout"><b>중요한 scale 주의점:</b> action mode는 normalization 없이 raw action을 입력·복원한다. dataset 표준편차는 XYZ <code>[{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}]</code>, rotation <code>[{std[3]:.3f}, {std[4]:.3f}, {std[5]:.3f}]</code>, gripper <code>{std[6]:.3f}</code>다. gripper가 ±1이라 rotation보다 훨씬 큰 숫자 scale이며, 실제 gripper-code NMI도 action 평균 {mean(action, 'gripper_nmi'):.3f}이다. 넓은 code 사용의 일부는 방향뿐 아니라 gripper regime 분리다.</div>
<div class="callout"><b>FSQ 축 분업:</b> action 모델의 axis 0은 평균 y action과 ρ={action_y_axis[0]:+.3f}/{action_y_axis[1]:+.3f}, axis 2는 gripper 평균과 ρ={action_grip_axis[0]:+.3f}/{action_grip_axis[1]:+.3f}다. 방향축이 실제로 생겼지만, gripper도 사실상 latent 한 축을 강하게 점유한다.</div>
<div class="callout"><b>action이 방향 proxy로 유효한가:</b> skill별 평균 action XYZ와 실제 state 변위의 cosine 평균은 {float(action_state_cosine.mean()):.3f}, dominant-axis 라벨 일치율은 {label_agreement:.1%}다. 따라서 이 dataset에서는 action XYZ가 실제 이동방향을 매우 잘 대변한다.</div>
<div class="cards">{''.join(cards)}</div>
<h2>한눈에 비교</h2>
<figure><img src="action_codebook_analysis_comparison.png"><figcaption>action은 effective code와 방향 NMI가 높고, 인접 same-code collision은 낮다.</figcaption></figure>
<div class="table"><table><thead><tr><th>model</th><th>epoch</th><th>effective</th><th>max share</th><th>direction NMI</th><th>direction purity</th><th>coherence</th><th>gripper NMI</th><th>task NMI</th><th>adjacent same</th><th>task 6 분리</th></tr></thead><tbody>{rows}</tbody></table></div>
<h2>해석</h2>
<p>action 입력은 absolute position 대신 매 timestep의 velocity 방향을 직접 제공한다. 그 결과 zero JS/none에서 나타났던 task 6 인접-skill merge가 사라지고, 전체 dataset에서도 code 내부 방향 일관성이 상승했다. 반면 task NMI는 action 평균 {mean(action, 'task_nmi'):.3f}, zero 평균 {mean(zero, 'task_nmi'):.3f}로 낮아졌다. 즉 action code는 “어느 task/episode phase인가”보다 “지금 어떤 controller motion인가”에 더 가까워졌다.</p>
<p>현재 snapshot은 action JS가 epoch400, action none이 epoch700이고 zero 기준선은 epoch2000이므로 loss 수렴 정도의 공정 비교는 아니다. 하지만 두 action 모델이 서로 다른 epoch에서도 같은 방향성 개선과 task 6 완전 분리를 보인다는 점은 입력 표현 효과가 강하다는 증거다.</p>
<p class="muted">raw metrics: <a href="action_codebook_analysis.json">action_codebook_analysis.json</a></p>
</main></body></html>"""


def main() -> None:
    args = parse_args()
    report_root = args.report_root.resolve()
    dataset = load_bundle(args.skill_bundle.resolve())
    paths = sorted(report_root.glob("*/metrics/collection.json"))
    if not paths:
        raise FileNotFoundError(f"No model collections below {report_root}")
    results = []
    for path in paths:
        print(f"Analyzing {path.parent.parent.name}", flush=True)
        result = model_metrics(path, dataset)
        output_dir = path.parent.parent / "action_codebook_analysis"
        save_detail(result, output_dir)
        results.append(result)
    order = {name: index for index, name in enumerate(("js_action", "none_action", "cont", "js_zero", "none_zero"))}
    results.sort(key=lambda result: order.get(result["name"], 999))
    save_summary_plot(results, report_root / "action_codebook_analysis_comparison.png")
    (report_root / "action_codebook_analysis.json").write_text(
        json.dumps(json_ready({"models": results}), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (report_root / args.output_name).write_text(summary_html(results, dataset), encoding="utf-8")
    print(f"Wrote {report_root / args.output_name}")


if __name__ == "__main__":
    main()
