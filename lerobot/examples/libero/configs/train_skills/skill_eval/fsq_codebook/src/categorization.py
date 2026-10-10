#!/usr/bin/env python3
"""Checkpoint-selectable FSQ categorization analysis for GT replay reports.

The analysis always uses the complete aligned training skill bundle.  The GT
replay task/episode selection is only a visual browser and never changes these
metrics.  Per-checkpoint results are cached beside each replay manifest so a
later report rebuild is normally a no-op.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pickletools
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.stats import rankdata


FORMAT = "fsq_gt_replay_categorization_v1"
FEATURE_FORMAT = "fsq_gt_replay_categorization_features_v1"
CACHE_NAME = "categorization_v1.json"
REPORT_NAME = "categorization.html"
DATA_NAME = "categorization.json"
NEIGHBOR_COUNT = 10
NEIGHBOR_QUERY_COUNT = 48
DIRECTION_LABELS = ("+x", "-x", "+y", "-y", "+z", "-z", "still")
DIRECTION_COLORS = ("#ef476f", "#9d2a46", "#06d6a0", "#038f6c", "#118ab2", "#07577a", "#8d99ae")

FEATURE_LABELS = {
    "start_x": "start x",
    "start_y": "start y",
    "start_z": "start z",
    "rv_start_x": "start rot x",
    "rv_start_y": "start rot y",
    "rv_start_z": "start rot z",
    "rel25_x": "relative 25% x",
    "rel25_y": "relative 25% y",
    "rel25_z": "relative 25% z",
    "rel50_x": "relative 50% x",
    "rel50_y": "relative 50% y",
    "rel50_z": "relative 50% z",
    "rel75_x": "relative 75% x",
    "rel75_y": "relative 75% y",
    "rel75_z": "relative 75% z",
    "disp_x": "displacement x",
    "disp_y": "displacement y",
    "disp_z": "displacement z",
    "net_xyz": "net XYZ distance",
    "path_xyz": "XYZ path length",
    "straightness": "path straightness",
    "rot_rel_x": "relative rotation x",
    "rot_rel_y": "relative rotation y",
    "rot_rel_z": "relative rotation z",
    "rot_net_angle": "net rotation angle",
    "rot_path_angle": "rotation path",
    "grip_mean": "state gripper mean",
    "grip_delta": "state gripper delta",
    "grip_range": "state gripper range",
    "grip_path": "state gripper path",
    "act_mean_x": "action mean x",
    "act_mean_y": "action mean y",
    "act_mean_z": "action mean z",
    "act_mean_rx": "action mean rx",
    "act_mean_ry": "action mean ry",
    "act_mean_rz": "action mean rz",
    "act_abs_x": "|action| mean x",
    "act_abs_y": "|action| mean y",
    "act_abs_z": "|action| mean z",
    "act_abs_rx": "|action| mean rx",
    "act_abs_ry": "|action| mean ry",
    "act_abs_rz": "|action| mean rz",
    "act_grip_mean": "action gripper mean",
    "act_grip_delta": "action gripper delta",
    "act_grip_transitions": "gripper transitions",
    "frames": "skill length",
    "skill_index": "skill index",
    "skill_order": "normalized skill order",
}

CORRELATION_FEATURES = (
    "start_x", "start_y", "start_z",
    "rv_start_x", "rv_start_y", "rv_start_z",
    "rel50_x", "rel50_y", "rel50_z",
    "disp_x", "disp_y", "disp_z",
    "path_xyz", "straightness",
    "rot_rel_x", "rot_rel_y", "rot_rel_z", "rot_path_angle",
    "act_mean_x", "act_mean_y", "act_mean_z",
    "act_mean_rx", "act_mean_ry", "act_mean_rz",
    "act_grip_mean", "act_grip_delta", "act_grip_transitions",
    "frames", "skill_order",
)

CODE_FEATURES = (
    "disp_x", "disp_y", "disp_z",
    "rel50_x", "rel50_y", "rel50_z",
    "path_xyz", "straightness",
    "rot_rel_x", "rot_rel_y", "rot_rel_z", "rot_path_angle",
    "act_grip_mean", "act_grip_delta", "frames",
)

MOTION_FEATURES = (
    "rel25_x", "rel25_y", "rel25_z",
    "rel50_x", "rel50_y", "rel50_z",
    "rel75_x", "rel75_y", "rel75_z",
    "disp_x", "disp_y", "disp_z",
    "net_xyz", "path_xyz", "straightness",
    "rot_rel_x", "rot_rel_y", "rot_rel_z",
    "rot_net_angle", "rot_path_angle",
)

FEATURE_GROUPS = {
    "Absolute start pose": (
        "start_x", "start_y", "start_z", "rv_start_x", "rv_start_y", "rv_start_z",
    ),
    "Relative translation": (
        "rel25_x", "rel25_y", "rel25_z", "rel50_x", "rel50_y", "rel50_z",
        "rel75_x", "rel75_y", "rel75_z", "disp_x", "disp_y", "disp_z",
        "net_xyz", "path_xyz", "straightness",
    ),
    "Relative rotation": (
        "rot_rel_x", "rot_rel_y", "rot_rel_z", "rot_net_angle", "rot_path_angle",
    ),
    "Gripper": (
        "grip_mean", "grip_delta", "grip_range", "grip_path",
        "act_grip_mean", "act_grip_delta", "act_grip_transitions",
    ),
    "Action delta": (
        "act_mean_x", "act_mean_y", "act_mean_z",
        "act_mean_rx", "act_mean_ry", "act_mean_rz",
        "act_abs_x", "act_abs_y", "act_abs_z",
        "act_abs_rx", "act_abs_ry", "act_abs_rz",
    ),
    "Length / order": ("frames", "skill_index", "skill_order"),
}


class CategorizationUnavailable(RuntimeError):
    """Expected missing artifact that should not fail the replay evaluation."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _tag_key(tag: str) -> tuple[int, int, str]:
    text = str(tag)
    if text.startswith("epoch") and text[5:].isdigit():
        return (0, int(text[5:]), "")
    return (1, 0, text)


def _interp(values: np.ndarray, fraction: float) -> np.ndarray:
    position = fraction * max(len(values) - 1, 0)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return (1.0 - weight) * values[lower] + weight * values[upper]


def _skill_order(episodes: np.ndarray, indices: np.ndarray) -> np.ndarray:
    result = np.zeros(len(episodes), dtype=np.float64)
    grouped: dict[int, list[int]] = defaultdict(list)
    for row, episode in enumerate(episodes):
        grouped[int(episode)].append(row)
    for members in grouped.values():
        members.sort(key=lambda row: (int(indices[row]), row))
        denominator = max(len(members) - 1, 1)
        for rank, row in enumerate(members):
            result[row] = rank / denominator
    return result


def _feature_row(states: np.ndarray, actions: np.ndarray, order: float, skill_index: int) -> dict[str, float]:
    position = states[:, :3].astype(np.float64)
    rotvec = states[:, 3:6].astype(np.float64)
    state_gripper = states[:, 6:8].astype(np.float64).mean(axis=1)
    action = actions.astype(np.float64)
    row: dict[str, float] = {}
    for axis, value in zip("xyz", position[0], strict=True):
        row[f"start_{axis}"] = float(value)
    for axis, value in zip("xyz", rotvec[0], strict=True):
        row[f"rv_start_{axis}"] = float(value)
    for fraction in (0.25, 0.5, 0.75):
        relative = _interp(position, fraction) - position[0]
        for axis, value in zip("xyz", relative, strict=True):
            row[f"rel{int(fraction * 100)}_{axis}"] = float(value)
    displacement = position[-1] - position[0]
    for axis, value in zip("xyz", displacement, strict=True):
        row[f"disp_{axis}"] = float(value)
    path = float(np.linalg.norm(np.diff(position, axis=0), axis=1).sum())
    net = float(np.linalg.norm(displacement))
    row["net_xyz"] = net
    row["path_xyz"] = path
    row["straightness"] = net / max(path, 1e-12)
    rotations = Rotation.from_rotvec(rotvec)
    relative_rotation = (rotations[0].inv() * rotations[-1]).as_rotvec()
    for axis, value in zip("xyz", relative_rotation, strict=True):
        row[f"rot_rel_{axis}"] = float(value)
    row["rot_net_angle"] = float(np.linalg.norm(relative_rotation))
    if len(rotations) > 1:
        increments = (rotations[:-1].inv() * rotations[1:]).as_rotvec()
        row["rot_path_angle"] = float(np.linalg.norm(increments, axis=1).sum())
    else:
        row["rot_path_angle"] = 0.0
    row["grip_mean"] = float(state_gripper.mean())
    row["grip_delta"] = float(state_gripper[-1] - state_gripper[0])
    row["grip_range"] = float(np.ptp(state_gripper))
    row["grip_path"] = float(np.abs(np.diff(state_gripper)).sum())
    axes = ("x", "y", "z", "rx", "ry", "rz")
    for column, axis in enumerate(axes):
        row[f"act_mean_{axis}"] = float(action[:, column].mean())
        row[f"act_abs_{axis}"] = float(np.abs(action[:, column]).mean())
    action_gripper = action[:, 6]
    row["act_grip_mean"] = float(action_gripper.mean())
    row["act_grip_delta"] = float(action_gripper[-1] - action_gripper[0])
    row["act_grip_transitions"] = float(np.count_nonzero(np.diff(np.signbit(action_gripper))))
    row["frames"] = float(len(states))
    row["skill_index"] = float(skill_index)
    row["skill_order"] = float(order)
    return row


def _bundle_fingerprint(bundle_path: Path) -> str:
    with np.load(bundle_path, allow_pickle=False) as bundle:
        if "fingerprint" in bundle:
            return str(bundle["fingerprint"].item())
    stat = bundle_path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _feature_cache_path(output_dir: Path) -> Path:
    return output_dir / "metrics" / "categorization_features_v1.npz"


def _load_or_build_features(bundle_path: Path, output_dir: Path) -> dict[str, np.ndarray]:
    fingerprint = _bundle_fingerprint(bundle_path)
    cache_path = _feature_cache_path(output_dir)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cache:
            if (
                str(cache["format"].item()) == FEATURE_FORMAT
                and str(cache["fingerprint"].item()) == fingerprint
            ):
                return {key: cache[key].copy() for key in cache.files}
    with np.load(bundle_path, allow_pickle=False) as bundle:
        states_cat = bundle["states_cat"]
        actions_cat = bundle["actions_cat"]
        state_lengths = bundle["states_len"].astype(np.int64)
        action_lengths = bundle["actions_len"].astype(np.int64)
        episodes = bundle["meta_episode_id"].astype(np.int64)
        tasks = bundle["meta_task_id"].astype(np.int64)
        skill_indices = bundle["meta_skill_index"].astype(np.int64)
        frame_starts = bundle["meta_frame_start"].astype(np.int64)
        frame_ends = bundle["meta_frame_end"].astype(np.int64)
        state_starts = np.concatenate(([0], np.cumsum(state_lengths[:-1])))
        action_starts = np.concatenate(([0], np.cumsum(action_lengths[:-1])))
        orders = _skill_order(episodes, skill_indices)
        rows = []
        for state_start, state_length, action_start, action_length, order, skill_index in zip(
            state_starts, state_lengths, action_starts, action_lengths, orders, skill_indices, strict=True
        ):
            rows.append(
                _feature_row(
                    states_cat[state_start : state_start + state_length],
                    actions_cat[action_start : action_start + action_length],
                    float(order),
                    int(skill_index),
                )
            )
    names = np.asarray(list(rows[0]), dtype="U40")
    matrix = np.asarray([[row[name] for name in names] for row in rows], dtype=np.float32)
    scale = matrix.astype(np.float64).std(axis=0)
    standardized = (
        (matrix.astype(np.float64) - matrix.astype(np.float64).mean(axis=0))
        / np.where(scale > 1e-9, scale, 1.0)
    ).astype(np.float32)
    name_to_column = {str(name): index for index, name in enumerate(names)}
    motion_columns = [name_to_column[name] for name in MOTION_FEATURES]
    motion = standardized[:, motion_columns]
    displacement = np.column_stack(
        [matrix[:, name_to_column[f"disp_{axis}"]] for axis in "xyz"]
    ).astype(np.float32)
    norms = np.linalg.norm(displacement, axis=1)
    directions = np.full(len(matrix), 6, dtype=np.int16)
    moving = norms >= 0.01
    dominant = np.argmax(np.abs(displacement[moving]), axis=1)
    negative = displacement[moving, dominant] < 0
    directions[moving] = (2 * dominant + negative.astype(np.int64)).astype(np.int16)
    grip_mean = matrix[:, name_to_column["act_grip_mean"]]
    grip_regime = np.where(grip_mean > 0.5, 2, np.where(grip_mean < -0.5, 0, 1)).astype(np.int16)
    query_count = min(max(NEIGHBOR_QUERY_COUNT, NEIGHBOR_COUNT + 1), len(motion))
    _, raw_neighbors = cKDTree(motion.astype(np.float64)).query(motion, k=query_count)
    if raw_neighbors.ndim == 1:
        raw_neighbors = raw_neighbors[:, None]
    neighbors = np.full((len(motion), NEIGHBOR_COUNT), -1, dtype=np.int32)
    for row, candidates in enumerate(raw_neighbors):
        selected = [
            int(candidate)
            for candidate in np.atleast_1d(candidates)
            if int(candidate) != row and episodes[int(candidate)] != episodes[row]
        ][:NEIGHBOR_COUNT]
        neighbors[row, : len(selected)] = selected
    payload: dict[str, np.ndarray] = {
        "format": np.asarray(FEATURE_FORMAT),
        "fingerprint": np.asarray(fingerprint),
        "feature_names": names,
        "matrix": matrix,
        "standardized": standardized,
        "motion": motion,
        "displacement": displacement,
        "direction": directions,
        "grip_regime": grip_regime,
        "episodes": episodes,
        "tasks": tasks,
        "skill_indices": skill_indices,
        "frame_starts": frame_starts,
        "frame_ends": frame_ends,
        "neighbors": neighbors,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(f".tmp{os.getpid()}.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(cache_path)
    return payload


def _normalized_mutual_info(first: np.ndarray, second: np.ndarray) -> float:
    _, first_inverse = np.unique(first, return_inverse=True)
    _, second_inverse = np.unique(second, return_inverse=True)
    rows = int(first_inverse.max()) + 1
    columns = int(second_inverse.max()) + 1
    contingency = np.bincount(
        first_inverse * columns + second_inverse, minlength=rows * columns
    ).reshape(rows, columns).astype(np.float64)
    total = contingency.sum()
    joint = contingency / max(total, 1.0)
    row_probability = joint.sum(axis=1)
    column_probability = joint.sum(axis=0)
    expected = row_probability[:, None] * column_probability[None, :]
    mask = joint > 0
    mutual_information = float(np.sum(joint[mask] * np.log(joint[mask] / expected[mask])))
    first_entropy = float(-np.sum(row_probability[row_probability > 0] * np.log(row_probability[row_probability > 0])))
    second_entropy = float(-np.sum(column_probability[column_probability > 0] * np.log(column_probability[column_probability > 0])))
    denominator = first_entropy + second_entropy
    return 0.0 if denominator <= 1e-12 else 2.0 * mutual_information / denominator


def _adjusted_rand(first: np.ndarray, second: np.ndarray) -> float:
    _, first_inverse = np.unique(first, return_inverse=True)
    _, second_inverse = np.unique(second, return_inverse=True)
    columns = int(second_inverse.max()) + 1
    contingency = np.bincount(
        first_inverse * columns + second_inverse,
        minlength=(int(first_inverse.max()) + 1) * columns,
    ).reshape(-1, columns).astype(np.int64)
    choose2 = lambda values: values * (values - 1) / 2.0
    sum_cells = float(choose2(contingency).sum())
    sum_rows = float(choose2(contingency.sum(axis=1)).sum())
    sum_columns = float(choose2(contingency.sum(axis=0)).sum())
    total_pairs = float(choose2(np.asarray([len(first)]))[0])
    if total_pairs <= 0:
        return 0.0
    expected = sum_rows * sum_columns / total_pairs
    maximum = 0.5 * (sum_rows + sum_columns)
    denominator = maximum - expected
    return 0.0 if abs(denominator) <= 1e-12 else (sum_cells - expected) / denominator


def _entropy_effective(counts: np.ndarray) -> float:
    probability = counts[counts > 0] / max(counts.sum(), 1)
    return float(np.exp(-(probability * np.log(probability)).sum()))


def _direction_coherence(tokens: np.ndarray, displacement: np.ndarray) -> float:
    norms = np.linalg.norm(displacement, axis=1)
    moving = norms >= 0.01
    unit = displacement[moving] / norms[moving, None]
    moving_tokens = tokens[moving]
    total = 0.0
    for token in np.unique(moving_tokens):
        members = unit[moving_tokens == token]
        total += len(members) * float(np.linalg.norm(members.mean(axis=0)))
    return total / max(len(unit), 1)


def _adjacent_indices(episodes: np.ndarray, skill_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for row, episode in enumerate(episodes):
        grouped[int(episode)].append(row)
    left: list[int] = []
    right: list[int] = []
    for members in grouped.values():
        members.sort(key=lambda row: (int(skill_indices[row]), row))
        for first, second in zip(members[:-1], members[1:], strict=True):
            if int(skill_indices[second]) == int(skill_indices[first]) + 1:
                left.append(first)
                right.append(second)
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def _motion_neighbor_consistency(tokens: np.ndarray, neighbors: np.ndarray) -> float:
    valid = neighbors >= 0
    if not valid.any():
        return 0.0
    observed = float((tokens[neighbors[valid]] == np.repeat(tokens, valid.sum(axis=1))).mean())
    counts = np.bincount(tokens)
    probability = counts / max(counts.sum(), 1)
    chance = float(np.square(probability).sum())
    return 0.0 if chance >= 1.0 - 1e-12 else (observed - chance) / (1.0 - chance)


def _motion_cohesion(tokens: np.ndarray, motion: np.ndarray) -> float:
    centroids = np.empty_like(motion)
    for token in np.unique(tokens):
        mask = tokens == token
        centroids[mask] = motion[mask].mean(axis=0)
    return float(1.0 - np.mean(np.square(motion - centroids)))


def _balanced_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    recalls = [float((predicted[actual == code] == code).mean()) for code in np.unique(actual)]
    return float(np.mean(recalls)) if recalls else 0.0


def _split_episodes(episodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(episodes)
    generator = np.random.default_rng(20260828)
    shuffled = unique.copy()
    generator.shuffle(shuffled)
    test_count = max(1, int(round(0.25 * len(shuffled))))
    test_episodes = set(int(value) for value in shuffled[:test_count])
    test = np.asarray([int(value) in test_episodes for value in episodes])
    return ~test, test


def _group_predictability(
    tokens: np.ndarray,
    matrix: np.ndarray,
    names: list[str],
    episodes: np.ndarray,
) -> dict[str, float]:
    lookup = {name: index for index, name in enumerate(names)}
    train, test = _split_episodes(episodes)
    result: dict[str, float] = {}
    for group, feature_names in FEATURE_GROUPS.items():
        columns = [lookup[name] for name in feature_names if name in lookup]
        values = matrix[:, columns].astype(np.float64)
        mean = values[train].mean(axis=0)
        scale = values[train].std(axis=0)
        values = (values - mean) / np.where(scale > 1e-9, scale, 1.0)
        classes = np.unique(tokens[train])
        centroids = np.vstack([values[train & (tokens == code)].mean(axis=0) for code in classes])
        test_values = values[test]
        distances = (
            np.square(test_values).sum(axis=1, keepdims=True)
            + np.square(centroids).sum(axis=1)[None, :]
            - 2.0 * test_values @ centroids.T
        )
        predicted = classes[np.argmin(distances, axis=1)]
        result[group] = _balanced_accuracy(tokens[test], predicted)
    return result


def _rank_standardized(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values, method="average")
    scale = ranks.std()
    return (ranks - ranks.mean()) / (scale if scale > 1e-12 else 1.0)


def _axis_associations(
    latents: np.ndarray,
    matrix: np.ndarray,
    names: list[str],
) -> tuple[list[str], list[list[float]], list[list[float]]]:
    lookup = {name: index for index, name in enumerate(names)}
    selected = [name for name in CORRELATION_FEATURES if name in lookup]
    values = matrix[:, [lookup[name] for name in selected]].astype(np.float64)
    feature_ranks = np.column_stack([_rank_standardized(values[:, column]) for column in range(values.shape[1])])
    correlations: list[list[float]] = []
    strengths: list[list[float]] = []
    centered = values - values.mean(axis=0)
    total = np.square(centered).sum(axis=0)
    for axis in range(latents.shape[1]):
        coordinate = latents[:, axis]
        axis_rank = _rank_standardized(coordinate)
        correlations.append((axis_rank @ feature_ranks / len(values)).tolist())
        between = np.zeros(values.shape[1], dtype=np.float64)
        for level in np.unique(coordinate):
            members = values[coordinate == level]
            between += len(members) * np.square(members.mean(axis=0) - values.mean(axis=0))
        strengths.append(np.divide(between, total, out=np.zeros_like(between), where=total > 1e-12).tolist())
    return selected, correlations, strengths


def _code_feature_means(
    tokens: np.ndarray,
    standardized: np.ndarray,
    names: list[str],
    codebook_size: int,
) -> tuple[list[str], list[list[float] | None]]:
    lookup = {name: index for index, name in enumerate(names)}
    selected = [name for name in CODE_FEATURES if name in lookup]
    values = standardized[:, [lookup[name] for name in selected]]
    rows: list[list[float] | None] = []
    for code in range(codebook_size):
        members = values[tokens == code]
        rows.append(None if not len(members) else members.mean(axis=0).tolist())
    return selected, rows


def _direction_composition(tokens: np.ndarray, directions: np.ndarray, codebook_size: int) -> list[list[float] | None]:
    rows: list[list[float] | None] = []
    for code in range(codebook_size):
        members = directions[tokens == code]
        if not len(members):
            rows.append(None)
        else:
            rows.append((np.bincount(members, minlength=len(DIRECTION_LABELS)) / len(members)).tolist())
    return rows


def _checkpoint_scalars(path: Path) -> dict[str, float | None]:
    result: dict[str, float | None] = {"validation_total": None, "validation_reconstruction": None}
    if not path.is_file():
        return result
    try:
        with zipfile.ZipFile(path) as archive:
            pickle_name = next(name for name in archive.namelist() if name.endswith("data.pkl"))
            operations = list(pickletools.genops(archive.read(pickle_name)))
    except (OSError, StopIteration, zipfile.BadZipFile):
        return result
    targets = {"val_loss": "validation_total", "val_select": "validation_reconstruction"}
    for index, (_, argument, _) in enumerate(operations):
        if argument not in targets:
            continue
        for operation, value, _ in operations[index + 1 : index + 5]:
            if operation.name in {"BINFLOAT", "BININT", "BININT1", "BININT2", "LONG1", "LONG4"}:
                result[targets[str(argument)]] = float(value)
                break
    return result


def _source_signature(
    latents_path: Path,
    bundle_fingerprint: str,
    boundary_sweep_path: Path | None = None,
    terminator_path: Path | None = None,
) -> dict[str, Any]:
    stat = latents_path.stat()
    result = {
        "format": FORMAT,
        "bundle_fingerprint": bundle_fingerprint,
        "latents_path": str(latents_path.resolve()),
        "latents_size": stat.st_size,
        "latents_mtime_ns": stat.st_mtime_ns,
    }
    if boundary_sweep_path is not None and boundary_sweep_path.is_file():
        sweep_stat = boundary_sweep_path.stat()
        result.update(
            {
                "boundary_sweep_path": str(boundary_sweep_path.resolve()),
                "boundary_sweep_size": sweep_stat.st_size,
                "boundary_sweep_mtime_ns": sweep_stat.st_mtime_ns,
            }
        )
    if terminator_path is not None and terminator_path.is_file():
        term_stat = terminator_path.stat()
        result.update(
            {
                "terminator_path": str(terminator_path.resolve()),
                "terminator_size": term_stat.st_size,
                "terminator_mtime_ns": term_stat.st_mtime_ns,
            }
        )
    return result


def _json_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _boundary_neighbors(features: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Exact previous/next skill rows, matched at a shared episode boundary."""
    episodes = features["episodes"].astype(np.int64)
    starts = features["frame_starts"].astype(np.int64)
    ends = features["frame_ends"].astype(np.int64)
    skill_indices = features["skill_indices"].astype(np.int64)
    by_start = {
        (int(episode), int(start), int(skill)): row
        for row, (episode, start, skill) in enumerate(
            zip(episodes, starts, skill_indices, strict=True)
        )
    }
    by_end = {
        (int(episode), int(end), int(skill)): row
        for row, (episode, end, skill) in enumerate(
            zip(episodes, ends, skill_indices, strict=True)
        )
    }
    previous = np.full(len(episodes), -1, dtype=np.int64)
    following = np.full(len(episodes), -1, dtype=np.int64)
    for row, (episode, start, end, skill) in enumerate(
        zip(episodes, starts, ends, skill_indices, strict=True)
    ):
        previous[row] = by_end.get((int(episode), int(start), int(skill) - 1), -1)
        following[row] = by_start.get((int(episode), int(end), int(skill) + 1), -1)
    return previous, following


def _boundary_sweep_metrics(
    path: Path,
    clean_tokens: np.ndarray,
    clean_latents: np.ndarray,
    features: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Summarize deterministic boundary shifts encoded by one FSQ checkpoint."""
    with np.load(path, allow_pickle=False) as sweep:
        if str(sweep["format"].item()) != "fsq_boundary_jitter_sweep_v1":
            raise ValueError(f"unsupported boundary sweep artifact: {path}")
        directions = [str(value) for value in sweep["directions"]]
        offsets = sweep["offsets"].astype(np.int64)
        valid = sweep["valid"].astype(bool)
        tokens = sweep["tokens"].astype(np.int64)
        latents = sweep["latents"].astype(np.float64)
        checks = {
            "episode_id": "episodes",
            "task_id": "tasks",
            "skill_index": "skill_indices",
            "frame_start": "frame_starts",
            "frame_end": "frame_ends",
        }
        for sweep_key, feature_key in checks.items():
            if not np.array_equal(sweep[sweep_key], features[feature_key]):
                raise ValueError(f"boundary-sweep/bundle alignment mismatch: {path}/{sweep_key}")
    expected = (len(directions), len(offsets), len(clean_tokens))
    if valid.shape != expected or tokens.shape != expected or latents.shape[:3] != expected:
        raise ValueError(f"invalid boundary sweep shapes in {path}")
    previous, following = _boundary_neighbors(features)

    codebook_size = int(clean_tokens.max()) + 1

    def summarize(scope_mask: np.ndarray) -> dict[str, Any]:
        metric_rows = {
            "code_retention": [],
            "latent_l1": [],
            "adjacent_code_attraction": [],
            "valid_count": [],
            "adjacent_valid_count": [],
        }
        code_rows = {
            "code_retention": [],
            "latent_l1": [],
            "adjacent_code_attraction": [],
            "valid_count": [],
        }
        for direction_index, direction in enumerate(directions):
            per_direction = {key: [] for key in metric_rows}
            per_code_direction = {key: [] for key in code_rows}
            neighbor = (
                previous
                if direction == "early_start"
                else following
                if direction == "late_end"
                else None
            )
            for offset_index in range(len(offsets)):
                mask = scope_mask & valid[direction_index, offset_index]
                rows = np.flatnonzero(mask)
                per_direction["valid_count"].append(int(len(rows)))
                per_direction["code_retention"].append(
                    _json_number(float((tokens[direction_index, offset_index, rows] == clean_tokens[rows]).mean()))
                    if len(rows)
                    else None
                )
                per_direction["latent_l1"].append(
                    _json_number(float(np.abs(latents[direction_index, offset_index, rows] - clean_latents[rows]).mean()))
                    if len(rows)
                    else None
                )
                adjacent_rows = np.asarray([], dtype=np.int64)
                if neighbor is not None:
                    adjacent_rows = rows[neighbor[rows] >= 0]
                    adjacent_rows = adjacent_rows[
                        clean_tokens[adjacent_rows] != clean_tokens[neighbor[adjacent_rows]]
                    ]
                per_direction["adjacent_valid_count"].append(int(len(adjacent_rows)))
                per_direction["adjacent_code_attraction"].append(
                    _json_number(float((tokens[direction_index, offset_index, adjacent_rows] == clean_tokens[neighbor[adjacent_rows]]).mean()))
                    if len(adjacent_rows)
                    else None
                )
                code_retention = []
                code_drift = []
                code_attraction = []
                code_counts = []
                for code in range(codebook_size):
                    code_members = rows[clean_tokens[rows] == code]
                    code_counts.append(int(len(code_members)))
                    code_retention.append(
                        _json_number(float((tokens[direction_index, offset_index, code_members] == code).mean()))
                        if len(code_members)
                        else None
                    )
                    code_drift.append(
                        _json_number(float(np.abs(latents[direction_index, offset_index, code_members] - clean_latents[code_members]).mean()))
                        if len(code_members)
                        else None
                    )
                    code_adjacent = adjacent_rows[clean_tokens[adjacent_rows] == code]
                    code_attraction.append(
                        _json_number(float((tokens[direction_index, offset_index, code_adjacent] == clean_tokens[neighbor[code_adjacent]]).mean()))
                        if len(code_adjacent)
                        else None
                    )
                per_code_direction["code_retention"].append(code_retention)
                per_code_direction["latent_l1"].append(code_drift)
                per_code_direction["adjacent_code_attraction"].append(code_attraction)
                per_code_direction["valid_count"].append(code_counts)
            for key in metric_rows:
                metric_rows[key].append(per_direction[key])
            for key in code_rows:
                code_rows[key].append(per_code_direction[key])
        return {"metrics": metric_rows, "codes": code_rows}

    tasks = features["tasks"].astype(np.int64)
    scopes = {"all": summarize(np.ones(len(clean_tokens), dtype=bool))}
    for task in np.unique(tasks):
        scopes[str(int(task))] = summarize(tasks == task)
    return {
        "format": "fsq_boundary_jitter_metrics_v1",
        "source_path": str(path.resolve()),
        "directions": directions,
        "offsets": offsets.astype(int).tolist(),
        **scopes["all"],
        "scopes": scopes,
    }


def _terminator_metrics(path: Path, codebook_size: int) -> dict[str, Any]:
    """Task/code-filterable GT-boundary termination traces and timing summaries."""
    with np.load(path, allow_pickle=False) as artifact:
        if str(artifact["format"].item()) != "fsq_terminator_diagnostics_v1":
            raise ValueError(f"unsupported terminator diagnostic artifact: {path}")
        lengths = artifact["probabilities_len"].astype(np.int64)
        starts = np.concatenate(([0], np.cumsum(lengths[:-1])))
        cat = artifact["probabilities_cat"].astype(np.float64)
        probabilities = [cat[start : start + length] for start, length in zip(starts, lengths, strict=True)]
        boundaries = artifact["boundary_index"].astype(np.int64)
        tokens = artifact["tokens"].astype(np.int64)
        tasks = artifact["task_id"].astype(np.int64)
        episodes = artifact["episode_id"].astype(np.int64)
        skill_indices = artifact["skill_index"].astype(np.int64)
        frame_starts = artifact["frame_start"].astype(np.int64)
        frame_ends = artifact["frame_end"].astype(np.int64)
        threshold = float(artifact["threshold"].item())
        sigma = float(artifact["end_target_sigma"].item())
    if not (
        len(probabilities)
        == len(boundaries)
        == len(tokens)
        == len(tasks)
    ):
        raise ValueError(f"invalid terminator diagnostic shapes in {path}")
    min_offset = max(-30, min(-int(boundary) for boundary in boundaries))
    max_offset = max(len(row) - 1 - int(boundary) for row, boundary in zip(probabilities, boundaries, strict=True))
    offsets = np.arange(min_offset, max_offset + 1, dtype=np.int64)

    def summarize(rows: np.ndarray) -> dict[str, Any]:
        curves = [[] for _ in offsets]
        boundary_values = []
        timing = []
        missed = 0
        for row in rows:
            values = probabilities[int(row)]
            boundary = int(boundaries[int(row)])
            boundary_values.append(float(values[boundary]))
            for column, offset in enumerate(offsets):
                index = boundary + int(offset)
                if 0 <= index < len(values):
                    curves[column].append(float(values[index]))
            hits = np.flatnonzero(values >= threshold)
            if len(hits):
                timing.append(int(hits[0]) - boundary)
            else:
                missed += 1
        mean = [float(np.mean(values)) if values else None for values in curves]
        p10 = [float(np.quantile(values, 0.1)) if values else None for values in curves]
        p90 = [float(np.quantile(values, 0.9)) if values else None for values in curves]
        return {
            "sample_count": int(len(rows)),
            "offsets": offsets.astype(int).tolist(),
            "mean_probability": mean,
            "p10_probability": p10,
            "p90_probability": p90,
            "valid_count": [len(values) for values in curves],
            "boundary_probability": _json_number(float(np.mean(boundary_values))) if boundary_values else None,
            "missed_fraction": float(missed / len(rows)) if len(rows) else None,
            "timing_mean": _json_number(float(np.mean(timing))) if timing else None,
            "timing_abs": _json_number(float(np.mean(np.abs(timing)))) if timing else None,
            "early_fraction": float(np.mean(np.asarray(timing) < 0)) if timing else None,
            "late_fraction": float(np.mean(np.asarray(timing) > 0)) if timing else None,
        }

    scopes: dict[str, Any] = {}
    for scope, scope_rows in {
        "all": np.arange(len(tokens), dtype=np.int64),
        **{
            str(int(task)): np.flatnonzero(tasks == task)
            for task in np.unique(tasks)
        },
    }.items():
        code_summaries = []
        for code in range(codebook_size):
            code_summaries.append(summarize(scope_rows[tokens[scope_rows] == code]))
        scopes[scope] = {"overall": summarize(scope_rows), "codes": code_summaries}
    samples = []
    for row, values in enumerate(probabilities):
        samples.append(
            {
                "token": int(tokens[row]),
                "task_id": int(tasks[row]),
                "episode_id": int(episodes[row]),
                "skill_index": int(skill_indices[row]),
                "frame_start": int(frame_starts[row]),
                "frame_end": int(frame_ends[row]),
                "boundary_index": int(boundaries[row]),
                "probabilities": values.astype(float).tolist(),
            }
        )
    return {
        "format": "fsq_terminator_metrics_v1",
        "source_path": str(path.resolve()),
        "threshold": threshold,
        "end_target_sigma": sigma,
        "scopes": scopes,
        "samples": samples,
    }


def _terminator_sample_key(sample: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(sample["task_id"]),
        int(sample["episode_id"]),
        int(sample["frame_start"]),
        int(sample["frame_end"]),
    )


def _attach_terminator_frame_pairs(
    payload: dict[str, Any],
    *,
    output_dir: Path,
    metas: list[dict[str, Any]],
) -> None:
    """Export one top-view start/end pair for every diagnostic sample.

    The terminator artifact already carries exact episode-relative frame bounds,
    so this is a report-only CPU/cache pass and never reruns model inference.
    """
    samples_by_key: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for model in payload.get("models", []):
        for checkpoint in model.get("checkpoints", []):
            terminator = checkpoint.get("terminator")
            if not terminator:
                continue
            for sample in terminator.get("samples", []):
                samples_by_key[_terminator_sample_key(sample)].append(sample)
    if not samples_by_key:
        return

    locations = {
        (
            str(meta.get("dataset_root_name") or "dataset_filtered"),
            str(meta.get("target_dataset") or ""),
        )
        for meta in metas
        if str(meta.get("target_dataset") or "")
    }
    if len(locations) != 1:
        print(
            "categorization analysis: terminator images unavailable because "
            f"dataset locations disagree: {sorted(locations)}"
        )
        return
    dataset_root_name, dataset_name = next(iter(locations))
    repository = Path(__file__).resolve().parents[8]
    raw_dataset_dir = repository / dataset_root_name / dataset_name
    if not raw_dataset_dir.is_dir():
        print(
            "categorization analysis: terminator images unavailable; "
            f"raw dataset is missing: {raw_dataset_dir}"
        )
        return

    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=f"local/{raw_dataset_dir.name}",
        root=raw_dataset_dir,
        video_keys_to_load=["observation.images.image"],
    )
    reader = dataset._ensure_reader()  # noqa: SLF001
    if reader.hf_dataset is None:
        reader.load_and_activate()
    cache = None
    cache_candidates = sorted(
        repository.glob(
            f".cache/fsq_frame_cache/*/{raw_dataset_dir.name}/rgb_zstd_v2/*/_SUCCESS"
        )
    )
    if cache_candidates:
        from fsq_frame_cache import RGBFrameCache

        cache = RGBFrameCache(cache_candidates[-1].parent, raw_dataset_dir)
        print(
            "categorization analysis: terminator thumbnails use frame cache "
            f"{cache_candidates[-1].parent}"
        )

    asset_dir = output_dir / "terminator_frames"
    asset_dir.mkdir(parents=True, exist_ok=True)
    camera_key = "observation.images.image"

    def save_frame(frame, path: Path) -> None:
        array = frame.detach().cpu().permute(1, 2, 0).numpy()
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating) and float(array.max()) <= 1.0:
                array = np.rint(array * 255.0)
            array = np.clip(array, 0, 255).astype(np.uint8)
        temporary = path.with_name(f".{path.stem}.tmp{os.getpid()}{path.suffix}")
        Image.fromarray(array).save(temporary, format="JPEG", quality=88, optimize=True)
        temporary.replace(path)

    exported = 0
    for key, copies in sorted(samples_by_key.items()):
        task_id, episode_id, frame_start, frame_end = key
        final_frame = max(frame_start, frame_end - 1)
        stem = f"task{task_id}_episode{episode_id}_frames{frame_start}-{frame_end}"
        start_path = asset_dir / f"{stem}_start.jpg"
        end_path = asset_dir / f"{stem}_end.jpg"
        try:
            if not start_path.is_file() or not end_path.is_file():
                episode = reader._meta.episodes[episode_id]  # noqa: SLF001
                episode_start = int(episode["dataset_from_index"])
                indices = [episode_start + frame_start, episode_start + final_frame]
                rows = reader.hf_dataset[indices]
                timestamps = [
                    float(value.item()) if hasattr(value, "item") else float(value)
                    for value in rows["timestamp"]
                ]
                video_path = reader.root / reader._meta.get_video_file_path(  # noqa: SLF001
                    episode_id, camera_key
                )
                from_timestamp = float(episode[f"videos/{camera_key}/from_timestamp"])
                queries = [from_timestamp + timestamp for timestamp in timestamps]
                if cache is not None:
                    frames = cache.get_frames(
                        video_path, queries, reader._tolerance_s  # noqa: SLF001
                    )
                else:
                    from lerobot.datasets.video_utils import decode_video_frames

                    frames = decode_video_frames(
                        video_path,
                        queries,
                        reader._tolerance_s,  # noqa: SLF001
                        reader._video_backend,  # noqa: SLF001
                        decoder_num_threads=1,
                    )
                if not start_path.is_file():
                    save_frame(frames[0], start_path)
                if not end_path.is_file():
                    save_frame(frames[1], end_path)
                exported += 1
            relative_start = start_path.relative_to(output_dir).as_posix()
            relative_end = end_path.relative_to(output_dir).as_posix()
            for sample in copies:
                sample["start_image"] = relative_start
                sample["end_image"] = relative_end
                sample["end_image_frame"] = final_frame
        except Exception as error:  # Keep the numeric report usable if one video is damaged.
            print(
                "categorization analysis: could not export terminator frame pair "
                f"{stem}: {error}"
            )
    print(
        "categorization analysis: terminator frame pairs ready "
        f"({len(samples_by_key)} unique, {exported} newly exported)"
    )


def _checkpoint_metrics(
    manifest_path: Path,
    manifest: dict[str, Any],
    features: dict[str, np.ndarray],
    bundle_fingerprint: str,
) -> dict[str, Any]:
    signature = manifest.get("signature") or {}
    latents_path = Path(str(signature.get("latents_path") or ""))
    if not latents_path.is_file():
        raise CategorizationUnavailable(f"latent artifact is missing: {latents_path}")
    raw_sweep_path = str(manifest.get("boundary_sweep_path") or "").strip()
    boundary_sweep_path = Path(raw_sweep_path) if raw_sweep_path else None
    raw_terminator_path = str(manifest.get("terminator_diagnostics_path") or "").strip()
    terminator_path = Path(raw_terminator_path) if raw_terminator_path else None
    source = _source_signature(
        latents_path, bundle_fingerprint, boundary_sweep_path, terminator_path
    )
    cache_path = manifest_path.parent / CACHE_NAME
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            cached = {}
        if cached.get("source") == source:
            return cached
    with np.load(latents_path, allow_pickle=False) as latent:
        tokens = latent["tokens"].astype(np.int64)
        quantized = latent["latents"].astype(np.float64)
        checks = {
            "episode_id": "episodes",
            "task_id": "tasks",
            "skill_index": "skill_indices",
            "frame_start": "frame_starts",
            "frame_end": "frame_ends",
        }
        for latent_key, feature_key in checks.items():
            if latent_key in latent and not np.array_equal(latent[latent_key], features[feature_key]):
                raise ValueError(f"latent/bundle alignment mismatch: {latents_path}/{latent_key}")
    levels = [int(value) for value in manifest["levels"]]
    codebook_size = int(np.prod(levels))
    counts = np.bincount(tokens, minlength=codebook_size)
    displacement = features["displacement"].astype(np.float64)
    episodes = features["episodes"].astype(np.int64)
    skill_indices = features["skill_indices"].astype(np.int64)
    left, right = _adjacent_indices(episodes, skill_indices)
    norms = np.linalg.norm(displacement, axis=1)
    valid = (norms[left] >= 0.01) & (norms[right] >= 0.01)
    cosine = np.zeros(len(left), dtype=np.float64)
    cosine[valid] = np.sum(displacement[left[valid]] * displacement[right[valid]], axis=1) / (
        norms[left[valid]] * norms[right[valid]]
    )
    opposite = valid & (cosine < 0.0)
    same = tokens[left] == tokens[right]
    names = [str(value) for value in features["feature_names"]]
    correlation_features, correlations, strengths = _axis_associations(
        quantized, features["matrix"], names
    )
    code_features, code_means = _code_feature_means(
        tokens, features["standardized"], names, codebook_size
    )
    direction_composition = _direction_composition(
        tokens, features["direction"].astype(np.int64), codebook_size
    )
    dominant_direction = []
    for composition in direction_composition:
        dominant_direction.append(None if composition is None else int(np.argmax(composition)))
    model_path = Path(str(signature.get("model_path") or ""))
    result: dict[str, Any] = {
        "format": FORMAT,
        "source": source,
        "epoch_tag": str(manifest["epoch_tag"]),
        "levels": levels,
        "sample_count": int(len(tokens)),
        "metrics": {
            "motion_neighbor_consistency": _json_number(
                _motion_neighbor_consistency(tokens, features["neighbors"].astype(np.int64))
            ),
            "motion_cohesion": _json_number(_motion_cohesion(tokens, features["motion"])),
            "direction_nmi": _json_number(
                _normalized_mutual_info(features["direction"], tokens)
            ),
            "direction_coherence": _json_number(_direction_coherence(tokens, displacement)),
            "opposite_adjacent_collision": _json_number(
                float(same[opposite].mean()) if opposite.any() else 0.0
            ),
            "adjacent_same_code": _json_number(float(same.mean()) if len(same) else 0.0),
            "task_nmi": _json_number(_normalized_mutual_info(features["tasks"], tokens)),
            "skill_index_nmi": _json_number(
                _normalized_mutual_info(features["skill_indices"], tokens)
            ),
            "gripper_nmi": _json_number(
                _normalized_mutual_info(features["grip_regime"], tokens)
            ),
            "used_codes": int(np.count_nonzero(counts)),
            "effective_codes": _json_number(_entropy_effective(counts)),
            "largest_code_share": _json_number(float(counts.max() / max(counts.sum(), 1))),
            **_checkpoint_scalars(model_path),
        },
        "group_predictability": _group_predictability(
            tokens, features["matrix"], names, episodes
        ),
        "correlation_features": correlation_features,
        "axis_correlations": correlations,
        "axis_strengths": strengths,
        "code_features": code_features,
        "code_feature_means": code_means,
        "direction_labels": list(DIRECTION_LABELS),
        "direction_colors": list(DIRECTION_COLORS),
        "direction_composition": direction_composition,
        "dominant_direction": dominant_direction,
        "counts": counts.astype(int).tolist(),
        "boundary_jitter": (
            _boundary_sweep_metrics(
                boundary_sweep_path, tokens, quantized, features
            )
            if boundary_sweep_path is not None and boundary_sweep_path.is_file()
            else None
        ),
        "terminator": (
            _terminator_metrics(terminator_path, codebook_size)
            if terminator_path is not None and terminator_path.is_file()
            else None
        ),
    }
    _atomic_json(cache_path, result)
    return result


def _global_dataset_root_name(repository: Path) -> str:
    """Read the active dataset root without hard-coding a server layout."""
    config_path = repository / "lerobot/examples/libero/configs/global_config.yaml"
    if not config_path.is_file():
        return ""
    try:
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return str(config.get("dataset_root") or "").strip()
    except ImportError:
        # The report environment normally has PyYAML.  Keep a small fallback so
        # bundle discovery still works in a bootstrap-only Python environment.
        for raw_line in config_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line.startswith("dataset_root:"):
                return line.split(":", 1)[1].split("#", 1)[0].strip().strip("\"'")
    return ""


def _skillset_relative_path(meta: dict[str, Any]) -> Path:
    required = ("target_dataset", "fsq_inputs_name", "skillset_seg_name", "skillset_name")
    missing = [key for key in required if not str(meta.get(key) or "")]
    if missing:
        raise CategorizationUnavailable(f"fsq_meta.json cannot locate skill bundle; missing {missing}")
    relative = Path()
    for key in required:
        relative /= str(meta[key])
    return relative


def _dataset_root_candidates(
    meta: dict[str, Any], repository: Path, fsq_dataset_root: Path
) -> list[Path]:
    if fsq_dataset_root.is_absolute():
        return [fsq_dataset_root]

    root_names: list[str] = []
    for value in (
        meta.get("dataset_root_name"),
        _global_dataset_root_name(repository),
        "dataset",
        "dataset_filtered",
    ):
        name = str(value or "").strip()
        if name and name not in root_names:
            root_names.append(name)
    # Older fsq_meta.json files did not record dataset_root_name.  Discover
    # custom roots such as dataset_ABC while still preferring the active global
    # root and the two historical defaults above.
    for path in sorted(repository.glob("dataset*")):
        if path.is_dir() and path.name not in root_names:
            root_names.append(path.name)
    return [repository / name / fsq_dataset_root for name in root_names]


def _build_missing_bundle(skills_dir: Path, bundle_path: Path) -> None:
    module_path = Path(__file__).resolve().parents[5] / "skills_bundle.py"
    if not module_path.is_file():
        raise CategorizationUnavailable(
            f"skill bundle builder is missing: {module_path}"
        )
    spec = importlib.util.spec_from_file_location("fsq_replay_skills_bundle", module_path)
    if spec is None or spec.loader is None:
        raise CategorizationUnavailable(f"could not load skill bundle builder: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(f"categorization analysis: building missing skill bundle from {skills_dir}")
    try:
        module.build_bundle(skills_dir, bundle_path)
    except Exception as error:
        raise CategorizationUnavailable(
            f"could not build skill bundle from {skills_dir}: {error}"
        ) from error


def _infer_bundle(
    model_path: Path, *, repository: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    meta_path = model_path.parent / "fsq_meta.json"
    if not meta_path.is_file():
        raise CategorizationUnavailable(f"fsq_meta.json is missing: {meta_path}")
    meta = json.loads(meta_path.read_text())
    repository = repository or Path(__file__).resolve().parents[8]
    fsq_dataset_root = Path(str(meta.get("fsq_dataset_root") or "FSQ_dataset"))
    skillset_relative = _skillset_relative_path(meta)
    attempted: list[Path] = []
    for dataset_root in _dataset_root_candidates(meta, repository, fsq_dataset_root):
        skillset_dir = dataset_root / skillset_relative
        attempted.append(skillset_dir)
        bundle = skillset_dir / "skills_bundle.npz"
        if bundle.is_file():
            return bundle, meta
        skills_dir = skillset_dir / "skills"
        if skills_dir.is_dir():
            _build_missing_bundle(skills_dir, bundle)
            if bundle.is_file():
                return bundle, meta
    rendered = "\n  - ".join(str(path) for path in attempted)
    raise CategorizationUnavailable(
        "could not locate the FSQ training skillset under any dataset root:\n"
        f"  - {rendered}"
    )


def _collection_sources(collection_dir: Path) -> tuple[str, str, list[tuple[Path, dict[str, Any]]]]:
    collection_path = collection_dir / "metrics" / "collection.json"
    if not collection_path.is_file():
        raise CategorizationUnavailable(f"collection is incomplete: {collection_path}")
    collection = json.loads(collection_path.read_text())
    checkpoints = []
    for checkpoint in collection.get("checkpoints") or []:
        tag = str(checkpoint["epoch_tag"])
        path = collection_dir / "checkpoints" / tag / "metrics" / "manifest.json"
        if not path.is_file():
            raise CategorizationUnavailable(f"checkpoint manifest is missing: {path}")
        checkpoints.append((path, json.loads(path.read_text())))
    if not checkpoints:
        raise CategorizationUnavailable(f"collection has no checkpoints: {collection_path}")
    return (
        str(collection.get("model_name") or collection_dir.name),
        str(collection.get("run_name") or collection_dir.name),
        checkpoints,
    )


def build_payload(
    collection_dirs: list[str | Path],
    *,
    output_dir: str | Path,
    skill_bundle: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    sources = [_collection_sources(Path(directory)) for directory in collection_dirs]
    inferred_bundles: list[Path] = []
    metas: list[dict[str, Any]] = []
    if skill_bundle is None:
        for _, _, checkpoints in sources:
            bundle, meta = _infer_bundle(Path(checkpoints[-1][1]["signature"]["model_path"]))
            inferred_bundles.append(bundle.resolve())
            metas.append(meta)
        if len(set(inferred_bundles)) != 1:
            raise CategorizationUnavailable(
                f"compared models use different skill bundles: {sorted(map(str, set(inferred_bundles)))}"
            )
        bundle_path = inferred_bundles[0]
    else:
        bundle_path = Path(skill_bundle).resolve()
        if not bundle_path.is_file():
            raise CategorizationUnavailable(f"skill bundle is missing: {bundle_path}")
        for _, _, checkpoints in sources:
            model_path = Path(checkpoints[-1][1]["signature"]["model_path"])
            meta_path = model_path.parent / "fsq_meta.json"
            metas.append(json.loads(meta_path.read_text()) if meta_path.is_file() else {})
    bundle_fingerprint = _bundle_fingerprint(bundle_path)
    features = _load_or_build_features(bundle_path, output_dir)
    models = []
    tokens_by_model_tag: dict[tuple[int, str], np.ndarray] = {}
    for model_index, ((name, run_name, checkpoints), meta) in enumerate(zip(sources, metas, strict=True)):
        results = []
        for manifest_path, manifest in checkpoints:
            result = _checkpoint_metrics(
                manifest_path, manifest, features, bundle_fingerprint
            )
            results.append(result)
            latents_path = Path(manifest["signature"]["latents_path"])
            with np.load(latents_path, allow_pickle=False) as latent:
                tokens_by_model_tag[(model_index, str(manifest["epoch_tag"]))] = latent["tokens"].astype(np.int64)
        results.sort(key=lambda item: _tag_key(item["epoch_tag"]))
        models.append(
            {
                "name": name,
                "run_name": run_name,
                "mode": str(meta.get("autoencoder_mode") or "unknown"),
                "pair_loss": str(meta.get("pair_loss") or "none"),
                "route_loss": bool(meta.get("route_loss", meta.get("reconstruction_route_loss", False))),
                "gripper_weight": float(meta.get("action_gripper_weight", 1.0)),
                "checkpoints": results,
            }
        )
    common_tags = sorted(
        set.intersection(
            *[
                {checkpoint["epoch_tag"] for checkpoint in model["checkpoints"]}
                for model in models
            ]
        ),
        key=_tag_key,
    ) if models else []
    pairwise: dict[str, dict[str, list[list[float]]]] = {}
    for tag in common_tags:
        count = len(models)
        nmi = np.eye(count, dtype=np.float64)
        ari = np.eye(count, dtype=np.float64)
        for first in range(count):
            for second in range(first + 1, count):
                left = tokens_by_model_tag[(first, tag)]
                right = tokens_by_model_tag[(second, tag)]
                nmi[first, second] = nmi[second, first] = _normalized_mutual_info(left, right)
                ari[first, second] = ari[second, first] = _adjusted_rand(left, right)
        pairwise[tag] = {"nmi": nmi.tolist(), "ari": ari.tolist()}
    payload = {
        "format": FORMAT,
        "title": Path(output_dir).parent.name if Path(output_dir).name == "compare" else Path(output_dir).name,
        "bundle_path": str(bundle_path),
        "bundle_fingerprint": bundle_fingerprint,
        "sample_count": int(len(features["episodes"])),
        "neighbor_count": NEIGHBOR_COUNT,
        "feature_labels": FEATURE_LABELS,
        "models": models,
        "common_checkpoints": common_tags,
        "pairwise": pairwise,
    }
    _attach_terminator_frame_pairs(payload, output_dir=output_dir, metas=metas)
    return payload


_DIAGNOSTIC_REPORT_STYLE = """
:root{--bg:#f4f6f9;--panel:#fff;--ink:#17202a;--muted:#667085;--line:#d4dbe6;--blue:#2878b5}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,Arial,"Noto Sans KR",sans-serif}header{position:sticky;top:0;z-index:5;padding:13px 20px;background:#fff;border-bottom:1px solid var(--line)}h1{margin:0 0 5px;font-size:21px}.subtitle{font-size:12px;color:var(--muted)}.toolbar{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin-top:9px}.control{display:flex;align-items:center;gap:5px;font-size:11px;font-weight:800}.control select,.control input{padding:5px 7px;border:1px solid var(--line);border-radius:7px;background:#fff}.nav{padding:5px 9px;border:1px solid #8fb6d3;border-radius:7px;background:#f2f8fc;color:#205f8c;text-decoration:none;font-size:12px;font-weight:800}.layout{display:grid;grid-template-columns:minmax(420px,560px) 1fr;gap:15px;padding:15px;align-items:start}.sidebar{position:sticky;top:115px;background:#fff;border:1px solid var(--line);border-radius:11px;padding:12px}.content{min-width:0}.panel{margin-bottom:14px;padding:13px;background:#fff;border:1px solid var(--line);border-radius:11px}h2{margin:0 0 6px;font-size:16px}h3{margin:8px 0 5px;font-size:13px}.hint,.status{color:var(--muted);font-size:11px;line-height:1.5}.status{padding:7px 9px;background:#f6f9fc;border-radius:7px}.cube{display:block;width:100%;height:auto}.legend{display:flex;gap:9px;flex-wrap:wrap;margin-top:8px;font-size:10px;color:var(--muted)}.legend i{display:inline-block;width:12px;height:9px;border-radius:2px;margin-right:3px}.chart{display:block;width:100%;min-width:650px;height:auto}.scroll{overflow:auto}.metrics{display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:8px}.metric{padding:10px;border:1px solid var(--line);border-radius:8px;background:#f9fbfd}.metric span{display:block;color:var(--muted);font-size:10px}.metric b{display:block;margin-top:4px;font-size:20px}table{border-collapse:collapse;width:100%;font-size:11px}th,td{padding:5px 7px;border:1px solid var(--line);text-align:right}th{background:#eef3fa}th:first-child,td:first-child{text-align:left}.sample-grid{display:grid;grid-template-columns:repeat(2,minmax(330px,1fr));gap:12px;margin-top:10px}.sample-card{overflow:hidden;border:1px solid var(--line);border-radius:10px;background:#f9fbfd}.sample-head{display:flex;justify-content:space-between;gap:10px;align-items:center;padding:9px 10px;font-size:12px}.sample-head span{flex:0 0 auto;padding:3px 7px;border-radius:999px;background:#e7f1f8;color:#205f8c;font-size:10px;font-weight:800}.frame-pair{display:grid;grid-template-columns:1fr 26px 1fr;align-items:center;padding:0 9px 10px}.frame-pair figure{min-width:0;margin:0}.frame-pair a{display:block;overflow:hidden;border-radius:7px;background:#e7ebf0}.frame-pair img{display:block;width:100%;aspect-ratio:1/1;object-fit:cover;transition:transform .15s ease}.frame-pair a:hover img{transform:scale(1.025)}.frame-pair figcaption{margin-top:4px;color:var(--muted);font-size:10px;text-align:center}.motion-arrow{color:#2878b5;font-size:22px;font-weight:900;text-align:center}.empty{padding:28px;text-align:center;color:var(--muted)}@media(max-width:1200px){.sample-grid{grid-template-columns:1fr}}@media(max-width:1050px){.layout{grid-template-columns:1fr}.sidebar{position:relative;top:auto}.metrics{grid-template-columns:1fr 1fr}.sample-grid{grid-template-columns:repeat(2,minmax(300px,1fr))}}@media(max-width:720px){.sample-grid{grid-template-columns:1fr}}
"""


def _write_boundary_jitter_report(output_dir: Path) -> Path:
    html = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FSQ boundary jitter</title><style>__STYLE__</style></head><body>
<header><h1>Boundary jitter diagnostics</h1><div class="subtitle" id="summary"></div><div class="toolbar"><label class="control">Model <select id="model"></select></label><label class="control">Checkpoint <select id="checkpoint"></select></label><label class="control">Task <select id="task"></select></label><label class="control">Direction <select id="direction"><option value="early_start">Early start</option><option value="late_start">Late start</option><option value="early_end">Early end</option><option value="late_end">Late end</option></select></label><label class="control">Metric <select id="metric"><option value="code_retention">Same-code retention</option><option value="latent_l1">Latent L1 drift</option><option value="adjacent_code_attraction">Adjacent-code attraction</option></select></label><label class="control">Offset <input id="offset" type="range" min="0" max="10" value="5"><span id="offsetLabel"></span></label><a class="nav" href="categorization.html">Categorization</a><a class="nav" href="terminator_diagnostics.html">Terminator</a></div></header>
<main class="layout"><aside class="sidebar"><h2>Codebook sensitivity cube</h2><p class="hint">점 색은 현재 task/direction/offset에서 clean code별 지표다. 점을 누르면 해당 code의 전체 offset curve를 본다.</p><svg id="cube" class="cube" viewBox="0 0 600 520"></svg><div class="legend"><span><i style="background:#d7dde8"></i>no sample</span><span><i style="background:hsl(8 72% 50%)"></i>unstable / large</span><span><i style="background:hsl(135 55% 42%)"></i>stable</span></div><div class="status" id="selected"></div></aside><section class="content"><div class="panel"><h2>Offset curve</h2><p class="hint">굵은 선은 선택 code, 점선은 같은 task의 전체 평균이다.</p><div class="scroll" id="curve"></div></div><div class="panel"><h2>Checkpoint table</h2><div class="scroll" id="checkpointTable"></div></div></section></main>
<script src="categorization-data.js"></script><script>
const DATA=window.FSQ_CATEGORIZATION_DATA,$=id=>document.getElementById(id),esc=v=>String(v).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));let mi=0,tag=null,task='all',direction='early_start',metric='code_retention',offset=5,token=0;const M=()=>DATA.models[mi],CP=(m=M(),t=tag)=>(m.checkpoints||[]).find(c=>c.epoch_tag===t),SW=cp=>cp&&cp.boundary_jitter,fmt=(v,pct=false)=>v==null?'·':pct?(100*v).toFixed(1)+'%':Number(v).toFixed(3);
function setup(){ $('model').innerHTML=DATA.models.map((m,i)=>`<option value="${i}">${esc(m.name)}</option>`).join('');$('model').onchange=()=>{mi=+$('model').value;tag=null;setCheckpoints();render()};$('checkpoint').onchange=()=>{tag=$('checkpoint').value;setTasks();render()};$('task').onchange=()=>{task=$('task').value;render()};$('direction').onchange=()=>{direction=$('direction').value;render()};$('metric').onchange=()=>{metric=$('metric').value;render()};$('offset').oninput=()=>{offset=+$('offset').value;render()};setCheckpoints();render()}
function setCheckpoints(){const tags=M().checkpoints.map(c=>c.epoch_tag);tag=tag&&tags.includes(tag)?tag:tags[tags.length-1];$('checkpoint').innerHTML=tags.map(t=>`<option${t===tag?' selected':''}>${t}</option>`).join('');setTasks()}
function setTasks(){const s=SW(CP()),tasks=s?Object.keys(s.scopes).sort((a,b)=>a==='all'?-1:b==='all'?1:+a-+b):['all'];if(!tasks.includes(task))task='all';$('task').innerHTML=tasks.map(t=>`<option value="${t}"${t===task?' selected':''}>${t==='all'?'All tasks':'Task '+t}</option>`).join('');if(s){$('offset').max=s.offsets.length-1;offset=Math.min(offset,s.offsets.length-1)} }
function coord(t,l){let c=[],b=1;for(const n of l){c.push(Math.floor(t/b)%n);b*=n}return c}function project(c,l){const v=l.map((n,i)=>((c[i]||0)-(n-1)/2)/Math.max(1,(Math.max(...l)-1)/2)),yaw=-.63,pitch=.46,cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),xr=cy*v[0]-sy*v[1],yr=sy*v[0]+cy*v[1],z=v[2]||0;return[300+xr*145,265+yr*145*sp-z*145*cp,yr*cp+z*sp]}
function codeValue(cp,tok){const s=SW(cp),scope=s&&s.scopes[task];if(!scope)return null;const d=s.directions.indexOf(direction),row=scope.codes[metric]&&scope.codes[metric][d]&&scope.codes[metric][d][offset];return row&&row[tok]!=null?Number(row[tok]):null}
function color(v,values){if(v==null)return'#d7dde8';if(metric==='code_retention')return`hsl(${8+127*v} ${58+12*v}% ${50-8*v}%)`;const max=Math.max(...values.filter(x=>x!=null),1e-9),q=Math.min(1,v/max);return`hsl(${135-127*q} ${55+17*q}% ${42+8*q}%)`}
function renderCube(){const cp=CP(),s=SW(cp),svg=$('cube'),NS='http://www.w3.org/2000/svg';svg.innerHTML='';if(!s){svg.innerHTML='<text x="300" y="260" text-anchor="middle" fill="#667085">Sweep unavailable</text>';return}const levels=cp.levels,total=levels.reduce((a,b)=>a*b,1),values=Array.from({length:total},(_,t)=>codeValue(cp,t)),make=(n,a)=>{const e=document.createElementNS(NS,n);Object.entries(a).forEach(([k,v])=>e.setAttribute(k,v));svg.appendChild(e);return e};for(let t=0;t<total;t++){const c=coord(t,levels);for(let d=0;d<Math.min(3,levels.length);d++)if(c[d]+1<levels[d]){const n=c.slice();n[d]++;const a=project(c,levels),b=project(n,levels);make('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:'rgba(80,90,105,.38)','stroke-width':1.2})}}const points=Array.from({length:total},(_,t)=>({t,p:project(coord(t,levels),levels)})).sort((a,b)=>a.p[2]-b.p[2]);for(const {t,p} of points){const circle=make('circle',{cx:p[0],cy:p[1],r:t===token?9:6,fill:color(values[t],values),stroke:t===token?'#111':'#334155','stroke-width':t===token?2.5:1,style:'cursor:pointer'});circle.onclick=()=>{token=t;render()};const title=document.createElementNS(NS,'title');title.textContent=`code #${t}: ${fmt(values[t],metric!=='latent_l1')}`;circle.appendChild(title)}}
function chart(cp){const s=SW(cp),scope=s&&s.scopes[task];if(!scope)return'<div class="empty">No sweep data.</div>';const d=s.directions.indexOf(direction),overall=scope.metrics[metric][d],selected=scope.codes[metric][d].map(row=>row[token]),all=[...overall,...selected].filter(v=>v!=null),ymax=metric==='code_retention'||metric==='adjacent_code_attraction'?1:Math.max(...all,1e-6)*1.08,w=900,h=320,L=55,R=18,T=18,B=54,x=i=>L+i/Math.max(s.offsets.length-1,1)*(w-L-R),y=v=>T+(ymax-v)/ymax*(h-T-B),path=a=>a.map((v,i)=>v==null?null:[i,v]).filter(Boolean).map((p,i)=>`${i?'L':'M'}${x(p[0])},${y(p[1])}`).join(' ');let out=`<svg class="chart" viewBox="0 0 ${w} ${h}">`;for(let i=0;i<=4;i++){const v=ymax*(1-i/4),yy=y(v);out+=`<line x1="${L}" y1="${yy}" x2="${w-R}" y2="${yy}" stroke="#dde3eb"/><text x="${L-6}" y="${yy+4}" text-anchor="end" font-size="10">${fmt(v,metric!=='latent_l1')}</text>`}out+=`<path d="${path(overall)}" fill="none" stroke="#8b95a5" stroke-width="2" stroke-dasharray="6 4"/><path d="${path(selected)}" fill="none" stroke="#2878b5" stroke-width="3"/>`;selected.forEach((v,i)=>{if(v!=null)out+=`<circle cx="${x(i)}" cy="${y(v)}" r="4" fill="#2878b5"><title>${s.offsets[i]} frames · ${fmt(v,metric!=='latent_l1')}</title></circle>`});out+=`<line x1="${L}" y1="${h-B}" x2="${w-R}" y2="${h-B}" stroke="#667085"/>`;s.offsets.forEach((value,i)=>{out+=`<line x1="${x(i)}" y1="${h-B}" x2="${x(i)}" y2="${h-B+4}" stroke="#667085"/><text x="${x(i)}" y="${h-B+17}" text-anchor="middle" font-size="10" fill="#667085">${value}</text>`});return out+`<text x="${(L+w-R)/2}" y="${h-8}" text-anchor="middle" font-size="11" fill="#667085">Offset (frames)</text></svg>`}
function table(){const rows=M().checkpoints.map(cp=>{const s=SW(cp),scope=s&&s.scopes[task],d=s?s.directions.indexOf(direction):-1,o=scope&&scope.metrics[metric][d][offset],c=scope&&scope.codes[metric][d][offset][token];return`<tr><td>${cp.epoch_tag}</td><td>${fmt(o,metric!=='latent_l1')}</td><td>${fmt(c,metric!=='latent_l1')}</td><td>${scope?scope.metrics.valid_count[d][offset]:'·'}</td></tr>`});$('checkpointTable').innerHTML=`<table><thead><tr><th>checkpoint</th><th>task overall</th><th>code #${token}</th><th>N</th></tr></thead><tbody>${rows.join('')}</tbody></table>`}
function render(){const cp=CP(),s=SW(cp);$('offsetLabel').textContent=offset+' frames';$('summary').textContent=`${DATA.title} · ${M().name} · ${tag} · ${task==='all'?'all tasks':'task '+task}`;$('selected').textContent=`code #${token} · ${direction} · ${offset} frames · ${metric}: ${fmt(codeValue(cp,token),metric!=='latent_l1')}`;renderCube();$('curve').innerHTML=chart(cp);table()};setup();
</script></body></html>""".replace("__STYLE__", _DIAGNOSTIC_REPORT_STYLE)
    path = output_dir / "boundary_jitter.html"
    path.write_text(html, encoding="utf-8")
    return path


def _write_terminator_report(output_dir: Path) -> Path:
    html = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FSQ terminator diagnostics</title><style>__STYLE__</style></head><body>
<header><h1>Terminator diagnostics</h1><div class="subtitle" id="summary"></div><div class="toolbar"><label class="control">Model <select id="model"></select></label><label class="control">Checkpoint <select id="checkpoint"></select></label><label class="control">Task <select id="task"></select></label><label class="control">Cube color <select id="metric"><option value="boundary_probability">P(end) at GT boundary</option><option value="timing_abs">|first trigger − boundary|</option><option value="missed_fraction">Missed fraction</option><option value="early_fraction">Early-trigger fraction</option></select></label><a class="nav" href="categorization.html">Categorization</a><a class="nav" href="boundary_jitter.html">Boundary jitter</a></div></header>
<main class="layout"><aside class="sidebar"><h2>Terminator codebook cube</h2><p class="hint">현재 task에서 code별 종료 성능을 색으로 표시한다. 점을 누르면 해당 code의 종료 곡선과 실제 motion 예시를 본다.</p><svg id="cube" class="cube" viewBox="0 0 600 520"></svg><div class="status" id="selected"></div></aside><section class="content"><div class="panel"><h2>GT boundary-aligned probability</h2><p class="hint">0이 GT skill-end frame이다. 음수는 boundary 이전, 양수는 다음 skill로 넘어간 뒤의 frame이다. 음영은 10–90 percentile, 점선은 threshold.</p><div class="scroll" id="curve"></div></div><div class="panel"><h2>Summary</h2><div class="metrics" id="metrics"></div></div><div class="panel"><h2>Checkpoint trend</h2><div class="scroll" id="checkpointTable"></div></div><div class="panel"><h2>Selected skill motions</h2><p class="hint">선택한 code/task에 속한 실제 skill의 시작과 마지막 top-view frame이다. 이미지를 누르면 원본 크기로 열린다.</p><div id="samples"></div></div></section></main>
<script src="categorization-data.js"></script><script>
const DATA=window.FSQ_CATEGORIZATION_DATA,$=id=>document.getElementById(id),esc=v=>String(v).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));let mi=Math.max(0,DATA.models.findIndex(m=>m.checkpoints.some(c=>c.terminator))),tag=null,task='all',metric='boundary_probability',token=0;const M=()=>DATA.models[mi],CP=(m=M(),t=tag)=>(m.checkpoints||[]).find(c=>c.epoch_tag===t),TERM=cp=>cp&&cp.terminator,fmt=(v,pct=false)=>v==null?'·':pct?(100*v).toFixed(1)+'%':Number(v).toFixed(2);
function setup(){$('model').innerHTML=DATA.models.map((m,i)=>`<option value="${i}"${i===mi?' selected':''}>${esc(m.name)}${m.checkpoints.some(c=>c.terminator)?'':' · no terminator'}</option>`).join('');$('model').onchange=()=>{mi=+$('model').value;tag=null;setCheckpoints();render()};$('checkpoint').onchange=()=>{tag=$('checkpoint').value;setTasks();render()};$('task').onchange=()=>{task=$('task').value;render()};$('metric').onchange=()=>{metric=$('metric').value;render()};setCheckpoints();render()}
function setCheckpoints(){const tags=M().checkpoints.map(c=>c.epoch_tag);tag=tag&&tags.includes(tag)?tag:tags[tags.length-1];$('checkpoint').innerHTML=tags.map(t=>`<option${t===tag?' selected':''}>${t}</option>`).join('');setTasks()}
function setTasks(){const t=TERM(CP()),tasks=t?Object.keys(t.scopes).sort((a,b)=>a==='all'?-1:b==='all'?1:+a-+b):['all'];if(!tasks.includes(task))task='all';$('task').innerHTML=tasks.map(v=>`<option value="${v}"${v===task?' selected':''}>${v==='all'?'All tasks':'Task '+v}</option>`).join('')}
function coord(t,l){let c=[],b=1;for(const n of l){c.push(Math.floor(t/b)%n);b*=n}return c}function project(c,l){const v=l.map((n,i)=>((c[i]||0)-(n-1)/2)/Math.max(1,(Math.max(...l)-1)/2)),yaw=-.63,pitch=.46,cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),xr=cy*v[0]-sy*v[1],yr=sy*v[0]+cy*v[1],z=v[2]||0;return[300+xr*145,265+yr*145*sp-z*145*cp,yr*cp+z*sp]}
function summary(cp,tok=null){const t=TERM(cp),scope=t&&t.scopes[task];return scope?(tok==null?scope.overall:scope.codes[tok]):null}function value(cp,tok){const s=summary(cp,tok);return s?s[metric]:null}function color(v,values){if(v==null)return'#d7dde8';const max=metric==='boundary_probability'||metric.includes('fraction')?1:Math.max(...values.filter(x=>x!=null),1),q=Math.min(1,v/max),good=metric==='boundary_probability'?q:1-q;return`hsl(${8+127*good} ${62}% ${49-7*good}%)`}
function renderCube(){const cp=CP(),term=TERM(cp),svg=$('cube'),NS='http://www.w3.org/2000/svg';svg.innerHTML='';if(!term){svg.innerHTML='<text x="300" y="260" text-anchor="middle" fill="#667085">No terminator artifact</text>';return}const levels=cp.levels,total=levels.reduce((a,b)=>a*b,1),values=Array.from({length:total},(_,t)=>value(cp,t)),make=(n,a)=>{const e=document.createElementNS(NS,n);Object.entries(a).forEach(([k,v])=>e.setAttribute(k,v));svg.appendChild(e);return e};for(let t=0;t<total;t++){const c=coord(t,levels);for(let d=0;d<Math.min(3,levels.length);d++)if(c[d]+1<levels[d]){const n=c.slice();n[d]++;const a=project(c,levels),b=project(n,levels);make('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:'rgba(80,90,105,.38)','stroke-width':1.2})}}for(const p of Array.from({length:total},(_,t)=>({t,p:project(coord(t,levels),levels)})).sort((a,b)=>a.p[2]-b.p[2])){const c=make('circle',{cx:p.p[0],cy:p.p[1],r:p.t===token?9:6,fill:color(values[p.t],values),stroke:p.t===token?'#111':'#334155','stroke-width':p.t===token?2.5:1,style:'cursor:pointer'});c.onclick=()=>{token=p.t;render()}}}
function curve(cp){const term=TERM(cp),s=summary(cp,token);if(!term||!s||!s.sample_count)return'<div class="empty">No samples for this code/task.</div>';const xs=s.offsets,mean=s.mean_probability,p10=s.p10_probability,p90=s.p90_probability,w=900,h=320,L=55,R=18,T=18,B=38,x=v=>L+(v-xs[0])/Math.max(xs[xs.length-1]-xs[0],1)*(w-L-R),y=v=>T+(1-v)*(h-T-B),points=(arr,rev=false)=>arr.map((v,i)=>v==null?null:[x(xs[i]),y(v)]).filter(Boolean).map(p=>p.join(',')).reverseIf;let band=[];p90.forEach((v,i)=>{if(v!=null)band.push(`${x(xs[i])},${y(v)}`)});for(let i=p10.length-1;i>=0;i--)if(p10[i]!=null)band.push(`${x(xs[i])},${y(p10[i])}`);const path=mean.map((v,i)=>v==null?null:[i,v]).filter(Boolean).map((p,i)=>`${i?'L':'M'}${x(xs[p[0]])},${y(p[1])}`).join(' ');let out=`<svg class="chart" viewBox="0 0 ${w} ${h}">`;for(let i=0;i<=4;i++){const v=1-i/4,yy=y(v);out+=`<line x1="${L}" y1="${yy}" x2="${w-R}" y2="${yy}" stroke="#dde3eb"/><text x="${L-6}" y="${yy+4}" text-anchor="end" font-size="10">${(100*v).toFixed(0)}%</text>`}out+=`<polygon points="${band.join(' ')}" fill="rgba(40,120,181,.16)"/><line x1="${x(0)}" y1="${T}" x2="${x(0)}" y2="${h-B}" stroke="#c43d3d" stroke-width="2"/><line x1="${L}" y1="${y(term.threshold)}" x2="${w-R}" y2="${y(term.threshold)}" stroke="#b66a16" stroke-dasharray="6 4"/><path d="${path}" fill="none" stroke="#2878b5" stroke-width="3"/>`;return out+`<text x="${x(0)+5}" y="${T+12}" font-size="10" fill="#c43d3d">GT end</text></svg>`}
function metrics(cp){const s=summary(cp,token),items=[['Samples',s&&s.sample_count,0],['P(end) @ boundary',s&&s.boundary_probability,'pct'],['|timing error|',s&&s.timing_abs,2],['Missed',s&&s.missed_fraction,'pct'],['Early',s&&s.early_fraction,'pct'],['Late',s&&s.late_fraction,'pct']];$('metrics').innerHTML=items.map(([k,v,p])=>`<div class="metric"><span>${k}</span><b>${p==='pct'?fmt(v,true):v==null?'·':Number(v).toFixed(p)}</b></div>`).join('')}
function table(){const rows=M().checkpoints.map(cp=>{const s=summary(cp,token);return`<tr><td>${cp.epoch_tag}</td><td>${s?s.sample_count:'·'}</td><td>${fmt(s&&s.boundary_probability,true)}</td><td>${fmt(s&&s.timing_abs)}</td><td>${fmt(s&&s.missed_fraction,true)}</td><td>${fmt(s&&s.early_fraction,true)}</td><td>${fmt(s&&s.late_fraction,true)}</td></tr>`});$('checkpointTable').innerHTML=`<table><thead><tr><th>checkpoint</th><th>N</th><th>P@end</th><th>|timing|</th><th>missed</th><th>early</th><th>late</th></tr></thead><tbody>${rows.join('')}</tbody></table>`}
function samples(cp){const term=TERM(cp);if(!term){$('samples').innerHTML='<div class="empty">No samples.</div>';return}const matched=term.samples.filter(s=>(task==='all'||String(s.task_id)===task)&&s.token===token),rows=matched.slice(0,20);$('samples').innerHTML=rows.length?`<div class="status">${matched.length}개 중 ${rows.length}개 표시 · code #${token}</div><div class="sample-grid">${rows.map(s=>{const title=`task ${s.task_id} · episode ${s.episode_id} · skill ${s.skill_index}`,prob=fmt(s.probabilities[s.boundary_index],true);if(!s.start_image||!s.end_image)return`<article class="sample-card"><div class="sample-head"><b>${title}</b><span>P@end ${prob}</span></div><div class="empty">Frame pair unavailable</div></article>`;return`<article class="sample-card"><div class="sample-head"><b>${title}</b><span>P@end ${prob}</span></div><div class="frame-pair"><figure><a href="${esc(s.start_image)}" target="_blank"><img loading="lazy" src="${esc(s.start_image)}" alt="${title} start"></a><figcaption>Start · frame ${s.frame_start}</figcaption></figure><div class="motion-arrow">→</div><figure><a href="${esc(s.end_image)}" target="_blank"><img loading="lazy" src="${esc(s.end_image)}" alt="${title} end"></a><figcaption>End · frame ${s.end_image_frame??s.frame_end-1}</figcaption></figure></div></article>`}).join('')}</div>`:'<div class="empty">No samples for selected code/task.</div>'}
function render(){const cp=CP(),s=summary(cp,token);$('summary').textContent=`${DATA.title} · ${M().name} · ${tag} · ${task==='all'?'all tasks':'task '+task}`;$('selected').textContent=`code #${token} · ${metric}: ${fmt(value(cp,token),metric!=='timing_abs')}`;renderCube();$('curve').innerHTML=curve(cp);metrics(cp);table();samples(cp)}setup();
</script></body></html>""".replace("__STYLE__", _DIAGNOSTIC_REPORT_STYLE)
    path = output_dir / "terminator_diagnostics.html"
    path.write_text(html, encoding="utf-8")
    return path


def write_report(output_dir: str | Path, payload: dict[str, Any]) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "metrics" / DATA_NAME, payload)
    data_path = output_dir / "categorization-data.js"
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    temporary_data = data_path.with_suffix(".js.tmp")
    temporary_data.write_text(f"window.FSQ_CATEGORIZATION_DATA={serialized};\n", encoding="utf-8")
    temporary_data.replace(data_path)
    document = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FSQ categorization analysis</title>
<style>
:root{--bg:#f4f6f9;--panel:#fff;--ink:#17202a;--muted:#667085;--line:#d4dbe6;--blue:#2878b5;--green:#27845b;--red:#c43d3d;--amber:#b66a16}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,Arial,"Noto Sans KR",sans-serif}header{position:sticky;top:0;z-index:5;padding:13px 20px;background:#fff;border-bottom:1px solid var(--line)}h1{margin:0 0 5px;font-size:21px}.subtitle{font-size:12px;color:var(--muted)}.toolbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:9px}.control{display:flex;align-items:center;gap:6px;font-size:12px;font-weight:800}.control select{max-width:420px;padding:5px 8px;border:1px solid var(--line);border-radius:7px;background:#fff}.nav{padding:5px 9px;border:1px solid #8fb6d3;border-radius:7px;background:#f2f8fc;color:#205f8c;text-decoration:none;font-size:12px;font-weight:800}main{max-width:1600px;margin:auto;padding:16px}.section{margin:0 0 15px;padding:14px;background:var(--panel);border:1px solid var(--line);border-radius:11px}h2{margin:0 0 5px;font-size:16px}h3{margin:15px 0 6px;font-size:13px}.hint{margin:3px 0 10px;color:var(--muted);font-size:11px;line-height:1.55}.metrics{display:grid;grid-template-columns:repeat(6,minmax(150px,1fr));gap:9px}.metric{padding:11px;border:1px solid var(--line);border-radius:9px;background:#f9fbfd}.metric span{display:block;color:var(--muted);font-size:10px;font-weight:800}.metric b{display:block;margin:4px 0 2px;font-size:22px}.metric small{font-size:10px;color:var(--muted)}.higher b{color:var(--green)}.lower b{color:var(--red)}.neutral b{color:var(--blue)}.grid{display:grid;grid-template-columns:1fr 1fr;gap:13px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:11px}th,td{padding:5px 7px;border:1px solid var(--line);text-align:right;white-space:nowrap}th{background:#eef3fa;font-weight:800}th:first-child,td:first-child{text-align:left}.heat td{min-width:64px;text-align:center;font-variant-numeric:tabular-nums}.heat td:first-child{position:sticky;left:0;z-index:1;background:#fff}.bars{display:grid;gap:8px}.bar-row{display:grid;grid-template-columns:150px 1fr 48px;align-items:center;gap:8px;font-size:11px}.bar-track{height:14px;border-radius:7px;background:#e9eef5;overflow:hidden}.bar-fill{height:100%;background:var(--blue);border-radius:7px}.direction{display:flex;width:180px;height:15px;border-radius:6px;overflow:hidden;background:#edf0f4}.direction i{display:block;height:100%}.code-table td{vertical-align:middle}.legend{display:flex;gap:9px;flex-wrap:wrap;color:var(--muted);font-size:10px}.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:3px}.chart{display:block;min-width:760px;width:100%;height:auto}.trend-chart-wrap{overflow-x:auto}.trend-status{min-height:30px;margin:9px 0;padding:7px 10px;border:1px solid #cbd8e6;border-radius:7px;background:#f6f9fc;color:#344054;font-size:11px;font-weight:700}.trend-legend{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:6px;margin-top:10px}.trend-legend-item{display:flex;align-items:center;gap:7px;min-width:0;padding:6px 8px;border:1px solid var(--line);border-radius:7px;background:#fff;color:#344054;text-align:left;font:inherit;font-size:10px;cursor:pointer}.trend-legend-item span:last-child{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.trend-swatch{width:20px;height:4px;border-radius:2px;flex:0 0 auto}.trend-series,.trend-legend-item{transition:opacity .12s ease,filter .12s ease,box-shadow .12s ease}.trend-series.is-muted,.trend-legend-item.is-muted{opacity:.14}.trend-series.is-active .trend-visible{stroke-width:4}.trend-series.is-active circle{stroke:#111;stroke-width:1.2}.trend-legend-item.is-active{border-color:#344054;box-shadow:0 0 0 1px #344054 inset;font-weight:800}.warning{padding:9px 11px;border-left:4px solid var(--amber);background:#fff6e8;color:#69400d;font-size:11px}.empty{padding:20px;text-align:center;color:var(--muted)}@media(max-width:1050px){.metrics{grid-template-columns:repeat(3,1fr)}.grid{grid-template-columns:1fr}}@media(max-width:620px){main{padding:10px}.metrics{grid-template-columns:1fr 1fr}.bar-row{grid-template-columns:115px 1fr 42px}.trend-legend{grid-template-columns:1fr}}
</style></head><body>
<header><h1>FSQ categorization analysis</h1><div class="subtitle" id="summary"></div><div class="toolbar"><label class="control">Model <select id="model"></select></label><label class="control">Checkpoint <select id="checkpoint"></select></label><a class="nav" href="index.html">GT replay</a><a class="nav" href="linked_codebooks.html">Linked codebooks</a><a class="nav" href="boundary_jitter.html">Boundary jitter</a><a class="nav" href="terminator_diagnostics.html">Terminator</a></div></header>
<main>
<section class="section"><h2>핵심 categorization 지표</h2><p class="hint">전체 training skillset 기준. Utilization을 품질 점수로 사용하지 않는다. Reconstruction은 preprocessing과 loss weight가 같은 모델끼리만 직접 비교할 수 있다.</p><div class="metrics" id="metrics"></div></section>
<section class="section"><h2>Checkpoint trend</h2><div class="toolbar"><label class="control">Metric <select id="trendMetric"></select></label></div><p class="hint">그래프의 선이나 아래 범례에 마우스를 올리면 모델명이 표시된다. 클릭하면 해당 모델을 고정 강조하고, 다시 클릭하면 해제한다.</p><div class="trend-status" id="trendStatus"></div><div id="trend"></div></section>
<section class="section"><h2>현재 checkpoint 모델 비교</h2><p class="hint">선택한 checkpoint가 존재하는 모델만 표시한다. 서로 다른 autoencoder mode의 reconstruction 값은 직접 순위화하지 않는다.</p><div class="scroll" id="comparison"></div></section>
<section class="section"><h2>Boundary jitter sweep</h2><div class="toolbar"><label class="control">Metric <select id="boundaryMetric"><option value="code_retention">Same-code retention</option><option value="latent_l1">Quantized latent L1 drift</option><option value="adjacent_code_attraction">Adjacent-code attraction</option></select></label></div><p class="hint">각 GT skill의 시작/끝 경계를 정확히 0–10 frame 이동해 다시 인코딩한다. early-start와 late-end는 이웃 skill이 섞이는 방향이며 adjacent attraction을 함께 볼 수 있다. 작은 offset에서 attraction이 높으면 경계 오염에 민감하다는 뜻이다.</p><div class="trend-status" id="boundaryStatus"></div><div class="grid" id="boundaryCharts"></div></section>
<div class="grid"><section class="section"><h2>Feature-group code predictability</h2><p class="hint">Episode-held-out nearest-centroid balanced accuracy. 높을수록 해당 특징 그룹만으로 code를 쉽게 예측할 수 있다.</p><div class="bars" id="groups"></div></section><section class="section"><h2>Shortcut / occupancy diagnostics</h2><p class="hint">좋고 나쁨을 직접 판정하지 않는 진단값이다.</p><div id="diagnostics"></div></section></div>
<section class="section"><h2>FSQ axis ↔ trajectory feature map</h2><p class="hint">Token 번호가 아니라 quantized FSQ scalar axis를 사용한다. 왼쪽 Spearman ρ는 방향, 오른쪽 η²는 비단조 관계까지 포함한 association strength다. Zero/action 모델의 입력에 없던 특징은 사후 설명용 proxy다.</p><div class="grid"><div><h3>Signed Spearman ρ</h3><div class="scroll" id="correlation"></div></div><div><h3>η² association strength</h3><div class="scroll" id="strength"></div></div></div></section>
<section class="section"><h2>Code × semantic feature heatmap</h2><p class="hint">각 code 내부 평균을 전체 dataset z-score로 표시한다. 양수는 전체 평균보다 크고, 음수는 작다는 뜻이다.</p><div class="scroll" id="codeHeatmap"></div></section>
<section class="section"><h2>Code별 실제 EE 방향 구성</h2><div class="legend" id="directionLegend"></div><div class="scroll" id="codes"></div></section>
<section class="section"><h2>모델 간 assignment agreement</h2><p class="hint">공통 checkpoint에서 동일한 전체 skill assignment를 비교한다. NMI/ARI는 두 partition이 얼마나 비슷한지만 나타내며 categorization 품질 점수는 아니다.</p><div class="grid"><div><h3>NMI</h3><div class="scroll" id="pairNmi"></div></div><div><h3>ARI</h3><div class="scroll" id="pairAri"></div></div></div></section>
<section class="warning">Motion-neighbor consistency는 다른 episode에서 찾은 최근접 10개 motion 이웃의 same-code 비율에서 전역 code occupancy에 따른 우연 일치를 보정한다. Motion cohesion은 상대 translation·rotation 특징을 표준화한 뒤 code centroid가 설명하는 분산 비율이다.</section>
</main><script src="categorization-data.js"></script><script>
const DATA=window.FSQ_CATEGORIZATION_DATA,modelSelect=document.getElementById('model'),checkpointSelect=document.getElementById('checkpoint'),trendSelect=document.getElementById('trendMetric'),boundarySelect=document.getElementById('boundaryMetric');
const esc=value=>String(value).replace(/[&<>'"]/g,char=>({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
const metricDefs={motion_neighbor_consistency:['Motion-neighbor consistency','higher',3],motion_cohesion:['Motion cohesion R²','higher',3],direction_nmi:['XYZ direction NMI','higher',3],direction_coherence:['Direction coherence','higher',3],opposite_adjacent_collision:['Opposite-adjacent collision','lower','pct'],validation_reconstruction:['Validation reconstruction','neutral',5]};
const diagnosticDefs={used_codes:'Active codes',effective_codes:'Effective codes',largest_code_share:'Largest-code share',task_nmi:'Task ID NMI',skill_index_nmi:'Skill-index NMI',gripper_nmi:'Gripper NMI',adjacent_same_code:'Adjacent same-code'};
const boundaryDefs={code_retention:['Same-code retention','pct'],latent_l1:['Quantized latent L1 drift',3],adjacent_code_attraction:['Adjacent-code attraction','pct']};
let modelIndex=0,checkpointTag=null,trendPinnedIndex=null;
const model=()=>DATA.models[modelIndex],checkpointFor=(item,tag=checkpointTag)=>(item.checkpoints||[]).find(cp=>cp.epoch_tag===tag),value=(cp,key)=>cp&&cp.metrics?cp.metrics[key]:null;
function format(number,spec=3){if(number==null||!Number.isFinite(Number(number)))return '·';return spec==='pct'?`${(100*Number(number)).toFixed(2)}%`:Number(number).toFixed(spec)}
function setup(){modelSelect.innerHTML=DATA.models.map((item,index)=>`<option value="${index}">${esc(item.name)}</option>`).join('');trendSelect.innerHTML=Object.entries(metricDefs).map(([key,definition])=>`<option value="${key}">${esc(definition[0])}</option>`).join('');modelSelect.addEventListener('change',()=>{modelIndex=Number(modelSelect.value);checkpointTag=null;setCheckpoints();render()});checkpointSelect.addEventListener('change',()=>{checkpointTag=checkpointSelect.value;render()});trendSelect.addEventListener('change',renderTrend);boundarySelect.addEventListener('change',renderBoundarySweep);setCheckpoints();render()}
function setCheckpoints(){const tags=model().checkpoints.map(cp=>cp.epoch_tag);if(!checkpointTag||!tags.includes(checkpointTag)){const common=DATA.common_checkpoints.filter(tag=>tags.includes(tag));checkpointTag=(common.length?common:tags)[(common.length?common:tags).length-1]}checkpointSelect.innerHTML=tags.map(tag=>`<option value="${esc(tag)}"${tag===checkpointTag?' selected':''}>${esc(tag)}</option>`).join('')}
function render(){const cp=checkpointFor(model());document.getElementById('summary').textContent=`${DATA.title} · complete training skillset ${DATA.sample_count.toLocaleString()} · ${model().mode} · pair ${model().pair_loss} · route ${model().route_loss?'ON':'OFF'}`;renderMetrics(cp);renderTrend();renderComparison();renderBoundarySweep();renderGroups(cp);renderDiagnostics(cp);renderAxisMaps(cp);renderCodeHeatmap(cp);renderCodes(cp);renderPairwise()}
function renderMetrics(cp){document.getElementById('metrics').innerHTML=Object.entries(metricDefs).map(([key,[label,direction,spec]])=>`<div class="metric ${direction}"><span>${esc(label)}</span><b>${format(value(cp,key),spec)}</b><small>${direction==='higher'?'higher is better':direction==='lower'?'lower is better':key==='validation_reconstruction'?'same mode only':'diagnostic'}</small></div>`).join('')}
function epoch(tag){const match=/^epoch(\\d+)$/.exec(tag);return match?Number(match[1]):NaN}
function trendColor(index){const hue=(207+index*137.508)%360,saturation=64+(index%3)*6,lightness=38+(index%2)*8;return `hsl(${hue.toFixed(1)} ${saturation}% ${lightness}%)`}
function updateTrendFocus(index){const root=document.getElementById('trend'),status=document.getElementById('trendStatus'),active=Number.isInteger(index)?index:null;root.querySelectorAll('[data-trend-index]').forEach(element=>{const current=Number(element.dataset.trendIndex),focused=active===current;element.classList.toggle('is-active',focused);element.classList.toggle('is-muted',active!==null&&!focused)});if(active===null){status.textContent=`모든 ${root.querySelectorAll('.trend-series').length}개 모델 표시 중 · 선 또는 범례를 hover/click하여 모델을 확인하세요.`;return}const prefix=trendPinnedIndex===active?'고정 선택':'Hover';status.textContent=`${prefix}: ${DATA.models[active].name}`}
function bindTrendInteractions(){const root=document.getElementById('trend');root.querySelectorAll('[data-trend-index]').forEach(element=>{const index=Number(element.dataset.trendIndex);element.addEventListener('mouseenter',()=>updateTrendFocus(index));element.addEventListener('mouseleave',()=>updateTrendFocus(trendPinnedIndex));element.addEventListener('focus',()=>updateTrendFocus(index));element.addEventListener('blur',()=>updateTrendFocus(trendPinnedIndex));element.addEventListener('click',event=>{event.stopPropagation();trendPinnedIndex=trendPinnedIndex===index?null:index;updateTrendFocus(trendPinnedIndex)});if(element.classList.contains('trend-series'))element.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();element.click()}})});updateTrendFocus(trendPinnedIndex)}
function renderTrend(){const key=trendSelect.value||Object.keys(metricDefs)[0],definition=metricDefs[key],series=DATA.models.map((item,index)=>({item,index,points:item.checkpoints.map(cp=>[epoch(cp.epoch_tag),value(cp,key)]).filter(point=>Number.isFinite(point[0])&&point[1]!=null)})).filter(item=>item.points.length);const target=document.getElementById('trend');if(!series.length){target.innerHTML='<div class="empty">No values.</div>';document.getElementById('trendStatus').textContent='이 metric에 표시할 모델이 없습니다.';return}if(trendPinnedIndex!==null&&!series.some(row=>row.index===trendPinnedIndex))trendPinnedIndex=null;const all=series.flatMap(item=>item.points),xmin=Math.min(...all.map(p=>p[0])),xmax=Math.max(...all.map(p=>p[0])),rawMin=Math.min(...all.map(p=>p[1])),rawMax=Math.max(...all.map(p=>p[1])),padding=(rawMax-rawMin||1)*.08,ymin=Math.min(0,rawMin-padding),ymax=rawMax+padding,width=1050,height=300,left=55,right=20,top=20,bottom=38,pw=width-left-right,ph=height-top-bottom,x=x=>left+(x-xmin)/Math.max(xmax-xmin,1)*pw,y=y=>top+(ymax-y)/Math.max(ymax-ymin,1e-9)*ph;let html=`<div class="trend-chart-wrap"><svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(definition[0])} checkpoint trend for ${series.length} models">`;for(let i=0;i<=4;i++){const yy=top+i*ph/4,val=ymax-i*(ymax-ymin)/4;html+=`<line x1="${left}" y1="${yy}" x2="${width-right}" y2="${yy}" stroke="#dde3eb"/><text x="${left-7}" y="${yy+4}" text-anchor="end" font-size="10" fill="#667085">${definition[2]==='pct'?(val*100).toFixed(1)+'%':val.toFixed(3)}</text>`}series.forEach(({item,index,points})=>{const path=points.map((point,i)=>`${i?'L':'M'}${x(point[0]).toFixed(1)},${y(point[1]).toFixed(1)}`).join(' '),color=trendColor(index);html+=`<g class="trend-series" data-trend-index="${index}" tabindex="0" role="button" aria-label="${esc(item.name)}"><title>${esc(item.name)}</title><path class="trend-visible" d="${path}" fill="none" stroke="${color}" stroke-width="2.2"/><path d="${path}" fill="none" stroke="transparent" stroke-width="14" pointer-events="stroke"/>`;points.forEach(point=>html+=`<circle cx="${x(point[0])}" cy="${y(point[1])}" r="3.5" fill="${color}"><title>${esc(item.name)} · epoch ${point[0]} · ${format(point[1],definition[2])}</title></circle>`);html+='</g>'});html+=`<line x1="${left}" y1="${top+ph}" x2="${width-right}" y2="${top+ph}" stroke="#667085"/>`;[xmin,Math.round((xmin+xmax)/2),xmax].forEach(t=>html+=`<text x="${x(t)}" y="${height-13}" text-anchor="middle" font-size="10" fill="#667085">${t}</text>`);html+='</svg></div><div class="trend-legend">';series.forEach(({item,index})=>{html+=`<button type="button" class="trend-legend-item" data-trend-index="${index}" title="${esc(item.name)}"><span class="trend-swatch" style="background:${trendColor(index)}"></span><span>${esc(item.name)}</span></button>`});target.innerHTML=html+'</div>';bindTrendInteractions()}
function renderComparison(){const rows=DATA.models.map((item,index)=>({item,index,cp:checkpointFor(item)})).filter(row=>row.cp);const keys=Object.keys(metricDefs);document.getElementById('comparison').innerHTML=`<table><thead><tr><th>model</th><th>mode</th>${keys.map(key=>`<th>${esc(metricDefs[key][0])}</th>`).join('')}<th>effective</th><th>max share</th></tr></thead><tbody>${rows.map(({item,index,cp})=>`<tr${index===modelIndex?' style="font-weight:800;background:#f3f8fc"':''}><td>${esc(item.name)}</td><td>${esc(item.mode)}</td>${keys.map(key=>`<td>${format(value(cp,key),metricDefs[key][2])}</td>`).join('')}<td>${format(value(cp,'effective_codes'),2)}</td><td>${format(value(cp,'largest_code_share'),'pct')}</td></tr>`).join('')}</tbody></table>`}
function boundaryDirectionChart(direction,key,definition){const labels={early_start:'Early start · prepend previous skill',late_start:'Late start · trim skill start',early_end:'Early end · trim skill end',late_end:'Late end · append next skill'},series=DATA.models.map((item,index)=>{const cp=checkpointFor(item),sweep=cp&&cp.boundary_jitter;if(!sweep)return null;const row=sweep.directions.indexOf(direction);if(row<0)return null;const values=(sweep.metrics[key]||[])[row]||[];return{item,index,points:sweep.offsets.map((offset,column)=>[Number(offset),values[column]]).filter(point=>point[1]!=null&&Number.isFinite(Number(point[1])))}}).filter(item=>item&&item.points.length);if(!series.length)return `<article class="metric"><h3>${esc(labels[direction])}</h3><div class="empty">${key==='adjacent_code_attraction'&&(direction==='late_start'||direction==='early_end')?'이 방향은 이웃 skill을 추가하지 않아 해당 지표를 계산하지 않음.':'Sweep artifact unavailable.'}</div></article>`;const all=series.flatMap(row=>row.points),xmin=Math.min(...all.map(p=>p[0])),xmax=Math.max(...all.map(p=>p[0])),rawMax=Math.max(...all.map(p=>Number(p[1]))),ymin=0,ymax=definition[1]==='pct'?1:Math.max(rawMax*1.08,1e-6),width=650,height=260,left=52,right=16,top=15,bottom=34,pw=width-left-right,ph=height-top-bottom,x=v=>left+(v-xmin)/Math.max(xmax-xmin,1)*pw,y=v=>top+(ymax-v)/Math.max(ymax-ymin,1e-9)*ph;let svg=`<article class="metric"><h3>${esc(labels[direction])}</h3><svg class="chart" viewBox="0 0 ${width} ${height}">`;for(let i=0;i<=4;i++){const yy=top+i*ph/4,val=ymax-i*(ymax-ymin)/4;svg+=`<line x1="${left}" y1="${yy}" x2="${width-right}" y2="${yy}" stroke="#dde3eb"/><text x="${left-6}" y="${yy+4}" text-anchor="end" font-size="10" fill="#667085">${format(val,definition[1])}</text>`}series.forEach(({item,index,points})=>{const path=points.map((point,i)=>`${i?'L':'M'}${x(point[0]).toFixed(1)},${y(Number(point[1])).toFixed(1)}`).join(' '),color=trendColor(index);svg+=`<path d="${path}" fill="none" stroke="${color}" stroke-width="2"><title>${esc(item.name)}</title></path>`;points.forEach(point=>svg+=`<circle cx="${x(point[0])}" cy="${y(Number(point[1]))}" r="3" fill="${color}"><title>${esc(item.name)} · ${point[0]} frames · ${format(point[1],definition[1])}</title></circle>`)});svg+=`<line x1="${left}" y1="${top+ph}" x2="${width-right}" y2="${top+ph}" stroke="#667085"/><text x="${left}" y="${height-10}" font-size="10" fill="#667085">0</text><text x="${width-right}" y="${height-10}" text-anchor="end" font-size="10" fill="#667085">${xmax} frames</text></svg><div class="legend">${series.map(({item,index})=>`<span><i style="background:${trendColor(index)}"></i>${esc(item.name)}</span>`).join('')}</div></article>`;return svg}
function renderBoundarySweep(){const key=boundarySelect.value||'code_retention',definition=boundaryDefs[key],directions=['early_start','late_start','early_end','late_end'];document.getElementById('boundaryCharts').innerHTML=directions.map(direction=>boundaryDirectionChart(direction,key,definition)).join('');const available=DATA.models.filter(item=>{const cp=checkpointFor(item);return cp&&cp.boundary_jitter}).length;document.getElementById('boundaryStatus').textContent=available?`${checkpointTag} · ${available}/${DATA.models.length} models with sweep data · ${definition[0]}`:'이 checkpoint의 sweep artifact가 아직 없습니다. GPU encoding prepass 완료 후 자동으로 채워집니다.'}
function renderGroups(cp){const entries=Object.entries(cp.group_predictability||{});document.getElementById('groups').innerHTML=entries.map(([label,score])=>`<div class="bar-row"><span>${esc(label)}</span><div class="bar-track"><div class="bar-fill" style="width:${Math.max(0,Math.min(100,score*100))}%"></div></div><b>${format(score,3)}</b></div>`).join('')}
function renderDiagnostics(cp){const cells=Object.entries(diagnosticDefs).map(([key,label])=>{const spec=key.includes('share')||key.includes('same')?'pct':key==='used_codes'?0:3;return `<div class="metric neutral"><span>${esc(label)}</span><b>${format(value(cp,key),spec)}</b></div>`}).join('');document.getElementById('diagnostics').innerHTML=`<div class="metrics" style="grid-template-columns:repeat(2,1fr)">${cells}</div>`}
function diverging(value,limit=1){const clipped=Math.max(-limit,Math.min(limit,value))/limit,amount=Math.abs(clipped),target=clipped>=0?[190,45,55]:[45,105,180],rgb=target.map(channel=>Math.round(255+(channel-255)*amount));return `rgb(${rgb.join(',')})`}
function sequential(value){const amount=Math.max(0,Math.min(1,value)),from=[245,249,252],to=[29,104,155],rgb=from.map((channel,index)=>Math.round(channel+(to[index]-channel)*amount));return `rgb(${rgb.join(',')})`}
function axisTable(cp,kind){const matrix=kind==='rho'?cp.axis_correlations:cp.axis_strengths,features=cp.correlation_features||[],axes=matrix.length;return `<table class="heat"><thead><tr><th>feature</th>${Array.from({length:axes},(_,axis)=>`<th>axis ${axis}</th>`).join('')}</tr></thead><tbody>${features.map((feature,row)=>`<tr><td>${esc(DATA.feature_labels[feature]||feature)}</td>${Array.from({length:axes},(_,axis)=>{const score=Number(matrix[axis][row]),color=kind==='rho'?diverging(score):sequential(score);return `<td style="background:${color};color:${Math.abs(score)>.58?'#fff':'#17202a'}" title="${esc(feature)} · axis ${axis}">${score.toFixed(3)}</td>`}).join('')}</tr>`).join('')}</tbody></table>`}
function renderAxisMaps(cp){document.getElementById('correlation').innerHTML=axisTable(cp,'rho');document.getElementById('strength').innerHTML=axisTable(cp,'eta')}
function renderCodeHeatmap(cp){const features=cp.code_features||[],rows=cp.code_feature_means||[];document.getElementById('codeHeatmap').innerHTML=`<table class="heat"><thead><tr><th>code</th>${features.map(feature=>`<th>${esc(DATA.feature_labels[feature]||feature)}</th>`).join('')}</tr></thead><tbody>${rows.map((values,code)=>values?`<tr><td>#${code} · ${cp.counts[code]}</td>${values.map((score,index)=>`<td style="background:${diverging(score,2.5)};color:${Math.abs(score)>1.45?'#fff':'#17202a'}" title="${esc(features[index])}">${Number(score).toFixed(2)}</td>`).join('')}</tr>`:'').join('')}</tbody></table>`}
function renderCodes(cp){const labels=cp.direction_labels||[],colorsHere=cp.direction_colors||[];document.getElementById('directionLegend').innerHTML=labels.map((label,index)=>`<span><i style="background:${colorsHere[index]}"></i>${esc(label)}</span>`).join('');const rows=(cp.direction_composition||[]).map((composition,code)=>{if(!composition)return '';const dominant=composition.indexOf(Math.max(...composition));return `<tr><td>#${code}</td><td>${cp.counts[code]}</td><td>${format(cp.counts[code]/cp.sample_count,'pct')}</td><td><div class="direction">${composition.map((share,index)=>`<i style="width:${share*100}%;background:${colorsHere[index]}" title="${esc(labels[index])} ${format(share,'pct')}"></i>`).join('')}</div></td><td>${esc(labels[dominant])} · ${format(composition[dominant],'pct')}</td></tr>`}).join('');document.getElementById('codes').innerHTML=`<table class="code-table"><thead><tr><th>code</th><th>count</th><th>share</th><th>direction composition</th><th>dominant</th></tr></thead><tbody>${rows}</tbody></table>`}
function pairTable(kind){const current=DATA.pairwise[checkpointTag];if(!current)return '<div class="empty">Every model does not share this checkpoint.</div>';const matrix=current[kind];return `<table class="heat"><thead><tr><th></th>${DATA.models.map(item=>`<th>${esc(item.name)}</th>`).join('')}</tr></thead><tbody>${DATA.models.map((item,row)=>`<tr><td>${esc(item.name)}</td>${DATA.models.map((_,column)=>{const score=matrix[row][column];return `<td style="background:${sequential(Math.max(0,score))};color:${score>.58?'#fff':'#17202a'}">${score.toFixed(3)}</td>`}).join('')}</tr>`).join('')}</tbody></table>`}
function renderPairwise(){document.getElementById('pairNmi').innerHTML=pairTable('nmi');document.getElementById('pairAri').innerHTML=pairTable('ari')}
setup();
</script></body></html>"""
    path = output_dir / REPORT_NAME
    temporary = path.with_suffix(".html.tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(path)
    _write_boundary_jitter_report(output_dir)
    _write_terminator_report(output_dir)
    return path


def build_report(
    collection_dirs: list[str | Path],
    *,
    output_dir: str | Path,
    skill_bundle: str | Path | None = None,
) -> Path:
    payload = build_payload(
        collection_dirs,
        output_dir=output_dir,
        skill_bundle=skill_bundle,
    )
    return write_report(output_dir, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skill-bundle", type=Path, default=None)
    args = parser.parse_args()
    path = build_report(
        args.compare,
        output_dir=args.output,
        skill_bundle=args.skill_bundle,
    )
    print(f"categorization report: {path}")


if __name__ == "__main__":
    main()
