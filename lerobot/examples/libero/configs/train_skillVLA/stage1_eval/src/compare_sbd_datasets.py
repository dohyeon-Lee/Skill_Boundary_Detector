#!/usr/bin/env python3
"""Compare two SkillVLA datasets without training or model inference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "lerobot").is_dir() and (parent / "dataset_filtered").is_dir():
            return parent
    raise RuntimeError("Could not locate the Skill_Boundary_Detector project root.")


PROJECT_ROOT = _project_root()
sys.path.insert(0, str(PROJECT_ROOT / "lerobot" / "src"))

from lerobot.policies.skillVLA.skill_jitter import choose_jitter_torch  # noqa: E402


COLUMNS = [
    "episode_index",
    "frame_index",
    "task_index",
    "skill_index",
    "skill_sequence",
    "skill_sequence_len",
    "skill_ds",
    "skill_de",
    "observation.state",
    "action",
]


@dataclass
class DatasetStats:
    label: str
    root: Path
    summary: dict[str, Any]
    token_rows: list[dict[str, Any]]
    task_rows: list[dict[str, Any]]
    episode_boundaries: dict[int, np.ndarray]
    episode_task: dict[int, int]
    action: np.ndarray
    episode: np.ndarray
    frame: np.ndarray


def _resolve(path: str) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _read_table(dataset_root: Path) -> pa.Table:
    files = sorted((dataset_root / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found below {dataset_root / 'data'}")
    return pa.concat_tables([pq.read_table(path, columns=COLUMNS) for path in files])


def _scalar(table: pa.Table, name: str, dtype: np.dtype) -> np.ndarray:
    return np.asarray(table[name].combine_chunks().to_numpy(zero_copy_only=False), dtype=dtype)


def _dense_list(table: pa.Table, name: str, dtype: np.dtype) -> np.ndarray:
    array = table[name].combine_chunks()
    offsets = np.asarray(array.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
    lengths = np.diff(offsets)
    if len(lengths) == 0:
        return np.empty((0, 0), dtype=dtype)
    if not np.all(lengths == lengths[0]):
        raise ValueError(f"{name} is not fixed-width: {np.unique(lengths).tolist()}")
    values = np.asarray(array.values.to_numpy(zero_copy_only=False), dtype=dtype)
    return values.reshape(len(lengths), int(lengths[0]))


def _group_sse(values: np.ndarray, groups: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    if groups.ndim == 1:
        _, inverse = np.unique(groups, return_inverse=True)
    else:
        _, inverse = np.unique(groups, axis=0, return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    sums = np.zeros((len(count), values.shape[1]), dtype=np.float64)
    sums_sq = np.zeros_like(sums)
    np.add.at(sums, inverse, values)
    np.add.at(sums_sq, inverse, values * values)
    return float(np.maximum(sums_sq - sums * sums / count[:, None], 0.0).sum())


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _token_entropy(counts: np.ndarray, vocab_size: int) -> tuple[float, float]:
    positive = counts[counts > 0].astype(np.float64)
    if not len(positive):
        return float("nan"), float("nan")
    probability = positive / positive.sum()
    entropy = float(-(probability * np.log(probability)).sum())
    normalized = entropy / math.log(vocab_size) if vocab_size > 1 else 1.0
    return normalized, float(math.exp(entropy))


def _jitter_mask_stats(
    *,
    skill_index: np.ndarray,
    ds: np.ndarray,
    de: np.ndarray,
    sequence_len: np.ndarray,
    next_length: np.ndarray,
    info: dict[str, Any],
    chunk_size: int,
    execution_steps: int,
    samples: int,
    seed: int,
) -> dict[str, float]:
    if samples <= 0:
        return {}
    import torch

    k = torch.from_numpy(np.array(skill_index, dtype=np.int64, copy=True))
    t_ds = torch.from_numpy(np.array(ds, dtype=np.int64, copy=True))
    t_de = torch.from_numpy(np.array(de, dtype=np.int64, copy=True))
    seq = torch.from_numpy(np.array(sequence_len, dtype=np.int64, copy=True))
    next_len = torch.from_numpy(np.array(next_length, dtype=np.int64, copy=True))
    valid_total = 0.0
    exec_total = 0.0
    full_total = 0.0
    directional = {
        "early_start_pmax": int(info.get("skill_jitter_early_start_pmax", -1)),
        "late_start_pmax": int(info.get("skill_jitter_late_start_pmax", -1)),
        "early_end_pmax": int(info.get("skill_jitter_early_end_pmax", -1)),
        "late_end_pmax": int(info.get("skill_jitter_late_end_pmax", -1)),
    }
    for sample in range(samples):
        torch.manual_seed(seed + sample)
        selected, offset = choose_jitter_torch(
            k,
            t_ds,
            t_de,
            seq,
            int(info.get("skill_pmax", 0)),
            str(info.get("skill_jitter_distribution", "half_normal")),
            **directional,
        )
        effective = t_de.clone()
        late = selected == k - 1
        early = selected == k + 1
        effective[late] = offset[late].abs() - 1 - t_ds[late]
        effective[early] = t_de[early] + next_len[early]
        if bool((effective < 0).any()):
            raise ValueError("Jitter simulation produced negative effective skill_de.")
        valid = torch.clamp(effective + 1, max=chunk_size)
        valid_total += float(valid.double().mean())
        exec_total += float((effective >= execution_steps - 1).double().mean())
        full_total += float((effective >= chunk_size - 1).double().mean())
    return {
        "jitter_valid_action_steps_mean": valid_total / samples,
        "jitter_masked_action_fraction": 1.0 - valid_total / samples / chunk_size,
        "jitter_execution_window_fully_valid_fraction": exec_total / samples,
        "jitter_chunk_fully_valid_fraction": full_total / samples,
    }


def _boundary_metrics(
    action_z: np.ndarray,
    episode: np.ndarray,
    start_rows: np.ndarray,
    window: int,
    peak_quantile: float,
) -> tuple[dict[str, float], dict[int, np.ndarray]]:
    delta = np.full(len(action_z), np.nan, dtype=np.float64)
    same_episode = episode[1:] == episode[:-1]
    delta[1:][same_episode] = np.linalg.norm(action_z[1:][same_episode] - action_z[:-1][same_episode], axis=1)
    valid_delta = delta[np.isfinite(delta)]
    threshold = float(np.quantile(valid_delta, peak_quantile))
    internal = start_rows[(start_rows > 0) & (episode[start_rows] == episode[start_rows - 1])]
    scores = delta[internal]
    sorted_delta = np.sort(valid_delta)
    ranks = np.searchsorted(sorted_delta, scores, side="right") / len(sorted_delta)

    episode_starts = np.r_[0, np.flatnonzero(episode[1:] != episode[:-1]) + 1]
    episode_ends = np.r_[episode_starts[1:], len(episode)]
    bounds = {int(episode[s]): (int(s), int(e)) for s, e in zip(episode_starts, episode_ends, strict=True)}
    local_peak = []
    mean_shift = []
    nearest_peak = []
    by_episode: dict[int, list[int]] = {}
    for row in internal:
        ep = int(episode[row])
        lo, hi = bounds[ep]
        left = max(lo + 1, int(row) - window)
        right = min(hi, int(row) + window + 1)
        local_peak.append(float(np.nanmax(delta[left:right])) >= threshold)
        pre = action_z[max(lo, int(row) - window) : int(row)]
        post = action_z[int(row) : min(hi, int(row) + window)]
        mean_shift.append(float(np.linalg.norm(post.mean(axis=0) - pre.mean(axis=0))))
        peaks = np.flatnonzero((episode == ep) & (delta >= threshold))
        nearest_peak.append(float(np.min(np.abs(peaks - row))) if len(peaks) else float("nan"))
        by_episode.setdefault(ep, []).append(int(row))

    result = {
        "boundary_count": int(len(internal)),
        "boundary_action_delta_mean": float(np.mean(scores)),
        "all_action_delta_mean": float(np.mean(valid_delta)),
        "boundary_action_delta_ratio": float(np.mean(scores) / np.mean(valid_delta)),
        "boundary_delta_percentile_mean": float(np.mean(ranks) * 100.0),
        "boundary_at_global_top10_fraction": float(np.mean(scores >= threshold)),
        "boundary_window_has_top10_fraction": float(np.mean(local_peak)),
        "boundary_nearest_top10_distance_mean": float(np.nanmean(nearest_peak)),
        "boundary_pre_post_action_shift_mean": float(np.mean(mean_shift)),
    }
    return result, {ep: np.asarray(rows, dtype=np.int64) for ep, rows in by_episode.items()}


def _analyse(label: str, dataset_root: Path, cfg: dict[str, Any]) -> DatasetStats:
    table = _read_table(dataset_root)
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    episode = _scalar(table, "episode_index", np.int64)
    frame = _scalar(table, "frame_index", np.int64)
    task = _scalar(table, "task_index", np.int64)
    skill_index = _scalar(table, "skill_index", np.int64)
    sequence_len = _scalar(table, "skill_sequence_len", np.int64)
    ds = _scalar(table, "skill_ds", np.int64)
    de = _scalar(table, "skill_de", np.int64)
    state = _dense_list(table, "observation.state", np.float64)
    action = _dense_list(table, "action", np.float64)

    start_rows = np.flatnonzero(ds == 0)
    sequences = table["skill_sequence"].combine_chunks().take(pa.array(start_rows)).to_pylist()
    occurrence_index = skill_index[start_rows]
    occurrence_code = np.asarray(
        [int(sequence[int(index)]) for sequence, index in zip(sequences, occurrence_index, strict=True)],
        dtype=np.int64,
    )
    occurrence_length = de[start_rows] + 1
    occurrence_end = start_rows + occurrence_length - 1
    if np.any(occurrence_end >= len(episode)) or np.any(episode[occurrence_end] != episode[start_rows]):
        raise ValueError(f"{label}: skill occurrence crosses an episode boundary.")

    frame_code = np.full(len(episode), -1, dtype=np.int64)
    next_length = np.zeros(len(episode), dtype=np.int64)
    for idx, (start, length, code) in enumerate(
        zip(start_rows, occurrence_length, occurrence_code, strict=True)
    ):
        stop = int(start + length)
        frame_code[start:stop] = code
        if idx + 1 < len(start_rows) and episode[start_rows[idx + 1]] == episode[start]:
            next_length[start:stop] = occurrence_length[idx + 1]
    if np.any(frame_code < 0):
        raise ValueError(f"{label}: skill occurrences do not cover every parquet row.")

    chunk_size = int(cfg["chunk_size"])
    execution_steps = int(cfg["execution_steps"])
    progress_bins = int(cfg["progress_bins"])
    canonical_valid = np.minimum(chunk_size, de + 1)
    skill_length = ds + de + 1
    progress = ds / np.maximum(skill_length - 1, 1)
    progress_bin = np.minimum((progress * progress_bins).astype(np.int64), progress_bins - 1)

    action_scale = np.std(action, axis=0)
    action_scale[action_scale < 1e-8] = 1.0
    action_z = (action - np.mean(action, axis=0)) / action_scale
    global_action_sse = _group_sse(action_z, np.zeros(len(action_z), dtype=np.int64))
    token_action_sse = _group_sse(action_z, frame_code)
    base_action_groups = np.stack([task, progress_bin], axis=1)
    conditioned_action_groups = np.stack([task, frame_code, progress_bin], axis=1)

    endpoint_xyz = state[occurrence_end, :3]
    occurrence_task = task[start_rows]
    task_endpoint_sse = _group_sse(endpoint_xyz, occurrence_task)
    token_endpoint_groups = np.stack([occurrence_task, occurrence_code], axis=1)

    boundary_summary, boundary_rows = _boundary_metrics(
        action_z,
        episode,
        start_rows,
        int(cfg["boundary_window"]),
        float(cfg["action_peak_quantile"]),
    )
    episode_boundaries = {
        ep: frame[rows].copy() for ep, rows in boundary_rows.items()
    }

    vocab_size = int(info.get("skill_num_embeddings", int(frame_code.max()) + 1))
    frame_counts = np.bincount(frame_code, minlength=vocab_size)[:vocab_size]
    occurrence_counts = np.bincount(occurrence_code, minlength=vocab_size)[:vocab_size]
    frame_entropy, frame_effective = _token_entropy(frame_counts, vocab_size)
    occurrence_entropy, occurrence_effective = _token_entropy(occurrence_counts, vocab_size)

    summary: dict[str, Any] = {
        "label": label,
        "dataset": str(dataset_root),
        "frames": int(len(episode)),
        "episodes": int(len(np.unique(episode))),
        "skill_occurrences": int(len(start_rows)),
        "skills_per_episode_mean": float(len(start_rows) / len(np.unique(episode))),
        "skill_length": _percentiles(occurrence_length.astype(np.float64)),
        "skill_shorter_than_execution_fraction": float(np.mean(occurrence_length < execution_steps)),
        "skill_shorter_than_chunk_fraction": float(np.mean(occurrence_length < chunk_size)),
        "canonical_valid_action_steps_mean": float(np.mean(canonical_valid)),
        "canonical_masked_action_fraction": float(1.0 - np.mean(canonical_valid) / chunk_size),
        "canonical_execution_window_fully_valid_fraction": float(np.mean(de >= execution_steps - 1)),
        "canonical_chunk_fully_valid_fraction": float(np.mean(de >= chunk_size - 1)),
        "token_frame_entropy_normalized": frame_entropy,
        "token_frame_effective_count": frame_effective,
        "token_frame_max_share": float(frame_counts.max() / frame_counts.sum()),
        "token_occurrence_entropy_normalized": occurrence_entropy,
        "token_occurrence_effective_count": occurrence_effective,
        "token_occurrence_max_share": float(occurrence_counts.max() / occurrence_counts.sum()),
        "action_within_token_variance_ratio": _ratio(token_action_sse, global_action_sse),
        "action_within_task_token_phase_variance_ratio": _ratio(
            _group_sse(action_z, conditioned_action_groups),
            _group_sse(action_z, base_action_groups),
        ),
        "end_xyz_within_task_token_variance_ratio": _ratio(
            _group_sse(endpoint_xyz, token_endpoint_groups),
            task_endpoint_sse,
        ),
        **boundary_summary,
    }
    summary.update(
        _jitter_mask_stats(
            skill_index=skill_index,
            ds=ds,
            de=de,
            sequence_len=sequence_len,
            next_length=next_length,
            info=info,
            chunk_size=chunk_size,
            execution_steps=execution_steps,
            samples=int(cfg["jitter_samples"]),
            seed=int(cfg["random_seed"]),
        )
    )

    token_rows = []
    for code in range(vocab_size):
        mask = occurrence_code == code
        token_rows.append(
            {
                "dataset": label,
                "token": code,
                "frame_count": int(frame_counts[code]),
                "frame_share": float(frame_counts[code] / frame_counts.sum()),
                "occurrence_count": int(occurrence_counts[code]),
                "occurrence_share": float(occurrence_counts[code] / occurrence_counts.sum()),
                "skill_length_mean": float(np.mean(occurrence_length[mask])) if np.any(mask) else "",
                "skill_length_median": float(np.median(occurrence_length[mask])) if np.any(mask) else "",
            }
        )

    task_rows = []
    for task_id in np.unique(task):
        row_mask = task == task_id
        occurrence_mask = occurrence_task == task_id
        task_rows.append(
            {
                "dataset": label,
                "task_index": int(task_id),
                "frames": int(row_mask.sum()),
                "skill_occurrences": int(occurrence_mask.sum()),
                "skill_length_mean": float(np.mean(occurrence_length[occurrence_mask])),
                "valid_action_steps_mean": float(np.mean(canonical_valid[row_mask])),
                "chunk_fully_valid_fraction": float(np.mean(de[row_mask] >= chunk_size - 1)),
            }
        )

    episode_task = {
        int(ep): int(task[np.flatnonzero(episode == ep)[0]]) for ep in np.unique(episode)
    }
    return DatasetStats(
        label=label,
        root=dataset_root,
        summary=summary,
        token_rows=token_rows,
        task_rows=task_rows,
        episode_boundaries=episode_boundaries,
        episode_task=episode_task,
        action=action,
        episode=episode,
        frame=frame,
    )


def _nearest_signed(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if not len(source) or not len(target):
        return np.empty(0, dtype=np.float64)
    position = np.searchsorted(target, source)
    left = target[np.maximum(position - 1, 0)]
    right = target[np.minimum(position, len(target) - 1)]
    choose_right = np.abs(right - source) < np.abs(left - source)
    nearest = np.where(choose_right, right, left)
    return nearest.astype(np.float64) - source.astype(np.float64)


def _compare(prev: DatasetStats, new: DatasetStats) -> dict[str, Any]:
    same_layout = (
        np.array_equal(prev.episode, new.episode)
        and np.array_equal(prev.frame, new.frame)
    )
    action_max_abs_difference = (
        float(np.max(np.abs(prev.action - new.action))) if same_layout else float("nan")
    )
    common = sorted(set(prev.episode_task) & set(new.episode_task))
    new_to_prev = []
    prev_to_new = []
    count_difference = []
    for ep in common:
        prev_b = prev.episode_boundaries.get(ep, np.empty(0, dtype=np.int64))
        new_b = new.episode_boundaries.get(ep, np.empty(0, dtype=np.int64))
        new_to_prev.extend(_nearest_signed(new_b, prev_b).tolist())
        prev_to_new.extend(_nearest_signed(prev_b, new_b).tolist())
        count_difference.append(len(new_b) - len(prev_b))
    signed = np.asarray(new_to_prev, dtype=np.float64)
    reverse = np.asarray(prev_to_new, dtype=np.float64)
    absolute = np.abs(signed)
    return {
        "same_episode_frame_layout": bool(same_layout),
        "action_max_abs_difference": action_max_abs_difference,
        "common_episodes": len(common),
        "new_minus_prev_boundaries_per_episode_mean": float(np.mean(count_difference)),
        "new_boundary_nearest_prev_signed_frames_mean": float(np.mean(signed)),
        "new_boundary_nearest_prev_abs_frames_mean": float(np.mean(absolute)),
        "new_boundary_nearest_prev_abs_frames_median": float(np.median(absolute)),
        "new_boundary_nearest_prev_abs_frames_p90": float(np.quantile(absolute, 0.90)),
        "new_boundary_matches_prev_exact_fraction": float(np.mean(absolute == 0)),
        "new_boundary_within_2_frames_of_prev_fraction": float(np.mean(absolute <= 2)),
        "new_boundary_within_5_frames_of_prev_fraction": float(np.mean(absolute <= 5)),
        "prev_boundary_within_5_frames_of_new_fraction": float(np.mean(np.abs(reverse) <= 5)),
    }


def _flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten(f"{prefix}.{key}" if prefix else key, child, output)
    else:
        output[prefix] = value


def _markdown(prev: dict[str, Any], new: dict[str, Any], comparison: dict[str, Any]) -> str:
    rows = [
        ("평균 skill 길이", "skill_length.mean", "higher"),
        ("평균 유효 action / 10", "canonical_valid_action_steps_mean", "higher"),
        ("마스킹 비율", "canonical_masked_action_fraction", "lower"),
        ("5-step 완전 유효 비율", "canonical_execution_window_fully_valid_fraction", "higher"),
        ("10-step 완전 유효 비율", "canonical_chunk_fully_valid_fraction", "higher"),
        ("Jitter 포함 평균 유효 action", "jitter_valid_action_steps_mean", "higher"),
        ("Token 최대 점유율", "token_frame_max_share", "lower"),
        ("Action 분산: token / 전체", "action_within_token_variance_ratio", "lower"),
        ("Action 분산: task+token+phase / task+phase", "action_within_task_token_phase_variance_ratio", "lower"),
        ("End XYZ 분산: task+token / task", "end_xyz_within_task_token_variance_ratio", "lower"),
        ("Boundary action-change percentile", "boundary_delta_percentile_mean", "higher"),
        ("Boundary ±window 내 top-10% 변화 비율", "boundary_window_has_top10_fraction", "higher"),
        ("Boundary 전후 action 평균 변화", "boundary_pre_post_action_shift_mean", "higher"),
    ]

    def get(source: dict[str, Any], dotted: str) -> float:
        value: Any = source
        for part in dotted.split("."):
            value = value[part]
        return float(value)

    lines = [
        "# SBD dataset comparison",
        "",
        "재학습 없이 parquet의 canonical skill 구간과 현재 jitter 계약을 분석한 결과입니다.",
        "",
        "| 지표 | prev | new | new-prev | 선호 방향 |",
        "|---|---:|---:|---:|---|",
    ]
    for name, key, direction in rows:
        p = get(prev, key)
        n = get(new, key)
        lines.append(f"| {name} | {p:.6f} | {n:.6f} | {n-p:+.6f} | {direction} |")
    lines.extend(
        [
            "",
            "## Boundary correspondence",
            "",
            f"- 동일 episode/frame/action: `{comparison['same_episode_frame_layout']}` / max action diff `{comparison['action_max_abs_difference']:.3g}`",
            f"- new boundary와 가장 가까운 prev boundary 거리: 평균 `{comparison['new_boundary_nearest_prev_abs_frames_mean']:.3f}` frame, median `{comparison['new_boundary_nearest_prev_abs_frames_median']:.3f}` frame",
            f"- ±2 frame 일치율: `{comparison['new_boundary_within_2_frames_of_prev_fraction']:.3%}`",
            f"- ±5 frame 일치율: `{comparison['new_boundary_within_5_frames_of_prev_fraction']:.3%}`",
            "",
            "## 해석 기준",
            "",
            "- 마스킹 가설은 유효 action 수와 5/10-step 완전 유효 비율 차이가 클 때 지지됩니다.",
            "- skill 표현 일관성은 action/end-XYZ 분산 비율이 낮을수록 좋습니다.",
            "- boundary-action 정렬은 percentile, local top-10%, 전후 action 변화가 높을수록 강합니다.",
            "- 이 통계는 상관 근거이며 단독으로 성공률 차이의 인과를 확정하지 않습니다.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text())
    analysis = config["analysis"]

    prev = _analyse("prev", _resolve(config["datasets"]["prev"]), analysis)
    new = _analyse("new", _resolve(config["datasets"]["new"]), analysis)
    comparison = _compare(prev, new)

    output_dir = _resolve(config["output"]["directory"]) / str(config["output"]["name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"config": config, "prev": prev.summary, "new": new.summary, "comparison": comparison}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    (output_dir / "report.md").write_text(_markdown(prev.summary, new.summary, comparison))
    _write_csv(output_dir / "token_stats.csv", prev.token_rows + new.token_rows)
    _write_csv(output_dir / "task_stats.csv", prev.task_rows + new.task_rows)

    flat_prev: dict[str, Any] = {}
    flat_new: dict[str, Any] = {}
    _flatten("", prev.summary, flat_prev)
    _flatten("", new.summary, flat_new)
    print(f"Wrote: {output_dir}")
    for key in (
        "canonical_valid_action_steps_mean",
        "jitter_valid_action_steps_mean",
        "action_within_task_token_phase_variance_ratio",
        "end_xyz_within_task_token_variance_ratio",
        "boundary_delta_percentile_mean",
        "boundary_window_has_top10_fraction",
    ):
        print(f"{key}: prev={flat_prev[key]:.6f} new={flat_new[key]:.6f} delta={flat_new[key]-flat_prev[key]:+.6f}")


if __name__ == "__main__":
    main()
