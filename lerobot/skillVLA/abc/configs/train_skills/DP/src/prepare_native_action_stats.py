#!/usr/bin/env python3
"""Cache transformed joint/delta-pose statistics for an ABC native task selection."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np

from lerobot.datasets.abc_native_io import (
    abc_native_raw_root,
    load_abc_native_episodes,
    load_abc_native_marker,
    load_abc_native_state_action,
)
from lerobot.processor.yam_kinematics import (
    YAM_DELTA_POSE_NAMES,
    to_yam_delta_pose_actions_numpy,
)


FORMATS = {
    "anchor_relative": "sbd_abc_sim_anchor_relative_stats/v1",
    "delta_joint": "sbd_abc_sim_delta_joint_stats/v1",
    "delta_pose": "sbd_abc_sim_delta_pose_stats/v1",
}
DEFAULT_SAMPLE_CAP = 1_000_000
ABC_EPISODE_SELECTION_FORMAT = "sbd_abc_sim_episode_selection/v1"


def _action_mask(action_names: list[str], exclude_tokens: list[str]) -> np.ndarray:
    lowered = [token.lower() for token in exclude_tokens if token]
    return np.asarray(
        [not any(token == name.lower() or token in name.lower() for token in lowered) for name in action_names],
        dtype=bool,
    )


def _fingerprint(
    marker: dict,
    episodes,
    tasks: list[str],
    chunk_size: int,
    exclude_tokens: list[str],
    action_names: list[str],
    action_mode: str,
) -> str:
    digest = hashlib.sha256()
    header = {
        "format": FORMATS[action_mode],
        "action_mode": action_mode,
        "dataset_request_sha256": marker.get("request_sha256"),
        "tasks": sorted(tasks),
        "task_sha256": {
            task: (marker.get("task_sha256") or {}).get(task) for task in sorted(tasks)
        },
        "chunk_size": chunk_size,
        "exclude_joints": exclude_tokens,
        "action_names": action_names,
    }
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    for _, row in episodes.sort_values("episode_index").iterrows():
        value = {
            "episode_index": int(row["episode_index"]),
            "length": int(row["length"]),
            "source_task": str(row["source_task"]),
            "states_actions_path": str(row["states_actions_path"]),
        }
        digest.update(b"\0")
        digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def _cache_path(dataset_dir: Path, tasks: list[str], fingerprint: str, action_mode: str) -> Path:
    if len(tasks) == 1:
        label = "task-" + re.sub(r"[^A-Za-z0-9_.-]+", "-", tasks[0]).strip("-")
    elif tasks:
        label = f"tasks-{len(tasks)}"
    else:
        label = "all"
    return dataset_dir / "meta" / f"{action_mode}_action_stats" / f"{label}-{fingerprint[:12]}.json"


def _iter_episode_relative_chunks(
    states: np.ndarray,
    actions: np.ndarray,
    chunk_size: int,
    mask: np.ndarray,
    action_mode: str,
) -> Iterator[np.ndarray]:
    """Yield the transformed arrays represented at every action-chunk offset.

    For delta modes, offsets after zero are suffixes of the same consecutive
    action-to-action deltas.  Compute those deltas once per episode instead of
    repeating FK (for delta_pose) or subtraction (for delta_joint) for every
    offset.  Yielding the suffixes in the original order preserves both the
    per-position weighting and deterministic quantile sampling.
    """
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    length = len(actions)
    offset_count = min(chunk_size, length)
    if offset_count == 0:
        return

    if action_mode == "delta_pose":
        yield to_yam_delta_pose_actions_numpy(actions, states[:length])
        if offset_count > 1:
            consecutive = to_yam_delta_pose_actions_numpy(actions[1:], actions[:-1])
            for offset in range(1, offset_count):
                yield consecutive[offset - 1 :]
        return

    if action_mode == "delta_joint":
        current = actions.copy()
        current[:, mask] -= states[:length, mask]
        yield current
        if offset_count > 1:
            consecutive = actions[1:].copy()
            consecutive[:, mask] -= actions[:-1, mask]
            for offset in range(1, offset_count):
                yield consecutive[offset - 1 :]
        return

    for offset in range(offset_count):
        relative = actions[offset:].copy()
        relative[:, mask] -= states[: length - offset, mask]
        yield relative


def _compute(
    dataset_dir: Path,
    marker: dict,
    episodes,
    chunk_size: int,
    mask: np.ndarray,
    sample_cap: int,
    action_mode: str,
) -> tuple[dict[str, list[float] | list[int]], int, int]:
    total_pairs = sum(
        sum(max(0, int(row["length"]) - offset) for offset in range(chunk_size))
        for _, row in episodes.iterrows()
    )
    if total_pairs <= 0:
        raise ValueError("ABC relative-action selection contains no valid action chunks.")
    stride = max(1, math.ceil(total_pairs / sample_cap))
    action_dim = int(marker["action_dim"])
    minimum = np.full(action_dim, np.inf, dtype=np.float64)
    maximum = np.full(action_dim, -np.inf, dtype=np.float64)
    total = np.zeros(action_dim, dtype=np.float64)
    total_sq = np.zeros(action_dim, dtype=np.float64)
    samples: list[np.ndarray] = []
    count = 0
    sample_offset = 0
    raw_root = abc_native_raw_root(dataset_dir, marker)

    for position, (_, row) in enumerate(
        episodes.sort_values("episode_index").iterrows(), start=1
    ):
        states, actions = load_abc_native_state_action(
            dataset_dir, row, marker=marker, raw_root=raw_root
        )
        for relative in _iter_episode_relative_chunks(
            states, actions, chunk_size, mask, action_mode
        ):
            minimum = np.minimum(minimum, relative.min(axis=0))
            maximum = np.maximum(maximum, relative.max(axis=0))
            total += relative.sum(axis=0)
            total_sq += np.square(relative).sum(axis=0)
            first = (-sample_offset) % stride
            if first < len(relative):
                samples.append(relative[first::stride].astype(np.float32))
            count += len(relative)
            sample_offset += len(relative)
        if position % 250 == 0 or position == len(episodes):
            print(
                f"[relative-stats] scanned {position}/{len(episodes)} episodes",
                file=sys.stderr,
                flush=True,
            )

    sampled = np.concatenate(samples, axis=0)
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    q01, q99 = np.quantile(sampled, [0.01, 0.99], axis=0)
    return {
        "min": minimum.tolist(),
        "max": maximum.tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "q01": q01.astype(np.float64).tolist(),
        "q99": q99.astype(np.float64).tolist(),
        "count": [count],
    }, stride, count


def prepare(
    dataset_dir: Path,
    tasks: list[str],
    chunk_size: int,
    exclude_tokens: list[str],
    action_mode: str = "anchor_relative",
    episode_selection_manifest: Path | None = None,
) -> Path:
    dataset_dir = dataset_dir.resolve()
    marker = load_abc_native_marker(dataset_dir)
    episodes = load_abc_native_episodes(dataset_dir)
    available = sorted(set(episodes["source_task"].astype(str)))
    if tasks:
        missing = sorted(set(tasks) - set(available))
        if missing:
            raise ValueError(f"Unknown ABC source task(s) {missing}; available={available}")
        episodes = episodes[episodes["source_task"].astype(str).isin(tasks)].copy()
    else:
        tasks = available
    selection_payload = None
    if episode_selection_manifest is not None:
        episode_selection_manifest = episode_selection_manifest.expanduser().resolve()
        selection_payload = json.loads(
            episode_selection_manifest.read_text(encoding="utf-8")
        )
        if selection_payload.get("format") != ABC_EPISODE_SELECTION_FORMAT:
            raise ValueError(
                f"Unsupported ABC episode selection format in {episode_selection_manifest}."
            )
        expected_request = marker.get("request_sha256")
        actual_request = selection_payload.get("dataset_request_sha256")
        if expected_request and actual_request and expected_request != actual_request:
            raise ValueError(
                f"Episode selection {episode_selection_manifest} belongs to a different dataset build."
            )
        manifest_tasks = [str(value) for value in selection_payload.get("tasks", [])]
        if manifest_tasks != tasks:
            raise ValueError(
                f"Episode selection tasks {manifest_tasks} do not match requested tasks {tasks}."
            )
        episode_ids = [int(value) for value in selection_payload.get("episodes", [])]
        if not episode_ids or len(episode_ids) != len(set(episode_ids)):
            raise ValueError(f"Invalid episode IDs in {episode_selection_manifest}.")
        available_ids = set(int(value) for value in episodes["episode_index"])
        missing_ids = sorted(set(episode_ids) - available_ids)
        if missing_ids:
            raise ValueError(
                f"Episode selection contains IDs outside the requested tasks: {missing_ids[:10]}."
            )
        episodes = episodes[episodes["episode_index"].isin(episode_ids)].copy()
        if len(episodes) != int(selection_payload.get("episode_count", -1)):
            raise ValueError(
                f"Episode selection resolved to {len(episodes)} rows, expected "
                f"{selection_payload.get('episode_count')}."
            )
    if episodes.empty:
        raise ValueError("ABC relative-action task selection is empty.")

    info = json.loads((dataset_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    action_names = list(info["features"]["action"].get("names") or [])
    if len(action_names) != int(marker["action_dim"]):
        raise ValueError("ABC action names are required to keep grippers absolute.")
    mask = _action_mask(action_names, exclude_tokens)
    if action_mode not in FORMATS:
        raise ValueError(f"action_mode must be one of {sorted(FORMATS)}, got {action_mode!r}")
    fingerprint = _fingerprint(
        marker, episodes, tasks, chunk_size, exclude_tokens, action_names, action_mode
    )
    output = _cache_path(dataset_dir, tasks, fingerprint, action_mode)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_suffix(output.suffix + ".lock")

    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if output.is_file():
            try:
                payload = json.loads(output.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict) and payload.get("fingerprint") == fingerprint:
                return output

        sample_cap = int(marker.get("quantile_sample_cap", DEFAULT_SAMPLE_CAP))
        stats, stride, count = _compute(
            dataset_dir, marker, episodes, chunk_size, mask, max(1, sample_cap), action_mode
        )
        payload = {
            "format": FORMATS[action_mode],
            "action_mode": action_mode,
            "fingerprint": fingerprint,
            "tasks": tasks,
            "episode_selection_manifest": (
                str(episode_selection_manifest) if episode_selection_manifest else None
            ),
            "episode_selection_fingerprint": (
                selection_payload.get("fingerprint") if selection_payload else None
            ),
            "episode_count": len(episodes),
            "chunk_size": chunk_size,
            "exclude_joints": exclude_tokens,
            "action_names": (
                list(YAM_DELTA_POSE_NAMES) if action_mode == "delta_pose" else action_names
            ),
            "source_action_names": action_names,
            "mask": mask.tolist(),
            "quantile_sample_stride": stride,
            "relative_pair_count": count,
            "action": stats,
        }
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--tasks-json", default="[]")
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--exclude-joints", default="gripper")
    parser.add_argument("--action-mode", choices=sorted(FORMATS), required=True)
    parser.add_argument("--episode-selection-manifest", type=Path)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    tasks = [str(value) for value in json.loads(args.tasks_json)]
    exclude_tokens = [value.strip() for value in args.exclude_joints.split(",") if value.strip()]
    output = prepare(
        args.dataset_dir,
        tasks,
        args.chunk_size,
        exclude_tokens,
        action_mode=args.action_mode,
        episode_selection_manifest=args.episode_selection_manifest,
    )
    print(output)


if __name__ == "__main__":
    main()
