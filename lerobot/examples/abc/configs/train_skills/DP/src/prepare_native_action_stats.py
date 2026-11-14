#!/usr/bin/env python3
"""Cache transformed joint-action statistics for an ABC native task selection."""

from __future__ import annotations

import argparse
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


FORMATS = {
    "anchor_relative": "sbd_abc_sim_anchor_relative_stats/v1",
    "delta_joint": "sbd_abc_sim_delta_joint_stats/v1",
}
DEFAULT_SAMPLE_CAP = 1_000_000


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
        length = len(actions)
        for offset in range(min(chunk_size, length)):
            relative = np.asarray(actions[offset:], dtype=np.float64).copy()
            if action_mode == "anchor_relative" or offset == 0:
                reference = np.asarray(states[: length - offset, mask], dtype=np.float64)
            else:
                reference = np.asarray(actions[offset - 1 : length - 1, mask], dtype=np.float64)
            relative[:, mask] -= reference
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
            "episode_count": len(episodes),
            "chunk_size": chunk_size,
            "exclude_joints": exclude_tokens,
            "action_names": action_names,
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
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    tasks = [str(value) for value in json.loads(args.tasks_json)]
    exclude_tokens = [value.strip() for value in args.exclude_joints.split(",") if value.strip()]
    output = prepare(
        args.dataset_dir, tasks, args.chunk_size, exclude_tokens, action_mode=args.action_mode
    )
    print(output)


if __name__ == "__main__":
    main()
