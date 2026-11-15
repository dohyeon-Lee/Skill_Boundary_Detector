"""Deterministic episode-selection manifests for ABC native datasets."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from lerobot.datasets.abc_native_io import (
    load_abc_native_episodes,
    load_abc_native_marker,
)


ABC_EPISODE_SELECTION_FORMAT = "sbd_abc_sim_episode_selection/v1"


def _episode_counts(tasks: list[str], value: Any) -> dict[str, int]:
    if isinstance(value, bool):
        raise ValueError("episodes_per_task must be a positive integer or a task-to-count mapping.")
    if isinstance(value, int):
        counts = {task: int(value) for task in tasks}
    elif isinstance(value, dict):
        normalized = {str(key).strip(): int(count) for key, count in value.items()}
        missing = sorted(set(tasks) - set(normalized))
        extra = sorted(set(normalized) - set(tasks))
        if missing or extra:
            raise ValueError(
                "episodes_per_task mapping must name every selected task exactly once; "
                f"missing={missing}, extra={extra}."
            )
        counts = {task: normalized[task] for task in tasks}
    else:
        raise ValueError("episodes_per_task must be a positive integer or a task-to-count mapping.")
    invalid = {task: count for task, count in counts.items() if count <= 0}
    if invalid:
        raise ValueError(f"episodes_per_task values must be positive, got {invalid}.")
    return counts


def _stable_random_rank(seed: int, task: str, episode_index: int) -> bytes:
    """Version-independent pseudorandom ordering for a task's episode IDs."""
    return hashlib.sha256(f"{seed}\0{task}\0{episode_index}".encode()).digest()


def _source_fingerprint(marker: dict, episodes: pd.DataFrame, tasks: list[str]) -> str:
    digest = hashlib.sha256()
    header = {
        "dataset_format": marker.get("format"),
        "dataset_request_sha256": marker.get("request_sha256"),
        "task_sha256": {
            task: (marker.get("task_sha256") or {}).get(task) for task in sorted(tasks)
        },
    }
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    for _, row in episodes.sort_values("episode_index").iterrows():
        record = {
            "episode_index": int(row["episode_index"]),
            "length": int(row["length"]),
            "source_task": str(row["source_task"]),
            "states_actions_path": str(row["states_actions_path"]),
        }
        digest.update(b"\0")
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def _selection_fingerprint(
    *,
    name: str,
    tasks: list[str],
    counts: dict[str, int],
    sampling: str,
    seed: int,
    source_fingerprint: str,
) -> str:
    payload = {
        "format": ABC_EPISODE_SELECTION_FORMAT,
        "name": name,
        "tasks": tasks,
        "episodes_per_task": counts,
        "sampling": sampling,
        "seed": seed,
        "source_fingerprint": source_fingerprint,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def ensure_abc_episode_selection_manifest(
    dataset_dir: str | Path,
    *,
    name: str,
    tasks: list[str],
    episodes_per_task: int | dict[str, int],
    sampling: str = "random",
    seed: int = 42,
) -> tuple[Path, dict[str, Any]]:
    """Create or validate an immutable deterministic ABC episode selection."""
    dataset_dir = Path(dataset_dir).resolve()
    name = str(name).strip()
    if not name:
        raise ValueError("ABC training-set name must not be empty.")
    tasks = [str(task).strip() for task in tasks]
    if not tasks or any(not task for task in tasks) or len(tasks) != len(set(tasks)):
        raise ValueError("ABC training-set tasks must be a non-empty list without duplicates.")
    sampling = str(sampling).strip().lower()
    if sampling != "random":
        raise ValueError("ABC training sets currently support sampling: random only.")
    seed = int(seed)
    counts = _episode_counts(tasks, episodes_per_task)

    marker = load_abc_native_marker(dataset_dir)
    all_episodes = load_abc_native_episodes(dataset_dir)
    task_series = all_episodes["source_task"].astype(str)
    available = sorted(set(task_series))
    missing = sorted(set(tasks) - set(available))
    if missing:
        raise ValueError(f"Unknown ABC training-set task(s) {missing}; available={available}.")
    candidates = all_episodes[task_series.isin(tasks)].copy()
    source_fingerprint = _source_fingerprint(marker, candidates, tasks)
    fingerprint = _selection_fingerprint(
        name=name,
        tasks=tasks,
        counts=counts,
        sampling=sampling,
        seed=seed,
        source_fingerprint=source_fingerprint,
    )

    output_dir = dataset_dir / "meta" / "episode_selections"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(character if character.isalnum() or character in "_.-" else "-" for character in name)
    output = output_dir / f"{safe_name}-{fingerprint[:12]}.json"
    lock_path = output.with_suffix(output.suffix + ".lock")

    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if output.is_file():
            payload = json.loads(output.read_text(encoding="utf-8"))
            if (
                payload.get("format") != ABC_EPISODE_SELECTION_FORMAT
                or payload.get("fingerprint") != fingerprint
            ):
                raise ValueError(f"ABC episode selection manifest is inconsistent: {output}")
            return output, payload

        selected_by_task: dict[str, list[int]] = {}
        selected_rows: list[pd.DataFrame] = []
        for task in tasks:
            task_rows = candidates[candidates["source_task"].astype(str) == task].copy()
            task_rows = task_rows.sort_values("episode_index")
            requested = counts[task]
            if len(task_rows) < requested:
                raise ValueError(
                    f"ABC training set {name!r} requests {requested} episodes from {task!r}, "
                    f"but only {len(task_rows)} are available."
                )
            ranked_indices = sorted(
                (int(value) for value in task_rows["episode_index"]),
                key=lambda episode_index: _stable_random_rank(seed, task, episode_index),
            )
            chosen_ids = sorted(ranked_indices[:requested])
            chosen = task_rows[task_rows["episode_index"].isin(chosen_ids)].copy()
            if len(chosen) != requested:
                raise RuntimeError(f"Failed to select {requested} unique episodes for {task!r}.")
            selected_by_task[task] = chosen_ids
            selected_rows.append(chosen)

        selected = pd.concat(selected_rows, ignore_index=True).sort_values("episode_index")
        episode_ids = [int(value) for value in selected["episode_index"]]
        payload = {
            "format": ABC_EPISODE_SELECTION_FORMAT,
            "name": name,
            "fingerprint": fingerprint,
            "source_fingerprint": source_fingerprint,
            "dataset_request_sha256": marker.get("request_sha256"),
            "tasks": tasks,
            "episodes_per_task": counts,
            "sampling": sampling,
            "seed": seed,
            "episode_count": len(episode_ids),
            "frame_count": int(selected["length"].sum()),
            "episodes": episode_ids,
            "episodes_by_task": selected_by_task,
        }
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    return output, payload

