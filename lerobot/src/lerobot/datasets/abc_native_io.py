"""Low-level episode I/O shared by ABC-native training utilities.

The storage-light ABC view keeps metadata in LeRobot form while state/action
rows and the stacked camera video remain in the official release tree.  These
helpers expose complete episodes without materializing a second frame table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from lerobot.datasets.abc_native_dataset import ABC_NATIVE_FORMAT


def is_abc_native_dataset(dataset_dir: str | Path) -> bool:
    marker_path = Path(dataset_dir) / "native_dataset.json"
    if not marker_path.is_file():
        return False
    try:
        return json.loads(marker_path.read_text(encoding="utf-8")).get("format") == ABC_NATIVE_FORMAT
    except (OSError, json.JSONDecodeError):
        return False


def load_abc_native_marker(dataset_dir: str | Path) -> dict:
    dataset_dir = Path(dataset_dir).resolve()
    marker_path = dataset_dir / "native_dataset.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("format") != ABC_NATIVE_FORMAT:
        raise ValueError(
            f"Unsupported ABC native format {marker.get('format')!r} in {marker_path}."
        )
    return marker


def abc_native_raw_root(dataset_dir: str | Path, marker: dict | None = None) -> Path:
    dataset_dir = Path(dataset_dir).resolve()
    marker = load_abc_native_marker(dataset_dir) if marker is None else marker
    raw_root = (dataset_dir / str(marker["raw_root"])).resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"ABC native raw root not found: {raw_root}")
    return raw_root


def load_abc_native_episodes(dataset_dir: str | Path) -> pd.DataFrame:
    files = sorted((Path(dataset_dir) / "meta" / "episodes").rglob("file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"ABC native episode metadata not found under {dataset_dir}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def _safe_source_path(raw_root: Path, relative: str, *, kind: str) -> Path:
    path = (raw_root / relative).resolve()
    try:
        path.relative_to(raw_root)
    except ValueError as error:
        raise ValueError(f"ABC native {kind} path escapes raw_root: {path}") from error
    if not path.is_file():
        raise FileNotFoundError(f"ABC native {kind} file not found: {path}")
    return path


def load_abc_native_state_action(
    dataset_dir: str | Path,
    episode_row: pd.Series | dict,
    *,
    marker: dict | None = None,
    raw_root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one complete episode as float32 ``(states, actions)`` arrays."""
    dataset_dir = Path(dataset_dir).resolve()
    marker = load_abc_native_marker(dataset_dir) if marker is None else marker
    raw_root = abc_native_raw_root(dataset_dir, marker) if raw_root is None else raw_root
    state_dim = int(marker["state_dim"])
    action_dim = int(marker["action_dim"])
    length = int(episode_row["length"])
    path = _safe_source_path(
        raw_root, str(episode_row["states_actions_path"]), kind="state/action"
    )
    expected_bytes = length * (state_dim + action_dim) * np.dtype("<f8").itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError(
            f"ABC native state/action size mismatch for episode "
            f"{int(episode_row['episode_index'])}: {path.stat().st_size} != {expected_bytes}"
        )
    values = np.memmap(
        path, mode="r", dtype="<f8", shape=(length, state_dim + action_dim)
    )
    return (
        np.asarray(values[:, :state_dim], dtype=np.float32),
        np.asarray(values[:, state_dim:], dtype=np.float32),
    )


def iter_abc_native_state_action(
    dataset_dir: str | Path,
    *,
    episode_ids: Iterable[int] | None = None,
    source_tasks: Iterable[str] | None = None,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield ``(episode_index, actions, states)`` in metadata order."""
    dataset_dir = Path(dataset_dir).resolve()
    marker = load_abc_native_marker(dataset_dir)
    raw_root = abc_native_raw_root(dataset_dir, marker)
    episodes = load_abc_native_episodes(dataset_dir)
    if episode_ids is not None:
        wanted = {int(value) for value in episode_ids}
        episodes = episodes[episodes["episode_index"].isin(wanted)]
    if source_tasks is not None:
        wanted_tasks = {str(value) for value in source_tasks}
        episodes = episodes[episodes["source_task"].astype(str).isin(wanted_tasks)]
    for _, row in episodes.sort_values("episode_index").iterrows():
        states, actions = load_abc_native_state_action(
            dataset_dir, row, marker=marker, raw_root=raw_root
        )
        yield int(row["episode_index"]), actions, states


def abc_native_task_indices(
    dataset_dir: str | Path, source_tasks: Iterable[str]
) -> tuple[list[int], int]:
    episodes = load_abc_native_episodes(dataset_dir)
    wanted = {str(value) for value in source_tasks}
    available = set(episodes["source_task"].astype(str))
    missing = sorted(wanted - available)
    if missing:
        raise ValueError(f"Unknown ABC source task(s) {missing}; available={sorted(available)}")
    selected = episodes[episodes["source_task"].astype(str).isin(wanted)]
    task_indices = sorted(
        {int(value) for values in selected["task_indices"] for value in values}
    )
    return task_indices, len(selected)


def abc_native_episode_frame_table(
    dataset_dir: str | Path, episode_ids: Iterable[int]
) -> pd.DataFrame:
    """Materialize only requested episodes in the DataFrame contract used by SBD."""
    dataset_dir = Path(dataset_dir).resolve()
    episodes = load_abc_native_episodes(dataset_dir).set_index("episode_index", drop=False)
    rows: list[pd.DataFrame] = []
    marker = load_abc_native_marker(dataset_dir)
    raw_root = abc_native_raw_root(dataset_dir, marker)
    fps = int(json.loads((dataset_dir / "meta" / "info.json").read_text())["fps"])
    for episode_id in (int(value) for value in episode_ids):
        if episode_id not in episodes.index:
            continue
        row = episodes.loc[episode_id]
        states, actions = load_abc_native_state_action(
            dataset_dir, row, marker=marker, raw_root=raw_root
        )
        length = len(states)
        task_indices = np.empty(length, dtype=np.int64)
        segments = json.loads(str(row["prompt_segments_json"]))
        for index, segment in enumerate(segments):
            start = int(segment["frame"])
            end = int(segments[index + 1]["frame"]) if index + 1 < len(segments) else length
            task_indices[start:end] = int(segment["task_index"])
        rows.append(
            pd.DataFrame(
                {
                    "episode_index": np.full(length, episode_id, dtype=np.int64),
                    "frame_index": np.arange(length, dtype=np.int64),
                    "timestamp": np.arange(length, dtype=np.float64) / fps,
                    "task_index": task_indices,
                    "observation.state": list(states),
                    "action": list(actions),
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

