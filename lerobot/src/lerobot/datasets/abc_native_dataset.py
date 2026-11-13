#!/usr/bin/env python

"""Random-access adapter for the storage-light ABC Sim native dataset.

The native view keeps LeRobot-compatible metadata, but deliberately references
the official ABC episode files instead of duplicating every state/action row and
camera stream into LeRobot parquet/video shards.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Callable
import fcntl
import hashlib
import logging
import math
import os
from pathlib import Path
import re

import numpy as np
import torch

from lerobot.datasets.dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import check_delta_timestamps, get_delta_indices
from lerobot.datasets.video_utils import decode_video_frames, get_safe_default_codec

ABC_NATIVE_FORMAT = "sbd_abc_sim_native/v1"
ABC_SELECTION_STATS_FORMAT = "sbd_abc_sim_selection_stats/v1"
_STATE_KEY = "observation.state"
_ACTION_KEY = "action"
_DEFAULT_QUANTILE_SAMPLE_CAP = 1_000_000

logger = logging.getLogger(__name__)


class ABCNativeDataset(torch.utils.data.Dataset):
    """Read official ABC Sim episodes through the ordinary LeRobot item contract.

    The constructor intentionally mirrors the read-side arguments of
    :class:`LeRobotDataset`, so the generic dataset factory can select this
    adapter without policy-specific code.

    ABC trains on complete 30-step action chunks and omits the final 29 possible
    start frames of each episode. ``require_complete_action_chunks`` applies the
    same rule for whichever positive action deltas the policy requests. Set it
    to ``False`` to recover standard LeRobot boundary-clamping plus ``is_pad``.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        episodes: list[int] | None = None,
        tasks: list[str] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        video_backend: str | None = None,
        video_keys_to_load: list[str] | None = None,
        require_complete_action_chunks: bool = True,
        memmap_cache_size: int = 32,
    ) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root).resolve()
        self.revision = revision or CODEBASE_VERSION
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = float(tolerance_s)
        self.video_backend = video_backend or get_safe_default_codec()
        self.video_keys_to_load = video_keys_to_load
        self.require_complete_action_chunks = require_complete_action_chunks
        self._memmap_cache_size = max(1, int(memmap_cache_size))
        self._memmaps: OrderedDict[int, np.memmap] = OrderedDict()

        self.meta = LeRobotDatasetMetadata(repo_id, root=self.root, revision=self.revision)
        self._marker = self._load_marker()
        self.raw_root = self._resolve_raw_root(self._marker)
        self.state_dim = int(self._marker["state_dim"])
        self.action_dim = int(self._marker["action_dim"])
        source_size = self._marker["source_image_size"]
        self.image_width, self.image_height = [int(value) for value in source_size]
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError(f"ABC native source_image_size must be positive, got {source_size!r}")

        self.delta_indices: dict[str, list[int]] | None = None
        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, self.meta.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, self.meta.fps)
            unknown = sorted(set(self.delta_indices) - set(self.meta.features))
            if unknown:
                raise KeyError(f"ABC native delta_timestamps contain unknown features: {unknown}")

        selected_tasks = None if tasks is None else [str(task).strip() for task in tasks]
        if selected_tasks is not None:
            if not selected_tasks or any(not task for task in selected_tasks):
                raise ValueError("tasks must contain at least one non-empty official source-task ID")
            if len(set(selected_tasks)) != len(selected_tasks):
                raise ValueError("tasks must not contain duplicates")

        if episodes is None:
            requested = list(range(self.meta.total_episodes))
        else:
            requested = [int(index) for index in episodes]
        if len(set(requested)) != len(requested):
            raise ValueError("episodes must not contain duplicates")
        if any(index < 0 or index >= self.meta.total_episodes for index in requested):
            raise IndexError(
                f"Episode selection must be within [0, {self.meta.total_episodes}), got {requested[:10]}"
            )
        if selected_tasks is not None:
            available_tasks = sorted(
                {
                    str(self.meta.episodes[index]["source_task"])
                    for index in range(self.meta.total_episodes)
                }
            )
            unknown = sorted(set(selected_tasks) - set(available_tasks))
            if unknown:
                raise ValueError(
                    f"Unknown ABC native source task(s): {unknown}. Available tasks: {available_tasks}"
                )
            selected_set = set(selected_tasks)
            requested = [
                index
                for index in requested
                if str(self.meta.episodes[index]["source_task"]) in selected_set
            ]
            if not requested:
                raise ValueError(
                    "ABC native task and episode filters have no episodes in common: "
                    f"tasks={selected_tasks}, episodes={episodes}"
                )
        self.tasks = selected_tasks
        self.episodes = requested if episodes is not None or selected_tasks is not None else None
        self._episode_rows = [self.meta.episodes[index] for index in requested]
        self.selection_stats_path: Path | None = None
        if selected_tasks is not None:
            self._apply_selection_stats(selected_tasks)

        action_lookahead = 0
        if self.require_complete_action_chunks and self.delta_indices is not None:
            action_lookahead = max(0, max(self.delta_indices.get(_ACTION_KEY, [0]), default=0))
        self._usable_lengths = np.asarray(
            [max(0, int(row["length"]) - action_lookahead) for row in self._episode_rows],
            dtype=np.int64,
        )
        self._cumulative_lengths = np.cumsum(self._usable_lengths)

        available_cameras = set(self.meta.camera_keys)
        if video_keys_to_load is None:
            self._camera_keys = list(self.meta.camera_keys)
        else:
            missing = sorted(set(video_keys_to_load) - available_cameras)
            if missing:
                raise KeyError(f"ABC native video_keys_to_load contains unknown cameras: {missing}")
            self._camera_keys = list(video_keys_to_load)

    @staticmethod
    def _stats_to_numpy(stats: dict) -> dict[str, dict[str, np.ndarray]]:
        return {
            feature: {name: np.asarray(value) for name, value in feature_stats.items()}
            for feature, feature_stats in stats.items()
        }

    def _selection_stats_fingerprint(self, selected_tasks: list[str]) -> str:
        digest = hashlib.sha256()
        header = {
            "format": ABC_SELECTION_STATS_FORMAT,
            "dataset_format": self._marker.get("format"),
            "dataset_request_sha256": self._marker.get("request_sha256"),
            "task_sha256": {
                task: (self._marker.get("task_sha256") or {}).get(task)
                for task in sorted(selected_tasks)
            },
            "tasks": sorted(selected_tasks),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "quantile_sample_cap": int(
                self._marker.get("quantile_sample_cap", _DEFAULT_QUANTILE_SAMPLE_CAP)
            ),
        }
        digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
        for row in self._episode_rows:
            episode = {
                "episode_index": int(row["episode_index"]),
                "length": int(row["length"]),
                "source_task": str(row["source_task"]),
                "states_actions_path": str(row["states_actions_path"]),
            }
            digest.update(b"\0")
            digest.update(json.dumps(episode, sort_keys=True, separators=(",", ":")).encode())
        return digest.hexdigest()

    def _selection_stats_cache_path(
        self, selected_tasks: list[str], fingerprint: str
    ) -> Path:
        if len(selected_tasks) == 1:
            label = "task-" + re.sub(r"[^A-Za-z0-9_.-]+", "-", selected_tasks[0]).strip("-")
        else:
            label = f"tasks-{len(selected_tasks)}"
        return self.root / "meta" / "selection_stats" / f"{label}-{fingerprint[:12]}.json"

    def _compute_selection_stats(self) -> tuple[dict, int]:
        total_frames = sum(int(row["length"]) for row in self._episode_rows)
        if total_frames <= 0:
            raise ValueError("Cannot compute ABC selection stats for an empty episode selection.")
        sample_cap = int(
            self._marker.get("quantile_sample_cap", _DEFAULT_QUANTILE_SAMPLE_CAP)
        )
        if sample_cap <= 0:
            raise ValueError("ABC native quantile_sample_cap must be positive.")
        sample_stride = max(1, math.ceil(total_frames / sample_cap))
        width = self.state_dim + self.action_dim
        count = 0
        total = np.zeros(width, dtype=np.float64)
        total_sq = np.zeros(width, dtype=np.float64)
        minimum = np.full(width, np.inf, dtype=np.float64)
        maximum = np.full(width, -np.inf, dtype=np.float64)
        samples: list[np.ndarray] = []
        global_offset = 0

        logger.info(
            "Computing ABC selection stats from %d episodes / %d frames...",
            len(self._episode_rows),
            total_frames,
        )
        for position, row in enumerate(self._episode_rows, start=1):
            values = np.asarray(self._memmap(row), dtype=np.float64)
            count += len(values)
            total += values.sum(axis=0)
            total_sq += np.square(values).sum(axis=0)
            minimum = np.minimum(minimum, values.min(axis=0))
            maximum = np.maximum(maximum, values.max(axis=0))
            first_sample = (-global_offset) % sample_stride
            if first_sample < len(values):
                samples.append(np.asarray(values[first_sample::sample_stride], dtype=np.float32))
            global_offset += len(values)
            if position % 1000 == 0 or position == len(self._episode_rows):
                logger.info(
                    "ABC selection stats: scanned %d/%d episodes (%d/%d frames)",
                    position,
                    len(self._episode_rows),
                    global_offset,
                    total_frames,
                )

        sampled = np.concatenate(samples, axis=0)
        mean = total / count
        variance = np.maximum(total_sq / count - np.square(mean), 0.0)
        quantiles = np.quantile(sampled, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)

        def feature_stats(start: int, end: int) -> dict[str, list[float] | list[int]]:
            return {
                "min": minimum[start:end].tolist(),
                "max": maximum[start:end].tolist(),
                "mean": mean[start:end].tolist(),
                "std": np.sqrt(variance[start:end]).tolist(),
                "count": [count],
                "q01": quantiles[0, start:end].astype(np.float64).tolist(),
                "q10": quantiles[1, start:end].astype(np.float64).tolist(),
                "q50": quantiles[2, start:end].astype(np.float64).tolist(),
                "q90": quantiles[3, start:end].astype(np.float64).tolist(),
                "q99": quantiles[4, start:end].astype(np.float64).tolist(),
            }

        return {
            _STATE_KEY: feature_stats(0, self.state_dim),
            _ACTION_KEY: feature_stats(self.state_dim, width),
        }, sample_stride

    def _apply_selection_stats(self, selected_tasks: list[str]) -> None:
        fingerprint = self._selection_stats_fingerprint(selected_tasks)
        cache_path = self._selection_stats_cache_path(selected_tasks, fingerprint)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")

        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            payload = None
            if cache_path.is_file():
                try:
                    candidate = json.loads(cache_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    candidate = None
                if (
                    isinstance(candidate, dict)
                    and candidate.get("format") == ABC_SELECTION_STATS_FORMAT
                    and candidate.get("fingerprint") == fingerprint
                ):
                    payload = candidate

            if payload is None:
                stats, sample_stride = self._compute_selection_stats()
                payload = {
                    "format": ABC_SELECTION_STATS_FORMAT,
                    "fingerprint": fingerprint,
                    "tasks": selected_tasks,
                    "episode_count": len(self._episode_rows),
                    "frame_count": sum(int(row["length"]) for row in self._episode_rows),
                    "quantile_sample_stride": sample_stride,
                    "stats": stats,
                }
                temporary = cache_path.with_name(
                    f".{cache_path.name}.{os.getpid()}.tmp"
                )
                try:
                    temporary.write_text(
                        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
                    )
                    os.replace(temporary, cache_path)
                finally:
                    temporary.unlink(missing_ok=True)
                logger.info("Cached ABC selection stats at %s", cache_path)
            else:
                logger.info("Loaded cached ABC selection stats from %s", cache_path)

        merged_stats = dict(self.meta.stats or {})
        merged_stats.update(self._stats_to_numpy(payload["stats"]))
        self.meta.stats = merged_stats
        self.selection_stats_path = cache_path

    def _load_marker(self) -> dict:
        marker_path = self.root / "native_dataset.json"
        if not marker_path.is_file():
            raise FileNotFoundError(f"Missing ABC native marker: {marker_path}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("format") != ABC_NATIVE_FORMAT:
            raise ValueError(
                f"Unsupported ABC native format {marker.get('format')!r}; expected {ABC_NATIVE_FORMAT!r}"
            )
        required = {"raw_root", "state_dim", "action_dim", "source_image_size"}
        missing = sorted(required - set(marker))
        if missing:
            raise ValueError(f"ABC native marker is missing fields: {missing}")
        return marker

    def _resolve_raw_root(self, marker: dict) -> Path:
        raw_value = str(marker["raw_root"]).strip()
        if not raw_value:
            raise ValueError("ABC native marker raw_root must not be empty")
        raw_root = (self.root / raw_value).resolve()
        if not raw_root.is_dir():
            raise FileNotFoundError(f"ABC native raw root not found: {raw_root}")
        return raw_root

    def _source_path(self, relative: str, *, kind: str, episode_index: int) -> Path:
        path = (self.raw_root / relative).resolve()
        try:
            path.relative_to(self.raw_root)
        except ValueError as error:
            raise ValueError(
                f"ABC native episode {episode_index} {kind} path escapes raw_root: {path}"
            ) from error
        if not path.is_file():
            raise FileNotFoundError(f"ABC native episode {episode_index} {kind} file not found: {path}")
        return path

    @property
    def num_frames(self) -> int:
        return len(self)

    @property
    def num_episodes(self) -> int:
        return len(self._episode_rows)

    @property
    def fps(self) -> int:
        return self.meta.fps

    @property
    def features(self) -> dict:
        return self.meta.features

    @property
    def camera_keys(self) -> list[str]:
        return self.meta.camera_keys

    @property
    def stats(self) -> dict:
        return self.meta.stats

    def __len__(self) -> int:
        return int(self._cumulative_lengths[-1]) if len(self._cumulative_lengths) else 0

    def _locate(self, index: int) -> tuple[int, dict, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"ABC native dataset index out of range: {index}")
        selected_index = int(np.searchsorted(self._cumulative_lengths, index, side="right"))
        previous = int(self._cumulative_lengths[selected_index - 1]) if selected_index else 0
        return selected_index, self._episode_rows[selected_index], int(index - previous)

    def _memmap(self, row: dict) -> np.memmap:
        episode_index = int(row["episode_index"])
        cached = self._memmaps.pop(episode_index, None)
        if cached is not None:
            self._memmaps[episode_index] = cached
            return cached
        path = self._source_path(
            str(row["states_actions_path"]), kind="state/action", episode_index=episode_index
        )
        expected_values = int(row["length"]) * (self.state_dim + self.action_dim)
        expected_bytes = expected_values * np.dtype("<f8").itemsize
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"ABC native episode {episode_index} state/action file is {actual_bytes} bytes; "
                f"expected {expected_bytes}"
            )
        array = np.memmap(
            path,
            mode="r",
            dtype="<f8",
            shape=(int(row["length"]), self.state_dim + self.action_dim),
        )
        self._memmaps[episode_index] = array
        while len(self._memmaps) > self._memmap_cache_size:
            self._memmaps.popitem(last=False)
        return array

    def _query_indices(
        self, frame_index: int, episode_length: int
    ) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        if self.delta_indices is None:
            return {}, {}
        queries: dict[str, list[int]] = {}
        padding: dict[str, torch.Tensor] = {}
        for key, offsets in self.delta_indices.items():
            queries[key] = [max(0, min(episode_length - 1, frame_index + offset)) for offset in offsets]
            padding[f"{key}_is_pad"] = torch.tensor(
                [frame_index + offset < 0 or frame_index + offset >= episode_length for offset in offsets],
                dtype=torch.bool,
            )
        return queries, padding

    @staticmethod
    def _prompt_at(row: dict, frame_index: int) -> tuple[int, str]:
        segments = json.loads(str(row["prompt_segments_json"]))
        starts = [int(segment["frame"]) for segment in segments]
        segment = segments[max(0, bisect_right(starts, frame_index) - 1)]
        task_index = int(segment["task_index"])
        tasks = list(row["tasks"])
        task_indices = [int(value) for value in row["task_indices"]]
        try:
            task = str(tasks[task_indices.index(task_index)])
        except ValueError as error:
            raise ValueError(
                f"Episode {row['episode_index']} prompt segment references unknown task_index {task_index}"
            ) from error
        return task_index, task

    def _query_cameras(
        self, row: dict, frame_index: int, queries: dict[str, list[int]]
    ) -> dict[str, torch.Tensor]:
        if not self._camera_keys:
            return {}
        episode_index = int(row["episode_index"])
        layout = json.loads(str(row["camera_layout_json"]))
        layouts = {f"observation.images.{entry['camera']}": entry for entry in layout}
        missing = sorted(set(self._camera_keys) - set(layouts))
        if missing:
            raise ValueError(f"ABC native episode {episode_index} has no camera layout for {missing}")

        per_camera_indices = {key: queries.get(key, [frame_index]) for key in self._camera_keys}
        unique_indices = sorted({value for values in per_camera_indices.values() for value in values})
        video_path = self._source_path(str(row["video_path"]), kind="video", episode_index=episode_index)
        stacked = decode_video_frames(
            video_path,
            [value / self.meta.fps for value in unique_indices],
            self.tolerance_s,
            self.video_backend,
            decoder_num_threads=1 if self.video_backend == "pyav" else None,
        )
        decoded = {frame: stacked[position] for position, frame in enumerate(unique_indices)}

        result: dict[str, torch.Tensor] = {}
        for key in self._camera_keys:
            entry = layouts[key]
            start = int(entry["row_start"])
            end = int(entry["row_end"])
            crops = torch.stack([decoded[value][:, start:end, :] for value in per_camera_indices[key]])
            actual_size = (int(crops.shape[-1]), int(crops.shape[-2]))
            expected_size = (self.image_width, self.image_height)
            if actual_size != expected_size:
                raise ValueError(
                    f"ABC native episode {episode_index} camera {key!r} has size "
                    f"{actual_size}, expected {expected_size}"
                )
            result[key] = crops if key in queries else crops.squeeze(0)
        return result

    def __getitem__(self, index: int) -> dict:
        _, row, frame_index = self._locate(int(index))
        episode_index = int(row["episode_index"])
        episode_length = int(row["length"])
        absolute_index = int(row["dataset_from_index"]) + frame_index
        values = self._memmap(row)
        queries, padding = self._query_indices(frame_index, episode_length)

        item: dict = {
            _STATE_KEY: torch.from_numpy(np.asarray(values[frame_index, : self.state_dim], dtype=np.float32)),
            _ACTION_KEY: torch.from_numpy(
                np.asarray(values[frame_index, self.state_dim :], dtype=np.float32)
            ),
            "timestamp": torch.tensor(frame_index / self.meta.fps, dtype=torch.float32),
            "frame_index": torch.tensor(frame_index, dtype=torch.int64),
            "episode_index": torch.tensor(episode_index, dtype=torch.int64),
            "index": torch.tensor(absolute_index, dtype=torch.int64),
        }
        for key, indices in queries.items():
            if key == _STATE_KEY:
                array = values[indices, : self.state_dim]
            elif key == _ACTION_KEY:
                array = values[indices, self.state_dim :]
            elif key in self.meta.camera_keys:
                continue
            else:
                raise NotImplementedError(f"ABC native temporal queries are not implemented for {key!r}")
            item[key] = torch.from_numpy(np.asarray(array, dtype=np.float32))
        item.update(padding)
        item.update(self._query_cameras(row, frame_index, queries))

        task_index, task = self._prompt_at(row, frame_index)
        item["task_index"] = torch.tensor(task_index, dtype=torch.int64)
        item["task"] = task

        if self.image_transforms is not None:
            for key in self._camera_keys:
                item[key] = self.image_transforms(item[key])
        return item
