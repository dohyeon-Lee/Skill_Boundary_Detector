"""Episode-exact skill occurrences and filtered-to-original frame alignment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ACTION_MATCH_THRESHOLD = 1e-3


@dataclass(frozen=True)
class SkillOccurrence:
    token: int
    episode_id: int
    task_id: int
    skill_index: int
    frame_start: int
    frame_end: int

    @property
    def length(self) -> int:
        return self.frame_end - self.frame_start

    @property
    def uid(self) -> str:
        return (
            f"task{self.task_id:02d}_ep{self.episode_id:05d}_"
            f"skill{self.skill_index:02d}_token{self.token:04d}_f{self.frame_start:04d}"
        )


@dataclass(frozen=True)
class EpisodeSource:
    episode_id: int
    task_id: int
    scene_file: str
    demo: str


@dataclass
class AlignedEpisode:
    source: EpisodeSource
    filtered_actions: np.ndarray
    filtered_states: np.ndarray
    original_action_indices: np.ndarray
    original_states: np.ndarray
    alignment_mean_error: float
    alignment_max_error: float

    def state_at(self, filtered_frame: int) -> np.ndarray:
        if not 0 <= filtered_frame < len(self.original_action_indices):
            raise IndexError(
                f"filtered frame {filtered_frame} is outside episode "
                f"[0, {len(self.original_action_indices) - 1}]"
            )
        original_frame = int(self.original_action_indices[filtered_frame])
        return np.asarray(self.original_states[original_frame], dtype=np.float64).copy()

    def original_frame_at(self, filtered_frame: int) -> int:
        return int(self.original_action_indices[filtered_frame])


def token_to_coord(token: int, levels: list[int] | tuple[int, ...]) -> list[int]:
    coord = []
    base = 1
    for level in levels:
        coord.append((int(token) // base) % int(level))
        base *= int(level)
    return coord


def _scene_task_ids(suite_name: str) -> dict[str, int]:
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[suite_name]()
    return {str(task.name): index for index, task in enumerate(suite.tasks)}


def _dataset_task_ids(dataset_dir: Path) -> dict[str, int]:
    """Task description -> task index, from the dataset's own task table.

    A dataset whose episodes come from several LIBERO suites has no single
    suite to number its tasks by, so its own table is the only numbering that
    covers every episode.
    """
    path = dataset_dir / "meta" / "tasks.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset task table not found: {path}")
    frame = pd.read_parquet(path).reset_index()
    missing = sorted({"task", "task_index"} - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing {missing}.")
    task_ids = {str(task): int(index) for task, index in zip(frame["task"], frame["task_index"], strict=True)}
    if len(task_ids) != len(frame):
        raise ValueError(f"{path} maps one task description to several indices.")
    return task_ids


def _load_episode_meta(dataset_dir: Path) -> pd.DataFrame:
    files = sorted((dataset_dir / "meta" / "episodes").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata under {dataset_dir / 'meta/episodes'}")
    native_abc = (dataset_dir / "native_dataset.json").is_file()
    columns = None if native_abc else [
        "episode_index",
        "data/chunk_index",
        "data/file_index",
        "length",
        "tasks",
    ]
    frame = pd.concat(
        [pd.read_parquet(path, columns=columns) for path in files],
        ignore_index=True,
    )
    if frame["episode_index"].duplicated().any():
        raise ValueError(f"Duplicate episode_index values in {dataset_dir / 'meta/episodes'}")
    return frame.set_index("episode_index", drop=False)


class SkillEvaluationDataset:
    """Join FSQ segments, exact demo mapping, filtered actions and HDF5 states."""

    def __init__(
        self,
        *,
        skill_dataset_dir: str | Path,
        skill_latents_path: str | Path,
        eval_init_states_path: str | Path | None,
        original_dataset_dir: str | Path | None,
        suite_name: str,
    ) -> None:
        self.skill_dataset_dir = Path(skill_dataset_dir)
        self.original_dataset_dir = (
            Path(original_dataset_dir) if original_dataset_dir else None
        )
        self.suite_name = str(suite_name)
        self.is_abc_native = (self.skill_dataset_dir / "native_dataset.json").is_file()
        self.episode_meta = _load_episode_meta(self.skill_dataset_dir)
        self._shard_cache: dict[tuple[int, int], pd.DataFrame] = {}
        self._aligned_cache: dict[int, AlignedEpisode] = {}

        latent_data = np.load(str(skill_latents_path), allow_pickle=True)
        required_latent = {
            "tokens",
            "episode_id",
            "skill_index",
            "frame_start",
            "frame_end",
        }
        missing_latent = sorted(required_latent - set(latent_data.files))
        if missing_latent:
            raise ValueError(f"skill_latents.npz is missing {missing_latent}.")
        lengths = {key: len(latent_data[key]) for key in required_latent}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"skill_latents arrays have inconsistent lengths: {lengths}")
        latent_task_ids = (
            np.asarray(latent_data["task_id"], dtype=np.int64)
            if "task_id" in latent_data.files
            else None
        )
        if latent_task_ids is not None and len(latent_task_ids) != next(iter(lengths.values())):
            raise ValueError("skill_latents task_id has an inconsistent length.")
        self._latent_rows = []
        for row_index, (token, episode, skill_index, frame_start, frame_end) in enumerate(
            zip(
                latent_data["tokens"],
                latent_data["episode_id"],
                latent_data["skill_index"],
                latent_data["frame_start"],
                latent_data["frame_end"],
                strict=True,
            )
        ):
            row = {
                "token": int(token),
                "episode_id": int(episode),
                "skill_index": int(skill_index),
                "frame_start": int(frame_start),
                "frame_end": int(frame_end),
            }
            if latent_task_ids is not None:
                row["task_id"] = int(latent_task_ids[row_index])
            self._latent_rows.append(row)

        # Task descriptions are only known up front in dataset-meta mode; suite
        # mode leaves them to the caller's LIBERO benchmark lookup.
        self.task_descriptions: dict[int, str] = {}
        if eval_init_states_path is not None:
            self.sources = self._sources_from_exact_map(eval_init_states_path)
        elif self.is_abc_native:
            self.sources = self._sources_from_abc_native_meta()
        else:
            self.sources = self._sources_from_dataset_meta()

        self._rows_by_episode: dict[int, list[dict]] = {}
        for row in self._latent_rows:
            self._rows_by_episode.setdefault(row["episode_id"], []).append(row)
        for rows in self._rows_by_episode.values():
            rows.sort(key=lambda row: (row["frame_start"], row["skill_index"]))
        if self.is_abc_native:
            for rows in self._rows_by_episode.values():
                for row in rows:
                    if "task_id" not in row:
                        row["task_id"] = self._abc_task_at_frame(
                            row["episode_id"], row["frame_start"]
                        )

    def _sources_from_abc_native_meta(self) -> dict[int, EpisodeSource]:
        """ABC episode provenance; skill rows retain their prompt-level task id."""
        task_ids = _dataset_task_ids(self.skill_dataset_dir)
        self.task_descriptions = {index: task for task, index in task_ids.items()}
        sources: dict[int, EpisodeSource] = {}
        for _, row in self.episode_meta.iterrows():
            episode_id = int(row["episode_index"])
            indices = [int(value) for value in np.atleast_1d(row["task_indices"])]
            if not indices:
                raise ValueError(f"ABC episode {episode_id} has no task_indices.")
            sources[episode_id] = EpisodeSource(
                episode_id=episode_id,
                task_id=indices[0],
                scene_file=str(row.get("source_task", "")),
                demo=str(row.get("source_episode", "")),
            )
        return sources

    def _abc_task_at_frame(self, episode_id: int, frame_index: int) -> int:
        row = self.episode_meta.loc[int(episode_id)]
        segments = json.loads(str(row["prompt_segments_json"]))
        selected = None
        for segment in segments:
            if int(segment["frame"]) > int(frame_index):
                break
            selected = int(segment["task_index"])
        if selected is None:
            raise ValueError(
                f"ABC episode {episode_id} has no prompt at frame {frame_index}."
            )
        return selected

    def available_episode_ids_by_task(self) -> dict[int, list[int]]:
        """Task -> episodes containing at least one encoded skill occurrence."""
        available: dict[int, set[int]] = {}
        for episode_id, rows in self._rows_by_episode.items():
            if episode_id not in self.episode_meta.index or episode_id not in self.sources:
                continue
            if self.is_abc_native:
                task_ids = {int(row["task_id"]) for row in rows}
            else:
                task_ids = {int(self.sources[episode_id].task_id)}
            for task_id in task_ids:
                available.setdefault(task_id, set()).add(int(episode_id))
        return {
            task_id: sorted(episode_ids)
            for task_id, episode_ids in available.items()
        }

    def _sources_from_exact_map(
        self, eval_init_states_path: str | Path
    ) -> dict[int, EpisodeSource]:
        """Episode provenance from the rendered episode-exact map.

        Episodes whose scene is not part of ``suite_name`` are dropped: without
        a suite task id they cannot be selected or grouped.
        """
        exact = np.load(str(eval_init_states_path), allow_pickle=True)
        required_exact = {"episode_index", "scene_file", "demo"}
        missing_exact = sorted(required_exact - set(exact.files))
        if missing_exact:
            raise ValueError(
                f"Episode-exact map is missing {missing_exact}; scene/demo provenance is mandatory."
            )
        scene_to_task = _scene_task_ids(self.suite_name)
        sources: dict[int, EpisodeSource] = {}
        for episode, scene_file, demo in zip(
            exact["episode_index"], exact["scene_file"], exact["demo"], strict=True
        ):
            scene_file = str(scene_file)
            task_name = scene_file.removesuffix("_demo.hdf5")
            task_id = scene_to_task.get(task_name)
            if task_id is None:
                continue
            episode_id = int(episode)
            if episode_id in sources:
                raise ValueError(f"Duplicate exact source for episode {episode_id}.")
            sources[episode_id] = EpisodeSource(
                episode_id=episode_id,
                task_id=int(task_id),
                scene_file=scene_file,
                demo=str(demo),
            )
        return sources

    def _sources_from_dataset_meta(self) -> dict[int, EpisodeSource]:
        """Episode provenance from the dataset's own task table.

        Every episode is covered, including datasets that draw their scenes from
        several LIBERO suites, but the original HDF5 demo is unknown -- so this
        mode serves selection and grouping only, never state alignment.
        """
        task_ids = _dataset_task_ids(self.skill_dataset_dir)
        self.task_descriptions = {index: task for task, index in task_ids.items()}
        sources: dict[int, EpisodeSource] = {}
        for episode, tasks in zip(
            self.episode_meta["episode_index"], self.episode_meta["tasks"], strict=True
        ):
            names = [str(name) for name in np.atleast_1d(tasks)]
            if len(names) != 1:
                raise ValueError(
                    f"Episode {int(episode)} carries {len(names)} task descriptions; "
                    "dataset-meta episode sourcing needs exactly one."
                )
            task_id = task_ids.get(names[0])
            if task_id is None:
                raise ValueError(
                    f"Episode {int(episode)} has task {names[0]!r}, which is absent from "
                    f"{self.skill_dataset_dir / 'meta/tasks.parquet'}."
                )
            sources[int(episode)] = EpisodeSource(
                episode_id=int(episode),
                task_id=task_id,
                scene_file="",
                demo="",
            )
        return sources

    def select_episodes(
        self,
        *,
        task_ids: list[int],
        episodes_per_task: int,
        selection: str,
        seed: int,
        explicit_episode_ids: list[int] | None = None,
    ) -> dict[int, list[int]]:
        task_ids = [int(task_id) for task_id in task_ids]
        all_available = self.available_episode_ids_by_task()
        available = {
            task_id: list(all_available.get(task_id, [])) for task_id in task_ids
        }

        explicit = [int(value) for value in (explicit_episode_ids or [])]
        if explicit:
            selected: dict[int, list[int]] = {task_id: [] for task_id in task_ids}
            missing = []
            for episode_id in explicit:
                matched_tasks = [
                    task_id
                    for task_id in task_ids
                    if episode_id in available.get(task_id, [])
                ]
                if not matched_tasks:
                    missing.append(episode_id)
                else:
                    for task_id in matched_tasks:
                        selected[task_id].append(episode_id)
            if missing:
                raise ValueError(
                    "Explicit episode_ids are not episode-exact skill episodes for the selected "
                    f"task_ids: {missing}."
                )
            return {task: values for task, values in selected.items() if values}

        rng = np.random.default_rng(int(seed))
        selected = {}
        for task_id in task_ids:
            candidates = available[task_id]
            if len(candidates) < int(episodes_per_task):
                raise ValueError(
                    f"task {task_id} has only {len(candidates)} episode-exact skill episodes; "
                    f"requested {episodes_per_task}."
                )
            if selection == "random":
                chosen = sorted(
                    int(value)
                    for value in rng.choice(candidates, size=episodes_per_task, replace=False)
                )
            else:
                chosen = candidates[:episodes_per_task]
            selected[task_id] = chosen
        return selected

    def occurrences(self, selected: dict[int, list[int]]) -> list[SkillOccurrence]:
        result = []
        for task_id, episode_ids in selected.items():
            for episode_id in episode_ids:
                for row in self._rows_by_episode[episode_id]:
                    if self.is_abc_native and int(row["task_id"]) != int(task_id):
                        continue
                    start, end = int(row["frame_start"]), int(row["frame_end"])
                    episode_length = int(self.episode_meta.loc[episode_id, "length"])
                    if not (0 <= start < end <= episode_length):
                        raise ValueError(
                            f"Invalid skill interval episode={episode_id}: [{start}, {end}) "
                            f"for length={episode_length}."
                        )
                    result.append(
                        SkillOccurrence(
                            token=int(row["token"]),
                            episode_id=int(episode_id),
                            task_id=int(task_id),
                            skill_index=int(row["skill_index"]),
                            frame_start=start,
                            frame_end=end,
                        )
                    )
        result.sort(
            key=lambda value: (
                value.task_id,
                value.episode_id,
                value.frame_start,
                value.skill_index,
            )
        )
        return result

    def _episode_frame(self, episode_id: int) -> pd.DataFrame:
        meta = self.episode_meta.loc[int(episode_id)]
        key = (int(meta["data/chunk_index"]), int(meta["data/file_index"]))
        if key not in self._shard_cache:
            path = (
                self.skill_dataset_dir
                / "data"
                / f"chunk-{key[0]:03d}"
                / f"file-{key[1]:03d}.parquet"
            )
            self._shard_cache[key] = pd.read_parquet(
                path,
                columns=["episode_index", "frame_index", "action", "observation.state"],
            )
        frame = self._shard_cache[key]
        frame = frame[frame["episode_index"] == int(episode_id)].sort_values("frame_index")
        expected = np.arange(len(frame), dtype=np.int64)
        actual = frame["frame_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(actual, expected):
            raise ValueError(
                f"Episode {episode_id} frame_index is not contiguous from zero: "
                f"first={actual[:3].tolist()}, last={actual[-3:].tolist()}."
            )
        return frame

    @staticmethod
    def align_actions(
        filtered_actions: np.ndarray,
        original_actions: np.ndarray,
        *,
        threshold: float = ACTION_MATCH_THRESHOLD,
    ) -> tuple[np.ndarray, float, float]:
        """Map each filtered action to its exact in-order original demo action."""
        filtered = np.asarray(filtered_actions, dtype=np.float32)
        original = np.asarray(original_actions, dtype=np.float32)
        if filtered.ndim != 2 or original.ndim != 2 or filtered.shape[1:] != original.shape[1:]:
            raise ValueError(
                f"Action shapes are incompatible: filtered={filtered.shape}, original={original.shape}."
            )
        indices: list[int] = []
        residuals: list[float] = []
        cursor = 0
        for action_index, action in enumerate(filtered):
            matched = False
            while cursor < len(original):
                residual = float(np.abs(action - original[cursor]).mean())
                original_index = cursor
                cursor += 1
                if residual <= threshold:
                    indices.append(original_index)
                    residuals.append(residual)
                    matched = True
                    break
            if not matched:
                raise ValueError(
                    f"Filtered action {action_index}/{len(filtered)} has no ordered exact match "
                    f"in original demo (threshold={threshold})."
                )
        return (
            np.asarray(indices, dtype=np.int32),
            float(np.mean(residuals)) if residuals else 0.0,
            float(np.max(residuals)) if residuals else 0.0,
        )

    def load_aligned_episode(self, episode_id: int) -> AlignedEpisode:
        episode_id = int(episode_id)
        if episode_id in self._aligned_cache:
            return self._aligned_cache[episode_id]
        source = self.sources.get(episode_id)
        if source is None:
            raise KeyError(f"Episode {episode_id} has no exact source mapping.")
        if self.original_dataset_dir is None or not source.scene_file:
            raise RuntimeError(
                "State alignment needs the episode-exact map: episodes sourced from "
                "the dataset task table carry no original HDF5 scene/demo."
            )
        frame = self._episode_frame(episode_id)
        filtered_actions = np.stack(frame["action"].to_numpy()).astype(np.float32)
        filtered_states = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
        hdf5_path = self.original_dataset_dir / source.scene_file
        if not hdf5_path.is_file():
            raise FileNotFoundError(f"Original HDF5 not found: {hdf5_path}")
        with h5py.File(hdf5_path, "r") as handle:
            demo_path = f"data/{source.demo}"
            if demo_path not in handle:
                raise KeyError(f"{hdf5_path} has no {demo_path}.")
            demo = handle[demo_path]
            original_actions = np.asarray(demo["actions"], dtype=np.float32)
            original_states = np.asarray(demo["states"], dtype=np.float64)
        indices, mean_error, max_error = self.align_actions(
            filtered_actions, original_actions
        )
        if int(indices[-1]) >= len(original_states):
            raise ValueError(
                f"Episode {episode_id} maps to state {indices[-1]} but original demo has "
                f"only {len(original_states)} states."
            )
        aligned = AlignedEpisode(
            source=source,
            filtered_actions=filtered_actions,
            filtered_states=filtered_states,
            original_action_indices=indices,
            original_states=original_states,
            alignment_mean_error=mean_error,
            alignment_max_error=max_error,
        )
        self._aligned_cache[episode_id] = aligned
        return aligned
