"""Episode-exact skill occurrences and filtered-to-original frame alignment."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pandas as pd

ACTION_MATCH_THRESHOLD = 1e-3


def _load_task_descriptions(dataset_dir: Path) -> dict[int, str]:
    """Return local dataset task IDs and their exact training languages."""
    path = dataset_dir / "meta" / "tasks.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Skill dataset task metadata not found: {path}")
    frame = pd.read_parquet(path)
    if "task_index" not in frame.columns:
        raise ValueError(f"Skill dataset task metadata lacks task_index: {path}")
    if "task" in frame.columns:
        languages = frame["task"].astype(str).tolist()
    elif frame.index.name in {"task", "language"}:
        languages = frame.index.astype(str).tolist()
    elif "language" in frame.columns:
        languages = frame["language"].astype(str).tolist()
    else:
        raise ValueError(
            f"Skill dataset task metadata has no task/language field: {path}"
        )
    result: dict[int, str] = {}
    for task_id, language in zip(frame["task_index"], languages, strict=True):
        task_id = int(task_id)
        previous = result.setdefault(task_id, str(language))
        if previous != str(language):
            raise ValueError(
                f"Dataset task {task_id} has conflicting languages in {path}."
            )
    return result


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

    @property
    def identity_uid(self) -> str:
        """Token-independent identity shared by aligned FSQ code spaces."""
        return (
            f"task{self.task_id:02d}_ep{self.episode_id:05d}_"
            f"skill{self.skill_index:02d}_f{self.frame_start:04d}_{self.frame_end:04d}"
        )


@dataclass(frozen=True)
class EpisodeSource:
    episode_id: int
    task_id: int
    scene_file: str
    demo: str
    kind: str = "libero_hdf5"
    init_state: np.ndarray | None = None
    init_state_index: int | None = None
    suite_name: str | None = None
    suite_task_id: int | None = None

    @property
    def env_task_id(self) -> int:
        """Simulator task ID (different from compact dataset IDs in LangGap)."""
        return int(self.suite_task_id if self.suite_task_id is not None else self.task_id)


@dataclass
class AlignedEpisode:
    source: EpisodeSource
    filtered_actions: np.ndarray
    filtered_states: np.ndarray
    original_action_indices: np.ndarray
    original_states: np.ndarray
    model_xml: str | None
    episode_start_xyz: np.ndarray
    alignment_mean_error: float
    alignment_max_error: float

    @property
    def requires_episode_replay(self) -> bool:
        return self.source.kind == "langgap_init"

    def restoration_at(
        self,
        filtered_frame: int,
    ) -> tuple[np.ndarray, np.ndarray, int | None]:
        """Return reset state, actions to replay, and optional LIBERO init ID."""
        if not 0 <= filtered_frame < len(self.filtered_actions):
            raise IndexError(
                f"filtered frame {filtered_frame} is outside episode "
                f"[0, {len(self.filtered_actions) - 1}]"
            )
        if not self.requires_episode_replay:
            return (
                self.state_at(filtered_frame),
                np.empty((0, self.filtered_actions.shape[1]), dtype=np.float32),
                None,
            )
        if self.source.init_state is None or self.source.init_state_index is None:
            raise ValueError(
                f"LangGap episode {self.source.episode_id} lacks exact init-state provenance."
            )
        return (
            np.asarray(self.source.init_state, dtype=np.float64).copy(),
            np.asarray(
                self.filtered_actions[:filtered_frame], dtype=np.float32
            ).copy(),
            int(self.source.init_state_index),
        )

    def state_at(self, filtered_frame: int) -> np.ndarray:
        if self.requires_episode_replay:
            raise RuntimeError(
                "LangGap does not provide per-frame MuJoCo states; use "
                "restoration_at() to reset the exact episode and replay actions."
            )
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


def _load_episode_meta(dataset_dir: Path) -> pd.DataFrame:
    files = sorted((dataset_dir / "meta" / "episodes").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata under {dataset_dir / 'meta/episodes'}")
    columns = [
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


def _episode_task_description(row: pd.Series, *, episode_id: int) -> str:
    """Read the language attached to one episode without assuming a task-ID order."""
    tasks = row["tasks"]
    if isinstance(tasks, np.ndarray):
        tasks = tasks.tolist()
    if isinstance(tasks, (list, tuple)):
        if len(tasks) != 1:
            raise ValueError(
                f"Episode {episode_id} must name exactly one task, got {tasks}."
            )
        tasks = tasks[0]
    return str(tasks)


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
        raw_dataset_dir: str | Path | None = None,
    ) -> None:
        self.skill_dataset_dir = Path(skill_dataset_dir)
        self.original_dataset_dir = (
            Path(original_dataset_dir)
            if original_dataset_dir is not None and str(original_dataset_dir).strip()
            else None
        )
        self.raw_dataset_dir = (
            Path(raw_dataset_dir)
            if raw_dataset_dir is not None and str(raw_dataset_dir).strip()
            else None
        )
        self.suite_name = str(suite_name)
        self.episode_meta = _load_episode_meta(self.skill_dataset_dir)
        self.raw_episode_meta = (
            _load_episode_meta(self.raw_dataset_dir)
            if self.raw_dataset_dir is not None
            else None
        )
        self.task_descriptions = _load_task_descriptions(self.skill_dataset_dir)
        self._shard_cache: dict[tuple[int, int], pd.DataFrame] = {}
        self._raw_shard_cache: dict[tuple[int, int], pd.DataFrame] = {}
        self._aligned_cache: dict[int, AlignedEpisode] = {}
        self._localized_model_xml_cache: dict[int, str] = {}
        info_path = self.skill_dataset_dir / "meta" / "info.json"
        info = json.loads(info_path.read_text()) if info_path.is_file() else {}
        self.proprio_grounding = str(
            info.get("proprio_grounding", "none") or "none"
        ).strip().lower().replace("-", "_")

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
        self._latent_rows = [
            {
                "token": int(token),
                "episode_id": int(episode),
                "skill_index": int(skill_index),
                "frame_start": int(frame_start),
                "frame_end": int(frame_end),
            }
            for token, episode, skill_index, frame_start, frame_end in zip(
                latent_data["tokens"],
                latent_data["episode_id"],
                latent_data["skill_index"],
                latent_data["frame_start"],
                latent_data["frame_end"],
                strict=True,
            )
        ]

        self.sources: dict[int, EpisodeSource] = {}
        self.has_episode_exact_map = eval_init_states_path is not None
        self.uses_langgap_replay = False
        if eval_init_states_path is None:
            language_to_task = {
                language: task_id
                for task_id, language in self.task_descriptions.items()
            }
            for episode_id, row in self.episode_meta.iterrows():
                language = _episode_task_description(row, episode_id=int(episode_id))
                task_id = language_to_task.get(language)
                if task_id is None:
                    raise ValueError(
                        f"Episode {episode_id} task {language!r} is absent from "
                        f"{self.skill_dataset_dir / 'meta/tasks.parquet'}."
                    )
                self.sources[int(episode_id)] = EpisodeSource(
                    episode_id=int(episode_id),
                    task_id=int(task_id),
                    scene_file="",
                    demo="",
                    kind="dataset_meta",
                )
        else:
            exact = np.load(str(eval_init_states_path), allow_pickle=True)
            langgap_exact_fields = {
                "episode_index",
                "dataset_task_id",
                "init_states",
                "suite_name",
                "suite_task_id",
                "init_state_index",
            }
            self.uses_langgap_replay = langgap_exact_fields.issubset(exact.files)
            required_exact = (
                langgap_exact_fields
                if self.uses_langgap_replay
                else {"episode_index", "scene_file", "demo"}
            )
            missing_exact = sorted(required_exact - set(exact.files))
            if missing_exact:
                raise ValueError(
                    f"Episode-exact map is missing {missing_exact}."
                )
        if eval_init_states_path is not None and self.uses_langgap_replay:
            for episode, dataset_task_id, init_state, suite_name, suite_task_id, init_index in zip(
                exact["episode_index"],
                exact["dataset_task_id"],
                exact["init_states"],
                exact["suite_name"],
                exact["suite_task_id"],
                exact["init_state_index"],
                strict=True,
            ):
                episode_id = int(episode)
                source_suite = str(suite_name)
                if source_suite != self.suite_name:
                    continue
                if episode_id in self.sources:
                    raise ValueError(f"Duplicate exact source for episode {episode_id}.")
                self.sources[episode_id] = EpisodeSource(
                    episode_id=episode_id,
                    task_id=int(dataset_task_id),
                    scene_file=f"{source_suite}:task_{int(suite_task_id):02d}",
                    demo=f"init_{int(init_index):03d}",
                    kind="langgap_init",
                    init_state=np.asarray(init_state, dtype=np.float64).copy(),
                    init_state_index=int(init_index),
                    suite_name=source_suite,
                    suite_task_id=int(suite_task_id),
                )
        elif eval_init_states_path is not None:
            scene_to_task = _scene_task_ids(self.suite_name)
            for episode, scene_file, demo in zip(
                exact["episode_index"], exact["scene_file"], exact["demo"], strict=True
            ):
                scene_file = str(scene_file)
                task_name = scene_file.removesuffix("_demo.hdf5")
                task_id = scene_to_task.get(task_name)
                if task_id is None:
                    continue
                episode_id = int(episode)
                if episode_id in self.sources:
                    raise ValueError(f"Duplicate exact source for episode {episode_id}.")
                self.sources[episode_id] = EpisodeSource(
                    episode_id=episode_id,
                    task_id=int(task_id),
                    scene_file=scene_file,
                    demo=str(demo),
                )

        self._rows_by_episode: dict[int, list[dict]] = {}
        for row in self._latent_rows:
            self._rows_by_episode.setdefault(row["episode_id"], []).append(row)
        for rows in self._rows_by_episode.values():
            rows.sort(key=lambda row: (row["frame_start"], row["skill_index"]))

    def episode_task_description(self, episode_id: int) -> str:
        """Return the episode's own language, independent of benchmark task IDs."""
        episode_id = int(episode_id)
        if episode_id not in self.episode_meta.index:
            raise KeyError(f"Skill dataset has no episode {episode_id}.")
        return _episode_task_description(
            self.episode_meta.loc[episode_id], episode_id=episode_id
        )

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
        available: dict[int, list[int]] = {task_id: [] for task_id in task_ids}
        for episode_id, source in self.sources.items():
            if (
                source.task_id in available
                and episode_id in self._rows_by_episode
                and episode_id in self.episode_meta.index
            ):
                available[source.task_id].append(episode_id)
        for values in available.values():
            values.sort()

        explicit = [int(value) for value in (explicit_episode_ids or [])]
        if explicit:
            selected: dict[int, list[int]] = {task_id: [] for task_id in task_ids}
            missing = []
            for episode_id in explicit:
                source = self.sources.get(episode_id)
                if (
                    source is None
                    or source.task_id not in selected
                    or episode_id not in self._rows_by_episode
                    or episode_id not in self.episode_meta.index
                ):
                    missing.append(episode_id)
                else:
                    selected[source.task_id].append(episode_id)
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

    def _raw_episode_start_xyz(self, episode_id: int) -> np.ndarray:
        """Read LangGap's ungrounded frame-0 EEF xyz from the source dataset."""
        if self.raw_dataset_dir is None or self.raw_episode_meta is None:
            raise RuntimeError(
                "LangGap episode_start_xyz evaluation requires the raw source "
                "dataset so its ungrounded frame-0 EEF xyz can be restored."
            )
        episode_id = int(episode_id)
        if episode_id not in self.raw_episode_meta.index:
            raise KeyError(
                f"Raw dataset {self.raw_dataset_dir} has no episode {episode_id}."
            )

        raw_meta = self.raw_episode_meta.loc[episode_id]
        skill_meta = self.episode_meta.loc[episode_id]
        raw_tasks = raw_meta["tasks"]
        skill_tasks = skill_meta["tasks"]
        if isinstance(raw_tasks, np.ndarray):
            raw_tasks = raw_tasks.tolist()
        if isinstance(skill_tasks, np.ndarray):
            skill_tasks = skill_tasks.tolist()
        if raw_tasks != skill_tasks:
            raise ValueError(
                "Raw and SkillVLA episode task metadata differ for episode "
                f"{episode_id}: raw={raw_tasks!r}, skillvla={skill_tasks!r}."
            )

        key = (
            int(raw_meta["data/chunk_index"]),
            int(raw_meta["data/file_index"]),
        )
        if key not in self._raw_shard_cache:
            path = (
                self.raw_dataset_dir
                / "data"
                / f"chunk-{key[0]:03d}"
                / f"file-{key[1]:03d}.parquet"
            )
            self._raw_shard_cache[key] = pd.read_parquet(
                path,
                columns=["episode_index", "frame_index", "observation.state"],
            )
        frame = self._raw_shard_cache[key]
        frame = frame[frame["episode_index"] == episode_id].sort_values(
            "frame_index"
        )
        if frame.empty or int(frame.iloc[0]["frame_index"]) != 0:
            raise ValueError(
                f"Raw dataset episode {episode_id} has no frame-0 state."
            )
        state = np.asarray(frame.iloc[0]["observation.state"], dtype=np.float32)
        if state.ndim != 1 or state.shape[0] < 3 or not np.isfinite(state[:3]).all():
            raise ValueError(
                "Raw LangGap frame-0 observation.state must contain finite EEF xyz, "
                f"got shape={state.shape} for episode {episode_id}."
            )
        return state[:3].copy()

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
        if source.kind == "dataset_meta":
            raise RuntimeError(
                f"Episode {episode_id} was sourced from dataset metadata only; "
                "state alignment requires an episode-exact map."
            )
        frame = self._episode_frame(episode_id)
        filtered_actions = np.stack(frame["action"].to_numpy()).astype(np.float32)
        filtered_states = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
        if source.kind == "langgap_init":
            if filtered_states.ndim != 2 or filtered_states.shape[1] < 3:
                raise ValueError(
                    "LangGap observation.state must provide episode-start xyz, got "
                    f"shape={filtered_states.shape}."
                )
            if self.proprio_grounding == "episode_start_xyz":
                episode_start_xyz = self._raw_episode_start_xyz(episode_id)
            elif self.proprio_grounding == "none":
                episode_start_xyz = filtered_states[0, :3].copy()
            else:
                raise ValueError(
                    "Unsupported SkillVLA proprio_grounding contract: "
                    f"{self.proprio_grounding!r}."
                )
            aligned = AlignedEpisode(
                source=source,
                filtered_actions=filtered_actions,
                filtered_states=filtered_states,
                original_action_indices=np.arange(
                    len(filtered_actions), dtype=np.int32
                ),
                original_states=np.empty((0, 0), dtype=np.float64),
                model_xml=None,
                episode_start_xyz=episode_start_xyz,
                alignment_mean_error=0.0,
                alignment_max_error=0.0,
            )
            self._aligned_cache[episode_id] = aligned
            return aligned
        if self.original_dataset_dir is None:
            raise RuntimeError(
                "LIBERO HDF5 state alignment requires original_dataset_dir."
            )
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
            model_xml_value = demo.attrs.get("model_file")
            if isinstance(model_xml_value, (bytes, np.bytes_)):
                model_xml = bytes(model_xml_value).decode("utf-8")
            elif model_xml_value is None:
                model_xml = None
            else:
                model_xml = str(model_xml_value)
            if "obs" not in demo or "ee_states" not in demo["obs"]:
                raise KeyError(
                    f"{hdf5_path}:{demo_path} has no obs/ee_states required "
                    "for episode-start proprio grounding."
                )
            original_ee_states = np.asarray(
                demo["obs"]["ee_states"], dtype=np.float32
            )
        indices, mean_error, max_error = self.align_actions(
            filtered_actions, original_actions
        )
        if int(indices[-1]) >= len(original_states):
            raise ValueError(
                f"Episode {episode_id} maps to state {indices[-1]} but original demo has "
                f"only {len(original_states)} states."
            )
        first_original_frame = int(indices[0])
        if (
            original_ee_states.ndim != 2
            or original_ee_states.shape[1] < 3
            or first_original_frame >= len(original_ee_states)
        ):
            raise ValueError(
                "Original obs/ee_states cannot provide the aligned episode-start "
                f"xyz: shape={original_ee_states.shape}, frame={first_original_frame}."
            )
        aligned = AlignedEpisode(
            source=source,
            filtered_actions=filtered_actions,
            filtered_states=filtered_states,
            original_action_indices=indices,
            original_states=original_states,
            model_xml=model_xml,
            episode_start_xyz=original_ee_states[first_original_frame, :3].copy(),
            alignment_mean_error=mean_error,
            alignment_max_error=max_error,
        )
        self._aligned_cache[episode_id] = aligned
        return aligned

    def exact_model_xml(self, episode_id: int) -> str | None:
        """Return the source demo XML with asset paths localized to this host."""
        episode_id = int(episode_id)
        aligned = self.load_aligned_episode(episode_id)
        if aligned.requires_episode_replay:
            return None
        if episode_id not in self._localized_model_xml_cache:
            if not aligned.model_xml:
                source = aligned.source
                raise ValueError(
                    "Episode-exact world restoration requires the original demo's "
                    f"model_file attribute: {source.scene_file}:{source.demo}."
                )
            self._localized_model_xml_cache[episode_id] = localize_model_xml_assets(
                aligned.model_xml
            )
        return self._localized_model_xml_cache[episode_id]


def _default_libero_assets_dir() -> Path:
    """Resolve LIBERO assets without embedding a cluster-specific prefix."""
    from libero.libero import get_libero_path

    configured = Path(get_libero_path("assets")).expanduser()
    if configured.is_dir():
        return configured.resolve()

    # A stale user-level LIBERO config can exist outside submitted jobs. The
    # imported package remains a portable fallback for tests and local runs.
    import libero

    packaged = Path(libero.__file__).resolve().parent / "libero" / "assets"
    if packaged.is_dir():
        return packaged.resolve()
    raise FileNotFoundError(
        "Could not resolve LIBERO assets from get_libero_path('assets') or the "
        f"imported package; configured={configured}, packaged={packaged}."
    )


def localize_model_xml_assets(
    model_xml: str,
    *,
    libero_assets_dir: str | Path | None = None,
    robosuite_package_dir: str | Path | None = None,
) -> str:
    """Rewrite source-machine asset paths in a recorded MuJoCo model XML.

    LIBERO demonstrations retain the full collection-time model XML, including
    fixture body positions and camera parameters. Asset file attributes are
    absolute paths on the collection machine, so only those paths are replaced;
    the recorded world geometry itself is left untouched.
    """
    if not str(model_xml).strip():
        raise ValueError("Recorded model XML must not be empty.")
    if libero_assets_dir is None:
        libero_root = _default_libero_assets_dir()
    else:
        libero_root = Path(libero_assets_dir).expanduser().resolve()
    if robosuite_package_dir is None:
        import robosuite

        robosuite_root = Path(robosuite.__file__).resolve().parent
    else:
        robosuite_root = Path(robosuite_package_dir).expanduser().resolve()

    root = ET.fromstring(model_xml)
    missing: list[tuple[str, str]] = []
    for element in root.findall(".//mesh") + root.findall(".//texture"):
        old_value = element.get("file")
        if not old_value:
            continue
        old_path = Path(old_value)
        if old_path.is_file():
            continue
        parts = old_path.parts
        candidate: Path | None = None
        robosuite_indices = [
            index for index, part in enumerate(parts) if part == "robosuite"
        ]
        if robosuite_indices:
            marker = robosuite_indices[-1]
            candidate = robosuite_root.joinpath(*parts[marker + 1 :])
        else:
            asset_indices = [
                index for index, part in enumerate(parts) if part == "assets"
            ]
            if asset_indices:
                marker = asset_indices[-1]
                candidate = libero_root.joinpath(*parts[marker + 1 :])
        if candidate is None or not candidate.is_file():
            missing.append((old_value, "" if candidate is None else str(candidate)))
            continue
        element.set("file", str(candidate.resolve()))
    if missing:
        preview = "; ".join(
            f"{old} -> {new or '<unresolved>'}" for old, new in missing[:5]
        )
        raise FileNotFoundError(
            f"Could not localize {len(missing)} recorded MuJoCo assets: {preview}"
        )
    return ET.tostring(root, encoding="unicode")
