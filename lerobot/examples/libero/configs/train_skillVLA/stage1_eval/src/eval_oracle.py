"""Dataset-to-LIBERO oracle alignment helpers for renewed Stage-1 evaluation."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def _norm_language(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _skill_rows(dataset_dir: Path):
    import pandas as pd  # imported lazily so config-only commands stay lightweight

    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    num_embeddings = int(info["skill_num_embeddings"])
    files = sorted((dataset_dir / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {dataset_dir / 'data'}")
    columns = [
        "episode_index",
        "frame_index",
        "task_index",
        "skill_sequence",
        "skill_length_sequence",
    ]
    frame = pd.concat(
        [pd.read_parquet(path, columns=columns) for path in files], ignore_index=True
    )
    first = frame.sort_values("frame_index").groupby("episode_index", as_index=False).first()
    return first.sort_values("episode_index"), num_embeddings


def _decode_skills(row, num_embeddings: int) -> list[dict]:
    sequence = np.asarray(row["skill_sequence"]).reshape(-1)
    lengths = np.asarray(row["skill_length_sequence"]).reshape(-1)
    return [
        {"token": int(sequence[index]), "gt_length": int(lengths[index])}
        for index in range(min(len(sequence), len(lengths)))
        if int(sequence[index]) < num_embeddings
    ]


def _episode_skill_action_chunks(
    dataset_dir: Path,
    chunk_size: int,
    *,
    target: str = "start_chunk",
) -> dict[int, dict[int, dict[str, np.ndarray | int]]]:
    """Load aligned main-route GT windows for every recorded skill.

    The action slice intentionally continues past the boundary into the next
    skill, matching the Stage-1 dataset's ordinary action-chunk construction.
    ``valid`` marks only offsets that still belong to the selected skill, which
    reproduces ``mask_actions_after_skill_end`` during oracle scoring. A
    full-skill target uses non-overlapping chunk-size windows covering the
    complete skill and records the current observation for each window.
    """
    import pandas as pd

    target = str(target).strip().lower()
    if target not in {"start_chunk", "full_skill"}:
        raise ValueError(
            "oracle latent target must be start_chunk|full_skill, "
            f"got {target!r}."
        )
    columns = ["episode_index", "frame_index", "skill_index", "action"]
    if target == "full_skill":
        columns.extend(("timestamp", "observation.state"))
    files = sorted((dataset_dir / "data").glob("**/*.parquet"))
    frame = pd.concat(
        [
            pd.read_parquet(
                path,
                columns=columns,
            )
            for path in files
        ],
        ignore_index=True,
    )
    result: dict[int, dict[int, dict[str, np.ndarray | int]]] = {}
    for episode, rows in frame.groupby("episode_index", sort=False):
        rows = rows.sort_values("frame_index")
        skill_indices = rows["skill_index"].to_numpy(dtype=np.int64)
        actions = np.stack(
            [np.asarray(value, dtype=np.float32) for value in rows["action"]]
        )
        by_skill: dict[int, dict[str, np.ndarray | int]] = {}
        for skill_index in np.unique(skill_indices):
            starts = np.flatnonzero(skill_indices == skill_index)
            if starts.size == 0:
                continue
            start = int(starts[0])
            skill_end = int(starts[-1]) + 1
            window_starts = (
                [start]
                if target == "start_chunk"
                else list(range(start, skill_end, chunk_size))
            )
            chunks = np.zeros(
                (len(window_starts), chunk_size, actions.shape[1]),
                dtype=np.float32,
            )
            valid = np.zeros((len(window_starts), chunk_size), dtype=bool)
            for window_index, window_start in enumerate(window_starts):
                end = min(window_start + chunk_size, actions.shape[0])
                count = end - window_start
                chunks[window_index, :count] = actions[window_start:end]
                valid[window_index, :count] = (
                    skill_indices[window_start:end] == skill_index
                )
            payload: dict[str, np.ndarray | int] = {
                "actions": chunks,
                "valid": valid,
                "episode_index": int(episode),
            }
            if target == "full_skill":
                payload["states"] = np.stack(
                    [
                        np.asarray(
                            rows.iloc[window_start]["observation.state"],
                            dtype=np.float32,
                        )
                        for window_start in window_starts
                    ]
                )
                payload["timestamps"] = np.asarray(
                    [
                        float(rows.iloc[window_start]["timestamp"])
                        for window_start in window_starts
                    ],
                    dtype=np.float64,
                )
                payload["episode_start_state"] = np.asarray(
                    rows.iloc[0]["observation.state"], dtype=np.float32
                )
            by_skill[int(skill_index)] = payload
        result[int(episode)] = by_skill
    return result


def load_sequences_by_language(dataset_dir: str | Path) -> dict[str, list[list[dict]]]:
    """Return normalized task language -> ordered per-episode GT skill sequences."""
    import pandas as pd  # noqa: F401 - keeps parquet dependency local

    dataset_dir = Path(dataset_dir)
    rows, num_embeddings = _skill_rows(dataset_dir)
    tasks = pd.read_parquet(dataset_dir / "meta" / "tasks.parquet")
    index_to_language = {
        int(task_index): str(language)
        for language, task_index in zip(tasks.index, tasks["task_index"], strict=True)
    }
    result: dict[str, list[list[dict]]] = defaultdict(list)
    for _, row in rows.iterrows():
        skills = _decode_skills(row, num_embeddings)
        if skills:
            result[_norm_language(index_to_language[int(row["task_index"])])].append(skills)
    return dict(result)


def map_sequences_to_tasks(
    task_descriptions: dict[int, str],
    sequences_by_language: dict[str, list[list[dict]]],
) -> dict[int, list[list[dict]]]:
    """Map a LIBERO task id to dataset episodes by normalized language."""
    mapped = {}
    for task_id, description in task_descriptions.items():
        sequences = sequences_by_language.get(_norm_language(description))
        if sequences:
            mapped[int(task_id)] = sequences
    return mapped


def _task_name_to_id(suite_name: str) -> dict[str, int]:
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[suite_name]()
    return {str(task.name): index for index, task in enumerate(suite.tasks)}


def load_episode_exact_data(
    dataset_dir: str | Path,
    init_states_path: str | Path,
    suite_name: str,
    *,
    action_chunk_size: int = 0,
    oracle_latent_target: str = "start_chunk",
) -> dict[int, list[dict]]:
    """Join GT skills with their exact MuJoCo init state, grouped by LIBERO task id."""
    dataset_dir = Path(dataset_dir)
    rows, num_embeddings = _skill_rows(dataset_dir)
    init_data = np.load(str(init_states_path), allow_pickle=True)
    init_by_episode = {
        int(episode): (state, str(scene_file))
        for episode, state, scene_file in zip(
            init_data["episode_index"],
            init_data["init_states"],
            init_data["scene_file"],
            strict=True,
        )
    }
    task_ids = _task_name_to_id(suite_name)
    action_chunks = (
        _episode_skill_action_chunks(
            dataset_dir,
            int(action_chunk_size),
            target=oracle_latent_target,
        )
        if int(action_chunk_size) > 0
        else {}
    )
    result: dict[int, list[dict]] = defaultdict(list)
    for _, row in rows.iterrows():
        episode = int(row["episode_index"])
        if episode not in init_by_episode:
            continue
        init_state, scene_file = init_by_episode[episode]
        task_name = scene_file.removesuffix("_demo.hdf5")
        task_id = task_ids.get(task_name)
        skills = _decode_skills(row, num_embeddings)
        if episode in action_chunks:
            for skill_order, skill in enumerate(skills):
                payload = action_chunks[episode].get(skill_order)
                if payload is None:
                    continue
                actions = np.asarray(payload["actions"])
                valid = np.asarray(payload["valid"])
                if oracle_latent_target == "start_chunk":
                    actions = actions[0]
                    valid = valid[0]
                skill["gt_actions"] = actions
                skill["gt_action_valid"] = valid
                for source_key, target_key in (
                    ("states", "gt_window_states"),
                    ("timestamps", "gt_window_timestamps"),
                    ("episode_start_state", "gt_episode_start_state"),
                ):
                    if source_key in payload:
                        skill[target_key] = np.asarray(payload[source_key])
                skill["gt_episode_index"] = int(payload["episode_index"])
        if task_id is not None and skills:
            result[task_id].append(
                {
                    "episode_index": episode,
                    "init_state": np.asarray(init_state, dtype=np.float64),
                    "skills": skills,
                }
            )
    for records in result.values():
        records.sort(key=lambda record: record["episode_index"])
    return dict(result)
