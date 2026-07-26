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


def _decode_skills(row, num_embeddings: int) -> list[dict[str, int]]:
    sequence = np.asarray(row["skill_sequence"]).reshape(-1)
    lengths = np.asarray(row["skill_length_sequence"]).reshape(-1)
    return [
        {"token": int(sequence[index]), "gt_length": int(lengths[index])}
        for index in range(min(len(sequence), len(lengths)))
        if int(sequence[index]) < num_embeddings
    ]


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
    result: dict[int, list[dict]] = defaultdict(list)
    for _, row in rows.iterrows():
        episode = int(row["episode_index"])
        if episode not in init_by_episode:
            continue
        init_state, scene_file = init_by_episode[episode]
        task_name = scene_file.removesuffix("_demo.hdf5")
        task_id = task_ids.get(task_name)
        skills = _decode_skills(row, num_embeddings)
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
