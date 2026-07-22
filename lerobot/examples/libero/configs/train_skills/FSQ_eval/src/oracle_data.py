"""Build episode-exact GT skill records directly from original FSQ artifacts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np


def build_task_name_to_id(suite_name: str) -> dict[str, int]:
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[suite_name]()
    return {str(task.name): index for index, task in enumerate(suite.tasks)}


def load_fsq_episode_data(
    latents_path: str | Path,
    init_states_path: str | Path,
    raw_dataset_dir: str | Path,
    suite_name: str,
) -> dict[int, list[dict]]:
    """Join encoded FSQ skills to exact LIBERO reset states by episode id."""
    from lerobot.datasets.io_utils import load_episodes

    with np.load(latents_path, allow_pickle=False) as encoded:
        required = {"tokens", "episode_id", "skill_index", "frame_start", "frame_end", "length"}
        missing = required.difference(encoded.files)
        if missing:
            raise KeyError(f"{latents_path} lacks FSQ eval fields: {sorted(missing)}")
        rows = [
            {
                "token": int(encoded["tokens"][i]),
                "episode_id": int(encoded["episode_id"][i]),
                "skill_index": int(encoded["skill_index"][i]),
                "frame_start": int(encoded["frame_start"][i]),
                "frame_end": int(encoded["frame_end"][i]),
                "length": int(encoded["length"][i]),
            }
            for i in range(len(encoded["tokens"]))
        ]

    init_data = np.load(init_states_path, allow_pickle=True)
    init_by_episode = {
        int(episode): (np.asarray(state, dtype=np.float64), str(scene_file))
        for episode, state, scene_file in zip(
            init_data["episode_index"],
            init_data["init_states"],
            init_data["scene_file"],
            strict=True,
        )
    }
    episode_meta = {
        int(row["episode_index"]): row
        for row in load_episodes(Path(raw_dataset_dir))
    }
    name_to_id = build_task_name_to_id(suite_name)

    skills_by_episode: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        episode = row["episode_id"]
        if episode not in episode_meta:
            raise KeyError(f"Raw dataset metadata has no episode {episode}.")
        if row["frame_end"] <= row["frame_start"]:
            raise ValueError(f"Invalid skill boundary in episode {episode}: {row}")
        skills_by_episode[episode].append(
            {
                "token": row["token"],
                "gt_length": row["length"],
                "skill_index": row["skill_index"],
                "frame_start": row["frame_start"],
                "frame_end": row["frame_end"],
            }
        )

    by_task: dict[int, list[dict]] = defaultdict(list)
    for episode, skills in skills_by_episode.items():
        if episode not in init_by_episode:
            continue
        init_state, scene_file = init_by_episode[episode]
        task_name = scene_file[: -len("_demo.hdf5")] if scene_file.endswith("_demo.hdf5") else scene_file
        task_id = name_to_id.get(task_name)
        if task_id is None:
            continue
        skills.sort(key=lambda item: (item["frame_start"], item["skill_index"]))
        by_task[task_id].append(
            {"episode_index": episode, "init_state": init_state, "skills": skills}
        )

    for records in by_task.values():
        records.sort(key=lambda item: item["episode_index"])
    return dict(by_task)
