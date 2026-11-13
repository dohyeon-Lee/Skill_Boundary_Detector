#!/usr/bin/env python3
# Inputs:
#   settings     : visualized_dataset_config.yaml
#   dataset path : {project_root}/{dataset_root}/{dataset}
# Reference model:
#   none
# Outputs:
#   terminal summary: total task count, task language commands, and episode count per task
"""Print task/language/episode counts for the dataset selected in the shared YAML.

Edit ``visualized_dataset_config.yaml`` and run ``python inspect_training_dataset.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

VISUALIZED_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VISUALIZED_DIR / "src"))

from visualized_dataset_config import (  # noqa: E402
    dataset_settings,
    load_config,
    reject_cli_arguments,
    visualization_settings,
)
from visualize_training_dataset import (  # noqa: E402
    group_episode_tasks,
    load_episodes,
    load_info,
    task_episode_counts,
    video_keys,
)


def main() -> None:
    reject_cli_arguments()
    config = load_config()
    settings = dataset_settings(config)
    visual = visualization_settings(config)
    dataset_dir = settings.dataset_dir
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    info = load_info(dataset_dir)
    tasks, episodes = group_episode_tasks(
        dataset_dir,
        load_episodes(dataset_dir, video_keys(info)),
        visual.task_grouping,
    )
    episode_counts = task_episode_counts(episodes, tasks)
    total_episodes = int(episodes["episode_index"].nunique())

    print("Dataset structure")
    print(f"  dataset        : {settings.dataset}")
    print(f"  path           : {dataset_dir}")
    print(f"  task grouping  : {visual.task_grouping}")
    print(f"  total tasks    : {len(tasks)}")
    print(f"  total episodes : {info.get('total_episodes', total_episodes)}")
    print(f"  total frames   : {info.get('total_frames', '?')}")
    print("")
    print("Tasks")

    for task_idx in sorted(tasks):
        count = int(episode_counts.get(task_idx, 0))
        language = tasks[task_idx]
        print(f"  task{task_idx:02d} | episodes={count:4d} | {language}")


if __name__ == "__main__":
    main()
