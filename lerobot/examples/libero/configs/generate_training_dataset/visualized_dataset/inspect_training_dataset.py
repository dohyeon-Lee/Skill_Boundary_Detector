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

import json
import sys
from pathlib import Path

import pandas as pd

VISUALIZED_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VISUALIZED_DIR / "src"))

from visualized_dataset_config import dataset_settings, load_config, reject_cli_arguments  # noqa: E402


def load_data_parquets(dataset_dir: Path) -> pd.DataFrame:
    paths = [
        path
        for chunk_dir in sorted((dataset_dir / "data").iterdir())
        for path in sorted(chunk_dir.glob("*.parquet"))
    ]
    if not paths:
        raise FileNotFoundError(f"No data parquet files found under {dataset_dir / 'data'}")
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def task_language_map(tasks_df: pd.DataFrame) -> dict[int, str]:
    if "task" in tasks_df.reset_index().columns:
        task_col = tasks_df.reset_index()["task"]
        index_col = tasks_df.reset_index()["task_index"]
        return {int(task_idx): str(task) for task_idx, task in zip(index_col, task_col, strict=True)}

    return {int(row["task_index"]): str(idx) for idx, row in tasks_df.iterrows()}


def main() -> None:
    reject_cli_arguments()
    settings = dataset_settings(load_config())
    dataset_dir = settings.dataset_dir
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    info_path = dataset_dir / "meta/info.json"
    tasks_path = dataset_dir / "meta/tasks.parquet"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json: {info_path}")
    if not tasks_path.exists():
        raise FileNotFoundError(f"Missing tasks.parquet: {tasks_path}")

    with open(info_path) as f:
        info = json.load(f)
    tasks_df = pd.read_parquet(tasks_path)
    data_df = load_data_parquets(dataset_dir)

    ep_task = data_df[["episode_index", "task_index"]].drop_duplicates()
    episode_counts = ep_task.groupby("task_index")["episode_index"].nunique().to_dict()
    language_by_task = task_language_map(tasks_df)

    print("Dataset structure")
    print(f"  dataset        : {settings.dataset}")
    print(f"  path           : {dataset_dir}")
    print(f"  total tasks    : {info.get('total_tasks', len(language_by_task))}")
    print(f"  total episodes : {info.get('total_episodes', ep_task['episode_index'].nunique())}")
    print(f"  total frames   : {info.get('total_frames', len(data_df))}")
    print("")
    print("Tasks")

    for task_idx in sorted(language_by_task):
        count = int(episode_counts.get(task_idx, 0))
        language = language_by_task[task_idx]
        print(f"  task{task_idx:02d} | episodes={count:4d} | {language}")

    missing = sorted(set(episode_counts) - set(language_by_task))
    if missing:
        print("")
        print(f"WARNING: data contains task ids missing from tasks.parquet: {missing}")


if __name__ == "__main__":
    main()
