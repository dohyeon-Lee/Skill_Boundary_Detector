#!/usr/bin/env python3
# Inputs:
#   dataset name : --dataset or inspect_dataset in training_dataset_config.yaml
#   dataset path : {project_root}/{dataset_root}/{dataset}
# Reference model:
#   none
# Outputs:
#   terminal summary: total task count, task language commands, and episode count per task
"""Print task/language/episode counts for a dataset under libero_dataset/.

Examples:
  python inspect_training_dataset.py --dataset libero_90_full_full
  python inspect_training_dataset.py --dataset libero_10
  python inspect_training_dataset.py --dataset libero_90_00to20_full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from training_dataset_config import DEFAULT_CONFIG_PATH, dataset_root_path, get_value, load_config


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
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    pre_args, _ = pre_parser.parse_known_args()
    cfg = load_config(pre_args.config)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[pre_parser],
    )
    parser.add_argument(
        "--dataset",
        default=str(get_value(cfg, "inspect_dataset", "libero_90")),
        help="Dataset name under libero_dataset/",
    )
    parser.add_argument("--root", type=Path, default=dataset_root_path(cfg), help="Dataset root directory")
    args = parser.parse_args()

    dataset_dir = args.root / args.dataset
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
    print(f"  dataset        : {args.dataset}")
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
