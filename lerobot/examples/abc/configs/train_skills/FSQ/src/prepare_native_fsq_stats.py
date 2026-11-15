#!/usr/bin/env python3
"""Create/reuse task-selection q01/q99 stats for ABC absolute-action FSQ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.datasets.abc_native_dataset import ABCNativeDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--tasks-json", required=True)
    args = parser.parse_args()
    tasks = [str(task) for task in json.loads(args.tasks_json)]
    dataset = ABCNativeDataset(
        repo_id="lerobot/abc_sim_all",
        root=args.dataset_dir,
        tasks=tasks,
        video_keys_to_load=[],
        require_complete_action_chunks=False,
    )
    if dataset.selection_stats_path is None:
        raise RuntimeError("ABC task selection did not produce a stats path.")
    print(dataset.selection_stats_path)


if __name__ == "__main__":
    main()
