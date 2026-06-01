#!/usr/bin/env python3
# Inputs:
#   LeRobot data : {project_root}/libero_original_dataset/{dataset}
#   config       : ../training_dataset_config.yaml
# Reference model:
#   none
# Outputs:
#   stats file   : {project_root}/libero_original_dataset/{dataset}/meta/stats.json
"""Check or recompute local LeRobot quantile stats for converted LIBERO data.

Converted datasets created by the current local LeRobot v3 writer already
include q01/q10/q50/q90/q99 in meta/stats.json. This script is mainly a guard:
it exits quickly when quantile stats already exist, and recomputes them only
when they are missing or --overwrite is passed.

Examples:
  python ensure_quantile_stats.py --dataset libero_90_full_full
  python ensure_quantile_stats.py --dataset libero_10_full_full --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
GENERATE_DIR = THIS_DIR.parent
CONFIG_PATH = GENERATE_DIR / "training_dataset_config.yaml"

sys.path.insert(0, str(GENERATE_DIR / "src"))
from training_dataset_config import load_config, project_root


QUANTILE_KEYS = ["q01", "q10", "q50", "q90", "q99"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--dataset", required=True, help="Dataset folder name under libero_original_dataset/")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Converted dataset root. Default: {project_root}/libero_original_dataset",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Local LeRobot repo id. Default: dohyeon/{dataset}",
    )
    parser.add_argument("--overwrite", action="store_true", help="Recompute stats even if quantiles exist.")
    return parser.parse_args()


def feature_stats_have_quantiles(stats: dict) -> bool:
    if not stats:
        return False
    missing: dict[str, list[str]] = {}
    for feature, feature_stats in stats.items():
        if not isinstance(feature_stats, dict):
            missing[feature] = QUANTILE_KEYS
            continue
        absent = [key for key in QUANTILE_KEYS if key not in feature_stats]
        if absent:
            missing[feature] = absent
    if missing:
        print("Missing quantile stats:")
        for feature, absent in missing.items():
            print(f"  {feature}: {', '.join(absent)}")
        return False
    return True


def load_stats(stats_path: Path) -> dict | None:
    if not stats_path.exists():
        return None
    with open(stats_path) as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root = args.root or (project_root(cfg) / "libero_original_dataset")
    dataset_dir = root / args.dataset
    stats_path = dataset_dir / "meta" / "stats.json"

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    stats = load_stats(stats_path)
    if stats is not None and feature_stats_have_quantiles(stats) and not args.overwrite:
        print("Quantile stats already exist")
        print(f"  dataset : {args.dataset}")
        print(f"  path    : {stats_path}")
        return

    project_dir = project_root(cfg)
    lerobot_src = project_dir / "lerobot" / "src"
    sys.path.insert(0, str(lerobot_src))

    from lerobot.datasets.io_utils import write_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.scripts.augment_dataset_quantile_stats import compute_quantile_stats_for_dataset

    repo_id = args.repo_id or f"dohyeon/{args.dataset}"
    print("Computing local quantile stats")
    print(f"  dataset : {args.dataset}")
    print(f"  root    : {dataset_dir}")
    print(f"  repo id : {repo_id}")
    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_dir)
    new_stats = compute_quantile_stats_for_dataset(dataset)
    write_stats(new_stats, dataset_dir)

    print("DONE")
    print(f"  updated : {stats_path}")


if __name__ == "__main__":
    main()
