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
QUANTILE_Q = [0.01, 0.10, 0.50, 0.90, 0.99]


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


def _video_keys(dataset_dir: Path) -> set[str]:
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    return {k for k, v in info["features"].items() if v.get("dtype") == "video"}


def recompute_global_stats(dataset_dir: Path, stats_path: Path, stats: dict) -> None:
    """비-비디오 feature의 stat(quantile 포함)을 data parquet 전체에서 GLOBAL exact로 계산해 병합.

    quantile은 전체 프레임을 모아 np.quantile로 한 번에 계산한다 — 에피소드별 quantile을 count-가중
    평균하는 lerobot aggregate는 짧은 에피소드들의 q01/q99를 평균내 전역 꼬리를 중심으로 끌어당겨
    좁게 만들기 때문(편향). min/max/mean/std도 전역으로 계산. 비디오 feature는 변환 시 값을 유지."""
    import numpy as np
    import pandas as pd

    video_keys = _video_keys(dataset_dir)
    files = sorted((dataset_dir / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data parquet under {dataset_dir}")
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    feats = [f for f in (stats.keys() or df.columns) if f not in video_keys and f in df.columns]
    for feat in feats:
        col = df[feat].to_numpy()
        arr = (np.stack(col) if isinstance(col[0], np.ndarray) else np.asarray(col).reshape(len(col), -1)).astype(np.float64)
        block = {"min": arr.min(0), "max": arr.max(0), "mean": arr.mean(0), "std": arr.std(0),
                 "count": np.array([arr.shape[0]])}
        for key, q in zip(QUANTILE_KEYS, QUANTILE_Q):
            block[key] = np.quantile(arr, q, axis=0)
        stats[feat] = {k: ([int(np.asarray(v).reshape(-1)[0])] if k == "count"
                           else [float(x) for x in np.asarray(v).ravel()]) for k, v in block.items()}
    stats_path.write_text(json.dumps(stats, indent=4))


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

    print("Computing GLOBAL exact quantile stats (non-video) from parquet")
    print(f"  dataset : {args.dataset}")
    print(f"  root    : {dataset_dir}")
    recompute_global_stats(dataset_dir, stats_path, stats or {})

    print("DONE")
    print(f"  updated : {stats_path}")


if __name__ == "__main__":
    main()
