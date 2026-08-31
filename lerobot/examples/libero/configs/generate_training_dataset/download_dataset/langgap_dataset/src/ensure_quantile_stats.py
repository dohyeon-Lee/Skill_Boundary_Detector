#!/usr/bin/env python3
# Inputs:
#   LeRobot data : {project_root}/{langgap_root}/{dataset}
#   config       : ../langgap_dataset_config.yaml  (stats_include_videos: fast/full 모드)
# Outputs:
#   stats file   : .../{dataset}/meta/stats.json
"""Check or recompute quantile stats for a LangGap dataset (q01/q10/q50/q90/q99).

filtered_dataset/src/ensure_quantile_stats.py 와 동일한 로직 (config 모듈만 langgap):
  fast (기본): 비-비디오 feature만 parquet에서 GLOBAL exact quantile 계산해 병합 — 몇 분.
  full: 전 feature를 비디오 디코딩 포함 재계산 — 수 시간.

python src/ensure_quantile_stats.py --dataset langgap_56_full_full [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from langgap_dataset_config import DEFAULT_CONFIG_PATH, langgap_root, load_config, project_root  # noqa: E402

QUANTILE_KEYS = ["q01", "q10", "q50", "q90", "q99"]
QUANTILE_Q = [0.01, 0.10, 0.50, 0.90, 0.99]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset", required=True, help="Dataset folder name under the langgap root")
    parser.add_argument("--root", type=Path, default=None,
                        help="Datasets root. Default: {project_root}/{langgap_root}")
    parser.add_argument("--repo-id", default=None, help="Local LeRobot repo id. Default: dohyeon/{dataset}")
    parser.add_argument("--include-videos", action="store_true",
                        help="full 모드 강제 (yaml stats_include_videos 무시)")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if quantiles exist.")
    return parser.parse_args()


def video_keys_of(dataset_dir: Path) -> set[str]:
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    return {k for k, v in info["features"].items() if v.get("dtype") == "video"}


def stats_sufficient(stats: dict | None, video_keys: set[str], include_videos: bool) -> bool:
    """fast 모드: 비-비디오 feature만 quantile 요구. full 모드: 전 feature 요구."""
    if not stats:
        return False
    for feat, fstats in stats.items():
        if not include_videos and feat in video_keys:
            continue  # 비디오는 min/max/mean/std만 있으면 충분
        if not isinstance(fstats, dict) or any(k not in fstats for k in QUANTILE_KEYS):
            return False
    return True


def fast_recompute(dataset_dir: Path, video_keys: set[str]) -> None:
    """비-비디오 feature의 stat(quantile 포함)을 data parquet 전체에서 GLOBAL exact로 계산해 병합.

    (filtered_dataset과 동일: 에피소드별 quantile의 count-가중 평균은 전역 꼬리를 좁히는
    편향이 있어 쓰지 않고, 전체 프레임을 모아 한 번에 np.quantile로 계산한다.)"""
    import pandas as pd

    stats_path = dataset_dir / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
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
    include_videos = args.include_videos or bool(cfg.get("stats_include_videos", False))
    root = args.root or langgap_root(cfg)
    dataset_dir = root / args.dataset
    stats_path = dataset_dir / "meta" / "stats.json"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    vkeys = video_keys_of(dataset_dir)
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else None
    if stats_sufficient(stats, vkeys, include_videos) and not args.overwrite:
        print(f"Stats already sufficient ({'full' if include_videos else 'fast'} mode): {stats_path}")
        return

    if not include_videos:
        print(f"Computing GLOBAL exact quantile stats (fast: non-video features only): {dataset_dir}")
        fast_recompute(dataset_dir, vkeys)
        print(f"DONE → {stats_path} (video features keep converted min/max/mean/std)")
        return

    sys.path.insert(0, str(project_root(cfg) / "lerobot" / "src"))
    from lerobot.datasets.io_utils import write_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.scripts.augment_dataset_quantile_stats import compute_quantile_stats_for_dataset

    repo_id = args.repo_id or f"dohyeon/{args.dataset}"
    print(f"Computing quantile stats (full, incl. videos): {dataset_dir} (repo id {repo_id})")
    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_dir)
    write_stats(compute_quantile_stats_for_dataset(dataset), dataset_dir)
    print(f"DONE → {stats_path}")


if __name__ == "__main__":
    main()
