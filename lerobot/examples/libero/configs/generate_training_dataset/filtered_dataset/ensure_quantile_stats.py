#!/usr/bin/env python3
# Inputs:
#   LeRobot data : {project_root}/{filtered_dataset_root}/{dataset}
#   config       : ./filtered_dataset_config.yaml  (stats_include_videos: fast/full 모드)
# Outputs:
#   stats file   : .../{dataset}/meta/stats.json
"""Check or recompute quantile stats for a FILTERED LIBERO dataset (q01/q10/q50/q90/q99).

두 모드 (yaml `stats_include_videos`, CLI --include-videos로 override):
  fast (false, 기본): 비-비디오 feature만 parquet에서 quantile(+min/max/mean/std) 계산해
    기존 stats.json에 병합. 비디오 feature는 변환 시 집계된 min/max/mean/std 유지 —
    DP(MEAN_STD)/pi05·skillVLA(IDENTITY)에 충분하며 비디오 디코딩이 없어 몇 분.
  full (true): 모든 feature를 에피소드별 비디오 디코딩 포함 전체 재계산 — 수 시간.

이미 충분한 stats가 있으면 빠르게 종료; --overwrite로 강제 재계산.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR / "src"))
from filtered_dataset_config import DEFAULT_CONFIG_PATH, filtered_root, load_config, project_root  # noqa: E402

QUANTILE_KEYS = ["q01", "q10", "q50", "q90", "q99"]
QUANTILE_Q = [0.01, 0.10, 0.50, 0.90, 0.99]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset", required=True, help="Dataset folder name under the filtered root")
    parser.add_argument("--root", type=Path, default=None,
                        help="Datasets root. Default: {project_root}/{filtered_dataset_root}")
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


def fast_recompute(dataset_dir: Path, video_keys: set[str], lerobot_src: Path) -> None:
    """비-비디오 feature의 stat(quantile 포함)을 data parquet에서 계산해 stats.json에 병합.

    lerobot 생태계 표준 방식 그대로: 에피소드별 get_feature_stats(히스토그램 quantile 추정)
    → aggregate_feature_stats(count-가중 평균). 이는 writer가 데이터셋 생성 시 쓰는 경로와
    동일하므로 기존 데이터셋들(libero_90, libero_90_full_full)의 stats와 값이 일치한다.
    (전역 exact quantile이 수학적으로는 더 정확하지만, 생태계 일관성을 위해 표준을 따름)"""
    import pandas as pd

    sys.path.insert(0, str(lerobot_src))
    from lerobot.datasets.compute_stats import (  # noqa: PLC0415
        DEFAULT_QUANTILES, aggregate_feature_stats, get_feature_stats,
    )

    stats_path = dataset_dir / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    files = sorted((dataset_dir / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data parquet under {dataset_dir}")
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"])
    feats = [f for f in (stats.keys() or df.columns) if f not in video_keys and f in df.columns]

    per_feat_eps: dict[str, list[dict]] = {f: [] for f in feats}
    for _, g in df.groupby("episode_index"):
        for feat in feats:
            col = g[feat].to_numpy()
            # augment의 process_single_episode와 동일한 모양/호출: 벡터 (T,D) → axis=0,
            # 스칼라 (T,) → axis=0 + keepdims (reshape 없이 그대로).
            arr = np.stack(col) if isinstance(col[0], np.ndarray) else np.asarray(col)
            per_feat_eps[feat].append(
                get_feature_stats(arr, axis=0, keepdims=arr.ndim == 1, quantile_list=DEFAULT_QUANTILES)
            )

    for feat in feats:
        agg = aggregate_feature_stats(per_feat_eps[feat])
        stats[feat] = {k: (np.asarray(v).tolist() if k != "count" else [int(np.asarray(v).reshape(-1)[0])])
                       for k, v in agg.items()}
    stats_path.write_text(json.dumps(stats, indent=4))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    include_videos = args.include_videos or bool(cfg.get("stats_include_videos", False))
    root = args.root or filtered_root(cfg)
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
        print(f"Computing quantile stats (fast: non-video features only): {dataset_dir}")
        fast_recompute(dataset_dir, vkeys, project_root(cfg) / "lerobot" / "src")
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
