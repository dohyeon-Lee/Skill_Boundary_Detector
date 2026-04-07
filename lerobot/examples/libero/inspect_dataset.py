"""
LeRobot 데이터셋 구조 검사 스크립트.

Usage:
    python examples/libero/inspect_dataset.py \
        --dataset_dir /scratch/mdorazi/Skill_Boundary_Detector/libero_dataset/libero_10
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tyro


@dataclass
class Args:
    dataset_dir: str = "/scratch/mdorazi/Skill_Boundary_Detector/libero_dataset/libero_90"
    episode_idx: int = 0
    """출력할 예시 에피소드 인덱스"""


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main(args: Args) -> None:
    dataset_dir = Path(args.dataset_dir)

    # ── info.json ──────────────────────────────────────────────────
    print_section("Dataset Info")
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    print(f"  codebase_version : {info.get('codebase_version')}")
    print(f"  robot_type       : {info.get('robot_type')}")
    print(f"  fps              : {info.get('fps')}")
    print(f"  total_episodes   : {info.get('total_episodes')}")
    print(f"  total_frames     : {info.get('total_frames')}")
    print(f"  total_tasks      : {info.get('total_tasks')}")

    # ── features ──────────────────────────────────────────────────
    print_section("Features")
    for key, feat in info.get("features", {}).items():
        dtype = feat.get("dtype")
        shape = feat.get("shape")
        print(f"  {key:<50} dtype={dtype}  shape={shape}")

    # ── tasks ──────────────────────────────────────────────────────
    print_section("Tasks (first 5)")
    tasks_path = dataset_dir / "meta" / "tasks.parquet"
    if tasks_path.exists():
        tasks = pd.read_parquet(tasks_path)
        print(tasks.head(5).to_string())
    else:
        print("  tasks.parquet 없음")

    # ── stats ──────────────────────────────────────────────────────
    print_section("Stats (action & state)")
    stats_path = dataset_dir / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        for key in ["action", "observation.state"]:
            if key in stats:
                s = stats[key]
                print(f"\n  [{key}]")
                for stat_name in ["min", "max", "mean", "std"]:
                    if stat_name in s:
                        vals = np.array(s[stat_name])
                        print(f"    {stat_name:5s}: {np.round(vals, 4).tolist()}")
    else:
        print("  stats.json 없음")

    # ── 샘플 에피소드 데이터 ───────────────────────────────────────
    print_section(f"Sample Episode (ep_idx={args.episode_idx})")
    data_files = sorted((dataset_dir / "data").rglob("*.parquet"))
    df_all = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)
    ep_df = df_all[df_all["episode_index"] == args.episode_idx]

    if ep_df.empty:
        print(f"  episode_index={args.episode_idx} 없음")
    else:
        print(f"  frames     : {len(ep_df)}")
        print(f"  columns    : {ep_df.columns.tolist()}")

        for col in ep_df.columns:
            sample = ep_df[col].iloc[0]
            if isinstance(sample, (list, np.ndarray)):
                arr = np.array(sample)
                print(f"\n  [{col}]  shape={arr.shape}")
                print(f"    first frame: {np.round(arr, 4).tolist()}")
            else:
                print(f"\n  [{col}]  value={sample}")

    # ── videos ─────────────────────────────────────────────────────
    print_section("Video Files")
    video_dir = dataset_dir / "videos"
    if video_dir.exists():
        cams = [d.name for d in video_dir.iterdir() if d.is_dir()]
        print(f"  cameras: {cams}")
        for cam in cams:
            mp4s = sorted(video_dir.glob(f"{cam}/**/*.mp4"))
            print(f"  {cam}: {len(mp4s)} mp4 files")
    else:
        print("  videos/ 없음")

    print("\n")


if __name__ == "__main__":
    main(tyro.cli(Args))
