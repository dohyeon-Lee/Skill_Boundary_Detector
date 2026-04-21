"""
Reconstruct a lerobot-format libero dataset into a smaller subset.

Usage:
  # Print dataset info only
  python reconstruct_dataset.py --dataset libero_10

  # Option 1: Keep 30% of episodes from ALL tasks
  python reconstruct_dataset.py --dataset libero_10 --option 1 --value 30
  → output: libero_small_dataset/libero_10_option1_30

  # Option 2: Keep ALL episodes from first 4 tasks only
  python reconstruct_dataset.py --dataset libero_10 --option 2 --value 4
  → output: libero_small_dataset/libero_10_option2_4
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path("/scratch/mdorazi/Skill_Boundary_Detector")
SRC_ROOT = BASE_DIR / "libero_dataset"
DST_ROOT = BASE_DIR / "libero_small_dataset"


# ──────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────

def load_dataset(dataset_name: str):
    src = SRC_ROOT / dataset_name
    assert src.exists(), f"Dataset not found: {src}"

    with open(src / "meta/info.json") as f:
        info = json.load(f)

    tasks_df = pd.read_parquet(src / "meta/tasks.parquet")

    # Concatenate all data parquets (handles multiple chunks)
    data_parts = []
    for chunk_dir in sorted((src / "data").iterdir()):
        for pf in sorted(chunk_dir.glob("*.parquet")):
            data_parts.append(pd.read_parquet(pf))
    data_df = pd.concat(data_parts, ignore_index=True)

    # Concatenate all episodes parquets
    ep_parts = []
    for chunk_dir in sorted((src / "meta/episodes").iterdir()):
        for pf in sorted(chunk_dir.glob("*.parquet")):
            ep_parts.append(pd.read_parquet(pf))
    episodes_df = pd.concat(ep_parts, ignore_index=True)

    return src, info, tasks_df, data_df, episodes_df


def print_info(dataset_name: str, data_df: pd.DataFrame, tasks_df: pd.DataFrame):
    ep_task = data_df[["episode_index", "task_index"]].drop_duplicates()
    task_counts = ep_task.groupby("task_index")["episode_index"].count()

    # tasks_df index is the task string, task_index is a column
    task_name_map = tasks_df["task_index"].reset_index().set_index("task_index")["task"].to_dict() \
        if "task" in tasks_df.reset_index().columns else {}

    print(f"\n{'='*60}")
    print(f"Dataset : {dataset_name}")
    print(f"Tasks   : {len(task_counts)}")
    print(f"Episodes: {ep_task['episode_index'].nunique()}")
    print(f"Frames  : {len(data_df)}")
    print(f"\nEpisodes per task:")
    for tid, cnt in task_counts.items():
        name = task_name_map.get(tid, "")
        print(f"  Task {tid:2d}: {cnt:3d} ep  |  {str(name)[:65]}")
    print("="*60)


# ──────────────────────────────────────────────────────────────
# Episode selection
# ──────────────────────────────────────────────────────────────

def select_option1(data_df: pd.DataFrame, percent: float) -> list[int]:
    """Keep ceil(percent%) episodes from each task, preserving order."""
    ep_task = data_df[["episode_index", "task_index"]].drop_duplicates()
    selected = []
    for _, group in ep_task.groupby("task_index"):
        eps = sorted(group["episode_index"].tolist())
        n_keep = max(1, math.ceil(len(eps) * percent / 100.0))
        selected.extend(eps[:n_keep])
    return sorted(selected)


def select_option2(data_df: pd.DataFrame, n_tasks: int) -> list[int]:
    """Keep ALL episodes from the first n_tasks tasks (task_index 0..n_tasks-1)."""
    ep_task = data_df[["episode_index", "task_index"]].drop_duplicates()
    keep = ep_task[ep_task["task_index"] < n_tasks]["episode_index"].tolist()
    return sorted(keep)


# ──────────────────────────────────────────────────────────────
# Stats computation
# ──────────────────────────────────────────────────────────────

def _col_stats(series: pd.Series) -> dict:
    vals = np.stack(series.values).astype(np.float64)
    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)
    n = vals.shape[0]
    return {
        "min":   vals.min(axis=0).tolist(),
        "max":   vals.max(axis=0).tolist(),
        "mean":  vals.mean(axis=0).tolist(),
        "std":   vals.std(axis=0).tolist(),
        "count": [int(n)] * vals.shape[1],
    }


def compute_stats(data_df: pd.DataFrame, orig_stats: dict, video_keys: list[str]) -> dict:
    numeric_cols = ["observation.state", "action", "timestamp",
                    "frame_index", "episode_index", "index", "task_index"]
    stats = {}
    for col in numeric_cols:
        if col in data_df.columns:
            try:
                stats[col] = _col_stats(data_df[col])
            except Exception:
                if col in orig_stats:
                    stats[col] = orig_stats[col]
    # Video pixel stats: copy from original (subset doesn't change per-pixel distribution much)
    for vkey in video_keys:
        if vkey in orig_stats:
            stats[vkey] = orig_stats[vkey]
    return stats


# ──────────────────────────────────────────────────────────────
# Build subset dataset
# ──────────────────────────────────────────────────────────────

def build_subset(
    dataset_name: str,
    selected_eps: list[int],
    option: int,
    value,
    src: Path,
    info: dict,
    tasks_df: pd.DataFrame,
    data_df: pd.DataFrame,
    episodes_df: pd.DataFrame,
) -> Path:
    out_name = f"{dataset_name}_option{option}_{value}"
    dst = DST_ROOT / out_name

    if dst.exists():
        print(f"Removing existing {dst}")
        shutil.rmtree(dst)

    print(f"\nBuilding '{out_name}' ...")

    # ── episode & task index maps ──────────────────────────────
    ep_map = {old: new for new, old in enumerate(selected_eps)}  # old → new

    if option == 2:
        n_tasks = int(value)
        task_map = {i: i for i in range(n_tasks)}
        new_tasks_df = tasks_df[tasks_df["task_index"] < n_tasks].copy()
    else:
        task_map = {i: i for i in tasks_df["task_index"].values}
        new_tasks_df = tasks_df.copy()

    # ── filter & remap data parquet ───────────────────────────
    subset_data = data_df[data_df["episode_index"].isin(selected_eps)].copy()
    subset_data["episode_index"] = subset_data["episode_index"].map(ep_map)
    subset_data["task_index"] = subset_data["task_index"].map(task_map)
    subset_data = (subset_data
                   .sort_values(["episode_index", "frame_index"])
                   .reset_index(drop=True))
    subset_data["index"] = np.arange(len(subset_data), dtype=np.int64)

    (dst / "data/chunk-000").mkdir(parents=True)
    subset_data.to_parquet(dst / "data/chunk-000/file-000.parquet", index=False)
    print(f"  data parquet written  ({len(subset_data):,} frames)")

    # ── copy videos ───────────────────────────────────────────
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    for vkey in video_keys:
        (dst / "videos" / vkey / "chunk-000").mkdir(parents=True)
        for old_ep, new_ep in ep_map.items():
            ep_row = episodes_df[episodes_df["episode_index"] == old_ep].iloc[0]
            orig_chunk = int(ep_row[f"videos/{vkey}/chunk_index"])
            orig_file  = int(ep_row[f"videos/{vkey}/file_index"])
            src_vid = src / "videos" / vkey / f"chunk-{orig_chunk:03d}" / f"file-{orig_file:03d}.mp4"
            dst_vid = dst / "videos" / vkey / "chunk-000" / f"file-{new_ep:03d}.mp4"
            shutil.copy(src_vid, dst_vid)
        print(f"  videos/{vkey}: {len(ep_map)} files copied")

    # ── build episodes metadata ───────────────────────────────
    subset_eps = episodes_df[episodes_df["episode_index"].isin(selected_eps)].copy()
    subset_eps["episode_index"] = subset_eps["episode_index"].map(ep_map)
    subset_eps = subset_eps.sort_values("episode_index").reset_index(drop=True)

    # Recompute frame ranges from new global index
    ep_ranges = (subset_data
                 .groupby("episode_index")["index"]
                 .agg(dataset_from_index="min", dataset_to_index="max")
                 .reset_index())
    ep_ranges["dataset_to_index"] += 1  # exclusive end

    subset_eps = subset_eps.drop(columns=["dataset_from_index", "dataset_to_index"], errors="ignore")
    subset_eps = subset_eps.merge(ep_ranges, on="episode_index", how="left")

    # Fix video chunk/file indices
    for vkey in video_keys:
        subset_eps[f"videos/{vkey}/chunk_index"] = 0
        subset_eps[f"videos/{vkey}/file_index"]  = subset_eps["episode_index"]

    # Fix data chunk/file indices (all in one parquet now)
    subset_eps["data/chunk_index"] = 0
    subset_eps["data/file_index"]  = 0

    # Fix meta/episodes chunk/file index
    subset_eps["meta/episodes/chunk_index"] = 0
    subset_eps["meta/episodes/file_index"]  = 0

    (dst / "meta/episodes/chunk-000").mkdir(parents=True)
    subset_eps.to_parquet(dst / "meta/episodes/chunk-000/file-000.parquet", index=False)
    print(f"  episodes metadata written  ({len(subset_eps)} episodes)")

    # ── tasks.parquet ─────────────────────────────────────────
    new_tasks_df.to_parquet(dst / "meta/tasks.parquet")

    # ── stats.json ────────────────────────────────────────────
    with open(src / "meta/stats.json") as f:
        orig_stats = json.load(f)
    stats = compute_stats(subset_data, orig_stats, video_keys)
    with open(dst / "meta/stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ── info.json ─────────────────────────────────────────────
    new_info = dict(info)
    new_info["total_episodes"] = len(selected_eps)
    new_info["total_frames"]   = int(len(subset_data))
    new_info["total_tasks"]    = int(len(new_tasks_df))
    new_info["splits"]         = {"train": f"0:{len(selected_eps)}"}
    new_info["chunks_size"]    = 1000
    new_info.pop("data_files_size_in_mb",  None)
    new_info.pop("video_files_size_in_mb", None)

    with open(dst / "meta/info.json", "w") as f:
        json.dump(new_info, f, indent=2)

    print(f"\n✓ Done: {dst}")
    print(f"  tasks={new_info['total_tasks']}  episodes={new_info['total_episodes']}  frames={new_info['total_frames']:,}")
    print("  NOTE: for pi05 training, run augment_dataset_quantile_stats.py on this dataset to add q01/q99 stats.")
    return dst


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct a lerobot libero dataset into a smaller subset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", default="libero_10",
                        help="Dataset name inside libero_dataset/ (default: libero_10)")
    parser.add_argument("--option", type=int, choices=[1, 2],
                        help="1 = percent of episodes per task, 2 = first N tasks")
    parser.add_argument("--value",
                        help="For option1: percentage (e.g. 30), for option2: number of tasks (e.g. 4)")
    args = parser.parse_args()

    src, info, tasks_df, data_df, episodes_df = load_dataset(args.dataset)
    print_info(args.dataset, data_df, tasks_df)

    if args.option is None:
        print("\n(No --option given, showing info only.)")
        return

    if args.value is None:
        parser.error("--value is required when --option is specified")

    DST_ROOT.mkdir(parents=True, exist_ok=True)

    if args.option == 1:
        percent = float(args.value)
        assert 0 < percent <= 100, "Percentage must be between 0 and 100"
        selected = select_option1(data_df, percent)
        print(f"\nOption 1 ({percent}% per task): {len(selected)} episodes selected")
        build_subset(args.dataset, selected, 1, int(percent),
                     src, info, tasks_df, data_df, episodes_df)

    elif args.option == 2:
        n_tasks = int(args.value)
        total_tasks = data_df["task_index"].nunique()
        assert 1 <= n_tasks <= total_tasks, f"n_tasks must be 1..{total_tasks}"
        selected = select_option2(data_df, n_tasks)
        print(f"\nOption 2 (first {n_tasks} tasks): {len(selected)} episodes selected")
        build_subset(args.dataset, selected, 2, n_tasks,
                     src, info, tasks_df, data_df, episodes_df)


if __name__ == "__main__":
    main()
