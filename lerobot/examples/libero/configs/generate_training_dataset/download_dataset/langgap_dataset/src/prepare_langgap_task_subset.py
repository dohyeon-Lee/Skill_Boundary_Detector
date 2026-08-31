#!/usr/bin/env python3
"""Prepare a task-filtered LangGap HF staging directory.

LangGap videos are packed: one MP4 may contain episodes from several tasks.  A
task-only download therefore first downloads the small metadata/parquet files,
uses them to find the minimum set of packed MP4 shards, and finally rewrites the
local staging metadata to expose only the requested tasks.  The packed source
files can still contain unused byte ranges; the canonical converter decodes only
the selected episode ranges and its final output contains exactly the subset.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from langgap_dataset_config import (
    DEFAULT_CONFIG_PATH,
    langgap_root,
    load_config,
    task_subset_ids,
)

MARKER_NAME = ".langgap_task_subset.json"


def _read_parquet_tree(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def _load_staging(staging: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    info_path = staging / "meta/info.json"
    tasks_path = staging / "meta/tasks.parquet"
    if not info_path.is_file() or not tasks_path.is_file():
        raise FileNotFoundError(
            f"Download meta/* and data/* before preparing the task subset: {staging}"
        )
    return (
        json.loads(info_path.read_text()),
        _read_parquet_tree(staging / "data"),
        _read_parquet_tree(staging / "meta/episodes"),
        pd.read_parquet(tasks_path),
    )


def _episode_task_table(data: pd.DataFrame) -> pd.DataFrame:
    pairs = data[["episode_index", "task_index"]].drop_duplicates()
    duplicated = pairs["episode_index"].duplicated(keep=False)
    if duplicated.any():
        bad = sorted(int(value) for value in pairs.loc[duplicated, "episode_index"].unique())
        raise ValueError(f"Episodes contain more than one task_index: {bad[:20]}")
    return pairs.sort_values("episode_index")


def _selection(
    data: pd.DataFrame, episodes: pd.DataFrame, task_ids: list[int]
) -> tuple[tuple[int, ...], pd.DataFrame]:
    pairs = _episode_task_table(data)
    available = {int(value) for value in pairs["task_index"].unique()}
    missing = sorted(set(task_ids) - available)
    if missing:
        raise ValueError(f"Source dataset does not contain task indices {missing}")
    selected = tuple(
        int(value)
        for value in pairs.loc[pairs["task_index"].isin(task_ids), "episode_index"]
    )
    metadata_episodes = {int(value) for value in episodes["episode_index"]}
    missing_metadata = sorted(set(selected) - metadata_episodes)
    if missing_metadata:
        raise ValueError(f"Selected episodes are missing metadata: {missing_metadata[:20]}")
    return selected, pairs


def required_video_files(staging: Path, task_ids: list[int]) -> list[str]:
    info, data, episodes, _ = _load_staging(staging)
    selected, _ = _selection(data, episodes, task_ids)
    selected_rows = episodes[episodes["episode_index"].isin(selected)]
    paths: set[str] = set()
    for key, feature in info.get("features", {}).items():
        if not isinstance(feature, dict) or feature.get("dtype") != "video":
            continue
        chunk_column = f"videos/{key}/chunk_index"
        file_column = f"videos/{key}/file_index"
        missing = sorted({chunk_column, file_column} - set(episodes.columns))
        if missing:
            raise ValueError(f"Episode metadata is missing video columns: {missing}")
        for chunk, file_index in selected_rows[[chunk_column, file_column]].itertuples(
            index=False, name=None
        ):
            paths.add(
                f"videos/{key}/chunk-{int(chunk):03d}/file-{int(file_index):03d}.mp4"
            )
    if not paths:
        raise ValueError("No video files were selected.")
    return sorted(paths)


def _refresh_episode_ranges(episodes: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    ranges = (
        data.groupby("episode_index")["index"]
        .agg(dataset_from_index="min", dataset_to_index="max")
        .reset_index()
    )
    ranges["dataset_to_index"] += 1
    episodes = episodes.drop(
        columns=["dataset_from_index", "dataset_to_index"], errors="ignore"
    ).merge(ranges, on="episode_index", how="left")
    episodes["data/chunk_index"] = 0
    episodes["data/file_index"] = 0
    episodes["meta/episodes/chunk_index"] = 0
    episodes["meta/episodes/file_index"] = 0
    return episodes


def finalize_subset(staging: Path, set_name: str, repo_id: str, task_ids: list[int]) -> dict[str, Any]:
    info, data, episodes, tasks = _load_staging(staging)
    selected, episode_tasks = _selection(data, episodes, task_ids)
    required_videos = required_video_files(staging, task_ids)
    missing_videos = [path for path in required_videos if not (staging / path).is_file()]
    if missing_videos:
        raise FileNotFoundError(f"Required video files have not been downloaded: {missing_videos}")

    selected_set = set(selected)
    episode_map = {source: output for output, source in enumerate(selected)}
    task_map = {source: output for output, source in enumerate(task_ids)}

    subset_data = data[data["episode_index"].isin(selected_set)].copy()
    subset_data["episode_index"] = subset_data["episode_index"].map(episode_map)
    subset_data["task_index"] = subset_data["task_index"].map(task_map)
    subset_data = subset_data.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    subset_data["index"] = np.arange(len(subset_data), dtype=np.int64)

    subset_episodes = episodes[episodes["episode_index"].isin(selected_set)].copy()
    subset_episodes["source_episode_index"] = subset_episodes["episode_index"]
    source_task_by_episode = episode_tasks.set_index("episode_index")["task_index"]
    subset_episodes["source_task_index"] = subset_episodes["episode_index"].map(
        source_task_by_episode
    )
    subset_episodes["episode_index"] = subset_episodes["episode_index"].map(episode_map)
    subset_episodes = subset_episodes.sort_values("episode_index").reset_index(drop=True)
    subset_episodes = _refresh_episode_ranges(subset_episodes, subset_data)

    subset_tasks = tasks[tasks["task_index"].isin(task_ids)].copy()
    subset_tasks["task_index"] = subset_tasks["task_index"].map(task_map)
    subset_tasks = subset_tasks.sort_values("task_index")
    if len(subset_tasks) != len(task_ids):
        raise ValueError(
            f"Task metadata contains {len(subset_tasks)} selected rows, expected {len(task_ids)}."
        )

    subset_info = dict(info)
    subset_info.update(
        {
            "total_episodes": len(selected),
            "total_frames": len(subset_data),
            "total_tasks": len(task_ids),
            "splits": {"train": f"0:{len(selected)}"},
            "source_repo_id": repo_id,
            "source_task_indices": task_ids,
            "source_episode_indices": list(selected),
            "download_derivation": "minimum_packed_video_shards_then_local_task_filter",
        }
    )

    temporary = Path(tempfile.mkdtemp(prefix=".task-subset-", dir=staging))
    try:
        data_path = temporary / "data/chunk-000/file-000.parquet"
        episodes_path = temporary / "episodes/chunk-000/file-000.parquet"
        data_path.parent.mkdir(parents=True)
        episodes_path.parent.mkdir(parents=True)
        subset_data.to_parquet(data_path, index=False)
        subset_episodes.to_parquet(episodes_path, index=False)
        subset_tasks.to_parquet(temporary / "tasks.parquet")
        (temporary / "info.json").write_text(json.dumps(subset_info, indent=2))

        shutil.rmtree(staging / "data")
        (temporary / "data").replace(staging / "data")
        shutil.rmtree(staging / "meta/episodes")
        (temporary / "episodes").replace(staging / "meta/episodes")
        (temporary / "tasks.parquet").replace(staging / "meta/tasks.parquet")
        (temporary / "info.json").replace(staging / "meta/info.json")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

    required_set = set(required_videos)
    for video_path in (staging / "videos").glob("**/*.mp4"):
        relative = video_path.relative_to(staging).as_posix()
        if relative not in required_set:
            video_path.unlink()

    marker = {
        "set": set_name,
        "source_repo_id": repo_id,
        "source_task_indices": task_ids,
        "output_task_indices": list(range(len(task_ids))),
        "episodes": len(selected),
        "frames": len(subset_data),
        "required_packed_video_files": required_videos,
        "note": "Packed files may contain unused episode byte ranges; canonical output is exact.",
    }
    marker_path = staging / MARKER_NAME
    marker_path.write_text(json.dumps(marker, indent=2))
    return marker


def subset_is_complete(staging: Path, task_ids: list[int]) -> bool:
    marker_path = staging / MARKER_NAME
    try:
        marker = json.loads(marker_path.read_text())
        info = json.loads((staging / "meta/info.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    if marker.get("source_task_indices") != task_ids:
        return False
    if info.get("source_task_indices") != task_ids or info.get("total_tasks") != len(task_ids):
        return False
    required = marker.get("required_packed_video_files") or []
    return bool(required) and all((staging / str(path)).is_file() for path in required)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--set", required=True)
    parser.add_argument(
        "--staging",
        type=Path,
        default=None,
        help="Override staging directory (mainly for validation).",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--print-video-files", action="store_true")
    action.add_argument("--finalize", action="store_true")
    action.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    sets = dict(cfg.get("langgap_sets") or {})
    if args.set not in sets:
        raise ValueError(f"Unknown LangGap set {args.set!r}; available={sorted(sets)}")
    task_ids = task_subset_ids(cfg, args.set)
    if not task_ids:
        raise ValueError(f"No download_task_ids_by_set entry for {args.set!r}")
    staging = args.staging or langgap_root(cfg) / "_hf" / args.set

    if args.check:
        raise SystemExit(0 if subset_is_complete(staging, task_ids) else 1)
    if args.print_video_files:
        for path in required_video_files(staging, task_ids):
            print(path)
        return

    marker = finalize_subset(staging, args.set, str(sets[args.set]), task_ids)
    print(
        f"Prepared {args.set}: tasks={len(task_ids)}, episodes={marker['episodes']}, "
        f"frames={marker['frames']}, packed_videos={len(marker['required_packed_video_files'])}"
    )


if __name__ == "__main__":
    main()
