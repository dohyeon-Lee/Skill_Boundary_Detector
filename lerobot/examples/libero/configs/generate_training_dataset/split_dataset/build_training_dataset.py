#!/usr/bin/env python3
# Inputs:
#   dataset name : --dataset or build_dataset in training_dataset_config.yaml
#   dataset path : {project_root}/{dataset_root}/{dataset}
#   task range   : --task-range or build_task_range
#   episodes/task: --episodes-per-task or build_episodes_per_task
# Reference model:
#   none
# Outputs:
#   dataset name : {dataset}_{task_range}_{episodes_per_task}[_scene], unless --output-name is set
#   dataset path : {project_root}/{dataset_root}/{output_name}
"""Build a task/episode subset under libero_dataset/.

Examples:
  python build_training_dataset.py --dataset libero_90_full_full --task-range 00to20 --episodes-per-task 10
  python build_training_dataset.py --dataset libero_10_full_full --task-range full --episodes-per-task 5
  python build_training_dataset.py --dataset libero_90_full_full --task-range task00-task30 --episodes-per-task full
  python build_training_dataset.py --dataset libero_90_full_full --task-range full --episodes-per-task firsthalf
  python build_training_dataset.py --dataset libero_90_full_full --task-range full --episodes-per-task lasthalf
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

THIS_DIR = Path(__file__).resolve().parent
# Prefer this checkout over a stale editable install/PYTHONPATH entry.  The
# dataset builder is commonly run directly from this folder, in which case
# Python would otherwise resolve ``lerobot`` from whichever checkout happened
# to be activated in the shell.
sys.path.insert(0, str(THIS_DIR.parents[4] / "src"))
sys.path.insert(0, str(THIS_DIR / "src"))

from training_dataset_config import (
    DEFAULT_CONFIG_PATH,
    as_bool,
    dataset_root_path,
    get_value,
    load_config,
)


def load_dataset(src: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not src.exists():
        raise FileNotFoundError(f"Dataset not found: {src}")

    with open(src / "meta/info.json") as f:
        info = json.load(f)

    tasks_df = pd.read_parquet(src / "meta/tasks.parquet")

    data_parts = [
        pd.read_parquet(path)
        for chunk_dir in sorted((src / "data").iterdir())
        for path in sorted(chunk_dir.glob("*.parquet"))
    ]
    if not data_parts:
        raise FileNotFoundError(f"No data parquet files found under {src / 'data'}")
    data_df = pd.concat(data_parts, ignore_index=True)

    episode_parts = [
        pd.read_parquet(path)
        for chunk_dir in sorted((src / "meta/episodes").iterdir())
        for path in sorted(chunk_dir.glob("*.parquet"))
    ]
    if not episode_parts:
        raise FileNotFoundError(f"No episode parquet files found under {src / 'meta/episodes'}")
    episodes_df = pd.concat(episode_parts, ignore_index=True)

    return info, tasks_df, data_df, episodes_df


def parse_task_range(raw: str, available_tasks: list[int]) -> tuple[list[int], str]:
    text = raw.strip().lower().replace("_", "").replace(" ", "")
    if text == "full":
        return available_tasks, "full"

    cleaned = text.replace("task", "")
    for sep in ("to", "-", ":"):
        if sep in cleaned:
            start_raw, end_raw = cleaned.split(sep, 1)
            start = int(start_raw)
            end = int(end_raw)
            break
    else:
        start = end = int(cleaned)

    if start > end:
        raise ValueError(f"Invalid task range {raw!r}: start must be <= end")

    requested = list(range(start, end + 1))
    missing = sorted(set(requested) - set(available_tasks))
    if missing:
        raise ValueError(f"Requested tasks are not in the dataset: {missing}")
    return requested, f"{start:02d}to{end:02d}"


def parse_episodes_per_task(raw: str) -> tuple[int | str | None, str]:
    text = raw.strip().lower()
    if text == "full":
        return None, "full"
    if text in ("firsthalf", "lasthalf"):
        return text, text
    count = int(text)
    if count <= 0:
        raise ValueError("--episodes-per-task must be a positive integer, 'full', 'firsthalf', or 'lasthalf'")
    return count, str(count)


def make_output_name(
    dataset: str,
    task_tag: str,
    episode_tag: str,
    *,
    scene_aware_halves: bool,
) -> str:
    """Build the default subset name, marking an actually scene-aware half split."""
    base = dataset[: -len("_full_full")] if dataset.endswith("_full_full") else dataset
    name = f"{base}_{task_tag}_{episode_tag}"
    return f"{name}_scene" if scene_aware_halves else name


def load_libero_scene_map(path: Path) -> dict[int, str]:
    """Load the content-matched source episode -> original LIBERO scene-task map."""
    if not path.is_file():
        raise FileNotFoundError(
            f"LIBERO scene-aware half split requires {path}. "
            "Build it first with train_skillVLA/stage1_eval/oracle_matching/run.sh "
            "<source_dataset>."
        )
    payload = np.load(path, allow_pickle=True)
    required = {"episode_index", "scene_file"}
    missing_keys = sorted(required - set(payload.files))
    if missing_keys:
        raise ValueError(f"LIBERO scene map {path} is missing arrays: {missing_keys}")

    episodes = np.asarray(payload["episode_index"]).reshape(-1)
    scenes = np.asarray(payload["scene_file"]).reshape(-1)
    if len(episodes) != len(scenes):
        raise ValueError(
            f"LIBERO scene map length mismatch: episode_index={len(episodes)}, "
            f"scene_file={len(scenes)} ({path})"
        )

    result: dict[int, str] = {}
    for episode, scene in zip(episodes, scenes, strict=True):
        episode = int(episode)
        scene = str(scene)
        previous = result.get(episode)
        if previous is not None and previous != scene:
            raise ValueError(
                f"Episode {episode} maps to conflicting LIBERO scenes: "
                f"{previous!r} vs {scene!r}"
            )
        result[episode] = scene
    return result


def write_subset_libero_scene_map(
    source_path: Path,
    destination: Path,
    selected_episodes: list[int],
) -> None:
    """Subset and reindex the oracle map alongside the newly indexed LeRobot dataset."""
    with np.load(source_path, allow_pickle=True) as payload:
        files = list(payload.files)
        source_episodes = np.asarray(payload["episode_index"]).reshape(-1)
        position = {int(episode): index for index, episode in enumerate(source_episodes)}
        missing = [episode for episode in selected_episodes if episode not in position]
        if missing:
            raise ValueError(
                f"Cannot write subset LIBERO scene map; {len(missing)} selected episode(s) "
                f"are absent from {source_path}."
            )
        indices = np.asarray([position[episode] for episode in selected_episodes], dtype=np.int64)
        output = {}
        for key in files:
            values = np.asarray(payload[key])
            output[key] = (
                values[indices]
                if values.ndim > 0 and len(values) == len(source_episodes)
                else values
            )

    output["source_episode_index"] = np.asarray(selected_episodes, dtype=np.int32)
    output["episode_index"] = np.arange(len(selected_episodes), dtype=np.int32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with open(temporary, "wb") as stream:
        np.savez(stream, **output)
    temporary.replace(destination)


def _select_group_halves(
    episodes: list[int],
    mode: str,
    group_by_episode: dict[int, str],
) -> list[int]:
    """Split each original LIBERO scene-task, preserving the source episode order."""
    missing = [episode for episode in episodes if episode not in group_by_episode]
    if missing:
        preview = ", ".join(map(str, missing[:10]))
        raise ValueError(
            "LIBERO scene map does not cover every candidate episode: "
            f"missing {len(missing)} episode(s), first=[{preview}]. "
            "Rebuild eval_init_states.npz from this exact source dataset."
        )

    grouped: dict[str, list[int]] = {}
    for episode in episodes:
        grouped.setdefault(group_by_episode[episode], []).append(episode)

    keep: set[int] = set()
    for scene, scene_episodes in sorted(grouped.items()):
        half = len(scene_episodes) // 2
        if half == 0 and mode == "firsthalf":
            raise ValueError(
                f"LIBERO scene-task {scene!r} has only {len(scene_episodes)} episode(s); "
                "firsthalf would be empty."
            )
        chosen = scene_episodes[:half] if mode == "firsthalf" else scene_episodes[half:]
        keep.update(chosen)

    # Keep the legacy task_index/episode_index ordering so downstream episode reindexing is stable.
    return [episode for episode in episodes if episode in keep]


def select_episodes(
    data_df: pd.DataFrame,
    task_ids: list[int],
    episodes_per_task: int | str | None,
    allow_fewer: bool,
    *,
    half_group_by_episode: dict[int, str] | None = None,
) -> list[int]:
    ep_task = (
        data_df[["episode_index", "task_index"]]
        .drop_duplicates()
        .sort_values(["task_index", "episode_index"])
    )
    candidate_episodes = [
        int(episode)
        for episode in ep_task.loc[ep_task["task_index"].isin(task_ids), "episode_index"]
    ]
    if (
        episodes_per_task in ("firsthalf", "lasthalf")
        and half_group_by_episode is not None
    ):
        return _select_group_halves(
            candidate_episodes,
            str(episodes_per_task),
            half_group_by_episode,
        )

    selected: list[int] = []
    for task_id in task_ids:
        episodes = ep_task.loc[ep_task["task_index"] == task_id, "episode_index"].tolist()
        if episodes_per_task is None:
            keep = episodes
        elif episodes_per_task in ("firsthalf", "lasthalf"):
            # 태스크별 에피소드를 절반으로 분할. 홀수면 lasthalf가 하나 더 가져간다
            # (n=50: first 0..24 / last 25..49, n=51: first 25개 / last 26개).
            # 두 분할은 서로소이며 합치면 full — 같은 소스로 두 번 돌려 train/heldout 쌍을 만든다.
            half = len(episodes) // 2
            if half == 0 and episodes_per_task == "firsthalf":
                raise ValueError(
                    f"Task {task_id:02d} has only {len(episodes)} episode(s) — "
                    "firsthalf would be empty."
                )
            keep = episodes[:half] if episodes_per_task == "firsthalf" else episodes[half:]
        else:
            if len(episodes) < episodes_per_task and not allow_fewer:
                raise ValueError(
                    f"Task {task_id:02d} only has {len(episodes)} episodes, "
                    f"but {episodes_per_task} were requested. Use --allow-fewer to cap."
                )
            keep = episodes[:episodes_per_task]
        selected.extend(int(ep) for ep in keep)
    return selected


def array_stats(series: pd.Series) -> dict:
    values = np.stack(series.values).astype(np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    count = int(values.shape[0])
    stats = {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [count] * values.shape[1],
    }
    for q in (0.01, 0.10, 0.50, 0.90, 0.99):
        stats[f"q{int(q * 100):02d}"] = np.quantile(values, q, axis=0).tolist()
    return stats


def compute_stats(data_df: pd.DataFrame, src_stats: dict, video_keys: list[str]) -> dict:
    stats = {}
    for col in [
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]:
        if col not in data_df.columns:
            continue
        try:
            stats[col] = array_stats(data_df[col])
        except Exception:
            if col in src_stats:
                stats[col] = src_stats[col]

    for key in video_keys:
        if key in src_stats:
            stats[key] = src_stats[key]
    return stats


def refresh_episode_stats(episodes_df: pd.DataFrame, data_df: pd.DataFrame) -> pd.DataFrame:
    data_cols = [
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    stat_names = ("min", "max", "mean", "std", "count")
    by_episode = {int(ep): group for ep, group in data_df.groupby("episode_index")}

    for row_idx, episode_index in episodes_df["episode_index"].items():
        group = by_episode[int(episode_index)]
        for data_col in data_cols:
            if data_col not in group.columns:
                continue
            required = [f"stats/{data_col}/{name}" for name in stat_names]
            if not any(col in episodes_df.columns for col in required):
                continue
            col_stats = array_stats(group[data_col])
            for name, value in col_stats.items():
                stat_col = f"stats/{data_col}/{name}"
                if stat_col in episodes_df.columns:
                    episodes_df.at[row_idx, stat_col] = value
    return episodes_df


def remap_tasks(tasks_df: pd.DataFrame, task_ids: list[int]) -> tuple[pd.DataFrame, dict[int, int]]:
    task_map = {old: new for new, old in enumerate(task_ids)}
    new_tasks = tasks_df[tasks_df["task_index"].isin(task_ids)].copy()
    new_tasks["task_index"] = new_tasks["task_index"].map(task_map)
    new_tasks = new_tasks.sort_values("task_index")
    return new_tasks, task_map


def _trim_episode_video(src_video: Path, dst_video: Path, from_ts: float, to_ts: float,
                        length: int, fps: float) -> None:
    """Extract ONLY this episode's [from_ts, to_ts] slice from the (multi-episode) source mp4 into a
    fresh per-episode mp4. The new LeRobot format packs MANY episodes per source file, so copying the
    whole file (the old bug) duplicated the entire source per episode → huge files + DINO decoding
    the full source for a tiny slice. Frames are selected by their exact episode timestamps: a raw
    range seek may include the preceding episode's final frame at floating-point boundaries."""
    from lerobot.datasets.video_utils import decode_episode_video_frames  # noqa: PLC0415
    from torchvision.io import write_video  # noqa: PLC0415

    frames = decode_episode_video_frames(
        src_video,
        from_ts,
        to_ts,
        length,
        fps,
        backend="pyav",
        # This is an offline, single-process conversion. Let the decoder use its
        # native thread pool; the one-thread guard is only needed inside forked
        # persistent training DataLoader workers.
        decoder_num_threads=None,
    )
    frames_thwc = (frames.permute(0, 2, 3, 1) * 255.0).round().clamp(0, 255).to(torch.uint8)
    write_video(str(dst_video), frames_thwc, fps=int(round(fps)))


def build_subset(
    src: Path,
    dst: Path,
    info: dict,
    tasks_df: pd.DataFrame,
    data_df: pd.DataFrame,
    episodes_df: pd.DataFrame,
    task_ids: list[int],
    selected_eps: list[int],
    overwrite: bool,
    split_metadata: dict | None = None,
) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {dst}. Pass --overwrite to replace it.")
        shutil.rmtree(dst)

    dst.mkdir(parents=True)
    ep_map = {old: new for new, old in enumerate(selected_eps)}
    new_tasks_df, task_map = remap_tasks(tasks_df, task_ids)

    subset_data = data_df[data_df["episode_index"].isin(selected_eps)].copy()
    subset_data["episode_index"] = subset_data["episode_index"].map(ep_map)
    subset_data["task_index"] = subset_data["task_index"].map(task_map)
    subset_data = subset_data.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    subset_data["index"] = np.arange(len(subset_data), dtype=np.int64)

    (dst / "data/chunk-000").mkdir(parents=True)
    subset_data.to_parquet(dst / "data/chunk-000/file-000.parquet", index=False)

    fps = float(info.get("fps", 20.0))
    video_keys = [key for key, value in info["features"].items() if value.get("dtype") == "video"]
    for video_key in video_keys:
        (dst / "videos" / video_key / "chunk-000").mkdir(parents=True)
        for done, (old_ep, new_ep) in enumerate(ep_map.items(), start=1):
            if done % 100 == 0 or done == len(ep_map):
                print(f"  trim {video_key}: {done}/{len(ep_map)}")
            ep_row = episodes_df[episodes_df["episode_index"] == old_ep].iloc[0]
            src_chunk = int(ep_row[f"videos/{video_key}/chunk_index"])
            src_file = int(ep_row[f"videos/{video_key}/file_index"])
            src_video = src / "videos" / video_key / f"chunk-{src_chunk:03d}" / f"file-{src_file:03d}.mp4"
            dst_video = dst / "videos" / video_key / "chunk-000" / f"file-{new_ep:03d}.mp4"
            # TRIM this episode's slice out of the packed source (NOT a whole-file copy — see helper).
            _trim_episode_video(src_video, dst_video,
                                float(ep_row[f"videos/{video_key}/from_timestamp"]),
                                float(ep_row[f"videos/{video_key}/to_timestamp"]),
                                int(ep_row["length"]), fps)

    subset_eps = episodes_df[episodes_df["episode_index"].isin(selected_eps)].copy()
    subset_eps["source_episode_index"] = subset_eps["episode_index"]
    subset_eps["source_task_index"] = subset_eps["source_episode_index"].map(
        data_df[["episode_index", "task_index"]].drop_duplicates().set_index("episode_index")["task_index"]
    )
    subset_eps["episode_index"] = subset_eps["episode_index"].map(ep_map)
    subset_eps = subset_eps.sort_values("episode_index").reset_index(drop=True)

    ranges = (
        subset_data.groupby("episode_index")["index"]
        .agg(dataset_from_index="min", dataset_to_index="max")
        .reset_index()
    )
    ranges["dataset_to_index"] += 1
    subset_eps = subset_eps.drop(columns=["dataset_from_index", "dataset_to_index"], errors="ignore")
    subset_eps = subset_eps.merge(ranges, on="episode_index", how="left")

    for video_key in video_keys:
        subset_eps[f"videos/{video_key}/chunk_index"] = 0
        subset_eps[f"videos/{video_key}/file_index"] = subset_eps["episode_index"]
        # Videos are now TRIMMED per episode → each starts at t=0, spans length/fps (not the old
        # offset into the packed source). Without this the reader would seek to a nonexistent offset.
        subset_eps[f"videos/{video_key}/from_timestamp"] = 0.0
        subset_eps[f"videos/{video_key}/to_timestamp"] = subset_eps["length"] / fps

    subset_eps["data/chunk_index"] = 0
    subset_eps["data/file_index"] = 0
    subset_eps["meta/episodes/chunk_index"] = 0
    subset_eps["meta/episodes/file_index"] = 0
    print("  episode metadata + stats ...")
    subset_eps = refresh_episode_stats(subset_eps, subset_data)

    (dst / "meta/episodes/chunk-000").mkdir(parents=True)
    subset_eps.to_parquet(dst / "meta/episodes/chunk-000/file-000.parquet", index=False)
    new_tasks_df.to_parquet(dst / "meta/tasks.parquet")

    with open(src / "meta/stats.json") as f:
        src_stats = json.load(f)
    with open(dst / "meta/stats.json", "w") as f:
        json.dump(compute_stats(subset_data, src_stats, video_keys), f, indent=2)

    new_info = dict(info)
    new_info["total_episodes"] = len(selected_eps)
    new_info["total_frames"] = int(len(subset_data))
    new_info["total_tasks"] = int(len(new_tasks_df))
    new_info["splits"] = {"train": f"0:{len(selected_eps)}"}
    new_info["chunks_size"] = 1000
    new_info["source_dataset"] = src.name
    new_info["source_task_indices"] = [int(task_id) for task_id in task_ids]
    new_info["source_episode_indices"] = [int(ep) for ep in selected_eps]
    if split_metadata:
        new_info.update(split_metadata)
    new_info.pop("data_files_size_in_mb", None)
    new_info.pop("video_files_size_in_mb", None)

    with open(dst / "meta/info.json", "w") as f:
        json.dump(new_info, f, indent=2)

    print(f"Built dataset: {dst}")
    print(f"  tasks    : {new_info['total_tasks']}")
    print(f"  episodes : {new_info['total_episodes']}")
    print(f"  frames   : {new_info['total_frames']}")


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    pre_args, _ = pre_parser.parse_known_args()
    cfg = load_config(pre_args.config)
    default_root = dataset_root_path(cfg)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[pre_parser],
    )
    parser.add_argument(
        "--dataset",
        default=str(get_value(cfg, "build_dataset", "libero_90")),
        help="Source dataset under libero_dataset/, e.g. libero_90",
    )
    parser.add_argument(
        "--task-range",
        default=str(get_value(cfg, "build_task_range", "full")),
        help="full, 00to20, 0-20, task00-task20, or a single task id",
    )
    parser.add_argument(
        "--episodes-per-task",
        default=str(get_value(cfg, "build_episodes_per_task", "full")),
        help="full, a number, or firsthalf/lasthalf (태스크별 앞/뒤 절반; 홀수면 lasthalf가 +1)",
    )
    parser.add_argument("--src-root", type=Path, default=default_root)
    parser.add_argument("--dst-root", type=Path, default=default_root)
    parser.add_argument(
        "--output-name",
        default=str(get_value(cfg, "build_output_name", "")),
        help="Override auto name: {dataset}_{range}_{episodes}[_scene]",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=as_bool(get_value(cfg, "build_overwrite", False)),
        help="Replace an existing output dataset",
    )
    parser.add_argument(
        "--allow-fewer",
        action=argparse.BooleanOptionalAction,
        default=as_bool(get_value(cfg, "build_allow_fewer", False)),
        help="Use all available episodes when a task has fewer than requested",
    )
    parser.add_argument(
        "--libero-scene-aware-halves",
        action=argparse.BooleanOptionalAction,
        default=as_bool(get_value(cfg, "build_libero_scene_aware_halves", True)),
        help=(
            "For LIBERO firsthalf/lasthalf, split each original scene-task separately "
            "instead of splitting merged language task_index groups"
        ),
    )
    configured_scene_map = str(get_value(cfg, "build_libero_scene_map", "") or "").strip()
    parser.add_argument(
        "--libero-scene-map",
        type=Path,
        default=Path(configured_scene_map).expanduser() if configured_scene_map else None,
        help=(
            "Source eval_init_states.npz. Empty uses "
            "<src-root>/skillvla_dataset/<dataset>/eval_init_states.npz"
        ),
    )
    args = parser.parse_args()

    src = args.src_root / args.dataset
    info, tasks_df, data_df, episodes_df = load_dataset(src)

    available_tasks = sorted(int(v) for v in data_df["task_index"].drop_duplicates().tolist())
    task_ids, task_tag = parse_task_range(args.task_range, available_tasks)
    episodes_per_task, episode_tag = parse_episodes_per_task(args.episodes_per_task)
    scene_map_path = None
    half_group_by_episode = None
    if (
        args.libero_scene_aware_halves
        and args.dataset.startswith("libero_")
        and episodes_per_task in ("firsthalf", "lasthalf")
    ):
        scene_map_path = args.libero_scene_map or (
            args.src_root / "skillvla_dataset" / args.dataset / "eval_init_states.npz"
        )
        if not scene_map_path.is_absolute():
            scene_map_path = Path(str(get_value(cfg, "project_root"))) / scene_map_path
        half_group_by_episode = load_libero_scene_map(scene_map_path)
    selected_eps = select_episodes(
        data_df,
        task_ids,
        episodes_per_task,
        args.allow_fewer,
        half_group_by_episode=half_group_by_episode,
    )

    if not selected_eps:
        raise ValueError("No episodes were selected")

    # Filtered suites are named "<suite>_full_full" (full tasks / full episodes). Strip that base tag
    # so a subset build doesn't double it. An actually active scene-aware half split gets a `_scene`
    # suffix so it cannot be confused with the legacy task_index-only half dataset. Explicit
    # --output-name/build_output_name always wins.
    output_name = args.output_name or make_output_name(
        args.dataset,
        task_tag,
        episode_tag,
        scene_aware_halves=scene_map_path is not None,
    )
    dst = args.dst_root / output_name

    print("Training dataset generation")
    print(f"  source       : {src}")
    print(f"  output       : {dst}")
    print(f"  task range   : {task_tag} ({len(task_ids)} tasks)")
    print(f"  episodes/task: {episode_tag}")
    if scene_map_path is not None:
        selected_scenes = {half_group_by_episode[episode] for episode in selected_eps}
        print(f"  half split   : LIBERO scene-aware ({len(selected_scenes)} scene-tasks)")
        print(f"  scene map    : {scene_map_path}")
    print(f"  selected eps : {len(selected_eps)}")

    build_subset(
        src=src,
        dst=dst,
        info=info,
        tasks_df=tasks_df,
        data_df=data_df,
        episodes_df=episodes_df,
        task_ids=task_ids,
        selected_eps=selected_eps,
        overwrite=args.overwrite,
        split_metadata=(
            {
                "episode_half_split_unit": "libero_scene_file",
                "libero_scene_aware_halves": True,
            }
            if scene_map_path is not None
            else None
        ),
    )
    if scene_map_path is not None:
        subset_scene_map = (
            args.dst_root / "skillvla_dataset" / output_name / "eval_init_states.npz"
        )
        write_subset_libero_scene_map(scene_map_path, subset_scene_map, selected_eps)
        print(f"  oracle map: {subset_scene_map}")
        stale_entries = [
            path
            for path in subset_scene_map.parent.iterdir()
            if path.name != subset_scene_map.name
        ]
        if stale_entries:
            print(
                "  [warning] Existing derived SkillVLA artifacts under this source are now stale; "
                "rebuild them from the new subset."
            )


if __name__ == "__main__":
    main()
