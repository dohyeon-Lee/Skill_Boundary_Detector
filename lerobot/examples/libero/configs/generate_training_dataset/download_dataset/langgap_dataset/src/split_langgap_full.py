#!/usr/bin/env python3
"""Split canonical langgap_56 into official40 and extension16 without video re-encoding.

The canonical full dataset keeps the two task groups in disjoint packed MP4 shards.
This script filters and reindexes parquet metadata, recomputes non-video statistics,
and hardlinks each whole video shard into the derived datasets.

Examples:
  python src/split_langgap_full.py --root dataset --dry-run
  python src/split_langgap_full.py --root dataset
  python src/split_langgap_full.py --root dataset --set langgap_original_full_full
  python src/split_langgap_full.py --root dataset --include-videos
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from langgap_dataset_config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    langgap_root,
    load_config,
    project_root,
)

NON_VIDEO_STATS_FEATURES = (
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)
STAT_QUANTILES = (("q01", 0.01), ("q10", 0.10), ("q50", 0.50), ("q90", 0.90), ("q99", 0.99))


@dataclass(frozen=True)
class SourceDataset:
    root: Path
    info: dict[str, Any]
    tasks: pd.DataFrame
    data: pd.DataFrame
    episodes: pd.DataFrame


@dataclass(frozen=True)
class SplitSpec:
    name: str
    source: str
    task_ids: tuple[int, ...]


def _read_parquet_tree(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def load_source(root: Path) -> SourceDataset:
    info_path = root / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Canonical source dataset not found: {info_path}")
    return SourceDataset(
        root=root,
        info=json.loads(info_path.read_text()),
        tasks=pd.read_parquet(root / "meta/tasks.parquet"),
        data=_read_parquet_tree(root / "data"),
        episodes=_read_parquet_tree(root / "meta/episodes"),
    )


def parse_task_range(raw: str) -> tuple[int, ...]:
    text = str(raw).strip().lower().replace("task", "").replace(" ", "")
    for separator in ("to", "-", ":"):
        if separator in text:
            first, last = text.split(separator, 1)
            start, end = int(first), int(last)
            break
    else:
        start = end = int(text)
    if start < 0 or start > end:
        raise ValueError(f"Invalid task range: {raw!r}")
    return tuple(range(start, end + 1))


def configured_splits(cfg: dict[str, Any]) -> dict[str, SplitSpec]:
    raw = cfg.get("langgap_derived_sets") or {}
    if not isinstance(raw, dict):
        raise TypeError("langgap_derived_sets must be a mapping.")
    result: dict[str, SplitSpec] = {}
    for name, value in raw.items():
        if not isinstance(value, dict) or not value.get("source") or not value.get("task_range"):
            raise ValueError(
                f"langgap_derived_sets[{name!r}] needs source and task_range."
            )
        result[str(name)] = SplitSpec(
            name=str(name),
            source=str(value["source"]),
            task_ids=parse_task_range(str(value["task_range"])),
        )
    return result


def resolve_root(cfg: dict[str, Any], override: Path | None) -> Path:
    if override is None:
        return langgap_root(cfg)
    return override if override.is_absolute() else project_root(cfg) / override


def video_keys(info: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        key
        for key, feature in info.get("features", {}).items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    )


def array_stats(series: pd.Series) -> dict[str, list[float] | list[int]]:
    raw = series.to_numpy()
    first = raw[0]
    if isinstance(first, (np.ndarray, list, tuple)):
        values = np.stack(raw)
    else:
        values = np.asarray(raw).reshape(len(raw), 1)
    values = values.astype(np.float64)
    count = int(values.shape[0])
    result: dict[str, list[float] | list[int]] = {
        "min": values.min(axis=0).reshape(-1).tolist(),
        "max": values.max(axis=0).reshape(-1).tolist(),
        "mean": values.mean(axis=0).reshape(-1).tolist(),
        "std": values.std(axis=0).reshape(-1).tolist(),
        "count": [count],
    }
    for key, quantile in STAT_QUANTILES:
        result[key] = np.quantile(values, quantile, axis=0).reshape(-1).tolist()
    return result


def recompute_global_stats(
    data: pd.DataFrame,
    source_stats: dict[str, Any],
    videos: tuple[str, ...],
) -> dict[str, Any]:
    """Recompute exact global parquet stats; retain video stats unless explicitly decoded later."""
    stats: dict[str, Any] = {}
    for feature in NON_VIDEO_STATS_FEATURES:
        if feature in data.columns:
            stats[feature] = array_stats(data[feature])
    for feature in videos:
        if feature in source_stats:
            stats[feature] = source_stats[feature]
    return stats


def refresh_episode_stats(episodes: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """Refresh per-episode parquet stats affected by episode/task/index remapping."""
    groups = {int(index): frame for index, frame in data.groupby("episode_index", sort=False)}
    for row_index, episode_index in episodes["episode_index"].items():
        frame = groups[int(episode_index)]
        for feature in NON_VIDEO_STATS_FEATURES:
            if feature not in frame.columns:
                continue
            block = array_stats(frame[feature])
            for stat, value in block.items():
                column = f"stats/{feature}/{stat}"
                if column in episodes.columns:
                    episodes.at[row_index, column] = value
    return episodes


def episode_task_table(data: pd.DataFrame) -> pd.DataFrame:
    pairs = data[["episode_index", "task_index"]].drop_duplicates()
    duplicated = pairs["episode_index"].duplicated(keep=False)
    if duplicated.any():
        raise ValueError("At least one episode contains more than one task_index.")
    return pairs.sort_values("episode_index")


def exclusive_video_file_maps(
    episodes: pd.DataFrame,
    selected: set[int],
    videos: tuple[str, ...],
) -> dict[str, dict[tuple[int, int], tuple[int, int]]]:
    """Require whole-shard ownership, then assign compact output file indices."""
    chosen_rows = episodes[episodes["episode_index"].isin(selected)]
    other_rows = episodes[~episodes["episode_index"].isin(selected)]
    mappings: dict[str, dict[tuple[int, int], tuple[int, int]]] = {}
    for key in videos:
        chunk_column = f"videos/{key}/chunk_index"
        file_column = f"videos/{key}/file_index"
        required = {chunk_column, file_column}
        missing = sorted(required - set(episodes.columns))
        if missing:
            raise ValueError(f"Episode metadata is missing video columns: {missing}")
        chosen = {
            (int(chunk), int(file))
            for chunk, file in chosen_rows[[chunk_column, file_column]].itertuples(
                index=False, name=None
            )
        }
        other = {
            (int(chunk), int(file))
            for chunk, file in other_rows[[chunk_column, file_column]].itertuples(
                index=False, name=None
            )
        }
        mixed = sorted(chosen & other)
        if mixed:
            raise ValueError(
                f"{key} has packed shards shared by selected and rejected episodes: {mixed}. "
                "Lossless splitting is unsafe; use the generic re-encoding builder."
            )
        mappings[key] = {
            source: (0, output_file)
            for output_file, source in enumerate(sorted(chosen))
        }
    return mappings


def plan_split(source: SourceDataset, spec: SplitSpec) -> dict[str, Any]:
    available = {int(value) for value in source.data["task_index"].unique()}
    missing = sorted(set(spec.task_ids) - available)
    if missing:
        raise ValueError(f"{source.root.name} does not contain task indices {missing}")
    pairs = episode_task_table(source.data)
    selected_episodes = tuple(
        int(value)
        for value in pairs.loc[pairs["task_index"].isin(spec.task_ids), "episode_index"]
    )
    selected_set = set(selected_episodes)
    episode_metadata = {int(value) for value in source.episodes["episode_index"]}
    absent_metadata = sorted(selected_set - episode_metadata)
    if absent_metadata:
        raise ValueError(f"Selected episodes missing metadata: {absent_metadata[:20]}")
    videos = video_keys(source.info)
    file_maps = exclusive_video_file_maps(source.episodes, selected_set, videos)
    frames = int(source.data["episode_index"].isin(selected_set).sum())
    return {
        "selected_episodes": selected_episodes,
        "selected_set": selected_set,
        "episode_tasks": pairs,
        "video_keys": videos,
        "video_file_maps": file_maps,
        "frames": frames,
    }


def _hardlink_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _recompute_full_video_stats(dataset_root: Path, repo_id: str, project: Path) -> None:
    sys.path.insert(0, str(project / "lerobot/src"))
    from lerobot.datasets.io_utils import write_stats  # noqa: PLC0415
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415
    from lerobot.scripts.augment_dataset_quantile_stats import (  # noqa: PLC0415
        compute_quantile_stats_for_dataset,
    )

    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)
    write_stats(compute_quantile_stats_for_dataset(dataset), dataset_root)


def build_split(
    source: SourceDataset,
    destination: Path,
    spec: SplitSpec,
    plan: dict[str, Any],
    *,
    include_videos: bool,
    project: Path,
) -> None:
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent)
    )
    selected_episodes: tuple[int, ...] = plan["selected_episodes"]
    selected_set: set[int] = plan["selected_set"]
    episode_map = {old: new for new, old in enumerate(selected_episodes)}
    task_map = {old: new for new, old in enumerate(spec.task_ids)}
    try:
        subset_data = source.data[source.data["episode_index"].isin(selected_set)].copy()
        subset_data["episode_index"] = subset_data["episode_index"].map(episode_map)
        subset_data["task_index"] = subset_data["task_index"].map(task_map)
        subset_data = subset_data.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
        subset_data["index"] = np.arange(len(subset_data), dtype=np.int64)
        data_path = temporary / "data/chunk-000/file-000.parquet"
        data_path.parent.mkdir(parents=True)
        subset_data.to_parquet(data_path, index=False)

        subset_episodes = source.episodes[
            source.episodes["episode_index"].isin(selected_set)
        ].copy()
        subset_episodes["source_episode_index"] = subset_episodes["episode_index"]
        source_tasks = plan["episode_tasks"].set_index("episode_index")["task_index"]
        subset_episodes["source_task_index"] = subset_episodes["episode_index"].map(source_tasks)
        subset_episodes["episode_index"] = subset_episodes["episode_index"].map(episode_map)
        subset_episodes = subset_episodes.sort_values("episode_index").reset_index(drop=True)

        ranges = (
            subset_data.groupby("episode_index")["index"]
            .agg(dataset_from_index="min", dataset_to_index="max")
            .reset_index()
        )
        ranges["dataset_to_index"] += 1
        subset_episodes = subset_episodes.drop(
            columns=["dataset_from_index", "dataset_to_index"], errors="ignore"
        ).merge(ranges, on="episode_index", how="left")
        subset_episodes["data/chunk_index"] = 0
        subset_episodes["data/file_index"] = 0
        subset_episodes["meta/episodes/chunk_index"] = 0
        subset_episodes["meta/episodes/file_index"] = 0

        link_modes: set[str] = set()
        for key, mapping in plan["video_file_maps"].items():
            chunk_column = f"videos/{key}/chunk_index"
            file_column = f"videos/{key}/file_index"
            old_pairs = list(
                zip(subset_episodes[chunk_column], subset_episodes[file_column], strict=True)
            )
            subset_episodes[chunk_column] = [
                mapping[(int(chunk), int(file))][0] for chunk, file in old_pairs
            ]
            subset_episodes[file_column] = [
                mapping[(int(chunk), int(file))][1] for chunk, file in old_pairs
            ]
            for (old_chunk, old_file), (new_chunk, new_file) in mapping.items():
                source_video = (
                    source.root / "videos" / key / f"chunk-{old_chunk:03d}" /
                    f"file-{old_file:03d}.mp4"
                )
                output_video = (
                    temporary / "videos" / key / f"chunk-{new_chunk:03d}" /
                    f"file-{new_file:03d}.mp4"
                )
                if not source_video.is_file():
                    raise FileNotFoundError(f"Missing packed video: {source_video}")
                link_modes.add(_hardlink_or_copy(source_video, output_video))

        subset_episodes = refresh_episode_stats(subset_episodes, subset_data)
        episodes_path = temporary / "meta/episodes/chunk-000/file-000.parquet"
        episodes_path.parent.mkdir(parents=True)
        subset_episodes.to_parquet(episodes_path, index=False)

        subset_tasks = source.tasks[source.tasks["task_index"].isin(spec.task_ids)].copy()
        subset_tasks["task_index"] = subset_tasks["task_index"].map(task_map)
        subset_tasks = subset_tasks.sort_values("task_index")
        subset_tasks.to_parquet(temporary / "meta/tasks.parquet")

        source_stats = json.loads((source.root / "meta/stats.json").read_text())
        stats = recompute_global_stats(subset_data, source_stats, plan["video_keys"])
        (temporary / "meta/stats.json").write_text(json.dumps(stats, indent=2))

        subset_info = dict(source.info)
        subset_info.update(
            {
                "total_episodes": len(selected_episodes),
                "total_frames": int(len(subset_data)),
                "total_tasks": len(spec.task_ids),
                "splits": {"train": f"0:{len(selected_episodes)}"},
                "source_dataset": source.root.name,
                "source_task_indices": list(spec.task_ids),
                "source_episode_indices": list(selected_episodes),
                "video_derivation": (
                    "hardlink_whole_shards"
                    if link_modes == {"hardlink"}
                    else "+".join(sorted(link_modes))
                ),
                "stats_derivation": {
                    "non_video": "recomputed_exact_from_subset_parquet",
                    "video": "recomputed_exact_with_decode" if include_videos else "inherited_from_source",
                },
            }
        )
        (temporary / "meta/info.json").write_text(json.dumps(subset_info, indent=2))
        provenance = {
            "source_dataset": source.root.name,
            "source_task_indices": list(spec.task_ids),
            "output_task_indices": list(range(len(spec.task_ids))),
            "episodes": len(selected_episodes),
            "frames": int(len(subset_data)),
            "video_file_maps": {
                key: {
                    f"{old[0]:03d}/{old[1]:03d}": f"{new[0]:03d}/{new[1]:03d}"
                    for old, new in mapping.items()
                }
                for key, mapping in plan["video_file_maps"].items()
            },
        }
        (temporary / "meta/langgap_split.json").write_text(json.dumps(provenance, indent=2))

        if include_videos:
            print(f"  {spec.name}: decoding videos for full stats (slow) ...", flush=True)
            _recompute_full_video_stats(temporary, f"local/{spec.name}", project)

        validate_output(temporary, source, spec, plan)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(f"Built {destination}")
    print(f"  tasks    : {len(spec.task_ids)} ({spec.task_ids[0]}..{spec.task_ids[-1]} -> 0..{len(spec.task_ids)-1})")
    print(f"  episodes : {len(selected_episodes)}")
    print(f"  frames   : {len(subset_data)}")
    print(f"  videos   : {sum(len(value) for value in plan['video_file_maps'].values())} packed files ({','.join(sorted(link_modes))})")


def validate_output(
    output: Path,
    source: SourceDataset,
    spec: SplitSpec,
    plan: dict[str, Any],
) -> None:
    info = json.loads((output / "meta/info.json").read_text())
    data = _read_parquet_tree(output / "data")
    episodes = _read_parquet_tree(output / "meta/episodes")
    tasks = pd.read_parquet(output / "meta/tasks.parquet")
    expected_episodes = len(plan["selected_episodes"])
    checks = {
        "info.total_tasks": info["total_tasks"] == len(spec.task_ids),
        "info.total_episodes": info["total_episodes"] == expected_episodes,
        "info.total_frames": info["total_frames"] == plan["frames"],
        "tasks rows": len(tasks) == len(spec.task_ids),
        "episode rows": len(episodes) == expected_episodes,
        "data rows": len(data) == plan["frames"],
        "episode indices": set(data["episode_index"]) == set(range(expected_episodes)),
        "task indices": set(data["task_index"]) == set(range(len(spec.task_ids))),
        "frame indices": np.array_equal(data["index"].to_numpy(), np.arange(len(data))),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Derived dataset validation failed: {failed}")
    for key, mapping in plan["video_file_maps"].items():
        for (old_chunk, old_file), (new_chunk, new_file) in mapping.items():
            old_path = source.root / "videos" / key / f"chunk-{old_chunk:03d}" / f"file-{old_file:03d}.mp4"
            new_path = output / "videos" / key / f"chunk-{new_chunk:03d}" / f"file-{new_file:03d}.mp4"
            if not new_path.is_file() or new_path.stat().st_size != old_path.stat().st_size:
                raise RuntimeError(f"Derived video validation failed: {new_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Dataset root; relative paths resolve from project_root. Default: LangGap/global config.",
    )
    parser.add_argument(
        "--set",
        action="append",
        dest="sets",
        help="Derived output name; repeat to select multiple. Default: all langgap_derived_sets.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate split boundaries without writing.")
    parser.add_argument(
        "--include-videos",
        action="store_true",
        help="Decode every output frame and recompute video stats too (slow; normally unnecessary).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = resolve_root(cfg, args.root)
    available = configured_splits(cfg)
    names = args.sets or list(available)
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError(f"Unknown derived sets {unknown}; available={sorted(available)}")
    specs = [available[name] for name in names]
    sources = {spec.source for spec in specs}
    if len(sources) != 1:
        raise ValueError(f"Selected splits must share one source dataset, got {sorted(sources)}")
    source_name = next(iter(sources))
    source = load_source(root / source_name)

    planned: list[tuple[SplitSpec, dict[str, Any]]] = []
    for spec in specs:
        destination = root / spec.name
        if destination.exists() and not args.dry_run:
            raise FileExistsError(f"Output already exists: {destination}")
        plan = plan_split(source, spec)
        planned.append((spec, plan))
        print(
            f"{spec.name}: tasks={spec.task_ids[0]}..{spec.task_ids[-1]}, "
            f"episodes={len(plan['selected_episodes'])}, frames={plan['frames']}, "
            f"video_files={sum(len(value) for value in plan['video_file_maps'].values())}"
        )

    if args.dry_run:
        print("DRY RUN OK: every selected group owns complete packed video shards.")
        return

    for spec, plan in planned:
        build_split(
            source,
            root / spec.name,
            spec,
            plan,
            include_videos=args.include_videos,
            project=project_root(cfg),
        )


if __name__ == "__main__":
    main()
