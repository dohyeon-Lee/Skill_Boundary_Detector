#!/usr/bin/env python3
"""Build a corrected LangGap dataset without adding an encoding generation.

The released ``langgap_full`` mixes two image conventions. Official tasks 0..39 are
already canonical, while author-collected ext tasks 40..55 are missing a W flip.

This utility creates a hardlink clone of the current 20 Hz final dataset, preserving the
official videos bit-for-bit. It rebuilds only ext video files directly from the original
HF staging videos, applying W flip and the 10 Hz-label -> real 20 Hz PTS correction in the
same encode. Ext images therefore have exactly the same encoding generation count as a
clean full conversion; this is not a re-encode of the current final ext videos.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR / "src"))
from langgap_dataset_config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    langgap_root,
    load_config,
)

CAMERA_PAIRS = (
    ("observation.images.image", "observation.images.image"),
    ("observation.images.image2", "observation.images.wrist_image"),
)


@dataclass(frozen=True)
class VideoBuild:
    source_camera: str
    target_camera: str
    source_relative_path: str
    target_relative_path: str
    source_chunk: int
    source_file: int
    target_chunk: int
    target_file: int


@dataclass(frozen=True)
class RepairPlan:
    videos: list[VideoBuild]
    updates: dict[int, dict[str, float | int]]
    total_episodes: int
    ext_episodes: int
    first_ext_episode: int
    ext_tasks: int
    source_fps: int
    target_fps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset", default="langgap_56_full_full")
    parser.add_argument(
        "--output-name",
        default=None,
        help="Default: <dataset>_canonical_orientation (source is never modified).",
    )
    parser.add_argument(
        "--ext-task-start",
        type=int,
        default=None,
        help="Default: convert_ext_task_start_by_set[dataset] from yaml.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--encoder-threads", type=int, default=8)
    parser.add_argument(
        "--crf",
        type=int,
        default=30,
        help="Matches LeRobot's normal libsvtav1 encode (not an extra generation).",
    )
    parser.add_argument("--preset", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_episodes(dataset_dir: Path) -> pd.DataFrame:
    files = sorted((dataset_dir / "meta/episodes").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata under {dataset_dir / 'meta/episodes'}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True).sort_values(
        "episode_index"
    )


def _episode_task(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected exactly one task per episode, got {value!r}")
        return str(value[0])
    try:
        if not isinstance(value, str) and len(value) == 1:
            return str(value[0])
    except TypeError:
        pass
    return str(value)


def _with_task_index(dataset_dir: Path, episodes: pd.DataFrame) -> pd.DataFrame:
    tasks = pd.read_parquet(dataset_dir / "meta/tasks.parquet")
    index_by_name = {str(name): int(row.task_index) for name, row in tasks.iterrows()}
    indices: list[int | None] = []
    missing = False
    for value in episodes["tasks"]:
        if value is None:
            indices.append(None)
            missing = True
            continue
        name = _episode_task(value)
        if name not in index_by_name:
            raise KeyError(f"Episode task {name!r} is absent from {dataset_dir / 'meta/tasks.parquet'}")
        indices.append(index_by_name[name])
    if missing:
        data_files = sorted((dataset_dir / "data").glob("**/*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"Cannot resolve missing episode tasks: no data under {dataset_dir}")
        frame_tasks = pd.concat(
            [pd.read_parquet(path, columns=["episode_index", "task_index"]) for path in data_files],
            ignore_index=True,
        )
        by_episode = frame_tasks.groupby("episode_index", sort=False)["task_index"].first().to_dict()
        indices = [
            int(by_episode[int(episode)]) if index is None else index
            for episode, index in zip(episodes["episode_index"], indices, strict=True)
        ]
    return episodes.assign(_task_index=indices)


def _video_columns(camera: str) -> tuple[str, str, str, str]:
    prefix = f"videos/{camera}"
    return (
        f"{prefix}/chunk_index",
        f"{prefix}/file_index",
        f"{prefix}/from_timestamp",
        f"{prefix}/to_timestamp",
    )


def build_plan(final_dir: Path, staging_dir: Path, ext_task_start: int) -> RepairPlan:
    final_info = json.loads((final_dir / "meta/info.json").read_text())
    staging_info = json.loads((staging_dir / "meta/info.json").read_text())
    target_fps = int(final_info["fps"])
    source_fps = int(staging_info["fps"])
    if target_fps != 20 or source_fps != 10:
        raise ValueError(f"Expected staging/final fps 10->20, got {source_fps}->{target_fps}.")
    time_scale = source_fps / target_fps

    final = _with_task_index(final_dir, _load_episodes(final_dir)).set_index("episode_index")
    staging = _with_task_index(staging_dir, _load_episodes(staging_dir)).set_index("episode_index")
    if set(final.index) != set(staging.index):
        raise ValueError("Final and staging episode_index sets differ; refusing a hybrid repair.")
    if not (final["length"].sort_index().to_numpy() == staging["length"].sort_index().to_numpy()).all():
        raise ValueError("Final and staging episode lengths differ.")

    final_ext = final[final["_task_index"] >= ext_task_start]
    staging_ext = staging[staging["_task_index"] >= ext_task_start]
    if set(final_ext.index) != set(staging_ext.index):
        raise ValueError("Final and staging disagree on which episodes are extension data.")
    if final_ext.empty:
        raise ValueError(f"No ext episodes found at task_index >= {ext_task_start}.")
    first_ext = int(final_ext.index.min())
    if set(final.loc[first_ext:].index) != set(final_ext.index):
        raise ValueError("Ext episodes are not a contiguous suffix of the final dataset.")

    videos: list[VideoBuild] = []
    updates: dict[int, dict[str, float | int]] = {int(ep): {} for ep in final_ext.index}
    for source_camera, target_camera in CAMERA_PAIRS:
        src_chunk, src_file, src_from, src_to = _video_columns(source_camera)
        dst_chunk, dst_file, dst_from, dst_to = _video_columns(target_camera)

        # The HF merge places ext episodes on fresh packed-video files. This clean boundary
        # is what lets us preserve every official frame without re-encoding it.
        source_groups = list(staging_ext.groupby([src_chunk, src_file], sort=True))
        for (chunk, file_index), _ in source_groups:
            all_rows = staging[
                (staging[src_chunk] == chunk) & (staging[src_file] == file_index)
            ]
            if bool((all_rows["_task_index"] < ext_task_start).any()):
                raise ValueError(
                    f"Staging {source_camera} chunk={chunk} file={file_index} mixes official and ext "
                    "episodes. Use the full task-aware converter instead."
                )

        official_final = final[final["_task_index"] < ext_task_start]
        chunks = sorted({int(value) for value in official_final[dst_chunk]})
        if chunks != [0]:
            raise ValueError(f"Expected official final videos in chunk 0, got chunks={chunks}.")
        target_chunk = 0
        next_target_file = int(official_final[dst_file].max()) + 1

        source_to_target: dict[tuple[int, int], tuple[int, int]] = {}
        for offset, ((chunk, file_index), _) in enumerate(source_groups):
            target_file = next_target_file + offset
            source_to_target[(int(chunk), int(file_index))] = (target_chunk, target_file)
            videos.append(
                VideoBuild(
                    source_camera=source_camera,
                    target_camera=target_camera,
                    source_relative_path=(
                        f"videos/{source_camera}/chunk-{int(chunk):03d}/file-{int(file_index):03d}.mp4"
                    ),
                    target_relative_path=(
                        f"videos/{target_camera}/chunk-{target_chunk:03d}/file-{target_file:03d}.mp4"
                    ),
                    source_chunk=int(chunk),
                    source_file=int(file_index),
                    target_chunk=target_chunk,
                    target_file=target_file,
                )
            )

        for episode, row in staging_ext.iterrows():
            target = source_to_target[(int(row[src_chunk]), int(row[src_file]))]
            updates[int(episode)].update(
                {
                    dst_chunk: target[0],
                    dst_file: target[1],
                    dst_from: float(row[src_from]) * time_scale,
                    dst_to: float(row[src_to]) * time_scale,
                }
            )

    return RepairPlan(
        videos=videos,
        updates=updates,
        total_episodes=int(len(final)),
        ext_episodes=int(len(final_ext)),
        first_ext_episode=first_ext,
        ext_tasks=int(final_ext["_task_index"].nunique()),
        source_fps=source_fps,
        target_fps=target_fps,
    )


def _link_or_copy(src: str, dst: str) -> str:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def clone_dataset(source: Path, output: Path, *, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output}. Pass --overwrite to replace it.")
        shutil.rmtree(output)
    shutil.copytree(source, output, copy_function=_link_or_copy)


def _probe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)["streams"][0]


def _build_video(
    build: VideoBuild,
    staging: Path,
    output: Path,
    *,
    source_fps: int,
    target_fps: int,
    crf: int,
    preset: int,
    encoder_threads: int,
) -> str:
    src = staging / build.source_relative_path
    dst = output / build.target_relative_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.stem}.rebuild-{os.getpid()}.mp4")
    before = _probe(src)
    pts_scale = source_fps / target_fps
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-vf",
        # Rebuild timestamps from frame ordinal. Scaling the source PTS directly leaves
        # the stream's nominal 10 Hz rate in some FFmpeg versions; forcing CFR with -r
        # then duplicates the terminal frame. This expression emits exactly N frames at
        # 20 Hz, which is the same frame/timestamp contract as the canonical writer.
        f"hflip,setpts=N/({target_fps}*TB),fps={target_fps}",
        "-fps_mode",
        "passthrough",
        "-an",
        "-c:v",
        "libsvtav1",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "2",
        "-crf",
        str(crf),
        "-preset",
        str(preset),
        "-svtav1-params",
        f"lp={encoder_threads}",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    try:
        subprocess.run(command, check=True)
        after = _probe(tmp)
        for field in ("width", "height", "nb_frames"):
            if str(after.get(field)) != str(before.get(field)):
                raise RuntimeError(
                    f"Video contract changed for {build.source_relative_path}: "
                    f"{field} {before.get(field)!r} -> {after.get(field)!r}"
                )
        if str(after.get("avg_frame_rate")) != f"{target_fps}/1":
            raise RuntimeError(
                f"Expected {target_fps} fps for {build.target_relative_path}, "
                f"got {after.get('avg_frame_rate')!r}."
            )
        expected_duration = float(before["duration"]) * pts_scale
        if abs(float(after["duration"]) - expected_duration) > 1.0 / target_fps:
            raise RuntimeError(
                f"Duration mismatch for {build.target_relative_path}: "
                f"expected {expected_duration:.6f}, got {after.get('duration')}."
            )
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
    return build.target_relative_path


def rewrite_episode_metadata(output: Path, updates: dict[int, dict[str, float | int]]) -> None:
    for path in sorted((output / "meta/episodes").glob("**/*.parquet")):
        table = pq.read_table(path)
        episodes = table.column("episode_index").to_pylist()
        touched = False
        relevant_columns = sorted({column for values in updates.values() for column in values})
        for column in relevant_columns:
            position = table.schema.get_field_index(column)
            values = table.column(position).to_pylist()
            changed = False
            for row_index, episode in enumerate(episodes):
                replacement = updates.get(int(episode), {}).get(column)
                if replacement is not None:
                    values[row_index] = replacement
                    changed = touched = True
            if changed:
                field = table.schema.field(position)
                table = table.set_column(position, field, pa.array(values, type=field.type))
        if touched:
            tmp = path.with_name(f".{path.name}.repair-{os.getpid()}")
            try:
                pq.write_table(table, tmp)
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)


def validate_output(output: Path, plan: RepairPlan) -> None:
    episodes = _load_episodes(output).set_index("episode_index")
    for episode, replacements in plan.updates.items():
        row = episodes.loc[episode]
        for column, expected in replacements.items():
            actual = row[column]
            if isinstance(expected, float):
                if abs(float(actual) - expected) > 1e-7:
                    raise RuntimeError(f"Episode {episode} {column}: {actual} != {expected}")
            elif int(actual) != expected:
                raise RuntimeError(f"Episode {episode} {column}: {actual} != {expected}")
    for build in plan.videos:
        if not (output / build.target_relative_path).is_file():
            raise FileNotFoundError(output / build.target_relative_path)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root = langgap_root(cfg)
    final = root / args.dataset
    staging = root / "_hf" / args.dataset
    output = root / (args.output_name or f"{args.dataset}_canonical_orientation")
    if final.resolve() == output.resolve():
        raise ValueError("In-place repair is intentionally unsupported; choose a separate output name.")
    for directory, label in ((final, "final"), (staging, "HF staging")):
        if not (directory / "meta/info.json").is_file():
            raise FileNotFoundError(f"{label} dataset not found: {directory}")

    starts = cfg.get("convert_ext_task_start_by_set", {}) or {}
    configured_start = starts.get(args.dataset) if isinstance(starts, dict) else None
    if args.ext_task_start is None and configured_start is None:
        raise ValueError(
            f"No ext task boundary configured for {args.dataset!r}; pass --ext-task-start."
        )
    ext_task_start = int(
        args.ext_task_start if args.ext_task_start is not None else configured_start
    )
    plan = build_plan(final, staging, ext_task_start)

    print("LangGap loss-aware ext orientation rebuild")
    print(f"  final base   : {final}")
    print(f"  HF staging  : {staging}")
    print(f"  output      : {output}")
    print(f"  ext tasks   : task_index >= {ext_task_start}")
    print(
        f"  episodes    : {plan.ext_episodes}/{plan.total_episodes} "
        f"(first={plan.first_ext_episode}, tasks={plan.ext_tasks})"
    )
    print(f"  video files : {len(plan.videos)} rebuilt from staging; official videos untouched")
    for build in plan.videos:
        print(f"    {build.source_relative_path} -> {build.target_relative_path}")
    if args.dry_run:
        return

    clone_dataset(final, output, overwrite=args.overwrite)
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {
            pool.submit(
                _build_video,
                build,
                staging,
                output,
                source_fps=plan.source_fps,
                target_fps=plan.target_fps,
                crf=int(args.crf),
                preset=int(args.preset),
                encoder_threads=max(1, int(args.encoder_threads)),
            ): build
            for build in plan.videos
        }
        for future in as_completed(futures):
            rebuilt = future.result()
            completed += 1
            print(f"  rebuilt {completed}/{len(plan.videos)}: {rebuilt}", flush=True)

    rewrite_episode_metadata(output, plan.updates)
    validate_output(output, plan)
    marker = {
        "final_base": str(final),
        "hf_staging": str(staging),
        "ext_task_start": ext_task_start,
        "extra_flip": "w",
        "quality_path": "official bitstream reused; ext encoded once from HF staging",
        "source_fps": plan.source_fps,
        "target_fps": plan.target_fps,
        "crf": int(args.crf),
        "preset": int(args.preset),
        "episodes": {
            "total": plan.total_episodes,
            "ext": plan.ext_episodes,
            "first_ext": plan.first_ext_episode,
            "ext_tasks": plan.ext_tasks,
        },
        "videos": [asdict(build) for build in plan.videos],
    }
    (output / "meta/orientation_repair.json").write_text(json.dumps(marker, indent=2))
    print(f"DONE: {output}")
    print("The source datasets were not modified; validate this output before promoting it.")


if __name__ == "__main__":
    main()
