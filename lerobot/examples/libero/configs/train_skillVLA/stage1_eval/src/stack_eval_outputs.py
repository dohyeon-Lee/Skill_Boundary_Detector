#!/usr/bin/env python3
"""Vertically stack matching Stage-1 side-by-side evaluation videos."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import yaml

from compare_videos import even, label_bar, load_font


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "stack_eval_outputs_config.yaml"
_EPISODE_PATTERN = re.compile(r"eval_episode_(\d+)\.mp4$")


@dataclass(frozen=True)
class Source:
    name: str
    label: str
    side_by_side_dir: Path


@dataclass(frozen=True)
class Settings:
    sources: tuple[Source, ...]
    output_dir: Path
    task_prefix: str
    task_ids: tuple[int, ...]
    episode_ids: tuple[int, ...] | None
    missing_policy: str
    overwrite: bool
    show_source_labels: bool
    label_height: int
    target_width: int | None


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _safe_output_name(value: object) -> str:
    name = str(value or "").strip()
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError(f"output_name must be one folder name, got {name!r}.")
    return name


def load_settings(config_path: Path) -> Settings:
    config_path = config_path.expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise TypeError("The stack config must be a YAML mapping.")

    outputs_root = _resolve_path(config.get("outputs_root", "outputs"), base=config_path.parent)
    raw_sources = config.get("sources")
    if not isinstance(raw_sources, list) or len(raw_sources) < 2:
        raise ValueError("sources must contain at least two eval output folders.")

    sources: list[Source] = []
    for index, raw_source in enumerate(raw_sources):
        if isinstance(raw_source, str):
            name = raw_source.strip()
            label = name
        elif isinstance(raw_source, dict):
            name = str(raw_source.get("folder", "")).strip()
            label = str(raw_source.get("label") or name).strip()
        else:
            raise TypeError(f"sources[{index}] must be a folder string or mapping.")
        if not name or not label:
            raise ValueError(f"sources[{index}] needs a non-empty folder and label.")
        source_root = _resolve_path(name, base=outputs_root)
        side_by_side_dir = source_root / "side_by_side"
        if not side_by_side_dir.is_dir():
            raise FileNotFoundError(f"Missing side_by_side directory: {side_by_side_dir}")
        sources.append(Source(name=name, label=label, side_by_side_dir=side_by_side_dir))

    output_name = _safe_output_name(config.get("output_name"))
    output_dir = outputs_root / output_name / "side_by_side"
    source_dirs = {source.side_by_side_dir.resolve() for source in sources}
    if output_dir.resolve() in source_dirs:
        raise ValueError("The output folder cannot also be one of the source folders.")

    task_prefix = str(config.get("task_prefix", "libero_90")).strip()
    if not task_prefix or "/" in task_prefix:
        raise ValueError(f"Invalid task_prefix: {task_prefix!r}")
    raw_task_ids = config.get("task_ids", list(range(10)))
    if not isinstance(raw_task_ids, list) or not raw_task_ids:
        raise ValueError("task_ids must be a non-empty list.")
    task_ids = tuple(int(task_id) for task_id in raw_task_ids)

    raw_episodes = config.get("episode_ids", "auto")
    if raw_episodes is None or raw_episodes == "auto":
        episode_ids = None
    elif isinstance(raw_episodes, list) and raw_episodes:
        episode_ids = tuple(int(episode_id) for episode_id in raw_episodes)
    else:
        raise ValueError("episode_ids must be 'auto' or a non-empty integer list.")

    missing_policy = str(config.get("missing_policy", "error")).strip().lower()
    if missing_policy not in {"error", "skip"}:
        raise ValueError("missing_policy must be 'error' or 'skip'.")

    video = config.get("video", {}) or {}
    if not isinstance(video, dict):
        raise TypeError("video must be a YAML mapping.")
    show_source_labels = bool(video.get("show_source_labels", True))
    label_height = even(max(2, int(video.get("label_height", 32))))
    raw_target_width = video.get("target_width")
    target_width = None if raw_target_width in {None, "auto"} else even(int(raw_target_width))
    if target_width is not None and target_width < 2:
        raise ValueError("video.target_width must be at least 2 pixels.")

    return Settings(
        sources=tuple(sources),
        output_dir=output_dir,
        task_prefix=task_prefix,
        task_ids=task_ids,
        episode_ids=episode_ids,
        missing_policy=missing_policy,
        overwrite=bool(config.get("overwrite", False)),
        show_source_labels=show_source_labels,
        label_height=label_height,
        target_width=target_width,
    )


def _episodes_in(task_dir: Path) -> set[int]:
    episodes: set[int] = set()
    if not task_dir.is_dir():
        return episodes
    for video in task_dir.glob("eval_episode_*.mp4"):
        match = _EPISODE_PATTERN.fullmatch(video.name)
        if match:
            episodes.add(int(match.group(1)))
    return episodes


def build_jobs(settings: Settings) -> tuple[list[tuple[list[Path], Path]], list[str]]:
    jobs: list[tuple[list[Path], Path]] = []
    missing_messages: list[str] = []
    for task_id in settings.task_ids:
        task_name = f"{settings.task_prefix}_{task_id}"
        task_dirs = [source.side_by_side_dir / task_name for source in settings.sources]
        if settings.episode_ids is None:
            episode_ids = sorted(set().union(*(_episodes_in(task_dir) for task_dir in task_dirs)))
        else:
            episode_ids = list(settings.episode_ids)
        if not episode_ids:
            missing_messages.append(f"{task_name}: no episode videos found in any source")
            continue

        for episode_id in episode_ids:
            filename = f"eval_episode_{episode_id}.mp4"
            inputs = [task_dir / filename for task_dir in task_dirs]
            missing_sources = [
                source.name
                for source, input_path in zip(settings.sources, inputs, strict=True)
                if not input_path.is_file()
            ]
            if missing_sources:
                missing_messages.append(f"{task_name}/{filename}: missing from {', '.join(missing_sources)}")
                continue
            jobs.append((inputs, settings.output_dir / task_name / filename))
    return jobs, missing_messages


def _fps(reader: imageio.Reader, path: Path) -> float:
    fps = float(reader.get_meta_data().get("fps", 0) or 0)
    if fps <= 0:
        raise ValueError(f"Could not determine video FPS: {path}")
    return fps


def _resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    frame = np.asarray(frame)[:, :, :3]
    frame_height, frame_width = frame.shape[:2]
    height = even(max(2, round(frame_height * width / frame_width)))
    if (frame_width, frame_height) == (width, height):
        return frame
    interpolation = cv2.INTER_AREA if width < frame_width else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, height), interpolation=interpolation)


def stack_video(
    inputs: list[Path],
    destination: Path,
    *,
    labels: tuple[str, ...],
    show_source_labels: bool,
    label_height: int,
    target_width: int | None,
) -> int:
    readers = [imageio.get_reader(str(path)) for path in inputs]
    writer = None
    temp_path = destination.with_name(f".{destination.stem}.writing{destination.suffix}")
    try:
        fps_values = [_fps(reader, path) for reader, path in zip(readers, inputs, strict=True)]
        if any(abs(fps - fps_values[0]) > 1e-3 for fps in fps_values[1:]):
            fps_by_path = dict(zip(map(str, inputs), fps_values, strict=True))
            raise ValueError(f"All source videos must have the same FPS: {fps_by_path}")

        iterators = [iter(reader) for reader in readers]
        current_frames = []
        for iterator, path in zip(iterators, inputs, strict=True):
            try:
                current_frames.append(np.asarray(next(iterator))[:, :, :3])
            except StopIteration as error:
                raise ValueError(f"Source video has no frames: {path}") from error

        width = target_width or even(max(frame.shape[1] for frame in current_frames))
        bars = None
        if show_source_labels:
            font = load_font(max(10, round(label_height * 0.58)))
            bars = [label_bar(width, label_height, label, font) for label in labels]

        destination.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            str(temp_path),
            fps=fps_values[0],
            codec="libx264",
            quality=8,
            macro_block_size=None,
            pixelformat="yuv420p",
            ffmpeg_params=["-movflags", "+faststart"],
        )
        active = [True] * len(iterators)
        frame_count = 0
        while True:
            rows = []
            for index, frame in enumerate(current_frames):
                row = _resize_to_width(frame, width)
                if bars is not None:
                    row = np.vstack([bars[index], row])
                rows.append(row)
            writer.append_data(np.vstack(rows))
            frame_count += 1

            advanced = False
            for index, iterator in enumerate(iterators):
                if not active[index]:
                    continue
                try:
                    current_frames[index] = np.asarray(next(iterator))[:, :, :3]
                    advanced = True
                except StopIteration:
                    active[index] = False
            if not advanced:
                break

        writer.close()
        writer = None
        temp_path.replace(destination)
        return frame_count
    finally:
        if writer is not None:
            writer.close()
        for reader in readers:
            reader.close()
        if temp_path.exists():
            temp_path.unlink()


def run(settings: Settings, *, dry_run: bool) -> None:
    jobs, missing = build_jobs(settings)
    if missing:
        report = "\n".join(f"  - {message}" for message in missing)
        if settings.missing_policy == "error":
            raise FileNotFoundError(
                f"Missing matching side-by-side videos:\n{report}\n"
                "Set missing_policy: skip to merge only complete matches."
            )
        print(f"Skipping {len(missing)} incomplete task/episode pairs:\n{report}")
    if not jobs:
        raise RuntimeError("No complete source video sets were found.")

    print(f"Sources : {len(settings.sources)}")
    for source in settings.sources:
        print(f"  {source.label}: {source.side_by_side_dir}")
    print(f"Output  : {settings.output_dir}")
    print(f"Videos  : {len(jobs)}")
    if dry_run:
        print("Dry run complete; no videos were written.")
        return

    written = 0
    skipped = 0
    for index, (inputs, destination) in enumerate(jobs, start=1):
        if destination.exists() and not settings.overwrite:
            print(f"[{index}/{len(jobs)}] skip existing {destination.relative_to(settings.output_dir)}")
            skipped += 1
            continue
        print(f"[{index}/{len(jobs)}] write {destination.relative_to(settings.output_dir)}")
        stack_video(
            inputs,
            destination,
            labels=tuple(source.label for source in settings.sources),
            show_source_labels=settings.show_source_labels,
            label_height=settings.label_height,
            target_width=settings.target_width,
        )
        written += 1

    settings.output_dir.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "sources": [
            {"folder": source.name, "label": source.label, "path": str(source.side_by_side_dir)}
            for source in settings.sources
        ],
        "output": str(settings.output_dir),
        "written": written,
        "skipped_existing": skipped,
        "missing_skipped": missing,
    }
    manifest_path = settings.output_dir.parent / "stack_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Done: wrote {written}, skipped existing {skipped}. Manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and list work without writing videos.",
    )
    args = parser.parse_args()
    run(load_settings(args.config), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
