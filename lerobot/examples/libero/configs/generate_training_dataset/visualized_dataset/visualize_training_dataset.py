#!/usr/bin/env python3
# Inputs:
#   settings     : visualized_dataset_config.yaml
# Outputs:
#   browser-ready VP9/H.264 clips, posters, and an HTML gallery under previews/
"""Build an HTML gallery for visually inspecting v3 training-dataset episodes.

LeRobot v3 stores multiple episodes in packed AV1 video files. This script selects
episodes from one task, extracts their timestamp ranges with ffmpeg, transcodes the
clips to VP9 WebM and H.264 MP4, and writes an HTML gallery with the selected camera.

Edit ``visualized_dataset_config.yaml`` and run ``python visualize_training_dataset.py``.
"""

from __future__ import annotations

import html
import json
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import pyarrow.parquet as pq

VISUALIZED_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VISUALIZED_DIR / "src"))

from visualized_dataset_config import (  # noqa: E402
    dataset_settings,
    load_config,
    reject_cli_arguments,
    visualization_settings,
)


PREVIEW_ROOT = VISUALIZED_DIR / "previews"
DEFAULT_CAMERA = "observation.images.image"


@dataclass(frozen=True)
class SourceClip:
    camera_key: str
    path: Path
    start: float
    end: float


@dataclass(frozen=True)
class PreviewClip:
    camera_key: str
    mp4_path: Path
    webm_path: Path
    poster_path: Path
    duration: float


@dataclass(frozen=True)
class EpisodeSample:
    task_index: int
    episode_index: int
    length: int
    clips: tuple[SourceClip, ...]


@dataclass(frozen=True)
class RenderedSample:
    task_index: int
    episode_index: int
    length: int
    clips: tuple[PreviewClip, ...]


def load_info(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / "meta" / "info.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {path}")
    with path.open(encoding="utf-8") as stream:
        info = json.load(stream)
    version = str(info.get("codebase_version", ""))
    if not version.startswith("v3"):
        raise ValueError(
            f"This visualizer expects a converted LeRobot v3 training dataset, but "
            f"{dataset_dir} reports codebase_version={version!r}."
        )
    return info


def load_tasks(dataset_dir: Path) -> dict[int, str]:
    path = dataset_dir / "meta" / "tasks.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing v3 task metadata: {path}")

    rows = pd.read_parquet(path).reset_index().to_dict("records")
    tasks: dict[int, str] = {}
    for row in rows:
        if "task_index" not in row or "task" not in row:
            raise ValueError(f"Task metadata must contain task_index and task: {row}")
        task_index = int(row["task_index"])
        if task_index in tasks:
            raise ValueError(f"Duplicate task_index {task_index} in {path}")
        tasks[task_index] = str(row["task"])
    if not tasks:
        raise ValueError(f"No tasks found in {path}")
    return tasks


def video_keys(info: dict[str, Any]) -> list[str]:
    keys = [
        str(key)
        for key, feature in info.get("features", {}).items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
    if not keys:
        raise ValueError("Dataset info.json contains no video features")
    return sorted(keys)


def _episode_columns(camera_keys: list[str]) -> list[str]:
    columns = ["episode_index", "tasks", "length", "task_index", "stats/task_index/min"]
    for key in camera_keys:
        prefix = f"videos/{key}"
        columns.extend(
            [
                f"{prefix}/chunk_index",
                f"{prefix}/file_index",
                f"{prefix}/from_timestamp",
                f"{prefix}/to_timestamp",
            ]
        )
    return columns


def load_episodes(dataset_dir: Path, camera_keys: list[str]) -> pd.DataFrame:
    paths = sorted((dataset_dir / "meta" / "episodes").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No v3 episode metadata found under {dataset_dir / 'meta' / 'episodes'}")

    wanted = _episode_columns(camera_keys)
    frames = []
    for path in paths:
        available = set(pq.read_schema(path).names)
        columns = [column for column in wanted if column in available]
        frames.append(pd.read_parquet(path, columns=columns))
    episodes = pd.concat(frames, ignore_index=True)

    required = {"episode_index", "tasks", "length"}
    missing = sorted(required - set(episodes.columns))
    if missing:
        raise ValueError(f"Episode metadata is missing required columns: {missing}")
    return episodes.sort_values("episode_index").reset_index(drop=True)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if pd.isna(value):
        return []
    return [value]


def _scalar_int(value: Any) -> int | None:
    values = _as_list(value)
    if not values:
        return None
    nested = _as_list(values[0])
    candidate = nested[0] if nested else values[0]
    try:
        return int(candidate)
    except (TypeError, ValueError):
        return None


def annotate_episode_tasks(episodes: pd.DataFrame, tasks: dict[int, str]) -> pd.DataFrame:
    language_to_index = {language: task_index for task_index, language in tasks.items()}

    def resolve_row(row: pd.Series) -> int | None:
        for column in ("task_index", "stats/task_index/min"):
            if column in row.index:
                task_index = _scalar_int(row[column])
                if task_index is not None:
                    return task_index
        indexes = {
            language_to_index[str(language)]
            for language in _as_list(row.get("tasks"))
            if str(language) in language_to_index
        }
        if len(indexes) == 1:
            return indexes.pop()
        return None

    result = episodes.copy()
    result["_resolved_task_index"] = [resolve_row(row) for _, row in result.iterrows()]
    unresolved = int(result["_resolved_task_index"].isna().sum())
    if unresolved:
        print(f"WARNING: could not resolve a task index for {unresolved} episode(s)", file=sys.stderr)
    return result


def task_episode_counts(episodes: pd.DataFrame, tasks: dict[int, str]) -> dict[int, int]:
    counts = episodes["_resolved_task_index"].value_counts().to_dict()
    return {task_index: int(counts.get(task_index, 0)) for task_index in tasks}


def resolve_task(selector: str, tasks: dict[int, str]) -> int:
    text = selector.strip()
    match = re.fullmatch(r"(?:task[\s_-]*)?(\d+)", text, flags=re.IGNORECASE)
    if match:
        task_index = int(match.group(1))
        if task_index not in tasks:
            indexes = ", ".join(str(index) for index in sorted(tasks))
            raise ValueError(f"Unknown task index {task_index}. Available indexes: {indexes}")
        return task_index

    folded = text.casefold()
    exact = [index for index, language in tasks.items() if language.casefold() == folded]
    if len(exact) == 1:
        return exact[0]

    partial = [index for index, language in tasks.items() if folded in language.casefold()]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        matches = ", ".join(f"task{index:02d}" for index in partial)
        raise ValueError(f"Task language selector is ambiguous; matches: {matches}")
    raise ValueError(f"No task language contains: {selector!r}")


def source_clip(
    dataset_dir: Path,
    info: dict[str, Any],
    row: pd.Series,
    camera_key: str,
) -> SourceClip:
    episode_index = int(row["episode_index"])
    prefix = f"videos/{camera_key}"
    required = [
        f"{prefix}/chunk_index",
        f"{prefix}/file_index",
        f"{prefix}/from_timestamp",
        f"{prefix}/to_timestamp",
    ]
    missing = [column for column in required if column not in row.index or pd.isna(row[column])]
    if missing:
        raise ValueError(
            f"Episode {episode_index} lacks packed-video metadata for {camera_key}: {missing}"
        )

    chunk_index = int(row[f"{prefix}/chunk_index"])
    file_index = int(row[f"{prefix}/file_index"])
    start = float(row[f"{prefix}/from_timestamp"])
    end = float(row[f"{prefix}/to_timestamp"])
    if end <= start:
        raise ValueError(
            f"Invalid timestamps for episode {episode_index}, {camera_key}: start={start}, end={end}"
        )

    template = info.get(
        "video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )
    relative_path = str(template).format(
        video_key=camera_key,
        chunk_index=chunk_index,
        file_index=file_index,
    )
    path = (dataset_dir / relative_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing video for episode {episode_index}: {path}")
    return SourceClip(camera_key=camera_key, path=path, start=start, end=end)


def select_samples(
    dataset_dir: Path,
    episodes: pd.DataFrame,
    info: dict[str, Any],
    task_indexes: list[int],
    sample_count: int,
    sampling: str,
    seed: int,
    camera_keys: list[str],
) -> list[EpisodeSample]:
    samples = []
    for task_index in task_indexes:
        matching = episodes[episodes["_resolved_task_index"] == task_index]
        rows = [row for _, row in matching.iterrows()]
        available = len(rows)
        count = min(sample_count, available)
        chosen = (
            random.Random(seed + task_index).sample(rows, count)
            if sampling == "random"
            else rows[:count]
        )
        samples.extend(
            EpisodeSample(
                task_index=task_index,
                episode_index=int(row["episode_index"]),
                length=int(row["length"]),
                clips=tuple(source_clip(dataset_dir, info, row, key) for key in camera_keys),
            )
            for row in chosen
        )
    return samples


def _camera_slug(camera_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", camera_key).strip("_")


def _clip_cache_key(clip: SourceClip, crf: int, preset: str) -> dict[str, Any]:
    source_stat = clip.path.stat()
    return {
        "source": str(clip.path),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "start": clip.start,
        "end": clip.end,
        "codec": "libx264",
        "crf": crf,
        "preset": preset,
        "pixel_format": "yuv420p",
    }


def _cached_clip_matches(output_path: Path, cache_key: dict[str, Any]) -> bool:
    manifest_path = Path(f"{output_path}.json")
    if not output_path.is_file() or output_path.stat().st_size == 0 or not manifest_path.is_file():
        return False
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8")) == cache_key
    except (json.JSONDecodeError, OSError):
        return False


def transcode_clip(
    clip: SourceClip,
    output_path: Path,
    ffmpeg: str,
    crf: int,
    preset: str,
    force: bool,
) -> bool:
    cache_key = _clip_cache_key(clip, crf=crf, preset=preset)
    if not force and _cached_clip_matches(output_path, cache_key):
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.tmp-{os.getpid()}.mp4")
    temporary.unlink(missing_ok=True)
    duration = clip.end - clip.start
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{clip.start:.9f}",
        "-i",
        str(clip.path),
        "-t",
        f"{duration:.9f}",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-avoid_negative_ts",
        "make_zero",
        str(temporary),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        temporary.unlink(missing_ok=True)
        details = result.stderr.strip() or "ffmpeg produced no error output"
        raise RuntimeError(f"ffmpeg failed for {clip.path} [{clip.start}, {clip.end}]:\n{details}")
    temporary.replace(output_path)
    Path(f"{output_path}.json").write_text(
        json.dumps(cache_key, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return True


def transcode_webm(
    clip: SourceClip,
    output_path: Path,
    ffmpeg: str,
    force: bool,
) -> bool:
    source_stat = clip.path.stat()
    cache_key = {
        "source": str(clip.path),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "start": clip.start,
        "end": clip.end,
        "codec": "libvpx-vp9",
        "crf": 32,
        "cpu_used": 5,
        "pixel_format": "yuv420p",
    }
    if not force and _cached_clip_matches(output_path, cache_key):
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.tmp-{os.getpid()}.webm")
    temporary.unlink(missing_ok=True)
    duration = clip.end - clip.start
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{clip.start:.9f}",
        "-i",
        str(clip.path),
        "-t",
        f"{duration:.9f}",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libvpx-vp9",
        "-crf",
        "32",
        "-b:v",
        "0",
        "-deadline",
        "good",
        "-cpu-used",
        "5",
        "-pix_fmt",
        "yuv420p",
        str(temporary),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        temporary.unlink(missing_ok=True)
        details = result.stderr.strip() or "ffmpeg produced no error output"
        raise RuntimeError(
            f"ffmpeg VP9 encoding failed for {clip.path} [{clip.start}, {clip.end}]:\n{details}"
        )
    temporary.replace(output_path)
    Path(f"{output_path}.json").write_text(
        json.dumps(cache_key, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return True


def generate_poster(
    video_path: Path,
    poster_path: Path,
    duration: float,
    ffmpeg: str,
    force: bool,
) -> bool:
    if (
        not force
        and poster_path.is_file()
        and poster_path.stat().st_size > 0
        and poster_path.stat().st_mtime_ns >= video_path.stat().st_mtime_ns
    ):
        return False

    temporary = poster_path.with_name(f".{poster_path.stem}.tmp-{os.getpid()}.jpg")
    temporary.unlink(missing_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{duration / 2:.9f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "3",
        str(temporary),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        temporary.unlink(missing_ok=True)
        details = result.stderr.strip() or "ffmpeg produced no error output"
        raise RuntimeError(f"ffmpeg poster generation failed for {video_path}:\n{details}")
    temporary.replace(poster_path)
    return True


def render_samples(
    samples: list[EpisodeSample],
    media_dir: Path,
    ffmpeg: str,
    crf: int,
    preset: str,
    force: bool,
) -> list[RenderedSample]:
    total = sum(len(sample.clips) for sample in samples)
    completed = 0
    rendered = []
    for sample in samples:
        preview_clips = []
        for clip in sample.clips:
            completed += 1
            stem = f"episode_{sample.episode_index:06d}__{_camera_slug(clip.camera_key)}"
            mp4_path = media_dir / f"{stem}.mp4"
            webm_path = media_dir / f"{stem}.webm"
            poster_path = media_dir / f"{stem}.jpg"
            mp4_changed = transcode_clip(
                clip,
                output_path=mp4_path,
                ffmpeg=ffmpeg,
                crf=crf,
                preset=preset,
                force=force,
            )
            webm_changed = transcode_webm(
                clip,
                output_path=webm_path,
                ffmpeg=ffmpeg,
                force=force,
            )
            poster_changed = generate_poster(
                video_path=mp4_path,
                poster_path=poster_path,
                duration=clip.end - clip.start,
                ffmpeg=ffmpeg,
                force=force,
            )
            status = "encoded" if mp4_changed or webm_changed or poster_changed else "cached"
            print(f"[{completed:>3}/{total}] {status}: episode {sample.episode_index}, {clip.camera_key}")
            preview_clips.append(
                PreviewClip(
                    camera_key=clip.camera_key,
                    mp4_path=mp4_path,
                    webm_path=webm_path,
                    poster_path=poster_path,
                    duration=clip.end - clip.start,
                )
            )
        rendered.append(
            RenderedSample(
                task_index=sample.task_index,
                episode_index=sample.episode_index,
                length=sample.length,
                clips=tuple(preview_clips),
            )
        )
    return rendered


def _media_url(path: Path, output_dir: Path) -> str:
    relative = os.path.relpath(path, output_dir.resolve())
    return quote(Path(relative).as_posix(), safe="/:@-._~")


def _camera_label(camera_key: str) -> str:
    return camera_key.removeprefix("observation.images.").replace("_", " ")


def build_html(
    dataset_name: str,
    dataset_dir: Path,
    tasks: dict[int, str],
    selected_task_indexes: list[int],
    sampling: str,
    seed: int,
    samples: list[RenderedSample],
    output_path: Path,
) -> str:
    cards = []
    sample_numbers: dict[int, int] = {}
    for sample in samples:
        sample_numbers[sample.task_index] = sample_numbers.get(sample.task_index, 0) + 1
        sample_number = sample_numbers[sample.task_index]
        camera_blocks = []
        for clip in sample.clips:
            mp4_url = _media_url(clip.mp4_path, output_path.parent)
            webm_url = _media_url(clip.webm_path, output_path.parent)
            poster_url = _media_url(clip.poster_path, output_path.parent)
            camera_blocks.append(
                f"""
                <figure>
                  <figcaption>{html.escape(_camera_label(clip.camera_key))}</figcaption>
                  <video class="clip" controls muted playsinline preload="none"
                         poster="{html.escape(poster_url, quote=True)}">
                    <source src="{html.escape(webm_url, quote=True)}" type="video/webm">
                    <source src="{html.escape(mp4_url, quote=True)}" type="video/mp4">
                    This browser cannot play VP9 WebM or H.264 MP4.
                  </video>
                </figure>"""
            )
        duration = sample.clips[0].duration
        cards.append(
            f"""
            <article class="card" data-task="{sample.task_index}">
              <header class="card-header">
                <div><span class="sample-no">TASK {sample.task_index:02d} · SAMPLE {sample_number:02d}</span>
                  <h2>Episode {sample.episode_index}</h2>
                  <p class="card-task">{html.escape(tasks[sample.task_index])}</p></div>
                <div class="chips"><span>{sample.length:,} frames</span><span>{duration:.2f} sec</span><span>VP9 / H.264</span></div>
              </header>
              <div class="camera-grid">{''.join(camera_blocks)}</div>
              <div class="card-actions">
                <button type="button" onclick="playCard(this)">Play cameras</button>
                <button type="button" class="secondary" onclick="resetCard(this)">Reset</button>
              </div>
            </article>"""
        )

    escaped_dataset = html.escape(dataset_name)
    escaped_path = html.escape(str(dataset_dir))
    all_tasks = len(selected_task_indexes) > 1
    selection_label = "all tasks" if all_tasks else f"task{selected_task_indexes[0]:02d}"
    subtitle = (
        f"{len(selected_task_indexes)} tasks · samples is a per-task maximum"
        if all_tasks
        else tasks[selected_task_indexes[0]]
    )
    seed_note = f" · seed {seed}" if sampling == "random" else ""
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_dataset} · {selection_label} preview</title>
  <style>
    :root {{ color-scheme:dark; --bg:#0b1020; --panel:#121a2c; --line:#26324b;
      --text:#edf2ff; --muted:#9aa8c5; --accent:#7dd3fc; --accent2:#a7f3d0; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:radial-gradient(circle at 15% 0,#172744 0,var(--bg) 34%);
      color:var(--text); font:15px/1.55 Inter,system-ui,-apple-system,"Noto Sans KR",sans-serif; }}
    main {{ width:min(1500px,calc(100% - 32px)); margin:0 auto; padding:44px 0 80px; }}
    .eyebrow,.sample-no {{ color:var(--accent); font-size:12px; font-weight:800; letter-spacing:.13em; }}
    h1 {{ margin:8px 0 6px; font-size:clamp(28px,4vw,52px); line-height:1.08; }}
    .task {{ max-width:1050px; margin:0 0 12px; color:var(--accent2); font-size:18px; }}
    .meta,.path-note {{ color:var(--muted); }}
    .path-note {{ overflow-wrap:anywhere; font-family:ui-monospace,SFMono-Regular,monospace; font-size:12px; }}
    .toolbar {{ display:flex; gap:8px; margin:22px 0; }}
    button {{ border:0; border-radius:9px; padding:9px 13px; background:var(--accent); color:#082032;
      font-weight:750; cursor:pointer; }}
    button:hover {{ filter:brightness(1.1); }} button.secondary {{ background:#26324b; color:var(--text); }}
    .gallery {{ display:grid; gap:22px; }}
    .card {{ border:1px solid var(--line); border-radius:16px; padding:18px;
      background:color-mix(in srgb,var(--panel) 92%,transparent); box-shadow:0 16px 50px #0004; }}
    .card-header {{ display:flex; align-items:flex-end; justify-content:space-between; gap:16px; margin-bottom:14px; }}
    h2 {{ margin:2px 0 0; font-size:22px; }}
    .card-task {{ max-width:900px; margin:3px 0 0; color:var(--accent2); font-size:13px; }}
    .chips {{ display:flex; flex-wrap:wrap; gap:7px; justify-content:flex-end; }}
    .chips span {{ border:1px solid var(--line); border-radius:999px; padding:4px 9px; color:var(--muted); font-size:12px; }}
    .camera-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr)); gap:12px; }}
    figure {{ margin:0; min-width:0; }} figcaption {{ margin:0 0 6px; color:var(--muted); font-weight:700; }}
    video {{ display:block; width:100%; max-height:560px; border-radius:10px; background:#000; }}
    .card-actions {{ display:flex; gap:8px; margin-top:13px; }}
    .notice {{ margin-top:26px; padding:12px 14px; border:1px solid var(--line); border-radius:10px; color:var(--muted); }}
    @media (max-width:680px) {{ main {{ width:min(100% - 20px,1500px); padding-top:28px; }}
      .card {{ padding:12px; }} .card-header {{ align-items:flex-start; flex-direction:column; }} .chips {{ justify-content:flex-start; }} }}
  </style>
</head>
<body>
<main>
  <div class="eyebrow">TRAINING DATASET PREVIEW</div>
  <h1>{escaped_dataset} · {selection_label}</h1>
  <p class="task">{html.escape(subtitle)}</p>
  <p class="meta">{len(samples)} episodes · {html.escape(sampling)} sampling{seed_note}</p>
  <p class="path-note">source: {escaped_path}</p>
  <div class="toolbar"><button type="button" class="secondary" onclick="pauseAll()">Pause all</button></div>
  <section class="gallery">{''.join(cards)}</section>
  <p class="notice">Each selected episode range is extracted from packed AV1 source video as VP9 WebM and H.264 MP4.</p>
</main>
<script>
  const clips = document.querySelectorAll('video.clip');
  function cardVideos(button) {{ return button.closest('.card').querySelectorAll('video.clip'); }}
  function playCard(button) {{
    const videos = cardVideos(button);
    videos.forEach(video => {{ video.pause(); video.currentTime = 0; }});
    Promise.allSettled(Array.from(videos, video => video.play()));
  }}
  function resetCard(button) {{
    cardVideos(button).forEach(video => {{ video.pause(); video.currentTime = 0; }});
  }}
  function pauseAll() {{ clips.forEach(video => video.pause()); }}
</script>
</body>
</html>
"""


def main() -> None:
    reject_cli_arguments()
    config = load_config()
    dataset_config = dataset_settings(config)
    settings = visualization_settings(config)

    dataset_dir = dataset_config.dataset_dir
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    info = load_info(dataset_dir)
    tasks = load_tasks(dataset_dir)
    available_cameras = video_keys(info)
    episodes = annotate_episode_tasks(load_episodes(dataset_dir, available_cameras), tasks)
    counts = task_episode_counts(episodes, tasks)

    if settings.list_tasks_only:
        print(f"Dataset: {dataset_config.dataset}")
        print(f"Path   : {dataset_dir}")
        for task_index, language in sorted(tasks.items()):
            print(f"task{task_index:02d} | episodes={counts[task_index]:4d} | {language}")
        return
    ffmpeg = shutil.which(settings.ffmpeg)
    if ffmpeg is None:
        raise FileNotFoundError(f"ffmpeg executable not found: {settings.ffmpeg}")

    if settings.task.casefold() in {"all", "*"}:
        selected_task_indexes = sorted(tasks)
    else:
        selected_task_indexes = [resolve_task(settings.task, tasks)]

    default_camera = DEFAULT_CAMERA if DEFAULT_CAMERA in available_cameras else available_cameras[0]
    cameras = list(settings.cameras) or [default_camera]
    unknown_cameras = sorted(set(cameras) - set(available_cameras))
    if unknown_cameras:
        raise ValueError(
            f"Unknown camera(s): {unknown_cameras}. Available cameras: {available_cameras}"
        )

    samples = select_samples(
        dataset_dir=dataset_dir,
        episodes=episodes,
        info=info,
        task_indexes=selected_task_indexes,
        sample_count=settings.samples,
        sampling=settings.sampling,
        seed=settings.seed,
        camera_keys=cameras,
    )
    if not samples:
        raise ValueError("No episodes found for the selected task(s)")

    if settings.output:
        output_path = settings.output
        media_dir = output_path.parent / f"{output_path.stem}_media"
    else:
        dataset_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_config.dataset).strip("_")
        sampling_slug = (
            settings.sampling
            if settings.sampling == "first"
            else f"random_seed{settings.seed}"
        )
        if len(selected_task_indexes) == 1:
            selection_slug = f"task{selected_task_indexes[0]:02d}"
            sample_slug = str(len(samples))
        else:
            selection_slug = "all_tasks"
            sample_slug = f"{settings.samples}_per_task"
        run_dir = PREVIEW_ROOT / dataset_slug / f"{selection_slug}_{sampling_slug}_{sample_slug}"
        output_path = run_dir / "index.html"
        media_dir = run_dir / "media"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rendered = render_samples(
        samples,
        media_dir=media_dir,
        ffmpeg=ffmpeg,
        crf=settings.crf,
        preset=settings.preset,
        force=settings.force,
    )
    document = build_html(
        dataset_name=dataset_config.dataset,
        dataset_dir=dataset_dir,
        tasks=tasks,
        selected_task_indexes=selected_task_indexes,
        sampling=settings.sampling,
        seed=settings.seed,
        samples=rendered,
        output_path=output_path,
    )
    output_path.write_text(document, encoding="utf-8")

    media_size = sum(
        clip.mp4_path.stat().st_size
        + clip.webm_path.stat().st_size
        + clip.poster_path.stat().st_size
        for sample in rendered
        for clip in sample.clips
    )
    print(f"Dataset : {dataset_dir}")
    if len(selected_task_indexes) == 1:
        task_index = selected_task_indexes[0]
        episode_ids = ", ".join(str(sample.episode_index) for sample in samples)
        print(f"Task    : task{task_index:02d} | {tasks[task_index]}")
        print(f"Episodes: {episode_ids}")
    else:
        print(f"Tasks   : all {len(selected_task_indexes)} tasks")
        print(f"Episodes: {len(samples)} total (up to {settings.samples} per task)")
    print(f"Media   : {media_dir} ({media_size / 1024**2:.1f} MiB)")
    print(f"HTML    : {output_path}")


if __name__ == "__main__":
    main()
