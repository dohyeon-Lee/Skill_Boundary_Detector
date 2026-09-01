#!/usr/bin/env python3
"""Discover repeated, continuous CALVIN long-horizon task combinations."""

from __future__ import annotations

import csv
import html
import json
import os
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any
from urllib.parse import quote

import numpy as np
from PIL import Image

from calvin_long_horizon_config import Settings


@dataclass(frozen=True)
class Annotation:
    annotation_index: int
    recording_index: int
    start: int
    end: int
    task_id: str
    language: str


@dataclass(frozen=True)
class Event:
    recording_index: int
    start: int
    end: int
    task_id: str
    annotation_indexes: tuple[int, ...]
    languages: tuple[str, ...]


@dataclass(frozen=True)
class Occurrence:
    recording_index: int
    start: int
    end: int
    task_ids: tuple[str, ...]
    events: tuple[Event, ...]
    gaps: tuple[int, ...]

    @property
    def total_frames(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class Candidate:
    task_ids: tuple[str, ...]
    raw_occurrence_count: int
    occurrences: tuple[Occurrence, ...]

    @property
    def occurrence_count(self) -> int:
        return len(self.occurrences)


def candidate_key(task_ids: tuple[str, ...]) -> str:
    """Return a stable, human-readable identifier independent of report rank."""
    if len(task_ids) < 2 or any(not task_id.strip() for task_id in task_ids):
        raise ValueError(f"invalid long-horizon task sequence: {task_ids!r}")
    if any("__" in task_id for task_id in task_ids):
        raise ValueError(
            "CALVIN task IDs used in a candidate key must not contain '__': "
            f"{task_ids!r}"
        )
    return f"{len(task_ids)}step__" + "__".join(task_ids)


def _load_numpy_object(path: Path) -> Any:
    value = np.load(path, allow_pickle=True)
    if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
        return value.item()
    return value


def load_recordings(source_dir: Path) -> list[tuple[int, int]]:
    path = source_dir / "ep_start_end_ids.npy"
    if not path.is_file():
        raise FileNotFoundError(f"CALVIN play recording boundaries not found: {path}")
    array = np.asarray(np.load(path, allow_pickle=False), dtype=np.int64)
    if array.shape == (2,):
        array = array.reshape(1, 2)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) == 0:
        raise ValueError(f"ep_start_end_ids.npy must have shape (N, 2), got {array.shape}")
    recordings = [tuple(map(int, pair)) for pair in array]
    for index, (start, end) in enumerate(recordings):
        if end < start:
            raise ValueError(f"recording {index} has descending interval [{start}, {end}]")
    ordered = sorted(enumerate(recordings), key=lambda item: item[1][0])
    for (left_index, (_, left_end)), (right_index, (right_start, _)) in zip(
        ordered, ordered[1:], strict=False
    ):
        if right_start <= left_end:
            raise ValueError(
                f"recordings overlap: {left_index} ends {left_end}, "
                f"{right_index} starts {right_start}"
            )
    return recordings


def load_annotations(
    source_dir: Path,
    annotation_folder: str,
    recordings: list[tuple[int, int]],
) -> tuple[list[Annotation], Path]:
    path = source_dir / annotation_folder / "auto_lang_ann.npy"
    if not path.is_file():
        raise FileNotFoundError(f"CALVIN language annotations not found: {path}")
    payload = _load_numpy_object(path)
    try:
        intervals = np.asarray(payload["info"]["indx"], dtype=np.int64)
        task_ids = np.asarray(payload["language"]["task"]).astype(str)
        languages = np.asarray(payload["language"]["ann"]).astype(str)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unexpected CALVIN language annotation structure: {path}") from exc
    if intervals.shape != (len(task_ids), 2) or len(languages) != len(task_ids):
        raise ValueError(
            "CALVIN annotation arrays disagree: "
            f"intervals={intervals.shape}, tasks={len(task_ids)}, languages={len(languages)}"
        )

    annotations: list[Annotation] = []
    for annotation_index, ((start_raw, end_raw), task_id, language) in enumerate(
        zip(intervals, task_ids, languages, strict=True)
    ):
        start, end = int(start_raw), int(end_raw)
        matches = [
            index
            for index, (recording_start, recording_end) in enumerate(recordings)
            if recording_start <= start and end <= recording_end
        ]
        if len(matches) != 1:
            raise ValueError(
                f"annotation {annotation_index} [{start}, {end}] belongs to "
                f"{len(matches)} play recordings"
            )
        annotations.append(
            Annotation(
                annotation_index=annotation_index,
                recording_index=matches[0],
                start=start,
                end=end,
                task_id=str(task_id),
                language=str(language),
            )
        )
    return annotations, path


def merge_overlapping_same_task(annotations: list[Annotation]) -> list[Event]:
    """Collapse overlapping windows for the same semantic task into one event."""
    grouped: dict[tuple[int, str], list[Annotation]] = defaultdict(list)
    for annotation in annotations:
        grouped[(annotation.recording_index, annotation.task_id)].append(annotation)

    events: list[Event] = []
    for (recording_index, task_id), rows in grouped.items():
        rows.sort(key=lambda row: (row.start, row.end, row.annotation_index))
        cluster: list[Annotation] = []
        cluster_end: int | None = None
        for row in rows:
            if cluster and cluster_end is not None and row.start > cluster_end:
                events.append(_event_from_cluster(recording_index, task_id, cluster))
                cluster = []
                cluster_end = None
            cluster.append(row)
            cluster_end = row.end if cluster_end is None else max(cluster_end, row.end)
        if cluster:
            events.append(_event_from_cluster(recording_index, task_id, cluster))
    return sorted(events, key=lambda event: (event.recording_index, event.start, event.end, event.task_id))


def _event_from_cluster(recording_index: int, task_id: str, rows: list[Annotation]) -> Event:
    return Event(
        recording_index=recording_index,
        start=min(row.start for row in rows),
        end=max(row.end for row in rows),
        task_id=task_id,
        annotation_indexes=tuple(row.annotation_index for row in rows),
        languages=tuple(dict.fromkeys(row.language for row in rows)),
    )


def _successors(events: list[Event], index: int, max_gap_frames: int) -> list[int]:
    """Return the earliest genuinely later event(s), ignoring overlapping co-labels."""
    current = events[index]
    later = [
        candidate
        for candidate in range(index + 1, len(events))
        if events[candidate].start > current.end
        and events[candidate].start - current.end - 1 <= max_gap_frames
    ]
    if not later:
        return []
    earliest_start = min(events[candidate].start for candidate in later)
    return [candidate for candidate in later if events[candidate].start == earliest_start]


def enumerate_occurrences(
    events: list[Event],
    sequence_steps: tuple[int, ...],
    min_total_frames: int,
    max_total_frames: int,
    max_gap_frames: int,
) -> dict[tuple[str, ...], list[Occurrence]]:
    by_recording: dict[int, list[Event]] = defaultdict(list)
    for event in events:
        by_recording[event.recording_index].append(event)

    grouped: dict[tuple[str, ...], list[Occurrence]] = defaultdict(list)
    signatures: set[tuple[Any, ...]] = set()
    wanted_steps = set(sequence_steps)
    max_steps = max(sequence_steps)

    for recording_index, recording_events in by_recording.items():
        recording_events.sort(key=lambda event: (event.start, event.end, event.task_id))
        edges = {
            index: _successors(recording_events, index, max_gap_frames)
            for index in range(len(recording_events))
        }

        def walk(path: list[int]) -> None:
            selected = tuple(recording_events[index] for index in path)
            if len(selected) in wanted_steps:
                task_ids = tuple(event.task_id for event in selected)
                if all(left != right for left, right in zip(task_ids, task_ids[1:], strict=False)):
                    start, end = selected[0].start, selected[-1].end
                    total_frames = end - start + 1
                    if min_total_frames <= total_frames <= max_total_frames:
                        signature = (recording_index, start, end, task_ids)
                        if signature not in signatures:
                            signatures.add(signature)
                            gaps = tuple(
                                right.start - left.end - 1
                                for left, right in zip(selected, selected[1:], strict=False)
                            )
                            grouped[task_ids].append(
                                Occurrence(
                                    recording_index=recording_index,
                                    start=start,
                                    end=end,
                                    task_ids=task_ids,
                                    events=selected,
                                    gaps=gaps,
                                )
                            )
            if len(path) == max_steps:
                return
            for successor in edges[path[-1]]:
                walk([*path, successor])

        for start_index in range(len(recording_events)):
            walk([start_index])
    return grouped


def independent_occurrences(occurrences: list[Occurrence]) -> list[Occurrence]:
    """Select a maximum-size set of non-overlapping spans within each recording."""
    by_recording: dict[int, list[Occurrence]] = defaultdict(list)
    for occurrence in occurrences:
        by_recording[occurrence.recording_index].append(occurrence)
    selected: list[Occurrence] = []
    for recording_index in sorted(by_recording):
        last_end: int | None = None
        for occurrence in sorted(
            by_recording[recording_index], key=lambda item: (item.end, item.start)
        ):
            if last_end is None or occurrence.start > last_end:
                selected.append(occurrence)
                last_end = occurrence.end
    return sorted(selected, key=lambda item: (item.recording_index, item.start, item.end))


def discover_candidates(
    events: list[Event],
    settings: Settings,
) -> tuple[list[Candidate], dict[str, int]]:
    raw = enumerate_occurrences(
        events,
        sequence_steps=settings.search.sequence_steps,
        min_total_frames=settings.search.min_total_frames,
        max_total_frames=settings.search.max_total_frames,
        max_gap_frames=settings.search.max_gap_frames,
    )
    candidates = []
    for task_ids, occurrences in raw.items():
        independent = independent_occurrences(occurrences)
        if len(independent) >= min(settings.search.min_occurrences):
            candidates.append(
                Candidate(
                    task_ids=task_ids,
                    raw_occurrence_count=len(occurrences),
                    occurrences=tuple(independent),
                )
            )
    candidates.sort(
        key=lambda candidate: (
            -candidate.occurrence_count,
            -median(item.total_frames for item in candidate.occurrences),
            candidate.task_ids,
        )
    )
    counts = Counter(str(len(candidate.task_ids)) for candidate in candidates)
    return candidates, dict(sorted(counts.items()))


def candidate_counts_by_min_occurrences(
    candidates: list[Candidate],
    min_occurrences: tuple[int, ...],
) -> dict[str, dict[str, Any]]:
    """Summarize nested occurrence thresholds without duplicating candidates."""
    result: dict[str, dict[str, Any]] = {}
    for threshold in min_occurrences:
        eligible = [
            candidate
            for candidate in candidates
            if candidate.occurrence_count >= threshold
        ]
        by_steps = Counter(str(len(candidate.task_ids)) for candidate in eligible)
        result[str(threshold)] = {
            "candidate_count": len(eligible),
            "candidate_count_by_steps": dict(sorted(by_steps.items())),
        }
    return result


def _occurrence_dict(occurrence: Occurrence, fps: int) -> dict[str, Any]:
    return {
        "recording_index": occurrence.recording_index,
        "source_start": occurrence.start,
        "source_end": occurrence.end,
        "total_frames": occurrence.total_frames,
        "total_seconds": occurrence.total_frames / fps,
        "gaps_frames": list(occurrence.gaps),
        "events": [asdict(event) for event in occurrence.events],
    }


def write_reports(
    output_dir: Path,
    candidates: list[Candidate],
    settings: Settings,
    annotation_path: Path,
    source_annotation_count: int,
    merged_event_count: int,
    by_steps: dict[str, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_min_occurrences = candidate_counts_by_min_occurrences(
        candidates, settings.search.min_occurrences
    )
    payload = {
        "schema_version": 3,
        "source_dir": str(settings.source.source_dir),
        "annotation_file": str(annotation_path),
        "source_annotations": source_annotation_count,
        "merged_semantic_events": merged_event_count,
        "candidate_count": len(candidates),
        "candidate_count_by_steps": by_steps,
        "candidate_counts_by_min_occurrences": by_min_occurrences,
        "search": {
            "sequence_steps": list(settings.search.sequence_steps),
            "min_total_frames": settings.search.min_total_frames,
            "max_total_frames": settings.search.max_total_frames,
            "max_gap_frames": settings.search.max_gap_frames,
            "min_occurrences": list(settings.search.min_occurrences),
            "fps": settings.source.fps,
            "event_deduplication": "merge overlapping annotations with the same task_id",
            "transition_rule": "earliest non-overlapping event within max_gap_frames",
            "occurrence_counting": "maximum non-overlapping occurrences per recording",
        },
        "candidates": [
            {
                "rank": rank,
                "candidate_id": f"candidate_{rank:03d}",
                "candidate_key": candidate_key(candidate.task_ids),
                "task_ids": list(candidate.task_ids),
                "language_draft": " then ".join(
                    task_id.replace("_", " ") for task_id in candidate.task_ids
                ),
                "occurrence_count": candidate.occurrence_count,
                "raw_occurrence_count": candidate.raw_occurrence_count,
                "eligible_min_occurrences": [
                    threshold
                    for threshold in settings.search.min_occurrences
                    if candidate.occurrence_count >= threshold
                ],
                "duration_frames": _summary(
                    [occurrence.total_frames for occurrence in candidate.occurrences]
                ),
                "duration_seconds": _summary(
                    [occurrence.total_frames / settings.source.fps for occurrence in candidate.occurrences]
                ),
                "gap_frames": _summary(
                    [gap for occurrence in candidate.occurrences for gap in occurrence.gaps]
                ),
                "occurrences": [
                    _occurrence_dict(occurrence, settings.source.fps)
                    for occurrence in candidate.occurrences
                ],
            }
            for rank, candidate in enumerate(candidates, start=1)
        ],
    }
    _write_json(output_dir / "candidates.json", payload)

    with (output_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "rank",
                "candidate_id",
                "candidate_key",
                "task_sequence",
                "steps",
                "occurrences",
                "raw_occurrences",
                "eligible_min_occurrences",
                "min_frames",
                "median_frames",
                "mean_frames",
                "max_frames",
                "median_seconds",
            ],
        )
        writer.writeheader()
        for rank, candidate in enumerate(candidates, start=1):
            lengths = [occurrence.total_frames for occurrence in candidate.occurrences]
            writer.writerow(
                {
                    "rank": rank,
                    "candidate_id": f"candidate_{rank:03d}",
                    "candidate_key": candidate_key(candidate.task_ids),
                    "task_sequence": " -> ".join(candidate.task_ids),
                    "steps": len(candidate.task_ids),
                    "occurrences": candidate.occurrence_count,
                    "raw_occurrences": candidate.raw_occurrence_count,
                    "eligible_min_occurrences": ",".join(
                        str(threshold)
                        for threshold in settings.search.min_occurrences
                        if candidate.occurrence_count >= threshold
                    ),
                    "min_frames": min(lengths),
                    "median_frames": median(lengths),
                    "mean_frames": mean(lengths),
                    "max_frames": max(lengths),
                    "median_seconds": median(lengths) / settings.source.fps,
                }
            )


def _summary(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "median": median(values),
        "mean": mean(values),
        "max": max(values),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _video_cache_key(
    occurrence: Occurrence,
    settings: Settings,
    width: int,
    height: int,
) -> dict[str, Any]:
    first = settings.source.source_dir / f"episode_{occurrence.start:07d}.npz"
    last = settings.source.source_dir / f"episode_{occurrence.end:07d}.npz"
    return {
        "source_dir": str(settings.source.source_dir),
        "source_start": occurrence.start,
        "source_end": occurrence.end,
        "first_size": first.stat().st_size,
        "first_mtime_ns": first.stat().st_mtime_ns,
        "last_size": last.stat().st_size,
        "last_mtime_ns": last.stat().st_mtime_ns,
        "camera": settings.visualization.camera,
        "width": width,
        "height": height,
        "fps": settings.source.fps,
        "codec": "libx264",
        "crf": settings.visualization.crf,
        "preset": settings.visualization.preset,
    }


def render_occurrence_video(
    occurrence: Occurrence,
    output_path: Path,
    settings: Settings,
    ffmpeg: str,
) -> bool:
    first_frame = _read_rgb(
        settings.source.source_dir / f"episode_{occurrence.start:07d}.npz",
        settings.visualization.camera,
    )
    if settings.visualization.image_size == "native":
        height, width = first_frame.shape[:2]
    else:
        width = height = int(settings.visualization.image_size)
    cache_key = _video_cache_key(occurrence, settings, width, height)
    cache_path = output_path.with_suffix(f"{output_path.suffix}.json")
    if (
        not settings.visualization.force
        and output_path.is_file()
        and output_path.stat().st_size > 0
        and cache_path.is_file()
    ):
        try:
            if json.loads(cache_path.read_text(encoding="utf-8")) == cache_key:
                return False
        except (json.JSONDecodeError, OSError):
            pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.tmp-{os.getpid()}.mp4")
    temporary.unlink(missing_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(settings.source.fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        settings.visualization.preset,
        "-crf",
        str(settings.visualization.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        assert process.stdin is not None
        for frame_index in range(occurrence.start, occurrence.end + 1):
            frame = _read_rgb(
                settings.source.source_dir / f"episode_{frame_index:07d}.npz",
                settings.visualization.camera,
            )
            if frame.shape[:2] != (height, width):
                frame = np.asarray(
                    Image.fromarray(frame).resize((width, height), Image.Resampling.BILINEAR),
                    dtype=np.uint8,
                )
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        process.stdin.close()
        assert process.stderr is not None
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(f"ffmpeg failed for [{occurrence.start}, {occurrence.end}]: {stderr}")
        temporary.replace(output_path)
        _write_json(cache_path, cache_key)
        return True
    except Exception:
        if process.poll() is None:
            process.kill()
            process.wait()
        temporary.unlink(missing_ok=True)
        raise


def _read_rgb(path: Path, camera: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"CALVIN timestep not found: {path}")
    with np.load(path, allow_pickle=False) as bundle:
        if camera not in bundle.files:
            raise ValueError(f"{path} has no camera field {camera!r}")
        frame = np.asarray(bundle[camera])
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Invalid {camera} frame in {path}: dtype={frame.dtype}, shape={frame.shape}")
    return np.ascontiguousarray(frame)


def choose_visualized_candidates(
    candidates: list[Candidate], settings: Settings
) -> list[tuple[int, Candidate]]:
    """Choose up to max_candidates independently for each sequence length."""
    limit = settings.visualization.max_candidates
    ranked = list(enumerate(candidates, start=1))
    if limit is None:
        return ranked
    selected: list[tuple[int, Candidate]] = []
    counts: Counter[int] = Counter()
    for rank, candidate in ranked:
        steps = len(candidate.task_ids)
        if counts[steps] >= limit:
            continue
        selected.append((rank, candidate))
        counts[steps] += 1
    return selected


def choose_occurrences(
    candidate: Candidate,
    rank: int,
    settings: Settings,
) -> list[Occurrence]:
    count = min(settings.visualization.samples_per_candidate, candidate.occurrence_count)
    rows = list(candidate.occurrences)
    if settings.visualization.sampling == "random":
        return random.Random(settings.visualization.seed + rank).sample(rows, count)
    return rows[:count]


def write_visualization(
    output_dir: Path,
    candidates: list[Candidate],
    settings: Settings,
) -> Path:
    ffmpeg = shutil.which(settings.visualization.ffmpeg)
    if ffmpeg is None:
        raise FileNotFoundError(
            f"ffmpeg executable not found: {settings.visualization.ffmpeg}"
        )
    selected = choose_visualized_candidates(candidates, settings)
    media_dir = output_dir / "media"
    rendered: list[tuple[int, Candidate, list[tuple[Occurrence, Path]]]] = []
    total = sum(
        min(settings.visualization.samples_per_candidate, candidate.occurrence_count)
        for _, candidate in selected
    )
    complete = 0
    for rank, candidate in selected:
        samples = []
        for sample_index, occurrence in enumerate(
            choose_occurrences(candidate, rank, settings), start=1
        ):
            complete += 1
            path = (
                media_dir
                / f"candidate_{rank:03d}"
                / f"sample_{sample_index:02d}_{occurrence.start}_{occurrence.end}.mp4"
            )
            changed = render_occurrence_video(occurrence, path, settings, ffmpeg)
            print(
                f"  video {complete:3d}/{total}: {'encoded' if changed else 'cached'} "
                f"candidate={rank} frames=[{occurrence.start}, {occurrence.end}]",
                flush=True,
            )
            samples.append((occurrence, path))
        rendered.append((rank, candidate, samples))
    html_path = output_dir / "index.html"
    html_path.write_text(_build_html(rendered, html_path, candidates, settings), encoding="utf-8")
    return html_path


def _url(path: Path, output_path: Path) -> str:
    return quote(os.path.relpath(path, output_path.parent).replace(os.sep, "/"), safe="/@-._~")


def _build_html(
    rendered: list[tuple[int, Candidate, list[tuple[Occurrence, Path]]]],
    output_path: Path,
    all_candidates: list[Candidate],
    settings: Settings,
) -> str:
    cards_by_steps: dict[int, list[str]] = defaultdict(list)
    index_rows_by_steps: dict[int, list[str]] = defaultdict(list)
    for rank, candidate, samples in rendered:
        steps = len(candidate.task_ids)
        sequence = " → ".join(candidate.task_ids)
        draft = " then ".join(task_id.replace("_", " ") for task_id in candidate.task_ids)
        lengths = [occurrence.total_frames for occurrence in candidate.occurrences]
        eligible_thresholds = [
            threshold
            for threshold in settings.search.min_occurrences
            if candidate.occurrence_count >= threshold
        ]
        threshold_text = ", ".join(f"≥{value}" for value in eligible_thresholds)
        anchor = f"candidate-{rank:03d}"
        stable_key = candidate_key(candidate.task_ids)
        index_rows_by_steps[steps].append(
            f"<tr><td>{rank:03d}</td><td>{steps}</td>"
            f"<td><a href=\"#{anchor}\">{html.escape(sequence)}</a></td>"
            f"<td><code>{html.escape(stable_key)}</code></td>"
            f"<td>{candidate.occurrence_count}</td><td>{html.escape(threshold_text)}</td>"
            f"<td>{median(lengths) / settings.source.fps:.2f}s</td></tr>"
        )
        sample_blocks = []
        for sample_index, (occurrence, path) in enumerate(samples, start=1):
            step_rows = []
            for step_index, event in enumerate(occurrence.events, start=1):
                prompts = " / ".join(event.languages[:3])
                step_rows.append(
                    f"<li><b>{step_index}. {html.escape(event.task_id)}</b> "
                    f"[{event.start}, {event.end}]<br>{html.escape(prompts)}</li>"
                )
            sample_blocks.append(
                f"""
                <article class="sample">
                  <h3>Sample {sample_index} · recording {occurrence.recording_index}</h3>
                  <p>{occurrence.total_frames} frames · {occurrence.total_frames / settings.source.fps:.2f}s ·
                  source [{occurrence.start}, {occurrence.end}] · gaps {list(occurrence.gaps)}</p>
                  <video controls muted playsinline preload="metadata" src="{html.escape(_url(path, output_path), quote=True)}"></video>
                  <ol>{''.join(step_rows)}</ol>
                </article>"""
            )
        cards_by_steps[steps].append(
            f"""
            <article class="candidate" id="{anchor}">
              <header><div><span class="rank">CANDIDATE {rank:03d}</span>
                <h2>{html.escape(sequence)}</h2>
                <p><code>{html.escape(stable_key)}</code>
                  <button class="copy-key" data-key="{html.escape(stable_key, quote=True)}">Copy key</button></p>
                <p class="draft">Draft: {html.escape(draft)}</p></div>
                <div class="metrics"><b>{candidate.occurrence_count} independent demos</b>
                  <span>meets {threshold_text} occurrence thresholds</span>
                  <span>median {median(lengths):.0f} frames / {median(lengths) / settings.source.fps:.2f}s</span></div></header>
              <div class="samples">{''.join(sample_blocks)}</div>
              <a class="back" href="#candidate-index">Back to candidate table ↑</a>
            </article>"""
        )
    all_counts = Counter(len(candidate.task_ids) for candidate in all_candidates)
    shown_counts = Counter(len(candidate.task_ids) for _, candidate, _ in rendered)
    step_values = sorted(
        set(settings.search.sequence_steps) | set(all_counts) | set(shown_counts)
    )
    by_min_occurrences = candidate_counts_by_min_occurrences(
        all_candidates, settings.search.min_occurrences
    )
    summary_header = "".join(f"<th>{steps}-step</th>" for steps in step_values)
    summary_rows = []
    for threshold in settings.search.min_occurrences:
        counts = by_min_occurrences[str(threshold)]
        step_counts = counts["candidate_count_by_steps"]
        cells = "".join(
            f"<td>{step_counts.get(str(steps), 0)}</td>" for steps in step_values
        )
        summary_rows.append(
            f"<tr><th>≥{threshold} occurrences</th>{cells}"
            f"<td>{counts['candidate_count']}</td></tr>"
        )

    index_groups = []
    video_groups = []
    hidden_total = 0
    for steps in step_values:
        rows = index_rows_by_steps.get(steps, [])
        shown = shown_counts.get(steps, 0)
        eligible = all_counts.get(steps, 0)
        hidden = eligible - shown
        hidden_total += hidden
        if rows:
            index_groups.append(
                f"<tr class=\"group-row\"><th colspan=\"7\">{steps}-step candidates "
                f"({shown} shown / {eligible} eligible)</th></tr>{''.join(rows)}"
            )
        cards = cards_by_steps.get(steps, [])
        if cards:
            video_groups.append(
                f"<section class=\"step-group\" id=\"step-{steps}\">"
                f"<div class=\"step-heading\"><div><span class=\"rank\">{steps}-STEP</span>"
                f"<h2>{steps}-step candidate videos</h2></div>"
                f"<a href=\"#candidate-index\">Candidate table ↑</a></div>"
                f"{''.join(cards)}</section>"
            )
    hidden_note = (
        f"<p class=\"note\">{hidden_total} additional eligible candidates are listed in "
        "candidates.csv/json. Increase max_candidates or leave it blank to render them.</p>"
        if hidden_total > 0
        else ""
    )
    per_step_text = (
        "all candidates per step"
        if settings.visualization.max_candidates is None
        else f"up to {settings.visualization.max_candidates} candidates per step"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CALVIN long-horizon candidates</title>
<style>
:root{{color-scheme:dark;--bg:#08111e;--panel:#111d2e;--line:#29405b;--text:#ecf5ff;--muted:#9eb1c8;--accent:#67e8f9;--green:#86efac}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,sans-serif}}
html{{scroll-behavior:smooth}} main{{width:min(1500px,calc(100% - 30px));margin:auto;padding:36px 0 80px}} h1{{font-size:38px;margin:4px 0}} .meta,.note{{color:var(--muted)}}
.summary-panel{{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px;margin:22px 0;overflow-x:auto}}
table{{width:100%;border-collapse:collapse}} th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left}} th{{color:var(--text)}}
td{{color:var(--muted)}} td a,.step-heading a,.back{{color:var(--accent)}} .group-row th{{padding-top:22px;color:var(--green);background:#0a1524}}
code{{color:#f9a8d4;white-space:nowrap}} .copy-key{{margin-left:8px;padding:3px 8px;border:1px solid var(--line);border-radius:6px;background:#0a1524;color:var(--accent);cursor:pointer}}
.step-group{{scroll-margin-top:16px;margin-top:42px}} .step-heading{{display:flex;align-items:end;justify-content:space-between;gap:16px;border-bottom:1px solid var(--line)}}
.step-heading h2{{font-size:30px;margin:2px 0 10px}} .back{{display:inline-block;margin-top:14px}}
.candidate{{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px;margin:22px 0}}
.candidate>header{{display:flex;justify-content:space-between;gap:18px;align-items:flex-end}} h2{{margin:4px 0;font-size:23px;color:var(--green)}}
.rank{{font-weight:800;letter-spacing:.12em;color:var(--accent)}} .draft{{margin:2px 0;color:var(--muted)}} .metrics{{display:grid;text-align:right;color:var(--muted)}}
.samples{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,350px),1fr));gap:14px;margin-top:16px}}
.sample{{border:1px solid var(--line);border-radius:12px;padding:12px;background:#0a1524}} .sample h3{{margin:0}} .sample p,li{{color:var(--muted)}}
video{{width:100%;border-radius:9px;background:#000}} ol{{padding-left:22px}} @media(max-width:700px){{.candidate>header{{display:block}}.metrics{{text-align:left}}}}
</style></head><body><main>
<span class="rank">CALVIN CONTINUOUS PLAY</span><h1>Long-horizon task candidates</h1>
<p class="meta">{len(all_candidates)} eligible semantic combinations · min occurrence thresholds {list(settings.search.min_occurrences)} ·
span {settings.search.min_total_frames}--{settings.search.max_total_frames} frames · max gap {settings.search.max_gap_frames} frames · {per_step_text}</p>
<section class="summary-panel"><h2>Eligibility summary</h2>
<table><thead><tr><th>Threshold</th>{summary_header}<th>Total</th></tr></thead>
<tbody>{''.join(summary_rows)}</tbody></table></section>
<section class="summary-panel" id="candidate-index"><h2>Visualized candidates</h2>
<p class="note">Click a task sequence to jump to its videos.</p>
<table><thead><tr><th>Rank</th><th>Steps</th><th>Task sequence</th><th>Candidate key</th><th>Demos</th><th>Thresholds</th><th>Median</th></tr></thead>
<tbody>{''.join(index_groups)}</tbody></table></section>
{hidden_note}{''.join(video_groups)}
</main><script>
document.querySelectorAll('.copy-key').forEach((button) => button.addEventListener('click', async () => {{
  await navigator.clipboard.writeText(button.dataset.key);
  const old = button.textContent; button.textContent = 'Copied';
  setTimeout(() => button.textContent = old, 1000);
}}));
</script></body></html>"""


def run(settings: Settings) -> tuple[list[Candidate], Path | None]:
    if not settings.source.source_dir.is_dir():
        raise FileNotFoundError(f"CALVIN source split not found: {settings.source.source_dir}")
    recordings = load_recordings(settings.source.source_dir)
    annotations, annotation_path = load_annotations(
        settings.source.source_dir,
        settings.source.annotation_folder,
        recordings,
    )
    events = merge_overlapping_same_task(annotations)
    candidates, by_steps = discover_candidates(events, settings)
    by_min_occurrences = candidate_counts_by_min_occurrences(
        candidates, settings.search.min_occurrences
    )

    output_dir = settings.visualization.output_dir
    write_reports(
        output_dir,
        candidates,
        settings,
        annotation_path,
        source_annotation_count=len(annotations),
        merged_event_count=len(events),
        by_steps=by_steps,
    )
    html_path = None
    if settings.visualization.enabled:
        html_path = write_visualization(output_dir, candidates, settings)

    print("CALVIN long-horizon candidate discovery", flush=True)
    print(f"  source annotations : {len(annotations)}", flush=True)
    print(f"  merged events      : {len(events)}", flush=True)
    print(f"  eligible candidates: {len(candidates)} ({by_steps})", flush=True)
    for threshold, counts in by_min_occurrences.items():
        print(
            f"    occurrences >= {threshold}: {counts['candidate_count']} "
            f"({counts['candidate_count_by_steps']})",
            flush=True,
        )
    print(f"  report JSON        : {output_dir / 'candidates.json'}", flush=True)
    print(f"  report CSV         : {output_dir / 'candidates.csv'}", flush=True)
    if html_path is not None:
        print(f"  visualization      : {html_path}", flush=True)
    return candidates, html_path
