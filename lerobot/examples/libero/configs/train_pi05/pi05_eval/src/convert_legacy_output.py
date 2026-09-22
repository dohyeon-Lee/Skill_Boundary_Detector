#!/usr/bin/env python3
"""Convert a pre-Stage-1-format pi05 eval output folder to the current layout, in place.

Legacy layout (one folder per label, raw videos with sidecar files)::

    panels/<label>/eval_info.json, eval_info_<tag>.json, task_success_rates.png
    panels/<label>/videos/<task>/eval_episode_<k>.mp4, language.txt, success.json

Current layout (what eval.sbatch now writes)::

    metrics/eval_info_<tag>.json = {label: {overall, per_task}}, metrics/eval_info_merged.json
    task_success_rates.png (+ checkpoint_success_rates.png for sweeps)
    panels/<NN>_<label>/videos/<task>/eval_episode_<k>.mp4   (SUCCESS/FAIL bar + caption baked in)
    side_by_side/<task>/eval_episode_<k>.mp4                 (only with >= 2 panels)

Videos are re-rendered from the legacy frames (rollout view only - the wrist camera was never
recorded). Legacy files are moved to ``.legacy_format/``, never deleted. No W&B logging.
Idempotent: already-converted outputs are left alone.

    python src/convert_legacy_output.py outputs/PT_libero90_30k [--labels a,b]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

_STAGE1_SRC = Path(__file__).resolve().parents[3] / "train_skillVLA" / "stage1_eval" / "src"
sys.path.insert(0, str(_STAGE1_SRC))
from compare_videos import read_video, stitch_panel_videos  # noqa: E402

_PANEL_DIR = re.compile(r"^\d{2}_")
_EPISODE = re.compile(r"^eval_episode_(\d+)\.mp4$")

Annotator = Callable[[np.ndarray, bool, str | None], np.ndarray]


def panel_dir_name(index: int, label: str) -> str:
    """Same rule as pi05_eval_config.panel_dir_name (kept import-free of the YAML resolver)."""
    safe = "".join(character if character.isalnum() or character in "._-" else "-" for character in label)
    return f"{index:02d}_{safe.strip('-_') or 'model'}"


def _stage1_merge():
    # Loaded by path: pi05_eval/src has its own merge_eval_chunks.py with the same module name.
    spec = importlib.util.spec_from_file_location("stage1_merge_eval_chunks", _STAGE1_SRC / "merge_eval_chunks.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _default_annotator() -> Annotator:
    from lerobot.scripts.lerobot_skillvla_eval import _annotate_eval_video  # noqa: PLC0415

    return lambda frames, success, language: _annotate_eval_video(frames, success, language)


def legacy_panels(out_dir: Path, labels: list[str] | None = None) -> list[str]:
    """Legacy panel folder names in display order (``labels`` first, then the rest sorted)."""
    panels_dir = out_dir / "panels"
    found = sorted(
        path.name for path in panels_dir.glob("*")
        if path.is_dir() and not _PANEL_DIR.match(path.name)
        and (any(path.glob("eval_info*.json")) or (path / "videos").is_dir())
    ) if panels_dir.is_dir() else []
    ordered = [label for label in (labels or []) if label in found]
    return ordered + [label for label in found if label not in ordered]


def _legacy_chunks(panel: Path) -> list[tuple[str, Path]]:
    """``[(tag, path)]``: per-worker chunk files, else the legacy merged file as one chunk."""
    chunks = [
        (path.stem.removeprefix("eval_info_"), path)
        for path in sorted(panel.glob("eval_info_*.json"))
    ]
    if not chunks and (panel / "eval_info.json").is_file():
        chunks = [("legacy", panel / "eval_info.json")]
    return chunks


def _relocate(value, old: str, new: str):
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_relocate(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _relocate(item, old, new) for key, item in value.items()}
    return value


def convert_metrics(out_dir: Path, label: str, legacy: Path, panel_dir: Path) -> int:
    """Write the label's chunks into ``metrics/eval_info_<tag>.json``. Returns chunks written."""
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    old_videos, new_videos = str(legacy / "videos"), str(panel_dir / "videos")
    written = 0
    for tag, path in _legacy_chunks(legacy):
        info = json.loads(path.read_text())
        target = metrics_dir / f"eval_info_{tag}.json"
        existing = json.loads(target.read_text()) if target.is_file() else {}
        if label in existing:
            continue
        existing[label] = _relocate(
            {"overall": info.get("overall", {}), "per_task": info.get("per_task", [])},
            old_videos, new_videos,
        )
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(json.dumps(existing, indent=2))
        temporary.replace(target)
        written += 1
    return written


def _successes_from_metrics(legacy: Path) -> dict[str, list[bool]]:
    """``{task folder: successes}`` from the metrics, for tasks missing their success.json."""
    result = {}
    for _, path in _legacy_chunks(legacy):
        for task in json.loads(path.read_text()).get("per_task", []):
            name = f"{task.get('task_group', '')}_{task.get('task_id', '')}"
            successes = task.get("metrics", {}).get("successes")
            if isinstance(successes, list):
                result[name] = [bool(value) for value in successes]
    return result


def convert_videos(legacy: Path, panel_dir: Path, annotate: Annotator) -> int:
    """Re-render legacy episode videos with the SUCCESS/FAIL bar and caption. Returns videos written."""
    fallback = _successes_from_metrics(legacy)
    written = 0
    for task_dir in sorted(path for path in (legacy / "videos").glob("*") if path.is_dir()):
        language_file = task_dir / "language.txt"
        language = language_file.read_text().strip() if language_file.is_file() else None
        success_file = task_dir / "success.json"
        successes = (
            [bool(value) for value in json.loads(success_file.read_text())]
            if success_file.is_file() else fallback.get(task_dir.name, [])
        )
        for video in sorted(task_dir.glob("eval_episode_*.mp4")):
            match = _EPISODE.match(video.name)
            if match is None:
                continue
            destination = panel_dir / "videos" / task_dir.name / video.name
            if destination.is_file() and destination.stat().st_size > 0:
                continue
            episode = int(match.group(1))
            if episode >= len(successes):
                print(f"convert: no success flag for {video}; skipped")
                continue
            frames, fps = read_video(video)
            if not frames:
                continue
            annotated = annotate(np.stack(frames), successes[episode], language)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f"{destination.stem}.tmp.mp4")
            imageio.mimsave(str(temporary), list(annotated), fps=fps)
            temporary.replace(destination)
            written += 1
    return written


def convert(
    out_dir: Path, *, labels: list[str] | None = None, annotate: Annotator | None = None,
    expected_tasks: int = 0, grid_columns: int = 0,
) -> list[str]:
    """Convert ``out_dir`` in place. Returns the converted labels in panel order."""
    order = legacy_panels(out_dir, labels)
    if not order:
        print(f"convert: no legacy panel folders under {out_dir / 'panels'}; nothing to do.")
        return []
    backup = out_dir / ".legacy_format"
    for index, label in enumerate(order):
        legacy = out_dir / "panels" / label
        panel_dir = out_dir / "panels" / panel_dir_name(index, label)
        chunks = convert_metrics(out_dir, label, legacy, panel_dir)
        videos = 0
        if (legacy / "videos").is_dir():
            if annotate is None:
                annotate = _default_annotator()
            videos = convert_videos(legacy, panel_dir, annotate)
        (backup / "panels").mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy), str(backup / "panels" / label))
        print(f"convert: {label} -> {panel_dir.name}: {chunks} metrics chunk(s), {videos} video(s)")

    _stage1_merge().run_merge(out_dir, expected_tasks=expected_tasks, labels=order)
    pairs = [(out_dir / "panels" / panel_dir_name(index, label) / "videos", label) for index, label in enumerate(order)]
    stitch_panel_videos(pairs, out_dir / "side_by_side", grid_columns)
    print(f"convert: legacy files kept under {backup}")
    return order


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--labels", default="", help="comma-separated panel order (default: sorted)")
    parser.add_argument("--expected_tasks", type=int, default=0)
    parser.add_argument("--grid_columns", type=int, default=0)
    args = parser.parse_args()
    convert(
        args.out_dir.resolve(),
        labels=[label.strip() for label in args.labels.split(",") if label.strip()],
        expected_tasks=args.expected_tasks,
        grid_columns=args.grid_columns,
    )


if __name__ == "__main__":
    main()
