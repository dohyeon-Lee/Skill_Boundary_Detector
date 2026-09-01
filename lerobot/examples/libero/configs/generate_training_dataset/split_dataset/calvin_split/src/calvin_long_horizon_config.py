#!/usr/bin/env python3
"""Validated YAML settings for CALVIN long-horizon candidate discovery."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SRC_DIR = Path(__file__).resolve().parent
CALVIN_SPLIT_DIR = SRC_DIR.parent
DEFAULT_CONFIG_PATH = CALVIN_SPLIT_DIR / "calvin_long_horizon_config.yaml"
ENCODING_PRESETS = {
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
}


@dataclass(frozen=True)
class SourceSettings:
    source_root: Path
    source_dir: Path
    annotation_folder: str
    fps: int


@dataclass(frozen=True)
class SearchSettings:
    sequence_steps: tuple[int, ...]
    min_total_frames: int
    max_total_frames: int
    max_gap_frames: int
    min_occurrences: tuple[int, ...]


@dataclass(frozen=True)
class VisualizationSettings:
    enabled: bool
    camera: str
    samples_per_candidate: int
    max_candidates: int | None
    sampling: str
    seed: int
    image_size: int | str
    output_dir: Path
    force: bool
    ffmpeg: str
    crf: int
    preset: str


@dataclass(frozen=True)
class Settings:
    source: SourceSettings
    search: SearchSettings
    visualization: VisualizationSettings


def _find_global(start: Path) -> Path | None:
    for directory in [start.resolve(), *start.resolve().parents]:
        candidate = directory / "global_config.yaml"
        if candidate.is_file():
            return candidate
    return None


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        local = yaml.safe_load(stream) or {}
    global_path = _find_global(config_path.parent)
    if global_path is None:
        return local
    with global_path.open(encoding="utf-8") as stream:
        global_config = yaml.safe_load(stream) or {}
    return {**global_config, **local}


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key) or {}
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a mapping")
    return value


def _bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise TypeError(f"{key} must be true or false, got {value!r}")


def _positive_int(value: Any, key: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{key} must be positive, got {parsed}")
    return parsed


def _optional_positive_int(value: Any, key: str) -> int | None:
    text = str(value or "").strip()
    return None if not text else _positive_int(text, key)


def _project_path(project_root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def settings(config: dict[str, Any]) -> Settings:
    project_value = str(config.get("project_root", "")).strip()
    if not project_value:
        raise ValueError("project_root is missing; set it in configs/global_config.yaml")
    project_root = Path(project_value).expanduser().resolve()

    source = _mapping(config, "source")
    source_dataset_root = _project_path(
        project_root, source.get("dataset_root", config.get("dataset_root", "dataset_calvin"))
    )
    raw_subdir = str(source.get("raw_subdir", "_calvin_raw")).strip()
    extracted_dir = str(source.get("extracted_dir", "task_D_D")).strip()
    split = str(source.get("split", "training")).strip().lower()
    if split not in {"training", "validation"}:
        raise ValueError("source.split must be training or validation")
    for key, value in (("source.raw_subdir", raw_subdir), ("source.extracted_dir", extracted_dir)):
        if not value or Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError(f"{key} must be a non-empty relative path without '..'")
    annotation_folder = str(source.get("annotation_folder", "lang_annotations")).strip()
    if Path(annotation_folder).name != annotation_folder or annotation_folder in {"", ".", ".."}:
        raise ValueError("source.annotation_folder must be one directory name")
    source_root = (source_dataset_root / raw_subdir / extracted_dir).resolve()
    source_settings = SourceSettings(
        source_root=source_root,
        source_dir=source_root / split,
        annotation_folder=annotation_folder,
        fps=_positive_int(source.get("fps", 30), "source.fps"),
    )

    search = _mapping(config, "search")
    raw_steps = search.get("sequence_steps", [2])
    if not isinstance(raw_steps, (list, tuple)):
        raw_steps = [raw_steps]
    sequence_steps = tuple(sorted({_positive_int(value, "search.sequence_steps") for value in raw_steps}))
    if not sequence_steps or any(value < 2 for value in sequence_steps):
        raise ValueError("search.sequence_steps must contain integers >= 2")
    min_frames = _positive_int(search.get("min_total_frames", 120), "search.min_total_frames")
    max_frames = _positive_int(search.get("max_total_frames", 450), "search.max_total_frames")
    if max_frames < min_frames:
        raise ValueError("search.max_total_frames must be >= search.min_total_frames")
    max_gap = int(search.get("max_gap_frames", 90))
    if max_gap < 0:
        raise ValueError("search.max_gap_frames must be >= 0")
    raw_min_occurrences = search.get("min_occurrences", [5])
    if not isinstance(raw_min_occurrences, (list, tuple)):
        raw_min_occurrences = [raw_min_occurrences]
    min_occurrences = tuple(
        sorted(
            {
                _positive_int(value, "search.min_occurrences")
                for value in raw_min_occurrences
            }
        )
    )
    if not min_occurrences:
        raise ValueError("search.min_occurrences must not be empty")
    search_settings = SearchSettings(
        sequence_steps=sequence_steps,
        min_total_frames=min_frames,
        max_total_frames=max_frames,
        max_gap_frames=max_gap,
        min_occurrences=min_occurrences,
    )

    visual = _mapping(config, "visualization")
    camera = str(visual.get("camera", "rgb_static")).strip()
    if camera not in {"rgb_static", "rgb_gripper"}:
        raise ValueError("visualization.camera must be rgb_static or rgb_gripper")
    sampling = str(visual.get("sampling", "first")).strip().lower()
    if sampling not in {"first", "random"}:
        raise ValueError("visualization.sampling must be first or random")
    raw_size = str(visual.get("image_size", "native")).strip().lower()
    image_size: int | str
    if raw_size == "native":
        image_size = "native"
    else:
        image_size = _positive_int(raw_size, "visualization.image_size")
    output_value = str(visual.get("output_dir", "outputs/calvin_long_horizon")).strip()
    if not output_value:
        raise ValueError("visualization.output_dir must not be blank")
    output_path = Path(output_value).expanduser()
    output_dir = (
        output_path.resolve()
        if output_path.is_absolute()
        else (CALVIN_SPLIT_DIR / output_path).resolve()
    )
    crf = int(visual.get("crf", 23))
    if not 0 <= crf <= 51:
        raise ValueError("visualization.crf must be between 0 and 51")
    preset = str(visual.get("preset", "veryfast")).strip().lower()
    if preset not in ENCODING_PRESETS:
        raise ValueError(f"visualization.preset must be one of {sorted(ENCODING_PRESETS)}")
    ffmpeg = str(visual.get("ffmpeg", "ffmpeg")).strip()
    if not ffmpeg:
        raise ValueError("visualization.ffmpeg must not be blank")
    visualization_settings = VisualizationSettings(
        enabled=_bool(visual.get("enabled", True), "visualization.enabled"),
        camera=camera,
        samples_per_candidate=_positive_int(
            visual.get("samples_per_candidate", 3), "visualization.samples_per_candidate"
        ),
        max_candidates=_optional_positive_int(
            visual.get("max_candidates", 30), "visualization.max_candidates"
        ),
        sampling=sampling,
        seed=int(visual.get("seed", 0)),
        image_size=image_size,
        output_dir=output_dir,
        force=_bool(visual.get("force", False), "visualization.force"),
        ffmpeg=ffmpeg,
        crf=crf,
        preset=preset,
    )
    return Settings(
        source=source_settings,
        search=search_settings,
        visualization=visualization_settings,
    )


def reject_cli_arguments() -> None:
    if len(sys.argv) > 1:
        raise SystemExit(
            f"CLI arguments are not used. Edit {DEFAULT_CONFIG_PATH} and run the Python file."
        )
