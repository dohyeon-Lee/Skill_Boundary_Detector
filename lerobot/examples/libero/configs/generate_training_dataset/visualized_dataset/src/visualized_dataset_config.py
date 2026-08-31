#!/usr/bin/env python3
"""Shared, validated YAML settings for the training-dataset inspection tools."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SRC_DIR = Path(__file__).resolve().parent
VISUALIZED_DIR = SRC_DIR.parent
DEFAULT_CONFIG_PATH = VISUALIZED_DIR / "visualized_dataset_config.yaml"
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
class DatasetSettings:
    dataset: str
    dataset_root: Path
    dataset_dir: Path


@dataclass(frozen=True)
class VisualizationSettings:
    list_tasks_only: bool
    task: str
    samples: int
    sampling: str
    seed: int
    cameras: tuple[str, ...]
    output: Path | None
    force: bool
    crf: int
    preset: str
    ffmpeg: str


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
    if global_path is None or global_path == config_path:
        return local
    with global_path.open(encoding="utf-8") as stream:
        global_config = yaml.safe_load(stream) or {}
    return {**global_config, **local}


def _bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise TypeError(f"{key} must be true or false, got {value!r}")


def _project_path(project_root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def dataset_settings(config: dict[str, Any]) -> DatasetSettings:
    project_value = str(config.get("project_root", "")).strip()
    if not project_value:
        raise ValueError("project_root is missing; set it in configs/global_config.yaml")
    project_root = Path(project_value).expanduser().resolve()

    override = str(config.get("dataset_root_override", "") or "").strip()
    root_value = override or str(config.get("dataset_root", "dataset")).strip()
    dataset_root = _project_path(project_root, root_value)

    dataset = str(config.get("dataset", "")).strip()
    if not dataset:
        raise ValueError("visualized_dataset_config.yaml: dataset must not be blank")
    dataset_dir = (dataset_root / dataset).resolve()
    try:
        dataset_dir.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError(
            f"dataset must resolve inside dataset_root {dataset_root}, got {dataset!r}"
        ) from error
    return DatasetSettings(dataset=dataset, dataset_root=dataset_root, dataset_dir=dataset_dir)


def visualization_settings(config: dict[str, Any]) -> VisualizationSettings:
    section = config.get("visualize") or {}
    if not isinstance(section, dict):
        raise TypeError("visualized_dataset_config.yaml: visualize must be a mapping")

    sampling = str(section.get("sampling", "first")).strip().lower()
    if sampling not in {"first", "random"}:
        raise ValueError(f"visualize.sampling must be first or random, got {sampling!r}")

    samples = int(section.get("samples", 1))
    if samples <= 0:
        raise ValueError(f"visualize.samples must be positive, got {samples}")

    crf = int(section.get("crf", 23))
    if not 0 <= crf <= 51:
        raise ValueError(f"visualize.crf must be between 0 and 51, got {crf}")

    preset = str(section.get("preset", "veryfast")).strip().lower()
    if preset not in ENCODING_PRESETS:
        raise ValueError(
            f"visualize.preset must be one of {sorted(ENCODING_PRESETS)}, got {preset!r}"
        )

    raw_cameras = section.get("cameras") or []
    if isinstance(raw_cameras, str):
        raw_cameras = [raw_cameras]
    if not isinstance(raw_cameras, (list, tuple)):
        raise TypeError("visualize.cameras must be a list")
    cameras = tuple(str(camera).strip() for camera in raw_cameras if str(camera).strip())

    output_value = str(section.get("output", "") or "").strip()
    output = None
    if output_value:
        output_path = Path(output_value).expanduser()
        output = (
            output_path.resolve()
            if output_path.is_absolute()
            else (VISUALIZED_DIR / output_path).resolve()
        )
        if output.suffix.lower() != ".html":
            raise ValueError(f"visualize.output must end in .html, got {output}")

    task = str(section.get("task", "")).strip()
    list_tasks_only = _bool(section.get("list_tasks_only", False), "visualize.list_tasks_only")
    if not list_tasks_only and not task:
        raise ValueError("visualize.task must not be blank unless list_tasks_only is true")

    ffmpeg = str(section.get("ffmpeg", "ffmpeg")).strip()
    if not ffmpeg:
        raise ValueError("visualize.ffmpeg must not be blank")

    return VisualizationSettings(
        list_tasks_only=list_tasks_only,
        task=task,
        samples=samples,
        sampling=sampling,
        seed=int(section.get("seed", 0)),
        cameras=cameras,
        output=output,
        force=_bool(section.get("force", False), "visualize.force"),
        crf=crf,
        preset=preset,
        ffmpeg=ffmpeg,
    )


def reject_cli_arguments() -> None:
    if len(sys.argv) > 1:
        raise SystemExit(
            f"CLI arguments are not used. Edit {DEFAULT_CONFIG_PATH} and run the Python file without arguments."
        )
