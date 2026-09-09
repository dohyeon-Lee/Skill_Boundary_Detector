#!/usr/bin/env python3
"""Resolve DrawSVG converter settings from local + global YAML config."""

from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "drawsvg_dataset_config.yaml"


def _find_global(start: Path) -> Path | None:
    for directory in [start.resolve(), *start.resolve().parents]:
        candidate = directory / "global_config.yaml"
        if candidate.is_file():
            return candidate
    return None


def load_config(path: Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        local = yaml.safe_load(stream) or {}
    if not isinstance(local, dict):
        raise ValueError(f"DrawSVG config must be a YAML mapping: {config_path}")

    global_path = _find_global(config_path.parent)
    if global_path is None or global_path == config_path:
        return local
    with global_path.open(encoding="utf-8") as stream:
        global_config = yaml.safe_load(stream) or {}
    if not isinstance(global_config, dict):
        raise ValueError(f"Global config must be a YAML mapping: {global_path}")
    return {**global_config, **local}


def project_root(config: dict[str, Any]) -> Path:
    value = config.get("project_root")
    if not value:
        raise ValueError("project_root is missing (set it in global_config.yaml)")
    return Path(str(value)).expanduser().resolve()


def resolve_project_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (project_root(config) / path).resolve()


def dataset_root(config: dict[str, Any]) -> Path:
    value = config.get("dataset_root")
    if not value:
        raise ValueError("dataset_root is missing (set it in global_config.yaml)")
    return resolve_project_path(config, str(value))


def source_root(config: dict[str, Any]) -> Path:
    """Locate the sibling DrawSVG pipeline without adding a local YAML knob."""
    override = os.environ.get("DRAWSVG_SOURCE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (project_root(config).parent / "drawsvg_pipeline" / "generated").resolve()


def include_groups(config: dict[str, Any]) -> list[str]:
    value = config.get("include_groups")
    if not isinstance(value, list) or not value:
        raise ValueError("include_groups must be a non-empty YAML list")
    groups = [str(item).strip() for item in value]
    if any(not group for group in groups):
        raise ValueError("include_groups cannot contain an empty name")
    if len(groups) != len(set(groups)):
        raise ValueError(f"include_groups contains duplicates: {groups}")
    for group in groups:
        if Path(group).name != group or group in {".", ".."}:
            raise ValueError(f"include_groups entries must be direct folder names: {group!r}")
    return groups


def output_name(config: dict[str, Any]) -> str:
    value = str(config.get("output_name", "")).strip()
    if not value:
        raise ValueError("output_name must be a non-empty folder name")
    if Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"output_name must be one folder name: {value!r}")
    return value


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def conversion_settings(config: dict[str, Any]) -> dict[str, Any]:
    root = project_root(config)
    groups = include_groups(config)
    name = output_name(config)
    partitions = ",".join(_as_list(config.get("train_partition")))
    if not partitions:
        raise ValueError("train_partition is missing/empty in global_config.yaml")

    return {
        "project_root": root,
        "python_bin": root / ".venv" / "bin" / "python",
        "drawsvg_source_root": source_root(config),
        "drawsvg_output_root": dataset_root(config),
        "drawsvg_output_name": name,
        "drawsvg_output_dir": dataset_root(config) / name,
        "drawsvg_include_groups": groups,
        "drawsvg_include_groups_json": json.dumps(groups, ensure_ascii=False),
        "drawsvg_convert_script": Path(__file__).resolve().parent / "convert_drawsvg_to_lerobot.py",
        "drawsvg_ensure_stats_script": (
            Path(__file__).resolve().parents[2]
            / "original_dataset"
            / "src"
            / "ensure_quantile_stats.py"
        ),
        # Dataset format is canonical and intentionally not exposed as local YAML knobs.
        "drawsvg_fps": 20,
        "drawsvg_image_size": 256,
        "drawsvg_vcodec": "libsvtav1",
        "drawsvg_image_writer_threads": 10,
        "drawsvg_image_writer_processes": 5,
        "drawsvg_convert_partition": partitions,
        "drawsvg_convert_qos": str(config.get("train_qos", "base_qos")),
        "drawsvg_convert_nodelist": str(config.get("train_nodelist", "") or ""),
        "drawsvg_convert_exclude_nodes": ",".join(_as_list(config.get("train_exclude_nodes"))),
        # Resource sizes follow the other full dataset converters.
        "drawsvg_convert_gres": "gpu:1",
        "drawsvg_convert_cpus_per_task": 16,
        "drawsvg_convert_mem": "64G",
        "drawsvg_convert_time": "24:00:00",
    }


def print_shell(settings: dict[str, Any]) -> None:
    for key, value in settings.items():
        if isinstance(value, list):
            value = json.dumps(value, ensure_ascii=False)
        print(f"export {key.upper()}={shlex.quote(str(value))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    settings = conversion_settings(load_config(args.config))
    if args.shell:
        print_shell(settings)
    else:
        for key, value in settings.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
