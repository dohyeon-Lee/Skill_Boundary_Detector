#!/usr/bin/env python3
"""Configuration resolver for the YAM real-dataset converter."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "yam_real_dataset_config.yaml"


def _find_global(start: Path) -> Path | None:
    for directory in [start.resolve(), *start.resolve().parents]:
        candidate = directory / "global_config.yaml"
        if candidate.exists():
            return candidate
    return None


def load_config(path: Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, encoding="utf-8") as stream:
        local = yaml.safe_load(stream) or {}
    global_path = _find_global(config_path.parent)
    if global_path is None or global_path == config_path:
        return local
    with open(global_path, encoding="utf-8") as stream:
        global_config = yaml.safe_load(stream) or {}
    return {**global_config, **local}


def project_root(config: dict[str, Any]) -> Path:
    value = config.get("project_root")
    if not value:
        raise ValueError("project_root is missing (set it in global_config.yaml)")
    return Path(str(value)).expanduser().resolve()


def resolve_project_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (project_root(config) / path).resolve()


def raw_root(config: dict[str, Any]) -> Path:
    return resolve_project_path(config, str(config.get("yam_raw_root", "dataset_YAM_raw")))


def output_root(config: dict[str, Any]) -> Path:
    return resolve_project_path(config, str(config.get("yam_dataset_root", "dataset_YAM")))


def dataset_sets(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    value = config.get("yam_sets") or {}
    if not isinstance(value, dict) or not value:
        raise ValueError("yam_sets must contain at least one named dataset")
    return {str(name): dict(entry or {}) for name, entry in value.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    values = {
        "project_root": project_root(config),
        "yam_raw_root": raw_root(config),
        "yam_output_root": output_root(config),
        "yam_set_names": " ".join(dataset_sets(config)),
    }
    if args.shell:
        for key, value in values.items():
            print(f"export {key.upper()}={shlex.quote(str(value))}")
    else:
        for key, value in values.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
