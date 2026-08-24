#!/usr/bin/env python3
"""Config resolver for the derived LIBERO anchor-relative dataset workflow."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "libero_anchor_relative_dataset_config.yaml"


def _find_global(start: Path) -> Path | None:
    for directory in [start.resolve(), *start.resolve().parents]:
        candidate = directory / "global_config.yaml"
        if candidate.exists():
            return candidate
    return None


def load_config(path: Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, encoding="utf-8") as file:
        cfg = yaml.safe_load(file) or {}
    global_path = _find_global(path.parent)
    if global_path is not None and global_path.resolve() != path.resolve():
        with open(global_path, encoding="utf-8") as file:
            cfg = {**(yaml.safe_load(file) or {}), **cfg}
    return cfg


def project_root(cfg: dict[str, Any]) -> Path:
    return Path(str(cfg["project_root"])).expanduser()


def dataset_root(cfg: dict[str, Any]) -> Path:
    raw = Path(str(cfg.get("anchor_relative_dataset_root") or cfg.get("dataset_root", "dataset")))
    return raw if raw.is_absolute() else project_root(cfg) / raw


def dataset_specs(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return dict(cfg.get("anchor_relative_datasets") or {})


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def shell_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    root = project_root(cfg)
    partitions = _as_list(cfg.get("train_partition", ["debug"]))
    return {
        "project_root": root,
        "lerobot_root": root / "lerobot",
        "anchor_relative_dataset_root": dataset_root(cfg),
        "anchor_relative_dataset_names": " ".join(dataset_specs(cfg)),
        "build_partition": ",".join(partitions) or "debug",
        "build_qos": str(cfg.get("train_qos", "base_qos")),
        "build_exclude_nodes": ",".join(_as_list(cfg.get("train_exclude_nodes", []))),
        "convert_gres": str(cfg.get("convert_gres", "gpu:1")),
        "convert_cpus_per_task": int(cfg.get("convert_cpus_per_task", 16)),
        "convert_mem": str(cfg.get("convert_mem", "64G")),
        "convert_time": str(cfg.get("convert_time", "24:00:00")),
        "convert_num_shards": int(cfg.get("convert_num_shards", 4)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    settings = shell_settings(load_config(args.config))
    if args.shell:
        for key, value in settings.items():
            print(f"export {key.upper()}={shlex.quote(str(value))}")
    else:
        for key, value in settings.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
