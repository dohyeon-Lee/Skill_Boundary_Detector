#!/usr/bin/env python3
"""Resolve ABC Sim dataset paths and scheduler settings."""

from __future__ import annotations

import argparse
import runpy
import shlex
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "ABC_sim_dataset_config.yaml"


def _find_global(start: Path) -> Path | None:
    for directory in [start.resolve(), *start.resolve().parents]:
        candidate = directory / "global_config.yaml"
        if candidate.is_file():
            return candidate
    return None


def _load_global(path: Path) -> dict[str, Any]:
    loader = next(
        directory / "src" / "global_config_loader.py"
        for directory in Path(__file__).resolve().parents
        if (directory / "src" / "global_config_loader.py").is_file()
    )
    return runpy.run_path(str(loader))["load_global_config"](path)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    global_path = _find_global(path.parent)
    if global_path is not None and global_path.resolve() != path.resolve():
        config = {**_load_global(global_path), **config}
    return config


def get_value(config: dict[str, Any], key: str, default: Any = None) -> Any:
    return config.get(key, default)


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def project_root(config: dict[str, Any]) -> Path:
    return Path(str(config["project_root"])).expanduser()


def dataset_root(config: dict[str, Any]) -> Path:
    return project_root(config) / str(get_value(config, "abc_sim_dataset_root", "dataset_ABC"))


def subsets(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return dict(config.get("abc_sim_subsets") or {})


def selected_subsets(config: dict[str, Any], only: str = "") -> dict[str, dict[str, Any]]:
    configured = subsets(config)
    names = {part for part in only.replace(",", " ").split() if part}
    if not names:
        return configured
    unknown = names - set(configured)
    if unknown:
        raise ValueError(f"unknown subset(s) {sorted(unknown)}; configured: {sorted(configured)}")
    return {name: spec for name, spec in configured.items() if name in names}


def build_settings(config: dict[str, Any]) -> dict[str, Any]:
    root = dataset_root(config)
    return {
        "project_root": project_root(config),
        "abc_sim_root": root,
        "abc_sim_manifest_url": str(get_value(config, "abc_sim_manifest_url")),
        "abc_sim_subset_names": " ".join(subsets(config)),
        "build_partition": ",".join(as_list(get_value(config, "train_partition", ["debug"]))) or "debug",
        "build_qos": str(get_value(config, "train_qos", "base_qos")),
        "build_exclude_nodes": ",".join(as_list(get_value(config, "train_exclude_nodes", []))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    settings = build_settings(load_config(args.config))
    if args.shell:
        for key, value in settings.items():
            print(f"export {key.upper()}={shlex.quote(str(value))}")
    else:
        for key, value in settings.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
