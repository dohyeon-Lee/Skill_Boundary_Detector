#!/usr/bin/env python3
"""Config resolver for FILTERED LIBERO dataset utilities (--shell emits bash exports)."""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "filtered_dataset_config.yaml"


def _find_global(start: Path) -> Path | None:
    """Walk up from `start` to find the nearest global_config.yaml (stops at first hit)."""
    for d in [start.resolve(), *start.resolve().parents]:
        candidate = d / "global_config.yaml"
        if candidate.exists():
            return candidate
    return None


def load_config(path: Path) -> dict[str, Any]:
    config_path = Path(path)
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    gpath = _find_global(config_path.parent)
    if gpath is not None and gpath.resolve() != config_path.resolve():
        with open(gpath, encoding="utf-8") as f:
            gcfg = yaml.safe_load(f) or {}
        cfg = {**gcfg, **cfg}
    return cfg


def get_value(cfg: dict[str, Any], key: str, default: Any = None, *, env: str | None = None) -> Any:
    if env and env in os.environ:
        return os.environ[env]
    return cfg.get(key, default)


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    return [p.strip() for p in text.split(",") if p.strip()] if text else []


def project_root(cfg: dict[str, Any]) -> Path:
    return Path(str(get_value(cfg, "project_root"))).expanduser()


def filtered_root(cfg: dict[str, Any]) -> Path:
    return project_root(cfg) / str(get_value(cfg, "filtered_dataset_root", "dataset_filtered"))


def build_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    proot = project_root(cfg)
    suites = cfg.get("filtered_suites") or {}
    return {
        "project_root": proot,
        "lerobot_root": proot / "lerobot",
        "filtered_root": filtered_root(cfg),
        # name=repo pairs, space-separated, for bash iteration
        "filtered_suites": " ".join(f"{n}={r}" for n, r in suites.items()),
        "data_file_size_in_mb": int(get_value(cfg, "data_file_size_in_mb", 100)),
        "video_file_size_in_mb": int(get_value(cfg, "video_file_size_in_mb", 200)),
        # Slurm (canonical train_* keys from global_config.yaml; CPU-only job)
        "build_partition": ",".join(as_list(get_value(cfg, "train_partition", ["debug"]))) or "debug",
        "build_qos": str(get_value(cfg, "train_qos", "base_qos")),
        "build_exclude_nodes": ",".join(as_list(get_value(cfg, "train_exclude_nodes", []))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--shell", action="store_true", help="Emit bash export lines.")
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
