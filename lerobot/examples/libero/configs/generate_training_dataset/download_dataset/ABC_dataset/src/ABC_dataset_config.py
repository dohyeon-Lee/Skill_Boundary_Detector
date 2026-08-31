#!/usr/bin/env python3
"""Config resolver for ABC dataset utilities (--shell emits bash exports).

Mirrors filtered_dataset/src/filtered_dataset_config.py: module yaml merged over the
nearest global_config.yaml (walk-up), canonical train_* Slurm keys reused for the
CPU build job. Subset details stay in the yaml (python tools read it directly);
--shell only exports what the bash wrappers need.
"""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "ABC_dataset_config.yaml"


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


def abc_root(cfg: dict[str, Any]) -> Path:
    return project_root(cfg) / str(get_value(cfg, "abc_dataset_root", "dataset_ABC"))


def abcdl_repo(cfg: dict[str, Any]) -> Path:
    """abcdl package repo; relative values resolve against project_root (cluster-portable)."""
    raw = Path(str(get_value(cfg, "abcdl_repo", "abcdl_RLLAB"))).expanduser()
    return raw if raw.is_absolute() else project_root(cfg) / raw


def subsets(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return dict(cfg.get("abc_subsets") or {})


def build_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    proot = project_root(cfg)
    return {
        "project_root": proot,
        "lerobot_root": proot / "lerobot",
        "abc_root": abc_root(cfg),
        "abcdl_repo": str(abcdl_repo(cfg)),
        "abc_hf_repo": str(get_value(cfg, "abc_hf_repo", "XDOF/ABC-130k")),
        "abc_subset_names": " ".join(subsets(cfg).keys()),
        "abc_image_size": int(get_value(cfg, "abc_image_size", 256)),
        "abc_fps": int(get_value(cfg, "abc_fps", 30)),
        "convert_workers": int(get_value(cfg, "convert_workers", 4)),
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
