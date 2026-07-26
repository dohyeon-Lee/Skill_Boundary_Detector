#!/usr/bin/env python3
"""Config resolver for LangGap dataset utilities (--shell emits bash exports)."""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "langgap_dataset_config.yaml"


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


def langgap_root(cfg: dict[str, Any]) -> Path:
    """Final dataset root. Blank langgap_dataset_root -> global dataset_root."""
    local = str(get_value(cfg, "langgap_dataset_root", "") or "").strip()
    sub = local or str(get_value(cfg, "dataset_root", "dataset"))
    return project_root(cfg) / sub


def all_sets(cfg: dict[str, Any]) -> dict[str, str]:
    """name -> HF repo id (모든 후보; 실제 대상은 download_sets가 고른다)."""
    sets = dict(cfg.get("langgap_sets") or {})
    sets.update(cfg.get("extra_sets") or {})  # backward-compat
    return sets


def default_set_names(cfg: dict[str, Any]) -> list[str]:
    names = as_list(get_value(cfg, "download_sets")) or as_list(get_value(cfg, "default_sets"))
    if not names:
        raise SystemExit("langgap_dataset_config.yaml: download_sets 를 지정하세요 (주석 해제).")
    unknown = [n for n in names if n not in all_sets(cfg)]
    if unknown:
        raise SystemExit(f"download_sets 에 langgap_sets 에 없는 이름이 있음: {unknown}")
    return names


def build_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    proot = project_root(cfg)
    sets = all_sets(cfg)
    return {
        "project_root": proot,
        "lerobot_root": proot / "lerobot",
        "langgap_root": langgap_root(cfg),
        # name=repo pairs, space-separated, for bash iteration
        "langgap_sets": " ".join(f"{n}={r}" for n, r in sets.items()),
        "default_sets": " ".join(default_set_names(cfg)),
        # Slurm (canonical train_* keys from global_config.yaml)
        "build_partition": ",".join(as_list(get_value(cfg, "train_partition", ["debug"]))) or "debug",
        "build_qos": str(get_value(cfg, "train_qos", "base_qos")),
        "build_exclude_nodes": ",".join(as_list(get_value(cfg, "train_exclude_nodes", []))),
        "convert_gres": str(get_value(cfg, "convert_gres", "gpu:1")),
        "convert_cpus_per_task": int(get_value(cfg, "convert_cpus_per_task", 32)),
        "convert_mem": str(get_value(cfg, "convert_mem", "128G")),
        "convert_time": str(get_value(cfg, "convert_time", "48:00:00")),
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
