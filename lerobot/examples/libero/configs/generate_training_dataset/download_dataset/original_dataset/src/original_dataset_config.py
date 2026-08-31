#!/usr/bin/env python3
"""Config resolver for original LIBERO dataset utilities."""

from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "original_dataset_config.yaml"


def _find_global(start: Path) -> Path | None:
    """Walk up from `start` to find the nearest global_config.yaml (stops at first hit)."""
    for d in [start.resolve(), *start.resolve().parents]:
        candidate = d / "global_config.yaml"
        if candidate.exists():
            return candidate
    return None


def load_config(path: Path) -> dict[str, Any]:
    config_path = Path(path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # Merge the nearest global_config.yaml as a base (module cfg keys win for roots).
    gpath = _find_global(config_path.parent)
    if gpath is not None and gpath.resolve() != config_path.resolve():
        with open(gpath, "r", encoding="utf-8") as f:
            gcfg = yaml.safe_load(f) or {}
        cfg = {**gcfg, **cfg}
    return cfg


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def get_value(cfg: dict[str, Any], key: str, default: Any = None, *, env: str | None = None) -> Any:
    if env and env in os.environ:
        return os.environ[env]
    return cfg.get(key, default)


def shell_value(value: Any) -> str:
    if isinstance(value, bool):
        value = "true" if value else "false"
    elif isinstance(value, (list, tuple, dict)):
        value = json.dumps(value)
    return shlex.quote(str(value))


def print_shell(settings: dict[str, Any]) -> None:
    for key, value in settings.items():
        print(f"export {key.upper()}={shell_value(value)}")


def optional_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def resolve_path(project_root: Path, value: Any, default: Path | str) -> Path:
    """Resolve module paths relative to the project selected by global_config.yaml."""
    raw = value if value not in (None, "", "null") else default
    path = Path(str(raw)).expanduser()
    return path if path.is_absolute() else project_root / path


def build_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(str(get_value(cfg, "project_root", env="PROJECT_ROOT"))).expanduser()
    dataset_root = resolve_path(
        project_root,
        get_value(cfg, "dataset_root", "dataset"),
        "dataset",
    )
    original_dataset_root = resolve_path(
        project_root,
        get_value(
            cfg,
            "original_dataset_root",
            "libero_original_dataset",
            env="LIBERO_ORIGINAL_DATASET_DIR",
        ),
        "libero_original_dataset",
    )
    download_tools_root = resolve_path(
        project_root,
        get_value(cfg, "original_dataset_tools_root", "tools", env="LIBERO_DOWNLOAD_TOOLS_DIR"),
        "tools",
    )
    suite = str(get_value(cfg, "convert_suite", "libero_90", env="CONVERT_SUITE"))
    source_root = resolve_path(
        project_root,
        get_value(cfg, "convert_source_root", "", env="CONVERT_SOURCE_ROOT"),
        original_dataset_root,
    )
    output_root = resolve_path(
        project_root,
        get_value(cfg, "convert_output_root", "", env="CONVERT_OUTPUT_ROOT"),
        dataset_root,
    )
    schema_reference = resolve_path(
        project_root,
        get_value(cfg, "convert_schema_reference", "", env="CONVERT_SCHEMA_REFERENCE"),
        "dataset_filtered/libero_90_full_full",
    )
    output_name = str(get_value(cfg, "convert_output_name", f"{suite}_full_full", env="CONVERT_OUTPUT_NAME"))

    # Slurm partition/qos/nodelist/exclude are canonical (global_config.yaml train_*); env vars still override.
    partitions = ",".join(as_list(get_value(cfg, "train_partition", ["debug"]))) or "debug"
    exclude = ",".join(as_list(get_value(cfg, "train_exclude_nodes", [])))

    return {
        "project_root": project_root,
        "python_bin": get_value(
            cfg,
            "python_bin",
            project_root / ".venv" / "bin" / "python",
            env="PYTHON_BIN",
        ),
        "config_path": DEFAULT_CONFIG_PATH,
        "convert_script": DEFAULT_CONFIG_PATH.parent / "src" / "convert_original_libero_to_lerobot.py",
        "ensure_stats_script": DEFAULT_CONFIG_PATH.parent / "src" / "ensure_quantile_stats.py",
        "original_dataset_root": original_dataset_root,
        "original_dataset_tools_root": download_tools_root,
        "original_datasets": str(get_value(
            cfg,
            "original_datasets",
            "libero_100",
            env="LIBERO_ORIGINAL_DATASETS",
        )),
        "convert_suite": suite,
        "convert_source_root": source_root,
        "convert_output_root": output_root,
        "convert_output_name": output_name,
        "convert_overwrite": as_bool(get_value(cfg, "convert_overwrite", False, env="CONVERT_OVERWRITE")),
        "convert_schema_reference": schema_reference,
        "convert_image_size": int(get_value(cfg, "convert_image_size", 256, env="CONVERT_IMAGE_SIZE")),
        "convert_vcodec": str(get_value(cfg, "convert_vcodec", "libsvtav1", env="CONVERT_VCODEC")),
        "convert_encoder_threads": optional_string(get_value(
            cfg, "convert_encoder_threads", "", env="CONVERT_ENCODER_THREADS"
        )),
        "convert_batch_encoding_size": int(get_value(
            cfg, "convert_batch_encoding_size", 1, env="CONVERT_BATCH_ENCODING_SIZE"
        )),
        "convert_streaming_encoding": as_bool(get_value(
            cfg, "convert_streaming_encoding", False, env="CONVERT_STREAMING_ENCODING"
        )),
        "convert_encoder_queue_maxsize": int(get_value(
            cfg, "convert_encoder_queue_maxsize", 30, env="CONVERT_ENCODER_QUEUE_MAXSIZE"
        )),
        "convert_image_writer_threads": int(get_value(
            cfg, "convert_image_writer_threads", 10, env="CONVERT_IMAGE_WRITER_THREADS"
        )),
        "convert_image_writer_processes": int(get_value(
            cfg, "convert_image_writer_processes", 5, env="CONVERT_IMAGE_WRITER_PROCESSES"
        )),
        "convert_max_tasks": optional_string(get_value(cfg, "convert_max_tasks", "", env="CONVERT_MAX_TASKS")),
        "convert_max_episodes_per_task": optional_string(get_value(
            cfg, "convert_max_episodes_per_task", "", env="CONVERT_MAX_EPISODES_PER_TASK"
        )),
        "convert_partition": partitions,
        "convert_qos": str(get_value(cfg, "train_qos", "base_qos", env="CONVERT_QOS")),
        "convert_gres": str(get_value(cfg, "convert_gres", "gpu:1", env="CONVERT_GRES")),
        "convert_cpus_per_task": int(get_value(cfg, "convert_cpus_per_task", 16, env="CONVERT_CPUS_PER_TASK")),
        "convert_mem": str(get_value(cfg, "convert_mem", "128G", env="CONVERT_MEM")),
        "convert_time": str(get_value(cfg, "convert_time", "48:00:00", env="CONVERT_TIME")),
        "convert_nodelist": str(get_value(cfg, "train_nodelist", "", env="CONVERT_NODELIST")),
        "convert_exclude_nodes": exclude,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    settings = build_settings(load_config(args.config))
    settings["config_path"] = args.config
    if args.shell:
        print_shell(settings)
        return
    for key, value in settings.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
