#!/usr/bin/env python3
"""Shared config helpers for generate_training_dataset scripts."""

from __future__ import annotations

import argparse
import ast
import os
import shlex
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "training_dataset_config.yaml"


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if " #" in text:
        text = text.split(" #", 1)[0].rstrip()
    low = text.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none"}:
        return None
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text.strip("\"'")


def _load_flat_yaml(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- ") and current_list_key is not None:
            out[current_list_key].append(_parse_scalar(line[2:]))
            continue
        if ":" not in line:
            current_list_key = None
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            out[key] = []
            current_list_key = key
        else:
            out[key] = _parse_scalar(value)
            current_list_key = None
    return out


def _read_yaml(config_path: Path) -> dict[str, Any]:
    try:
        import yaml

        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        return _load_flat_yaml(config_path)


def _find_global(start: Path) -> Path | None:
    """Walk up from `start` to find the nearest global_config.yaml (stops at first hit)."""
    for d in [start.resolve(), *start.resolve().parents]:
        candidate = d / "global_config.yaml"
        if candidate.exists():
            return candidate
    return None


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = _read_yaml(config_path)
    # Merge the nearest global_config.yaml as a base (module cfg keys win for roots).
    gpath = _find_global(config_path.parent)
    if gpath is not None and gpath.resolve() != config_path.resolve():
        cfg = {**_read_yaml(gpath), **cfg}
    return cfg


def get_value(cfg: dict[str, Any], key: str, default: Any = None, env: str | None = None) -> Any:
    env_key = env or key.upper()
    if env_key in os.environ:
        return os.environ[env_key]
    return cfg.get(key, default)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [v.strip() for v in str(value).split(",") if v.strip()]


def project_root(cfg: dict[str, Any]) -> Path:
    return Path(str(get_value(cfg, "project_root"))).expanduser()


def dataset_root_path(cfg: dict[str, Any]) -> Path:
    return project_root(cfg) / str(get_value(cfg, "dataset_root", "libero_dataset"))


DINO_VISUAL_BACKBONE = "dinov3_vits16"
DINO_IMAGE_MODEL_DIR = "dinov3-vits16"
DINO_N_PATCH_RAW = 196
DINO_IMAGE_SIZE = 224


def _resolve_dino_image_keys(cfg: dict[str, Any]) -> list[str]:
    """Camera selection for the frame-DINO precompute. `dino_camera` is the ergonomic knob
    (image | wrist | both, or a raw comma-separated key list); falls back to `dino_image_keys`.
    Independent of DP — DP reads its own dino_image_keys from train_skills_config."""
    cam_map = {
        "image": ["observation.images.image"],
        "third": ["observation.images.image"],
        "3rd":   ["observation.images.image"],
        "wrist": ["observation.images.wrist_image"],
        "eye_in_hand": ["observation.images.wrist_image"],
        "both":  ["observation.images.image", "observation.images.wrist_image"],
        "all":   ["observation.images.image", "observation.images.wrist_image"],
    }
    camera = str(get_value(cfg, "dino_camera", "", env="DINO_CAMERA")).strip()
    if not camera:
        return as_list(get_value(cfg, "dino_image_keys", ["observation.images.image"], env="DINO_IMAGE_KEYS"))
    if camera.lower() in cam_map:
        return cam_map[camera.lower()]
    return [k.strip() for k in camera.split(",") if k.strip()]  # raw key(s)


def dino_settings(cfg: dict[str, Any], dataset: str | None = None) -> dict[str, Any]:
    target_dataset = dataset or str(get_value(cfg, "dino_target_dataset", "libero_90", env="TARGET_DATASET"))
    root = project_root(cfg)
    dataset_root = dataset_root_path(cfg)
    patch_grid = int(get_value(cfg, "dino_patch_grid", 8, env="DINO_PATCH_GRID"))

    return {
        "project_root": root,
        "lerobot_root": root / "lerobot",
        "dataset_root": dataset_root,
        "dataset_root_name": str(get_value(cfg, "dataset_root", "libero_dataset")),
        "dataset": target_dataset,
        "dataset_dir": dataset_root / target_dataset,
        "derived_data_dir": dataset_root / f"{target_dataset}_data",
        "output_dir": dataset_root / f"{target_dataset}_DINO" / f"pg{patch_grid}",
        "visual_backbone": DINO_VISUAL_BACKBONE,
        "image_model_dir": DINO_IMAGE_MODEL_DIR,
        "image_model_path": root / "models" / DINO_IMAGE_MODEL_DIR,
        "n_patch_raw": DINO_N_PATCH_RAW,
        "image_keys": _resolve_dino_image_keys(cfg),
        "patch_grid": patch_grid,
        "image_size": DINO_IMAGE_SIZE,
        "batch_size": int(get_value(cfg, "dino_batch_size", 1024, env="DINO_BATCH_SIZE")),
        "dtype": str(get_value(cfg, "dino_dtype", "float16", env="DINO_DTYPE")),
        "wandb_project": str(get_value(cfg, "dino_wandb_project", "", env="DINO_WANDB_PROJECT")),
        # Slurm partition/qos/exclude are canonical (global_config.yaml train_*); env vars still override.
        "partitions": as_list(get_value(cfg, "train_partition", ["debug"], env="DINO_PARTITIONS")),
        "exclude_nodes": as_list(get_value(cfg, "train_exclude_nodes", [], env="DINO_EXCLUDE_NODES")),
        "qos": str(get_value(cfg, "train_qos", "big_qos", env="DINO_QOS")),
        "gpu_reserve": int(get_value(cfg, "dino_gpu_reserve", 0, env="DINO_GPU_RESERVE")),
        "gpu_max_per_node": int(get_value(cfg, "dino_gpu_max_per_node", 7, env="DINO_GPU_MAX_PER_NODE")),
        "max_workers": int(get_value(cfg, "dino_max_workers", 0, env="DINO_MAX_WORKERS")),
    }


def print_shell(settings: dict[str, Any]) -> None:
    for key, value in settings.items():
        shell_key = key.upper()
        if isinstance(value, Path):
            value = str(value)
        elif isinstance(value, list):
            value = ",".join(str(v) for v in value)
        print(f"export {shell_key}={shlex.quote(str(value))}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--shell-dino", action="store_true")
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.shell_dino:
        print_shell(dino_settings(cfg, dataset=args.dataset))
        return

    for key, value in cfg.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
