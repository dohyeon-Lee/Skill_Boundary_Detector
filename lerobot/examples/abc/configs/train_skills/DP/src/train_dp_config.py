#!/usr/bin/env python3
"""Resolve the ABC Diffusion Policy config into portable shell settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shlex
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "dp_config.yaml"
ABC_NATIVE_FORMAT = "sbd_abc_sim_native/v1"
YAM_JOINT_NAMES = [
    *(f"left_joint_{index}" for index in range(6)),
    "left_gripper",
    *(f"right_joint_{index}" for index in range(6)),
    "right_gripper",
]
UNET_DIMS = {
    "small": [128, 256, 512],
    "base": [256, 512, 1024],
    "large": [512, 1024, 2048],
}
ACTION_MODES = {"dataset", "anchor_relative", "delta_joint"}


def _find_global(start: Path) -> Path | None:
    for directory in [start.resolve(), *start.resolve().parents]:
        candidate = directory / "global_config.yaml"
        if candidate.is_file():
            return candidate
    return None


def load_config(path: Path) -> dict[str, Any]:
    path = Path(path)
    local = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    global_path = _find_global(path.parent)
    if global_path is None or global_path.resolve() == path.resolve():
        return local
    # A submission snapshot contains a private global_config.yaml but not the
    # Python loader. Resolve code from the checked-out ABC config tree.
    loader = next(
        directory / "src" / "global_config_loader.py"
        for directory in Path(__file__).resolve().parents
        if (directory / "src" / "global_config_loader.py").is_file()
    )
    global_config = runpy.run_path(str(loader))["load_global_config"](global_path)
    return {**global_config, **local}


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = yaml.safe_load(text)
        if not isinstance(parsed, list):
            raise ValueError(f"dp_tasks must be a task, group, or list, got {value!r}.")
        return as_list(parsed)
    return [part.strip() for part in text.split(",") if part.strip()]


def task_selection(value: Any) -> list[str]:
    tasks = as_list(value)
    if not tasks or tasks == ["all"]:
        return []
    if any(task.lower() == "all" for task in tasks):
        raise ValueError("dp_tasks cannot mix 'all' with explicit selectors.")
    if len(tasks) != len(set(tasks)):
        raise ValueError("dp_tasks contains duplicate selectors.")
    return tasks


def configured_task_groups(cfg: dict[str, Any]) -> dict[str, list[str]]:
    raw = cfg.get("abc_sim_task_groups", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("abc_sim_task_groups must be a mapping.")
    groups: dict[str, list[str]] = {}
    for raw_name, raw_tasks in raw.items():
        name = str(raw_name).strip()
        tasks = as_list(raw_tasks)
        if not name or not tasks or len(tasks) != len(set(tasks)):
            raise ValueError(f"Invalid ABC task group {raw_name!r}.")
        groups[name] = tasks
    return groups


def expand_task_selectors(selectors: list[str], groups: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    tasks: list[str] = []
    used_groups: list[str] = []
    for selector in selectors:
        if selector in groups:
            used_groups.append(selector)
            candidates = groups[selector]
        elif selector.startswith("group_"):
            raise ValueError(f"Unknown ABC Sim task group {selector!r}; choose one of {sorted(groups)}.")
        else:
            candidates = [selector]
        for task in candidates:
            if task not in tasks:
                tasks.append(task)
    return tasks, used_groups


def _task_suffix(tasks: list[str], selectors: list[str], groups: dict[str, list[str]]) -> str:
    if not tasks:
        return "all"
    if len(selectors) == 1 and selectors[0] in groups:
        return selectors[0]
    if len(tasks) == 1:
        return f"task-{tasks[0]}"
    digest = hashlib.sha256("\0".join(sorted(tasks)).encode()).hexdigest()[:8]
    return f"tasks{len(tasks)}-{digest}"


def _native_task_ids(dataset_dir: Path) -> list[str]:
    marker_path = dataset_dir / "native_dataset.json"
    if not marker_path.is_file():
        raise FileNotFoundError(f"ABC native marker not found: {marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("format") != ABC_NATIVE_FORMAT:
        raise ValueError(f"Unsupported ABC dataset format in {marker_path}.")
    hashes = marker.get("task_sha256") or {}
    if hashes:
        return sorted(str(task) for task in hashes)
    return sorted(
        {
            str(source.get("path", "")).split("/", 1)[0]
            for source in marker.get("sources", [])
            if isinstance(source, dict) and str(source.get("path", "")).strip()
        }
    )


def _validate_dataset(dataset_dir: Path, tasks: list[str]) -> dict[str, Any]:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"ABC metadata not found: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if int(info.get("fps", 0)) != 30:
        raise ValueError(f"ABC DP expects 30 Hz data, got fps={info.get('fps')!r}.")
    for key in ("observation.state", "action"):
        feature = (info.get("features") or {}).get(key) or {}
        if feature.get("shape") != [14] or feature.get("names") != YAM_JOINT_NAMES:
            raise ValueError(f"ABC DP expects {key} to be the canonical 14D absolute YAM joint vector.")
    available = _native_task_ids(dataset_dir)
    missing = sorted(set(tasks) - set(available))
    if missing:
        raise ValueError(f"Unknown dp_tasks {missing}; available tasks: {available}")
    return info


def _resolve_path(project_root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else project_root / path


def build_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    # Keep the old boolean spellings retired: the explicit action-mode option
    # makes checkpoint/run naming and downstream SBD contracts unambiguous.
    retired = sorted(
        key for key in ("dp_proprio_grounding", "dp_relative", "dp_use_relative_actions") if key in cfg
    )
    if retired:
        raise ValueError(
            f"ABC DP does not expose legacy grounding/relative booleans: {retired}. "
            "Use dp_action_mode: dataset|anchor_relative|delta_joint."
        )

    project_root = Path(str(cfg["project_root"])).expanduser().resolve()
    dataset_root = str(cfg.get("dp_dataset_root", cfg.get("dataset_root", "dataset_ABC")))
    dataset_name = str(os.environ.get("DP_DATASET", cfg.get("dp_dataset", "abc_sim_all")))
    dataset_dir = project_root / dataset_root / dataset_name

    selectors = task_selection(os.environ.get("DP_TASKS", cfg.get("dp_tasks", "all")))
    groups = configured_task_groups(cfg)
    tasks, used_groups = expand_task_selectors(selectors, groups)
    info = _validate_dataset(dataset_dir, tasks)

    action_mode = str(cfg.get("dp_action_mode", "dataset")).strip().lower()
    if action_mode not in ACTION_MODES:
        raise ValueError(f"dp_action_mode must be one of {sorted(ACTION_MODES)}, got {action_mode!r}.")
    relative_exclude_joints = as_list(cfg.get("dp_relative_exclude_joints", ["gripper"]))
    if action_mode != "dataset" and relative_exclude_joints != ["gripper"]:
        raise ValueError(
            f"ABC {action_mode} DP currently requires dp_relative_exclude_joints: [gripper] "
            "to keep both gripper dimensions absolute while only the 12 arm joints become relative."
        )

    n_obs_steps = int(cfg.get("dp_n_obs_steps", 30))
    future_horizon = int(cfg.get("dp_future_action_horizon", 36))
    if n_obs_steps != 30 or future_horizon != 36:
        raise ValueError("ABC DP currently fixes 30 Hz horizons to obs=30 and future=36 steps.")
    history_encoder = str(cfg.get("dp_history_encoder", "transformer"))
    history_dim = int(cfg.get("dp_history_encoder_dim", 128))
    if history_encoder != "transformer" or history_dim != 128:
        raise ValueError("ABC DP currently uses the 128D Transformer history encoder.")
    unet_size = str(cfg.get("dp_unet_size", "small"))
    if unet_size not in UNET_DIMS:
        raise ValueError(f"dp_unet_size must be one of {sorted(UNET_DIMS)}.")
    if unet_size != "small":
        raise ValueError("The first ABC DP run is fixed to dp_unet_size: small.")
    lr_scheduler = str(cfg.get("dp_lr_scheduler", "constant"))
    if lr_scheduler != "constant":
        raise ValueError("ABC DP is fixed to a constant learning rate.")
    learning_rate = float(cfg.get("dp_lr", 1e-4))
    if learning_rate != 1e-4:
        raise ValueError("ABC DP is fixed to learning rate 1e-4.")
    if as_bool(cfg.get("dp_amp", False)):
        raise ValueError("ABC DP is fixed to FP32 (dp_amp: false).")

    steps = int(cfg.get("dp_steps", 100000))
    save_freq = int(cfg.get("dp_save_freq", 10000))
    if steps != 100000 or save_freq != 10000:
        raise ValueError("ABC DP is configured for 100000 steps with 10000-step checkpoints.")

    task_suffix = _task_suffix(tasks, selectors, groups)
    action_tag = {
        "dataset": "absolute",
        "anchor_relative": "anchor-relative-joint",
        "delta_joint": "delta-joint",
    }[action_mode]
    run_name = (
        f"{dataset_name}_{task_suffix}_{action_tag}_future_only_"
        f"obs{n_obs_steps}_future{future_horizon}_transformer_constLR_small"
    )
    outputs_root = project_root / str(cfg.get("outputs_root", "outputs_ABC")) / "DP"
    output_dir = outputs_root / run_name
    base_config = _resolve_path(project_root, cfg.get("dp_base_config"))

    partition = ",".join(as_list(cfg.get("train_partition", ["debug"]))) or "debug"
    exclude = ",".join(as_list(cfg.get("train_exclude_nodes", [])))
    resolved_tasks = json.dumps(tasks)
    return {
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        "config_path": str(cfg.get("_config_path", DEFAULT_CONFIG_PATH)),
        "train_dp": as_bool(cfg.get("train_DP", True)),
        "dp_dataset": dataset_name,
        "dp_dataset_dir": dataset_dir,
        "dp_task_selectors": selectors,
        "dp_tasks": tasks,
        "dp_used_task_groups": used_groups,
        "dp_task_filter": bool(tasks),
        "dp_resolved_tasks": resolved_tasks,
        "dp_tasks_label": ",".join(selectors) if selectors else "all",
        "dp_resolved_tasks_label": ",".join(tasks) if tasks else "all",
        "dp_task_count": len(tasks) if tasks else int(info.get("total_tasks", 0)),
        "dp_action_mode": action_mode,
        "dp_use_relative_actions": action_mode == "anchor_relative",
        "dp_use_delta_joint_actions": action_mode == "delta_joint",
        "dp_relative_exclude_joints": relative_exclude_joints,
        "dp_relative_exclude_joints_csv": ",".join(relative_exclude_joints),
        "dp_run_name": run_name,
        "dp_output_dir": output_dir,
        "dp_final_checkpoint": output_dir / "checkpoints/100000/pretrained_model",
        "dp_base_config": base_config,
        "dp_n_obs_steps": n_obs_steps,
        "dp_future_action_horizon": future_horizon,
        "dp_trajectory_horizon": future_horizon,
        "dp_n_action_steps": future_horizon,
        "dp_drop_n_last_frames": future_horizon - 1,
        "dp_history_encoder": history_encoder,
        "dp_history_encoder_dim": history_dim,
        "dp_unet_size": unet_size,
        "dp_down_dims_arg": json.dumps(UNET_DIMS[unet_size]),
        "dp_lr_scheduler": lr_scheduler,
        "dp_lr": learning_rate,
        "dp_amp": False,
        "dp_steps": steps,
        "dp_batch_size": int(cfg.get("dp_batch_size", 64)),
        "dp_num_workers": int(cfg.get("dp_num_workers", 8)),
        "dp_seed": int(cfg.get("dp_seed", 42)),
        "dp_log_freq": int(cfg.get("dp_log_freq", 200)),
        "dp_save_freq": save_freq,
        "dp_wandb_enable": as_bool(cfg.get("dp_wandb_enable", True)),
        "dp_wandb_project": str(cfg.get("dp_wandb_project", "DP_ABC")),
        "dp_partition": partition,
        "dp_nodelist": str(cfg.get("train_nodelist", "")),
        "dp_exclude_nodes": exclude,
        "dp_qos": str(cfg.get("train_qos", "base_qos")),
        "dp_gres": str(cfg.get("dp_gres", "gpu:1")),
        "dp_cpus_per_task": int(cfg.get("dp_cpus_per_task", 8)),
        "dp_mem": str(cfg.get("dp_mem", "64G")),
        "dp_time": str(cfg.get("dp_time", "48:00:00")),
    }


def _shell_value(value: Any) -> str:
    if isinstance(value, bool):
        value = "true" if value else "false"
    elif isinstance(value, (list, tuple, dict)):
        value = json.dumps(value)
    return shlex.quote(str(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["_config_path"] = str(args.config.resolve())
    settings = build_settings(cfg)
    if args.shell:
        for key, value in settings.items():
            print(f"export {key.upper()}={_shell_value(value)}")
    else:
        print(json.dumps(settings, indent=2, default=str))


if __name__ == "__main__":
    main()
