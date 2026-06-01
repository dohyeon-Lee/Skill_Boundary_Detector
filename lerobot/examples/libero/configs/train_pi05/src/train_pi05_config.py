#!/usr/bin/env python3
"""Config resolver for configs/train_pi05.

Emits shell exports for pi05 PT/FT/eval sbatch files without depending on the
legacy configs/data_generation/pipeline_config.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "train_pi05_config.yaml"


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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


def run_name(prefix: str, dataset: str, batch_size: int, exp: str) -> str:
    name = f"{prefix}_{dataset}_pi05_batch{batch_size}"
    if exp:
        name = f"{name}_{exp}"
    return name


def slurm_settings(cfg: dict[str, Any], prefix: str, *, cpus: int, mem: str, time: str, qos: str) -> dict[str, Any]:
    return {
        f"{prefix}_partition": ",".join(as_list(get_value(cfg, f"{prefix}_partition", ["debug"]))) or "debug",
        f"{prefix}_nodelist": str(get_value(cfg, f"{prefix}_nodelist", "")),
        f"{prefix}_exclude_nodes": ",".join(as_list(get_value(cfg, f"{prefix}_exclude_nodes", []))),
        f"{prefix}_qos": str(get_value(cfg, f"{prefix}_qos", qos)),
        f"{prefix}_gres": str(get_value(cfg, f"{prefix}_gres", "gpu:1")),
        f"{prefix}_cpus_per_task": int(get_value(cfg, f"{prefix}_cpus_per_task", cpus)),
        f"{prefix}_mem": str(get_value(cfg, f"{prefix}_mem", mem)),
        f"{prefix}_time": str(get_value(cfg, f"{prefix}_time", time)),
    }


def build_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    lerobot_root = project_root / "lerobot"
    pi05_outputs_root = project_root / str(get_value(cfg, "pi05_outputs_root", "pi05_outputs"))

    pt_dataset = str(get_value(cfg, "pt_dataset", "libero_90", env="PT_DATASET"))
    pt_dataset_root = str(get_value(cfg, "pt_dataset_root", get_value(cfg, "dataset_root", "libero_dataset"), env="PT_DATASET_ROOT"))
    pt_batch_size = int(get_value(cfg, "pt_batch_size", 32, env="PT_BATCH_SIZE"))
    pt_num_gpus = int(get_value(cfg, "pt_num_gpus", 1, env="PT_NUM_GPUS"))
    pt_exp = str(get_value(cfg, "pt_exp", "exp1", env="PT_EXP")).strip()
    pt_run_name = run_name("PT", pt_dataset, pt_batch_size, pt_exp)

    ft_dataset = str(get_value(cfg, "ft_dataset", "libero_10_op1_10", env="FT_DATASET"))
    ft_dataset_root = str(get_value(cfg, "ft_dataset_root", get_value(cfg, "dataset_root", "libero_dataset"), env="FT_DATASET_ROOT"))
    ft_batch_size = int(get_value(cfg, "ft_batch_size", 32, env="FT_BATCH_SIZE"))
    ft_exp = str(get_value(cfg, "ft_exp", "exp2", env="FT_EXP")).strip()
    ft_pre_dataset = str(get_value(cfg, "ft_pretrained_dataset", pt_dataset, env="FT_PRETRAINED_DATASET"))
    ft_pre_batch = int(get_value(cfg, "ft_pretrained_batch_size", pt_batch_size, env="FT_PRETRAINED_BATCH_SIZE"))
    ft_pre_exp = str(get_value(cfg, "ft_pretrained_exp", pt_exp, env="FT_PRETRAINED_EXP")).strip()
    ft_pre_ckpt = str(get_value(cfg, "ft_pretrained_checkpoint", "050000", env="FT_PRETRAINED_CHECKPOINT"))
    ft_pre_run_name = run_name("PT", ft_pre_dataset, ft_pre_batch, ft_pre_exp)
    ft_run_name = f"FT_{ft_pre_ckpt}PT_{ft_dataset}_pi05_batch{ft_batch_size}"
    if ft_exp:
        ft_run_name = f"{ft_run_name}_{ft_exp}"

    eval_stage = str(get_value(cfg, "eval_stage", "FT", env="EVAL_STAGE")).upper()
    eval_checkpoint = str(get_value(cfg, "eval_checkpoint", "001000", env="CHECKPOINT"))
    eval_model = os.environ.get("MODEL", "")
    if not eval_model:
        eval_model = ft_run_name if eval_stage == "FT" else pt_run_name
    eval_target_task = str(get_value(cfg, "eval_target_task", "libero_90", env="TARGET_TASK"))
    eval_wandb_run = str(os.environ.get("WANDB_RUN_NAME", f"{eval_model}_{eval_checkpoint}_{eval_target_task}"))

    settings = {
        "project_root": project_root,
        "lerobot_root": lerobot_root,
        "python_bin": project_root / ".venv" / "bin" / "python",
        "train_bin": project_root / ".venv" / "bin" / "lerobot-train",
        "eval_bin": project_root / ".venv" / "bin" / "lerobot-eval",
        "pi05_outputs_root": pi05_outputs_root,
        "pi_base": str(get_value(cfg, "pi_base", "lerobot/pi05_base")),
        # PT
        "pt_dataset": pt_dataset,
        "pt_dataset_root": pt_dataset_root,
        "pt_dataset_dir": project_root / pt_dataset_root / pt_dataset,
        "pt_batch_size": pt_batch_size,
        "pt_num_gpus": pt_num_gpus,
        "pt_num_workers": int(get_value(cfg, "pt_num_workers", 4, env="PT_NUM_WORKERS")),
        "pt_exp": pt_exp,
        "pt_lr": float(get_value(cfg, "pt_lr_base", 2.5e-05, env="PT_LR_BASE")) * pt_num_gpus,
        "pt_steps": int(get_value(cfg, "pt_steps", 100000, env="PT_STEPS")),
        "pt_save_freq": int(get_value(cfg, "pt_save_freq", 5000, env="PT_SAVE_FREQ")),
        "pt_wandb_project": str(get_value(cfg, "pt_wandb_project", "VLA_posttrain", env="PT_WANDB_PROJECT")),
        "pt_run_name": pt_run_name,
        "pt_output_dir": pi05_outputs_root / pt_run_name,
        # FT
        "ft_dataset": ft_dataset,
        "ft_dataset_root": ft_dataset_root,
        "ft_dataset_dir": project_root / ft_dataset_root / ft_dataset,
        "ft_batch_size": ft_batch_size,
        "ft_num_workers": int(get_value(cfg, "ft_num_workers", 4, env="FT_NUM_WORKERS")),
        "ft_exp": ft_exp,
        "ft_lr": str(get_value(cfg, "ft_lr", 2.5e-05, env="FT_LR")),
        "ft_steps": int(get_value(cfg, "ft_steps", 5000, env="FT_STEPS")),
        "ft_save_freq": int(get_value(cfg, "ft_save_freq", 500, env="FT_SAVE_FREQ")),
        "ft_wandb_project": str(get_value(cfg, "ft_wandb_project", "VLA_Finetune", env="FT_WANDB_PROJECT")),
        "ft_run_name": ft_run_name,
        "ft_output_dir": pi05_outputs_root / ft_run_name,
        "ft_pretrained_run_name": ft_pre_run_name,
        "ft_pretrained_checkpoint": ft_pre_ckpt,
        "ft_pretrained_model_path": pi05_outputs_root / ft_pre_run_name / "checkpoints" / ft_pre_ckpt / "pretrained_model",
        "ft_freeze_vision_encoder": as_bool(get_value(cfg, "ft_freeze_vision_encoder", False, env="PI05_FT_FREEZE_VISION_ENCODER")),
        "ft_train_expert_only": as_bool(get_value(cfg, "ft_train_expert_only", False, env="PI05_FT_TRAIN_EXPERT_ONLY")),
        # Eval
        "eval_target_task": eval_target_task,
        "eval_task_ids": str(get_value(cfg, "eval_task_ids", "[0,1,2,3,4,5,6,7,8,9]", env="TASK_IDS")),
        "eval_n_episodes": int(get_value(cfg, "eval_n_episodes", 5, env="N_EPISODES")),
        "eval_episode_offset": int(get_value(cfg, "eval_episode_offset", 25, env="EPISODE_OFFSET")),
        "eval_n_action_steps": int(get_value(cfg, "eval_n_action_steps", 5, env="N_ACTION_STEPS")),
        "eval_batch_size": int(get_value(cfg, "eval_batch_size", 1, env="EVAL_BATCH_SIZE")),
        "eval_stage": eval_stage,
        "eval_checkpoint": eval_checkpoint,
        "eval_model": eval_model,
        "eval_policy_path": Path(os.environ.get("POLICY_PATH", str(pi05_outputs_root / eval_model / "checkpoints" / eval_checkpoint / "pretrained_model"))),
        "eval_max_parallel_tasks": int(get_value(cfg, "eval_max_parallel_tasks", 1, env="MAX_PARALLEL_TASKS")),
        "eval_max_videos_per_task": int(get_value(cfg, "eval_max_videos_per_task", 1, env="MAX_VIDEOS_PER_TASK")),
        "eval_video_frame_stride": int(get_value(cfg, "eval_video_frame_stride", 2, env="VIDEO_FRAME_STRIDE")),
        "eval_video_fps": int(get_value(cfg, "eval_video_fps", 10, env="VIDEO_FPS")),
        "eval_wandb_project": str(get_value(cfg, "eval_wandb_project", "VLA_eval", env="WANDB_PROJECT")),
        "eval_wandb_run_name": eval_wandb_run,
    }
    settings.update(slurm_settings(cfg, "pt", cpus=16, mem="128G", time="48:00:00", qos="big_qos"))
    settings.update(slurm_settings(cfg, "ft", cpus=16, mem="128G", time="48:00:00", qos="pro6000_qos"))
    settings.update(slurm_settings(cfg, "eval", cpus=8, mem="32G", time="48:00:00", qos="base_qos"))
    return settings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--shell", action="store_true")
    args = ap.parse_args()
    settings = build_settings(load_config(args.config))
    if args.shell:
        print_shell(settings)
    else:
        for key, value in settings.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
