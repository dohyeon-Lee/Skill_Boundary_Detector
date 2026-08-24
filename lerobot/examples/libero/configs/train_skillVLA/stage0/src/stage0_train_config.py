#!/usr/bin/env python3
"""Resolve the compact renewed SkillVLA Stage-0 YAML into shell exports."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import as_bool, as_list, load_config, print_shell, resolve_path  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "stage0_train_config.yaml"


def _at(cfg: dict, *path: str, default=None):
    value = cfg
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _local_model(project_root: Path, value) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        return Path(resolve_path(project_root, path))
    if path.exists() or "models" not in path.parts:
        return path
    return project_root.joinpath(*path.parts[path.parts.index("models"):])


def _load_fsq_config(project_root: Path, fsq_path: Path):
    sys.path.insert(0, str(project_root / "lerobot" / "examples" / "libero"))
    from FSQ import FORMAT_VERSION, SplineFSQAEConfig  # noqa: PLC0415

    checkpoint = torch.load(str(fsq_path), map_location="cpu", weights_only=False)
    fsq_cfg = checkpoint.get("cfg")
    if fsq_cfg is None:
        raise ValueError(f"FSQ checkpoint has no cfg: {fsq_path}")
    if isinstance(fsq_cfg, dict):
        fsq_cfg = SplineFSQAEConfig(**fsq_cfg)
    if int(getattr(fsq_cfg, "format_version", 0)) != int(FORMAT_VERSION):
        raise ValueError(
            f"Stage-0 expects the current FSQ format v{FORMAT_VERSION}, got "
            f"{getattr(fsq_cfg, 'format_version', None)} from {fsq_path}."
        )
    return fsq_cfg


def build_settings(cfg: dict) -> dict:
    project_root = Path(str(cfg["project_root"])).expanduser()
    dataset_root = project_root / str(cfg.get("dataset_root", "dataset"))
    outputs_root = project_root / str(cfg.get("outputs_root", "outputs"))
    skillvla_root = dataset_root / str(_at(cfg, "dataset", "skillvla_root", default="skillvla_dataset"))
    source = str(_at(cfg, "dataset", "source"))
    run_tag = str(_at(cfg, "dataset", "run"))
    run_dir = skillvla_root / source / run_tag

    fsq_value = _at(cfg, "warm_start", "fsq", default="")
    fsq_path = Path(resolve_path(project_root, fsq_value)) if str(fsq_value).strip() else run_dir / "FSQ.pt"
    if not fsq_path.is_file():
        raise FileNotFoundError(f"Stage-0 FSQ checkpoint not found: {fsq_path}")
    fsq_cfg = _load_fsq_config(project_root, fsq_path)

    levels = [int(x) for x in fsq_cfg.fsq_levels]
    match = re.search(r"FSQ(\d+)", run_tag)
    if match and [int(x) for x in match.group(1)] != levels:
        raise ValueError(f"Dataset run says FSQ{match.group(1)}, but {fsq_path} contains levels={levels}.")

    pi_base = Path(resolve_path(project_root, _at(cfg, "warm_start", "pi_base", default="models/pi05_base")))
    tokenizer = Path(resolve_path(
        project_root,
        _at(cfg, "warm_start", "tokenizer", default="models/paligemma-3b-pt-224-tokenizer"),
    ))
    dino_model = _local_model(project_root, _at(cfg, "cond", "dino_model", default=fsq_cfg.dino_model_path))
    state_cond_mode = str(_at(cfg, "expert", "skill_cond_mode", default=getattr(fsq_cfg, "skill_cond_mode", "broadcast")))
    if state_cond_mode not in {"state", "state_skill", "broadcast"}:
        raise ValueError(f"expert.skill_cond_mode must be state|state_skill|broadcast, got {state_cond_mode!r}.")

    cond_weight = float(_at(cfg, "loss", "conditional", "weight", default=1.0))
    uncond_weight = float(_at(cfg, "loss", "unconditional", "weight", default=0.5))
    conditional_objective = str(_at(
        cfg, "loss", "conditional", "objective", default="flow")).strip().lower()
    gradient_routing = str(_at(cfg, "loss", "gradient_routing", default="split")).strip().lower()
    uncond_start = float(_at(cfg, "loss", "unconditional", "skill_start_weight", default=1.0))
    uncond_end = float(_at(cfg, "loss", "unconditional", "skill_end_weight", default=1.0))
    if min(cond_weight, uncond_weight) < 0.0 or min(uncond_start, uncond_end) <= 0.0:
        raise ValueError("Stage-0 loss weights must satisfy cond/uncond >= 0 and timestep weights > 0.")
    if gradient_routing not in {"shared", "split"}:
        raise ValueError(f"loss.gradient_routing must be shared|split, got {gradient_routing!r}.")
    if conditional_objective not in {"flow", "endpoint_xyz"}:
        raise ValueError(
            "loss.conditional.objective must be flow|endpoint_xyz, "
            f"got {conditional_objective!r}."
        )
    language_ranking_enabled = as_bool(_at(
        cfg, "loss", "language_ranking", "enabled", default=False))
    language_ranking_weight = float(_at(
        cfg, "loss", "language_ranking", "weight", default=0.1))
    language_ranking_relative_margin = float(_at(
        cfg, "loss", "language_ranking", "relative_margin", default=0.01))
    if language_ranking_enabled and language_ranking_weight <= 0.0:
        raise ValueError("loss.language_ranking.weight must be > 0 when enabled.")
    if language_ranking_enabled and conditional_objective != "flow":
        raise ValueError(
            "Exp4-1 endpoint_xyz and Exp4-2 language_ranking are separate objectives; enable only one."
        )
    if language_ranking_relative_margin < 0.0:
        raise ValueError("loss.language_ranking.relative_margin must be >= 0.")
    if language_ranking_enabled and not as_bool(_at(cfg, "token_access", "language", default=True)):
        raise ValueError("loss.language_ranking.enabled=true requires token_access.language=true.")

    alpha_min = float(_at(cfg, "vlm_residual", "alpha_min", default=0.1))
    alpha_max = float(_at(cfg, "vlm_residual", "alpha_max", default=0.2))
    init_alpha = float(_at(cfg, "vlm_residual", "init_alpha", default=0.15))
    if not 0.0 <= alpha_min < init_alpha < alpha_max:
        raise ValueError(
            "Need 0 <= vlm_residual.alpha_min < init_alpha < alpha_max, got "
            f"{alpha_min}, {init_alpha}, and {alpha_max}."
        )
    vlm_residual_enabled = as_bool(_at(cfg, "vlm_residual", "enabled", default=True))

    train_skill_predictor = as_bool(_at(cfg, "skill_predictor", "train", default=False))
    skill_predictor_weight = float(_at(cfg, "skill_predictor", "weight", default=0.5))
    skill_predictor_lr_scale = float(_at(cfg, "skill_predictor", "lr_scale", default=1.0))
    skill_predictor_all_layers = as_bool(_at(cfg, "skill_predictor", "all_layers", default=False))
    skill_predictor_attend_image = as_bool(
        _at(cfg, "skill_predictor", "token_access", "image", default=True))
    skill_predictor_attend_language = as_bool(
        _at(cfg, "skill_predictor", "token_access", "language", default=True))
    if train_skill_predictor and not vlm_residual_enabled:
        raise ValueError("skill_predictor.train=true requires vlm_residual.enabled=true in renewed Stage-0.")
    if train_skill_predictor and min(skill_predictor_weight, skill_predictor_lr_scale) <= 0.0:
        raise ValueError("skill_predictor.weight and skill_predictor.lr_scale must be > 0 when enabled.")
    if train_skill_predictor and not (skill_predictor_attend_image or skill_predictor_attend_language):
        raise ValueError("skill_predictor.token_access must enable image and/or language when training.")

    batch_size = int(_at(cfg, "training", "dataloader", "batch_size", default=16))
    num_gpus = int(_at(cfg, "training", "dataloader", "gpus", default=1))
    same_skill_batch_enabled = as_bool(_at(
        cfg, "training", "dataloader", "same_skill_different_task", "enabled", default=False))
    same_skill_batch_fraction = float(_at(
        cfg, "training", "dataloader", "same_skill_different_task", "grouped_fraction", default=0.5))
    same_skill_progress_temperature = float(_at(
        cfg, "training", "dataloader", "same_skill_different_task", "progress_temperature", default=0.1))
    if not 0.0 <= same_skill_batch_fraction <= 1.0:
        raise ValueError("same_skill_different_task.grouped_fraction must be in [0, 1].")
    if same_skill_progress_temperature <= 0.0:
        raise ValueError("same_skill_different_task.progress_temperature must be > 0.")
    if same_skill_batch_enabled and batch_size < 4:
        raise ValueError("same_skill_different_task needs dataloader.batch_size >= 4.")
    if (same_skill_batch_enabled
            and as_bool(_at(cfg, "terminator", "train", default=False))
            and int(batch_size * same_skill_batch_fraction) // 2 * 2 >= batch_size):
        raise ValueError(
            "terminator.train=true needs at least one random dataloader slot; lower "
            "same_skill_different_task.grouped_fraction."
        )
    suffix = str(_at(cfg, "run", "suffix", default="")).strip().strip("_")
    run_name = f"{run_tag}_{suffix}" if suffix else run_tag

    settings = {
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        "source_dataset": source,
        "run_tag": run_tag,
        "skillvla_dataset_dir": run_dir / "skillvla",
        "repo_id": f"dohyeon/{source}",
        "pi_base": pi_base,
        "tokenizer_path": tokenizer,
        "fsq_ckpt": fsq_path,
        "dino_model_path": dino_model,
        "vision_backbone": str(fsq_cfg.vision_backbone),
        "cond_encoder_variant": str(_at(cfg, "cond", "encoder_variant", default="gemma_300m")),
        "state_cond_mode": state_cond_mode,
        "cond_state_adarms": as_bool(_at(cfg, "cond", "state_adarms", default=False)),
        "action_expert_variant": str(_at(cfg, "expert", "variant", default="gemma_300m")),
        "max_state_dim": int(fsq_cfg.max_state_dim),
        "max_action_dim": int(fsq_cfg.max_action_dim),
        "chunk_size": int(fsq_cfg.chunk_size),
        "min_period": 4e-3,
        "max_period": 4.0,
        "time_sampling_beta_alpha": 1.5,
        "time_sampling_beta_beta": 1.0,
        "time_sampling_scale": 0.999,
        "time_sampling_offset": 0.001,
        "skill_fsq_levels": "[" + ",".join(str(x) for x in levels) + "]",
        "dino_image_size": int(fsq_cfg.dino_image_size),
        "siglip_image_size": int(fsq_cfg.siglip_image_size),
        "attend_image": as_bool(_at(cfg, "token_access", "image", default=True)),
        "attend_language": as_bool(_at(cfg, "token_access", "language", default=True)),
        "vlm_cond": as_bool(_at(cfg, "connections", "vlm_to_cond", default=False)),
        "cond_expert": as_bool(_at(cfg, "connections", "cond_to_expert", default=True)),
        "vlm_expert": False,
        "stage0_vlm_residual": vlm_residual_enabled,
        "stage0_vlm_residual_heads": str(_at(cfg, "vlm_residual", "heads", default="auto")),
        "stage0_vlm_residual_dropout": float(_at(cfg, "vlm_residual", "dropout", default=0.0)),
        "stage0_vlm_residual_alpha_min": alpha_min,
        "stage0_vlm_residual_alpha_max": alpha_max,
        "stage0_vlm_residual_init_alpha": init_alpha,
        "stage0_vlm_residual_zero_init_output": as_bool(
            _at(cfg, "vlm_residual", "zero_init_output", default=True)),
        "stage0_train_skill_predictor": train_skill_predictor,
        "stage0_skill_predictor_weight": skill_predictor_weight,
        "stage0_skill_predictor_lr_scale": skill_predictor_lr_scale,
        "stage0_skill_predictor_all_layers": skill_predictor_all_layers,
        "stage0_skill_predictor_detach_vlm": True,
        "skill_reader_all_layers": skill_predictor_all_layers,
        "reader_attend_image": skill_predictor_attend_image,
        "reader_attend_language": skill_predictor_attend_language,
        "stage0_conditional_loss_weight": cond_weight,
        "stage0_unconditional_loss_weight": uncond_weight,
        "stage0_conditional_objective": conditional_objective,
        "stage0_gradient_routing": gradient_routing,
        "stage0_uncond_skill_start_loss_weight": uncond_start,
        "stage0_uncond_skill_end_loss_weight": uncond_end,
        "stage0_language_ranking_enabled": language_ranking_enabled,
        "stage0_language_ranking_weight": language_ranking_weight,
        "stage0_language_ranking_relative_margin": language_ranking_relative_margin,
        "stage0_freeze_vlm_llm": as_bool(_at(cfg, "freeze", "vlm_llm", default=True)),
        "stage0_freeze_vlm_vision": as_bool(_at(cfg, "freeze", "vlm_vision", default=True)),
        "stage0_freeze_cond": as_bool(_at(cfg, "freeze", "cond", default=False)),
        "stage0_freeze_cond_vision": as_bool(_at(cfg, "freeze", "cond_vision", default=False)),
        "stage0_freeze_expert": as_bool(_at(cfg, "freeze", "expert", default=False)),
        "stage0_freeze_skill_reader": as_bool(_at(cfg, "freeze", "skill_reader", default=True)),
        "stage0_freeze_skill_head": as_bool(_at(cfg, "freeze", "skill_head", default=True)),
        "train_terminator": as_bool(_at(cfg, "terminator", "train", default=False)),
        "terminator_freeze_vision_encoder": as_bool(
            _at(cfg, "terminator", "freeze_vision", default=False)),
        "track_param_drift": as_bool(_at(cfg, "logging", "param_drift", default=True)),
        "pt_run_name": run_name,
        "pt_output_dir": outputs_root / "skillVLA_stage0" / run_name,
        "batch_size": batch_size,
        "num_workers": int(_at(cfg, "training", "dataloader", "workers", default=2)),
        "num_gpus": num_gpus,
        "same_skill_batch_enabled": same_skill_batch_enabled,
        "same_skill_batch_fraction": same_skill_batch_fraction,
        "same_skill_progress_temperature": same_skill_progress_temperature,
        "lr": float(_at(cfg, "training", "optimizer", "base_lr", default=2.5e-5)) * num_gpus,
        "cond_lr_scale": float(_at(cfg, "training", "optimizer", "cond_lr_scale", default=1.0)),
        "expert_lr_scale": float(_at(cfg, "training", "optimizer", "expert_lr_scale", default=1.0)),
        "terminator_lr_scale": float(_at(cfg, "training", "optimizer", "terminator_lr_scale", default=1.0)),
        "gradient_checkpointing": as_bool(_at(cfg, "training", "gradient_checkpointing", default=True)),
        "steps": int(_at(cfg, "training", "schedule", "steps", default=50000)),
        "save_freq": int(_at(cfg, "training", "schedule", "save_every", default=2500)),
        "wandb_enable": as_bool(_at(cfg, "logging", "wandb", "enable", default=True)),
        "wandb_project": str(_at(cfg, "logging", "wandb", "project", default="VLA_stage0")),
        "train_partition": ",".join(as_list(cfg.get("train_partition", ["big"]))) or "big",
        "train_qos": str(cfg.get("train_qos", "big_qos")),
        "train_gres": str(_at(cfg, "slurm", "gres", default="gpu:1")),
        "train_cpus_per_task": int(_at(cfg, "slurm", "cpus", default=16)),
        "train_mem": str(_at(cfg, "slurm", "memory", default="256G")),
        "train_time": str(_at(cfg, "slurm", "time", default="48:00:00")),
        "train_nodelist": str(cfg.get("train_nodelist", "")),
        "train_exclude_nodes": ",".join(as_list(cfg.get("train_exclude_nodes", []))),
    }
    if math.prod(levels) <= 1:
        raise ValueError(f"Invalid FSQ levels: {levels}")
    if settings["vision_backbone"] == "dino" and not dino_model.is_dir():
        raise FileNotFoundError(f"Cond DINO model not found: {dino_model}")
    return settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    settings = build_settings(load_config(args.config))
    if args.shell:
        print_shell(settings)
    else:
        for key, value in settings.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
