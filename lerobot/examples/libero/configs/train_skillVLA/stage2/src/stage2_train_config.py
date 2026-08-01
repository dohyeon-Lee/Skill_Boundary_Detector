#!/usr/bin/env python3
"""Resolve clean BayesVLA-style Stage-2 training settings."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import as_bool, as_list, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "stage2_train_config.yaml"


def _at(config: dict, *path: str, default=None):
    value = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _validate_auxiliary_config(config: dict) -> None:
    auxiliary = config.get("auxiliary") or {}
    if not isinstance(auxiliary, dict):
        raise ValueError("auxiliary must be a mapping.")
    unknown_auxiliaries = sorted(set(auxiliary) - {"skill_predictor", "terminator"})
    if unknown_auxiliaries:
        raise ValueError(
            f"Unknown Stage-2 auxiliary entries: {unknown_auxiliaries}."
        )
    for name in ("skill_predictor", "terminator"):
        settings = auxiliary.get(name) or {}
        if not isinstance(settings, dict):
            raise ValueError(f"auxiliary.{name} must be a mapping.")
        unknown = sorted(set(settings) - {"train"})
        if unknown:
            raise ValueError(
                f"auxiliary.{name} only accepts 'train'; {unknown} must be "
                "inherited from the Stage-1 checkpoint."
            )


def _local_path(project_root: Path, value: object, *, marker: str | None = None) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        return project_root / path
    if path.exists() or marker is None or marker not in path.parts:
        return path
    return project_root.joinpath(*path.parts[path.parts.index(marker) :])


def _read_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return json.loads(path.read_text())


def _dataset_contract(dataset_dir: Path) -> dict:
    info = _read_json(dataset_dir / "meta" / "info.json", "Stage-2 dataset metadata")
    levels = [int(value) for value in info.get("skill_fsq_levels", [])]
    if not levels or any(value <= 1 for value in levels):
        raise ValueError(f"Invalid Stage-2 skill_fsq_levels: {levels}")
    features = info.get("features", {})
    return {
        "levels": levels,
        "state_dim": int(features["observation.state"]["shape"][0]),
        "action_dim": int(features["action"]["shape"][0]),
    }


def _require_stage1_contract(config: dict, checkpoint: Path) -> None:
    if config.get("type") != "skill_expert":
        raise ValueError(
            f"Stage 2 requires policy.type=skill_expert, got {config.get('type')!r} at {checkpoint}."
        )
    has_predictor = as_bool(config.get("train_skill_predictor", False)) or str(
        config.get("training_skill_source", "gt")
    ).strip().lower() == "predictor"
    if not has_predictor:
        raise ValueError("Stage-1 checkpoint must contain the trained frozen-VLM skill predictor.")
    if not config.get("train_terminator", False):
        raise ValueError("Stage-1 checkpoint must contain the co-trained terminator.")
    if config.get("action_expert_variant") != "gemma_300m":
        raise ValueError("Stage 2 expects the 18-layer gemma_300m action expert.")
    if config.get("cond_encoder_variant") != "gemma_300m":
        raise ValueError("Stage 2 expects the 18-layer gemma_300m condition encoder.")
    conditioning_route = str(config.get("conditioning_route", "")).strip().lower()
    if conditioning_route == "skill_cond":
        conditioning_route = "skillonly_cond"
    if conditioning_route not in {
        "state_cond",
        "state_skill_cond",
        "state_skill_only_cond",
        "stateonly_cond",
        "skillonly_cond",
        "visiononly_cond",
    }:
        raise ValueError(
            "Stage 2 expects conditioning_route="
            "state_cond|state_skill_cond|state_skill_only_cond|stateonly_cond|"
            "skillonly_cond|visiononly_cond."
        )
    config["conditioning_route"] = conditioning_route
    if not (
        config.get("skill_predictor_attend_image", False)
        and config.get("skill_predictor_attend_language", False)
    ):
        raise ValueError("Stage 2 needs both image and language tokens in the frozen VLM memory.")


def _stage1_dataset_run(checkpoint: Path) -> str:
    """Recover the dataset run directory recorded when Stage 1 was trained."""
    train_config = _read_json(
        checkpoint / "train_config.json", "Stage-1 training config"
    )
    dataset_path = Path(str((train_config.get("dataset") or {}).get("root") or ""))
    if dataset_path.name != "skillvla" or not dataset_path.parent.name:
        raise ValueError(
            "Stage-1 train_config.json must record dataset.root ending in "
            f"<run>/skillvla, got {str(dataset_path)!r}."
        )
    return dataset_path.parent.name


def build_settings(config: dict) -> dict:
    project_root = Path(str(config["project_root"])).expanduser()
    dataset_root = project_root / str(config.get("dataset_root", "dataset"))
    outputs_root = project_root / str(config.get("outputs_root", "outputs"))

    stage1_run = str(_at(config, "warm_start", "stage1_run")).strip()
    stage1_checkpoint = str(
        _at(config, "warm_start", "checkpoint", default="last")
    ).strip()
    if not stage1_run:
        raise ValueError("warm_start.stage1_run must be the exact Stage-1 output directory name.")
    stage1_path = (
        outputs_root
        / "skillVLA_stage1"
        / stage1_run
        / "checkpoints"
        / stage1_checkpoint
        / "pretrained_model"
    )
    stage1_config = _read_json(stage1_path / "config.json", "Stage-1 policy config")
    if not (stage1_path / "model.safetensors").is_file():
        raise FileNotFoundError(f"Stage-1 weights not found: {stage1_path / 'model.safetensors'}")
    _require_stage1_contract(stage1_config, stage1_path)

    source = str(_at(config, "dataset", "source")).strip()
    if not source:
        raise ValueError("dataset.source is required because the Stage-2 split may differ from Stage 1.")
    stage1_dataset_run = _stage1_dataset_run(stage1_path)
    configured_run = str(_at(config, "dataset", "run", default="") or "").strip()
    if configured_run and configured_run != stage1_dataset_run:
        raise ValueError(
            "dataset.run must match the Stage-1 dataset run when explicitly set: "
            f"stage1={stage1_dataset_run!r}, configured={configured_run!r}."
        )
    run_tag = stage1_dataset_run
    skillvla_root = dataset_root / str(
        _at(config, "dataset", "skillvla_root", default="skillvla_dataset")
    )
    dataset_dir = skillvla_root / source / run_tag / "skillvla"
    contract = _dataset_contract(dataset_dir)
    stage1_levels = [int(value) for value in stage1_config["skill_fsq_levels"]]
    if contract["levels"] != stage1_levels:
        raise ValueError(
            f"Stage-2 dataset FSQ levels {contract['levels']} do not match Stage 1 {stage1_levels}."
        )
    if contract["state_dim"] > int(stage1_config["max_state_dim"]):
        raise ValueError("Stage-2 state dimension exceeds the Stage-1 projection size.")
    if contract["action_dim"] > int(stage1_config["max_action_dim"]):
        raise ValueError("Stage-2 action dimension exceeds the Stage-1 projection size.")

    dino_path = _local_path(
        project_root, stage1_config["dino_model_path"], marker="models"
    )
    tokenizer_path = _local_path(
        project_root, stage1_config["tokenizer_path"], marker="models"
    )
    terminator_dino_path = _local_path(
        project_root,
        stage1_config.get("terminator_dino_model_path")
        or stage1_config["dino_model_path"],
        marker="models",
    )
    fsq_path = _local_path(project_root, stage1_config["fsq_path"])
    for path, label in (
        (dino_path, "DINO model"),
        (tokenizer_path, "PaliGemma tokenizer"),
        (terminator_dino_path, "terminator DINO model"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Inherited {label} not found: {path}")
    if not fsq_path.is_file():
        raise FileNotFoundError(f"Inherited FSQ checkpoint not found: {fsq_path}")

    likelihood_layers = int(_at(config, "likelihood", "layers", default=4))
    if likelihood_layers != 4:
        raise ValueError("BayesVLA-matched Stage 2 fixes likelihood.layers=4.")
    skill_source = str(
        _at(config, "likelihood", "training_skill_source", default="gt")
    ).strip().lower()
    if skill_source not in {"gt", "predictor"}:
        raise ValueError("likelihood.training_skill_source must be gt or predictor.")
    action_loss_mode = str(config.get("loss", "flow")).strip().lower()
    if action_loss_mode not in {"flow", "flow_endpoint_xyz"}:
        raise ValueError(
            "loss must be flow|flow_endpoint_xyz, got "
            f"{action_loss_mode!r}."
        )
    _validate_auxiliary_config(config)
    finetune_skill_predictor = as_bool(
        _at(config, "auxiliary", "skill_predictor", "train", default=False)
    )
    finetune_terminator = as_bool(
        _at(config, "auxiliary", "terminator", "train", default=False)
    )
    batch_size = int(
        _at(config, "training", "dataloader", "batch_size", default=16)
    )
    same_skill_batch_enabled = as_bool(
        _at(
            config,
            "training",
            "dataloader",
            "same_skill_different_task",
            "enabled",
            default=False,
        )
    )
    same_skill_batch_fraction = float(
        _at(
            config,
            "training",
            "dataloader",
            "same_skill_different_task",
            "grouped_fraction",
            default=0.5,
        )
    )
    same_skill_progress_temperature = float(
        _at(
            config,
            "training",
            "dataloader",
            "same_skill_different_task",
            "progress_temperature",
            default=0.1,
        )
    )
    if not 0.0 <= same_skill_batch_fraction <= 1.0:
        raise ValueError(
            "same_skill_different_task.grouped_fraction must be in [0, 1]."
        )
    if same_skill_progress_temperature <= 0.0:
        raise ValueError(
            "same_skill_different_task.progress_temperature must be > 0."
        )
    if same_skill_batch_enabled and batch_size < 4:
        raise ValueError(
            "same_skill_different_task needs dataloader.batch_size >= 4."
        )
    grouped_samples = int(batch_size * same_skill_batch_fraction) // 2 * 2
    if same_skill_batch_enabled and finetune_terminator and grouped_samples >= batch_size:
        raise ValueError(
            "auxiliary.terminator.train=true needs at least one random dataloader "
            "slot; lower same_skill_different_task.grouped_fraction."
        )
    suffix = str(_at(config, "run", "suffix", default="")).strip().strip("_")
    loss_tag = "flow" if action_loss_mode == "flow" else "endpoint"
    batch_tag = "batchON" if same_skill_batch_enabled else "batchOFF"
    run_name = (
        f"{stage1_run}_{stage1_checkpoint}_{loss_tag}_{skill_source}_{batch_tag}"
    )
    if suffix:
        run_name += f"_{suffix}"

    num_gpus = int(_at(config, "training", "dataloader", "gpus", default=1))
    base_lr = float(_at(config, "training", "optimizer", "base_lr", default=2.5e-5))
    settings = {
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        "skillvla_dataset_dir": dataset_dir,
        "repo_id": f"dohyeon/{source}",
        "stage1_checkpoint_path": stage1_path,
        "dino_model_path": dino_path,
        "tokenizer_path": tokenizer_path,
        "fsq_path": fsq_path,
        "action_expert_variant": stage1_config["action_expert_variant"],
        "cond_encoder_variant": stage1_config["cond_encoder_variant"],
        "chunk_size": int(stage1_config["chunk_size"]),
        "n_action_steps": int(stage1_config["n_action_steps"]),
        "max_state_dim": int(stage1_config["max_state_dim"]),
        "max_action_dim": int(stage1_config["max_action_dim"]),
        "num_inference_steps": int(stage1_config["num_inference_steps"]),
        "min_period": float(stage1_config["min_period"]),
        "max_period": float(stage1_config["max_period"]),
        "time_sampling_beta_alpha": float(stage1_config["time_sampling_beta_alpha"]),
        "time_sampling_beta_beta": float(stage1_config["time_sampling_beta_beta"]),
        "time_sampling_scale": float(stage1_config["time_sampling_scale"]),
        "time_sampling_offset": float(stage1_config["time_sampling_offset"]),
        "dino_image_size": int(stage1_config["dino_image_size"]),
        "freeze_vision_encoder": as_bool(stage1_config["freeze_vision_encoder"]),
        "conditioning_route": stage1_config["conditioning_route"],
        "skill_vocab_size": math.prod(stage1_levels),
        "skill_fsq_levels": "[" + ",".join(str(value) for value in stage1_levels) + "]",
        "transition_jitter_pmax": int(stage1_config["transition_jitter_pmax"]),
        "transition_jitter_distribution": stage1_config["transition_jitter_distribution"],
        "train_skill_predictor": True,
        "skill_predictor_weight": float(stage1_config["skill_predictor_weight"]),
        "skill_predictor_lr_scale": float(
            stage1_config["skill_predictor_lr_scale"]
        ),
        "skill_predictor_all_layers": as_bool(stage1_config["skill_predictor_all_layers"]),
        "skill_predictor_detach_vlm": as_bool(
            stage1_config.get("skill_predictor_detach_vlm", True)
        ),
        "skill_predictor_lora": as_bool(
            stage1_config.get("skill_predictor_lora", False)
        ),
        "skill_predictor_lora_targets": str(
            stage1_config.get("skill_predictor_lora_targets", "q,k,v,o")
        ),
        "skill_predictor_lora_rank": int(
            stage1_config.get("skill_predictor_lora_rank", 8)
        ),
        "skill_predictor_lora_alpha": float(
            stage1_config.get("skill_predictor_lora_alpha", 16.0)
        ),
        "skill_predictor_lora_dropout": float(
            stage1_config.get("skill_predictor_lora_dropout", 0.0)
        ),
        "skill_predictor_lora_lr_scale": float(
            stage1_config.get("skill_predictor_lora_lr_scale", 10.0)
        ),
        "skill_predictor_vlm_variant": stage1_config["skill_predictor_vlm_variant"],
        "skill_predictor_image_size": int(stage1_config["skill_predictor_image_size"]),
        "skill_predictor_reader_tokens": int(stage1_config["skill_predictor_reader_tokens"]),
        "skill_predictor_reader_depth": int(stage1_config["skill_predictor_reader_depth"]),
        "skill_predictor_reader_heads": int(stage1_config["skill_predictor_reader_heads"]),
        "skill_predictor_deadzone_frac": float(stage1_config["skill_predictor_deadzone_frac"]),
        "skill_predictor_attend_image": True,
        "skill_predictor_attend_language": True,
        "tokenizer_max_length": int(stage1_config["tokenizer_max_length"]),
        "train_terminator": True,
        "terminator_freeze_vision_encoder": as_bool(
            stage1_config["terminator_freeze_vision_encoder"]
        ),
        "terminator_dino_model_path": terminator_dino_path,
        "terminator_lr_scale": float(stage1_config["terminator_lr_scale"]),
        "terminator_end_target_sigma": float(stage1_config["terminator_end_target_sigma"]),
        "terminator_end_pos_weight": float(stage1_config["terminator_end_pos_weight"]),
        "likelihood_num_layers": likelihood_layers,
        "likelihood_cross_attention_heads": 8,
        "training_skill_source": skill_source,
        "action_loss_mode": action_loss_mode,
        "finetune_skill_predictor": finetune_skill_predictor,
        "finetune_terminator": finetune_terminator,
        "same_skill_batch_enabled": same_skill_batch_enabled,
        "same_skill_batch_fraction": same_skill_batch_fraction,
        "same_skill_progress_temperature": same_skill_progress_temperature,
        "gradient_checkpointing": as_bool(
            _at(config, "training", "gradient_checkpointing", default=True)
        ),
        "lr": base_lr * num_gpus,
        "batch_size": batch_size,
        "num_workers": int(
            _at(config, "training", "dataloader", "workers", default=2)
        ),
        "num_gpus": num_gpus,
        "steps": int(_at(config, "training", "schedule", "steps", default=50000)),
        "log_freq": int(
            _at(config, "training", "schedule", "log_every", default=100)
        ),
        "save_freq": int(
            _at(config, "training", "schedule", "save_every", default=5000)
        ),
        "pt_run_name": run_name,
        "pt_output_dir": outputs_root / "skillVLA_stage2" / run_name,
        "wandb_enable": as_bool(
            _at(config, "logging", "wandb", "enable", default=True)
        ),
        "wandb_project": str(
            _at(config, "logging", "wandb", "project", default="VLA_stage2")
        ),
        "train_partition": ",".join(as_list(config.get("train_partition", ["big"]))) or "big",
        "train_qos": str(config.get("train_qos", "big_qos")),
        "train_gres": str(_at(config, "slurm", "gres", default="gpu:1")),
        "train_cpus_per_task": int(_at(config, "slurm", "cpus", default=16)),
        "train_mem": str(_at(config, "slurm", "memory", default="256G")),
        "train_time": str(_at(config, "slurm", "time", default="48:00:00")),
        "train_nodelist": str(config.get("train_nodelist", "")),
        "train_exclude_nodes": ",".join(as_list(config.get("train_exclude_nodes", []))),
    }
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
