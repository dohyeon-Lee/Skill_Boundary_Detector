#!/usr/bin/env python3
"""Resolve the clean Stage-1 VSA YAML into shell exports."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import as_bool, as_list, load_config, print_shell, resolve_path  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "stage1_train_config.yaml"


def _at(config: dict, *path: str, default=None):
    value = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _local_model_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        return Path(resolve_path(project_root, path))
    if path.exists() or "models" not in path.parts:
        return path
    return project_root.joinpath(*path.parts[path.parts.index("models") :])


def _read_dataset_contract(dataset_dir: Path, run_tag: str) -> dict:
    """Read skill geometry and jitter metadata without opening an FSQ checkpoint."""
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Stage-1 dataset metadata not found: {info_path}")
    info = json.loads(info_path.read_text())
    levels = [int(level) for level in info.get("skill_fsq_levels", [])]
    if not levels or any(level <= 1 for level in levels):
        raise ValueError(f"Invalid skill_fsq_levels in {info_path}: {levels}")
    match = re.search(r"FSQ(\d+)", run_tag)
    if match and [int(digit) for digit in match.group(1)] != levels:
        raise ValueError(
            f"Dataset run says FSQ{match.group(1)}, but {info_path} says levels={levels}."
        )
    features = info.get("features", {})
    state_dim = int(features["observation.state"]["shape"][0])
    action_dim = int(features["action"]["shape"][0])
    return {
        "levels": levels,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "jitter_pmax": int(info.get("skill_pmax", 0)),
        "jitter_distribution": str(
            info.get("skill_jitter_distribution", "half_normal")
        ).replace("-", "_"),
    }


def _validate_predictor_checkpoint(
    checkpoint: Path,
    *,
    levels: list[int],
    predictor_contract: dict,
) -> None:
    """Fail before submission when a frozen predictor cannot be loaded exactly."""
    config_path = checkpoint / "config.json"
    weights_path = checkpoint / "model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"Stage-1 predictor config not found: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Stage-1 predictor weights not found: {weights_path}")
    source = json.loads(config_path.read_text())
    if source.get("type") != "skill_expert":
        raise ValueError(
            "Predictor checkpoint must be policy.type=skill_expert, got "
            f"{source.get('type')!r} at {checkpoint}."
        )
    if not source.get("train_skill_predictor", False):
        raise ValueError(f"Stage-1 checkpoint has no trained predictor: {checkpoint}")

    expected = {
        "skill_fsq_levels": levels,
        "skill_vocab_size": math.prod(levels),
        **predictor_contract,
    }
    mismatches = [
        f"{field}: checkpoint={source.get(field)!r}, requested={value!r}"
        for field, value in expected.items()
        if source.get(field) != value
    ]
    if mismatches:
        raise ValueError("Predictor checkpoint contract mismatch: " + "; ".join(mismatches))


def build_settings(config: dict) -> dict:
    project_root = Path(str(config["project_root"])).expanduser()
    dataset_root = project_root / str(config.get("dataset_root", "dataset"))
    outputs_root = project_root / str(config.get("outputs_root", "outputs"))
    source = str(_at(config, "dataset", "source"))
    run_tag = str(_at(config, "dataset", "run"))
    skillvla_root = dataset_root / str(
        _at(config, "dataset", "skillvla_root", default="skillvla_dataset")
    )
    dataset_dir = skillvla_root / source / run_tag / "skillvla"
    contract = _read_dataset_contract(dataset_dir, run_tag)

    pi_base = _local_model_path(
        project_root, str(_at(config, "warm_start", "pi_base", default="models/pi05_base"))
    )
    tokenizer_path = _local_model_path(
        project_root,
        str(
            _at(
                config,
                "warm_start",
                "tokenizer",
                default="models/paligemma-3b-pt-224-tokenizer",
            )
        ),
    )
    dino_model = _local_model_path(
        project_root, str(_at(config, "vision", "dino_model", default="models/dinov3-vitl16"))
    )
    fsq_value = str(_at(config, "warm_start", "fsq", default="")).strip()
    fsq_path = (
        _local_model_path(project_root, fsq_value)
        if fsq_value
        else dataset_dir.parent / "FSQ.pt"
    )
    if not (pi_base / "model.safetensors").is_file():
        raise FileNotFoundError(f"pi0.5 base checkpoint not found: {pi_base}")
    if not dino_model.is_dir():
        raise FileNotFoundError(f"DINO model not found: {dino_model}")
    train_skill_predictor = as_bool(
        _at(config, "skill_predictor", "train", default=True)
    )
    training_skill_source = str(
        _at(config, "action_conditioning", "training_skill_source", default="gt")
    ).strip().lower()
    if training_skill_source not in {"gt", "predictor"}:
        raise ValueError(
            "action_conditioning.training_skill_source must be gt or predictor."
        )
    predictor_checkpoint_value = str(
        _at(config, "warm_start", "predictor_checkpoint", default="") or ""
    ).strip()
    predictor_checkpoint = (
        Path(resolve_path(project_root, predictor_checkpoint_value))
        if predictor_checkpoint_value
        else None
    )
    if training_skill_source == "predictor" and predictor_checkpoint is None:
        raise ValueError(
            "training_skill_source=predictor requires warm_start.predictor_checkpoint."
        )
    uses_skill_predictor = train_skill_predictor or training_skill_source == "predictor"
    skill_predictor_lora = as_bool(
        _at(config, "skill_predictor", "lora", "enabled", default=True)
    )
    skill_predictor_lora_targets = str(
        _at(config, "skill_predictor", "lora", "targets", default="q,k,v,o")
    ).strip()
    skill_predictor_lora_rank = int(
        _at(config, "skill_predictor", "lora", "rank", default=8)
    )
    skill_predictor_lora_alpha = float(
        _at(config, "skill_predictor", "lora", "alpha", default=16.0)
    )
    skill_predictor_lora_dropout = float(
        _at(config, "skill_predictor", "lora", "dropout", default=0.0)
    )
    skill_predictor_lora_lr_scale = float(
        _at(config, "skill_predictor", "lora", "lr_scale", default=10.0)
    )
    if skill_predictor_lora and (
        not skill_predictor_lora_targets
        or skill_predictor_lora_rank <= 0
        or skill_predictor_lora_alpha <= 0.0
        or skill_predictor_lora_dropout < 0.0
        or skill_predictor_lora_lr_scale <= 0.0
    ):
        raise ValueError("Invalid skill_predictor.lora configuration.")
    train_terminator = as_bool(_at(config, "terminator", "train", default=True))
    if train_terminator and not fsq_path.is_file():
        raise FileNotFoundError(
            f"Stage-1 terminator FSQ checkpoint not found: {fsq_path}"
        )
    if uses_skill_predictor:
        required = ("config.json", "tokenizer_config.json", "tokenizer.json")
        missing = [name for name in required if not (tokenizer_path / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Local PaliGemma tokenizer is incomplete at {tokenizer_path}: missing {missing}."
            )
    freeze_vision_encoder = as_bool(
        _at(config, "vision", "freeze", default=False)
    )

    expert_variant = str(
        _at(config, "architecture", "expert_variant", default="gemma_300m")
    )
    cond_variant = str(
        _at(config, "architecture", "cond_variant", default=expert_variant)
    )
    if expert_variant != cond_variant:
        raise ValueError("Stage 1 requires the same 18-layer variant for cond and expert.")
    conditioning_route = str(
        _at(config, "architecture", "conditioning_route", default="state_cond")
    ).strip().lower()
    # Backward-compatible spelling used by existing Stage-1 configs/checkpoints.
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
            "architecture.conditioning_route must be "
            "state_cond|state_skill_cond|state_skill_only_cond|stateonly_cond|"
            "skillonly_cond|visiononly_cond."
        )

    max_state_dim = int(_at(config, "architecture", "max_state_dim", default=32))
    max_action_dim = int(_at(config, "architecture", "max_action_dim", default=32))
    if contract["state_dim"] > max_state_dim or contract["action_dim"] > max_action_dim:
        raise ValueError(
            "Dataset state/action dimensions exceed the configured pi0.5 projections: "
            f"dataset=({contract['state_dim']}, {contract['action_dim']}), "
            f"configured=({max_state_dim}, {max_action_dim})."
        )

    chunk_size = int(_at(config, "architecture", "chunk_size", default=10))
    action_loss_mode = str(config.get("loss", "flow")).strip().lower()
    if action_loss_mode not in {"flow", "flow_endpoint_xyz"}:
        raise ValueError(
            "loss must be flow|flow_endpoint_xyz, got "
            f"{action_loss_mode!r}."
        )
    n_action_steps = int(
        _at(config, "execution", "action_steps", default=chunk_size)
    )
    transition_jitter = as_bool(
        _at(config, "transition_randomization", "enabled", default=True)
    )
    jitter_pmax = contract["jitter_pmax"] if transition_jitter else 0
    jitter_distribution = contract["jitter_distribution"] if transition_jitter else "half_normal"
    if jitter_distribution not in {"half_normal", "uniform"}:
        raise ValueError(f"Unsupported dataset jitter distribution: {jitter_distribution!r}")

    batch_size = int(_at(config, "training", "dataloader", "batch_size", default=16))
    num_gpus = int(_at(config, "training", "dataloader", "gpus", default=1))
    base_lr = float(_at(config, "training", "optimizer", "base_lr", default=2.5e-5))
    dino_lr = _at(config, "training", "optimizer", "dino_lr", default=None)
    dino_lr = "" if dino_lr in (None, "", "null") else str(float(dino_lr))
    if freeze_vision_encoder and dino_lr:
        raise ValueError("training.optimizer.dino_lr must be null when vision.freeze=true.")

    predictor_contract = {
        "skill_predictor_vlm_variant": "gemma_2b",
        "skill_predictor_image_size": 224,
        "skill_predictor_reader_tokens": int(
            _at(config, "skill_predictor", "reader", "tokens", default=4)
        ),
        "skill_predictor_reader_depth": int(
            _at(config, "skill_predictor", "reader", "depth", default=2)
        ),
        "skill_predictor_reader_heads": int(
            _at(config, "skill_predictor", "reader", "heads", default=8)
        ),
        "skill_predictor_all_layers": as_bool(
            _at(config, "skill_predictor", "all_layers", default=True)
        ),
        "skill_predictor_detach_vlm": not skill_predictor_lora,
        "skill_predictor_lora": skill_predictor_lora,
        "skill_predictor_lora_targets": skill_predictor_lora_targets,
        "skill_predictor_lora_rank": skill_predictor_lora_rank,
        "skill_predictor_lora_alpha": skill_predictor_lora_alpha,
        "skill_predictor_lora_dropout": skill_predictor_lora_dropout,
        "skill_predictor_deadzone_frac": float(
            _at(config, "skill_predictor", "reader", "deadzone_frac", default=0.8)
        ),
        "skill_predictor_attend_image": as_bool(
            _at(config, "skill_predictor", "token_access", "image", default=True)
        ),
        "skill_predictor_attend_language": as_bool(
            _at(config, "skill_predictor", "token_access", "language", default=True)
        ),
        "tokenizer_max_length": int(
            _at(config, "skill_predictor", "tokenizer_max_length", default=200)
        ),
    }
    if predictor_checkpoint is not None:
        _validate_predictor_checkpoint(
            predictor_checkpoint,
            levels=contract["levels"],
            predictor_contract=predictor_contract,
        )
    suffix = str(_at(config, "run", "suffix", default="")).strip().strip("_")
    vision_tag = (
        "no_vision"
        if conditioning_route == "state_skill_only_cond"
        else ("dino_frozen" if freeze_vision_encoder else "dino_tuned")
    )
    run_name = (
        f"{source}_{run_tag}_{vision_tag}_{conditioning_route}_{action_loss_mode}"
    )
    if training_skill_source == "predictor":
        run_name = f"{run_name}_pretrained_predictor"
    if suffix:
        run_name = f"{run_name}_{suffix}"

    levels = contract["levels"]
    settings = {
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        "skillvla_dataset_dir": dataset_dir,
        "repo_id": f"dohyeon/{source}",
        "pi_base": pi_base,
        "fsq_path": fsq_path,
        "tokenizer_path": tokenizer_path,
        "dino_model_path": dino_model,
        "dino_image_size": int(_at(config, "vision", "image_size", default=224)),
        "freeze_vision_encoder": freeze_vision_encoder,
        "dino_lr": dino_lr,
        "action_expert_variant": expert_variant,
        "cond_encoder_variant": cond_variant,
        "conditioning_route": conditioning_route,
        "skill_fsq_levels": "[" + ",".join(str(level) for level in levels) + "]",
        "skill_vocab_size": math.prod(levels),
        "transition_jitter_pmax": jitter_pmax,
        "transition_jitter_distribution": jitter_distribution,
        "training_skill_source": training_skill_source,
        "skill_predictor_checkpoint_path": predictor_checkpoint or "",
        "train_skill_predictor": train_skill_predictor,
        "skill_predictor_weight": float(
            _at(config, "skill_predictor", "weight", default=0.5)
        ),
        "skill_predictor_lr_scale": float(
            _at(config, "skill_predictor", "lr_scale", default=1.0)
        ),
        "skill_predictor_all_layers": predictor_contract["skill_predictor_all_layers"],
        "skill_predictor_detach_vlm": not skill_predictor_lora,
        "skill_predictor_lora": skill_predictor_lora,
        "skill_predictor_lora_targets": skill_predictor_lora_targets,
        "skill_predictor_lora_rank": skill_predictor_lora_rank,
        "skill_predictor_lora_alpha": skill_predictor_lora_alpha,
        "skill_predictor_lora_dropout": skill_predictor_lora_dropout,
        "skill_predictor_lora_lr_scale": skill_predictor_lora_lr_scale,
        "skill_predictor_reader_tokens": predictor_contract["skill_predictor_reader_tokens"],
        "skill_predictor_reader_depth": predictor_contract["skill_predictor_reader_depth"],
        "skill_predictor_reader_heads": predictor_contract["skill_predictor_reader_heads"],
        "skill_predictor_deadzone_frac": predictor_contract["skill_predictor_deadzone_frac"],
        "skill_predictor_attend_image": predictor_contract["skill_predictor_attend_image"],
        "skill_predictor_attend_language": predictor_contract["skill_predictor_attend_language"],
        "tokenizer_max_length": predictor_contract["tokenizer_max_length"],
        "train_terminator": train_terminator,
        "terminator_freeze_vision_encoder": as_bool(
            _at(config, "terminator", "freeze_vision", default=True)
        ),
        "terminator_dino_model_path": dino_model,
        "terminator_lr_scale": float(
            _at(config, "training", "optimizer", "terminator_lr_scale", default=1.0)
        ),
        "terminator_end_target_sigma": float(
            _at(config, "terminator", "end_target_sigma", default=2.0)
        ),
        "terminator_end_pos_weight": float(
            _at(config, "terminator", "end_pos_weight", default=1.0)
        ),
        "max_state_dim": max_state_dim,
        "max_action_dim": max_action_dim,
        "chunk_size": chunk_size,
        "action_loss_mode": action_loss_mode,
        "n_action_steps": n_action_steps,
        "min_period": float(_at(config, "flow", "min_period", default=4e-3)),
        "max_period": float(_at(config, "flow", "max_period", default=4.0)),
        "time_sampling_beta_alpha": float(
            _at(config, "flow", "beta_alpha", default=1.5)
        ),
        "time_sampling_beta_beta": float(
            _at(config, "flow", "beta_beta", default=1.0)
        ),
        "time_sampling_scale": float(_at(config, "flow", "scale", default=0.999)),
        "time_sampling_offset": float(_at(config, "flow", "offset", default=0.001)),
        "pt_run_name": run_name,
        "pt_output_dir": outputs_root / "skillVLA_stage1" / run_name,
        "batch_size": batch_size,
        "num_workers": int(
            _at(config, "training", "dataloader", "workers", default=2)
        ),
        "num_gpus": num_gpus,
        "gradient_checkpointing": as_bool(
            _at(config, "training", "gradient_checkpointing", default=True)
        ),
        "lr": base_lr * num_gpus,
        "steps": int(_at(config, "training", "schedule", "steps", default=50000)),
        "log_freq": int(
            _at(config, "training", "schedule", "log_every", default=100)
        ),
        "save_freq": int(
            _at(config, "training", "schedule", "save_every", default=5000)
        ),
        "wandb_enable": as_bool(
            _at(config, "logging", "wandb", "enable", default=True)
        ),
        "wandb_project": str(
            _at(config, "logging", "wandb", "project", default="VLA_stage1")
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
