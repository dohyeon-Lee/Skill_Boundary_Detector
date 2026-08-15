#!/usr/bin/env python3
"""Resolve the auxiliary-only terminator/predictor YAML into shell exports."""

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

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "terminator_train_config.yaml"


def _at(config: dict, *path: str, default=None):
    value = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _local_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        return Path(resolve_path(project_root, path))
    if path.exists() or "models" not in path.parts:
        return path
    return project_root.joinpath(*path.parts[path.parts.index("models") :])


def _dataset_contract(dataset_dir: Path, run_tag: str) -> dict:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Auxiliary dataset metadata not found: {info_path}")
    info = json.loads(info_path.read_text())
    levels = [int(level) for level in info.get("skill_fsq_levels", [])]
    if not levels or any(level <= 1 for level in levels):
        raise ValueError(f"Invalid skill_fsq_levels in {info_path}: {levels}")
    match = re.search(r"FSQ(\d+)", run_tag)
    if match and [int(digit) for digit in match.group(1)] != levels:
        raise ValueError(
            f"Dataset run says FSQ{match.group(1)}, but metadata says levels={levels}."
        )
    features = info.get("features", {})
    return {
        "levels": levels,
        "state_dim": int(features["observation.state"]["shape"][0]),
        "action_dim": int(features["action"]["shape"][0]),
    }


def _predictor_contract(config: dict) -> dict:
    lora_enabled = as_bool(_at(config, "skill_predictor", "lora", "enabled", default=True))
    return {
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
        "skill_predictor_detach_vlm": not lora_enabled,
        "skill_predictor_lora": lora_enabled,
        "skill_predictor_lora_targets": str(
            _at(config, "skill_predictor", "lora", "targets", default="q,k,v,o")
        ),
        "skill_predictor_lora_rank": int(
            _at(config, "skill_predictor", "lora", "rank", default=8)
        ),
        "skill_predictor_lora_alpha": float(
            _at(config, "skill_predictor", "lora", "alpha", default=16.0)
        ),
        "skill_predictor_lora_dropout": float(
            _at(config, "skill_predictor", "lora", "dropout", default=0.0)
        ),
        "skill_predictor_deadzone_frac": float(
            _at(config, "skill_predictor", "reader", "deadzone_frac", default=0.8)
        ),
        "skill_predictor_attend_image": as_bool(
            _at(config, "skill_predictor", "token_access", "image", default=True)
        ),
        "skill_predictor_attend_language": as_bool(
            _at(config, "skill_predictor", "token_access", "language", default=True)
        ),
        "tokenizer_max_length": 200,
    }


def _validate_predictor_checkpoint(
    checkpoint: Path, levels: list[int], contract: dict
) -> None:
    config_path = checkpoint / "config.json"
    weights_path = checkpoint / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"Incomplete predictor checkpoint: {checkpoint}")
    source = json.loads(config_path.read_text())
    if source.get("type") not in {"skill_expert", "skill_aux"}:
        raise ValueError(f"Unsupported predictor checkpoint type {source.get('type')!r}.")
    if not source.get("train_skill_predictor", False):
        raise ValueError(f"Checkpoint has no trained skill predictor: {checkpoint}")
    expected = {
        "skill_fsq_levels": levels,
        "skill_vocab_size": math.prod(levels),
        **contract,
    }
    mismatches = [
        f"{key}: checkpoint={source.get(key)!r}, requested={value!r}"
        for key, value in expected.items()
        if source.get(key) != value
    ]
    if mismatches:
        raise ValueError("Predictor checkpoint contract mismatch: " + "; ".join(mismatches))


def build_settings(config: dict) -> dict:
    project_root = Path(str(config["project_root"])).expanduser()
    dataset_root = project_root / str(config.get("dataset_root", "dataset"))
    outputs_root = project_root / str(config.get("outputs_root", "outputs"))
    source = str(_at(config, "dataset", "source"))
    run_tag = str(_at(config, "dataset", "run"))
    dataset_dir = (
        dataset_root
        / str(_at(config, "dataset", "skillvla_root", default="skillvla_dataset"))
        / source
        / run_tag
        / "skillvla"
    )
    dataset = _dataset_contract(dataset_dir, run_tag)

    train_terminator = as_bool(_at(config, "terminator", "train", default=False))
    train_image_terminator = as_bool(
        _at(config, "image_only_terminator", "train", default=False)
    )
    train_wrist_terminator = as_bool(
        _at(config, "wrist_only_terminator", "train", default=False)
    )
    train_predictor = as_bool(_at(config, "skill_predictor", "train", default=False))
    if not (
        train_terminator
        or train_image_terminator
        or train_wrist_terminator
        or train_predictor
    ):
        raise ValueError(
            "Set terminator.train, image_only_terminator.train, "
            "wrist_only_terminator.train, and/or skill_predictor.train to true; "
            "all false is empty."
        )
    enabled = []
    if train_terminator:
        enabled.append("terminator")
    if train_image_terminator:
        enabled.append("image_terminator")
    if train_wrist_terminator:
        enabled.append("wrist_terminator")
    if train_predictor:
        enabled.append("predictor")
    mode = "_".join(enabled)

    pi_base = _local_path(
        project_root, str(_at(config, "warm_start", "pi_base", default="models/pi05_base"))
    )
    tokenizer = _local_path(
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
    fsq_value = str(_at(config, "warm_start", "fsq", default="") or "").strip()
    fsq_path = _local_path(project_root, fsq_value) if fsq_value else dataset_dir.parent / "FSQ.pt"
    predictor_value = str(
        _at(config, "warm_start", "predictor_checkpoint", default="") or ""
    ).strip()
    predictor_checkpoint = _local_path(project_root, predictor_value) if predictor_value else None
    predictor_contract = _predictor_contract(config)

    if (
        train_terminator
        or train_image_terminator
        or train_wrist_terminator
    ) and not fsq_path.is_file():
        raise FileNotFoundError(f"FSQ terminator checkpoint not found: {fsq_path}")
    if train_predictor:
        if not (pi_base / "model.safetensors").is_file():
            raise FileNotFoundError(f"pi0.5 predictor base not found: {pi_base}")
        required_tokenizer = ("config.json", "tokenizer_config.json", "tokenizer.json")
        missing = [name for name in required_tokenizer if not (tokenizer / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Tokenizer is incomplete at {tokenizer}: missing={missing}")
        if predictor_checkpoint is not None:
            _validate_predictor_checkpoint(
                predictor_checkpoint, dataset["levels"], predictor_contract
            )

    suffix = str(_at(config, "run", "suffix", default="") or "").strip().strip("_")
    if suffix and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", suffix) is None:
        raise ValueError("run.suffix contains unsupported characters.")
    batch_size = int(_at(config, "training", "dataloader", "batch_size", default=32))
    num_gpus = int(_at(config, "training", "dataloader", "gpus", default=1))
    steps = int(_at(config, "training", "schedule", "steps", default=100000))
    warmup_steps = int(
        _at(config, "training", "schedule", "warmup_steps", default=1000)
    )
    decay_steps = int(
        _at(config, "training", "schedule", "lr_decay_steps", default=30000)
    )
    scheduler_mode = str(
        _at(config, "training", "schedule", "lr_mode", default="warmup_constant")
    ).strip().lower()
    if min(batch_size, num_gpus, steps) <= 0:
        raise ValueError("Batch size, GPU count, and steps must be positive.")
    if scheduler_mode not in {"warmup_constant", "cosine_decay"}:
        raise ValueError("training.schedule.lr_mode must be warmup_constant or cosine_decay.")
    if warmup_steps < 0 or decay_steps <= 0:
        raise ValueError("Invalid scheduler step counts.")

    run_name = f"bs{batch_size}_{source}_{run_tag}"
    if suffix:
        run_name += f"_{suffix}"
    base_lr = float(_at(config, "training", "optimizer", "base_lr", default=2.5e-5))
    settings = {
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        "skillvla_dataset_dir": dataset_dir,
        "repo_id": f"dohyeon/{source}",
        "pi_base": pi_base,
        "tokenizer_path": tokenizer,
        "fsq_path": fsq_path,
        "predictor_checkpoint_path": predictor_checkpoint or "",
        "train_terminator": train_terminator,
        "train_image_only_terminator": train_image_terminator,
        "train_wrist_only_terminator": train_wrist_terminator,
        "train_skill_predictor": train_predictor,
        "training_mode": mode,
        "skill_fsq_levels": "[" + ",".join(str(level) for level in dataset["levels"]) + "]",
        "skill_vocab_size": math.prod(dataset["levels"]),
        "max_state_dim": dataset["state_dim"],
        "max_action_dim": dataset["action_dim"],
        "terminator_freeze_vision_encoder": as_bool(
            _at(config, "terminator", "freeze_vision", default=True)
        ),
        "terminator_end_target_sigma": float(
            _at(config, "terminator", "end_target_sigma", default=2.0)
        ),
        "terminator_end_pos_weight": float(
            _at(config, "terminator", "end_pos_weight", default=1.0)
        ),
        "terminator_termination_only": as_bool(
            _at(config, "terminator", "termination_only", default=False)
        ),
        "image_only_terminator_freeze_vision_encoder": as_bool(
            _at(config, "image_only_terminator", "freeze_vision", default=True)
        ),
        "image_only_terminator_end_target_sigma": float(
            _at(config, "image_only_terminator", "end_target_sigma", default=2.0)
        ),
        "image_only_terminator_end_pos_weight": float(
            _at(config, "image_only_terminator", "end_pos_weight", default=1.0)
        ),
        "image_only_terminator_termination_only": as_bool(
            _at(config, "image_only_terminator", "termination_only", default=False)
        ),
        "wrist_only_terminator_freeze_vision_encoder": as_bool(
            _at(config, "wrist_only_terminator", "freeze_vision", default=True)
        ),
        "wrist_only_terminator_end_target_sigma": float(
            _at(config, "wrist_only_terminator", "end_target_sigma", default=2.0)
        ),
        "wrist_only_terminator_end_pos_weight": float(
            _at(config, "wrist_only_terminator", "end_pos_weight", default=1.0)
        ),
        "wrist_only_terminator_termination_only": as_bool(
            _at(config, "wrist_only_terminator", "termination_only", default=False)
        ),
        **predictor_contract,
        "skill_predictor_lr_scale": float(
            _at(config, "training", "optimizer", "predictor_lr_scale", default=1.0)
        ),
        "skill_predictor_lora_lr_scale": float(
            _at(config, "training", "optimizer", "predictor_lora_lr_scale", default=10.0)
        ),
        "terminator_lr_scale": float(
            _at(config, "training", "optimizer", "terminator_lr_scale", default=1.0)
        ),
        "image_only_terminator_lr_scale": float(
            _at(
                config,
                "training",
                "optimizer",
                "image_only_terminator_lr_scale",
                default=1.0,
            )
        ),
        "wrist_only_terminator_lr_scale": float(
            _at(
                config,
                "training",
                "optimizer",
                "wrist_only_terminator_lr_scale",
                default=1.0,
            )
        ),
        "optimizer_grad_clip_norm": float(
            _at(config, "training", "optimizer", "grad_clip_norm", default=1.0)
        ),
        "gradient_checkpointing": as_bool(
            _at(config, "training", "gradient_checkpointing", default=False)
        ),
        "lr": base_lr * num_gpus,
        "batch_size": batch_size,
        "num_workers": int(_at(config, "training", "dataloader", "workers", default=4)),
        "num_gpus": num_gpus,
        "steps": steps,
        "scheduler_mode": scheduler_mode,
        "scheduler_warmup_steps": warmup_steps,
        "scheduler_decay_steps": decay_steps,
        "log_freq": int(_at(config, "training", "schedule", "log_every", default=100)),
        "save_freq": int(_at(config, "training", "schedule", "save_every", default=5000)),
        "pt_run_name": run_name,
        "pt_output_dir": outputs_root / "skillVLA_terminator" / run_name,
        "wandb_enable": as_bool(_at(config, "logging", "wandb", "enable", default=True)),
        "wandb_project": str(
            _at(config, "logging", "wandb", "project", default="VLA_terminator")
        ),
        "train_partition": ",".join(as_list(config.get("train_partition", ["big"]))) or "big",
        "train_qos": str(config.get("train_qos", "big_qos")),
        "train_gres": str(_at(config, "slurm", "gres", default="gpu:1")),
        "train_cpus_per_task": int(_at(config, "slurm", "cpus", default=10)),
        "train_mem": str(_at(config, "slurm", "memory", default="128G")),
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
