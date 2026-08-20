#!/usr/bin/env python3
"""Resolve Stage-2 fine-tuning from a complete Stage-2 checkpoint."""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import as_bool, as_list, get_value, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "ft_train_config.yaml"


def _at(config: dict, *path: str, default=None):
    value = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _safe_name(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text or text in {".", ".."} or "/" in text or "\0" in text:
        raise ValueError(f"{field} must be a non-empty folder name, got {text!r}.")
    return text


def _relocate_project_path(project_root: Path, value: object) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute():
        return project_root / path
    if path.exists():
        return path
    for anchor in (
        "dataset",
        "dataset_filtered",
        "dataset_ABC",
        "models",
        "outputs",
        "outputs_filtered",
        "outputs_ABC",
    ):
        if anchor in path.parts:
            return project_root.joinpath(*path.parts[path.parts.index(anchor) :])
    return path


def _read_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return json.loads(path.read_text())


def _dataset_contract(dataset_dir: Path) -> dict:
    info = _read_json(dataset_dir / "meta/info.json", "FT dataset metadata")
    levels = [int(value) for value in info.get("skill_fsq_levels", [])]
    if not levels or any(value <= 1 for value in levels):
        raise ValueError(f"Invalid FT dataset skill_fsq_levels: {levels}")
    features = info.get("features", {})
    return {
        "levels": levels,
        "state_dim": int(features["observation.state"]["shape"][0]),
        "action_dim": int(features["action"]["shape"][0]),
        "repo_id": str(info.get("repo_id") or ""),
    }


def _require_same_fsq(left: Path, right: Path, *, label: str) -> None:
    if not left.is_file():
        raise FileNotFoundError(f"Stage-2 FSQ checkpoint not found: {left}")
    if not right.is_file():
        raise FileNotFoundError(f"{label} FSQ checkpoint not found: {right}")
    if not filecmp.cmp(left, right, shallow=False):
        raise ValueError(
            f"{label} FSQ.pt is not byte-identical to the Stage-2 checkpoint FSQ.pt. "
            "Equal FSQ levels are insufficient because token IDs would mean different skills."
        )


def build_settings(config: dict) -> dict:
    project_root = Path(str(get_value(config, "project_root"))).expanduser()
    dataset_root = project_root / str(get_value(config, "dataset_root", "dataset"))
    outputs_root = project_root / str(get_value(config, "outputs_root", "outputs"))

    outputs_subdir = _safe_name(
        _at(config, "warm_start", "outputs_subdir", default="skillVLA_stage2"),
        field="warm_start.outputs_subdir",
    )
    stage2_run = _safe_name(
        _at(config, "warm_start", "stage2_run", default=""),
        field="warm_start.stage2_run",
    )
    checkpoint = _safe_name(
        _at(config, "warm_start", "checkpoint", default="last"),
        field="warm_start.checkpoint",
    )
    stage2_path = (
        outputs_root
        / outputs_subdir
        / stage2_run
        / "checkpoints"
        / checkpoint
        / "pretrained_model"
    )
    for name in (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "train_config.json",
    ):
        if not (stage2_path / name).is_file():
            raise FileNotFoundError(f"Incomplete Stage-2 warm start; missing {stage2_path / name}")
    parent = _read_json(stage2_path / "config.json", "Stage-2 warm-start config")
    if parent.get("type", parent.get("model_type")) != "skill_vla_stage2":
        raise ValueError(f"FT requires a skill_vla_stage2 checkpoint: {stage2_path}")
    stage2_mode = str(parent.get("stage2_mode", "likelihood")).strip().lower()
    if stage2_mode not in {"likelihood", "dsbc"}:
        raise ValueError(f"Invalid parent stage2_mode={stage2_mode!r}.")
    if as_bool(parent.get("train_terminator", False)):
        raise ValueError("Stage-2 FT expects a terminator-free parent checkpoint.")
    training_skill_source = str(parent.get("training_skill_source", "gt")).lower()
    if training_skill_source != "gt":
        raise ValueError(
            "This FT pipeline teacher-forces dataset skill codes and requires "
            "a Stage-2 checkpoint with training_skill_source='gt'."
        )
    levels = [int(value) for value in parent.get("skill_fsq_levels", [])]
    if not levels or math.prod(levels) != int(parent.get("skill_vocab_size", 0)):
        raise ValueError("Invalid Stage-2 FSQ geometry in the warm-start checkpoint.")

    parent_fsq = _relocate_project_path(project_root, parent.get("fsq_path"))
    source = _safe_name(_at(config, "dataset", "source", default=""), field="dataset.source")
    configured_run = str(_at(config, "dataset", "run", default="") or "").strip()
    if configured_run:
        configured_run = _safe_name(configured_run, field="dataset.run")
    run_tag = configured_run or parent_fsq.parent.name
    if configured_run and configured_run != parent_fsq.parent.name:
        raise ValueError(
            "dataset.run must match the Stage-2 FSQ run: "
            f"configured={configured_run!r}, stage2={parent_fsq.parent.name!r}."
        )
    skillvla_root = dataset_root / str(
        _at(config, "dataset", "skillvla_root", default="skillvla_dataset")
    )
    dataset_dir = skillvla_root / source / run_tag / "skillvla"
    contract = _dataset_contract(dataset_dir)
    if contract["levels"] != levels:
        raise ValueError(
            f"FT dataset FSQ levels {contract['levels']} do not match Stage-2 {levels}."
        )
    if contract["state_dim"] > int(parent["max_state_dim"]):
        raise ValueError("FT dataset state dimension exceeds the Stage-2 projection size.")
    if contract["action_dim"] > int(parent["max_action_dim"]):
        raise ValueError("FT dataset action dimension exceeds the Stage-2 projection size.")
    _require_same_fsq(parent_fsq, dataset_dir.parent / "FSQ.pt", label="FT dataset")

    scheduler_mode = str(
        _at(config, "training", "schedule", "lr_mode", default="warmup_constant")
    ).strip().lower()
    if scheduler_mode not in {"warmup_constant", "cosine_decay"}:
        raise ValueError("training.schedule.lr_mode must be warmup_constant|cosine_decay.")
    warmup_steps = int(
        _at(config, "training", "schedule", "warmup_steps", default=1000)
    )
    decay_steps = int(
        _at(config, "training", "schedule", "lr_decay_steps", default=30000)
    )
    if warmup_steps < 0 or decay_steps <= 0:
        raise ValueError("Scheduler warmup must be non-negative and decay steps positive.")
    num_gpus = int(_at(config, "training", "dataloader", "gpus", default=1))
    batch_size = int(
        _at(config, "training", "dataloader", "batch_size", default=32)
    )
    if num_gpus <= 0 or batch_size <= 0:
        raise ValueError("training dataloader gpus and batch_size must be positive.")
    base_lr = float(
        _at(config, "training", "optimizer", "base_lr", default=2.5e-5)
    )
    if base_lr <= 0.0:
        raise ValueError("training.optimizer.base_lr must be positive.")
    num_workers = int(
        _at(config, "training", "dataloader", "workers", default=8)
    )
    steps = int(_at(config, "training", "schedule", "steps", default=100000))
    log_freq = int(
        _at(config, "training", "schedule", "log_every", default=100)
    )
    save_freq = int(
        _at(config, "training", "schedule", "save_every", default=10000)
    )
    if num_workers < 0:
        raise ValueError("training.dataloader.workers must be non-negative.")
    if steps <= 0 or log_freq <= 0 or save_freq <= 0:
        raise ValueError("Training steps, log_every, and save_every must be positive.")

    explicit_run = str(_at(config, "run", "name", default="") or "").strip()
    if explicit_run:
        run_name = _safe_name(explicit_run, field="run.name")
    else:
        parent_id = hashlib.sha1(stage2_run.encode()).hexdigest()[:8]
        run_name = f"{source}_{stage2_mode}_ft_{checkpoint}_{parent_id}"

    return {
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        "skillvla_dataset_dir": dataset_dir,
        "repo_id": contract["repo_id"] or f"skillvla/{source}",
        "stage2_checkpoint_path": stage2_path,
        "parent_stage2_run": stage2_run,
        "parent_stage2_checkpoint": checkpoint,
        "stage2_mode": stage2_mode,
        "training_skill_source": training_skill_source,
        "gradient_checkpointing": as_bool(
            _at(config, "training", "gradient_checkpointing", default=False)
        ),
        "lr": base_lr * num_gpus,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "num_gpus": num_gpus,
        "scheduler_mode": scheduler_mode,
        "scheduler_warmup_steps": warmup_steps,
        "scheduler_decay_steps": decay_steps,
        "steps": steps,
        "log_freq": log_freq,
        "save_freq": save_freq,
        "pt_run_name": run_name,
        "pt_output_dir": outputs_root / "skillVLA_FT" / run_name,
        "wandb_enable": as_bool(
            _at(config, "logging", "wandb", "enable", default=True)
        ),
        "wandb_project": str(
            _at(config, "logging", "wandb", "project", default="VLA_stage2_FT")
        ),
        "train_partition": ",".join(
            as_list(get_value(config, "train_partition", ["debug"]))
        )
        or "debug",
        "train_qos": str(get_value(config, "train_qos", "base_qos")),
        "train_gres": str(_at(config, "slurm", "gres", default="gpu:1")),
        "train_cpus_per_task": int(
            _at(config, "slurm", "cpus", default=12)
        ),
        "train_mem": str(_at(config, "slurm", "memory", default="256G")),
        "train_time": str(_at(config, "slurm", "time", default="48:00:00")),
        "train_nodelist": str(get_value(config, "train_nodelist", "")),
        "train_exclude_nodes": ",".join(
            as_list(get_value(config, "train_exclude_nodes", []))
        ),
    }


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
