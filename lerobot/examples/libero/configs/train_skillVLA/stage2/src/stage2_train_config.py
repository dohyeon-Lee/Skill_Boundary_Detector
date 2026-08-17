#!/usr/bin/env python3
"""Resolve clean BayesVLA-style Stage-2 training settings.

Stage 2 assembles two independently trained frozen sources:

* ``warm_start.stage1_run``: a cond_gemma (arch0-family) Stage-1 prior. It
  needs neither predictor nor terminator.
* ``warm_start.predictor``: any Stage-1/skill_aux run whose config records
  ``train_skill_predictor=true``; its frozen VLM provides the cross-attention
  memory and its architecture fields are inherited verbatim.

Terminators stay out of Stage 2 entirely; evaluation attaches one externally.
"""

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

_CONDITIONING_ROUTES = {
    "state_cond",
    "state_skill_cond",
    "state_skill_only_cond",
    "stateonly_cond",
    "skillonly_cond",
    "visiononly_cond",
}

# Module-shape fields adopted verbatim from the predictor checkpoint's config.
_PREDICTOR_MODULE_FIELDS = (
    "skill_predictor_vlm_variant",
    "skill_predictor_image_size",
    "skill_predictor_reader_tokens",
    "skill_predictor_reader_depth",
    "skill_predictor_reader_heads",
    "skill_predictor_all_layers",
    "skill_predictor_detach_vlm",
    "skill_predictor_lora",
    "skill_predictor_lora_targets",
    "skill_predictor_lora_rank",
    "skill_predictor_lora_alpha",
    "skill_predictor_lora_dropout",
    "skill_predictor_deadzone_frac",
    "tokenizer_max_length",
)


def _at(config: dict, *path: str, default=None):
    value = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


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


def _normalize_route(route: object) -> str:
    normalized = str(route or "").strip().lower()
    return "skillonly_cond" if normalized == "skill_cond" else normalized


def _require_stage1_prior_contract(config: dict, checkpoint: Path) -> None:
    if config.get("type") != "skill_expert":
        raise ValueError(
            f"Stage 2 requires policy.type=skill_expert, got {config.get('type')!r} at {checkpoint}."
        )
    architecture = config.get("architecture") or (
        "cond_gemma" if "conditioning_route" in config else ""
    )
    if architecture != "cond_gemma":
        raise ValueError(
            "Stage 2 is implemented on the cond_gemma (arch0-family) Stage-1 "
            f"prior; got architecture={architecture!r} at {checkpoint}."
        )
    if config.get("action_expert_variant") != "gemma_300m":
        raise ValueError("Stage 2 expects the 18-layer gemma_300m action expert.")
    if config.get("cond_encoder_variant") != "gemma_300m":
        raise ValueError("Stage 2 expects the 18-layer gemma_300m condition encoder.")
    conditioning_route = _normalize_route(config.get("conditioning_route"))
    if conditioning_route not in _CONDITIONING_ROUTES:
        raise ValueError(
            f"Stage 2 expects conditioning_route in {sorted(_CONDITIONING_ROUTES)}, "
            f"got {conditioning_route!r}."
        )
    config["conditioning_route"] = conditioning_route


def _require_predictor_contract(config: dict, checkpoint: Path) -> None:
    if config.get("type") not in {"skill_expert", "skill_aux"}:
        raise ValueError(
            "Predictor source must be a skill_expert or skill_aux checkpoint, got "
            f"{config.get('type')!r} at {checkpoint}."
        )
    if not as_bool(config.get("train_skill_predictor", False)):
        raise ValueError(
            f"Predictor run has no trained predictor (train_skill_predictor=false): {checkpoint}"
        )
    if not (
        as_bool(config.get("skill_predictor_attend_image", False))
        and as_bool(config.get("skill_predictor_attend_language", False))
    ):
        raise ValueError(
            "Stage 2 needs both image and language tokens in the frozen VLM memory."
        )
    missing = [field for field in _PREDICTOR_MODULE_FIELDS if field not in config]
    if missing:
        raise ValueError(
            f"Predictor checkpoint config is missing module fields {missing}: {checkpoint}"
        )


def _fsq_run_tag(config: dict, label: str) -> str:
    """Return the FSQ dataset-run directory name recorded in fsq_path."""
    fsq_path = Path(str(config.get("fsq_path") or ""))
    if not fsq_path.name:
        raise ValueError(f"{label} config records no fsq_path.")
    return fsq_path.parent.name


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


def _pretrained_model_dir(
    outputs_root: Path, run: str, checkpoint: str, label: str
) -> Path:
    path = (
        outputs_root
        / "skillVLA_stage1"
        / run
        / "checkpoints"
        / checkpoint
        / "pretrained_model"
    )
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"{label} config not found: {path / 'config.json'}")
    if not (path / "model.safetensors").is_file():
        raise FileNotFoundError(f"{label} weights not found: {path / 'model.safetensors'}")
    return path


def build_settings(config: dict) -> dict:
    if "auxiliary" in config:
        raise ValueError(
            "Stage 2 no longer trains auxiliaries; remove the 'auxiliary' section. "
            "Predictor and terminator stay frozen (terminators attach at evaluation)."
        )
    project_root = Path(str(config["project_root"])).expanduser()
    dataset_root = project_root / str(config.get("dataset_root", "dataset"))
    outputs_root = project_root / str(config.get("outputs_root", "outputs"))
    stage2_mode = str(config.get("stage2_mode", "likelihood")).strip().lower()
    if stage2_mode not in {"likelihood", "dsbc"}:
        raise ValueError("stage2_mode must be likelihood|dsbc.")

    stage1_run = str(_at(config, "warm_start", "stage1_run", default="") or "").strip()
    stage1_checkpoint = str(
        _at(config, "warm_start", "checkpoint", default="last")
    ).strip()
    if not stage1_run:
        raise ValueError("warm_start.stage1_run must be the exact Stage-1 output directory name.")
    stage1_path = _pretrained_model_dir(
        outputs_root, stage1_run, stage1_checkpoint, "Stage-1 prior"
    )
    stage1_config = _read_json(stage1_path / "config.json", "Stage-1 policy config")
    _require_stage1_prior_contract(stage1_config, stage1_path)

    predictor_path_setting = str(
        _at(config, "warm_start", "predictor", "path", default="") or ""
    ).strip()
    if predictor_path_setting:
        predictor_path = _local_path(project_root, predictor_path_setting)
        for name in ("config.json", "model.safetensors"):
            if not (predictor_path / name).is_file():
                raise FileNotFoundError(
                    f"Predictor {name} not found: {predictor_path / name}"
                )
    else:
        predictor_run = str(
            _at(config, "warm_start", "predictor", "run", default="") or ""
        ).strip()
        if not predictor_run:
            raise ValueError(
                "warm_start.predictor.run (or .path) must name the run providing "
                "the trained frozen-VLM skill predictor."
            )
        predictor_checkpoint = str(
            _at(config, "warm_start", "predictor", "checkpoint", default="last")
        ).strip()
        predictor_path = _pretrained_model_dir(
            outputs_root, predictor_run, predictor_checkpoint, "Predictor"
        )
    predictor_config = _read_json(predictor_path / "config.json", "Predictor policy config")
    _require_predictor_contract(predictor_config, predictor_path)

    stage1_levels = [int(value) for value in stage1_config["skill_fsq_levels"]]
    predictor_levels = [int(value) for value in predictor_config["skill_fsq_levels"]]
    if predictor_levels != stage1_levels:
        raise ValueError(
            f"Predictor FSQ levels {predictor_levels} do not match Stage 1 {stage1_levels}."
        )
    skill_source = str(
        _at(config, "likelihood", "training_skill_source", default="gt")
    ).strip().lower()
    if skill_source not in {"gt", "predictor"}:
        raise ValueError("likelihood.training_skill_source must be gt or predictor.")
    stage1_fsq_run = _fsq_run_tag(stage1_config, "Stage-1")
    predictor_fsq_run = _fsq_run_tag(predictor_config, "Predictor")
    if predictor_fsq_run != stage1_fsq_run:
        raise ValueError(
            "Predictor FSQ run does not match the Stage-1 prior: "
            f"stage1={stage1_fsq_run!r}, predictor={predictor_fsq_run!r}. "
            "Codes with the same index mean different skills across FSQ runs, "
            "so the assembled policy would predict skills in the wrong space."
        )

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
        project_root, predictor_config["tokenizer_path"], marker="models"
    )
    fsq_path = _local_path(
        project_root, stage1_config["fsq_path"], marker=dataset_root.name
    )
    for path, label in (
        (dino_path, "DINO model"),
        (tokenizer_path, "PaliGemma tokenizer"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Inherited {label} not found: {path}")
    if not fsq_path.is_file():
        raise FileNotFoundError(f"Inherited FSQ checkpoint not found: {fsq_path}")

    likelihood_layers = int(_at(config, "likelihood", "layers", default=4))
    if likelihood_layers != 4:
        raise ValueError("BayesVLA-matched Stage 2 fixes likelihood.layers=4.")
    likelihood_vlm_memory = str(
        _at(config, "likelihood", "vlm_memory", default="last")
    ).strip().lower()
    if likelihood_vlm_memory not in {"last", "layer_mix"}:
        raise ValueError(
            "likelihood.vlm_memory must be last|layer_mix, got "
            f"{likelihood_vlm_memory!r}."
        )
    likelihood_gate_lr_scale = float(
        _at(config, "likelihood", "gate_lr_scale", default=1.0)
    )
    if likelihood_gate_lr_scale <= 0.0:
        raise ValueError("likelihood.gate_lr_scale must be positive.")
    dsbc_noise_output_mode = str(
        _at(config, "dsbc", "noise_output_mode", default="shared")
    ).strip().lower()
    if dsbc_noise_output_mode not in {"shared", "per_step"}:
        raise ValueError("dsbc.noise_output_mode must be shared|per_step.")
    dsbc_frs_num_steps = int(
        _at(
            config,
            "dsbc",
            "frs_num_steps",
            default=int(stage1_config["num_inference_steps"]),
        )
    )
    if dsbc_frs_num_steps <= 0:
        raise ValueError("dsbc.frs_num_steps must be positive.")
    dsbc_anchor_seed = int(_at(config, "dsbc", "anchor_seed", default=0))
    if dsbc_anchor_seed < 0:
        raise ValueError("dsbc.anchor_seed must be non-negative.")
    scheduler_mode = str(
        _at(config, "training", "schedule", "lr_mode", default="cosine_decay")
    ).strip().lower()
    if scheduler_mode not in {"cosine_decay", "warmup_constant"}:
        raise ValueError(
            "training.schedule.lr_mode must be 'cosine_decay' or "
            f"'warmup_constant', got {scheduler_mode!r}."
        )
    scheduler_warmup_steps = int(
        _at(config, "training", "schedule", "warmup_steps", default=1000)
    )
    if scheduler_warmup_steps < 0:
        raise ValueError("training.schedule.warmup_steps must be non-negative.")
    scheduler_decay_steps = int(
        _at(config, "training", "schedule", "lr_decay_steps", default=30000)
    )
    if scheduler_decay_steps <= 0:
        raise ValueError("training.schedule.lr_decay_steps must be positive.")
    if "loss" in config:
        raise ValueError(
            "Stage 2 no longer has a 'loss' selector; the flow loss is fixed. "
            "Configure only cumulative_xyz_loss.enabled/weight, like Stage 1."
        )
    cumulative_xyz_config = config.get("cumulative_xyz_loss", {})
    if not isinstance(cumulative_xyz_config, dict):
        raise ValueError("cumulative_xyz_loss must be a mapping.")
    cumulative_xyz_loss_enabled = as_bool(
        cumulative_xyz_config.get("enabled", False)
    )
    cumulative_xyz_loss_weight = float(cumulative_xyz_config.get("weight", 0.5))
    if not math.isfinite(cumulative_xyz_loss_weight) or cumulative_xyz_loss_weight <= 0:
        raise ValueError("cumulative_xyz_loss.weight must be finite and positive.")
    if stage2_mode == "dsbc" and cumulative_xyz_loss_enabled:
        raise ValueError(
            "cumulative_xyz_loss is unavailable in DSBC mode; DSBC trains on FRS noise."
        )
    stage1_skill_end_mask = as_bool(
        stage1_config.get("mask_actions_after_skill_end", False)
    )
    mask_override = config.get("mask_actions_after_skill_end")
    mask_actions_after_skill_end = (
        stage1_skill_end_mask if mask_override is None else as_bool(mask_override)
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
    suffix = str(_at(config, "run", "suffix", default="")).strip().strip("_")
    batch_tag = "batchON" if same_skill_batch_enabled else "batchOFF"
    run_name = f"{stage1_run}_{stage1_checkpoint}_{skill_source}_{batch_tag}"
    if stage2_mode == "dsbc":
        run_name += f"_dsbc_{dsbc_noise_output_mode}_frs{dsbc_frs_num_steps}"
    if mask_actions_after_skill_end != stage1_skill_end_mask:
        # Keep overridden-mask runs out of the inherited-mask output directory.
        run_name += "_nomask" if not mask_actions_after_skill_end else "_endmask"
    if cumulative_xyz_loss_enabled:
        cumulative_weight_label = f"{cumulative_xyz_loss_weight:g}".replace(".", "p")
        run_name += f"_cumxyz{cumulative_weight_label}"
    if likelihood_vlm_memory == "layer_mix":
        run_name += "_layermix"
    if likelihood_gate_lr_scale != 1.0:
        run_name += f"_glr{likelihood_gate_lr_scale:g}".replace(".", "p")
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
        "predictor_checkpoint_path": predictor_path,
        "dino_model_path": dino_path,
        "tokenizer_path": tokenizer_path,
        "fsq_path": fsq_path,
        "architecture": "cond_gemma",
        "architecture_revision": str(
            stage1_config.get("architecture_revision", "skillvla_real_v1")
        ),
        "architecture_label": str(stage1_config.get("architecture_label", "")),
        "stage2_mode": stage2_mode,
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
        "mask_actions_after_skill_end": mask_actions_after_skill_end,
        "num_visual_latents_per_camera": int(
            stage1_config.get("num_visual_latents_per_camera", 32)
        ),
        "visual_perceiver_width": int(
            stage1_config.get("visual_perceiver_width", 1024)
        ),
        "skill_vocab_size": math.prod(stage1_levels),
        "skill_fsq_levels": "[" + ",".join(str(value) for value in stage1_levels) + "]",
        "transition_jitter_pmax": int(stage1_config["transition_jitter_pmax"]),
        "transition_jitter_distribution": stage1_config["transition_jitter_distribution"],
        "train_skill_predictor": True,
        "skill_predictor_vlm_variant": predictor_config["skill_predictor_vlm_variant"],
        "skill_predictor_image_size": int(predictor_config["skill_predictor_image_size"]),
        "skill_predictor_reader_tokens": int(predictor_config["skill_predictor_reader_tokens"]),
        "skill_predictor_reader_depth": int(predictor_config["skill_predictor_reader_depth"]),
        "skill_predictor_reader_heads": int(predictor_config["skill_predictor_reader_heads"]),
        "skill_predictor_all_layers": as_bool(predictor_config["skill_predictor_all_layers"]),
        "skill_predictor_detach_vlm": as_bool(predictor_config["skill_predictor_detach_vlm"]),
        "skill_predictor_lora": as_bool(predictor_config["skill_predictor_lora"]),
        "skill_predictor_lora_targets": str(predictor_config["skill_predictor_lora_targets"]),
        "skill_predictor_lora_rank": int(predictor_config["skill_predictor_lora_rank"]),
        "skill_predictor_lora_alpha": float(predictor_config["skill_predictor_lora_alpha"]),
        "skill_predictor_lora_dropout": float(predictor_config["skill_predictor_lora_dropout"]),
        "skill_predictor_deadzone_frac": float(predictor_config["skill_predictor_deadzone_frac"]),
        "skill_predictor_attend_image": True,
        "skill_predictor_attend_language": True,
        "tokenizer_max_length": int(predictor_config["tokenizer_max_length"]),
        "train_terminator": False,
        "likelihood_num_layers": likelihood_layers,
        "likelihood_cross_attention_heads": 8,
        "likelihood_vlm_memory": likelihood_vlm_memory,
        "likelihood_gate_lr_scale": likelihood_gate_lr_scale,
        "dsbc_noise_output_mode": dsbc_noise_output_mode,
        "dsbc_frs_num_steps": dsbc_frs_num_steps,
        "dsbc_anchor_seed": dsbc_anchor_seed,
        "training_skill_source": skill_source,
        "cumulative_xyz_loss_enabled": cumulative_xyz_loss_enabled,
        "cumulative_xyz_loss_weight": cumulative_xyz_loss_weight,
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
        "scheduler_mode": scheduler_mode,
        "scheduler_warmup_steps": scheduler_warmup_steps,
        "scheduler_decay_steps": scheduler_decay_steps,
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
