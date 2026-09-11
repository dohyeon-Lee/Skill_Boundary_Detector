#!/usr/bin/env python3
"""Resolve Stage-2 training on a fine-tuning SkillVLA dataset.

Two initialization contracts are supported:

* ``stage2`` continues a complete DSBC checkpoint's noise predictor and its
  optional latent or self-routed skill predictor.
* ``stage1`` runs the ordinary Stage-2 recipe from a Stage-1 checkpoint while
  replacing only its training dataset with the FT dataset selected here.

FT intentionally does not define a second copy of the DSBC architecture
options.  The selected Stage-2 checkpoint is the architecture contract, so its
predictor type, reader, LoRA topology, losses, and freeze policy remain exactly
the same on the fine-tuning dataset.
"""

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

_PROPRIO_GROUNDING_MODES = {"none", "episode_start_xyz"}


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


def _relocate_checkpoint_reference(
    project_root: Path,
    value: object,
    *,
    field: str,
    require_local: bool = False,
) -> str:
    """Rebase a checkpoint-owned local path without rewriting Hub references."""
    text = str(value or "").strip()
    if not text or text == "null":
        return ""

    path = Path(text).expanduser()
    if path.is_absolute():
        resolved = _relocate_project_path(project_root, path)
        if require_local and not resolved.exists():
            raise FileNotFoundError(
                f"Rebased Stage-2 {field} not found: {resolved} "
                f"(checkpoint value: {text})"
            )
        return str(resolved)

    # Project-relative checkpoint paths are local when their rebased target
    # exists. Otherwise preserve the original string: values such as
    # ``namespace/model`` may intentionally be Hugging Face repo IDs.
    candidate = project_root / path
    if candidate.exists():
        return str(candidate)
    return text


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
    proprio_grounding = str(
        info.get("proprio_grounding", "none") or "none"
    ).strip().lower().replace("-", "_")
    if proprio_grounding not in _PROPRIO_GROUNDING_MODES:
        raise ValueError(
            "Invalid FT dataset proprio_grounding: "
            f"{proprio_grounding!r} at {dataset_dir}."
        )
    return {
        "levels": levels,
        "state_dim": int(features["observation.state"]["shape"][0]),
        "action_dim": int(features["action"]["shape"][0]),
        "repo_id": str(info.get("repo_id") or ""),
        "proprio_grounding": proprio_grounding,
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


def _select_ft_dataset(
    source_root: Path,
    configured_run: str,
    parent_fsq: Path,
) -> Path:
    """Select an explicit run or the unique byte-compatible FT dataset.

    PT and FT builders intentionally use different human-readable run suffixes
    (for example ``_pt_grounded`` versus ``_ft_grounded``). The stable skill
    identity is the serialized FSQ module, not that folder name.
    """
    if configured_run:
        return source_root / configured_run / "skillvla"
    if not source_root.is_dir():
        raise FileNotFoundError(f"FT dataset source not found: {source_root}")
    if not parent_fsq.is_file():
        raise FileNotFoundError(f"Stage-2 FSQ checkpoint not found: {parent_fsq}")
    matches = [
        run_dir / "skillvla"
        for run_dir in sorted(source_root.iterdir())
        if run_dir.is_dir()
        and (run_dir / "skillvla/meta/info.json").is_file()
        and (run_dir / "FSQ.pt").is_file()
        and filecmp.cmp(parent_fsq, run_dir / "FSQ.pt", shallow=False)
    ]
    if len(matches) == 1:
        return matches[0]
    available = sorted(
        child.name for child in source_root.iterdir() if child.is_dir()
    )
    if not matches:
        raise FileNotFoundError(
            "No byte-compatible FT SkillVLA run found below "
            f"{source_root}. Available runs: {available}. Set dataset.run "
            "explicitly after building the dataset with the Stage-2 FSQ.pt."
        )
    raise ValueError(
        "Multiple byte-compatible FT SkillVLA runs were found below "
        f"{source_root}: {[path.parent.name for path in matches]}. "
        "Set dataset.run explicitly."
    )


def _build_from_stage2(config: dict) -> dict:
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
    if stage2_mode != "dsbc":
        raise ValueError(
            "FT continues the Stage-2 DSBC predictor modules and "
            f"therefore requires a DSBC parent, got stage2_mode={stage2_mode!r}."
        )
    latent_predictor_enabled = as_bool(
        parent.get("dsbc_latent_predictor_enabled", False)
    )
    skill_predictor_enabled = as_bool(
        parent.get("dsbc_skill_predictor_enabled", False)
    )
    if latent_predictor_enabled and skill_predictor_enabled:
        raise ValueError(
            "Invalid Stage-2 checkpoint: latent and self-routed skill "
            "predictors cannot both be enabled."
        )
    skill_predictor_all_layers = as_bool(
        parent.get("skill_predictor_all_layers", False)
    )
    skill_predictor_lora = as_bool(parent.get("skill_predictor_lora", False))
    skill_predictor_freeze_lora = as_bool(
        parent.get("dsbc_skill_predictor_freeze_lora", False)
    )
    skill_hard_weight = float(parent.get("dsbc_skill_hard_weight", 1.0))
    skill_ste_weight = float(parent.get("dsbc_skill_ste_weight", 0.1))
    skill_timesteps = int(parent.get("dsbc_skill_timesteps", 2))
    skill_samples_per_skill = int(
        parent.get("dsbc_skill_samples_per_skill", 3)
    )
    if skill_predictor_enabled:
        if as_bool(parent.get("skill_flow_latent_best_of_n_enabled", False)):
            raise ValueError(
                "Invalid Stage-2 checkpoint: the self-routed skill predictor "
                "requires a latent-free Stage-1 prior."
            )
        if not math.isfinite(skill_hard_weight) or skill_hard_weight < 0.0:
            raise ValueError(
                "Invalid Stage-2 checkpoint dsbc_skill_hard_weight."
            )
        if not math.isfinite(skill_ste_weight) or skill_ste_weight < 0.0:
            raise ValueError(
                "Invalid Stage-2 checkpoint dsbc_skill_ste_weight."
            )
        if skill_hard_weight + skill_ste_weight <= 0.0:
            raise ValueError(
                "The Stage-2 skill predictor needs a positive hard or STE "
                "loss weight."
            )
        if skill_timesteps <= 0 or skill_samples_per_skill <= 0:
            raise ValueError(
                "The Stage-2 skill predictor timesteps and samples_per_skill "
                "must be positive."
            )
        if skill_predictor_freeze_lora and not skill_predictor_lora:
            raise ValueError(
                "Invalid Stage-2 checkpoint: freezing skill-predictor LoRA "
                "requires skill_predictor_lora=true."
            )
    if as_bool(parent.get("train_terminator", False)):
        raise ValueError("Stage-2 FT expects a terminator-free parent checkpoint.")
    training_skill_source = str(parent.get("training_skill_source", "gt")).lower()
    if training_skill_source != "gt":
        raise ValueError(
            "FT requires a Stage-2 checkpoint with training_skill_source='gt'. "
            "For self-routed skill learning, GT is the local-search center; "
            "the predicted hard code still drives the noise predictor."
        )
    levels = [int(value) for value in parent.get("skill_fsq_levels", [])]
    if not levels or math.prod(levels) != int(parent.get("skill_vocab_size", 0)):
        raise ValueError("Invalid Stage-2 FSQ geometry in the warm-start checkpoint.")
    parent_proprio_grounding = str(
        parent.get("proprio_grounding", "none") or "none"
    ).strip().lower().replace("-", "_")
    if parent_proprio_grounding not in _PROPRIO_GROUNDING_MODES:
        raise ValueError(
            "Invalid Stage-2 proprio_grounding in the warm-start checkpoint: "
            f"{parent_proprio_grounding!r}."
        )

    parent_fsq = _relocate_project_path(project_root, parent.get("fsq_path"))
    policy_dino_model_path = _relocate_checkpoint_reference(
        project_root,
        parent.get("dino_model_path", "models/dinov3-vitl16"),
        field="dino_model_path",
        require_local=True,
    )
    policy_tokenizer_path = _relocate_checkpoint_reference(
        project_root,
        parent.get("tokenizer_path"),
        field="tokenizer_path",
        require_local=True,
    )
    policy_vlm_base_path = _relocate_checkpoint_reference(
        project_root,
        parent.get("vlm_base_path", "models/pi05_base"),
        field="vlm_base_path",
        require_local=True,
    )
    policy_stage1_checkpoint_path = _relocate_checkpoint_reference(
        project_root,
        parent.get("stage1_checkpoint_path"),
        field="stage1_checkpoint_path",
    )
    policy_skill_predictor_checkpoint_path = _relocate_checkpoint_reference(
        project_root,
        parent.get("skill_predictor_checkpoint_path"),
        field="skill_predictor_checkpoint_path",
    )
    policy_terminator_dino_model_path = _relocate_checkpoint_reference(
        project_root,
        parent.get("terminator_dino_model_path"),
        field="terminator_dino_model_path",
        require_local=True,
    )
    source = _safe_name(_at(config, "dataset", "source", default=""), field="dataset.source")
    configured_run = str(_at(config, "dataset", "run", default="") or "").strip()
    if configured_run:
        configured_run = _safe_name(configured_run, field="dataset.run")
    skillvla_root = dataset_root / str(
        _at(config, "dataset", "skillvla_root", default="skillvla_dataset")
    )
    dataset_dir = _select_ft_dataset(
        skillvla_root / source,
        configured_run,
        parent_fsq,
    )
    contract = _dataset_contract(dataset_dir)
    if contract["levels"] != levels:
        raise ValueError(
            f"FT dataset FSQ levels {contract['levels']} do not match Stage-2 {levels}."
        )
    if contract["state_dim"] > int(parent["max_state_dim"]):
        raise ValueError("FT dataset state dimension exceeds the Stage-2 projection size.")
    if contract["action_dim"] > int(parent["max_action_dim"]):
        raise ValueError("FT dataset action dimension exceeds the Stage-2 projection size.")
    if contract["proprio_grounding"] != parent_proprio_grounding:
        raise ValueError(
            "FT dataset/Stage-2 proprio grounding mismatch: "
            f"dataset={contract['proprio_grounding']!r}, "
            f"stage2={parent_proprio_grounding!r}."
        )
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
    suffix = str(_at(config, "run", "suffix", default="") or "").strip().strip("_")
    if explicit_run:
        run_name = _safe_name(explicit_run, field="run.name")
    else:
        run_name = f"{stage2_run}_{checkpoint}_{source}_ft"
    if suffix:
        run_name += f"_{_safe_name(suffix, field='run.suffix')}"
    if len(run_name.encode()) > 240:
        parent_id = hashlib.sha1(stage2_run.encode()).hexdigest()[:8]
        run_name = f"{source}_{stage2_mode}_{checkpoint}_{parent_id}_ft"
        if suffix:
            run_name += f"_{_safe_name(suffix, field='run.suffix')}"

    train_scope = ["noise_predictor"]
    if latent_predictor_enabled:
        train_scope.append("latent_predictor")
    if skill_predictor_enabled:
        train_scope.append("skill_predictor")

    return {
        "initialization_mode": "stage2",
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        "skillvla_dataset_dir": dataset_dir,
        "repo_id": contract["repo_id"] or f"skillvla/{source}",
        "stage2_checkpoint_path": stage2_path,
        "parent_stage2_run": stage2_run,
        "parent_stage2_checkpoint": checkpoint,
        "stage2_mode": stage2_mode,
        "ft_train_scope": "+".join(train_scope),
        "dsbc_noise_output_mode": str(
            parent.get("dsbc_noise_output_mode", "per_step")
        ),
        "dsbc_noise_vlm_tokens": str(
            parent.get(
                "dsbc_noise_vlm_tokens",
                "full"
                if as_bool(parent.get("dsbc_noise_vlm_enabled", False))
                else "none",
            )
        ),
        "dsbc_noise_vlm_enabled": as_bool(
            parent.get("dsbc_noise_vlm_enabled", False)
        ),
        "dsbc_latent_predictor_mode": str(
            parent.get("dsbc_latent_predictor_mode", "skill_start")
        ),
        "dsbc_latent_predictor_vlm_tokens": str(
            parent.get("dsbc_latent_predictor_vlm_tokens", "image_language")
        ),
        "dsbc_latent_predictor_lora": as_bool(
            parent.get("dsbc_latent_predictor_lora", False)
        ),
        "dsbc_latent_predictor_enabled": latent_predictor_enabled,
        "dsbc_skill_predictor_enabled": skill_predictor_enabled,
        "skill_predictor_all_layers": skill_predictor_all_layers,
        "skill_predictor_lora": skill_predictor_lora,
        "dsbc_skill_predictor_freeze_lora": skill_predictor_freeze_lora,
        "dsbc_skill_hard_weight": skill_hard_weight,
        "dsbc_skill_ste_weight": skill_ste_weight,
        "dsbc_skill_timesteps": skill_timesteps,
        "dsbc_skill_samples_per_skill": skill_samples_per_skill,
        "training_skill_source": training_skill_source,
        # A complete Stage-2 checkpoint carries its architecture, but historical
        # checkpoints may contain absolute paths from the machine that created
        # them. Export rebased overrides so both fresh FT and resume are portable.
        "policy_dino_model_path": policy_dino_model_path,
        "policy_tokenizer_path": policy_tokenizer_path,
        "policy_vlm_base_path": policy_vlm_base_path,
        "policy_stage1_checkpoint_path": policy_stage1_checkpoint_path,
        "policy_skill_predictor_checkpoint_path": policy_skill_predictor_checkpoint_path,
        "policy_fsq_path": parent_fsq,
        "policy_terminator_dino_model_path": policy_terminator_dino_model_path,
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


def _build_from_stage1(config: dict) -> dict:
    """Use a Stage-2 checkpoint as a recipe, but initialize from its Stage 1."""
    settings = _build_from_stage2(config)
    predictor_scope = str(settings["ft_train_scope"])
    stage2_path = Path(settings["stage2_checkpoint_path"])
    train_config_path = stage2_path / "train_config.json"
    train_config = _read_json(train_config_path, "Stage-2 training config")
    train_policy = train_config.get("policy")
    if not isinstance(train_policy, dict):
        raise ValueError(
            f"Stage-2 train_config has no policy mapping: {train_config_path}"
        )
    if train_policy.get("type", train_policy.get("model_type")) != "skill_vla_stage2":
        raise ValueError(
            f"Stage-2 train_config has the wrong policy type: {train_config_path}"
        )

    # config_path reconstructs a fresh SkillVLAStage2Policy. Because resume is
    # forced false and --policy.path is never passed, no Stage-2 tensors are
    # loaded; the recorded stage1_checkpoint_path performs the sole warm start.
    run_name = str(settings["pt_run_name"])
    if not str(_at(config, "run", "name", default="") or "").strip():
        run_name += "_fresh"
    output_dir = Path(settings["pt_output_dir"]).parent / run_name
    scheduler = train_config.get("scheduler") or {}
    wandb = train_config.get("wandb") or {}
    optimizer = train_config.get("optimizer") or {}
    settings.update(
        {
            "initialization_mode": "stage1",
            "stage2_train_config_path": train_config_path,
            "ft_train_scope": (
                f"fresh_{predictor_scope}_from_recorded_stage1"
            ),
            "pt_run_name": run_name,
            "pt_output_dir": output_dir,
            "batch_size": int(train_config.get("batch_size", settings["batch_size"])),
            "num_workers": int(
                train_config.get("num_workers", settings["num_workers"])
            ),
            "steps": int(train_config.get("steps", settings["steps"])),
            "log_freq": int(train_config.get("log_freq", settings["log_freq"])),
            "save_freq": int(
                train_config.get("save_freq", settings["save_freq"])
            ),
            "lr": float(optimizer.get("lr", settings["lr"])),
            "scheduler_mode": str(
                scheduler.get("type", settings["scheduler_mode"])
            ),
            "scheduler_warmup_steps": int(
                scheduler.get(
                    "num_warmup_steps", settings["scheduler_warmup_steps"]
                )
            ),
            "wandb_enable": as_bool(
                wandb.get("enable", settings["wandb_enable"])
            ),
            "wandb_project": str(
                wandb.get("project", settings["wandb_project"])
            ),
        }
    )
    return settings


def build_settings(config: dict) -> dict:
    initialization = _at(config, "initialization", default={})
    if initialization is None:
        initialization = {}
    if not isinstance(initialization, dict):
        raise ValueError("initialization must be a mapping with mode: stage2|stage1.")
    mode = str(initialization.get("mode", "stage2")).strip().lower()
    if mode == "stage2":
        return _build_from_stage2(config)
    if mode == "stage1":
        return _build_from_stage1(config)
    raise ValueError("initialization.mode must be stage2|stage1.")


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
