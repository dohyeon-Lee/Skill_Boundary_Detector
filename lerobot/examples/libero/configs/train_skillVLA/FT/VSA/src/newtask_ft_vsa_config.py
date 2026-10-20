#!/usr/bin/env python3
"""Resolve NewTask FT for a Stage-1 VSA with a frozen skill-only route.

Every architecture setting is inherited from the source checkpoint's
``config.json``; this resolver only selects the checkpoint, the new-task
dataset, Expert freezing, the optional skill-flow loss, and the optimizer schedule.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_TRAIN_SKILLVLA = _HERE.parents[3]
sys.path.insert(0, str(_TRAIN_SKILLVLA.parent / "train_skills" / "src"))
from train_skills_config import (  # noqa: E402
    as_bool,
    as_list,
    load_stage1_component_config,
    print_shell,
    resolve_path,
    resolve_run_checkpoint,
    resolve_skillvla_dataset_run,
    resolve_transition_randomization,
)

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "vsa_ft_config.yaml"
OUTPUT_GROUP = "skillVLA_FT"
# Arch0--Arch3 share trainable parameters between the skill-only and deployed
# routes, so the frozen-core contract cannot hold for them.  Do not maintain a
# positive architecture-name allow-list here: Stage-1 keeps gaining new
# core-exit families, and their checkpoint contract is the source of truth.
_SHARED_ROUTE_ARCHITECTURES = frozenset(
    f"arch{number}" for number in range(4)
)


def _load_sibling(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_STAGE1 = _load_sibling(
    "stage1_train_config", _TRAIN_SKILLVLA / "PT/src/stage1_train_config.py"
)
_AUX = _load_sibling(
    "terminator_train_config",
    _TRAIN_SKILLVLA / "PT/Terminator/src/terminator_train_config.py",
)
_at = _STAGE1._at


def _checkpoint_dir(project_root: Path, outputs_root: Path, value) -> Path:
    """``{run, checkpoint}`` (searched across output groups) or a legacy path string."""
    if isinstance(value, dict):
        path = resolve_run_checkpoint(outputs_root, "VSA", value, field="warm_start.vsa_checkpoint")
    else:
        path = Path(resolve_path(project_root, str(value).strip())) if str(value or "").strip() else None
    if path is None:
        raise ValueError("warm_start.vsa_checkpoint is required for NewTask FT.")
    if (path / "pretrained_model" / "config.json").is_file():
        path = path / "pretrained_model"
    for name in ("config.json", "model.safetensors", "policy_preprocessor.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(f"Incomplete VSA checkpoint, missing {name}: {path}")
    return path


def _checkpoint_run_and_step(checkpoint: Path) -> tuple[str, str]:
    # <run>/checkpoints/<step>/pretrained_model
    step_dir = checkpoint.parent
    if step_dir.parent.name != "checkpoints":
        raise ValueError(
            "warm_start.vsa_checkpoint must be <run>/checkpoints/<step>/pretrained_model, "
            f"got {checkpoint}."
        )
    step = step_dir.resolve().name if step_dir.name == "last" else step_dir.name
    return step_dir.parent.parent.name, _step_label(step)


def _step_label(step: str) -> str:
    """``170000`` -> ``170k`` (same convention as relabeled_85k); others unchanged."""
    if step.isdigit() and int(step) > 0 and int(step) % 1000 == 0:
        return f"{int(step) // 1000}k"
    return step


def _suffix(config: dict) -> str:
    run_config = config.get("run", {})
    if not isinstance(run_config, dict):
        raise ValueError("run must be a mapping.")
    unknown = set(run_config) - {"suffix", "common_suffix"}
    if unknown:
        raise ValueError(f"Unsupported run settings: {sorted(unknown)}.")
    suffixes = []
    for field in ("suffix", "common_suffix"):
        suffix = str(run_config.get(field, "") or "").strip().strip("_")
        if suffix and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", suffix) is None:
            raise ValueError(f"run.{field} contains unsupported characters.")
        if suffix and suffix not in suffixes:
            suffixes.append(suffix)
    return "_".join(suffixes)


def build_settings(config: dict, *, _allow_joint_adaptation: bool = False) -> dict:
    if config.get("stage1_component") != "VSA" or not as_bool(config.get("newtask_ft", False)):
        raise ValueError("NewTask FT VSA requires stage1_component: VSA and newtask_ft: true.")
    inherited_sections = {"architecture", "vision", "skill_flow", "termination_loss",
                          "cumulative_xyz_loss", "mask_actions_after_skill_end"} & set(config)
    if inherited_sections:
        raise ValueError(
            "NewTask FT inherits the model contract from the checkpoint; remove "
            f"{sorted(inherited_sections)} from the YAML."
        )
    project_root = Path(str(config["project_root"])).expanduser()
    dataset_root = project_root / str(config.get("dataset_root", "dataset"))
    outputs_root = project_root / str(config.get("outputs_root", "outputs"))

    checkpoint = _checkpoint_dir(
        project_root, outputs_root, _at(config, "warm_start", "vsa_checkpoint", default="")
    )
    source = json.loads((checkpoint / "config.json").read_text())
    label = str(source.get("architecture_label", "")).strip().lower()
    if label.endswith("_dgd"):
        label = label[:-4] + "_dGD"
    if source.get("type") != "skill_expert":
        raise ValueError(f"VSA checkpoint must be policy.type=skill_expert: {checkpoint}")
    if not label:
        raise ValueError(f"VSA checkpoint has no architecture_label: {checkpoint}")
    architecture_family = label.split("_", 1)[0]
    if architecture_family in _SHARED_ROUTE_ARCHITECTURES:
        raise ValueError(
            "NewTask FT requires a core-exit architecture with a separable "
            f"skill-only route; {label!r} belongs to the shared-route Arch0--Arch3 family."
        )
    if source.get("training_skill_source", "gt") != "gt":
        raise ValueError("NewTask FT requires a checkpoint trained with GT skills.")
    if source.get("skill_flow_latent_best_of_n_enabled", False):
        raise ValueError("NewTask FT does not support latent Best-of-N checkpoints.")

    dataset_source = str(_at(config, "dataset", "source"))
    skillvla_root = dataset_root / str(
        _at(config, "dataset", "skillvla_root", default="skillvla_dataset")
    )
    run_tag, dataset_relabeled = resolve_skillvla_dataset_run(
        skillvla_root / dataset_source,
        str(_at(config, "dataset", "run")),
        _at(config, "dataset", "relabeled", default=""),
    )
    dataset_dir = skillvla_root / dataset_source / run_tag / "skillvla"
    contract = _STAGE1._read_dataset_contract(dataset_dir, run_tag)

    # --- new dataset vs. frozen checkpoint contract --------------------------
    mismatches = []
    if source.get("skill_fsq_levels") != contract["levels"]:
        mismatches.append(
            f"skill_fsq_levels: checkpoint={source.get('skill_fsq_levels')}, dataset={contract['levels']}"
        )
    if str(source.get("proprio_grounding", "none")) != contract["proprio_grounding"]:
        mismatches.append(
            f"proprio_grounding: checkpoint={source.get('proprio_grounding')!r}, "
            f"dataset={contract['proprio_grounding']!r}"
        )
    for feature, key, group in (
        ("observation.state", "state_dim", "input_features"),
        ("action", "action_dim", "output_features"),
    ):
        saved = _at(source, group, feature, "shape")
        if saved is not None and int(saved[0]) != contract[key]:
            mismatches.append(f"{feature} dim: checkpoint={saved[0]}, dataset={contract[key]}")
    if mismatches:
        raise ValueError("NewTask FT dataset/checkpoint mismatch: " + "; ".join(mismatches))
    _AUX.newtask_ft_code_space_id(
        source,
        checkpoint,
        dataset_dir.parent,
        verify_fsq_source=as_bool(_at(config, "fsq", "verify_source", default=True)),
    )
    needs_focus_uv = _STAGE1.requires_focus_uv(
        label, foveated_vision_enabled=bool(source.get("foveated_vision_enabled", False))
    )
    if needs_focus_uv and contract["focus_uv_path"] is None:
        raise FileNotFoundError(
            f"{label} needs skill_focus_uv.npz in the new-task dataset "
            f"({dataset_dir.parent}); rebuild it with focus_uv.enabled=true."
        )
    if needs_focus_uv and contract["focus_uv_normalization"] != "minus_one_to_one":
        raise ValueError(
            f"{label} requires skill_focus_uv_normalization='minus_one_to_one', got "
            f"{contract['focus_uv_normalization']!r}."
        )

    dino_model = _STAGE1._local_model_path(project_root, str(source["dino_model_path"]))
    tokenizer = _STAGE1._local_model_path(project_root, str(source["tokenizer_path"]))
    fsq_path = dataset_dir.parent / "FSQ.pt"
    if not dino_model.is_dir():
        raise FileNotFoundError(f"DINO model not found: {dino_model}")
    if not fsq_path.is_file():
        raise FileNotFoundError(f"FSQ checkpoint not found: {fsq_path}")

    adaptation = config.get("adaptation", {}) or {}
    if not isinstance(adaptation, dict):
        raise ValueError("adaptation must be a mapping.")
    legacy_keys = {
        "train_dino", "dino_lr_scale", "unfreeze_action_head", "full_unfreeze", "dsbc",
    }
    unknown_adaptation = set(adaptation) - ({"skill_flow_loss", "freeze_expert"} | legacy_keys)
    if unknown_adaptation:
        raise ValueError(
            f"Unsupported adaptation settings: {sorted(unknown_adaptation)}. "
            "Standalone VSA FT exposes adaptation.skill_flow_loss and adaptation.freeze_expert."
        )

    skill_flow_loss = as_bool(adaptation.get("skill_flow_loss", False))
    freeze_expert = as_bool(adaptation.get("freeze_expert", False))
    if freeze_expert and skill_flow_loss:
        raise ValueError("adaptation.freeze_expert=true requires adaptation.skill_flow_loss=false.")
    if _allow_joint_adaptation:
        if freeze_expert:
            raise ValueError("adaptation.freeze_expert is supported only by standalone VSA FT.")
        # Joint owns these choices and reuses this resolver only for the VSA
        # checkpoint/dataset contract. They are intentionally absent from the
        # standalone VSA YAML.
        train_dino = as_bool(adaptation.get("train_dino", False))
        dino_lr_scale = float(adaptation.get("dino_lr_scale", 1.0))
        unfreeze_action_head = as_bool(adaptation.get("unfreeze_action_head", False))
        unfreeze_skill_layers = False
        full_unfreeze = as_bool(adaptation.get("full_unfreeze", False))
        dsbc_enabled = as_bool(adaptation.get("dsbc", False))
        if full_unfreeze and unfreeze_action_head:
            raise ValueError(
                "adaptation.full_unfreeze already trains the action head; set only one of "
                "full_unfreeze / unfreeze_action_head."
            )
    else:
        # Preserve old default-valued config snapshots, but reject removed
        # standalone choices instead of silently changing their behavior.
        fixed_defaults = {
            "train_dino": False,
            "dino_lr_scale": 1.0,
            "unfreeze_action_head": False,
            "full_unfreeze": False,
            "dsbc": False,
        }
        for key, expected in fixed_defaults.items():
            if key not in adaptation:
                continue
            actual = float(adaptation[key]) if key == "dino_lr_scale" else as_bool(adaptation[key])
            if actual != expected:
                raise ValueError(
                    f"adaptation.{key} is no longer a standalone VSA FT choice; "
                    f"remove it (fixed value: {expected!r})."
                )
        train_dino = False
        dino_lr_scale = 1.0
        # Enabling skill-flow adapts the complete Expert stack
        # used by the skill-flow objective, plus its shared decoding head.
        unfreeze_action_head = skill_flow_loss
        unfreeze_skill_layers = skill_flow_loss
        full_unfreeze = False
        dsbc_enabled = False

    if skill_flow_loss and not source.get("skill_flow_enabled", False):
        raise ValueError(
            "adaptation.skill_flow_loss=true requires a checkpoint with skill_flow_enabled=true."
        )
    skill_flow_active = bool(
        (unfreeze_action_head or full_unfreeze)
        and skill_flow_loss
        and source.get("skill_flow_enabled", False)
    )
    if skill_flow_active:
        # The skill-flow loss returns, so the new dataset must fit the checkpoint's trajectory horizon.
        horizon = int(source.get("skill_flow_max_length", 0) or 0)
        if source.get("skill_flow_target", "canonical") == "canonical" and (
            contract["skill_observed_max_length"] > horizon
        ):
            raise ValueError(
                "The new dataset's longest skill does not fit the checkpoint's skill-flow horizon: "
                f"dataset={contract['skill_observed_max_length']}, checkpoint={horizon}."
            )
    if dino_lr_scale <= 0:
        raise ValueError("adaptation.dino_lr_scale must be positive.")

    transition_jitter = resolve_transition_randomization(config, contract)

    steps = int(_at(config, "training", "schedule", "steps", default=20000))
    scheduler_mode = str(
        _at(config, "training", "schedule", "lr_mode", default="warmup_constant")
    ).strip().lower()
    warmup_steps = int(_at(config, "training", "schedule", "warmup_steps", default=500))
    decay_steps = int(_at(config, "training", "schedule", "lr_decay_steps", default=steps))
    batch_size = int(_at(config, "training", "dataloader", "batch_size", default=32))
    num_gpus = int(_at(config, "training", "dataloader", "gpus", default=1))
    if min(steps, batch_size, num_gpus, decay_steps) <= 0 or warmup_steps < 0:
        raise ValueError("Invalid training schedule/dataloader values.")
    if scheduler_mode not in {"cosine_decay", "warmup_constant"}:
        raise ValueError("training.schedule.lr_mode must be cosine_decay or warmup_constant.")
    debug_every = int(_at(config, "training", "vsa_debug", "every", default=0))
    debug_initial = [int(v) for v in as_list(_at(config, "training", "vsa_debug", "initial", default=[]))]
    if debug_every < 0 or any(v <= 0 for v in debug_initial):
        raise ValueError("training.vsa_debug values must be non-negative/positive.")
    periodic = range(debug_every, steps + 1, debug_every) if debug_every > 0 else ()
    debug_schedule = sorted(set(debug_initial).union(periodic))

    source_run, source_step = _checkpoint_run_and_step(checkpoint)
    run_name = f"{source_run}_{source_step}_{dataset_source}_ft_bs{batch_size}"
    if _allow_joint_adaptation:
        if train_dino:
            run_name += "_dino"
        if unfreeze_action_head:
            run_name += "_head"
        if full_unfreeze:
            run_name += "_full"
        if (unfreeze_action_head or full_unfreeze) and not skill_flow_loss and source.get("skill_flow_enabled", False):
            run_name += "_noskill"
        if dsbc_enabled:
            run_name += "_dsbc"
    elif freeze_expert:
        run_name += "_frozenexpert"
    elif skill_flow_loss:
        run_name += "_skillflow"
    suffix = _suffix(config)
    if suffix:
        run_name += f"_{suffix}"
    if len(run_name) > 240:
        raise ValueError(f"Run name exceeds the filesystem limit ({len(run_name)} chars): {run_name}")

    return {
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        "skillvla_dataset_dir": dataset_dir,
        "dataset_relabeled": dataset_relabeled,
        "repo_id": f"dohyeon/{dataset_source}",
        "vsa_checkpoint_path": checkpoint,
        "architecture_label": label,
        "trainable_expert_layers": 0 if freeze_expert else int(source.get("visual_bridge_last_n_layers", 1)),
        "fsq_path": fsq_path,
        "dino_model_path": dino_model,
        "tokenizer_path": tokenizer,
        "freeze_vision_encoder": not train_dino,
        "freeze_expert": freeze_expert,
        "unfreeze_action_head": unfreeze_action_head,
        "unfreeze_skill_layers": unfreeze_skill_layers,
        "full_unfreeze": full_unfreeze,
        "skill_flow_loss": skill_flow_loss,
        "skill_flow_active": skill_flow_active,
        "dsbc_enabled": dsbc_enabled,
        "dino_lr_scale": dino_lr_scale,
        "transition_jitter_pmax": transition_jitter["jitter_pmax"],
        "transition_jitter_pmax_percent": transition_jitter["jitter_pmax_percent"],
        "transition_jitter_preserve_gripper": transition_jitter["jitter_preserve_gripper"],
        "transition_jitter_early_start_pmax": transition_jitter["jitter_early_start_pmax"],
        "transition_jitter_late_start_pmax": transition_jitter["jitter_late_start_pmax"],
        "transition_jitter_early_end_pmax": transition_jitter["jitter_early_end_pmax"],
        "transition_jitter_late_end_pmax": transition_jitter["jitter_late_end_pmax"],
        "transition_jitter_distribution": transition_jitter["jitter_distribution"],
        "gradient_checkpointing": as_bool(
            _at(config, "training", "gradient_checkpointing", default=False)
        ),
        "vsa_debug_schedule": "[" + ",".join(str(v) for v in debug_schedule) + "]",
        "lr": float(_at(config, "training", "optimizer", "base_lr", default=2.5e-5)) * num_gpus,
        "batch_size": batch_size,
        "num_workers": int(_at(config, "training", "dataloader", "workers", default=8)),
        "num_gpus": num_gpus,
        "steps": steps,
        "scheduler_mode": scheduler_mode,
        "scheduler_warmup_steps": warmup_steps,
        "scheduler_decay_steps": decay_steps,
        "log_freq": int(_at(config, "training", "schedule", "log_every", default=100)),
        "save_freq": int(_at(config, "training", "schedule", "save_every", default=2000)),
        "run_name": run_name,
        "output_dir": outputs_root / OUTPUT_GROUP / "VSA" / run_name,
        "wandb_enable": as_bool(_at(config, "logging", "wandb", "enable", default=True)),
        "wandb_project": str(_at(config, "logging", "wandb", "project", default="VLA_NewTask_FT")),
        "train_partition": ",".join(as_list(config.get("train_partition", ["big"]))) or "big",
        "train_qos": str(config.get("train_qos", "big_qos")),
        "train_gres": str(_at(config, "slurm", "gres", default="gpu:1")),
        "train_cpus_per_task": int(_at(config, "slurm", "cpus", default=12)),
        "train_mem": str(_at(config, "slurm", "memory", default="128G")),
        "train_time": str(_at(config, "slurm", "time", default="48:00:00")),
        "train_nodelist": str(config.get("train_nodelist", "")),
        "train_exclude_nodes": ",".join(as_list(config.get("train_exclude_nodes", []))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    settings = build_settings(load_stage1_component_config(args.config))
    if args.shell:
        print_shell(settings)
    else:
        for key, value in settings.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
