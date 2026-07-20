#!/usr/bin/env python3
"""Resolve SkillVLA Stage-3 training from a frozen Stage-0 or Stage-2 parent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import as_bool, as_levels, as_list, get_value, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "stage3_train_config.yaml"


def _at(cfg: dict, *keys: str, default=None):
    value = cfg
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _local_model_path(project_root: Path, value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return raw
    path = Path(raw).expanduser()
    if path.is_absolute() and path.exists():
        return str(path)
    project_path = project_root / path
    if project_path.exists():
        return str(project_path)
    candidate = project_root / "models" / path.name
    return str(candidate if candidate.exists() else path)


def build_settings(cfg: dict) -> dict:
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    dataset_root = project_root / str(get_value(cfg, "dataset_root", "dataset"))
    outputs_root = project_root / str(get_value(cfg, "outputs_root", "outputs"))
    skillvla_root = dataset_root / str(
        _at(cfg, "dataset", "skillvla_root", default=get_value(cfg, "skillvla_dataset_root", "skillvla_dataset"))
    )

    # New grouped UI. Old stage2_* keys remain accepted for existing config snapshots.
    parent_stage_value = _at(cfg, "warm_start", "stage", default=None)
    if parent_stage_value in (None, ""):
        parent_stage_value = "stage0" if get_value(cfg, "stage0_run_name", None) else "stage2"
    parent_stage = str(parent_stage_value).strip().lower()
    if parent_stage not in {"stage0", "stage2"}:
        raise ValueError(f"warm_start.stage must be stage0|stage2, got {parent_stage!r}.")

    legacy_run_key = f"{parent_stage}_run_name"
    parent_run = str(
        _at(cfg, "warm_start", "run", default=get_value(cfg, legacy_run_key, "")) or ""
    ).strip()
    if not parent_run:
        raise ValueError("warm_start.run must name a trained Stage-0 or Stage-2 run.")
    legacy_checkpoint_key = f"{parent_stage}_checkpoint"
    parent_checkpoint = str(
        _at(cfg, "warm_start", "checkpoint", default=get_value(cfg, legacy_checkpoint_key, "last")) or "last"
    ).strip()

    parent_root = outputs_root / f"skillVLA_{parent_stage}"
    parent_ckpt = parent_root / parent_run / "checkpoints" / parent_checkpoint / "pretrained_model"
    parent_cfg_json = parent_ckpt / "config.json"
    parent_train_json = parent_ckpt / "train_config.json"
    if not parent_cfg_json.is_file():
        raise FileNotFoundError(f"Stage-3 parent config not found: {parent_cfg_json}")
    if not parent_train_json.is_file():
        raise FileNotFoundError(f"Stage-3 parent train config not found: {parent_train_json}")

    parent_cfg = json.loads(parent_cfg_json.read_text())
    parent_train_cfg = json.loads(parent_train_json.read_text())

    def _inh_bool(key: str, default=False) -> bool:
        return as_bool(parent_cfg.get(key, default))

    def _inh_num(key: str, default, cast):
        return cast(parent_cfg.get(key, default))

    def _inh_tri(key: str) -> str:
        value = parent_cfg.get(key, None)
        return "" if value is None else ("true" if as_bool(value) else "false")

    # Dataset and FSQ geometry always follow the parent checkpoint.
    parent_dataset = parent_train_cfg.get("dataset") or {}
    ds_root = str(parent_dataset.get("root") or "")
    ds_repo = str(parent_dataset.get("repo_id") or "")
    if not ds_root:
        raise ValueError(f"Cannot derive parent dataset.root from {parent_train_json}")
    if not Path(ds_root).is_dir():
        old = Path(ds_root)
        ds_root = str(skillvla_root / old.parent.parent.name / old.parent.name / old.name)
    if not Path(ds_root).is_dir():
        raise FileNotFoundError(f"Stage-3 dataset not found after local re-anchoring: {ds_root}")
    run_dir = Path(ds_root).parent
    source_dataset = run_dir.parent.name
    if not ds_repo:
        ds_repo = f"dohyeon/{source_dataset}"

    # Prefer the parent's exported co-trained terminator. Fall back to its configured FSQ, then the
    # dataset-embedded FSQ.pt. This preserves Stage-0/2 terminator work while Stage-3 freezes it.
    parent_ft_fsq = parent_ckpt.parent / "FSQ_ft.pt"
    parent_cfg_fsq = Path(str(parent_cfg.get("fsq_path") or "")).expanduser()
    if parent_ft_fsq.is_file():
        fsq_ckpt = parent_ft_fsq
        fsq_source = "parent_cotrained"
    elif parent_cfg_fsq.is_file():
        fsq_ckpt = parent_cfg_fsq
        fsq_source = "parent_config"
    else:
        fsq_ckpt = run_dir / "FSQ.pt"
        fsq_source = "dataset_base"
    if not fsq_ckpt.is_file():
        raise FileNotFoundError(f"Stage-3 FSQ checkpoint not found: {fsq_ckpt}")

    stage1_checkpoint_path = str(parent_cfg.get("stage1_checkpoint_path") or "")
    skill_fsq_levels = list(as_levels(parent_cfg.get("skill_fsq_levels", [5, 5, 5])))
    s1_vision_backbone = str(parent_cfg.get("s1_vision_backbone", "siglip"))
    s1_state_cond_mode = str(parent_cfg.get("s1_state_cond_mode", "state"))

    # Exact parent architecture. Stage-0 needs the scratch-side fields because it has no Stage-1 config.
    tokenizer_path = _local_model_path(project_root, parent_cfg.get("tokenizer_path", ""))
    s1_dino_model_path = _local_model_path(
        project_root, parent_cfg.get("s1_dino_model_path", "models/dinov3-vits16")
    )
    architecture = {
        "paligemma_variant": str(parent_cfg.get("paligemma_variant", "gemma_2b")),
        "action_expert_variant": str(parent_cfg.get("action_expert_variant", "gemma_300m")),
        "s1_cond_encoder_variant": str(parent_cfg.get("s1_cond_encoder_variant") or "gemma_300m"),
        "s1_dino_model_path": s1_dino_model_path,
        "s1_dino_image_size": _inh_num("s1_dino_image_size", 224, int),
        "s1_siglip_image_size": _inh_num("s1_siglip_image_size", 224, int),
        "max_state_dim": _inh_num("max_state_dim", 32, int),
        "max_action_dim": _inh_num("max_action_dim", 32, int),
        "chunk_size": _inh_num("chunk_size", 10, int),
        "min_period": _inh_num("min_period", 4e-3, float),
        "max_period": _inh_num("max_period", 4.0, float),
        "time_sampling_beta_alpha": _inh_num("time_sampling_beta_alpha", 1.5, float),
        "time_sampling_beta_beta": _inh_num("time_sampling_beta_beta", 1.0, float),
        "time_sampling_scale": _inh_num("time_sampling_scale", 0.999, float),
        "time_sampling_offset": _inh_num("time_sampling_offset", 0.001, float),
    }

    # Topology and predictor architecture must match the parent weights.
    topology = {
        "attend_language": _inh_bool("attend_language", False),
        "attend_image": _inh_bool("attend_image", True),
        "reader_attend_image": _inh_tri("reader_attend_image"),
        "reader_attend_language": _inh_tri("reader_attend_language"),
        "vlm_cond": _inh_bool("vlm_cond", True),
        "cond_expert": _inh_bool("cond_expert", True),
        "vlm_expert": _inh_bool("vlm_expert", False),
        "num_reader_tokens": _inh_num("num_reader_tokens", 4, int),
        "reader_depth": _inh_num("reader_depth", 2, int),
        "reader_heads": _inh_num("reader_heads", 8, int),
        "skill_reader_all_layers": _inh_bool("skill_reader_all_layers", False),
    }

    # Rebuild every inherited adapter before full-loading the parent; Stage-3 freezes all of them and
    # trains only the freshly zero-initialized skill adapter plus reader/head.
    adapters = {
        "vlm_lora": _inh_bool("vlm_lora", _inh_bool("lora_cond_vlm", True)),
        "cond_lora": _inh_bool("cond_lora", _inh_bool("lora_cond_bridge", True)),
        "lora_rank": _inh_num("lora_rank", 8, int),
        "lora_alpha": _inh_num("lora_alpha", 16.0, float),
        "lora_dropout": _inh_num("lora_dropout", 0.0, float),
        "lora_targets": str(parent_cfg.get("lora_targets", "q,k,v,o")),
        "lang_bridge": _inh_bool("lang_bridge", False),
        "lang_bridge_rank": _inh_num("lang_bridge_rank", 64, int),
        "lang_bridge_lr_scale": _inh_num("lang_bridge_lr_scale", 10.0, float),
        "stage0_expert_source": str(parent_cfg.get("stage0_expert_source", "fsq")),
        "stage0_cond_state_adarms": _inh_bool("stage0_cond_state_adarms", False),
        "stage0_expert_lora": _inh_bool("stage0_expert_lora", False) if parent_stage == "stage0" else False,
        "stage0_expert_lora_targets": str(
            parent_cfg.get("stage0_expert_lora_targets", "q,k,v,o,mlp,action_out")
        ),
        "stage0_expert_lora_rank": _inh_num("stage0_expert_lora_rank", 8, int),
        "stage0_expert_lora_alpha": _inh_num("stage0_expert_lora_alpha", 16.0, float),
        "stage0_expert_lora_dropout": _inh_num("stage0_expert_lora_dropout", 0.0, float),
        "stage0_expert_lora_lr_scale": _inh_num("stage0_expert_lora_lr_scale", 10.0, float),
    }

    skill_loss_weight = _at(cfg, "objective", "skill_loss", "weight", default=get_value(cfg, "skill_loss_weight", None))
    if skill_loss_weight in (None, "", "null"):
        skill_loss_weight = parent_cfg.get("skill_loss_weight", 0.1)
    deadzone = _at(cfg, "objective", "skill_loss", "deadzone_frac", default=get_value(cfg, "skill_deadzone_frac", None))
    if deadzone in (None, "", "null"):
        deadzone = parent_cfg.get("skill_deadzone_frac", 0.0)
    skill_action_grounding = as_bool(
        _at(cfg, "objective", "action_grounding", default=get_value(cfg, "skill_action_grounding", False))
    )
    lora_lr_scale = float(
        _at(cfg, "skill_path", "lora_lr_scale", default=get_value(cfg, "lora_lr_scale", parent_cfg.get("lora_lr_scale", 1.0)))
    )

    train_terminator = as_bool(
        _at(cfg, "terminator", "train", default=get_value(cfg, "train_terminator", False))
    )
    terminator_dino_model_path = _local_model_path(
        project_root,
        _at(
            cfg,
            "terminator",
            "dino_model",
            default=get_value(
                cfg,
                "terminator_dino_model_path",
                parent_cfg.get("terminator_dino_model_path")
                or parent_cfg.get("s1_dino_model_path")
                or "models/dinov3-vits16",
            ),
        ),
    )

    batch_size = int(_at(cfg, "training", "dataloader", "batch_size", default=get_value(cfg, "batch_size", 16)))
    num_workers = int(_at(cfg, "training", "dataloader", "workers", default=get_value(cfg, "num_workers", 4)))
    num_gpus = int(_at(cfg, "training", "dataloader", "gpus", default=get_value(cfg, "num_gpus", 1)))
    lr_base = float(_at(cfg, "training", "optimizer", "base_lr", default=get_value(cfg, "lr_base", 2.5e-5)))
    steps = int(_at(cfg, "training", "schedule", "steps", default=get_value(cfg, "steps", 30000)))
    save_freq = int(_at(cfg, "training", "schedule", "save_every", default=get_value(cfg, "save_freq", 5000)))

    manual_name = str(_at(cfg, "run", "name", default="") or "").strip()
    suffix = str(_at(cfg, "run", "suffix", default=get_value(cfg, "exp", "")) or "").strip().strip("_")
    mode = "b" if skill_action_grounding else "a"
    run_name = manual_name or f"{parent_run}_{parent_checkpoint}__s3{mode}_{'s0' if parent_stage == 'stage0' else 's2'}"
    if suffix:
        run_name = f"{run_name}_{suffix}"
    vla_root = outputs_root / "skillVLA_stage3"

    settings: dict = {
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        "source_dataset": source_dataset,
        "skillvla_dataset_dir": ds_root,
        "repo_id": ds_repo,
        "fsq_ckpt": fsq_ckpt,
        "fsq_source": fsq_source,
        "terminator_dino_model_path": terminator_dino_model_path,
        "transition_pack": run_dir / "transitions.npz",
        "parent_stage": parent_stage,
        "parent_run_name": parent_run,
        "parent_checkpoint": parent_checkpoint,
        "parent_checkpoint_path": parent_ckpt,
        "stage1_checkpoint_path": stage1_checkpoint_path,
        "tokenizer_path": tokenizer_path,
        "s1_vision_backbone": s1_vision_backbone,
        "s1_state_cond_mode": s1_state_cond_mode,
        "skill_fsq_levels": "[" + ",".join(str(x) for x in skill_fsq_levels) + "]",
        **architecture,
        **topology,
        **adapters,
        "pt_stage": "skill",
        "stage3_parent_stage": parent_stage,
        "skill_action_grounding": skill_action_grounding,
        "skill_loss_weight": str(skill_loss_weight),
        "skill_deadzone_frac": float(deadzone or 0.0),
        "lora_skill": True,
        "lora_lr_scale": lora_lr_scale,
        "train_terminator": train_terminator,
        "terminator_lr_scale": float(
            _at(cfg, "terminator", "lr_scale", default=get_value(cfg, "terminator_lr_scale", 1.0))
        ),
        "terminator_end_target_sigma": float(
            _at(cfg, "terminator", "target_sigma", default=get_value(cfg, "terminator_end_target_sigma", 1.0))
        ),
        "terminator_end_pos_weight": float(
            _at(cfg, "terminator", "end_weight", default=get_value(cfg, "terminator_end_pos_weight", 1.0))
        ),
        "track_param_drift": as_bool(
            _at(cfg, "logging", "param_drift", default=get_value(cfg, "track_param_drift", False))
        ),
        "skillvla_outputs_root": vla_root,
        "pt_run_name": run_name,
        "pt_output_dir": vla_root / run_name,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "num_gpus": num_gpus,
        "lr": lr_base * num_gpus,
        "steps": steps,
        "save_freq": save_freq,
        "wandb_enable": as_bool(
            _at(cfg, "logging", "wandb", "enable", default=get_value(cfg, "wandb_enable", True))
        ),
        "wandb_project": str(
            _at(cfg, "logging", "wandb", "project", default=get_value(cfg, "wandb_project", "VLA_stage3"))
        ),
    }

    part_value = _at(cfg, "slurm", "partition", default=get_value(cfg, "train_partition", ["big"]))
    exclude_value = _at(cfg, "slurm", "exclude_nodes", default=get_value(cfg, "train_exclude_nodes", []))
    settings.update({
        "train_partition": ",".join(as_list(part_value)) or "big",
        "train_qos": str(_at(cfg, "slurm", "qos", default=get_value(cfg, "train_qos", "big_qos"))),
        "train_gres": str(_at(cfg, "slurm", "gres", default=get_value(cfg, "train_gres", "gpu:1"))),
        "train_cpus_per_task": int(
            _at(cfg, "slurm", "cpus", default=get_value(cfg, "train_cpus_per_task", 16))
        ),
        "train_mem": str(_at(cfg, "slurm", "memory", default=get_value(cfg, "train_mem", "128G"))),
        "train_time": str(_at(cfg, "slurm", "time", default=get_value(cfg, "train_time", "48:00:00"))),
        "train_nodelist": str(
            _at(cfg, "slurm", "nodelist", default=get_value(cfg, "train_nodelist", ""))
        ),
        "train_exclude_nodes": ",".join(as_list(exclude_value)),
    })
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
