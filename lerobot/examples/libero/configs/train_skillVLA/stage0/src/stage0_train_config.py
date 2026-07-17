#!/usr/bin/env python3
"""Resolve the compact SkillVLA Stage-0 YAML into shell exports."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

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


def _csv(value) -> str:
    return ",".join(as_list(value))


def _alpha(value, rank: int) -> float:
    return float(2 * rank if str(value).strip().lower() in ("", "auto", "none") else value)


def _local_model(project_root: Path, value) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        return Path(resolve_path(project_root, path))
    if path.exists() or "models" not in path.parts:
        return path
    return project_root.joinpath(*path.parts[path.parts.index("models"):])


_BRANCH_COMPONENTS = (
    ("vlm", "vlm"),
    ("cond", "cond"),
    ("expert", "expert"),
    ("vlm_lora", "vlm_lora"),
    ("cond_lora", "cond_lora"),
    ("expert_lora", "expert_lora"),
    ("language_bridge", "lang_bridge"),
    ("vlm_vision", "vlm_vision"),
    ("cond_vision", "cond_vision"),
    ("skill_reader", "skill_reader"),
    ("skill_head", "skill_head"),
)


def _branch_settings(
    cfg: dict, branch: str, enabled_components: dict[str, bool],
) -> tuple[float, bool, str]:
    freeze_groups = _at(cfg, "regime", branch, "freeze")
    if not isinstance(freeze_groups, dict):
        raise ValueError(f"regime.{branch}.freeze must explicitly list every Stage-0 component.")
    freeze = {}
    for group, entries in freeze_groups.items():
        if not isinstance(entries, dict):
            raise ValueError(f"regime.{branch}.freeze.{group} must be a mapping.")
        duplicates = set(freeze) & set(entries)
        if duplicates:
            raise ValueError(
                f"regime.{branch}.freeze contains duplicate components: {sorted(duplicates)}.")
        freeze.update(entries)
    required = {yaml_name for yaml_name, _ in _BRANCH_COMPONENTS}
    missing = sorted(required - set(freeze))
    extra = sorted(set(freeze) - required)
    if missing or extra:
        raise ValueError(
            f"regime.{branch}.freeze matrix mismatch: missing={missing}, extra={extra}.")
    train = [
        internal
        for yaml_name, internal in _BRANCH_COMPONENTS
        if enabled_components.get(yaml_name, True) and not as_bool(freeze[yaml_name])
    ]
    probability = float(_at(cfg, "regime", branch, "probability"))
    drop_vlm = as_bool(_at(cfg, "regime", branch, "drop_vlm"))
    return probability, drop_vlm, ",".join(train)


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

    sys.path.insert(0, str(project_root / "lerobot" / "examples" / "libero"))
    from FSQ import load_fsq_action_expert_state  # noqa: PLC0415

    _, fsq_cfg = load_fsq_action_expert_state(fsq_path)
    levels = [int(x) for x in fsq_cfg.fsq_levels]
    match = re.search(r"FSQ(\d+)", run_tag)
    if match and [int(x) for x in match.group(1)] != levels:
        raise ValueError(f"Dataset run says FSQ{match.group(1)}, but {fsq_path} contains levels={levels}.")

    pi_base = Path(resolve_path(project_root, _at(cfg, "warm_start", "pi_base", default="models/pi05_base")))
    tokenizer = Path(resolve_path(
        project_root,
        _at(cfg, "warm_start", "tokenizer", default="models/paligemma-3b-pt-224-tokenizer"),
    ))
    dino_value = _at(cfg, "cond", "dino_model", default=getattr(fsq_cfg, "dino_model_path", ""))
    dino_model = _local_model(project_root, dino_value)

    vlm_rank = int(_at(cfg, "vlm_lora", "rank", default=8))
    expert_rank = int(_at(cfg, "expert_lora", "rank", default=8))
    bridge_rank = int(_at(cfg, "language_bridge", "rank", default=64))
    reader_tokens = int(_at(cfg, "skill_reader", "architecture", "tokens", default=4))
    reader_depth = int(_at(cfg, "skill_reader", "architecture", "depth", default=2))
    reader_heads = int(_at(cfg, "skill_reader", "architecture", "heads", default=8))
    if min(reader_tokens, reader_depth, reader_heads) <= 0:
        raise ValueError(
            "SkillReader tokens, depth, and heads must all be positive, got "
            f"{(reader_tokens, reader_depth, reader_heads)}.")
    vlm_targets = _csv(_at(cfg, "vlm_lora", "targets", default=["q", "k", "v", "o"]))
    expert_targets = _csv(_at(
        cfg, "expert_lora", "targets", default=["q", "k", "v", "o", "mlp", "action_out"]))
    adapter_enabled = {
        name: as_bool(_at(cfg, "components", "adapters", name, default=default))
        for name, default in (
            ("vlm_lora", True),
            ("cond_lora", False),
            ("expert_lora", True),
            ("language_bridge", True),
        )
    }
    p_a, a_drop_vlm, a_train_components = _branch_settings(cfg, "A", adapter_enabled)
    p_b, b_drop_vlm, b_train_components = _branch_settings(cfg, "B", adapter_enabled)
    wrong_weight = float(_at(cfg, "regime", "wrong_language", "weight", default=0.0))
    wrong_margin = float(_at(cfg, "regime", "wrong_language", "margin", default=0.02))
    a_start_weight = float(_at(cfg, "loss", "A", "skill_start_weight", default=1.0))
    a_end_weight = float(_at(cfg, "loss", "A", "skill_end_weight", default=1.0))
    b_start_weight = float(_at(cfg, "loss", "B", "skill_start_weight", default=1.0))
    b_end_weight = float(_at(cfg, "loss", "B", "skill_end_weight", default=1.0))
    if not 0.0 <= p_a <= 1.0 or not 0.0 <= p_b <= 1.0 or abs(p_a + p_b - 1.0) > 1e-8:
        raise ValueError(f"Stage-0 A/B probabilities must be in [0,1] and sum to 1, got A={p_a}, B={p_b}.")
    if a_drop_vlm or not b_drop_vlm:
        raise ValueError("Stage-0 topology requires A.drop_vlm=false and B.drop_vlm=true.")
    forbidden_train = {"vlm", "expert", "skill_reader", "skill_head"}
    for branch, spec in (("A", a_train_components), ("B", b_train_components)):
        invalid = set(spec.split(",")) & forbidden_train
        if invalid:
            raise ValueError(
                f"Stage-0 {branch} must keep fixed bases/unused modules frozen, but trains {sorted(invalid)}.")
    if wrong_weight < 0.0 or wrong_margin < 0.0:
        raise ValueError("wrong-language weight and margin must be non-negative.")
    loss_weights = (a_start_weight, a_end_weight, b_start_weight, b_end_weight)
    if any(weight <= 0.0 for weight in loss_weights):
        raise ValueError(f"All Stage-0 A/B loss weights must be > 0, got {loss_weights}.")

    batch_size = int(_at(cfg, "training", "dataloader", "batch_size", default=128))
    num_gpus = int(_at(cfg, "training", "dataloader", "gpus", default=1))
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
        "cond_encoder_variant": str(fsq_cfg.cond_encoder_variant or fsq_cfg.action_expert_variant),
        "state_cond_mode": str(fsq_cfg.state_cond_mode),
        "action_expert_variant": str(fsq_cfg.action_expert_variant),
        "max_state_dim": int(fsq_cfg.max_state_dim),
        "max_action_dim": int(fsq_cfg.max_action_dim),
        "chunk_size": int(fsq_cfg.chunk_size),
        "min_period": float(fsq_cfg.min_period),
        "max_period": float(fsq_cfg.max_period),
        "time_sampling_beta_alpha": float(fsq_cfg.time_sampling_beta_alpha),
        "time_sampling_beta_beta": float(fsq_cfg.time_sampling_beta_beta),
        "time_sampling_scale": float(fsq_cfg.time_sampling_scale),
        "time_sampling_offset": float(fsq_cfg.time_sampling_offset),
        "skill_fsq_levels": "[" + ",".join(str(x) for x in levels) + "]",
        "dino_image_size": int(fsq_cfg.dino_image_size),
        "siglip_image_size": int(fsq_cfg.siglip_image_size),
        "attend_image": as_bool(_at(cfg, "token_access", "image", default=False)),
        "attend_language": as_bool(_at(cfg, "token_access", "language", default=True)),
        "num_reader_tokens": reader_tokens,
        "reader_depth": reader_depth,
        "reader_heads": reader_heads,
        "skill_reader_all_layers": as_bool(_at(
            cfg, "skill_reader", "architecture", "all_layers", default=True)),
        "reader_attend_image": as_bool(_at(
            cfg, "skill_reader", "token_access", "image", default=True)),
        "reader_attend_language": as_bool(_at(
            cfg, "skill_reader", "token_access", "language", default=True)),
        "vlm_cond": as_bool(_at(cfg, "connections", "vlm_to_cond", default=True)),
        "cond_expert": as_bool(_at(cfg, "connections", "cond_to_expert", default=True)),
        "vlm_expert": as_bool(_at(cfg, "connections", "vlm_to_expert", default=False)),
        "vlm_lora": adapter_enabled["vlm_lora"],
        "cond_lora": adapter_enabled["cond_lora"],
        "stage0_expert_lora": adapter_enabled["expert_lora"],
        "vlm_lora_targets": vlm_targets,
        "vlm_lora_rank": vlm_rank,
        "vlm_lora_alpha": _alpha(_at(cfg, "vlm_lora", "alpha", default="auto"), vlm_rank),
        "vlm_lora_dropout": float(_at(cfg, "vlm_lora", "dropout", default=0.0)),
        "vlm_lora_lr_scale": float(_at(cfg, "vlm_lora", "lr_scale", default=10.0)),
        "expert_lora_targets": expert_targets,
        "expert_lora_rank": expert_rank,
        "expert_lora_alpha": _alpha(_at(cfg, "expert_lora", "alpha", default="auto"), expert_rank),
        "expert_lora_dropout": float(_at(cfg, "expert_lora", "dropout", default=0.0)),
        "expert_lora_lr_scale": float(_at(cfg, "expert_lora", "lr_scale", default=10.0)),
        "lang_bridge": adapter_enabled["language_bridge"],
        "lang_bridge_rank": bridge_rank,
        "lang_bridge_lr_scale": float(_at(cfg, "language_bridge", "lr_scale", default=10.0)),
        "stage0_vlm_severed_prob": p_b,
        "stage0_a_drop_vlm": a_drop_vlm,
        "stage0_b_drop_vlm": b_drop_vlm,
        "stage0_a_train_components": a_train_components,
        "stage0_b_train_components": b_train_components,
        "wrong_language_weight": wrong_weight,
        "wrong_language_margin": wrong_margin,
        "a_skill_start_loss_weight": a_start_weight,
        "a_skill_end_loss_weight": a_end_weight,
        "b_skill_start_loss_weight": b_start_weight,
        "b_skill_end_loss_weight": b_end_weight,
        "train_terminator": as_bool(_at(cfg, "terminator", "train", default=False)),
        "terminator_freeze_vision_encoder": as_bool(
            _at(cfg, "terminator", "freeze_vision", default=False)),
        "track_param_drift": as_bool(_at(cfg, "logging", "param_drift", default=True)),
        "pt_run_name": run_name,
        "pt_output_dir": outputs_root / "skillVLA_stage0" / run_name,
        "batch_size": batch_size,
        "num_workers": int(_at(cfg, "training", "dataloader", "workers", default=2)),
        "num_gpus": num_gpus,
        "lr": float(_at(cfg, "training", "optimizer", "base_lr", default=2.5e-5)) * num_gpus,
        "cond_lr_scale": float(_at(cfg, "training", "optimizer", "cond_lr_scale", default=1.0)),
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
