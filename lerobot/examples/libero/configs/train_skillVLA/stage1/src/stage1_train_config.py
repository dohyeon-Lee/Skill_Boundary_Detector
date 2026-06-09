#!/usr/bin/env python3
"""Config for SkillVLA Stage-1 training (standalone action expert, policy.type=skill_expert).

Trains the action expert by flow matching on the build_data skillvla dataset (raw images +
skill_sequence/skill_index + actions). No FSQ / DINO-token / VLM artifacts are needed here:
the expert encodes raw images with its OWN trainable DINOv3 and reads the GT skill code from
the dataset. Weights can warm-start from pi05_base (action-expert motion prior) or train from
scratch. All roots are declared in this yaml (standalone). Emits shell exports (--shell).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import as_bool, as_levels, as_list, get_value, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "stage1_train_config.yaml"


def build_settings(cfg: dict) -> dict:
    # Standalone: every root is declared in this yaml (no build_data dependency).
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    dataset_root = project_root / str(get_value(cfg, "dataset_root", "dataset"))
    skillvla_root = dataset_root / str(get_value(cfg, "skillvla_dataset_root", "skillvla_dataset"))
    lerobot_root = project_root / "lerobot"

    source_dataset = str(get_value(cfg, "source_dataset"))
    run_tag = str(get_value(cfg, "run_tag"))
    run_dir = skillvla_root / source_dataset / run_tag
    # FSQ codebook structure is encoded in run_tag's FSQ<levels> tag (e.g. FSQ88 → [8,8]).
    _m = re.search(r"FSQ(\d+)", run_tag)
    build_fsq_levels = [int(d) for d in _m.group(1)] if _m else [5, 5, 5]

    batch_size = int(get_value(cfg, "batch_size", 32))
    num_gpus = int(get_value(cfg, "num_gpus", 1))
    lr_base = float(get_value(cfg, "lr_base", 2.5e-05))
    exp = str(get_value(cfg, "exp", "")).strip()

    # Conditioning architecture: "fused" (one self-attn stream) or "joint" (cond-encoder ⊥ expert).
    expert_arch = str(get_value(cfg, "expert_arch", "fused")).strip() or "fused"
    cond_encoder_variant = str(get_value(cfg, "cond_encoder_variant", "")).strip()
    if cond_encoder_variant.lower() in ("none", "null"):  # blank yaml → omit (use action expert's variant)
        cond_encoder_variant = ""

    init_from_pi05 = as_bool(get_value(cfg, "init_from_pi05", True))
    pi_base = str(get_value(cfg, "pi_base", "lerobot/pi05_base")) if init_from_pi05 else ""

    dino_lr = get_value(cfg, "dino_lr", None)
    dino_lr_str = "" if dino_lr in (None, "", "null") else str(dino_lr)
    siglip_lr = get_value(cfg, "siglip_lr", None)
    siglip_lr_str = "" if siglip_lr in (None, "", "null") else str(siglip_lr)

    # run-name vision tag: <backbone>_<freeze|unfreeze> (the backbone trained + whether it was frozen).
    vision_backbone = (str(get_value(cfg, "vision_backbone", "dino")).strip().lower() or "dino")
    if vision_backbone == "siglip":
        vfrozen = as_bool(get_value(cfg, "freeze_siglip", False))
    else:
        vfrozen = as_bool(get_value(cfg, "freeze_dino", False))
    vis_tag = f"{vision_backbone}_{'freeze' if vfrozen else 'unfreeze'}"

    # run-name arch tag: joint→A, fused→B. init_from_pi05 is fixed true → not in the name.
    arch_tag = {"joint": "A", "fused": "B"}.get(expert_arch, expert_arch)
    run_name = f"{source_dataset}_{run_tag}_{vis_tag}_batch{batch_size}_{arch_tag}"
    if exp:
        run_name = f"{run_name}_{exp}"
    # Single outputs root from yaml; the per-stage subdir is fixed here (not in yaml).
    outputs_root = project_root / str(get_value(cfg, "outputs_root", "outputs"))
    vla_root = outputs_root / "skillVLA_stage1"
    output_dir = vla_root / run_name   # under skillVLA_stage1/, so no extra stage prefix

    # FSQ skill structure: auto-match the FSQ the dataset was built with (parsed from run_tag's FSQ<levels> tag).
    # The expert only needs the codebook size (= prod(levels)) for its skill embedding; levels are
    # kept for the eval cube viz. Either may be overridden in the stage1 yaml.
    skill_fsq_levels = list(as_levels(get_value(cfg, "skill_fsq_levels", build_fsq_levels)))
    vocab_default = 1
    for _lvl in skill_fsq_levels:
        vocab_default *= _lvl
    skill_vocab_size = int(get_value(cfg, "skill_vocab_size", vocab_default))

    settings: dict = {
        # roots
        "project_root": project_root,
        "lerobot_root": lerobot_root,
        # dataset (raw skillvla dataset: images + skill columns + actions)
        "source_dataset": source_dataset,
        "run_tag": run_tag,
        "skillvla_dataset_dir": run_dir / "skillvla",
        "repo_id": f"dohyeon/{source_dataset}",
        # conditioning architecture
        "expert_arch": expert_arch,               # "fused" | "joint"
        "cond_encoder_variant": cond_encoder_variant,  # "" → same as action_expert_variant
        # model init
        "pi_base": pi_base,                       # "" → train the expert from scratch
        # vision encoder: "dino" or "siglip" (siglip warm-starts from pi_base's vision_tower)
        "vision_backbone": str(get_value(cfg, "vision_backbone", "dino")),
        "dino_model_path": str(get_value(cfg, "dino_model_path", "/data2/dohyeon/SBD/models/dinov3-vits16")),
        "dino_lr": dino_lr_str,                   # "" → same LR as the rest
        "freeze_dino": as_bool(get_value(cfg, "freeze_dino", False)),
        "freeze_siglip": as_bool(get_value(cfg, "freeze_siglip", True)),
        "siglip_lr": siglip_lr_str,               # "" → same LR as the rest (when unfrozen)
        "siglip_image_size": int(get_value(cfg, "siglip_image_size", 224)),
        "skill_vocab_size": skill_vocab_size,
        "skill_fsq_levels": "[" + ",".join(str(x) for x in skill_fsq_levels) + "]",
        # output
        "skillvla_outputs_root": vla_root,
        "pt_run_name": run_name,
        "pt_output_dir": output_dir,
        # optimization
        "batch_size": batch_size,
        "num_workers": int(get_value(cfg, "num_workers", 8)),
        "num_gpus": num_gpus,
        "lr": lr_base * num_gpus,
        "steps": int(get_value(cfg, "steps", 100000)),
        "save_freq": int(get_value(cfg, "save_freq", 5000)),
        # wandb
        "wandb_enable": as_bool(get_value(cfg, "wandb_enable", True)),
        "wandb_project": str(get_value(cfg, "wandb_project", "VLA_stage1")),
    }

    part = ",".join(as_list(get_value(cfg, "train_partition", ["big"]))) or "big"
    excl = ",".join(as_list(get_value(cfg, "train_exclude_nodes", [])))
    settings.update({
        "train_partition": part,
        "train_qos": str(get_value(cfg, "train_qos", "big_qos")),
        "train_gres": str(get_value(cfg, "train_gres", "gpu:1")),
        "train_cpus_per_task": int(get_value(cfg, "train_cpus_per_task", 16)),
        "train_mem": str(get_value(cfg, "train_mem", "128G")),
        "train_time": str(get_value(cfg, "train_time", "48:00:00")),
        "train_nodelist": str(get_value(cfg, "train_nodelist", "")),
        "train_exclude_nodes": excl,
    })
    return settings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--shell", action="store_true")
    args = ap.parse_args()
    settings = build_settings(load_config(args.config))
    if args.shell:
        print_shell(settings)
    else:
        for k, v in settings.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
