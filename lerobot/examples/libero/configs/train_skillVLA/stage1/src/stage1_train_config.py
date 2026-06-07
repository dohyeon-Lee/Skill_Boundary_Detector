#!/usr/bin/env python3
"""Config for SkillVLA Stage-1 training (standalone action expert, policy.type=skill_expert).

Trains the action expert by flow matching on the build_data skillvla dataset (raw images +
skill_sequence/skill_index + actions). No FSQ / DINO-token / VLM artifacts are needed here:
the expert encodes raw images with its OWN trainable DINOv3 and reads the GT skill code from
the dataset. Weights can warm-start from pi05_base (action-expert motion prior) or train from
scratch. Roots come from build_data's yaml. Emits shell exports (--shell).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import as_bool, as_levels, as_list, get_value, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "stage1_train_config.yaml"
BUILD_DATA_CONFIG_PATH = _HERE.parent.parent.parent / "build_data" / "train_skillVLA_config.yaml"


def _roots() -> tuple[Path, Path, Path, Path, list[int]]:
    """(project_root, dataset_root, skillvla_root, lerobot_root, fsq_levels) from build_data yaml.
    fsq_levels = the FSQ codebook structure the skillvla dataset was built with."""
    bc = load_config(str(BUILD_DATA_CONFIG_PATH))
    project_root = Path(str(get_value(bc, "project_root"))).expanduser()
    dataset_root = project_root / str(get_value(bc, "dataset_root", "libero_dataset"))
    skillvla_root = dataset_root / str(get_value(bc, "skillvla_dataset_root", "skillvla_dataset"))
    fsq_levels = list(as_levels(get_value(bc, "fsq_levels", [5, 5, 5])))  # robust to "[5,5,5]" env round-trip
    return project_root, dataset_root, skillvla_root, project_root / "lerobot", fsq_levels


def build_settings(cfg: dict) -> dict:
    project_root, dataset_root, skillvla_root, lerobot_root, build_fsq_levels = _roots()

    source_dataset = str(get_value(cfg, "source_dataset"))
    run_tag = str(get_value(cfg, "run_tag"))
    run_dir = skillvla_root / source_dataset / run_tag

    batch_size = int(get_value(cfg, "batch_size", 32))
    num_gpus = int(get_value(cfg, "num_gpus", 1))
    lr_base = float(get_value(cfg, "lr_base", 2.5e-05))
    exp = str(get_value(cfg, "exp", "")).strip()

    init_from_pi05 = as_bool(get_value(cfg, "init_from_pi05", True))
    pi_base = str(get_value(cfg, "pi_base", "lerobot/pi05_base")) if init_from_pi05 else ""

    dino_lr = get_value(cfg, "dino_lr", None)
    dino_lr_str = "" if dino_lr in (None, "", "null") else str(dino_lr)
    siglip_lr = get_value(cfg, "siglip_lr", None)
    siglip_lr_str = "" if siglip_lr in (None, "", "null") else str(siglip_lr)

    init_tag = "init" if init_from_pi05 else "scratch"
    run_name = f"{source_dataset}_{run_tag}_batch{batch_size}_stage1_{init_tag}"
    if exp:
        run_name = f"{run_name}_{exp}"
    vla_root = project_root / str(get_value(cfg, "skillvla_outputs_root", "skillVLA_outputs"))
    output_dir = vla_root / f"S1_{run_name}"

    # FSQ skill structure: auto-match the FSQ the dataset was built with (build_data fsq_levels).
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
