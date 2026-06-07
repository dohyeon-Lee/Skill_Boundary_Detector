#!/usr/bin/env python3
"""Config for SkillVLA Stage-2 training (policy.type=skill_vla).

A PaliGemma VLM (warm-started from pi05_base) predicts the skill from the skill-START obs; an action
expert warm-started from a Stage-1 ``skill_expert`` checkpoint flow-matches the action chunk. The
Stage-1 checkpoint's config supplies expert_arch (A=joint / B=fused), vision_backbone,
action_expert_variant and skill_vocab_size — the model reads them itself, so here we only point to
it. Roots + FSQ levels come from build_data's yaml; FSQ.pt (eval terminator) lives in the run dir.
Emits shell exports (--shell).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import as_bool, as_levels, as_list, get_value, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "stage2_train_config.yaml"
BUILD_DATA_CONFIG_PATH = _HERE.parent.parent.parent / "build_data" / "train_skillVLA_config.yaml"


def _roots() -> tuple[Path, Path, Path, Path, list[int]]:
    """(project_root, dataset_root, skillvla_root, lerobot_root, fsq_levels) from build_data yaml."""
    bc = load_config(str(BUILD_DATA_CONFIG_PATH))
    project_root = Path(str(get_value(bc, "project_root"))).expanduser()
    dataset_root = project_root / str(get_value(bc, "dataset_root", "libero_dataset"))
    skillvla_root = dataset_root / str(get_value(bc, "skillvla_dataset_root", "skillvla_dataset"))
    fsq_levels = list(as_levels(get_value(bc, "fsq_levels", [5, 5, 5])))
    return project_root, dataset_root, skillvla_root, project_root / "lerobot", fsq_levels


def _stage1_arch(ckpt_dir: Path) -> str:
    """expert_arch (joint/fused) from the Stage-1 checkpoint config — for the run-name tag only
    (the model itself re-reads the full Stage-1 config at load time)."""
    cfg = ckpt_dir / "train_config.json"
    if cfg.is_file():
        try:
            return str(json.loads(cfg.read_text()).get("policy", {}).get("expert_arch", "")) or "fused"
        except Exception:  # noqa: BLE001
            pass
    return "fused"


def build_settings(cfg: dict) -> dict:
    project_root, dataset_root, skillvla_root, lerobot_root, build_fsq_levels = _roots()

    source_dataset = str(get_value(cfg, "source_dataset"))
    run_tag = str(get_value(cfg, "run_tag"))
    run_dir = skillvla_root / source_dataset / run_tag

    vla_root = project_root / str(get_value(cfg, "skillvla_outputs_root", "skillVLA_outputs"))
    stage1_run_name = str(get_value(cfg, "stage1_run_name")).strip()
    stage1_checkpoint = str(get_value(cfg, "stage1_checkpoint", "last")).strip() or "last"
    stage1_ckpt = vla_root / stage1_run_name / "checkpoints" / stage1_checkpoint / "pretrained_model"
    arch = _stage1_arch(stage1_ckpt)
    arch_tag = "A" if arch == "joint" else "B"

    batch_size = int(get_value(cfg, "batch_size", 16))
    num_gpus = int(get_value(cfg, "num_gpus", 1))
    lr_base = float(get_value(cfg, "lr_base", 2.5e-05))
    exp = str(get_value(cfg, "exp", "")).strip()

    # FSQ skill structure: auto-match the FSQ the dataset was built with (build_data fsq_levels).
    skill_fsq_levels = list(as_levels(get_value(cfg, "skill_fsq_levels", build_fsq_levels)))

    run_name = f"{source_dataset}_{run_tag}_batch{batch_size}_stage2_{arch_tag}"
    if exp:
        run_name = f"{run_name}_{exp}"
    output_dir = vla_root / f"S2_{run_name}"

    settings: dict = {
        # roots
        "project_root": project_root,
        "lerobot_root": lerobot_root,
        # dataset (skillvla dataset: current + skill-start obs + skill_code + actions)
        "source_dataset": source_dataset,
        "run_tag": run_tag,
        "skillvla_dataset_dir": run_dir / "skillvla",
        "fsq_ckpt": run_dir / "FSQ.pt",            # eval-time terminator (recorded in the checkpoint)
        "repo_id": f"dohyeon/{source_dataset}",
        # warm-start: pi05 → VLM, Stage-1 skill_expert → action expert / cond side
        "pi_base": str(get_value(cfg, "pi_base", "lerobot/pi05_base")),
        "stage1_run_name": stage1_run_name,
        "stage1_checkpoint": stage1_checkpoint,
        "stage1_checkpoint_path": stage1_ckpt,
        "expert_arch": arch,                        # informational (read from Stage-1 ckpt)
        # skill head / FSQ codebook
        "skill_fsq_levels": "[" + ",".join(str(x) for x in skill_fsq_levels) + "]",
        "skill_loss_weight": str(get_value(cfg, "skill_loss_weight", 0.5)),
        # freeze toggles (all parts otherwise trained)
        "freeze_vlm": as_bool(get_value(cfg, "freeze_vlm", False)),
        "freeze_vlm_vision": as_bool(get_value(cfg, "freeze_vlm_vision", False)),
        "freeze_cond_encoder": as_bool(get_value(cfg, "freeze_cond_encoder", True)),
        "freeze_action_expert": as_bool(get_value(cfg, "freeze_action_expert", False)),
        "freeze_expert_vision": as_bool(get_value(cfg, "freeze_expert_vision", False)),
        # output
        "skillvla_outputs_root": vla_root,
        "pt_run_name": run_name,
        "pt_output_dir": output_dir,
        # optimization
        "batch_size": batch_size,
        "num_workers": int(get_value(cfg, "num_workers", 4)),
        "num_gpus": num_gpus,
        "lr": lr_base * num_gpus,
        "steps": int(get_value(cfg, "steps", 100000)),
        "save_freq": int(get_value(cfg, "save_freq", 2500)),
        # wandb
        "wandb_enable": as_bool(get_value(cfg, "wandb_enable", True)),
        "wandb_project": str(get_value(cfg, "wandb_project", "VLA_stage2")),
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
