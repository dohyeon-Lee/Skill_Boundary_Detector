#!/usr/bin/env python3
"""Config for SkillVLA Stage-2 training (policy.type=skill_vla).

A PaliGemma VLM (warm-started from pi05_base) predicts the skill from the skill-START obs; an action
expert warm-started from a Stage-1 ``skill_expert`` checkpoint flow-matches the action chunk. The
Stage-1 checkpoint's config supplies vision_backbone, action_expert_variant, skill_vocab_size and
state_n_bins — the model reads them itself, so here we only point to it. All roots are declared in
this yaml (standalone); source/run_tag/FSQ levels are parsed from stage1_run_name; FSQ.pt (eval
terminator) lives in the run dir. Emits shell exports (--shell).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import as_bool, as_levels, as_list, get_value, load_config, print_shell, resolve_path  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "stage2_train_config.yaml"


def build_settings(cfg: dict) -> dict:
    # Standalone: every root is declared in this yaml (no build_data dependency).
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    dataset_root = project_root / str(get_value(cfg, "dataset_root", "dataset"))
    skillvla_root = dataset_root / str(get_value(cfg, "skillvla_dataset_root", "skillvla_dataset"))
    lerobot_root = project_root / "lerobot"

    # Single outputs root from yaml; per-stage subdirs fixed here. Warm-start lives in stage1's.
    outputs_root = project_root / str(get_value(cfg, "outputs_root", "outputs"))
    stage1_vla_root = outputs_root / "skillVLA_stage1"
    vla_root = outputs_root / "skillVLA_stage2"

    stage1_run_name = str(get_value(cfg, "stage1_run_name")).strip()
    stage1_checkpoint = str(get_value(cfg, "stage1_checkpoint", "last")).strip() or "last"
    stage1_ckpt = stage1_vla_root / stage1_run_name / "checkpoints" / stage1_checkpoint / "pretrained_model"

    # Everything is parsed from the stage1_run_name:
    #   {source}_{run_tag}_[{dino|siglip}_{freeze|unfreeze}_]batch{N}[_np][_exp][_c{N}]
    # → the skillvla dataset (source + run_tag), the FSQ levels, and the Stage-1 policy vision tag
    # (backbone + freeze, e.g. "dino_unfreeze"/"siglip_freeze") captured below so the Stage-2 run name
    # records which vision encoder it warm-started from. (Arch is always joint now — no A/B tag.)
    _rt = re.search(
        r"(FSQ\d+_dino\d+.*?)(?:_((?:dino|siglip)_(?:freeze|unfreeze)))?_batch\d+", stage1_run_name)
    if not _rt:
        raise ValueError(f"stage1_run_name must embed a 'FSQ..._dino..._batch<N>' run tag, got: {stage1_run_name}")
    run_tag = _rt.group(1)
    s1_vis_tag = _rt.group(2) or ""   # Stage-1 vision: dino_freeze / siglip_unfreeze / ... ("" if absent)
    source_dataset = stage1_run_name[: _rt.start()].rstrip("_")
    run_dir = skillvla_root / source_dataset / run_tag
    build_fsq_levels = [int(d) for d in re.search(r"FSQ(\d+)", run_tag).group(1)]

    batch_size = int(get_value(cfg, "batch_size", 16))
    num_gpus = int(get_value(cfg, "num_gpus", 1))
    lr_base = float(get_value(cfg, "lr_base", 2.5e-05))
    exp = str(get_value(cfg, "exp", "")).strip()
    freeze_cond_encoder = as_bool(get_value(cfg, "freeze_cond_encoder", True))
    freeze_expert_vision = as_bool(get_value(cfg, "freeze_expert_vision", False))

    # FSQ skill structure: auto-match the FSQ the dataset was built with (parsed from run_tag).
    skill_fsq_levels = list(as_levels(get_value(cfg, "skill_fsq_levels", build_fsq_levels)))

    # run_name = {source}_{run_tag}_[{s1_vis_tag}_]{stage1_checkpoint}_{freeze|unfreeze}[_{exp}]:
    #   s1_vis_tag             = Stage-1 vision the expert warm-started from (dino_unfreeze/siglip_freeze/…)
    #   trailing {freeze|unfreeze} = Stage-2's own freeze_expert_vision choice.
    # batch{N}/arch(A/B) are dropped (arch is re-read from the Stage-1 ckpt at load time); exp is last.
    vis_tag = "freeze" if freeze_expert_vision else "unfreeze"
    parts = [source_dataset, run_tag] + ([s1_vis_tag] if s1_vis_tag else []) + [stage1_checkpoint, vis_tag]
    run_name = "_".join(parts)
    if exp:
        run_name = f"{run_name}_{exp}"
    output_dir = vla_root / run_name   # under skillVLA_stage2/, so no extra stage prefix

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
        "pi_base": resolve_path(project_root, get_value(cfg, "pi_base", "models/pi05_base")),
        "stage1_run_name": stage1_run_name,
        "stage1_checkpoint": stage1_checkpoint,
        "stage1_checkpoint_path": stage1_ckpt,
        # skill head / FSQ codebook
        "skill_fsq_levels": "[" + ",".join(str(x) for x in skill_fsq_levels) + "]",
        "skill_loss_weight": str(get_value(cfg, "skill_loss_weight", 0.5)),
        "cond_attend_language": as_bool(get_value(cfg, "cond_attend_language", False)),
        # terminator co-training (same as FT): adapt the FSQ terminator on this dataset's GT signals
        # (disjoint from the SkillVLA params). Warm-starts from fsq_ckpt; exported per-checkpoint for eval.
        "train_terminator": as_bool(get_value(cfg, "train_terminator", False)),
        "terminator_lr_scale": float(get_value(cfg, "terminator_lr_scale", 1.0)),
        "terminator_end_target_sigma": float(get_value(cfg, "terminator_end_target_sigma", 2.0)),
        "terminator_end_pos_weight": float(get_value(cfg, "terminator_end_pos_weight", 1.0)),
        "dino_tokens_path": run_dir / "dino.npz",   # current-frame DINO tokens for the terminator
        # freeze toggles (all parts otherwise trained)
        "freeze_vlm": as_bool(get_value(cfg, "freeze_vlm", False)),
        "freeze_vlm_vision": as_bool(get_value(cfg, "freeze_vlm_vision", False)),
        "freeze_cond_encoder": freeze_cond_encoder,
        "freeze_action_expert": as_bool(get_value(cfg, "freeze_action_expert", False)),
        "freeze_expert_vision": freeze_expert_vision,
        # output
        "skillvla_outputs_root": vla_root,
        "pt_run_name": run_name,
        "pt_output_dir": output_dir,
        # optimization
        "batch_size": batch_size,
        "num_workers": int(get_value(cfg, "num_workers", 4)),
        "num_gpus": num_gpus,
        "lr": lr_base * num_gpus,
        "expert_lr_scale": float(get_value(cfg, "expert_lr_scale", 1.0)),
        "cond_lr_scale": float(get_value(cfg, "cond_lr_scale", 1.0)),
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
