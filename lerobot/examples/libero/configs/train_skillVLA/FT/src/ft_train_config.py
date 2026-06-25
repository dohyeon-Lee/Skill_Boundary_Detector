#!/usr/bin/env python3
"""Config for SkillVLA FINETUNING (FT) — adapt a post-trained Stage-2 model to a NEW task.

Warm-starts the WHOLE skill_vla policy from a Stage-2 checkpoint (``pretrained_path`` → the model's
``from_pretrained`` takes the is_stage2 branch and full-loads VLM + cond + expert + skill head), then
continues training on the new task's skillvla dataset. The Stage-1 checkpoint path / skill_fsq_levels
are read from the Stage-2 checkpoint's config.json (the model still needs the Stage-1 *config* for its
architecture; no Stage-1 weights are reloaded). FT-specific behaviour:

  * cond_skill_source=pred : the action prefix is conditioned on the VLM's OWN predicted skill (STE),
                             matching inference; the flow loss backprops into the VLM trunk.
  * freeze: VLM (trunk + vision + skill-query) UNFROZEN; skill_head + cond_encoder + action expert
            (+ its vision) FROZEN — keep the motor repertoire, re-ground obs→skill for the new task.
  * train_terminator       : co-train the FSQ terminator on the new task's GT signals (disjoint graph)
                             and export an adapted FSQ checkpoint for eval.

Output: {project_root}/{outputs_root}/skillVLA_FT/{run_name}/. Emits shell exports (--shell).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import as_bool, as_levels, as_list, get_value, load_config, print_shell, resolve_path  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "ft_train_config.yaml"


def build_settings(cfg: dict) -> dict:
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    dataset_root = project_root / str(get_value(cfg, "dataset_root", "dataset"))
    skillvla_root = dataset_root / str(get_value(cfg, "skillvla_dataset_root", "skillvla_dataset"))
    lerobot_root = project_root / "lerobot"
    outputs_root = project_root / str(get_value(cfg, "outputs_root", "outputs"))

    # ── Warm-start: a trained Stage-2 run + checkpoint (full-loaded by the policy) ──
    stage2_vla_root = outputs_root / "skillVLA_stage2"
    stage2_run_name = str(get_value(cfg, "stage2_run_name")).strip()
    stage2_checkpoint = str(get_value(cfg, "stage2_checkpoint", "last")).strip() or "last"
    stage2_ckpt = stage2_vla_root / stage2_run_name / "checkpoints" / stage2_checkpoint / "pretrained_model"

    # The Stage-2 checkpoint config is the source of truth for the architecture: stage1_checkpoint_path
    # (Stage-1 config → vision/skill_vocab/state_cond_mode; no weights reloaded) and skill_fsq_levels.
    s2_cfg: dict = {}
    s2_cfg_json = stage2_ckpt / "config.json"
    if s2_cfg_json.is_file():
        s2_cfg = json.loads(s2_cfg_json.read_text())
    stage1_checkpoint_path = str(get_value(cfg, "stage1_checkpoint_path", "") or s2_cfg.get("stage1_checkpoint_path") or "")
    skill_fsq_levels = list(as_levels(get_value(cfg, "skill_fsq_levels", s2_cfg.get("skill_fsq_levels", [5, 5, 5]))))

    # ── New-task skillvla dataset (built by configs/train_skillVLA/build_data with the SAME FSQ) ──
    source_dataset = str(get_value(cfg, "source_dataset")).strip()
    run_tag = str(get_value(cfg, "run_tag")).strip()
    run_dir = skillvla_root / source_dataset / run_tag
    # FT terminator warm-start + current-frame DINO tokens live in the new task's run dir (same codebook).
    fsq_ckpt = run_dir / "FSQ.pt"
    dino_tokens_path = run_dir / "dino.npz"

    batch_size = int(get_value(cfg, "batch_size", 16))
    num_gpus = int(get_value(cfg, "num_gpus", 1))
    lr_base = float(get_value(cfg, "lr_base", 2.5e-05))
    exp = str(get_value(cfg, "exp", "")).strip()

    # skill_loss_weight: blank in the yaml → inherit the Stage-2 checkpoint's value.
    slw = get_value(cfg, "skill_loss_weight", None)
    skill_loss_weight = s2_cfg.get("skill_loss_weight", 0.1) if slw in (None, "", "null") else slw

    cond_skill_source = str(get_value(cfg, "cond_skill_source", "pred")).strip() or "pred"
    if cond_skill_source not in ("gt", "pred"):
        raise ValueError(f"cond_skill_source must be 'gt' or 'pred', got {cond_skill_source!r}")
    train_terminator = as_bool(get_value(cfg, "train_terminator", True))

    # Attention toggles: INHERIT from the forked Stage-2 checkpoint (must MATCH it — pretrained_path full-
    # loads the weights, and the action expert was trained with that attention mask). Blank yaml → inherit
    # from the Stage-2 config.json; set explicitly only to ablate.
    def _inherit(key):
        v = get_value(cfg, key, None)
        return as_bool(s2_cfg.get(key, False)) if v in (None, "", "null") else as_bool(v)
    action_attend_vlm = _inherit("action_attend_vlm")
    cond_attend_language = _inherit("cond_attend_language")

    # run_name: new task + run_tag + ft{Stage-2 ckpt} + the Stage-2 SOURCE variant (so FT forks of DIFFERENT
    # Stage-2 models don't collide — backbone / state_cond_mode / cum) + action↔VLM + the FT variant tags.
    s2tags = []
    for pat in (r"_(siglip|dino)_(?:freeze|unfreeze)",   # Stage-1 vision backbone of the forked Stage-2
                r"_(state_skill|state)(?:_|$)",          # state_cond_mode
                r"_(cum_(?:ep|all))(?:_|$)"):            # cumulative-loss variant
        m = re.search(pat, stage2_run_name)
        if m:
            s2tags.append(m.group(1))
    if action_attend_vlm:                                # resolved value (not name-parse) → run-name matches config
        s2tags.append("actvlm")
    tags = [cond_skill_source]          # skill source ALWAYS tagged: "pred" | "gt"
    if train_terminator:
        tags.append("term")
    run_name = f"{source_dataset}_{run_tag}_ft{stage2_checkpoint}"
    parts = s2tags + tags
    if parts:
        run_name = run_name + "_" + "_".join(parts)
    if exp:
        run_name = f"{run_name}_{exp}"
    vla_root = outputs_root / "skillVLA_FT"
    output_dir = vla_root / run_name

    settings: dict = {
        # roots
        "project_root": project_root,
        "lerobot_root": lerobot_root,
        # dataset (new task)
        "source_dataset": source_dataset,
        "run_tag": run_tag,
        "skillvla_dataset_dir": run_dir / "skillvla",
        "fsq_ckpt": fsq_ckpt,                       # terminator warm-start + (eval terminator base)
        "dino_tokens_path": dino_tokens_path,       # current-frame DINO tokens for terminator co-train
        "repo_id": f"dohyeon/{source_dataset}",
        # warm-start (full policy from Stage-2) + architecture config (from its config.json)
        "stage2_run_name": stage2_run_name,
        "stage2_checkpoint": stage2_checkpoint,
        "stage2_checkpoint_path": stage2_ckpt,
        "stage1_checkpoint_path": stage1_checkpoint_path,
        "skill_fsq_levels": "[" + ",".join(str(x) for x in skill_fsq_levels) + "]",
        # attention — inherited from the Stage-2 ckpt (must match the forked weights)
        "action_attend_vlm": action_attend_vlm,
        "cond_attend_language": cond_attend_language,
        # FT behaviour
        "cond_skill_source": cond_skill_source,
        "train_terminator": train_terminator,
        "terminator_lr_scale": float(get_value(cfg, "terminator_lr_scale", 1.0)),
        "terminator_end_target_sigma": float(get_value(cfg, "terminator_end_target_sigma", 2.0)),
        "terminator_end_pos_weight": float(get_value(cfg, "terminator_end_pos_weight", 1.0)),
        "skill_loss_weight": str(skill_loss_weight),
        # freeze toggles (FT default: VLM-only finetune)
        "freeze_vlm": as_bool(get_value(cfg, "freeze_vlm", False)),
        "freeze_vlm_vision": as_bool(get_value(cfg, "freeze_vlm_vision", False)),
        "freeze_skill_head": as_bool(get_value(cfg, "freeze_skill_head", True)),
        "freeze_cond_encoder": as_bool(get_value(cfg, "freeze_cond_encoder", True)),
        "freeze_action_expert": as_bool(get_value(cfg, "freeze_action_expert", True)),
        "freeze_expert_vision": as_bool(get_value(cfg, "freeze_expert_vision", True)),
        # output
        "skillvla_outputs_root": vla_root,
        "pt_run_name": run_name,
        "pt_output_dir": output_dir,
        # optimization
        "batch_size": batch_size,
        "num_workers": int(get_value(cfg, "num_workers", 4)),
        "num_gpus": num_gpus,
        "lr": lr_base * num_gpus,
        "steps": int(get_value(cfg, "steps", 30000)),
        "save_freq": int(get_value(cfg, "save_freq", 5000)),
        # wandb
        "wandb_enable": as_bool(get_value(cfg, "wandb_enable", True)),
        "wandb_project": str(get_value(cfg, "wandb_project", "VLA_FT")),
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
