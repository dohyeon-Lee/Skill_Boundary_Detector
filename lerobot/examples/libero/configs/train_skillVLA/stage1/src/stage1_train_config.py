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
from train_skills_config import as_bool, as_levels, as_list, get_value, load_config, print_shell, resolve_path  # noqa: E402

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

    # Action chunk horizon. Longer chunks make the far future under-determined by the current
    # obs alone, pushing the flow-matching loss to actually use the skill (z_q) condition.
    # Weights are chunk-length agnostic (per-step projections + RoPE), so Stage-2 may
    # warm-start from a Stage-1 checkpoint trained with a different chunk_size.
    chunk_size = int(get_value(cfg, "chunk_size", 10))
    n_action_steps = get_value(cfg, "n_action_steps", None)
    n_action_steps = chunk_size if n_action_steps in (None, "", "null") else int(n_action_steps)

    # Conditioning: joint (cond-encoder ⊥ action expert); skill+progress ride the action-stream prefix.
    cond_encoder_variant = str(get_value(cfg, "cond_encoder_variant", "")).strip()
    if cond_encoder_variant.lower() in ("none", "null"):  # blank yaml → omit (use action expert's variant)
        cond_encoder_variant = ""
    state_cond_mode = str(get_value(cfg, "state_cond_mode", "state_skill")).strip().lower()  # state | state_skill
    skill_end_w = float(get_value(cfg, "skill_end_loss_weight", 1.0))      # R end weighting (action_weight only)
    action_weight = as_bool(get_value(cfg, "action_weight", False))        # per-sample sw-weight the action MSE

    init_from_pi05 = as_bool(get_value(cfg, "init_from_pi05", True))
    pi_base = resolve_path(project_root, get_value(cfg, "pi_base", "models/pi05_base")) if init_from_pi05 else ""

    dino_lr = get_value(cfg, "dino_lr", None)
    dino_lr_str = "" if dino_lr in (None, "", "null") else str(dino_lr)
    siglip_lr = get_value(cfg, "siglip_lr", None)
    siglip_lr_str = "" if siglip_lr in (None, "", "null") else str(siglip_lr)

    vision_backbone = (str(get_value(cfg, "vision_backbone", "dino")).strip().lower() or "dino")
    freeze_vision_encoder = as_bool(get_value(cfg, "freeze_vision_encoder", False))  # SELECTED backbone (NOT in run-name)

    # run-name: run_tag + backbone + batch + state-cond [+ weighted] [+ exp]. source_dataset /
    # freeze_vision_encoder / chunk_size are OMITTED (fixed — never swept). action_weight → "_weighted" ONLY
    # when true (plain = no tag).
    run_name = f"{run_tag}_{vision_backbone}_batch{batch_size}_{state_cond_mode}"
    if action_weight:
        run_name = f"{run_name}_weighted"
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

    # ── Co-trained terminator inputs (build_data products under run_dir → AUTO-derived from run_tag).
    # DINO token paths are attached ONLY when train_terminator (else "" → factory skips the wrappers). ──
    train_terminator = as_bool(get_value(cfg, "train_terminator", False))
    fsq_path_raw = str(get_value(cfg, "fsq_path", "")).strip()
    fsq_path = (resolve_path(project_root, fsq_path_raw)
                if fsq_path_raw and fsq_path_raw.lower() not in ("null", "none") else run_dir / "FSQ.pt")
    skill_decoder_dino_tokens_path = ""
    skill_decoder_dino_wrist_tokens_path = ""
    if train_terminator:
        sdd_raw = str(get_value(cfg, "skill_decoder_dino_tokens_path", "")).strip()
        skill_decoder_dino_tokens_path = (resolve_path(project_root, sdd_raw)
            if sdd_raw and sdd_raw.lower() not in ("null", "none") else run_dir / "dino.npz")
        # Wrist tokens: only a "both" FSQ build produces dino_wrist.npz → AUTO-attach when it EXISTS.
        sddw_raw = str(get_value(cfg, "skill_decoder_dino_wrist_tokens_path", "")).strip()
        if sddw_raw and sddw_raw.lower() not in ("null", "none"):
            skill_decoder_dino_wrist_tokens_path = resolve_path(project_root, sddw_raw)
        else:
            _wrist_cand = run_dir / "dino_wrist.npz"
            skill_decoder_dino_wrist_tokens_path = _wrist_cand if _wrist_cand.exists() else ""

    settings: dict = {
        # roots
        "project_root": project_root,
        "lerobot_root": lerobot_root,
        # dataset (raw skillvla dataset: images + skill columns + actions)
        "source_dataset": source_dataset,
        "run_tag": run_tag,
        "skillvla_dataset_dir": run_dir / "skillvla",
        "repo_id": f"dohyeon/{source_dataset}",
        # conditioning (joint only; skill+progress on the action prefix)
        "cond_encoder_variant": cond_encoder_variant,  # "" → same as action_expert_variant
        "state_cond_mode": state_cond_mode,       # state (skill=prefix token) | state_skill (skill→AdaRMS too)
        # model init
        "pi_base": pi_base,                       # "" → train the expert from scratch
        # vision encoder: "dino" or "siglip" (siglip warm-starts from pi_base's vision_tower)
        "vision_backbone": str(get_value(cfg, "vision_backbone", "dino")),
        "dino_model_path": resolve_path(project_root, get_value(cfg, "dino_model_path", "models/dinov3-vits16")),
        "dino_lr": dino_lr_str,                   # "" → same LR as the rest
        "freeze_vision_encoder": freeze_vision_encoder,  # freeze the SELECTED backbone (dino|siglip)
        "siglip_lr": siglip_lr_str,               # "" → same LR as the rest (when unfrozen)
        "siglip_image_size": int(get_value(cfg, "siglip_image_size", 224)),
        "skill_vocab_size": skill_vocab_size,
        "skill_fsq_levels": "[" + ",".join(str(x) for x in skill_fsq_levels) + "]",
        # output
        "skillvla_outputs_root": vla_root,
        "pt_run_name": run_name,
        "pt_output_dir": output_dir,
        # action chunk horizon
        "chunk_size": chunk_size,
        "n_action_steps": n_action_steps,
        # loss: optional cumulative-position term (λ) + end weighting (R) + aggregation mode. λ=0 → off.
        "skill_end_loss_weight": skill_end_w,    # R — end weighting (action_weight only)
        "action_weight": action_weight,          # per-sample sw-weight the action MSE → wandb loss_weighted
        # co-trained FSQ terminator (gradient-disjoint; AUTO-derived from run_tag)
        "train_terminator": train_terminator,
        "fsq_path": fsq_path,                     # {run_dir}/FSQ.pt (warm-start the terminator)
        "terminator_end_target_sigma": float(get_value(cfg, "terminator_end_target_sigma", 1.0)),
        "terminator_end_pos_weight": float(get_value(cfg, "terminator_end_pos_weight", 1.0)),
        "terminator_lr_scale": float(get_value(cfg, "terminator_lr_scale", 1.0)),
        "skill_decoder_dino_tokens_path": skill_decoder_dino_tokens_path,        # "" unless train_terminator
        "skill_decoder_dino_wrist_tokens_path": skill_decoder_dino_wrist_tokens_path,  # "" unless dino_wrist.npz exists
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
