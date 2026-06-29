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


def _resolve_opt(project_root, raw) -> str:
    """Resolve an OPTIONAL path: blank / null / none → "" (omitted); else project_root-relative resolve."""
    s = str(raw).strip()
    return resolve_path(project_root, s) if s and s.lower() not in ("null", "none") else ""


def build_settings(cfg: dict, mode: str = "joint", loss_mode_arg: str = "plain") -> dict:
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
    skill_end_w = float(get_value(cfg, "skill_end_loss_weight", 1.0))      # R = progress-weighting strength (early/late)

    # ── Stage-1 experiment design: connector on/off + schedule + (joint) loss_mode (3 runs) ──
    use_connector = as_bool(get_value(cfg, "use_connector", True))
    schedule = str(mode).strip().lower()                                      # joint | staged (from the .sbatch file)
    if schedule not in ("joint", "staged"):
        raise ValueError(f"--mode must be 'joint' or 'staged' (got {mode!r}).")
    loss_mode = str(loss_mode_arg).strip().lower()                           # plain | weighted_gated (from --loss-mode)
    if loss_mode not in ("plain", "weighted_gated"):
        raise ValueError(f"--loss-mode must be 'plain' or 'weighted_gated' (got {loss_mode_arg!r}).")
    # Terminator inputs are build_data products under run_dir → AUTO-derived from run_tag (like the
    # dataset). Override only if a yaml path is given. (Same as Stage-2: FSQ.pt + dino.npz beside skillvla/.)
    fsq_path_raw = str(get_value(cfg, "fsq_path", "")).strip()
    fsq_path = (
        resolve_path(project_root, fsq_path_raw)
        if fsq_path_raw and fsq_path_raw.lower() not in ("null", "none") else run_dir / "FSQ.pt"
    )
    sdd_raw = str(get_value(cfg, "skill_decoder_dino_tokens_path", "")).strip()
    skill_decoder_dino_tokens_path = (
        resolve_path(project_root, sdd_raw)
        if sdd_raw and sdd_raw.lower() not in ("null", "none") else run_dir / "dino.npz"
    )
    # Wrist tokens (dual terminator). Only a "both" FSQ build produces dino_wrist.npz → AUTO-attach when
    # it EXISTS (a 3rd-only "wow" build has none, so leave blank). Override with an explicit yaml path.
    sddw_raw = str(get_value(cfg, "skill_decoder_dino_wrist_tokens_path", "")).strip()
    if sddw_raw and sddw_raw.lower() not in ("null", "none"):
        skill_decoder_dino_wrist_tokens_path = resolve_path(project_root, sddw_raw)
    else:
        _wrist_cand = run_dir / "dino_wrist.npz"
        skill_decoder_dino_wrist_tokens_path = _wrist_cand if _wrist_cand.exists() else ""

    init_from_pi05 = as_bool(get_value(cfg, "init_from_pi05", True))
    pi_base = resolve_path(project_root, get_value(cfg, "pi_base", "models/pi05_base")) if init_from_pi05 else ""

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

    # joint + ae + discretized state is the only arch now → no arch/inject tag. init_from_pi05 fixed true.
    run_name = f"{source_dataset}_{run_tag}_{vis_tag}_batch{batch_size}"
    run_name = f"{run_name}_{state_cond_mode}"   # state | state_skill (what rides the expert AdaRMS) — never collide
    # experiment tag so the 3 runs (joint-plain / joint-gated / staged) never collide.
    exp_tag = ("conn" if use_connector else "noconn") + f"_{schedule}"
    if schedule == "joint":
        exp_tag += f"_{loss_mode}"
    run_name = f"{run_name}_{exp_tag}"
    if exp:
        run_name = f"{run_name}_{exp}"
    # Single outputs root from yaml; the per-stage subdir is fixed here (not in yaml).
    outputs_root = project_root / str(get_value(cfg, "outputs_root", "outputs"))
    vla_root = outputs_root / "skillVLA_stage1"
    output_dir = vla_root / run_name   # under skillVLA_stage1/, so no extra stage prefix

    # ── Staged = TWO sequential runs in the sbatch → SEPARATE folders {run_name}/1-1 and
    # {run_name}/1-2_<warmstart-tag>. Both phases run `steps`. 1-1: expert-vision + action
    # (use_connector=false, late-weighted). 1-2: freeze expert-vision, add connector (plain),
    # warm-started from a CHOSEN 1-1 checkpoint (the warm-start tag lets different choices coexist). ──
    staged = (schedule == "staged")
    phase1_output = phase2_output = ""
    phase1_weighting = "late"
    phase2_warmstart = ""
    if staged:
        phase1_weighting = str(get_value(cfg, "staged_phase1_weighting", "late")).strip().lower()
        steps_total = int(get_value(cfg, "steps", 100000))           # both phases run this many steps
        phase1_output = output_dir / f"1-1_{phase1_weighting}"       # weighting tag → late/plain don't collide
        ws = str(get_value(cfg, "staged_phase2_warmstart", "last")).strip()
        if ws.lower() in ("", "last", "null", "none"):
            ws_tag = "last"
            phase2_warmstart = phase1_output / "checkpoints" / "last" / "pretrained_model"
        elif ws.isdigit():                                            # a 1-1 step number → padded ckpt dir
            ws_tag = f"step{int(ws)}"
            phase2_warmstart = phase1_output / "checkpoints" / f"{int(ws):0{len(str(steps_total))}d}" / "pretrained_model"
        else:                                                         # explicit checkpoint path
            _p = Path(ws)
            _name = _p.parent.name if _p.name == "pretrained_model" else _p.name
            ws_tag = re.sub(r"[^A-Za-z0-9]+", "_", _name).strip("_") or "custom"
            phase2_warmstart = _p
        phase2_output = output_dir / f"1-2_{ws_tag}"                  # warm-start tag → choices don't collide

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
        # conditioning (joint only; skill+progress on the action prefix)
        "cond_encoder_variant": cond_encoder_variant,  # "" → same as action_expert_variant
        "state_cond_mode": state_cond_mode,       # state (skill=prefix token) | state_skill (skill→AdaRMS too)
        # model init
        "pi_base": pi_base,                       # "" → train the expert from scratch
        # vision encoder: "dino" or "siglip" (siglip warm-starts from pi_base's vision_tower)
        "vision_backbone": str(get_value(cfg, "vision_backbone", "dino")),
        "dino_model_path": resolve_path(project_root, get_value(cfg, "dino_model_path", "models/dinov3-vits16")),
        "dino_lr": dino_lr_str,                   # "" → same LR as the rest
        "freeze_dino": as_bool(get_value(cfg, "freeze_dino", False)),
        "freeze_siglip": as_bool(get_value(cfg, "freeze_siglip", True)),
        "siglip_lr": siglip_lr_str,               # "" → same LR as the rest (when unfrozen)
        "siglip_image_size": int(get_value(cfg, "siglip_image_size", 224)),
        "skill_vocab_size": skill_vocab_size,
        "skill_fsq_levels": "[" + ",".join(str(x) for x in skill_fsq_levels) + "]",
        # ── Connector (Stage-1 future-conditioning → VAE latent z) ──
        "use_connector": use_connector,
        "connector_dino_model_path": resolve_path(project_root, get_value(cfg, "connector_dino_model_path", "models/dinov3-vitb16")),
        "connector_dino_image_size": int(get_value(cfg, "connector_dino_image_size", 224)),
        "connector_width": int(get_value(cfg, "connector_width", 768)),
        "connector_depth": int(get_value(cfg, "connector_depth", 4)),
        "connector_n_heads": int(get_value(cfg, "connector_n_heads", 8)),
        "connector_n_latents": int(get_value(cfg, "connector_n_latents", 4)),
        "connector_z_dim": int(get_value(cfg, "connector_z_dim", 64)),
        "connector_free_bits": float(get_value(cfg, "connector_free_bits", 0.1)),
        "connector_kl_weight": float(get_value(cfg, "connector_kl_weight", 1e-3)),
        "connector_z_consistency_weight": float(get_value(cfg, "connector_z_consistency_weight", 0.0)),
        "z_ablation_every": int(get_value(cfg, "z_ablation_every", 0)),
        # ── Stage-1 experiment design — action_weighting & freeze_expert_vision are decided by the
        # .sbatch (joint→plain/false, staged_1→phase1_weighting/false, staged_2→plain/true). ──
        "loss_mode": loss_mode,                   # plain | weighted_gated
        "gate_prob": float(get_value(cfg, "gate_prob", 0.5)),
        "boundary_mode": str(get_value(cfg, "boundary_mode", "hold")).strip().lower(),
        # staged orchestration (sbatch runs phase 1-1 then 1-2 → separate folders)
        "staged": staged,
        "staged_phase1_weighting": phase1_weighting,
        "phase1_output_dir": phase1_output,
        "phase2_output_dir": phase2_output,
        "phase2_warmstart": phase2_warmstart,
        "train_terminator": as_bool(get_value(cfg, "train_terminator", False)),
        "fsq_path": fsq_path,                     # "" → terminator co-train off / no eval terminator
        "terminator_end_target_sigma": float(get_value(cfg, "terminator_end_target_sigma", 1.0)),
        "terminator_end_pos_weight": float(get_value(cfg, "terminator_end_pos_weight", 1.0)),
        "terminator_lr_scale": float(get_value(cfg, "terminator_lr_scale", 1.0)),
        # current-frame FSQ-grid DINO tokens for the terminator (attached by SkillVLADinoTokenDataset)
        "skill_decoder_dino_tokens_path": skill_decoder_dino_tokens_path,   # auto: run_dir/dino.npz
        "skill_decoder_dino_output_key": str(get_value(cfg, "skill_decoder_dino_output_key", "skill_decoder_dino")),
        "skill_decoder_dino_cache_path": _resolve_opt(project_root, get_value(cfg, "skill_decoder_dino_cache_path", "")),
        "skill_decoder_dino_build_cache": as_bool(get_value(cfg, "skill_decoder_dino_build_cache", True)),
        # wrist tokens for a dual (use_wrist) terminator — "" unless run_dir/dino_wrist.npz exists
        "skill_decoder_dino_wrist_tokens_path": skill_decoder_dino_wrist_tokens_path,
        # output
        "skillvla_outputs_root": vla_root,
        "pt_run_name": run_name,
        "pt_output_dir": output_dir,
        # action chunk horizon
        "chunk_size": chunk_size,
        "n_action_steps": n_action_steps,
        # loss: optional cumulative-position term (λ) + end weighting (R) + aggregation mode. λ=0 → off.
        "skill_end_loss_weight": skill_end_w,     # R = progress-weighting strength (early/late)
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
    ap.add_argument("--mode", default="joint", choices=["joint", "staged"])         # which .sbatch is running
    ap.add_argument("--loss-mode", default="plain", choices=["plain", "weighted_gated"])  # joint only
    ap.add_argument("--shell", action="store_true")
    args = ap.parse_args()
    settings = build_settings(load_config(args.config), mode=args.mode, loss_mode_arg=args.loss_mode)
    if args.shell:
        print_shell(settings)
    else:
        for k, v in settings.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
