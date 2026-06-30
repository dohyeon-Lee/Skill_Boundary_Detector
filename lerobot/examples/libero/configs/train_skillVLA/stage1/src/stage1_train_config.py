#!/usr/bin/env python3
"""Config for SkillVLA Stage-1 training (standalone action expert, policy.type=skill_expert).

Trains the action expert by flow matching on the build_data skillvla dataset (raw images +
skill columns + actions). Two experiments, each picked by WHICH .sbatch you submit:

  staged  → train_staged_1.sbatch (1-1: VSA = vision+state+skill, no Oracle) then
            train_staged_2.sbatch (1-2: freeze the VSA base, add the Oracle + r, CFG A/B dropout,
            warm-started from a chosen 1-1 checkpoint).
  single  → train_single.sbatch  (one run, Oracle on from scratch, CFG A/B dropout; no freeze).

The Oracle sweep knobs (r_dim, kl_weight, dropout_p) tag the 1-2 / single output folder so sweeps
never collide; the shared 1-1 (no Oracle) is reused across them. Emits shell exports (--shell).
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


def _fmt(x: float) -> str:
    """Compact float tag for folder names: 0.001 → '0.001', 0.5 → '0.5'."""
    return ("%g" % float(x))


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

    chunk_size = int(get_value(cfg, "chunk_size", 10))
    n_action_steps = get_value(cfg, "n_action_steps", None)
    n_action_steps = chunk_size if n_action_steps in (None, "", "null") else int(n_action_steps)

    cond_encoder_variant = str(get_value(cfg, "cond_encoder_variant", "")).strip()
    if cond_encoder_variant.lower() in ("none", "null"):  # blank yaml → omit (use action expert's variant)
        cond_encoder_variant = ""
    state_cond_mode = str(get_value(cfg, "state_cond_mode", "state")).strip().lower()  # state | state_skill

    # ── Oracle (Stage-1 residual r) hyperparameters + sweep knobs ──
    oracle_resample_n = int(get_value(cfg, "oracle_resample_n", 30))
    oracle_spline_degree = int(get_value(cfg, "oracle_spline_degree", 3))
    oracle_input_source = (str(get_value(cfg, "oracle_input_source", "state")).strip() or "state")
    oracle_width = int(get_value(cfg, "oracle_width", 512))
    oracle_depth = int(get_value(cfg, "oracle_depth", 3))
    oracle_n_heads = int(get_value(cfg, "oracle_n_heads", 8))
    oracle_n_tokens = int(get_value(cfg, "oracle_n_tokens", 1))
    oracle_r_dim = int(get_value(cfg, "oracle_r_dim", 16))           # SWEEP
    oracle_free_bits = float(get_value(cfg, "oracle_free_bits", 0.1))
    oracle_kl_weight = float(get_value(cfg, "oracle_kl_weight", 1e-3))  # SWEEP (β)
    oracle_dropout_p = float(get_value(cfg, "oracle_dropout_p", 0.5))   # SWEEP
    r_ablation_every = int(get_value(cfg, "r_ablation_every", 0))
    # Oracle sweep tag → distinguishes 1-2 / single output folders across a sweep. n{tokens}r{dim} is
    # the latent SHAPE (total bottleneck = n_tokens × r_dim); kl/dp are the other swept knobs.
    # Tag the Oracle input source (state | action) in the folder/run name so the two coexist distinctly.
    oracle_tag = (f"n{oracle_n_tokens}r{oracle_r_dim}_kl{_fmt(oracle_kl_weight)}"
                  f"_dp{_fmt(oracle_dropout_p)}_{oracle_input_source}")

    # Terminator inputs are build_data products under run_dir → AUTO-derived from run_tag.
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

    vision_backbone = (str(get_value(cfg, "vision_backbone", "dino")).strip().lower() or "dino")
    freeze_vision_encoder = as_bool(get_value(cfg, "freeze_vision_encoder", False))
    vis_tag = f"{vision_backbone}_{'freeze' if freeze_vision_encoder else 'unfreeze'}"

    # ── Output tree: {base}/staged/{1-1, 1-2_<ws>_<oracle>_<fv>} and {base}/single/<oracle>_<fv> ──
    base_name = f"{source_dataset}_{run_tag}_{vis_tag}_batch{batch_size}_{state_cond_mode}"
    if exp:
        base_name = f"{base_name}_{exp}"
    outputs_root = project_root / str(get_value(cfg, "outputs_root", "outputs"))
    vla_root = outputs_root / "skillVLA_stage1"
    base_dir = vla_root / base_name

    steps_total = int(get_value(cfg, "steps", 100000))                   # all phases run this many steps
    staged_1_1_dir = base_dir / "staged" / "1-1"                         # VSA base (no Oracle), shared across sweeps
    # 1-2 warm-starts from a CHOSEN 1-1 checkpoint (last | step number | explicit path).
    ws = str(get_value(cfg, "staged_phase2_warmstart", "last")).strip()
    if ws.lower() in ("", "last", "null", "none"):
        ws_tag, ws_path = "last", staged_1_1_dir / "checkpoints" / "last" / "pretrained_model"
    elif ws.isdigit():
        ws_tag = f"step{int(ws)}"
        # checkpoint dirs are zero-padded to max(6, len(str(total_steps))) digits (mirrors lerobot
        # train_utils.get_step_checkpoint_dir) → e.g. step 15000 → "015000", NOT "15000".
        ws_path = staged_1_1_dir / "checkpoints" / f"{int(ws):0{max(6, len(str(steps_total)))}d}" / "pretrained_model"
    else:
        _p = Path(ws)
        _name = _p.parent.name if _p.name == "pretrained_model" else _p.name
        ws_tag = re.sub(r"[^A-Za-z0-9]+", "_", _name).strip("_") or "custom"
        ws_path = _p
    # freeze_vsa_vision is a sweep knob in BOTH 1-2 and single → tag both folders so the variants coexist.
    fv_tag = "visfrozen" if as_bool(get_value(cfg, "freeze_vsa_vision", True)) else "visadapt"
    staged_1_2_dir = base_dir / "staged" / f"1-2_{ws_tag}_{oracle_tag}_{fv_tag}"  # ws + sweep + vision tags
    single_dir = base_dir / "single" / f"{oracle_tag}_{fv_tag}"          # Oracle from scratch: sweep + vision tags

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
        # conditioning
        "cond_encoder_variant": cond_encoder_variant,  # "" → same as action_expert_variant
        "state_cond_mode": state_cond_mode,            # state (skill=prefix) | state_skill (skill→AdaRMS)
        # model init
        "pi_base": pi_base,                            # "" → train the expert from scratch
        # vision encoder
        "vision_backbone": vision_backbone,
        "dino_model_path": resolve_path(project_root, get_value(cfg, "dino_model_path", "models/dinov3-vits16")),
        "dino_lr": dino_lr_str,
        "freeze_vision_encoder": freeze_vision_encoder,
        "siglip_lr": siglip_lr_str,
        "siglip_image_size": int(get_value(cfg, "siglip_image_size", 224)),
        "skill_vocab_size": skill_vocab_size,
        "skill_fsq_levels": "[" + ",".join(str(x) for x in skill_fsq_levels) + "]",
        # ── Oracle (state-trajectory → 1-token VAE residual r) ──
        "oracle_resample_n": oracle_resample_n,
        "oracle_spline_degree": oracle_spline_degree,
        "oracle_input_source": oracle_input_source,
        "oracle_width": oracle_width,
        "oracle_depth": oracle_depth,
        "oracle_n_heads": oracle_n_heads,
        "oracle_n_tokens": oracle_n_tokens,
        "oracle_r_dim": oracle_r_dim,
        "oracle_free_bits": oracle_free_bits,
        "oracle_kl_weight": oracle_kl_weight,
        "oracle_dropout_p": oracle_dropout_p,
        "r_ablation_every": r_ablation_every,
        "freeze_vsa_vision": as_bool(get_value(cfg, "freeze_vsa_vision", True)),  # staged 1-2: also freeze vision?
        "boundary_mode": str(get_value(cfg, "boundary_mode", "hold")).strip().lower(),
        # ── Output dirs (the sbatch picks the one for its experiment) ──
        "base_name": base_name,
        "staged_1_1_dir": staged_1_1_dir,
        "staged_1_2_dir": staged_1_2_dir,
        "staged_2_warmstart": ws_path,
        "single_dir": single_dir,
        "staged_1_1_name": f"{base_name}_staged_1-1",
        "staged_1_2_name": f"{base_name}_staged_1-2_{ws_tag}_{oracle_tag}_{fv_tag}",
        "single_name": f"{base_name}_single_{oracle_tag}_{fv_tag}",
        # ── Terminator co-train (orthogonal to the Oracle) ──
        "train_terminator": as_bool(get_value(cfg, "train_terminator", False)),
        "fsq_path": fsq_path,
        "terminator_end_target_sigma": float(get_value(cfg, "terminator_end_target_sigma", 1.0)),
        "terminator_end_pos_weight": float(get_value(cfg, "terminator_end_pos_weight", 1.0)),
        "terminator_lr_scale": float(get_value(cfg, "terminator_lr_scale", 1.0)),
        "skill_decoder_dino_tokens_path": skill_decoder_dino_tokens_path,   # auto: run_dir/dino.npz
        "skill_decoder_dino_output_key": str(get_value(cfg, "skill_decoder_dino_output_key", "skill_decoder_dino")),
        "skill_decoder_dino_cache_path": _resolve_opt(project_root, get_value(cfg, "skill_decoder_dino_cache_path", "")),
        "skill_decoder_dino_build_cache": as_bool(get_value(cfg, "skill_decoder_dino_build_cache", True)),
        "skill_decoder_dino_wrist_tokens_path": skill_decoder_dino_wrist_tokens_path,  # "" unless dino_wrist.npz exists
        # action chunk horizon
        "chunk_size": chunk_size,
        "n_action_steps": n_action_steps,
        # optimization
        "batch_size": batch_size,
        "num_workers": int(get_value(cfg, "num_workers", 8)),
        "num_gpus": num_gpus,
        "lr": lr_base * num_gpus,
        "steps": steps_total,
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
