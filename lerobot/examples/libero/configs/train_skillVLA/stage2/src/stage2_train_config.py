#!/usr/bin/env python3
"""Config for SkillVLA Stage-2 training (policy.type=skill_vla).

A PaliGemma VLM (warm-started from pi05_base) predicts the skill from the skill-START obs; an action
expert warm-started from a Stage-1 ``skill_expert`` checkpoint flow-matches the action chunk. The
Stage-1 checkpoint's config supplies vision_backbone, action_expert_variant, skill_vocab_size and
state_cond_mode — the model reads them itself, so here we only point to it. All roots are declared in
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
    #   {source}_{run_tag}_[{dino|siglip}_{freeze|unfreeze}_]batch{N}[_exp][_c{N}]
    # → the skillvla dataset (source + run_tag), the FSQ levels, and the Stage-1 policy vision tag
    # (backbone + freeze, e.g. "dino_unfreeze"/"siglip_freeze") captured below so the Stage-2 run name
    # records which vision encoder it warm-started from. (Arch is always joint now — no A/B tag.)
    # Vision tag is {dino|siglip}[_{freeze|unfreeze}] — the DP-branch training emitter drops the
    # freeze/unfreeze suffix (just "siglip"), so it is OPTIONAL inside the group.
    _rt = re.search(
        r"(FSQ\d+_dino\d+.*?)(?:_((?:dino|siglip)(?:_(?:freeze|unfreeze))?))?_batch\d+", stage1_run_name)
    if not _rt:
        raise ValueError(f"stage1_run_name must embed a 'FSQ..._dino..._batch<N>' run tag, got: {stage1_run_name}")
    run_tag = _rt.group(1)
    s1_vis_tag = _rt.group(2) or ""   # Stage-1 vision: dino_freeze / siglip_unfreeze / siglip / ... ("" if absent)
    # source_dataset: OLD naming embedded it as the stage1_run_name prefix ({source}_{run_tag}_...); the
    # DP-branch training emitter drops that prefix ({run_tag}_{backbone}_batch{N}_{state_cond}), so it's empty.
    # Prefer the prefix when present (back-compat), else fall back to the yaml's source_dataset.
    source_from_dir = stage1_run_name[: _rt.start()].rstrip("_")
    source_dataset = source_from_dir or str(get_value(cfg, "source_dataset", "")).strip()
    if not source_dataset:
        # DP naming has no prefix (and source_dataset isn't set) → AUTO-DERIVE: the run_tag lives under exactly
        # one source dir in the skillvla root. Set source_dataset in the yaml only if the SAME run_tag exists
        # under more than one source.
        cands = sorted({p.parent.name for p in skillvla_root.glob(f"*/{run_tag}") if p.is_dir()})
        if len(cands) == 1:
            source_dataset = cands[0]
        else:
            raise ValueError(
                f"Could not auto-derive source_dataset for run_tag {run_tag!r} under {skillvla_root}/*/ "
                f"(found {cands or 'none'}); set source_dataset in the Stage-2 yaml.")
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

    # run_name = {source}_{run_tag}_[{s1_vis_tag}_]{stage1_checkpoint}_{freeze|unfreeze}[_<loss tags>][_{exp}]:
    #   s1_vis_tag             = Stage-1 vision the expert warm-started from (dino_unfreeze/siglip_freeze/…)
    #   trailing {freeze|unfreeze} = Stage-2's own freeze_expert_vision choice.
    #   mode tag               = Stage-1 state_cond_mode (state / state_skill) — KEPT so the two modes don't
    #                            collide (their loss tags are otherwise identical).
    #   loss tags              = the Stage-1 loss-recipe tags (cum_ep/cum_all + ac/ac_w/ac_x + weighted),
    #                            KEPT so different Stage-1 loss recipes land in DISTINCT Stage-2 folders
    #                            (e.g. DP "_state" vs "_state_weighted" must not collide).
    #   connection tags       = inter-module attention edges (deviations from the default chain): ve
    #                            (vlm_expert on), noVc (vlm_cond off), noCe (cond_expert off), lang (attend_language on).
    # batch{N}/c{N}/np are dropped (arch is re-read from the Stage-1 ckpt at load time); exp is last.
    vis_tag = "freeze" if freeze_expert_vision else "unfreeze"
    # Inter-module attention connections (defaults: vlm_cond=T, cond_expert=T, vlm_expert=F, attend_language=F).
    attend_language = as_bool(get_value(cfg, "attend_language", False))
    vlm_cond = as_bool(get_value(cfg, "vlm_cond", True))
    cond_expert = as_bool(get_value(cfg, "cond_expert", True))
    vlm_expert = as_bool(get_value(cfg, "vlm_expert", False))    # action ← VLM directly (was action_attend_vlm)
    conn_tags = ([("ve" if vlm_expert else "")] + [("noVc" if not vlm_cond else "")]
                 + [("noCe" if not cond_expert else "")] + [("lang" if attend_language else "")])
    conn_tags = [t for t in conn_tags if t]
    vlm_dropout_p = float(get_value(cfg, "vlm_dropout_p", 0.0))               # CFG VLM dropout (0 = off)
    # Skill reader (separate post-VLM joint concat-KV probe module) + SkillHead dead-zone.
    skill_deadzone_frac = float(get_value(cfg, "skill_deadzone_frac", 0.0))   # 0 = plain MSE
    num_reader_tokens = int(get_value(cfg, "num_reader_tokens", 4))
    reader_depth = int(get_value(cfg, "reader_depth", 2))
    reader_heads = int(get_value(cfg, "reader_heads", 8))
    reader_tags = ([f"dz{skill_deadzone_frac:g}"] if skill_deadzone_frac > 0 else [])
    reader_tags += ([f"rp{num_reader_tokens}d{reader_depth}"] if (num_reader_tokens != 4 or reader_depth != 2) else [])
    # Per-regime freeze (nested dicts freeze_vlm_vsa / freeze_vsa → flat config fields). Only used when p>0.
    _fa = get_value(cfg, "freeze_vlm_vsa", {}) or {}   # A = VLM_VSA batches
    _fb = get_value(cfg, "freeze_vsa", {}) or {}       # B = VSA batches
    frz = {f"freeze_vlm_vsa_{k}": as_bool(_fa.get(k, False)) for k in ("expert", "cond", "vlm")}
    frz.update({f"freeze_vsa_{k}": as_bool(_fb.get(k, False)) for k in ("expert", "cond", "vlm")})
    # compact run-name tag so distinct freeze configs land in distinct folders (e.g. A freezes expert → frzAe)
    _aF = "".join(k[0] for k in ("expert", "cond", "vlm") if frz[f"freeze_vlm_vsa_{k}"])
    _bF = "".join(k[0] for k in ("expert", "cond", "vlm") if frz[f"freeze_vsa_{k}"])
    frz_tag = "frz" + (f"A{_aF}" if _aF else "") + (f"B{_bF}" if _bF else "")
    # state_skill checked first (it contains the substring "_state"); else state; else "" (older naming).
    mode_tag = ("state_skill" if "_state_skill" in stage1_run_name
                else "state" if "_state" in stage1_run_name else "")
    loss_tags = []
    for pat in (r"_(cum_(?:ep|all))(?:_|$)", r"_(ac_w|ac_x|ac)(?:_|$)", r"_(weighted)(?:_|$)"):
        m = re.search(pat, stage1_run_name)
        if m:
            loss_tags.append(m.group(1))
    parts = ([source_dataset, run_tag] + ([s1_vis_tag] if s1_vis_tag else [])
             + [stage1_checkpoint, vis_tag] + ([mode_tag] if mode_tag else []) + loss_tags
             + conn_tags + reader_tags
             + ([f"vdrop{vlm_dropout_p:g}"] if vlm_dropout_p > 0 else [])
             + ([frz_tag] if (vlm_dropout_p > 0 and (_aF or _bF)) else []))
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
        # skill reader (post-VLM probe module) + SkillHead regression dead-zone
        "skill_deadzone_frac": skill_deadzone_frac,
        "num_reader_tokens": num_reader_tokens,
        "reader_depth": reader_depth,
        "reader_heads": reader_heads,
        # inter-module attention connections (VLM/cond/expert edges + the language master switch)
        "attend_language": attend_language,
        "vlm_cond": vlm_cond,
        "cond_expert": cond_expert,
        "vlm_expert": vlm_expert,                   # action tokens read the VLM directly (was action_attend_vlm)
        "vlm_dropout_p": vlm_dropout_p,            # CFG-style VLM dropout prob (train only; 0 = off)
        **frz,                                      # per-regime freeze (freeze_{vlm_vsa,vsa}_{expert,cond,vlm})
        # terminator co-training (same as FT): adapt the FSQ terminator on this dataset's GT signals
        # (disjoint from the SkillVLA params). Warm-starts from fsq_ckpt; exported per-checkpoint for eval.
        "train_terminator": as_bool(get_value(cfg, "train_terminator", False)),
        "terminator_lr_scale": float(get_value(cfg, "terminator_lr_scale", 1.0)),
        "terminator_end_target_sigma": float(get_value(cfg, "terminator_end_target_sigma", 2.0)),
        "terminator_end_pos_weight": float(get_value(cfg, "terminator_end_pos_weight", 1.0)),
        "dino_tokens_path": run_dir / "dino.npz",   # current-frame DINO tokens for the terminator
        # Wrist DINO tokens — attached ONLY for a dual (terminator_use_wrist) FSQ, and only if built ("" else).
        "dino_wrist_tokens_path": (run_dir / "dino_wrist.npz") if (run_dir / "dino_wrist.npz").exists() else "",
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
