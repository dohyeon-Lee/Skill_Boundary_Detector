#!/usr/bin/env python3
"""Config for SkillVLA FINETUNING (FT) — adapt a post-trained Stage-2 model to a NEW task.

Warm-starts the WHOLE skill_vla policy from a Stage-2 checkpoint (``pretrained_path`` → the model's
``from_pretrained`` takes the is_stage2 branch and full-loads VLM + cond + expert + skill head), then
continues training on the new task's skillvla dataset. The Stage-1 checkpoint path / skill_fsq_levels
are read from the Stage-2 checkpoint's config.json (the model still needs the Stage-1 *config* for its
architecture; no Stage-1 weights are reloaded). FT-specific behaviour:

  * cond_skill_source=pred : the action prefix is conditioned on the VLM's OWN predicted skill (STE),
                             matching inference; the flow loss backprops into the VLM trunk.
  * freeze: via the freeze_vlm_vsa dict (FT is always p=0 → it applies statically; same single source
            as Stage-2). Default: VLM (trunk + vision) UNFROZEN, cond pipeline frozen, skill decoder
            (reader + head) frozen — keep the motor repertoire, re-ground obs→skill for the new task.
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
    def _inherit(key, default=False):
        v = get_value(cfg, key, None)
        return as_bool(s2_cfg.get(key, default)) if v in (None, "", "null") else as_bool(v)
    # Inter-module attention connections — inherited from the forked Stage-2 ckpt (must match its weights).
    attend_language = _inherit("attend_language", False)
    attend_image = _inherit("attend_image", True)
    vlm_cond = _inherit("vlm_cond", True)
    cond_expert = _inherit("cond_expert", True)
    vlm_expert = _inherit("vlm_expert", False)

    # Skill reader ARCHITECTURE — inherited from the Stage-2 ckpt (the reader weights are full-loaded and
    # frozen in FT; its shape MUST match). skill_deadzone_frac is a loss param (head frozen → no grad); keep.
    def _inherit_num(key, default, cast):
        v = get_value(cfg, key, None)
        return cast(s2_cfg.get(key, default)) if v in (None, "", "null") else cast(v)
    num_reader_tokens = _inherit_num("num_reader_tokens", 4, int)
    reader_depth = _inherit_num("reader_depth", 2, int)
    reader_heads = _inherit_num("reader_heads", 8, int)
    skill_deadzone_frac = _inherit_num("skill_deadzone_frac", 0.0, float)

    # SCRATCH-parent support: a Stage-2 trained WITHOUT Stage-1 has stage1_checkpoint_path="" in its
    # config — the FT model then synthesizes the Stage-1-side architecture from these two knobs, which
    # must therefore be inherited from the parent (they default to siglip/state if the parent predates them).
    scratch_parent = not stage1_checkpoint_path
    s1_vision_backbone = str(s2_cfg.get("s1_vision_backbone", "siglip"))
    s1_state_cond_mode = str(s2_cfg.get("s1_state_cond_mode", "state"))

    # Per-regime freeze (stage2와 동일한 단일 소스): FT도 vlm_dropout p>0이면 배치마다 A/B 토글,
    # p=0이면 A 딕셔너리가 정적 적용. FT 기본 A = {expert:t, cond:t, llm:f, vlm_vision:f} (모터 동결,
    # VLM 재접지). "vlm"은 llm의 레거시 별칭으로 허용.
    _RKEYS = ("expert", "cond", "llm", "vlm_vision")
    _FA_DEF = {"expert": True, "cond": True, "llm": False, "vlm_vision": False}
    _fa = get_value(cfg, "freeze_vlm_vsa", {}) or {}
    _fb = get_value(cfg, "freeze_vsa", {}) or {}
    def _grp(d, k, dflt):
        return as_bool(d.get(k, d.get("vlm", dflt) if k == "llm" else dflt))
    frz = {f"freeze_vlm_vsa_{k}": _grp(_fa, k, _FA_DEF[k]) for k in _RKEYS}
    frz.update({f"freeze_vsa_{k}": _grp(_fb, k, False) for k in _RKEYS})

    # VLM dropout (+선형 스케줄) — stage2와 동일. FT에선 pred-skill 모드와 조합되어 B 배치 =
    # "자기 예측 skill만으로 VLM 없이 실행" 훈련이 됨.
    vlm_dropout_p = float(get_value(cfg, "vlm_dropout_p", 0.0))
    _pe = get_value(cfg, "vlm_dropout_p_end", None)
    vlm_dropout_p_end = None if str(_pe).lower() in ("none", "", "null", "~") else float(_pe)
    vlm_dropout_decay_steps = int(get_value(cfg, "vlm_dropout_decay_steps", 0) or 0)
    sched_on = vlm_dropout_p_end is not None and vlm_dropout_decay_steps > 0
    regimes_ever = vlm_dropout_p > 0 or (sched_on and vlm_dropout_p_end > 0)

    # run_name = {ft_source}_{ft_run_tag}_ft{ckpt}[_scratch]_{pred|gt}[_term][_{P}p_{A}_{B}]__{parent_regime}[_{exp}]
    #   parent_regime = 부모 Stage-2 폴더명의 "__" 뒤 태그({conn}_{P}p_{A}[_{B}])를 그대로 승계.
    #   FT 자신이 dropout을 쓰면({P}p 태그) 자기 p/freeze(A/B tf문자열)도 tags에 추가됨.
    parent_regime = stage2_run_name.split("__", 1)[1] if "__" in stage2_run_name else ""
    if sched_on:
        p_tag = (f"{int(round(vlm_dropout_p * 100))}to{int(round(vlm_dropout_p_end * 100))}p"
                 f"{vlm_dropout_decay_steps // 1000}k")
    else:
        p_tag = f"{int(round(vlm_dropout_p * 100))}p"
    _tf = lambda pfx: "".join("t" if frz[f"freeze_{pfx}_{k}"] else "f" for k in _RKEYS)
    tags = [cond_skill_source] + (["term"] if train_terminator else [])   # skill source ALWAYS tagged
    if regimes_ever:
        tags.append(f"{p_tag}_{_tf('vlm_vsa')}_{_tf('vsa')}")
    parts = ([source_dataset, run_tag, f"ft{stage2_checkpoint}"]
             + (["scratch"] if scratch_parent else []) + tags)
    run_name = "_".join(parts) + (f"__{parent_regime}" if parent_regime else "")
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
        "stage1_checkpoint_path": stage1_checkpoint_path,   # "" = scratch-parent (arch from the s1_* knobs)
        "s1_vision_backbone": s1_vision_backbone,
        "s1_state_cond_mode": s1_state_cond_mode,
        "skill_fsq_levels": "[" + ",".join(str(x) for x in skill_fsq_levels) + "]",
        # attention connections — inherited from the Stage-2 ckpt (must match the forked weights)
        "attend_language": attend_language,
        "attend_image": attend_image,
        "vlm_cond": vlm_cond,
        "cond_expert": cond_expert,
        "vlm_expert": vlm_expert,
        # skill reader architecture (inherited from the Stage-2 ckpt — reader is loaded + frozen in FT)
        "num_reader_tokens": num_reader_tokens,
        "reader_depth": reader_depth,
        "reader_heads": reader_heads,
        "skill_deadzone_frac": skill_deadzone_frac,
        # FT behaviour
        "cond_skill_source": cond_skill_source,
        "train_terminator": train_terminator,
        "terminator_lr_scale": float(get_value(cfg, "terminator_lr_scale", 1.0)),
        "terminator_end_target_sigma": float(get_value(cfg, "terminator_end_target_sigma", 2.0)),
        "terminator_end_pos_weight": float(get_value(cfg, "terminator_end_pos_weight", 1.0)),
        "skill_loss_weight": str(skill_loss_weight),
        # freeze — stage2와 동일한 단일 소스 (p=0 → A 딕셔너리 정적; p>0 → 배치별 A/B 토글)
        **frz,
        "freeze_skill_head": as_bool(get_value(cfg, "freeze_skill_head", True)),
        # VLM dropout (+스케줄) — stage2와 동일
        "vlm_dropout_p": vlm_dropout_p,
        "vlm_dropout_p_end": "" if vlm_dropout_p_end is None else vlm_dropout_p_end,
        "vlm_dropout_decay_steps": vlm_dropout_decay_steps,
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
