#!/usr/bin/env python3
"""Config for SkillVLA STAGE 3 — the skill-path PT (pt_stage="skill").

Pipeline: stage1 (VSA) → stage2 (②③ + vlm_vision on the flow loss) → STAGE 3 (this) → FT.
Warm-starts the WHOLE policy from a STAGE-2 checkpoint (``pretrained_path`` → from_pretrained's
is_stage2 full-load) and trains ONLY the skill path — adapter ① ("skill" @ VLM LLM) + skill_reader +
skill_head — on the skill loss, over the SAME dataset stage 2 trained on (derived from its
train_config.json; no dataset key here). Everything else is frozen, vision included, so ① learns on
the FINAL vision (no moving target); structurally identical to FT's SKILL regime (its rehearsal).

Adapter ① is injected fresh at load (the stage-2 checkpoint has no ① keys → it starts at the B=0
no-op); ②③ are re-injected to match the checkpoint keys and stay frozen. LoRA structure (rank/alpha/
targets, ②③ flags) is INHERITED from the stage-2 config.json — only the ①-side lr scale is a knob.

Output: {project_root}/{outputs_root}/skillVLA_stage3/{run_name}/. Emits shell exports (--shell).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import as_bool, as_levels, as_list, get_value, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "stage3_train_config.yaml"


def build_settings(cfg: dict) -> dict:
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    dataset_root = project_root / str(get_value(cfg, "dataset_root", "dataset"))
    skillvla_root = dataset_root / str(get_value(cfg, "skillvla_dataset_root", "skillvla_dataset"))
    lerobot_root = project_root / "lerobot"
    outputs_root = project_root / str(get_value(cfg, "outputs_root", "outputs"))

    # ── Warm-start: a trained STAGE-2 run + checkpoint (full-loaded by the policy) ──
    stage2_vla_root = outputs_root / "skillVLA_stage2"
    stage2_run_name = str(get_value(cfg, "stage2_run_name")).strip()
    stage2_checkpoint = str(get_value(cfg, "stage2_checkpoint", "last")).strip() or "last"
    stage2_ckpt = stage2_vla_root / stage2_run_name / "checkpoints" / stage2_checkpoint / "pretrained_model"

    # Stage-2 config.json = source of truth for the architecture AND the LoRA structure (①'s rank/alpha
    # must equal ②③'s — one shared lora_rank — and ②③ must be re-injected to match the checkpoint keys).
    s2_cfg: dict = {}
    s2_cfg_json = stage2_ckpt / "config.json"
    if s2_cfg_json.is_file():
        s2_cfg = json.loads(s2_cfg_json.read_text())
    stage1_checkpoint_path = str(s2_cfg.get("stage1_checkpoint_path") or "")
    skill_fsq_levels = list(as_levels(s2_cfg.get("skill_fsq_levels", [5, 5, 5])))
    s1_vision_backbone = str(s2_cfg.get("s1_vision_backbone", "siglip"))
    s1_state_cond_mode = str(s2_cfg.get("s1_state_cond_mode", "state"))

    def _inh_bool(key, default=False):
        return as_bool(s2_cfg.get(key, default))

    def _inh_num(key, default, cast):
        return cast(s2_cfg.get(key, default))

    # Attention topology + reader architecture: MUST match the loaded weights → always inherited.
    attend_language = _inh_bool("attend_language", False)
    attend_image = _inh_bool("attend_image", True)
    # reader 전용 read-set (tri-state 상속): s2 config.json에 없거나 null(옛 ckpt) → ""(미전달, 상속 유지).
    # 학습(①/reader)과 추론이 같은 read-set을 써야 하므로 반드시 s2 값을 따라감.
    def _inh_tri(key):
        v = s2_cfg.get(key, None)
        return "" if v is None else ("true" if as_bool(v) else "false")
    reader_attend_image = _inh_tri("reader_attend_image")
    reader_attend_language = _inh_tri("reader_attend_language")
    vlm_cond = _inh_bool("vlm_cond", True)
    cond_expert = _inh_bool("cond_expert", True)
    vlm_expert = _inh_bool("vlm_expert", False)
    num_reader_tokens = _inh_num("num_reader_tokens", 4, int)
    reader_depth = _inh_num("reader_depth", 2, int)
    reader_heads = _inh_num("reader_heads", 8, int)
    skill_reader_all_layers = _inh_bool("skill_reader_all_layers", False)
    # LoRA structure — inherited so Stage-2 vlm_lora/cond_lora tensors have matching wrappers when the
    # parent checkpoint is loaded. The frozen Stage-1 expert_lora is rebuilt from its Stage-1 config.
    vlm_lora = _inh_bool("vlm_lora", _inh_bool("lora_cond_vlm", True))
    cond_lora = _inh_bool("cond_lora", _inh_bool("lora_cond_bridge", True))
    lora_rank = _inh_num("lora_rank", 8, int)
    lora_alpha = _inh_num("lora_alpha", 16.0, float)
    lora_dropout = _inh_num("lora_dropout", 0.0, float)
    lora_targets = str(s2_cfg.get("lora_targets", "q,k,v,o"))
    # ①-side knobs (this stage's own): adapter LR scale + the skill-loss params.
    lora_lr_scale = float(get_value(cfg, "lora_lr_scale", s2_cfg.get("lora_lr_scale", 1.0)))
    skill_loss_weight = get_value(cfg, "skill_loss_weight", None)
    if skill_loss_weight in (None, "", "null"):
        skill_loss_weight = s2_cfg.get("skill_loss_weight", 0.1)
    # NB: a BLANK yaml key yields None (not the get_value default) → explicit None-check, like above.
    _sdz = get_value(cfg, "skill_deadzone_frac", None)
    if _sdz in (None, "", "null"):
        _sdz = s2_cfg.get("skill_deadzone_frac", 0.0)
    skill_deadzone_frac = float(_sdz or 0.0)

    # ── Dataset: the SAME one stage 2 trained on (derived from its train_config.json) ──
    s2_train_json = stage2_ckpt / "train_config.json"
    ds_root = ""
    ds_repo = ""
    if s2_train_json.is_file():
        _tc_ds = (json.loads(s2_train_json.read_text()).get("dataset") or {})
        ds_root = str(_tc_ds.get("root") or "")
        ds_repo = str(_tc_ds.get("repo_id") or "")
    if not ds_root:
        raise ValueError(f"Cannot derive the stage-2 dataset: missing dataset.root in {s2_train_json}")
    # 이식 면역: 다른 서버에서 학습된 부모면 그 서버의 절대경로 → 이 서버의 skillvla_root로 재앵커.
    if not Path(ds_root).is_dir():
        _p = Path(ds_root)
        ds_root = str(skillvla_root / _p.parent.parent.name / _p.parent.name / _p.name)
    run_dir = Path(ds_root).parent                      # {skillvla_root}/{source}/{run_tag}
    source_dataset = run_dir.parent.name
    if not ds_repo:
        ds_repo = f"dohyeon/{source_dataset}"
    fsq_ckpt = run_dir / "FSQ.pt"
    # terminator co-training은 ONLINE DINO (배치 현재 프레임 라이브 토큰화) — dino.npz 불필요.
    terminator_dino_model_path = get_value(cfg, "terminator_dino_model_path", None) or str(
        project_root / "models" / "dinov3-vits16")
    # Transition pack = stage 3's actual training set (segment-level; the sbatch lazy-builds it if
    # missing — same pattern as FT's skill_code_freq.npz). Also the future FT-replay buffer.
    transition_pack = run_dir / "transitions.npz"

    batch_size = int(get_value(cfg, "batch_size", 16))
    num_gpus = int(get_value(cfg, "num_gpus", 1))
    lr_base = float(get_value(cfg, "lr_base", 2.5e-05))
    exp = str(get_value(cfg, "exp", "")).strip()
    train_terminator = as_bool(get_value(cfg, "train_terminator", False))
    # mode b: ground the STE skill prediction through the frozen motor on the FULL dataset (flow main +
    # GT-code anchor). mode a (false): transition-pack GT-code regression only.
    skill_action_grounding = as_bool(get_value(cfg, "skill_action_grounding", False))

    # run_name = {stage2_run_name}_{s2ckpt}__s3{a|b}[_{exp}] → outputs/skillVLA_stage3/
    #   a = transition-pack GT-code / b = action-grounding. ckpt 마커는 유지(같은 부모의 다른 스텝에서
    #   파생 시 충돌 방지 — FT의 PT-ckpt 마커와 같은 철학); lora_lr_scale 등은 config.json이 진실.
    run_name = f"{stage2_run_name}_{stage2_checkpoint}__s3{'b' if skill_action_grounding else 'a'}"
    if exp:
        run_name = f"{run_name}_{exp}"
    vla_root = outputs_root / "skillVLA_stage3"
    output_dir = vla_root / run_name

    settings: dict = {
        "project_root": project_root,
        "lerobot_root": lerobot_root,
        # dataset (INHERITED from the stage-2 run — same data, now supervising the skill loss)
        "source_dataset": source_dataset,
        "skillvla_dataset_dir": ds_root,
        "repo_id": ds_repo,
        "fsq_ckpt": fsq_ckpt,
        "terminator_dino_model_path": terminator_dino_model_path,
        "transition_pack": transition_pack,
        # warm-start (full policy from stage 2) + architecture config (from its config.json)
        "stage2_run_name": stage2_run_name,
        "stage2_checkpoint": stage2_checkpoint,
        "stage2_checkpoint_path": stage2_ckpt,
        "stage1_checkpoint_path": stage1_checkpoint_path,
        "s1_vision_backbone": s1_vision_backbone,
        "s1_state_cond_mode": s1_state_cond_mode,
        "skill_fsq_levels": "[" + ",".join(str(x) for x in skill_fsq_levels) + "]",
        "skill_loss_weight": str(skill_loss_weight),
        "skill_deadzone_frac": skill_deadzone_frac,
        "num_reader_tokens": num_reader_tokens,
        "reader_depth": reader_depth,
        "reader_heads": reader_heads,
        "skill_reader_all_layers": skill_reader_all_layers,
        "attend_language": attend_language,
        "attend_image": attend_image,
        "reader_attend_image": reader_attend_image,       # ""=상속(미전달) | true/false — s2 ckpt에서 상속
        "reader_attend_language": reader_attend_language,
        "vlm_cond": vlm_cond,
        "cond_expert": cond_expert,
        "vlm_expert": vlm_expert,
        # STAGE 3: every batch is a SKILL batch; ① trains (fresh B=0 inject), vlm_lora/cond_lora frozen.
        "pt_stage": "skill",
        "skill_action_grounding": skill_action_grounding,
        "lora_skill": True,
        "vlm_lora": vlm_lora,
        "cond_lora": cond_lora,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_targets": lora_targets,
        "lora_lr_scale": lora_lr_scale,
        # terminator co-training (optional; stage 2 usually already adapted it on this same dataset)
        "train_terminator": train_terminator,
        "terminator_lr_scale": float(get_value(cfg, "terminator_lr_scale", 1.0)),
        "terminator_end_target_sigma": float(get_value(cfg, "terminator_end_target_sigma", 1.0)),
        "terminator_end_pos_weight": float(get_value(cfg, "terminator_end_pos_weight", 1.0)),
        # per-component update tracking
        "track_param_drift": as_bool(get_value(cfg, "track_param_drift", False)),
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
        "wandb_project": str(get_value(cfg, "wandb_project", "VLA_stage3")),
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
        for key, value in settings.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
