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
    # run_tag는 입력받지 않는다 — FT 데이터셋은 부모 Stage-2와 같은 FSQ/파이프라인(run_tag)으로 빌드돼야
    # 하므로, 부모 체크포인트의 train_config.json(dataset.root = .../{src}/{run_tag}/skillvla)에서 유도.
    source_dataset = str(get_value(cfg, "source_dataset")).strip()
    s2_train_json = stage2_ckpt / "train_config.json"
    s2_ds_root = ""
    s2_ds_repo = ""
    if s2_train_json.is_file():
        _tc_ds = (json.loads(s2_train_json.read_text()).get("dataset") or {})
        s2_ds_root = str(_tc_ds.get("root") or "")
        s2_ds_repo = str(_tc_ds.get("repo_id") or "")
    if not s2_ds_root:
        raise ValueError(f"Cannot derive run_tag: missing dataset.root in {s2_train_json}")
    # 이식 면역: 부모 Stage-2가 다른 서버에서 학습됐으면 그 서버의 절대경로가 박혀 있음 → 존재하지
    # 않으면 이 서버의 skillvla_root/{source}/{run_tag}/skillvla로 재앵커 (경로 꼬리 3단은 서버 불문).
    if not Path(s2_ds_root).is_dir():
        _p = Path(s2_ds_root)
        s2_ds_root = str(skillvla_root / _p.parent.parent.name / _p.parent.name / _p.name)
    run_tag = Path(s2_ds_root).parent.name
    run_dir = skillvla_root / source_dataset / run_tag
    # FT terminator warm-start + current-frame DINO tokens live in the new task's run dir (same codebook).
    fsq_ckpt = run_dir / "FSQ.pt"
    dino_tokens_path = run_dir / "dino.npz"
    # Wrist DINO tokens — dual(terminator_use_wrist) FSQ면 필수; 빌드돼 있을 때만 경로 전달 ("" else).
    dino_wrist_tokens_path = (run_dir / "dino_wrist.npz") if (run_dir / "dino_wrist.npz").exists() else ""

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
    # Reader read-set (final vs all VLM layers) — MUST match the parent (the loaded reader weights expect
    # the same KV layout). Inherited from the Stage-2 ckpt; set explicitly only to ablate a matching parent.
    skill_reader_all_layers = _inherit("skill_reader_all_layers", False)

    # SCRATCH-parent support: a Stage-2 trained WITHOUT Stage-1 has stage1_checkpoint_path="" in its
    # config — the FT model then synthesizes the Stage-1-side architecture from these two knobs, which
    # must therefore be inherited from the parent (they default to siglip/state if the parent predates them).
    scratch_parent = not stage1_checkpoint_path
    s1_vision_backbone = str(s2_cfg.get("s1_vision_backbone", "siglip"))
    s1_state_cond_mode = str(s2_cfg.get("s1_state_cond_mode", "state"))

    # LoRA-continual FT regimes: [SKILL, MOTOR_conn, MOTOR_sev] (REPLACES the legacy A/B/C freeze
    # system — trainability is design-fixed: expert trains, ②③/bases/vision frozen, skill path per
    # ft_train_skill; see SkillVLAPolicy._apply_continual_freezes). The LoRA structure itself (①②③,
    # rank/alpha/targets) is INHERITED from the parent Stage-2 checkpoint's config.json.
    _rp = get_value(cfg, "regime_probs_ft", None)
    if not (isinstance(_rp, (list, tuple)) and len(_rp) == 3):
        raise ValueError("regime_probs_ft must be a 3-element list [SKILL, MOTOR_conn, MOTOR_sev] (합 1) "
                         "— the A/B/C regime_probs/freeze_A/B/C legacy was replaced.")
    regime_probs_ft = [float(x) for x in _rp]
    ft_train_skill = as_bool(get_value(cfg, "ft_train_skill", False))
    if regime_probs_ft[0] > 0 and not ft_train_skill:
        raise ValueError("SKILL regime prob > 0 but ft_train_skill=false — nothing is trainable on SKILL "
                         "batches (①/reader/head frozen). Set ft_train_skill=true or SKILL prob to 0.")
    if ft_train_skill and regime_probs_ft[0] <= 0:
        print("[warn] ft_train_skill=true but SKILL prob = 0 — the skill path is unfrozen yet never trained.")

    # ── PT-forgetting probe: 부모 Stage-2의 학습 데이터셋(위에서 유도)에서 고정 배치를 뽑아
    # probe_every 스텝마다 forward-only로 loss를 재측정 (노이즈까지 seed 고정 = 같은 자) → wandb probe/*.
    probe_pt_forgetting = as_bool(get_value(cfg, "probe_pt_forgetting", False))
    probe_dataset_root = s2_ds_root if probe_pt_forgetting else ""
    probe_repo_id = s2_ds_repo if probe_pt_forgetting else ""

    # run_name = {source}_{부모pre}_{s2ckpt}__{부모regime}__ft{P_SKILL}{P_conn}{P_sev}[_ts]{distill_tag}[_nohold][_exp]
    #   부모pre/부모regime = 부모 Stage-2 폴더명을 "__" 기준으로 나눈 앞/뒤 (예: 부모regime=pt5050_Ltttr8).
    #   ft 세그먼트 = 확률%: SKILL/MOTOR_conn/MOTOR_sev; _ts = ft_train_skill(스킬 경로 학습). term 제외.
    parent_pre, _, parent_regime = stage2_run_name.partition("__")
    ps, pcn, psv = (int(round(x * 100)) for x in regime_probs_ft)
    ft_regime = f"ft{ps}{pcn}{psv}" + ("_ts" if ft_train_skill else "")
    # Distinct output dir so ablations don't collide: distill{L}{G}w{λ} — VSA (frozen-PT) distillation on.
    distill_tag = ""
    if as_bool(get_value(cfg, "vsa_distill", False)):
        _dw = int(round(float(get_value(cfg, "vsa_distill_weight", 0.2)) * 100))
        distill_tag = (f"_distill{int(get_value(cfg, 'vsa_distill_n_local', 2))}"
                       f"{int(get_value(cfg, 'vsa_distill_n_global', 2))}w{_dw}")
    severed_hold_target = as_bool(get_value(cfg, "severed_hold_target", True))
    run_name = (f"{source_dataset}_{parent_pre}_{stage2_checkpoint}"
                + (f"__{parent_regime}" if parent_regime else "")
                + f"__{ft_regime}{distill_tag}")
    if not severed_hold_target:
        run_name = f"{run_name}_nohold"        # ablation: severed VSA keeps the real cross-skill tail
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
        "dino_wrist_tokens_path": dino_wrist_tokens_path,   # dual FSQ용 wrist 토큰 ("" = 없음/불필요)
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
        "skill_reader_all_layers": skill_reader_all_layers,   # inherited: must match parent reader KV layout
        "skill_deadzone_frac": skill_deadzone_frac,
        # FT behaviour
        "cond_skill_source": cond_skill_source,
        "train_terminator": train_terminator,
        "terminator_lr_scale": float(get_value(cfg, "terminator_lr_scale", 1.0)),
        "terminator_end_target_sigma": float(get_value(cfg, "terminator_end_target_sigma", 2.0)),
        "terminator_end_pos_weight": float(get_value(cfg, "terminator_end_pos_weight", 1.0)),
        "skill_loss_weight": str(skill_loss_weight),
        # LoRA-continual FT regimes (trainability is design-fixed by _apply_continual_freezes;
        # the LoRA structure/rank rides in from the parent Stage-2 config.json — not re-emitted here)
        "regime_probs_ft": ",".join(str(x) for x in regime_probs_ft),
        "ft_train_skill": ft_train_skill,
        # PT-forgetting probe ("" = off; root/repo_id는 부모 Stage-2 train_config.json에서 자동 유도)
        "probe_dataset_root": probe_dataset_root,
        "probe_dataset_repo_id": probe_repo_id,
        "probe_every": int(get_value(cfg, "probe_every", 250)),
        "probe_batches": int(get_value(cfg, "probe_batches", 4)),
        "probe_seed": int(get_value(cfg, "probe_seed", 12345)),
        "probe_vsa": as_bool(get_value(cfg, "probe_vsa", True)),
        # per-component update tracking (wandb param_drift/* + param_drift_rel/*)
        "track_param_drift": as_bool(get_value(cfg, "track_param_drift", False)),
        # continual-learning VSA distillation (anti-forgetting; B batches only)
        "vsa_distill": as_bool(get_value(cfg, "vsa_distill", False)),
        "vsa_distill_weight": float(get_value(cfg, "vsa_distill_weight", 0.2)),
        "vsa_distill_n_local": int(get_value(cfg, "vsa_distill_n_local", 2)),
        "vsa_distill_n_global": int(get_value(cfg, "vsa_distill_n_global", 2)),
        "vsa_distill_neighbor_radius": int(get_value(cfg, "vsa_distill_neighbor_radius", 1)),
        "severed_hold_target": severed_hold_target,   # severed VSA BC → Stage-1 stop+hold past skill_de

        # Motion-counter / global-sampler histograms (skill_code_freq.npz per dataset; the sbatch
        # auto-builds them). vsa_distill_freq_path = PARENT PT dataset → the first-FT SEED + global weight
        # base. vsa_ft_freq_path = THIS FT dataset → added to the cumulative counter + defines the codes
        # excluded from distill (unless prior clears the pct). Emitted only when distill is on.
        "vsa_distill_freq_path": (str(Path(s2_ds_root).parent / "skill_code_freq.npz")
                                  if as_bool(get_value(cfg, "vsa_distill", False)) else ""),
        "vsa_ft_freq_path": (str(run_dir / "skill_code_freq.npz")
                             if as_bool(get_value(cfg, "vsa_distill", False)) else ""),
        "vsa_distill_prior_pct": float(get_value(cfg, "vsa_distill_prior_pct", 20.0)),
        # PT/FT skillvla dirs the sbatch builds the histograms from (build_skill_code_freq.py)
        "vsa_pt_skillvla_dir": s2_ds_root,
        "vsa_ft_skillvla_dir": str(run_dir / "skillvla"),
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
