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
import json
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

    # SCRATCH mode: stage1_run_name이 "none"으로 시작하면 Stage-1 warm-start 없음 (fresh expert/cond;
    # VSA는 vlm_dropout B 배치가 조각). 한 줄 포맷 none_{run_tag}_{siglip|dino}_{state|state_skill}에서
    # 데이터셋(run_tag)과 Stage-1측 아키텍처를 전부 파싱하며, stage1_checkpoint는 자동 무시됨.
    #   예: stage1_run_name: none_FSQ555_dino8_both_1000_siglip_state
    stage1_run_name = str(get_value(cfg, "stage1_run_name") or "").strip()
    scratch = stage1_run_name.lower().startswith("none") or stage1_run_name.lower() in ("", "false", "null", "~")
    stage1_checkpoint = str(get_value(cfg, "stage1_checkpoint", "last")).strip() or "last"

    _s1_vb = _s1_scm = None            # scratch 한 줄 포맷에서 파싱된 vision/state_cond_mode (없으면 yaml 키)
    if scratch:
        stage1_ckpt = ""
        stage1_checkpoint = ""         # scratch: stage1_checkpoint는 무시 (있어도 의미 없음)
        spec = re.sub(r"^none_?", "", stage1_run_name, flags=re.IGNORECASE)
        if spec:                       # one-liner: none_{run_tag}_{vision}_{mode}
            m = re.match(r"(FSQ\d+_dino\d+.*?)_(dino|siglip)_(state_skill|state)$", spec)
            if not m:
                raise ValueError(
                    "SCRATCH one-liner must be none_{run_tag}_{siglip|dino}_{state|state_skill} "
                    f"(e.g. none_FSQ555_dino8_both_1000_siglip_state), got: {stage1_run_name}")
            run_tag, _s1_vb, _s1_scm = m.group(1), m.group(2), m.group(3)
        else:                          # bare "none": legacy separate keys (run_tag + s1_* below)
            run_tag = str(get_value(cfg, "run_tag", "")).strip()
            if not run_tag:
                raise ValueError("SCRATCH mode needs the one-liner (stage1_run_name: none_{run_tag}_"
                                 "{vision}_{mode}) or an explicit `run_tag:` key.")
        s1_vis_tag = ""
        source_from_dir = ""
    else:
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
    if not source_dataset and not scratch:
        # DISAMBIGUATION PROBE: the Stage-1 checkpoint's train_config.json records the dataset it
        # trained on (dataset.root = .../{source}/{run_tag}/skillvla) → derive source from there.
        # Robust even when the SAME run_tag exists under several sources (e.g. after building an FT
        # dataset with the parent's run_tag). Same pattern as FT's run_tag derivation.
        _s1_tc = stage1_ckpt / "train_config.json"
        if _s1_tc.is_file():
            _root = str(((json.loads(_s1_tc.read_text()).get("dataset") or {}).get("root")) or "")
            if _root and Path(_root).name == "skillvla" and Path(_root).parent.name == run_tag:
                source_dataset = Path(_root).parent.parent.name
    if not source_dataset:
        # Last resort (scratch, or no readable Stage-1 train_config.json): the run_tag lives under
        # exactly one source dir in the skillvla root. If it exists under several, set source_dataset
        # in the yaml.
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

    # FSQ skill structure: auto-match the FSQ the dataset was built with (parsed from run_tag).
    skill_fsq_levels = list(as_levels(get_value(cfg, "skill_fsq_levels", build_fsq_levels)))

    # run_name = {STAGE-1 정보}__{pA}{Afreeze}_{pB}{Bfreeze}_{pC}{Cfreeze}[_{exp}]
    #   STAGE-1 정보 = {run_tag}_[{s1_vis_tag}_]{stage1_checkpoint}[_{mode}][_<loss tags>]
    #     (source / reader / freeze·unfreeze 태그는 이름에서 제외 — config.json이 진실)
    #   "__" 뒤 = 3-way 레짐: A=연결+freeze_A, B=연결+freeze_B, C=절단+freeze_C.
    #     각 블록 = {확률%}{freeze t/f: expert,cond,llm,vlm_vision 순}, 예: 30ttff_50fftt_20fftt.
    #   연결 토폴로지(attend_*/vlm_*)는 모델이 소비하되 이름엔 넣지 않음 — config.json이 진실.
    # batch{N}/c{N}/np are dropped (arch is re-read from the Stage-1 ckpt at load time); exp is last.
    # Inter-module attention connections (defaults: vlm_cond=T, cond_expert=T, vlm_expert=F, attend_language=F,
    # attend_image=T). attend_image/attend_language pick the VLM read-set (image-only / +lang / language-only).
    attend_language = as_bool(get_value(cfg, "attend_language", False))
    attend_image = as_bool(get_value(cfg, "attend_image", True))
    vlm_cond = as_bool(get_value(cfg, "vlm_cond", True))
    cond_expert = as_bool(get_value(cfg, "cond_expert", True))
    vlm_expert = as_bool(get_value(cfg, "vlm_expert", False))    # action ← VLM directly (was action_attend_vlm)
    if not attend_image and not attend_language:
        raise ValueError("attend_image=False AND attend_language=False leaves the VLM unreadable; "
                         "use vlm_cond/vlm_expert=False to go VLM-blind instead.")
    # Skill reader (separate post-VLM joint concat-KV probe module) + SkillHead dead-zone.
    skill_deadzone_frac = float(get_value(cfg, "skill_deadzone_frac", 0.0))   # 0 = plain MSE
    num_reader_tokens = int(get_value(cfg, "num_reader_tokens", 4))
    reader_depth = int(get_value(cfg, "reader_depth", 2))
    reader_heads = int(get_value(cfg, "reader_heads", 8))
    # Per-regime freeze: 3-way A/B/C from nested dicts freeze_A / freeze_B / freeze_C → flat config
    # fields freeze_vlm_vsa_*(A) / freeze_vsa_*(B) / freeze_c_*(C). regime_probs is REQUIRED (2-way
    # binary-dropout legacy removed).
    # Group keys: llm = the VLM's Gemma LLM trunk ONLY (accepts "vlm" as an alias in the dict);
    # vlm_vision = the PaliGemma SigLIP tower.
    _REGIME_KEYS = ("expert", "cond", "llm", "vlm_vision")
    def _grp(d, k):
        return d.get(k, d.get("vlm", False)) if k == "llm" else d.get(k, False)   # "vlm" = alias for llm
    _rp = get_value(cfg, "regime_probs", None)
    if not (isinstance(_rp, (list, tuple)) and len(_rp) == 3):
        raise ValueError("regime_probs must be a 3-element list [A, B, C] (합 1). 2-way binary-dropout "
                         "legacy가 제거됨 — regime_probs를 반드시 지정하세요.")
    regime_probs = [float(x) for x in _rp]
    _fa = get_value(cfg, "freeze_A", {}) or {}
    _fb = get_value(cfg, "freeze_B", {}) or {}
    _fc = get_value(cfg, "freeze_C", {}) or {}
    frz = {f"freeze_vlm_vsa_{k}": as_bool(_grp(_fa, k)) for k in _REGIME_KEYS}
    frz.update({f"freeze_vsa_{k}": as_bool(_grp(_fb, k)) for k in _REGIME_KEYS})
    frz.update({f"freeze_c_{k}": as_bool(_grp(_fc, k)) for k in _REGIME_KEYS})
    # C (severed) batches use the Stage-1 hold BC target (stop+hold past skill_de). Default on; the name
    # gets a "_nohold" tag only when disabled (ablation).
    severed_hold_target = as_bool(get_value(cfg, "severed_hold_target", True))
    # EMA-self distillation (forgetting prep): pin sampled non-GT skills to the model's own weight-EMA.
    ema_self_distill = as_bool(get_value(cfg, "ema_self_distill", False))
    ema_self_alpha = float(get_value(cfg, "ema_self_alpha", 0.999))
    ema_self_weight = float(get_value(cfg, "ema_self_weight", 0.5))
    ema_self_n_local = int(get_value(cfg, "ema_self_n_local", 2))
    ema_self_n_global = int(get_value(cfg, "ema_self_n_global", 2))
    # SCRATCH-side arch knobs (model uses them only when stage1_checkpoint_path is empty). The one-liner
    # (none_{run_tag}_{vision}_{mode}) takes precedence over the separate yaml keys.
    s1_vision_backbone = _s1_vb or str(get_value(cfg, "s1_vision_backbone", "siglip")).strip()
    s1_state_cond_mode = _s1_scm or str(get_value(cfg, "s1_state_cond_mode", "state")).strip()
    if scratch and not (regime_probs[1] > 0 or regime_probs[2] > 0):
        raise ValueError("SCRATCH mode needs VSA training — regime_probs의 B 또는 C가 >0 이어야 함 "
                         "(B/C 배치가 VSA를 학습).")
    # state_skill checked first (it contains the substring "_state"); else state; else "" (older naming).
    if scratch:
        mode_tag, loss_tags = s1_state_cond_mode, []
    else:
        mode_tag = ("state_skill" if "_state_skill" in stage1_run_name
                    else "state" if "_state" in stage1_run_name else "")
        loss_tags = []
        for pat in (r"_(cum_(?:ep|all))(?:_|$)", r"_(ac_w|ac_x|ac)(?:_|$)", r"_(weighted)(?:_|$)"):
            m = re.search(pat, stage1_run_name)
            if m:
                loss_tags.append(m.group(1))
    # {STAGE-1 정보}__{pA}{Afreeze}_{pB}{Bfreeze}_{pC}{Cfreeze}:
    #   각 블록 = {확률%}{freeze t/f in (expert, cond, llm, vlm_vision) order}. 예: 30ttff_50fftt_20fftt.
    #   A=연결+freeze_A, B=연결+freeze_B, C=절단+freeze_C.
    # SCRATCH: {run_tag}_{vision}_scratch_{state_cond_mode}__... (Stage-1 체크포인트 정보가 없으므로).
    _tf = lambda prefix: "".join("t" if frz[f"freeze_{prefix}_{k}"] else "f" for k in _REGIME_KEYS)
    pa, pb, pc = (int(round(x * 100)) for x in regime_probs)
    regime_tag = f"{pa}{_tf('vlm_vsa')}_{pb}{_tf('vsa')}_{pc}{_tf('c')}"
    if scratch:
        parts = [run_tag, s1_vision_backbone, "scratch", s1_state_cond_mode]
    else:
        parts = ([run_tag] + ([s1_vis_tag] if s1_vis_tag else [])
                 + [stage1_checkpoint] + ([mode_tag] if mode_tag else []) + loss_tags)
    run_name = "_".join(parts) + "__" + regime_tag
    if not severed_hold_target:
        run_name = f"{run_name}_nohold"        # ablation: severed batches keep the real cross-skill tail
    if ema_self_distill:
        # EMA-self on → short suffix: α (decimals) + n_local/n_global + weight×100. off → name unchanged.
        _a = str(ema_self_alpha).split(".")[-1] or "0"
        run_name = f"{run_name}_ema{_a}_{ema_self_n_local}{ema_self_n_global}w{int(round(ema_self_weight * 100))}"
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
        "attend_image": attend_image,
        "vlm_cond": vlm_cond,
        "cond_expert": cond_expert,
        "vlm_expert": vlm_expert,                   # action tokens read the VLM directly (was action_attend_vlm)
        # 3-way regime probs [A,B,C] as a comma string (REQUIRED — governs A/B/C sampling in the model).
        "regime_probs": ",".join(str(x) for x in regime_probs),
        "severed_hold_target": severed_hold_target,   # C (severed) BC → Stage-1 stop+hold past skill_de
        # EMA-self distillation (Stage-2 forgetting prep)
        "ema_self_distill": ema_self_distill,
        "ema_self_alpha": ema_self_alpha,
        "ema_self_weight": ema_self_weight,
        "ema_self_n_local": ema_self_n_local,
        "ema_self_n_global": ema_self_n_global,

        # per-component update tracking (wandb param_drift/* + param_drift_rel/*)
        "track_param_drift": as_bool(get_value(cfg, "track_param_drift", False)),
        # scratch mode (no Stage-1): fresh expert/cond; arch from the two knobs below
        "scratch": scratch,
        "s1_vision_backbone": s1_vision_backbone,
        "s1_state_cond_mode": s1_state_cond_mode,
        **frz,                                      # freeze (freeze_{vlm_vsa,vsa}_{expert,cond,vlm,vlm_vision}; p=0 → A dict static)
        # terminator co-training (same as FT): adapt the FSQ terminator on this dataset's GT signals
        # (disjoint from the SkillVLA params). Warm-starts from fsq_ckpt; exported per-checkpoint for eval.
        "train_terminator": as_bool(get_value(cfg, "train_terminator", False)),
        "terminator_lr_scale": float(get_value(cfg, "terminator_lr_scale", 1.0)),
        "terminator_end_target_sigma": float(get_value(cfg, "terminator_end_target_sigma", 2.0)),
        "terminator_end_pos_weight": float(get_value(cfg, "terminator_end_pos_weight", 1.0)),
        "dino_tokens_path": run_dir / "dino.npz",   # current-frame DINO tokens for the terminator
        # Wrist DINO tokens — attached ONLY for a dual (terminator_use_wrist) FSQ, and only if built ("" else).
        "dino_wrist_tokens_path": (run_dir / "dino_wrist.npz") if (run_dir / "dino_wrist.npz").exists() else "",
        # freezing is fully governed by the per-regime dicts (**frz above; p=0 → A dict applies statically)
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
