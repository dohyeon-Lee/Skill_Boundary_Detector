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
    skill_reader_all_layers = as_bool(get_value(cfg, "skill_reader_all_layers", False))
    # STAGE 2 = cond-path PT: every batch is COND/flow (pt_stage="cond" — the interleaved [SKILL, COND]
    # mixing was retired; the skill path trains in STAGE 3). Trainability is design-fixed: ②③ adapters +
    # vlm_vision train, everything else frozen; see SkillVLAPolicy._apply_continual_freezes.
    lora_cond_vlm = as_bool(get_value(cfg, "lora_cond_vlm", True))
    lora_cond_bridge = as_bool(get_value(cfg, "lora_cond_bridge", True))
    lora_expert = as_bool(get_value(cfg, "lora_expert", False))   # ④ connected-only 소비자 적응 (run name 3번째 문자)
    lora_rank = int(get_value(cfg, "lora_rank", 8))
    # lora_alpha: "auto"(또는 빈칸) → 2×rank — rank만 바꿔도 effective scale(alpha/r)이 2로 유지되어
    # rank 효과와 scale 효과가 섞이지 않음. 숫자를 쓰면 그대로 (명시 오버라이드).
    # stage3/FT는 이 resolve된 숫자를 체크포인트 config.json에서 상속하므로 여기 한 곳만 처리하면 됨.
    _alpha_raw = get_value(cfg, "lora_alpha", "auto")
    lora_alpha = (2.0 * lora_rank if _alpha_raw is None or str(_alpha_raw).strip().lower() in ("", "auto")
                  else float(_alpha_raw))
    lora_dropout = float(get_value(cfg, "lora_dropout", 0.0))
    lora_targets = str(get_value(cfg, "lora_targets", "q,k,v,o"))
    lora_lr_scale = float(get_value(cfg, "lora_lr_scale", 1.0))   # adapter-only LR × (vlm_vision stays base)
    if not (lora_cond_vlm or lora_cond_bridge):
        raise ValueError("Both cond adapters (②lora_cond_vlm/③lora_cond_bridge) are off — nothing would "
                         "train on stage-2 COND batches besides vlm_vision (expert/bases are frozen).")
    # SCRATCH-side arch knobs (model uses them only when stage1_checkpoint_path is empty). The one-liner
    # (none_{run_tag}_{vision}_{mode}) takes precedence over the separate yaml keys.
    s1_vision_backbone = _s1_vb or str(get_value(cfg, "s1_vision_backbone", "siglip")).strip()
    s1_state_cond_mode = _s1_scm or str(get_value(cfg, "s1_state_cond_mode", "state")).strip()
    if scratch:
        raise ValueError("SCRATCH mode is incompatible with the LoRA-continual PT: the expert/cond bases "
                         "are FROZEN by design (stage-1 완성 전제) — a fresh-init VSA would never train. "
                         "Warm-start from a Stage-1 checkpoint instead.")
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
    # {STAGE-1 정보}__s2_L{②③[④] t/f}r{rank}[lr{scale}][_{exp}] — 예: ...state__s2_Lttr16lr10.
    #   s2 = cond-path PT stage; Ltt = ②cond_vlm ③cond_bridge on/off; r = shared rank (stage3 inherits).
    #   ④ lora_expert는 켰을 때만 3번째 문자 t가 붙음 (Lttt...) — 기존 ④-off 런 이름과 호환.
    _l = "".join("t" if b else "f" for b in (lora_cond_vlm, lora_cond_bridge))
    if lora_expert:
        _l += "t"
    regime_tag = f"s2_L{_l}r{lora_rank}"
    if lora_lr_scale != 1.0:                     # adapter LR multiplier (e.g. lr10 = 10× base)
        regime_tag += f"lr{int(lora_lr_scale) if lora_lr_scale == int(lora_lr_scale) else lora_lr_scale}"
    cond_severed_prob = float(get_value(cfg, "cond_severed_prob", 0.0) or 0.0)
    if not 0.0 <= cond_severed_prob < 1.0:
        raise ValueError(f"cond_severed_prob must be in [0, 1) — got {cond_severed_prob}")
    if cond_severed_prob > 0.0:                  # severed 정규화 배치 (conduit gating) — sv50 = p 0.5
        regime_tag += f"sv{int(round(cond_severed_prob * 100))}"
    # vf = vlm_vision(+projector) FROZEN in stage 2 (LLM freeze와 같은 논리 — 사전학습 지각은 이미
    # 일반적; ②③ 라우팅 어댑터만 학습). 학습 파라미터 스코프가 바뀌므로 run name에 태그.
    freeze_vlm_vision = as_bool(get_value(cfg, "freeze_vlm_vision", False))
    if freeze_vlm_vision:
        regime_tag += "vf"
    # 제출-시점 fail-fast: dual FSQ("both" — terminator가 wrist DINO도 씀) + train_terminator인데
    # dino_wrist.npz가 없으면, 모델 로딩(~5분) 후 첫 배치 런타임에서야 터짐 — 여기서 미리 차단.
    # (다른 클러스터로 옮길 때 28GB짜리 dino_wrist.npz 복사가 누락되는 케이스.)
    train_terminator = as_bool(get_value(cfg, "train_terminator", False))
    dino_wrist = run_dir / "dino_wrist.npz"
    if train_terminator and "both" in run_tag and not dino_wrist.exists():
        raise ValueError(
            f"train_terminator=true + dual FSQ('both')인데 wrist DINO 토큰이 없음: {dino_wrist} — "
            "원본 빌드에서 dino_wrist.npz(가능하면 dino_wrist.features.npy도)를 복사하거나, "
            "train_terminator: false로 두세요.")

    parts = ([run_tag] + ([s1_vis_tag] if s1_vis_tag else [])
             + [stage1_checkpoint] + ([mode_tag] if mode_tag else []) + loss_tags)
    run_name = "_".join(parts) + "__" + regime_tag
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
        "skill_reader_all_layers": skill_reader_all_layers,   # reader reads ALL VLM layers (vs final only)
        # inter-module attention connections (VLM/cond/expert edges + the language master switch)
        "attend_language": attend_language,
        "attend_image": attend_image,
        "vlm_cond": vlm_cond,
        "cond_expert": cond_expert,
        "vlm_expert": vlm_expert,                   # action tokens read the VLM directly (was action_attend_vlm)
        # STAGE 2: pt_stage=cond (every batch COND/flow) + the two cond adapters (① lives in stage3).
        "pt_stage": "cond",
        "lora_skill": False,
        "lora_cond_vlm": lora_cond_vlm,
        "lora_cond_bridge": lora_cond_bridge,
        "lora_expert": lora_expert,               # ④ connected-only 소비자 적응 (severed에는 구조적 무영향)
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_targets": lora_targets,
        "lora_lr_scale": lora_lr_scale,
        "cond_severed_prob": cond_severed_prob,   # severed 정규화 배치 확률 (③만 활성; conduit gating)
        "freeze_vlm_vision": freeze_vlm_vision,   # stage2에서도 vision 동결 (②③ 어댑터만 학습; run name "vf")

        # per-component update tracking (wandb param_drift/* + param_drift_rel/*)
        "track_param_drift": as_bool(get_value(cfg, "track_param_drift", False)),
        # scratch mode (no Stage-1): fresh expert/cond; arch from the two knobs below
        "scratch": scratch,
        "s1_vision_backbone": s1_vision_backbone,
        "s1_state_cond_mode": s1_state_cond_mode,
        # (trainability is design-fixed by _apply_continual_freezes — no per-regime freeze flags emitted)
        # terminator co-training (same as FT): adapt the FSQ terminator on this dataset's GT signals
        # (disjoint from the SkillVLA params). Warm-starts from fsq_ckpt; exported per-checkpoint for eval.
        "train_terminator": train_terminator,
        "terminator_lr_scale": float(get_value(cfg, "terminator_lr_scale", 1.0)),
        "terminator_end_target_sigma": float(get_value(cfg, "terminator_end_target_sigma", 2.0)),
        "terminator_end_pos_weight": float(get_value(cfg, "terminator_end_pos_weight", 1.0)),
        "dino_tokens_path": run_dir / "dino.npz",   # current-frame DINO tokens for the terminator
        # Wrist DINO tokens — attached ONLY for a dual (terminator_use_wrist) FSQ, and only if built ("" else).
        "dino_wrist_tokens_path": dino_wrist if dino_wrist.exists() else "",
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
