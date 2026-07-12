#!/usr/bin/env python3
"""Config for SkillVLA closed-loop EVAL (PT) on LIBERO sim.

Pick the model to evaluate by its OUTPUT FOLDER NAME + checkpoint (under
{project_root}/{outputs_root}/skillVLA_stage2/). The trained policy's params (FSQ path,
vision_backbone, skill_vocab, state_cond_mode, ...) live in the checkpoint's config.json and are
restored automatically when lerobot loads --policy.path, so the eval never re-specifies them.
The FSQ / skill_latents / raw-dataset paths used for the run are derived from that config.

Emits shell exports (--shell). All roots are declared in this yaml (standalone).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import as_bool, as_list, get_value, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "stage2_eval_config.yaml"


def _resolve_model(vla_root: Path, model_dir: str, checkpoint: str, eval_terminator: str) -> dict:
    """Resolve ONE Stage-2 run folder → its checkpoint + terminator + dataset-derived artifact paths.
    The checkpoint config.json is the single source of truth (FSQ path → skill_latents / raw-dataset /
    gt_skill dirs; train_terminator → the auto terminator resolution)."""
    policy_path = vla_root / model_dir / "checkpoints" / checkpoint / "pretrained_model"
    pol: dict = {}
    cfg_json = policy_path / "config.json"
    if cfg_json.is_file():
        pol = json.loads(cfg_json.read_text())
    base_fsq = str(pol.get("fsq_path") or "")
    train_terminator_used = as_bool(pol.get("train_terminator", False))
    skill_latents_path = raw_dataset_dir = gt_skill_dataset_dir = eval_init_states_path = ""
    if base_fsq:
        fp = Path(base_fsq)  # {root}/{dataset_root}/skillvla_dataset/{source}/{run_tag}/FSQ.pt
        skill_latents_path = str(fp.parent / "skill_latents.npz")
        gt_skill_dataset_dir = str(fp.parent / "skillvla")  # skillvla dataset (skill_sequence) for oracle GT
        # EPISODE-EXACT init states: FSQ-independent, shared per SOURCE (built by stage1_eval/oracle_matching)
        eval_init_states_path = str(fp.parents[1] / "eval_init_states.npz")
        try:
            raw_dataset_dir = str(fp.parents[3] / fp.parents[1].name)  # {dataset_root}/{source}
        except IndexError:
            raw_dataset_dir = ""
    if eval_terminator == "cotrained" and not train_terminator_used:
        raise ValueError(
            f"eval_terminator='cotrained' but {model_dir} was trained with train_terminator=false "
            "(no co-trained terminator exists). Use 'base' or 'auto'.")
    resolved_term = eval_terminator if eval_terminator != "auto" else ("cotrained" if train_terminator_used else "base")
    return {
        "model_dir": model_dir, "checkpoint": checkpoint, "policy_path": policy_path,
        # warm-start 부모(stage1) ckpt 경로 — include_stage1 패널이 여기서 자동 유도 (scratch면 "")
        "stage1_checkpoint_path": str(pol.get("stage1_checkpoint_path") or ""),
        "base_fsq": base_fsq, "train_terminator_used": train_terminator_used, "resolved_term": resolved_term,
        "ft_fsq_path": policy_path.parent / "FSQ_ft.pt", "ft_run_dir": vla_root / model_dir,
        "skill_latents_path": skill_latents_path, "raw_dataset_dir": raw_dataset_dir,
        "gt_skill_dataset_dir": gt_skill_dataset_dir, "eval_init_states_path": eval_init_states_path,
    }


def _auto_labels(model_dirs: list[str]) -> list[str]:
    """Distinguishing middle token(s) of each model_dir (strip the common leading + trailing _-tokens),
    used as the side-by-side panel label when a model has no explicit label. (Same as stage1_eval.)"""
    if len(model_dirs) <= 1:
        return list(model_dirs)
    toks = [d.split("_") for d in model_dirs]
    short = min(len(t) for t in toks)
    p = 0
    while p < short and all(t[p] == toks[0][p] for t in toks):
        p += 1
    s = 0
    while s < short - p and all(t[-1 - s] == toks[0][-1 - s] for t in toks):
        s += 1
    return ["_".join(t[p: len(t) - s]) or d for t, d in zip(toks, model_dirs)]


def build_settings(cfg: dict) -> dict:
    # Standalone: roots declared in this yaml (no build_data dependency).
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    outputs_root = project_root / str(get_value(cfg, "outputs_root", "outputs"))
    # skillVLA_stage2 by default; set models_root: skillVLA_stage3 in the eval yaml to eval STAGE-3 runs
    # (the full model with ①) — same folder layout, same checkpoint structure.
    vla_root = outputs_root / str(get_value(cfg, "models_root", "skillVLA_stage2"))
    default_ckpt = str(get_value(cfg, "checkpoint", "last"))
    target_task = str(get_value(cfg, "target_task", "libero_90"))
    image_key = str(get_value(cfg, "image_key", "observation.images.image"))
    eval_terminator = str(get_value(cfg, "eval_terminator", "auto")).lower()
    if eval_terminator not in ("auto", "cotrained", "base"):
        raise ValueError(f"eval_terminator must be 'auto', 'cotrained' or 'base' (got {eval_terminator!r}).")

    # `models` (list of {model_dir, checkpoint?, label?}) → MULTI-model side-by-side (same scenes via the
    # shared seed; per-task videos stitched into a labelled grid). Otherwise the single `model_dir`.
    models_yaml = get_value(cfg, "models", None)
    if isinstance(models_yaml, list) and models_yaml:
        entries = [{"model_dir": str(get_value(e, "model_dir")),
                    "checkpoint": str(get_value(e, "checkpoint", default_ckpt)),
                    "label": str(get_value(e, "label", "")).strip()} for e in models_yaml]
    else:  # back-compat: a single top-level model_dir (a 1-entry `models` list behaves identically)
        md = get_value(cfg, "model_dir", None)
        if not md:
            raise ValueError("Set `models` (a list of {model_dir, checkpoint?, label?}) — or a single "
                             "`model_dir` — in the eval yaml.")
        entries = [{"model_dir": str(md), "checkpoint": default_ckpt, "label": ""}]

    resolved = [_resolve_model(vla_root, e["model_dir"], e["checkpoint"], eval_terminator) for e in entries]
    autos = _auto_labels([e["model_dir"] for e in entries])
    labels: list[str] = []
    for e, a in zip(entries, autos):
        lbl = (e["label"] or a).replace("/", "_").replace(" ", "_")
        while lbl in labels:                      # panel dirs are keyed by label → must be unique
            lbl += "x"
        labels.append(lbl)

    # eval_dropout 프로브: 각 모델을 같은 체크포인트의 세 패널로 복제 —
    #   "{label}_full"     : VLM 엣지 + LoRA(②③) 다 켜짐 (온전한 모델)
    #   "{label}_drop_vlm" : 진짜 VLM만 절단, LoRA(②③)는 그대로 (③ cond_bridge의 이미지-단독 conduit 격리)
    #   "{label}_VSA"      : VLM 절단 + 어댑터도 OFF → pure Stage-1 VSA
    # 스킬 출처는 모든 패널이 use_gt_skill을 따름(gt=GT 주입 / pred=VLM이 별도 실행해 skill만). 멀티모델
    # 기계 전체 재사용 — 패널 순서는 모델별 [full, drop_vlm, VSA] 3연이라 models_per_row: 3이면 행당 한 모델.
    # (drop_vlm은 eval_drop_vlm=true + keep_adapters=true, VSA는 eval_drop_vlm=true + keep_adapters=false.)
    eval_dropout = as_bool(get_value(cfg, "eval_dropout", False))
    drop_flags = [False] * len(resolved)
    keepad_flags = [False] * len(resolved)
    dropcmp_base = list(labels)
    if eval_dropout:
        trip_resolved, trip_labels, drop_flags, keepad_flags = [], [], [], []
        for lbl, r in zip(labels, resolved):
            pfx = "" if len(resolved) == 1 else f"{lbl}_"   # 단일 모델이면 그냥 full / drop_vlm / VSA
            trip_resolved += [r, dict(r), dict(r)]
            trip_labels += [f"{pfx}full", f"{pfx}drop_vlm", f"{pfx}VSA"]
            drop_flags += [False, True, True]
            keepad_flags += [False, True, False]
        resolved, labels = trip_resolved, trip_labels

    use_gt_skill = as_bool(get_value(cfg, "use_gt_skill", False))
    advance_mode = str(get_value(cfg, "skill_advance_mode", "terminator"))

    # ── include_stage1: 각 모델의 warm-start 부모(stage1)를 STAGE1 자신의 평가 경로(stage1_eval의
    # run_eval.py)로 같은 scene(episode-exact)·같은 GT 스킬에서 평가해 격자의 추가 패널로 붙임 —
    # "VSA(순수 severed) ≡ stage1" 주장의 시각 대조군. 부모는 ckpt config.json의 stage1_checkpoint_path
    # 에서 자동. 패널 순서는 모델별 [full, drop_vlm, VSA, stage1] 그룹(models_per_row: 4 → 행당 한 모델);
    # 같은 부모를 공유하는 뒤 모델의 stage1 패널은 alias(sbatch가 심볼릭 링크 — 재평가 없음).
    # stage1은 GT 주입 전용이라 use_gt_skill=true 필수. terminator는 s2 패널과 동일 파일(위 참고).
    include_stage1 = as_bool(get_value(cfg, "include_stage1", False))
    if include_stage1:
        if not use_gt_skill:
            raise ValueError("include_stage1=true requires use_gt_skill=true (stage1 has no skill predictor).")
        group = 3 if eval_dropout else 1          # [full, drop_vlm, VSA] per model (+ stage1 appended)
        seen_parent: dict[str, str] = {}          # stage1 ckpt path → 첫 패널 label (alias target)
        new_r, new_l, new_d, new_k = [], [], [], []
        for gi in range(0, len(resolved), group):
            grp_r, grp_l = resolved[gi:gi + group], labels[gi:gi + group]
            grp_d, grp_k = drop_flags[gi:gi + group], keepad_flags[gi:gi + group]
            new_r += grp_r; new_l += grp_l; new_d += grp_d; new_k += grp_k
            r0 = grp_r[0]
            s1 = r0.get("stage1_checkpoint_path", "")
            if not s1 or not Path(s1).is_dir():
                raise ValueError(f"include_stage1: {r0['model_dir']} config.json에 stage1_checkpoint_path가 "
                                 f"없거나 경로가 존재하지 않음({s1!r}) — scratch 학습이면 include_stage1을 끄세요.")
            pfx = grp_l[0][:-len("full")] if eval_dropout else ("" if len(entries) == 1 else f"{grp_l[0]}_")
            lbl = f"{pfx}stage1"
            while lbl in new_l:
                lbl += "x"
            entry = dict(r0)
            if s1 in seen_parent:
                entry.update({"runner": "alias", "alias_target": seen_parent[s1]})
            else:
                # stage1 패널은 s2 패널과 "똑같은 터미네이터 파일"을 씀 — 터미네이터는 STAGE-2 체크포인트
                # 기준 eval_terminator로 resolve(cotrained → s2 co-trained FSQ_ft.pt, base → 원본 데이터셋
                # FSQ.pt). entry가 r0(dict)를 복사하므로 resolved_term/ft_fsq_path/ft_run_dir/base_fsq/
                # checkpoint가 그대로 실려 sbatch의 resolve_fsq가 s2와 동일 FSQ를 골라 --policy.fsq_path로
                # 넘긴다. stage1 자기 co-trained은 쓰지 않으므로 use_trained_terminator=False(전달된 파일을
                # 그대로 터미네이터로 로드) → stage1 ↔ vlm_drop이 항상 동일 터미네이터 → 스킬 경계 일치,
                # severed VSA(=stage1 파라미터) 대조가 터미네이터 변수 없이 성립. (cotrained에서 s2가
                # train_terminator=false면 _resolve_model이 이미 에러.)
                entry.update({
                    "runner": "stage1", "policy_path": Path(s1),
                    "use_trained_terminator": False,
                    "terminator_dino": str(project_root / "models" / "dinov3-vits16"),
                })
                seen_parent[s1] = lbl
            new_r.append(entry); new_l.append(lbl); new_d.append(False); new_k.append(False)
        resolved, labels, drop_flags, keepad_flags = new_r, new_l, new_d, new_k

    m0 = resolved[0]
    multi = len(resolved) >= 2

    model_dir, checkpoint, policy_path = m0["model_dir"], m0["checkpoint"], m0["policy_path"]
    base_fsq, train_terminator_used, resolved_term = m0["base_fsq"], m0["train_terminator_used"], m0["resolved_term"]
    ft_fsq_path, skill_latents_path = m0["ft_fsq_path"], m0["skill_latents_path"]
    raw_dataset_dir, gt_skill_dataset_dir = m0["raw_dataset_dir"], m0["gt_skill_dataset_dir"]

    # EPISODE-EXACT eval (stage1 방식, 항상 ON): 각 롤아웃을 데이터셋 에피소드의 MuJoCo init_state로 리셋.
    # eval_init_states.npz(source당 공유; stage1_eval/oracle_matching이 생성)가 없으면 자동으로
    # seed-기반 리셋으로 폴백(경고). use_gt_skill이면 GT 시퀀스도 그 에피소드에 정렬됨.
    missing = [r["model_dir"] for r in resolved
               if not (r["eval_init_states_path"] and Path(r["eval_init_states_path"]).is_file())]
    if missing:
        print(f"[stage2_eval] eval_init_states.npz missing for {missing} — falling back to seed-based "
              "resets. Build it: stage1_eval/oracle_matching/run.sh <source>", file=sys.stderr)
        for r in resolved:
            r["eval_init_states_path"] = ""
    # Folder suffix = _{skill_src}[_{eval_exp}]: skill_src = 스킬 출처 — "pred"(모델의 skill 예측) |
    # "gt"(GT 시퀀스 주입; advance_mode≠terminator면 gt-{mode}). terminator ft/base·episode-exact ee·
    # skill_end_mode(항상 or) 태그는 모두 제거 — 표준값에서 벗어난 ablation은 eval_exp로 구분.
    skill_src = "gt" if use_gt_skill else "pred"
    if use_gt_skill and advance_mode != "terminator":
        skill_src = f"gt-{advance_mode}"
    eval_exp = str(get_value(cfg, "eval_exp", "")).strip()   # free-form folder tag (e.g. "and" / "baseterm")
    suffix = skill_src + ("_s1" if include_stage1 else "") + (f"_{eval_exp}" if eval_exp else "")
    if multi:
        if eval_dropout:   # self-compare per checkpoint → base model name(s) + _dropcmp
            core = model_dir if len(dropcmp_base) == 1 else "compare_" + "_vs_".join(dropcmp_base)
            run_name = f"{core}_{checkpoint}_{target_task}_{suffix}_dropcmp"
        else:
            run_name = "compare_" + "_vs_".join(labels) + f"_{checkpoint}_{target_task}_{suffix}"
        models_json = json.dumps([
            {"label": lbl, "policy_path": str(r["policy_path"]), "base_fsq": r["base_fsq"],
             "ft_fsq_path": str(r["ft_fsq_path"]), "ft_run_dir": str(r["ft_run_dir"]),
             "checkpoint": r["checkpoint"], "resolved_term": r["resolved_term"],
             "gt_skill_dataset_dir": r["gt_skill_dataset_dir"],
             "eval_init_states_path": r["eval_init_states_path"],
             "drop_vlm": bool(d), "keep_adapters": bool(k),   # keep_adapters: severed인데 LoRA(②③) 유지 (drop_vlm 패널)
             # include_stage1 패널용: runner "s2"(기본) | "stage1"(부모를 stage1_eval 경로로 평가) |
             # "alias"(같은 부모 공유 → alias_target 패널 디렉토리로 심볼릭 링크)
             "runner": r.get("runner", "s2"), "alias_target": r.get("alias_target", ""),
             "terminator_dino": r.get("terminator_dino", ""),
             "use_trained_terminator": bool(r.get("use_trained_terminator", False))}
            for lbl, r, d, k in zip(labels, resolved, drop_flags, keepad_flags)])
    else:
        run_name = f"{model_dir}_{checkpoint}_{target_task}_{suffix}"
        models_json = ""
    eval_out_dir = _HERE.parent.parent / "outputs" / run_name

    settings: dict = {
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        # model-0 aliases (submit's artifact checks + the single-model path use these directly).
        # MULTI: the full per-model list rides MODELS_JSON (read by eval.sbatch's panel loop).
        "models_json": models_json,
        "models_labels": json.dumps(labels) if multi else "",
        "models_per_row": int(get_value(cfg, "models_per_row", 0) or 0),
        "policy_path": policy_path,
        "checkpoint": checkpoint,
        # terminator: eval sbatch picks FSQ_FOR_EVAL by EVAL_TERMINATOR — "cotrained" → ft_fsq_path
        # (lazy-exported from THIS checkpoint if missing), "base" → base_fsq.
        "base_fsq": base_fsq,                 # dataset FSQ (pre co-train; existence check + export base)
        "ft_fsq_path": ft_fsq_path,           # per-checkpoint adapted terminator (checkpoints/<ckpt>/FSQ_ft.pt)
        "ft_run_dir": vla_root / model_dir,   # for export_ft_terminator.py --ft_run_dir
        "eval_terminator": resolved_term,     # resolved base|cotrained → eval.sbatch FSQ pick + folder tag
        "train_terminator_used": train_terminator_used,
        "skill_latents_path": skill_latents_path,
        "raw_dataset_dir": raw_dataset_dir,
        # EPISODE-EXACT: dataset-episode MuJoCo init states ("" = seed-based resets; model-0 alias)
        "eval_init_states_path": m0["eval_init_states_path"],
        "image_key": image_key,
        # eval rollout — task_ids is env-overridable (TASK_IDS): submit_eval.sh splits it across
        # eval_num_gpus Slurm jobs (1 GPU each), all writing into the SAME out dir (disjoint tasks).
        "target_task": target_task,
        "task_ids": str(get_value(cfg, "task_ids", "[0,1,2,3,4,5,6,7,8,9]")),
        "eval_num_gpus": int(get_value(cfg, "eval_num_gpus", 1)),
        "n_episodes": int(get_value(cfg, "n_episodes", 1)),
        "n_action_steps": int(get_value(cfg, "n_action_steps", 5)),
        "eval_batch_size": int(get_value(cfg, "eval_batch_size", 1)),
        "max_parallel_tasks": int(get_value(cfg, "max_parallel_tasks", 1)),
        "max_videos_per_task": int(get_value(cfg, "max_videos_per_task", 1)),
        "video_frame_stride": int(get_value(cfg, "video_frame_stride", 2)),
        "video_fps": int(get_value(cfg, "video_fps", 10)),
        # skill_html now honored in MULTI too (per-panel skill traces under panels/{label}/skill_html/;
        # the sbatch auto-disables when skill_latents.npz is missing). Costs a bit per panel — set
        # skill_html: false in the yaml to skip.
        "skill_html": as_bool(get_value(cfg, "skill_html", True)),
        "skill_html_train_samples": int(get_value(cfg, "skill_html_train_samples", 10)),
        # inference knobs (eval-time tuning; model structure comes from the checkpoint)
        "skill_end_mode": str(get_value(cfg, "skill_end_mode", "termination")),
        "skill_end_threshold": str(get_value(cfg, "skill_end_threshold", 0.5)),
        "skill_end_progress_threshold": str(get_value(cfg, "skill_end_progress_threshold", 0.9)),
        "inference_skill_max_length": int(get_value(cfg, "inference_skill_max_length", 200)),
        # oracle eval: teacher-force GT skills + pick transition timing (gt vs terminator)
        "use_gt_skill": use_gt_skill,
        "gt_skill_dataset_dir": gt_skill_dataset_dir,
        "skill_advance_mode": advance_mode,
        # output / wandb — chunked submission (TASK_TAG, e.g. "t0-4") → distinct wandb run per chunk
        "wandb_project": str(get_value(cfg, "wandb_project", "VLA_eval")),
        "wandb_run_name": run_name + (f"_{os.environ['TASK_TAG']}" if os.environ.get("TASK_TAG") else ""),
        "eval_out_dir": eval_out_dir,
    }

    # Slurm partition/qos/nodelist/exclude are canonical (global_config.yaml train_*).
    part = ",".join(as_list(get_value(cfg, "train_partition", ["debug"]))) or "debug"
    excl = ",".join(as_list(get_value(cfg, "train_exclude_nodes", [])))
    settings.update({
        "eval_partition": part,
        "eval_qos": str(get_value(cfg, "train_qos", "base_qos")),
        "eval_gres": str(get_value(cfg, "eval_gres", "gpu:1")),
        "eval_cpus_per_task": int(get_value(cfg, "eval_cpus_per_task", 8)),
        "eval_mem": str(get_value(cfg, "eval_mem", "32G")),
        "eval_time": str(get_value(cfg, "eval_time", "1:00:00")),
        "eval_nodelist": str(get_value(cfg, "train_nodelist", "")),
        "eval_exclude_nodes": excl,
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
