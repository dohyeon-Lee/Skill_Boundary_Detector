#!/usr/bin/env python3
"""Config for Conditional/Unconditional Stage-0 SkillVLA closed-loop eval on LIBERO sim.

Pick the model to evaluate by its OUTPUT FOLDER NAME + checkpoint (under
{project_root}/{outputs_root}/skillVLA_stage0/). The trained policy's params (FSQ path,
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
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import as_bool, as_list, get_value, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "stage0_eval_config.yaml"
DEFAULT_STAGE3_EVAL_CONFIG_PATH = _HERE.parent.parent.parent / "stage3_eval" / "stage3_eval_config.yaml"


def _resolve_stage3_predictor_path(cfg: dict, project_root: Path) -> Path:
    """Resolve the single Stage-3 checkpoint selected by stage3_eval_config.yaml."""
    raw_path = str(get_value(cfg, "stage3_eval_config", "") or "").strip()
    config_path = Path(raw_path).expanduser() if raw_path else DEFAULT_STAGE3_EVAL_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = (_HERE.parent.parent / config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Stage-3 eval config not found: {config_path}")

    stage3_cfg = load_config(config_path)
    stage3_project = Path(str(get_value(stage3_cfg, "project_root", project_root))).expanduser()
    stage3_outputs = stage3_project / str(get_value(stage3_cfg, "outputs_root", "outputs"))
    stage3_root = stage3_outputs / str(get_value(stage3_cfg, "models_root", "skillVLA_stage3"))
    default_checkpoint = str(get_value(stage3_cfg, "checkpoint", "last"))
    models = get_value(stage3_cfg, "models", None)
    if isinstance(models, list) and models:
        specs = [
            (
                str(get_value(model, "model_dir", "") or "").strip(),
                str(get_value(model, "checkpoint", default_checkpoint) or default_checkpoint).strip(),
            )
            for model in models
        ]
    else:
        specs = [
            (
                str(get_value(stage3_cfg, "model_dir", "") or "").strip(),
                default_checkpoint,
            )
        ]
    if any(not model_dir for model_dir, _ in specs):
        raise ValueError(f"Stage-3 eval config must select a model_dir: {config_path}")

    candidates = [
        stage3_root / model_dir / "checkpoints" / checkpoint / "pretrained_model"
        for model_dir, checkpoint in specs
    ]
    unique = {candidate.resolve(): candidate for candidate in candidates}
    if len(unique) != 1:
        selected = ", ".join(str(path) for path in unique.values())
        raise ValueError(
            "skill_source=stage3 requires stage3_eval_config.yaml to select exactly one "
            f"distinct Stage-3 checkpoint; got: {selected}"
        )

    policy_path = next(iter(unique.values()))
    config_json = policy_path / "config.json"
    if not config_json.is_file():
        raise FileNotFoundError(f"Stage-3 predictor checkpoint not found: {config_json}")
    policy_cfg = json.loads(config_json.read_text())
    if policy_cfg.get("pt_stage") != "skill":
        raise ValueError(
            "skill_source=stage3 requires a Stage-3 checkpoint with pt_stage='skill', "
            f"got {policy_cfg.get('pt_stage')!r}: {config_json}"
        )
    return policy_path


def _resolve_model(vla_root: Path, model_dir: str, checkpoint: str, eval_terminator: str) -> dict:
    """Resolve one Stage-0 run folder to its checkpoint, terminator, and dataset artifacts.
    The checkpoint config.json is the single source of truth (FSQ path → skill_latents / raw-dataset /
    gt_skill dirs; train_terminator → the auto terminator resolution)."""
    policy_path = vla_root / model_dir / "checkpoints" / checkpoint / "pretrained_model"
    pol: dict = {}
    cfg_json = policy_path / "config.json"
    if cfg_json.is_file():
        pol = json.loads(cfg_json.read_text())
        if not as_bool(pol.get("stage0_vlm_residual", False)):
            raise ValueError(
                f"{model_dir}/{checkpoint} is not a renewed Stage-0 VLM-residual checkpoint."
            )
        legacy_vlm_edges = [
            name for name in ("vlm_cond", "vlm_expert") if as_bool(pol.get(name, False))
        ]
        if legacy_vlm_edges:
            raise ValueError(
                f"{model_dir}/{checkpoint} enables {legacy_vlm_edges}; Conditional/Unconditional "
                "eval requires the VLM residual to be the only VLM-to-action pathway."
            )
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
        "terminator_dino_model_path": str(pol.get("terminator_dino_model_path") or ""),
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


def _stage0_modes(raw: str) -> tuple[str, ...]:
    """Parse the renewed Stage-0 action routes."""
    aliases = {
        "conditional": "conditional", "cond": "conditional",
        "unconditional": "unconditional", "uncond": "unconditional", "base": "unconditional",
    }
    items = [s.strip().lower() for s in str(raw or "conditional,unconditional").split(",") if s.strip()]
    modes = tuple(aliases.get(s, "") for s in items)
    if not modes or any(not s for s in modes) or len(set(modes)) != len(modes):
        raise ValueError(
            "Stage-0 modes must be a comma list without duplicates from "
            "conditional,unconditional (aliases: cond,uncond,base), got " + repr(raw)
        )
    return modes


def _expand_stage0_panels(models: list[dict]) -> list[dict]:
    """Expand each checkpoint into matched Conditional/Unconditional rollout panels."""
    panels: list[dict] = []
    labels = {"conditional": "Conditional", "unconditional": "Unconditional"}
    for model in models:
        for mode in _stage0_modes(model.get("modes", "")):
            panel = dict(model)
            panel["label"] = f"{model['label']} [{labels[mode]}]"
            panel["eval_mode"] = mode
            # The renewed Stage-0 base was trained without any VLM edge. eval_drop_vlm therefore
            # toggles exactly the final VLM residual: false=conditional, true=unconditional.
            panel["drop_vlm"] = mode == "unconditional"
            panel["keep_adapters"] = False
            panels.append(panel)
    return panels


def _resolve_output_name(cfg: dict) -> str:
    """Use the requested output folder verbatim, or the config resolution time when blank."""
    output_name = str(get_value(cfg, "output_name", "") or "").strip()
    if not output_name:
        output_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if output_name in {".", ".."} or "/" in output_name or "\0" in output_name:
        raise ValueError(
            "output_name must be a single folder name, not a path "
            f"(got {output_name!r})."
        )
    return output_name


def build_settings(cfg: dict) -> dict:
    # Standalone: roots declared in this yaml (no build_data dependency).
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    outputs_root = project_root / str(get_value(cfg, "outputs_root", "outputs"))
    vla_root = outputs_root / str(get_value(cfg, "models_root", "skillVLA_stage0"))
    default_ckpt = str(get_value(cfg, "checkpoint", "last"))  # legacy top-level fallback
    target_task = str(get_value(cfg, "target_task", "libero_90"))
    image_key = str(get_value(cfg, "image_key", "observation.images.image"))
    default_terminator = str(get_value(
        cfg, "terminator_source", get_value(cfg, "eval_terminator", "auto")
    )).lower()
    if default_terminator not in ("auto", "cotrained", "base"):
        raise ValueError(
            f"terminator_source must be auto|cotrained|base (got {default_terminator!r})."
        )
    default_advance = str(get_value(cfg, "skill_advance_mode", "terminator")).lower()  # legacy fallback
    if default_advance not in {"gt", "terminator"}:
        raise ValueError(f"skill_advance_mode must be gt|terminator, got {default_advance!r}.")
    skill_source = str(get_value(cfg, "skill_source", "gt")).strip().lower()
    skill_source_aliases = {
        "gt": "gt",
        "pred": "predicted",
        "predictor": "predicted",
        "predicted": "predicted",
        "stage3": "stage3",
    }
    skill_source = skill_source_aliases.get(skill_source, "")
    if not skill_source:
        raise ValueError("skill_source must be gt|predicted|stage3 (aliases: pred,predictor).")
    use_gt_skill = skill_source == "gt"
    stage3_predictor_path = (
        _resolve_stage3_predictor_path(cfg, project_root) if skill_source == "stage3" else None
    )

    # Every model owns checkpoint/advance_mode and expands to the selected Conditional/Unconditional panels.
    models_yaml = get_value(cfg, "models", None)
    default_modes = "conditional,unconditional"
    global_modes = str(get_value(cfg, "modes", default_modes) or default_modes)
    if isinstance(models_yaml, list) and models_yaml:
        entries = [{
            "model_dir": str(get_value(e, "model_dir")),
            "checkpoint": str(get_value(e, "checkpoint", default_ckpt)),
            "advance_mode": str(get_value(e, "advance_mode", default_advance)).lower(),
            # `terminator` is the pre-rename per-entry alias kept for frozen YAML snapshots.
            "terminator_source": str(get_value(
                e, "terminator_source", get_value(e, "terminator", default_terminator)
            )).lower(),
            "modes": str(get_value(e, "modes", global_modes) or global_modes),
            "label": str(get_value(e, "label", "")).strip(),
        } for e in models_yaml]
    else:  # back-compat: a single top-level model_dir (a 1-entry `models` list behaves identically)
        md = get_value(cfg, "model_dir", None)
        if not md:
            raise ValueError("Set `models` (a list of {model_dir, checkpoint?, label?}) — or a single "
                             "`model_dir` — in the eval yaml.")
        entries = [{"model_dir": str(md), "checkpoint": default_ckpt,
                    "advance_mode": default_advance, "terminator_source": default_terminator,
                    "modes": global_modes, "label": ""}]

    for e in entries:
        if e["advance_mode"] not in {"gt", "terminator"}:
            raise ValueError(f"models[].advance_mode must be gt|terminator, got {e['advance_mode']!r}.")
        if e["terminator_source"] not in {"auto", "cotrained", "base"}:
            raise ValueError(
                "models[].terminator_source must be auto|cotrained|base, "
                f"got {e['terminator_source']!r}."
            )
    if not use_gt_skill and any(e["advance_mode"] == "gt" for e in entries):
        raise ValueError(
            "models[].advance_mode=gt requires skill_source=gt; predicted skills must use terminator."
        )
    for e in entries:
        _stage0_modes(e["modes"])

    resolved_models = [
        _resolve_model(vla_root, e["model_dir"], e["checkpoint"], e["terminator_source"])
        for e in entries
    ]
    autos = _auto_labels([e["model_dir"] for e in entries])
    used_base_labels: list[str] = []
    for e, r, auto in zip(entries, resolved_models, autos):
        label = (e["label"] or auto).replace("/", "_").strip()
        while label in used_base_labels:
            label += "x"
        used_base_labels.append(label)
        r.update({"label": label, "modes": e["modes"], "advance_mode": e["advance_mode"]})
    resolved = _expand_stage0_panels(resolved_models)
    labels = [r["label"] for r in resolved]
    drop_flags = [bool(r["drop_vlm"]) for r in resolved]
    keepad_flags = [bool(r["keep_adapters"]) for r in resolved]

    m0 = resolved[0]
    # The single-panel path has no route override and therefore represents Conditional only.
    # A lone Unconditional panel must still use MODELS_JSON so eval_drop_vlm is honored.
    multi = len(resolved) >= 2 or m0.get("eval_mode") != "conditional"

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
        print(f"[stage0_eval] eval_init_states.npz missing for {missing} — falling back to seed-based "
              "resets. Build it: stage1_eval/oracle_matching/run.sh <source>", file=sys.stderr)
        for r in resolved:
            r["eval_init_states_path"] = ""
    run_name = _resolve_output_name(cfg)
    if multi:
        models_json = json.dumps([
            {"label": lbl, "policy_path": str(r["policy_path"]), "base_fsq": r["base_fsq"],
             "ft_fsq_path": str(r["ft_fsq_path"]), "ft_run_dir": str(r["ft_run_dir"]),
             "checkpoint": r["checkpoint"], "resolved_term": r["resolved_term"],
             "gt_skill_dataset_dir": r["gt_skill_dataset_dir"],
             "eval_init_states_path": r["eval_init_states_path"],
             "advance_mode": r["advance_mode"],
             "drop_vlm": bool(d), "keep_adapters": bool(k)}
            for lbl, r, d, k in zip(labels, resolved, drop_flags, keepad_flags)])
    else:
        models_json = ""
    eval_out_dir = _HERE.parent.parent / "outputs" / run_name

    settings: dict = {
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        # model-0 aliases (submit's artifact checks + the single-model path use these directly).
        # MULTI: the full per-model list rides MODELS_JSON (read by eval.sbatch's panel loop).
        "models_json": models_json,
        "models_labels": json.dumps(labels) if multi else "",
        "models_per_row": int(get_value(cfg, "models_per_row", 2) or 0),
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
        "skill_advance_mode": m0["advance_mode"],
        # Blank uses this Stage-0 checkpoint's predictor. A Stage-3 path swaps only the skill-code
        # producer; action generation and the FSQ terminator remain owned by the Stage-0 checkpoint.
        "eval_skill_predictor_path": stage3_predictor_path or "",
        # output / wandb — chunked submission (TASK_TAG, e.g. "t0-4") → distinct wandb run per chunk
        "output_name": run_name,
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
