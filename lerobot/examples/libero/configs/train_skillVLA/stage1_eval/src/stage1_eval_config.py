#!/usr/bin/env python3
"""Config for SkillVLA Stage-1 (skill_expert) closed-loop oracle EVAL.

Point at a trained Stage-1 run by OUTPUT FOLDER NAME + checkpoint. The FSQ terminator
(FSQ.pt) and the GT skill sequences (skillvla dataset) come from the same {run_dir} the
model was trained on (source_dataset + run_tag). All roots are declared in this yaml
(standalone). Env tasks are matched to dataset tasks by language. Emits shell exports (--shell).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import as_bool, as_list, get_value, load_config, print_shell, resolve_path  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "stage1_eval_config.yaml"


_RUN_TAG_RE = re.compile(r"(FSQ\d+_dino\d+.*?_(?:\d+|best))_(?:[a-zA-Z][^_]*_)*batch\d+")


def _resolve_model(model_dir: str, checkpoint: str, *, skillvla_root: Path, vla_root: Path,
                   source_yaml: str) -> dict:
    """Resolve one training run's folder name → its run_tag + artifact paths (policy checkpoint, FSQ.pt,
    skillvla dataset, skill_latents, eval_init_states). source_dataset = the model_dir prefix if present
    (old naming {source}_{run_tag}_...), else the yaml's source_dataset (DP naming drops the prefix)."""
    m = _RUN_TAG_RE.search(model_dir)
    if not m:
        raise ValueError(f"model_dir must embed a 'FSQ..._dino..._<ckpt>_batch<N>' run tag, got: {model_dir}")
    run_tag = m.group(1)
    source = model_dir[: m.start()].rstrip("_") or source_yaml
    if not source:
        raise ValueError(
            "Could not determine source_dataset: model_dir has no '{source}_' prefix and 'source_dataset' "
            f"is not set in the eval yaml (e.g. libero_90_full_full). model_dir={model_dir}")
    run_dir = skillvla_root / source / run_tag
    return {
        "model_dir": model_dir, "checkpoint": checkpoint, "run_tag": run_tag, "source": source,
        "run_dir": run_dir,
        "policy_path": vla_root / model_dir / "checkpoints" / checkpoint / "pretrained_model",
        "fsq_path": run_dir / "FSQ.pt",
        "skill_label_dataset_dir": run_dir / "skillvla",
        "skill_latents_path": run_dir / "skill_latents.npz",
        "eval_init_states_path": skillvla_root / source / "eval_init_states.npz",
    }


def _auto_labels(model_dirs: list[str]) -> list[str]:
    """Distinguishing middle token(s) of each model_dir (strip the common leading + trailing _-tokens),
    used as the side-by-side panel label when a model has no explicit label."""
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
    # Standalone: every root is declared in this yaml (no build_data dependency).
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    dataset_root = project_root / str(get_value(cfg, "dataset_root", "dataset"))
    skillvla_root = dataset_root / str(get_value(cfg, "skillvla_dataset_root", "skillvla_dataset"))
    lerobot_root = project_root / "lerobot"
    outputs_root = project_root / str(get_value(cfg, "outputs_root", "outputs"))
    vla_root = outputs_root / "skillVLA_stage1"
    source_yaml = str(get_value(cfg, "source_dataset", "")).strip()
    default_ckpt = str(get_value(cfg, "checkpoint", "last"))

    # `models` (a list of {model_dir, checkpoint?, label?}) → MULTI-model side-by-side eval; otherwise the
    # single `model_dir`/`checkpoint`. A 1-entry list behaves like the single case.
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

    resolved = [_resolve_model(e["model_dir"], e["checkpoint"], skillvla_root=skillvla_root,
                               vla_root=vla_root, source_yaml=source_yaml) for e in entries]
    for r, a in zip(resolved, _auto_labels([e["model_dir"] for e in entries])):
        r["label"] = next(e["label"] for e in entries if e["model_dir"] == r["model_dir"]) or a
    m0 = resolved[0]
    multi = len(resolved) >= 2

    advance_mode = str(get_value(cfg, "skill_advance_mode", "terminator"))
    eval_exp = str(get_value(cfg, "eval_exp", "")).strip()
    # Optional terminator override: load the terminator from a DIFFERENT run's FSQ.pt (single-model only).
    terminator_path = str(get_value(cfg, "terminator_path", "")).strip()

    if multi:
        # MULTI: env-side artifacts (POLICY_PATH etc.) point at model 0 so cfg.policy + the sbatch's artifact
        # checks are valid; the FULL per-model list rides MODELS_JSON (read by run_eval). Combined folder.
        folder = "compare_" + "_vs_".join(r["label"] for r in resolved) + f"_{m0['checkpoint']}"
        if eval_exp:
            folder = f"{folder}_{eval_exp}"
        models_json = json.dumps([
            {"policy_path": str(r["policy_path"]), "fsq_path": str(r["fsq_path"]),
             "skill_label_dataset_dir": str(r["skill_label_dataset_dir"]), "label": r["label"]}
            for r in resolved])
        fsq_ckpt = m0["fsq_path"]
    else:
        folder = f"{m0['model_dir']}_{m0['checkpoint']}"
        if advance_mode != "terminator":   # terminator is the default → keep the folder name clean; tag only gt
            folder = f"{folder}_adv-{advance_mode}"
        if eval_exp:
            folder = f"{folder}_{eval_exp}"
        if terminator_path:  # distinct folder so a refined-terminator run doesn't clobber the FSQ-term run
            _ts = re.search(r"checkpoints/([^/]+)/", terminator_path)
            folder = f"{folder}_refterm{_ts.group(1) if _ts else ''}"
        models_json = ""
        fsq_ckpt = resolve_path(project_root, terminator_path) if terminator_path else m0["fsq_path"]

    # Model-0 aliases (env-side single exports; also the single-model path uses these directly).
    policy_path = m0["policy_path"]
    run_dir = m0["run_dir"]
    source_dataset = m0["source"]
    stage1_eval_dir = _HERE.parent.parent
    eval_out_dir = stage1_eval_dir / "outputs" / folder

    settings: dict = {
        "project_root": project_root,
        "lerobot_root": lerobot_root,
        # model + eval artifacts (FSQ + skillvla dataset from the training run_dir). For MULTI-model
        # side-by-side, these are model 0's; the full per-model list is in models_json (env → run_eval).
        "policy_path": policy_path,
        "models_json": models_json,
        # side-by-side grid: N panels per ROW (0 = one row; 3 → 6 models = 2×3, 9 = 3×3, ...)
        "models_per_row": int(get_value(cfg, "models_per_row", 0) or 0),
        "fsq_ckpt": fsq_ckpt,
        "skill_label_dataset_dir": run_dir / "skillvla",
        # Episode-exact eval: per-episode MuJoCo init_state + scene (FSQ-independent → lives at the source
        # parent, shared by every FSQ run). Built by stage1_eval/oracle_matching/run.sh {source}.
        "eval_init_states_path": skillvla_root / source_dataset / "eval_init_states.npz",
        "source_dataset": source_dataset,
        # FSQ terminator's raw-image DINO (the policy's own backbone comes from the checkpoint).
        "terminator_dino_model_path": resolve_path(
            project_root, get_value(cfg, "terminator_dino_model_path", "models/dinov3-vits16")),
        "eval_out_dir": eval_out_dir,
        # skill HTML (FSQ cube + used skills + per-skill progress + FSQ-space samples)
        "skill_html": as_bool(get_value(cfg, "skill_html", True)),
        "skill_html_train_samples": int(get_value(cfg, "skill_html_train_samples", 6)),
        "skill_latents_path": run_dir / "skill_latents.npz",
        "skill_html_raw_dataset_dir": run_dir / "skillvla",
        "image_key": "observation.images.image",
        # wandb
        "wandb_enable": as_bool(get_value(cfg, "wandb_enable", True)),
        # env / rollout — task_ids is env-overridable (TASK_IDS): submit_eval.sh splits it across
        # eval_num_gpus Slurm jobs (1 GPU each), all writing into the SAME out dir (disjoint tasks).
        "target_task": str(get_value(cfg, "target_task", "libero_90")),
        "task_ids": str(get_value(cfg, "task_ids", "[0,1,2,3,4,5,6,7,8,9]")),
        "eval_num_gpus": int(get_value(cfg, "eval_num_gpus", 1)),
        "n_episodes": int(get_value(cfg, "n_episodes", 5)),
        "eval_batch_size": int(get_value(cfg, "eval_batch_size", 1)),
        "max_parallel_tasks": int(get_value(cfg, "max_parallel_tasks", 1)),
        "n_action_steps": int(get_value(cfg, "n_action_steps", 5)),
        "max_videos_per_task": int(get_value(cfg, "max_videos_per_task", 1)),
        "video_frame_stride": int(get_value(cfg, "video_frame_stride", 2)),
        "video_fps": int(get_value(cfg, "video_fps", 10)),
        # terminator
        "skill_advance_mode": advance_mode,
        "skill_end_mode": str(get_value(cfg, "skill_end_mode", "termination")),
        "skill_end_threshold": str(get_value(cfg, "skill_end_threshold", 0.5)),
        "skill_end_progress_threshold": str(get_value(cfg, "skill_end_progress_threshold", 0.9)),
        "inference_skill_max_length": int(get_value(cfg, "inference_skill_max_length", 200)),
        # Use the checkpoint's co-trained terminator; a terminator_path override forces the raw FSQ.pt.
        "eval_use_trained_terminator": as_bool(get_value(cfg, "eval_use_trained_terminator", True)) and not terminator_path,
        # wandb
        "wandb_project": str(get_value(cfg, "wandb_project", "VLA_stage1_eval")),
        # Chunked submission (TASK_TAG, e.g. "t0-4") → distinct wandb run per chunk.
        "wandb_run_name": f"S1eval_{folder}" + (f"_{os.environ['TASK_TAG']}" if os.environ.get("TASK_TAG") else ""),
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
