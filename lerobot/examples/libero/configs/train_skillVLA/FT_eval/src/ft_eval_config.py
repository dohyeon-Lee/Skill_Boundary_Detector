#!/usr/bin/env python3
"""Config for SkillVLA FT closed-loop EVAL on LIBERO sim.

Mirrors stage2_eval but points at a FINETUNED run under {outputs_root}/skillVLA_FT/ and swaps in the
FT-adapted terminator: if an ``FSQ_ft.pt`` exists in the run dir (exported by FT training) it overrides
the checkpoint's ``fsq_path`` so the closed loop gates skill transitions with the terminator that was
co-trained on the new task. The model structure is otherwise restored from the checkpoint config.json;
``train_terminator`` is forced false at eval (no co-training terminator is built). Emits shell exports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import as_bool, as_list, get_value, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "ft_eval_config.yaml"


def build_settings(cfg: dict) -> dict:
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    outputs_root = project_root / str(get_value(cfg, "outputs_root", "outputs"))
    vla_root = outputs_root / "skillVLA_FT"          # FT runs (matches FT training output)
    model_dir = str(get_value(cfg, "model_dir"))
    checkpoint = str(get_value(cfg, "checkpoint", "last"))
    target_task = str(get_value(cfg, "target_task", "libero_10"))
    image_key = str(get_value(cfg, "image_key", "observation.images.image"))

    run_root = vla_root / model_dir
    policy_path = run_root / "checkpoints" / checkpoint / "pretrained_model"

    # The checkpoint config.json is the source of truth for the trained model. fsq_path (dataset FSQ)
    # → derive the skill_latents + raw-dataset paths used for skill_html / oracle GT.
    pol: dict = {}
    cfg_json = policy_path / "config.json"
    if cfg_json.is_file():
        pol = json.loads(cfg_json.read_text())
    base_fsq = str(pol.get("fsq_path") or "")
    train_terminator_used = as_bool(pol.get("train_terminator", False))
    skill_latents_path = ""
    raw_dataset_dir = ""
    gt_skill_dataset_dir = ""
    if base_fsq:
        fp = Path(base_fsq)  # {root}/{dataset_root}/skillvla_dataset/{source}/{run_tag}/FSQ.pt
        skill_latents_path = str(fp.parent / "skill_latents.npz")
        gt_skill_dataset_dir = str(fp.parent / "skillvla")
        try:
            raw_dataset_dir = str(fp.parents[3] / fp.parents[1].name)
        except IndexError:
            raw_dataset_dir = ""

    # FT-adapted terminator from the SAME checkpoint: checkpoints/<ckpt>/FSQ_ft.pt (exported at train
    # time; the eval sbatch lazy-exports it from this checkpoint if missing). The closed loop uses it
    # when present (train_terminator runs), else falls back to the base FSQ recorded in the config.
    ft_fsq_path = policy_path.parent / "FSQ_ft.pt"

    # Stage-2 terminator: FT warm-starts the WHOLE policy (incl. the co-trained terminator) from a
    # Stage-2 checkpoint, recorded in the config as pretrained_path. eval_terminator=stage2 evaluates
    # with THAT terminator — the one FT INHERITED, before FT's own co-training adapted it (an ablation
    # of FT's terminator adaptation). Stage-2 training exports it per checkpoint to
    # <stage2_ckpt>/FSQ_ft.pt; the eval sbatch lazy-exports from the Stage-2 checkpoint if missing.
    stage2_pretrained = str(pol.get("pretrained_path") or "")
    if stage2_pretrained:
        _s2 = Path(stage2_pretrained).parent          # …/skillVLA_stage2/<run>/checkpoints/<step>
        stage2_fsq_path = _s2 / "FSQ_ft.pt"
        stage2_run_dir = _s2.parents[1]                # …/skillVLA_stage2/<run>
        stage2_checkpoint = _s2.name                   # <step>
    else:
        stage2_fsq_path = stage2_run_dir = stage2_checkpoint = ""

    # Which terminator decides skill transitions at eval (action chunk always from the expert):
    #   ft     → the FT-adapted terminator co-trained INTO this checkpoint (FSQ_ft.pt); falls back to
    #            base if FT didn't co-train. (default — matches the model you finetuned)
    #   base   → the new task's ORIGINAL dataset FSQ.pt (terminator before ANY in-policy co-training)
    #   stage2 → the terminator FT inherited from its Stage-2 source checkpoint (before FT adapted it)
    eval_terminator = str(get_value(cfg, "eval_terminator", "ft")).strip().lower()
    if eval_terminator not in ("ft", "base", "stage2"):
        raise ValueError(f"eval_terminator must be 'ft', 'base', or 'stage2', got {eval_terminator!r}")
    if eval_terminator == "stage2" and not stage2_pretrained:
        raise ValueError("eval_terminator=stage2 but the FT checkpoint config has no pretrained_path "
                         "(the Stage-2 source). Can't locate the Stage-2 terminator.")

    use_gt_skill = as_bool(get_value(cfg, "use_gt_skill", False))
    advance_mode = str(get_value(cfg, "skill_advance_mode", "terminator"))
    # Folder suffix = _{term}_{mode}: term is the RESOLVED terminator (ft / base / s2), mode is the
    # skill_end_mode (and / pro / ter). Oracle runs get a gtskill prefix so they don't collide.
    if eval_terminator == "stage2":
        term_tag = "s2"
    elif eval_terminator == "base":
        term_tag = "base"
    else:  # ft → resolves to base when FT didn't co-train a terminator
        term_tag = "ft" if train_terminator_used else "base"
    mode_tag = {"and": "and", "progress": "pro", "termination": "ter"}.get(
        str(get_value(cfg, "skill_end_mode", "termination")), "ter")
    suffix = f"{term_tag}_{mode_tag}"
    if use_gt_skill:
        suffix = f"gtskill-{advance_mode}_{suffix}"
    run_name = f"{model_dir}_{checkpoint}_{target_task}_{suffix}"
    eval_out_dir = _HERE.parent.parent / "outputs" / run_name

    settings: dict = {
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        # model (structure restored from the checkpoint config on --policy.path)
        "policy_path": policy_path,
        "checkpoint": checkpoint,
        # terminator: the eval sbatch resolves FSQ_FOR_EVAL = ft_fsq_path (lazy-exported from THIS
        # checkpoint if missing) when train_terminator ran, else base_fsq.
        "base_fsq": base_fsq,                 # always-present dataset FSQ (prereq check + fallback + export base)
        "ft_fsq_path": ft_fsq_path,           # per-checkpoint adapted terminator (checkpoints/<ckpt>/FSQ_ft.pt)
        "ft_run_dir": run_root,               # for export_ft_terminator.py --ft_run_dir
        "train_terminator_used": train_terminator_used,
        # terminator choice (ft | base | stage2) + the Stage-2 source terminator paths
        "eval_terminator": eval_terminator,
        "stage2_fsq_path": str(stage2_fsq_path),     # Stage-2 ckpt's FSQ_ft.pt (eval_terminator=stage2)
        "stage2_run_dir": str(stage2_run_dir),       # for lazy export_ft_terminator.py --ft_run_dir
        "stage2_checkpoint": str(stage2_checkpoint), # Stage-2 ckpt step
        "skill_latents_path": skill_latents_path,
        "raw_dataset_dir": raw_dataset_dir,
        "image_key": image_key,
        # eval rollout
        "target_task": target_task,
        "task_ids": str(get_value(cfg, "task_ids", "[0,1,2,3,4,5,6,7,8,9]")),
        "n_episodes": int(get_value(cfg, "n_episodes", 5)),
        "n_action_steps": int(get_value(cfg, "n_action_steps", 5)),
        "eval_batch_size": int(get_value(cfg, "eval_batch_size", 1)),
        "max_parallel_tasks": int(get_value(cfg, "max_parallel_tasks", 1)),
        "max_videos_per_task": int(get_value(cfg, "max_videos_per_task", 5)),
        "video_frame_stride": int(get_value(cfg, "video_frame_stride", 2)),
        "video_fps": int(get_value(cfg, "video_fps", 10)),
        "skill_html": as_bool(get_value(cfg, "skill_html", True)),
        "skill_html_train_samples": int(get_value(cfg, "skill_html_train_samples", 10)),
        # inference knobs (model structure comes from the checkpoint)
        "skill_end_mode": str(get_value(cfg, "skill_end_mode", "termination")),
        "skill_end_threshold": str(get_value(cfg, "skill_end_threshold", 0.5)),
        "skill_end_progress_threshold": str(get_value(cfg, "skill_end_progress_threshold", 0.9)),
        "inference_skill_max_length": int(get_value(cfg, "inference_skill_max_length", 200)),
        # oracle eval
        "use_gt_skill": use_gt_skill,
        "gt_skill_dataset_dir": gt_skill_dataset_dir,
        "skill_advance_mode": advance_mode,
        # output / wandb
        "wandb_project": str(get_value(cfg, "wandb_project", "VLA_eval")),
        "wandb_run_name": run_name,
        "eval_out_dir": eval_out_dir,
    }

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
