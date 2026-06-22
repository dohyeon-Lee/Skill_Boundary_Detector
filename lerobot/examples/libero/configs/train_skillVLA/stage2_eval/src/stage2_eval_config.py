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
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import as_bool, as_list, get_value, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "stage2_eval_config.yaml"


def build_settings(cfg: dict) -> dict:
    # Standalone: roots declared in this yaml (no build_data dependency).
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    outputs_root = project_root / str(get_value(cfg, "outputs_root", "outputs"))
    vla_root = outputs_root / "skillVLA_stage2"   # fixed subdir (matches stage2 training)
    model_dir = str(get_value(cfg, "model_dir"))
    checkpoint = str(get_value(cfg, "checkpoint", "last"))
    target_task = str(get_value(cfg, "target_task", "libero_90"))
    image_key = str(get_value(cfg, "image_key", "observation.images.image"))

    policy_path = vla_root / model_dir / "checkpoints" / checkpoint / "pretrained_model"

    # The checkpoint config.json is the single source of truth for the trained model.
    # FSQ path → derive the skill_latents + raw-dataset paths used for skill_html.
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
        gt_skill_dataset_dir = str(fp.parent / "skillvla")  # skillvla dataset (skill_sequence) for oracle GT
        try:
            raw_dataset_dir = str(fp.parents[3] / fp.parents[1].name)  # {dataset_root}/{source}
        except IndexError:
            raw_dataset_dir = ""

    # Terminator co-trained in Stage-2 → the adapted FSQ from THIS checkpoint (checkpoints/<ckpt>/
    # FSQ_ft.pt; the eval sbatch lazy-exports it if missing). The base dataset FSQ is the pre-co-train one.
    ft_fsq_path = policy_path.parent / "FSQ_ft.pt"

    # Which terminator to use at eval (user toggle): auto = co-trained if the run trained one else base;
    # cotrained / base force one. resolved_term (base|cotrained) drives both the eval.sbatch FSQ pick and
    # the folder tag, so base vs cotrained evals land side-by-side.
    eval_terminator = str(get_value(cfg, "eval_terminator", "auto")).lower()
    if eval_terminator not in ("auto", "cotrained", "base"):
        raise ValueError(f"eval_terminator must be 'auto', 'cotrained' or 'base' (got {eval_terminator!r}).")
    if eval_terminator == "cotrained" and not train_terminator_used:
        raise ValueError(
            "eval_terminator='cotrained' but this checkpoint was trained with train_terminator=false "
            "(no co-trained terminator exists). Use 'base' or 'auto'.")
    resolved_term = eval_terminator if eval_terminator != "auto" else ("cotrained" if train_terminator_used else "base")

    # Skill-source tag so oracle (GT) vs VLM-predicted and the advance mode land in DISTINCT folders.
    use_gt_skill = as_bool(get_value(cfg, "use_gt_skill", False))
    advance_mode = str(get_value(cfg, "skill_advance_mode", "terminator"))
    skill_tag = f"gtskill-{advance_mode}" if use_gt_skill else "pred"
    term_tag = "ftterm" if resolved_term == "cotrained" else "baseterm"
    run_name = f"{model_dir}_{checkpoint}_{target_task}_{skill_tag}_{term_tag}_eval"
    eval_out_dir = _HERE.parent.parent / "outputs" / run_name

    settings: dict = {
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        # model (everything else is restored from the checkpoint config on --policy.path)
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
        "image_key": image_key,
        # eval rollout
        "target_task": target_task,
        "task_ids": str(get_value(cfg, "task_ids", "[0,1,2,3,4,5,6,7,8,9]")),
        "n_episodes": int(get_value(cfg, "n_episodes", 1)),
        "n_action_steps": int(get_value(cfg, "n_action_steps", 5)),
        "eval_batch_size": int(get_value(cfg, "eval_batch_size", 1)),
        "max_parallel_tasks": int(get_value(cfg, "max_parallel_tasks", 1)),
        "max_videos_per_task": int(get_value(cfg, "max_videos_per_task", 1)),
        "video_frame_stride": int(get_value(cfg, "video_frame_stride", 2)),
        "video_fps": int(get_value(cfg, "video_fps", 10)),
        "skill_html": as_bool(get_value(cfg, "skill_html", True)),
        "skill_html_train_samples": int(get_value(cfg, "skill_html_train_samples", 10)),
        # inference knobs (eval-time tuning; model structure comes from the checkpoint)
        "skill_end_threshold": str(get_value(cfg, "skill_end_threshold", 0.5)),
        "inference_skill_max_length": int(get_value(cfg, "inference_skill_max_length", 200)),
        # oracle eval: teacher-force GT skills + pick transition timing (gt vs terminator)
        "use_gt_skill": use_gt_skill,
        "gt_skill_dataset_dir": gt_skill_dataset_dir,
        "skill_advance_mode": advance_mode,
        # output / wandb
        "wandb_project": str(get_value(cfg, "wandb_project", "VLA_eval")),
        "wandb_run_name": run_name,
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
