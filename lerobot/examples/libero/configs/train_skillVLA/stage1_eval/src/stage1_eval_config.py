#!/usr/bin/env python3
"""Config for SkillVLA Stage-1 (skill_expert) closed-loop oracle EVAL.

Point at a trained Stage-1 run by OUTPUT FOLDER NAME + checkpoint. The FSQ terminator
(FSQ.pt) and the GT skill sequences (skillvla dataset) come from the same {run_dir} the
model was trained on (source_dataset + run_tag). All roots are declared in this yaml
(standalone). Env tasks are matched to dataset tasks by language. Emits shell exports (--shell).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import as_bool, as_list, get_value, load_config, print_shell, resolve_path  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "stage1_eval_config.yaml"


def build_settings(cfg: dict) -> dict:
    # Standalone: every root is declared in this yaml (no build_data dependency).
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    dataset_root = project_root / str(get_value(cfg, "dataset_root", "dataset"))
    skillvla_root = dataset_root / str(get_value(cfg, "skillvla_dataset_root", "skillvla_dataset"))
    lerobot_root = project_root / "lerobot"

    model_dir = str(get_value(cfg, "model_dir"))           # e.g. libero_90_full_full_FSQ88_dino8_1000_batch32_A
    # model_dir fully identifies the trained run: {source}_{run_tag}_batch{N}_{A|B}[_exp].
    # Parse both run_tag (FSQ..._dino..._{ckpt}[_exp]) and source_dataset (the prefix) from it.
    # run_tag ends with _<ckpt> (numeric or "best"); vis_tag follows (e.g. siglip_freeze).
    # Pattern: FSQ..._dino..._<ckpt> then optional letter-starting tokens, then _batch<N>.
    _rt = re.search(r"(FSQ\d+_dino\d+.*?_(?:\d+|best))_(?:[a-zA-Z][^_]*_)*batch\d+", model_dir)
    if not _rt:
        raise ValueError(f"model_dir must embed a 'FSQ..._dino..._<ckpt>_batch<N>' run tag, got: {model_dir}")
    run_tag = _rt.group(1)
    source_dataset = model_dir[: _rt.start()].rstrip("_")  # the part before the run tag
    run_dir = skillvla_root / source_dataset / run_tag

    # Single outputs root from yaml; the stage-1 subdir is fixed here (matches stage1 training).
    outputs_root = project_root / str(get_value(cfg, "outputs_root", "outputs"))
    vla_root = outputs_root / "skillVLA_stage1"
    checkpoint = str(get_value(cfg, "checkpoint", "last"))
    policy_path = vla_root / model_dir / "checkpoints" / checkpoint / "pretrained_model"

    # Output folder name = model + checkpoint + skill-advance mode + optional free-form tag,
    # so terminator vs gt (and any ablation) runs land in distinct folders.
    advance_mode = str(get_value(cfg, "skill_advance_mode", "terminator"))
    eval_exp = str(get_value(cfg, "eval_exp", "")).strip()
    run_tag = f"{model_dir}_{checkpoint}_adv-{advance_mode}"
    if eval_exp:
        run_tag = f"{run_tag}_{eval_exp}"

    # Results under stage1_eval/outputs/{run_tag}/
    stage1_eval_dir = _HERE.parent.parent
    eval_out_dir = stage1_eval_dir / "outputs" / run_tag

    settings: dict = {
        "project_root": project_root,
        "lerobot_root": lerobot_root,
        # model + eval artifacts (FSQ + skillvla dataset from the training run_dir)
        "policy_path": policy_path,
        "fsq_ckpt": run_dir / "FSQ.pt",
        "skill_label_dataset_dir": run_dir / "skillvla",
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
        # env / rollout
        "target_task": str(get_value(cfg, "target_task", "libero_90")),
        "task_ids": str(get_value(cfg, "task_ids", "[0,1,2,3,4,5,6,7,8,9]")),
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
        "inference_skill_max_length": int(get_value(cfg, "inference_skill_max_length", 200)),
        # wandb
        "wandb_project": str(get_value(cfg, "wandb_project", "VLA_stage1_eval")),
        "wandb_run_name": f"S1eval_{run_tag}",
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
