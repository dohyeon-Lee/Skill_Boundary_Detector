#!/usr/bin/env python3
"""Config for SkillVLA closed-loop EVAL (PT) on LIBERO sim.

Imports the PT training config for the model path + FSQ path + roots, then merges
eval-only knobs. Pick which trained model to evaluate by NAME + sub-folder
(+ batch_size / exp / checkpoint); blank fields inherit
../skillVLA/skillVLA_train_config.yaml. Emits shell exports (--shell).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent / "train_skills" / "src"))
sys.path.insert(0, str(_HERE.parent.parent.parent / "skillVLA" / "src"))
from train_skills_config import as_bool, as_list, get_value, load_config, print_shell  # noqa: E402
from skillVLA_train_config import _is_set, build_settings as train_settings  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "skillVLA_eval_config.yaml"
TRAIN_CONFIG_PATH = _HERE.parent.parent.parent / "skillVLA" / "skillVLA_train_config.yaml"

# Fields that identify which PT run to evaluate (blank → inherit the train yaml).
MODEL_SELECTORS = ("source_dataset", "run_tag", "batch_size", "exp")


def build_settings(cfg: dict, train_cfg_path: Path | None = None) -> dict:
    train_cfg = load_config(str(train_cfg_path or TRAIN_CONFIG_PATH))
    for key in MODEL_SELECTORS:
        if _is_set(cfg.get(key)):
            train_cfg[key] = cfg[key]
    tr = train_settings(train_cfg)

    checkpoint = str(get_value(cfg, "checkpoint", "last"))
    target_task = str(get_value(cfg, "target_task", "libero_90"))
    policy_path = Path(str(tr["pt_output_dir"])) / "checkpoints" / checkpoint / "pretrained_model"
    run_name = f"{tr['pt_run_name']}_{checkpoint}_{target_task}_fsq_eval"
    eval_out_dir = _HERE.parent.parent / "outputs" / run_name

    settings: dict = {
        "project_root": tr["project_root"],
        "lerobot_root": tr["lerobot_root"],
        # model + FSQ (from the PT run)
        "policy_path": policy_path,
        "pt_run_name": tr["pt_run_name"],
        "checkpoint": checkpoint,
        "fsq_ckpt": tr["fsq_ckpt"],
        "image_model_path": tr["image_model_path"],
        "skill_decoder_state_indices": tr["skill_decoder_state_indices"],
        "use_fsq_latent_suffix": tr["use_fsq_latent_suffix"],
        "raw_dataset_dir": tr["raw_dataset_dir"],
        "image_key": tr["image_key"],
        "skill_latents_path": tr["skill_latents_path"],
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
        # inference knobs (eval-only)
        "skill_decoder_end_threshold": str(get_value(cfg, "skill_decoder_end_threshold", 0.5)),
        "inference_skill_max_length": int(get_value(cfg, "inference_skill_max_length", 200)),
        # output / wandb
        "wandb_project": str(get_value(cfg, "wandb_project", "VLA_eval")),
        "wandb_run_name": run_name,
        "eval_out_dir": eval_out_dir,
    }

    part = ",".join(as_list(get_value(cfg, "eval_partition", ["debug"]))) or "debug"
    excl = ",".join(as_list(get_value(cfg, "eval_exclude_nodes", [])))
    settings.update({
        "eval_partition": part,
        "eval_qos": str(get_value(cfg, "eval_qos", "base_qos")),
        "eval_gres": str(get_value(cfg, "eval_gres", "gpu:1")),
        "eval_cpus_per_task": int(get_value(cfg, "eval_cpus_per_task", 8)),
        "eval_mem": str(get_value(cfg, "eval_mem", "32G")),
        "eval_time": str(get_value(cfg, "eval_time", "1:00:00")),
        "eval_nodelist": str(get_value(cfg, "eval_nodelist", "")),
        "eval_exclude_nodes": excl,
    })
    return settings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--train_config", type=Path, default=None)
    ap.add_argument("--shell", action="store_true")
    args = ap.parse_args()
    settings = build_settings(load_config(args.config), train_cfg_path=args.train_config)
    if args.shell:
        print_shell(settings)
    else:
        for k, v in settings.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
