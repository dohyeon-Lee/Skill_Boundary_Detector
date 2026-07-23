#!/usr/bin/env python3
"""skill_eval config: emit evaluation-only knobs + slurm settings as shell exports.

Root paths come from train_skills_config.py; this only owns eval-specific knobs.
Owns the eval_run_fsq / eval_run_dp toggles and the DP-eval knobs in addition to
the FSQ-eval knobs.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# reuse the train_skills config helpers (yaml load + shell emitter)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from train_skills_config import as_bool, as_list, get_value, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "eval_config.yaml"


def build_settings(config_path: str | None = None) -> dict:
    cfg = load_config(config_path or DEFAULT_CONFIG_PATH)
    # Slurm partition/qos/nodelist/exclude are canonical (global_config.yaml train_*).
    exclude = as_list(get_value(cfg, "train_exclude_nodes", []))
    output_suffix = str(get_value(cfg, "dp_eval_output_suffix", "")).strip()
    if output_suffix and not re.fullmatch(r"[A-Za-z0-9._-]+", output_suffix):
        raise ValueError(
            "dp_eval_output_suffix may contain only letters, digits, '.', '_' and '-', "
            f"got {output_suffix!r}"
        )
    random_far_fraction = float(get_value(cfg, "fsq_eval_random_far_fraction", 0.1))
    if not 0.0 < random_far_fraction <= 1.0:
        raise ValueError(
            f"fsq_eval_random_far_fraction must be in (0,1], got {random_far_fraction}."
        )
    return {
        # which eval(s) to run
        "eval_run_fsq":            str(as_bool(get_value(cfg, "eval_run_fsq", True))).lower(),
        "eval_run_dp":             str(as_bool(get_value(cfg, "eval_run_dp", True))).lower(),
        # DP selection: blank = follow train_skills_config; else the DP folder name (+ checkpoint) to eval.
        "eval_dp_run_name":        str(get_value(cfg, "eval_dp_run_name", "")),
        "eval_dp_checkpoint":      str(get_value(cfg, "eval_dp_checkpoint", "")),
        "dp_eval_skillset_dir":    str(get_value(cfg, "dp_eval_skillset_dir", "")),
        "dp_eval_output_suffix":   output_suffix,
        "skillset_boundary_threshold_mode": str(
            get_value(cfg, "skillset_boundary_threshold_mode", "episode_mean")
        ),
        # DP skill-boundary eval knobs
        "dp_eval_n_episodes":      int(get_value(cfg, "dp_eval_n_episodes", 10)),
        "dp_eval_task_ids":        " ".join(as_list(get_value(cfg, "dp_eval_task_ids", []))),
        "dp_eval_skill_video":     str(as_bool(get_value(cfg, "dp_eval_skill_video", False))).lower(),
        "dp_eval_show_start_end_frames": str(
            as_bool(get_value(cfg, "dp_eval_show_start_end_frames", True))
        ).lower(),
        "dp_eval_show_cos_graph":  str(as_bool(get_value(cfg, "dp_eval_show_cos_graph", True))).lower(),
        "dp_eval_show_gripper_graph": str(
            as_bool(get_value(cfg, "dp_eval_show_gripper_graph", True))
        ).lower(),
        "fsq_eval_run_name":       str(get_value(cfg, "fsq_eval_run_name", "")),
        "fsq_eval_dino_model_path": str(
            get_value(cfg, "fsq_eval_dino_model_path", "models/dinov3-vitl16")
        ),
        "fsq_eval_checkpoint":     str(get_value(cfg, "fsq_eval_checkpoint", 0)),
        "fsq_eval_n_action_steps": int(get_value(cfg, "fsq_eval_n_action_steps", 5)),
        "fsq_eval_n_samples":      int(get_value(cfg, "fsq_eval_n_samples", 5)),
        "fsq_eval_max_entries":    int(get_value(cfg, "fsq_eval_max_entries", 0)),
        "fsq_eval_decoder_scope":  str(get_value(cfg, "fsq_eval_decoder_scope", "samples")),
        "fsq_eval_end_threshold":  str(get_value(cfg, "fsq_eval_end_threshold", 0.5)),
        "fsq_eval_random_far_skill": str(
            as_bool(get_value(cfg, "fsq_eval_random_far_skill", True))
        ).lower(),
        "fsq_eval_random_far_fraction": str(random_far_fraction),
        "fsq_eval_thumb_size":     int(get_value(cfg, "fsq_eval_thumb_size", 160)),
        "fsq_eval_image_key":      str(get_value(cfg, "fsq_eval_image_key", "observation.images.image")),
        "fsq_eval_wandb_project":  str(get_value(cfg, "fsq_eval_wandb_project", "VAE_eval")),
        "fsq_eval_partition":      ",".join(as_list(get_value(cfg, "train_partition", ["debug"]))) or "debug",
        "fsq_eval_qos":            str(get_value(cfg, "train_qos", "base_qos")),
        "fsq_eval_gres":           str(get_value(cfg, "fsq_eval_gres", "gpu:1")),
        "fsq_eval_cpus_per_task":  int(get_value(cfg, "fsq_eval_cpus_per_task", 4)),
        "fsq_eval_mem":            str(get_value(cfg, "fsq_eval_mem", "32G")),
        "fsq_eval_time":           str(get_value(cfg, "fsq_eval_time", "02:00:00")),
        "fsq_eval_nodelist":       str(get_value(cfg, "train_nodelist", "")),
        "fsq_eval_exclude_nodes":  ",".join(exclude),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--shell", action="store_true")
    args = ap.parse_args()
    settings = build_settings(args.config)
    if args.shell:
        print_shell(settings)
    else:
        for k, v in settings.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
