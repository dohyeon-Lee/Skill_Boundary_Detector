#!/usr/bin/env python3
"""skill_eval config: emit evaluation-only knobs + slurm settings as shell exports.

Root paths come from train_skills_config.py; this only owns eval-specific knobs.
Owns the eval_run_fsq / eval_run_dp toggles and the DP-eval knobs in addition to
the FSQ-eval knobs.
"""

from __future__ import annotations

import argparse
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
    return {
        # which eval(s) to run
        "eval_run_fsq":            str(as_bool(get_value(cfg, "eval_run_fsq", True))).lower(),
        "eval_run_dp":             str(as_bool(get_value(cfg, "eval_run_dp", True))).lower(),
        # DP selection: blank = follow train_skills_config; else the DP folder name (+ checkpoint) to eval.
        "eval_dp_run_name":        str(get_value(cfg, "eval_dp_run_name", "")),
        "eval_dp_checkpoint":      str(get_value(cfg, "eval_dp_checkpoint", "")),
        # DP skill-boundary eval knobs
        "dp_eval_n_episodes":      int(get_value(cfg, "dp_eval_n_episodes", 10)),
        "dp_eval_task_ids":        " ".join(as_list(get_value(cfg, "dp_eval_task_ids", []))),
        "fsq_eval_run_name":       str(get_value(cfg, "fsq_eval_run_name", "")),
        "fsq_eval_checkpoint":     str(get_value(cfg, "fsq_eval_checkpoint", 0)),
        "fsq_eval_n_action_steps": int(get_value(cfg, "fsq_eval_n_action_steps", 5)),
        "fsq_eval_n_samples":      int(get_value(cfg, "fsq_eval_n_samples", 5)),
        "fsq_eval_max_entries":    int(get_value(cfg, "fsq_eval_max_entries", 0)),
        "fsq_eval_end_threshold":  str(get_value(cfg, "fsq_eval_end_threshold", 0.5)),
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
