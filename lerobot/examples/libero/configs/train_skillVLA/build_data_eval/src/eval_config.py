#!/usr/bin/env python3
"""Emit eval-only knobs + slurm for configs/train_skillVLA evals as shell exports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import as_bool, as_list, get_value, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "eval_config.yaml"


def build_settings(config_path: str | None = None) -> dict:
    cfg = load_config(config_path or DEFAULT_CONFIG_PATH)
    # Slurm partition/qos/nodelist/exclude are canonical (global_config.yaml train_*).
    exclude = as_list(get_value(cfg, "train_exclude_nodes", []))
    settings = {
        "eval_run_dino":       str(as_bool(get_value(cfg, "eval_run_dino", True))).lower(),
        "eval_run_skillset":   str(as_bool(get_value(cfg, "eval_run_skillset", True))).lower(),
        "eval_run_fsq_patch":  str(as_bool(get_value(cfg, "eval_run_fsq_patch", True))).lower(),
        "eval_run_fsq_recon":  str(as_bool(get_value(cfg, "eval_run_fsq_recon", True))).lower(),
        "eval_n_episodes":     int(get_value(cfg, "eval_n_episodes", 12)),
        "eval_task_ids":       " ".join(str(int(t)) for t in as_list(get_value(cfg, "eval_task_ids", []))),
        "eval_n_samples":      int(get_value(cfg, "eval_n_samples", 10)),
        "eval_max_entries":    int(get_value(cfg, "eval_max_entries", 0)),
        "eval_n_action_steps": int(get_value(cfg, "eval_n_action_steps", 5)),
        "eval_end_threshold":  str(get_value(cfg, "eval_end_threshold", 0.5)),
        "eval_thumb_size":     int(get_value(cfg, "eval_thumb_size", 160)),
        "eval_seed":           int(get_value(cfg, "eval_seed", 42)),
        "eval_wandb_project":  str(get_value(cfg, "eval_wandb_project", "VAE_eval")),
        "eval_wandb_enable":   str(as_bool(get_value(cfg, "eval_wandb_enable", False))).lower(),
        "eval_partition":      ",".join(as_list(get_value(cfg, "train_partition", ["debug"]))) or "debug",
        "eval_qos":            str(get_value(cfg, "train_qos", "base_qos")),
        "eval_gres":           str(get_value(cfg, "eval_gres", "gpu:1")),
        "eval_cpus_per_task":  int(get_value(cfg, "eval_cpus_per_task", 8)),
        "eval_mem":            str(get_value(cfg, "eval_mem", "64G")),
        "eval_time":           str(get_value(cfg, "eval_time", "04:00:00")),
        "eval_nodelist":       str(get_value(cfg, "train_nodelist", "")),
        "eval_exclude_nodes":  ",".join(exclude),
    }

    # Optional: point the eval at an explicit run folder, independent of the build_data yaml.
    # submit_eval.sh / eval.sbatch source train_skillVLA_config.py FIRST, then this; re-emitting the
    # same path vars here overrides them (later bash `export` wins). Empty → keep the build_data paths.
    #
    # eval_run_dir accepts EITHER:
    #   • a path (absolute, or with a "/" → relative to project_root): used as-is
    #   • a bare run name (no "/", e.g. "FSQ865_dino8_wow_200"): expanded to the standard layout
    #     {dataset_root}/{skillvla_dataset_root}/{source}/{name}, with source = eval_source_dataset
    #     (or "libero_90_full_full" by default).
    run_dir = str(get_value(cfg, "eval_run_dir", "") or "").strip()
    if run_dir:
        project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
        dataset_root = project_root / str(get_value(cfg, "dataset_root", "dataset"))
        src_override = str(get_value(cfg, "eval_source_dataset", "") or "").strip()
        p = Path(run_dir).expanduser()
        if p.is_absolute() or "/" in run_dir:
            rd = p if p.is_absolute() else project_root / p
            source_dataset = src_override or rd.parent.name  # standard layout puts source one level up
        else:
            # bare run name → expand under the standard skillvla layout
            source_dataset = src_override or "libero_90_full_full"
            skillvla_root = str(get_value(cfg, "skillvla_dataset_root", "skillvla_dataset"))
            rd = dataset_root / skillvla_root / source_dataset / run_dir
        settings.update({
            "skillvla_dataset_dir": rd / "skillvla",
            "dino_model_path":      project_root / "models" / "dinov3-vits16",  # ONLINE DINO
            "fsq_copy_path":        rd / "FSQ.pt",
            "raw_dataset_dir":      dataset_root / source_dataset,   # raw video at {dataset_root}/{source}
            "eval_dir":             rd / "eval",
        })

    return settings


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
