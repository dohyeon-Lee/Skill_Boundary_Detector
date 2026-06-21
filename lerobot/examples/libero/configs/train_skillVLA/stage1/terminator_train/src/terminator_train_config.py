#!/usr/bin/env python3
"""Submit-time resolver for the standalone terminator trainer (emits shell exports via --shell).

ONLY resolves submission plumbing — SLURM args (partition/qos from global_config.yaml), the venv/cd
roots, and the existence-check paths (FSQ.pt / dino.npz / skillvla). The actual hyperparameters live in
terminator_train_config.yaml and are read directly by train_terminator.py (single source of truth)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[4] / "train_skills" / "src"))
from train_skills_config import as_list, get_value, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "terminator_train_config.yaml"


def build_settings(cfg: dict) -> dict:
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    dataset_root = project_root / str(get_value(cfg, "dataset_root", "dataset"))
    skillvla_root = dataset_root / str(get_value(cfg, "skillvla_dataset_root", "skillvla_dataset"))
    source = str(get_value(cfg, "source_dataset"))
    run_tag = str(get_value(cfg, "run_tag"))
    run_dir = skillvla_root / source / run_tag

    excl = ",".join(as_list(get_value(cfg, "train_exclude_nodes", [])))
    return {
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        # existence-check paths (submit_train.sh fails loudly if missing)
        "skillvla_dataset_dir": run_dir / "skillvla",
        "fsq_path": run_dir / "FSQ.pt",
        "dino_path": run_dir / "dino.npz",
        # SLURM (partition/qos from global_config; the rest from this yaml)
        "train_partition": ",".join(as_list(get_value(cfg, "train_partition", ["big"]))) or "big",
        "train_qos": str(get_value(cfg, "train_qos", "big_qos")),
        "train_gres": str(get_value(cfg, "train_gres", "gpu:1")),
        "train_cpus_per_task": int(get_value(cfg, "train_cpus_per_task", 16)),
        "train_mem": str(get_value(cfg, "train_mem", "64G")),
        "train_time": str(get_value(cfg, "train_time", "12:00:00")),
        "train_nodelist": str(get_value(cfg, "train_nodelist", "")),
        "train_exclude_nodes": excl,
    }


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
