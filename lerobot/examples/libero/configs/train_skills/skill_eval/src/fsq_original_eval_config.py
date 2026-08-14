#!/usr/bin/env python3
"""Resolve FSQ-original evaluation settings.

--shell           : export global paths, run/checkpoint lists, and Slurm knobs.
--resolve-skills  : print the skills directory recorded in a run's
                    fsq_original_meta.json (resolved against dataset_root).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "src"))

from train_skills_config import (  # noqa: E402
    as_list,
    get_value,
    load_config,
    print_shell,
)

DEFAULT_CONFIG_PATH = _HERE.parent / "fsq_original_eval_config.yaml"


def _checkpoint_tags(cfg: dict) -> list[str]:
    tags = []
    for value in as_list(get_value(cfg, "fsq_orig_eval_checkpoint", "200")):
        text = str(value).strip()
        if not text:
            continue
        tags.append(f"epoch{int(text):04d}" if text.isdigit() else text)
    if not tags:
        raise ValueError("fsq_orig_eval_checkpoint must name at least one checkpoint.")
    if len(tags) != len(set(tags)):
        raise ValueError(f"fsq_orig_eval_checkpoint contains duplicates: {tags}.")
    return tags


def resolve(cfg: dict) -> dict:
    root = Path(str(get_value(cfg, "project_root"))).expanduser()
    outputs_root = root / str(get_value(cfg, "outputs_root", "outputs"))
    runs = [r for r in as_list(get_value(cfg, "fsq_orig_eval_run_name", "")) if r]
    if not runs:
        raise ValueError("fsq_orig_eval_run_name must name at least one FSQ run.")
    partitions = as_list(get_value(cfg, "train_partition", ["debug"]))
    return {
        "project_root": root,
        "lerobot_root": root / "lerobot",
        "fsq_outputs_root": outputs_root / "FSQ",
        "fsq_orig_eval_runs": ",".join(runs),
        "fsq_orig_eval_checkpoints": " ".join(_checkpoint_tags(cfg)),
        "fsq_orig_eval_term_threshold": str(get_value(cfg, "fsq_orig_eval_term_threshold", 0.5)),
        "fsq_orig_eval_val_split": str(get_value(cfg, "fsq_orig_eval_val_split", 0.1)),
        "eval_partition": ",".join(partitions) or "debug",
        "eval_qos": str(get_value(cfg, "train_qos", "base_qos")),
        "eval_nodelist": str(get_value(cfg, "train_nodelist", "")),
        "eval_exclude_nodes": as_list(get_value(cfg, "train_exclude_nodes", [])),
        "eval_gres": str(get_value(cfg, "eval_gres", "gpu:1")),
        "eval_cpus": int(get_value(cfg, "eval_cpus", 8)),
        "eval_mem": str(get_value(cfg, "eval_mem", "32G")),
        "eval_time": str(get_value(cfg, "eval_time", "1:00:00")),
    }


def resolve_skills(cfg: dict, run_dir: Path) -> Path:
    meta_path = Path(run_dir) / "fsq_original_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Not an FSQ-original run (fsq_original_meta.json missing): {run_dir}"
        )
    meta = json.loads(meta_path.read_text())
    root = Path(str(get_value(cfg, "project_root"))).expanduser()
    dataset_root = root / str(get_value(cfg, "dataset_root", "libero_dataset"))
    skills = (
        dataset_root
        / str(meta["fsq_dataset_root"])
        / str(meta["target_dataset"])
        / str(meta["fsq_inputs_name"])
        / str(meta["skillset_seg_name"])
        / str(meta["skillset_name"])
        / "skills"
    )
    if not skills.is_dir():
        raise FileNotFoundError(f"Skills dir from {meta_path} not found: {skills}")
    return skills


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--shell", action="store_true")
    parser.add_argument("--resolve-skills", type=Path, default=None, metavar="RUN_DIR")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.resolve_skills is not None:
        print(resolve_skills(cfg, args.resolve_skills))
        return
    settings = resolve(cfg)
    if args.shell:
        print_shell(settings)
        return
    for key, value in settings.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
