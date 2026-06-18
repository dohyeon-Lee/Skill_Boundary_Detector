#!/usr/bin/env python3
"""Config + path resolution for the STATE-influence probe / pi05-vs-cond comparison (state_probe.py).

Resolves two checkpoints to compare on the SAME skillvla dataset:
  * cond : a SkillVLA Stage-1 skill_expert run (1-token state) — model_dir parsed like stage1_eval.
  * pi05 : a plain pi05 run trained on the same data (state discretized into the prompt).
The shared dataset (and the cond checkpoint) come from the cond model_dir's {source}/{run_tag}.
OFFLINE — only needs the checkpoints + skillvla dataset. Emits shell exports (--shell).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[4] / "train_skills" / "src"))
from train_skills_config import as_list, get_value, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parents[1] / "state_probe_config.yaml"


def build_settings(cfg: dict) -> dict:
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    dataset_root = project_root / str(get_value(cfg, "dataset_root", "dataset"))
    skillvla_root = dataset_root / str(get_value(cfg, "skillvla_dataset_root", "skillvla_dataset"))
    outputs_root = project_root / str(get_value(cfg, "outputs_root", "outputs"))

    # ── cond (SkillVLA stage-1 skill_expert): model_dir → run_tag/source → dataset + checkpoint ──
    cond_model_dir = str(get_value(cfg, "cond_model_dir"))
    _rt = re.search(r"(FSQ\d+_dino\d+.*?_(?:\d+|best))_(?:[a-zA-Z][^_]*_)*batch\d+", cond_model_dir)
    if not _rt:
        raise ValueError(f"cond_model_dir must embed a 'FSQ..._dino..._<ckpt>_batch<N>' run tag, got: {cond_model_dir}")
    run_tag = _rt.group(1)
    source_dataset = cond_model_dir[: _rt.start()].rstrip("_")
    run_dir = skillvla_root / source_dataset / run_tag
    cond_ckpt = str(get_value(cfg, "cond_checkpoint", "last"))
    cond_policy_path = outputs_root / "skillVLA_stage1" / cond_model_dir / "checkpoints" / cond_ckpt / "pretrained_model"

    # ── pi05 (plain): pi05_model_dir under {outputs_root}/pi05_PT/ ──
    pi05_model_dir = str(get_value(cfg, "pi05_model_dir", "")).strip()
    pi05_ckpt = str(get_value(cfg, "pi05_checkpoint", "last"))
    pi05_subdir = str(get_value(cfg, "pi05_outputs_subdir", "pi05_PT"))
    pi05_policy_path = ""
    if pi05_model_dir:
        pi05_policy_path = outputs_root / pi05_subdir / pi05_model_dir / "checkpoints" / pi05_ckpt / "pretrained_model"

    tag = f"{cond_model_dir}_{cond_ckpt}"
    if pi05_model_dir:
        tag = f"{tag}__vs__pi05_{pi05_ckpt}"
    out_dir = _HERE.parents[1] / "outputs" / tag

    settings: dict = {
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        "cond_policy_path": cond_policy_path,
        "pi05_policy_path": pi05_policy_path,           # "" → cond-only probe
        "dataset_dir": run_dir / "skillvla",            # shared (cond's skillvla; pi05 ignores skill cols)
        "output_dir": out_dir,
        "n_frames": int(get_value(cfg, "n_frames", 512)),
        "batch_size": int(get_value(cfg, "batch_size", 32)),
        "gauss_sigmas": str(get_value(cfg, "gauss_sigmas", "0.5,1.0,2.0")),
        "seed": int(get_value(cfg, "seed", 1000)),
    }

    part = ",".join(as_list(get_value(cfg, "train_partition", ["debug"]))) or "debug"
    excl = ",".join(as_list(get_value(cfg, "train_exclude_nodes", [])))
    settings.update({
        "eval_partition": part,
        "eval_qos": str(get_value(cfg, "train_qos", "base_qos")),
        "eval_gres": str(get_value(cfg, "eval_gres", "gpu:1")),
        "eval_cpus_per_task": int(get_value(cfg, "eval_cpus_per_task", 8)),
        "eval_mem": str(get_value(cfg, "eval_mem", "48G")),
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
