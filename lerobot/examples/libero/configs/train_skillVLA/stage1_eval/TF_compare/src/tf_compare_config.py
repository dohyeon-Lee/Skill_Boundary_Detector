#!/usr/bin/env python3
"""Config + path resolution for the Stage-1 teacher-forced COMPARISON across many models.

Mirrors build_data_eval/FSQ_utilized_eval: a yaml `targets` list (name → {model_dir, checkpoint})
is evaluated one model at a time with the existing offline teacher_forced.py, results are cached
to outputs/{name}/metrics.json, and report.py renders a single comparison summary.md.

Each model_dir self-identifies the run ({source}_{run_tag}_batch{N}_{A|B}[_exp]); the skillvla
dataset (GT skill codes) and the checkpoint path are parsed from it — same logic as
stage1_eval/src/stage1_eval_config.py. Teacher-forced eval is OFFLINE: it only needs the policy
checkpoint + the skillvla dataset, no FSQ terminator / simulator.

Emits shell exports (--shell) for the submit/run scripts.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[4] / "train_skills" / "src"))
from train_skills_config import as_list, get_value, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parents[1] / "tf_compare_config.yaml"
OUTPUTS_DIR = _HERE.parents[1] / "outputs"

# Run-tag regex shared with stage1_eval: FSQ..._dino..._<ckpt> then optional vis tokens then _batch<N>.
_RUN_TAG_RE = re.compile(r"(FSQ\d+_dino\d+.*?_(?:\d+|best))_(?:[a-zA-Z][^_]*_)*batch\d+")


def load(config_path: str | Path | None = None) -> dict:
    return load_config(config_path or DEFAULT_CONFIG_PATH)


def resolve_paths(cfg: dict, spec: dict) -> tuple[Path, Path]:
    """(policy_path, skillvla_dataset_dir) for one target, parsed from its model_dir.

    A target may override dataset_root / outputs_root (and skillvla_dataset_root) to reach models
    that live under a different root than the global config — e.g. prev-server runs under
    dataset/ + outputs/ while the current global is dataset_filtered/ + outputs_filtered/."""
    model_dir = str(spec["model_dir"])
    checkpoint = str(spec.get("checkpoint", "last"))
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    dataset_root = project_root / str(spec.get("dataset_root") or get_value(cfg, "dataset_root", "dataset"))
    outputs_root = project_root / str(spec.get("outputs_root") or get_value(cfg, "outputs_root", "outputs"))
    skillvla_root = dataset_root / str(
        spec.get("skillvla_dataset_root") or get_value(cfg, "skillvla_dataset_root", "skillvla_dataset"))
    vla_root = outputs_root / "skillVLA_stage1"

    m = _RUN_TAG_RE.search(model_dir)
    if not m:
        raise ValueError(f"model_dir must embed a 'FSQ..._dino..._<ckpt>_batch<N>' run tag, got: {model_dir}")
    run_tag = m.group(1)
    source_dataset = model_dir[: m.start()].rstrip("_")
    run_dir = skillvla_root / source_dataset / run_tag

    policy_path = vla_root / model_dir / "checkpoints" / checkpoint / "pretrained_model"
    dataset_dir = run_dir / "skillvla"
    return policy_path, dataset_dir


def iter_targets(cfg: dict):
    """Yield (name, spec_dict) for each target, preserving yaml order."""
    targets = get_value(cfg, "targets", {}) or {}
    for name, spec in targets.items():
        yield str(name), dict(spec)


def build_settings(cfg: dict) -> dict:
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    part = ",".join(as_list(get_value(cfg, "train_partition", ["debug"]))) or "debug"
    excl = ",".join(as_list(get_value(cfg, "train_exclude_nodes", [])))
    return {
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        "eval_partition": part,
        "eval_qos": str(get_value(cfg, "train_qos", "base_qos")),
        "eval_gres": str(get_value(cfg, "eval_gres", "gpu:1")),
        "eval_cpus_per_task": int(get_value(cfg, "eval_cpus_per_task", 8)),
        "eval_mem": str(get_value(cfg, "eval_mem", "32G")),
        "eval_time": str(get_value(cfg, "eval_time", "4:00:00")),
        "eval_nodelist": str(get_value(cfg, "train_nodelist", "")),
        "eval_exclude_nodes": excl,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--shell", action="store_true")
    args = ap.parse_args()
    settings = build_settings(load(args.config))
    if args.shell:
        print_shell(settings)
    else:
        for k, v in settings.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
