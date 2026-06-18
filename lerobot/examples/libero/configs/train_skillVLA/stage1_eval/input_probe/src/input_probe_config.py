#!/usr/bin/env python3
"""Config + path resolution for the INPUT-influence probe (input_probe.py).

Compares N models' state/skill/progress sensitivity on the SAME skillvla dataset. Each `targets` entry is
{label, kind, model_dir, checkpoint}:
  * kind=cond : a SkillVLA Stage-1 skill_expert run under {outputs_root}/skillVLA_stage1/ (1-token or
                discretized state). model_dir is parsed for {source}/{run_tag} (the shared dataset).
  * kind=pi05 : a plain pi05 run under {outputs_root}/{pi05_outputs_subdir} (state in the prompt).
The shared dataset comes from the FIRST cond target. (Back-compat: if `targets` is absent, builds a
2-entry list from cond_model_dir + pi05_model_dir.) OFFLINE. Emits shell exports (--shell).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[4] / "train_skills" / "src"))
from train_skills_config import as_list, get_value, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parents[1] / "input_probe_config.yaml"
_RUN_TAG_RE = r"(FSQ\d+_dino\d+.*?_(?:\d+|best))_(?:[a-zA-Z][^_]*_)*batch\d+"


def _parse_cond(model_dir: str) -> tuple[str, str]:
    """cond model_dir → (source_dataset, run_tag)."""
    m = re.search(_RUN_TAG_RE, model_dir)
    if not m:
        raise ValueError(f"cond model_dir must embed a 'FSQ..._dino..._<ckpt>_batch<N>' run tag, got: {model_dir}")
    return model_dir[: m.start()].rstrip("_"), m.group(1)


def _targets_from_cfg(cfg: dict) -> list[dict]:
    """The yaml `targets` list, or a 2-entry fallback from cond_model_dir/pi05_model_dir."""
    targets = get_value(cfg, "targets", None)
    if targets:
        return [dict(t) for t in targets]
    out = []
    cm = str(get_value(cfg, "cond_model_dir", "")).strip()
    if cm:
        out.append({"label": "cond", "kind": "cond", "model_dir": cm,
                    "checkpoint": str(get_value(cfg, "cond_checkpoint", "last"))})
    pm = str(get_value(cfg, "pi05_model_dir", "")).strip()
    if pm:
        out.append({"label": "pi05", "kind": "pi05", "model_dir": pm,
                    "checkpoint": str(get_value(cfg, "pi05_checkpoint", "last")),
                    "pi05_outputs_subdir": str(get_value(cfg, "pi05_outputs_subdir", "pi05_PT"))})
    return out


def build_settings(cfg: dict) -> dict:
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    dataset_root = project_root / str(get_value(cfg, "dataset_root", "dataset"))
    skillvla_root = dataset_root / str(get_value(cfg, "skillvla_dataset_root", "skillvla_dataset"))
    outputs_root = project_root / str(get_value(cfg, "outputs_root", "outputs"))

    targets_cfg = _targets_from_cfg(cfg)
    if not targets_cfg:
        raise ValueError("No targets: set a `targets:` list (or cond_model_dir/pi05_model_dir).")

    resolved: list[dict] = []
    dataset_dir = None
    for i, t in enumerate(targets_cfg):
        kind = str(t.get("kind", "cond")).strip().lower()
        model_dir = str(t["model_dir"]).strip()
        ckpt = str(t.get("checkpoint", "last")).strip() or "last"
        label = str(t.get("label") or kind).strip()
        if kind == "pi05":
            subdir = str(t.get("pi05_outputs_subdir", get_value(cfg, "pi05_outputs_subdir", "pi05_PT")))
            path = outputs_root / subdir / model_dir / "checkpoints" / ckpt / "pretrained_model"
        else:  # cond (SkillVLA stage-1 skill_expert)
            source, run_tag = _parse_cond(model_dir)
            path = outputs_root / "skillVLA_stage1" / model_dir / "checkpoints" / ckpt / "pretrained_model"
            if dataset_dir is None:  # shared dataset = the FIRST cond target's run dir
                dataset_dir = skillvla_root / source / run_tag / "skillvla"
        resolved.append({"label": label, "policy_path": str(path)})

    if dataset_dir is None:
        raise ValueError("Need at least one kind=cond target (its source/run_tag gives the shared dataset).")

    tag = "_vs_".join(t["label"] for t in resolved)
    out_dir = _HERE.parents[1] / "outputs" / tag

    settings: dict = {
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        "targets_json": json.dumps(resolved),     # [{label, policy_path}, ...] for input_probe.py --targets
        "dataset_dir": dataset_dir,                # shared skillvla dataset (cond's; pi05 ignores skill cols)
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
