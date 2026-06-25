#!/usr/bin/env python3
"""Config for the Stage-2 VLM attention heatmap (attn_viz).

Pick a trained Stage-2 run by its OUTPUT FOLDER NAME + checkpoint (under
{project_root}/{outputs_root}/skillVLA_stage2/). The skillvla dataset (frames + task language) is
derived from the checkpoint config's fsq_path. Emits shell exports (--shell).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import get_value, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "attn_viz_config.yaml"


def build_settings(cfg: dict) -> dict:
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    outputs_root = project_root / str(get_value(cfg, "outputs_root", "outputs"))
    vla_root = outputs_root / "skillVLA_stage2"
    model_dir = str(get_value(cfg, "model_dir"))
    checkpoint = str(get_value(cfg, "checkpoint", "last"))
    policy_path = vla_root / model_dir / "checkpoints" / checkpoint / "pretrained_model"

    # skillvla dataset (frames + task language) from the checkpoint config's fsq_path.
    pol: dict = {}
    cfg_json = policy_path / "config.json"
    if cfg_json.is_file():
        pol = json.loads(cfg_json.read_text())
    base_fsq = str(pol.get("fsq_path") or "")
    dataset_dir = str(Path(base_fsq).parent / "skillvla") if base_fsq else ""

    # Tag the output by the heatmap knobs so different layers / weighting don't overwrite each other.
    attn_layers = str(get_value(cfg, "attn_layers", "last"))
    attn_weighting = str(get_value(cfg, "attn_weighting", "value"))
    layer_tag = attn_layers.replace(",", "-")
    output_dir = _HERE.parent.parent / "outputs" / f"{model_dir}_{checkpoint}_L{layer_tag}_{attn_weighting}"

    settings: dict = {
        "project_root": project_root,
        "lerobot_root": project_root / "lerobot",
        "policy_path": policy_path,
        "dataset_dir": dataset_dir,
        "output_dir": output_dir,
        "n_samples": int(get_value(cfg, "n_samples", 8)),
        "attn_layers": attn_layers,
        "attn_weighting": attn_weighting,
        "seed": int(get_value(cfg, "seed", 1000)),
        # slurm
        "eval_gres": str(get_value(cfg, "eval_gres", "gpu:1")),
        "eval_cpus_per_task": int(get_value(cfg, "eval_cpus_per_task", 8)),
        "eval_mem": str(get_value(cfg, "eval_mem", "32G")),
        "eval_time": str(get_value(cfg, "eval_time", "1:00:00")),
        "eval_partition": ",".join(str(x) for x in (get_value(cfg, "train_partition", ["big"]) or ["big"]))
        if isinstance(get_value(cfg, "train_partition", ["big"]), list) else str(get_value(cfg, "train_partition", "big")),
        "eval_qos": str(get_value(cfg, "train_qos", "big_qos")),
        "eval_nodelist": str(get_value(cfg, "train_nodelist", "")),
        "eval_exclude_nodes": ",".join(str(x) for x in (get_value(cfg, "train_exclude_nodes", []) or []))
        if isinstance(get_value(cfg, "train_exclude_nodes", []), list) else str(get_value(cfg, "train_exclude_nodes", "")),
    }
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
