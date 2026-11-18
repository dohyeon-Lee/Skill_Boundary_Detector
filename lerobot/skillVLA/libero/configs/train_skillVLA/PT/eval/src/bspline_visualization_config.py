#!/usr/bin/env python3
"""Read the shared eval YAML without editing it; visualize one episode/task."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import stage1_eval_config as base


def visualization_config(config: dict) -> dict:
    config = deepcopy(config)
    config.update(n_episodes=1, eval_batch_size=1, max_parallel_tasks=1,
                  skill_html_train_samples=0)
    # GPU count and per-GPU worker packing follow the shared eval YAML.
    name = str(config.get("output_name", "") or "bspline")
    config["output_name"] = name + "_bspline_visualization"
    config["motion_plan"] = {**(config.get("motion_plan") or {}), "enabled": True,
                             "independent_base": False}
    config["video"] = {**(config.get("video") or {}), "max_per_task": 1, "skill_videos": False}
    # Do not publish single-episode debugging numbers as an SR experiment.
    config["logging"] = {**(config.get("logging") or {})}
    config["logging"]["wandb"] = {**(config["logging"].get("wandb") or {}), "enable": False}
    return config


def build_settings(config: dict) -> dict:
    settings = base.build_settings(visualization_config(config))
    models = json.loads(settings["models_json"])
    unsupported = [model["label"] for model in models
                   if not model["architecture_label"].startswith("fullskill_bspline_")]
    if unsupported:
        raise ValueError(f"Bspline visualization requires Bspline checkpoints; unsupported panels: {unsupported}.")
    return settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=base.DEFAULT_CONFIG_PATH)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    settings = build_settings(base.load_config(args.config))
    if args.shell:
        base.print_shell(settings)
    else:
        for key, value in settings.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
