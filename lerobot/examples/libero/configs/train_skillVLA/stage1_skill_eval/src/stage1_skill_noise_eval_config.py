#!/usr/bin/env python3
"""Resolve exact-start, repeated-noise Stage-1 skill trajectory evaluation."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from stage1_skill_eval_config import (
    build_settings as _build_comparison_settings,
)

import sys

_HERE = Path(__file__).resolve()
_TRAIN_SKILLS_SRC = _HERE.parent.parent.parent.parent / "train_skills" / "src"
sys.path.insert(0, str(_TRAIN_SKILLS_SRC))

from train_skills_config import get_value, load_config, print_shell  # noqa: E402


DEFAULT_CONFIG_PATH = _HERE.parent.parent / "stage1_skill_noise_eval_config.yaml"


def _code_probe_mode(value) -> str:
    if isinstance(value, bool):
        return "neighbor" if value else "off"
    text = str(value).strip().lower()
    aliases = {
        "true": "neighbor",
        "false": "off",
        "none": "off",
        "assigned": "off",
        "neighbors": "neighbor",
    }
    mode = aliases.get(text, text)
    if mode not in {"off", "neighbor", "all"}:
        raise ValueError("neighbor_code_probe must be off|neighbor|all.")
    return mode


def build_settings(config: dict) -> dict:
    """Reuse the production model resolver while replacing the rollout plan."""
    resolved_config = deepcopy(config)
    envs_per_task = int(
        get_value(
            config,
            "envs_per_task",
            get_value(config, "episodes_per_task", 1),
        )
    )
    if envs_per_task <= 0:
        raise ValueError("envs_per_task must be positive.")
    if "episodes_per_task" in config and int(config["episodes_per_task"]) != envs_per_task:
        raise ValueError(
            "Set only envs_per_task for noise evaluation, or make "
            "episodes_per_task identical."
        )
    resolved_config["episodes_per_task"] = envs_per_task

    noise_rollouts = int(get_value(config, "noise_rollouts_per_env", 50))
    if noise_rollouts <= 0:
        raise ValueError("noise_rollouts_per_env must be positive.")
    trajectory_stride = int(get_value(config, "trajectory_stride", 1))
    if trajectory_stride <= 0:
        raise ValueError("trajectory_stride must be positive.")
    code_probe_mode = _code_probe_mode(
        get_value(config, "neighbor_code_probe", "off")
    )

    settings = _build_comparison_settings(resolved_config)
    work_units = int(settings["eval_work_unit_count"]) * noise_rollouts
    requested_gpus = int(get_value(config, "eval_num_gpus", 1))
    settings.update(
        {
            "envs_per_task": envs_per_task,
            "noise_rollouts_per_env": noise_rollouts,
            "neighbor_code_probe": code_probe_mode,
            "trajectory_stride": trajectory_stride,
            "eval_work_unit_count": work_units,
            "eval_num_gpus": min(requested_gpus, work_units),
            "eval_out_dir": (
                Path(settings["stage1_skill_eval_dir"])
                / "noise_outputs"
                / Path(settings["eval_out_dir"]).name
            ),
        }
    )
    return settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    settings = build_settings(load_config(args.config))
    if args.shell:
        print_shell(settings)
    else:
        for key, value in settings.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
