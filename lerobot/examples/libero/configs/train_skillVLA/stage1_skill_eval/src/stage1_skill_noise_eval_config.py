#!/usr/bin/env python3
"""Resolve exact-start, repeated-noise Stage-1 skill trajectory evaluation."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from stage1_skill_eval_config import (
    build_settings as _build_comparison_settings,
)

import sys

_HERE = Path(__file__).resolve()
_TRAIN_SKILLS_SRC = _HERE.parent.parent.parent.parent / "train_skills" / "src"
sys.path.insert(0, str(_TRAIN_SKILLS_SRC))

from train_skills_config import as_bool, get_value, load_config, print_shell  # noqa: E402


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
        "neighbor+opposite": "neighbor_and_opposite",
        "neighbors_and_opposite": "neighbor_and_opposite",
    }
    mode = aliases.get(text, text)
    if mode not in {"off", "neighbor", "neighbor_and_opposite", "all"}:
        raise ValueError(
            "neighbor_code_probe must be "
            "off|neighbor|neighbor_and_opposite|all."
        )
    return mode


def _compact_checkpoint(value: object) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("model_defaults.checkpoint must not be empty.")
    if not text.isdigit():
        return text.lower().replace(" ", "_")
    number = int(text)
    if number >= 1000:
        return f"{number / 1000:g}".replace(".", "p") + "k"
    return str(number)


def _rollout_randomization_mode(value: object) -> str:
    """Normalize which stochastic Stage-1 input changes across rollouts."""
    text = str(value or "noise").strip().lower().replace("-", "_")
    aliases = {
        "mode": "latent",
        "mode_latent": "latent",
        "z": "latent",
        "all": "both",
        "noise_and_latent": "both",
        "latent_and_noise": "both",
    }
    mode = aliases.get(text, text)
    if mode not in {"noise", "latent", "both"}:
        raise ValueError("rollout_randomization must be noise|latent|both.")
    return mode


def _probe_suffixed_output_name(
    output_name: str,
    checkpoint_suffix: str,
    probe_mode: str,
    skill_only_rollout_probe: bool = False,
    rollout_randomization: str = "noise",
) -> str:
    """Expose checkpoint, code probe, and stochastic-source contracts."""
    base = str(output_name).strip()
    if not base:
        raise ValueError("output_name must not be empty.")
    randomization = _rollout_randomization_mode(rollout_randomization)
    suffix = f"_{checkpoint_suffix}_{probe_mode}_rand{randomization}"
    if skill_only_rollout_probe:
        suffix += "_skillonly"
    return f"{base}{suffix}"


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
    skill_only_rollout_probe = as_bool(
        get_value(config, "skill_only_rollout_probe", False)
    )
    rollout_randomization = _rollout_randomization_mode(
        get_value(config, "rollout_randomization", "noise")
    )

    settings = _build_comparison_settings(resolved_config)
    if skill_only_rollout_probe:
        supported = {
            "arch0_skill",
            "arch0_skill_chunk",
            "arch0_2_skill_chunk",
        }
        unsupported = [
            f"{model['label']} ({model.get('architecture_label', '')})"
            for model in json.loads(settings["models_json"])
            if str(model.get("architecture_label", "")) not in supported
        ]
        if unsupported:
            raise ValueError(
                "skill_only_rollout_probe requires a trained auxiliary route; "
                "unsupported model(s): " + ", ".join(unsupported)
            )
    work_units = int(settings["eval_work_unit_count"]) * noise_rollouts
    requested_gpus = int(get_value(config, "eval_num_gpus", 1))
    model_defaults = config.get("model_defaults", {})
    if not isinstance(model_defaults, dict):
        raise ValueError("model_defaults must be a mapping.")
    output_name = _probe_suffixed_output_name(
        Path(settings["eval_out_dir"]).name,
        _compact_checkpoint(model_defaults.get("checkpoint", "")),
        code_probe_mode,
        skill_only_rollout_probe,
        rollout_randomization,
    )
    settings.update(
        {
            "envs_per_task": envs_per_task,
            "noise_rollouts_per_env": noise_rollouts,
            "neighbor_code_probe": code_probe_mode,
            "skill_only_rollout_probe": skill_only_rollout_probe,
            "rollout_randomization": rollout_randomization,
            "trajectory_stride": trajectory_stride,
            "eval_work_unit_count": work_units,
            "eval_num_gpus": min(requested_gpus, work_units),
            "eval_out_dir": (
                Path(settings["stage1_skill_eval_dir"])
                / "noise_outputs"
                / output_name
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
