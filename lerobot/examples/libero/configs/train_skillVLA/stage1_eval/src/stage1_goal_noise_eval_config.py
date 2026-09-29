#!/usr/bin/env python3
"""Expand a normal Stage-1 eval config into paired goal-noise panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import stage1_eval_config as base


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "stage1_goal_noise_eval_config.yaml"


def _as_list(value, *, field: str) -> list:
    if isinstance(value, list):
        return value
    if value is None:
        raise ValueError(f"goal_noise.{field} is required.")
    return [value]


def _number_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def build_settings(config: dict) -> dict:
    settings = base.build_settings(config)
    noise = config.get("goal_noise")
    if not isinstance(noise, dict):
        raise ValueError("goal_noise must be a mapping.")
    distribution = str(noise.get("distribution", "gaussian")).strip().lower()
    if distribution != "gaussian":
        raise ValueError("goal_noise.distribution currently supports only gaussian.")
    stds_mm = [float(value) for value in _as_list(noise.get("std_mm"), field="std_mm")]
    if not stds_mm or any(value < 0 for value in stds_mm):
        raise ValueError("goal_noise.std_mm must contain non-negative values.")
    targets = [str(value).strip().lower() for value in _as_list(noise.get("targets", "end"), field="targets")]
    if any(target not in {"end", "start", "both"} for target in targets):
        raise ValueError("goal_noise.targets entries must be end|start|both.")
    seed = int(noise.get("seed", 0))

    conditions: list[tuple[str, float]] = []
    for std_mm in stds_mm:
        if std_mm == 0:
            if ("none", 0.0) not in conditions:
                conditions.append(("none", 0.0))
        else:
            for target in targets:
                condition = (target, std_mm)
                if condition not in conditions:
                    conditions.append(condition)
    if not conditions:
        raise ValueError("goal_noise produced no evaluation conditions.")

    base_specs = json.loads(settings["models_json"])
    expanded = []
    for target, std_mm in conditions:
        suffix = "clean" if std_mm == 0 else f"{target}_{_number_tag(std_mm)}mm"
        for spec in base_specs:
            clone = dict(spec)
            clone["label"] = f"{spec['label']}__{suffix}"
            clone["goal_noise_target"] = target
            clone["goal_noise_std_m"] = std_mm / 1000.0
            clone["goal_noise_seed"] = seed
            expanded.append(clone)

    settings["models_json"] = json.dumps(expanded, separators=(",", ":"))
    settings["panel_count"] = len(expanded)
    settings["checkpoint_count"] = int(settings["checkpoint_count"]) * len(conditions)
    settings["model_architectures"] = ", ".join(
        f"{spec['label']}={spec['architecture_label']}" for spec in expanded
    )
    settings["goal_noise_conditions"] = ",".join(
        "clean" if std_mm == 0 else f"{target}:{std_mm:g}mm"
        for target, std_mm in conditions
    )
    settings["goal_noise_seed"] = seed
    return settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
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
