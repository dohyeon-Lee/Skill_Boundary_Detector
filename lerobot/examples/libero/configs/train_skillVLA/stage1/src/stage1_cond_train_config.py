#!/usr/bin/env python3
"""Compatibility shim for already-submitted legacy Cond-Gemma Slurm jobs.

New runs must use stage1_train_config.py and the single stage1_train_config.yaml.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from stage1_train_config import (
    DEFAULT_CONFIG_PATH,
    build_settings as build_unified_settings,
    load_config,
    print_shell,
)

_LEGACY_ROUTES = {
    "state_cond": ("state_cond", "state_cond"),
    "viso_stateo_skillo": ("state_skill_cond", "VisO_StateO_SkillO"),
    "visx_stateo_skillo": ("state_skill_only_cond", "VisX_StateO_SkillO"),
    "viso_stateo_skillx": ("stateonly_cond", "VisO_StateO_SkillX"),
    "viso_statex_skillo": ("skillonly_cond", "VisO_StateX_SkillO"),
    "viso_statex_skillx": ("visiononly_cond", "VisO_StateX_SkillX"),
}


def _translate_legacy_cond_config(
    config: dict,
) -> tuple[dict, float | None, bool, tuple[str, str] | None]:
    translated = deepcopy(config)
    architecture = translated.setdefault("architecture", {})
    if not isinstance(architecture, dict):
        raise ValueError("architecture must be a mapping.")
    architecture["name"] = "arch0"
    nested_arch1 = architecture.pop("arch1", {})
    if not isinstance(nested_arch1, dict):
        raise ValueError("architecture.arch1 must be a mapping when present.")
    legacy_config = any(
        key in architecture for key in ("revision", "cond_variant", "conditioning_route")
    ) or bool(nested_arch1)
    cond_variant = architecture.pop(
        "cond_variant", nested_arch1.get("cond_variant", architecture.get("expert_variant"))
    )
    if legacy_config and cond_variant != architecture.get("expert_variant", "gemma_300m"):
        raise ValueError("Legacy Arch1 requires matching cond/expert variants.")
    route_value = str(
        architecture.pop(
            "conditioning_route",
            nested_arch1.get("conditioning_route", "VisO_StateO_SkillO"),
        )
    ).strip().lower()
    architecture.pop("revision", None)
    legacy_route = None
    if legacy_config:
        if route_value not in _LEGACY_ROUTES:
            raise ValueError(f"Unsupported legacy Arch1 conditioning route: {route_value!r}.")
        legacy_route = _LEGACY_ROUTES[route_value]

    optimizer = translated.setdefault("training", {}).setdefault("optimizer", {})
    had_legacy_dino_key = "dino_lr" in optimizer
    legacy_dino_lr = optimizer.pop("dino_lr", None)
    base_lr = float(optimizer.get("base_lr", 2.5e-5))
    num_gpus = int(
        translated.get("training", {}).get("dataloader", {}).get("gpus", 1)
    )
    policy_lr = base_lr * num_gpus
    if had_legacy_dino_key:
        optimizer["dino_lr_scale"] = (
            float(legacy_dino_lr) / policy_lr
            if legacy_dino_lr not in (None, "", "null")
            else 1.0
        )
    freeze_vision = bool(translated.setdefault("vision", {}).get("freeze", False))
    return translated, legacy_dino_lr, freeze_vision, legacy_route


def build_settings(config: dict) -> dict:
    translated, legacy_dino_lr, freeze_vision, legacy_route = (
        _translate_legacy_cond_config(config)
    )
    settings = build_unified_settings(translated)
    # Preserve the exact metadata/output contract of already-submitted legacy
    # Cond jobs even though new runs call the same revision Arch0.
    settings["architecture_label"] = "arch1"
    if settings["pt_run_name"].endswith("_arch0"):
        settings["pt_run_name"] = settings["pt_run_name"][:-6] + "_arch1"
        settings["pt_output_dir"] = (
            Path(settings["pt_output_dir"]).parent / settings["pt_run_name"]
        )
    if legacy_route is not None:
        internal_route, route_label = legacy_route
        settings["conditioning_route"] = internal_route
        settings["pt_run_name"] = f"{settings['pt_run_name']}_{route_label}"
        settings["pt_output_dir"] = (
            Path(settings["pt_output_dir"]).parent / settings["pt_run_name"]
        )
    # train_cond.sbatch expects the historical absolute DINO_LR export. Supplying
    # the policy LR when the old value was null reproduces its former base group.
    if freeze_vision:
        settings["dino_lr"] = ""
    elif legacy_dino_lr not in (None, "", "null"):
        settings["dino_lr"] = str(float(legacy_dino_lr))
    else:
        settings["dino_lr"] = str(settings["lr"] * settings["dino_lr_scale"])
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
