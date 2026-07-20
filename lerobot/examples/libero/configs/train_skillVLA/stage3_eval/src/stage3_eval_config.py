#!/usr/bin/env python3
"""Stage-3 closed-loop eval config: predicted skills with Stage-0/2 motor routes."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve()
_BASE_PATH = _HERE.parent.parent.parent / "stage2_eval" / "src" / "stage2_eval_config.py"
_SPEC = importlib.util.spec_from_file_location("skillvla_stage2_eval_config", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load shared eval config: {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "stage3_eval_config.yaml"

_MODE_ALIASES = {
    "a": "a", "full": "a", "connected": "a",
    "b": "b", "severed": "b", "drop_vlm": "b",
}
_TERMINATOR_ALIASES = {
    "auto": "auto",
    "stage3": "cotrained", "cotrained": "cotrained",
    "parent": "base", "stage0": "base", "stage2": "base", "base": "base",
}


def _normalize_modes(raw: object) -> str:
    items = [item.strip().lower() for item in str(raw or "a").split(",") if item.strip()]
    modes = [_MODE_ALIASES.get(item, "") for item in items]
    if not modes or any(not mode for mode in modes) or len(set(modes)) != len(modes):
        raise ValueError(
            "Stage-3 modes must be a comma list without duplicates from a,b "
            f"(aliases: full,connected,severed,drop_vlm), got {raw!r}."
        )
    return ",".join(modes)


def _normalize_terminator(raw: object) -> str:
    value = str(raw or "auto").strip().lower()
    resolved = _TERMINATOR_ALIASES.get(value)
    if resolved is None:
        raise ValueError(
            "terminator_source must be auto|stage3|parent "
            f"(legacy aliases cotrained|base are accepted), got {value!r}."
        )
    return resolved


def _output_name(cfg: dict) -> str:
    name = str(_BASE.get_value(cfg, "output_name", "") or "").strip()
    if not name:
        name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if name in {".", ".."} or "/" in name or "\0" in name:
        raise ValueError(f"output_name must be one folder name, got {name!r}.")
    return name


def _prepare_config(cfg: dict) -> dict:
    prepared = copy.deepcopy(cfg)
    requested_source = str(_BASE.get_value(prepared, "skill_source", "pred") or "pred").lower()
    if requested_source != "pred":
        raise ValueError("Stage-3 eval always uses the trained skill predictor; skill_source must be pred.")

    prepared["models_root"] = "skillVLA_stage3"
    prepared["skill_source"] = "pred"
    prepared["include_stage1"] = False
    prepared["modes"] = _normalize_modes(_BASE.get_value(prepared, "modes", "a"))
    prepared["terminator_source"] = _normalize_terminator(
        _BASE.get_value(prepared, "terminator_source", "auto")
    )

    models = _BASE.get_value(prepared, "models", None)
    if isinstance(models, list):
        for model in models:
            advance = str(_BASE.get_value(model, "advance_mode", "terminator") or "terminator").lower()
            if advance != "terminator":
                raise ValueError("Stage-3 predicted-skill eval requires models[].advance_mode=terminator.")
            model["advance_mode"] = "terminator"
            model["modes"] = _normalize_modes(_BASE.get_value(model, "modes", prepared["modes"]))
            model["terminator_source"] = _normalize_terminator(
                _BASE.get_value(model, "terminator_source", prepared["terminator_source"])
            )
    else:
        prepared["skill_advance_mode"] = "terminator"
    return prepared


def _validate_stage3_checkpoints(settings: dict) -> None:
    paths = [Path(settings["policy_path"])]
    models_json = settings.get("models_json", "")
    if models_json:
        paths = [Path(model["policy_path"]) for model in json.loads(models_json)]
    for policy_path in dict.fromkeys(paths):
        config_path = policy_path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Stage-3 checkpoint config not found: {config_path}")
        policy_cfg = json.loads(config_path.read_text())
        if policy_cfg.get("pt_stage") != "skill":
            raise ValueError(
                f"Expected a Stage-3 checkpoint (pt_stage='skill'), got "
                f"{policy_cfg.get('pt_stage')!r}: {config_path}"
            )


def build_settings(cfg: dict) -> dict:
    prepared = _prepare_config(cfg)
    settings = _BASE.build_settings(prepared)
    _validate_stage3_checkpoints(settings)

    name = _output_name(cfg)
    settings["output_name"] = name
    settings["eval_out_dir"] = _HERE.parent.parent / "outputs" / name
    settings["wandb_project"] = str(_BASE.get_value(cfg, "wandb_project", "VLA_stage3_eval"))
    settings["wandb_run_name"] = name + (
        f"_{os.environ['TASK_TAG']}" if os.environ.get("TASK_TAG") else ""
    )
    return settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    settings = build_settings(_BASE.load_config(args.config))
    if args.shell:
        _BASE.print_shell(settings)
    else:
        for key, value in settings.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
