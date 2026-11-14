#!/usr/bin/env python3
"""ABC wrapper around the maintained shared SBD evaluator configuration."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def _shared_module():
    path = next(
        parent
        / "libero/configs/train_skills/skill_eval/skill_boundary_detection/src/config.py"
        for parent in Path(__file__).resolve().parents
        if (
            parent
            / "libero/configs/train_skills/skill_eval/skill_boundary_detection/src/config.py"
        ).is_file()
    )
    spec = importlib.util.spec_from_file_location("sbd_shared_boundary_eval_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared SBD evaluator config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_settings(config_path: str) -> dict:
    shared = _shared_module()
    return shared.build_settings(config_path, analysis_domain="abc")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    shared = _shared_module()
    settings = build_settings(args.config)
    if args.shell:
        shared.print_shell(settings)
    else:
        print(json.dumps(settings, indent=2))


if __name__ == "__main__":
    main()
