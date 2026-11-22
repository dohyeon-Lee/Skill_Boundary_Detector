#!/usr/bin/env python3
"""LIBERO's compact DP evaluation config, using the shared evaluation workflow."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def workflow_dir():
    return next(parent / "abc/configs/train_skills/DP/eval/src"
                for parent in Path(__file__).resolve().parents
                if (parent / "abc/configs/train_skills/DP/eval/src").is_dir())


def load_workflow(name):
    spec = importlib.util.spec_from_file_location(f"libero_dp_eval_{name}", workflow_dir() / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_settings(config_path):
    return load_workflow("config").build_settings(
        str(config_path), analysis_domain="libero", eval_dir_override=Path(__file__).resolve().parents[1],
    )


def load_config(config_path):
    return load_workflow("config")._shared_module().load_config(config_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    settings = build_settings(args.config)
    if args.shell:
        load_workflow("config")._shared_module().print_shell(settings)
    else:
        print(json.dumps(settings, indent=2))


if __name__ == "__main__":
    main()
