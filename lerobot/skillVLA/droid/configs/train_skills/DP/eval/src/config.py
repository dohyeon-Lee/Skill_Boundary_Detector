#!/usr/bin/env python3
"""DROID adapter for the maintained LIBERO/ABC SBD evaluation workflow."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def workflow_dir():
    return next(parent / 'abc/configs/train_skills/DP/eval/src'
                for parent in Path(__file__).resolve().parents
                if (parent / 'abc/configs/train_skills/DP/eval/src').is_dir())


def load_workflow(name):
    spec = importlib.util.spec_from_file_location(f'droid_dp_eval_{name}', workflow_dir() / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_settings(config_path):
    return load_workflow('config').build_settings(
        str(config_path), analysis_domain='droid', eval_dir_override=Path(__file__).resolve().parents[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--shell', action='store_true')
    args = parser.parse_args()
    settings = build_settings(args.config)
    if args.shell:
        workflow = load_workflow('config')
        workflow._shared_module().print_shell(workflow.shell_settings(settings))
    else:
        print(json.dumps(settings, indent=2))


if __name__ == '__main__':
    main()
