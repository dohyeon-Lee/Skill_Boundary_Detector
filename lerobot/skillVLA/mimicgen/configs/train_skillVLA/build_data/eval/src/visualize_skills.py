#!/usr/bin/env python3
"""Inspect Coffee build boundaries with the shared LIBERO HTML renderer."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'lerobot/src/lerobot').is_dir())
BUILD = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BUILD / 'src'))
from build_dataset import load_config, resolve  # noqa: E402


def build_settings(config_path):
    paths = resolve(load_config(config_path))
    return {'raw_dataset_dir': paths['raw'], 'skillvla_run_dir': paths['output']}


if __name__ == '__main__':
    source = ROOT / 'lerobot/skillVLA/libero/configs/train_skillVLA/build_data/eval/src/visualize_skills.py'
    spec = importlib.util.spec_from_file_location('shared_build_visualization', source)
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    shared.main(settings_loader=build_settings, default_config=BUILD / 'eval/visualize_config.yaml')
