#!/usr/bin/env python3
"""Use the existing LIBERO v3 video gallery with MimicGen-local settings and output."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
VISUALIZED_DIR = HERE.parent
ROOT = next(parent for parent in HERE.parents if (parent / 'lerobot/src/lerobot').is_dir())
SHARED = ROOT / 'lerobot/skillVLA/libero/configs/generate_training_dataset/visualized_dataset'


def shared_visualizer():
    # Load the shared implementation, while keeping all mutable paths/settings
    # local to MimicGen. No LIBERO YAML or generated reports are changed.
    spec = importlib.util.spec_from_file_location('_mimicgen_video_gallery', SHARED / 'visualize_training_dataset.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    settings_module = sys.modules[module.dataset_settings.__module__]
    settings_module.VISUALIZED_DIR = VISUALIZED_DIR
    settings_module.DEFAULT_CONFIG_PATH = VISUALIZED_DIR / 'visualized_dataset_config.yaml'
    module.VISUALIZED_DIR = VISUALIZED_DIR
    module.PREVIEW_ROOT = VISUALIZED_DIR / 'outputs'
    return module, settings_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--list-tasks', action='store_true')
    args = parser.parse_args()
    module, settings = shared_visualizer()
    config = settings.load_config(VISUALIZED_DIR / 'visualized_dataset_config.yaml')
    local = config.setdefault('visualize', {})
    if local.get('ffmpeg', 'auto') == 'auto':
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg is None:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        local['ffmpeg'] = ffmpeg
    if args.list_tasks:
        local['list_tasks_only'] = True
    dataset = settings.dataset_settings(config)
    visual = settings.visualization_settings(config)
    if args.dry_run:
        print(f'Dataset: {dataset.dataset_dir}\nHTML: {visual.output}\nFFmpeg: {visual.ffmpeg}\nCPU only')
        return
    module.load_config = lambda: config
    sys.argv = [sys.argv[0]]
    module.main()


if __name__ == '__main__':
    main()
