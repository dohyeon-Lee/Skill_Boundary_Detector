"""Use the shared server profiles with DROID's own storage defaults."""
from pathlib import Path
import argparse
import json
import os
import runpy
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'lerobot/src/lerobot').is_dir())
_shared = runpy.run_path(str(ROOT / 'lerobot/skillVLA/libero/configs/src/global_config_loader.py'))
_filter = runpy.run_path(str(Path(__file__).with_name('idle_filter.py')))
_groups = runpy.run_path(str(Path(__file__).with_name('dataset_groups.py')))
_datasets = runpy.run_path(str(Path(__file__).with_name('dataset_registry.py')))


def load_global_config(path, server=None, *, resolve_dataset=True):
    config = _shared['load_global_config'](path, server)
    if resolve_dataset:
        config = _datasets['configure'](config)
    # Validate stage spellings and real YAML booleans here, before any training
    # can start. Conversion still ignores these training-only switches.
    config['idle_filter'] = _filter['validate_settings'](config)
    _groups['definition'](config)
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=os.environ.get('SBD_GLOBAL_CONFIG',
                        str(Path(__file__).resolve().parents[1] / 'global_config.yaml')))
    parser.add_argument('--server', default=None)
    parser.add_argument('--key', default=None)
    args = parser.parse_args()
    config = load_global_config(args.config, args.server)
    value = config.get(args.key, '') if args.key else config
    print(json.dumps(value, indent=2, default=str) if isinstance(value, (list, dict)) else value)
