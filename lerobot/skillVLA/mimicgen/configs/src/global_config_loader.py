"""Common server loader with MimicGen's own storage defaults."""
from pathlib import Path
import os
import runpy
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'lerobot/src/lerobot').is_dir())
_shared = runpy.run_path(str(ROOT / 'lerobot/skillVLA/libero/configs/src/global_config_loader.py'))
load_global_config = _shared['load_global_config']
if __name__ == '__main__':
    if '--config' not in sys.argv:
        sys.argv.extend(['--config', os.environ.get('SBD_GLOBAL_CONFIG', str(Path(__file__).resolve().parents[1] / 'global_config.yaml'))])
    _shared['main']()
