"""Delegate GPU packing to the shared implementation."""
from pathlib import Path
import runpy

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'lerobot/src/lerobot').is_dir())
if __name__ == '__main__':
    runpy.run_path(str(ROOT / 'lerobot/skillVLA/libero/configs/src/eval_gpu_packing.py'), run_name='__main__')
