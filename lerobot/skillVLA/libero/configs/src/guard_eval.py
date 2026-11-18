#!/usr/bin/env python3
"""Supervise one evaluation group without limiting its workers' allocators."""
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "lerobot/src"))

from lerobot.utils.gpu_memory_guard import Guard, Nvidia


if __name__ == "__main__":
    folder = ROOT / ".cache/gpu_memory_guard" / uuid.uuid4().hex
    raise SystemExit(Guard(["bash", *sys.argv[1:]], os.environ, folder, Nvidia(os.environ)).run())
