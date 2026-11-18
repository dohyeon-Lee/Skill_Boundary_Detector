#!/usr/bin/env python3
"""ABC entrypoint for the shared LIBERO/ABC Stage-1 resolver."""

from __future__ import annotations

import runpy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[7]
SHARED = (
    PROJECT_ROOT
    / "lerobot/skillVLA/libero/configs/train_skillVLA/PT/src/stage1_train_config.py"
)
runpy.run_path(str(SHARED), run_name="__main__")

