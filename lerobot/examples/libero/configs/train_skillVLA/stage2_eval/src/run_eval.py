#!/usr/bin/env python
"""Stage-2 entry point for the shared skill-conditioned LIBERO evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

_COMMON_SRC = Path(__file__).resolve().parents[2] / "stage1_eval" / "src"
sys.path.insert(0, str(_COMMON_SRC))

from run_eval import eval_main  # noqa: E402


if __name__ == "__main__":
    eval_main()
