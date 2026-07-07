#!/usr/bin/env python3
"""FT entry point for the per-component parameter-drift plot — a thin shim that runs the canonical
implementation (stage2_eval/src/plot_param_drift.py) with --stage ft. Outputs land under
FT_eval/outputs/update/{model_dir}/. See that file's docstring for details and flags.

  plot_param_drift.py --all [--wandb]
  plot_param_drift.py --model_dir <FT folder name> [--model_dir <another> ...] [--wandb]
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_TARGET = Path(__file__).resolve().parent.parent.parent / "stage2_eval" / "src" / "plot_param_drift.py"

if __name__ == "__main__":
    # Default to --stage ft; a user-supplied --stage still wins (argparse takes the last occurrence).
    sys.argv = [str(_TARGET), "--stage", "ft", *sys.argv[1:]]
    runpy.run_path(str(_TARGET), run_name="__main__")
