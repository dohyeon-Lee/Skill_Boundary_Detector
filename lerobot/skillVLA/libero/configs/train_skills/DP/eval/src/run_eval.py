#!/usr/bin/env python3
"""Run the same cache/build/report workflow as ABC with LIBERO's config/source."""
from pathlib import Path
import runpy
import sys

from config import workflow_dir

if __name__ == "__main__":
    sys.path.append(str(workflow_dir()))
    runpy.run_path(str(workflow_dir() / "run_eval.py"), run_name="__main__")
