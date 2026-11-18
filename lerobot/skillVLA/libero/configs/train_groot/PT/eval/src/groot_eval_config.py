#!/usr/bin/env python3
"""Resolve GR00T evaluation settings; model imports stay in the isolated venv."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[4] / "src/baseline_eval"))
from config import main

if __name__ == "__main__":
    if "--config" not in sys.argv and not any(a.startswith("--config=") for a in sys.argv):
        sys.argv += ["--config", str(HERE.parents[1] / "groot_eval_config.yaml")]
    if "--work-dir" not in sys.argv:
        sys.argv += ["--work-dir", str(HERE.parents[1])]
    main()
