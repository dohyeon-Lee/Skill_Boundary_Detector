#!/usr/bin/env python3
"""Use the shared synchronized comparison player on LIBERO evaluation outputs."""
from pathlib import Path
from config import load_workflow

if __name__ == "__main__":
    module = load_workflow("together_report")
    module.EVAL_DIR = Path(__file__).resolve().parents[1]
    module.main()
