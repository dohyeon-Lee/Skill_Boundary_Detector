#!/usr/bin/env python3
"""ABC SkillVLA entry point; reuse native simulator rollout and success metrics."""
from pathlib import Path
import sys

SHARED_SRC = Path(__file__).resolve().parents[4] / "train_pi05/PT/eval/src"
sys.path.insert(0, str(SHARED_SRC))
from run_eval import main

if __name__ == "__main__":
    main()
