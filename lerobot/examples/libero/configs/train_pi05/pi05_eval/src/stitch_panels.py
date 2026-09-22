#!/usr/bin/env python3
"""Stitch pi05 panel videos into ``<out>/side_by_side`` exactly like the Stage-1 evaluator.

Each panel's episode video already carries its SUCCESS/FAIL bar and language caption; the grid
only adds the panel label above each tile. Skipped when there is a single panel.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "train_skillVLA" / "stage1_eval" / "src"))
from compare_videos import stitch_panel_videos  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--grid_columns", type=int, default=0, help="panels per row (0 = one row)")
    args = parser.parse_args()
    panels = json.loads(os.environ.get("MODELS_JSON", "") or "[]")
    pairs = [(args.out_dir / "panels" / panel["panel_dir"] / "videos", panel["label"]) for panel in panels]
    if len(pairs) < 2:
        print("stitch: single panel; no side-by-side video")
        return
    stitch_panel_videos(pairs, args.out_dir / "side_by_side", args.grid_columns)


if __name__ == "__main__":
    main()
