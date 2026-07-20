#!/usr/bin/env python
"""Merge sharded compare_summary_shard*.json (SLURM array run) into the final
compare_summary.json + compare_chart.png. Runs as a dependency job after the array."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_compare import draw_chart  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    shards = sorted(args.out_dir.glob("compare_summary_shard*.json"))
    if not shards:
        raise SystemExit(f"no shard summaries in {args.out_dir}")

    merged = json.loads(shards[0].read_text())
    tasks = []
    for p in shards:
        tasks.extend(json.loads(p.read_text())["tasks"])
    tasks.sort(key=lambda t: t["task_id"])
    merged["tasks"] = tasks
    merged["overall"] = {
        lbl: float(np.mean([t["success_rate"].get(lbl, 0.0) for t in tasks]))
        for lbl in merged["models"]
    }
    merged["n_shards_merged"] = len(shards)

    (args.out_dir / "compare_summary.json").write_text(json.dumps(merged, indent=2))
    draw_chart(merged, args.out_dir / "compare_chart.png")

    print(f"merged {len(shards)} shards, {len(tasks)} tasks")
    print("OVERALL  " + "  ".join(f"{k}={v:.3f}" for k, v in merged["overall"].items()))


if __name__ == "__main__":
    main()
