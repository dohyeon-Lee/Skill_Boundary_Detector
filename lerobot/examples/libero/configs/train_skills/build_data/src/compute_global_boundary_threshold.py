"""Compute the single global boundary threshold from staged episode curves.

The curve-collection array writes one ``curves/ep*.npz`` per raw episode.  This
script intentionally averages every SG-smoothed replanning point (not per-task
or per-array means), so array sharding cannot alter the global threshold.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--expected_episodes", type=int, required=True,
        help="Fail if the curve array did not produce exactly this many episode curves.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    curves_dir = Path(args.curves_dir)
    curve_paths = sorted(curves_dir.glob("ep*.npz"))
    if len(curve_paths) != args.expected_episodes:
        raise RuntimeError(
            f"Expected {args.expected_episodes} episode curves under {curves_dir}, "
            f"found {len(curve_paths)}. The global threshold would be incomplete."
        )

    total, count = 0.0, 0
    episode_ids: set[int] = set()
    for curve_path in curve_paths:
        with np.load(curve_path, allow_pickle=False) as curve:
            ep_id = int(curve["episode_id"])
            values = np.asarray(curve["sg_vals"], dtype=np.float64)
        if ep_id in episode_ids:
            raise RuntimeError(f"Duplicate episode curve for ep{ep_id:05d}: {curve_path}")
        if values.size == 0 or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid SG curve: {curve_path}")
        episode_ids.add(ep_id)
        total += float(values.sum())
        count += int(values.size)

    if count == 0:
        raise RuntimeError(f"No replanning values found under {curves_dir}")
    global_mean = total / count
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "boundary_threshold_mode": "global_mean",
        "global_mean": global_mean,
        "n_episodes": len(episode_ids),
        "n_replan_points": count,
        "aggregation": "mean_over_all_episode_sg_replan_points",
    }
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(temp_path, output_path)
    print(
        f"global_mean={global_mean:.8f} from {len(episode_ids)} episodes / "
        f"{count} SG replanning points → {output_path}"
    )


if __name__ == "__main__":
    main()
