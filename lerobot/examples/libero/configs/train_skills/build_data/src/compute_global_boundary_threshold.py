"""Compute one scaled global-mean threshold from staged episode curves.

The curve-collection array writes one ``curves/ep*.npz`` per raw episode.  This
The script pools every SG-smoothed replanning point (rather than weighting
tasks or array shards equally), so sharding cannot alter the threshold.
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
    parser.add_argument("--dp_run_name", default="")
    parser.add_argument("--dp_checkpoint", default="")
    parser.add_argument("--eval_at_step", default="")
    parser.add_argument("--n_gmm_components", default="")
    parser.add_argument("--replan_interval", default="")
    parser.add_argument("--smooth_window", default="")
    parser.add_argument("--savgol_polyorder", default="")
    parser.add_argument("--threshold_scale", type=float, default=1.0)
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
    if args.threshold_scale <= 0.0:
        raise ValueError("threshold_scale must be positive.")
    global_threshold = global_mean * args.threshold_scale
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "boundary_threshold_mode": "global_mean",
        # global_mean is the raw statistic; global_threshold is the scaled
        # detector cutoff. Old scale=1 readers remain compatible.
        "global_mean": global_mean,
        "global_threshold": global_threshold,
        "boundary_threshold_scale": args.threshold_scale,
        "n_episodes": len(episode_ids),
        "n_replan_points": count,
        "aggregation": "scaled_mean_over_all_episode_sg_replan_points",
        "provenance": {
            "dp_run_name": args.dp_run_name,
            "dp_checkpoint": args.dp_checkpoint,
            "eval_at_step": args.eval_at_step,
            "n_gmm_components": args.n_gmm_components,
            "replan_interval": args.replan_interval,
            "smooth_window": args.smooth_window,
            "savgol_polyorder": args.savgol_polyorder,
        },
    }
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(temp_path, output_path)
    print(
        f"global_threshold={global_threshold:.8f} "
        f"(mean={global_mean:.8f} × scale={args.threshold_scale:g}) "
        f"from {len(episode_ids)} episodes / "
        f"{count} SG replanning points → {output_path}"
    )


if __name__ == "__main__":
    main()
