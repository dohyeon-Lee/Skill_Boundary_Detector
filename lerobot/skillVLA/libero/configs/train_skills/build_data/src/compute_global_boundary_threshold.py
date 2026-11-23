"""Compute one scaled global-mean threshold from staged episode curves.

The curve-collection array writes one ``curves/ep*.npz`` per raw episode.  This
The script pools every eligible SG-smoothed replanning point (rather than
weighting tasks or array shards equally). Terminal-masked points are excluded,
so sharding cannot alter the threshold or leak episode-end artifacts into it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sbd_build_contract import validate_curve, validate_manifest, validate_reducer_args


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
    parser.add_argument("--probe_mode", default="")
    parser.add_argument("--probe_generation", default="")
    parser.add_argument("--probe_count", default="")
    parser.add_argument("--denoise_output", default="")
    parser.add_argument("--score_metric", default="")
    parser.add_argument("--score_power", default="")
    parser.add_argument("--terminal_mask_frames", default="")
    parser.add_argument("--terminal_mask_mode", default="fixed_frames")
    parser.add_argument("--terminal_release_pre_offset_frames", default="8")
    parser.add_argument("--threshold_scale", type=float, default=1.0)
    parser.add_argument("--segmentation_contract_json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    curves_dir = Path(args.curves_dir)
    contract = json.loads(args.segmentation_contract_json) if args.segmentation_contract_json else None
    if contract is not None:
        validate_reducer_args(args, contract)
        validate_manifest(curves_dir.parent / "skillset_manifest.json", contract)
    curve_paths = sorted(curves_dir.glob("ep*.npz"))
    too_short_paths = sorted(curves_dir.glob("ep*.too_short.json"))
    accounted_episodes = len(curve_paths) + len(too_short_paths)
    if accounted_episodes != args.expected_episodes:
        raise RuntimeError(
            f"Expected {args.expected_episodes} accounted episodes under {curves_dir}, "
            f"found {len(curve_paths)} curves and {len(too_short_paths)} "
            "too-short markers. The global threshold would be incomplete."
        )

    total, count = 0.0, 0
    initial_masks: set[int] = set()
    episode_ids: set[int] = set()
    skipped_episode_ids: set[int] = set()
    no_anchor_episode_ids: set[int] = set()
    for marker_path in too_short_paths:
        marker = json.loads(marker_path.read_text())
        if marker.get("reason") != "episode_shorter_than_dp_future_horizon":
            raise RuntimeError(f"Invalid too-short marker reason: {marker_path}")
        ep_id = int(marker["episode_id"])
        if ep_id in skipped_episode_ids:
            raise RuntimeError(f"Duplicate too-short marker for ep{ep_id:05d}")
        if int(marker["n_frames"]) >= int(marker["future_horizon"]):
            raise RuntimeError(f"Invalid too-short marker lengths: {marker_path}")
        skipped_episode_ids.add(ep_id)

    for curve_path in curve_paths:
        with np.load(curve_path, allow_pickle=False) as curve:
            if contract is not None:
                validate_curve(curve, contract, curve_path)
            ep_id = int(curve["episode_id"])
            values = np.asarray(curve["sg_vals"], dtype=np.float64)
            valid_start = int(np.asarray(curve["threshold_valid_start"]).item()) if "threshold_valid_start" in curve else 0
            initial_mask = int(np.asarray(curve["initial_mask_frames"]).item()) if "initial_mask_frames" in curve else 0
            initial_masks.add(initial_mask)
            valid_count = (
                int(np.asarray(curve["threshold_valid_count"]).item())
                if "threshold_valid_count" in curve
                else len(values)
            )
            if not (0 <= valid_start <= len(values) and 0 <= valid_count <= len(values)):
                raise RuntimeError(
                    f"Invalid threshold_valid_count={valid_count} for "
                    f"{len(values)} points: {curve_path}"
                )
            if initial_mask > 0:
                if "replan_ts" not in curve or valid_start != int(np.searchsorted(curve["replan_ts"], initial_mask, side="left")):
                    raise RuntimeError(f"Missing/inconsistent initial mask indices: {curve_path}")
            no_anchors = valid_count <= valid_start
            marked_empty = bool(np.asarray(curve["no_valid_boundary_anchors"]).item()) if "no_valid_boundary_anchors" in curve else False
            if no_anchors != marked_empty or (no_anchors and not np.isnan(values).all()):
                raise RuntimeError(f"Missing/inconsistent empty-mask marker: {curve_path}")
            values = values[valid_start:valid_count]
        if ep_id in episode_ids:
            raise RuntimeError(f"Duplicate episode curve for ep{ep_id:05d}: {curve_path}")
        if ep_id in skipped_episode_ids:
            raise RuntimeError(
                f"Episode ep{ep_id:05d} has both a curve and a too-short marker."
            )
        if (values.size == 0 and not no_anchors) or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid SG curve: {curve_path}")
        episode_ids.add(ep_id)
        if no_anchors:
            no_anchor_episode_ids.add(ep_id)
        total += float(values.sum())
        count += int(values.size)

    if count == 0:
        raise RuntimeError(f"No replanning values found under {curves_dir}")
    if len(initial_masks) != 1:
        raise RuntimeError("Cannot mix curves with different initial history masks.")
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
        "n_skipped_too_short": len(skipped_episode_ids),
        "n_no_valid_boundary_anchors": len(no_anchor_episode_ids),
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
            "probe_mode": args.probe_mode,
            "probe_generation": args.probe_generation,
            "probe_count": args.probe_count,
            "denoise_output": args.denoise_output,
            "score_metric": args.score_metric,
            "score_power": args.score_power,
            "initial_mask_frames": next(iter(initial_masks)),
            "terminal_mask_frames": args.terminal_mask_frames,
            "terminal_mask_mode": args.terminal_mask_mode,
            "terminal_release_pre_offset_frames": (
                args.terminal_release_pre_offset_frames
            ),
        },
    }
    if contract is not None:
        payload['segmentation_contract'] = contract['segmentation']
        # Keep legacy flat provenance readable, including every Gaussian/GMM
        # choice, while the structured recipe supports strict compatibility.
        payload['provenance'].update(contract['segmentation']['detector'])
        payload['provenance'].update({
            ('probe_generation' if key == 'generation' else 'probe_count' if key == 'count' else key): value
            for key, value in contract['segmentation']['probe'].items()
        })
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
