#!/usr/bin/env python3
"""④-b: RELATIVE action statistics for a LeRobot-v3 dataset (pi-family relative training).

The pi pipeline is `relative → normalize`, so the normalizer needs stats of the RELATIVE
distribution. This script computes them with the OFFICIAL lerobot machinery — the very
same `to_relative_actions` function and `RelativeActionsProcessorStep._build_mask`
(exclude-by-joint-NAME semantics) that the training pipeline runs — so the stats describe
bit-for-bit the tensors the normalizer will see:

    rel_k(t) = to_relative_actions(action[t+k], state[t], mask),  k ∈ [0, chunk_size)

Excluded joints (e.g. "gripper") need NO special-casing: the official function leaves those
dims absolute, so the pooled distribution for them is simply the absolute action
distribution — exactly what the normalizer will see for those dims too.

The dataset is NEVER modified (relative is a train-time transform; a stored column is
impossible — the value depends on the sampling anchor t). Output goes to a SEPARATE file so
meta/stats.json stays untouched and absolute consumers (DP MIN_MAX 등) are structurally
isolated:

    meta/relative_action_stats.json
    {"chunk_size", "exclude_joints", "action_names", "mask",
     "quantile_anchor_stride", "num_frames", "action": {min/max/mean/std/q01/q99}}

min/max/mean/std are EXACT over every (t, k) pair (streaming). q01/q99 pool every k with
stride-subsampled anchors t (cap ~20M samples) — tail-quantile error is negligible there
and memory stays flat (full materialization would be frames × chunk × dim).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# examples/libero/configs/generate_training_dataset/download_dataset/ABC_dataset/src
# → lerobot repo /src
_LEROBOT_SRC = Path(__file__).resolve().parents[7] / "src"
sys.path.insert(0, str(_LEROBOT_SRC))
from lerobot.processor.relative_action_processor import (  # noqa: E402
    RelativeActionsProcessorStep,
    to_relative_actions,
)

QUANTILE_SAMPLE_CAP = 20_000_000  # pooled rel-samples used for q01/q99 (per dim)


def load_state_action(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(states, actions, episode_index) over the whole dataset, frame order within episodes."""
    files = sorted((dataset_dir / "data").rglob("file-*.parquet")) or sorted(
        (dataset_dir / "data").rglob("episode_*.parquet"))
    if not files:
        raise SystemExit(f"no data parquet under {dataset_dir / 'data'}")
    cols = ["observation.state", "action", "episode_index", "frame_index"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"], kind="stable")
    states = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    actions = np.stack(df["action"].to_numpy()).astype(np.float64)
    return states, actions, df["episode_index"].to_numpy()


def episode_slices(episode_index: np.ndarray) -> list[slice]:
    starts = np.flatnonzero(np.r_[True, episode_index[1:] != episode_index[:-1]])
    ends = np.r_[starts[1:], len(episode_index)]
    return [slice(int(s), int(e)) for s, e in zip(starts, ends)]


def official_mask(dataset_dir: Path, exclude_joints: list[str], action_dim: int) -> tuple[list[bool], list[str] | None]:
    """Build the convert-mask EXACTLY like the training-time step will (name matching)."""
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    action_names = (info["features"].get("action") or {}).get("names")
    if exclude_joints and not action_names:
        raise SystemExit(
            "exclude_joints requires per-dim action `names` in meta/info.json (missing) — "
            "rebuild the dataset with the names-emitting converter (③), or pass --exclude-joints ''")
    step = RelativeActionsProcessorStep(exclude_joints=list(exclude_joints), action_names=action_names)
    return step._build_mask(action_dim), action_names  # noqa: SLF001 — single source of truth


def compute_relative_stats(states: np.ndarray, actions: np.ndarray, episode_index: np.ndarray,
                           chunk_size: int, mask: list[bool]) -> tuple[dict[str, np.ndarray], int]:
    """Exact streaming min/max/mean/std over all (t,k); stride-anchored quantiles.

    Uses the official `to_relative_actions` per (episode, k) so semantics (mask handling,
    dim alignment) are inherited from the training pipeline, not re-implemented."""
    D = actions.shape[1]
    slices = episode_slices(episode_index)

    n_pool = sum(max(0, (sl.stop - sl.start) - k) for sl in slices for k in range(chunk_size))
    if n_pool == 0:
        raise SystemExit("no (t, k) pairs — empty episodes?")
    stride = max(1, int(np.ceil(n_pool / QUANTILE_SAMPLE_CAP)))

    mn = np.full(D, np.inf)
    mx = np.full(D, -np.inf)
    s1 = np.zeros(D)
    s2 = np.zeros(D)
    count = 0
    q_chunks: list[np.ndarray] = []
    for ei, sl in enumerate(slices):
        A = torch.from_numpy(actions[sl])
        S = torch.from_numpy(states[sl])
        T = A.shape[0]
        for k in range(min(chunk_size, T)):
            # rel_k for every anchor t: action[t+k] − state[t] on masked dims (official fn)
            diff = to_relative_actions(A[k:], S[: T - k], mask).numpy()
            mn = np.minimum(mn, diff.min(axis=0))
            mx = np.maximum(mx, diff.max(axis=0))
            s1 += diff.sum(axis=0)
            s2 += (diff ** 2).sum(axis=0)
            count += diff.shape[0]
            # de-alias the fixed stride across (episode, k) so anchor phase varies
            q_chunks.append(diff[((ei * 31 + k) % stride):: stride].astype(np.float32))
    pooled = np.concatenate(q_chunks, axis=0)
    mean = s1 / count
    var = np.maximum(s2 / count - mean ** 2, 0.0)
    q01, q99 = np.quantile(pooled, [0.01, 0.99], axis=0)
    return {
        "min": mn, "max": mx, "mean": mean, "std": np.sqrt(var),
        "q01": q01.astype(np.float64), "q99": q99.astype(np.float64),
    }, stride


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="datasets root (e.g. .../dataset_ABC)")
    parser.add_argument("--dataset", required=True, help="dataset folder name under root")
    parser.add_argument("--chunk-size", type=int, default=50,
                        help="action horizon the relative pipeline trains with (pi05 default 50)")
    parser.add_argument("--exclude-joints", default="gripper",
                        help="comma-separated joint-NAME tokens kept ABSOLUTE (upstream "
                             "relative_exclude_joints semantics; '' = convert all dims)")
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute even if an up-to-date output exists")
    args = parser.parse_args()

    dataset_dir = args.root / args.dataset
    exclude_joints = [t for t in (s.strip() for s in args.exclude_joints.split(",")) if t]
    out_path = dataset_dir / "meta" / "relative_action_stats.json"

    if out_path.exists() and not args.overwrite:
        prev = json.loads(out_path.read_text())
        if int(prev.get("chunk_size", -1)) == args.chunk_size \
                and list(prev.get("exclude_joints", [])) == exclude_joints:
            print(f"[relative-stats] up-to-date (chunk={args.chunk_size}, "
                  f"exclude={exclude_joints}) → {out_path}  (--overwrite to recompute)")
            return
        print(f"[relative-stats] params changed (was chunk={prev.get('chunk_size')}, "
              f"exclude={prev.get('exclude_joints')}) — recomputing")

    states, actions, episode_index = load_state_action(dataset_dir)
    if states.shape != actions.shape:
        raise SystemExit(f"state {states.shape} vs action {actions.shape} mismatch — "
                         "relative(action − state) needs the same joint space")
    mask, action_names = official_mask(dataset_dir, exclude_joints, actions.shape[1])

    print(f"[relative-stats] {args.dataset}: {actions.shape[0]} frames, dim {actions.shape[1]}, "
          f"episodes {len(np.unique(episode_index))}, chunk {args.chunk_size}")
    print(f"[relative-stats] exclude {exclude_joints} → absolute-kept dims "
          f"{[i for i, m in enumerate(mask) if not m]} (official name-mask)")
    rel, stride = compute_relative_stats(states, actions, episode_index, args.chunk_size, mask)

    payload = {
        "chunk_size": args.chunk_size,
        "exclude_joints": exclude_joints,
        "action_names": action_names,
        "mask": mask,
        "quantile_anchor_stride": stride,
        "num_frames": int(actions.shape[0]),
        "action": {k: np.asarray(v, dtype=np.float64).tolist() for k, v in rel.items()},
    }
    out_path.write_text(json.dumps(payload, indent=2))
    with np.printoptions(precision=4, suppress=True):
        print(f"[relative-stats] q01 {np.asarray(rel['q01'])}")
        print(f"[relative-stats] q99 {np.asarray(rel['q99'])}")
    print(f"[relative-stats] ✅ → {out_path}  (meta/stats.json untouched)")


if __name__ == "__main__":
    main()
