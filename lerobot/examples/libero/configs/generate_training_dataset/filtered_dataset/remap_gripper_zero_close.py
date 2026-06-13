#!/usr/bin/env python3
# Inputs:
#   LeRobot v3 dataset : --dataset-dir (meta/ + data/ + videos/)
# Outputs:
#   data parquet       : action[..., 6] remapped in place (new = 1 - 2*old)
#   meta/episodes      : per-episode action stats columns transformed analytically
#   meta/stats.json    : action stats transformed analytically
#   meta/info.json     : 'note' field appended (idempotency marker)
"""Remap the gripper action from the hub's {0,1} convention to robosuite's ±1 ("zero_close").

The IPEC/OpenVLA LIBERO datasets store the gripper action as 0/1 with 0 = close. The local
training/eval pipeline (and the LIBERO env at rollout) expects -1 = open / +1 = close, so we
apply ``new = 1 - 2*old`` to action dim 6 — the same mapping recorded in the existing
libero_dataset/libero_90 ("gripper remapped with mapping=zero_close"). Verified against that
dataset: hub gripper mean 0.532 → 1-2*0.532 = -0.064 ≈ local mean -0.058.

Stats are transformed analytically (exact for an affine map with a sign flip):
  min' = 1-2*max   max' = 1-2*min   mean' = 1-2*mean   std' = 2*std
  q01' = 1-2*q99   q10' = 1-2*q90   q50' = 1-2*q50   q90' = 1-2*q10   q99' = 1-2*q01
Run ensure_quantile_stats.py --overwrite afterwards if you prefer recomputed-from-data stats.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

GRIPPER_DIM = 6
NOTE = "gripper remapped with mapping=zero_close"
# stat name -> (source stat name, scale on the value). new = 1 - 2*source for value-like stats.
QUANTILE_FLIP = {"min": "max", "max": "min", "q01": "q99", "q10": "q90", "q50": "q50",
                 "q90": "q10", "q99": "q01", "mean": "mean"}


def remap_vec(vec):
    v = np.asarray(vec, dtype=np.float32).copy()
    v[..., GRIPPER_DIM] = 1.0 - 2.0 * v[..., GRIPPER_DIM]
    return v


def remap_data(root: Path) -> tuple[float, float]:
    files = sorted((root / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data parquet under {root}")
    before, after = [], []
    for f in files:
        df = pd.read_parquet(f)
        acts = np.stack(df["action"].to_numpy()).astype(np.float32)
        before.append(acts[:, GRIPPER_DIM].mean())
        acts[:, GRIPPER_DIM] = 1.0 - 2.0 * acts[:, GRIPPER_DIM]
        after.append(acts[:, GRIPPER_DIM].mean())
        df["action"] = list(acts)
        df.to_parquet(f, index=False)
    return float(np.mean(before)), float(np.mean(after))


def remap_episode_stats(root: Path) -> int:
    """Transform per-episode 'stats/action/*' columns in meta/episodes parquet files."""
    files = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    n = 0
    for f in files:
        df = pd.read_parquet(f)
        cols = {c for c in df.columns if c.startswith("stats/action/")}
        if not cols:
            continue
        orig = {c: [np.asarray(v, dtype=np.float64).copy() for v in df[c]] for c in cols}
        for new_stat, src_stat in QUANTILE_FLIP.items():
            dst, src = f"stats/action/{new_stat}", f"stats/action/{src_stat}"
            if dst not in cols or src not in cols:
                continue
            vals = []
            for i in range(len(df)):
                v = np.asarray(orig[dst][i], dtype=np.float64).copy()
                s = np.asarray(orig[src][i], dtype=np.float64).reshape(v.shape)
                v[..., GRIPPER_DIM] = 1.0 - 2.0 * s[..., GRIPPER_DIM]
                vals.append(v)
            df[dst] = vals
        if "stats/action/std" in cols:
            df["stats/action/std"] = [
                (lambda v: (v.__setitem__((..., GRIPPER_DIM), 2.0 * v[..., GRIPPER_DIM]) or v))(
                    np.asarray(v, dtype=np.float64).copy())
                for v in orig["stats/action/std"]
            ]
        df.to_parquet(f, index=False)
        n += 1
    return n


def remap_stats_json(root: Path) -> None:
    path = root / "meta" / "stats.json"
    if not path.exists():
        return
    stats = json.loads(path.read_text())
    a = stats.get("action")
    if not a:
        return
    orig = {k: list(v) for k, v in a.items() if isinstance(v, list)}
    for new_stat, src_stat in QUANTILE_FLIP.items():
        if new_stat in orig and src_stat in orig:
            a[new_stat][GRIPPER_DIM] = 1.0 - 2.0 * orig[src_stat][GRIPPER_DIM]
    if "std" in orig:
        a["std"][GRIPPER_DIM] = 2.0 * orig["std"][GRIPPER_DIM]
    path.write_text(json.dumps(stats, indent=4))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="LeRobot v3 dataset root")
    args = parser.parse_args()
    root = args.dataset_dir

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    if NOTE in str(info.get("note", "")):
        print(f"Already remapped (note present), skipping: {root}")
        return

    before, after = remap_data(root)
    n_eps_files = remap_episode_stats(root)
    remap_stats_json(root)
    info["note"] = (info.get("note", "") + "; " if info.get("note") else "") + NOTE
    info_path.write_text(json.dumps(info, indent=4))

    print("DONE gripper remap (zero_close)")
    print(f"  dataset       : {root}")
    print(f"  gripper mean  : {before:.3f} -> {after:.3f}  (expect ~0.5 -> ~-0.05)")
    print(f"  episode-stats : {n_eps_files} parquet file(s) transformed")


if __name__ == "__main__":
    main()
