"""
Convert an existing LeRobot v3.0 dataset (libero_90) to state-action delta format.
Replaces raw teleop actions with EEF delta actions computed from observation.state.

Action space (7D): [delta_eef_pos/0.05 (3), delta_eef_ori/0.5 (3), gripper (1)]
  - Same convention as convert_libero_stateaction_to_lerobot_dp.py

Does NOT re-encode videos — symlinks them from the source dataset.

Usage:
  PYTHONPATH=/data2/dohyeon/SBD/lerobot/src \
    /data2/dohyeon/SBD/.venv/bin/python \
    /data2/dohyeon/SBD/lerobot/examples/libero/convert_lerobot_to_stateaction.py \
    --src_dir /data2/dohyeon/SBD/libero_dataset/libero_90 \
    --dst_dir /data2/dohyeon/SBD/libero_dataset/libero_90_stateaction
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tyro

OSC_POS_SCALE = 0.05
OSC_ORI_SCALE = 0.5


@dataclass
class Args:
    src_dir: str = "/data2/dohyeon/SBD/libero_dataset/libero_90"
    dst_dir: str = "/data2/dohyeon/SBD/libero_dataset/libero_90_stateaction"


def compute_delta_actions_for_episode(ep_df: pd.DataFrame) -> list:
    """Compute delta EEF actions for a single episode (grouped rows)."""
    states = np.stack(ep_df["observation.state"].values)  # (T, 8)
    actions = np.stack(ep_df["action"].values)             # (T, 7)

    new_actions = []
    for i in range(len(ep_df)):
        curr_eef = states[i, :6].astype(np.float32)
        next_eef = states[i + 1, :6].astype(np.float32) if i + 1 < len(ep_df) else curr_eef
        delta = next_eef - curr_eef
        delta[:3] /= OSC_POS_SCALE
        delta[3:6] /= OSC_ORI_SCALE
        gripper = actions[i, 6:7].astype(np.float32)
        new_actions.append(np.concatenate([delta, gripper]))

    return new_actions


def update_episodes_stats(dst_dir: Path, df: pd.DataFrame) -> None:
    """Recompute per-episode action stats and write to episodes_stats.jsonl."""
    stats_path = dst_dir / "meta" / "episodes_stats.jsonl"
    if not stats_path.exists():
        return

    # Load existing stats indexed by episode_index
    existing = {}
    for line in stats_path.read_text().splitlines():
        if line.strip():
            entry = json.loads(line)
            existing[entry["episode_index"]] = entry

    # Recompute action stats per episode from the new df
    for ep_idx, ep_df in df.groupby("episode_index"):
        if ep_idx not in existing:
            continue
        actions = np.stack(ep_df["action"].values)
        existing[ep_idx]["stats"]["action"] = {
            "min": actions.min(axis=0).tolist(),
            "max": actions.max(axis=0).tolist(),
            "mean": actions.mean(axis=0).tolist(),
            "std": actions.std(axis=0).tolist(),
            "count": [len(actions)],
        }

    lines = [json.dumps(existing[k]) for k in sorted(existing.keys())]
    stats_path.write_text("\n".join(lines) + "\n")


def main(args: Args) -> None:
    src = Path(args.src_dir)
    dst = Path(args.dst_dir)

    if dst.exists():
        print(f"Removing existing output at {dst}")
        shutil.rmtree(dst)

    # Copy meta/
    print("Copying meta/...")
    shutil.copytree(src / "meta", dst / "meta")

    # Symlink videos/ (no re-encoding needed)
    print("Symlinking videos/...")
    (dst / "videos").symlink_to((src / "videos").resolve())

    # Process parquet files (v3.0: file-{index}.parquet, all episodes in one or few files)
    parquet_files = sorted((src / "data").rglob("file-*.parquet"))
    print(f"Processing {len(parquet_files)} parquet file(s)...")

    for src_parquet in parquet_files:
        print(f"  Reading {src_parquet.name} ...")
        df = pd.read_parquet(src_parquet)

        # Compute delta actions per episode (respect episode boundaries)
        print("  Computing delta actions...")
        new_actions = []
        for _, ep_df in df.groupby("episode_index", sort=False):
            new_actions.extend(compute_delta_actions_for_episode(ep_df))
        df["action"] = new_actions

        dst_parquet = dst / "data" / src_parquet.relative_to(src / "data")
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst_parquet, index=False)
        print(f"  Saved {dst_parquet}")

    print("Updating action stats in episodes_stats.jsonl...")
    # Re-read the written parquet for stats (or reuse df if single file)
    all_dfs = [pd.read_parquet(dst / "data" / p.relative_to(src / "data"))
               for p in parquet_files]
    full_df = pd.concat(all_dfs) if len(all_dfs) > 1 else all_dfs[0]
    update_episodes_stats(dst, full_df)

    print(f"Done. Dataset saved to {dst}")


if __name__ == "__main__":
    main(tyro.cli(Args))
