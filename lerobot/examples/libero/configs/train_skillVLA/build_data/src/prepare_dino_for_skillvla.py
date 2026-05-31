#!/usr/bin/env python3
"""Slice per-episode DINO features for a SkillVLA source dataset.

Copies/symlinks frame-level DINO npz shards from a base full-dataset DINO dir
into the SkillVLA intermediate workspace, using the source dataset's
meta/episodes `source_episode_index` mapping (falls back to episode_index).

  base : {base_dino_dir}/{image_key}/episode_{source_ep:07d}.npz
  dst  : {dst_dir}/{image_key}/episode_{target_ep:07d}.npz

This mirrors train_skills/DP/src/prepare_dino_for_training_dataset.py but takes
explicit paths so it can be reused outside the train_skills config.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# reuse the copy helpers from the train_skills DINO prepare script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "train_skills" / "DP" / "src"))
from prepare_dino_for_training_dataset import copy_one, load_episodes, safe_key  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True, help="Source LeRobot dataset (has meta/episodes)")
    p.add_argument("--base_dino_dir", required=True, help="Base full-dataset DINO dir (per-episode npz)")
    p.add_argument("--dst_dir", required=True, help="Destination per-episode DINO dir")
    p.add_argument("--image_keys", default="observation.images.image", help="Comma-separated image keys")
    p.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default="symlink")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    dataset_dir = Path(args.dataset_dir)
    base_dir = Path(args.base_dino_dir)
    dst_dir = Path(args.dst_dir)
    image_keys = [k.strip() for k in args.image_keys.split(",") if k.strip()]

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Source dataset not found: {dataset_dir}")
    if not base_dir.exists():
        raise FileNotFoundError(
            f"Base DINO dir not found: {base_dir}\n"
            f"Generate the full-dataset DINO first (configs/generate_training_dataset)."
        )

    episodes = load_episodes(dataset_dir).sort_values("episode_index").reset_index(drop=True)
    if "source_episode_index" not in episodes.columns:
        episodes["source_episode_index"] = episodes["episode_index"]

    print("Slice per-episode DINO")
    print(f"  source dataset : {dataset_dir}")
    print(f"  base DINO      : {base_dir}")
    print(f"  dst DINO       : {dst_dir}")
    print(f"  image keys     : {image_keys}")
    print(f"  mode           : {args.mode}")

    counts = {"write": 0, "skip": 0}
    missing: list[str] = []
    for _, row in episodes.iterrows():
        target_ep = int(row["episode_index"])
        source_ep = int(row["source_episode_index"])
        for image_key in image_keys:
            key_dir = safe_key(image_key)
            src_file = base_dir / key_dir / f"episode_{source_ep:07d}.npz"
            dst_file = dst_dir / key_dir / f"episode_{target_ep:07d}.npz"
            if not src_file.exists():
                missing.append(str(src_file))
                continue
            result = copy_one(src_file, dst_file, mode=args.mode, overwrite=args.overwrite)
            counts[result] = counts.get(result, 0) + 1

    if missing:
        preview = "\n  ".join(missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} base DINO files. First few:\n  {preview}")

    # extract_skill_dino_tokens.py needs manifest.json for backbone metadata.
    import shutil
    base_manifest = base_dir / "manifest.json"
    if base_manifest.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base_manifest, dst_dir / "manifest.json")
        print(f"  manifest       : copied from {base_manifest}")
    else:
        print(f"  [warn] no manifest.json in {base_dir} — extract_skill_dino_tokens may fail")

    print(f"  wrote/skipped  : {counts}")
    print("DONE")


if __name__ == "__main__":
    main()
