#!/usr/bin/env python3
"""Merge per-episode DINO npz shards into one frame-level dino.npz.

Reads the per-episode DINO directory (sliced by prepare_dino_for_skillvla.py):
  {frame_dino_dir}/{image_key}/episode_XXXXXXX.npz  (each: features (T, n_tokens, feat_dim))

Writes a single npz in the format consumed by SkillVLA training
(skillvla_dino_token_dataset.py) and add_skill_latents_to_dataset.load_dino_feature_map:
  features    (N_total, n_tokens, feat_dim) float16
  offsets     (E+1,) int64    per-episode cumulative offsets
  episode_id  (E,)  int64
  frame_start (E,)  int64     0 for each episode (LeRobot frame_index resets per episode)
  length      (E,)  int64
  n_tokens, feat_dim          int64 scalars
  image_key, image_model_name, patch_grid   metadata
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def safe_key(image_key: str) -> str:
    return image_key.replace("/", "_").replace(".", "_")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frame_dino_dir", required=True, help="per-episode DINO dir (manifest + episode npz)")
    p.add_argument("--image_key", default="observation.images.image")
    p.add_argument("--output_path", required=True)
    args = p.parse_args()

    dino_dir = Path(args.frame_dino_dir)
    key_dir = dino_dir / safe_key(args.image_key)
    files = sorted(key_dir.glob("episode_*.npz"), key=lambda f: int(f.stem.split("_")[1]))
    if not files:
        raise FileNotFoundError(f"No episode_*.npz under {key_dir}")

    manifest = {}
    if (dino_dir / "manifest.json").exists():
        manifest = json.loads((dino_dir / "manifest.json").read_text())

    feats_list, offsets, ep_ids, frame_starts, lengths = [], [0], [], [], []
    for f in files:
        ep = int(f.stem.split("_")[1])
        feat = np.load(str(f))["features"]   # (T, n_tokens, feat_dim)
        feats_list.append(feat)
        offsets.append(offsets[-1] + len(feat))
        ep_ids.append(ep)
        frame_starts.append(0)
        lengths.append(len(feat))

    features = np.concatenate(feats_list, axis=0).astype(np.float16)
    n_tokens = int(features.shape[1])
    feat_dim = int(features.shape[2])

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out),
        features=features,
        offsets=np.array(offsets, dtype=np.int64),
        episode_id=np.array(ep_ids, dtype=np.int64),
        frame_start=np.array(frame_starts, dtype=np.int64),
        length=np.array(lengths, dtype=np.int64),
        n_tokens=np.array(n_tokens, dtype=np.int64),
        feat_dim=np.array(feat_dim, dtype=np.int64),
        image_key=str(args.image_key),
        image_model_name=str(manifest.get("image_model_name", "")),
        patch_grid=np.array(int(manifest.get("patch_grid", 0)), dtype=np.int64),
    )
    print(f"[merge] {features.shape} from {len(files)} episodes → {out}")


if __name__ == "__main__":
    main()
