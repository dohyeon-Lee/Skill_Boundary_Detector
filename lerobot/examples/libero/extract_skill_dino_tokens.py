"""
Extract skill-level DINO tokens from pre-computed frame-level features.

Reads per-episode npz files produced by precompute_frame_dino_features.py and
slices the relevant frame windows for each skill, outputting the same format
as precompute_dino_features.py (N_total, n_tokens, feat_dim) float16.

Usage:
    python extract_skill_dino_tokens.py \
        --frame_dino_dir .../libero_90_frame_dino_features/dinov3_vits16_pg8 \
        --skills_dir     .../libero_90_skillset/skills \
        --output_path    .../libero_90_skillset_dinov3_vits16_tokens.npz \
        --image_key      observation.images.image
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from precompute_dino_features import (
    fit_feature_length,
    load_skill_metadata,
    _save_npz,
)


@dataclass
class Args:
    frame_dino_dir: str
    """Directory produced by precompute_frame_dino_features.py for one backbone/patch_grid.
    Must contain manifest.json and {image_key_norm}/episode_XXXXXXX.npz files."""
    skills_dir: str
    """Skill npz directory (skillset/skills)."""
    output_path: str
    """Output npz path (same format as precompute_dino_features.py)."""
    image_key: str = "observation.images.image"


def _image_key_norm(image_key: str) -> str:
    return image_key.replace(".", "_").replace("/", "_")


def main(args: Args) -> None:
    frame_dino_dir = Path(args.frame_dino_dir)
    skills_dir     = Path(args.skills_dir)
    output_path    = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read manifest for backbone metadata
    manifest_path = frame_dino_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {frame_dino_dir}")
    with open(manifest_path) as f:
        manifest = json.load(f)

    patch_grid  = int(manifest["patch_grid"])
    model_name  = manifest["image_model_name"]
    image_key   = args.image_key

    ep_dir = frame_dino_dir / _image_key_norm(image_key)
    if not ep_dir.exists():
        available = [p.name for p in frame_dino_dir.iterdir() if p.is_dir()]
        raise FileNotFoundError(
            f"Camera directory '{_image_key_norm(image_key)}' not found in {frame_dino_dir}.\n"
            f"Available: {available}"
        )

    metadata = load_skill_metadata(skills_dir)
    print(f"[extract] {len(metadata)} skills  |  frame_dino_dir={frame_dino_dir}")
    print(f"[extract] output → {output_path}")

    # Group skills by episode for sequential file access
    by_episode: dict[int, list[int]] = {}
    for idx, m in enumerate(metadata):
        by_episode.setdefault(int(m["episode_id"]), []).append(idx)

    # Verify all episode files exist up front
    missing = [ep for ep in by_episode if not (ep_dir / f"episode_{ep:07d}.npz").exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} episode npz files missing from {ep_dir}. "
            f"First few: {[f'episode_{e:07d}.npz' for e in missing[:5]]}\n"
            "Run DP_train/run_frame_dino_parallel.sh first."
        )

    chunks: list[np.ndarray | None] = [None] * len(metadata)
    t0 = time.perf_counter()

    for ep_id, skill_ids in tqdm(sorted(by_episode.items()), desc="Extracting skill tokens"):
        ep_data     = np.load(str(ep_dir / f"episode_{ep_id:07d}.npz"))
        ep_features = ep_data["features"]  # (T_ep, n_tokens, feat_dim) float16

        for idx in skill_ids:
            m  = metadata[idx]
            fs = int(m["frame_start"])
            fe = int(m["frame_end"])
            clip = ep_features[fs : min(fe, len(ep_features))]
            chunks[idx] = fit_feature_length(clip, int(m["length"]))

    # Concatenate in order
    ordered = []
    offsets = [0]
    for idx, feat in enumerate(chunks):
        if feat is None:
            raise RuntimeError(f"No features for skill index {idx}")
        ordered.append(feat)
        offsets.append(offsets[-1] + len(feat))

    features = np.concatenate(ordered, axis=0)
    _save_npz(output_path, features, offsets, metadata, image_key, model_name, patch_grid)

    elapsed = max(time.perf_counter() - t0, 1e-6)
    print(f"[extract] Saved {features.shape} (float16) → {output_path}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main(tyro.cli(Args))
