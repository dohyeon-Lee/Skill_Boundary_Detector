"""Merge partial frozen visual feature files produced by parallel workers."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).parent))

from precompute_dino_features import (
    load_skill_metadata,
    _load_episodes_meta,
    _resolve_image_key,
    _save_npz,
    _partial_path,
)


@dataclass
class Args:
    skills_dir: str
    dataset_dir: str
    output_path: str
    num_workers: int
    image_key: str = "observation.images.image"
    image_model_name: str = "/data2/dohyeon/SBD/models/dinov3-vits16"
    patch_grid: int = 8


def main(args: Args) -> None:
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = load_skill_metadata(Path(args.skills_dir))
    episodes_meta = _load_episodes_meta(Path(args.dataset_dir))
    image_key = _resolve_image_key(episodes_meta, args.image_key)

    print(f"[MERGE] skills={len(metadata)}, workers={args.num_workers}")

    feature_chunks: list[np.ndarray | None] = [None] * len(metadata)

    for worker_id in range(args.num_workers):
        partial_path = _partial_path(output_path, worker_id, args.num_workers)
        if not partial_path.exists():
            raise FileNotFoundError(f"Missing partial file: {partial_path}")

        data = np.load(str(partial_path))
        skill_indices = data["skill_indices"]
        features = data["features"]
        offsets = data["offsets"]

        for rank, idx in enumerate(skill_indices):
            feature_chunks[idx] = features[offsets[rank] : offsets[rank + 1]]

        print(f"[MERGE] Loaded worker {worker_id}: {len(skill_indices)} skills from {partial_path.name}")

    missing = [i for i, f in enumerate(feature_chunks) if f is None]
    if missing:
        raise RuntimeError(f"Missing features for {len(missing)} skills: {missing[:10]}...")

    ordered_chunks = list(feature_chunks)
    offsets_out = [0]
    for feat in ordered_chunks:
        offsets_out.append(offsets_out[-1] + len(feat))

    features_all = np.concatenate(ordered_chunks, axis=0).astype(np.float16)
    _save_npz(output_path, features_all, offsets_out, metadata, image_key, args.image_model_name, args.patch_grid)

    print(f"[MERGE] Saved {features_all.shape} → {output_path}")

    for worker_id in range(args.num_workers):
        partial_path = _partial_path(output_path, worker_id, args.num_workers)
        partial_path.unlink(missing_ok=True)
    print("[MERGE] Cleaned up partial files.")


if __name__ == "__main__":
    main(tyro.cli(Args))
