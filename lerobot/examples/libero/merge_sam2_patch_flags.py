"""Merge per-skill SAM2 mask files into one FSQ training npz.

Input files are produced by precompute_sam2_masks.py:

  sam2_masks_dir/task{task_id}/ep{episode_id:05d}_skill{skill_index:03d}.npz

Each file stores temporal patch masks. FSQ uses timestep-aligned flags, so
this script preserves the full sequence in a flat array plus offsets:

  patch_flags[offsets[i]:offsets[i + 1]]: (T_i, n_patches, 2)
  channels = [changed, green]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from tqdm import tqdm

from precompute_dino_features import load_skill_metadata


@dataclass
class Args:
    skills_dir: str
    """Skill npz directory, recursively searched."""

    sam2_masks_dir: str
    """Directory with per-skill SAM2 mask npz files."""

    output_path: str
    """Merged npz output path."""

    n_patches: int = 64
    """Expected patch count. For 8x8 pooled DINO patches, use 64."""


def _mask_path(root: Path, meta: dict) -> Path:
    task_id = int(meta["task_id"])
    ep_id = int(meta["episode_id"])
    skill_index = int(meta["skill_index"])
    return root / f"task{task_id}" / f"ep{ep_id:05d}_skill{skill_index:03d}.npz"


def main(args: Args) -> None:
    skills_dir = Path(args.skills_dir)
    sam2_root = Path(args.sam2_masks_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = load_skill_metadata(skills_dir)
    flags: list[np.ndarray] = []
    offsets = [0]
    missing = np.zeros((len(metadata),), dtype=bool)

    for i, meta in enumerate(tqdm(metadata, desc="Merging SAM2 flags")):
        path = _mask_path(sam2_root, meta)
        expected_len = int(meta["length"])
        if not path.exists():
            missing[i] = True
            flags.append(np.zeros((expected_len, args.n_patches, 2), dtype=np.uint8))
            offsets.append(offsets[-1] + expected_len)
            continue

        data = np.load(str(path))
        patch_masks = data["patch_masks"]  # (T, H_p, W_p, 2)
        if len(patch_masks) == 0:
            missing[i] = True
            flags.append(np.zeros((expected_len, args.n_patches, 2), dtype=np.uint8))
            offsets.append(offsets[-1] + expected_len)
            continue

        temporal = patch_masks.astype(np.uint8).reshape(len(patch_masks), -1, 2)
        if temporal.shape[1] != args.n_patches:
            raise ValueError(
                f"{path} has {temporal.shape[1]} patches after flattening, expected {args.n_patches}"
            )
        if len(temporal) != expected_len:
            raise ValueError(
                f"{path} length mismatch: patch_masks has {len(temporal)}, "
                f"metadata length is {expected_len}"
            )
        flags.append(temporal)
        offsets.append(offsets[-1] + len(temporal))

    flat_flags = (
        np.concatenate(flags, axis=0)
        if flags
        else np.zeros((0, args.n_patches, 2), dtype=np.uint8)
    )

    np.savez_compressed(
        str(output_path),
        patch_flags=flat_flags,
        offsets=np.array(offsets, dtype=np.int64),
        missing=missing,
        episode_id=np.array([m["episode_id"] for m in metadata], dtype=np.int64),
        task_id=np.array([m["task_id"] for m in metadata], dtype=np.int64),
        skill_index=np.array([m["skill_index"] for m in metadata], dtype=np.int64),
        frame_start=np.array([m["frame_start"] for m in metadata], dtype=np.int64),
        frame_end=np.array([m["frame_end"] for m in metadata], dtype=np.int64),
        length=np.array([m["length"] for m in metadata], dtype=np.int64),
        n_patches=np.array(args.n_patches, dtype=np.int64),
        channels=np.array(["changed", "green"]),
    )

    n_missing = int(missing.sum())
    print(f"[merge_sam2] Saved {flat_flags.shape} with offsets ({len(offsets) - 1} skills) -> {output_path}")
    if n_missing:
        print(f"[merge_sam2] Missing/empty skills: {n_missing}/{len(metadata)}")


if __name__ == "__main__":
    main(tyro.cli(Args))
