#!/usr/bin/env python3
# Inputs:
#   dataset name : --dataset, TRAIN_DATA, or target_dataset in train_skills_config.yaml
#   dataset path : {project_root}/{dataset_root}/{target_dataset}
#   source DINO  : {project_root}/{dataset_root}/{dino_source_dataset}_DINO/pg{dino_patch_grid}
# Reference model:
#   none
# Outputs:
#   prepared DINO: {project_root}/{dataset_root}/FSQ_dataset/{target_dataset}/DINO/pg{dino_patch_grid}
"""Prepare target-dataset DINO features for DP/FSQ training.

This copies frame-level DINO npz shards from a full/base dataset DINO directory
into the target training workspace:

  libero_dataset/FSQ_dataset/{target_dataset}/DINO/pg{patch_grid}/

For generated subsets, build_training_dataset.py stores source_episode_index in
meta/episodes. That mapping is used so target episode_0000000.npz points to the
correct source episode from the base dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from train_skills_config import DEFAULT_CONFIG_PATH, load_config, train_settings


def safe_key(image_key: str) -> str:
    return image_key.replace("/", "_").replace(".", "_")


def load_episodes(dataset_dir: Path) -> pd.DataFrame:
    paths = sorted((dataset_dir / "meta" / "episodes").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode parquet files found under {dataset_dir / 'meta/episodes'}")
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def copy_one(src: Path, dst: Path, mode: str, overwrite: bool) -> str:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return "skip"
        dst.unlink()

    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    else:
        raise ValueError(f"Unsupported copy mode: {mode}. Use copy, hardlink, or symlink.")
    return "write"


def write_manifest(
    dst_dir: Path,
    src_dir: Path,
    settings: dict,
    mapping: list[dict],
    counts: dict[str, int],
    image_keys: list[str],
    patch_grid: int,
) -> None:
    source_manifest = src_dir / "manifest.json"
    manifest = {}
    if source_manifest.exists():
        with open(source_manifest) as f:
            manifest = json.load(f)

    manifest.update(
        {
            "dataset_dir": str(settings["raw_dataset_dir"]),
            "source_dino_dir": str(src_dir),
            "target_dino_dir": str(dst_dir),
            "target_dataset": settings["target_dataset"],
            "source_dataset": settings["dino_source_dataset"],
            "image_keys": image_keys,
            "patch_grid": patch_grid,
            "format": "per_episode_npz",
            "episode_mapping": mapping,
            "copy_counts": counts,
        }
    )
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset", default=None, help="Override target_dataset from yaml")
    parser.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default=None)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing prepared DINO files")
    parser.add_argument(
        "--variant", choices=["dp", "fsq"], default="dp",
        help="Which grid to prepare: 'dp' = pg{dino_patch_grid} (DP/skillset), "
             "'fsq' = pg{fsq_patch_grid} (FSQ skill-token extraction). Same dir when grids match.",
    )
    parser.add_argument(
        "--image_keys", default=None,
        help="Comma-separated camera keys to copy (overrides yaml dino_image_keys). Use this to "
             "stage BOTH cameras for the FSQ terminator without changing the DP's dino_image_keys, "
             "e.g. observation.images.image,observation.images.wrist_image",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    settings = train_settings(cfg, dataset=args.dataset)

    mode = args.mode or str(settings["dino_copy_mode"])
    overwrite = bool(args.overwrite or settings["dino_overwrite_prepared"])
    dataset_dir = Path(settings["raw_dataset_dir"])
    if args.variant == "fsq":
        src_dir = Path(settings["base_fsq_dino_feature_dir"])
        dst_dir = Path(settings["fsq_dino_feature_dir"])
        patch_grid = settings["fsq_patch_grid"]
    else:
        src_dir = Path(settings["base_dino_feature_dir"])
        dst_dir = Path(settings["dino_feature_dir"])
        patch_grid = settings["dino_patch_grid"]
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")
    if not src_dir.exists():
        raise FileNotFoundError(f"Base DINO feature dir not found: {src_dir}")

    # Which camera subdirs to copy. Default: auto-detect every camera present in the base DINO
    # dir (wrist absent → copy 3rd only; both present → copy both). --image_keys restricts to a
    # specific set. DP is unaffected either way — it reads its own dino_image_keys at train time.
    if args.image_keys:
        key_dirs = [safe_key(k.strip()) for k in args.image_keys.split(",") if k.strip()]
        for kd in key_dirs:
            if not (src_dir / kd).is_dir():
                raise FileNotFoundError(f"Requested camera not found in base DINO: {src_dir / kd}")
    else:
        key_dirs = sorted(p.name for p in src_dir.iterdir() if p.is_dir())
        if not key_dirs:
            raise FileNotFoundError(f"No camera subdirs found in base DINO: {src_dir}")

    episodes = load_episodes(dataset_dir).sort_values("episode_index").reset_index(drop=True)
    if "source_episode_index" not in episodes.columns:
        episodes["source_episode_index"] = episodes["episode_index"]
    if "source_task_index" not in episodes.columns and "task_index" in episodes.columns:
        episodes["source_task_index"] = episodes["task_index"]

    counts = {"write": 0, "skip": 0}
    mapping: list[dict] = []
    missing: list[str] = []

    print("Prepare DINO features")
    print(f"  target dataset : {settings['target_dataset']}")
    print(f"  source dataset : {settings['dino_source_dataset']}")
    print(f"  source DINO    : {src_dir}")
    print(f"  target DINO    : {dst_dir}")
    print(f"  cameras        : {key_dirs}")
    print(f"  mode           : {mode}")
    print(f"  overwrite      : {overwrite}")

    for _, row in episodes.iterrows():
        target_ep = int(row["episode_index"])
        source_ep = int(row["source_episode_index"])
        entry = {"target_episode_index": target_ep, "source_episode_index": source_ep}
        if "source_task_index" in row:
            entry["source_task_index"] = int(row["source_task_index"])
        mapping.append(entry)

        for key_dir in key_dirs:
            src_file = src_dir / key_dir / f"episode_{source_ep:07d}.npz"
            dst_file = dst_dir / key_dir / f"episode_{target_ep:07d}.npz"
            if not src_file.exists():
                missing.append(str(src_file))
                continue
            result = copy_one(src_file, dst_file, mode=mode, overwrite=overwrite)
            counts[result] = counts.get(result, 0) + 1

    if missing:
        preview = "\n  ".join(missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} source DINO files. First missing files:\n  {preview}"
        )

    write_manifest(dst_dir, src_dir, settings, mapping, counts, key_dirs, patch_grid)
    print(f"  wrote/skipped  : {counts}")
    print("DONE")


if __name__ == "__main__":
    main()
