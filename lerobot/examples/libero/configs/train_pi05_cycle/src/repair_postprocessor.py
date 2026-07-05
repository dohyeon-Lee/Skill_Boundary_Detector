#!/usr/bin/env python
"""Repair cycle-PT checkpoints whose saved postprocessor inherited pi05_base's unnormalizer
stats instead of the training dataset's (train-script bug fixed 2026-07-05; weights are fine).

Rebuilds pre+post processors with the dataset's stats and overwrites them in every
checkpoints/*/pretrained_model of the given run dirs.

Usage:
  python repair_postprocessor.py --dataset_dir .../dataset_filtered/libero_90_full_full \
      --runs outputs_filtered/pi05_cycle_PT/PTcyc_... [more run dirs...]
"""

import argparse
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=Path, required=True)
    ap.add_argument("--repo_id", default="lerobot/libero_90_full_full")
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    args = ap.parse_args()

    meta = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_dir)

    for run in args.runs:
        ckpts = sorted((run / "checkpoints").glob("*/pretrained_model"))
        if not ckpts:
            print(f"!! no checkpoints under {run}")
            continue
        for ckpt in ckpts:
            policy_cfg = PreTrainedConfig.from_pretrained(str(ckpt))
            policy_cfg.pretrained_path = ckpt
            pre, post = make_pre_post_processors(
                policy_cfg=policy_cfg,
                pretrained_path=str(ckpt),
                preprocessor_overrides={
                    # login node has no GPU; every eval consumer overrides device anyway
                    "device_processor": {"device": "cpu"},
                    "normalizer_processor": {
                        "stats": meta.stats,
                        "features": {**policy_cfg.input_features, **policy_cfg.output_features},
                        "norm_map": policy_cfg.normalization_mapping,
                    },
                },
                postprocessor_overrides={
                    "unnormalizer_processor": {
                        "stats": meta.stats,
                        "features": policy_cfg.output_features,
                        "norm_map": policy_cfg.normalization_mapping,
                    },
                },
            )
            pre.save_pretrained(str(ckpt))
            post.save_pretrained(str(ckpt))
            print(f"repaired {ckpt}")


if __name__ == "__main__":
    main()
