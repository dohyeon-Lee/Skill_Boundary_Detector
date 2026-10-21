"""Our LIBERO layout/statistics, using the unmodified native GR00T processor steps.

Native pretrained stats may describe OTHER robots, absolute coordinates, or a
7D state. Our dataset has grounded/raw XYZ3 + axis-angle3 + finger2 and already
delta EEF action3+rotation3+gripper1. Never subtract state from action again.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

STATE_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "left_finger", "right_finger"]
ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def dataset_assets(config, stats: dict[str, Any], base, *, horizon: int):
    from lerobot.policies.groot import processor_groot as pg

    def groups(key, names):
        source = stats.get(key, {})
        if "q01" not in source or "q99" not in source:
            raise ValueError(f"GR00T requires dataset {key} q01/q99 statistics.")
        low, high = (torch.as_tensor(source[name]).flatten() for name in ("q01", "q99"))
        if low.numel() != len(names) or high.numel() != len(names):
            raise ValueError(f"GR00T {key} must have {len(names)} statistic dimensions.")
        if not bool(torch.isfinite(low).all() & torch.isfinite(high).all()) or bool((high < low).any()):
            raise ValueError(f"Invalid GR00T {key} quantiles.")
        return {name: {stat: ([float(torch.as_tensor(value).flatten()[0])] if stat == "count"
                             else [float(torch.as_tensor(value).flatten()[index])])
                       for stat, value in source.items()
                       if stat == "count" or torch.as_tensor(value).numel() == len(names)}
                for index, name in enumerate(names)}

    raw = {"state": groups("observation.state", STATE_KEYS), "action": groups("action", ACTION_KEYS)}
    layout = {
        "video": {"delta_indices": [0], "modality_keys": ["image", "wrist_image"]},
        "state": {"delta_indices": [0], "modality_keys": STATE_KEYS},
        "action": {"delta_indices": list(range(horizon)), "modality_keys": ACTION_KEYS,
                   # ABSOLUTE here means do not re-difference the ALREADY delta commands.
                   "action_configs": [{"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT"}
                                      for _ in ACTION_KEYS]},
    }
    flat = {key: {"min": torch.as_tensor(stats[key]["q01"]).tolist(),
                  "max": torch.as_tensor(stats[key]["q99"]).tolist()}
            for key in ("observation.state", "action")}
    fields = dict(stats=flat, raw_stats=raw, modality_config=layout,
                  valid_action_horizon=horizon, max_action_horizon=40, video_horizon=1,
                  use_percentiles=True, use_relative_action=False,
                  video_modality_keys=["image", "wrist_image"])
    if base is not None:
        return replace(base, **fields,
                       embodiment_mapping={**pg.N1_7_EMBODIMENT_MAPPING, **base.embodiment_mapping})
    return pg._GrootN17CheckpointProcessorAssets(
        **fields, embodiment_mapping=dict(pg.N1_7_EMBODIMENT_MAPPING),
        formalize_language=True, state_dropout_prob=0.0, clip_outliers=True,
        image_crop_size=[230, 230], image_target_size=[256, 256],
        shortest_image_edge=None, crop_fraction=None,
        use_albumentations=False, letter_box_transform=False,
    )


def make_dataset_processors(config, stats, meta, *, horizon):
    from lerobot.policies.groot import processor_groot as pg

    original = pg._load_n1_7_checkpoint_processor_assets
    assets = dataset_assets(config, stats, original(config), horizon=horizon)
    # A process-local construction hook. Saved pipelines serialize the native
    # steps, our exact layout, quantiles and masks; inference needs no hook.
    try:
        pg._load_n1_7_checkpoint_processor_assets = lambda _: assets
        return pg.make_groot_pre_post_processors(config, dataset_stats=stats, dataset_meta=meta)
    finally:
        pg._load_n1_7_checkpoint_processor_assets = original
