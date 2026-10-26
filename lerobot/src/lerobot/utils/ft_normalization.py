"""Checkpoint-owned normalization for the FT statistics ablation.

No LeRobot or torch imports: isolated baseline environments can share this
reader without importing the main checkout's policy classes.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path


def normalization_stats_mode(value="ft"):
    if value not in ("ft", "pt"):
        raise ValueError("training.normalization_stats must be 'ft' or 'pt'.")
    return value


def _pipeline_stats(root, filename):
    import numpy as np
    from safetensors.numpy import load_file

    pipeline = json.loads((root / filename).read_text())
    result = {}
    for step in pipeline["steps"]:
        registry = step.get("registry_name", "")
        if registry in ("normalizer_processor", "unnormalizer_processor",
                        "molmoact2_masked_normalizer", "molmoact2_masked_unnormalizer"):
            state_file = step.get("state_file")
            if not state_file:
                raise ValueError(f"Missing normalization state_file in {root / filename}.")
            for key, value in load_file(str(root / state_file)).items():
                feature, stat = key.rsplit(".", 1)
                result.setdefault(feature, {})[stat] = value.copy()
        elif registry in ("groot_n1_7_pack_inputs_v1", "groot_n1_7_action_decode_v1"):
            raw = step["config"]["raw_stats"]
            for group, feature, names in (
                ("state", "observation.state", ["x", "y", "z", "roll", "pitch", "yaw", "left_finger", "right_finger"]),
                ("action", "action", ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]),
            ):
                if group not in raw:
                    continue
                if set(raw[group]) != set(names):
                    raise ValueError(f"Unexpected {group} layout in {root / filename}.")
                shared = set.intersection(*(set(raw[group][name]) for name in names))
                result[feature] = {
                    stat: np.asarray(raw[group][names[0]][stat]).copy() if stat == "count"
                    else np.asarray([np.asarray(raw[group][name][stat]).item() for name in names])
                    for stat in shared
                }
    return result


def checkpoint_normalization_stats(pretrained_path):
    """Read saved state/action statistics and check encoder/decoder agreement."""
    import numpy as np

    root = Path(pretrained_path)
    pre = _pipeline_stats(root, "policy_preprocessor.json")
    post = _pipeline_stats(root, "policy_postprocessor.json")
    if "action" not in post:
        raise ValueError(f"No saved action unnormalization statistics in {root}.")
    if "action" in pre:
        for stat in set(pre["action"]) & set(post["action"]):
            if not np.array_equal(pre["action"][stat], post["action"][stat]):
                raise ValueError(f"Pre/post action statistics disagree for {stat} in {root}.")
    result = {**pre, "action": post["action"]}
    for feature in ("observation.state", "action"):
        block = result.get(feature, {})
        if "q01" not in block or "q99" not in block:
            raise ValueError(f"Missing saved {feature} q01/q99 in {root}.")
        low, high = np.asarray(block["q01"]), np.asarray(block["q99"])
        if low.ndim != 1 or low.shape != high.shape or not low.size:
            raise ValueError(f"Invalid saved {feature} quantile shapes in {root}.")
        if not (np.isfinite(low).all() and np.isfinite(high).all() and (low <= high).all()):
            raise ValueError(f"Invalid saved {feature} quantiles in {root}.")
    return result


def with_checkpoint_normalization(dataset_stats, pretrained_path):
    result = copy.deepcopy(dataset_stats)
    saved = checkpoint_normalization_stats(pretrained_path)
    for feature in ("observation.state", "action"):
        current = dataset_stats.get(feature, {})
        expected = current.get("q01", current.get("mean"))
        if expected is not None and len(expected) != len(saved[feature]["q01"]):
            raise ValueError(f"Checkpoint and dataset {feature} dimensions differ.")
    result.update(saved)
    return result


def validate_resume_normalization(checkpoint, requested, metadata_name):
    """Historical FT checkpoints used FT stats; never change mode on resume."""
    path = Path(checkpoint) / "pretrained_model" / metadata_name
    saved = json.loads(path.read_text()).get("normalization_stats", "ft")
    if saved != requested:
        raise ValueError(f"Cannot change normalization_stats from {saved!r} to {requested!r} "
                         "while resuming; use a new run.suffix.")

