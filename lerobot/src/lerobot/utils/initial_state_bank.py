"""Read pi05-generated scene banks without using demonstration annotations."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def bank_identity(path):
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def inspect_bank(path, *, suite, task_ids, episodes):
    """Validate metadata/coverage at submission, without importing LIBERO."""
    identity = bank_identity(path)
    with np.load(identity["path"], allow_pickle=True) as data:
        required = {"metadata", "episode_index", "init_states", "scene_file", "model_body_poses"}
        if not required.issubset(data.files):
            raise ValueError("initial_state_bank must be a generated pi05 bank, not a demo GT archive")
        metadata = json.loads(str(data["metadata"].item()))
        if metadata.get("version") != 1 or metadata.get("suite") != suite:
            raise ValueError(f"initial_state_bank suite/version mismatch (requested {suite})")
        if not set(task_ids).issubset(set(metadata.get("task_ids", []))):
            raise ValueError("initial_state_bank does not contain every requested task_id")
        if episodes <= 0 or int(metadata.get("episodes", 0)) < episodes:
            raise ValueError("initial_state_bank has fewer episodes than requested n_episodes")
        count = len(data["episode_index"])
        if any(len(data[key]) != count for key in required - {"metadata"}):
            raise ValueError("initial_state_bank arrays have inconsistent lengths")
        scenes = Counter(str(scene) for scene in data["scene_file"])
        if len(scenes) != len(metadata["task_ids"]) or any(n < episodes for n in scenes.values()):
            raise ValueError("initial_state_bank scene coverage is inconsistent with its metadata")
    return {**identity, "random_range": float(metadata["random_range"]), "suite": suite,
            "n_episodes": episodes, "sampling_seed": int(metadata["seed"])}


def load_bank(identity):
    """Load sorted rows only if the exact submission-time bytes still exist."""
    if bank_identity(identity["path"]) != {key: identity[key] for key in ("path", "sha256")}:
        raise ValueError("initial_state_bank changed after config resolution; refusing to mix scenes")
    scenes = {}
    with np.load(identity["path"], allow_pickle=True) as data:
        for episode, state, scene, poses in zip(
            data["episode_index"], data["init_states"], data["scene_file"], data["model_body_poses"], strict=True
        ):
            scenes.setdefault(str(scene).removesuffix("_demo.hdf5"), []).append({
                "episode_index": int(episode), "state": np.asarray(state, dtype=np.float64),
                "model_body_poses": json.loads(str(poses)),
            })
    for rows in scenes.values():
        rows.sort(key=lambda row: row["episode_index"])
        if len({row["episode_index"] for row in rows}) != len(rows):
            raise ValueError("initial_state_bank has duplicate episode indices within a scene")
    return scenes


def require_predictor_only(specs):
    """Fail instead of silently broadcasting stale GT over randomized scenes."""
    for spec in specs:
        uses_gt_uv = (spec.get("needs_focus_uv") or spec.get("terminator_needs_goal_uv")) and spec.get("focus_source") != "predictor"
        needs_pose = spec.get("needs_end_pose") or spec.get("terminator_needs_goal_xyz") or spec.get("expert_xyz_source") == "gt"
        if (
            spec.get("skill_source") not in {"own", "external"}
            or spec.get("advance_mode") not in {"own", "external", "original"}
            or uses_gt_uv or (needs_pose and spec.get("pose_source") != "predictor")
            or spec.get("latent_source") == "oracle" or spec.get("predictor_diagnostics")
            or spec.get("task0_intervention")
        ):
            raise ValueError(
                f"initial_state_bank requires predictor-only skill/UV/pose and learned termination; "
                f"GT/oracle/diagnostic/intervention dependency in {spec.get('label', '<model>')}."
            )
