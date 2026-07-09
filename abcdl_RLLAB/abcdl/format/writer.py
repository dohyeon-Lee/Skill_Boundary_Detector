"""Write an Episode to the abcdl MP4+binary on-disk layout."""

from __future__ import annotations

import json
import os

import numpy as np

from abcdl.constants import TICK_NS
from abcdl.episode import Episode
from abcdl.format.encode import encode_strict_h264


def write_abcdl(episode: Episode, out_dir: str) -> None:
    episode.validate()
    os.makedirs(out_dir, exist_ok=True)
    T = episode.num_steps

    sa = np.concatenate([np.asarray(episode.states, np.float64),
                         np.asarray(episode.actions, np.float64)], axis=1)
    sa.astype("<f8").tofile(os.path.join(out_dir, "states_actions.bin"))

    stacks = []
    h = w = None
    for name in episode.meta.cameras:
        frames = np.asarray(episode.cameras[name].frames, np.uint8)
        if frames.shape[0] != T:
            raise ValueError(f"camera {name} has {frames.shape[0]} frames, expected {T}")
        if h is None:
            h, w = frames.shape[1], frames.shape[2]
        elif (frames.shape[1], frames.shape[2]) != (h, w):
            raise ValueError("all cameras must share H,W for vertical stacking")
        stacks.append(frames)
    combined = np.concatenate(stacks, axis=1)  # vstack along height
    encode_strict_h264(combined, os.path.join(out_dir, "combined_camera-images-rgb.mp4"))

    # Named per-frame features (reward, mc_return, success, embeddings, …): each stored
    # as its own raw, contiguous ``ff_<name>.bin`` so the loader can MEMMAP it and read
    # only the touched frame rows (no decode, no whole-episode load) — the fast path for
    # large-batch / high-iteration RL training. dtype + per-frame shape go in metadata.
    frame_features_meta: dict = {}
    if episode.frame_features:
        for name, arr in episode.frame_features.items():
            a = np.ascontiguousarray(arr)
            if a.shape[0] != T:
                raise ValueError(f"frame_feature {name!r} has length {a.shape[0]}, expected {T}")
            a.tofile(os.path.join(out_dir, f"ff_{name}.bin"))
            frame_features_meta[name] = {"dtype": str(a.dtype), "shape": list(a.shape[1:])}

    tick_ns = int(episode.meta.tick_ns or TICK_NS)
    meta = {
        "task_name": episode.meta.task,
        "cameras": list(episode.meta.cameras),
        "camera_resolutions": {k: list(v) for k, v in episode.meta.camera_resolutions.items()},
        "alignment": episode.meta.alignment,
        "t0_ns": int(episode.meta.t0_ns or 0),
        "tick_ns": tick_ns,
        # Store fps explicitly so readers can recover the exact value without
        # floating-point drift from 1e9/tick_ns (e.g. 30 Hz stored as tick_ns
        # =33_333_333 gives 30.0000003, not 30.0).
        "fps": float(episode.meta.fps) if episode.meta.fps is not None else 1e9 / tick_ns,
        "num_steps": T,
        "state_dim": int(episode.states.shape[1]),
        "action_dim": int(episode.actions.shape[1]),
        "operator_id": episode.meta.operator_id,
        "session_id": episode.meta.session_id,
        "frame_feature_keys": list(frame_features_meta),   # back-compat list
        "frame_features": frame_features_meta,             # {name: {dtype, shape}}
    }
    with open(os.path.join(out_dir, "episode_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
