"""SkillExpert (Stage-1) training dataset — LeRobotDataset + the current skill's STATE TRAJECTORY.

On top of the standard (current-frame) sample, each item gains the Oracle's input: the current
skill's full state trajectory (start→end), resampled to a fixed length:

  skill_state_traj : (resample_n, state_dim) float32

This is fetched at runtime from the existing dataset — no rebuild needed. The current skill's frame
range is [idx - skill_ds, idx + skill_de] (skill_ds/skill_de are per-frame parquet columns written by
add_skill_latents_to_dataset.py; a skill is contiguous within an episode). The states over that range
are linearly resampled to a fixed N (endpoints anchored). The range is the SAME for every frame of a
skill, so the trajectory — and hence the Oracle's r — is constant within a skill (matches Stage-2,
where the VLM emits r once at the skill start and holds it).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import make_interp_spline

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Batch key this dataset adds (the policy's forward feeds it to the Oracle).
SKILL_STATE_TRAJ = "skill_state_traj"

STATE_KEY = "observation.state"
# Trailing gripper-STATE dims (LIBERO observation.state = [x,y,z,roll,pitch,yaw, gripper, gripper]).
# MIRRORS FSQ.py: gripper dims stay ABSOLUTE + use a degree-1 spline (near-step → no cubic overshoot);
# pose dims are zero-grounded + use the pose spline degree. Keep in sync with FSQ.py spline_encode.
N_GRIPPER_DIMS = 2


def _scalar(x) -> int:
    return int(x.item() if torch.is_tensor(x) else np.asarray(x).reshape(-1)[0])


def _spline_resample(traj: np.ndarray, n: int, degree: int) -> np.ndarray:
    """Resample a (L, D) state trajectory to (n, D) control points EXACTLY as the FSQ encoder
    (FSQ.py spline_encode): zero-ground the pose (gripper stays absolute), then per-dim interpolating
    spline (gripper dims degree 1, pose dims `degree`) sampled at n uniform points. The Oracle thus sees
    the trajectory in the same representation the skill code z was derived from → r is a clean residual."""
    traj = np.asarray(traj, dtype=np.float32)
    L, D = traj.shape
    if L < 2:                                                          # degenerate skill → tile the lone frame
        return np.broadcast_to(traj, (n, D)).astype(np.float32).copy()
    offset = traj[0].copy()
    offset[-N_GRIPPER_DIMS:] = 0.0                                     # zero-ground pose; gripper absolute
    traj = traj - offset
    t_orig = np.linspace(0.0, 1.0, L)
    t_ctrl = np.linspace(0.0, 1.0, n)
    ctrl = np.zeros((n, D), dtype=np.float32)
    for d in range(D):
        k = 1 if d >= D - N_GRIPPER_DIMS else degree                  # gripper: degree 1; pose: `degree`
        ctrl[:, d] = make_interp_spline(t_orig, traj[:, d], k=min(k, L - 1))(t_ctrl)
    return ctrl


class SkillExpertDataset(LeRobotDataset):
    """LeRobotDataset that also yields the current skill's resampled state trajectory (Oracle input)."""

    def __init__(self, *args, resample_n: int = 30, spline_degree: int = 3, **kwargs):
        self.resample_n = int(resample_n)
        self.spline_degree = int(spline_degree)
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx) -> dict:
        item = super().__getitem__(idx)

        ds = _scalar(item["skill_ds"])      # frames since the current skill's start (start = 0)
        de = _scalar(item["skill_de"])      # frames to the current skill's end (end = 0)
        frame_index = _scalar(item["frame_index"])
        ep_idx = _scalar(item["episode_index"])

        # Skill span in GLOBAL indices (contiguous within the episode); clamp to the episode bounds.
        ep_start = int(idx) - frame_index
        ep_len = _scalar(self.meta.episodes[ep_idx]["length"])
        start = max(ep_start, int(idx) - ds)
        end = min(ep_start + ep_len - 1, int(idx) + de)                 # inclusive last frame

        rows = self.hf_dataset[start : end + 1][STATE_KEY]             # states over the skill
        if isinstance(rows, list):
            traj = np.stack([np.asarray(s, dtype=np.float32) for s in rows])
        else:
            traj = np.asarray(rows, dtype=np.float32)                  # (L, state_dim)

        ctrl = _spline_resample(traj, self.resample_n, self.spline_degree)  # FSQ-style control points
        item[SKILL_STATE_TRAJ] = torch.from_numpy(ctrl)
        return item
