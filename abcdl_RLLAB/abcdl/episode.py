"""Neutral in-memory representation of one episode (the IR all formats share)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


@dataclass(frozen=True)
class StateLayout:
    arm_dof: int
    gripper_dof: int
    n_arms: int

    @property
    def per_arm(self) -> int:
        return self.arm_dof + self.gripper_dof

    @property
    def state_dim(self) -> int:
        return self.n_arms * self.per_arm

    @property
    def action_dim(self) -> int:
        return self.state_dim


StateLayout.YAM = StateLayout(arm_dof=6, gripper_dof=1, n_arms=2)


@dataclass
class CameraStream:
    frames: Union[list, np.ndarray]   # list[bytes] (encoded Annex-B) OR (T,H,W,3) uint8
    timestamps: np.ndarray            # (Tc,) int64 ns
    width: int
    height: int
    codec: str                        # "h264" | "h265" | "raw"

    def __len__(self) -> int:
        return len(self.frames)


@dataclass
class EpisodeMeta:
    task: str
    fps: float
    cameras: list[str]
    camera_resolutions: dict[str, tuple]
    camera_codecs: dict[str, str]
    operator_id: Optional[str] = None
    alignment: str = "native"
    t0_ns: Optional[int] = None
    tick_ns: Optional[int] = None
    session_id: Optional[str] = None


@dataclass
class Episode:
    states: np.ndarray                # (T, state_dim) float64
    actions: np.ndarray               # (T, action_dim) float64
    timestamps: np.ndarray            # (T,) int64 ns
    cameras: dict[str, "CameraStream"]  # {name: CameraStream}
    meta: EpisodeMeta
    ee_poses: Optional[dict] = None   # {side: (T,4,4)}
    extras: Optional[dict] = None     # {side: {"velocity":..., "torque":...}}
    subtasks: Optional[list] = None   # [(t_ns, label)]
    # Named per-frame scalar/vector features, each leading-dim T (e.g. reward,
    # mc_return, success). Stored as frame_features.npz; surfaced as dataset item keys.
    frame_features: Optional[dict] = None  # {name: (T,) or (T, d) array}

    @property
    def num_steps(self) -> int:
        return int(self.states.shape[0])

    def validate(self) -> None:
        T = self.num_steps
        if self.actions.shape[0] != T:
            raise ValueError(f"actions length {self.actions.shape[0]} != states {T}")
        if self.timestamps.shape[0] != T:
            raise ValueError(f"timestamps length {self.timestamps.shape[0]} != states {T}")
        for name in self.meta.cameras:
            if name not in self.cameras:
                raise ValueError(f"meta lists camera {name!r} but it is missing from cameras")
