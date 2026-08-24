# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""SE(3)-aware relative actions for LIBERO's OSC pose controller.

The derived LIBERO dataset stores an absolute commanded EEF target per frame as
``[position(3), absolute_axis_angle(3), gripper(1)]``.  The policy is trained on
the target relative to one current-pose anchor:

* position: ``target_position - anchor_position``;
* rotation: ``Log(R_target @ R_anchor.T)`` (world/left convention);
* gripper: kept absolute.

At inference the inverse transform is followed by the exact robosuite OSC input
conversion.  The OSC controller uses ``R_goal = Exp(delta) @ R_current``.

This module is intentionally separate from ``relative_action_processor.py``.
That processor owns ABC's elementwise joint-space contract and must remain
unchanged for existing datasets and checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE

from .pipeline import ProcessorStep, ProcessorStepRegistry

__all__ = [
    "EefRelativeActionsProcessorStep",
    "EefRelativeToOscActionsProcessorStep",
    "absolute_eef_actions_to_osc",
    "matrix_to_rotation_vector",
    "osc_actions_to_absolute_eef",
    "rotation_vector_to_matrix",
    "to_eef_absolute_actions",
    "to_eef_relative_actions",
]


def _check_pose_action(action: Tensor, state: Tensor) -> None:
    if action.shape[-1] != 7:
        raise ValueError(f"EEF pose action must be 7D, got shape {tuple(action.shape)}")
    if state.shape[-1] < 6:
        raise ValueError(f"EEF state needs at least position+rotation (6D), got {tuple(state.shape)}")


def _broadcast_state(state: Tensor, action: Tensor) -> Tensor:
    """Broadcast one anchor state over an optional action time dimension."""
    state = state.to(device=action.device, dtype=action.dtype)
    if action.ndim == state.ndim + 1:
        state = state.unsqueeze(-2)
    return state


def _hat(vector: Tensor) -> Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero),
        dim=-1,
    ).reshape(*vector.shape[:-1], 3, 3)


def rotation_vector_to_matrix(rotation_vector: Tensor) -> Tensor:
    """Convert an axis-angle rotation vector to a rotation matrix.

    ``rotation_vector`` uses exponential coordinates: its direction is the axis
    and its norm is the angle in radians.
    """
    if rotation_vector.shape[-1] != 3:
        raise ValueError(f"rotation vector must end in 3, got {tuple(rotation_vector.shape)}")
    theta2 = (rotation_vector * rotation_vector).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2)
    small = theta2 < 1e-8
    # Stable Taylor branches around zero. torch.where evaluates both branches,
    # so clamp denominators in the general expression as well.
    a = torch.where(
        small,
        1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
        torch.sin(theta) / theta.clamp_min(1e-12),
    )
    b = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-12),
    )
    k = _hat(rotation_vector)
    eye = torch.eye(3, dtype=rotation_vector.dtype, device=rotation_vector.device)
    eye = eye.expand(*rotation_vector.shape[:-1], 3, 3)
    return eye + a.unsqueeze(-1) * k + b.unsqueeze(-1) * (k @ k)


def _matrix_to_quaternion_wxyz(matrix: Tensor) -> Tensor:
    """Robustly convert rotation matrices to unit quaternions in ``wxyz`` order."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix must end in (3,3), got {tuple(matrix.shape)}")
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]

    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                (
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ),
                dim=-1,
            ),
            min=0.0,
        )
    )
    # Each row is a quaternion candidate whose named component is dominant.
    candidates = torch.stack(
        (
            torch.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), dim=-1),
            torch.stack((m10 - m01, m02 + m20, m12 + m21, q_abs[..., 3] ** 2), dim=-1),
        ),
        dim=-2,
    )
    candidates = candidates / (2.0 * q_abs.clamp_min(1e-8).unsqueeze(-1))
    index = q_abs.argmax(dim=-1)
    gather_index = index[..., None, None].expand(*index.shape, 1, 4)
    quat = candidates.gather(dim=-2, index=gather_index).squeeze(-2)
    quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-12)
    # q and -q encode the same rotation. Prefer w >= 0 so Log returns angles <= pi.
    return torch.where(quat[..., :1] < 0, -quat, quat)


def matrix_to_rotation_vector(matrix: Tensor) -> Tensor:
    """Convert a rotation matrix to the principal axis-angle vector."""
    quat = _matrix_to_quaternion_wxyz(matrix)
    w = quat[..., :1].clamp(-1.0, 1.0)
    xyz = quat[..., 1:]
    sin_half = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, w)
    scale = torch.where(
        sin_half < 1e-8,
        2.0 + (sin_half * sin_half) / 3.0,
        angle / sin_half.clamp_min(1e-12),
    )
    return xyz * scale


def to_eef_relative_actions(actions: Tensor, state: Tensor) -> Tensor:
    """Convert absolute EEF targets to one-anchor SE(3)-relative actions."""
    _check_pose_action(actions, state)
    anchor = _broadcast_state(state, actions)
    out = actions.clone()
    out[..., :3] = actions[..., :3] - anchor[..., :3]
    target_rotation = rotation_vector_to_matrix(actions[..., 3:6])
    anchor_rotation = rotation_vector_to_matrix(anchor[..., 3:6])
    relative_rotation = target_rotation @ anchor_rotation.transpose(-1, -2)
    out[..., 3:6] = matrix_to_rotation_vector(relative_rotation)
    # Gripper is an absolute command and is intentionally untouched.
    return out


def to_eef_absolute_actions(actions: Tensor, state: Tensor) -> Tensor:
    """Invert :func:`to_eef_relative_actions` using the same world/left convention."""
    _check_pose_action(actions, state)
    anchor = _broadcast_state(state, actions)
    out = actions.clone()
    out[..., :3] = actions[..., :3] + anchor[..., :3]
    relative_rotation = rotation_vector_to_matrix(actions[..., 3:6])
    anchor_rotation = rotation_vector_to_matrix(anchor[..., 3:6])
    target_rotation = relative_rotation @ anchor_rotation
    out[..., 3:6] = matrix_to_rotation_vector(target_rotation)
    return out


def osc_actions_to_absolute_eef(
    actions: Tensor,
    state: Tensor,
    *,
    position_scale: float = 0.05,
    rotation_scale: float = 0.5,
    clip: bool = True,
) -> Tensor:
    """Apply robosuite OSC deltas to current EEF states and return absolute targets."""
    _check_pose_action(actions, state)
    if position_scale <= 0 or rotation_scale <= 0:
        raise ValueError("OSC position and rotation scales must be positive")
    current = _broadcast_state(state, actions)
    command = actions.clamp(-1.0, 1.0) if clip else actions
    out = actions.clone()
    out[..., :3] = current[..., :3] + command[..., :3] * position_scale
    delta_rotation = rotation_vector_to_matrix(command[..., 3:6] * rotation_scale)
    current_rotation = rotation_vector_to_matrix(current[..., 3:6])
    out[..., 3:6] = matrix_to_rotation_vector(delta_rotation @ current_rotation)
    out[..., 6] = actions[..., 6]
    return out


def absolute_eef_actions_to_osc(
    actions: Tensor,
    state: Tensor,
    *,
    position_scale: float = 0.05,
    rotation_scale: float = 0.5,
    clip: bool = True,
) -> Tensor:
    """Convert absolute EEF targets into normalized LIBERO OSC delta inputs."""
    _check_pose_action(actions, state)
    if position_scale <= 0 or rotation_scale <= 0:
        raise ValueError("OSC position and rotation scales must be positive")
    current = _broadcast_state(state, actions)
    out = actions.clone()
    out[..., :3] = (actions[..., :3] - current[..., :3]) / position_scale
    target_rotation = rotation_vector_to_matrix(actions[..., 3:6])
    current_rotation = rotation_vector_to_matrix(current[..., 3:6])
    # robosuite applies R_goal = Exp(delta) @ R_current.
    delta_rotation = target_rotation @ current_rotation.transpose(-1, -2)
    out[..., 3:6] = matrix_to_rotation_vector(delta_rotation) / rotation_scale
    out[..., 6] = actions[..., 6]
    if clip:
        out[..., :6] = out[..., :6].clamp(-1.0, 1.0)
    return out


@ProcessorStepRegistry.register("eef_relative_actions_processor")
@dataclass
class EefRelativeActionsProcessorStep(ProcessorStep):
    """Train on absolute EEF targets relative to the current observation pose."""

    enabled: bool = False
    _last_state: Tensor | None = field(default=None, init=False, repr=False)

    @staticmethod
    def _current_state(state: Tensor) -> Tensor:
        return state[..., -1, :] if state.ndim >= 3 else state

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {}) or {}
        state = observation.get(OBS_STATE)
        current = self._current_state(state) if state is not None else None
        if current is not None:
            self._last_state = current
        if not self.enabled:
            return transition
        action = transition.get(TransitionKey.ACTION)
        if action is None or current is None:
            return transition.copy()
        out = transition.copy()
        out[TransitionKey.ACTION] = to_eef_relative_actions(action, current)
        return out

    def get_cached_state(self) -> Tensor | None:
        return self._last_state

    def reset(self) -> None:
        self._last_state = None

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("eef_relative_to_osc_actions_processor")
@dataclass
class EefRelativeToOscActionsProcessorStep(ProcessorStep):
    """Convert an unnormalized relative EEF prediction into a LIBERO OSC input."""

    enabled: bool = False
    position_scale: float = 0.05
    rotation_scale: float = 0.5
    clip: bool = True
    relative_step: EefRelativeActionsProcessorStep | None = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        if self.relative_step is None:
            raise RuntimeError("EEF relative-to-OSC processor has no paired relative preprocessor")
        state = self.relative_step.get_cached_state()
        if state is None:
            raise RuntimeError("EEF relative-to-OSC processor has no cached current EEF state")
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition.copy()
        absolute = to_eef_absolute_actions(action, state)
        out = transition.copy()
        out[TransitionKey.ACTION] = absolute_eef_actions_to_osc(
            absolute,
            state,
            position_scale=self.position_scale,
            rotation_scale=self.rotation_scale,
            clip=self.clip,
        )
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "position_scale": self.position_scale,
            "rotation_scale": self.rotation_scale,
            "clip": self.clip,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
