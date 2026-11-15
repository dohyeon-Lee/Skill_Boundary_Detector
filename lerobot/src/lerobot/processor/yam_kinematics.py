"""Batched YAM forward kinematics and joint-to-delta-pose conversion.

The constants below are the six arm-body transforms and the ``grasp_site``
transform from ABC's official ``i2rt_yam/yam.xml``.  ABC stores each arm as
six revolute joints plus one gripper command, in the order

``[left joint1..6, left gripper, right joint1..6, right gripper]``.

This module deliberately implements forward kinematics only.  It is used to
express recorded absolute joint targets in an EEF delta-pose coordinate system
for DP training and SBD analysis; it does not turn predictions back into robot
commands and therefore has no Mink/i2rt/DAQP (IK) dependency.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

YAM_ACTION_DIM = 14
YAM_ARM_JOINT_INDICES = ((0, 1, 2, 3, 4, 5), (7, 8, 9, 10, 11, 12))
YAM_GRIPPER_INDICES = (6, 13)
YAM_DELTA_POSE_NAMES = (
    "left_delta_x",
    "left_delta_y",
    "left_delta_z",
    "left_delta_rot_x",
    "left_delta_rot_y",
    "left_delta_rot_z",
    "left_gripper",
    "right_delta_x",
    "right_delta_y",
    "right_delta_z",
    "right_delta_rot_x",
    "right_delta_rot_y",
    "right_delta_rot_z",
    "right_gripper",
)

# Body-local transforms for link_1 ... link_6, copied from the official model.
_LINK_POSITIONS = (
    (0.0, 0.0, 0.0631),
    (0.000025, -0.02, 0.0409),
    (0.264, 0.0, 0.0),
    (-0.245, -0.06, 0.0),
    (-0.074, -0.0395, 0.000025),
    (0.0, 0.0353, 0.0395),
)
_LINK_QUATERNIONS_WXYZ = (
    (1.0, 0.0, 0.0, 1.0),
    (1.0, 1.0, 1.0, 1.0),
    (0.0, 1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (1.0, -1.0, 1.0, 1.0),
    (1.0, -1.0, 0.0, 0.0),
)
_GRASP_SITE_POSITION = (0.0, 0.0, 0.1347)
_GRASP_SITE_QUATERNION_WXYZ = (1.0, 0.0, 0.0, -1.0)


def _quaternion_wxyz_to_matrix(quaternion: Tensor) -> Tensor:
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def _rotation_z(angle: Tensor) -> Tensor:
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    zero = torch.zeros_like(angle)
    one = torch.ones_like(angle)
    return torch.stack(
        (cosine, -sine, zero, sine, cosine, zero, zero, zero, one), dim=-1
    ).reshape(*angle.shape, 3, 3)


def rotation_matrix_to_rotvec(rotation: Tensor) -> Tensor:
    """Return the shortest SO(3) logarithm for a batch of rotation matrices."""
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    vee = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    sine = torch.sin(angle)
    small = angle.abs() < 1e-5
    scale = torch.where(
        small,
        torch.full_like(angle, 0.5),
        angle / (2.0 * sine.clamp_min(1e-7)),
    )
    return vee * scale.unsqueeze(-1)


def yam_arm_grasp_pose(joints: Tensor) -> tuple[Tensor, Tensor]:
    """Forward-kinematics pose of ABC's YAM ``grasp_site``.

    Args:
        joints: Tensor shaped ``[..., 6]`` in radians.

    Returns:
        Position ``[..., 3]`` and rotation matrix ``[..., 3, 3]`` in the arm
        base frame.  ABC mounts both arms with the same fixed orientation, so
        fixed world translations do not affect the delta-pose representation.
    """
    if joints.shape[-1] != 6:
        raise ValueError(f"YAM arm FK expects [...,6] joints, got {tuple(joints.shape)}")
    positions = joints.new_tensor(_LINK_POSITIONS)
    quaternions = joints.new_tensor(_LINK_QUATERNIONS_WXYZ)
    fixed_rotations = _quaternion_wxyz_to_matrix(quaternions)
    position = torch.zeros(*joints.shape[:-1], 3, dtype=joints.dtype, device=joints.device)
    rotation = torch.eye(3, dtype=joints.dtype, device=joints.device).expand(
        *joints.shape[:-1], 3, 3
    )
    for index in range(6):
        position = position + torch.matmul(rotation, positions[index]).squeeze(-1)
        rotation = torch.matmul(
            torch.matmul(rotation, fixed_rotations[index]), _rotation_z(joints[..., index])
        )
    site_position = joints.new_tensor(_GRASP_SITE_POSITION)
    site_rotation = _quaternion_wxyz_to_matrix(
        joints.new_tensor(_GRASP_SITE_QUATERNION_WXYZ)
    )
    position = position + torch.matmul(rotation, site_position).squeeze(-1)
    rotation = torch.matmul(rotation, site_rotation)
    return position, rotation


def yam_bimanual_grasp_pose(joints: Tensor) -> tuple[Tensor, Tensor]:
    """Return left/right grasp poses from canonical 14D ABC joint vectors."""
    if joints.shape[-1] != YAM_ACTION_DIM:
        raise ValueError(f"YAM bimanual FK expects [...,14], got {tuple(joints.shape)}")
    arm_indices = torch.as_tensor(YAM_ARM_JOINT_INDICES, device=joints.device)
    arm_joints = joints[..., arm_indices]
    return yam_arm_grasp_pose(arm_joints)


def to_yam_delta_pose_actions(actions: Tensor, state: Tensor) -> Tensor:
    """Convert absolute joint targets into consecutive bimanual EEF deltas.

    Translation and SO(3) rotation-vector deltas are expressed in the fixed arm
    base frame.  The first target is compared with the current observed joint
    state; each later target is compared with the previous target.  Gripper
    commands remain absolute.  Input and output are both 14D.
    """
    if actions.ndim not in {2, 3} or actions.shape[-1] != YAM_ACTION_DIM:
        raise ValueError(
            "YAM delta-pose actions must be [B,14] or [B,T,14], "
            f"got {tuple(actions.shape)}"
        )
    if state.shape[-1] != YAM_ACTION_DIM:
        raise ValueError(f"YAM delta-pose state must end in 14 dims, got {tuple(state.shape)}")
    # [T,14] with a single [14] anchor is an unbatched temporal chunk.  In
    # contrast, [B,14] with [B,14] states represents independent one-step
    # samples (the form used by the streaming stats computation).
    if actions.ndim == 2 and state.ndim == 1:
        return to_yam_delta_pose_actions(actions.unsqueeze(0), state.unsqueeze(0)).squeeze(0)
    state = state.to(device=actions.device, dtype=actions.dtype)
    target_position, target_rotation = yam_bimanual_grasp_pose(actions)
    state_position, state_rotation = yam_bimanual_grasp_pose(state)
    if actions.ndim == 2:
        previous_position = state_position
        previous_rotation = state_rotation
    else:
        previous_position = torch.cat(
            (state_position.unsqueeze(-3), target_position[..., :-1, :, :]), dim=-3
        )
        previous_rotation = torch.cat(
            (state_rotation.unsqueeze(-4), target_rotation[..., :-1, :, :, :]), dim=-4
        )
    delta_position = target_position - previous_position
    # Spatial/base-frame angular increment, paired with base-frame translation.
    delta_rotation = torch.matmul(target_rotation, previous_rotation.transpose(-1, -2))
    delta_rotvec = rotation_matrix_to_rotvec(delta_rotation)
    gripper_indices = torch.as_tensor(YAM_GRIPPER_INDICES, device=actions.device)
    grippers = actions[..., gripper_indices]
    converted = torch.cat((delta_position, delta_rotvec, grippers.unsqueeze(-1)), dim=-1)
    # [..., arm=2, 7] -> canonical [left7, right7]
    return converted.flatten(start_dim=-2)


def to_yam_delta_pose_actions_numpy(actions: np.ndarray, state: np.ndarray) -> np.ndarray:
    """NumPy wrapper used by stats preparation and SBD PCA fitting."""
    action_tensor = torch.as_tensor(np.asarray(actions), dtype=torch.float64)
    state_tensor = torch.as_tensor(np.asarray(state), dtype=torch.float64)
    return (
        to_yam_delta_pose_actions(action_tensor, state_tensor)
        .cpu()
        .numpy()
        .astype(np.float32)
    )
