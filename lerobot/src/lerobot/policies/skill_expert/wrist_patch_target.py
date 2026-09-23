"""Where the skill-end EEF lands in the WRIST image, as a patch label for the alignment head.

The wrist camera is bolted to the gripper, so its pose is ``EEF pose o constant offset``: with the
per-frame EEF pose (raw ``observation.state``: xyz + axis-angle) and the skill-end xyz the policy is
already conditioned on, the target pixel follows from geometry alone — no dataset artifact and no
simulator. Both points share the same grounding, so the absolute origin cancels.

The label is the DINO patch that holds that pixel. Frames where the goal is out of view, behind the
camera, or inside the gripper's own patch carry no loss at all -- the same masking idea as
``mask_actions_after_skill_end``: "out of frame" is a single constant answer, so it teaches the
encoder nothing about *where* to look while being computable from the conditioning alone.

Nothing here is LIBERO-specific except the default constants: a real robot supplies its hand-eye
calibration and intrinsics through ``WristCamera`` instead (fx/fy/cx/cy and radial distortion
included).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

# LIBERO's Panda: robot0_eye_in_hand sits 5 cm ahead of the hand body and 9.7 cm behind the grip
# site, aligned with the hand frame (see stage1_eval/wrist_uv_probe, verified against MuJoCo).
LIBERO_WRIST_OFFSET = (0.05, 0.0, -0.097)
# Hand frame -> camera frame: the camera looks back along the hand's -z with x/y swapped
# (XML quat "0 0.707108 0.707108 0"), read once from the recorded model.
LIBERO_WRIST_ROTATION = (0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0)
LIBERO_WRIST_FOVY = 75.0


@dataclass(frozen=True)
class WristCamera:
    """Gripper-mounted camera: constant pose in the EEF rotation frame plus intrinsics.

    ``position``/``rotation`` are expressed in the frame of the recorded EEF *rotation* (LIBERO
    records the hand body) and relative to the recorded EEF *position* (the grip site).
    ``fovy`` fills square intrinsics; give ``fx``/``fy``/``cx``/``cy`` (and ``distortion``:
    ``(k1, k2)``) to use a calibrated real camera instead.
    """

    position: tuple[float, float, float] = LIBERO_WRIST_OFFSET
    rotation: tuple[float, ...] = LIBERO_WRIST_ROTATION
    fovy: float | None = LIBERO_WRIST_FOVY
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    distortion: tuple[float, float] = (0.0, 0.0)
    mirror_x: bool = True            # LIBERO stores the rendered image mirrored in x

    def intrinsics(self, *, height: int, width: int) -> tuple[float, float, float, float]:
        if self.fx is not None and self.fy is not None:
            return self.fx, self.fy, (self.cx if self.cx is not None else width / 2.0), (
                self.cy if self.cy is not None else height / 2.0
            )
        if self.fovy is None:
            raise ValueError("WristCamera needs fovy or fx/fy.")
        focal = 0.5 * height / math.tan(math.radians(self.fovy) / 2.0)
        return focal, focal, width / 2.0, height / 2.0


def rotation_from_axis_angle(axis_angle: Tensor) -> Tensor:
    """Batched Rodrigues rotation of ``[batch, 3]`` vectors whose norm is the angle."""
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / angle.clamp_min(1e-12)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    cross = torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
    ).reshape(*axis.shape[:-1], 3, 3)
    identity = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).expand_as(cross)
    sin, cos = torch.sin(angle).unsqueeze(-1), torch.cos(angle).unsqueeze(-1)
    rotation = identity + sin * cross + (1.0 - cos) * (cross @ cross)
    return torch.where(angle.unsqueeze(-1) < 1e-12, identity, rotation)


def project_into_wrist(
    eef_xyz: Tensor,
    eef_axis_angle: Tensor,
    target_xyz: Tensor,
    camera: WristCamera,
    *,
    height: int,
    width: int,
) -> tuple[Tensor, Tensor]:
    """(pixel ``[batch, 2]``, depth ``[batch]``) of ``target_xyz`` in the wrist image.

    Depth <= 0 means the goal is behind the camera; the pixel is meaningless there.
    """
    eef_xyz, eef_axis_angle, target_xyz = (
        tensor.to(dtype=torch.float32) for tensor in (eef_xyz, eef_axis_angle, target_xyz)
    )
    rotation = rotation_from_axis_angle(eef_axis_angle)
    offset = torch.tensor(camera.position, device=eef_xyz.device, dtype=torch.float32)
    camera_rotation = rotation @ torch.tensor(
        camera.rotation, device=eef_xyz.device, dtype=torch.float32
    ).reshape(3, 3)
    camera_xyz = eef_xyz + (rotation @ offset.unsqueeze(-1)).squeeze(-1)

    # MuJoCo cameras look down their own -z with +y up; robosuite's renderer flips both axes.
    axis_correction = torch.diag(
        torch.tensor([1.0, -1.0, -1.0], device=eef_xyz.device, dtype=torch.float32)
    )
    world_to_camera = (camera_rotation @ axis_correction).transpose(-1, -2)
    local = (world_to_camera @ (target_xyz - camera_xyz).unsqueeze(-1)).squeeze(-1)

    depth = local[:, 2]
    safe_depth = depth.clamp_min(1e-8)
    x, y = local[:, 0] / safe_depth, local[:, 1] / safe_depth
    k1, k2 = camera.distortion
    if k1 or k2:
        radius = x * x + y * y
        scale = 1.0 + k1 * radius + k2 * radius * radius
        x, y = x * scale, y * scale
    fx, fy, cx, cy = camera.intrinsics(height=height, width=width)
    column, row = fx * x + cx, fy * y + cy
    if camera.mirror_x:
        column = (width - 1) - column
    return torch.stack([column, row], dim=-1), depth


def patch_labels(
    eef_xyz: Tensor,
    eef_axis_angle: Tensor,
    target_xyz: Tensor,
    camera: WristCamera,
    *,
    grid: int,
    height: int,
    width: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Classification labels for one batch of frames.

    Returns ``(label, pixel, valid)``: ``label`` indexes the ``grid*grid`` cells row-major, and
    ``valid`` marks the frames the loss applies to. A frame is dropped when the goal is behind the
    camera or outside the frame (no spatial answer to learn) and when it falls in the gripper's own
    cell (the answer is then the fixed gripper pixel, so no looking is required). ``label`` is
    ``-1`` wherever ``valid`` is False.
    """
    pixel, depth = project_into_wrist(
        eef_xyz, eef_axis_angle, target_xyz, camera, height=height, width=width
    )
    column = torch.floor(pixel[:, 0] / width * grid).long()
    row = torch.floor(pixel[:, 1] / height * grid).long()
    in_view = (
        (depth > 1e-8)
        & (row >= 0) & (row < grid)
        & (column >= 0) & (column < grid)
    )
    cell = row.clamp(0, grid - 1) * grid + column.clamp(0, grid - 1)

    eef_pixel, _ = project_into_wrist(
        eef_xyz, eef_axis_angle, eef_xyz, camera, height=height, width=width
    )
    eef_cell = (
        torch.floor(eef_pixel[:, 1] / height * grid).long().clamp(0, grid - 1) * grid
        + torch.floor(eef_pixel[:, 0] / width * grid).long().clamp(0, grid - 1)
    )
    valid = in_view & (cell != eef_cell)
    return torch.where(valid, cell, torch.full_like(cell, -1)), pixel, valid


def soft_targets(label: Tensor, *, grid: int, sigma: float) -> Tensor:
    """``[batch, grid*grid]`` target distribution: a Gaussian of ``sigma`` cells around the label.

    ``label`` must hold real cells only (pass the rows ``patch_labels`` marked valid).
    """
    if bool(((label < 0) | (label >= grid * grid)).any()):
        raise ValueError("soft_targets expects valid cell labels; mask the dropped frames first.")
    targets = torch.zeros(label.shape[0], grid * grid, device=label.device, dtype=torch.float32)
    if sigma <= 0:
        targets[torch.arange(label.shape[0], device=label.device), label] = 1.0
        return targets
    axis = torch.arange(grid, device=label.device, dtype=torch.float32)
    cell_row = (label // grid).to(torch.float32)
    cell_column = (label % grid).to(torch.float32)
    row_weight = torch.exp(-((axis[None, :] - cell_row[:, None]) ** 2) / (2.0 * sigma**2))
    column_weight = torch.exp(-((axis[None, :] - cell_column[:, None]) ** 2) / (2.0 * sigma**2))
    weights = (row_weight[:, :, None] * column_weight[:, None, :]).reshape(-1, grid * grid)
    return weights / weights.sum(dim=-1, keepdim=True)
