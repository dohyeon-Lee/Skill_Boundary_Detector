"""Wrist patch labels for the alignment head (geometry only, no dataset and no simulator)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lerobot.policies.skill_expert.wrist_patch_target import (
    WristCamera,
    patch_labels,
    project_into_wrist,
    rotation_from_axis_angle,
    soft_targets,
)

GRID, SIZE = 14, 256
CAMERA = WristCamera()


def _poses(count: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    xyz = torch.rand(count, 3, generator=generator) * 0.4 - 0.2
    axis_angle = torch.rand(count, 3, generator=generator) * 2.0 - 1.0
    axis_angle[:, 0] += np.pi                       # gripper roughly facing down, like LIBERO
    return xyz, axis_angle


def _forward(axis_angle: torch.Tensor) -> torch.Tensor:
    """Where the wrist camera looks: along the hand frame's +z, towards the gripper tip."""
    return rotation_from_axis_angle(axis_angle)[0] @ torch.tensor([0.0, 0.0, 1.0])


def test_rotation_matches_scipy_rodrigues() -> None:
    _, axis_angle = _poses(5)
    rotation = rotation_from_axis_angle(axis_angle)
    for index in range(axis_angle.shape[0]):
        vector = axis_angle[index].numpy()
        angle = np.linalg.norm(vector)
        axis = vector / angle
        cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        expected = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
        np.testing.assert_allclose(rotation[index].numpy(), expected, atol=1e-5)
    np.testing.assert_allclose(rotation_from_axis_angle(torch.zeros(1, 3)).numpy(), np.eye(3)[None], atol=1e-7)


def test_the_eef_is_always_the_same_pixel() -> None:
    """The camera is bolted to the gripper, so the EEF cannot move in its own image."""
    xyz, axis_angle = _poses(16)
    pixel, depth = project_into_wrist(xyz, axis_angle, xyz, CAMERA, height=SIZE, width=SIZE)
    assert torch.allclose(pixel, pixel[:1].expand_as(pixel), atol=1e-3)
    assert torch.allclose(depth, torch.full_like(depth, abs(CAMERA.position[2])), atol=1e-6)
    assert float(pixel[0, 1]) > SIZE / 2                    # the gripper sits low in the wrist view


def test_labels_hold_the_cell_and_drop_what_carries_no_signal() -> None:
    xyz, axis_angle = _poses(1)
    forward = _forward(axis_angle)

    ahead = xyz + 0.25 * forward
    label, pixel, valid = patch_labels(xyz, axis_angle, ahead, CAMERA, grid=GRID, height=SIZE, width=SIZE)
    assert bool(valid) and 0 <= int(label) < GRID * GRID
    assert int(label) == int(pixel[0, 1] // (SIZE / GRID)) * GRID + int(pixel[0, 0] // (SIZE / GRID))

    for dropped, goal in (
        ("behind the camera", xyz - 0.25 * forward),
        ("outside the frame", xyz + 0.25 * forward + 0.6 * (rotation := rotation_from_axis_angle(axis_angle)[0]) @ torch.tensor([1.0, 0.0, 0.0])),
        ("the gripper's own cell", xyz),
    ):
        label, _, valid = patch_labels(xyz, axis_angle, goal, CAMERA, grid=GRID, height=SIZE, width=SIZE)
        assert not bool(valid), f"{dropped} must carry no loss"
        assert int(label) == -1
    assert rotation.shape == (3, 3)


def test_soft_targets_are_a_distribution_peaked_on_the_cell() -> None:
    label = torch.tensor([GRID * GRID // 2 + 3, 0])
    targets = soft_targets(label, grid=GRID, sigma=0.7)
    assert targets.shape == (2, GRID * GRID)
    np.testing.assert_allclose(targets.sum(dim=-1).numpy(), [1.0, 1.0], atol=1e-6)
    assert int(targets[0].argmax()) == int(label[0])              # peak on the true cell
    assert 0.0 < float(targets[0, int(label[0]) + 1]) < float(targets[0, int(label[0])])
    hard = soft_targets(label, grid=GRID, sigma=0.0)
    assert hard[0, int(label[0])] == 1.0
    with pytest.raises(ValueError, match="mask the dropped frames"):
        soft_targets(torch.tensor([-1]), grid=GRID, sigma=0.7)


def test_a_calibrated_real_camera_replaces_the_simulator_constants() -> None:
    """Real robots pass hand-eye calibration + intrinsics instead of MuJoCo's fovy."""
    camera = WristCamera(position=(0.0, 0.0, -0.1), rotation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                         fovy=None, fx=200.0, fy=210.0, cx=120.0, cy=130.0,
                         distortion=(0.1, 0.0), mirror_x=False)
    xyz = torch.zeros(1, 3)
    axis_angle = torch.zeros(1, 3)
    pixel, depth = project_into_wrist(
        xyz, axis_angle, torch.tensor([[0.0, 0.0, -0.4]]), camera, height=SIZE, width=SIZE
    )
    assert float(depth) == pytest.approx(0.3, abs=1e-6)
    np.testing.assert_allclose(pixel.numpy(), [[120.0, 130.0]], atol=1e-4)   # on the principal point
