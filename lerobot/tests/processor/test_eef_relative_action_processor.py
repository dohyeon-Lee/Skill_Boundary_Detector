import math

import pytest
import torch

from lerobot.processor.eef_relative_action_processor import (
    absolute_eef_actions_to_osc,
    matrix_to_rotation_vector,
    osc_actions_to_absolute_eef,
    rotation_vector_to_matrix,
    to_eef_absolute_actions,
    to_eef_relative_actions,
)


def test_rotation_vector_matrix_round_trip_including_near_pi():
    vectors = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1e-7, -2e-7, 3e-7],
            [0.2, -0.3, 0.4],
            [math.pi - 1e-5, 0.0, 0.0],
            [(math.pi - 1e-5) / math.sqrt(3)] * 3,
        ],
        dtype=torch.float64,
    )
    matrices = rotation_vector_to_matrix(vectors)
    restored = rotation_vector_to_matrix(matrix_to_rotation_vector(matrices))
    assert torch.allclose(restored, matrices, atol=2e-6, rtol=2e-6)


def test_eef_relative_round_trip_uses_left_world_rotation():
    anchor = torch.tensor([[0.4, -0.2, 1.1, 0.2, -0.1, 2.9, 0.01, -0.01]])
    relative_rotation = torch.tensor([0.15, -0.2, 0.1])
    anchor_rotation = rotation_vector_to_matrix(anchor[:, 3:6])
    target_rotation = rotation_vector_to_matrix(relative_rotation) @ anchor_rotation
    target = torch.tensor([[[0.43, -0.25, 1.12, 0.0, 0.0, 0.0, -1.0]]])
    target[..., 3:6] = matrix_to_rotation_vector(target_rotation).unsqueeze(1)

    relative = to_eef_relative_actions(target, anchor)
    assert torch.allclose(relative[..., :3], torch.tensor([[[0.03, -0.05, 0.02]]]), atol=1e-6)
    assert torch.allclose(
        rotation_vector_to_matrix(relative[..., 3:6]),
        rotation_vector_to_matrix(relative_rotation).unsqueeze(0).unsqueeze(0),
        atol=1e-6,
    )
    assert relative[..., 6].item() == -1.0
    restored = to_eef_absolute_actions(relative, anchor)
    assert torch.allclose(restored[..., :3], target[..., :3], atol=1e-6)
    assert torch.allclose(
        rotation_vector_to_matrix(restored[..., 3:6]), target_rotation.unsqueeze(1), atol=1e-6
    )


def test_osc_absolute_round_trip_matches_robosuite_scale_and_composition():
    state = torch.tensor([[0.3, 0.1, 1.0, 0.1, -0.2, 3.0, 0.02, -0.02]])
    osc = torch.tensor([[0.4, -0.5, 0.25, 0.2, -0.3, 0.1, 1.0]])
    absolute = osc_actions_to_absolute_eef(osc, state)
    restored = absolute_eef_actions_to_osc(absolute, state)
    assert torch.allclose(restored, osc, atol=2e-5, rtol=2e-5)


def test_osc_conversion_clips_only_pose_command_and_keeps_gripper_absolute():
    state = torch.zeros(1, 8)
    osc = torch.tensor([[2.0, -2.0, 0.0, 0.0, 0.0, 2.0, -1.0]])
    absolute = osc_actions_to_absolute_eef(osc, state)
    assert torch.allclose(absolute[0, :3], torch.tensor([0.05, -0.05, 0.0]))
    assert absolute[0, 6].item() == -1.0
    restored = absolute_eef_actions_to_osc(absolute, state)
    assert restored.abs().max().item() <= 1.0


def test_invalid_pose_shapes_fail_loudly():
    with pytest.raises(ValueError, match="7D"):
        to_eef_relative_actions(torch.zeros(2, 6), torch.zeros(2, 8))
