import torch

from lerobot.processor.yam_kinematics import (
    to_yam_delta_pose_actions,
    yam_arm_grasp_pose,
)


def test_yam_zero_joint_grasp_pose_matches_official_xml():
    position, rotation = yam_arm_grasp_pose(torch.zeros(6, dtype=torch.float64))
    expected_position = torch.tensor([0.245, 0.0, 0.164], dtype=torch.float64)
    expected_rotation = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    assert torch.allclose(position, expected_position, atol=1e-12)
    assert torch.allclose(rotation, expected_rotation, atol=1e-12)


def test_yam_delta_pose_is_sequential_and_keeps_absolute_grippers():
    state = torch.zeros(1, 14, dtype=torch.float64)
    actions = state[:, None].repeat(1, 3, 1)
    actions[:, 0, 0] = 0.1
    actions[:, 1, 0] = 0.2
    actions[:, 2, 0] = 0.15
    actions[..., 6] = torch.tensor([0.1, 0.2, 0.3])
    actions[..., 13] = torch.tensor([-0.1, -0.2, -0.3])
    converted = to_yam_delta_pose_actions(actions, state)

    # joint1 is a base-frame Z rotation, so its spatial rotation increments are
    # exactly the consecutive joint increments.
    assert torch.allclose(
        converted[0, :, 5], torch.tensor([0.1, 0.1, -0.05], dtype=torch.float64), atol=1e-7
    )
    assert torch.allclose(converted[0, :, 6], actions[0, :, 6])
    assert torch.allclose(converted[0, :, 13], actions[0, :, 13])

    # The SBD path passes an unbatched [T,14] chunk; it must have exactly the
    # same sequential interpretation as the training batch.
    unbatched = to_yam_delta_pose_actions(actions[0], state[0])
    assert torch.allclose(unbatched, converted[0], atol=1e-12)
