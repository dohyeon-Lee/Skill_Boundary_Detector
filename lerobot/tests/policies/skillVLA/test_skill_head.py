import torch

from lerobot.policies.skillVLA.skill_head import SkillHead


def test_skill_head_coordinate_quantization_has_hard_forward_and_ste_gradient() -> None:
    head = SkillHead(hidden_dim=4, fsq_levels=[3, 3, 3])
    coordinates = torch.tensor(
        [[-0.8, 0.2, 0.9], [0.49, -0.51, 0.0]], requires_grad=True
    )

    code, hard, ste = head.quantize_coordinates(coordinates)

    torch.testing.assert_close(
        hard,
        torch.tensor([[-1.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
    )
    torch.testing.assert_close(ste, hard)
    torch.testing.assert_close(code, torch.tensor([21, 10]))
    ste.sum().backward()
    torch.testing.assert_close(coordinates.grad, torch.ones_like(coordinates))
