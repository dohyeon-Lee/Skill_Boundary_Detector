"""NewTask FT freeze contract, checked on a parameter-only stand-in model."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from torch import nn

from lerobot.policies.skill_expert.modeling_skill_expert import apply_newtask_ft_freeze


class _Expert(nn.Module):
    def __init__(self, depth: int):
        super().__init__()
        self.model = nn.Module()
        self.model.config = SimpleNamespace(num_hidden_layers=depth)
        self.model.layers = nn.ModuleList(nn.Linear(2, 2) for _ in range(depth))
        self.model.norm = nn.Linear(2, 2)


class _Model(nn.Module):
    def __init__(self, depth: int = 4):
        super().__init__()
        self.gemma_expert = _Expert(depth)
        self.cond_encoder = nn.Linear(2, 2)
        for name in (
            "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out",
            "skill_proj", "end_pose_condition",                      # skill-only route
            "image_proj", "state_proj", "visual_bridge_attention",
            "layerwise_condition_readers", "end_xyz_condition",
            "cond_end_pose_condition", "focus_uv_head", "termination_head",
        ):
            setattr(self, name, nn.Linear(2, 2))
        self.visual_bridge_gates = nn.Parameter(nn.Linear(2, 2).weight[0].detach().clone())
        self.mode_latent_mlp = None


def _trainable(model: nn.Module) -> set[str]:
    return {name.split(".weight")[0].split(".bias")[0] for name, p in model.named_parameters() if p.requires_grad}


@pytest.mark.parametrize("last_n", [1, 2])
def test_freezes_exactly_the_skill_only_route(last_n: int) -> None:
    model = _Model(depth=4)
    report = apply_newtask_ft_freeze(model, SimpleNamespace(visual_bridge_last_n_layers=last_n))
    assert report == {"frozen_expert_layers": 4 - last_n, "trainable_expert_layers": last_n}
    expected = {
        "cond_encoder", "image_proj", "state_proj", "visual_bridge_attention",
        "layerwise_condition_readers", "end_xyz_condition", "cond_end_pose_condition",
        "focus_uv_head", "termination_head", "visual_bridge_gates",
    } | {f"gemma_expert.model.layers.{index}" for index in range(4 - last_n, 4)}
    assert _trainable(model) == expected


def test_requires_a_frozen_core_layer() -> None:
    with pytest.raises(ValueError, match="at least one frozen"):
        apply_newtask_ft_freeze(_Model(depth=2), SimpleNamespace(visual_bridge_last_n_layers=2))
