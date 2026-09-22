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
            "skill_proj", "end_pose_condition", "start_pose_condition",   # skill-only route
            "image_proj", "state_proj", "visual_bridge_attention",
            "layerwise_condition_readers", "end_xyz_condition",
            "cond_end_pose_condition", "focus_uv_head", "termination_head", "bridge_proprio_condition",
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
    assert report == {
        "frozen_expert_layers": 4 - last_n, "trainable_expert_layers": last_n, "action_head_trainable": 0,
    }
    expected = {
        "cond_encoder", "image_proj", "state_proj", "visual_bridge_attention",
        "layerwise_condition_readers", "end_xyz_condition", "cond_end_pose_condition",
        "focus_uv_head", "termination_head", "visual_bridge_gates", "bridge_proprio_condition",
    } | {f"gemma_expert.model.layers.{index}" for index in range(4 - last_n, 4)}
    assert _trainable(model) == expected


def test_requires_a_frozen_core_layer() -> None:
    with pytest.raises(ValueError, match="at least one frozen"):
        apply_newtask_ft_freeze(_Model(depth=2), SimpleNamespace(visual_bridge_last_n_layers=2))


def test_unfreezing_the_action_head_adds_only_the_final_norm_and_head() -> None:
    frozen, unfrozen = _Model(depth=4), _Model(depth=4)
    apply_newtask_ft_freeze(frozen, SimpleNamespace(visual_bridge_last_n_layers=1))
    report = apply_newtask_ft_freeze(
        unfrozen,
        SimpleNamespace(visual_bridge_last_n_layers=1, newtask_ft_unfreeze_action_head=True),
    )
    assert report["action_head_trainable"] == 1
    assert _trainable(unfrozen) - _trainable(frozen) == {"action_out_proj", "gemma_expert.model.norm"}
    # The rest of the skill-only route, including the Expert-side end-pose projection, stays frozen.
    assert not {"action_in_proj", "time_mlp_in", "time_mlp_out", "skill_proj", "end_pose_condition",
                "gemma_expert.model.layers.0"} & _trainable(unfrozen)


@pytest.mark.parametrize(("enabled", "unfreeze", "skipped"), [(False, False, False), (True, False, True), (True, True, False)])
def test_skill_flow_is_skipped_only_while_its_route_is_frozen(enabled: bool, unfreeze: bool, skipped: bool) -> None:
    from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy

    policy = SimpleNamespace(config=SimpleNamespace(
        newtask_ft_enabled=enabled, newtask_ft_unfreeze_action_head=unfreeze))
    assert SkillExpertPolicy._newtask_ft_skips_skill_flow(policy) is skipped


def test_full_unfreeze_freezes_nothing_and_keeps_the_skill_flow_loss() -> None:
    from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy

    model = _Model(depth=4)
    before = _trainable(model)
    report = apply_newtask_ft_freeze(
        model, SimpleNamespace(visual_bridge_last_n_layers=1, newtask_ft_full_unfreeze=True)
    )
    assert report == {"frozen_expert_layers": 0, "trainable_expert_layers": 4, "action_head_trainable": 1}
    assert _trainable(model) == before
    assert {"gemma_expert.model.layers.0", "action_in_proj", "skill_proj", "end_pose_condition"} <= before
    policy = SimpleNamespace(config=SimpleNamespace(
        newtask_ft_enabled=True, newtask_ft_unfreeze_action_head=False, newtask_ft_full_unfreeze=True))
    assert SkillExpertPolicy._newtask_ft_skips_skill_flow(policy) is False


@pytest.mark.parametrize(
    ("head", "full", "skill_flow_loss", "skipped"),
    [
        (False, False, True, True),    # default variant: frozen route, never computed
        (False, False, False, True),
        (True, False, True, False),
        (True, False, False, True),    # action loss only
        (False, True, True, False),
        (False, True, False, True),    # action loss only
    ],
)
def test_skill_flow_loss_switch(head: bool, full: bool, skill_flow_loss: bool, skipped: bool) -> None:
    from lerobot.datasets.factory import _newtask_ft_skips_skill_flow as dataset_rule
    from lerobot.policies.skill_expert.modeling_skill_expert import newtask_ft_skips_skill_flow

    config = SimpleNamespace(
        newtask_ft_enabled=True, newtask_ft_unfreeze_action_head=head,
        newtask_ft_full_unfreeze=full, newtask_ft_skill_flow_loss=skill_flow_loss,
    )
    assert newtask_ft_skips_skill_flow(config) is skipped
    assert dataset_rule(config) is skipped                      # targets follow the loss
    config.newtask_ft_enabled = False                           # ordinary Stage-1 training
    assert newtask_ft_skips_skill_flow(config) is False and dataset_rule(config) is False
