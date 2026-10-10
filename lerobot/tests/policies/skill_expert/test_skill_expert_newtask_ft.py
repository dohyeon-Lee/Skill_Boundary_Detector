"""NewTask FT freeze contract, checked on a parameter-only stand-in model."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.skill_expert.modeling_skill_expert import (
    _build_state_dict,
    apply_newtask_ft_freeze,
)


def test_native_checkpoint_keeps_dsbc_head_at_policy_root() -> None:
    action = torch.ones(1)
    dsbc = torch.zeros(1)
    state, is_pi05 = _build_state_dict(
        {
            "gemma_expert.weight": action,
            "newtask_dsbc_head.output.weight": dsbc,
        },
        architecture="baseline",
    )

    assert is_pi05 is False
    assert state == {
        "model.gemma_expert.weight": action,
        "newtask_dsbc_head.output.weight": dsbc,
    }


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


def test_joint_keeps_canonical_targets_but_not_the_ordinary_vsa_skill_loss() -> None:
    from lerobot.datasets.factory import _newtask_ft_skips_skill_flow as dataset_rule
    from lerobot.policies.skill_expert.modeling_skill_expert import newtask_ft_skips_skill_flow

    config = SimpleNamespace(
        newtask_ft_enabled=True,
        newtask_joint_enabled=True,
        newtask_ft_unfreeze_action_head=False,
        newtask_ft_full_unfreeze=False,
    )
    # False here means the dataset retains canonical trajectories. Joint's
    # private VSA forward independently suppresses the ordinary skill loss.
    assert newtask_ft_skips_skill_flow(config) is False
    assert dataset_rule(config) is False


class _IsolatedTerminator(nn.Module):
    context_mode = "none"
    camera_mode = "top"
    goal_xyz = True
    state_dim = 8

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, z_q, context, top, wrist, *, goal_xyz):
        del context, wrist
        signal = z_q.float().sum(dim=-1) + goal_xyz.float().sum(dim=-1)
        signal = signal + top.float().flatten(1).mean(dim=-1)
        logits = self.scale * signal
        return torch.zeros_like(logits), logits


class _IsolatedJointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fsq_term_train = _IsolatedTerminator()
        self.register_buffer("_fsq_strides", torch.ones(1, dtype=torch.long))

    @staticmethod
    def _code_to_zq(code):
        return code.float().reshape(-1, 1)


def test_joint_terminator_loss_cannot_backpropagate_to_predictor_outputs() -> None:
    from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy

    policy = object.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.model = _IsolatedJointModel()
    policy.config = SimpleNamespace(
        newtask_joint_enabled=True,
        newtask_joint_terminator_enabled=True,
        terminator_start_proprio=False,
        terminator_goal_noise_max_m=0.0,
        terminator_end_target_sigma=2.0,
        terminator_end_pos_weight=1.0,
    )
    predicted_code = torch.tensor([1.0, 2.0], requires_grad=True)
    predicted_xyz = torch.randn(2, 3, requires_grad=True)
    per_sample, _ = policy._joint_terminator_loss(
        {
            "skill_de": torch.tensor([0, 3]),
            "observation.images.image": torch.randn(2, 3, 4, 4),
        },
        predicted_code=predicted_code,
        predicted_xyz=predicted_xyz,
    )
    per_sample.mean().backward()

    assert policy.model.fsq_term_train.scale.grad is not None
    assert predicted_code.grad is None
    assert predicted_xyz.grad is None
    assert policy.isolated_main_optimizer_grad_groups() == {
        "terminator": [policy.model.fsq_term_train.scale]
    }
    policy.newtask_dsbc_head = nn.Linear(1, 1)
    isolated = policy.isolated_main_optimizer_grad_groups()
    assert isolated["terminator"] == [policy.model.fsq_term_train.scale]
    assert isolated["dsbc"] == list(policy.newtask_dsbc_head.parameters())


def test_reset_initializes_legacy_predictor_metrics_for_joint_forward() -> None:
    from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy

    policy = object.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(n_action_steps=4)
    policy.reset()

    assert policy._last_predicted_skill_accuracy is None
    assert policy._last_predicted_diff_from_current is None
    assert policy._last_unique_predicted_skills is None
