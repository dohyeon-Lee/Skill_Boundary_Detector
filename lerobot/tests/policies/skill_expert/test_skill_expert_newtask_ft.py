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


@pytest.mark.parametrize("last_n", [1, 9])
@pytest.mark.parametrize("window", [False, True])
def test_freeze_expert_changes_only_terminal_layer_trainability(last_n, window) -> None:
    baseline, frozen = _Model(depth=18), _Model(depth=18)
    common = dict(visual_bridge_last_n_layers=last_n, uses_motion_window=window)
    apply_newtask_ft_freeze(baseline, SimpleNamespace(**common))
    report = apply_newtask_ft_freeze(
        frozen, SimpleNamespace(**common, newtask_ft_freeze_expert=True),
    )
    assert report == {
        "frozen_expert_layers": 18, "trainable_expert_layers": 0, "action_head_trainable": 0,
    }
    assert not any(p.requires_grad for p in frozen.gemma_expert.parameters())
    terminal = {f"gemma_expert.model.layers.{index}" for index in range(18 - last_n, 18)}
    assert _trainable(frozen) == _trainable(baseline) - terminal
    assert {"cond_encoder", "visual_bridge_attention", "layerwise_condition_readers"} <= _trainable(frozen)


def test_frozen_expert_keeps_action_gradients_to_cond_and_bridge_and_cannot_move() -> None:
    model = _Model(depth=4)
    apply_newtask_ft_freeze(
        model, SimpleNamespace(visual_bridge_last_n_layers=1, newtask_ft_freeze_expert=True),
    )
    before = {key: value.detach().clone() for key, value in model.gemma_expert.state_dict().items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    hidden = model.action_in_proj(torch.randn(3, 2))
    for layer in model.gemma_expert.model.layers[:-1]:
        hidden = layer(hidden)
    visual = model.visual_bridge_attention(model.cond_encoder(torch.randn(3, 2)))
    hidden = model.gemma_expert.model.layers[-1](hidden + visual)
    output = model.action_out_proj(model.gemma_expert.model.norm(hidden))
    output.square().mean().backward()
    assert model.cond_encoder.weight.grad.abs().sum() > 0
    assert model.visual_bridge_attention.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.gemma_expert.parameters())
    optimizer.step()
    for key, value in model.gemma_expert.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize("unfreeze", [
    "newtask_ft_unfreeze_action_head", "newtask_ft_unfreeze_skill_layers", "newtask_ft_full_unfreeze",
])
def test_frozen_expert_rejects_unfreezing_flags(unfreeze) -> None:
    config = SimpleNamespace(visual_bridge_last_n_layers=1, newtask_ft_freeze_expert=True, **{unfreeze: True})
    with pytest.raises(ValueError, match="cannot be combined"):
        apply_newtask_ft_freeze(_Model(), config)


@pytest.mark.parametrize("enabled", [False, True])
def test_frozen_expert_skill_loss_and_targets_follow_explicit_switch(enabled) -> None:
    from lerobot.datasets.factory import _newtask_ft_skips_skill_flow as dataset_rule
    from lerobot.policies.skill_expert.modeling_skill_expert import newtask_ft_skips_skill_flow

    config = SimpleNamespace(newtask_ft_enabled=True, newtask_ft_freeze_expert=True,
                             newtask_ft_skill_flow_loss=enabled)
    assert newtask_ft_skips_skill_flow(config) is (not enabled)
    assert dataset_rule(config) is (not enabled)


def test_requires_a_frozen_core_layer() -> None:
    with pytest.raises(ValueError, match="at least one frozen"):
        apply_newtask_ft_freeze(_Model(depth=2), SimpleNamespace(visual_bridge_last_n_layers=2))


@pytest.mark.parametrize("unfreeze_head", [False, True])
@pytest.mark.parametrize("bridge_layers", [1, 9, 17])
def test_window_ft_freezes_shared_base_proprio_and_respects_head_option(unfreeze_head, bridge_layers):
    model = _Model(depth=18)
    apply_newtask_ft_freeze(model, SimpleNamespace(visual_bridge_last_n_layers=bridge_layers,
        uses_motion_window=True, newtask_ft_unfreeze_action_head=unfreeze_head))
    assert not any(p.requires_grad for p in model.bridge_proprio_condition.parameters())
    assert not any(p.requires_grad for layer in model.gemma_expert.model.layers[:18 - bridge_layers]
                   for p in layer.parameters())
    assert all(p.requires_grad for layer in model.gemma_expert.model.layers[18 - bridge_layers:]
               for p in layer.parameters())
    assert all(p.requires_grad == unfreeze_head for p in model.action_out_proj.parameters())


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


def test_skill_flow_unfreezes_all_expert_layers_but_not_route_inputs() -> None:
    model = _Model(depth=4)
    report = apply_newtask_ft_freeze(
        model,
        SimpleNamespace(
            visual_bridge_last_n_layers=1,
            newtask_ft_unfreeze_action_head=True,
            newtask_ft_unfreeze_skill_layers=True,
        ),
    )
    assert report == {
        "frozen_expert_layers": 0,
        "trainable_expert_layers": 4,
        "action_head_trainable": 1,
    }
    trainable = _trainable(model)
    assert {f"gemma_expert.model.layers.{index}" for index in range(4)} <= trainable
    assert {"gemma_expert.model.norm", "action_out_proj"} <= trainable
    assert not {"action_in_proj", "time_mlp_in", "time_mlp_out", "skill_proj"} & trainable


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


def test_reset_initializes_legacy_predictor_metrics_for_joint_forward() -> None:
    from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy

    policy = object.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(n_action_steps=4)
    policy.reset()

    assert policy._last_predicted_skill_accuracy is None
    assert policy._last_predicted_diff_from_current is None
    assert policy._last_unique_predicted_skills is None


class _RouteModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("_fsq_levels", torch.tensor([3]))
        self.register_buffer("_fsq_strides", torch.tensor([1]))
        self.calls = []

    @staticmethod
    def _code_to_zq(code):
        return code.float().reshape(-1, 1) - 1

    @staticmethod
    def sample_noise(shape, device):
        return torch.zeros(shape, device=device)

    @staticmethod
    def sample_time(batch, device):
        return torch.full((batch,), 0.5, device=device)

    def skill_only_flow_residual(self, actions, code, mask, *, end_pose, time, noise, **kwargs):
        self.calls.append((kwargs, noise, time, end_pose.requires_grad))
        coordinates = code if code.ndim == 2 else self._code_to_zq(code)
        return (coordinates[:, 0] + end_pose.sum(-1))[:, None, None].expand_as(actions)


@pytest.mark.parametrize("state_conditioned", [False, True])
def test_joint_route_losses_keep_canonical_values_and_detach_hard_targets(state_conditioned):
    from lerobot.configs.types import PolicyFeature, FeatureType
    from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy
    from lerobot.policies.skillVLA.dataset_skillVLA import SKILL_CANONICAL_ACTIONS, SKILL_CANONICAL_ACTION_IS_PAD
    policy = object.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.model = _RouteModel()
    policy.config = SimpleNamespace(
        skill_flow_target="residual_window" if state_conditioned else "canonical",
        skill_flow_state_conditioned=state_conditioned,
        max_action_dim=4, max_state_dim=4,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(4,))},
        newtask_joint_route_timesteps=2, newtask_joint_route_ste_weight=0.1,
    )
    policy._joint_route_end_pose = lambda batch, xyz: xyz
    coordinates = torch.full((2, 1), 0.25, requires_grad=True)
    xyz = torch.full((2, 3), 0.1, requires_grad=True)
    hard, ste, metrics = policy._joint_route_losses(
        {SKILL_CANONICAL_ACTIONS: torch.zeros(2, 8, 4),
         SKILL_CANONICAL_ACTION_IS_PAD: torch.zeros(2, 8, dtype=torch.bool),
         "observation.state": torch.ones(2, 3)},
        gt_skill=torch.ones(2, dtype=torch.long), predicted_coordinates=coordinates,
        ste_coordinates=coordinates, predicted_xyz=xyz,
    )
    torch.testing.assert_close(hard, torch.full((2,), 0.25**2))
    torch.testing.assert_close(ste, torch.full((2,), 0.55**2))
    assert metrics["skill_predictor/route_target_vs_gt_accuracy"] == 1.0
    assert torch.autograd.grad(hard.sum(), xyz, allow_unused=True, retain_graph=True)[0] is None
    (hard.mean() + ste.mean()).backward()
    assert coordinates.grad.abs().sum() > 0 and xyz.grad.abs().sum() > 0
    assert len(policy.model.calls) == 8  # 3 candidates x 2 times, plus 2 STE evaluations
    assert all(call[1] is policy.model.calls[0][1] for call in policy.model.calls)
    assert [call[3] for call in policy.model.calls] == [False]*6 + [True]*2
    for kwargs, *_ in policy.model.calls:
        assert ("state" in kwargs) == state_conditioned
        if state_conditioned:
            torch.testing.assert_close(kwargs["state"], torch.tensor([[1., 1., 1., 0.]]).expand(2, -1))


class _PostEndTerminator(_IsolatedTerminator):
    def _prepare_image_tokens(self, top, wrist):
        return top.flatten(1).mean(-1, keepdim=True)

    def _forward_from_image_tokens(self, z_q, context, tokens, *, goal_xyz, start_state):
        logits = self.scale * (z_q.sum(-1) + tokens.sum(-1) + goal_xyz.sum(-1) + start_state.sum(-1))
        return torch.zeros_like(logits), logits


def test_joint_newterm_post_end_and_terminal_weight_are_isolated_from_predictor():
    from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy
    from lerobot.policies.skillVLA.newterm_supervision import POST_END_VALID, POST_END_START_STATE, TERMINAL_FRAME
    policy = object.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.model = _IsolatedJointModel()
    policy.model.fsq_term_train = _PostEndTerminator()
    policy.config = SimpleNamespace(
        newtask_joint_enabled=True, newtask_joint_terminator_enabled=True,
        terminator_start_proprio=True, terminator_goal_noise_max_m=0.,
        terminator_end_target_sigma=1., terminator_end_pos_weight=2., terminator_post_end_steps=2,
    )
    predicted_code = torch.tensor([1., 2.], requires_grad=True)
    predicted_xyz = torch.zeros(2, 3, requires_grad=True)
    past_codes = torch.tensor([0., 1., 0.], requires_grad=True)
    past_xyz = torch.zeros(3, 3, requires_grad=True)
    selected = []
    def predict(batch, rows, slots):
        selected.append((rows.tolist(), slots.tolist()))
        return past_codes, past_xyz
    policy._joint_post_end_predictor_outputs = predict
    batch = {
        "skill_de": torch.tensor([0, 3]), "observation.images.image": torch.zeros(2, 3, 4, 4),
        "terminator_start_state": torch.zeros(2, 8),
        POST_END_VALID: torch.tensor([[True, True], [True, False]]),
        POST_END_START_STATE: torch.zeros(2, 2, 8), TERMINAL_FRAME: torch.tensor([True, False]),
    }
    per_sample, metrics = policy._joint_terminator_loss(
        batch, predicted_code=predicted_code, predicted_xyz=predicted_xyz,
    )
    target = torch.exp(-batch["skill_de"].float().square() / 2)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        predicted_code.detach(), target, pos_weight=torch.tensor(2.), reduction="none",
    ) * torch.tensor([3., 1.]) + 2 * torch.log(torch.tensor(2.))
    torch.testing.assert_close(per_sample, expected)
    assert selected == [([0, 0, 1], [0, 1, 0])]
    assert metrics["terminator/post_end_conditions_per_frame"] == 1.0  # same-code past condition removed
    per_sample.mean().backward()
    assert policy.model.fsq_term_train.scale.grad is not None
    assert all(value.grad is None for value in (predicted_code, predicted_xyz, past_codes, past_xyz))
