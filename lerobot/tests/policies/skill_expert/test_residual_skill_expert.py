from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.skill_expert import residual_skill_expert as module
from lerobot.policies.skill_expert.residual_skill_expert import ResidualPatchSkillExpert
from lerobot.policies.skillVLA.dataset_skillVLA import SkillVLADataset


def tiny_model(monkeypatch, variant=1, checkpoint_layers=0):
    def initialize(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.width = 4
        self.gemma_expert = nn.Module()
        self.gemma_expert.model = nn.Module()
        self.gemma_expert.model.config = SimpleNamespace(num_hidden_layers=18)
        self.gemma_expert.model.norm = nn.Identity()
        self.action_out_proj = nn.Linear(4, 4)
        self.base_weight = nn.Parameter(torch.tensor(0.01))
        self.last_weight = nn.Parameter(torch.tensor(0.02))
        self.bridge_proprio_condition = nn.Linear(4, 4, bias=False)
        self.end_pose_condition = nn.Linear(3, 4, bias=False)
        self.start_pose_condition = nn.Linear(3, 4, bias=False)
        self._gradient_checkpointing = False
        self._vsa_debug_active = False
        self.calls = []
    monkeypatch.setattr(module.Both3SkillExpert, "__init__", initialize)
    monkeypatch.setattr(module, "layernorm_forward", lambda norm, hidden, condition: (norm(hidden), None))
    config = SimpleNamespace(architecture_label=f"residual_patch_{variant}", max_action_dim=4,
        chunk_size=2, visual_bridge_last_n_layers=1, skill_flow_checkpoint_layers=checkpoint_layers,
        cumulative_xyz_loss_enabled=False, num_inference_steps=3)
    model = ResidualPatchSkillExpert(config)
    model._time_condition = lambda t: t[:, None].expand(-1, 4)
    model._skill_broadcasts = lambda code: (None, code.float()[:, None].expand(-1, 4))
    model._action_tokens = lambda x: x
    model._action_velocity = lambda hidden: model.action_out_proj(hidden)
    model.sample_time = lambda batch, device: torch.full((batch,), 0.5, device=device)
    model.sample_noise = lambda shape, device: torch.randn(shape, device=device)
    model._residual_visual_cache = lambda *a: [None] * 18
    def geometry(hidden, gemma):
        del gemma
        b, n, _ = hidden.shape
        pos = torch.arange(n)[None].expand(b, -1)
        return torch.zeros(b, 1, n, n), pos, (pos[..., None].expand(-1, -1, 4),) * 2
    model._sequence_geometry = geometry
    def layer(self, i, hidden, mask, pos, cond, skill, latent, rotary):
        self.calls.append((i, hidden.shape[1], mask.shape[-1], pos[0].tolist()))
        weight = self.base_weight if i < 17 else self.last_weight
        return hidden + weight * (hidden.mean(dim=1, keepdim=True) + cond[:, None] + skill[:, None])
    model._expert_layer_with_latent_bridge = MethodType(layer, model)
    return model


@pytest.mark.parametrize("variant", range(1, 7))
def test_notion_direct_input_contracts(monkeypatch, variant):
    model = tiny_model(monkeypatch, variant)
    state = torch.randn(1, 4, requires_grad=True)
    goal = torch.randn(1, 6, requires_grad=True)
    base, last = model._residual_conditions(torch.ones(1), state, goal)
    base_grad = torch.autograd.grad(base.sum(), (state, goal), retain_graph=True, allow_unused=True)
    assert base_grad[0].abs().sum() > 0
    if variant == 6:
        assert base_grad[1] is None
    else:
        assert base_grad[1][:, :3].abs().sum() > 0
        assert bool(base_grad[1][:, 3:].abs().sum() > 0) == (variant <= 4)
    if not last.requires_grad:
        assert variant in {4, 5, 6}
        return
    last_grad = torch.autograd.grad(last.sum(), (state, goal), allow_unused=True)
    assert (last_grad[0] is not None) == (variant in {1, 3})
    assert (last_grad[1] is not None) == (variant in {1, 2})


@pytest.mark.parametrize("checkpoint_layers", [0, 2])
def test_zero_init_joint_forward_and_gradient_routing(monkeypatch, checkpoint_layers):
    torch.manual_seed(42)
    model = tiny_model(monkeypatch, checkpoint_layers=checkpoint_layers)
    actions = torch.randn(1, 2, 4)
    window = torch.randn(1, 8, 4)
    error = model([], torch.randn(1, 4), torch.tensor([1]), actions,
                  noise=torch.ones_like(actions), end_pose=torch.randn(1, 6), window_actions=window)
    skill_error = model.take_skill_window_residual()
    torch.testing.assert_close(error, skill_error[:, 2:4])
    assert [c[1] for c in model.calls] == [8] * 17 + [2]
    assert model.calls[-1][2:] == (2, [2, 3])
    main_grad = torch.autograd.grad(error.square().mean(), model.base_weight, retain_graph=True)[0]
    assert main_grad.abs() > 0  # no detach from action loss to base
    (error.square().mean() + skill_error[:, 1:5].square().mean()).backward()
    assert model.residual_action_head.weight.grad.abs().sum() > 0
    assert model.last_weight.grad == 0  # expected until zero-init head has moved
    metrics = model.residual_diagnostics(torch.ones(1, 2, dtype=torch.bool), 4)
    assert metrics["residual/residual_velocity_rms"] == 0
    assert metrics["residual/mse_improvement"] == 0


def test_rollout_denoises_full_window_and_returns_only_current_chunk(monkeypatch):
    model = tiny_model(monkeypatch)
    nn.Module.train(model, False)
    result = model.sample_actions([], torch.zeros(1, 4), torch.tensor([1]),
        noise=torch.ones(1, 8, 4), end_pose=torch.ones(1, 6), num_steps=3)
    assert result.shape == (1, 2, 4)
    assert len(model.calls) == 18 * 3
    assert all(c[1] == 8 for c in model.calls if c[0] < 17)


def test_trained_residual_head_survives_checkpoint_roundtrip(monkeypatch):
    first = tiny_model(monkeypatch, variant=6)
    with torch.no_grad():
        first.residual_action_head.weight.fill_(0.01)
        first.residual_action_head.bias.fill_(0.02)
    restored = tiny_model(monkeypatch, variant=6)
    restored.load_state_dict(first.state_dict(), strict=True)
    kwargs = dict(noise=torch.ones(1, 8, 4), end_pose=torch.ones(1, 6), num_steps=2)
    expected = first.sample_actions([], torch.zeros(1, 4), torch.tensor([1]), **kwargs)
    actual = restored.sample_actions([], torch.zeros(1, 4), torch.tensor([1]), **kwargs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_policy_combines_masked_base_and_unchanged_action_loss(monkeypatch):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.skill_expert.configuration_skill_expert import (
        SkillExpertConfig, LAYERWISE_COND_BOTTLENECK_ARCHITECTURE,
        LAYERWISE_COND_BOTTLENECK_CROSS_ATTENTION, RESIDUAL_PATCH_REVISION,
    )
    from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy
    model = tiny_model(monkeypatch)
    config = SkillExpertConfig(
        architecture_label="residual_patch_1", architecture=LAYERWISE_COND_BOTTLENECK_ARCHITECTURE,
        architecture_revision=RESIDUAL_PATCH_REVISION,
        vision_conditioning_mode=LAYERWISE_COND_BOTTLENECK_CROSS_ATTENTION,
        max_state_dim=4, max_action_dim=4, chunk_size=2, n_action_steps=1,
        visual_bottleneck_tokens=100, skill_flow_enabled=True,
        skill_flow_target="residual_window", skill_flow_max_length=8, skill_flow_state_conditioned=True,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(4,))},
        dtype="float32", device="cpu",
    )
    model.config = config
    policy = SkillExpertPolicy.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.config = config
    policy.model = model
    policy._collect_images = lambda batch: []
    policy._wrist_patch_alignment_loss = lambda batch, **kwargs: (None, {})
    batch = {
        "action": torch.randn(1, 2, 4), "observation.state": torch.randn(1, 4),
        "skill_code": torch.tensor([1]), "skill_start_state": torch.randn(1, 8),
        "skill_end_state": torch.randn(1, 8), "skill_end_state_valid": torch.tensor([True]),
        "skill_canonical_actions": torch.randn(1, 8, 4),
        "skill_canonical_action_is_pad": torch.tensor([[True, False, False, False, True, True, True, True]]),
        "action_is_pad": torch.tensor([[False, False]]),
    }
    loss, metrics = policy(batch)
    assert len(model.calls) == 18  # shared prefix, no second skill-flow pass
    assert metrics["residual/base_valid_fraction"] == 3 / 8
    assert metrics["residual/final_velocity_mse"] == pytest.approx(metrics["action_loss"])
    assert loss.item() == pytest.approx(metrics["action_loss"] + metrics["skill_flow/weighted"])
    assert model._last_skill_window_residual is None
    loss.backward()
    assert model.base_weight.grad.abs() > 0
    assert model.residual_action_head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("current,start,length,expected", [
    (0, 0, 5, [False, False, True, True, True, True, True, False]),
    (3, 3, 4, [False, False, True, True, True, True, False, False]),
    (6, 3, 4, [True, True, True, False, False, False, False, False]),
    (9, 3, 4, [False] * 8),
])
def test_window_loss_mask_handles_start_end_and_jitter(current, start, length, expected):
    dataset = object.__new__(SkillVLADataset)
    dataset._residual_action_horizon = 2
    dataset._canonical_action_cache = torch.arange(10).float()[:, None]
    dataset.meta = SimpleNamespace(episodes={0: {"length": 10, "dataset_from_index": 0}})
    dataset.reader = SimpleNamespace(_absolute_to_relative_idx=None)
    actions, pad = dataset._residual_skill_actions(episode_index=0, current_frame=current,
        skill_start_frame=start, skill_length=length)
    assert (~pad).tolist() == expected
    assert actions.shape == (8, 1)
    assert torch.count_nonzero(actions[pad]) == 0
