from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.skill_expert import residual_skill_expert as module
from lerobot.policies.skill_expert.residual_skill_expert import ResidualPatchSkillExpert
from lerobot.policies.skill_expert.direct_skill_expert import DirectPatchSkillExpert
from lerobot.policies.skillVLA.dataset_skillVLA import SkillVLADataset


def tiny_model(monkeypatch, variant=1, checkpoint_layers=0, family="residual"):
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
    config = SimpleNamespace(architecture_label=f"{family}_patch_{variant}", max_action_dim=4,
        chunk_size=2, visual_bridge_last_n_layers=1, skill_flow_checkpoint_layers=checkpoint_layers,
        cumulative_xyz_loss_enabled=False, num_inference_steps=3)
    model_class = DirectPatchSkillExpert if family == "direct" else ResidualPatchSkillExpert
    model = model_class(config)
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
@pytest.mark.parametrize("family", ["residual", "direct"])
def test_notion_direct_input_contracts(monkeypatch, variant, family):
    model = tiny_model(monkeypatch, variant, family=family)
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


@pytest.mark.parametrize("variant", range(1, 7))
@pytest.mark.parametrize("checkpoint_layers", [0, 2])
def test_direct_shared_head_readouts_and_first_step_gradients(monkeypatch, variant, checkpoint_layers):
    torch.manual_seed(42)
    model = tiny_model(monkeypatch, variant, checkpoint_layers, family="direct")
    assert not hasattr(model, "residual_action_head")
    assert not any("residual_action_head" in k for k in model.state_dict())
    readouts = []
    hook = model.action_out_proj.register_forward_hook(
        lambda module, inputs, output: readouts.append((inputs[0], output))
    )
    state, goal = torch.randn(1, 4), torch.randn(1, 6)
    actions = torch.randn(1, 2, 4)
    error = model([], state, torch.tensor([1]), actions,
                  noise=torch.ones_like(actions), end_pose=goal, window_actions=torch.randn(1, 8, 4))
    hook.remove()
    skill_error = model.take_skill_window_residual()
    assert [x[0].shape[1] for x in readouts] == [8, 2]  # same head invoked twice
    target = torch.ones_like(actions) - actions
    torch.testing.assert_close(error, target - readouts[1][1])
    # Not base + direct: base is nonzero but is not added to the deployed velocity.
    assert not torch.allclose(error, target - readouts[0][1][:, 2:4] - readouts[1][1])
    assert [c[1] for c in model.calls] == [8] * 17 + [2]
    assert model.calls[-1][2:] == (2, [2, 3])
    main = error.square().mean()
    auxiliary = skill_error.square().mean()
    action_grads = torch.autograd.grad(main, (model.base_weight, model.last_weight,
        model.action_out_proj.weight), retain_graph=True)
    assert all(g.abs().sum() > 0 for g in action_grads)  # no detach, no zero-init head
    base_grads = torch.autograd.grad(auxiliary, (model.base_weight, model.last_weight,
        model.action_out_proj.weight), retain_graph=True, allow_unused=True)
    assert base_grads[0].abs() > 0 and base_grads[2].abs().sum() > 0
    assert base_grads[1] is None  # base loss does not train the terminal layer
    (main + auxiliary).backward()
    metrics = model.residual_diagnostics(torch.ones(1, 2, dtype=torch.bool), 4)
    assert set(k.split("/")[0] for k in metrics) == {"direct"}
    assert metrics["direct/final_velocity_mse"] == pytest.approx(main.item())


def test_direct_rollout_replaces_central_velocity_in_one_shared_window(monkeypatch):
    model = tiny_model(monkeypatch, family="direct")
    windows = []

    def velocity(window, *args):
        windows.append(window.clone())
        base = torch.full_like(window, 2)
        direct = torch.full_like(window[:, 2:4], 7)
        combined = base.clone()
        combined[:, 2:4] = model._combine_central_velocity(base[:, 2:4], direct)
        return combined, base, direct

    model._run_residual_velocity = velocity
    result = model.sample_actions([], torch.zeros(1, 4), torch.tensor([1]),
        noise=torch.ones(1, 8, 4), end_pose=torch.ones(1, 6), num_steps=2)
    assert len(windows) == 2
    torch.testing.assert_close(result, torch.full_like(result, -6))  # 1 - 7, NOT 1 - (2 + 7)
    torch.testing.assert_close(windows[1][:, :2], torch.zeros(1, 2, 4))
    torch.testing.assert_close(windows[1][:, 4:], torch.zeros(1, 4, 4))
    torch.testing.assert_close(windows[1][:, 2:4], torch.full((1, 2, 4), -2.5))


def test_direct_shared_head_checkpoint_roundtrip(monkeypatch):
    first = tiny_model(monkeypatch, variant=6, family="direct")
    restored = tiny_model(monkeypatch, variant=6, family="direct")
    restored.load_state_dict(first.state_dict(), strict=True)
    kwargs = dict(noise=torch.ones(1, 8, 4), end_pose=torch.ones(1, 6), num_steps=2)
    expected = first.sample_actions([], torch.zeros(1, 4), torch.tensor([1]), **kwargs)
    actual = restored.sample_actions([], torch.zeros(1, 4), torch.tensor([1]), **kwargs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("family", ["residual", "direct"])
def test_policy_combines_masked_base_and_unchanged_action_loss(monkeypatch, family):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.skill_expert.configuration_skill_expert import (
        SkillExpertConfig, LAYERWISE_COND_BOTTLENECK_ARCHITECTURE,
        LAYERWISE_COND_BOTTLENECK_CROSS_ATTENTION, RESIDUAL_PATCH_REVISION, DIRECT_PATCH_REVISION,
    )
    from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy
    model = tiny_model(monkeypatch, family=family)
    config = SkillExpertConfig(
        architecture_label=f"{family}_patch_1", architecture=LAYERWISE_COND_BOTTLENECK_ARCHITECTURE,
        architecture_revision=DIRECT_PATCH_REVISION if family == "direct" else RESIDUAL_PATCH_REVISION,
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
    assert metrics[f"{family}/base_valid_fraction"] == 3 / 8
    assert metrics[f"{family}/final_velocity_mse"] == pytest.approx(metrics["action_loss"])
    assert loss.item() == pytest.approx(metrics["action_loss"] + metrics["skill_flow/weighted"])
    assert model._last_skill_window_residual is None
    loss.backward()
    assert model.base_weight.grad.abs() > 0
    head = model.action_out_proj if family == "direct" else model.residual_action_head
    assert head.weight.grad.abs().sum() > 0
    if family == "direct":
        assert model.last_weight.grad.abs() > 0
        assert "direct/shared_head_grad_norm" in policy.training_debug_metrics()
        assert not any(k.startswith("residual/") for k in metrics)


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
