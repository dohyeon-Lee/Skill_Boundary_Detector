from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.skill_expert import residual_skill_expert as module
from lerobot.policies.skill_expert.residual_skill_expert import ResidualPatchSkillExpert
from lerobot.policies.skill_expert.direct_skill_expert import DirectPatchSkillExpert
from lerobot.policies.skillVLA.dataset_skillVLA import SkillVLADataset


def tiny_model(monkeypatch, variant=1, checkpoint_layers=0, family="residual", dgd=False, bridge_layers=1):
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
    config = SimpleNamespace(architecture_label=f"{family}_patch_{variant}" + ("_dGD" if dgd else ""), max_action_dim=4,
        chunk_size=2, visual_bridge_last_n_layers=bridge_layers, skill_flow_checkpoint_layers=checkpoint_layers,
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
        weight = self.base_weight if i < self.visual_bridge_start_layer else self.last_weight
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
@pytest.mark.parametrize("bridge_layers", [1, 9, 17])
def test_zero_init_joint_forward_and_gradient_routing(monkeypatch, checkpoint_layers, bridge_layers):
    torch.manual_seed(42)
    model = tiny_model(monkeypatch, checkpoint_layers=checkpoint_layers, bridge_layers=bridge_layers)
    actions = torch.randn(1, 2, 4)
    window = torch.randn(1, 8, 4)
    error = model([], torch.randn(1, 4), torch.tensor([1]), actions,
                  noise=torch.ones_like(actions), end_pose=torch.randn(1, 6), window_actions=window)
    skill_error = model.take_skill_window_residual()
    torch.testing.assert_close(error, skill_error[:, 2:4])
    assert [c[1] for c in model.calls] == [8] * (18 - bridge_layers) + [2] * bridge_layers
    assert model.calls[-1][2:] == (2, [2, 3])
    main_grad = torch.autograd.grad(error.square().mean(), model.base_weight, retain_graph=True)[0]
    assert main_grad.abs() > 0  # no detach from action loss to base
    (error.square().mean() + skill_error[:, 1:5].square().mean()).backward()
    assert model.residual_action_head.weight.grad.abs().sum() > 0
    assert model.last_weight.grad == 0  # expected until zero-init head has moved
    metrics = model.residual_diagnostics(torch.ones(1, 2, dtype=torch.bool), 4)
    assert metrics["residual/residual_velocity_rms"] == 0
    assert metrics["residual/mse_improvement"] == 0


@pytest.mark.parametrize("family", ["direct", "residual"])
@pytest.mark.parametrize("bridge_layers", [1, 9, 17])
def test_rollout_denoises_full_window_and_returns_only_current_chunk(monkeypatch, family, bridge_layers):
    model = tiny_model(monkeypatch, family=family, bridge_layers=bridge_layers)
    nn.Module.train(model, False)
    result = model.sample_actions([], torch.zeros(1, 4), torch.tensor([1]),
        noise=torch.ones(1, 8, 4), end_pose=torch.ones(1, 6), num_steps=3)
    assert result.shape == (1, 2, 4)
    assert len(model.calls) == 18 * 3
    assert all(c[1] == 8 for c in model.calls if c[0] < 18 - bridge_layers)
    assert all(c[1] == 2 and c[3] == [2, 3] for c in model.calls if c[0] >= 18 - bridge_layers)


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
@pytest.mark.parametrize("bridge_layers", [1, 9])
def test_direct_shared_head_readouts_and_first_step_gradients(monkeypatch, variant, checkpoint_layers, bridge_layers):
    torch.manual_seed(42)
    model = tiny_model(monkeypatch, variant, checkpoint_layers, family="direct", bridge_layers=bridge_layers)
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
    assert [c[1] for c in model.calls] == [8] * (18 - bridge_layers) + [2] * bridge_layers
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


def tiny_dgd_model(monkeypatch, variant, family, dgd, checkpoint_mode=0, bridge_layers=1):
    """Exercise the REAL AdaRMS/read-only decoder with trainable base inputs."""
    from lerobot.policies.pi_gemma import PiGemmaRMSNorm, layernorm_forward

    model = tiny_model(monkeypatch, variant, checkpoint_layers=checkpoint_mode, family=family, dgd=dgd,
                       bridge_layers=bridge_layers)
    monkeypatch.setattr(module, "layernorm_forward", layernorm_forward)
    model.gemma_expert.model.norm = PiGemmaRMSNorm(4, cond_dim=4)
    model.action_in_proj = nn.Linear(4, 4)
    model.time_mlp_in = nn.Linear(1, 4)
    model.time_mlp_out = nn.Linear(4, 4)
    model.skill_proj = nn.Linear(1, 4)
    model.visual_weight = nn.Parameter(torch.randn(4) * 0.1)
    model._action_tokens = lambda x: model.action_in_proj(x)
    model._time_condition = lambda t: model.time_mlp_out(torch.nn.functional.silu(model.time_mlp_in(t[:, None])))
    model._skill_broadcasts = lambda code: (None, model.skill_proj(code.float()[:, None]))
    model._residual_visual_cache = lambda *args: [None] * (18 - bridge_layers) + [model.visual_weight] * bridge_layers
    original = model._expert_layer_with_latent_bridge

    def layer(self, i, hidden, mask, pos, cond, skill, latent, rotary):
        hidden = original(i, hidden, mask, pos, cond, skill, latent, rotary)
        return hidden + latent if latent is not None else hidden

    model._expert_layer_with_latent_bridge = MethodType(layer, model)
    with torch.no_grad():
        model.gemma_expert.model.norm.dense.weight.normal_(std=0.03)
        if family == "residual":
            model.residual_action_head.weight.normal_(std=0.03)
    return model


@pytest.mark.parametrize("family", ["direct", "residual"])
@pytest.mark.parametrize("variant", range(1, 7))
@pytest.mark.parametrize("dgd", [False, True])
@pytest.mark.parametrize("bridge_layers", [1, 9])
def test_joint_route_matches_real_base_and_keeps_predictor_gradients(
    monkeypatch, family, variant, dgd, bridge_layers,
):
    torch.manual_seed(179)
    model = tiny_dgd_model(monkeypatch, variant, family, dgd, bridge_layers=bridge_layers)
    window, noise = torch.randn(2, 8, 4), torch.randn(2, 8, 4)
    state = torch.randn(2, 4, requires_grad=True)
    goal = torch.randn(2, 6, requires_grad=True)
    skill = torch.tensor([0.2, -0.3], requires_grad=True)
    time = torch.tensor([0.4, 0.8])
    model.sample_noise = lambda shape, device: noise.clone()
    model([], state, skill, window[:, 2:4], noise=noise[:, 2:4], time=time,
          end_pose=goal, window_actions=window)
    expected = model.take_skill_window_residual()
    model.calls.clear()
    # Freeze parameters, NOT the skill/XYZ input graph used to train Predictor.
    model.requires_grad_(False)
    mask = torch.tensor([[True, False, False, False, False, True, True, True]]).expand(2, -1)
    actual = model.skill_only_flow_residual(
        window, skill, mask, time=time, noise=noise, state=state, end_pose=goal,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(model.calls) == 18 - bridge_layers
    assert all(call[1:3] == (8, 8) for call in model.calls)
    # Padding belongs to loss only, never to teacher attention or token positions.
    unmasked = model.skill_only_flow_residual(
        window, skill, torch.zeros_like(mask), time=time, noise=noise, state=state, end_pose=goal,
    )
    torch.testing.assert_close(actual, unmasked, rtol=0, atol=0)
    from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy
    loss = SkillExpertPolicy._joint_masked_route_error(actual, mask, 4).mean()
    skill_grad, goal_grad, state_grad = torch.autograd.grad(loss, (skill, goal, state), allow_unused=True)
    assert skill_grad.abs().sum() > 0
    assert state_grad.abs().sum() > 0
    if variant == 6:
        assert goal_grad is None
    else:
        assert goal_grad[:, :3].abs().sum() > 0
        assert bool(goal_grad[:, 3:].abs().sum()) == (variant <= 4)
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("family", ["direct", "residual"])
@pytest.mark.parametrize("dgd", [False, True])
def test_nine_terminal_layers_read_matching_cond_latents(monkeypatch, family, dgd):
    model = tiny_dgd_model(monkeypatch, 1, family, dgd, bridge_layers=9)
    latents = [torch.randn(4, requires_grad=True) for _ in range(18)]
    model._residual_visual_cache = lambda *args: latents
    original = model._expert_layer_with_latent_bridge
    seen = []

    def layer(self, i, hidden, mask, pos, cond, skill, latent, rotary):
        if i < 9:
            assert latent is None
        else:
            assert latent is latents[i]
            seen.append(i)
        return original(i, hidden, mask, pos, cond, skill, latent, rotary)

    model._expert_layer_with_latent_bridge = MethodType(layer, model)
    error = model([], torch.randn(2, 4), torch.tensor([1, 2]), torch.randn(2, 2, 4),
                  end_pose=torch.randn(2, 6), window_actions=torch.randn(2, 8, 4))
    base_error = model.take_skill_window_residual()
    assert seen == list(range(9, 18))
    action_grads = torch.autograd.grad(error.square().mean(), latents, retain_graph=True, allow_unused=True)
    base_grads = torch.autograd.grad(base_error.square().mean(), latents, allow_unused=True)
    assert all(grad is None for grad in action_grads[:9])
    assert all(grad is not None and grad.abs().sum() > 0 for grad in action_grads[9:])
    assert all(grad is None for grad in base_grads)


@pytest.mark.parametrize("family", ["residual", "direct"])
@pytest.mark.parametrize("variant", range(1, 7))
@pytest.mark.parametrize("checkpoint_mode", [0, 2, "full"])
@pytest.mark.parametrize("bridge_layers", [1, 9])
def test_dgd_blocks_all_action_to_base_gradients_but_keeps_base_loss_and_terminal_learning(
    monkeypatch, family, variant, checkpoint_mode, bridge_layers,
):
    torch.manual_seed(713)
    models = [tiny_dgd_model(monkeypatch, variant, family, dgd, 0 if checkpoint_mode == "full" else checkpoint_mode,
                            bridge_layers=bridge_layers)
              for dgd in (False, True)]
    models[1].load_state_dict(models[0].state_dict(), strict=True)
    assert models[0].state_dict().keys() == models[1].state_dict().keys()
    args = ([], torch.randn(2, 4), torch.tensor([1, 2]), torch.randn(2, 2, 4))
    kwargs = dict(noise=torch.randn(2, 2, 4), time=torch.tensor([0.4, 0.8]),
                  end_pose=torch.randn(2, 6), window_actions=torch.randn(2, 8, 4))
    results = []
    for model in models:
        model._gradient_checkpointing = checkpoint_mode == "full"
        torch.manual_seed(19)
        error = model(*args, **kwargs)
        skill_error = model.take_skill_window_residual()
        parameters = dict(model.named_parameters())
        values = tuple(parameters.values())
        main = error.square().mean()
        base = skill_error.square().mean()
        action_grads = torch.autograd.grad(main, values, retain_graph=True, allow_unused=True)
        base_grads = torch.autograd.grad(base, values, retain_graph=True, allow_unused=True)
        combined_grads = torch.autograd.grad(main + base, values, allow_unused=True)
        action_grads = dict(zip(parameters, action_grads, strict=True))
        base_grads = dict(zip(parameters, base_grads, strict=True))
        combined_grads = dict(zip(parameters, combined_grads, strict=True))
        results.append((error.detach(), skill_error.detach(), action_grads, base_grads))
        terminal_names = {"last_weight", "visual_weight"}
        if family == "residual":
            terminal_names |= {"residual_action_head.weight", "residual_action_head.bias"}
        for name in terminal_names:
            assert action_grads[name] is not None and action_grads[name].abs().sum() > 0
            assert base_grads[name] is None
        if model.detach_action_base:
            for name in parameters.keys() - terminal_names:
                g = action_grads[name]
                assert g is None or torch.count_nonzero(g) == 0, name
                expected = base_grads[name]
                actual = combined_grads[name]
                if expected is None:
                    assert actual is None or torch.count_nonzero(actual) == 0, name
                else:
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            assert action_grads["base_weight"].abs() > 0
            assert action_grads["gemma_expert.model.norm.dense.weight"].abs().sum() > 0
            assert action_grads["action_out_proj.weight"].abs().sum() > 0
        for name in ("base_weight", "action_in_proj.weight", "time_mlp_in.weight", "skill_proj.weight",
                     "bridge_proprio_condition.weight", "gemma_expert.model.norm.dense.weight", "action_out_proj.weight"):
            assert base_grads[name] is not None and base_grads[name].abs().sum() > 0, name
        assert all(p.requires_grad for p in model.parameters())  # no global freeze toggles
    for i in (0, 1):
        torch.testing.assert_close(results[1][i], results[0][i], rtol=0, atol=0)
    for name, g in results[0][3].items():
        other = results[1][3][name]
        assert (g is None) == (other is None)
        if g is not None:
            torch.testing.assert_close(other, g, rtol=0, atol=0)


@pytest.mark.parametrize("family", ["residual", "direct"])
@pytest.mark.parametrize("variant", range(1, 7))
def test_dgd_rollout_checkpoint_and_frozen_ft_match_attached(monkeypatch, family, variant):
    torch.manual_seed(713)
    models = [tiny_dgd_model(monkeypatch, variant, family, dgd) for dgd in (False, True)]
    models[1].load_state_dict(models[0].state_dict(), strict=True)
    args = ([], torch.randn(2, 4), torch.tensor([1, 2]))
    kwargs = dict(noise=torch.randn(2, 8, 4), end_pose=torch.randn(2, 6), num_steps=3)
    torch.testing.assert_close(models[1].sample_actions(*args, **kwargs),
                               models[0].sample_actions(*args, **kwargs), rtol=0, atol=0)
    train_args = args + (torch.randn(2, 2, 4),)
    kwargs = dict(noise=torch.randn(2, 2, 4), end_pose=kwargs["end_pose"], window_actions=torch.randn(2, 8, 4))
    results = []
    for model in models:
        terminal_names = {"last_weight", "visual_weight"}
        if family == "residual":
            terminal_names |= {"residual_action_head.weight", "residual_action_head.bias"}
        for name, p in model.named_parameters():
            p.requires_grad_(name in terminal_names)
        torch.manual_seed(19)
        error = model(*train_args, **kwargs)
        error.square().mean().backward()
        results.append((error.detach(), {n: p.grad for n, p in model.named_parameters()}))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for name, g in results[0][1].items():
        other = results[1][1][name]
        assert (g is None) == (other is None)
        if g is not None:
            torch.testing.assert_close(other, g, rtol=0, atol=0)


@pytest.mark.parametrize("family", ["direct", "residual"])
@pytest.mark.parametrize("variant", range(1, 7))
@pytest.mark.parametrize("bridge_layers", [1, 9])
def test_dgd_action_only_optimizer_cannot_decay_or_update_base(monkeypatch, family, variant, bridge_layers):
    model = tiny_dgd_model(monkeypatch, variant, family, True, bridge_layers=bridge_layers)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    error = model([], torch.randn(2, 4), torch.tensor([1, 2]), torch.randn(2, 2, 4),
                  end_pose=torch.randn(2, 6), window_actions=torch.randn(2, 8, 4))
    error.square().mean().backward()
    terminal_names = {"last_weight", "visual_weight"}
    if family == "residual":
        terminal_names |= {"residual_action_head.weight", "residual_action_head.bias"}
    for name, p in model.named_parameters():
        assert (p.grad is not None) == (name in terminal_names), name
    optimizer.step()
    for name, p in model.named_parameters():
        if name in terminal_names:
            assert not torch.equal(before[name], p), name
        else:
            torch.testing.assert_close(before[name], p, rtol=0, atol=0)


def tiny_visual_path(model):
    """Use the real window visual-cache/probe methods with small Cond tensors."""
    model._condition_tokens = lambda images, **kwargs: torch.cat(images, dim=1)
    model._record_visual_debug = lambda tokens: None
    model._residual_visual_cache = MethodType(ResidualPatchSkillExpert._residual_visual_cache, model)

    def project(self, state, focus, skill, goal):
        self._bridge_condition_shift = self.bridge_proprio_condition(state)
        return self._bridge_condition_shift + skill[:, None].float() + goal[:, :3].sum(-1, keepdim=True)

    def encode(self, tokens, projected):
        torch.rand(())  # Deliberately consume RNG, without affecting the toy readout.
        latent = tokens + projected[:, None]
        self._final_agent_patch_tokens = tokens[:, :2]
        self._final_wrist_patch_tokens = tokens[:, 2:]
        self._final_dual_patch_query = latent[:, :2]
        return [latent] * 18

    model._project_condition_state = MethodType(project, model)
    model._encode_layerwise_latents = MethodType(encode, model)
    original_layer = model._expert_layer_with_latent_bridge

    def layer(self, i, hidden, mask, pos, cond, skill, latent, rotary):
        hidden = original_layer(i, hidden, mask, pos, cond, skill, latent, rotary)
        return hidden + self.last_weight * latent.mean(1, keepdim=True) if latent is not None else hidden

    model._expert_layer_with_latent_bridge = MethodType(layer, model)


@pytest.mark.parametrize("family", ["direct", "residual"])
@pytest.mark.parametrize("variant", range(1, 7))
@pytest.mark.parametrize("dgd", [False, True])
def test_window_shuffle_probes_preserve_training_graph_rng_and_readouts(monkeypatch, family, variant, dgd):
    torch.manual_seed(4)
    regular = tiny_model(monkeypatch, variant, checkpoint_layers=2, family=family, dgd=dgd)
    debug = tiny_model(monkeypatch, variant, checkpoint_layers=2, family=family, dgd=dgd)
    if family == "residual":
        with torch.no_grad():
            regular.residual_action_head.weight.fill_(0.03)
    debug.load_state_dict(regular.state_dict())
    for model in (regular, debug):
        tiny_visual_path(model)
    state, goal = torch.randn(2, 4), torch.randn(2, 6)
    images = [torch.randn(2, 2, 4), torch.randn(2, 2, 4)]
    actions, window = torch.randn(2, 2, 4), torch.randn(2, 8, 4)
    results = []
    for model, enabled in ((regular, False), (debug, True)):
        model._vsa_debug_active = enabled
        torch.manual_seed(19)
        error = model(images, state, torch.tensor([1, 5]), actions,
                      end_pose=goal, window_actions=window)
        # The original cache remains differentiable for the auxiliary patch loss.
        query = model._final_dual_patch_query
        assert query.requires_grad
        loss = error.square().mean() + model.take_skill_window_residual().square().mean() + query.square().mean()
        rng = torch.random.get_rng_state().clone()
        loss.backward()
        gradients = {name: p.grad.clone() if p.grad is not None else None for name, p in model.named_parameters()}
        results.append((error.detach(), loss.detach(), rng, gradients, query.detach()))
        assert model._window_debug_condition_tokens is None
        assert model._vsa_debug_active is enabled
        assert model._gradient_checkpointing is False
    for actual, expected in zip(results[1][:3], results[0][:3]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(results[1][4], results[0][4], rtol=0, atol=0)
    for name, expected in results[0][3].items():
        actual = results[1][3][name]
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not regular._last_vsa_debug_stats
    for name in ("top_image_shuffle", "wrist_image_shuffle", "both_images_shuffle",
                 "state_shuffle", "skill_shuffle", "end_pose_shuffle"):
        assert f"sensitivity/{name}/output_delta_rms" in debug._last_vsa_debug_stats
        value = debug._last_vsa_debug_stats[f"sensitivity/{name}/relative_output_delta"]
        assert isinstance(value, float) and value >= 0
    assert debug._last_vsa_debug_stats["sensitivity/state_shuffle/relative_output_delta"] > 0
    assert debug._last_vsa_debug_stats["sensitivity/skill_shuffle/relative_output_delta"] > 0


@pytest.mark.parametrize("family", ["direct", "residual"])
def test_window_probe_singleton_batch_is_skipped(monkeypatch, family):
    model = tiny_model(monkeypatch, family=family)
    tiny_visual_path(model)
    model._vsa_debug_active = True
    model([torch.ones(1, 2, 4)] * 2, torch.ones(1, 4), torch.tensor([1]),
          torch.ones(1, 2, 4), end_pose=torch.ones(1, 6), window_actions=torch.ones(1, 8, 4))
    assert len(model.calls) == 18
    assert not model._last_vsa_debug_stats
    assert model._window_debug_condition_tokens is None


def test_window_probe_failure_restores_caches_flags_and_rng(monkeypatch):
    model = tiny_model(monkeypatch, family="direct")
    model._vsa_debug_active = True
    model._gradient_checkpointing = True
    names = ("_bridge_condition_shift", "_final_agent_patch_tokens",
             "_final_wrist_patch_tokens", "_final_dual_patch_query")
    saved = {name: torch.randn(2, 4, requires_grad=True) for name in names}
    for name, tensor in saved.items():
        setattr(model, name, tensor)

    def fail(*args, **kwargs):
        for name in names:
            setattr(model, name, torch.zeros(2, 4))
        torch.rand(())
        raise RuntimeError("probe failed")

    model._predict_window_velocity_from_condition = fail
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="probe failed"):
        model._window_input_sensitivity_stats(
            predicted_velocity=torch.zeros(2, 2, 4), condition_tokens=torch.ones(2, 4, 4),
            noisy_actions=torch.ones(2, 8, 4), state=torch.ones(2, 4),
            skill_code=torch.tensor([1, 5]), time=torch.ones(2), end_pose=torch.ones(2, 6),
        )
    assert model._vsa_debug_active and model._gradient_checkpointing
    assert torch.equal(torch.random.get_rng_state(), rng)
    for name, tensor in saved.items():
        assert getattr(model, name) is tensor


@pytest.mark.parametrize("family", ["residual", "direct"])
@pytest.mark.parametrize("dgd", [False, True])
@pytest.mark.parametrize("bridge_layers", [1, 9])
def test_policy_combines_masked_base_and_unchanged_action_loss(monkeypatch, family, dgd, bridge_layers):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.skill_expert.configuration_skill_expert import (
        SkillExpertConfig, LAYERWISE_COND_BOTTLENECK_ARCHITECTURE,
        LAYERWISE_COND_BOTTLENECK_CROSS_ATTENTION, RESIDUAL_PATCH_REVISION, DIRECT_PATCH_REVISION,
    )
    from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy
    model = tiny_model(monkeypatch, family=family, dgd=dgd, bridge_layers=bridge_layers)
    config = SkillExpertConfig(
        architecture_label=f"{family}_patch_1" + ("_dGD" if dgd else ""), architecture=LAYERWISE_COND_BOTTLENECK_ARCHITECTURE,
        architecture_revision=DIRECT_PATCH_REVISION if family == "direct" else RESIDUAL_PATCH_REVISION,
        vision_conditioning_mode=LAYERWISE_COND_BOTTLENECK_CROSS_ATTENTION,
        max_state_dim=4, max_action_dim=4, chunk_size=2, n_action_steps=1,
        visual_bottleneck_tokens=100, visual_bridge_last_n_layers=bridge_layers, skill_flow_enabled=True,
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
