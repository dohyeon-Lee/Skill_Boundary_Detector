"""Real tiny Gemma attention: SDPA preserves Cond masks, outputs and gradients."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.skill_expert import modeling_utils
from lerobot.policies.skill_expert.layerwise_cond_bottleneck import (
    LayerwiseCondBottleneckSkillExpert,
    CoreExitLayerwiseCondBottleneckSkillExpert,
    _LayerwiseConditionReader,
)


def _tiny_model(monkeypatch, backend, checkpoint=False, model_class=LayerwiseCondBottleneckSkillExpert):
    monkeypatch.setattr(
        modeling_utils, "get_gemma_config",
        lambda variant: SimpleNamespace(
            width=32, depth=2, mlp_dim=64, num_heads=4, num_kv_heads=2, head_dim=8,
        ),
    )
    model = object.__new__(model_class)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(cond_attention_backend=backend)
    model.cond_encoder = modeling_utils.build_gemma(
        "gemma_300m", use_adarms=True, attention_backend=backend,
    )
    # Avoid a vacuous comparison: zero-initialized AdaRMS gates would otherwise
    # bypass every attention/MLP update, including an accidentally causal SDPA.
    with torch.no_grad():
        for layer in model.cond_encoder.model.layers:
            nn.init.normal_(layer.input_layernorm.dense.weight, std=0.02)
            nn.init.normal_(layer.post_attention_layernorm.dense.weight, std=0.02)
            layer.input_layernorm.dense.bias[-32:].fill_(0.7)
            layer.post_attention_layernorm.dense.bias[-32:].fill_(0.6)
    model.layerwise_latent_queries = nn.Parameter(torch.randn(5, 16))
    model.layerwise_condition_memory_norm = nn.LayerNorm(32)
    model.layerwise_condition_readers = nn.ModuleList([
        _LayerwiseConditionReader(16, 32, 2) for _ in range(2)
    ])
    model._gradient_checkpointing = checkpoint
    model._vsa_debug_active = False
    return model


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sdpa_matches_eager_for_start_and_current_outputs_and_gradients(monkeypatch, checkpoint, dtype):
    torch.manual_seed(9)
    eager = _tiny_model(monkeypatch, "eager", checkpoint)
    sdpa = _tiny_model(monkeypatch, "sdpa", checkpoint)
    sdpa.load_state_dict(eager.state_dict(), strict=True)
    # Match production: Gemma uses BF16, narrow bottleneck readers stay FP32.
    for model in [eager, sdpa]:
        model.cond_encoder.to(dtype=dtype)
    tokens = [torch.randn(2, 7, 32).to(dtype) for _ in range(2)]
    states = [torch.randn(2, 32).to(dtype) for _ in range(2)]
    rtol, atol = (2e-2, 2e-3) if dtype == torch.bfloat16 else (1e-4, 2e-6)
    results = []
    for model in [eager, sdpa]:
        inputs = [t.clone().requires_grad_() for t in tokens]
        conditions = [s.clone().requires_grad_() for s in states]
        start, current = [
            model._encode_layerwise_latents(t, s)[-1]
            for t, s in zip(inputs, conditions, strict=True)
        ]
        (start.square().mean() + (start * current).mean() + current.square().mean()).backward()
        results.append((start, current, inputs, conditions))
    for first, second in zip(results[0][:2], results[1][:2], strict=True):
        torch.testing.assert_close(first, second, rtol=rtol, atol=atol)
    for first, second in zip(eager.parameters(), sdpa.parameters(), strict=True):
        assert (first.grad is None) == (second.grad is None)
        if first.grad is not None:
            torch.testing.assert_close(first.grad, second.grad, rtol=rtol, atol=atol)
    for group in [2, 3]:
        for first, second in zip(results[0][group], results[1][group], strict=True):
            assert first.grad.abs().sum() > 0
            torch.testing.assert_close(first.grad, second.grad, rtol=rtol, atol=atol)


def test_sdpa_uses_mask_free_bidirectional_attention_not_causal_lm(monkeypatch):
    model = _tiny_model(monkeypatch, "sdpa")
    calls = []
    original = torch.nn.functional.scaled_dot_product_attention

    def record(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", record)
    model._encode_layerwise_latents(torch.randn(2, 7, 32), torch.randn(2, 32))
    # Bottleneck MHA also calls SDPA; select Gemma calls by its explicit scale.
    cond_calls = [call for call in calls if "scale" in call]
    assert len(cond_calls) == 2
    assert all(call["attn_mask"] is None and call["is_causal"] is False for call in cond_calls)


def test_sdpa_does_not_change_default_expert_backend(monkeypatch):
    model = _tiny_model(monkeypatch, "sdpa")
    expert = modeling_utils.build_gemma("gemma_300m", use_adarms=True)
    assert all(layer.self_attn.config._attn_implementation == "sdpa"
               for layer in model.cond_encoder.model.layers)
    assert all(layer.self_attn.config._attn_implementation == "eager"
               for layer in expert.model.layers)


def _tiny_expert(monkeypatch, backend, checkpoint=False):
    model = _tiny_model(monkeypatch, backend, checkpoint, CoreExitLayerwiseCondBottleneckSkillExpert)
    model.gemma_expert = copy.deepcopy(model.cond_encoder)
    model.action_out_proj = nn.Linear(32, 7)
    model.config.visual_bridge_last_n_layers = 1
    model.config.skill_flow_checkpoint_layers = 1 if checkpoint else 0
    model._action_tokens = lambda x: x
    model._layer_uses_visual_bridge = lambda i: False
    return model


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("route", ["chunk", "fullskill"])
def test_expert_sdpa_matches_eager_with_padding_and_checkpointing(monkeypatch, checkpoint, dtype, route):
    torch.manual_seed(17)
    eager = _tiny_expert(monkeypatch, "eager", checkpoint)
    sdpa = _tiny_expert(monkeypatch, "sdpa", checkpoint)
    sdpa.load_state_dict(eager.state_dict(), strict=True)
    for model in [eager, sdpa]:
        model.gemma_expert.to(dtype=dtype)
    tokens = torch.randn(2, 7, 32).to(dtype)
    condition, skill = [torch.randn(2, 32).to(dtype) for _ in range(2)]
    valid = torch.tensor([[True] * 5 + [False] * 2, [True] * 3 + [False] * 4])
    mask = torch.where((valid[:, None, :, None] & valid[:, None, None, :]), 0., -2.3819763e38)
    positions = (valid.long().cumsum(-1) - 1).clamp_min(0)
    results = []
    for model in [eager, sdpa]:
        inputs = [x.clone().requires_grad_() for x in [tokens, condition, skill]]
        if route == "chunk":
            hidden = model._run_expert_with_layerwise_latents(*inputs, [None] * 2)
        else:
            hidden = model._skill_only_expert_hidden(inputs[0], mask, positions, *inputs[1:])
        output = model._action_velocity(hidden)
        output[valid].square().mean().backward()
        results.append((output, inputs))
    rtol, atol = (3e-2, 3e-3) if dtype == torch.bfloat16 else (1e-4, 2e-6)
    torch.testing.assert_close(results[0][0][valid], results[1][0][valid], rtol=rtol, atol=atol)
    for first, second in zip(eager.parameters(), sdpa.parameters(), strict=True):
        assert (first.grad is None) == (second.grad is None)
        if first.grad is not None:
            torch.testing.assert_close(first.grad, second.grad, rtol=rtol, atol=atol)
    for first, second in zip(results[0][1], results[1][1], strict=True):
        assert first.grad.abs().sum() > 0
        torch.testing.assert_close(first.grad, second.grad, rtol=rtol, atol=atol)


def test_sdpa_fullskill_padding_cannot_affect_valid_tokens(monkeypatch):
    model = _tiny_expert(monkeypatch, "sdpa")
    # The fixture intentionally omits DINO and the auxiliary modules accessed
    # by the full model's train()/eval() override. This probe only uses Gemma.
    nn.Module.train(model, False)
    tokens = torch.randn(2, 7, 32)
    valid = torch.tensor([[True] * 5 + [False] * 2, [True] * 3 + [False] * 4])
    mask = torch.where(valid[:, None, :, None] & valid[:, None, None, :], 0., -2.3819763e38)
    positions = (valid.long().cumsum(-1) - 1).clamp_min(0)
    cond, skill = torch.randn(2, 32), torch.randn(2, 32)
    original = model._skill_only_expert_hidden(tokens, mask, positions, cond, skill)
    changed = tokens.clone()
    changed[~valid] = torch.randn_like(changed[~valid]) * 100
    altered = model._skill_only_expert_hidden(changed, mask, positions, cond, skill)
    torch.testing.assert_close(original[valid], altered[valid])
