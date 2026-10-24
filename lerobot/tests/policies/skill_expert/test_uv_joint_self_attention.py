"""Joint bottleneck ablation: camera scopes, real attention/gradients and old contracts."""

import pytest
import torch
from torch import nn

from lerobot.configs.uv_fullskill import UV_BSPLINE_JOINT_SELF_ATTENTION_LABEL as LABEL
from lerobot.policies.skill_expert.fullskill_bspline import FullskillBsplineSkillExpert
from lerobot.policies.skill_expert.layerwise_cond_bottleneck import _LayerwiseConditionReader
from test_cond_sdpa import _tiny_model
from test_fullskill_bspline import tiny_model
from test_stage1_vsa import _skill_config


BASELINE = "uv_fullskill_bspline_3_lastlayer_lastvis_current_preskill_diffaux"


def test_new_identity_inherits_baseline_config_and_rejects_old_revision():
    from dataclasses import asdict

    baseline, joint = _skill_config(BASELINE), _skill_config(LABEL)
    before, after = asdict(baseline), asdict(joint)
    assert {key for key in before if before[key] != after[key]} == {
        "architecture_label", "architecture_revision",
    }
    assert joint.uses_joint_bottleneck_self_attention
    assert not baseline.uses_joint_bottleneck_self_attention
    assert joint.uses_current_bspline_vision and joint.uses_bspline_preskill_memory
    assert joint.trains_agent_end_state_prediction and joint.uses_top_uv_goal
    with pytest.raises(ValueError, match="revision"):
        _skill_config(LABEL, architecture_revision=baseline.architecture_revision)


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_camera_reads_stay_separate_but_auxiliary_gradients_cross_all_queries(monkeypatch, joint, checkpoint):
    # Two real Cond-Gemma layers, with the production recurrent update and MHA.
    torch.manual_seed(42)
    model = _tiny_model(monkeypatch, "sdpa", checkpoint, model_class=FullskillBsplineSkillExpert)
    model.config.uses_joint_bottleneck_self_attention = joint
    model.layerwise_latent_queries = nn.Parameter(torch.randn(100, 16))
    cross_calls, self_calls = {}, {}
    for index, reader in enumerate(model.layerwise_condition_readers):
        cross_calls[index], self_calls[index] = [], []
        reader.cross_attention.register_forward_pre_hook(
            lambda module, args, index=index: cross_calls[index].append(
                (args[0].shape[1], args[1].detach().clone())
            )
        )
        reader.self_attention.register_forward_pre_hook(
            lambda module, args, index=index: self_calls[index].append(args[0].shape[1])
        )
    outputs = model._encode_layerwise_latents(torch.randn(2, 8, 32), torch.randn(2, 32))
    assert len(outputs) == 2 and outputs[-1].shape == (2, 100, 16)
    for index in cross_calls:
        assert [query for query, _ in cross_calls[index]] == [1, 1, 98]
        top, wrist, both = [memory for _, memory in cross_calls[index]]
        torch.testing.assert_close(top, both[:, :4])
        torch.testing.assert_close(wrist, both[:, 4:])
        assert self_calls[index] == ([100] if joint else [1, 1, 98])
    # Supervise only the Top auxiliary token: cross-group gradients are new.
    outputs[-1][:, 0].square().mean().backward()
    gradient = model.layerwise_latent_queries.grad
    assert torch.isfinite(gradient).all() and gradient[0].abs().sum() > 0
    if joint:
        assert gradient[1].abs().sum() > 0
        assert gradient[2:].abs().sum() > 0
    else:
        assert gradient[1:].count_nonzero() == 0


@pytest.mark.parametrize("bridge_layers", [1, 9, 17])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_joint_model_keeps_bridge_routes_preskill_detach_and_parameter_layout(monkeypatch, bridge_layers, checkpoint):
    options = dict(start_position="lastlayer", current_route="_lastvis_current_preskill",
                   uv=True, diffaux=True, bridge_layers=bridge_layers, checkpoint=checkpoint)
    baseline = tiny_model(monkeypatch, 3, **options)
    model = tiny_model(monkeypatch, 3, architecture_label=LABEL, **options)
    # Strict loading proves that no new attention block, gate or head was added.
    model.load_state_dict(baseline.state_dict(), strict=True)
    # The fixture caches a projected state during construction; rebuild it
    # after loading, just as production does on its next condition forward.
    model._bridge_condition_shift = model.bridge_proprio_condition(torch.randn(2, 32))
    split = 18 - bridge_layers
    assert model.start_bridge_layer == baseline.start_bridge_layer == split - 1
    assert model.current_bridge_layers == baseline.current_bridge_layers == tuple(range(split, 18))
    assert model.direct_current_proprio and model.variant == 3
    assert model.cond_end_pose_condition[0].in_features == 2
    assert model.end_pose_condition[0].in_features == 2
    hidden = torch.randn(2, 10, 32, requires_grad=True)
    latent = torch.randn(2, 100, 16, requires_grad=True)
    previous = torch.randn(2, 30, 16, requires_grad=True)
    model.set_skill_start_latent(latent)
    model.set_preskill_plan(previous, torch.tensor([False, True]))
    output = model._run_expert_with_layerwise_latents(
        hidden, torch.randn(2, 32), torch.randn(2, 32), [latent] * 18,
    )
    prediction = model.take_bspline_prediction()
    assert output.shape == hidden.shape and prediction.shape == (2, 30, 8)
    (output.square().mean() + prediction.square().mean()).backward()
    assert hidden.grad is not None and latent.grad is not None
    assert previous.grad is None


@pytest.mark.parametrize("label", [LABEL, BASELINE])
def test_preskill_encoder_uses_its_architecture_attention_scope(monkeypatch, label):
    model = tiny_model(monkeypatch, 3, architecture_label=label,
                       start_position="lastlayer", current_route="_lastvis_current_preskill",
                       uv=True, diffaux=True)
    model.config.chunk_size = 10
    model.config.max_action_dim = 32
    model._condition_tokens = lambda *args, **kwargs: torch.randn(2, 8, 32)
    model._project_condition_state = lambda *args, **kwargs: torch.randn(2, 32)
    model._expert_condition = lambda *args, **kwargs: torch.randn(2, 32)
    model._skill_broadcasts = lambda *args: (None, torch.randn(2, 32))
    model.layerwise_condition_readers = nn.ModuleList([
        # Existing reader parameters, reduced widths only for this CPU test.
        _LayerwiseConditionReader(16, 32, 2)
    ])
    model.layerwise_latent_queries = nn.Parameter(torch.randn(100, 16))
    model._encode_layerwise_latents = lambda tokens, condition: [
        model._update_layerwise_latent(0, model.layerwise_latent_queries[None].expand(2, -1, -1), tokens)
    ]
    calls = []
    model.layerwise_condition_readers[0].self_attention.register_forward_pre_hook(
        lambda module, args: calls.append(args[0].shape[1])
    )
    with torch.no_grad():
        plan = model.encode_preskill_plan([], torch.randn(2, 32), torch.tensor([1, 2]), torch.randn(2, 6))
    assert plan.shape == (2, 30, 16) and not plan.requires_grad
    assert calls == ([100] if label == LABEL else [1, 1, 98])
