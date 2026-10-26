"""Last-only vision must preserve the prefix, memory, heads and action depth."""

import pytest
import torch

from lerobot.configs.uv_fullskill import (
    UV_BSPLINE_DIFFAUX4_LAST_LAYER_BASE_LABELS,
    UV_BSPLINE_LAST_LAYER_BASE_LABELS,
)
from test_fullskill_bspline import tiny_model
from test_stage1_vsa import _skill_config

LAST_LAYER_BASELINES = {
    **UV_BSPLINE_LAST_LAYER_BASE_LABELS,
    **UV_BSPLINE_DIFFAUX4_LAST_LAYER_BASE_LABELS,
}


@pytest.mark.parametrize("label,baseline", LAST_LAYER_BASELINES.items())
def test_last_only_checkpoint_roundtrip_and_revision_isolation(tmp_path, label, baseline):
    from dataclasses import asdict
    from lerobot.configs.policies import PreTrainedConfig

    config, original = _skill_config(label), _skill_config(baseline)
    before, after = asdict(original), asdict(config)
    assert {key for key in before if before[key] != after[key]} == {
        "architecture_label", "architecture_revision",
    }
    with pytest.raises(ValueError, match="revision"):
        _skill_config(label, architecture_revision=original.architecture_revision)
    config.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.architecture_label == label
    assert restored.architecture_revision == config.architecture_revision
    assert restored.uses_current_bspline_vision == original.uses_current_bspline_vision
    assert restored.uses_bspline_preskill_memory == original.uses_bspline_preskill_memory
    assert restored.trains_agent_end_state_prediction == original.trains_agent_end_state_prediction


@pytest.mark.parametrize("label,baseline", LAST_LAYER_BASELINES.items())
@pytest.mark.parametrize("checkpoint", [False, True])
def test_last_only_vision_forward_backward_preserves_baseline(monkeypatch, label, baseline, checkpoint):
    current = "_current" in baseline
    diffaux = "_diffaux" in baseline
    route = "_lastvis_current_preskill" if current else "_lastvis"
    kwargs = dict(start_position="lastlayer", current_route=route, uv=True,
                  diffaux=diffaux, checkpoint=checkpoint)
    torch.manual_seed(17)
    original = tiny_model(monkeypatch, 3, architecture_label=baseline, **kwargs)
    torch.manual_seed(17)
    model = tiny_model(monkeypatch, 3, architecture_label=label, **kwargs)
    assert model.state_dict().keys() == original.state_dict().keys()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original.state_dict()[key], rtol=0, atol=0)
    config, old_config = _skill_config(label), _skill_config(baseline)
    for flag in ("uses_top_uv_goal", "uses_current_bspline_vision", "uses_bspline_preskill_memory",
                 "trains_agent_end_state_prediction", "uses_joint_bottleneck_self_attention",
                 "uses_predicted_expert_xyz", "reads_bspline_plan", "uses_gt_expert_xyz",
                 "uses_gemma_xyz_readout", "uses_uv_readout_token", "goal_input_noise_max_m",
                 "uses_start_xyz_auxiliary"):
        assert getattr(config, flag) == getattr(old_config, flag)
    if config.uses_gemma_xyz_readout:
        assert config.uses_joint_bottleneck_self_attention and config.uses_gt_expert_xyz
        assert not config.uses_uv_readout_token
        assert hasattr(model, "xyz_readout_token")
        assert model.agent_end_state_head[-1].out_features == 3
    assert not config.uses_predicted_expert_xyz
    assert model.start_bridge_layer == 8
    assert model.current_bridge_layers == (17,)
    assert model.current_vision_route == "final_last"
    assert original.current_bridge_layers == tuple(range(9, 18))
    start = torch.randn(2, 100, 16, requires_grad=True)
    latents = [torch.randn(2, 100, 16, requires_grad=True) for _ in range(18)]
    if not current:
        model.set_skill_start_latent(start)
    previous = torch.randn(2, 30, 16, requires_grad=True)
    if current:
        model.set_preskill_plan(previous, torch.tensor([False, True]))
    noise = torch.randn(2, 10, 32, requires_grad=True)
    condition = torch.randn(2, 32, requires_grad=True)
    skill = torch.randn(2, 32, requires_grad=True)
    calls, plans, reads = [], [], []
    run_layer = model._expert_layer_with_latent_bridge

    def record(index, hidden, mask, positions, cond, code, latent, rotary):
        calls.append((index, latent, cond))
        return run_layer(index, hidden, mask, positions, cond, code, latent, rotary)

    model._expert_layer_with_latent_bridge = record
    model.bspline_generator.register_forward_hook(lambda *args: plans.append(len(calls)))
    model.bspline_reader.register_forward_pre_hook(lambda module, args: reads.append(args[1]))
    output = model._run_expert_with_layerwise_latents(noise, condition, skill, latents)
    prediction = model.take_bspline_prediction()
    assert [index for index, _, _ in calls] == list(range(18))
    assert [index for index, latent, _ in calls if latent is not None] == [8, 17]
    assert calls[8][1] is (latents[-1] if current else start)
    assert calls[17][1] is latents[-1]
    assert all(torch.equal(cond, calls[9][2]) for _, _, cond in calls[9:])
    assert plans == [9] and len(reads) == 1
    assert output.shape == noise.shape and prediction.shape == (2, 30, 8)
    if current:
        torch.testing.assert_close(reads[0], previous)
        assert not reads[0].requires_grad
    # Checkpoint backward must retain its per-layer inputs after caches change.
    model.set_skill_start_latent(None)
    model._bridge_condition_shift = model._bspline_time_condition = None
    (output.square().mean() + prediction.square().mean()).backward()
    assert latents[-1].grad is not None and latents[-1].grad.abs().sum() > 0
    assert all(latent.grad is None for latent in latents[:-1])
    assert previous.grad is None
    assert (start.grad is None) == current
    assert model.visual_bridge_gates.grad[:17].count_nonzero() == 0
    assert model.visual_bridge_gates.grad[17].abs() > 0
    assert model.bspline_queries.grad is not None
    assert model.bridge_proprio_condition[-1].weight.grad is not None


@pytest.mark.parametrize("label", LAST_LAYER_BASELINES)
@pytest.mark.parametrize("suffix_depth", [1, 17])
def test_last_only_vision_keeps_configured_split(monkeypatch, label, suffix_depth):
    model = tiny_model(monkeypatch, 3, start_position="lastlayer", uv=True,
                       bridge_layers=suffix_depth, architecture_label=label)
    assert model.split_layer == 18 - suffix_depth
    assert model.start_bridge_layer == 17 - suffix_depth
    assert model.current_bridge_layers == (17,)
