"""Three dedicated predictions, camera reads and the unchanged Expert routes."""

from dataclasses import asdict

import pytest
import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.uv_fullskill import UV_BSPLINE_DIFFAUX3_LABELS as LABELS
from lerobot.configs.uv_fullskill import UV_BSPLINE_GT_XYZ_LABELS
from lerobot.policies.skill_expert import modeling_skill_expert as impl
from lerobot.policies.skill_expert.fullskill_bspline import FullskillBsplineSkillExpert
from lerobot.policies.skill_expert.layerwise_cond_bottleneck import WristPatchAlignmentHead
from test_cond_sdpa import _tiny_model
from test_fullskill_bspline import tiny_model
from test_stage1_vsa import _skill_config
from test_uv_fullskill_bspline import policy_for


@pytest.mark.parametrize("label", LABELS)
def test_diffaux3_checkpoint_identity_and_roundtrip(tmp_path, label):
    config = _skill_config(label)
    old_label = "uv_bspline_lastlayer_current_preskill_diffaux2_detach"
    previous = _skill_config(old_label)
    before, after = asdict(previous), asdict(config)
    assert {key for key in before if before[key] != after[key]} == {
        "architecture_label", "architecture_revision",
    }
    assert config.trains_agent_patch_alignment and config.trains_agent_end_state_prediction
    assert config.uses_bspline_preskill_memory == ("_current_preskill_" in label)
    assert config.uses_current_bspline_vision == ("_current_preskill_" in label)
    assert config.uses_joint_bottleneck_self_attention and config.uses_gt_expert_xyz
    assert not config.uses_predicted_expert_xyz
    assert config.agent_end_state_loss_weight == (1. if "_xyzw1" in label else .1)
    with pytest.raises(ValueError, match="revision"):
        _skill_config(label, architecture_revision=previous.architecture_revision)
    config.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.architecture_label == label
    assert restored.uses_triple_auxiliary_prediction
    for key in ("agent_patch_align_head.query_proj.weight", "agent_end_state_head.1.weight"):
        assert impl._allowed_pi05_missing_key("model." + key, restored)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_diffaux3_camera_reads_and_joint_attention_each_layer(monkeypatch, checkpoint):
    model = _tiny_model(monkeypatch, "sdpa", checkpoint, model_class=FullskillBsplineSkillExpert)
    model.config.uses_joint_bottleneck_self_attention = True
    model.config.uses_triple_auxiliary_prediction = True
    model._alignment_token_count = 3
    model.layerwise_latent_queries = nn.Parameter(torch.randn(100, 16))
    cross, mixing = {}, {}
    for i, reader in enumerate(model.layerwise_condition_readers):
        cross[i], mixing[i] = [], []
        reader.cross_attention.register_forward_pre_hook(
            lambda module, args, i=i: cross[i].append((args[0].shape[1], args[1].detach().clone()))
        )
        reader.self_attention.register_forward_pre_hook(
            lambda module, args, i=i: mixing[i].append(args[0].shape[1])
        )
    latents = model._encode_layerwise_latents(torch.randn(2, 8, 32), torch.randn(2, 32))
    assert all(z.shape == (2, 100, 16) for z in latents)
    assert model._final_dual_patch_query.shape == (2, 3, 16)
    for i, calls in cross.items():
        assert [n for n, _ in calls] == [1, 1, 1, 97]
        top, wrist, xyz, action = [memory for _, memory in calls]
        torch.testing.assert_close(top, xyz[:, :4])
        torch.testing.assert_close(wrist, xyz[:, 4:])
        torch.testing.assert_close(xyz, action)
        assert mixing[i] == [100]
    latents[-1][:, 2].square().mean().backward()
    # XYZ auxiliary supervision reaches both camera tokens and action queries.
    gradient = model.layerwise_latent_queries.grad
    assert gradient[:2].abs().sum() > 0 and gradient[3:].abs().sum() > 0


@pytest.mark.parametrize("label", LABELS)
def test_diffaux3_three_heads_and_bridge_partition(monkeypatch, label):
    model = tiny_model(monkeypatch, 3, uv=True, diffaux=True, architecture_label=label,
                       start_position="lastlayer", current_route="_lastvis_current_preskill")
    assert model._alignment_token_count == 3
    assert model.start_bridge_layer == 8
    assert model.current_bridge_layers == ((17,) if label.endswith("_last") else tuple(range(9, 18)))
    model.agent_patch_align_head = WristPatchAlignmentHead(16, 32)
    model.wrist_patch_align_head = WristPatchAlignmentHead(16, 32)
    latent = torch.randn(2, 100, 16, requires_grad=True)
    top, wrist = (torch.randn(2, 4, 32, requires_grad=True) for _ in range(2))
    model._on_final_bottleneck_latent(latent)
    model._final_agent_patch_tokens, model._final_wrist_patch_tokens = top, wrist
    top_logits, wrist_logits, xyz = model.predict_training_top_wrist_patch_and_xyz()
    assert top_logits.shape == wrist_logits.shape == (2, 4)
    torch.testing.assert_close(xyz, model.agent_end_state_head(latent[:, 2]))
    assert model._final_dual_patch_query is None
    assert model._final_agent_patch_tokens is None and model._final_wrist_patch_tokens is None
    with pytest.raises(RuntimeError, match="preceding"):
        model.predict_training_top_wrist_patch_and_xyz()
    (top_logits.square().mean() + wrist_logits.square().mean() + xyz.square().mean()).backward()
    assert all(latent.grad[:, i].abs().sum() > 0 for i in range(3))
    assert latent.grad[:, 3:].count_nonzero() == 0
    reads = []
    for head in (model.start_bridge_attention, model.start_align_bridge_attention,
                 model.visual_bridge_attention, model.visual_align_bridge_attention):
        head.register_forward_pre_hook(lambda module, args: reads.append(args[1].shape[1]))
    hidden = torch.randn(2, 10, 32)
    model._visual_bridge_update(8, hidden, latent.detach())
    model._visual_bridge_update(model.current_bridge_layers[0], hidden, latent.detach())
    assert reads == [97, 3, 97, 3]


@pytest.mark.parametrize("label", LABELS)
@pytest.mark.parametrize("top_k", [1, 2])
def test_diffaux3_invisible_patches_do_not_mask_xyz_loss(monkeypatch, label, top_k):
    policy = policy_for(label)
    policy.config.chunk_end_state_q01, policy.config.chunk_end_state_q99 = [0.] * 8, [2.] * 8
    xyz = torch.ones(2 * top_k, 3, requires_grad=True)
    patch = torch.ones(2 * top_k, 4, requires_grad=True)
    policy.model = nn.Module()
    policy.model.predict_training_top_wrist_patch_and_xyz = lambda: (patch, patch, xyz)
    monkeypatch.setattr(impl, "patch_labels", lambda *args, **kwargs: (
        torch.zeros(2 * top_k, dtype=torch.long), None, torch.zeros(2 * top_k, dtype=torch.bool)))
    # Exercise the real patch loss/masking and real XYZ quantile MSE.
    batch = {
        "skill_decoder_state": torch.zeros(2, 8), "skill_end_state": torch.ones(2, 8),
        "skill_end_state_valid": torch.ones(2, dtype=torch.bool),
        "skill_focus_uv": torch.full((2, 2), float("nan")),
        "skill_focus_valid": torch.zeros(2, dtype=torch.bool),
    }
    loss, metrics = policy._wrist_patch_alignment_loss(batch, top_k=top_k)
    weight = 1. if "_xyzw1" in label else .1
    torch.testing.assert_close(loss, torch.tensor(weight))
    assert metrics["agent_patch/raw_loss"] == 0.
    assert metrics["wrist_patch/raw_loss"] == 0.
    loss.backward()
    assert xyz.grad.abs().sum() > 0 and patch.grad.count_nonzero() == 0


@pytest.mark.parametrize("label", LABELS)
def test_diffaux3_clipped_top_uv_keeps_border_patch_supervision(monkeypatch, label):
    policy = policy_for(label)
    policy.config.chunk_end_state_q01, policy.config.chunk_end_state_q99 = [0.] * 8, [2.] * 8
    xyz = torch.zeros(1, 3, requires_grad=True)
    top = torch.zeros(1, 4, requires_grad=True)
    wrist = torch.zeros(1, 4, requires_grad=True)
    policy.model = nn.Module()
    policy.model.predict_training_top_wrist_patch_and_xyz = lambda: (top, wrist, xyz)
    monkeypatch.setattr(impl, "patch_labels", lambda *args, **kwargs: (
        torch.zeros(1, dtype=torch.long), None, torch.zeros(1, dtype=torch.bool)))
    batch = {
        "skill_decoder_state": torch.zeros(1, 8),
        "skill_end_state": torch.ones(1, 8),
        "skill_end_state_valid": torch.ones(1, dtype=torch.bool),
        "skill_focus_uv": torch.tensor([[1., -.4]]),
        # False means the raw projection was clipped, not that the border UV is unusable.
        "skill_focus_valid": torch.zeros(1, dtype=torch.bool),
    }
    loss, metrics = policy._wrist_patch_alignment_loss(batch, top_k=1)
    assert metrics["agent_patch/valid_fraction"] == 1.
    assert metrics["wrist_patch/valid_fraction"] == 0.
    loss.backward()
    assert top.grad.abs().sum() > 0
    assert wrist.grad.count_nonzero() == 0


@pytest.mark.parametrize("label", UV_BSPLINE_GT_XYZ_LABELS)
@pytest.mark.parametrize("checkpoint", [False, True])
def test_diffaux3_bfloat16_action_plan_and_detached_memory(monkeypatch, label, checkpoint):
    model = tiny_model(monkeypatch, 3, uv=True, diffaux=True, architecture_label=label,
                       checkpoint=checkpoint, start_position="lastlayer",
                       current_route="_lastvis_current_preskill")
    model.gemma_expert.to(dtype=torch.bfloat16)
    nn.init.normal_(model.end_pose_condition[-1].weight, std=.1)
    previous = torch.randn(2, 30, 16, requires_grad=True)
    model.set_preskill_plan(previous, torch.tensor([False, True]))
    latent = torch.randn(2, 100, 16, requires_grad=True)
    start = torch.randn(2, 100, 16, requires_grad=True)
    model.set_skill_start_latent(start)
    calls, plans = [], []
    run_layer = model._expert_layer_with_latent_bridge

    def record(index, hidden, mask, positions, cond, code, vision, rotary):
        calls.append((index, vision))
        return run_layer(index, hidden, mask, positions, cond, code, vision, rotary)

    model._expert_layer_with_latent_bridge = record
    model.bspline_reader.register_forward_pre_hook(lambda module, args: plans.append(args[1]))
    noise = torch.randn(2, 10, 32, dtype=torch.bfloat16, requires_grad=True)
    condition, skill = (torch.randn(2, 32, dtype=torch.bfloat16) for _ in range(2))
    output = model._run_expert_with_layerwise_latents(noise, condition, skill, [latent] * 18)
    plan = model.take_bspline_prediction()
    assert output.dtype == torch.bfloat16 and plan.shape == (2, 30, 8)
    assert torch.isfinite(output).all() and torch.isfinite(plan).all()
    suffix = [17] if label.endswith("_last") else list(range(9, 18))
    assert [index for index, _ in calls] == list(range(18))
    assert [index for index, vision in calls if vision is not None] == [8, *suffix]
    current = "_current_preskill_" in label
    assert calls[8][1] is (latent if current else start)
    assert len(plans) == 1
    if current:
        torch.testing.assert_close(plans[0], previous)
        assert not plans[0].requires_grad
    else:
        assert plans[0].requires_grad
    # Backward may recompute after mutable state was overwritten by another pass.
    model._bridge_condition_shift = model._bspline_time_condition = None
    model.set_skill_start_latent(None)
    (output.float().square().mean() + plan.square().mean()).backward()
    assert previous.grad is None and latent.grad.abs().sum() > 0
    assert model.bspline_proprio_head[-1].weight.grad.abs().sum() > 0
    assert noise.grad.abs().sum() > 0
    assert (start.grad is None) == current
    if label.endswith("_last"):
        assert model.visual_bridge_gates.grad[:17].count_nonzero() == 0
        assert model.visual_bridge_gates.grad[17].abs() > 0
    assert model.agent_end_state_head[-1].weight.grad is None
