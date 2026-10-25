"""Shared metre noise, GT conditioning, and Gemma XYZ readout contracts."""

import pytest
import torch
from torch import nn

from lerobot.configs.uv_fullskill import (
    UV_BSPLINE_GT_XYZ_LABELS as LABELS, UV_BSPLINE_DIFFAUX4_LABELS,
)
from lerobot.policies.skill_aux.terminator_uv import TOP_CAMERA, project_goal_uv
from lerobot.policies.skill_expert import fullskill_bspline, modeling_skill_expert as impl
from lerobot.policies.skill_expert.fullskill_bspline import FullskillBsplineSkillExpert
from lerobot.policies.skill_expert.wrist_patch_alignment import WristPatchAlignmentHead
from test_cond_sdpa import _tiny_model
from test_fullskill_bspline import tiny_model
from test_stage1_vsa import _skill_config
from test_uv_fullskill_bspline import policy_for


def goal_batch():
    return {
        "skill_end_state": torch.tensor([[.1, .2, 1., 0., 0., 0., 0., 0.]]).repeat(2, 1),
        "skill_start_state": torch.zeros(2, 8),
        "skill_end_state_valid": torch.ones(2, dtype=torch.bool),
        # Deliberately stale cached UV: training must reproject XYZ instead.
        "skill_focus_uv": torch.ones(2, 2), "skill_focus_valid": torch.ones(2, dtype=torch.bool),
        TOP_CAMERA: torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]]).repeat(2, 1, 1),
    }


@pytest.mark.parametrize("label", LABELS)
def test_gt_input_noise_is_shared_before_projection_and_disabled_at_eval(monkeypatch, label):
    policy = policy_for(label)
    batch = goal_batch()
    original = {k: v.clone() for k, v in batch.items()}
    monkeypatch.setattr(torch, "rand_like", lambda x: torch.full_like(x, .75))
    goal = policy._skill_delta_goal(batch, require_valid=True)
    assert goal.shape == (2, 9)
    torch.testing.assert_close(goal[:, 6:], batch["skill_end_state"][:, :3] + .005)
    uv, _ = project_goal_uv(goal[:, 6:], batch[TOP_CAMERA])
    torch.testing.assert_close(goal[:, :2], uv)
    torch.testing.assert_close(goal[:, 3:6], batch["skill_start_state"][:, :3])
    for key, value in original.items():
        torch.testing.assert_close(batch[key], value)
    with pytest.raises(KeyError, match="projection"):
        policy._skill_delta_goal({k: v for k, v in batch.items() if k != TOP_CAMERA})
    policy.eval()
    clean = policy._skill_delta_goal(batch)
    torch.testing.assert_close(clean[:, 6:], batch["skill_end_state"][:, :3])
    uv, _ = project_goal_uv(batch["skill_end_state"][:, :3], batch[TOP_CAMERA])
    torch.testing.assert_close(clean[:, :2], uv)
    runtime = policy._skill_delta_goal({k: v for k, v in batch.items() if k != TOP_CAMERA})
    torch.testing.assert_close(runtime[:, :2], batch["skill_focus_uv"])


@pytest.mark.parametrize("label", LABELS)
def test_expert_receives_gt_metres_and_never_auxiliary_predictions(monkeypatch, label):
    model = tiny_model(monkeypatch, 3, uv=True, diffaux=True, architecture_label=label,
                       start_position="lastlayer", current_route="_lastvis_current_preskill")
    model._time_condition = lambda t: torch.zeros(t.shape[0], 32)
    model.start_pose_condition = nn.Linear(3, 32)
    packet = policy_for(label)._skill_delta_goal(goal_batch())
    observed = []
    model.end_pose_condition.register_forward_pre_hook(lambda module, args: observed.append(args[0]))
    condition = model._expert_condition(torch.ones(2), end_pose=packet)
    torch.testing.assert_close(observed[0], packet[:, 6:])
    assert model._condition_on_predicted_xyz(condition, torch.randn(2, 100, 16)) is condition
    condition.square().mean().backward()
    assert model.agent_end_state_head[-1].weight.grad is None
    assert model.end_pose_condition[-1].weight.grad is not None


@pytest.mark.parametrize("label", UV_BSPLINE_DIFFAUX4_LABELS)
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gemma_readout_token_attention_is_separate_from_bottleneck(monkeypatch, label, checkpoint, dtype):
    model = tiny_model(monkeypatch, 3, uv=True, diffaux=True, architecture_label=label,
                       start_position="lastlayer", current_route="_lastvis_current_preskill")
    core = _tiny_model(monkeypatch, "sdpa", checkpoint, model_class=FullskillBsplineSkillExpert)
    model.cond_encoder = core.cond_encoder.to(dtype=dtype)
    model.layerwise_condition_memory_norm = core.layerwise_condition_memory_norm
    model.layerwise_condition_readers = core.layerwise_condition_readers
    model.layerwise_latent_queries = nn.Parameter(torch.randn(100, 16))
    model._gradient_checkpointing = checkpoint
    model.agent_patch_align_head = WristPatchAlignmentHead(16, 32)
    model.wrist_patch_align_head = WristPatchAlignmentHead(16, 32)
    assert model.agent_end_state_head[-1].in_features == 32
    packet = policy_for(label)._skill_delta_goal(goal_batch())
    monkeypatch.setattr(fullskill_bspline._LITGoalFreeCondMixin, "_project_condition_state",
                        lambda self, state, *args: torch.zeros(state.shape[0], 32))
    if not model.config.uses_uv_readout_token:
        nn.init.normal_(model.cond_end_pose_condition[-1].weight, std=.1)
    projected = model._project_condition_state(torch.zeros(2, 32), end_pose=packet)
    changed = packet.clone(); changed[:, :2] += .2
    other = model._project_condition_state(torch.zeros(2, 32), end_pose=changed)
    if model.config.uses_uv_readout_token:
        assert isinstance(model.cond_end_pose_condition, nn.Identity)
        torch.testing.assert_close(projected, other)
    else:
        assert not torch.equal(projected, other)
    model._project_condition_state(torch.zeros(2, 32), end_pose=packet)
    reads, mixing, gemma_lengths = [], [], []
    for reader in model.layerwise_condition_readers:
        reader.cross_attention.register_forward_pre_hook(
            lambda module, args: reads.append((args[0].shape[1], args[1].shape[1])))
        reader.self_attention.register_forward_pre_hook(lambda module, args: mixing.append(args[0].shape[1]))
    model.cond_encoder.model.layers[0].self_attn.register_forward_pre_hook(
        lambda module, args: gemma_lengths.append(args[0].shape[1]))
    images = torch.randn(2, 10, 32, dtype=dtype, requires_grad=True)
    latents = model._encode_layerwise_latents(images, projected.to(dtype))
    assert gemma_lengths[0] == 11
    assert reads == [(1, 5), (1, 5), (98, 10)] * 2
    assert mixing == [100, 100]
    top, wrist, xyz = model.predict_training_top_wrist_patch_and_xyz()
    assert top.shape == wrist.shape == (2, 4) and xyz.shape == (2, 3)
    xyz.square().mean().backward(retain_graph=True)
    assert model.layerwise_latent_queries.grad is None
    assert images.grad[:, :5].abs().sum() > 0 and images.grad[:, 5:].abs().sum() > 0
    model.zero_grad(set_to_none=True)
    (top.square().mean() + wrist.square().mean() + xyz.square().mean() + latents[-1].square().mean()).backward()
    token_param = model.xyz_uv_token[0].weight if model.config.uses_uv_readout_token else model.xyz_readout_token
    assert token_param.grad is not None and torch.isfinite(token_param.grad).all()
    assert token_param.grad.abs().sum() > 0
    assert model.layerwise_latent_queries.grad.abs().sum() > 0
    assert model._final_xyz_readout is None


@pytest.mark.parametrize("label", LABELS)
def test_clean_xyz_loss_and_exact_three_auxiliary_weights(monkeypatch, label):
    policy = policy_for(label)
    policy.config.chunk_end_state_q01, policy.config.chunk_end_state_q99 = [0.] * 8, [2.] * 8
    xyz = torch.ones(2, 3, requires_grad=True)
    patch = torch.ones(2, 4, requires_grad=True)
    policy.model = nn.Module()
    policy.model.predict_training_top_wrist_patch_and_xyz = lambda: (patch, patch, xyz)
    monkeypatch.setattr(impl, "patch_labels", lambda *args, **kwargs: (
        torch.zeros(2, dtype=torch.long), None, torch.ones(2, dtype=torch.bool)))
    monkeypatch.setattr(impl, "wrist_patch_alignment_loss", lambda logits, *args, **kwargs: (logits.square().mean(), {}))
    batch = goal_batch(); batch["skill_end_state"][:] = 1.; batch["skill_decoder_state"] = torch.zeros(2, 8)
    batch["skill_end_state"][:, 3:] = float("nan")
    loss, metrics = policy._wrist_patch_alignment_loss(batch, top_k=1)
    torch.testing.assert_close(loss, torch.tensor(.3))
    assert policy.config.patch_alignment_objective_weight == 1.
    assert metrics["agent_end_state/weight"] == .1
    loss.backward()
    torch.testing.assert_close(xyz.grad, torch.full_like(xyz, .1 / 3))


@pytest.mark.parametrize("label", UV_BSPLINE_DIFFAUX4_LABELS)
def test_gemma_variants_config_roundtrip_and_warm_start(tmp_path, label):
    from lerobot.configs.policies import PreTrainedConfig
    config = _skill_config(label)
    config.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.uses_gemma_xyz_readout and restored.uses_gt_expert_xyz
    assert not restored.uses_triple_auxiliary_prediction and not restored.uses_predicted_expert_xyz
    assert restored.uses_uv_readout_token == label.endswith("_uvtoken")
    for key in ("model.agent_end_state_head.1.weight", "model.xyz_readout_token", "model.xyz_uv_token.0.weight"):
        assert impl._allowed_pi05_missing_key(key, restored)


@pytest.mark.parametrize("label", [x for x in LABELS if "_current_preskill_" in x])
def test_preskill_uses_its_own_gt_endpoint_and_shared_noise(monkeypatch, label):
    from lerobot.policies.skillVLA import preskill_plan as mem
    policy = policy_for(label)
    policy.model = nn.Module()
    policy.model.anchor = nn.Parameter(torch.zeros(1))
    observed, saved = [], []
    policy.model.encode_preskill_plan = lambda images, state, code, goal: (
        observed.append(goal.detach().clone()) or torch.ones(
            state.shape[0], policy.config.bspline_tokens, policy.config.visual_bottleneck_width))
    policy.model.set_preskill_plan = lambda plan, valid: saved.append((plan, valid))
    monkeypatch.setattr(policy, "_collect_images", lambda batch: [])
    monkeypatch.setattr(torch, "rand_like", lambda x: torch.full_like(x, .75))
    batch = goal_batch()
    previous_end = batch["skill_end_state"].clone(); previous_end[:, 0] = -.4
    batch.update({mem.VALID: torch.tensor([False, True]), mem.IMAGE: torch.zeros(2, 3, 4, 4),
                  mem.WRIST_IMAGE: torch.zeros(2, 3, 4, 4), mem.START_STATE: torch.zeros(2, 8),
                  mem.END_STATE: previous_end, mem.FOCUS_UV: torch.zeros(2, 2),
                  mem.FOCUS_VALID: torch.ones(2, dtype=torch.bool),
                  mem.STATE_NORMALIZED: torch.zeros(2, 8), mem.CODE: torch.ones(2, dtype=torch.long)})
    policy._prepare_preskill_memory(batch)
    torch.testing.assert_close(observed[0][:, 6:], previous_end[1:, :3] + .005)
    uv, _ = project_goal_uv(observed[0][:, 6:], batch[TOP_CAMERA][1:])
    torch.testing.assert_close(observed[0][:, :2], uv)
    assert saved[0][0][0].count_nonzero() == 0 and not saved[0][0].requires_grad
