"""Diffaux2/3 deployment scale, stop-gradient boundary and independent loss weights."""

import pytest
import torch
from torch import nn

from lerobot.configs.uv_fullskill import UV_BSPLINE_PREDICTED_XYZ_LABELS
from lerobot.policies.skill_expert import fullskill_bspline, modeling_skill_expert as impl
from test_fullskill_bspline import tiny_model
from test_stage1_vsa import _skill_config
from test_uv_fullskill_bspline import policy_for

LABELS = UV_BSPLINE_PREDICTED_XYZ_LABELS


@pytest.mark.parametrize("label", LABELS)
def test_eval_exposes_the_exact_xyz_passed_to_the_expert_without_repredicting(monkeypatch, label):
    model = tiny_model(monkeypatch, 3, architecture_label=label, uv=True, diffaux=True,
                       start_position="lastlayer", current_route="_lastvis_current_preskill").eval()
    model.config.chunk_end_state_q01 = [.1, .2, .3] + [-1.] * 5
    model.config.chunk_end_state_q99 = [.5, .6, .7] + [1.] * 5
    observed = []
    model.end_pose_condition.register_forward_pre_hook(lambda module, args: observed.append(args[0]))
    with torch.no_grad():
        model._condition_on_predicted_xyz(torch.zeros(2, 32), torch.randn(2, 100, 16))
    policy = policy_for(label)
    policy.model = model
    xyz = policy.get_inference_goal_xyz()
    torch.testing.assert_close(xyz, observed[-1])
    assert not xyz.requires_grad
    xyz.zero_()
    torch.testing.assert_close(policy.get_inference_goal_xyz(), observed[-1])


@pytest.mark.parametrize("label", LABELS)
def test_predicted_xyz_scale_and_expert_gradient_boundary(monkeypatch, label):
    model = tiny_model(monkeypatch, 3, architecture_label=label, uv=True, diffaux=True,
                       start_position="lastlayer", current_route="_lastvis_current_preskill")
    config = _skill_config(label)
    assert config.uses_joint_bottleneck_self_attention and config.uses_predicted_expert_xyz
    assert config.detaches_predicted_expert_xyz == ("_detach" in label)
    assert model.agent_end_state_head[-1].out_features == 3
    assert model.cond_end_pose_condition[0].in_features == 2
    assert model.end_pose_condition[0].in_features == 3
    model.config.chunk_end_state_q01 = [.1, .2, .3] + [-1.] * 5
    model.config.chunk_end_state_q99 = [.5, .6, .7] + [1.] * 5
    # Exercise the adapter beyond its intentional zero-initialized output.
    nn.init.normal_(model.end_pose_condition[-1].weight, std=.1)
    latent = torch.randn(2, 100, 16, requires_grad=True)
    xyz_index = 2 if config.uses_triple_auxiliary_prediction else 0
    normalized = model.agent_end_state_head(latent[:, xyz_index])
    observed = []
    model.end_pose_condition.register_forward_pre_hook(lambda module, args: observed.append(args[0]))
    condition = model._condition_on_predicted_xyz(torch.zeros(2, 32), latent)
    expected = (normalized + 1) * .2 + normalized.new_tensor([.1, .2, .3])
    torch.testing.assert_close(observed[0], expected)
    condition.square().mean().backward()
    assert model.end_pose_condition[-1].weight.grad.abs().sum() > 0
    if config.detaches_predicted_expert_xyz:
        assert model.agent_end_state_head[-1].weight.grad is None
    else:
        assert model.agent_end_state_head[-1].weight.grad.abs().sum() > 0
    # Auxiliary supervision still trains the head in BOTH variants.
    model.zero_grad(set_to_none=True)
    model.agent_end_state_head(latent[:, xyz_index]).square().mean().backward()
    assert model.agent_end_state_head[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("label", LABELS)
def test_xyz_target_only_and_exact_auxiliary_coefficients(monkeypatch, label):
    policy = policy_for(label)
    policy.config.chunk_end_state_q01 = [0.] * 8
    policy.config.chunk_end_state_q99 = [2.] * 8
    prediction = torch.ones(2, 3, requires_grad=True)
    wrist = torch.ones(2, 4, requires_grad=True)
    top = torch.ones(2, 4, requires_grad=True)

    class Readouts(nn.Module):
        def predict_training_end_state_and_wrist_patch(self):
            return prediction, wrist

        def predict_training_top_wrist_patch_and_xyz(self):
            return top, wrist, prediction

    policy.model = Readouts()
    monkeypatch.setattr(impl, "patch_labels", lambda *args, **kwargs: (
        torch.zeros(2, dtype=torch.long), None, torch.ones(2, dtype=torch.bool)))
    monkeypatch.setattr(impl, "wrist_patch_alignment_loss", lambda logits, *args, **kwargs: (
        logits.square().mean(), {}))
    target = torch.ones(2, 8)
    target[:, 3:] = float("nan")  # rotation/gripper must not enter XYZ3 MSE.
    batch = {"skill_decoder_state": torch.zeros(2, 8), "skill_end_state": target,
             "skill_end_state_valid": torch.ones(2, dtype=torch.bool),
             "skill_focus_uv": torch.zeros(2, 2), "skill_focus_valid": torch.ones(2, dtype=torch.bool)}
    loss, metrics = policy._wrist_patch_alignment_loss(batch, top_k=1)
    top_weight = 1.0 if label.removesuffix("_last").endswith(("_topw1", "_xyzw1")) else .1
    patch_sum = .2 if policy.config.uses_triple_auxiliary_prediction else .1
    torch.testing.assert_close(loss * policy.config.patch_alignment_objective_weight,
                               torch.tensor(patch_sum + top_weight))
    assert metrics["agent_end_state/raw_loss"] == pytest.approx(1.)
    assert metrics["agent_end_state/weight"] == top_weight
    assert "agent_end_state/rotation_mae_normalized" not in metrics
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.full_like(prediction, top_weight / 3))
    torch.testing.assert_close(wrist.grad, torch.full_like(wrist, .1 / 4))
    if policy.config.uses_triple_auxiliary_prediction:
        torch.testing.assert_close(top.grad, torch.full_like(top, .1 / 4))
        assert metrics["agent_patch/weight"] == .1
    else:
        assert top.grad is None


@pytest.mark.parametrize("label", LABELS)
def test_training_and_sampling_use_current_prediction_without_gt_xyz(monkeypatch, label):
    model = tiny_model(monkeypatch, 3, architecture_label=label, uv=True, diffaux=True,
                       start_position="lastlayer", current_route="_lastvis_current_preskill")
    model.config.chunk_size, model.config.max_action_dim = 10, 32
    model.config.skill_flow_checkpoint_layers = 0
    model.start_pose_condition = nn.Linear(3, 32)
    model._time_condition = lambda time: torch.zeros(time.shape[0], 32)
    model._skill_broadcasts = lambda *args: (None, torch.zeros(2, 32))
    model._action_velocity = lambda hidden: hidden
    monkeypatch.setattr(fullskill_bspline._LITGoalFreeCondMixin, "_project_condition_state",
                        lambda self, state, *args: torch.zeros(state.shape[0], 32))
    latent = torch.randn(2, 100, 16)
    # Held-start models retain a distinct prefix source; the deployed XYZ
    # prediction still comes from the current observation, as in diffaux2.
    if not model.config.uses_current_bspline_vision:
        model.set_skill_start_latent(torch.randn_like(latent))
    encodes = []
    def encode(tokens, state):
        encodes.append(1)
        return [latent] * 18
    model._encode_layerwise_latents = encode
    model.set_preskill_plan(torch.zeros(2, 30, 16), torch.zeros(2, dtype=torch.bool))
    observations = []
    model.end_pose_condition.register_forward_pre_hook(
        lambda module, args: observations.append(args[0].detach().clone()))
    # The goal packet has UV/valid/start only. No GT endpoint is available.
    policy = policy_for(label)
    batch = {"skill_focus_uv": torch.tensor([[.2, -.3]]).expand(2, -1),
             "skill_focus_valid": torch.ones(2, dtype=torch.bool),
             "skill_start_state": torch.tensor([[.1, .2, .3, 0., 0., 0., 0., 0.]]).expand(2, -1)}
    goal = policy._skill_delta_goal(batch)
    # Even poisoned GT XYZ cannot enter the deployed goal transport.
    batch["skill_end_state"] = torch.full((2, 8), float("nan"))
    torch.testing.assert_close(policy._skill_delta_goal(batch), goal)
    state, tokens, actions = torch.randn(2, 32), torch.randn(2, 8, 32), torch.randn(2, 10, 32)
    model._predict_velocity_from_condition(tokens, actions, state, torch.tensor([1, 2]), torch.ones(2), end_pose=goal)
    assert len(observations) == 1 and len(encodes) == 1
    xyz_index = 2 if model.config.uses_triple_auxiliary_prediction else 0
    torch.testing.assert_close(observations[0], model.agent_end_state_head(latent[:, xyz_index]))  # q01=-1,q99=1
    model.eval()
    with torch.no_grad():
        model._sample_with_condition_cache(tokens, actions, state, torch.tensor([1, 2]), 2, end_pose=goal)
    assert len(encodes) == 2 and len(observations) == 3
    torch.testing.assert_close(observations[-1], observations[0])
    model._condition_tokens = lambda *args, **kwargs: tokens
    plan = model.encode_preskill_plan([], state, torch.tensor([1, 2]), goal)
    assert plan.shape == (2, 30, 16) and not plan.requires_grad
    assert len(encodes) == 3 and len(observations) == 4
    torch.testing.assert_close(observations[-1], observations[0])
