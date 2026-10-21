"""UV goal transport, isolation, masking and all four inherited action routes."""

import pytest
import torch
from torch import nn

from lerobot.configs.uv_fullskill import UV_BSPLINE_BASE_LABELS, UV_BSPLINE_DIFFAUX_LABELS
from lerobot.policies.skill_expert import fullskill_bspline
from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy
from lerobot.policies.skillVLA import preskill_plan as memory
from test_fullskill_bspline import tiny_model
from test_stage1_vsa import _skill_config


def policy_for(label):
    policy = SkillExpertPolicy.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.config = _skill_config(label)
    return policy


@pytest.mark.parametrize("label", UV_BSPLINE_BASE_LABELS)
def test_uv_goal_needs_no_end_xyz_and_masks_invalid_goals(label):
    policy = policy_for(label)
    uv = torch.tensor([[.2, -.4], [float("nan"), 2.]], requires_grad=True)
    start = torch.ones(2, 8, requires_grad=True)
    batch = {"skill_focus_uv": uv, "skill_focus_valid": torch.tensor([True, False]),
             "skill_start_state": start}
    goal = policy._skill_delta_goal(batch, require_valid=True)
    torch.testing.assert_close(goal[:, :3], torch.tensor([[.2, -.4, 1.], [0., 0., 0.]]))
    torch.testing.assert_close(goal[:, 3:], start[:, :3])
    batch["skill_end_state"] = torch.full((2, 8), float("nan"), requires_grad=True)
    torch.testing.assert_close(policy._skill_delta_goal(batch), goal)
    goal.sum().backward()
    assert batch["skill_end_state"].grad is None
    assert uv.grad[0].abs().sum() > 0 and uv.grad[1].count_nonzero() == 0
    assert start.grad[:, :3].abs().sum() > 0
    with pytest.raises(KeyError, match="skill_focus_valid"):
        policy._skill_delta_goal({k: v for k, v in batch.items() if k != "skill_focus_valid"}, require_valid=True)


@pytest.mark.parametrize("route", ["_lastvis", "_lastvis_current", "_lastvis_current_none", "_lastvis_current_preskill"])
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("diffaux", [False, True])
def test_uv_routes_backward_and_conditioning(monkeypatch, route, checkpoint, diffaux):
    model = tiny_model(monkeypatch, 3, checkpoint=checkpoint, start_position="lastlayer", current_route=route, uv=True, diffaux=diffaux)
    assert model.variant == 3 and model.current_vision_route == "final_all"
    assert model.cond_end_pose_condition[0].in_features == 2
    assert model.end_pose_condition[0].in_features == 2
    model.start_pose_condition = nn.Linear(3, 32)
    model._time_condition = lambda t: torch.zeros(t.shape[0], 32)
    for name in ("cond_end_pose_condition", "end_pose_condition"):
        nn.init.normal_(getattr(model, name)[-1].weight, std=.1)
    monkeypatch.setattr(fullskill_bspline._LITGoalFreeCondMixin, "_project_condition_state",
                        lambda self, state, *args: torch.zeros(state.shape[0], 32))
    goal = torch.tensor([[.2, -.3, 1., .1, .2, .3], [.4, .5, 0., .1, .2, .3]])
    state = torch.randn(2, 32)
    projected = model._project_condition_state(state, skill_code=torch.tensor([1, 2]), end_pose=goal)
    expert = model._expert_condition(torch.ones(2), end_pose=goal)
    changed = goal.clone()
    changed[:, :2] += .2
    projected2 = model._project_condition_state(state, skill_code=torch.tensor([1, 2]), end_pose=changed)
    expert2 = model._expert_condition(torch.ones(2), end_pose=changed)
    assert not torch.equal(projected[0], projected2[0])
    assert not torch.equal(expert[0], expert2[0])
    torch.testing.assert_close(projected[1], projected2[1])
    torch.testing.assert_close(expert[1], expert2[1])
    # Late Expert and output norm must not regain direct UV or start XYZ.
    model._bridge_condition_shift = model.bridge_proprio_condition(state)
    late = model._action_layer_condition(9, expert)
    torch.testing.assert_close(late, model._action_layer_condition(9, expert2))
    model.train()
    hidden = torch.randn(2, 10, 32, requires_grad=True)
    latent = torch.randn(2, 100, 16, requires_grad=True)
    model.set_skill_start_latent(latent)
    previous = torch.randn(2, 30, 16, requires_grad=True)
    model.set_preskill_plan(previous, torch.tensor([False, True]))
    output = model._run_expert_with_layerwise_latents(hidden, expert, torch.randn(2, 32), [latent] * 18)
    prediction = model.take_bspline_prediction()
    assert output.shape == (2, 10, 32) and prediction.shape == (2, 30, 8)
    (output.float().square().mean() + prediction.square().mean() + projected.square().mean()).backward()
    assert latent.grad is not None and hidden.grad is not None
    assert previous.grad is None
    assert model.end_pose_condition[-1].weight.grad is not None
    assert model.cond_end_pose_condition[-1].weight.grad is not None
    assert model.start_pose_condition.weight.grad is not None
    assert model.bspline_proprio_head[-1].weight.grad is not None
    if route.endswith("_none"):
        assert model.bspline_reader.out_proj.weight.grad is None


def test_uv_preskill_encoder_receives_previous_not_current_uv():
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
        def encode_preskill_plan(self, images, state, code, goal):
            self.goal = goal.clone()
            return torch.zeros(state.shape[0], 30, 256)
        def set_preskill_plan(self, plan, valid):
            self.plan = plan
    policy = policy_for("uv_fullskill_bspline_3_lastlayer_lastvis_current_preskill")
    policy.model = Encoder()
    batch = {memory.IMAGE: torch.ones(1, 3, 4, 4), memory.WRIST_IMAGE: torch.ones(1, 3, 4, 4),
             memory.STATE_NORMALIZED: torch.ones(1, 8), memory.START_STATE: torch.full((1, 8), .3),
             memory.END_STATE: torch.full((1, 8), float("nan")), memory.CODE: torch.tensor([5]),
             memory.VALID: torch.tensor([True]), memory.FOCUS_UV: torch.tensor([[.4, -.2]]),
             memory.FOCUS_VALID: torch.tensor([True]), "skill_focus_uv": torch.tensor([[-.8, .9]])}
    policy._prepare_preskill_memory(batch)
    torch.testing.assert_close(policy.model.goal, torch.tensor([[.4, -.2, 1., .3, .3, .3]]))
    assert not policy.model.plan.requires_grad


@pytest.mark.parametrize("label", UV_BSPLINE_DIFFAUX_LABELS)
def test_diffaux_identity_keeps_base_routes_and_separate_checkpoint_head(label):
    from lerobot.policies.skill_expert.modeling_skill_expert import _allowed_pi05_missing_key

    config = policy_for(label).config
    base = policy_for(label.removesuffix("_diffaux")).config
    assert config.uses_top_uv_goal and config.trains_agent_end_state_prediction
    assert config.trains_wrist_patch_alignment and not config.trains_agent_patch_alignment
    assert config.uses_current_bspline_vision == base.uses_current_bspline_vision
    assert config.uses_bspline_preskill_memory == base.uses_bspline_preskill_memory
    assert config.reads_bspline_plan == base.reads_bspline_plan
    assert config.architecture_revision != base.architecture_revision
    assert _allowed_pi05_missing_key("model.agent_end_state_head.1.weight", config)
    assert not _allowed_pi05_missing_key("model.agent_end_state_head.1.weight", base)


def test_diffaux_head_reads_only_top_aux_query_and_consumes_both_stashes(monkeypatch):
    model = tiny_model(monkeypatch, 3, start_position="lastlayer", current_route="_lastvis", uv=True, diffaux=True)

    class WristHead(nn.Module):
        def forward(self, query, patches):
            return query[:, 0, :4] + patches[:, 0, :4]

    model.wrist_patch_align_head = WristHead()
    query = torch.randn(2, 2, 16, requires_grad=True)
    model._final_dual_patch_query = query
    model._final_agent_patch_tokens = torch.randn(2, 4, 16)
    model._final_wrist_patch_tokens = torch.randn(2, 4, 16, requires_grad=True)
    prediction, wrist = model.predict_training_end_state_and_wrist_patch()
    assert prediction.shape == (2, 8) and wrist.shape == (2, 4)
    assert not list(model.agent_patch_align_head.parameters())
    prediction.square().mean().backward()
    assert query.grad[:, 0].abs().sum() > 0
    assert query.grad[:, 1].count_nonzero() == 0
    assert model.agent_end_state_head[-1].weight.grad is not None
    assert model._final_dual_patch_query is None and model._final_agent_patch_tokens is None
    assert model._final_wrist_patch_tokens is None
    with pytest.raises(RuntimeError, match="preceding training"):
        model.predict_training_end_state_and_wrist_patch()


@pytest.mark.parametrize("top_k", [1, 2])
def test_diffaux_state_mse_normalizes_all_eight_dimensions_and_masks_nan(top_k):
    policy = policy_for(UV_BSPLINE_DIFFAUX_LABELS[0])
    # Legacy pose INPUT selector must not truncate the new auxiliary TARGET.
    assert policy.config.skill_end_pose_mode == "xyz"
    policy.config.chunk_end_state_q01 = list(range(8))
    policy.config.chunk_end_state_q99 = [v + 2 for v in range(8)]
    # The valid raw midpoint normalizes to zero; error is one in EVERY axis.
    target = torch.tensor([[v + 1. for v in range(8)], [float("nan")] * 8])
    prediction = torch.ones(2 * top_k, 8, requires_grad=True)
    batch = {"skill_end_state": target, "skill_end_state_valid": torch.tensor([True, False]),
             # Endpoint state supervision does not depend on goal UV visibility.
             "skill_focus_valid": torch.tensor([False, False])}
    loss, metrics = policy._agent_end_state_alignment_loss(batch, prediction, top_k=top_k)
    torch.testing.assert_close(loss, torch.tensor(1.))
    assert metrics["agent_end_state/valid_fraction"] == .5
    for group in ("xyz", "rotation", "gripper"):
        assert metrics[f"agent_end_state/{group}_mae_normalized"] == 1.
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[:top_k].abs().sum() > 0
    assert prediction.grad[top_k:].count_nonzero() == 0
    batch["skill_end_state_valid"][:] = False
    zero, _ = policy._agent_end_state_alignment_loss(batch, prediction, top_k=top_k)
    torch.testing.assert_close(zero, torch.tensor(0.))
    prediction.grad = None
    zero.backward()
    assert prediction.grad.count_nonzero() == 0


def test_diffaux_combines_unchanged_wrist_loss_with_state_mse(monkeypatch):
    from lerobot.policies.skill_expert import modeling_skill_expert as impl
    policy = policy_for(UV_BSPLINE_DIFFAUX_LABELS[0])
    prediction = torch.ones(2, 8, requires_grad=True)
    wrist = torch.randn(2, 4, requires_grad=True)

    class Readouts(nn.Module):
        def predict_training_end_state_and_wrist_patch(self):
            return prediction, wrist

    policy.model = Readouts()
    monkeypatch.setattr(impl, "patch_labels", lambda *args, **kwargs: (
        torch.zeros(2, dtype=torch.long), None, torch.ones(2, dtype=torch.bool)))
    monkeypatch.setattr(impl, "wrist_patch_alignment_loss", lambda logits, *args, **kwargs: (
        logits.square().mean(), {"wrist_patch/valid_fraction": 1.}))
    batch = {"skill_decoder_state": torch.zeros(2, 8), "skill_end_state": torch.zeros(2, 8),
             "skill_end_state_valid": torch.ones(2, dtype=torch.bool)}
    loss, metrics = policy._wrist_patch_alignment_loss(batch, top_k=1)
    torch.testing.assert_close(loss, wrist.square().mean() + prediction.square().mean())
    assert "agent_patch/raw_loss" not in metrics and "agent_end_state/raw_loss" in metrics
    loss.backward()
    assert wrist.grad is not None and prediction.grad is not None


def test_diffaux_start_aux_uses_same_endpoint_not_start_proprio(monkeypatch):
    policy = policy_for(UV_BSPLINE_DIFFAUX_LABELS[0])
    end, start = torch.randn(2, 8), torch.randn(2, 8)
    batch = {"skill_end_state": end, "skill_start_state": start,
             "skill_start_focus_uv": torch.zeros(2, 2),
             "skill_start_focus_valid": torch.ones(2, dtype=torch.bool)}
    captured = {}
    monkeypatch.setattr(policy, "_wrist_patch_alignment_loss", lambda values, **kwargs: captured.update(values))
    policy._skill_start_patch_alignment_loss(batch)
    assert captured["skill_end_state"] is end
    assert captured["skill_decoder_state"] is start
