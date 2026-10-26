"""Diffaux3/4 rollout uses cached start-head XYZ, never a GT endpoint."""

import pytest
import torch
from torch import nn
from types import SimpleNamespace

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.configs.uv_fullskill import UV_BSPLINE_GT_XYZ_LABELS
from lerobot.policies.skill_expert.fullskill_bspline import FullskillBsplineSkillExpert
from test_cond_sdpa import _tiny_model
from test_fullskill_bspline import tiny_model
from test_stage1_vsa import _skill_config
from test_uv_fullskill_bspline import policy_for


class StartHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalized_xyz = nn.Parameter(torch.zeros(3))
        self.encodings, self.goals = [], []
        self.held_start = None

    def encode_skill_start(self, images, state, code, goal):
        self.encodings.append(([x.clone() for x in images], state.clone(), code.clone(), goal.clone()))
        self.count = state.shape[0]
        return torch.ones(self.count, 100, 16)

    def predict_start_xyz(self):
        return self.normalized_xyz.expand(self.count, -1)

    def set_skill_start_latent(self, value):
        self.held_start = value

    def set_preskill_plan(self, *args):
        pass

    def sample_actions(self, images, state, skill_code, **kwargs):
        self.goals.append(kwargs["end_pose"].clone())
        return torch.zeros(state.shape[0], 2, 7)


def rollout_policy(monkeypatch, label):
    policy = policy_for(label)
    policy.config = _skill_config(label, chunk_size=2, n_action_steps=1,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))})
    policy.config.chunk_end_state_q01 = [1., 2., 3., 0., 0., 0., 0., 0.]
    policy.config.chunk_end_state_q99 = [3., 6., 9., 1., 1., 1., 1., 1.]
    policy.model = StartHead()
    # An external UV Predictor may omit wrist; the VSA start head still uses both.
    policy.model.skill_predictor = SimpleNamespace(uses_wrist=False)
    monkeypatch.setattr(policy, "_prepare_preskill_memory", lambda *a, **kw: None)
    policy.reset()
    return policy


def batch():
    return {"observation.state": torch.full((2, 8), .2), "skill_code": torch.tensor([3, 3]),
        "observation.images.image": torch.full((2, 3, 2, 2), 9.),
        "observation.images.wrist_image": torch.full((2, 3, 2, 2), 8.),
        "skill_start_image": torch.ones(2, 3, 2, 2),
        "skill_start_wrist_image": torch.full((2, 3, 2, 2), 2.),
        "skill_start_state": torch.zeros(2, 8),
        "skill_start_state_normalized": torch.full((2, 8), .1),
        "skill_start_cache_id": torch.tensor([0, 0]),
        "skill_focus_uv": torch.tensor([[.2, -.4], [.3, -.5]]),
        "skill_focus_valid": torch.ones(2, dtype=torch.bool)}


@pytest.mark.parametrize("label", UV_BSPLINE_GT_XYZ_LABELS)
def test_rollout_uses_start_head_metres_without_gt_and_caches_occurrences(monkeypatch, label):
    policy = rollout_policy(monkeypatch, label)
    inputs = batch()
    policy.predict_action_chunk(inputs)  # No skill_end_state or camera calibration.
    expected = torch.tensor([[2., 4., 6.], [2., 4., 6.]])
    torch.testing.assert_close(policy.model.goals[-1][:, 6:], expected)
    torch.testing.assert_close(policy.get_inference_goal_xyz(), expected)
    assert len(policy.model.encodings) == 1  # Held vision reuses this same pass.
    images, state, code, goal = policy.model.encodings[0]
    torch.testing.assert_close(images[0], inputs["skill_start_image"])
    torch.testing.assert_close(images[1], inputs["skill_start_wrist_image"])
    torch.testing.assert_close(state[:, :8], inputs["skill_start_state_normalized"])
    assert goal.shape == (2, 6)  # XYZ must not be smuggled into its own prediction.
    assert (policy.model.held_start is None) is policy.config.uses_current_bspline_vision
    # Poisoned GT must not affect either the cache or the Expert goal packet.
    inputs["skill_end_state"] = torch.full((2, 8), float("nan"))
    inputs["skill_end_state_valid"] = torch.zeros(2, dtype=torch.bool)
    inputs["observation.state"].add_(5)
    inputs["observation.images.image"].add_(5)
    with torch.no_grad():
        policy.model.normalized_xyz.fill_(1.)
    policy.predict_action_chunk(inputs)
    assert len(policy.model.encodings) == 1
    torch.testing.assert_close(policy.model.goals[-1][:, 6:], expected)
    # Refresh just one environment, including consecutive identical FSQ codes.
    inputs["skill_start_cache_id"][1] = 1
    policy.predict_action_chunk(inputs)
    assert len(policy.model.encodings) == 2
    assert policy.model.encodings[-1][0][0].shape[0] == 1
    expected[1] = torch.tensor([3., 6., 9.])
    torch.testing.assert_close(policy.model.goals[-1][:, 6:], expected)
    assert not policy.get_inference_goal_xyz().requires_grad
    policy.reset()
    policy.predict_action_chunk(inputs)
    assert len(policy.model.encodings) == 3
    torch.testing.assert_close(policy.model.goals[-1][:, 6:], torch.tensor([[3., 6., 9.], [3., 6., 9.]]))


@pytest.mark.parametrize("label", UV_BSPLINE_GT_XYZ_LABELS)
def test_supplied_uv_source_is_preserved_and_invalid_prediction_has_no_gt_fallback(monkeypatch, label):
    policy = rollout_policy(monkeypatch, label)
    inputs = batch()
    inputs["skill_end_state"] = torch.full((2, 8), 999.)
    policy.predict_action_chunk(inputs)
    torch.testing.assert_close(policy.model.goals[-1][:, :2], inputs["skill_focus_uv"])
    inputs["skill_focus_uv"].neg_()  # A different selected UV source remains effective.
    policy.predict_action_chunk(inputs)
    torch.testing.assert_close(policy.model.goals[-1][:, :2], inputs["skill_focus_uv"])
    inputs["skill_start_cache_id"].add_(1)
    with torch.no_grad():
        policy.model.normalized_xyz.fill_(float("nan"))
    with pytest.raises(RuntimeError, match="no GT fallback"):
        policy.predict_action_chunk(inputs)


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("label", ["uv_bspline_lastlayer_current_preskill_diffaux4_start_last",
                                   "uv_bspline_lastlayer_current_preskill_diffaux4_uvtoken",
                                   "uv_bspline_lastlayer_current_preskill_diffaux3_last"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_actual_gemma_xyz_head_is_available_in_eval(monkeypatch, checkpoint, label, dtype):
    model = tiny_model(monkeypatch, 3, uv=True, diffaux=True, architecture_label=label)
    core = _tiny_model(monkeypatch, "sdpa", checkpoint, model_class=FullskillBsplineSkillExpert)
    model.cond_encoder = core.cond_encoder.to(dtype=dtype)
    model.layerwise_condition_memory_norm = core.layerwise_condition_memory_norm
    model.layerwise_condition_readers = core.layerwise_condition_readers
    model.layerwise_latent_queries = nn.Parameter(torch.randn(100, 16))
    if model.config.uses_uv_readout_token:
        model._xyz_token_input = (torch.zeros(2, 2), torch.ones(2, 1))
    model.eval()
    with torch.no_grad():
        model._encode_layerwise_latents(torch.randn(2, 10, 32, dtype=dtype), torch.zeros(2, 32, dtype=dtype))
        xyz = model.predict_start_xyz()
    assert xyz.shape == (2, 3) and torch.isfinite(xyz).all()
    assert getattr(model, "_final_xyz_readout", None) is None
    assert model._inference_bottleneck_xyz is None


def test_previous_skill_plan_retains_its_cached_predicted_endpoint():
    from lerobot.policies.skillVLA import preskill_plan as mem
    policy = policy_for("uv_bspline_lastlayer_current_preskill_diffaux4")
    policy.model = StartHead()
    policy.eval()
    goals = []
    policy.model.encode_preskill_plan = lambda images, state, code, goal: (
        goals.append(goal.clone()) or torch.ones(state.shape[0], 30, 256))
    inputs = batch()
    predicted = torch.tensor([[2., 4., 6.], [3., 6., 9.]])
    inputs.update({mem.VALID: torch.tensor([False, True]), mem.CACHE_ID: torch.tensor([0, 1]),
        mem.IMAGE: inputs["skill_start_image"], mem.WRIST_IMAGE: inputs["skill_start_wrist_image"],
        mem.STATE_NORMALIZED: torch.zeros(2, 8), mem.START_STATE: torch.zeros(2, 8),
        mem.END_STATE: torch.cat((predicted, torch.zeros(2, 5)), -1),
        mem.CODE: torch.tensor([2, 2]), mem.FOCUS_UV: inputs["skill_focus_uv"],
        mem.FOCUS_VALID: torch.ones(2, dtype=torch.bool)})
    policy._prepare_preskill_memory(inputs, inference=True)
    torch.testing.assert_close(goals[0][:, 6:], predicted[1:])
