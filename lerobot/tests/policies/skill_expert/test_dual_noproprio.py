"""Direct-proprio and early-vision ablations of the dual bottleneck."""
from dataclasses import replace

import pytest
import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.configs.uv_fullskill import (
    DUAL_BOTTLENECK_NOPROPRIO_LABEL as NO_PROPRIO,
    DUAL_BOTTLENECK_NOPROPRIO_LAST_LABEL as NO_PROPRIO_LAST,
    DUAL_BOTTLENECK_NOPROPRIO_WOVIS_LABEL as NO_EARLY_VISION,
)
from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy, _allowed_pi05_missing_key
from test_dual_bottleneck import inputs
from test_pi05_vlm_vsa import tiny_vlm_model
from test_stage1_vsa import _skill_config
from test_uv_gt_xyz import goal_batch

LABELS = [NO_PROPRIO, NO_PROPRIO_LAST, NO_EARLY_VISION]


def packed_goal(label, goal):
    return torch.cat((goal[:, :3], torch.randn(len(goal), 8)), dim=-1) if label == NO_EARLY_VISION else goal


@pytest.mark.parametrize("label", LABELS)
def test_config_roundtrip_and_missing_pretrained_parameters(tmp_path, label):
    cfg = _skill_config(label)
    assert cfg.uses_dual_bottleneck and cfg.uses_pi05_vlm_vision
    assert cfg.dual_omits_current_expert_proprio
    assert cfg.dual_uses_start_proprio == (label == NO_EARLY_VISION)
    assert cfg.uses_skill_prefix_vision == (label != NO_EARLY_VISION)
    assert cfg.omits_start_vision and not cfg.reads_bspline_plan
    cfg.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored == cfg
    if label == NO_PROPRIO_LAST:
        assert cfg.architecture_revision != _skill_config(NO_PROPRIO).architecture_revision
    assert _allowed_pi05_missing_key("model.start_proprio_condition.0.weight", cfg) == (label == NO_EARLY_VISION)
    with pytest.raises(ValueError, match="9\\+9"):
        replace(cfg, visual_bridge_last_n_layers=17)


@pytest.mark.parametrize("label", LABELS)
def test_only_current_proprio_direct_path_is_removed(monkeypatch, label):
    model = tiny_vlm_model(monkeypatch, label).eval()
    images, state, skill, actions, goal = inputs()
    goal = packed_goal(label, goal)
    assert isinstance(model.bridge_proprio_condition, nn.Identity)
    assert not any(key.startswith("bridge_proprio_condition.") for key in model.state_dict())
    time = torch.full((2,), .4)
    representation = model._project_condition_state(state, skill_code=skill, end_pose=goal)
    torch.testing.assert_close(representation[0], state)  # Proprio remains in the VLM prefix.
    first = model._expert_condition(time, skill_code=skill, end_pose=goal)
    model._project_condition_state(state + .5, skill_code=skill, end_pose=goal)
    second = model._expert_condition(time, skill_code=skill, end_pose=goal)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(model._action_layer_condition(9, second), model._time_condition(time))
    if label == NO_EARLY_VISION:
        nn.init.normal_(model.start_proprio_condition[-1].weight, std=.1)
        first = model._expert_condition(time, end_pose=goal)
        for column in range(8):
            changed = goal.clone(); changed[:, column + 3] += .7
            assert not torch.equal(first, model._expert_condition(time, end_pose=changed))
        with pytest.raises(ValueError, match="proprio8"):
            model._expert_condition(time, end_pose=goal[:, :6])


@pytest.mark.parametrize("label", LABELS)
@pytest.mark.parametrize("checkpoint,dtype", [(False, torch.float32), (True, torch.bfloat16)])
def test_training_losses_and_actual_bridge_routes(monkeypatch, label, checkpoint, dtype):
    model = tiny_vlm_model(monkeypatch, label, dtype=dtype)
    for field in ("action_expert", "skill_planner", "vlm"):
        setattr(model.config, field + "_checkpointing", checkpoint)
    model.config.vision_checkpointing = False
    model.configure_stream_checkpointing()
    images, state, skill, actions, goal = inputs(); goal = packed_goal(label, goal)
    bridges = []; original = model._visual_bridge_update
    def capture(index, *args):
        bridges.append(index)
        return original(index, *args)
    model._visual_bridge_update = capture
    output = model(images, state, skill, actions, end_pose=goal, fullskill_target=torch.randn(2, 30, 8))
    flow = model.take_fullskill_flow_loss()
    assert torch.isfinite(output).all() and torch.isfinite(flow).all()
    expected = ([8, 17, 8] if label == NO_PROPRIO_LAST else
                list(range(9, 18)) if label == NO_EARLY_VISION else list(range(18)) + list(range(9)))
    assert bridges == expected
    expected_layers = ([8, 17] if label == NO_PROPRIO_LAST else
                       list(range(9, 18)) if label == NO_EARLY_VISION else list(range(18)))
    assert [i for i in range(18) if model._layer_uses_visual_bridge(i)] == expected_layers
    assert model._active_visual_bridge_gates().numel() == len(expected_layers)
    vision_parameter = model.cond_encoder.vlm.vision_tower.weight
    vision_grad = torch.autograd.grad(flow.mean(), vision_parameter, retain_graph=True, allow_unused=True)[0]
    if label == NO_EARLY_VISION:
        assert vision_grad is None  # Current images cannot enter the fullskill branch.
    else:
        assert vision_grad is not None
    upper, lower = model.predict_training_dual_auxiliaries()
    loss = output.square().mean() + flow.mean() + .05 * sum(v.square().mean() for tower in (upper, lower) for v in tower)
    loss.backward()
    for parameter in (vision_parameter, model.lower_layerwise_latent_queries, model.bspline_noise_input.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    if label == NO_EARLY_VISION:
        assert model.lower_visual_bridge_attention.q_proj_weight.grad is None
        assert model.start_proprio_condition[-1].weight.grad is not None
    model.eval()
    assert model.sample_actions(images, state, skill, num_steps=2, end_pose=goal).shape == (2, 10, 32)


def test_wovis_fullskill_is_independent_of_current_vision_and_current_proprio(monkeypatch):
    model = tiny_vlm_model(monkeypatch, NO_EARLY_VISION).eval()
    _, state, skill, _, goal = inputs(); goal = packed_goal(NO_EARLY_VISION, goal)
    time = torch.full((2,), .4); tokens = torch.randn(2, 30, model.width)
    model._project_condition_state(state, skill_code=skill, end_pose=goal)
    first = model._planner_forward(tokens, time, skill, goal, torch.randn(2, 100, 16))
    model._project_condition_state(state + 20, skill_code=skill, end_pose=goal)
    second = model._planner_forward(tokens, time, skill, goal, torch.randn(2, 100, 16) * 20)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_policy_transports_normalized_start_state8_and_keeps_existing_uv_noise(monkeypatch):
    model = tiny_vlm_model(monkeypatch, NO_EARLY_VISION)
    model.config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    policy = SkillExpertPolicy.__new__(SkillExpertPolicy); nn.Module.__init__(policy)
    policy.config, policy.model = model.config, model
    images, state, skill, actions, _ = inputs()
    policy._collect_images = lambda batch: images
    batch = goal_batch()
    normalized = torch.randn(2, 8)
    batch.update(action=actions[:, :, :7], skill_code=skill, skill_start_state_normalized=normalized,
                 **{"observation.state": state, "skill_decoder_state": torch.zeros(2, 8),
                    "skill_bspline_states": torch.randn(2, 30, 8)})
    goal = policy._skill_delta_goal(batch, require_valid=True)
    assert goal.shape == (2, 11)
    torch.testing.assert_close(goal[:, 3:], normalized)
    assert model.config.goal_input_noise_max_m == .01
    loss, metrics = policy(batch)
    assert torch.isfinite(loss) and "bspline/loss" in metrics
    assert "vision_aux/lower/agent_patch/raw_loss" in metrics
    loss.backward()
    policy.eval()
    runtime = {"skill_focus_uv": torch.zeros(2, 2), "skill_focus_valid": torch.ones(2, dtype=torch.bool),
               "skill_start_state_normalized": normalized}
    torch.testing.assert_close(policy._skill_delta_goal(runtime)[:, 3:], normalized)
    runtime.update(skill_code=skill, **{'observation.state': state})
    actions = policy.predict_action_chunk(runtime, num_steps=2)
    assert actions.shape == (2, 10, 7) and torch.isfinite(actions).all()
    runtime.pop("skill_start_state_normalized")
    with pytest.raises(KeyError, match="normalized"):
        policy._skill_delta_goal(runtime)
