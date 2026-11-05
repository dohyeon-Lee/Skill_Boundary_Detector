from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.uv_fullskill import VANILA_PI05_VLM_BASE_LABELS as LABELS
from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy
from test_pi05_vlm_vsa import tiny_vlm_model
from test_vanila import config_for
from lerobot.configs.uv_fullskill import vanila_bridge_layers


@pytest.mark.parametrize("label", LABELS)
def test_vlm_vanila_config_and_policy_dispatch(monkeypatch, tmp_path, label):
    from lerobot.policies.skill_expert import pi05_vlm_vision

    config = config_for(label, tokenizer_path="unused", device="cpu", dtype="float32")
    assert config.uses_vanila_expert and config.uses_pi05_vlm_vision
    assert not config.uses_bspline_plan and not config.uses_independent_skill_planner
    assert not config.reads_bspline_plan and not config.skill_flow_enabled
    assert config.omits_skill_expert_uv_goal == ("nouv" in label)
    config.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.cond_encoder_variant == "gemma_2b" and restored.vision_backbone == "pi05vlm"
    dummy = nn.Module()
    dummy.skill_proj, dummy.action_in_proj = nn.Linear(3, 32), nn.Linear(32, 32)
    dummy.gemma_expert = nn.Linear(32, 32)
    dummy.gemma_expert.model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=18))
    dummy.working_dtype, dummy.dino, dummy.fsq_term_train = torch.float32, None, None
    monkeypatch.setattr(pi05_vlm_vision, "Pi05VLMVanilaSkillExpert", lambda config: dummy)
    assert SkillExpertPolicy(config).model is dummy


@pytest.mark.parametrize("label", LABELS)
@pytest.mark.parametrize("dtype,checkpoint", [(torch.float32, False), (torch.bfloat16, True)])
def test_vlm_vanila_last_only_training_aux_and_inference(monkeypatch, label, dtype, checkpoint):
    model = tiny_vlm_model(monkeypatch, label, dtype=dtype, checkpoint=checkpoint, bridge_layers=vanila_bridge_layers(label))
    assert model.dino is None
    assert len(model.gemma_expert.model.layers) == len(model.cond_encoder.model.layers) == 18
    assert model.split_layer == (9 if label == 'vanila_last_pi05vlm_9_9' else 1)
    assert model.visual_bridge_start_layer == 17
    assert not any("bspline" in key or "planner" in key or "start_bridge" in key for key in model.state_dict())
    assert model.config.omits_skill_expert_uv_goal == isinstance(model.end_pose_condition, nn.Identity)
    images, state, code = [torch.rand(2, 3, 4, 4)] * 2, torch.zeros(2, 32), torch.tensor([1, 2])
    goal = torch.tensor([[.2, -.3, 1., .1, .1, .1]] * 2)
    assert model.config.omits_start_vision and not model.config.uses_symmetric_start_vision
    seen = []
    original = model._visual_bridge_update
    def capture(index, hidden, latent):
        seen.append(index)
        return original(index, hidden, latent)
    monkeypatch.setattr(model, "_visual_bridge_update", capture)
    model.config.vsa_debug_schedule = [1]
    model.set_training_step(1)
    residual = model(images, state, code, torch.randn(2, 10, 32), end_pose=goal)
    aux = model.predict_training_top_wrist_patch_and_xyz()
    assert set(seen) == {17}
    assert aux[0].shape == aux[1].shape == (2, 4) and aux[2].shape == (2, 3)
    loss = residual.square().mean() + .1 * sum(x.square().mean() for x in aux)
    loss.backward()
    assert model.action_out_proj.weight.grad is not None
    assert model.agent_patch_align_head.query_proj.weight.grad is not None
    assert model.cond_encoder.model.layers[0].self_attn.q_proj.weight.grad is not None
    assert "activation/state_input_rms" in model._last_vsa_debug_stats
    condition = model._expert_condition(torch.ones(2), end_pose=goal)
    late = model._action_layer_condition(model.split_layer, condition)
    assert all(model._action_layer_condition(i, condition) is condition for i in range(model.split_layer))
    for index in range(model.split_layer, 18):
        torch.testing.assert_close(model._action_layer_condition(index, condition), late)
    # No cached start observation or planner is required at inference.
    model.eval()
    output = model.sample_actions(images, state, code, num_steps=1, end_pose=goal)
    assert output.shape == (2, 10, 32) and torch.isfinite(output).all()


@pytest.mark.parametrize("label", LABELS)
def test_split_inputs_do_not_leak_into_late_adarms_or_final_norm(monkeypatch, label):
    from lerobot.policies.skill_expert import pi05_vlm_vision

    model = tiny_vlm_model(monkeypatch, label, bridge_layers=vanila_bridge_layers(label))
    # Nonzero adapters expose routing mistakes hidden by warm-start zero init.
    nn.init.normal_(model.bridge_proprio_condition[-1].weight, std=.1)
    nn.init.normal_(model.start_pose_condition[-1].weight, std=.1)
    if not model.config.omits_skill_expert_uv_goal:
        nn.init.normal_(model.end_pose_condition[-1].weight, std=.1)
    state, code = torch.randn(2, 32), torch.tensor([1, 2])
    goal = torch.tensor([[.2, -.3, 1., .1, .2, .3]] * 2)
    time = torch.ones(2)
    def conditions(proprio, packed):
        model._project_condition_state(proprio, skill_code=code, end_pose=packed)
        early = model._expert_condition(time, end_pose=packed)
        return early, model._action_layer_condition(model.split_layer, early)
    early, late = conditions(state, goal)
    early_state, late_state = conditions(state + .5, goal)
    torch.testing.assert_close(early, early_state)
    assert not torch.equal(late, late_state)
    changed = goal.clone(); changed[:, :2] += .3; changed[:, 3:] += .2
    early_goal, late_goal = conditions(state, changed)
    assert not torch.equal(early, early_goal)
    torch.testing.assert_close(late, late_goal)
    uv_only = goal.clone(); uv_only[:, :2] += .3
    early_uv, _ = conditions(state, uv_only)
    assert torch.equal(early, early_uv) == model.config.omits_skill_expert_uv_goal
    conditions(state, goal)
    layer_calls, norm_calls = [], []
    original_layer = model._expert_layer_with_latent_bridge
    original_norm = pi05_vlm_vision.layernorm_forward
    def layer(*args):
        layer_calls.append(args)
        return original_layer(*args)
    def norm(module, hidden, condition):
        norm_calls.append(condition)
        return original_norm(module, hidden, condition)
    monkeypatch.setattr(model, "_expert_layer_with_latent_bridge", layer)
    monkeypatch.setattr(pi05_vlm_vision, "layernorm_forward", norm)
    skill = model._skill_broadcasts(code)[1]
    latents = [torch.randn(2, 100, 16) for _ in range(18)]
    model._run_expert_with_layerwise_latents(torch.randn(2, 10, 32), early, skill, latents)
    assert all(call[5] is skill for call in layer_calls[:model.split_layer])
    assert all(call[5] is None for call in layer_calls[model.split_layer:])
    assert all(call[6] is None for call in layer_calls[:-1])
    assert layer_calls[-1][6] is latents[-1]
    torch.testing.assert_close(norm_calls[-1], late)
