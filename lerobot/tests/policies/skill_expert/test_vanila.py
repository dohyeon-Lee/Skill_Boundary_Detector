"""Vanila: no planner, all-layer nonvisual inputs, final-Z-only bridges."""
from dataclasses import replace

import pytest
import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.uv_fullskill import VANILA_LABELS, VANILA_REVISIONS
from lerobot.policies.skill_expert.vanila import VanilaSkillExpert
from lerobot.policies.skill_expert.layerwise_cond_bottleneck import _LITGoalFreeCondMixin
from lerobot.policies.skill_expert.layerwise_cond_bottleneck import _LayerwiseConditionReader
from lerobot.policies.skill_expert.layerwise_cond_bottleneck import WristSkillEndPoseLayerwiseCondBottleneckSkillExpert
from lerobot.policies.skill_expert.wrist_patch_alignment import WristPatchAlignmentHead
from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy, _allowed_pi05_missing_key
from test_fullskill_bspline import tiny_model
from test_stage1_vsa import _skill_config
from test_uv_gt_xyz import goal_batch


def config_for(label, **kwargs):
    return replace(_skill_config('uv_bspline_skill_denoising_none'), **{
        'architecture_label': label, 'architecture_revision': VANILA_REVISIONS[label],
        'visual_bridge_last_n_layers': 1 if label.endswith('_last') else 18,
        **kwargs,
    })


@pytest.mark.parametrize('label', VANILA_LABELS)
def test_contract_serialization(tmp_path, label):
    cfg = config_for(label)
    assert cfg.uses_vanila_expert and cfg.uses_triple_auxiliary_prediction
    assert cfg.uses_joint_bottleneck_self_attention and cfg.predicts_xyz_only
    assert not cfg.uses_bspline_plan and not cfg.uses_independent_skill_planner
    assert not cfg.skill_flow_enabled and not cfg.reads_bspline_plan
    assert not cfg.uses_start_predicted_expert_xyz and not cfg.uses_gemma_xyz_readout
    assert cfg.uses_symmetric_start_vision and not cfg.uses_skill_prefix_vision
    cfg.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.architecture_label == label
    assert restored.omits_skill_expert_uv_goal == ('nouv' in label or 'vision_only' in label)
    for key in ('bridge_proprio_condition.0.weight', 'start_pose_condition.0.weight',
                'cond_end_pose_condition.0.weight', 'agent_end_state_head.1.weight'):
        assert _allowed_pi05_missing_key('model.' + key, cfg)


def model_for(monkeypatch, label, checkpoint):
    # Reuse the existing 18-layer tiny Gemma fixture, never pretrained GPU assets.
    tiny_model(monkeypatch, 3, checkpoint=checkpoint)
    cfg = config_for(label)
    # Production geometry is fixed at 256/8; downsize only the test fixture.
    cfg.visual_bottleneck_width, cfg.visual_bridge_heads = 16, 4
    model = VanilaSkillExpert(cfg)
    if not cfg.uses_vision_only_expert:
        model.start_pose_condition = nn.Linear(3, 32)
    model._time_condition = lambda t: t[:, None].expand(-1, 32).to(model.working_dtype)
    monkeypatch.setattr(_LITGoalFreeCondMixin, '_project_condition_state',
                        lambda self, state, *args: torch.zeros(len(state), 32))
    monkeypatch.setattr(WristSkillEndPoseLayerwiseCondBottleneckSkillExpert, '_project_condition_state',
                        lambda self, state, focus, code, goal: state + code[:, None].float())
    if not cfg.uses_vision_only_expert:
        nn.init.normal_(model.bridge_proprio_condition[-1].weight, std=.1)
    if not cfg.omits_skill_expert_uv_goal:
        nn.init.normal_(model.end_pose_condition[-1].weight, std=.1)
    nn.init.normal_(model.cond_end_pose_condition[-1].weight, std=.1)
    return model


@pytest.mark.parametrize('label', VANILA_LABELS)
@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_all_inputs_and_last_z_bridge_gradients(monkeypatch, label, checkpoint, dtype):
    model = model_for(monkeypatch, label, checkpoint)
    model.gemma_expert.to(dtype=dtype)
    assert not any('bspline' in n or 'planner' in n or 'start_bridge' in n for n, _ in model.named_parameters())
    state = torch.randn(2, 32, requires_grad=True)
    goal = torch.tensor([[.2, -.3, 1., .1, .2, .3]] * 2, requires_grad=True)
    projected = model._project_condition_state(state, skill_code=torch.tensor([1, 2]), end_pose=goal)
    condition = model._expert_condition(torch.ones(2), end_pose=goal)
    changed = goal.detach().clone()
    changed[:, :2] += .2
    cond_changed = model._expert_condition(torch.ones(2), end_pose=changed)
    assert torch.equal(condition, cond_changed) == model.config.omits_skill_expert_uv_goal
    assert all(model._action_layer_condition(i, condition) is condition for i in range(18))
    latents = [torch.randn(2, 100, 16, requires_grad=True) for _ in range(18)]
    seen = []
    original = model._visual_bridge_update
    def capture(index, hidden, latent):
        seen.append(index)
        assert latent is latents[-1]
        return original(index, hidden, latent)
    monkeypatch.setattr(model, '_visual_bridge_update', capture)
    hidden = torch.randn(2, 10, 32, dtype=dtype, requires_grad=True)
    skill = (model._skill_broadcasts(torch.tensor([1, 2]))[1] if model.config.uses_vision_only_expert
             else torch.randn(2, 32, dtype=dtype))
    out = model._run_expert_with_layerwise_latents(hidden, condition, skill, latents)
    assert sorted(set(seen)) == ([17] if label.endswith('_last') else list(range(18)))
    (out.square().mean() + projected.square().mean()).backward()
    assert all(x.grad is None for x in latents[:-1])
    assert latents[-1].grad is not None and latents[-1].grad.abs().sum() > 0
    assert state.grad is not None and state.grad.abs().sum() > 0
    if not model.config.uses_vision_only_expert:
        assert model.start_pose_condition.weight.grad is not None


@pytest.mark.parametrize('label', VANILA_LABELS)
def test_goal_noise_and_inference_without_start_encoding(label):
    policy = SkillExpertPolicy.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.config = config_for(label)
    policy.model = nn.Identity()
    # Inference never asks for a full-skill/start-vision plan or GT endpoint XYZ.
    policy.eval()
    policy._prepare_start_vision({}, None, None, inference=True)
    runtime = {'skill_focus_uv': torch.tensor([[1.2, -.2]]),
               'skill_start_state': torch.zeros(1, 8)}
    goal = policy._skill_delta_goal(runtime)
    torch.testing.assert_close(goal[:, :3], torch.tensor([[1., -.2, 1.]]))
    policy.train()
    batch = goal_batch()
    torch.manual_seed(21)
    noisy = policy._skill_delta_goal(batch, require_valid=True)
    assert noisy.shape[-1] == 6


def test_preprocessing_supplies_normalized_start_without_fullskill_target():
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.skill_expert.processor_skill_expert import make_skill_expert_pre_post_processors
    cfg = config_for('vanila', input_features={
        'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }, output_features={'action': PolicyFeature(type=FeatureType.ACTION, shape=(7,))})
    stats = {'observation.state': {'q01': torch.zeros(8), 'q99': torch.full((8,), 2.)},
             'action': {'q01': -torch.ones(7), 'q99': torch.ones(7)}}
    pre, _ = make_skill_expert_pre_post_processors(cfg, stats)
    batch = pre({'observation.state': torch.ones(2, 8), 'action': torch.zeros(2, 10, 7),
                 'skill_start_state': torch.ones(2, 8)})
    torch.testing.assert_close(batch['skill_start_state_normalized'], torch.zeros(2, 8), atol=1e-6, rtol=0)
    assert 'skill_bspline_states' not in batch


def test_joint_camera_scopes_and_auxiliary_gradients(monkeypatch):
    model = model_for(monkeypatch, 'vanila', False)
    model.layerwise_condition_readers = nn.ModuleList([_LayerwiseConditionReader(16, 32, 4)])
    reader = model.layerwise_condition_readers[0]
    scopes = []
    original = reader.read_condition
    def read(query, memory):
        scopes.append((query.shape[1], memory.shape[1]))
        return original(query, memory)
    monkeypatch.setattr(reader, 'read_condition', read)
    latent = torch.randn(2, 100, 16, requires_grad=True)
    memory = torch.randn(2, 10, 32, requires_grad=True)
    result = model._update_layerwise_latent(0, latent, memory)
    assert scopes == [(1, 5), (1, 5), (1, 10), (97, 10)]
    # Joint self-attention lets action-query information reach XYZ readout.
    xyz_grad = torch.autograd.grad(result[:, 2].square().mean(), latent, retain_graph=True)[0]
    assert xyz_grad[:, 3:].abs().sum() > 0
    model.agent_patch_align_head = WristPatchAlignmentHead(16, 32)
    model.wrist_patch_align_head = WristPatchAlignmentHead(16, 32)
    model._final_dual_patch_query = result[:, :3]
    model._final_agent_patch_tokens = memory[:, 1:5]
    model._final_wrist_patch_tokens = memory[:, 6:10]
    top, wrist, xyz = model.predict_training_top_wrist_patch_and_xyz()
    assert top.shape == wrist.shape == (2, 4) and xyz.shape == (2, 3)
    (.1 * (top.square().mean() + wrist.square().mean() + xyz.square().mean())).backward()
    assert memory.grad is not None and memory.grad.abs().sum() > 0
    assert model.agent_end_state_head[-1].weight.grad is not None
    with pytest.raises(RuntimeError, match='readout'):
        model.predict_training_top_wrist_patch_and_xyz()


def test_vision_only_removes_direct_conditions_but_keeps_vision_inputs(monkeypatch):
    model = model_for(monkeypatch, 'vanila_vision_only', False).eval()
    direct = ('skill_proj.', 'end_pose_condition.', 'start_pose_condition.', 'bridge_proprio_condition.')
    assert not any(name.startswith(direct) for name, _ in model.named_parameters())
    state = torch.randn(2, 32, requires_grad=True)
    goal = torch.tensor([[.2, -.3, 1., .1, .2, .3]] * 2, requires_grad=True)
    code = torch.tensor([1, 2])
    projected = model._project_condition_state(state, skill_code=code, end_pose=goal)
    assert not torch.equal(projected, model._project_condition_state(state + 1, skill_code=code, end_pose=goal))
    assert not torch.equal(projected, model._project_condition_state(state, skill_code=code + 1, end_pose=goal))
    changed_goal = goal.detach().clone()
    changed_goal[:, :2] += .5
    assert not torch.equal(projected, model._project_condition_state(state, skill_code=code, end_pose=changed_goal))
    time = torch.ones(2)
    before = model._expert_condition(time, end_pose=goal)
    after = model._expert_condition(time, projected_state=projected + 5,
                                     skill_code=code + 20, end_pose=None)
    torch.testing.assert_close(before, after)
    first = model._skill_broadcasts(code)[1]
    second = model._skill_broadcasts(code + 20)[1]
    assert first.count_nonzero() == 0
    torch.testing.assert_close(first, second)
    hidden = torch.randn(2, 10, 32)
    latents = [torch.randn(2, 100, 16) for _ in range(18)]
    output = model._run_expert_with_layerwise_latents(hidden, before, first, latents)
    # Hold the visual representation fixed to isolate the removed direct paths.
    other = model._run_expert_with_layerwise_latents(hidden, after, second, latents)
    torch.testing.assert_close(output, other, atol=0, rtol=0)
    direct_grads = torch.autograd.grad(output.sum(), (state, goal), allow_unused=True)
    assert all(g is None for g in direct_grads)


def test_vision_only_inference_does_not_read_start_xyz():
    policy = SkillExpertPolicy.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.config = config_for('vanila_vision_only')
    policy.eval()
    batch = {'skill_focus_uv': torch.tensor([[.2, -.3]])}
    first = policy._skill_delta_goal(batch)
    second = policy._skill_delta_goal({**batch, 'skill_start_state': torch.full((1, 8), float('nan'))})
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first[:, 3:], torch.zeros(1, 3))
