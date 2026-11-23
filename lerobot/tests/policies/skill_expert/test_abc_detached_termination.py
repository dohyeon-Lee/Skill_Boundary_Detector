"""ABC termination keeps both arm goals and isolates the private reader gradients."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.configs.uv_fullskill import DUAL_BOTTLENECK_TERM_DETACH_LABEL
from lerobot.policies.skillVLA.abc_native_skillvla_dataset import ABCNativeSkillVLADataset
from lerobot.policies.skillVLA.vlm_termination_supervision import (
    DISTANCE, TERMINAL, PAST_CODES, PAST_VALID, PAST_UV, PAST_UV_VALID,
    PAST_TERMINAL, PAST_DISTANCE,
)
from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy
from test_pi05_vlm_vsa import tiny_vlm_model


def test_abc_jittered_termination_retains_each_skills_two_uvs():
    dataset = object.__new__(ABCNativeSkillVLADataset)
    dataset._vlm_termination_post_end_steps = 10
    def spatial(episode, rank, start, frame):
        assert episode == 5 and start == rank * 100
        return {'uv': np.array([[rank * .2, .1], [-.3, rank * .4]], dtype=np.float32),
                'valid': np.array([True, rank == 0])}
    dataset._spatial = SimpleNamespace(targets=spatial)
    item = {}
    # A held late: current A is already positive, B keeps its GT negative.
    dataset._add_vlm_termination_targets(item, 5, 103, 0, [0, 100], [100, 100], [4, 7], 200)
    assert item[DISTANCE].item() == -4
    assert item[PAST_CODES][item[PAST_VALID]].tolist() == [7]
    assert item[PAST_DISTANCE][item[PAST_VALID]].tolist() == [96]
    torch.testing.assert_close(item[PAST_UV][item[PAST_VALID]], torch.tensor([[[.2, .1], [-.3, .4]]]))
    assert item[PAST_UV_VALID][item[PAST_VALID]].tolist() == [[True, False]]
    # B selected: the extra condition is the previous A, with A's own UVs.
    dataset._add_vlm_termination_targets(item, 5, 103, 1, [0, 100], [100, 100], [4, 7], 200)
    assert item[DISTANCE].item() == 96
    assert item[PAST_CODES][item[PAST_VALID]].tolist() == [4]
    assert item[PAST_DISTANCE][item[PAST_VALID]].tolist() == [-4]
    torch.testing.assert_close(item[PAST_UV][item[PAST_VALID]], torch.tensor([[[0., .1], [-.3, 0.]]]))
    assert item[PAST_UV_VALID][item[PAST_VALID]].tolist() == [[True, True]]
    dataset._add_vlm_termination_targets(item, 5, 110, 1, [0, 100], [100, 100], [4, 7], 200)
    assert not item[PAST_VALID].any()
    dataset._add_vlm_termination_targets(item, 5, 199, 1, [0, 100], [100, 100], [4, 7], 200)
    assert item[TERMINAL].item()


@pytest.mark.parametrize('checkpoint,dtype', [(False, torch.float32), (True, torch.bfloat16)])
def test_abc_detached_policy_loss_past_conditions_and_deployed_prefix(monkeypatch, tmp_path, checkpoint, dtype):
    keys = ['observation.images.top', 'observation.images.left_wrist', 'observation.images.right_wrist']
    torch.manual_seed(43)
    model = tiny_vlm_model(
        monkeypatch, DUAL_BOTTLENECK_TERM_DETACH_LABEL, checkpoint=checkpoint, dtype=dtype,
        device='cpu', bimanual_spatial_auxiliary=True, skill_plan_state_dim=14,
        chunk_end_state_dim=14, chunk_end_state_q01=[-1.] * 14, chunk_end_state_q99=[1.] * 14,
        bimanual_goal_xyz_q01=[-1.] * 6, bimanual_goal_xyz_q99=[1.] * 6,
        skill_image_keys=keys, skill_fsq_levels=[3, 3, 3, 2], skill_vocab_size=54,
        chunk_size=30, output_features={'action': PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
    )
    images = [torch.rand(2, 3, 4, 4) for _ in keys]
    state = torch.zeros(2, 32)
    code = torch.tensor([1, 53])
    goal = torch.tensor([[.2, -.3, .4, -.5, 1., 1.], [.1, .2, .3, .4, 1., 0.]])
    model(images, state, code, torch.randn(2, 30, 32), end_pose=goal)
    logits, memory = model.take_vlm_termination_outputs()
    changed = state.clone(); changed[:, 14:] = 100
    deployed = model.vlm_termination_from_image_tokens(memory, changed, code, goal)
    torch.testing.assert_close(logits, deployed, rtol=0, atol=0)
    deployed.square().mean().backward()
    branch = model.vlm_termination_bottleneck
    assert branch.queries.grad.abs().sum() > 0
    assert all(p.grad is None for name, p in model.named_parameters()
               if not name.startswith('vlm_termination_bottleneck.'))
    model.zero_grad(set_to_none=True)
    policy = SkillExpertPolicy(model.config)
    policy.model = model
    batch = dict(zip(keys, images, strict=True))
    batch.update({
        'observation.state': state[:, :14], 'skill_code': code,
        'skill_decoder_state': torch.zeros(2, 14), 'skill_start_state': torch.zeros(2, 14),
        'action': torch.randn(2, 30, 14), 'skill_effective_de': torch.tensor([4, 0]),
        'skill_bspline_states': torch.randn(2, 30, 14),
        'skill_end_xyz': torch.zeros(2, 2, 3), 'skill_end_xyz_valid': torch.ones(2, 2, dtype=torch.bool),
        'skill_focus_uv': goal[:, :4].reshape(2, 2, 2), 'skill_focus_valid': goal[:, 4:].bool(),
        'skill_wrist_focus_uv': torch.zeros(2, 2, 2), 'skill_wrist_focus_valid': torch.ones(2, 2, dtype=torch.bool),
        'skill_wrist_focus_clipped': torch.zeros(2, 2, dtype=torch.bool),
        DISTANCE: torch.tensor([-4, 0]), TERMINAL: torch.zeros(2, dtype=torch.bool),
        PAST_CODES: torch.zeros(2, 12, dtype=torch.long), PAST_VALID: torch.zeros(2, 12, dtype=torch.bool),
        PAST_UV: torch.zeros(2, 12, 2, 2), PAST_UV_VALID: torch.zeros(2, 12, 2, dtype=torch.bool),
        PAST_TERMINAL: torch.zeros(2, 12, dtype=torch.bool), PAST_DISTANCE: torch.zeros(2, 12, dtype=torch.long),
    })
    batch[PAST_VALID][1, 0] = True
    batch[PAST_CODES][1, 0] = 3
    batch[PAST_UV][1, 0] = torch.tensor([[.6, -.4], [-.2, .5]])
    batch[PAST_UV_VALID][1, 0] = torch.tensor([False, True])
    observed = []
    original = model.vlm_termination_from_image_tokens
    def record(memory, state, code, goal):
        observed.append((code.detach().clone(), goal.detach().clone()))
        return original(memory, state, code, goal)
    monkeypatch.setattr(model, 'vlm_termination_from_image_tokens', record)
    loss, metrics = policy(batch)
    assert torch.isfinite(loss)
    assert observed[0][0].tolist() == [3]
    torch.testing.assert_close(observed[0][1], torch.tensor([[.6, -.4, -.2, .5, 0., 1.]]))
    assert metrics['vlm_termination/post_end_conditions_per_frame'] == .5
    loss.backward()
    assert branch.head[-1].weight.grad.abs().sum() > 0
    assert model.gemma_expert.model.layers[-1].self_attn.q_proj.weight.grad.abs().sum() > 0
    isolated = policy.isolated_main_optimizer_grad_groups()['vlm_termination']
    assert {id(p) for p in isolated} == {id(p) for p in branch.parameters()}
    saved = replace(model.config, visual_bottleneck_width=256, visual_bridge_heads=8)
    saved.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.uses_detached_termination_bottleneck and restored.bimanual_spatial_auxiliary
    assert restored.vlm_termination_post_end_steps == 10
    probability = policy.predict_termination_probability(batch)
    assert probability.shape == (2,) and torch.isfinite(probability).all()
