"""The plan's action command stays distinct from the observed finger positions."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.uv_fullskill import DUAL_BOTTLENECK_NOPROPRIO_LABEL as LABEL
from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy
from lerobot.policies.skillVLA.dataset_skillVLA import SkillVLADataset
from lerobot.policies.skillVLA.proprio_spline import resample_skill_proprio, resample_skill_with_gripper_action
from test_pi05_vlm_vsa import tiny_vlm_model
from test_stage1_vsa import _skill_config


def test_commands_use_hold_not_spline_and_keep_pose_endpoints():
    states = np.arange(40, dtype=np.float32).reshape(5, 8)
    commands = np.array([-1, -1, 1, 1, -1], dtype=np.float32)
    target = resample_skill_with_gripper_action(states, commands, 30)
    assert target.shape == (30, 7)
    np.testing.assert_allclose(target[:, :6], resample_skill_proprio(states, 30)[:, :6])
    indices = np.arange(30) * 4 // 29
    np.testing.assert_array_equal(target[:, 6], commands[indices])
    # A one-frame skill is repeated, including the command.
    single = resample_skill_with_gripper_action(states[:1], commands[:1], 30)
    np.testing.assert_array_equal(single, np.repeat(single[:1], 30, axis=0))


def test_dataset_aligns_commands_after_episode_selection_and_keeps_proprio():
    dataset = object.__new__(SkillVLADataset)
    dataset._bspline_gripper_action = True
    dataset._bspline_skill_state_tokens, dataset._bspline_target_cache = 30, {}
    dataset.meta = SimpleNamespace(episodes={5: {"length": 4, "dataset_from_index": 20}})
    dataset.reader = SimpleNamespace(_absolute_to_relative_idx={i: i - 20 for i in range(20, 24)})
    dataset._chunk_end_state_cache = torch.arange(32).reshape(4, 8).float()
    dataset._bspline_gripper_action_cache = torch.tensor([-1., 1., -1., 1.])
    target = dataset._bspline_skill_states(episode_index=5, skill_start_frame=1, skill_length=2)
    torch.testing.assert_close(target[0, :6], dataset._chunk_end_state_cache[1, :6])
    assert target[0, 6] == 1 and target[-1, 6] == -1
    assert dataset._chunk_end_state_cache.shape == (4, 8)
    dataset._bspline_gripper_action, dataset._bspline_target_cache = False, {}
    legacy = dataset._bspline_skill_states(episode_index=5, skill_start_frame=1, skill_length=2)
    assert legacy.shape == (30, 8)
    torch.testing.assert_close(legacy[0], dataset._chunk_end_state_cache[1])


def test_checkpoint_and_loss_normalization_separate_action_from_finger_statistics(tmp_path):
    cfg = _skill_config(LABEL, bspline_gripper_action=True, skill_plan_state_dim=7,
                        chunk_end_state_q01=[-2.] * 6 + [-.04, -.01],
                        chunk_end_state_q99=[2.] * 6 + [.01, .04])
    cfg.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.bspline_gripper_action and restored.chunk_end_state_dim == 8
    assert restored.bspline_plan_quantiles == ([-2.] * 6 + [-1.], [2.] * 6 + [1.])
    policy = object.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.config = restored
    target = torch.zeros(1, 30, 7)
    target[:, :, 6] = 1
    normalized = policy._prepare_independent_skill_plan(
        {"skill_bspline_states": target}, torch.tensor([1]), None,
    )
    torch.testing.assert_close(normalized, target)
    with pytest.raises(ValueError, match="7D LIBERO"):
        replace(cfg, skill_plan_state_dim=8)
    legacy = replace(cfg, bspline_gripper_action=False, skill_plan_state_dim=8)
    assert legacy.bspline_plan_quantiles == (cfg.chunk_end_state_q01, cfg.chunk_end_state_q99)


@pytest.mark.parametrize('spatial_dim,gripper_range', [(8, [-1., 1.]), (6, [0., 1.])])
def test_seven_dimensional_flow_trains_and_visualizes_with_eight_dimensional_proprio(monkeypatch, spatial_dim, gripper_range):
    torch.manual_seed(17)
    model = tiny_vlm_model(monkeypatch, LABEL, bspline_gripper_action=True,
                           skill_plan_state_dim=7, chunk_end_state_dim=spatial_dim,
                           chunk_end_state_q01=[-2.]*spatial_dim, chunk_end_state_q99=[2.]*spatial_dim,
                           bspline_gripper_action_range=gripper_range, device="cpu")
    images = [torch.rand(1, 3, 4, 4), torch.rand(1, 3, 4, 4)]
    state, code = torch.randn(1, 32), torch.tensor([1])
    goal = torch.tensor([[.2, -.3, 1., .1, .2, .3]])
    residual = model(images, state, code, torch.randn(1, 10, 32), end_pose=goal,
                     fullskill_target=torch.randn(1, 30, 7))
    flow = model.take_fullskill_flow_loss()
    (residual.square().mean() + flow.mean()).backward()
    assert model.bspline_proprio_head[-1].out_features == 7
    assert model.bspline_noise_input.in_features == 7
    assert model.bspline_proprio_head[-1].weight.grad[-1].abs().sum() > 0
    model.eval()
    with torch.no_grad():
        model.sample_actions(images, state, code, num_steps=1, end_pose=goal)
        plan = model.generate_visualization_plan(code, goal)
    assert plan.shape == (1, 30, 7) and torch.isfinite(plan).all()
