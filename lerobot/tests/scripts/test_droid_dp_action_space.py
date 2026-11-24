"""DROID EEF coordinates, selected-corpus stats, real CPU checkpoint and resume."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pandas as pd
import pytest

from test_droid_dataset import source
from test_droid_training import view, tiny_dp_config, TRAIN, VIEW, CONFIGS

STATS = runpy.run_path(str(CONFIGS / 'src/dp_action_stats.py'))


def test_pose_rotation_composition_is_not_vector_subtraction():
    import torch
    from scipy.spatial.transform import Rotation
    from lerobot.processor.eef_delta_pose import to_eef_delta_pose_actions, to_eef_delta_pose_actions_numpy
    rng = np.random.default_rng(42)
    actions = rng.normal(size=(3, 5, 7))
    state = rng.normal(size=(3, 7))
    actions[..., 6] = rng.uniform(size=(3, 5))
    # Includes a near-pi increment and noncommuting rotations.
    state[0, 3:6] = [0, 0, 0]
    actions[0, 0, 3:6] = [np.pi - 1e-8, 0, 0]
    actual = to_eef_delta_pose_actions(torch.tensor(actions), torch.tensor(state)).numpy()
    previous = np.concatenate([state[:, None], actions[:, :-1]], axis=1)
    expected = np.concatenate([actions[..., :3] - previous[..., :3],
        (Rotation.from_rotvec(actions[..., 3:6].reshape(-1, 3)) *
         Rotation.from_rotvec(previous[..., 3:6].reshape(-1, 3)).inv()).as_rotvec().reshape(3, 5, 3),
        actions[..., 6:7]], axis=-1)
    np.testing.assert_allclose(actual, expected, atol=1e-9)
    np.testing.assert_allclose(to_eef_delta_pose_actions_numpy(actions, state), actual, atol=1e-9)
    np.testing.assert_allclose(to_eef_delta_pose_actions(torch.tensor(actions[0]), torch.tensor(state[0])), actual[0])
    independent = to_eef_delta_pose_actions(torch.tensor(actions[:, 0]), torch.tensor(state)).numpy()
    np.testing.assert_allclose(independent, actual[:, 0])
    zeros = to_eef_delta_pose_actions(torch.zeros(2, 3, 7), torch.zeros(2, 7))
    assert torch.equal(zeros, torch.zeros_like(zeros))
    # SBD target probes must see precisely the same 7D coordinates as training.
    manifold = runpy.run_path(str(CONFIGS.parents[1] / 'libero/configs/train_skills/build_data/src/action_manifold.py'))
    probe = manifold['to_model_action_chunk'](actions[0], state[0], action_mode='delta_pose')
    np.testing.assert_allclose(probe[:, [0, 1, 2, 6]], actual[0][:, [0, 1, 2, 6]], atol=1e-6)
    # SBD uses float32. Around pi, equivalent rotation vectors can switch sign.
    np.testing.assert_allclose(Rotation.from_rotvec(probe[:, 3:6]).as_matrix(),
        Rotation.from_rotvec(actual[0, :, 3:6]).as_matrix(), atol=1e-6)


def test_eef_view_preserves_source_and_videos_and_uses_recorded_targets(view, tmp_path):
    from scipy.spatial.transform import Rotation
    original, joint = view
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in original.rglob('*') if p.is_file()}
    result = VIEW['prepare_view'](original, joint.with_name('eef_view'), mode='delta_pose')
    assert (result / 'videos').resolve() == original / 'videos'
    data = pd.read_parquet(result / 'data/chunk-000/file-000.parquet')
    target = np.stack(data['source.action.cartesian_position'])
    actual = np.stack(data.action)
    np.testing.assert_allclose(actual[:, :3], target[:, :3])
    np.testing.assert_allclose(actual[:, 3:6], Rotation.from_euler('xyz', target[:, 3:]).as_rotvec(), atol=1e-7)
    np.testing.assert_allclose(actual[:, 6], data['source.action.gripper_position'])
    np.testing.assert_allclose(np.stack(data['observation.state'])[:, :6], np.stack(data['observation.states.ee_state']))
    assert json.loads((result / 'meta/info.json').read_text())['features']['action']['shape'] == [7]
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in original.rglob('*') if p.is_file()}
    assert VIEW['prepare_view'](original, result, mode='delta_pose') == result
    with pytest.raises(FileExistsError, match='contract'):
        VIEW['prepare_joint_view'](original, result)


def test_normalization_matches_training_chunks_and_excludes_other_episodes(view, tmp_path):
    from lerobot.processor.eef_delta_pose import to_eef_delta_pose_actions_numpy
    original, joint = view
    dataset = VIEW['prepare_view'](original, joint.with_name('eef_stats'), mode='delta_pose')
    expected = []
    # The second episode is never part of this normalization corpus.
    episode = pd.read_parquet(dataset / 'data/chunk-000/file-000.parquet')
    actions, states = np.stack(episode.action), np.stack(episode['observation.state'])
    for t in range(len(actions)-2):
        expected.append(to_eef_delta_pose_actions_numpy(actions[t:t+3], states[t]))
    expected = np.concatenate(expected)
    target = tmp_path / 'stats.json'
    STATS['prepare'](dataset, target, horizon=3, episode_ids=[0])
    result = json.loads(target.read_text())
    assert result['pose_input'] == 'eef_pose' and result['used_episode_ids'] == [0]
    assert result['action_samples'] == len(expected)
    for key, values in [('mean', expected.mean(0)), ('std', expected.std(0)), ('min', expected.min(0)), ('max', expected.max(0))]:
        np.testing.assert_allclose(result['action'][key], values, atol=1e-7)
    before = target.read_bytes()
    STATS['prepare'](dataset, target, horizon=3, episode_ids=[0])
    assert target.read_bytes() == before
    with pytest.raises(ValueError, match='different corpus/horizon'):
        STATS['prepare'](dataset, target, horizon=4, episode_ids=[0])


def test_dp_selector_changes_only_its_view_and_output_and_resume_is_full_state(view, tmp_path, monkeypatch):
    import torch
    torch.set_num_threads(1)
    original, _ = view
    cfg = tiny_dp_config(view, tmp_path)
    joint = TRAIN['resolve'](cfg, 'DP')
    assert '_delta_joint_future' in Path(joint['output']).name
    assert joint['action_space'] == 'joint_velocity'
    legacy = dict(cfg)
    legacy.pop('dp_action_space')
    assert '_delta_joint_future' not in Path(TRAIN['resolve'](legacy, 'DP')['output']).name
    cfg['dp_action_space'] = 'delta_pose'
    plan = TRAIN['resolve'](cfg, 'DP')
    assert plan['dataset'] != joint['dataset'] and plan['output'] != joint['output']
    assert plan['action_dim'] == plan['state_dim'] == 7 and plan['proprio_space'] == 'eef_pose'
    assert cfg['droid_action_space'] == 'joint_velocity'  # FSQ/VLA's global interface stays independent.
    VIEW['prepare_view'](original, plan['dataset'], mode='delta_pose')
    TRAIN['_contract'](cfg, plan)
    desired = TRAIN['build_dp_config'](cfg, plan)
    assert desired.policy.use_delta_pose_actions and desired.policy.delta_pose_input == 'eef_pose'
    monkeypatch.setattr(sys, 'argv', ['droid_eef_training_test'])
    monkeypatch.setenv('SBD_GPU_MEMORY_GUARD', 'false')
    TRAIN['run_dp'](cfg, plan)
    last = Path(plan['output']) / 'checkpoints/last'
    assert json.loads((last / 'training_state/training_step.json').read_text())['step'] == 1
    assert (last / 'training_state/optimizer_state.safetensors').is_file()
    cfg['dp_steps'] = 2
    TRAIN['run_dp'](cfg, plan)
    assert json.loads((last / 'training_state/training_step.json').read_text())['step'] == 2
    saved = json.loads((last / 'pretrained_model/config.json').read_text())
    assert saved['delta_pose_input'] == 'eef_pose'


def test_invalid_action_selector_fails_before_submission(view, tmp_path):
    cfg = tiny_dp_config(view, tmp_path)
    cfg['dp_action_space'] = 'cartesian_velocity'
    with pytest.raises(ValueError, match='dp_action_space'):
        TRAIN['resolve'](cfg, 'DP')
