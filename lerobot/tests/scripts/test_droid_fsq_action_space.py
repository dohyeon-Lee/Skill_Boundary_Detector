"""Independent DROID FSQ action choices on unchanged SBD boundaries."""
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.spatial.transform import Rotation

from test_droid_training import view, source, TRAIN, CONFIGS

ADAPTER = runpy.run_path(str(CONFIGS / 'src/fsq_action_data.py'))
REP = runpy.run_path(str(TRAIN['SHARED'] / 'FSQ/src/action_representation.py'))


@pytest.fixture
def skills(view, tmp_path):
    _, dataset = view
    root = tmp_path / 'skillset'
    (root / 'skills').mkdir(parents=True)
    (root / 'skillset_manifest.json').write_text('{"test": "original joint SBD"}')
    (root / '.complete').write_text('complete')
    for ep in range(2):
        path = dataset / f'data/chunk-000/file-{ep:03d}.parquet'
        rows = pd.read_parquet(path)
        n = len(rows)
        # Rotation crosses the +/-pi branch; subtraction would be incorrect.
        targets = np.zeros((n, 6), dtype=np.float32)
        targets[:, 0] = np.arange(n)*.02
        targets[:, 5] = np.deg2rad(179 + np.arange(n)*2)
        rows['source.action.cartesian_position'] = list(targets)
        rows.to_parquet(path)
        for rank, start in enumerate((0, 3)):
            np.savez(root / 'skills' / f'ep{ep}_skill{rank}.npz',
                actions=np.stack(rows.action)[start:start+3],
                states=np.stack(rows['observation.state'])[start:start+3],
                episode_id=ep, task_id=ep, skill_index=rank, frame_start=start, frame_end=start+3)
    return dataset, root


def config_for(dataset, mode):
    cfg = TRAIN['load_config'](CONFIGS / 'train_skills/FSQ/fsq_config.yaml')
    cfg['droid_dataset']['group'] = 'all'
    cfg.update(project_root=str(dataset.parent), dataset_root=str(dataset.parent),
        target_dataset=dataset.name, source_dataset=dataset.name,
        fsq_action_space=mode, fsq_device='cpu')
    return cfg


def test_switch_changes_representation_and_output_but_keeps_segmentation(skills):
    dataset, _ = skills
    cfg = config_for(dataset, 'joint_velocity')
    joint = TRAIN['resolve'](cfg, 'FSQ')
    pose = TRAIN['resolve'](dict(cfg, fsq_action_space='delta_pose'), 'FSQ')
    assert pose['dataset'] == joint['dataset'] and pose['skillset'] == joint['skillset']
    assert joint['action_dim'] == joint['state_dim'] == 8
    assert pose['action_dim'] == pose['state_dim'] == 7
    assert pose['proprio_space'] == 'eef_pose'
    assert pose['output'] != joint['output'] and 'delta_pose' in pose['output']
    with pytest.raises(ValueError, match='fsq_action_space'):
        TRAIN['resolve'](dict(cfg, fsq_action_space='delta_joint'), 'FSQ')


def test_pose_adapter_preserves_selection_boundaries_and_matches_dp_convention(skills):
    dataset, root = skills
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'skills').glob('*.npz')}
    _, states, actions, metadata = ADAPTER['load_pose_skills'](root/'skills', dataset, episode_ids=[1])
    prepare_fsq_actions = REP['prepare_fsq_actions']
    pack_action_state_context = REP['pack_action_state_context']
    unpack_delta_action_context = REP['unpack_delta_action_context']
    from lerobot.processor.eef_delta_pose import to_eef_delta_pose_actions_numpy
    assert [m['episode_id'] for m in metadata] == [1, 1]
    assert [(m['frame_start'], m['frame_end']) for m in metadata] == [(0, 3), (3, 6)]
    for a, s in zip(actions, states):
        delta = prepare_fsq_actions(a, s, 'delta_pose', [6])
        np.testing.assert_allclose(delta, to_eef_delta_pose_actions_numpy(a, s[0]), atol=1e-6)
        np.testing.assert_array_equal(delta[:, 6], a[:, 6])
        rotations = Rotation.from_rotvec(a[1:, 3:6]) * Rotation.from_rotvec(a[:-1, 3:6]).inv()
        np.testing.assert_allclose(delta[1:, 3:6], rotations.as_rotvec(), atol=1e-6)
        packed = pack_action_state_context(a, s)
        shifted = unpack_delta_action_context(packed[1:], 7, [6], 'delta_pose')
        np.testing.assert_allclose(shifted, to_eef_delta_pose_actions_numpy(a[1:], s[1]), atol=1e-6)
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'skills').glob('*.npz')}
    path = root/'skills/ep1_skill0.npz'
    with np.load(path) as data:
        broken = dict(data)
    broken['actions'] = broken['actions'] + 1
    np.savez(path, **broken)
    with pytest.raises(ValueError, match='disagree'):
        ADAPTER['load_pose_skills'](root/'skills', dataset, episode_ids=[1])


@pytest.mark.parametrize('mode,dim,gripper', [('joint_velocity', 8, 7), ('delta_pose', 7, 6)])
@pytest.mark.parametrize('levels', [[3, 3, 3], [3, 3, 3, 2]])
def test_real_training_and_resume_with_unchanged_skill_files(skills, tmp_path, monkeypatch, mode, dim, gripper, levels):
    torch.set_num_threads(1)
    from lerobot.utils import gpu_memory_guard
    guard_dir = tmp_path / 'guard'
    guard_dir.mkdir()
    monkeypatch.setenv('SBD_GPU_GUARD_CHILD', '1')
    monkeypatch.setenv('SBD_GPU_GUARD_RUN_DIR', str(guard_dir))
    monkeypatch.setattr(gpu_memory_guard, '_ready_written', False)
    dataset, root = skills
    cfg = config_for(dataset, mode)
    cfg.update(fsq_hidden_dim=16, fsq_num_layers=1, fsq_num_epochs=1, fsq_batch_size=1,
        fsq_num_workers=0, fsq_checkpoint_every=1, fsq_wandb_project=None,
        fsq_boundary_aug_pmax_step=1, fsq_quantizer={'type': 'fsq', 'levels': levels})
    output = tmp_path / mode
    plan = dict(dataset=str(dataset), skillset=str(root), output=str(output), chunk_size=3,
        action_space=mode, dataset_selection=dict(episode_ids=[0, 1]))
    if mode == 'delta_pose':
        plan['action_stats_path'] = str(tmp_path / 'contracts/pose.stats.json')
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'skills').glob('*.npz')}
    TRAIN['run_fsq'](cfg, plan)
    assert (guard_dir / 'ready').is_file()
    saved = torch.load(output/'FSQ_epoch0001.pt', map_location='cpu', weights_only=False)
    assert saved['cfg'].action_dim == dim and saved['cfg'].state_dim == dim
    assert list(saved['cfg'].action_gripper_indices) == [gripper]
    assert saved['cfg'].action_space == ('delta_pose' if mode == 'delta_pose' else 'dataset')
    assert 'optim_state' in saved and 'scheduler_state' in saved
    cfg['fsq_num_epochs'] = 2
    TRAIN['run_fsq'](cfg, plan)
    resumed = torch.load(output/'FSQ_epoch0002.pt', map_location='cpu', weights_only=False)
    assert resumed['epoch'] == 2
    if mode == 'delta_pose':
        stats = json.loads(Path(plan['action_stats_path']).read_text())
        assert len(stats['action']['q01']) == 7
        assert len(stats['observation.state']['q01']) == 7
        assert stats['contract']['episode_ids'] == [0, 1]
        import FSQ as fsq
        encoder, saved_config = fsq.load_fsq_encoder(output/'FSQ_epoch0002.pt', 'cpu')
        _, states, actions, metadata = ADAPTER['load_pose_skills'](root/'skills', dataset, episode_ids=[0, 1])
        delta = REP['prepare_fsq_actions'](actions[0], states[0], 'delta_pose', [6])
        assert np.isfinite(encoder.encode_actions_numpy(delta)).all()
        assert 0 <= encoder.encode_actions_index(delta) < int(np.prod(levels))
        contexts = fsq.build_boundary_augmentation_contexts(
            [REP['pack_action_state_context'](a, s) for a, s in zip(actions, states)], metadata, pmax=1)
        monkeypatch.setattr(fsq, 'sample_boundary_augmented_segment', lambda context, **kwargs:
            (context.trajectory[context.start+1:context.end].copy(), 0, 1))
        training = fsq.FSQTrajectoryDataset(states, states, actions, metadata, str(dataset), saved_config,
            training=True, boundary_contexts=contexts,
            adjacent_skill_indices=fsq.build_adjacent_skill_indices(metadata))
        item = training[0]
        expected = REP['prepare_fsq_actions'](actions[0][1:], states[0][1:], 'delta_pose', [6])
        np.testing.assert_allclose(item['augmented_action_seq'][:2],
            fsq.normalize_action_sequence(expected, saved_config.action_q01, saved_config.action_q99,
                gripper_weight=.1, gripper_indices=[6], clip=True), atol=1e-6)
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'skills').glob('*.npz')}
