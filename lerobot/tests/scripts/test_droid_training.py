"""DROID plans, no-video-reencode 8D view, and tiny real CPU training."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import subprocess

import numpy as np
import pandas as pd
import pytest
import yaml

from test_droid_dataset import source, prepare

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / 'lerobot/skillVLA/droid/configs'
TRAIN = runpy.run_path(str(CONFIGS / 'train_skills/src/train_skills_config.py'))
VIEW = runpy.run_path(str(CONFIGS / 'src/dataset_view.py'))


@pytest.fixture
def view(source):
    selected, excluded, hashes = prepare.discover(source)
    original = prepare.convert(source, selected, excluded, hashes)
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in original.rglob('*') if p.is_file()}
    result = VIEW['prepare_joint_view'](original, original.with_name('joint_view'))
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in original.rglob('*') if p.is_file()}
    assert before == after
    return original, result


def test_joint_training_view_has_correct_values_stats_and_original_videos(view):
    original, result = view
    df = pd.read_parquet(result / 'data/chunk-000/file-000.parquet')
    np.testing.assert_allclose(np.stack(df.action)[:, :7], np.stack(df['source.action.joint_velocity']))
    np.testing.assert_allclose(np.stack(df['observation.state'])[:, :7], np.stack(df['observation.states.joint_state']))
    np.testing.assert_allclose(np.stack(df.action)[:, 7], df['source.action.gripper_position'])
    assert (result / 'videos').resolve() == original / 'videos'
    info = json.loads((result / 'meta/info.json').read_text())
    stats = json.loads((result / 'meta/stats.json').read_text())
    assert info['features']['action']['shape'] == [8] and len(stats['action']['q01']) == 8
    assert len(stats['observation.state']['q99']) == 8
    assert VIEW['prepare_joint_view'](original, result) == result
    with pytest.raises(ValueError, match='fps'):
        VIEW['prepare_joint_view'](original, result.with_name('wrong_fps'), fps=20)


def test_plan_reads_global_switch_and_freezes_independent_contract(tmp_path):
    path = CONFIGS / 'train_skills/DP/dp_config.yaml'
    config = TRAIN['load_config'](path)
    assert TRAIN['resolve'](config, 'DP')['idle_filter']['enabled'] is False
    frozen = tmp_path / 'frozen'
    frozen.mkdir()
    # Standalone frozen configs do not read live server profiles or live switches.
    global_cfg = {'workflow_domain': 'droid', 'project_root': str(tmp_path), 'dataset_root': 'datasets',
        'outputs_root': 'outputs', 'droid_fps': 15, 'droid_action_space': 'joint_velocity',
        'droid_proprio_space': 'joint_position', 'vla_chunk_size': 15, 'vla_replan_steps': 8,
        'idle_filter': {'DP': True, 'FSQ': False}}
    (frozen / 'global_config.yaml').write_text(yaml.safe_dump(global_cfg))
    # An old frozen request carries explicit datasets rather than today's registry.
    old_local = yaml.safe_load(path.read_text())
    old_local.update(source_dataset='converted', target_dataset='joint_view')
    (frozen / 'dp_config.yaml').write_text(yaml.safe_dump(old_local))
    cfg = TRAIN['load_config'](frozen / 'dp_config.yaml')
    plan = TRAIN['resolve'](cfg, 'DP')
    assert plan['idle_filter']['enabled'] and plan['output'].endswith('_idle')
    assert TRAIN['filter_settings'](cfg, 'DP')['idle_filter'] == {'DP': True}
    cfg['idle_filter']['FSQ'] = True
    assert TRAIN['filter_settings'](cfg, 'DP')['idle_filter'] == {'DP': True}


def tiny_dp_config(view, tmp_path):
    _, dataset = view
    config = TRAIN['load_config'](CONFIGS / 'train_skills/DP/dp_config.yaml')
    config['droid_dataset']['group'] = 'all'  # Tiny fixture UUIDs are not public DROID episodes.
    config.update(project_root=str(tmp_path), dataset_root=str(dataset.parent), target_dataset=dataset.name,
        source_dataset='converted', outputs_root='outputs', dp_device='cpu', dp_steps=1, dp_batch_size=2,
        dp_num_workers=0, dp_n_obs_steps=3, dp_future_action_horizon=3,
        dp_history_encoder_dim=16, dp_down_dims=[8, 16], dp_wandb_enable=False, dp_save_freq=1, dp_log_freq=1)
    return config


def test_dp_real_cpu_update_checkpoint_and_full_state_resume(view, tmp_path, monkeypatch):
    import torch
    torch.set_num_threads(1)
    config = tiny_dp_config(view, tmp_path)
    plan = TRAIN['resolve'](config, 'DP')
    monkeypatch.setattr(sys, 'argv', ['droid_training_test'])
    monkeypatch.setenv('SBD_GPU_MEMORY_GUARD', 'false')
    TRAIN['run_dp'](config, plan)
    last = Path(plan['output']) / 'checkpoints/last'
    assert json.loads((last / 'training_state/training_step.json').read_text())['step'] == 1
    assert (last / 'training_state/optimizer_state.safetensors').is_file()
    config['dp_steps'] = 2
    TRAIN['run_dp'](config, plan)
    assert json.loads((last / 'training_state/training_step.json').read_text())['step'] == 2


def test_fsq_real_cpu_epoch_with_training_only_mask(view, tmp_path):
    import torch
    torch.set_num_threads(1)
    _, dataset = view
    skills = tmp_path / 'skillset/skills'
    skills.mkdir(parents=True)
    for ep in range(2):
        df = pd.read_parquet(dataset / f'data/chunk-000/file-{ep:03d}.parquet')
        for rank, start in enumerate((0, 3)):
            np.savez(skills / f'ep{ep}_skill{rank}.npz', actions=np.stack(df.action)[start:start+3],
                states=np.stack(df['observation.state'])[start:start+3], episode_id=ep,
                task_id=ep, skill_index=rank, frame_start=start, frame_end=start+3)
    module = TRAIN['fsq_module']()
    config = TRAIN['load_config'](CONFIGS / 'train_skills/FSQ/fsq_config.yaml')
    config.update(fsq_device='cpu', fsq_hidden_dim=16, fsq_num_layers=1, fsq_num_epochs=1,
        fsq_batch_size=1, fsq_num_workers=0, fsq_checkpoint_every=1, fsq_wandb_project=None,
        fsq_pair_loss={'type': 'none', 'weight': .1, 'inv_temperature': 5},
        fsq_route_loss={'enabled': False}, fsq_boundary_aug_pmax_step=0)
    plan = {'dataset': str(dataset), 'skillset': str(skills.parent), 'output': str(tmp_path / 'fsq'), 'chunk_size': 3}
    args = TRAIN['build_fsq_args'](config, plan, module)
    mask = tmp_path / 'training_mask.npy'
    np.save(mask, np.asarray([False, True, True, True]))
    args.training_skill_mask_path = str(mask)
    module.main(args)
    saved = torch.load(tmp_path / 'fsq/FSQ_epoch0001.pt', map_location='cpu', weights_only=False)
    assert saved['cfg'].action_dim == 8
    assert saved['cfg'].state_dim == 8
    assert 'optim_state' in saved
    args.resume_from = str(tmp_path / 'fsq/FSQ_epoch0001.pt')
    args.epochs = 2
    module.main(args)
    resumed = torch.load(tmp_path / 'fsq/FSQ_epoch0002.pt', map_location='cpu', weights_only=False)
    assert resumed['epoch'] == 2 and 'optim_state' in resumed


def test_filter_adapters_use_global_flags_and_original_episode_commands(tmp_path):
    import h5py
    from datasets import Dataset
    adapter = runpy.run_path(str(CONFIGS / 'src/training_filter.py'))
    root = tmp_path / 'dataset'
    meta = root / 'meta/droid/episodes'
    meta.mkdir(parents=True)
    raw = tmp_path / 'raw'
    raw.mkdir()
    v = np.arange(420, dtype=np.float32).reshape(60, 7) / 100
    v[25:50] = 0
    with h5py.File(raw / 'trajectory.h5', 'w') as stream:
        stream['action/joint_velocity'] = v
    (meta / '000000.json').write_text(json.dumps({'directory': str(raw)}))
    original = np.asarray([i for i in range(60) if i not in (3, 9)])
    np.savez(meta / '000000.npz', source_frame_index=original)
    class Data:
        def __len__(self): return len(original)
    data = Data()
    data.root = root
    data.hf_dataset = Dataset.from_dict({'episode_index': [0]*len(original), 'frame_index': list(range(len(original)))})
    class BaseSampler:
        indices = list(range(len(original)-3))
    base = BaseSampler()
    off = {'workflow_domain': 'droid', 'idle_filter': {'DP': False}, 'stage': 'DP'}
    assert adapter['apply_frame_sampler'](data, off, base) is base
    on = {**off, 'idle_filter': {'DP': True}}
    result = adapter['apply_frame_sampler'](data, on, base)
    reference = runpy.run_path(str(CONFIGS / 'src/idle_filter.py'))['nonidle_start_mask'](v)[original]
    assert sorted(result.indices) == [i for i in base.indices if reference[i]]
    skills = [{'episode_id': 0, 'frame_start': 0, 'frame_end': 15},
              {'episode_id': 0, 'frame_start': 30, 'frame_end': 40}]
    assert adapter['fsq_skill_mask'](root, skills, {'workflow_domain': 'droid', 'idle_filter': {'FSQ': False}}).all()
    np.testing.assert_equal(adapter['fsq_skill_mask'](root, skills, {'workflow_domain': 'droid', 'idle_filter': {'FSQ': True}}), [True, False])


@pytest.mark.parametrize('stage,script', [('DP', 'submit_train_dp.sh'), ('FSQ', 'submit_train_fsq.sh')])
def test_dry_run_with_existing_job_selector_never_submits(stage, script):
    folder = CONFIGS / 'train_skills' / stage
    snapshots = sorted(str(p) for p in folder.glob('.config_snapshots/*'))
    result = subprocess.run(['bash', str(folder / script), '--75000', '--dry-run'],
        capture_output=True, text=True, timeout=20, env={**os.environ, 'SBD_SERVER': 'rllab'})
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan['stage'] == stage and plan['action_dim'] == plan['state_dim'] == 8
    assert not plan['idle_filter']['enabled'] and plan['vla_chunk_size'] == 15
    assert sorted(str(p) for p in folder.glob('.config_snapshots/*')) == snapshots
