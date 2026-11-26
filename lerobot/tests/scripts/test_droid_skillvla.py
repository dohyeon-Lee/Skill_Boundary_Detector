"""Tiny real FSQ -> DROID SkillVLA build, geometry and shared PT contracts."""
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import pytest

from test_droid_fsq_eval import evaluated_models
from test_droid_fsq_action_space import skills
from test_droid_training import view, source, CONFIGS

BUILD = runpy.run_path(str(CONFIGS / 'train_skillVLA/build_data/src/build_dataset.py'))
ROOT = CONFIGS.parents[3]
SHARED = CONFIGS.parents[1] / 'libero/configs'


@pytest.mark.parametrize('mode', ['joint_velocity', 'delta_pose'])
def test_fsq_build_preserves_controller_and_uses_real_eef_geometry(evaluated_models, mode):
    cfg, dataset, skillset = evaluated_models
    checkpoint = Path(cfg['outputs_root']) / 'FSQ' / ('tiny_' + mode) / 'FSQ_epoch0001.pt'
    plan = BUILD['settings'](dict(cfg, fsq_checkpoint=str(checkpoint), output_name='skillvla_' + mode))
    originals = {p: BUILD['sha'](p) for p in [*dataset.glob('data/**/*.parquet'), *skillset.glob('skills/*.npz')]}
    output = BUILD['build'](plan, device='cpu')
    result = BUILD['verify'](output)
    assert result['episode_count'] == 2 and result['skill_count'] == 4
    info = json.loads((output/'skillvla/meta/info.json').read_text())
    assert info['features']['action']['shape'] == [8]
    assert info['features']['observation.state']['shape'] == [8]
    assert info['skill_spatial_state_key'] == 'observation.states.ee_state'
    assert info['skill_gripper_action_index'] == 7 and info['proprio_grounding'] == 'none'
    assert info['skill_gripper_action_range'] == [0., 1.]
    assert len([k for k in info['features'] if k.startswith('observation.images.')]) == 2
    for path in dataset.glob('data/**/*.parquet'):
        before = pd.read_parquet(path)
        after = pd.read_parquet(output/'skillvla'/path.relative_to(dataset))
        for key in ('action', 'observation.state', 'observation.states.ee_state', 'frame_index'):
            np.testing.assert_array_equal(np.stack(before[key]), np.stack(after[key]))
    assert (output/'skillvla/videos').resolve() == (dataset/'videos').resolve()
    with np.load(output/'skill_initial_state.npz') as iss:
        assert iss['iss_windows'].shape[-1] == 6
        assert int(iss['state_dim']) == 6
    with np.load(output/'skill_focus_uv.npz') as focus:
        assert len(focus['focus_uv']) == 4
        camera = focus['camera_grounded_transform'][0]
        rows = pd.read_parquet(next((dataset/'data').rglob('*.parquet')))
        xyz = np.stack(rows['observation.states.ee_state'])[:, :3]
        with np.load(dataset/'meta/droid/episodes/000000.npz') as calib:
            k, pose = calib['exterior_1.intrinsics'], calib['exterior_1.camera_to_base'][0]
        proj = np.eye(4); proj[:3, :3] = k
        projected = np.c_[xyz, np.ones(len(xyz))] @ (proj @ np.linalg.inv(pose)).T
        pixels = projected[:, :2] / projected[:, 2:3]
        height, width = int(focus['image_height']), int(focus['image_width'])
        expected = (2*np.clip(pixels, [0, 0], [width-1, height-1])/[width-1, height-1]-1).astype(np.float32)
        from lerobot.policies.skillVLA.chunk_patch_target import project_top_points
        actual, _ = project_top_points(xyz, camera, height=height, width=width)
        np.testing.assert_allclose(actual, expected, atol=1e-6)
    stamp = (output/'skill_latents.npz').stat().st_mtime_ns
    assert BUILD['build'](plan, device='cpu') == output
    assert (output/'skill_latents.npz').stat().st_mtime_ns == stamp
    assert originals == {p: BUILD['sha'](p) for p in originals}
    # Exercise the real shared training dataset, including jitter, calibration
    # and the 15-step controller chunk (no fake EEF from joint[:3]).
    from lerobot.policies.skillVLA.dataset_skillVLA import (
        SkillVLADataset, SKILL_BSPLINE_STATES, SKILL_CHUNK_END_STATE, SKILL_END_XYZ,
    )
    training = SkillVLADataset('droid/test', root=output/'skillvla', episodes=[0, 1],
        video_backend='pyav', delta_timestamps={'action': [i/15 for i in range(15)]},
        include_predictor_start_inputs=True, bspline_skill_state_tokens=30,
        bspline_gripper_action=True, include_chunk_end_state_target=True,
        include_skill_end_xyz_target=True,
        include_chunk_end_patch_target=True, chunk_end_state_horizon=15,
        jitter_pmax_percent=25, jitter_preserve_gripper=True)
    item = training[1]
    assert item['action'].shape == (15, 8)
    assert item['observation.state'].shape == (8,)
    assert item[SKILL_BSPLINE_STATES].shape == (30, 7)
    assert item[SKILL_CHUNK_END_STATE].shape == (6,)
    assert item[SKILL_END_XYZ].shape == (3,)
    assert (item[SKILL_BSPLINE_STATES][:, -1] >= 0).all()
    # All components resolve from the same built dataset and frozen DROID prior.
    import yaml
    common = yaml.safe_load((CONFIGS/'train_skillVLA/PT/stage1_common_config.yaml').read_text())
    prior = output.parent/'prior'
    prior.mkdir(exist_ok=True)
    (prior/'model.safetensors').touch()
    for component, filename in [('VSA', 'vsa'), ('Predictor', 'predictor')]:
        path = CONFIGS/f'train_skillVLA/PT/{component}/{filename}_train_config.yaml'
        local = yaml.safe_load(path.read_text())
        params = dict(cfg, **local)
        params['dataset'] = dict(source=dataset.name, run=output.name, skillvla_root='skillvla')
        params['transition_randomization'] = common['transition_randomization']
        params['warm_start'] = dict(common['warm_start'], **local.get('warm_start', {}), pi_base=str(prior))
        params['warm_start']['tokenizer'] = str(CONFIGS.parents[3]/'models/paligemma-3b-pt-224-tokenizer')
        params['workflow_domain'] = 'droid'
        params['architecture'] = dict(local.get('architecture', {}), chunk_size=15)
        params['execution'] = dict(action_steps=8)
        params['idle_filter'] = {'PT_VSA': True, 'PT_PRED': True, 'PT_TERM': True}
        params.setdefault('vision', {})['dino_model'] = str(CONFIGS.parents[3]/'models/dinov3-vits16')
        if component == 'VSA':
            resolver = runpy.run_path(str(SHARED/'train_skillVLA/PT/src/stage1_train_config.py'))
        else:
            resolver = runpy.run_path(str(SHARED/'train_skillVLA/PT/src/auxiliary/auxiliary_train_config.py'))
        settings = (resolver['build_launch_settings'](params, path) if component == 'VSA'
                    else resolver['build_launch_settings'](params))
        assert settings['pi_base'] == prior
        assert settings['dataset_staging_dereference'] is True
        assert settings['skip_dataset_staging'] is False
        assert json.loads(settings['droid_idle_filter'])['stage'] == 'PT_' + {'VSA':'VSA', 'Predictor':'PRED'}[component]
        if component == 'VSA':
            assert settings['chunk_size'] == 15 and settings['n_action_steps'] == 8
            assert settings['chunk_end_state_dim'] == 6 and settings['skill_plan_state_dim'] == 7
            assert json.loads(settings['bspline_gripper_action_range']) == [0., 1.]
            assert settings['dataloader_pin_memory'] and settings['dataloader_persistent_workers']
            assert settings['dataloader_prefetch_factor'] == 4


def test_unselected_fsq_and_pt_fail_before_scheduling():
    with pytest.raises(ValueError, match='Select fsq_checkpoint'):
        BUILD['settings']({'fsq_checkpoint': ''})
    shared = runpy.run_path(str(SHARED / 'train_skills/src/train_skills_config.py'))
    for component, filename in [('VSA', 'vsa'), ('Predictor', 'predictor')]:
        with pytest.raises(ValueError, match='Select dataset.run'):
            shared['_merge_global'](CONFIGS / f'train_skillVLA/PT/{component}/{filename}_train_config.yaml',
                dict(stage1_component=component, dataset={'run': ''}))


def test_droid_pose6_gripper_target_and_quantiles():
    from lerobot.policies.skillVLA.proprio_spline import resample_skill_with_gripper_action
    from lerobot.policies.skill_expert.configuration_skill_expert import SkillExpertConfig
    pose = np.ones((4, 6), dtype=np.float32)
    commands = np.array([0., 0., 1., 1.], dtype=np.float32)
    target = resample_skill_with_gripper_action(pose, commands, 7)
    assert target.shape == (7, 7) and set(target[:, -1]) == {0., 1.}
    property_fn = SkillExpertConfig.bspline_plan_quantiles.fget
    from types import SimpleNamespace
    config = SimpleNamespace(bspline_gripper_action=True, bspline_gripper_action_range=[0., 1.],
        chunk_end_state_q01=[-2.]*6, chunk_end_state_q99=[2.]*6)
    assert property_fn(config) == ([-2.]*6+[0.], [2.]*6+[1.])


def test_resubmitting_frozen_component_preserves_shared_yaml(tmp_path):
    import os
    import subprocess
    config = tmp_path/'configs'
    component = config/'PT/VSA/vsa_train_config.yaml'
    component.parent.mkdir(parents=True)
    global_path = config/'global_config.yaml'
    global_path.write_text('workflow_domain: snapshot_test\n')
    common = component.parent.parent/'stage1_common_config.yaml'
    common.write_text('warm_start: {pi_base: frozen_prior}\n')
    component.write_text('stage1_common: true\n')
    script = SHARED/'src/snapshot_config.sh'
    command = 'source "$1"; first=$(snapshot_config "$2"); export SBD_GLOBAL_CONFIG="$(dirname "$first")/global_config.yaml"; snapshot_config "$first"'
    env = dict(os.environ, SBD_GLOBAL_CONFIG=str(global_path))
    result = subprocess.run(['bash', '-c', command, 'snapshot_test', str(script), str(component)],
        env=env, text=True, capture_output=True, check=True)
    snapshot = Path(result.stdout.strip())
    assert snapshot.with_name('stage1_common_config.yaml').read_text() == common.read_text()
