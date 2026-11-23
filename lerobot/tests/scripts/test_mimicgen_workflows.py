"""Domain boundary contracts: camera orientation, checkpoint roots and launch routing."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_coffee_processor_preserves_pixels_and_libero_axis_angle():
    from lerobot.processor.env_processor import LiberoProcessorStep, MimicGenProcessorStep
    image = torch.arange(12).reshape(1, 3, 2, 2).float()
    obs = {
        'observation.images.image': image,
        'observation.robot_state': {
            'eef': {'pos': torch.tensor([[.1, .2, .3]]), 'quat': torch.tensor([[0., .8, 0., -.6]])},
            'gripper': {'qpos': torch.tensor([[.03, -.03]])},
        },
    }
    coffee = MimicGenProcessorStep().observation(obs)
    libero = LiberoProcessorStep().observation(obs)
    assert torch.equal(coffee['observation.images.image'], image)
    assert torch.equal(libero['observation.images.image'], image.flip((2, 3)))
    assert torch.equal(coffee['observation.state'], libero['observation.state'])
    assert coffee['observation.state'][0, 4] > np.pi
    assert not coffee['observation.wrist_camera_mirror_x'].item()


def test_wrist_projection_supports_mixed_domain_batches():
    from lerobot.policies.skill_expert.wrist_patch_target import camera_for_batch, project_into_wrist
    pos = torch.zeros(2, 3)
    angle = torch.zeros(2, 3)
    goal = torch.tensor([[.02, .03, .3], [.02, .03, .3]])
    camera = camera_for_batch({'observation.wrist_camera_mirror_x': torch.tensor([True, False])})
    pixels, depth = project_into_wrist(pos, angle, goal, camera, height=256, width=256)
    assert torch.allclose(pixels[0, 0] + pixels[1, 0], torch.tensor(255.))
    assert torch.allclose(pixels[0, 1], pixels[1, 1])
    assert torch.allclose(depth[0], depth[1])


def test_coffee_camera_targets_use_shared_projection_without_mirroring(tmp_path):
    import json
    import pandas as pd
    builder = load('lerobot/skillVLA/mimicgen/configs/train_skillVLA/build_data/src/build_dataset.py', 'coffee_uv_test')
    raw = tmp_path / 'raw'; (raw / 'meta').mkdir(parents=True); (raw / 'data').mkdir()
    dataset = tmp_path / 'run/skillvla'; (dataset / 'meta').mkdir(parents=True)
    projection = np.eye(4); projection[0, 0] = projection[1, 1] = 100
    projection[0, 2] = projection[1, 2] = 128
    (raw / 'meta/mimicgen_camera.json').write_text(json.dumps({'width':256,'height':256,'top_world_to_pixel':projection.tolist()}))
    states = np.array([[.1,.1,1.,0,0,0,.03,-.03],[.2,.2,1.,0,0,0,.03,-.03]],np.float32)
    pd.DataFrame({'episode_index':[0,0],'frame_index':[0,1],'observation.state':list(states)}).to_parquet(raw / 'data/frames.parquet')
    (dataset / 'meta/info.json').write_text(json.dumps({'proprio_grounding':'episode_start_xyz'}))
    latents = tmp_path / 'latents.npz'
    np.savez(latents,episode_id=[0],task_id=[0],skill_index=[0],frame_start=[0],frame_end=[2])
    assert builder.camera_targets(raw,dataset,latents) == 1
    with np.load(dataset.parent / 'skill_focus_uv.npz') as artifact:
        assert np.allclose(artifact['start_focus_uv_raw_pixels'][0], [138,138])
        assert np.allclose(artifact['focus_uv_raw_pixels'][0], [148,148])
        from lerobot.policies.skillVLA.focus_projection import project_eef
        endpoint = project_eef(states[1,:3]-states[0,:3],artifact['camera_grounded_transform'][0],height=256,width=256)
        assert np.allclose(endpoint.raw_xy,[148,148])


def test_coffee_rejects_demo_exact_reset():
    from lerobot.envs.configs import MimicGenEnv
    assert MimicGenEnv().type == 'mimicgen'
    with pytest.raises(ValueError,match='no per-demo'):
        MimicGenEnv(init_states=True)


def test_yonsei_selects_mimicgen_resolvers_not_abc_or_libero_defaults():
    remote = load('tools/remote/yonsei.py','coffee_remote_test')
    config,resolver,extra = remote.config_and_resolver(
        ROOT / 'lerobot/skillVLA/mimicgen/configs/train_skills/DP/submit_train_dp.sh',[],{},ROOT)
    assert '/mimicgen/' in str(config)
    assert str(resolver).endswith('/libero/configs/train_skills/src/train_skills_config.py')
    assert extra == ['--stage','dp']


def test_coffee_yaml_is_authoritative_over_stale_workflow_exports(monkeypatch):
    config_module = load('lerobot/skillVLA/libero/configs/train_skills/src/train_skills_config.py', 'coffee_yaml_authority')
    monkeypatch.setenv('DP_CHECKPOINT', '050000')
    monkeypatch.setenv('DP_RUN_NAME', 'stale_other_domain')
    cfg = {'workflow_domain': 'mimicgen', 'dp_checkpoint': '010000'}
    assert config_module.get_value(cfg, 'dp_checkpoint') == '010000'
    assert config_module.get_value(cfg, 'dp_run_name', 'current') == 'current'
