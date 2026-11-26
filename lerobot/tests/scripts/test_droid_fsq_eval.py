"""Real tiny FSQ train -> encode -> DROID video HTML, with immutable sources."""
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import shutil
import sys

import numpy as np
import pytest

from test_droid_fsq_action_space import skills, config_for
from test_droid_training import source, view, TRAIN, CONFIGS

SRC = CONFIGS / 'train_skills/FSQ/eval/src'
PLAN = runpy.run_path(str(SRC / 'config.py'))
spec = importlib.util.spec_from_file_location('test_droid_fsq_eval_runner', SRC / 'run_eval.py')
RUN = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RUN)


@pytest.fixture
def evaluated_models(skills, tmp_path):
    import torch
    torch.set_num_threads(1)
    dataset, original = skills
    skillset = dataset.parent / 'FSQ_dataset' / dataset.name / 'FSQ_inputs/test_segments/skillset'
    shutil.copytree(original, skillset)
    configs = []
    for mode, levels in [('joint_velocity', [3, 3, 3]), ('delta_pose', [3, 3, 3, 2])]:
        cfg = config_for(dataset, mode)
        cfg.update(outputs_root=str(tmp_path / 'outputs_DROID'), skillset_seg_name='test_segments',
            fsq_run_name='tiny_' + mode, fsq_quantizer={'type': 'fsq', 'levels': levels},
            fsq_hidden_dim=16, fsq_num_layers=1, fsq_num_epochs=1, fsq_batch_size=1,
            fsq_num_workers=0, fsq_checkpoint_every=1, fsq_wandb_project=None,
            fsq_boundary_aug_pmax_step=1)
        plan = TRAIN['resolve'](cfg, 'FSQ')
        TRAIN['_contract'](cfg, plan)
        TRAIN['run_fsq'](cfg, plan)
        configs.append((cfg, plan))
    cfg = dict(configs[0][0], models=[Path(p['output']).name for _, p in configs], checkpoint='last',
        output_name='tiny_eval', task_ids='all', max_tasks=0, episodes_per_task=1,
        boundary_jitter_sweep=True, resume=True, device='cpu', camera='exterior_2')
    return cfg, dataset, skillset


def test_two_actions_real_encoding_video_gallery_compare_and_resume(evaluated_models):
    cfg, dataset, skillset = evaluated_models
    protected = [*skillset.rglob('*.npz'), *dataset.glob('data/**/*.parquet')]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    plan = PLAN['build_settings'](cfg)
    assert [r['action_space'] for r in plan['models']] == ['joint_velocity', 'delta_pose']
    RUN.evaluate(plan)
    assert (Path(plan['output']) / 'compare/index.html').is_file()
    snapshots = {}
    for row in plan['models']:
        out = Path(row['output'])
        summary = json.loads((out / 'evaluation_complete.json').read_text())
        assert summary['verified'] and summary['episode_count'] == 2 and summary['skill_occurrences'] == 4
        assert summary['levels'] == row['levels'] and summary['camera'] == 'exterior_2'
        tokens = np.load(out / 'artifacts/skill_latents.npz')['tokens']
        assert len(tokens) == 4 and ((0 <= tokens) & (tokens < np.prod(row['levels']))).all()
        assert (out / 'artifacts/skill_boundary_jitter_sweep.npz').is_file()
        manifest = json.loads((out / 'metrics/manifest.json').read_text())
        for record in manifest['records'].values():
            video = out / record['video_path']
            assert video.is_file() and not video.is_symlink()
        snapshots[out] = (out / 'artifacts/skill_latents.npz').stat().st_mtime_ns
    RUN.evaluate(plan)
    assert all((out / 'artifacts/skill_latents.npz').stat().st_mtime_ns == stamp for out, stamp in snapshots.items())
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    changed = PLAN['build_settings'](dict(cfg, camera='wrist'))
    with pytest.raises(ValueError, match='different request'):
        RUN.evaluate(changed)
    with pytest.raises(FileNotFoundError, match='checkpoint missing'):
        PLAN['build_settings'](dict(cfg, checkpoint=999))
    (skillset / 'skillset_manifest.json').write_text('changed')
    with pytest.raises(ValueError, match='SHA256'):
        PLAN['build_settings'](cfg)


@pytest.mark.parametrize('change', [dict(output_name='../escape'), dict(camera='top'),
    dict(episodes_per_task=0), dict(resume='false'), dict(task_ids=[1, 1])])
def test_invalid_config_before_any_model_or_submission(change):
    cfg = TRAIN['load_config'](CONFIGS / 'train_skills/FSQ/fsq_config.yaml')
    cfg.update(models=['dummy'], **change)
    with pytest.raises(ValueError):
        PLAN['build_settings'](cfg)


def test_gallery_cap_keeps_rare_tasks_and_random_selection_is_reproducible():
    available = {0: [7], 1: [2, 4], 2: [11, 12, 13, 14, 15, 16]}
    plan = dict(task_ids='all', max_tasks=0, seed=42, episodes_per_task=5, episode_selection='first')
    assert RUN.gallery_selection(available, plan) == {0: [7], 1: [2, 4], 2: [11, 12, 13, 14, 15]}
    plan['episode_selection'] = 'random'
    assert RUN.gallery_selection(available, plan) == RUN.gallery_selection(available, plan)
    plan['task_ids'] = [99]
    with pytest.raises(ValueError, match='no selected'):
        RUN.gallery_selection(available, plan)
