"""One UUID contract across training, PCA, episode workers and frozen launches."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess

import numpy as np
import pytest
import yaml

from test_droid_dp_eval import setup, payload

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / 'lerobot/skillVLA/droid/configs'
GROUPS = runpy.run_path(str(CONFIGS / 'src/dataset_groups.py'))
TRAIN = runpy.run_path(str(CONFIGS / 'train_skills/src/train_skills_config.py'))
SHARED = runpy.run_path(str(ROOT / 'lerobot/skillVLA/libero/configs/train_skills/src/train_skills_config.py'))


@pytest.fixture
def grouped(setup, tmp_path):
    path, cfg = setup
    dataset = tmp_path / 'dataset_DROID/tiny'
    uuids = ['cup-A', 'towel-A', 'cup-B', 'paper-A', 'unknown-A']
    (dataset / 'conversion_manifest.json').write_text(json.dumps({'source_episode_ids': uuids}))
    manifest = tmp_path / 'without_deformable.json'
    manifest.write_text(json.dumps(dict(group='without_deformable', source_episode_ids=['cup-A', 'cup-B'])))
    cfg.update(workflow_domain='droid', droid_dataset_group='without_deformable',
               droid_dataset_groups={'without_deformable': dict(manifest=str(manifest),
                   sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())})
    path.write_text(yaml.safe_dump(cfg))
    return path, cfg, dataset, manifest


def test_real_group_has_broad_screen_and_excludes_unknowns():
    cfg = yaml.safe_load((CONFIGS / 'global_config.yaml').read_text())
    cfg['droid_dataset']['group'] = 'without_deformable'
    definition = GROUPS['definition'](cfg)
    data = json.loads(definition['path'].read_text())
    assert len(definition['source_episode_ids']) == 55865
    assert data['unknown_policy'] == 'exclude'
    assert set(data['exclude_patterns']) == {'textile_clothing', 'paper_foil', 'bags_pouches', 'cords_elastics', 'soft_objects'}


def test_episode_ids_are_stable_and_group_is_separate_from_idle(grouped):
    _, cfg, dataset, manifest = grouped
    before = hashlib.sha256((dataset / 'conversion_manifest.json').read_bytes()).hexdigest()
    selection = GROUPS['select'](cfg, dataset)
    assert selection['episode_ids'] == [0, 2]
    assert selection['source_episode_ids'] == ['cup-A', 'cup-B']
    assert selection['selected_episodes'] == 2 and selection['total_episodes'] == 5
    for stage in ['DP', 'FSQ', 'PT_VSA', 'PT_PRED', 'PT_TERM', 'FT_VSA', 'FT_PRED', 'FT_TERM']:
        for idle in [False, True]:
            configured = dict(cfg, idle_filter={stage: idle})
            assert GROUPS['selected_episodes'](configured, dataset) == [0, 2]
            assert SHARED['resolve_training_episodes'](configured, dataset) == [0, 2]
    limited = dict(cfg, dataset={'n_episodes': 1})
    assert SHARED['resolve_training_episodes'](limited, dataset) == [0]
    assert before == hashlib.sha256((dataset / 'conversion_manifest.json').read_bytes()).hexdigest()
    assert GROUPS['select']({}, dataset / 'nonexistent') is None
    assert GROUPS['selected_episodes']({}, dataset, [2, 0]) == [2, 0]
    manifest.write_text(manifest.read_text() + ' ')
    with pytest.raises(ValueError, match='SHA256'):
        GROUPS['select'](cfg, dataset)


def test_dp_loader_and_output_receive_exact_group(grouped, tmp_path):
    _, cfg, dataset, _ = grouped
    local = yaml.safe_load((CONFIGS / 'train_skills/DP/dp_config.yaml').read_text())
    cfg = dict(cfg, **local)
    cfg.update(source_dataset='tiny', target_dataset='tiny', outputs_root='outputs_DROID',
        droid_fps=15, droid_action_space='joint_velocity', droid_proprio_space='joint_position',
        vla_chunk_size=15, vla_replan_steps=8)
    plan = TRAIN['resolve'](cfg, 'DP')
    desired = TRAIN['build_dp_config'](cfg, plan)
    assert desired.dataset.episodes == [0, 2]
    assert plan['output'].endswith('_without_deformable')
    path = tmp_path / 'selection.json'
    GROUPS['write_selection'](path, plan['dataset_selection'])
    GROUPS['write_selection'](path, plan['dataset_selection'])
    with pytest.raises(ValueError, match='selection changed'):
        GROUPS['write_selection'](path, dict(plan['dataset_selection'], episode_ids=[0]))


def test_fsq_filters_whole_corpus_before_train_validation_split(grouped, tmp_path):
    _, cfg, dataset, _ = grouped
    skills = tmp_path / 'skills'; skills.mkdir()
    for ep in range(5):
        np.savez(skills / f'ep{ep}.npz', actions=np.zeros((4, 8)), states=np.ones((4, 8)),
                 episode_id=ep, skill_index=0, frame_start=0, frame_end=4)
    module = TRAIN['fsq_module']()
    segments, states, actions, metadata = module.load_skill_files(skills,
        episode_ids=GROUPS['selected_episodes'](cfg, dataset))
    assert [m['episode_id'] for m in metadata] == [0, 2]
    assert len(segments) == len(states) == len(actions) == 2
    assert all(len(s) == 4 for s in segments)
    assert len(list(skills.glob('*.npz'))) == 5


def test_fsq_requires_build_completion_before_training(grouped):
    _, cfg, dataset, _ = grouped
    local = yaml.safe_load((CONFIGS / 'train_skills/FSQ/fsq_config.yaml').read_text())
    cfg = dict(cfg, **local)
    cfg.update(source_dataset='tiny', target_dataset='tiny', outputs_root='outputs_DROID',
        droid_fps=15, droid_action_space='joint_velocity', droid_proprio_space='joint_position',
        vla_chunk_size=15, vla_replan_steps=8)
    plan = TRAIN['resolve'](cfg, 'FSQ')
    skillset = Path(plan['skillset'])
    (skillset / 'skills').mkdir(parents=True)
    (skillset / 'skillset_manifest.json').write_text('{}')
    (dataset / 'training_view_manifest.json').write_text('{}')
    assert TRAIN['resolve'](cfg, 'FSQ')['missing_prerequisites'] == [str(skillset / '.complete')]
    (skillset / '.complete').write_text('complete')
    assert TRAIN['resolve'](cfg, 'FSQ')['missing_prerequisites'] == []


def test_eval_filters_selection_pca_and_threshold_as_one_group(grouped):
    path, cfg, _, _ = grouped
    cfg.update(task_ids='all', n_tasks='all', n_episodes=5, pca_scope='dataset')
    path.write_text(yaml.safe_dump(cfg))
    result = payload(path)
    manifest = result['source_manifest']
    assert [row['episode_index'] for row in manifest['evaluation_episodes']] == [0, 2]
    assert manifest['pca_episode_ids'] == [0, 2]
    assert result['threshold_calibration']['expected_episodes'] == 2
    assert '_without_deformable' in result['artifact_dir']
    assert manifest['dataset_selection']['selected_episodes'] == 2
    cfg['task_ids'] = [2]; path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='absent'):
        payload(path)


def test_build_resolves_separate_group_with_global_pca_contract(grouped):
    _, cfg, dataset, _ = grouped
    local = yaml.safe_load((CONFIGS / 'train_skills/build_data/build_data_config.yaml').read_text())
    cfg = dict(cfg, **local)
    cfg.update(target_dataset='tiny', outputs_root='outputs_DROID',
               dp_run_name='tiny_future_only_obs30_future18_transformer_constLR_small')
    settings = SHARED['build_data_settings'](cfg)
    selection = GROUPS['select'](cfg, dataset)
    assert selection['episode_ids'] == [0, 2]
    assert settings['droid_selected_episodes'] == 2
    assert settings['droid_selection_sha256'] == selection['selection_sha256']
    assert settings['skillset_dir'].parent.name.endswith('_without_deformable')
    assert settings['raw_dataset_dir'] == dataset
    # An explicit checkpoint remains the user's checkpoint; only outputs change.
    assert '_without_deformable' not in settings['dp_policy']


def test_compact_build_names_follow_settings_and_respect_frozen_recipes(grouped):
    _, cfg, _, _ = grouped
    local = yaml.safe_load((CONFIGS / 'train_skills/build_data/build_data_config.yaml').read_text())
    cfg = dict(cfg, **local)
    cfg.update(target_dataset='tiny', outputs_root='outputs_DROID',
               dp_run_name='tiny_future_only_obs30_future18_transformer_constLR_small')
    compact = SHARED['build_data_settings'](cfg)
    changed = SHARED['build_data_settings'](dict(cfg, skillset_smooth_window=9, skillset_savgol_polyorder=6))
    assert 'sg13p10' in compact['skillset_dir'].parent.name
    assert 'sg9p6' in changed['skillset_dir'].parent.name
    assert compact['dp_policy_path'] == changed['dp_policy_path']
    assert compact['droid_selection_sha256'] == changed['droid_selection_sha256']
    # Explicit names from frozen runs must remain resumable after this change.
    frozen = SHARED['build_data_settings'](dict(cfg, skillset_mode='gaussian_x0hat',
        skillset_output_suffix='existing_recipe', skillset_gripper_indices=[6]))
    assert frozen['skillset_dir'].parent.name.endswith('existing_recipe_without_deformable')
    assert frozen['skillset_gripper_indices'] == '6'
    old_without_suffix = SHARED['build_data_settings'](dict(cfg, skillset_mode='gaussian_x0hat'))
    assert '_h18_' not in old_without_suffix['skillset_dir'].parent.name


def test_compact_build_name_uses_checkpoint_horizon_not_stale_run_label(grouped):
    _, cfg, _, _ = grouped
    local = yaml.safe_load((CONFIGS / 'train_skills/build_data/build_data_config.yaml').read_text())
    cfg = dict(cfg, **local)
    cfg.update(target_dataset='tiny', outputs_root='outputs_DROID',
               dp_run_name='tiny_future_only_obs30_future18_transformer_constLR_small')
    original = SHARED['build_data_settings'](cfg)
    policy = original['dp_policy_path'] / 'config.json'
    policy.parent.mkdir(parents=True, exist_ok=True)
    policy.write_text(json.dumps(dict(n_obs_steps=30, horizon=36, action_sequence_mode='future_only')))
    resolved = SHARED['build_data_settings'](cfg)
    assert '_h36_' in resolved['skillset_dir'].parent.name
    assert resolved['dp_policy'] == original['dp_policy']


def test_build_worker_can_resolve_exported_settings_without_path_changes(grouped, monkeypatch, capsys):
    _, cfg, _, _ = grouped
    local = yaml.safe_load((CONFIGS / 'train_skills/build_data/build_data_config.yaml').read_text())
    cfg = dict(cfg, **local)
    cfg.update(target_dataset='tiny', outputs_root='outputs_DROID',
               dp_run_name='tiny_future_only_obs30_future18_transformer_constLR_small')
    settings = SHARED['build_data_settings'](cfg)
    SHARED['print_shell'](settings)
    exports = capsys.readouterr().out
    for line in exports.splitlines():
        assignment = shlex.split(line)[1]
        key, value = assignment.split('=', 1)
        monkeypatch.setenv(key, value)
    assert SHARED['build_data_settings'](cfg) == settings


def test_snapshot_carries_immutable_group_even_if_live_manifest_changes(grouped, tmp_path):
    _, cfg, _, live = grouped
    folder = tmp_path / 'launch'; folder.mkdir()
    global_path = folder / 'global_config.yaml'; global_path.write_text(yaml.safe_dump(cfg))
    local = folder / 'dp.yaml'; local.write_text('target_dataset: tiny\n')
    shell = ROOT / 'lerobot/skillVLA/libero/configs/src/snapshot_config.sh'
    process = subprocess.run(['bash', '-c', 'source "$1"; snapshot_config "$2"', 'test', str(shell), str(local)],
        env=dict(os.environ, SBD_GLOBAL_CONFIG=str(global_path)), capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    snapshot = Path(process.stdout.strip())
    saved = yaml.safe_load((snapshot.parent / 'global_config.yaml').read_text())
    frozen = GROUPS['definition'](saved)
    assert frozen['path'].parent == snapshot.parent
    live.write_text('{}')
    assert GROUPS['definition'](saved)['source_episode_ids'] == ['cup-A', 'cup-B']


def test_selected_dataset_propagates_from_conversion_to_training_and_builds(tmp_path):
    from test_droid_dataset import prepare
    global_cfg = yaml.safe_load((CONFIGS / 'global_config.yaml').read_text())
    global_cfg['droid_dataset']['name'] = 'new_download'
    global_cfg['droid_dataset']['sources']['new_download'] = dict(raw_root='/data5/new_droid',
        calibration_root='/data5/new_calibration', annotations_path='/data5/new_annotations.json',
        converted_dataset='new_converted', training_dataset='new_joint_velocity')
    global_cfg.update(project_root=str(ROOT))
    folder = tmp_path / 'configs'; folder.mkdir()
    (folder / 'global_config.yaml').write_text(yaml.safe_dump(global_cfg))
    paths = {}
    for stage, filename in [('DP', 'dp_config.yaml'), ('FSQ', 'fsq_config.yaml')]:
        path = folder / filename
        path.write_text((CONFIGS / f'train_skills/{stage}/{filename}').read_text())
        cfg = TRAIN['load_config'](path)
        assert cfg['source_dataset'] == 'new_converted' and cfg['target_dataset'] == 'new_joint_velocity'
        paths[stage] = cfg
    assert paths['FSQ']['skillset_seg_name'].startswith('seg_new_converted_delta_joint_')
    path = folder / 'build.yaml'
    path.write_text((CONFIGS / 'train_skills/build_data/build_data_config.yaml').read_text())
    cfg = SHARED['load_config'](path)
    assert cfg['dp_run_name'].startswith('new_converted_delta_joint_')
    # This path-resolution test has no converted episodes for the new family.
    # The grouped fixture above verifies selection against real tiny metadata.
    cfg['droid_dataset']['group'] = 'all'
    assert SHARED['build_data_settings'](cfg)['raw_dataset_dir'].name == 'new_joint_velocity'
    conversion = yaml.safe_load((CONFIGS / 'generate_training_dataset/download_dataset/dataset_config.yaml').read_text())
    # External conversion YAMLs can select the same named family. Explicit
    # paths in old resolved snapshots continue to take priority.
    conversion.update(droid_dataset=global_cfg['droid_dataset'])
    path = folder / 'conversion.yaml'; path.write_text(yaml.safe_dump(conversion))
    resolved = prepare.load_settings(path)
    assert resolved['raw_root'] == '/data5/new_droid'
    assert resolved['calibration_root'] == '/data5/new_calibration'
    assert Path(resolved['output_path']).name == 'new_converted'
    conversion.update(raw_root='/data4/old_frozen', output_name='old_output')
    path.write_text(yaml.safe_dump(conversion))
    assert prepare.load_settings(path)['raw_root'] == '/data4/old_frozen'
    assert Path(prepare.load_settings(path)['output_path']).name == 'old_output'


def test_vsa_and_auxiliary_training_hooks_have_same_group_and_distinct_outputs(grouped):
    _, cfg, dataset, _ = grouped
    outputs = []
    for run_key, output_key in [('pt_run_name', 'pt_output_dir'), ('run_name', 'output_dir')]:
        settings = {run_key: 'model', output_key: Path('/outputs/model')}
        result = SHARED['apply_droid_training_group'](cfg, settings, dataset,
            run_key=run_key, output_key=output_key)
        assert json.loads(result['dataset_episodes']) == [0, 2]
        assert '_without_deformable_sel' in result[run_key]
        assert result[output_key].name == result[run_key]
        outputs.append(result[output_key])
    assert outputs[0] == outputs[1]
    unknown = dict(cfg, droid_dataset_group='typo')
    with pytest.raises(ValueError, match='Unknown DROID dataset group'):
        GROUPS['select'](unknown, dataset)


def test_build_workers_partition_only_group_and_reduce_once(tmp_path, monkeypatch):
    from test_libero_sbd_single_gpu import builder
    selection = tmp_path / 'selection.json'
    selection.write_text(json.dumps(dict(group='without_deformable', episode_ids=[1, 3])))
    runner = builder(tmp_path, DROID_SELECTION_PATH=str(selection),
                     PROJECT_ROOT=str(ROOT), LIBERO_EXPECTED_EPISODES='2')
    captures, stages = [], []
    def command(argv, label, *, env=None, **kwargs):
        ids = json.loads(Path(env['LIBERO_EPISODE_SHARD_PATH']).read_text())['episode_ids']
        full = json.loads(Path(env['DROID_SELECTION_PATH']).read_text())['episode_ids']
        assert full == [1, 3]  # The PCA corpus remains global across both shards.
        captures.append((env['CURVES_ONLY'], ids))
        return 0, ''
    monkeypatch.setattr(runner, '_command', command)
    monkeypatch.setattr(runner, '_stage', stages.append)
    runner.run()
    for phase in ['true', 'false']:
        combined = [ep for current, ids in captures if current == phase for ep in ids]
        assert sorted(combined) == [1, 3] and len(set(combined)) == 2
    assert stages == ['compute_global_boundary_threshold.sbatch', 'mark_skillset_complete.sbatch']
    assert json.loads((Path(runner.env['SKILLSET_DIR']) / 'dataset_selection.json').read_text())['episode_ids'] == [1, 3]
