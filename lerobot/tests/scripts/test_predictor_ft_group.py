"""FT groups inherit two different PT contracts without changing either source."""
import copy
import json
from pathlib import Path

import pytest

from test_newtask_ft_config import AUX, _aux_setup, _checkpoint


def group_config(tmp_path):
    config = _aux_setup(tmp_path, 'Predictor')
    original = Path(config['warm_start']['predictor_checkpoint'])
    source = json.loads((original / 'config.json').read_text())
    source.update(skill_predictor_mode='mode3', skill_predictor_focus_uv_enabled=True,
                  skill_predictor_end_state_mode='off', skill_predictor_lora=False,
                  skill_predictor_freeze_vlm=False)
    (original / 'config.json').write_text(json.dumps(source))
    _checkpoint(tmp_path, 'skillVLA_PT/Predictor', 'previous',
                {**source, 'skill_predictor_mode': 'mode3_preskill'})
    info = tmp_path / 'dataset/skillvla_dataset/new_task/FSQ333_ft/skillvla/meta/info.json'
    metadata = json.loads(info.read_text())
    metadata['skill_focus_uv_camera'] = 'agentview'
    info.write_text(json.dumps(metadata))
    config['warm_start']['predictor_checkpoint'] = [
        {'run': 'p', 'checkpoint': '050000'},
        {'run': 'previous', 'checkpoint': '050000'},
    ]
    config['predictor_ft'] = {'unfreeze': 'all', 'lora': {'enabled': False}}
    config['training'] = {'dataloader': {'gpus': 1}, 'schedule': {'steps': 10000}}
    config['slurm'] = {'gres': 'gpu:1'}
    return config


def test_ft_group_matches_each_scalar_and_worker_checkpoint(tmp_path):
    config = group_config(tmp_path)
    original = copy.deepcopy(config)
    parent = AUX.build_launch_settings(config)
    assert config == original
    assert parent['predictor_parallel_count'] == 2
    assert parent['num_gpus'] == 1
    manifest = json.loads(parent['predictor_parallel_runs'])
    assert [r['mode'] for r in manifest] == ['mode3', 'mode3_preskill']
    for source, entry in zip(config['warm_start']['predictor_checkpoint'], manifest, strict=True):
        single = copy.deepcopy(config)
        single['warm_start']['predictor_checkpoint'] = source
        expected = AUX.build_settings(single)
        worker = AUX.build_launch_settings(config, predictor_mode_override=entry['mode'])
        assert worker['predictor_parallel_count'] == 1
        assert worker['output_dir'] == expected['output_dir']
        assert str(worker['predictor_checkpoint_path']) == entry['predictor_checkpoint_path']
        assert not worker['skill_predictor_freeze_vlm']
        assert worker['batch_size'] == 16 and worker['steps'] == 10000
        assert worker['skill_predictor_focus_uv_enabled']


def test_ft_group_singleton_and_unknown_worker(tmp_path):
    config = group_config(tmp_path)
    config['warm_start']['predictor_checkpoint'].pop()
    assert AUX.build_launch_settings(config)['predictor_parallel_count'] == 1
    with pytest.raises(ValueError, match='not selected'):
        AUX.build_launch_settings(config, predictor_mode_override='mode3_preskill')


def test_ft_group_rejects_duplicate_source_modes(tmp_path):
    config = group_config(tmp_path)
    config['warm_start']['predictor_checkpoint'][1] = config['warm_start']['predictor_checkpoint'][0]
    with pytest.raises(ValueError, match='distinct'):
        AUX.build_launch_settings(config)


@pytest.mark.parametrize('value', [[], [None], [''], [{}]])
def test_ft_group_rejects_empty_sources(tmp_path, value):
    config = group_config(tmp_path)
    config['warm_start']['predictor_checkpoint'] = value
    with pytest.raises(ValueError, match='non-empty'):
        AUX.build_launch_settings(config)


@pytest.mark.parametrize('change', ['pt', 'wrong_component', 'multiple_gpus', 'multi_gres', 'mode_list'])
def test_ft_group_rejects_invalid_launch(tmp_path, change):
    config = group_config(tmp_path)
    if change == 'pt':
        config['mode'] = 'pt'
    elif change == 'wrong_component':
        config['stage1_component'] = 'Joint'
    elif change == 'multiple_gpus':
        config['training']['dataloader']['gpus'] = 2
    elif change == 'multi_gres':
        config['slurm']['gres'] = 'gpu:2'
    else:
        config['skill_predictor'] = {'mode': ['mode3', 'mode3_preskill']}
    with pytest.raises(ValueError):
        AUX.build_launch_settings(config)


def test_terminator_ignores_predictor_group_from_shared_config(tmp_path):
    config = group_config(tmp_path)
    config['stage1_component'] = 'Terminator'
    settings = AUX.build_launch_settings(config)
    assert settings['train_terminator'] and not settings['train_skill_predictor']
    assert settings['predictor_parallel_count'] == 1


def test_shipped_predictor_inherits_shared_checkpoint_list():
    import yaml
    configs = Path(AUX.__file__).parents[3] / 'FT'
    predictor = configs / 'Predictor/predictor_ft_config.yaml'
    common = yaml.safe_load((configs / 'newtask_ft_common_config.yaml').read_text())
    component = yaml.safe_load(predictor.read_text())
    assert 'warm_start' not in component
    merged = AUX.load_stage1_component_config(predictor)
    assert merged['warm_start']['predictor_checkpoint'] == common['warm_start']['predictor_checkpoint']
    assert len(merged['warm_start']['predictor_checkpoint']) == 2
