"""Episode workers must never calibrate on their own partial selections."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'lerobot/skillVLA/abc/configs/train_skills/skill_eval/skill_boundary_detection/src'
spec = importlib.util.spec_from_file_location('abc_parallel_ablation_test', SRC / 'run_parallel_ablation.py')
MOD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MOD)


def test_cache_worker_stops_before_any_partial_threshold_or_report(tmp_path, monkeypatch):
    calls = []
    skill = SimpleNamespace(load_policy=lambda *a, **kw: 'policy')
    monkeypatch.setitem(sys.modules, 'skill_divider', skill)
    original_selected = lambda *a, **kw: [1, 2, 3, 4]
    original_calibration = lambda *a, **kw: [1, 2, 3, 4]
    derive = lambda: calls.append('global threshold')
    module = SimpleNamespace(_selected_episode_ids=original_selected,
                             _threshold_calibration_episode_ids=original_calibration, _derive=derive)
    settings = {'threshold_calibration': {'source': 'selected', 'expected_episodes': 4}}
    def run(cfg):
        assert cfg['threshold_calibration']['expected_episodes'] == 4
        assert module._selected_episode_ids() == [1, 3]
        assert module._threshold_calibration_episode_ids() == [1, 3]
        assert skill.load_policy() == 'policy'
        calls.append('raw and cluster caches')
        module._derive()
        calls.append('report')
    module.run = run
    MOD.cache_worker(module, settings, [1, 3], tmp_path, 0)
    assert calls == ['raw and cluster caches']
    assert module._selected_episode_ids is original_selected
    assert module._threshold_calibration_episode_ids is original_calibration
    assert module._derive is derive
    assert json.loads((tmp_path / 'worker_000_complete.json').read_text())['episodes'] == [1, 3]


def test_real_worker_error_is_not_reported_as_cache_success(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'skill_divider', SimpleNamespace(load_policy=lambda: None))
    def fail(cfg):
        raise ValueError('invalid raw descriptor')
    module = SimpleNamespace(_selected_episode_ids=lambda: [],
                             _threshold_calibration_episode_ids=lambda: [], _derive=lambda: None, run=fail)
    with pytest.raises(ValueError, match='invalid raw descriptor'):
        MOD.cache_worker(module, {}, [1], tmp_path, 0)
    assert not (tmp_path / 'worker_000_complete.json').exists()


def test_global_merge_uses_original_selection_and_only_finished_caches(tmp_path, monkeypatch):
    import lerobot.datasets.abc_native_io as native
    rows = pd.DataFrame({'episode_index': [1, 2, 3, 4], 'length': [5, 10, 50, 100]})
    monkeypatch.setattr(native, 'load_abc_native_episodes', lambda *a: rows)
    settings = {'workers': 2, 'cpu_budget': 2, 'skillset_dir': str(tmp_path),
                'task_ids': [0], 'task_id_space': 'dataset', 'target_task': 'grab', 'n_episodes': 4,
                'threshold_calibration': {'source': 'selected', 'expected_episodes': 4}}
    selected = lambda *a, **kw: [1, 2, 3, 4]
    module = SimpleNamespace(_selected_episode_ids=selected,
                             _threshold_calibration_episode_ids=lambda *a: [1, 2, 3, 4],
                             _load_manifest=lambda *a: {'dataset_dir': str(tmp_path)},
                             _portable_path=lambda *a: tmp_path,
                             _probe_episode=lambda: 'unexpected inference')
    cache = {}
    module._cluster_cache_load = lambda ep: cache.get(ep)
    def merge(cfg):
        assert cfg is settings
        assert module._selected_episode_ids() == [1, 2, 3, 4]
        assert cfg['threshold_calibration']['expected_episodes'] == 4
        assert [module._cluster_cache_load(ep) for ep in selected()] == [10, 20, 30, 40]
        with pytest.raises(RuntimeError, match='Raw cache miss'):
            module._probe_episode()
        return tmp_path / 'index.html'
    module.run = merge
    runner = MOD.ParallelAblation(settings, tmp_path / 'workers')
    def worker(index, path):
        for ep in json.loads(path.read_text()):
            cache[ep] = ep * 10
        return 0
    monkeypatch.setattr(runner, '_worker', worker)
    assert runner.run(module) == tmp_path / 'index.html'
    assert set(cache) == {1, 2, 3, 4}
    assert module._selected_episode_ids is selected


def test_joint_action_mask_reaches_probe_collector_and_preserves_grippers(tmp_path, monkeypatch):
    import numpy as np
    # Importing the shared implementation here also supplies its real action
    # transform; the fake collector only replaces the expensive DP forward pass.
    evaluator = MOD.load_evaluator()
    from action_manifold import to_model_action_chunk
    info = tmp_path / 'meta/info.json'
    info.parent.mkdir()
    names = [f'left_joint_{i}' for i in range(6)] + ['left_gripper'] + [f'right_joint_{i}' for i in range(6)] + ['right_gripper']
    info.write_text(json.dumps({'features': {'action': {'names': names}}}))
    manifest = {'dataset_dir': str(tmp_path), 'action': {'dim': 14, 'mode': 'delta_joint',
                'relative_exclude_joints': ['gripper'], 'gripper_indices': [6, 13]}}
    facade = SimpleNamespace(_load_manifest=lambda *a: manifest, _portable_path=lambda *a: tmp_path)
    seen = {}
    def collect(*args, **kw):
        seen.update(kw)
        return to_model_action_chunk(kw['actions'], kw['state'], kw['action_mode'], kw['relative_mask'])
    skill = SimpleNamespace(run_vf_analysis=collect)
    monkeypatch.setitem(sys.modules, 'skill_divider', skill)
    state = np.zeros(14, dtype=np.float32)
    actions = np.stack([state + 1, state + 3, state + 4])
    actions[:, 6] = [.1, .2, .3]
    actions[:, 13] = [.4, .5, .6]
    with MOD.action_contract(facade, {'skillset_dir': str(tmp_path)}):
        result = skill.run_vf_analysis(action_mode='delta_joint', actions=actions, state=state)
    np.testing.assert_allclose(result[:, 0], [1, 2, 1])
    np.testing.assert_array_equal(result[:, [6, 13]], actions[:, [6, 13]])
    assert not seen['relative_mask'][6] and not seen['relative_mask'][13]
    assert skill.run_vf_analysis is collect


def test_dataset_action_contract_keeps_legacy_path_unchanged(tmp_path, monkeypatch):
    manifest = {'action': {'mode': 'dataset'}}
    facade = SimpleNamespace(_load_manifest=lambda *a: manifest)
    seen = {}
    def collect(**kw):
        seen.update(kw)
        return 42
    skill = SimpleNamespace(run_vf_analysis=collect)
    monkeypatch.setitem(sys.modules, 'skill_divider', skill)
    with MOD.action_contract(facade, {'skillset_dir': str(tmp_path)}):
        assert skill.run_vf_analysis(action_mode='dataset') == 42
    assert 'relative_mask' not in seen


def test_task_shard_preserves_parallel_worker_budget(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('abc_shard_worker_config', SRC / 'config.py')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    output = tmp_path / 'full_report'
    payload = {'n_episodes': 5, 'task_ids': [1, 2, 3],
               'threshold_calibration': {'source': 'selected', 'expected_episodes': 15}}
    shared = SimpleNamespace(
        build_settings=lambda *a, **kw: {'dp_ablation_output_dir': str(output),
                                        'dp_ablation_settings_json': json.dumps(payload),
                                        'dp_ablation_cpus': 8},
        load_config=lambda *a: {'eval_workers': 4},
        get_value=lambda cfg, key, default: cfg.get(key, default))
    monkeypatch.setattr(config, '_shared_module', lambda: shared)
    monkeypatch.setenv('ABC_SBD_SHARD_TAG', 'tasks1_3')
    monkeypatch.setenv('ABC_SBD_TASK_IDS', '1,3')
    settings = config.build_settings('config.yaml')
    result = json.loads(settings['dp_ablation_settings_json'])
    assert settings['dp_ablation_workers'] == result['workers'] == 4
    assert result['cpu_budget'] == 8
    assert result['task_ids'] == [1, 3]
    assert result['threshold_calibration']['expected_episodes'] == 10
    assert settings['dp_ablation_output_dir'] == str(output / 'shards/tasks1_3')
    assert result['cache_source_dirs'][0] == str(output)
