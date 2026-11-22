"""Episode workers must never calibrate on their own partial selections."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'lerobot/skillVLA/abc/configs/train_skills/DP/eval/src'
spec = importlib.util.spec_from_file_location('abc_parallel_ablation_test', SRC / 'run_parallel_ablation.py')
MOD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MOD)


@pytest.mark.parametrize(('workers', 'episode_worker', 'expected'), [
    (1, False, 8), (2, True, 4), (2, False, 8),
])
def test_cpu_budget_limits_already_loaded_native_pools(workers, episode_worker, expected):
    code = f'''
import importlib.util, json, os
import numpy as np
np.ones((8, 8)) @ np.ones((8, 8))
spec = importlib.util.spec_from_file_location('parallel', {str(SRC / 'run_parallel_ablation.py')!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.configure_cpu_threads({{'workers': {workers}, 'cpu_budget': 8}}, episode_worker={episode_worker!r})
import torch
from threadpoolctl import threadpool_info
assert torch.get_num_threads() == {expected}
assert torch.get_num_interop_threads() == 1
assert all(os.environ[name] == str({expected}) for name in module.THREAD_ENV_NAMES)
pools = threadpool_info()
assert pools and all(pool['num_threads'] <= {expected} for pool in pools), pools
'''
    env = dict(os.environ, OPENBLAS_NUM_THREADS='32', OMP_NUM_THREADS='32', MKL_NUM_THREADS='32')
    subprocess.run([sys.executable, '-c', code], env=env, check=True, capture_output=True, text=True)


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
    module._cluster_cache_load = lambda ep, *args: cache.get(ep)
    def merge(cfg):
        assert cfg is settings
        assert module._selected_episode_ids() == [1, 2, 3, 4]
        assert cfg['threshold_calibration']['expected_episodes'] == 4
        assert [module._cluster_cache_load(ep) for ep in selected()] == [10, 20, 30, 40]
        assert module._cluster_cache_load('final10', {'descriptor_key': 'final_descriptors'}) is None
        with pytest.raises(RuntimeError, match='Cluster cache miss'):
            module._cluster_cache_load('missing_primary', {'schema': 2})
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
