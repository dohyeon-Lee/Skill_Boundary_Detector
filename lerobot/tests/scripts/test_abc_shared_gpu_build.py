from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
ABC_BUILD_SRC = ROOT / "lerobot/skillVLA/abc/configs/train_skills/build_data/src"
ABC_CONFIG_SRC = ROOT / "lerobot/skillVLA/abc/configs/train_skills/src/train_skills_config.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _load(ABC_BUILD_SRC / "run_single_gpu_workers.py", "abc_single_gpu_workers")
CONFIG = _load(ABC_CONFIG_SRC, "abc_train_skills_config")


@pytest.mark.parametrize('mode', ['srun_existing', 'srun', 'sbatch'])
def test_single_gpu_pipeline_uses_numeric_batch_ids_only_for_sbatch(tmp_path, mode):
    source = (ABC_BUILD_SRC.parent / 'submit_build_data.sh').read_text()
    start = source.index('if [ "${SKILLSET_SINGLE_GPU_WORKERS}" -gt 0 ]; then')
    branch = source[start:source.index('\nelse\n', start)] + '\nfi\n'
    harness = '''
set -euo pipefail
submit_job_scheduler() { echo slurm; }
submit_job_slurm_mode() { echo "$TEST_MODE"; }
submit_job() {
  [ "$ABC_SINGLE_GPU_BUILD" = true ]
  printf '%s\\n' "$@" > "$TEST_CAPTURE"
  if [ "$TEST_MODE" != sbatch ]; then
    for arg in "$@"; do [ "$arg" != --parsable ] || return 2; done
  fi
  echo submitted-test-session
}
SKILLSET_SINGLE_GPU_WORKERS=24
ABC_EXPECTED_EPISODES=2203
SCRIPT_DIR=/example/build_data
GPU_ARGS=(--cpus-per-task=24 --gres=gpu:1)
'''
    capture = tmp_path / 'argv'
    result = subprocess.run(['bash', '-c', harness + branch],
        env={**os.environ, 'TEST_MODE': mode, 'TEST_CAPTURE': str(capture)},
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    args = capture.read_text().splitlines()
    assert ('--parsable' in args) == (mode == 'sbatch')
    assert '--cpus-per-task=24' in args
    assert args[-1] == '/example/build_data/src/build_skillset.sbatch'


def test_abc_libero_standard_keeps_abc_terminal_masking() -> None:
    profile = CONFIG.SBD_PROFILES["libero_standard"]

    assert profile["skillset_score_power"] == 2.0
    assert profile["skillset_smooth_window"] == 7
    assert profile["skillset_savgol_polyorder"] == 4
    assert profile["skillset_terminal_mask_frames"] == 0
    assert profile["skillset_terminal_mask_mode"] == "abc_terminal_release"
    assert profile["skillset_terminal_release_pre_offset_frames"] == 8


def test_abc_standard_uses_the_same_odd_smoothing_window() -> None:
    profile = CONFIG.SBD_PROFILES["abc_standard"]
    assert profile["skillset_smooth_window"] == 7
    assert profile["skillset_savgol_polyorder"] == 4
    assert profile["skillset_score_power"] == 1.0


@pytest.mark.parametrize("workers", [0, 1, 6])
def test_abc_build_parallelism_accepts_array_or_shared_gpu(workers: int) -> None:
    settings = CONFIG.resolve_build_parallelism(
        {"skillset_tasks_per_job": 4, "skillset_single_gpu_workers": workers}
    )
    assert settings == {
        "skillset_tasks_per_job": 4,
        "skillset_single_gpu_workers": workers,
        "skillset_workers_per_gpu": 1,
        "skillset_max_gpu_jobs": 0,
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("skillset_tasks_per_job", 0),
        ("skillset_single_gpu_workers", -1),
        ("skillset_workers_per_gpu", 0),
        ("skillset_max_gpu_jobs", -1),
    ],
)
def test_abc_build_parallelism_rejects_invalid_values(key: str, value: int) -> None:
    config = {"skillset_tasks_per_job": 4, "skillset_single_gpu_workers": 0}
    config[key] = value
    with pytest.raises(ValueError, match=key):
        CONFIG.resolve_build_parallelism(config)


def test_shared_gpu_pool_bounds_workers_and_covers_every_shard(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    state = tmp_path / "state.json"

    # The fake shard process updates a file under an advisory lock, allowing
    # the real subprocess pool to prove its peak concurrency and coverage.
    worker = src / "fake_worker.py"
    worker.write_text(
        """
import fcntl, json, os, time
from pathlib import Path
p = Path(os.environ['TEST_STATE'])
with p.with_suffix('.lock').open('a') as stream:
    fcntl.flock(stream, fcntl.LOCK_EX)
    data = json.loads(p.read_text()) if p.exists() else {'active': 0, 'peak': 0, 'seen': []}
    data['active'] += 1
    data['peak'] = max(data['peak'], data['active'])
    data['seen'].append(int(os.environ['ABC_WORKER_SHARD_INDEX']))
    data.setdefault('threads', []).append(os.environ['OMP_NUM_THREADS'])
    p.write_text(json.dumps(data))
time.sleep(0.12)
with p.with_suffix('.lock').open('a') as stream:
    fcntl.flock(stream, fcntl.LOCK_EX)
    data = json.loads(p.read_text())
    data['active'] -= 1
    p.write_text(json.dumps(data))
""",
        encoding="utf-8",
    )
    (src / "build_skillset.sbatch").write_text(
        f"#!/usr/bin/env bash\nexec {sys.executable} {worker}\n", encoding="utf-8"
    )
    env = dict(
        os.environ,
        SKILLSET_SINGLE_GPU_WORKERS="2",
        SKILLSET_TASKS_PER_JOB="2",
        ABC_TASK_INDICES_JSON=json.dumps([59, 60, 61, 62, 63, 64, 65]),
        TEST_STATE=str(state),
        CUDA_VISIBLE_DEVICES="0",
        CURVES_ONLY="true",
        SLURM_CPUS_PER_TASK="8",
        SKILLSET_CPUS_PER_TASK="4",
    )
    builder = RUNNER.ABCSharedGPUBuilder(env, src=src, log_dir=tmp_path / "logs")
    builder._pool(phase="curves")

    result = json.loads(state.read_text())
    assert result["peak"] == 2
    assert result["active"] == 0
    assert sorted(result["seen"]) == [0, 1, 2, 3]


def test_array_gpu_pool_runs_two_logical_shards_per_allocation(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    seen = tmp_path / "seen"
    worker = src / "build_skillset.sbatch"
    worker.write_text(
        "#!/usr/bin/env bash\n"
        "mkdir -p \"$TEST_SEEN\"\n"
        "touch \"$TEST_SEEN/$ABC_WORKER_SHARD_INDEX\"\n",
        encoding="utf-8",
    )
    env = dict(
        os.environ,
        ABC_ARRAY_GPU_BUILD="true",
        SKILLSET_WORKERS_PER_GPU="2",
        SKILLSET_SINGLE_GPU_WORKERS="0",
        SKILLSET_TASKS_PER_JOB="2",
        ABC_TASK_INDICES_JSON=json.dumps(list(range(12))),
        SLURM_ARRAY_TASK_ID="1",
        TEST_SEEN=str(seen),
        SLURM_CPUS_PER_TASK="8",
    )
    builder = RUNNER.ABCSharedGPUBuilder(env, src=src, log_dir=tmp_path / "logs")
    builder.run_array_group()
    assert sorted(path.name for path in seen.iterdir()) == ["2", "3"]


def test_max_gpu_jobs_derives_put_relative_shard_size() -> None:
    settings = CONFIG.resolve_build_parallelism(
        {
            "skillset_tasks_per_job": 4,
            "skillset_workers_per_gpu": 2,
            "skillset_max_gpu_jobs": 6,
            "skillset_single_gpu_workers": 0,
        },
        task_count=1822,
    )
    assert settings["skillset_tasks_per_job"] == 152
    assert settings["skillset_workers_per_gpu"] == 2
    assert settings["skillset_max_gpu_jobs"] == 6


def test_episode_subset_filters_metadata_pca_and_records_provenance(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pandas as pd
    import lerobot.datasets.abc_native_io as native

    entry = _load(ABC_BUILD_SRC / 'build_skill_dataset.py', 'abc_subset_builder_test')
    payload = {'fingerprint': 'selected-dp', 'episode_count': 2, 'episodes': [1, 3]}
    monkeypatch.setattr(entry, 'load_episode_selection', lambda *a: payload)
    monkeypatch.setattr(native, 'iter_abc_native_state_action',
                        lambda *a, episode_ids, **kw: iter((e, None, None) for e in [1, 2, 3] if e in episode_ids))
    rows = pd.DataFrame({'episode_index': [1, 2, 3]})
    shared = SimpleNamespace(
        load_episodes_meta=lambda *a: rows,
        _iter_state_action_episodes=lambda *a: iter([]),
        _skillset_manifest=lambda **kw: {'source_tasks': ['task']},
        get_or_fit_action_pca=lambda path, meta, fit: fit(),
    )
    original = shared.load_episodes_meta
    def build(args):
        assert shared.load_episodes_meta(None).episode_index.tolist() == [1, 3]
        assert [e for e, *_ in shared._iter_state_action_episodes(None)] == [1, 3]
        meta = {}
        shared.get_or_fit_action_pca(None, meta, lambda: None)
        assert meta['episode_selection']['episode_count'] == 2
        assert shared._skillset_manifest()['episode_selection'] == entry.selection_contract(payload)
    shared.main = build
    args = SimpleNamespace(dataset_dir=str(tmp_path), output_dir=str(tmp_path / 'out'),
                           source_tasks='task', episode_selection_manifest='selected.json')
    entry.run(shared, args)
    assert shared.load_episodes_meta is original
    (tmp_path / 'out').mkdir()
    (tmp_path / 'out/skillset_manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='different ABC episode selection'):
        entry.run(shared, args)


def test_episode_balance_covers_skewed_tasks_without_duplicate_work():
    episodes = [(ep, 100 if ep < 10 else 1000) for ep in range(30)]
    shards = RUNNER.balance_episodes(episodes, 6)
    assert len(shards) == 6
    assert sorted(ep for shard in shards for ep in shard) == list(range(30))
    weights = dict(episodes)
    loads = [sum(weights[ep] for ep in shard) for shard in shards]
    assert max(loads) - min(loads) <= max(weights.values())
    assert RUNNER.balance_episodes([(7, 10)], 6) == [[7]]


def test_episode_pool_can_run_more_workers_than_prompt_tasks(tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    result = tmp_path / 'result'
    result.mkdir()
    (src / 'fake_worker.py').write_text('''
import json, os
from pathlib import Path
p = Path(os.environ['ABC_EPISODE_SHARD_PATH'])
data = json.loads(p.read_text())
Path(os.environ['RESULT'], p.name).write_text(json.dumps(data['episode_ids']))
''')
    (src / 'build_skillset.sbatch').write_text(f'exec {sys.executable} {src / "fake_worker.py"}\n')
    out = tmp_path / 'output'
    (out / 'curves').mkdir(parents=True)
    env = dict(os.environ, SKILLSET_SINGLE_GPU_WORKERS='6', SKILLSET_TASKS_PER_JOB='1',
               ABC_TASK_INDICES_JSON='[9]', SLURM_CPUS_PER_TASK='12',
               SKILLSET_CPUS_PER_TASK='12', SKILLSET_DIR=str(out), RESULT=str(result))
    builder = RUNNER.ABCSharedGPUBuilder(env, src=src, log_dir=tmp_path / 'logs')
    builder.episode_mode = True
    builder.episodes = [(ep, (ep + 1) * 10) for ep in range(30)]
    assert builder.shard_count == 6
    builder._pool(phase='curves')
    assert len(list(result.iterdir())) == 6
    ids = [ep for file in result.iterdir() for ep in json.loads(file.read_text())]
    assert sorted(ids) == list(range(30))


def test_worker_subset_keeps_full_dp_pca_and_manifest_contract(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pandas as pd
    import lerobot.datasets.abc_native_io as native
    entry = _load(ABC_BUILD_SRC / 'build_skill_dataset.py', 'abc_worker_subset_test')
    payload = {'fingerprint': 'dp-selection', 'episode_count': 3, 'episodes': [1, 2, 3]}
    monkeypatch.setattr(entry, 'load_episode_selection', lambda *a: payload)
    def iterator(*a, episode_ids, source_tasks):
        assert set(episode_ids) == {1, 2, 3}
        assert source_tasks == ['one', 'two']
        return iter((ep, None, None) for ep in [1, 2, 3])
    monkeypatch.setattr(native, 'iter_abc_native_state_action', iterator)
    rows = pd.DataFrame({'episode_index': [1, 2, 3, 4], 'source_task': ['one', 'two', 'two', 'other']})
    shared = SimpleNamespace(load_episodes_meta=lambda *a: rows,
                             _iter_state_action_episodes=lambda *a: iter([]),
                             _skillset_manifest=lambda **kw: {'source_tasks': ['one']},
                             get_or_fit_action_pca=lambda path, meta, fit: fit())
    def build(args):
        assert args.source_tasks == 'one'
        assert shared.load_episodes_meta(None).episode_index.tolist() == [1]
        assert [ep for ep, *_ in shared._iter_state_action_episodes(None, ('one',))] == [1, 2, 3]
        meta = {'source_tasks': ['one']}
        shared.get_or_fit_action_pca(None, meta, lambda: None)
        assert meta['source_tasks'] == ['one', 'two']
        assert shared._skillset_manifest()['source_tasks'] == ['one', 'two']
    shared.main = build
    ids = tmp_path / 'worker.json'
    ids.write_text('{"episode_ids": [1]}')
    args = SimpleNamespace(dataset_dir=str(tmp_path), output_dir=str(tmp_path / 'out'),
                           source_tasks='one,two', episode_selection_manifest='selected.json',
                           episode_ids_path=str(ids))
    entry.run(shared, args)
    assert args.source_tasks == 'one,two'
    ids.write_text('{"episode_ids": [4]}')
    with pytest.raises(ValueError, match='outside the DP selection'):
        entry.run(shared, args)
