"""One-allocation SBD routing, balanced work and stage barriers, without a GPU."""
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[2] / "skillVLA/libero/configs/train_skills/build_data/src"
spec = importlib.util.spec_from_file_location('libero_sbd_workers_test', SRC / 'run_single_gpu_workers.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def builder(tmp_path, **overrides):
    src = tmp_path / 'build/src'; src.mkdir(parents=True)
    raw = tmp_path / 'raw'
    meta = raw / 'meta/episodes/chunk-000'; meta.mkdir(parents=True)
    pd.DataFrame({'episode_index': [1, 2, 3, 4], 'length': [100, 90, 10, 20]}).to_parquet(meta / 'file-000.parquet')
    env = dict(os.environ, RAW_DATASET_DIR=str(raw), LIBERO_EXPECTED_EPISODES='4',
               SKILLSET_DIR=str(tmp_path / 'skillset'), SKILLSET_SINGLE_GPU_WORKERS='2',
               SKILLSET_TASKS_PER_JOB='5', SKILLSET_CPUS_PER_TASK='4', SLURM_CPUS_PER_TASK='8',
               SKILLSET_BOUNDARY_THRESHOLD_MODE='global_mean', SKILLSET_GLOBAL_THRESHOLD_SOURCE='',
               CUDA_VISIBLE_DEVICES='3')
    env.update(overrides)
    return runner.LIBEROSharedGPUBuilder(env, src=src, log_dir=tmp_path / 'logs')


def test_balanced_pool_inherits_one_gpu_and_covers_episodes_once(tmp_path):
    b = builder(tmp_path)
    capture = tmp_path / 'seen.jsonl'
    worker = tmp_path / 'worker.py'
    worker.write_text('''
import fcntl,json,os,time
from pathlib import Path
assert os.environ['CUDA_VISIBLE_DEVICES']=='3'
assert os.environ['CURVES_ONLY']=='true' and os.environ['USE_CACHED_CURVES']=='false'
assert os.environ['OMP_NUM_THREADS']==os.environ['MKL_NUM_THREADS']==os.environ['OPENBLAS_NUM_THREADS']=='2'
ids=json.loads(Path(os.environ['LIBERO_EPISODE_SHARD_PATH']).read_text())['episode_ids']
with Path(os.environ['TEST_CAPTURE']).open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps(ids)+'\\n')
time.sleep(.05)
''')
    (b.src / 'build_skillset.sbatch').write_text(f'exec {shlex.quote(sys.executable)} {shlex.quote(str(worker))}\n')
    b.env['TEST_CAPTURE'] = str(capture)
    b._pool(phase='curves')
    shards = [json.loads(line) for line in capture.read_text().splitlines()]
    assert sorted(ep for shard in shards for ep in shard) == [1, 2, 3, 4]
    assert sorted(shards) == [[1, 3], [2, 4]]


@pytest.mark.parametrize('mode,reference,curves_only,expected', [
    ('global_mean', '', False, ['curves', 'compute_global_boundary_threshold.sbatch', 'cached_segment', 'mark_skillset_complete.sbatch']),
    ('global_mean', 'reference.json', False, ['curves', 'cached_segment', 'mark_skillset_complete.sbatch']),
    ('episode_mean', '', False, ['segment', 'mark_skillset_complete.sbatch']),
    ('global_mean', '', True, ['curves']),
])
def test_pipeline_barriers(tmp_path, monkeypatch, mode, reference, curves_only, expected):
    b = builder(tmp_path, SKILLSET_BOUNDARY_THRESHOLD_MODE=mode,
                SKILLSET_GLOBAL_THRESHOLD_SOURCE=reference, CURVES_ONLY=str(curves_only).lower())
    seen = []
    monkeypatch.setattr(b, '_pool', lambda *, phase: seen.append(phase))
    monkeypatch.setattr(b, '_stage', seen.append)
    b.run()
    assert seen == expected and b.env['EXPECTED_EPISODES'] == '4'


def test_failed_curves_cannot_mark_complete(tmp_path, monkeypatch):
    b = builder(tmp_path)
    def fail(**kwargs): raise RuntimeError('worker failed')
    monkeypatch.setattr(b, '_pool', fail)
    monkeypatch.setattr(b, '_stage', lambda *a: pytest.fail('Must not reduce or mark failed build'))
    with pytest.raises(RuntimeError, match='worker failed'): b.run()


@pytest.mark.parametrize('mode', ['srun_existing', 'srun', 'sbatch'])
def test_submitter_uses_one_allocation_without_array_dependencies(tmp_path, mode):
    b = builder(tmp_path)
    (b.dataset / 'meta/tasks.parquet').parent.mkdir(exist_ok=True)
    pd.DataFrame({'task_index': [0, 1]}).to_parquet(b.dataset / 'meta/tasks.parquet')
    policy = tmp_path / 'policy'; policy.mkdir()
    config = tmp_path / 'build.yaml'; config.write_text('{}')
    capture = tmp_path / 'args'
    source = (SRC.parent / 'submit_build_data.sh').read_text()
    start = source.index('if [ "${SKILLSET_SINGLE_GPU_WORKERS}" -gt 0 ]; then')
    branch = source[start:source.index("\nIFS=", start)]
    harness = '''
set -euo pipefail
submit_job_scheduler() { echo slurm; }
submit_job_slurm_mode() { echo "$TEST_MODE"; }
submit_job() {
 [ "$LIBERO_SINGLE_GPU_BUILD" = true ]
 [ -f "$LIBERO_BUILD_ENV" ]
 printf '%s\\n' "$@" > "$TEST_CAPTURE"
 echo 12345
}
'''
    env = dict(b.env, TEST_MODE=mode, TEST_CAPTURE=str(capture), SCRIPT_DIR=str(b.cwd),
               BUILD_SRC_DIR=str(b.src), CONFIG_PATH=str(config), BOOTSTRAP_PYTHON=sys.executable,
               DP_POLICY_PATH=str(policy), TARGET_DATASET='test', SKILLSET_DONE_PATH=str(tmp_path / '.complete'),
               RESOLVED_SETTINGS='export TEST_FROZEN=1', SLURM_PARTITION='debug', SLURM_QOS='normal',
               SLURM_GRES='gpu:B200:1', SKILLSET_MEM='64G', SKILLSET_TIME='08:00:00',
               SLURM_NODELIST='', SLURM_EXCLUDE_NODES='')
    result = subprocess.run(['bash', '-c', harness + branch], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    args = capture.read_text().splitlines()
    assert '--gres=gpu:B200:1' in args and '--cpus-per-task=4' in args
    assert ('--parsable' in args) == (mode == 'sbatch')
    assert not any(arg.startswith(('--array', '--dependency')) for arg in args)
    assert args[-1] == str(b.src / 'build_single_gpu.sbatch')
