"""Existing-allocation DROID CLI is tested without submitting real Slurm work."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from test_droid_dataset import source, prepare  # shared tiny raw-video fixture

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'lerobot/skillVLA/droid/configs/generate_training_dataset/download_dataset/src'
sys.path.insert(0, str(SRC))
try:
    spec = importlib.util.spec_from_file_location('droid_launcher_test', SRC / 'launch_prepare_dataset.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
finally:
    sys.path.pop(0)


@pytest.mark.parametrize('tokens,job,workers', [
    ([], None, None),
    (['--75000'], 75000, None),
    (['--75000', '--16'], 75000, 16),
    (['--75000', '--64'], 75000, 64),
    (['--job-id', '75000', '--workers', '32'], 75000, 32),
    (['--workers=24'], None, 24),
])
def test_numeric_selectors(tokens, job, workers):
    args = launcher.arguments(tokens)
    assert (args.job_id, args.workers) == (job, workers)


@pytest.mark.parametrize('tokens', [
    ['--0'], ['--075000'], ['--75000', '--0'], ['--75000', '--16', '--8'],
    ['--75000', '--job-id', '123'], ['--75000', '--16', '--workers', '32'],
    ['--workers', '-1'], ['--dry-run', '--verify'],
])
def test_invalid_selectors_never_dispatch(tokens):
    with pytest.raises(SystemExit) as error:
        launcher.arguments(tokens)
    assert error.value.code == 2


def config(tmp_path):
    cfg = prepare.load_settings(SRC.parent / 'dataset_config.yaml')
    cfg.update(dataset_root=str(tmp_path / 'datasets'), output_path=str(tmp_path / 'datasets/output'))
    return cfg


def test_worker_budget_includes_encoder_threads(tmp_path):
    args = launcher.arguments(['--75000', '--32'])
    cfg = config(tmp_path)
    allocation = {'NumCPUs': '64', 'NodeList': 'node200', 'SubmitTime': 'identity'}
    assert launcher.build_plan(args, cfg, allocation)['cpu_budget'] == 32
    cfg['encoder_threads'] = 2
    assert launcher.build_plan(args, cfg, allocation)['cpu_budget'] == 64
    with pytest.raises(ValueError, match='need 64 CPUs'):
        launcher.build_plan(args, cfg, {**allocation, 'NumCPUs': '16'})


def test_cpu_dispatch_freezes_config_and_uses_original_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, 'ROOT', tmp_path)
    cfg = config(tmp_path)
    args = launcher.arguments(['--75000', '--64'])
    allocation = {'NumCPUs': '64', 'NodeList': 'node200', 'SubmitTime': 'identity'}
    calls = []
    def private_json(path, data):
        path.write_text(json.dumps(data))
    runner = {'private_json': private_json, 'submit': lambda command, existing_job_id: calls.append((command, existing_job_id)) or 0}
    plan = launcher.build_plan(args, cfg, allocation)
    assert launcher.queue(args, cfg, plan, runner) == 0
    command, job = calls.pop()
    assert job == '75000'
    assert '--cpus-per-task=64' in command
    assert not any(x.startswith(('--gres=', '--array=', '--dependency=')) for x in command)
    assert str(SRC / 'prepare_in_allocation.sh') in command
    snapshot = next((tmp_path / '.cache/droid_conversions').glob('*/dataset_config.yaml'))
    frozen = yaml.safe_load(snapshot.read_text())
    assert frozen['workers'] == 64
    assert frozen['action_mode'] == cfg['action_mode']
    assert snapshot.stat().st_mode & 0o077 == 0
    assert cfg['workers'] == 4  # live settings were not rewritten


def test_dry_run_validates_but_does_not_submit_or_create_capsule(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    monkeypatch.setattr(launcher, 'ROOT', tmp_path)
    monkeypatch.setattr(launcher, 'load_settings', lambda _: cfg)
    submitted, commands = [], []
    allocation = {'NumCPUs': '64', 'NodeList': 'node200', 'SubmitTime': 'identity'}
    monkeypatch.setattr(launcher, 'shared_runner', lambda: {
        'existing_allocation': lambda job: allocation,
        'submit': lambda *args, **kwargs: submitted.append(args)})
    monkeypatch.setattr(launcher.subprocess, 'call', lambda command: commands.append(command) or 0)
    assert launcher.main(['--75000', '--16', '--dry-run']) == 0
    assert not submitted and not (tmp_path / '.cache').exists()
    assert commands[0][-3:] == ['--workers', '16', '--dry-run']


def test_dead_allocation_has_no_local_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, 'load_settings', lambda _: config(tmp_path))
    def rejected(_):
        raise ValueError('requires a RUNNING allocation')
    monkeypatch.setattr(launcher, 'shared_runner', lambda: {'existing_allocation': rejected})
    calls = []
    monkeypatch.setattr(launcher.subprocess, 'call', lambda command: calls.append(command))
    with pytest.raises(ValueError, match='RUNNING'):
        launcher.main(['--75000', '--16'])
    assert not calls


def test_allocation_entrypoint_rejects_execution_outside_allocation():
    env = {k: v for k, v in os.environ.items() if not k.startswith('SLURM_')}
    result = subprocess.run(['bash', str(SRC / 'prepare_in_allocation.sh'), '75000', 'missing.yaml', '16'],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'outside the requested original' in result.stderr


def test_allocation_entrypoint_converts_fixture_on_cpu(source, tmp_path):
    cfg = {**source, 'workers': 2}
    frozen = tmp_path / 'frozen.yaml'
    frozen.write_text(yaml.safe_dump(cfg))
    env = {**os.environ, 'SLURM_JOB_ID': '123456',
           'SLURMD_NODENAME': os.uname().nodename.split('.')[0],
           'CUDA_VISIBLE_DEVICES': 'must-be-hidden'}
    result = subprocess.run(['bash', str(SRC / 'prepare_in_allocation.sh'), '123456', str(frozen), '2'],
                            env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert 'CPUs=2, GPU disabled' in result.stdout
    info = json.loads((Path(cfg['output_path']) / 'meta/info.json').read_text())
    assert info['total_episodes'] == 2 and info['total_frames'] == 12


def test_shell_entrypoint_routes_numeric_options_without_slurm(monkeypatch, tmp_path):
    # -h must be handled by the new CLI rather than rejected by the converter.
    result = subprocess.run(['bash', str(SRC.parent / 'prepare_dataset.sh'), '--help'],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert '--job-id' in result.stdout and '--workers' in result.stdout
