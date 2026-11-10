from pathlib import Path
import json

import huggingface_hub
import pytest

from tools.runpod import train_watchdog as wd


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setenv('RUNPOD_POD_ID', 'test-pod')
    monkeypatch.setattr(wd.shutil, 'which', lambda _: '/fake/runpodctl')
    monkeypatch.setattr(huggingface_hub, 'get_token', lambda: 'test-token')
    config = {'hf_checkpoint_repo': 'me/models', 'hf_dataset_repo': 'me/data',
              'runpod_watchdog': {'enabled': True, 'keep': 1, 'protect': [50000, 100000]}}
    runs = [{'output_dir': str(tmp_path / 'outputs_filtered/Predictor' / mode)} for mode in ('mode1', 'mode2')]
    resolved = {'project_root': tmp_path, 'steps': 100000, 'predictor_parallel_runs': json.dumps(runs)}
    return tmp_path, config, resolved


def test_prepare_keeps_two_modes_and_has_no_deadline(prepared):
    root, config, resolved = prepared
    folder = wd.prepare(root / 'config.yaml', loader=lambda _: (config, resolved))
    plan = json.loads((folder / 'plan.json').read_text())
    assert plan['keep'] == 1 and plan['protect'] == [50000, 100000]
    assert plan['done_step'] == 100000 and plan['done_time'] is None
    assert plan['expected_runs'] == ['outputs_filtered/Predictor/mode1', 'outputs_filtered/Predictor/mode2']
    assert 'goal_only' not in plan  # failures keep the existing shutdown behavior


@pytest.mark.parametrize('missing', ['token', 'runpodctl', 'pod'])
def test_prepare_fails_before_training_if_shutdown_unavailable(prepared, monkeypatch, missing):
    root, config, resolved = prepared
    if missing == 'token':
        monkeypatch.setattr(huggingface_hub, 'get_token', lambda: None)
    elif missing == 'runpodctl':
        monkeypatch.setattr(wd.shutil, 'which', lambda _: None)
    else:
        monkeypatch.delenv('RUNPOD_POD_ID')
    with pytest.raises(ValueError):
        wd.prepare(root / 'config.yaml', loader=lambda _: (config, resolved))


def test_prepare_refuses_outputs_outside_project(prepared):
    root, config, resolved = prepared
    resolved['predictor_parallel_runs'] = json.dumps([{'output_dir': str(root.parent / 'other')}])
    with pytest.raises(ValueError):
        wd.prepare(root / 'config.yaml', loader=lambda _: (config, resolved))


def test_launch_reuses_live_watchdog_and_validates_pod(prepared, monkeypatch):
    root, config, resolved = prepared
    folder = wd.prepare(root / 'config.yaml', loader=lambda _: (config, resolved))
    monkeypatch.setattr(wd, 'active', lambda _: {'pid': 123})
    monkeypatch.setattr(wd.subprocess, 'Popen', lambda *a, **k: pytest.fail('duplicate watcher'))
    wd.launch(folder, '1234')
    monkeypatch.setattr(wd, 'active', lambda _: None)
    monkeypatch.setenv('RUNPOD_POD_ID', 'wrong-pod')
    with pytest.raises(ValueError, match='different Pod'):
        wd.launch(folder, '1234')


@pytest.mark.parametrize('on_pod', [False, True])
def test_submitter_attaches_watchdog_only_to_local_runpod(tmp_path, on_pod):
    import os
    import subprocess
    source = wd.ROOT / 'lerobot/examples/libero/configs/train_skillVLA/PT/Terminator/src/submit_auxiliary.sh'
    tail = source.read_text().split('WATCHDOG_BUNDLE=""', 1)[1]
    fake_python = tmp_path / 'python'
    fake_python.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALLS"\nif [ "$2" = prepare ]; then echo /fake/bundle; fi\n')
    fake_python.chmod(0o755)
    script = '''set -euo pipefail
submit_job_scheduler() { echo local; }
submit_job() { printf 'submit %s\\n' "$*" >> "$CALLS"; echo 123; }
SBATCH_ARGS=(--time=72:00:00)
WATCHDOG_BUNDLE=""
''' + tail
    env = {**os.environ, 'RUNPOD_POD_ID': 'test' if on_pod else '', 'RUNPOD_WATCHDOG_ENABLED': 'true',
           'PROJECT_ROOT': str(tmp_path), 'BOOTSTRAP_PYTHON': str(fake_python),
           'CONFIG_PATH': '/fake/config.yaml', 'SRC_DIR': '/fake/src', 'CALLS': str(tmp_path / 'calls')}
    result = subprocess.run(['bash', '-c', script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = (tmp_path / 'calls').read_text()
    assert (' prepare ' in calls) == on_pod
    assert (' launch ' in calls) == on_pod
    assert ('--parsable' in calls) == on_pod
    assert '--done-time' not in calls
