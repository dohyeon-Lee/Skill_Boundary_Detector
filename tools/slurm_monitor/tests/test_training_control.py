import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from tools.slurm_monitor import training_control as control


def test_stop_signals_only_matching_run_and_keeps_other_training_alive():
    children = []
    try:
        for run in ('a' * 32, 'b' * 32):
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                env={**os.environ, 'SBD_TRAINING_PARENT_ID': run, 'SLURM_JOB_ID': '123'})
            children.append(child)
        assert control.stop_processes('a' * 32, '123', grace=1) == 1
        assert children[0].wait(timeout=3) == -signal.SIGTERM
        assert children[1].poll() is None
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=3)


def test_stale_status_and_wrong_allocation_never_signal(tmp_path, monkeypatch):
    path = tmp_path / '.cache/training_events/one.json'
    path.parent.mkdir(parents=True)
    target = {'run_id': 'a' * 32, 'job_id': '123', 'started_at': 'start',
              'status_path': str(path), 'node': os.uname().nodename.split('.')[0]}
    path.write_text(json.dumps({**target, 'state': 'RUNNING', 'run_id': 'b' * 32}))
    with pytest.raises(ValueError, match='변경'):
        control.verify(tmp_path, target)
    monkeypatch.setattr(control, 'stop_processes', lambda *a: pytest.fail('must not signal'))
    monkeypatch.setenv('SLURM_JOB_ID', '999')
    receipt = tmp_path / 'result.json'
    assert control.execute(tmp_path, target, receipt) == 1
    assert control.read(receipt)['state'] == 'FAILED'


def test_request_uses_existing_worker_without_new_step_or_gpu(tmp_path, monkeypatch):
    path = tmp_path / '.cache/training_events/one.json'
    path.parent.mkdir(parents=True)
    target = {'run_id': 'a' * 32, 'job_id': '123', 'started_at': 'start',
              'status_path': str(path), 'terminal': False}
    path.write_text(json.dumps({**target, 'state': 'RUNNING'}))
    queued = []
    worker = {'directory_for': lambda root, allocation: root / '.workers/test',
              'ready': lambda directory: True,
              'enqueue': lambda directory, launch: queued.append(launch)}
    session = {'existing_allocation': lambda identifier: {'JobId': identifier,
        'SubmitTime': 'allocation-start', 'NodeList': 'node200'},
        'allocation_worker': lambda: worker}
    monkeypatch.setattr(control.runpy, 'run_path', lambda path: session)
    receipt = Path(control.request(tmp_path, target))
    assert control.read(receipt)['state'] == 'QUEUED'
    assert len(queued) == 1
    launch = control.read(queued[0])
    assert launch['training_control']['run_id'] == target['run_id']
    assert launch['existing_job_id'] == '123'
    assert 'command' not in launch  # A control message, not a new training launch.


def test_control_branch_runs_before_training_launch():
    source = Path('lerobot/skillVLA/libero/configs/src/srun_session.py').read_text()
    assert source.index("if data.get('training_control'):") < source.index('environment = os.environ.copy()', source.index('def run_training'))


def test_process_identity_excludes_wrong_run_job_and_owner(tmp_path, monkeypatch):
    process = tmp_path / '42'
    process.mkdir()
    # /proc/stat fields after the closing parenthesis: state, ppid, ... starttime.
    (process / 'stat').write_text('42 (python) S ' + ' '.join(['0'] * 18 + ['1234']))
    (process / 'environ').write_bytes(b'SBD_TRAINING_PARENT_ID=' + b'a' * 32 + b'\0SLURM_JOB_ID=123\0')
    assert control.identity(process, 'a' * 32, '123') == '1234'
    assert control.identity(process, 'b' * 32, '123') is None
    assert control.identity(process, 'a' * 32, '456') is None
    monkeypatch.setattr(control.os, 'getuid', lambda: process.stat().st_uid + 1)
    assert control.identity(process, 'a' * 32, '123') is None


def test_legacy_signal_rechecks_process_birth_before_signalling(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(control, 'identity', lambda *args: 'new-birth')
    monkeypatch.setattr(control.os, 'kill', lambda *args: calls.append(args))
    control.send((12345, 'old-birth', None), signal.SIGTERM, 'a' * 32, '123', tmp_path)
    assert calls == []
