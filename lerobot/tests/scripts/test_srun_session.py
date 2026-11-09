"""Persistent launch tests use an isolated tmux server and a fake srun, never Slurm."""
from __future__ import annotations

import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from test_submit_job import SUBMIT_JOB, _env, _job_script, _submit

SCRIPT = SUBMIT_JOB.with_name("srun_session.py")
SPEC = importlib.util.spec_from_file_location("srun_session", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def wait_for(predicate, timeout=10):
    end = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < end, "timed out"
        time.sleep(.05)


def test_request_overrides_resources_without_interpreting_script_arguments(tmp_path):
    script = _job_script(tmp_path, "exit 0")
    literal = '$(touch should-not-exist) "quoted"'
    options, command = MODULE.request(["--gres=gpu:B200:1", "--nodelist=node200", str(script), literal])
    assert options["--gres"] == "gpu:B200:1" and options["--job-name"] == "demo"
    assert options["--nodelist"] == "node200" and command[-1] == literal
    assert not (tmp_path / "should-not-exist").exists()


def test_reserve_only_records_gpu_without_starting_training(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    path = tmp_path / "launch.json"
    path.write_text(json.dumps({"name": "standby", "cwd": str(tmp_path), "reserve_only": True,
                               "command": ["must-not-run"], "environment": {}, "options": {}}))
    assert MODULE.run_training(path) == 0
    assert json.loads((tmp_path / "allocation_status.json").read_text())["job_id"] == "123"
    assert not (tmp_path / "training_status.json").exists()


def test_guard_configuration_is_frozen_and_explicit_opt_out_is_preserved(monkeypatch):
    monkeypatch.setattr(MODULE.runpy,'run_path',lambda _: {'load_global_config':lambda _: {'srun_gpu_memory_guard':True,'srun_gpu_reserve_gib':6}})
    env=MODULE.memory_guard_environment({})
    assert env['SBD_GPU_MEMORY_GUARD']=='true' and env['SBD_GPU_GUARD_RESERVE_GIB']=='6'
    assert MODULE.memory_guard_environment({'SBD_GPU_MEMORY_GUARD':'0'})=={'SBD_GPU_MEMORY_GUARD':'0'}


@pytest.mark.parametrize("option", ["--parsable", "--array=0-2", "--dependency=afterok:123"])
def test_batch_workflows_keep_numeric_ids_and_dependencies(tmp_path, option):
    script = _job_script(tmp_path, "exit 0")
    result = _submit(tmp_path, _env(tmp_path, "slurm", SBD_SLURM_SUBMIT_MODE="srun"), option, str(script))
    assert result.returncode == 0 and result.stdout.startswith("sbatch ")


@pytest.mark.skipif(not shutil.which("tmux"), reason="tmux not installed")
def test_unique_sessions_keep_gpu_after_failure_retry_and_reuse(tmp_path):
    socket = "sbd-test-" + uuid.uuid4().hex
    sessions = tmp_path / "sessions"
    home = tmp_path / "home"
    home.mkdir()
    env = _env(tmp_path, "slurm", SBD_SLURM_SUBMIT_MODE="srun", SBD_TMUX_SOCKET=socket,
               SBD_SRUN_SESSIONS_DIR=str(sessions), HOME=str(home), CUDA_VISIBLE_DEVICES="wrong-submit-gpu",
               SRUN_TEST_ROOT=str(tmp_path))
    fake = tmp_path / "bin/srun"
    fake.write_text('#!' + sys.executable + '''
import json,os,sys
from pathlib import Path
root=Path(os.environ['SRUN_TEST_ROOT'])
with (root/'srun_calls').open('a') as f:
 f.write(json.dumps(sys.argv[1:])+'\\n')
env=os.environ.copy()
env.update(SLURM_JOB_ID=str(os.getpid()), SLURMD_NODENAME='fake-node', CUDA_VISIBLE_DEVICES='assigned-gpu')
rc=sys.argv[sys.argv.index('--rcfile')+1]
os.execvpe('bash',['bash','--noprofile','--rcfile',rc,'-i'],env)
''')
    fake.chmod(0o755)
    script = _job_script(tmp_path, 'echo "job=$SLURM_JOB_ID gpu=$CUDA_VISIBLE_DEVICES foo=$FOO"; exit 7')
    tmux = ["tmux", "-L", socket]
    def send(session, command):
        subprocess.run([*tmux, "send-keys", "-t", session, "-l", command], check=True)
        subprocess.run([*tmux, "send-keys", "-t", session, "Enter"], check=True)
    try:
        records = []
        for _ in range(2):
            result = _submit(tmp_path, env, "--nodelist=node200", "--cpus-per-task=48", str(script))
            assert result.returncode == 0, result.stderr
            wait_for(lambda: len(list(sessions.glob('*/launch.json'))) > len(records))
            path = next(p for p in sessions.glob('*/launch.json') if p not in records)
            records.append(path)
            wait_for(lambda: (path.parent / "training_status.json").exists()
                     and json.loads((path.parent / "training_status.json").read_text()).get("state") == "FAILED")
            data = json.loads(path.read_text())
            status = json.loads((path.parent / "training_status.json").read_text())
            assert 'assigned-gpu' in Path(status['output']).read_text()
            assert 'foo=bar' in Path(status['output']).read_text()
            assert path.stat().st_mode & 0o077 == 0
            subprocess.run([*tmux, "has-session", "-t", data['session']], check=True)
            assert subprocess.check_output([*tmux, 'show-options', '-w', '-v', '-t',
                                             data['session'], 'remain-on-exit'], text=True).strip() == 'off'
            # A user's global retain-dead-panes setting must not keep our sessions.
            subprocess.run([*tmux, 'set-option', '-g', 'remain-on-exit', 'on'], check=True)
        names = [json.loads(p.read_text())["session"] for p in records]
        assert len(set(names)) == 2
        calls = (tmp_path / 'srun_calls').read_text().splitlines()
        assert len(calls) == 2
        assert '--nodelist=node200' in json.loads(calls[0])
        assert '--cpus-per-task=48' in json.loads(calls[0])

        path, session = records[0], names[0]
        status = json.loads((path.parent / 'training_status.json').read_text())
        output = Path(status['output'])
        wait_for(lambda: output.read_text().count('job=') == 1)
        send(session, 'sbd_retry')
        wait_for(lambda: output.read_text().count('job=') == 2)
        wait_for(lambda: len(list((tmp_path / 'training_events').glob('*.json'))) >= 3)
        attempts = [json.loads(p.read_text()) for p in (tmp_path / 'training_events').glob('*.json')]
        assert len({attempt['run_id'] for attempt in attempts}) == len(attempts)
        assert all(attempt['started_at'].endswith('+00:00') for attempt in attempts)
        assert sum(attempt['state'] == 'FAILED' for attempt in attempts) >= 2
        assert len((tmp_path / 'srun_calls').read_text().splitlines()) == 2

        # A fresh submit command in the allocated shell reuses it too.
        command = 'source ' + shlex.quote(str(SUBMIT_JOB)) + '; FOO=fresh submit_job ' + shlex.quote(str(script))
        send(session, command)
        wait_for(lambda: 'foo=fresh' in output.read_text())
        assert len((tmp_path / 'srun_calls').read_text().splitlines()) == 2
        assert len(subprocess.check_output([*tmux, 'list-sessions'], text=True).splitlines()) == 2
        send(session, 'exit')
        wait_for(lambda: (path.parent / 'allocation_exit.json').exists())
        wait_for(lambda: subprocess.run([*tmux, 'has-session', '-t', session],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0)
        assert json.loads((path.parent / 'allocation_exit.json').read_text())['exit_code'] == 7
        assert output.is_file()  # Logs survive the automatic tmux cleanup.
        subprocess.run([*tmux, 'has-session', '-t', names[1]], check=True)
    finally:
        subprocess.run([*tmux, 'kill-server'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@pytest.mark.skipif(not shutil.which('tmux'), reason='tmux not installed')
@pytest.mark.parametrize('exit_code', [1, 124, 137])
def test_allocation_failure_or_timeout_closes_tmux_but_keeps_exit_record(tmp_path, exit_code):
    socket = 'sbd-exit-test-' + uuid.uuid4().hex
    sessions = tmp_path / 'sessions'
    env = _env(tmp_path, 'slurm', SBD_SLURM_SUBMIT_MODE='srun', SBD_TMUX_SOCKET=socket,
               SBD_SRUN_SESSIONS_DIR=str(sessions))
    fake = tmp_path / 'bin/srun'
    fake.write_text(f'#!/bin/bash\nexit {exit_code}\n')
    fake.chmod(0o755)
    tmux = ['tmux', '-L', socket]
    try:
        result = _submit(tmp_path, env, str(_job_script(tmp_path, 'exit 0')))
        assert result.returncode == 0, result.stderr
        wait_for(lambda: bool(list(sessions.glob('*/allocation_exit.json'))))
        record = next(sessions.glob('*/allocation_exit.json'))
        assert json.loads(record.read_text())['exit_code'] == exit_code
        session = json.loads(record.with_name('launch.json').read_text())['session']
        wait_for(lambda: subprocess.run([*tmux, 'has-session', '-t', session],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0)
    finally:
        subprocess.run([*tmux, 'kill-server'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_gpu_guard_does_not_requeue_managed_interactive_allocation(tmp_path):
    env = _env(tmp_path, 'slurm', SBD_SRUN_SESSION='1', SLURM_JOB_ID='123',
               GPU_GUARD_BAD_NODE_FILE=str(tmp_path/'bad_nodes'))
    script = SUBMIT_JOB.with_name('gpu_guard.sh')
    result = subprocess.run(['bash', '-c', 'source "$1"; guard_requeue_current_job TEST failure', '_', str(script)],
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert 'leaving the allocation intact' in result.stderr


@pytest.mark.skipif(not Path('/dev/full').exists(), reason='requires Linux /dev/full')
def test_log_write_failure_does_not_block_the_training_pipe(tmp_path):
    env = _env(tmp_path, 'slurm', SBD_SLURM_SUBMIT_MODE='srun', SBD_SRUN_SESSION='1',
               SLURM_JOB_ID='123', SBD_SRUN_SESSIONS_DIR=str(tmp_path/'sessions'))
    script = _job_script(tmp_path, shlex.join([sys.executable, '-c', "print('x'*262144)"]))
    result = _submit(tmp_path, env, '--output=/dev/full', str(script))
    assert result.returncode == 0
    assert 'continuing to drain output' in result.stderr
    assert 'remains available' in result.stdout


def fake_allocation_bin(tmp_path, *, state='RUNNING', uid=None, gpus=1):
    binary = tmp_path / 'bin/scontrol'
    owner = os.getuid() if uid is None else uid
    text = (f'JobId=8801 UserId=test({owner}) JobState={state} NumNodes=1 NumCPUs=48 '
            f'TRES=cpu=48,gres/gpu={gpus} SubmitTime=2026-11-03T12:00:00 NodeList=fake-node')
    binary.write_text('#!/bin/bash\nprintf "%s\\n" ' + shlex.quote(text) + '\n')
    binary.chmod(0o755)


@pytest.mark.parametrize('state,foreign,gpus', [('PENDING', False, 1), ('COMPLETED', False, 1),
                                               ('RUNNING', True, 1), ('RUNNING', False, 2)])
def test_existing_mode_rejects_unusable_targets_without_submitting(tmp_path, state, foreign, gpus):
    env = _env(tmp_path, 'slurm', SBD_SLURM_SUBMIT_MODE='srun_existing', SBD_SRUN_JOB_ID='8801',
               SBD_SRUN_SESSIONS_DIR=str(tmp_path/'sessions'))
    fake_allocation_bin(tmp_path, state=state, uid=os.getuid()+1 if foreign else None, gpus=gpus)
    result = _submit(tmp_path, env, str(_job_script(tmp_path, 'exit 0')))
    assert result.returncode != 0 and 'sbatch ' not in result.stdout
    assert not (tmp_path/'sessions').exists()


@pytest.mark.skipif(not shutil.which('tmux'), reason='tmux not installed')
def test_existing_mode_shares_only_target_allocation_and_keeps_logs_separate(tmp_path):
    socket = 'sbd-existing-test-' + uuid.uuid4().hex
    sessions = tmp_path/'sessions'
    home = tmp_path/'home'
    home.mkdir()
    env = _env(tmp_path, 'slurm', SBD_SLURM_SUBMIT_MODE='srun_existing', SBD_SRUN_JOB_ID='8801',
               SBD_TMUX_SOCKET=socket, SBD_SRUN_SESSIONS_DIR=str(sessions), HOME=str(home),
               SRUN_TEST_ROOT=str(tmp_path))
    fake_allocation_bin(tmp_path)
    binary = tmp_path/'bin/srun'
    binary.write_text('#!'+sys.executable+'''
import json,os,sys
from pathlib import Path
Path(os.environ['SRUN_TEST_ROOT'],'step_args.json').write_text(json.dumps(sys.argv[1:]))
assert '--jobid=8801' in sys.argv and '--overlap' in sys.argv
env=os.environ.copy()
env.update(SLURM_JOB_ID='8801',SLURM_STEP_ID='7',CUDA_VISIBLE_DEVICES='original-allocation-gpu')
rc=sys.argv[sys.argv.index('--rcfile')+1]
os.execvpe('bash',['bash','--noprofile','--rcfile',rc,'-i'],env)
''')
    binary.chmod(0o755)
    script = _job_script(tmp_path, 'echo "gpu=$CUDA_VISIBLE_DEVICES"; exit 0')
    (tmp_path/'logs').mkdir()
    original = tmp_path/'logs/demo_8801.out'
    original.write_text('original training log\n')
    tmux = ['tmux','-L',socket]
    try:
        result = _submit(tmp_path, env, '--partition=debug', '--qos=base_qos', '--nodelist=node200', str(script))
        assert result.returncode == 0, result.stderr
        wait_for(lambda: bool(list(sessions.glob('*/training_status.json'))))
        path = next(sessions.glob('*/training_status.json'))
        wait_for(lambda: json.loads(path.read_text()).get('state') == 'COMPLETED')
        status = json.loads(path.read_text())
        assert status['job_id']=='8801'
        assert Path(status['output']).name == 'demo_8801.step7.out'
        assert 'original-allocation-gpu' in Path(status['output']).read_text()
        assert original.read_text() == 'original training log\n'
        args=json.loads((tmp_path/'step_args.json').read_text())
        assert '--jobid=8801' in args and '--overlap' in args
        assert not any(arg.startswith(('--partition=','--qos=','--nodelist=','--time=')) for arg in args)
        launch=json.loads((path.parent/'launch.json').read_text())
        subprocess.run([*tmux,'has-session','-t',launch['session']],check=True)
    finally:
        subprocess.run([*tmux,'kill-server'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


def test_existing_mode_never_falls_back_to_sbatch_for_batch_only_options(tmp_path):
    env = _env(tmp_path, 'slurm', SBD_SLURM_SUBMIT_MODE='srun_existing', SBD_SRUN_JOB_ID='8801')
    result = _submit(tmp_path, env, '--parsable', str(_job_script(tmp_path,'exit 0')))
    assert result.returncode != 0 and 'no new allocation' in result.stderr
    assert not result.stdout.startswith('sbatch ')


@pytest.mark.parametrize('identifier',['','--hold','8801;echo bad','8801_2'])
def test_existing_mode_requires_an_unambiguous_numeric_id(identifier):
    with pytest.raises(ValueError,match='positive numeric'):
        MODULE.existing_allocation(identifier)
