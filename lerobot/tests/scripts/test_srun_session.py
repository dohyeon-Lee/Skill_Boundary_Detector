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
        assert len((tmp_path / 'srun_calls').read_text().splitlines()) == 2

        # A fresh submit command in the allocated shell reuses it too.
        command = 'source ' + shlex.quote(str(SUBMIT_JOB)) + '; FOO=fresh submit_job ' + shlex.quote(str(script))
        send(session, command)
        wait_for(lambda: 'foo=fresh' in output.read_text())
        assert len((tmp_path / 'srun_calls').read_text().splitlines()) == 2
        assert len(subprocess.check_output([*tmux, 'list-sessions'], text=True).splitlines()) == 2
        send(session, 'exit')
        wait_for(lambda: (path.parent / 'allocation_exit.json').exists())
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
