#!/usr/bin/env python3
"""Dispatch frozen launches as children of an already allocated Slurm step."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid

SELF = Path(__file__).resolve()


def write_json(path, data):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(data, stream, indent=2)
    temporary.replace(path)


def directory_for(root, allocation):
    identity = hashlib.sha256(allocation['SubmitTime'].encode()).hexdigest()[:12]
    return root / '.workers' / f"{allocation['JobId']}-{identity}"


def ready(directory):
    try:
        data = json.loads((directory / 'worker_status.json').read_text())
        return data.get('state') == 'RUNNING' and 0 <= time.time() - data['heartbeat'] < 30
    except (OSError, ValueError, KeyError):
        return False


def start(directory, allocation, runner):
    if os.environ.get('SLURM_JOB_ID') != allocation['JobId']:
        raise ValueError('The allocation worker must start inside the original srun shell.')
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('The original srun shell has no assigned visible GPU.')
    if os.environ.get('SLURMD_NODENAME') != os.uname().nodename.split('.')[0]:
        raise ValueError('Start the worker on the allocated compute node, inside its original srun shell.')
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not ready(directory):
        with (directory / 'worker.log').open('ab', buffering=0) as log:
            subprocess.Popen([sys.executable, str(SELF), 'serve', str(directory),
                              allocation['JobId'], allocation['SubmitTime'], str(runner)],
                             stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                             start_new_session=True)
    deadline = time.monotonic() + 5
    while not ready(directory):
        if time.monotonic() > deadline:
            raise ValueError(f'Allocation worker did not start; inspect {directory / "worker.log"}')
        time.sleep(.05)
    return directory


def ensure(root, allocation, runner, tmux):
    directory = directory_for(root, allocation)
    if ready(directory):
        return directory
    node = os.environ.get('SLURMD_NODENAME', '')
    if os.environ.get('SLURM_JOB_ID') == allocation['JobId'] and node == os.uname().nodename.split('.')[0]:
        return start(directory, allocation, runner)
    # Upgrade an older managed allocation only when its original shell is idle.
    # Never send commands into an actively running trainer or open another step.
    for status_path in root.glob('*/allocation_status.json'):
        try:
            status = json.loads(status_path.read_text())
            launch = json.loads(status_path.with_name('launch.json').read_text())
            if status.get('job_id') != allocation['JobId'] or launch.get('existing_job_id'):
                continue
            if status_path.with_name('allocation_exit.json').exists():
                continue
            target = launch['session']
            command = subprocess.check_output([*tmux, 'display-message', '-p', '-t', target,
                                                '#{pane_current_command}'], text=True,
                                               stderr=subprocess.DEVNULL).strip()
            if command not in {'bash', 'sh', 'zsh'}:
                screen = subprocess.check_output([*tmux, 'capture-pane', '-p', '-t', target,
                                                   '-S', '-5'], text=True)
                lines = [line.strip() for line in screen.splitlines() if line.strip()]
                if not lines or re.search(r'[$#%]\s*$', lines[-1]) is None:
                    continue
            bootstrap = shlex.join([sys.executable, str(SELF), 'start-current', str(root), str(runner)])
            subprocess.run([*tmux, 'send-keys', '-t', target, '-l', bootstrap], check=True)
            subprocess.run([*tmux, 'send-keys', '-t', target, 'Enter'], check=True)
            deadline = time.monotonic() + 6
            while time.monotonic() < deadline:
                if ready(directory):
                    return directory
                time.sleep(.1)
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
    bootstrap = shlex.join([sys.executable, str(SELF), 'start-current', str(root), str(runner)])
    raise ValueError('No worker is running inside the original allocation. In its existing srun shell, run: '
                     + bootstrap + '. No new job or GPU step was requested.')


def enqueue(directory, path):
    if not ready(directory):
        raise ValueError('The original allocation worker is unavailable; no job was submitted.')
    status_path = path.parent / 'dispatch_status.json'
    if status_path.exists() and json.loads(status_path.read_text()).get('state') in {'QUEUED', 'RUNNING'}:
        raise ValueError('This frozen launch is already queued or running in the original allocation.')
    requests = directory / 'requests'
    requests.mkdir(mode=0o700, exist_ok=True)
    write_json(path.parent / 'dispatch_status.json', {'state': 'QUEUED', 'worker': str(directory)})
    write_json(requests / (path.parent.name + '.json'), {'launch_path': str(path)})


def serve(directory, identifier, submit_time, runner):
    if os.environ.get('SLURM_JOB_ID') != identifier or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Refusing to dispatch outside the original GPU allocation.')
    lock = (directory / 'worker.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    requests = directory / 'requests'
    requests.mkdir(mode=0o700, exist_ok=True)
    children = {}
    status = {'state': 'RUNNING', 'job_id': identifier, 'allocation_submit_time': submit_time,
              'node': os.environ.get('SLURMD_NODENAME', ''), 'step_id': os.environ.get('SLURM_STEP_ID', ''),
              'pid': os.getpid()}
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    last_heartbeat = 0
    try:
        while not stopping:
            if time.monotonic() - last_heartbeat > 2:
                status['heartbeat'] = time.time()
                write_json(directory / 'worker_status.json', status)
                last_heartbeat = time.monotonic()
            for request in sorted(requests.glob('*.json')):
                path = None
                try:
                    payload = json.loads(request.read_text())
                    path = Path(payload['launch_path']).resolve()
                    if path.name != 'launch.json' or path.parent.parent != directory.parent.parent:
                        raise ValueError('Launch must belong to the private session directory.')
                    data = json.loads(path.read_text())
                    if data.get('existing_job_id') != identifier or data.get('allocation_submit_time') != submit_time:
                        raise ValueError('Allocation identity does not match the queued launch.')
                    request.rename(request.with_suffix('.claimed'))
                    with (path.parent / 'srun_console.log').open('ab', buffering=0) as log:
                        child = subprocess.Popen([sys.executable, str(runner), 'run', str(path)],
                                                 stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                                 start_new_session=True)
                    children[child.pid] = (child, path)
                    write_json(path.parent / 'dispatch_status.json', {
                        'state': 'RUNNING', 'pid': child.pid, 'job_id': identifier,
                        'step_id': status['step_id'], 'worker': str(directory)})
                except (OSError, ValueError, KeyError) as error:
                    if path and path.parent.parent == directory.parent.parent:
                        write_json(path.parent / 'dispatch_status.json', {'state': 'FAILED', 'error': str(error)})
                    if request.exists():
                        request.rename(request.with_suffix('.rejected'))
            for pid, (child, path) in list(children.items()):
                code = child.poll()
                if code is not None:
                    write_json(path.parent / 'dispatch_status.json', {
                        'state': 'COMPLETED' if code == 0 else 'FAILED', 'exit_code': code,
                        'job_id': identifier, 'step_id': status['step_id'], 'worker': str(directory)})
                    del children[pid]
            time.sleep(.2)
    finally:
        for child, path in children.values():
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        status.update(state='STOPPED', heartbeat=time.time())
        write_json(directory / 'worker_status.json', status)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['start-current', 'serve'])
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.mode == 'serve':
        directory, identifier, submit_time, runner = args.arguments
        serve(Path(directory), identifier, submit_time, Path(runner))
    else:
        root, runner = map(Path, args.arguments)
        identifier = os.environ.get('SLURM_JOB_ID', '')
        if not identifier.isdigit():
            raise ValueError('Run start-current inside the original srun shell.')
        text = subprocess.check_output(['scontrol', 'show', 'job', '-o', identifier], text=True, timeout=15,
                                       env={**os.environ, 'TZ': 'UTC'})
        allocation = dict(part.split('=', 1) for part in text.split() if '=' in part)
        if allocation.get('JobId') != identifier or allocation.get('JobState') != 'RUNNING':
            raise ValueError('The original allocation is no longer running.')
        print('Allocation worker ready: ' + str(start(directory_for(root, allocation), allocation, runner)))


if __name__ == '__main__':
    main()
