"""Stop one explicitly identified run inside its existing allocation, never scancel."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import runpy
import signal
import time
import uuid


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream)
    temporary.replace(path)


def supported(row):
    return bool(re.fullmatch(r'[a-f0-9]{32}', row.get('run_id', ''))
                and re.fullmatch(r'[1-9][0-9]*', row.get('job_id', ''))
                and row.get('status_path') and not row.get('terminal'))


def verify(root, target):
    path = Path(target['status_path'])
    if path.is_symlink() or path.stat().st_uid != os.getuid():
        raise ValueError('학습 상태 파일의 소유자가 일치하지 않습니다.')
    path.resolve().relative_to(Path(root).resolve() / '.cache')
    record = read(path)
    for field in ('run_id', 'job_id', 'started_at'):
        if record.get(field) != target.get(field):
            raise ValueError('학습 실행 정보가 변경되었습니다. 새로고침 후 확인하세요.')
    if record.get('state') != 'RUNNING':
        raise ValueError('이미 종료됐거나 종료 상태를 확인할 수 없는 실행입니다.')
    return record


def request(root, target):
    root = Path(root).resolve()
    if not supported(target):
        raise ValueError('실행 ID가 확인된 개별 작업만 종료할 수 있습니다.')
    verify(root, target)
    runner = root / 'lerobot/skillVLA/libero/configs/src/srun_session.py'
    session = runpy.run_path(str(runner))
    allocation = session['existing_allocation'](target['job_id'])
    worker = session['allocation_worker']()
    directory = worker['directory_for'](root / '.cache/srun_sessions', allocation)
    if not worker['ready'](directory):
        raise ValueError('기존 할당의 실행 관리자가 응답하지 않습니다. 전체 할당은 취소하지 않았습니다.')
    receipt = root / '.cache/training_stops' / (target['run_id'] + '.json')
    launch = root / '.cache/srun_sessions' / ('desk-stop-' + uuid.uuid4().hex) / 'launch.json'
    launch.parent.mkdir(parents=True, mode=0o700)
    payload = {k: target[k] for k in ('run_id', 'job_id', 'started_at', 'status_path')}
    payload['node'] = allocation.get('NodeList', '')
    data = {'name': 'SlurmDesk_stop_run', 'existing_job_id': target['job_id'],
            'allocation_submit_time': allocation['SubmitTime'],
            'training_control': payload, 'control_receipt': str(receipt),
            'control_root': str(root)}
    write(launch, data)
    write(receipt, {'state': 'QUEUED', 'launch_path': str(launch), 'requested_at': time.time()})
    try:
        worker['enqueue'](directory, launch)
    except Exception as error:
        write(receipt, {'state': 'FAILED', 'error': str(error)})
        raise
    return str(receipt)


def identity(path, run_id, job_id):
    """Keep only a matching run identity, with a PID birth check around /proc reads."""
    if path.stat().st_uid != os.getuid():
        return None
    before = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    if before[0] == 'Z':
        return None
    values = {}
    for item in (path / 'environ').read_bytes().split(b'\0'):
        key, _, value = item.partition(b'=')
        if key in (b'SBD_TRAINING_PARENT_ID', b'SLURM_JOB_ID'):
            values[key] = value.decode()
    after = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    return before[19] if (before[19] == after[19] and
        values.get(b'SBD_TRAINING_PARENT_ID') == run_id and
        values.get(b'SLURM_JOB_ID') == job_id) else None


def alive(path, birth):
    try:
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == birth
    except (OSError, IndexError):
        return False


def send(handle, sig, run_id, job_id, proc):
    pid, birth, fd = handle
    try:
        if fd is not None:
            signal.pidfd_send_signal(fd, sig)
        elif identity(proc / str(pid), run_id, job_id) == birth:
            # Older cluster kernels lack pidfd. Recheck owner, run, allocation
            # and process birth immediately before each individual signal.
            os.kill(pid, sig)
    except (OSError, ValueError, IndexError):
        pass


def stop_processes(run_id, job_id, proc=Path('/proc'), grace=8):
    handles = []
    use_pidfd = hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal')
    try:
        for path in proc.iterdir():
            if not path.name.isdigit() or int(path.name) == os.getpid():
                continue
            fd = None
            try:
                if use_pidfd:
                    fd = os.pidfd_open(int(path.name))
                birth = identity(path, run_id, job_id)
                if birth:
                    handles.append((int(path.name), birth, fd))
                    fd = None
            except (OSError, ValueError, IndexError):
                pass
            finally:
                if fd is not None:
                    os.close(fd)
        if not handles:
            raise ValueError('해당 실행의 프로세스를 찾지 못했습니다. 다른 작업에는 신호를 보내지 않았습니다.')
        for handle in handles:
            send(handle, signal.SIGTERM, run_id, job_id, proc)
        remaining = handles[:]
        deadline = time.monotonic() + grace
        while remaining and time.monotonic() < deadline:
            remaining = [h for h in remaining if alive(proc / str(h[0]), h[1])]
            if remaining:
                time.sleep(.1)
        for handle in remaining:
            send(handle, signal.SIGKILL, run_id, job_id, proc)
        deadline = time.monotonic() + 3
        while remaining and time.monotonic() < deadline:
            remaining = [h for h in remaining if alive(proc / str(h[0]), h[1])]
            if remaining:
                time.sleep(.1)
        if remaining:
            raise ValueError('일부 프로세스의 종료 확인을 기다리고 있습니다.')
        # Catch late forked workers without ever claiming a partial stop completed.
        for path in proc.iterdir():
            if path.name.isdigit():
                try:
                    if identity(path, run_id, job_id):
                        raise ValueError('추가 자식 프로세스의 종료 확인이 필요합니다. 새로고침 후 다시 요청하세요.')
                except (OSError, IndexError):
                    pass
        return len(handles)
    finally:
        for _, _, fd in handles:
            if fd is not None:
                os.close(fd)


def execute(root, target, receipt):
    """Runs only via the existing allocation dispatcher; never opens SSH/a GPU step."""
    try:
        if (os.environ.get('SLURM_JOB_ID') != target['job_id'] or
                os.uname().nodename.split('.')[0] != target['node']):
            raise ValueError('개별 종료 요청이 지정된 원래 할당 안에서 실행되지 않았습니다.')
        verify(root, target)
        count = stop_processes(target['run_id'], target['job_id'])
        write(receipt, {'state': 'STOPPED', 'processes': count, 'ended_at': time.time()})
        return 0
    except Exception as error:
        write(receipt, {'state': 'FAILED', 'error': str(error), 'ended_at': time.time()})
        return 1
