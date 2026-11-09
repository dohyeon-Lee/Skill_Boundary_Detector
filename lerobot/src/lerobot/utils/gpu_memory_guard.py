"""Single-GPU admission and newest-first preemption inside an existing srun.

The supervisor never submits/cancels Slurm jobs or signals other supervisors.
Each owner stops only its own isolated trainer process group. CUDA allocator
limits and a polled free-memory threshold are best-effort protection, not GPU
partitioning. The supervisor itself uses only stdlib/nvidia-smi, not CUDA.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import uuid

GIB = 1024**3
OOM = re.compile(r"CUDA out of memory|CUDA error: out of memory|(?:torch\.)?(?:cuda\.)?OutOfMemoryError|CUDNN_STATUS_ALLOC_FAILED", re.I)
STORAGE = re.compile(r"No space left on device|Disk quota exceeded|\bENOSPC\b|\bEDQUOT\b", re.I)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, default=str)
    temporary.replace(path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


@dataclass
class Settings:
    reserve: int = 8 * GIB
    stop_below: int = 4 * GIB
    minimum: int = 8 * GIB
    retry_margin: int = 2 * GIB
    interval: float = 1.0
    grace: float = 2.0

    @classmethod
    def from_env(cls):
        values = {}
        for name in ('reserve', 'stop_below', 'minimum', 'retry_margin'):
            values[name] = int(float(os.environ.get('SBD_GPU_GUARD_' + name.upper() + '_GIB', getattr(cls(), name) / GIB)) * GIB)
        values['interval'] = float(os.environ.get('SBD_GPU_GUARD_INTERVAL', '1'))
        result = cls(**values)
        if not 0 < result.stop_below < result.reserve or result.minimum <= 0 or result.retry_margin <= 0 or result.interval < .1:
            raise ValueError('GPU guard requires 0 < stop_below < reserve, positive budgets, interval >= 0.1s')
        return result


class Nvidia:
    def __init__(self, env):
        visible = env.get('CUDA_VISIBLE_DEVICES', '')
        assigned = env.get('SLURM_JOB_GPUS', '')
        if not visible or ',' in visible or visible == '-1' or visible.startswith('MIG-'):
            raise ValueError('GPU memory guard requires exactly one visible GPU')
        self.selector = visible if visible.startswith('GPU-') else assigned if assigned and ',' not in assigned else visible

    def sample(self):
        text = subprocess.check_output(['nvidia-smi', '-i', self.selector,
            '--query-gpu=uuid,memory.total,memory.free', '--format=csv,noheader,nounits'],
            text=True, timeout=5)
        rows = [line.split(',') for line in text.strip().splitlines()]
        if len(rows) != 1 or len(rows[0]) != 3:
            raise ValueError('Cannot identify one assigned GPU')
        identifier, total, free = (part.strip() for part in rows[0])
        return identifier, int(float(total) * 1024**2), int(float(free) * 1024**2)


def birth(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return stat[19] if stat[0] != 'Z' else None
    except (OSError, IndexError):
        return None


class Registry:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.folder.parent.is_symlink() or self.folder.parent.stat().st_uid != os.getuid():
            raise ValueError('GPU guard registry parent must be owned by this user')
        self.folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.folder.is_symlink() or self.folder.stat().st_uid != os.getuid() or self.folder.stat().st_mode & 0o077:
            raise ValueError('GPU guard registry must be a private directory owned by this user')

    @contextlib.contextmanager
    def locked(self):
        with (self.folder / 'lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.folder / 'registry.json'
            data = read_json(path) or {'sequence': 0, 'runs': {}}
            data['runs'] = {k: v for k, v in data['runs'].items() if birth(v['pid']) == v['birth']}
            yield data
            write_json(path, data)

    def register(self, key, submitted=None):
        with self.locked() as data:
            data['sequence'] += 1
            data['runs'][key] = {'pid': os.getpid(), 'birth': birth(os.getpid()),
                'sequence': data['sequence'], 'submitted': submitted or time.time_ns(), 'state': 'waiting'}

    def remove(self, key):
        with self.locked() as data:
            data['runs'].pop(key, None)


def retry_command(command, metadata, run_dir, validator=None):
    """Resume latest complete local save, or preserve this attempt's fresh output."""
    output = Path(metadata['output'])
    if validator is None:
        from lerobot.utils.checkpoint_retention import _is_training_checkpoint
        validator = lambda p: _is_training_checkpoint(p, policy_type=None)
    checkpoints = output / 'checkpoints'
    candidates = sorted((p for p in checkpoints.glob('[0-9]*') if p.name.isdigit() and validator(p)),
                        key=lambda p: int(p.name))
    if candidates:
        checkpoint = candidates[-1]
        cfg = metadata['config']
        arguments = ['--resume=true', '--config_path=' + str(checkpoint / 'pretrained_model/train_config.json')]
        for key in ('steps', 'batch_size', 'num_workers', 'save_freq', 'log_freq', 'eval_freq',
                    'checkpoint_archive', 'keep_checkpoints', 'protect_checkpoints',
                    'dataloader_timeout_s', 'dataloader_prefetch_factor', 'dataloader_persistent_workers'):
            if key in cfg:
                value = cfg[key]
                arguments.append('--' + key + '=' + (json.dumps(value) if not isinstance(value, str) else value))
        if cfg['dataset'].get('root') is not None:
            arguments.append('--dataset.root=' + str(cfg['dataset']['root']))
        for key in ('gradient_checkpointing', 'action_expert_checkpointing', 'skill_planner_checkpointing',
                    'vlm_checkpointing', 'vision_checkpointing'):
            if key in cfg['policy']:
                arguments.append('--policy.' + key + '=' + json.dumps(cfg['policy'][key]))
        return metadata['entrypoint'] + arguments
    if output.exists():
        if not metadata.get('owns_fresh_output') or output.is_symlink():
            raise ValueError('No complete checkpoint; existing output is preserved and needs manual recovery: ' + str(output))
        marker = output / '.gpu_guard_owner'
        if marker.is_symlink() or not marker.is_file() or marker.read_text() != str(run_dir):
            raise ValueError('Output ownership marker changed; refusing to move it')
        target = output.with_name(output.name + '.gpu-paused-' + Path(run_dir).name + '-' + uuid.uuid4().hex[:8])
        output.rename(target)  # Never delete artifacts or checkpoints.
        print('GPU memory guard: preserved incomplete fresh output at ' + str(target), flush=True)
    return command


class Guard:
    def __init__(self, command, env, run_dir, probe, settings=None, registry=None, retry=retry_command):
        self.command, self.env, self.folder = list(command), dict(env), Path(run_dir)
        self.probe, self.settings, self.retry = probe, settings or Settings.from_env(), retry
        self.folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        gpu, self.total, _ = probe.sample()
        scope = re.sub(r'[^A-Za-z0-9_-]', '_', env.get('SLURM_JOB_ID', 'local') + '_' + gpu)
        self.registry = registry or Registry(Path(env.get('SBD_GPU_GUARD_REGISTRY_ROOT', f'/tmp/sbd-gpu-guard-{os.getuid()}')) / scope)
        self.key = uuid.uuid4().hex
        self.process = None
        self.cancelled = False
        self.attempt = 0
        self.errors = ''
        self.error_lock = threading.Lock()
        self.child_stopped = True

    def status(self, state, reason='', **values):
        write_json(self.folder / 'status.json', {'state': state, 'reason': reason,
            'attempt': self.attempt, 'updated_at': datetime.now(timezone.utc).isoformat(), **values})
        if state == 'FAILED' and reason:
            print('GPU memory guard: ' + reason, file=sys.stderr, flush=True)

    def attach(self):
        paths = []
        if self.env.get('SBD_TRAINING_STATUS_PATH'):
            paths.append(Path(self.env['SBD_TRAINING_STATUS_PATH']))
        if self.env.get('SBD_TRAINING_EVENTS_DIR') and self.env.get('SBD_TRAINING_PARENT_ID'):
            paths.append(Path(self.env['SBD_TRAINING_EVENTS_DIR']) / (self.env['SBD_TRAINING_PARENT_ID'] + '.json'))
        for path in paths:
            record = read_json(path)
            if record and record.get('state') == 'RUNNING':
                record['resource_status_path'] = str(self.folder / 'status.json')
                write_json(path, record)

    def stop_child(self):
        if self.process is None or self.child_stopped:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=self.settings.grace)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.process.wait()
        self.child_stopped = True

    def drain(self, source, destination):
        while chunk := os.read(source.fileno(), 4096):
            with self.error_lock:
                self.errors = (self.errors + chunk.decode(errors='replace'))[-131072:]
            try:
                destination.buffer.write(chunk)
                destination.buffer.flush()
            except OSError:
                pass

    def run(self):
        self.registry.register(self.key, int(self.env.get('SBD_GPU_GUARD_ORDER_NS', '0')) or None)
        self.attach()
        minimum = self.settings.minimum
        command = self.command
        handlers = {}
        def cancel(*_):
            self.cancelled = True
        if threading.current_thread() is threading.main_thread():
            handlers = {sig: signal.signal(sig, cancel) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            while not self.cancelled:
                try:
                    _, total, free = self.probe.sample()
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    self.status('WAITING_FOR_MEMORY', 'GPU 상태 확인 실패; 실행을 보류합니다: ' + str(error))
                    time.sleep(self.settings.interval)
                    continue
                budget = free - self.settings.reserve
                # Admission is serialized, including trainer initialization. A
                # later process cannot reserve the same not-yet-allocated bytes.
                with self.registry.locked() as data:
                    others = [v for k, v in data['runs'].items() if k != self.key]
                    blocked = any(v['state'] == 'starting' for v in others)
                    if budget >= minimum and not blocked:
                        data['runs'][self.key].update(state='starting', budget=budget)
                    else:
                        data['runs'][self.key]['state'] = 'waiting'
                if budget < minimum or blocked:
                    self.status('WAITING_FOR_MEMORY', '메모리 확보 또는 먼저 시작한 학습의 초기화를 기다립니다.', required_bytes=minimum, available_budget_bytes=max(0, budget))
                    time.sleep(self.settings.interval)
                    continue
                self.attempt += 1
                self.errors = ''
                child_env = {**self.env, 'SBD_GPU_GUARD_CHILD': '1', 'SBD_GPU_GUARD_RUN_DIR': str(self.folder),
                             'SBD_GPU_BUDGET_BYTES': str(budget)}
                self.status('RUNNING', budget_bytes=budget)
                self.process = subprocess.Popen(command, env=child_env, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, start_new_session=True)
                self.child_stopped = False
                threads = [threading.Thread(target=self.drain, args=pair, daemon=True) for pair in
                    ((self.process.stdout, sys.stdout), (self.process.stderr, sys.stderr))]
                for thread in threads:
                    thread.start()
                protected = False
                while self.process.poll() is None and not self.cancelled:
                    metadata = read_json(self.folder / 'trainer.json')
                    if metadata.get('output'):
                        output = Path(metadata['output'])
                        if metadata.get('owns_fresh_output') and output.is_dir():
                            marker = output / '.gpu_guard_owner'
                            if not marker.exists():
                                marker.write_text(str(self.folder))
                    try:
                        _, _, free = self.probe.sample()
                    except (OSError, ValueError, subprocess.SubprocessError):
                        free = None
                    with self.registry.locked() as data:
                        me = data['runs'][self.key]
                        # The trainer reports after its first optimizer update;
                        # initialization usually contains the largest transients.
                        if (self.folder / 'ready').exists():
                            me['state'] = 'running' if me['state'] == 'starting' else me['state']
                        active = [(k, v) for k, v in data['runs'].items() if v['state'] in ('starting', 'running', 'stop_requested')]
                        if free is not None and free < self.settings.stop_below and active and time.monotonic() >= data.get('cooldown_until', 0):
                            if not any(v['state'] == 'stop_requested' for _, v in active):
                                newest, _ = max(active, key=lambda pair: (pair[1]['submitted'], pair[1]['sequence']))
                                data['runs'][newest]['state'] = 'stop_requested'
                        protected = me['state'] == 'stop_requested'
                    if protected:
                        self.stop_child()
                        break
                    time.sleep(self.settings.interval)
                self.stop_child()  # also reap descendants left after a normal exit
                for thread in threads:
                    thread.join(timeout=3)
                code = self.process.returncode
                self.process.stdout.close()
                self.process.stderr.close()
                with self.registry.locked() as data:
                    was_requested = data['runs'][self.key]['state'] == 'stop_requested'
                    data['runs'][self.key]['state'] = 'waiting'
                    if protected or was_requested:
                        data['cooldown_until'] = time.monotonic() + max(3, self.settings.interval * 3)
                if self.cancelled:
                    self.status('CANCELLED', '실행자가 학습 중단을 요청했습니다.')
                    return 143
                if STORAGE.search(self.errors):
                    self.status('FAILED', '저장공간 부족: 자동 재실행 금지')
                    return code or 1
                if not protected and not (code and OOM.search(self.errors)):
                    self.status('COMPLETED' if code == 0 else 'FAILED')
                    return code
                minimum = max(budget + self.settings.retry_margin, int(budget * 1.1))
                if minimum > total - self.settings.reserve:
                    self.status('FAILED', 'GPU 전체 예산으로도 추가 여유 확보 불가; 설정 변경 후 재개가 필요합니다.')
                    return 95
                self.status('WAITING_FOR_MEMORY', '최근 추가한 학습을 메모리 보호를 위해 중단했습니다. 더 큰 예산 확보 후 재개합니다.', required_bytes=minimum)
                print('GPU memory guard: newest trainer paused; allocation and older trainers remain held.', flush=True)
                # A repeated restart needs strictly more headroom; an impossible
                # budget is left waiting instead of an OOM/restart loop.
                metadata = read_json(self.folder / 'trainer.json')
                if not metadata:
                    self.status('FAILED', '학습 설정을 기록하기 전에 중단됨; 생성물을 보존하고 수동 재개가 필요합니다.')
                    return 95
                try:
                    command = self.retry(self.command, metadata, self.folder)
                except (OSError, ValueError) as error:
                    self.status('FAILED', str(error))
                    return 95
                (self.folder / 'ready').unlink(missing_ok=True)
            self.status('CANCELLED', '실행자가 학습 중단을 요청했습니다.')
            return 143
        finally:
            self.stop_child()
            self.registry.remove(self.key)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def supervise_if_needed(cfg):
    if os.environ.get('SBD_GPU_GUARD_CHILD') == '1' or os.environ.get('SBD_GPU_MEMORY_GUARD', '0').lower() not in ('1', 'true'):
        return None
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SBD_SRUN_SESSION') != '1' or getattr(cfg.policy, 'device', 'cuda') == 'cpu':
        return None
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        print('GPU memory guard: multi-process training is not managed.', flush=True)
        return None
    arguments = list(sys.orig_argv[1:])
    command = [sys.executable, *arguments]
    repo = Path(__file__).resolve().parents[4]
    folder = repo / '.cache/gpu_memory_guard' / uuid.uuid4().hex
    return Guard(command, os.environ, folder, Nvidia(os.environ)).run()


def apply_budget():
    if os.environ.get('SBD_GPU_GUARD_CHILD') != '1':
        return
    import torch
    device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory
    fraction = int(os.environ['SBD_GPU_BUDGET_BYTES']) / total
    previous = torch.cuda.memory.get_per_process_memory_fraction(device)
    torch.cuda.set_per_process_memory_fraction(min(previous, fraction, 1.0), device)


def record_config(cfg):
    if os.environ.get('SBD_GPU_GUARD_CHILD') != '1':
        return
    args = list(sys.orig_argv[1:])
    count = len(sys.argv) - 1
    prefix = [sys.executable, *(args[:-count] if count else args)]
    folder = Path(os.environ['SBD_GPU_GUARD_RUN_DIR'])
    write_json(folder / 'trainer.json', {'output': str(Path(cfg.output_dir).absolute()),
        'owns_fresh_output': not cfg.resume and not Path(cfg.output_dir).exists(),
        'entrypoint': prefix, 'config': cfg.to_dict()})


_ready_written = False


def mark_ready():
    global _ready_written
    if not _ready_written and os.environ.get('SBD_GPU_GUARD_CHILD') == '1':
        (Path(os.environ['SBD_GPU_GUARD_RUN_DIR']) / 'ready').touch()
        _ready_written = True
