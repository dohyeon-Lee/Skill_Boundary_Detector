"""Exercise protection using real CPU subprocesses and fake GPU telemetry only."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[2] / 'src/lerobot/utils/gpu_memory_guard.py'
SPEC = importlib.util.spec_from_file_location('memory_guard_test_module', SOURCE)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def wait(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, 'timed out'
        time.sleep(.01)


class Probe:
    free = 1000
    def sample(self):
        return 'GPU-test', 1000, self.free


def settings():
    return M.Settings(reserve=100, stop_below=50, minimum=100, retry_margin=40, interval=.01, grace=.1)


def test_default_budget_is_valid_without_environment_overrides(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith('SBD_GPU_GUARD_'):
            monkeypatch.delenv(key)
    result = M.Settings.from_env()
    assert result.reserve == result.minimum == result.retry_margin == int(0.3 * M.GIB)
    assert result.stop_below == int(0.15 * M.GIB)
    assert result.interval == 0.5


def child_script(tmp_path, name, body):
    p = tmp_path / (name + '.py')
    p.write_text('''import json,os,time,sys
from pathlib import Path
folder=Path(os.environ['SBD_GPU_GUARD_RUN_DIR'])
(folder/'trainer.json').write_text(json.dumps({'output':str(folder/'output')}))
(folder/'ready').touch()
''' + body)
    return [sys.executable, str(p)]


def test_memory_pressure_pauses_newest_only_and_waits_for_larger_budget(tmp_path):
    probe = Probe()
    registry = M.Registry(tmp_path / 'registry')
    old_cmd = child_script(tmp_path, 'old', f"while not Path({str(tmp_path/'finish_old')!r}).exists(): time.sleep(.01)\n")
    new_cmd = child_script(tmp_path, 'new', '''counter=folder/'attempts'
n=int(counter.read_text())+1 if counter.exists() else 1
counter.write_text(str(n))
if n==1:
 import subprocess
 descendant=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
 (folder/'descendant.pid').write_text(str(descendant.pid))
 while True: time.sleep(.01)
''')
    retry_calls = []
    def retry(command, *_):
        retry_calls.append(True)
        return command
    old = M.Guard(old_cmd, {**os.environ, 'SBD_GPU_GUARD_ORDER_NS': '1'}, tmp_path/'old_run', probe, settings(), registry, retry)
    new = M.Guard(new_cmd, {**os.environ, 'SBD_GPU_GUARD_ORDER_NS': '2'}, tmp_path/'new_run', probe, settings(), registry, retry)
    results = {}
    a = threading.Thread(target=lambda: results.update(old=old.run()))
    b = threading.Thread(target=lambda: results.update(new=new.run()))
    a.start()
    try:
        wait(lambda: (old.folder/'ready').exists())
        probe.free = 900  # older trainer already occupies part of the GPU
        b.start()
        wait(lambda: (new.folder/'descendant.pid').exists())
        probe.free = 30
        wait(lambda: M.read_json(new.folder/'status.json').get('state') == 'WAITING_FOR_MEMORY')
        probe.free = 900  # original budget isn't enough to retry
        wait(lambda: len(retry_calls) == 1)
        time.sleep(.05)
        assert old.process.poll() is None
        assert M.birth(int((new.folder/'descendant.pid').read_text())) is None
        assert (new.folder/'attempts').read_text() == '1'
        (tmp_path/'finish_old').touch()
        wait(lambda: 'old' in results)
        probe.free = 1000
        wait(lambda: 'new' in results)
        assert results == {'old': 0, 'new': 0}
        assert (new.folder/'attempts').read_text() == '2'
        assert new.attempt == 2
    finally:
        old.cancelled = new.cancelled = True
        a.join(timeout=3)
        if b.ident:
            b.join(timeout=3)


def test_oom_can_finish_only_this_run_for_completion_handler_recovery(tmp_path):
    command = child_script(tmp_path, 'oom_once', "print('CUDA out of memory',file=sys.stderr);sys.exit(1)\n")
    guard = M.Guard(command, {**os.environ, 'SBD_GPU_GUARD_TERMINAL_ON_OOM': '1'},
                    tmp_path/'run', Probe(), settings(), M.Registry(tmp_path/'registry'),
                    retry=lambda *args: pytest.fail('Completion handler owns this retry'))
    assert guard.run() == 95
    status = M.read_json(guard.folder/'status.json')
    assert status['state'] == 'FAILED' and status['memory_failure'] is True
    assert guard.attempt == 1


def test_storage_failure_never_retries_even_with_oom_in_the_log(tmp_path):
    command = child_script(tmp_path, 'disk_full', "print('CUDA out of memory; No space left on device',file=sys.stderr);sys.exit(1)\n")
    guard = M.Guard(command, os.environ, tmp_path/'run', Probe(), settings(), M.Registry(tmp_path/'registry'),
                    lambda *_: pytest.fail('Storage failures must never retry'))
    assert guard.run() == 1
    assert guard.attempt == 1
    assert '저장공간 부족' in M.read_json(guard.folder/'status.json')['reason']


def test_allocator_oom_waits_until_budget_increases(tmp_path):
    probe = Probe()
    probe.free = 800
    command = child_script(tmp_path, 'oom', '''counter=folder/'attempts'
n=int(counter.read_text())+1 if counter.exists() else 1
counter.write_text(str(n))
if n==1:
 print('torch.OutOfMemoryError: CUDA out of memory',file=sys.stderr)
 sys.exit(1)
''')
    guard = M.Guard(command, os.environ, tmp_path/'run', probe, settings(), M.Registry(tmp_path/'registry'),lambda command,*_:command)
    result = []
    thread = threading.Thread(target=lambda: result.append(guard.run()))
    thread.start()
    try:
        wait(lambda: M.read_json(guard.folder/'status.json').get('state')=='WAITING_FOR_MEMORY')
        time.sleep(.05)
        assert guard.attempt == 1
        probe.free = 1000
        wait(lambda: bool(result))
        assert result == [0] and guard.attempt == 2
    finally:
        guard.cancelled = True
        thread.join(timeout=3)


def test_normal_failure_does_not_loop_or_submit_jobs(tmp_path):
    command = child_script(tmp_path, 'bad', 'sys.exit(7)\n')
    guard = M.Guard(command, os.environ, tmp_path/'run', Probe(), settings(), M.Registry(tmp_path/'registry'))
    assert guard.run() == 7
    assert guard.attempt == 1


def test_retry_uses_checkpoint_and_keeps_current_runtime_settings(tmp_path):
    output = tmp_path/'model'
    cp = output/'checkpoints/030000'
    cp.mkdir(parents=True)
    metadata = {'output': str(output), 'entrypoint': ['python', '-m', 'trainer'],
                'config': {'steps': 100000, 'batch_size': 16, 'num_workers': 8,
                           'protect_checkpoints': [50000,100000], 'dataset': {'root': '/new/stage'},
                           'policy': {'gradient_checkpointing': False}}}
    command = M.retry_command(['original'], metadata, tmp_path/'run', validator=lambda _: True)
    assert command[:3] == ['python', '-m', 'trainer']
    assert '--resume=true' in command and '--steps=100000' in command
    assert '--dataset.root=/new/stage' in command
    assert '--policy.gradient_checkpointing=false' in command
    assert '--config_path='+str(cp/'pretrained_model/train_config.json') in command
    assert cp.exists()


@pytest.mark.parametrize('enabled', [False, True])
def test_retry_preserves_current_oversampling_opt_out_and_settings(tmp_path, enabled):
    output = tmp_path/'model'
    cp = output/'checkpoints/030000'
    cp.mkdir(parents=True)
    config = {'dataset': {'root': '/dataset'}, 'policy': {},
              'transition_oversampling': {'enabled': enabled, 'percent': 25, 'distribution': 'uniform', 'weight': 2.0}}
    metadata = {'output': str(output), 'entrypoint': ['python', '-m', 'trainer'], 'config': config}
    command = M.retry_command([], metadata, tmp_path/'run', validator=lambda _: True)
    assert '--transition_oversampling.enabled='+str(enabled).lower() in command
    assert '--transition_oversampling.percent=25' in command
    assert '--transition_oversampling.distribution=uniform' in command
    assert '--transition_oversampling.weight=2.0' in command


def test_fresh_output_is_archived_only_with_matching_ownership_marker(tmp_path):
    output = tmp_path/'model'
    output.mkdir()
    (output/'keep_me').write_text('preserved')
    run = tmp_path/'run'
    metadata = {'output': str(output), 'owns_fresh_output': True}
    with pytest.raises(ValueError, match='ownership'):
        M.retry_command(['fresh'], metadata, run, validator=lambda _: False)
    (output/'.gpu_guard_owner').write_text(str(run))
    assert M.retry_command(['fresh'], metadata, run, validator=lambda _: False) == ['fresh']
    assert not output.exists()
    archives = list(tmp_path.glob('model.gpu-paused-*'))
    assert len(archives) == 1 and (archives[0]/'keep_me').read_text() == 'preserved'


@pytest.mark.parametrize('name', ['m' * 230, '가' * 80])
def test_fresh_retry_preserves_long_run_names_within_filesystem_limit(tmp_path, name):
    output = tmp_path / name
    output.mkdir()
    (output / 'keep_me').write_text('preserved')
    run = tmp_path / ('a' * 32)
    (output / '.gpu_guard_owner').write_text(str(run))
    metadata = {'output': str(output), 'owns_fresh_output': True}
    assert M.retry_command(['fresh'], metadata, run, validator=lambda _: False) == ['fresh']
    assert not output.exists()
    preserved = list(tmp_path.iterdir())
    assert len(preserved) == 1
    assert len(preserved[0].name.encode()) <= os.pathconf(tmp_path, 'PC_NAME_MAX')
    assert (preserved[0] / 'keep_me').read_text() == 'preserved'


def test_budget_preserves_a_stricter_existing_allocator_limit(tmp_path, monkeypatch):
    calls = []
    cuda = SimpleNamespace(current_device=lambda: 0, get_device_properties=lambda _: SimpleNamespace(total_memory=1000),
                           memory=SimpleNamespace(get_per_process_memory_fraction=lambda _: .5),
                           set_per_process_memory_fraction=lambda value, device: calls.append((value,device)))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    monkeypatch.setenv('SBD_GPU_GUARD_CHILD','1')
    monkeypatch.setenv('SBD_GPU_BUDGET_BYTES','800')
    M.apply_budget()
    assert calls == [(.5,0)]


def test_bad_or_ambiguous_gpu_telemetry_does_not_pick_another_device(monkeypatch):
    with pytest.raises(ValueError):
        M.Nvidia({'CUDA_VISIBLE_DEVICES':'0,1'})
    with pytest.raises(ValueError):
        M.Nvidia({'CUDA_VISIBLE_DEVICES':''})
    assert M.Nvidia({'CUDA_VISIBLE_DEVICES':'0','SLURM_JOB_GPUS':'6'}).selector == '6'


def test_record_config_preserves_original_module_entrypoint_for_resume(tmp_path, monkeypatch):
    monkeypatch.setenv('SBD_GPU_GUARD_CHILD','1')
    monkeypatch.setenv('SBD_GPU_GUARD_RUN_DIR',str(tmp_path/'guard'))
    monkeypatch.setattr(sys,'orig_argv',[sys.executable,'-m','trainer','--steps=200000'])
    monkeypatch.setattr(sys,'argv',['module.py','--steps=200000'])
    cfg=SimpleNamespace(output_dir=tmp_path/'output',resume=False,to_dict=lambda:{'steps':200000})
    M.record_config(cfg)
    saved=M.read_json(tmp_path/'guard/trainer.json')
    assert saved['entrypoint']==[sys.executable,'-m','trainer']
    assert saved['owns_fresh_output'] and saved['config']['steps']==200000
