import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest
import torch
from safetensors.torch import save_file

from lerobot.utils.yonsei_checkpoint_sync import publish_checkpoint
from tools.remote import checkpoints, desk, probe


def setup(tmp_path, monkeypatch):
    base = tmp_path / 'remote'
    work = base / '.cache/yonsei_launches/test-training'
    work.mkdir(parents=True)
    root = tmp_path / 'lab'
    receipt = root / '.cache/yonsei_launches/test-training'
    receipt.mkdir(parents=True)
    manifest = {'host': 'fake', 'profile': {'project': str(base)}, 'project': str(work / 'project'),
                'training': {'checkpoint_sync': True, 'protected': []}, 'args': [],
                'script': 'submit_train.sh', 'created_at': time.time()}
    (receipt / 'launch.json').write_text(json.dumps(manifest))
    (receipt / 'remote.json').write_text('{}')
    (work / 'launch.json').write_text(json.dumps(manifest))
    monkeypatch.setenv('SBD_YONSEI_CHECKPOINT_OUTBOX', str(work / 'checkpoint_events'))
    monkeypatch.setenv('SBD_YONSEI_LAUNCH_ID', work.name)
    monkeypatch.setenv('SBD_YONSEI_PROJECT', str(work / 'project'))
    monkeypatch.setenv('SBD_YONSEI_WORKSPACE', str(base))
    monkeypatch.setenv('SBD_YONSEI_PROTECTED_CHECKPOINTS', '[]')
    monkeypatch.setenv('SBD_YONSEI_DATASET_RELATIVE', 'dataset_filtered/train')
    return root, base, work, receipt, manifest


def saved(run, step):
    path = run / 'checkpoints' / f'{step:06d}'
    pre, state = path / 'pretrained_model', path / 'training_state'
    pre.mkdir(parents=True)
    state.mkdir()
    for file, value in [(pre/'config.json', {'type': 'diffusion'}),
                        (pre/'train_config.json', {'scheduler': {}, 'output_dir': str(run), 'dataset': {'root': '/tmp/staged-data'}}),
                        (pre/'policy_preprocessor.json', {'steps': []}),
                        (pre/'policy_postprocessor.json', {'steps': []}),
                        (state/'training_step.json', {'step': step}),
                        (state/'optimizer_param_groups.json', []), (state/'scheduler_state.json', {})]:
        file.write_text(json.dumps(value))
    for file in [pre/'model.safetensors', state/'optimizer_state.safetensors', state/'rng_state.safetensors']:
        save_file({'weight': torch.ones(1)}, str(file))
    return path


def local_transport(monkeypatch):
    actual = subprocess.run
    def run(args, **kwargs):
        args = args[:]
        if args[0] == 'rsync':
            i = args.index('-e')
            del args[i:i+2]
            args[-2] = args[-2].removeprefix('fake:')
        return actual(args, **kwargs)
    monkeypatch.setattr(checkpoints.subprocess, 'run', run)


@pytest.mark.parametrize('protected,expected', [([], [1000, 50000, 100000]), ([50000, 100000], [50000, 100000])])
def test_only_complete_selected_saves_are_published_and_survive_retention(tmp_path, monkeypatch, protected, expected):
    root, base, work, receipt, manifest = setup(tmp_path, monkeypatch)
    run = base / 'outputs_filtered/DP/model'
    for step in [1000, 50000, 100000]:
        path = saved(run, step)
        publish_checkpoint(path, run, step, protected)
        shutil.rmtree(path)
    events = [json.loads(p.read_text()) for p in (work/'checkpoint_events').glob('*.json')]
    assert sorted(e['step'] for e in events) == expected
    for event in events:
        assert (Path(event['stage'])/'pretrained_model/model.safetensors').is_file()


def test_copy_checksum_resume_state_and_ack_cleanup_keep_original(tmp_path, monkeypatch):
    root, base, work, receipt, manifest = setup(tmp_path, monkeypatch)
    run = base / 'outputs_filtered/DP/model'
    original = saved(run, 50000)
    publish_checkpoint(original, run, 50000, [50000, 100000])
    event = json.loads(next((work/'checkpoint_events').glob('*.json')).read_text())
    local_transport(monkeypatch)
    target = Path(checkpoints.collect_one(root, receipt, manifest, event))
    assert target == root/'outputs_filtered/DP/model_yonsei/checkpoints/050000'
    assert (target/'training_state/optimizer_state.safetensors').is_file()
    assert (target.parent/'last').resolve() == target
    config = json.loads((target/'pretrained_model/train_config.json').read_text())
    assert config['output_dir'] == str(target.parent.parent)
    assert config['dataset']['root'] == str(root/'dataset_filtered/train')
    assert checkpoints.collect_one(root, receipt, manifest, event) == str(target)
    result = subprocess.run([sys.executable, '-c', checkpoints.ACK], input=json.dumps({
        'base': str(base), 'launch': receipt.name, 'event': event['id']}), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert original.is_dir() and not Path(event['stage']).exists()
    assert (work/'checkpoint_acks'/event['id']).exists()


def test_fsq_periodic_save_and_existing_local_run_are_preserved(tmp_path, monkeypatch):
    root, base, work, receipt, manifest = setup(tmp_path, monkeypatch)
    run = base/'outputs_filtered/FSQ/model'
    run.mkdir(parents=True)
    path = run/'FSQ_epoch0050.pt'
    path.write_bytes(b'completed-fsq-checkpoint')
    monkeypatch.setenv('SBD_YONSEI_PROTECTED_CHECKPOINTS', '[50]')
    publish_checkpoint(path, run, 50, kind='fsq')
    event = json.loads(next((work/'checkpoint_events').glob('*.json')).read_text())
    existing = root/'outputs_filtered/FSQ/model_yonsei'
    existing.mkdir(parents=True)
    (existing/'keep.txt').write_text('other run')
    local_transport(monkeypatch)
    target = Path(checkpoints.collect_one(root, receipt, manifest, event))
    assert target.parent != existing and target.parent.name.endswith('_yonsei')
    assert target.read_bytes() == path.read_bytes()
    assert (existing/'keep.txt').read_text() == 'other run'


def test_partial_save_is_not_announced_and_non_yonsei_save_is_unchanged(tmp_path, monkeypatch):
    root, base, work, receipt, manifest = setup(tmp_path, monkeypatch)
    run = base/'outputs_filtered/DP/model'
    partial = run/'checkpoints/050000'
    partial.mkdir(parents=True)
    publish_checkpoint(partial, run, 50000)
    assert not list((work/'checkpoint_events').glob('*.json'))
    monkeypatch.delenv('SBD_YONSEI_CHECKPOINT_OUTBOX')
    publish_checkpoint(partial, run, 50000)
    assert not list((work/'checkpoint_events').glob('*.json'))


def test_running_training_is_collected_before_job_completion(tmp_path, monkeypatch):
    root, base, work, receipt, manifest = setup(tmp_path, monkeypatch)
    run = base/'outputs_filtered/DP/model'
    publish_checkpoint(saved(run, 1000), run, 1000)
    event = json.loads(next((work/'checkpoint_events').glob('*.json')).read_text())
    calls = []
    monkeypatch.setattr(checkpoints, 'collect_one', lambda *a: calls.append(a[-1]['step']) or '/copied/001000')
    monkeypatch.setattr(checkpoints, 'acknowledge', lambda *a: None)
    row = {'id': receipt.name, 'state': 'RUNNING', 'checkpoints': [event]}
    worker = desk.Worker(root, probe=lambda *a: [row])
    worker.poll()
    for _ in range(100):
        if not worker.inflight:
            break
        time.sleep(.01)
    assert calls == [1000]
    status = json.loads((receipt/'checkpoint_sync.json').read_text())
    assert len(status['completed']) == 1 and status['state'] == 'WATCHING'
