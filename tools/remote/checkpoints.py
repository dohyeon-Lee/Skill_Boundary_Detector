"""Verified, incremental collection of sealed training checkpoints."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time


ACK = r'''
import json, shutil, sys
from pathlib import Path
r=json.load(sys.stdin)
work=Path(r['base'])/'.cache/yonsei_launches'/r['launch']
event=json.loads((work/'checkpoint_events'/(r['event']+'.json')).read_text())
if event['id']!=r['event'] or event['launch_id']!=r['launch']:raise ValueError('Checkpoint identity mismatch')
owner=Path(event['stage']).parent
if owner.name!=r['event'] or owner.parent.name!=r['launch'] or owner.parent.parent.name!='.yonsei-checkpoint-transfer':
    raise ValueError('Refusing checkpoint cleanup outside owned staging folder')
if owner.exists():
    if owner.is_symlink():raise ValueError('Invalid staging symlink')
    marker=json.loads((owner/'owner.json').read_text())
    if marker!={'event_id':r['event'],'launch_id':r['launch']}:raise ValueError('Staging owner mismatch')
    shutil.rmtree(owner)
folder=work/'checkpoint_acks';folder.mkdir(exist_ok=True)
(folder/r['event']).write_text('collected')
print('acknowledged')
'''


def load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def validate(event, receipt):
    if event.get('launch_id') != receipt.name or not re.fullmatch(r'[a-f0-9]{32}', event.get('id', '')):
        raise ValueError('Checkpoint launch identity mismatch')
    output = Path(event['output'])
    if output.is_absolute() or '..' in output.parts or len(output.parts) < 2 or not output.parts[0].startswith('outputs'):
        raise ValueError('Invalid training output path')
    entry = event['entry']
    if not (event['kind'] == 'lerobot' and re.fullmatch(r'checkpoints/[0-9]+', entry) or
            event['kind'] == 'fsq' and re.fullmatch(r'FSQ_epoch[0-9]+\.pt', entry)):
        raise ValueError('Invalid checkpoint entry')
    if type(event.get('step')) is not int or event['step'] < 0:
        raise ValueError('Invalid checkpoint step')
    if event['kind'] == 'lerobot' and int(Path(entry).name) != event['step']:
        raise ValueError('Checkpoint step mismatch')
    stage = Path(event['stage'])
    if (stage.name != 'checkpoint' or stage.parent.name != event['id'] or
            stage.parent.parent.name != receipt.name or
            stage.parent.parent.parent.name != '.yonsei-checkpoint-transfer'):
        raise ValueError('Invalid sealed checkpoint staging path')


def destination(root, receipt, event):
    relative = Path(event['output'])
    parent = root / relative.parent
    parent.resolve().relative_to((root / relative.parts[0]).resolve())
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / '.yonsei-checkpoint-sync.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        base = relative.name.removesuffix('_yonsei')
        target = parent / (base + '_yonsei')
        marker = 'yonsei_training_origin.json'
        if target.exists() and load(target / marker).get('launch_id') != receipt.name:
            suffix = '_' + receipt.name + '_yonsei'
            prefix = base.encode()[:255-len(suffix.encode())].decode('utf-8', errors='ignore')
            target = parent / (prefix + suffix)
        if target.exists() and load(target / marker).get('launch_id') != receipt.name:
            raise ValueError('Training result destination belongs to another launch')
        target.mkdir(exist_ok=True)
        (target / marker).write_text(json.dumps({'launch_id': receipt.name, 'remote_output': event['output']}))
    return target


def collect_one(root, receipt, manifest, event):
    from tools.remote.yonsei import SSH_OPTIONS
    from tools.remote.desk import repair_paths
    validate(event, receipt)
    run = destination(root, receipt, event)
    target = run / event['entry']
    marker = (target / 'yonsei_checkpoint.json' if event['kind'] == 'lerobot' else
              run / (target.name + '.yonsei.json'))
    if target.exists() and load(marker).get('event_id') == event['id']:
        return str(target)
    if target.exists():
        raise FileExistsError('Existing checkpoint is preserved: ' + str(target))
    container = run / ('.yonsei-' + event['id'] + '.partial')
    stage = container / target.name if event['kind'] == 'lerobot' else container
    stage.mkdir(parents=True, exist_ok=True)
    transport = 'ssh ' + shlex.join(SSH_OPTIONS)
    common = ['rsync', '-rlt', '--safe-links', '--protect-args', '--timeout=120', '-e', transport]
    source = manifest['host'] + ':' + event['stage'].rstrip('/') + '/'
    with (receipt / 'checkpoint_collection.log').open('ab', buffering=0) as log:
        subprocess.run([*common, '--partial-dir=.yonsei-partial', source, str(stage) + '/'],
                       stdout=log, stderr=log, check=True, timeout=24*3600)
        result = subprocess.run([*common, '--checksum', '--dry-run', '--itemize-changes', source, str(stage) + '/'],
                                capture_output=True, text=True, check=True, timeout=24*3600)
    if result.stdout.strip():
        raise ValueError('Checkpoint checksum verification failed; retrying later')
    if event['kind'] == 'lerobot':
        from lerobot.utils.checkpoint_retention import _is_training_checkpoint
        # Validator checks the numeric step directory name as well as tensor headers.
        candidate = stage
        if not _is_training_checkpoint(candidate, policy_type=None):
            raise ValueError('Transferred checkpoint is incomplete; not published')
        for prefix in (event['project'], event['workspace']):
            repair_paths(candidate, str(Path(prefix) / event['output']), run)
            repair_paths(candidate, prefix, root)
        if event.get('dataset'):
            dataset = Path(event['dataset'])
            if dataset.is_absolute() or '..' in dataset.parts or not dataset.parts[0].startswith('dataset'):
                raise ValueError('Invalid original training dataset path')
            config_path = candidate / 'pretrained_model/train_config.json'
            config = json.loads(config_path.read_text())
            config['dataset']['root'] = str(root / dataset)
            config_path.write_text(json.dumps(config, indent=2))
        (candidate / 'yonsei_checkpoint.json').write_text(json.dumps({'event_id': event['id'], 'step': event['step']}))
        target.parent.mkdir(parents=True, exist_ok=True)
        candidate.rename(target)
        container.rmdir()
        last = target.parent / 'last'
        current = int(last.resolve().name) if last.is_symlink() and last.resolve().name.isdigit() else -1
        if event['step'] >= current:
            temporary = target.parent / ('.last-' + event['id'])
            temporary.symlink_to(target.name)
            temporary.replace(last)
    else:
        marker.write_text(json.dumps({'event_id': event['id'], 'step': event['step']}))
        (stage / target.name).rename(target)
        stage.rmdir()
    return str(target)


def acknowledge(manifest, receipt, event):
    from tools.remote.yonsei import SSH_OPTIONS
    result = subprocess.run(['ssh', *SSH_OPTIONS, manifest['host'], 'python3 -c ' + shlex.quote(ACK)],
        input=json.dumps({'base': manifest['profile']['project'], 'launch': receipt.name, 'event': event['id']}),
        capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError((result.stderr.strip().splitlines() or ['Checkpoint acknowledgement failed'])[-1][:500])
