"""Publish completed saves to a local outbox; network transfers run on the lab host."""
from __future__ import annotations

import json
import hashlib
import logging
import os
from pathlib import Path
import shutil
import time
import uuid


def report_progress(output_dir, step, total, unit='step'):
    outbox = os.environ.get('SBD_YONSEI_CHECKPOINT_OUTBOX')
    if not outbox:
        return
    try:
        output = str(Path(output_dir).absolute())
        folder = Path(outbox).parent / 'training_progress'
        folder.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(output.encode()).hexdigest()[:24]
        temporary = folder / ('.' + key + '.' + uuid.uuid4().hex)
        temporary.write_text(json.dumps({'output': output, 'step': int(step), 'total': int(total),
                                        'unit': unit, 'updated_at': time.time(), 'approximate': False}))
        temporary.replace(folder / (key + '.json'))
    except Exception:
        logging.exception('Could not publish Yonsei training progress')


def publish_checkpoint(path, output_dir, step, protected=None, *, kind='lerobot'):
    outbox = os.environ.get('SBD_YONSEI_CHECKPOINT_OUTBOX')
    if not outbox:
        return
    try:
        selected = (list(protected) if protected else
                    json.loads(os.environ.get('SBD_YONSEI_PROTECTED_CHECKPOINTS', '[]')))
        if selected and step not in selected:
            return
        path, output = Path(path).absolute(), Path(output_dir).absolute()
        project = Path(os.environ['SBD_YONSEI_PROJECT'])
        workspace = Path(os.environ['SBD_YONSEI_WORKSPACE'])
        relative = None
        for base in (project, workspace):
            try:
                relative = output.relative_to(base)
                break
            except ValueError:
                continue
        if relative is None or not relative.parts[0].startswith('outputs'):
            raise ValueError('Checkpoint output is outside the portable output layout: ' + str(output))
        launch = os.environ['SBD_YONSEI_LAUNCH_ID']
        event_id = uuid.uuid4().hex
        # Same filesystem hard links keep serialized saves alive through native
        # retention without duplicating tensor data or blocking on network I/O.
        owner = output / '.yonsei-checkpoint-transfer' / launch / event_id
        owner.mkdir(parents=True, mode=0o700)
        (owner / 'owner.json').write_text(json.dumps({'launch_id': launch, 'event_id': event_id}))
        stage = owner / 'checkpoint'
        if kind == 'lerobot':
            if not (path / 'training_state/training_step.json').is_file():
                raise ValueError('Checkpoint save is incomplete')
            shutil.copytree(path, stage, copy_function=os.link, symlinks=False)
            entry = 'checkpoints/' + path.name
        else:
            stage.mkdir()
            os.link(path, stage / path.name)
            entry = path.name
        event = {'id': event_id, 'launch_id': launch, 'output': relative.as_posix(),
                 'entry': entry, 'step': int(step), 'kind': kind, 'stage': str(stage),
                 'project': str(project), 'workspace': str(workspace),
                 'dataset': os.environ.get('SBD_YONSEI_DATASET_RELATIVE', ''),
                 'protected': selected, 'created_at': time.time()}
        folder = Path(outbox)
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / ('.' + event_id + '.tmp')
        temporary.write_text(json.dumps(event))
        temporary.replace(folder / (event_id + '.json'))
    except Exception:
        logging.exception('Yonsei checkpoint transfer could not be queued; original save is preserved')
