import json
import os
from types import SimpleNamespace

import pytest

from tools.remote import cancel, yonsei


@pytest.mark.parametrize('foreign,wrong_path', [(False, False), (True, False), (False, True)])
def test_cancel_rechecks_owner_and_frozen_submission_before_scancel(tmp_path, monkeypatch, foreign, wrong_path):
    work = tmp_path / '.cache/yonsei_launches/test-one'
    work.mkdir(parents=True)
    (work / 'launch.json').write_text('{"args":[]}')
    (work / 'jobs.jsonl').write_text('{"job_id":"123","tasks":[0,1]}\n')
    commands = []
    def run(args, **kwargs):
        commands.append(args)
        if args[0] == 'scontrol':
            uid = os.getuid() + int(foreign)
            command = '/other/run.sh' if wrong_path else str(work / 'job_status/batch.sh')
            return SimpleNamespace(returncode=0, stdout=f'JobId=124 ArrayJobId=123 UserId=me({uid}) JobState=RUNNING Command={command} WorkDir={work}/project/eval', stderr='')
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(cancel.subprocess, 'run', run)
    if foreign or wrong_path:
        with pytest.raises(ValueError): cancel.cancel(tmp_path, 'test-one')
        assert not any(c[0] == 'scancel' for c in commands)
    else:
        assert cancel.cancel(tmp_path, 'test-one')['job_ids'] == ['123']
        assert commands[-1][-1] == '123'
        assert commands[-1][0] == 'scancel'


def test_old_automatic_snapshot_does_not_override_current_yaml(capsys):
    env = {'STAGE1_EVAL_CONFIG': '/project/.config_snapshots/old/config.yaml',
           'STAGE1_EVAL_JOB_NAME': 'old5k', 'TRAIN_DATA': 'libero_10'}
    selected = yonsei.selectors(env)
    assert selected == {'TRAIN_DATA': 'libero_10'}
    assert 'Ignoring inherited automatic snapshot' in capsys.readouterr().err


def test_existing_allocation_cannot_be_cancelled_from_launch_card(tmp_path, monkeypatch):
    work = tmp_path / '.cache/yonsei_launches/test-one'
    work.mkdir(parents=True)
    (work / 'launch.json').write_text('{"args":["--123"]}')
    monkeypatch.setattr(cancel.subprocess, 'run', lambda *a, **k: pytest.fail('must not contact scheduler'))
    with pytest.raises(ValueError, match='기존 할당'):
        cancel.cancel(tmp_path, 'test-one')
