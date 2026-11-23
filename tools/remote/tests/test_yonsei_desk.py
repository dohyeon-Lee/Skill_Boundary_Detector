import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tools.remote import desk, probe, yonsei


def fixture(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    base = tmp_path / 'remote'
    identifier = '20261122-test'
    work = base / '.cache/yonsei_launches' / identifier
    work.mkdir(parents=True)
    output = 'lerobot/skillVLA/libero/configs/train_skillVLA/FT/eval/outputs/run'
    spec = {'output': output, 'tasks': [0, 1], 'labels': ['model'], 'episodes': 2}
    manifest = {'source': str(root), 'host': 'fake', 'project': str(work / 'project'),
                'script': 'eval/submit_eval.sh', 'evaluation': spec, 'created_at': time.time(),
                'args': [], 'profile': {'project': str(base)}}
    (work / 'launch.json').write_text(json.dumps(manifest))
    receipt = root / '.cache/yonsei_launches' / identifier
    receipt.mkdir(parents=True)
    (receipt / 'launch.json').write_text(json.dumps(manifest))
    (receipt / 'remote.json').write_text('{}')
    return root, base, work, receipt, manifest


def test_submission_success_is_not_evaluation_completion_and_missing_accounting_waits(tmp_path, monkeypatch):
    root, base, work, receipt, manifest = fixture(tmp_path)
    (work / 'jobs.jsonl').write_text('{"job_id":"123"}\n')
    (work / 'status.json').write_text('{"state":"COMPLETED","exit_code":0}')
    monkeypatch.setattr(probe, 'scheduler', lambda _: ({}, {'123'}))
    assert probe.snapshot(base, [receipt.name])[0]['state'] == 'RUNNING'
    monkeypatch.setattr(probe, 'scheduler', lambda _: ({}, set()))
    assert probe.snapshot(base, [receipt.name])[0]['state'] == 'ACCOUNTING'
    rows = {'123': [{'id': '123', 'state': 'COMPLETED', 'exit_code': '0:0'}]}
    monkeypatch.setattr(probe, 'scheduler', lambda _: (rows, set()))
    assert probe.snapshot(base, [receipt.name])[0]['state'] == 'COMPLETED'
    rows['123'].append({'id': '123_1', 'state': 'OUT_OF_MEMORY', 'exit_code': '1:0'})
    assert probe.snapshot(base, [receipt.name])[0]['state'] == 'FAILED'


def test_scheduler_tracks_array_master_ids_instead_of_raw_allocation_ids(monkeypatch):
    from types import SimpleNamespace
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        if command[0] == 'squeue':
            return SimpleNamespace(returncode=0, stdout='123_[2-3]|PENDING\n', stderr='')
        return SimpleNamespace(stdout='123_0|COMPLETED|0:0\n123_1|OUT_OF_MEMORY|1:0\n', returncode=0)
    monkeypatch.setattr(probe.subprocess, 'run', run)
    rows, active = probe.scheduler({'123'})
    assert active == {'123'}
    assert '--format=JobID,State,ExitCode' in commands[1]
    assert {r['state'] for r in rows['123']} == {'COMPLETED', 'OUT_OF_MEMORY', 'PENDING'}


def test_scheduler_uses_controller_when_accounting_database_is_unavailable(monkeypatch):
    from types import SimpleNamespace
    def run(command, **kwargs):
        if command[0] == 'squeue':
            return SimpleNamespace(returncode=0, stdout='', stderr='')
        if command[0] == 'sacct':
            return SimpleNamespace(returncode=1, stdout='', stderr='Connection refused')
        return SimpleNamespace(returncode=0, stdout='JobId=124 ArrayJobId=123 ArrayTaskId=0 JobState=FAILED ExitCode=1:0\n')
    monkeypatch.setattr(probe.subprocess, 'run', run)
    rows, active = probe.scheduler({'123'})
    assert not active and rows['123'] == [{'id':'123_0','state':'FAILED','exit_code':'1:0'}]


def test_durable_array_exits_survive_controller_retention_and_require_every_element(tmp_path, monkeypatch):
    root, base, work, receipt, manifest = fixture(tmp_path)
    (work/'status.json').write_text('{"state":"COMPLETED","exit_code":0}')
    (work/'jobs.jsonl').write_text('{"job_id":"123","tasks":[0,1]}\n')
    folder=work/'job_status';folder.mkdir()
    (folder/'123_0.exit').write_text('0')
    monkeypatch.setattr(probe,'scheduler',lambda _:({},set()))
    assert probe.snapshot(base,[receipt.name])[0]['state']=='ACCOUNTING'
    (folder/'123_1.exit').write_text('1')
    assert probe.snapshot(base,[receipt.name])[0]['state']=='FAILED'
    (folder/'123_1.exit').write_text('0')
    assert probe.snapshot(base,[receipt.name])[0]['state']=='COMPLETED'


def test_existing_allocation_requires_its_own_inner_worker_completion(tmp_path, monkeypatch):
    root, base, work, receipt, manifest = fixture(tmp_path)
    manifest['args'] = ['--74311']
    (work / 'launch.json').write_text(json.dumps(manifest))
    (work / 'status.json').write_text('{"state":"COMPLETED","exit_code":0}')
    monkeypatch.setattr(probe, 'scheduler', lambda _: ({}, set()))
    assert probe.snapshot(base, [receipt.name])[0]['state'] == 'RUNNING'
    status = work / 'project/.cache/srun_sessions/one/training_status.json'
    status.parent.mkdir(parents=True)
    status.write_text('{"state":"RUNNING"}')
    assert probe.snapshot(base, [receipt.name])[0]['state'] == 'RUNNING'
    status.write_text('{"state":"COMPLETED"}')
    assert probe.snapshot(base, [receipt.name])[0]['state'] == 'COMPLETED'


@pytest.mark.parametrize('shared', [False, True])
def test_partial_episode_progress_and_final_metrics_include_cached_tasks(tmp_path, shared):
    root, base, work, receipt, manifest = fixture(tmp_path)
    project = work / 'project'
    if shared:
        manifest['evaluation']['storage_project'] = str(base)
    out = (base if shared else project) / manifest['evaluation']['output']
    record = out / 'panels/00_model/videos/libero_90_0/eval_progress.json'
    record.parent.mkdir(parents=True)
    record.write_text('{"successes":[true,false],"total_episodes":2}')
    p = probe.progress(project, manifest['evaluation'])
    assert (p['completed'], p['total'], p['percent'], p['success_pct']) == (2, 4, 50, 50)
    metric = out / 'metrics/eval_info_merged.json'
    metric.parent.mkdir()
    metric.write_text(json.dumps({'model': {'per_task': [
        {'task_id': 1, 'metrics': {'successes': [True, True]}}]}}))
    p = probe.progress(project, manifest['evaluation'])
    assert (p['completed'], p['successes'], p['success_pct']) == (4, 3, 75)


def test_sbatch_recorder_handles_arrays_parsable_output_and_errors(tmp_path):
    real = tmp_path / 'actual'
    wrapper = tmp_path / 'sbatch'
    wrapper.write_text(yonsei.SBATCH_RECORDER)
    record = tmp_path / 'jobs.jsonl'
    import os
    env = {**os.environ, 'SBD_YONSEI_REAL_SBATCH': str(real), 'SBD_YONSEI_JOB_FILE': str(record)}
    for output in ('Submitted batch job 123', '124;cluster'):
        real.write_text('#!/bin/sh\nprintf "%s\\n" "' + output + '"\n')
        real.chmod(0o755)
        result = subprocess.run([sys.executable, str(wrapper), '--array=0-3'], env=env, capture_output=True, text=True)
        assert result.returncode == 0 and result.stdout.strip() == output
    assert [json.loads(line)['job_id'] for line in record.read_text().splitlines()] == ['123', '124']
    real.write_text('#!/bin/sh\nexit 1\n')
    assert subprocess.run([sys.executable, str(wrapper)], env=env).returncode == 1
    assert len(record.read_text().splitlines()) == 2


def test_sbatch_wrapper_records_worker_exit_and_keeps_original_directives(tmp_path):
    import os
    real, wrapper, original = tmp_path/'actual', tmp_path/'sbatch', tmp_path/'job.sh'
    marker = tmp_path/'generated-wrapper-path'
    real.write_text('#!/bin/bash\nprintf "%s" "${@: -1}" > '+str(marker)+'\necho "Submitted batch job 123"\n')
    real.chmod(0o755)
    wrapper.write_text(yonsei.SBATCH_RECORDER)
    original.write_text('#!/bin/bash\n#SBATCH --output=logs/%x_%j.out\nexit 7\n')
    env={**os.environ,'SBD_YONSEI_REAL_SBATCH':str(real),
         'SBD_YONSEI_JOB_FILE':str(tmp_path/'jobs.jsonl'),
         'SBD_YONSEI_JOB_STATUS_DIR':str(tmp_path/'status')}
    subprocess.run([sys.executable,str(wrapper),'--array=0-1%2',str(original)],env=env,check=True)
    wrapped=Path(marker.read_text())
    assert '#SBATCH --output=logs/%x_%j.out' in wrapped.read_text()
    result=subprocess.run(['bash',str(wrapped)],env={**env,'SLURM_ARRAY_JOB_ID':'123',
                          'SLURM_ARRAY_TASK_ID':'1','SLURM_JOB_ID':'124'})
    assert result.returncode==7
    assert (tmp_path/'status/123_1.exit').read_text().strip()=='7'
    assert json.loads((tmp_path/'jobs.jsonl').read_text())['tasks']==[0,1]


@pytest.mark.parametrize('shared', [False, True])
def test_collection_copies_video_repairs_paths_suffixes_and_preserves_collision(tmp_path, monkeypatch, shared):
    root, base, work, receipt, manifest = fixture(tmp_path)
    if shared:
        manifest['evaluation']['storage_project'] = str(base)
    out = Path(base if shared else manifest['project']) / manifest['evaluation']['output']
    out.mkdir(parents=True)
    (out / 'index.html').write_text('<video src="' + str(out) + '/video.mp4"></video>')
    (out / 'video.mp4').write_bytes(b'video-fixture')
    if shared:
        private = Path(manifest['project']) / manifest['evaluation']['output']
        (out / 'private.html').write_text('<video src="' + str(private) + '/video.mp4"></video>')
    target = desk.destination(root, receipt, manifest)
    target.mkdir(parents=True)
    (target / 'keep.txt').write_text('existing local result')
    calls = []
    actual_run = subprocess.run
    def local_rsync(command, **kwargs):
        calls.append(command)
        assert command[0] == 'rsync'
        # Exercise actual rsync + checksum while replacing only its SSH transport.
        local = command[:]
        index = local.index('-e')
        del local[index:index+2]
        local[-2] = local[-2].removeprefix('fake:')
        return actual_run(local, **kwargs)
    monkeypatch.setattr(desk.subprocess, 'run', local_rsync)
    copied = Path(desk.collect(root, receipt, manifest))
    assert copied != target and copied.name.endswith('_yonsei')
    assert (target / 'keep.txt').read_text() == 'existing local result'
    assert (copied / 'video.mp4').read_bytes() == b'video-fixture'
    assert str(copied) + '/video.mp4' in (copied / 'index.html').read_text()
    if shared:
        assert str(copied) + '/video.mp4' in (copied / 'private.html').read_text()
    assert '--checksum' in calls[1] and '--dry-run' in calls[1]
    assert desk.collect(root, receipt, manifest) == str(copied)


def test_collection_failure_retries_without_publishing_and_monitor_survives_disconnect(tmp_path):
    root, base, work, receipt, manifest = fixture(tmp_path)
    replies = [{'id': receipt.name, 'state': 'COMPLETED', 'jobs': [], 'job_ids': []}]
    def fail(*args):
        raise OSError('connection reset')
    worker = desk.Worker(root, probe=lambda *args: replies, collector=fail)
    worker.poll()
    for _ in range(100):
        if desk.load(receipt / 'collection.json', {}).get('state') == 'RETRY':
            break
        time.sleep(.01)
    assert desk.load(receipt / 'collection.json')['state'] == 'RETRY'
    assert not desk.destination(root, receipt, manifest).exists()
    worker.probe = fail
    state = worker.poll()
    assert state['jobs'][0]['state'] == 'COMPLETED'
    assert state['jobs'][0]['connection_error'] == 'connection reset'


def test_output_spec_never_selects_whole_outputs_tree(tmp_path):
    script = tmp_path / 'DP/eval/submit_eval.sh'
    assert yonsei.evaluation_spec(script, {'eval_out_dir': str(tmp_path / 'eval/outputs/run')}, tmp_path)['output'] == 'eval/outputs/run'
    with pytest.raises(ValueError):
        yonsei.evaluation_spec(script, {'eval_out_dir': str(tmp_path / 'eval/outputs')}, tmp_path)


def test_frozen_source_wins_over_venv_editable_checkout(tmp_path, monkeypatch):
    root, base, work, receipt, manifest = fixture(tmp_path)
    source = work / 'project/lerobot/src/lerobot/utils'
    source.mkdir(parents=True)
    (source.parent / '__init__.py').write_text('')
    (source / '__init__.py').write_text('')
    (source / 'sbd_smoothing.py').write_text('MARKER="new snapshot"\n')
    launcher = work / 'project/eval/submit_eval.sh'
    launcher.parent.mkdir(parents=True)
    launcher.write_text('#!/bin/bash\n"' + sys.executable + '" -c "from lerobot.utils.sbd_smoothing import MARKER; print(MARKER)"\n')
    manifest.update(cwd=str(launcher.parent), environment={}, profile={'slurm_submit_mode':'sbatch'})
    (work / 'launch.json').write_text(json.dumps(manifest))
    monkeypatch.setenv('PYTHONPATH', '/old/checkout/lerobot/src')
    assert yonsei.execute(work / 'launch.json') == 0


def test_code_snapshot_excludes_partial_model_directories(tmp_path):
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    model = tmp_path / 'models/dinov3-vits16/config.json'
    model.parent.mkdir(parents=True)
    model.write_text('{}')
    code = tmp_path / 'lerobot/src/lerobot/fresh.py'
    code.parent.mkdir(parents=True)
    code.write_text('')
    assert yonsei.code_files(tmp_path) == {code.relative_to(tmp_path)}


def test_training_progress_exact_outbox_and_legacy_log_fallback(tmp_path):
    work = tmp_path/'launch'
    project = work/'project'
    logs = project/'DP/logs'
    logs.mkdir(parents=True)
    (work/'jobs.jsonl').write_text('{"job_id":"123"}\n')
    manifest = {'cwd': str(project/'DP'), 'script':'DP/submit_train_dp.sh',
                'training':{'total':100000}}
    (logs/'ABC_DP_123.err').write_text("{'steps': 100000}\nstep: 1.2K smpl: 1200\n")
    progress = probe.training_progress(work, manifest)
    assert len(progress) == 1
    assert progress[0]['step'] == 1200
    assert progress[0]['percent'] == 1.2
    assert progress[0]['approximate'] is True
    from lerobot.utils.yonsei_checkpoint_sync import report_progress
    from unittest.mock import patch
    with patch.dict('os.environ', {'SBD_YONSEI_CHECKPOINT_OUTBOX':str(work/'checkpoint_events')}):
        report_progress('/outputs/DP/model', 1273, 100000)
    exact = probe.training_progress(work, manifest)
    assert len(exact) == 1
    assert exact[0]['step'] == 1273
    assert exact[0]['approximate'] is False
    assert exact[0]['percent'] == 1.273


def test_training_preparing_and_fsq_epoch_counters(tmp_path):
    work = tmp_path/'launch'
    project = work/'project'
    logs = project/'FSQ/logs'
    logs.mkdir(parents=True)
    (work/'jobs.jsonl').write_text('{"job_id":"123"}\n')
    manifest = {'cwd':str(project/'FSQ'), 'script':'FSQ/submit_train_fsq.sh',
                'training':{'total':2000,'unit':'epoch'}}
    (logs/'FSQ_123.out').write_text('Preparing dataset')
    item, = probe.training_progress(work, manifest)
    assert item['step'] is None and item['percent'] is None
    assert item['total'] == 2000
    (logs/'FSQ_123.out').write_text('[FSQ-v3] 50/2000 train loss=0.3')
    item, = probe.training_progress(work, manifest)
    assert (item['unit'], item['step'], item['total'], item['percent']) == ('epoch',50,2000,2.5)


def test_training_ignores_unrelated_old_logs_and_coalesces_current_streams(tmp_path):
    work = tmp_path/'launch'
    project = work/'project'
    logs = project/'FT/VSA/logs'
    logs.mkdir(parents=True)
    (work/'jobs.jsonl').write_text('{"job_id":"2848722"}\n')
    manifest = {'cwd':str(project/'FT/VSA'), 'script':'FT/VSA/submit_train.sh',
                'created_at':time.time()-10, 'training':{'total':20000}}
    # A pending launch may see old shared logs, including recently copied ones.
    (logs/'newtask_ft_vsa_2675083.err').write_text('schedule: steps=10000\nCheckpoint policy after step 10000')
    item, = probe.training_progress(work, manifest)
    assert item['step'] is None and item['percent'] is None
    assert item['total'] == 20000
    (logs/'newtask_ft_vsa_2848722.err').write_text('step:1.2K smpl:1M')
    (logs/'newtask_ft_vsa_2848722.out').write_text('Checkpoint policy after step 1000')
    item, = probe.training_progress(work, manifest)
    assert (item['step'], item['total'], item['percent']) == (1200,20000,6)
    assert item['output'] == 'newtask_ft_vsa_2848722'
    # A matching historical job-id alone is insufficient if Slurm IDs get reused.
    manifest['created_at'] = time.time()+10
    item, = probe.training_progress(work, manifest)
    assert item['step'] is None


def test_training_never_scans_shared_logs_without_this_launch_job_identity(tmp_path):
    work = tmp_path/'launch'
    logs = work/'project/DP/logs'
    logs.mkdir(parents=True)
    (logs/'DP_123.err').write_text('step:100K smpl:1M')
    item, = probe.training_progress(work, {'script':'DP/submit_train.sh', 'training':{'total':20000}})
    assert item['step'] is None and item['percent'] is None
