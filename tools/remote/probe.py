"""Bounded, read-only Yonsei login-host probe. Sent over SSH; stdlib only."""
import json
from pathlib import Path
import re
import subprocess
import sys

TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def read(path, default=None):
    try:
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError('Progress JSON is too large')
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def scheduler(ids):
    """A submit command exiting zero is not evidence its GPU jobs finished."""
    if not ids:
        return {}, set()
    joined = ','.join(sorted(set(ids)))
    active = subprocess.run(['squeue', '-h', '-j', joined, '-o', '%i|%T'],
                            capture_output=True, text=True, timeout=25)
    # Slurm can return nonzero for job IDs which have just left the queue.
    if active.returncode and active.stderr.strip() and 'Invalid job id' not in active.stderr:
        raise RuntimeError('squeue unavailable: ' + active.stderr.strip()[:300])
    live = {line.split('|')[0].split('_')[0] for line in active.stdout.splitlines() if '|' in line}
    result = subprocess.run(['sacct', '-n', '-P', '-j', joined,
                             '--format=JobID,State,ExitCode'],
                            capture_output=True, text=True, timeout=25)
    rows = {}
    for line in result.stdout.splitlines() if result.returncode == 0 else []:
        parts = line.split('|')
        if len(parts) < 3 or '.' in parts[0]:
            continue
        identifier, state, code = parts[:3]
        state = state.split()[0].rstrip('+')
        root = identifier.split('_')[0]
        if root in ids:
            rows.setdefault(root, []).append({'id': identifier, 'state': state, 'exit_code': code})
    # Some Yonsei login hosts have no working slurmdbd/sacct connection.
    # The controller still retains recent completed jobs and full array membership.
    if result.returncode:
        for job in sorted(ids):
            control = subprocess.run(['scontrol', '-o', 'show', 'job', job],
                                     capture_output=True, text=True, timeout=20)
            if control.returncode:
                continue  # Accounting remains unknown; never infer success from an empty queue.
            for line in control.stdout.splitlines():
                fields = dict(re.findall(r'(?:^|\s)([A-Za-z]+)=([^\s]+)', line))
                if 'JobState' not in fields:
                    continue
                master = fields.get('ArrayJobId', fields.get('JobId'))
                if master != job:
                    continue
                index = fields.get('ArrayTaskId')
                identifier = job + '_' + index if index else fields['JobId']
                rows.setdefault(job, []).append({'id': identifier,
                    'state': fields['JobState'], 'exit_code': fields.get('ExitCode', '')})
    for line in active.stdout.splitlines():
        parts = line.split('|')
        if len(parts) == 2 and parts[0].split('_')[0] in ids:
            rows.setdefault(parts[0].split('_')[0], []).append(
                {'id': parts[0], 'state': parts[1], 'exit_code': ''})
    return rows, live


def outcomes(path):
    value = read(path, {})
    successes = value.get('successes')
    if not isinstance(successes, list) or not all(type(x) is bool for x in successes):
        return None
    return successes


def progress(project, spec):
    """Count explicit episode outcomes, never infer success from simulator steps."""
    storage = Path(spec.get('storage_project', project))
    out = storage / spec['output']
    # Output must belong to this recipe, including workspace output aliases.
    out.resolve().relative_to(storage.resolve())
    if spec.get('kind') == 'abc':
        for prefix in ('src/metrics', 'metrics'):
            value = read(out / prefix / 'summary.json') or read(out / prefix / 'partial_summary.json')
            if not value:
                continue
            done, total = int(value['num_worlds']), int(value['expected_worlds'])
            successes = int(value['successes'])
            return {'completed': done, 'total': total, 'successes': successes,
                    'percent': 100 * done / total if total else None,
                    'success_pct': 100 * successes / done if done else None,
                    'models': [], 'output_exists': True}
    tasks = [int(t) for t in spec.get('tasks', [])]
    labels = spec.get('labels') or ['eval']
    episodes = int(spec.get('episodes') or 0)
    observations = {}
    patterns = [('panels/*/videos/*/eval_progress.json', True),
                ('videos/*/eval_progress.json', False)]
    for pattern, multi in patterns:
        for path in list(out.glob(pattern))[:10000]:
            path.resolve().relative_to(out.resolve())
            task_match = re.search(r'_(\d+)$', path.parent.name)
            panel_match = re.match(r'(\d+)_', path.parents[2].name) if multi else None
            if not task_match or (multi and not panel_match):
                continue
            key = (int(panel_match[1]) if multi else 0, int(task_match[1]))
            values = outcomes(path)
            if values is not None:
                observations[key] = values
    # Final metrics also cover cached episodes and evaluations without video progress.
    metrics = out / 'metrics'
    paths = [metrics / 'eval_info_merged.json', metrics / 'eval_info.json']
    paths += list(metrics.glob('eval_info_*.json'))[:100]
    for path in dict.fromkeys(paths):
        value = read(path, {})
        if 'per_task' in value:
            value = {labels[0]: value}
        for label, info in value.items():
            if label not in labels or not isinstance(info, dict):
                continue
            index = labels.index(label)
            for task in info.get('per_task', []):
                values = task.get('metrics', task).get('successes')
                if isinstance(values, list) and all(type(x) is bool for x in values):
                    key = (index, int(task['task_id']))
                    if len(values) >= len(observations.get(key, [])):
                        observations[key] = values
    models = []
    for index, label in enumerate(labels):
        rows = []
        for task in tasks:
            values = observations.get((index, task), [])[:episodes]
            rows.append({'task_id': task, 'completed': len(values), 'total': episodes,
                         'successes': sum(values),
                         'success_pct': 100 * sum(values) / len(values) if values else None})
        done, total = sum(r['completed'] for r in rows), sum(r['total'] for r in rows)
        successes = sum(r['successes'] for r in rows)
        models.append({'label': label, 'tasks': rows, 'completed': done, 'total': total,
                       'successes': successes,
                       'success_pct': 100 * successes / done if done else None})
    done, total = sum(m['completed'] for m in models), sum(m['total'] for m in models)
    successes = sum(m['successes'] for m in models)
    return {'completed': done, 'total': total, 'successes': successes,
            'percent': 100 * done / total if total else None,
            'success_pct': 100 * successes / done if done else None, 'models': models,
            'output_exists': out.is_dir()}


def snapshot(base, identifiers):
    launches, all_ids = [], set()
    for identifier in identifiers[:200]:
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', identifier):
            raise ValueError('Invalid launch identity')
        work = base / '.cache/yonsei_launches' / identifier
        manifest = read(work / 'launch.json', {})
        jobs, expected = [], set()
        record = work / 'jobs.jsonl'
        if record.exists():
            if record.stat().st_size > 1024 * 1024:
                raise ValueError('Too many submitted jobs')
            for line in record.read_text().splitlines():
                if line.strip():
                    entry = json.loads(line)
                    job = str(entry['job_id'])
                    if not job.isdigit():
                        raise ValueError('Invalid Slurm identity')
                    jobs.append(job)
                    for task in entry.get('tasks', []):
                        if task is not None and (type(task) is not int or task < 0):
                            raise ValueError('Invalid Slurm array index')
                        expected.add(job if task is None else job + '_' + str(task))
        all_ids.update(jobs)
        row = {'id': identifier, 'launcher': read(work / 'status.json', {}),
               'job_ids': sorted(set(jobs)), 'jobs': [], 'error': '', 'expected_jobs': sorted(expected)}
        row['cancel'] = read(work / 'cancel.json', {})
        if manifest.get('training', {}).get('checkpoint_sync'):
            row['checkpoints'] = [read(p) for p in sorted((work / 'checkpoint_events').glob('*.json'))
                                  if not (work / 'checkpoint_acks' / p.stem).exists()]
        row['durable_exits'] = []
        for identity in expected:
            marker = identity if '_' in identity else identity + '_solo'
            path = work / 'job_status' / (marker + '.exit')
            if path.is_file():
                code = int(path.read_text().strip())
                row['durable_exits'].append({'id': identity, 'state': 'COMPLETED' if code == 0 else 'FAILED',
                                              'exit_code': str(code) + ':0'})
        spec = manifest.get('evaluation')
        if spec:
            try:
                row['progress'] = progress(work / 'project', spec)
            except (OSError, ValueError, KeyError, TypeError) as error:
                row['error'] = 'Progress unavailable: ' + str(error)[:300]
        # An existing allocation dispatch returns before its internal worker exits.
        sessions = work / 'project/.cache/srun_sessions'
        session_states = []
        for path in list(sessions.glob('*/training_status.json'))[:100]:
            value = read(path, {})
            session_states.append(str(value.get('state', value.get('status', 'UNKNOWN'))).upper())
        row['session_states'] = session_states
        row['existing_allocation'] = any(re.fullmatch(r'--[1-9][0-9]*', str(a))
                                         for a in manifest.get('args', []))
        launches.append(row)
    accounting, live = scheduler(all_ids)
    for row in launches:
        for job in row['job_ids']:
            row['jobs'].extend(accounting.get(job, []))
        actual = {j['id']: j for j in row['jobs']}
        actual.update({j['id']: j for j in row.pop('durable_exits')})
        row['jobs'] = list(actual.values())
        launcher = row['launcher']
        states = [j['state'] for j in row['jobs']]
        sessions = row['session_states']
        if launcher.get('state') == 'FAILED':
            state = 'FAILED'
        elif live.intersection(row['job_ids']):
            state = 'PENDING' if states and all(s == 'PENDING' for s in states) else 'RUNNING'
        elif row['job_ids']:
            known = (all(identity in actual for identity in row['expected_jobs']) if row['expected_jobs']
                     else all(accounting.get(job) for job in row['job_ids']))
            if not known or not states or any(s not in TERMINAL for s in states):
                state = 'ACCOUNTING'
            elif any(s != 'COMPLETED' for s in states) or any(j['exit_code'] != '0:0' for j in row['jobs']):
                state = 'CANCELLED' if row['cancel'].get('status') == 'requested' else 'FAILED'
            elif launcher.get('state') == 'COMPLETED':
                state = 'COMPLETED'
            else:
                state = 'FINALIZING'
        elif row['existing_allocation']:
            # Never mistake the outer long-lived allocation for this evaluation.
            if launcher.get('state') == 'COMPLETED' and sessions and all(s in {'COMPLETED', 'DONE', 'SUCCEEDED'} for s in sessions):
                state = 'COMPLETED'
            elif any(s in {'FAILED', 'CANCELLED', 'TIMEOUT'} for s in sessions):
                state = 'FAILED'
            else:
                state = 'RUNNING'
        else:
            state = launcher.get('state', 'LAUNCHING')
        row['state'] = state
    return launches


if __name__ == '__main__':
    request = json.load(sys.stdin)
    print(json.dumps(snapshot(Path(request['base']), request['ids']), allow_nan=False))
