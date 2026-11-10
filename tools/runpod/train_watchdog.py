"""Connect auxiliary RunPod training to the existing HF checkpoint watchdog."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]


def settings(path):
    sys.path.insert(0, str(ROOT / 'lerobot/examples/libero/configs/train_skillVLA/PT/Terminator/src'))
    from terminator_train_config import load_stage1_component_config, build_launch_settings
    config = load_stage1_component_config(path)
    return config, build_launch_settings(config)


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def birth(pid):
    try:
        values = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
        return values[19] if values[0] != 'Z' else None
    except (OSError, IndexError):
        return None


def active(path):
    try:
        record = json.loads(Path(path).read_text())
        return record if birth(record['pid']) == record['birth'] else None
    except (OSError, ValueError, KeyError):
        return None


def prepare(path, *, loader=settings):
    from huggingface_hub import get_token
    if not os.environ.get('RUNPOD_POD_ID') or shutil.which('runpodctl') is None:
        raise ValueError('Automatic training shutdown requires a RunPod Pod and runpodctl.')
    if not get_token():
        raise ValueError('HF login is required before launching training with automatic checkpoint upload.')
    config, resolved = loader(Path(path))
    options = config.get('runpod_watchdog') or {}
    if not isinstance(options, dict) or set(options) - {'enabled', 'keep', 'protect'}:
        raise ValueError('runpod_watchdog accepts enabled, keep and protect only.')
    keep, protected = options.get('keep', 1), options.get('protect', [50000, 100000])
    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
        raise ValueError('runpod_watchdog.keep must be a positive integer.')
    if not isinstance(protected, list) or any(type(step) is not int or step < 0 for step in protected):
        raise ValueError('runpod_watchdog.protect must be a list of non-negative steps.')
    project = Path(resolved['project_root'])
    runs = json.loads(resolved.get('predictor_parallel_runs', '[]'))
    outputs = [Path(item['output_dir']) for item in runs] or [Path(resolved['output_dir'])]
    selected = sorted(str(output.relative_to(project)) for output in outputs)
    roots = {str(Path(run).parent) for run in selected}
    if len(roots) != 1 or len(selected) != len(set(selected)):
        raise ValueError('Selected training runs must be distinct and share one output group.')
    steps = int(resolved['steps'])
    if steps <= 0 or any(step > steps for step in protected):
        raise ValueError('Protected checkpoints must fit the training step target.')
    repo = (os.environ.get('SBD_HF_CHECKPOINT_REPO') or config.get('hf_checkpoint_repo') or '').strip()
    if not repo:
        raise ValueError('hf_checkpoint_repo is required for the training watchdog.')
    manifest = dict(project_root=str(project), repo=repo,
                    log_repo=(os.environ.get('SBD_HF_DATASET_REPO') or config.get('hf_dataset_repo') or '').strip(),
                    root=roots.pop(), expected_runs=selected, keep=keep, protect=sorted(set(protected)),
                    done_step=steps, done_time=None,
                    pod_id=os.environ['RUNPOD_POD_ID'])
    encoded = json.dumps(manifest, sort_keys=True).encode()
    key = hashlib.sha256(json.dumps([manifest['pod_id'], selected]).encode()).hexdigest()[:20]
    folder = project / '.cache/runpod_training_watchdogs' / key
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest['signature'] = hashlib.sha256(encoded).hexdigest()
    with (folder / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        running = active(folder / 'active.json')
        if running and running['signature'] != manifest['signature']:
            raise ValueError('Another watchdog is already managing these runs with different settings.')
        if not running:
            write(folder / 'plan.json', {**manifest, 'config': str(Path(path).resolve())})
    return folder


def launch(folder, job_id):
    folder = Path(folder)
    if not str(job_id).isdigit():
        raise ValueError('Expected a numeric local training job ID.')
    with (folder / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if active(folder / 'active.json'):
            return
        plan = json.loads((folder / 'plan.json').read_text())
        if plan['pod_id'] != os.environ.get('RUNPOD_POD_ID'):
            raise ValueError('Watchdog belongs to a different Pod.')
        write(folder / 'training_job.json', {'job_id': str(job_id)})
        with (folder / 'watchdog.log').open('ab', buffering=0) as log:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'run', str(folder)],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     start_new_session=True, close_fds=True)
        write(folder / 'active.json', {'pid': child.pid, 'birth': birth(child.pid),
                                      'signature': plan['signature'], 'job_id': str(job_id)})


def run(folder):
    folder = Path(folder)
    plan = json.loads((folder / 'plan.json').read_text())
    if plan['pod_id'] != os.environ.get('RUNPOD_POD_ID'):
        raise ValueError('Watchdog belongs to a different Pod.')
    sys.path.insert(0, str(ROOT / 'lerobot/examples/libero/configs/src'))
    from hf_checkpoints import watch
    with (folder / 'watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                write(folder / 'status.json', {'state': 'WATCHING', 'target_steps': plan['done_step'],
                                              'expected_runs': plan['expected_runs'], 'updated_at': time.time()})
                watch(Path(plan['project_root']), plan['repo'], root=plan['root'],
                      keep=plan['keep'], protect=set(plan['protect']), prune_local=True,
                      done_step=plan['done_step'], done_time=None, log_repo=plan['log_repo'],
                      expected_runs=set(plan['expected_runs']), assume_yes=True)
                break
            except Exception as error:
                # Retry transfers, never training. Failed training still triggers
                # shutdown once remaining checkpoints have reached the Hub.
                write(folder / 'status.json', {'state': 'TRANSFER_RETRY', 'error': str(error),
                                              'updated_at': time.time()})
                traceback.print_exc()
                time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'launch', 'run'))
    parser.add_argument('path', type=Path)
    parser.add_argument('job_id', nargs='?')
    args = parser.parse_args()
    if args.mode == 'prepare':
        print(prepare(args.path))
    elif args.mode == 'launch':
        launch(args.path, args.job_id)
        print(f'RunPod HF watchdog: {args.path / "watchdog.log"}')
    else:
        run(args.path)


if __name__ == '__main__':
    main()
