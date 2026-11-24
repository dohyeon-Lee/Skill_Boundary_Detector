#!/usr/bin/env python3
"""Local CPU conversion or execution inside an existing managed Slurm allocation.

Example: ./prepare_dataset.sh --75000 --16
The first numeric selector is the existing job, the second the episode workers.
No new GPU allocation, Slurm step or direct compute-node SSH is requested.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import uuid

import yaml

from prepare_dataset import HERE, ROOT, load_settings

RUNNER = ROOT / 'lerobot/skillVLA/libero/configs/src/srun_session.py'


def positive(value):
    if not re.fullmatch(r'[1-9][0-9]*', str(value)):
        raise argparse.ArgumentTypeError('Use a positive integer without leading zeroes')
    return int(value)


class Once(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest) is not None:
            parser.error(f'{option_string} was specified more than once')
        setattr(namespace, self.dest, values)


def arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id', type=positive, action=Once, help='Existing RUNNING managed Slurm job; never creates a new allocation')
    parser.add_argument('--workers', type=positive, action=Once, help='Episode workers; overrides the YAML')
    parser.add_argument('--config', type=Path, default=HERE.parent / 'dataset_config.yaml')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Validate and print the plan without enqueueing anything')
    mode.add_argument('--verify', action='store_true', help='Decode and verify an existing completed dataset')
    expanded, numeric, value_next = [], 0, False
    for token in argv:
        if value_next:
            expanded.append(token)
            value_next = False
        elif re.fullmatch(r'--[0-9]+', token):
            numeric += 1
            if numeric > 2:
                parser.error('Use at most --JOB_ID --WORKERS')
            expanded.extend(['--job-id' if numeric == 1 else '--workers', token[2:]])
        else:
            expanded.append(token)
            value_next = token in ('--config', '--job-id', '--workers')
    return parser.parse_args(expanded)


def shared_runner():
    return runpy.run_path(str(RUNNER))


def build_plan(args, cfg, allocation=None):
    workers = args.workers if args.workers is not None else cfg['workers']
    cpus = workers * cfg['encoder_threads']
    if allocation and cpus > int(allocation['NumCPUs']):
        raise ValueError(f'{workers} workers x {cfg["encoder_threads"]} encoder threads need {cpus} CPUs; '
                         f'job {args.job_id} has only {allocation["NumCPUs"]}. Nothing was submitted.')
    return {'mode': 'existing_allocation_cpu' if allocation else 'local_cpu',
            'job_id': str(args.job_id) if args.job_id is not None else None,
            'node': allocation.get('NodeList') if allocation else os.uname().nodename,
            'allocation_submit_time': allocation.get('SubmitTime') if allocation else None,
            'workers': workers, 'encoder_threads': cfg['encoder_threads'], 'cpu_budget': cpus,
            'image_transform': cfg['image_transform'], 'image_size': cfg['image_size'],
            'output': cfg['output_path'], 'config': str(args.config.expanduser().resolve()),
            'gpu_compute': False, 'new_allocation': False, 'new_slurm_step': False}


def write_snapshot(path, cfg, workers):
    # Freeze the resolved conversion inputs, independent of future live YAML edits.
    keys = ('raw_root', 'calibration_root', 'annotations_path', 'dataset_root', 'output_name',
            'max_episodes', 'episode_ids', 'success_only', 'missing_calibration', 'cameras',
            'image_size', 'image_transform', 'fps', 'remove_skipped_steps', 'video_tail_policy', 'action_mode',
            'external_extrinsics', 'encoder_threads', 'video_crf')
    frozen = {key: cfg[key] for key in keys}
    for key in ('recording_timestamps_root', 'empty_episode_policy'):
        if key in cfg:
            frozen[key] = cfg[key]
    frozen['workers'] = workers
    with path.open('x', encoding='utf-8') as stream:
        yaml.safe_dump(frozen, stream, sort_keys=False)
    path.chmod(0o600)


def queue(args, cfg, plan, runner):
    # Shared runner validates owner, RUNNING state and exact allocation identity
    # again, then asks the original srun shell's worker to launch this child.
    capsule = ROOT / '.cache/droid_conversions' / uuid.uuid4().hex
    capsule.mkdir(parents=True, mode=0o700)
    config = capsule / 'dataset_config.yaml'
    write_snapshot(config, cfg, plan['workers'])
    runner['private_json'](capsule / 'plan.json', plan)
    options = [f'--job-name=DROID_prepare_{plan["workers"]}workers',
               f'--cpus-per-task={plan["cpu_budget"]}',
               f'--output={capsule}/convert_%j.out', f'--error={capsule}/convert_%j.err',
               str(HERE / 'prepare_in_allocation.sh'), str(args.job_id), str(config),
               str(plan['cpu_budget'])]
    if args.verify:
        options.append('--verify')
    print(json.dumps(plan, indent=2), flush=True)
    code = runner['submit'](options, existing_job_id=str(args.job_id))
    print(f'Frozen conversion settings / logs: {capsule}', flush=True)
    return code


def main(argv=None):
    args = arguments(sys.argv[1:] if argv is None else argv)
    cfg = load_settings(args.config.expanduser().resolve())
    runner, allocation = None, None
    if args.job_id is not None:
        runner = shared_runner()
        allocation = runner['existing_allocation'](str(args.job_id))
    plan = build_plan(args, cfg, allocation)
    command = [sys.executable, str(HERE / 'prepare_dataset.py'), '--config', str(args.config.expanduser().resolve()),
               '--workers', str(plan['workers'])]
    if args.dry_run:
        print(json.dumps(plan, indent=2), flush=True)
        # Selection and calibration checks happen locally, without a worker request.
        return subprocess.call([*command, '--dry-run'])
    if allocation:
        return queue(args, cfg, plan, runner)
    if args.verify:
        command.append('--verify')
    return subprocess.call(command)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f'DROID preparation: {error}', file=sys.stderr)
        raise SystemExit(1)
