#!/usr/bin/env python3
"""Parallel ABC probe/GMM caches with one global calibration and report pass."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading


THREAD_ENV_NAMES = (
    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'ARROW_NUM_THREADS',
)


def configure_cpu_threads(settings, *, episode_worker=False):
    """Apply the CPU budget even in the in-process, single-worker path."""
    workers = int(settings.get('workers', 1))
    budget = int(settings.get('cpu_budget', workers))
    if not 1 <= workers <= budget:
        raise ValueError('Worker count must be between 1 and the CPU budget')
    threads = max(1, budget // workers) if episode_worker else budget
    for name in THREAD_ENV_NAMES:
        os.environ[name] = str(threads)
    # NumPy may already be imported by config/cache validation. Environment
    # variables alone cannot resize its existing native thread pools.
    from threadpoolctl import threadpool_limits
    import torch
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    threadpool_limits(limits=threads)
    print(f'[cpu] threads={threads}, interop=1, budget={budget}, '
          f'episode_workers={workers}', flush=True)
    return threads


def load_evaluator():
    path = next(parent / 'libero/configs/train_skills/DP/eval/src/evaluate.py'
                for parent in Path(__file__).resolve().parents
                if (parent / 'libero/configs/train_skills/DP/eval/src/evaluate.py').is_file())
    spec = importlib.util.spec_from_file_location('abc_shared_ablation', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def episode_selection(evaluator, settings):
    skillset = Path(settings['skillset_dir'])
    selected = evaluator._selected_episode_ids(
        skillset, task_ids=settings['task_ids'], task_id_space=settings['task_id_space'],
        target_task=settings['target_task'], n_episodes=settings['n_episodes'])
    calibration = evaluator._threshold_calibration_episode_ids(settings, skillset, selected)
    if not selected or len(set(calibration)) != len(calibration):
        raise ValueError('Ablation episode selection is empty or contains duplicates')
    return selected, calibration


def ready(work_dir, index):
    run = os.environ.get('SBD_TRAINING_PARENT_ID', 'local')
    (work_dir / f'policy_ready_{run}_{index:03d}.json').write_text(json.dumps({'pid': os.getpid()}))


@contextmanager
def action_contract(evaluator, settings):
    """Forward ABC's joint/gripper mask to the shared descriptor collector."""
    import skill_divider
    from action_manifold import relative_action_mask
    manifest = evaluator._load_manifest(Path(settings['skillset_dir']))
    action = manifest['action']
    mode = action.get('mode', 'dataset')
    mask = None
    if mode in {'anchor_relative', 'delta_joint'}:
        root = evaluator._portable_path(manifest['dataset_dir'], 'dataset_ABC')
        info = json.loads((root / 'meta/info.json').read_text())
        names = info['features']['action']['names']
        mask = relative_action_mask(action['dim'], names, action.get('relative_exclude_joints', []))
        if any(mask[int(index)] for index in action.get('gripper_indices', [])):
            raise ValueError('ABC transformed-action probes must keep grippers absolute')
    original = skill_divider.run_vf_analysis
    def collect(*args, **kwargs):
        if kwargs.get('action_mode', 'dataset') != mode:
            raise ValueError('Probe action mode differs from the source skillset')
        if mask is not None and kwargs.get('relative_mask') is None:
            kwargs['relative_mask'] = mask.copy()
        return original(*args, **kwargs)
    skill_divider.run_vf_analysis = collect
    try:
        yield
    finally:
        skill_divider.run_vf_analysis = original


class CachesReady(Exception):
    """End a worker at the cache/derivation boundary, before calibration or HTML."""


def cache_worker(evaluator, settings, episode_ids, work_dir, index):
    if settings.get('rollout', {}).get('enabled') or settings.get('prediction_loss', {}).get('enabled'):
        raise ValueError('Parallel workers currently support probe ablations; use eval_workers: 1 for rollout/loss')
    if not episode_ids or len(set(episode_ids)) != len(episode_ids):
        raise ValueError('Worker episodes must be nonempty and unique')
    originals = {name: getattr(evaluator, name) for name in (
        '_selected_episode_ids', '_threshold_calibration_episode_ids', '_derive', 'load_policy')
        if hasattr(evaluator, name)}
    # run() imports load_policy from skill_divider locally. Hook that definition
    # to publish admission readiness only after its CUDA model is resident.
    import skill_divider
    original_policy = skill_divider.load_policy
    def load_policy(*args, **kwargs):
        result = original_policy(*args, **kwargs)
        ready(work_dir, index)
        print(f'[worker {index}] policy ready', flush=True)
        return result
    def end_cache_stage(*args, **kwargs):
        raise CachesReady()
    evaluator._selected_episode_ids = lambda *a, **kw: list(episode_ids)
    evaluator._threshold_calibration_episode_ids = lambda *a, **kw: list(episode_ids)
    evaluator._derive = end_cache_stage
    skill_divider.load_policy = load_policy
    print(f'[worker {index}] cache-only scope: {len(episode_ids)} episodes; '
          f'global calibration remains {settings.get("threshold_calibration", {})}; '
          'no local threshold or report will be published', flush=True)
    try:
        evaluator.run(settings)
    except CachesReady:
        (work_dir / f'worker_{index:03d}_complete.json').write_text(
            json.dumps({'state': 'CACHES_READY', 'episodes': episode_ids}))
        print(f'[worker {index}] raw and GMM caches ready; no local threshold/report generated', flush=True)
    else:
        raise RuntimeError('Worker did not stop at the cache-stage boundary')
    finally:
        skill_divider.load_policy = original_policy
        for name, value in originals.items():
            setattr(evaluator, name, value)


class ParallelAblation:
    def __init__(self, settings, work_dir):
        self.settings = settings
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.workers = int(settings.get('workers', 1))
        self.cpu_budget = int(settings.get('cpu_budget', self.workers))
        if not 1 <= self.workers <= self.cpu_budget:
            raise ValueError('Worker count must be between 1 and the CPU budget')
        self.lock = threading.RLock()
        self.children = set()
        self.stopping = False

    def stop(self, *_):
        with self.lock:
            self.stopping = True
            for child in self.children:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass

    def _worker(self, index, shard_path):
        env = dict(os.environ)
        threads = str(max(1, self.cpu_budget // self.workers))
        for name in THREAD_ENV_NAMES:
            env[name] = threads
        with (self.work_dir / f'worker_{index:03d}.out').open('wb') as out, \
             (self.work_dir / f'worker_{index:03d}.err').open('wb') as err:
            with self.lock:
                if self.stopping:
                    return 143
                child = subprocess.Popen([
                    sys.executable, '-u', str(Path(__file__).resolve()),
                    '--settings-file', str(self.work_dir / 'settings.json'),
                    '--episode-shard', str(shard_path), '--worker-index', str(index),
                    '--work-dir', str(self.work_dir),
                ], env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    start_new_session=True)
                self.children.add(child)
            try:
                code = child.wait()
            finally:
                with self.lock:
                    self.children.discard(child)
        if code:
            text = (self.work_dir / f'worker_{index:03d}.err').read_text(errors='replace')[-12000:]
            print(f'[worker {index}] failed ({code})\n{text}', file=sys.stderr, flush=True)
        return code

    def run(self, evaluator):
        selected, calibration = episode_selection(evaluator, self.settings)
        (self.work_dir / 'selection.json').write_text(json.dumps({
            'evaluation_episodes': selected, 'calibration_episodes': calibration}, indent=2))
        if self.workers == 1:
            import skill_divider
            original = skill_divider.load_policy
            def policy(*args, **kwargs):
                result = original(*args, **kwargs)
                ready(self.work_dir, 0)
                return result
            skill_divider.load_policy = policy
            try:
                return evaluator.run(self.settings)
            finally:
                skill_divider.load_policy = original
        if self.settings.get('rollout', {}).get('enabled') or self.settings.get('prediction_loss', {}).get('enabled'):
            raise ValueError('Parallel execution supports probe ablations; set eval_workers: 1 for rollout/loss')
        balance_path = Path(__file__).resolve().parents[3] / 'build_data/src/run_single_gpu_workers.py'
        spec = importlib.util.spec_from_file_location('abc_episode_balancer', balance_path)
        balancer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(balancer)
        manifest = evaluator._load_manifest(Path(self.settings['skillset_dir']))
        if manifest.get('evaluation_source') in {'libero_subset', 'mimicgen_subset'}:
            lengths = {int(row['episode_index']): int(row['length']) for row in manifest['evaluation_episodes']}
        else:
            from lerobot.datasets.abc_native_io import load_abc_native_episodes
            dataset_dir = evaluator._portable_path(manifest['dataset_dir'], 'dataset_ABC')
            rows = load_abc_native_episodes(dataset_dir).set_index('episode_index')
            lengths = {ep: int(rows.loc[ep, 'length']) for ep in calibration}
        shards = balancer.balance_episodes([(ep, lengths[ep]) for ep in calibration], self.workers)
        (self.work_dir / 'settings.json').write_text(json.dumps(self.settings, indent=2))
        paths = []
        for index, ids in enumerate(shards):
            path = self.work_dir / f'episodes_{index:03d}.json'
            path.write_text(json.dumps(ids))
            paths.append(path)
        assert sorted(ep for ids in shards for ep in ids) == sorted(calibration)
        print(f'SBD ablation: {len(shards)} episode workers, {len(calibration)} calibration episodes; '
              'pooled threshold and report run once after all caches are ready', flush=True)
        try:
            with ThreadPoolExecutor(max_workers=len(shards)) as pool:
                futures = [pool.submit(self._worker, index, path) for index, path in enumerate(paths)]
                for future in as_completed(futures):
                    code = future.result()
                    if code:
                        self.stop()
                        raise RuntimeError(f'Ablation cache worker failed ({code}); partial caches preserved')
        finally:
            if self.stopping:
                self.stop()
        if self.stopping:
            raise RuntimeError('Ablation was stopped; final report was not started')
        # The untouched full selection is used here. Cache metadata must match;
        # never hide a partitioning bug by silently re-probing in this pass.
        original_probe = evaluator._probe_episode
        original_load = evaluator._cluster_cache_load
        def unexpected_probe(*a, **kw):
            raise RuntimeError('Raw cache miss during global merge; inspect worker results')
        def checked_cluster(*a, **kw):
            cached = original_load(*a, **kw)
            # Workers prepare SBD clusters. The optional final-10 comparison
            # is fitted on CPU during reporting, independently of SBD cuts.
            metadata = a[1] if len(a) > 1 else kw.get('metadata', {})
            comparison_only = metadata.get('descriptor_key') == 'final_descriptors'
            if cached is None and not comparison_only:
                raise RuntimeError(f'Cluster cache miss during global merge: {a[0]}')
            return cached
        evaluator._probe_episode = unexpected_probe
        evaluator._cluster_cache_load = checked_cluster
        print('[global merge] all workers finished; calibrating once on the full selection', flush=True)
        try:
            return evaluator.run(self.settings)
        finally:
            evaluator._probe_episode = original_probe
            evaluator._cluster_cache_load = original_load


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--settings-json')
    src.add_argument('--settings-file', type=Path)
    parser.add_argument('--episode-shard', type=Path)
    parser.add_argument('--worker-index', type=int, default=0)
    parser.add_argument('--work-dir', type=Path)
    args = parser.parse_args()
    settings = json.loads(args.settings_json if args.settings_json else args.settings_file.read_text())
    configure_cpu_threads(settings, episode_worker=args.episode_shard is not None)
    from dataset_source import prepare_source
    prepare_source(settings)
    from video_report import prepare_report_layout
    prepare_report_layout(settings)
    work_dir = args.work_dir or Path(os.environ.get('ABC_SBD_WORK_DIR', str(Path(settings['output_dir']) / '_parallel')))
    work_dir.mkdir(parents=True, exist_ok=True)
    evaluator = load_evaluator()
    with action_contract(evaluator, settings):
        if args.episode_shard:
            cache_worker(evaluator, settings, json.loads(args.episode_shard.read_text()), work_dir, args.worker_index)
        else:
            runner = ParallelAblation(settings, work_dir)
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, runner.stop)
            runner.run(evaluator)


if __name__ == '__main__':
    main()
