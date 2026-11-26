"""Batched DROID encoding and the shared FSQ codebook/video HTML report."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

_CONFIG = runpy.run_path(str(Path(__file__).with_name('config.py')))
CONFIGS, ROOT = _CONFIG['CONFIGS'], _CONFIG['ROOT']
build_settings, load_config = _CONFIG['build_settings'], _CONFIG['load_config']


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def gallery_selection(available, plan):
    """Cover rare tasks too; the YAML episode count is a per-task upper bound."""
    import numpy as np
    tasks = sorted(available) if plan['task_ids'] == 'all' else plan['task_ids']
    if plan['task_ids'] != 'all' and any(not available.get(task) for task in tasks):
        raise ValueError('Requested task has no selected DROID skill episodes')
    if plan['task_ids'] == 'all' and plan['max_tasks']:
        tasks = tasks[:plan['max_tasks']]
    if not tasks:
        raise ValueError('No selected DROID skill episodes')
    rng, selected = np.random.default_rng(plan['seed']), {}
    for task in tasks:
        candidates = sorted(available[task])
        n = min(plan['episodes_per_task'], len(candidates))
        selected[task] = (sorted(int(ep) for ep in rng.choice(candidates, n, replace=False))
                          if plan['episode_selection'] == 'random' else candidates[:n])
    return selected


def evaluate(plan):
    import fcntl
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.evaluation.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This FSQ evaluation output is already running; nothing duplicated') from None
        return _evaluate(plan)


def _evaluate(plan):
    import numpy as np
    import torch
    from lerobot.utils import gpu_memory_guard as guard
    if plan['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; refusing CPU fallback')
    torch.set_num_threads(min(plan['cpus'], 8))
    shared = ROOT / 'lerobot/skillVLA/libero/configs/train_skills'
    sys.path[:0] = [str(shared / 'FSQ/src'), str(shared / 'skill_eval/fsq_codebook/src')]
    encoder = module(ROOT / 'lerobot/skillVLA/libero/configs/train_skillVLA/build_data/src/encode_FSQ_skills.py', 'droid_fsq_encoder')
    replay = module(shared / 'skill_eval/fsq_codebook/src/evaluate.py', 'droid_fsq_replay')
    pose = runpy.run_path(str(CONFIGS / 'src/fsq_action_data.py'))
    from FSQ import load_fsq_encoder
    from action_representation import prepare_fsq_actions
    from train_FSQ import load_skill_files
    print('output: ' + plan['output'], flush=True)
    data_cache, reader_cache, collections = {}, {}, []
    for row in plan['models']:
        output = Path(row['output']); output.mkdir(parents=True, exist_ok=True)
        identity = dict(format='droid_fsq_eval_v2', checkpoint_sha256=hashlib.sha256(Path(row['model_path']).read_bytes()).hexdigest(),
            source_contract_sha256=hashlib.sha256(Path(row['contract_path']).read_bytes()).hexdigest(),
            episode_ids=row['episode_ids'], camera=plan['camera'], task_ids=plan['task_ids'],
            max_tasks=plan['max_tasks'], episodes_per_task=plan['episodes_per_task'],
            episode_selection=plan['episode_selection'], seed=plan['seed'])
        owner = output / 'evaluation_contract.json'
        if owner.exists() and (not plan['resume'] or json.loads(owner.read_text()) != identity):
            raise ValueError(f'Evaluation output belongs to a different request; preserve it: {output}')
        if not owner.exists() and any(output.iterdir()):
            raise FileExistsError(f'Unowned evaluation output; preserve and inspect: {output}')
        atomic_json(owner, identity)
        key = (row['dataset_dir'], row['skillset'], row['action_space'], tuple(row['episode_ids'] or []))
        if key not in data_cache:
            skills = Path(row['skillset']) / 'skills'
            if row['action_space'] == 'delta_pose':
                values = pose['load_pose_skills'](skills, row['dataset_dir'], episode_ids=row['episode_ids'])
            else:
                values = load_skill_files(skills, episode_ids=row['episode_ids'])
            data_cache[key] = values
        segments, states, actions, metadata = data_cache[key]
        model, cfg = load_fsq_encoder(row['model_path'], plan['device'])
        expected_action = 'delta_pose' if row['action_space'] == 'delta_pose' else 'dataset'
        if (cfg.action_space != expected_action or cfg.action_dim != (7 if expected_action == 'delta_pose' else 8)
                or list(cfg.fsq_levels) != row['levels']):
            raise ValueError('Checkpoint architecture/action space differs from its DROID training contract')
        artifacts = output / 'artifacts'; artifacts.mkdir(exist_ok=True)
        latents_path, sweep = artifacts / 'skill_latents.npz', artifacts / 'skill_boundary_jitter_sweep.npz'
        if not latents_path.is_file():
            source = ([prepare_fsq_actions(a, s, 'delta_pose', [6]) for a, s in zip(actions, states, strict=True)]
                      if expected_action == 'delta_pose' else actions)
            with torch.inference_mode():
                latents, tokens = encoder._encode_trajectory_batch(model, source, device=plan['device'], batch_size=128)
            guard.mark_ready()
            arrays = dict(latents=latents, tokens=tokens, skill_order=np.asarray(encoder._compute_skill_orders(metadata), dtype=np.float32))
            for name in ('episode_id', 'task_id', 'skill_index', 'frame_start', 'frame_end', 'length'):
                arrays[name] = np.asarray([m[name] for m in metadata])
            temporary = artifacts / '.latents.tmp.npz'
            np.savez_compressed(temporary, **arrays); temporary.replace(latents_path)
        else:
            with np.load(latents_path, allow_pickle=False) as cached:
                if len(cached['tokens']) != len(metadata):
                    raise ValueError('Cached latent count differs from the selected training skills')
            guard.mark_ready()
        if plan['boundary_jitter_sweep'] and not sweep.is_file():
            args = encoder.Args(skills_dir=str(Path(row['skillset']) / 'skills'), device=plan['device'], encode_batch_size=128)
            with torch.inference_mode():
                encoder._encode_boundary_sweep(args, model=model, output_path=sweep, clean_latents_path=latents_path,
                    device=plan['device'], segments=segments, skill_actions=actions, metadata=metadata, dec_states=states)
        del model
        if plan['device'] == 'cuda':
            torch.cuda.empty_cache()
        dataset = replay.SkillEvaluationDataset(skill_dataset_dir=row['dataset_dir'], skill_latents_path=latents_path,
            eval_init_states_path=None, original_dataset_dir=None, suite_name='droid')
        selected = gallery_selection(dataset.available_episode_ids_by_task(), plan)
        selection_path = Path(plan['output']) / 'episode_selection.json'
        selection = {str(task): ids for task, ids in selected.items()}
        if selection_path.is_file() and json.loads(selection_path.read_text()) != selection:
            raise ValueError('Models do not select identical evaluation episodes; refusing a mixed comparison')
        atomic_json(selection_path, selection)
        print(f"Gallery: {len(selected)} tasks / {sum(map(len, selected.values()))} episodes", flush=True)
        reader_key = (row['dataset_dir'], plan['video_key'])
        if reader_key not in reader_cache:
            reader_cache[reader_key] = replay._EpisodeFrameReader(Path(row['dataset_dir']), video_key=plan['video_key'])
            reader_cache[reader_key].materialize_videos = True
        args = SimpleNamespace(model_path=Path(row['model_path']), latents_path=latents_path,
            boundary_sweep_path=sweep if plan['boundary_jitter_sweep'] else None, terminator_diagnostics_path=None,
            skill_dataset_dir=Path(row['dataset_dir']), eval_init_states_path='', original_dataset_dir='',
            target_task='droid', task_ids=json.dumps(list(selected)),
            episode_ids=json.dumps(sorted({ep for ids in selected.values() for ep in ids})),
            episodes_per_task=1, episode_selection=plan['episode_selection'],
            output_dir=output, collection_dir=Path(row['collection']),
            expected_epoch_tags=json.dumps([f"epoch{r['checkpoint']:04d}" for r in plan['models'] if r['run_name'] == row['run_name']]),
            run_name=row['run_name'], model_name=row['run_name'], report_title='DROID FSQ codebook',
            epoch_tag=f"epoch{row['checkpoint']:04d}", worker_index=0, worker_count=1,
            seed=plan['seed'], resume=plan['resume'], compare_dir='', video_playback=True)
        replay.run_one(args, {'reader': reader_cache[reader_key], 'descriptions': dataset.task_descriptions})
        manifest = json.loads((output / 'metrics/manifest.json').read_text())
        records = manifest['records']
        if not manifest['completed'] or not records:
            raise ValueError('Incomplete FSQ replay manifest')
        for record in records.values():
            for field in ('start_image_path', 'final_image_path', 'video_path'):
                if not (output / record[field]).is_file():
                    raise FileNotFoundError(f'Missing replay asset: {record[field]}')
        summary = dict(run_name=row['run_name'], checkpoint=row['checkpoint'], action_space=row['action_space'],
            levels=row['levels'], camera=plan['camera'], training_skill_count=len(metadata),
            episodes_per_task_cap=plan['episodes_per_task'],
            task_count=len(manifest['signature']['selected_episodes']),
            episode_count=sum(map(len, manifest['signature']['selected_episodes'].values())),
            skill_occurrences=len(records), codebook_used=manifest['train_codebook_used'],
            codebook_size=int(np.prod(row['levels'])), codebook_effective=manifest['train_codebook_effective'])
        atomic_json(output / 'summary.json', summary)
        atomic_json(output / 'evaluation_complete.json', {**summary, 'verified': True})
        collections.append(row['collection'])
        print('Verified: ' + json.dumps(summary), flush=True)
    collections = list(dict.fromkeys(collections))
    if len(collections) > 1:
        replay.maybe_build_compare(collections, output_dir=Path(plan['output']) / 'compare')
    print('DONE -> ' + plan['output'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    plan = build_settings(load_config(args.config))
    from lerobot.utils import gpu_memory_guard as guard
    if plan['device'] == 'cuda':
        exit_code = guard.supervise_if_needed(SimpleNamespace(policy=SimpleNamespace(device='cuda')))
        if exit_code is not None:
            raise SystemExit(exit_code)
        guard.apply_budget()
    logging.basicConfig(level=logging.INFO)
    evaluate(plan)


if __name__ == '__main__':
    main()
