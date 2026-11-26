"""DROID FSQ replay plans from immutable training contracts, not live training YAML."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy
import shlex

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'lerobot/src/lerobot').is_dir())
CONFIGS = ROOT / 'lerobot/skillVLA/droid/configs'
TRAIN = runpy.run_path(str(CONFIGS / 'train_skills/src/train_skills_config.py'))
CAMERAS = {'exterior_1': 'observation.images.image', 'wrist': 'observation.images.wrist_image',
           'exterior_2': 'observation.images.image2'}


def folder(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError(f'Expected one safe folder name: {value!r}')
    return value


def load_config(path):
    return TRAIN['load_config'](path)


def build_settings(cfg):
    root = Path(cfg.get('project_root', ROOT)).resolve()
    dataset_root, outputs_root = root / cfg['dataset_root'], root / cfg['outputs_root']
    models = cfg.get('models', [])
    if not isinstance(models, list) or not models:
        raise ValueError('models must be a nonempty list of FSQ run names')
    camera = cfg.get('camera', 'exterior_1')
    if camera not in CAMERAS:
        raise ValueError('camera must be exterior_1|wrist|exterior_2')
    for key in ('resume', 'boundary_jitter_sweep'):
        if type(cfg.get(key, True)) is not bool:
            raise ValueError(f'{key} must be true or false')
    episodes, limit = cfg.get('episodes_per_task', 3), cfg.get('max_tasks', 10)
    if type(episodes) is not int or episodes < 1 or type(limit) is not int or limit < 0:
        raise ValueError('episodes_per_task must be positive; max_tasks must be nonnegative')
    tasks = cfg.get('task_ids', 'all')
    if tasks != 'all' and (not isinstance(tasks, list) or not tasks or
            any(type(t) is not int or t < 0 for t in tasks) or len(set(tasks)) != len(tasks)):
        raise ValueError('task_ids must be all or unique nonnegative dataset task IDs')
    if cfg.get('episode_selection', 'first') not in {'first', 'random'}:
        raise ValueError('episode_selection must be first|random')
    requested = cfg.get('checkpoint', 'last')
    requested = requested if isinstance(requested, list) else [requested]
    if not requested or len(set(map(str, requested))) != len(requested):
        raise ValueError('checkpoint must select unique epochs')
    collection = root / 'lerobot/skillVLA/droid/configs/train_skills/FSQ/eval/outputs' / folder(cfg.get('output_name', 'droid_fsq'))
    rows, names = [], set()
    for entry in models:
        if not isinstance(entry, str) and not (isinstance(entry, dict) and isinstance(entry.get('run_name'), str)):
            raise ValueError('Each model must be a run name or a pinned run/checkpoints mapping')
        name = folder(entry if isinstance(entry, str) else entry['run_name'])
        if name in names:
            raise ValueError('models contains a duplicate run')
        names.add(name)
        model_dir = outputs_root / 'FSQ' / name
        contract_path = model_dir.parent / '.contracts' / (name + '.json')
        if not contract_path.is_file():
            raise FileNotFoundError(f'DROID FSQ training contract missing: {contract_path}')
        saved = json.loads(contract_path.read_text())
        if saved.get('stage') != 'FSQ':
            raise ValueError(f'Not an FSQ training contract: {contract_path}')
        # Relocate copied Yonsei contracts by dataset name, preserving source SHA.
        dataset = dataset_root / Path(saved['dataset']).name
        if dataset.name != cfg['target_dataset']:
            raise ValueError('Selected global dataset differs from the FSQ training dataset')
        view = dataset / 'training_view_manifest.json'
        if not view.is_file() or json.loads(view.read_text())['source_sha256'] != saved['source_sha256']:
            raise ValueError(f'DROID source dataset provenance differs: {view}')
        segment = folder(saved['skillset_seg_name'])
        group = saved.get('dataset_selection', {}).get('group', 'all')
        if group != 'all' and not segment.endswith('_' + group):
            segment += '_' + group
        skillset = dataset_root / 'FSQ_dataset' / dataset.name / 'FSQ_inputs' / segment / 'skillset'
        manifest = skillset / 'skillset_manifest.json'
        if not (skillset / '.complete').is_file() or not manifest.is_file() or not (skillset / 'skills').is_dir():
            raise FileNotFoundError(f'Completed FSQ source skillset required: {skillset}')
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != saved['skillset_sha256']:
            raise ValueError(f'FSQ source skillset SHA256 differs: {manifest}')
        selected = TRAIN['GROUPS']['select'](cfg, dataset)
        ids = saved.get('dataset_selection', {}).get('episode_ids')
        if selected:
            allowed = None if ids is None else set(ids)
            ids = [ep for ep in selected['episode_ids'] if allowed is None or ep in allowed]
        if ids == []:
            raise ValueError('Global group and FSQ training selection do not intersect')
        source = saved.get('fsq_action_space', 'joint_velocity')
        if source not in {'joint_velocity', 'delta_pose'} or saved.get('action_dim') != (7 if source == 'delta_pose' else 8):
            raise ValueError(f'Unsupported DROID FSQ action contract: {contract_path}')
        info = json.loads((dataset / 'meta/info.json').read_text())
        if CAMERAS[camera] not in info['features'] or not (dataset / 'videos').is_dir():
            raise FileNotFoundError(f'DROID camera is unavailable: {dataset} / {camera}')
        available = {int(m[1]): p for p in model_dir.glob('FSQ_epoch*.pt')
                     if (m := re.fullmatch(r'FSQ_epoch(\d+)\.pt', p.name))}
        epochs = []
        checkpoints = entry.get('checkpoints', requested) if isinstance(entry, dict) else requested
        if not isinstance(checkpoints, list) or not checkpoints:
            raise ValueError('Pinned model checkpoints must be a nonempty list')
        for value in checkpoints:
            if str(value) == 'last':
                if not available:
                    raise FileNotFoundError(f'No saved FSQ epoch checkpoint: {model_dir}')
                epoch = max(available)
            elif str(value).isdigit() and int(value) > 0:
                epoch = int(value)
            else:
                raise ValueError('checkpoint must be last or a positive epoch')
            if epoch not in available:
                raise FileNotFoundError(f'FSQ checkpoint missing: {model_dir / f"FSQ_epoch{epoch:04d}.pt"}')
            if epoch in epochs:
                raise ValueError('checkpoint entries resolve to the same epoch')
            epochs.append(epoch)
            rows.append(dict(run_name=name, checkpoint=epoch, model_path=str(available[epoch]),
                contract_path=str(contract_path), dataset_dir=str(dataset), skillset=str(skillset),
                action_space=source, levels=saved['fsq_quantizer']['levels'], episode_ids=ids,
                collection=str(collection / name), output=str(collection / name / 'checkpoints' / f'epoch{epoch:04d}')))
    slurm = cfg.get('slurm', {})
    cpus = slurm.get('cpus', 8)
    if type(cpus) is not int or cpus < 1:
        raise ValueError('slurm.cpus must be positive')
    device = cfg.get('device', 'cuda')
    if device not in {'cpu', 'cuda'}:
        raise ValueError('device must be cpu|cuda')
    return dict(models=rows, output=str(collection), camera=camera, video_key=CAMERAS[camera],
        task_ids=tasks, max_tasks=limit, episodes_per_task=episodes,
        episode_selection=cfg.get('episode_selection', 'first'), seed=int(cfg.get('seed', 42)),
        boundary_jitter_sweep=cfg.get('boundary_jitter_sweep', True), resume=cfg.get('resume', True),
        device=device, cpus=cpus, memory=slurm.get('memory', '32G'), time=slurm.get('time', '04:00:00'),
        partition=cfg.get('train_partition', ''), qos=cfg.get('train_qos', ''),
        nodelist=cfg.get('train_nodelist', ''), exclude=cfg.get('train_exclude_nodes', []))


def print_shell(plan):
    values = {'DROID_FSQ_MODELS_JSON': plan['models'], 'EVAL_OUTPUT_DIR': plan['output']}
    for key in ('device', 'cpus', 'memory', 'time', 'partition', 'qos', 'nodelist', 'exclude'):
        values['DROID_FSQ_EVAL_' + key.upper()] = plan[key]
    for key, value in values.items():
        if isinstance(value, list):
            value = json.dumps(value) if key == 'DROID_FSQ_MODELS_JSON' else ','.join(map(str, value))
        print(f'export {key}={shlex.quote(str(value))}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--shell', action='store_true')
    parser.add_argument('--freeze', action='store_true', help='Pin last to numeric epochs in a private snapshot')
    args = parser.parse_args()
    plan = build_settings(load_config(args.config))
    if args.freeze:
        import yaml
        config = yaml.safe_load(args.config.read_text())
        names = list(dict.fromkeys(r['run_name'] for r in plan['models']))
        config['models'] = [dict(run_name=name, checkpoints=[r['checkpoint'] for r in plan['models']
            if r['run_name'] == name]) for name in names]
        temporary = args.config.with_name('.' + args.config.name + '.tmp')
        temporary.write_text(yaml.safe_dump(config, sort_keys=False))
        temporary.replace(args.config)
        return
    if args.shell:
        print_shell(plan)
    else:
        # Keep dry runs readable: never print thousands of UUIDs/episode IDs.
        print(json.dumps({**plan, 'models': [{**r, 'episode_ids': None if r['episode_ids'] is None else
            f"{len(r['episode_ids'])} selected episodes"} for r in plan['models']]}, indent=2))


if __name__ == '__main__':
    main()
