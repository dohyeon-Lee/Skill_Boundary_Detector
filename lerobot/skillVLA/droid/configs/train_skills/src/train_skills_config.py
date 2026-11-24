"""DROID DP/FSQ plans and training, using the shared model/trainer implementations."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shlex
import sys

import numpy as np
import yaml

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'lerobot/src/lerobot').is_dir())
CONFIGS = ROOT / 'lerobot/skillVLA/droid/configs'
SHARED = ROOT / 'lerobot/skillVLA/libero/configs/train_skills'
GLOBAL = runpy.run_path(str(CONFIGS / 'src/global_config_loader.py'))
FILTER = runpy.run_path(str(CONFIGS / 'src/idle_filter.py'))
VIEW = runpy.run_path(str(CONFIGS / 'src/dataset_view.py'))
GROUPS = runpy.run_path(str(CONFIGS / 'src/dataset_groups.py'))
DATASETS = runpy.run_path(str(CONFIGS / 'src/dataset_registry.py'))


def load_config(path):
    path = Path(path).resolve()
    local = yaml.safe_load(path.read_text()) or {}
    if 'idle_filter' in local:
        raise ValueError('Set idle_filter only in DROID global_config.yaml')
    global_path = next((p / 'global_config.yaml' for p in path.parents
                        if (p / 'global_config.yaml').is_file()), None)
    if global_path is None:
        raise FileNotFoundError('DROID config requires its live or frozen global_config.yaml')
    config = {**GLOBAL['load_global_config'](global_path, resolve_dataset=False), **local}
    config = DATASETS['configure'](config)
    GROUPS['definition'](config)
    FILTER['validate_settings'](config)
    if config.get('droid_action_space') != 'joint_velocity' or config.get('droid_proprio_space') != 'joint_position':
        raise ValueError('This DROID adapter implements joint_velocity actions and joint_position proprioception')
    for name in ('droid_fps', 'vla_chunk_size', 'vla_replan_steps'):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if config['vla_replan_steps'] > config['vla_chunk_size']:
        raise ValueError('vla_replan_steps cannot exceed vla_chunk_size')
    return config


def resolve(config, stage):
    if stage not in ('DP', 'FSQ'):
        raise ValueError('Choose DP or FSQ')
    root = Path(config.get('project_root', ROOT)).resolve()
    dataset = root / config['dataset_root'] / config['target_dataset']
    selector = FILTER['TrainingIdleFilter'](config, stage)
    if stage == 'DP':
        action_space = config.get('dp_action_space', 'joint_velocity')
        if action_space not in {'joint_velocity', 'delta_pose'}:
            raise ValueError('dp_action_space must be joint_velocity|delta_pose')
        if action_space == 'delta_pose':
            dataset = dataset.with_name(dataset.name.removesuffix('_joint_velocity') + '_delta_pose')
        obs, future = config['dp_n_obs_steps'], config['dp_future_action_horizon']
        if type(obs) is not int or obs < 1 or type(future) is not int or future < 1:
            raise ValueError('DP history and future horizon must be positive integers')
        tags = dict(config)
        tags['target_dataset'] = dataset.name
        if action_space == 'joint_velocity' and 'dp_action_space' in config:
            # New runs use the requested delta_joint label. Keep the stored
            # view's precise joint_velocity name and old frozen run names.
            tags['target_dataset'] = dataset.name.removesuffix('_joint_velocity') + '_delta_joint'
        tags['dp_lr_tag'] = 'constLR' if config.get('dp_lr_scheduler') == 'constant' else config.get('dp_lr_scheduler', 'cosine')
        dims = config.get('dp_down_dims', [128, 256, 512])
        tags['dp_unet_tag'] = 'small' if dims == [128, 256, 512] else 'dims' + '-'.join(map(str, dims))
        name = config['dp_policy_name'].format(**tags)
        source = root / config['dataset_root'] / config['source_dataset']
        required = [source / 'conversion_manifest.json']
    else:
        name = config['fsq_run_name'].format(target_dataset=config['target_dataset'])
        source = dataset
        skills = dataset.parent / config['fsq_dataset_root'] / config['target_dataset'] / config['fsq_inputs_name'] / GROUPS['tagged'](config['skillset_seg_name'], config) / config['skillset_name']
        required = [dataset / 'training_view_manifest.json', skills / 'skillset_manifest.json', skills / 'skills']
    if selector.enabled:
        name += '_idle'
    name = GROUPS['tagged'](name, config)
    if Path(name).name != name or name in ('', '.', '..'):
        raise ValueError('Training output name must be one folder name')
    output = root / config['outputs_root'] / stage / name
    prefix = 'dp' if stage == 'DP' else 'fsq_train'
    plan = {'stage': stage, 'project_root': str(root), 'dataset': str(dataset),
            'source_dataset': str(source), 'output': str(output), 'idle_filter': selector.contract(),
            'fps': config['droid_fps'], 'action_dim': 8, 'state_dim': 8,
            'action_space': 'joint_velocity', 'proprio_space': 'joint_position',
            'vla_chunk_size': config['vla_chunk_size'], 'vla_replan_steps': config['vla_replan_steps'],
            'cpus': int(config.get(prefix + '_cpus_per_task', 8)),
            'mem': config.get(prefix + '_mem', '64G'), 'time': config.get(prefix + '_time', '48:00:00'),
            'partition': config.get('train_partition', ''), 'qos': config.get('train_qos', ''),
            'nodelist': config.get('train_nodelist', ''), 'exclude': config.get('train_exclude_nodes', []),
            'missing_prerequisites': [str(p) for p in required if not p.exists()]}
    if stage == 'DP':
        plan.update(history_steps=obs, prediction_steps=future)
        if action_space == 'delta_pose':
            plan.update(action_space='delta_pose', proprio_space='eef_pose', action_dim=7, state_dim=7,
                action_stats_path=str(output.parent / '.contracts' / (name + '.action_stats.json')))
    else:
        plan.update(skillset=str(skills), chunk_size=int(config.get('fsq_chunk_size', config['vla_chunk_size'])))
        if plan['chunk_size'] < 1:
            raise ValueError('FSQ chunk size must be positive')
    if plan['cpus'] < 1:
        raise ValueError('Training CPU budget must be positive')
    selection = GROUPS['select'](config, dataset if dataset.exists() else source)
    if selection:
        plan['dataset_selection'] = selection
    return plan


def filter_settings(config, stage):
    return {'workflow_domain': 'droid', 'idle_filter': {stage: FILTER['validate_settings'](config)[stage]}, 'stage': stage}


def build_dp_config(config, plan):
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    obs, horizon = plan['history_steps'], plan['prediction_steps']
    policy = DiffusionConfig(device=config.get('dp_device', 'cuda'), push_to_hub=False,
        state_only=True, proprio_grounding='none', history_conditioning='state',
        history_encoder=config.get('dp_history_encoder', 'transformer'),
        history_encoder_dim=int(config.get('dp_history_encoder_dim', 128)),
        n_obs_steps=obs, horizon=horizon, n_action_steps=horizon,
        action_sequence_mode='future_only', drop_n_last_frames=horizon - 1,
        down_dims=tuple(config.get('dp_down_dims', [128, 256, 512])),
        scheduler_name=config.get('dp_lr_scheduler', 'constant'), use_amp=bool(config.get('dp_amp', False)))
    if plan.get('action_space') == 'delta_pose':
        policy.use_delta_pose_actions = True
        policy.delta_pose_input = 'eef_pose'
        policy.delta_pose_stats_path = plan['action_stats_path']
    return TrainPipelineConfig(dataset=DatasetConfig(repo_id='droid/' + Path(plan['dataset']).name, root=plan['dataset'], video_backend='pyav',
        episodes=plan.get('dataset_selection', {}).get('episode_ids')),
        policy=policy, output_dir=Path(plan['output']), job_name=Path(plan['output']).name,
        steps=int(config.get('dp_steps', 100000)), batch_size=int(config.get('dp_batch_size', 256)),
        num_workers=int(config.get('dp_num_workers', 8)), seed=int(config.get('dp_seed', 42)),
        save_freq=int(config.get('dp_save_freq', 10000)), log_freq=int(config.get('dp_log_freq', 200)),
        eval_freq=0, dataloader_multiprocessing_context='spawn',
        droid_idle_filter=filter_settings(config, 'DP'),
        wandb=WandBConfig(enable=bool(config.get('dp_wandb_enable', True)), project=config.get('dp_wandb_project', 'DROID_DP')))


def _contract(config, plan):
    contract = {key: plan[key] for key in ('stage', 'dataset', 'idle_filter', 'fps', 'action_dim', 'state_dim')}
    if 'dataset_selection' in plan:
        contract['dataset_selection'] = plan['dataset_selection']
    if plan.get('action_space') == 'delta_pose':
        contract.update(dp_action_space='delta_pose', delta_pose_input='eef_pose')
    contract['source_sha256'] = json.loads((Path(plan['dataset']) / 'training_view_manifest.json').read_text())['source_sha256']
    names = ('dp_n_obs_steps', 'dp_future_action_horizon', 'dp_history_encoder', 'dp_history_encoder_dim', 'dp_down_dims', 'dp_lr_scheduler') if plan['stage'] == 'DP' else ('fsq_quantizer', 'fsq_autoencoder', 'fsq_start_state_conditioning', 'fsq_pair_loss', 'fsq_route_loss', 'fsq_hidden_dim', 'fsq_num_layers', 'fsq_lr_schedule', 'fsq_boundary_aug_pmax_step', 'fsq_boundary_aug_pmax_percent', 'skillset_seg_name')
    contract.update({name: config.get(name) for name in names})
    if plan['stage'] == 'FSQ':
        contract['chunk_size'] = plan['chunk_size']
        manifest = Path(plan['skillset']) / 'skillset_manifest.json'
        contract['skillset_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    path = Path(plan['output']).parent / '.contracts' / (Path(plan['output']).name + '.json')
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError(f'Cannot resume with changed DROID data/filter/model contract: {path}')
    VIEW['atomic_json'](path, contract)


def run_dp(config, plan):
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.scripts.lerobot_train import train
    desired = build_dp_config(config, plan)
    if plan.get('action_space') == 'delta_pose':
        stats = runpy.run_path(str(CONFIGS / 'src/dp_action_stats.py'))
        stats['prepare'](plan['dataset'], plan['action_stats_path'], horizon=plan['prediction_steps'],
                         episode_ids=desired.dataset.episodes)
    checkpoint = Path(plan['output']) / 'checkpoints/last/pretrained_model/train_config.json'
    if checkpoint.is_file():
        cfg = TrainPipelineConfig.from_pretrained(checkpoint)
        if cfg.droid_idle_filter != desired.droid_idle_filter:
            raise ValueError('Saved DP idle-filter settings differ; choose a separate output name')
        if cfg.dataset.episodes != desired.dataset.episodes:
            raise ValueError('Saved DP episode group differs; choose a separate output name')
        if (cfg.policy.use_delta_pose_actions, cfg.policy.delta_pose_input) != (
                desired.policy.use_delta_pose_actions, desired.policy.delta_pose_input):
            raise ValueError('Saved DP action representation differs; choose a separate output name')
        cfg.resume = True
        cfg.steps, cfg.num_workers, cfg.batch_size = desired.steps, desired.num_workers, desired.batch_size
        # validate() restores model/optimizer/scheduler/RNG using this exact checkpoint.
        sys.argv.append('--config_path=' + str(checkpoint))
    else:
        cfg = desired
    train(cfg)


def fsq_module():
    path = SHARED / 'FSQ/src/train_FSQ.py'
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('droid_shared_fsq_train', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_fsq_args(config, plan, module):
    args = module.Args(skills_dir=str(Path(plan['skillset']) / 'skills'), raw_dataset_dir=plan['dataset'],
        output_dir=plan['output'], device=config.get('fsq_device', 'cuda'),
        autoencoder_mode=config['fsq_autoencoder']['mode'],
        action_gripper_weight=float(config['fsq_autoencoder']['gripper_weight']),
        action_space='dataset', action_gripper_indices=[7], chunk_size=plan['chunk_size'],
        start_state_conditioning=config.get('fsq_start_state_conditioning', 'none'),
        decoder_reconstructor=True, decoder_terminator_progress=False, decoder_terminator_termination=False,
        fsq_levels=config['fsq_quantizer']['levels'], quantizer='fsq',
        pair_loss=config['fsq_pair_loss']['type'], pair_weight=float(config['fsq_pair_loss']['weight']),
        pair_inv_temperature=float(config['fsq_pair_loss']['inv_temperature']),
        route_loss=config['fsq_route_loss']['enabled'],
        wandb_project=config.get('fsq_wandb_project', 'DROID_FSQ'), wandb_run_name=Path(plan['output']).name)
    args.episode_ids = plan.get('dataset_selection', {}).get('episode_ids')
    mapping = {'fsq_hidden_dim': 'hidden_dim', 'fsq_num_layers': 'num_layers',
        'fsq_lr': 'lr', 'fsq_lr_schedule': 'lr_schedule', 'fsq_batch_size': 'batch_size',
        'fsq_num_epochs': 'epochs', 'fsq_samples_per_skill': 'samples_per_skill',
        'fsq_num_workers': 'num_workers', 'fsq_val_num_workers': 'val_num_workers',
        'fsq_val_every': 'val_every', 'fsq_checkpoint_every': 'checkpoint_every',
        'fsq_save_best_model': 'save_best_model', 'fsq_gradient_checkpointing': 'gradient_checkpointing',
        'fsq_seed': 'seed', 'fsq_boundary_aug_pmax_step': 'boundary_aug_pmax',
        'fsq_boundary_aug_pmax_percent': 'boundary_aug_pmax_percent'}
    for source, target in mapping.items():
        if source in config:
            setattr(args, target, config[source])
    return args


def run_fsq(config, plan):
    module = fsq_module()
    args = build_fsq_args(config, plan, module)
    checkpoints = list(Path(plan['output']).glob('FSQ_epoch*.pt'))
    if checkpoints:
        args.resume_from = str(max(checkpoints, key=lambda p: int(p.stem.removeprefix('FSQ_epoch'))))
    elif Path(plan['output']).exists() and any(Path(plan['output']).iterdir()):
        raise FileExistsError('FSQ output exists without a resumable checkpoint; preserve and inspect')
    if FILTER['TrainingIdleFilter'](config, 'FSQ').enabled:
        adapter = runpy.run_path(str(CONFIGS / 'src/training_filter.py'))
        _, _, _, metadata = module.load_skill_files(Path(args.skills_dir), episode_ids=args.episode_ids)
        mask = adapter['fsq_skill_mask'](plan['dataset'], metadata, config)
        target = Path(plan['output']).parent / '.contracts' / (Path(plan['output']).name + '.mask.npy')
        np.save(target, mask)
        args.training_skill_mask_path = str(target)
    module.main(args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['DP', 'FSQ'], required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--shell', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--config_path')  # Shared trainer re-execution on full-state resume.
    args = parser.parse_args(argv)
    config = load_config(args.config)
    plan = resolve(config, args.stage)
    if (args.check or args.run) and plan['missing_prerequisites']:
        raise FileNotFoundError('Missing prerequisites (nothing submitted): ' + ', '.join(plan['missing_prerequisites']))
    if args.run:
        import torch
        from types import SimpleNamespace
        from lerobot.utils import gpu_memory_guard
        device = config.get('dp_device' if args.stage == 'DP' else 'fsq_device', 'cuda')
        guarded = gpu_memory_guard.supervise_if_needed(SimpleNamespace(policy=SimpleNamespace(device=device)))
        if guarded is not None:
            raise SystemExit(guarded)
        if device != 'cpu' and not torch.cuda.is_available():
            raise RuntimeError('DROID training requested CUDA but CUDA is unavailable; refusing CPU fallback')
        gpu_memory_guard.apply_budget()
        if args.stage == 'DP':
            VIEW['prepare_view'](plan['source_dataset'], plan['dataset'], fps=plan['fps'], mode=plan['action_space'])
        _contract(config, plan)
        print('output: ' + plan['output'], flush=True)
        print(f'schedule: steps={config.get("dp_steps", 100000)}' if args.stage == 'DP'
              else f'schedule: epochs={config.get("fsq_num_epochs", 2000)}', flush=True)
        print('DROID training plan: ' + json.dumps(plan), flush=True)
        if args.stage == 'DP':
            run_dp(config, plan)
        else:
            run_fsq(config, plan)
    elif args.shell:
        values = {'PROJECT_ROOT': plan['project_root'], 'DROID_OUTPUT_DIR': plan['output'],
                  'DROID_CPUS': plan['cpus'], 'DROID_MEM': plan['mem'], 'DROID_TIME': plan['time'],
                  'DROID_PARTITION': plan['partition'], 'DROID_QOS': plan['qos'],
                  'DROID_NODELIST': plan['nodelist'], 'DROID_EXCLUDE': plan['exclude']}
        for name, value in values.items():
            if isinstance(value, list):
                value = ','.join(map(str, value))
            print(f'export {name}={shlex.quote(str(value))}')
    else:
        print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
