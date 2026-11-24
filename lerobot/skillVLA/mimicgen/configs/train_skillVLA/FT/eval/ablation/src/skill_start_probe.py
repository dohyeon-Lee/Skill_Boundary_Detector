"""GT-prefix reconstruction followed by an isolated GT-conditioned Coffee skill.

Uses the deployed VSA wrapper/processors. Never substitutes GT actions after the
handoff in policy trials. The GT suffix is a separately labelled positive control.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'setup_env.sh').is_file())
HERE = Path(__file__).resolve().parents[1]
SHARED = ROOT / 'lerobot/skillVLA/libero/configs/train_skillVLA/PT/eval/src'
sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(ROOT / 'lerobot/skillVLA/libero/configs/train_skills/src'))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, default=str))
    temporary.replace(path)


def freeze(path):
    import yaml
    from train_skills_config import load_config

    recipe = yaml.safe_load(path.read_text())
    mode = recipe.get('initialization', 'gt_prefix')
    if mode not in {'gt_prefix', 'rollout_entry', 'transition_timing'}:
        raise ValueError('initialization must be gt_prefix, rollout_entry or transition_timing')
    if mode == 'rollout_entry' and (type(recipe.get('max_prefix_steps')) is not int or recipe['max_prefix_steps'] <= 0):
        raise ValueError('max_prefix_steps must be a positive integer')
    for key in ['episode_index', 'skill_index', 'n_trials', 'seed', 'max_skill_steps', 'n_action_steps']:
        value = recipe[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (1 if key in {'n_trials', 'max_skill_steps', 'n_action_steps'} else 0):
            raise ValueError(f'{key}: invalid nonnegative/positive integer')
    cfg = load_config(HERE.parent / 'ft_eval_config.yaml')
    cfg['aux_training_configs'] = {}
    advance_mode = 'external' if mode in {'rollout_entry', 'transition_timing'} else 'gt'
    cfg['model_defaults'] = dict(skill_source='gt', focus_source='gt', pose_source='gt', advance_mode=advance_mode)
    ignored = ('predictor_', 'external_predictor_') if advance_mode == 'external' else ('predictor_', 'terminator_', 'external_')
    cfg['models'] = [{k: v for k, v in cfg['models'][0].items() if not k.startswith(ignored)}]
    cfg['models'][0].update(label='GT_start_VSA', vsa_checkpoint=str(recipe['vsa_checkpoint']),
                            skill_source='gt', focus_source='gt', pose_source='gt', advance_mode=advance_mode)
    cfg['oracle'] = dict(episode_exact=True, repeat_episodes=True)
    cfg['n_action_steps'] = recipe['n_action_steps']
    os.environ['STAGE1_EVAL_WORK_DIR'] = str(HERE.parent)
    resolver = module(HERE.parents[2] / 'PT/eval/src/config.py', 'coffee_probe_config')
    settings = resolver.build_settings(cfg)
    spec = json.loads(settings['models_json'])[0]
    for key in ['episode_index', 'skill_index']:
        if key not in recipe: raise ValueError(f'Missing {key}')
    name = str(recipe['output_name'])
    if Path(name).name != name or name in {'', '.', '..'}:
        raise ValueError('output_name must be one folder name')
    output = HERE / 'outputs' / name
    payload = dict(recipe=recipe, spec=spec, output=str(output),
                   resolved_settings={k: str(v) for k, v in settings.items()})
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
    directory = ROOT / '.cache/coffee_skill_start_probes' / signature
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory / 'settings.json'
    save(snapshot, payload)
    print(f'# {mode} -> VSA skill {recipe["skill_index"]}; checkpoint {recipe["vsa_checkpoint"]}', file=sys.stderr)
    print(f'# Report: {output}/index.html', file=sys.stderr)
    for key, value in dict(PROBE_SETTINGS=snapshot, PROBE_GRES=recipe['slurm']['gres'],
                           PROBE_CPUS=recipe['slurm']['cpus'], PROBE_MEMORY=recipe['slurm']['memory'],
                           PROBE_TIME=recipe['slurm']['time']).items():
        print('export ' + key + '=' + shlex.quote(str(value)))


def run(settings_path):
    import copy
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from lerobot.envs.configs import MimicGenEnv
    from lerobot.envs.mimicgen import MimicGenGymEnv, INSTRUCTION
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.envs.utils import preprocess_observation
    from lerobot.utils.io_utils import write_video
    from lerobot.utils.eval_gpu_memory import apply_limit
    from lerobot.utils.random_utils import set_seed
    import run_eval
    from probe_presentation import render_overlay, write_report

    payload = json.loads(settings_path.read_text())
    recipe, spec = payload['recipe'], payload['spec']
    output = Path(payload['output'])
    output.mkdir(parents=True, exist_ok=True)
    src = output / 'src'
    src.mkdir(exist_ok=True)
    if (src / 'results.json').is_file() and not (src / 'results_before_overlay.json').exists():
        (src / 'results_before_overlay.json').write_bytes((src / 'results.json').read_bytes())
    save(src / 'settings.json', payload)
    results = []
    torch.set_num_threads(int(recipe['slurm']['cpus']))
    apply_limit()
    episode = recipe['episode_index']
    raw = Path(spec['raw_dataset_dir'])
    tables = []
    for path in sorted(raw.glob('data/**/*.parquet')):
        table = pq.read_table(path, filters=[('episode_index', '=', episode)],
                              columns=['episode_index', 'frame_index', 'action', 'observation.state'])
        if len(table): tables.append(table.to_pandas())
    import pandas as pd
    rows = pd.concat(tables).sort_values('frame_index')
    if rows.frame_index.tolist() != list(range(len(rows))):
        raise ValueError('Source episode frame indices must be contiguous from zero')
    actions = np.stack(rows.action)
    states = np.stack(rows['observation.state'])
    latents = np.load(spec['skill_latents_path'])
    chosen = np.flatnonzero((latents['episode_id'] == episode) & (latents['skill_index'] == recipe['skill_index']))
    if len(chosen) != 1: raise ValueError('Expected exactly one selected skill')
    row = int(chosen[0]); start = int(latents['frame_start'][row]); end = int(latents['frame_end'][row])
    records = run_eval.load_episode_exact_data(spec['skill_dataset_dir'], spec['eval_init_states_path'], 'Coffee_D0')
    record = next(r for r in records[0] if r['episode_index'] == episode)
    skill = copy.deepcopy(record['skills'][recipe['skill_index']])
    if skill['token'] != int(latents['tokens'][row]): raise ValueError('Skill artifacts disagree')
    info = dict(source_episode=episode, skill_index=recipe['skill_index'], token=skill['token'],
                frame_start=start, frame_end=end, gt_length=end-start, focus_uv=np.asarray(skill['focus_uv']).tolist(),
                initialization='verified episode-start XML/state + GT actions [0,frame_start)',
                grounding='original episode-start xyz; not the selected skill-start xyz',
                predictor=False, terminator=False, recipe=recipe)
    save(src / 'probe.json', info)

    def report(*, publish=False):
        summary = dict(**info, results=results, completed_vsa_trials=sum(r['kind']=='vsa' for r in results))
        save(src / 'results.json', summary)
        if publish or not (output/'index.html').exists():
            write_report(output, info, results)
    report()

    env_cfg = MimicGenEnv(dataset_dir=str(raw), terminate_on_success=False,
                         episode_length=max(600, start+recipe['max_skill_steps']))
    context = None
    control_boundary = None
    for trial in range(-1 if recipe['gt_replay_control'] else 0, recipe['n_trials']):
        control = trial == -1
        label = 'GT_action_control' if control else f'VSA_seed{recipe["seed"]+trial}'
        set_seed(recipe['seed']+max(0,trial))
        print(f'{label}: reconstructing frame {start}', flush=True)
        env = MimicGenGymEnv(env_cfg)
        try:
            env.init_states = True
            env._init_states = [dict(state=record['init_state'], model_xml=record['model_xml'])]
            obs, _ = env.reset(seed=1)
            for action in actions[:start]: obs, _, _, _, _ = env.step(action)
            boundary = env._last['state'].copy()
            if control_boundary is None: control_boundary = boundary
            boundary_difference = float(np.abs(boundary-control_boundary).max())
            if boundary_difference > 1e-5: raise RuntimeError(f'Reconstructed boundary changed: {boundary_difference}')
            xyz_error = float(np.abs(boundary[:3]-states[start,:3]).max())
            if xyz_error > 0.01: raise RuntimeError(f'Reconstructed start differs from source by {xyz_error} m')
            if not control:
                if context is None:
                    os.environ.update(SKILL_END_MODE='termination', SKILL_END_THRESHOLD='0.5',
                                      SKILL_END_PROGRESS_THRESHOLD='1', INFERENCE_SKILL_MAX_LENGTH='0')
                    checkpoint_config = json.loads((Path(spec['policy_path'])/'config.json').read_text())
                    base = SimpleNamespace(use_amp=checkpoint_config.get('use_amp', False),
                                           n_action_steps=recipe['n_action_steps'])
                    cfg = SimpleNamespace(policy=base, env=env_cfg,
                                          rename_map={'observation.images.image2': 'observation.images.wrist_image'})
                    context = run_eval._build_context(spec, cfg, torch.device('cuda'))
                wrapper, pre, post = (context[k] for k in ['policy','preprocessor','postprocessor'])
                env_pre, env_post = make_env_pre_post_processors(env_cfg, context['config'])
                selected = copy.deepcopy(skill)
                # Hold this one skill throughout the bounded probe, including recovery time.
                selected['gt_length'] = recipe['max_skill_steps']+1
                wrapper.set_forced_skill_token_sequences([[selected]])
                grounders = [s for s in pre.steps if callable(getattr(s,'set_reference_xyz',None))]
                if context['config'].proprio_grounding == 'episode_start_xyz' and len(grounders) != 1:
                    raise RuntimeError('Expected exactly one episode grounding processor')
                for grounder in grounders: grounder.set_reference_xyz(states[0,:3])
                wrapper.gripper_signal_enabled = True
                wrapper.set_capture_fullskill_video(recipe.get('video', {}).get('fullskill_overlay', True))
                diagnostic_post = copy.deepcopy(post)
            frames, executed, chunks = [], [], []
            success = False
            horizon = end-start if control else recipe['max_skill_steps']
            for step in range(horizon):
                frames.append(np.concatenate([obs['pixels']['image'],obs['pixels']['image2']],axis=1))
                if control:
                    action = actions[start+step]
                else:
                    def batch(value):
                        return {k:batch(v) for k,v in value.items()} if isinstance(value,dict) else np.expand_dims(value,0)
                    inputs = preprocess_observation(batch(obs)); inputs['task'] = [INSTRUCTION]
                    inputs = pre(env_pre(inputs))
                    with torch.inference_mode(), torch.autocast('cuda', enabled=context['config'].use_amp):
                        prediction = wrapper.select_action(inputs)
                    action_tensor = env_post({'action':post(prediction)})['action']
                    action = action_tensor[0].detach().float().cpu().numpy()
                    wrapper.record_executed_action(action_tensor)
                    for chunk in wrapper.pop_gripper_action_chunks():
                        values = diagnostic_post(chunk['actions'].reshape(-1,chunk['actions'].shape[-1]))
                        chunks.append(dict(step=step,actions=values.detach().float().cpu().numpy().tolist()))
                executed.append(action.tolist())
                obs, _, _, _, step_info = env.step(action)
                success |= bool(step_info['is_success'])
                if success: break
                if (step+1)%25 == 0: print(f'{label}: {step+1}/{horizon}', flush=True)
            frames.append(np.concatenate([obs['pixels']['image'],obs['pixels']['image2']],axis=1))
            name = label+'.mp4'
            temporary = src/(label+'.tmp.mp4')
            write_video(temporary,np.stack(frames),20)
            temporary.replace(src/name)
            result = dict(label=label, kind='gt_control' if control else 'vsa', success=success,
                          steps=len(executed), video=name, boundary_xyz_error_m=xyz_error,
                          boundary_repeat_max_error=boundary_difference, actions=executed, chunks=chunks)
            if not control: result['skill_trace'] = wrapper.get_skill_trace()
            result['top_camera_transform'] = env.camera_transform(height=env_cfg.observation_height,
                                                                  width=env_cfg.observation_width).tolist()
            render_overlay(result, src, result['top_camera_transform'],
                           require_plan=not control and recipe.get('video', {}).get('fullskill_overlay', True),
                           uv_enabled=recipe.get('video', {}).get('uv_overlay', True))
            results.append(result); report()
            print(f'{label}: success={success}, {len(executed)} steps', flush=True)
            if control and not success: raise RuntimeError('GT suffix control failed; VSA trials would be inconclusive')
        finally:
            env.close()
    report(publish=True)
    print(f'Report: {output}/index.html', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path)
    parser.add_argument('--settings',type=Path)
    args = parser.parse_args()
    if args.freeze: freeze(args.freeze)
    elif args.settings:
        payload = json.loads(args.settings.read_text())
        if payload['recipe'].get('initialization', 'gt_prefix') == 'rollout_entry':
            from rollout_entry_probe import run as run_entry
            run_entry(args.settings)
        elif payload['recipe'].get('initialization') == 'transition_timing':
            from transition_timing_probe import run as run_transition
            run_transition(args.settings)
        else:
            run(args.settings)
    else: parser.error('Use --freeze YAML or --settings JSON')
