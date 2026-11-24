"""GT-prefix skill-chain probe: exact action boundaries versus Terminator.

Only the transition rule differs within each seed pair. Both suffixes use the
same deployed VSA, GT skill/UV, original episode grounding and final-skill hold.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace


def run(settings_path):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import torch
    from lerobot.envs.configs import MimicGenEnv
    from lerobot.envs.mimicgen import MimicGenGymEnv, INSTRUCTION
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.envs.utils import preprocess_observation
    from lerobot.utils.io_utils import write_video
    from lerobot.utils.eval_gpu_memory import apply_limit
    from lerobot.utils.random_utils import set_seed
    from skill_start_probe import save
    from probe_presentation import render_overlay, write_report
    import run_eval

    payload = json.loads(settings_path.read_text())
    recipe, spec, settings = payload['recipe'], payload['spec'], payload['resolved_settings']
    output = Path(payload['output'])
    src = output / 'src'
    src.mkdir(parents=True, exist_ok=True)
    if (src / 'results.json').exists():
        raise FileExistsError(f'{output}: choose a new output_name to preserve earlier results')
    save(src / 'settings.json', payload)
    torch.set_num_threads(int(recipe['slurm']['cpus']))
    apply_limit()
    env_keys = {
        'SKILL_END_MODE': ('skill_end_mode', 'termination'),
        'SKILL_END_THRESHOLD': ('skill_end_threshold', '.5'),
        'SKILL_END_PROGRESS_THRESHOLD': ('skill_end_progress_threshold', '.95'),
        'INFERENCE_SKILL_MAX_LENGTH': ('inference_skill_max_length', '200'),
        'IMMEDIATE_REPLAN_ON_SKILL_END': ('immediate_replan_on_skill_end', 'true'),
        'GT_TERMINATION_MIN_FRACTION': ('gt_termination_min_fraction', '.5'),
        'MIN_SKILL_STEPS': ('min_skill_steps', '0'),
    }
    for env_key, (key, default) in env_keys.items():
        os.environ[env_key] = str(settings.get(key, default)).lower()
    if os.environ['IMMEDIATE_REPLAN_ON_SKILL_END'] != 'true':
        raise ValueError('This experiment requires immediate replan at the transition')
    records = run_eval.load_episode_exact_data(
        spec['skill_dataset_dir'], spec['eval_init_states_path'], 'Coffee_D0')
    record = next(r for r in records[0] if r['episode_index'] == recipe['episode_index'])
    target = recipe['skill_index']
    if not 0 <= target < len(record['skills']) - 1:
        raise ValueError('Select a starting skill with at least one following skill')
    selected_skills = record['skills'][target:]
    durations = [int(s['gt_length']) for s in selected_skills[:-1]]
    if any(length <= 0 for length in durations):
        raise ValueError('GT transition lengths must be positive')
    boundaries = np.cumsum(durations).tolist()
    final_cursor = len(selected_skills) - 1
    start = sum(int(s['gt_length']) for s in record['skills'][:target])
    raw = Path(spec['raw_dataset_dir'])
    tables = [pq.read_table(path, filters=[('episode_index', '=', recipe['episode_index'])],
                           columns=['frame_index', 'action', 'observation.state'])
              for path in sorted(raw.glob('data/**/*.parquet'))]
    rows = pd.concat([table.to_pandas() for table in tables if len(table)]).sort_values('frame_index')
    if rows.frame_index.tolist() != list(range(len(rows))):
        raise ValueError('Source episode must have contiguous frame indices')
    actions, source_states = np.stack(rows.action), np.stack(rows['observation.state'])
    if start + sum(durations) >= len(actions):
        raise ValueError('Source episode does not contain the selected skill chain')
    limit = recipe['max_skill_steps']
    insertion_limit = int(os.environ['INFERENCE_SKILL_MAX_LENGTH'])
    if insertion_limit <= 0:
        raise ValueError('The Terminator condition needs a finite insertion timeout')
    env_cfg = MimicGenEnv(dataset_dir=str(raw), terminate_on_success=False,
                         episode_length=start + max(sum(durations), insertion_limit*final_cursor) + limit + 2)
    base = json.loads((Path(spec['policy_path']) / 'config.json').read_text())
    cfg = SimpleNamespace(policy=SimpleNamespace(use_amp=base.get('use_amp', False),
                                                 n_action_steps=recipe['n_action_steps']),
                          env=env_cfg, rename_map={'observation.images.image2': 'observation.images.wrist_image'})
    context = run_eval._build_context(spec, cfg, torch.device('cuda'))
    wrapper, pre, post = (context[key] for key in ['policy', 'preprocessor', 'postprocessor'])
    env_pre, env_post = make_env_pre_post_processors(env_cfg, context['config'])
    grounders = [p for p in pre.steps if callable(getattr(p, 'set_reference_xyz', None))]
    if context['config'].proprio_grounding == 'episode_start_xyz' and len(grounders) != 1:
        raise RuntimeError('Expected one episode grounding processor')
    wrapper.gripper_signal_enabled = True
    wrapper.set_capture_fullskill_video(recipe.get('video', {}).get('fullskill_overlay', True))
    info = dict(source_episode=recipe['episode_index'], skill_index=target,
                token=record['skills'][target]['token'], next_token=record['skills'][target+1]['token'],
                skill_numbers=list(range(target+1, len(record['skills'])+1)),
                tokens=[s['token'] for s in selected_skills],
                frame_start=start, gt_transition_actions=boundaries, gt_skill_durations=durations,
                initialization='transition_timing', recipe=recipe,
                protocol=f'GT actions [0,{start}) reconstruct the selected skill start. '
                         f'A switches after cumulative action counts {boundaries}; '
                         'B uses the configured Terminator at each nonfinal skill, including its timeout. '
                         f'Both hold the final GT skill/UV without Terminator for up to {limit} actions. '
                         'Original episode-start grounding and paired sampling seeds are retained.',
                transition_environment={key: os.environ[key] for key in env_keys})
    results, pairs = [], []

    def publish(final=False):
        save(src / 'results.json', dict(**info, results=results, pairs=pairs))
        if final or not (output / 'index.html').exists():
            write_report(output, info, results)

    def batch(value):
        return {key: batch(v) for key, v in value.items()} if isinstance(value, dict) else np.expand_dims(value, 0)

    publish()
    reference = None
    for trial in range(recipe['n_trials']):
        seed = recipe['seed'] + trial
        pair_results = []
        for mode in ['fixed', 'terminator']:
            condition = 'A_GT' + '_'.join(map(str, durations)) if mode == 'fixed' else 'B_Terminator'
            label = f'seed{seed}_{condition}'
            env = MimicGenGymEnv(env_cfg)
            try:
                env.init_states = True
                env._init_states = [dict(state=record['init_state'], model_xml=record['model_xml'])]
                obs, _ = env.reset(seed=1)
                origin = env._last['state'][:3].copy()
                for action in actions[:start]:
                    obs, _, _, _, _ = env.step(action)
                snapshot = env._rpc('snapshot')
                initial_image = np.concatenate([obs['pixels']['image'], obs['pixels']['image2']], axis=1)
                if reference is None:
                    reference = (snapshot, initial_image.copy())
                    save(src / 'gt_skill_start.json', snapshot)
                errors = {key: float(np.max(np.abs(np.asarray(snapshot[key]) - np.asarray(reference[0][key]))))
                          for key in ['sim_state', 'qacc_warmstart', 'body_positions', 'controller_initial_joints']}
                errors['image_pixel'] = int(np.max(np.abs(initial_image.astype(int) - reference[1].astype(int))))
                xyz_error = float(np.max(np.abs(env._last['state'][:3] - source_states[start, :3])))
                if any(v > 1e-7 for key, v in errors.items() if key != 'image_pixel') or errors['image_pixel'] or xyz_error > .01:
                    raise RuntimeError(f'GT skill-start validation failed: {errors}, xyz={xyz_error}m')
                sequence = copy.deepcopy(selected_skills)
                hold_length = insertion_limit*final_cursor + sum(durations) + limit + 10
                # A's boundary is enforced by executed-action count below, not
                # the wrapper's observation counter (which fires one frame early).
                if mode == 'fixed':
                    for item in sequence[:-1]:
                        item['gt_length'] = hold_length
                sequence[-1]['gt_length'] = hold_length
                wrapper.advance_mode = 'gt' if mode == 'fixed' else 'external'
                wrapper.max_skill_length = 0 if mode == 'fixed' else insertion_limit
                wrapper.set_forced_skill_token_sequences([sequence])
                for grounder in grounders:
                    grounder.set_reference_xyz(origin)
                # Initial model noise is paired across A/B, independent of model
                # construction and environment reset's random-number consumption.
                set_seed(seed)
                frames, executed, states, joints, chunks, snapshots = [], [], [], [], [], []
                diagnostic_post = copy.deepcopy(post)
                transition_step = transition_snapshot = transition_reason = None
                transitions = []
                success = False
                horizon = (sum(durations) if mode == 'fixed' else insertion_limit*final_cursor) + limit + 1
                print(f'{label}: GT frame {start}, source XYZ max error={xyz_error:.6f}m', flush=True)
                for step in range(horizon):
                    cursor = wrapper._cursor[0]
                    if mode == 'fixed' and cursor < final_cursor and step == boundaries[cursor]:
                        wrapper._pending_advance.add(0)
                    previous_cursor = wrapper._cursor[0]
                    inputs = preprocess_observation(batch(obs))
                    inputs['task'] = [INSTRUCTION]
                    inputs = pre(env_pre(inputs))
                    with torch.inference_mode(), torch.autocast('cuda', enabled=context['config'].use_amp):
                        prediction = wrapper.select_action(inputs)
                    tensor = env_post({'action': post(prediction)})['action']
                    action = tensor[0].detach().float().cpu().numpy()
                    if wrapper._cursor[0] != previous_cursor:
                        next_cursor = wrapper._cursor[0]
                        if next_cursor != previous_cursor + 1:
                            raise RuntimeError('Expected one skill advance per action')
                        current_step = len(executed)
                        current_snapshot = env._rpc('snapshot')
                        prior = wrapper.get_skill_trace()[previous_cursor]
                        fired_prob = prior.get('end_probs', [])[-1].get('prob') if prior.get('end_probs') else None
                        reason = ('exact_action_count' if mode == 'fixed' else
                                             'probability' if fired_prob is not None and fired_prob >= wrapper.end_threshold else
                                             'timeout')
                        if mode == 'fixed' and current_step != boundaries[previous_cursor]:
                            raise RuntimeError(f'Wrong fixed boundary: {current_step} != {boundaries[previous_cursor]}')
                        transitions.append(dict(step=current_step, from_skill_number=target+previous_cursor+1,
                                                to_skill_number=target+next_cursor+1, reason=reason,
                                                probability=fired_prob, snapshot=current_snapshot))
                        if next_cursor == final_cursor:
                            transition_step, transition_snapshot, transition_reason = current_step, current_snapshot, reason
                            wrapper.advance_mode = 'gt'
                            wrapper.max_skill_length = 0
                            wrapper._pending_episode_done.clear()
                        print(f'{label}: skill {target+next_cursor+1} at action {current_step}, reason={reason}, prob={fired_prob}', flush=True)
                    frames.append(np.concatenate([obs['pixels']['image'], obs['pixels']['image2']], axis=1))
                    states.append(env._last['state'].tolist())
                    joints.append(env._last['joints'].tolist())
                    if step % recipe['n_action_steps'] == 0:
                        snapshots.append(dict(step=step, snapshot=env._rpc('snapshot')))
                    wrapper.record_executed_action(tensor)
                    for chunk in wrapper.pop_gripper_action_chunks():
                        values = diagnostic_post(chunk['actions'].reshape(-1, chunk['actions'].shape[-1]))
                        chunks.append(dict(step=step, actions=values.detach().float().cpu().numpy().tolist()))
                    executed.append(action.tolist())
                    obs, _, _, _, step_info = env.step(action)
                    success |= bool(step_info['is_success'])
                    if success or (transition_step is not None and len(executed) - transition_step >= limit):
                        break
                    if (step + 1) % 25 == 0:
                        print(f'{label}: {step+1} actions, skill {wrapper._cursor[0]}', flush=True)
                if transition_step is None and not success:
                    raise RuntimeError('No transition to the final skill within the bounded horizon')
                frames.append(np.concatenate([obs['pixels']['image'], obs['pixels']['image2']], axis=1))
                temp = src / (label + '.tmp.mp4')
                write_video(temp, np.stack(frames), 20)
                temp.replace(src / (label + '.mp4'))
                trace = copy.deepcopy(wrapper.get_skill_trace())
                for item in trace:
                    item['source_skill_index'] = target + item['skill_index']
                result = dict(label=label, kind='vsa', seed=seed, condition=mode, success=success,
                              steps=len(executed), video=label+'.mp4', boundary_xyz_error_m=xyz_error,
                              boundary_repeat_max_error=max(errors.values()), replay_validation=errors,
                              transition_step=transition_step, transition_reason=transition_reason,
                              transitions=transitions, sim_fps=20,
                              termination_threshold=wrapper.end_threshold,
                              termination_graph=mode == 'terminator',
                              final_skill_steps=0 if transition_step is None else len(executed)-transition_step,
                              transition_snapshot=transition_snapshot, end_snapshot=env._rpc('snapshot'),
                              states=states, joints=joints, actions=executed, chunks=chunks,
                              snapshots=snapshots, skill_trace=trace,
                              top_camera_transform=env.camera_transform(height=env_cfg.observation_height,
                                                                         width=env_cfg.observation_width).tolist())
                render_overlay(result, src, result['top_camera_transform'],
                               require_plan=recipe.get('video', {}).get('fullskill_overlay', True),
                               uv_enabled=recipe.get('video', {}).get('uv_overlay', True))
                results.append(result)
                pair_results.append(result)
                publish()
                print(f'{label}: success={success}, transition={transition_step}, final steps={result["final_skill_steps"]}', flush=True)
            finally:
                env.close()
        a, b = pair_results
        overlap = min(a['transitions'][0]['step'] if a['transitions'] else a['steps'],
                      b['transitions'][0]['step'] if b['transitions'] else b['steps'])
        shared_diff = float(np.max(np.abs(np.asarray(a['actions'][:overlap])-np.asarray(b['actions'][:overlap])))) if overlap else None
        pair = dict(seed=seed, fixed_success=a['success'], terminator_success=b['success'],
                    fixed_transition_step=a['transition_step'], terminator_transition_step=b['transition_step'],
                    fixed_transitions=[t['step'] for t in a['transitions']],
                    terminator_transitions=[t['step'] for t in b['transitions']],
                    terminator_reason=b['transition_reason'], pre_transition_action_max_difference=shared_diff)
        pairs.append(pair)
        publish(final=True)
        print(f'Paired result: {json.dumps(pair)}', flush=True)
    publish(final=True)
    print(f'Report: {output}/index.html', flush=True)
