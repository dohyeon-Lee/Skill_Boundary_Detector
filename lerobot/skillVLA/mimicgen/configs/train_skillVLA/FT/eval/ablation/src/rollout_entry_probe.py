"""Paired continuation/reset probe at a policy-generated Coffee skill boundary.

Replay actual executed actions from the verified initial state, rather than
restoring MuJoCo state alone (which omits controller and gripper history).
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace


def run(settings_path):
    import numpy as np
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
    recipe, spec = payload['recipe'], payload['spec']
    if spec['advance_mode'] != 'external' or spec['skill_source'] != 'gt':
        raise ValueError('Actual prefix requires GT skills and the configured external Terminator')
    output = Path(payload['output'])
    src = output / 'src'
    src.mkdir(parents=True, exist_ok=True)
    # Never combine partial results from different invocations.
    if (src / 'results.json').exists():
        raise FileExistsError(f'{output}: choose a new output_name to preserve the prior experiment')
    save(src / 'settings.json', payload)
    torch.set_num_threads(int(recipe['slurm']['cpus']))
    apply_limit()
    records = run_eval.load_episode_exact_data(
        spec['skill_dataset_dir'], spec['eval_init_states_path'], 'Coffee_D0')
    record = next(r for r in records[0] if r['episode_index'] == recipe['episode_index'])
    target = recipe['skill_index']
    if target != len(record['skills']) - 1:
        raise ValueError('This paired probe holds the final skill; select the final skill_index')
    selected = copy.deepcopy(record['skills'][target])
    settings = payload['resolved_settings']
    transition_keys = {
        'SKILL_END_MODE': ('skill_end_mode', 'termination'),
        'SKILL_END_THRESHOLD': ('skill_end_threshold', '.5'),
        'SKILL_END_PROGRESS_THRESHOLD': ('skill_end_progress_threshold', '.95'),
        'INFERENCE_SKILL_MAX_LENGTH': ('inference_skill_max_length', '200'),
        'IMMEDIATE_REPLAN_ON_SKILL_END': ('immediate_replan_on_skill_end', 'true'),
        'GT_TERMINATION_MIN_FRACTION': ('gt_termination_min_fraction', '.5'),
        'MIN_SKILL_STEPS': ('min_skill_steps', '0'),
    }
    for env_key, (key, default) in transition_keys.items():
        os.environ[env_key] = str(settings.get(key, default)).lower()
    env_cfg = MimicGenEnv(dataset_dir=spec['raw_dataset_dir'], terminate_on_success=False,
                         episode_length=int(recipe['max_prefix_steps']) + recipe['max_skill_steps'] + 1)
    base_config = json.loads((Path(spec['policy_path']) / 'config.json').read_text())
    cfg = SimpleNamespace(
        policy=SimpleNamespace(use_amp=base_config.get('use_amp', False), n_action_steps=recipe['n_action_steps']),
        env=env_cfg, rename_map={'observation.images.image2': 'observation.images.wrist_image'})
    context = run_eval._build_context(spec, cfg, torch.device('cuda'))
    wrapper, pre, post = (context[k] for k in ['policy', 'preprocessor', 'postprocessor'])
    env_pre, env_post = make_env_pre_post_processors(env_cfg, context['config'])
    grounders = [p for p in pre.steps if callable(getattr(p, 'set_reference_xyz', None))]
    if context['config'].proprio_grounding == 'episode_start_xyz' and len(grounders) != 1:
        raise RuntimeError('Expected one episode grounding processor')
    wrapper.gripper_signal_enabled = True
    wrapper.set_capture_fullskill_video(recipe.get('video', {}).get('fullskill_overlay', True))
    info = dict(source_episode=recipe['episode_index'], skill_index=target, token=selected['token'],
                frame_start=sum(int(s['gt_length']) for s in record['skills'][:target]),
                initialization='rollout_entry', recipe=recipe,
                protocol='VSA + configured Terminator prefix; identical actual-action replay for fresh reset; '
                         'same GT skill/UV and original episode grounding; both suffixes hold the last skill without Terminator',
                prefix_transition_environment={key: os.environ[key] for key in transition_keys})
    results, pairs = [], []

    def publish(final=False):
        save(src / 'results.json', dict(**info, results=results, pairs=pairs))
        if final or not (output / 'index.html').exists():
            write_report(output, info, results)

    def reset_environment():
        env = MimicGenGymEnv(env_cfg)
        try:
            env.init_states = True
            env._init_states = [dict(state=record['init_state'], model_xml=record['model_xml'])]
            obs, _ = env.reset(seed=1)
            return env, obs
        except BaseException:
            env.close()
            raise

    def reset_policy(sequence, *, mode, origin):
        wrapper.advance_mode = mode
        wrapper.max_skill_length = int(os.environ['INFERENCE_SKILL_MAX_LENGTH']) if mode == 'external' else 0
        wrapper.set_forced_skill_token_sequences([copy.deepcopy(sequence)])
        if origin is not None:
            for grounder in grounders:
                grounder.set_reference_xyz(origin)

    def batch(value):
        return {k: batch(v) for k, v in value.items()} if isinstance(value, dict) else np.expand_dims(value, 0)

    def predict(obs):
        inputs = preprocess_observation(batch(obs))
        inputs['task'] = [INSTRUCTION]
        inputs = pre(env_pre(inputs))
        with torch.inference_mode(), torch.autocast('cuda', enabled=context['config'].use_amp):
            prediction = wrapper.select_action(inputs)
        return env_post({'action': post(prediction)})['action']

    def frame(obs):
        return np.concatenate([obs['pixels']['image'], obs['pixels']['image2']], axis=1)

    def numpy_action(tensor):
        return tensor[0].detach().float().cpu().numpy()

    def rng_state():
        import random
        return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                    numpy=np.random.get_state(), python=random.getstate())

    def restore_rng(state):
        import random
        torch.set_rng_state(state['torch'])
        torch.cuda.set_rng_state(state['cuda'])
        np.random.set_state(state['numpy'])
        random.setstate(state['python'])

    def local_trace(start):
        def shift(value):
            if isinstance(value, list):
                return [shift(v) for v in value]
            if isinstance(value, dict):
                return {k: int(v)-start if k == 'episode_timestep' else shift(v) for k, v in value.items()}
            return value
        return shift(copy.deepcopy([s for s in wrapper.get_skill_trace() if s['episode_timestep'] >= start]))

    def suffix(env, obs, label, entry_step, *, first_action=None, replay_error=0.0):
        frames, executed, states, chunks = [], [], [], []
        diagnostic_post = copy.deepcopy(post)
        success = False
        for step in range(recipe['max_skill_steps']):
            frames.append(frame(obs))
            states.append(env._last['state'].tolist())
            tensor = first_action if step == 0 and first_action is not None else predict(obs)
            action = numpy_action(tensor)
            wrapper.record_executed_action(tensor)
            for chunk in wrapper.pop_gripper_action_chunks():
                values = diagnostic_post(chunk['actions'].reshape(-1, chunk['actions'].shape[-1]))
                chunks.append(dict(step=step, actions=values.detach().float().cpu().numpy().tolist()))
            executed.append(action.tolist())
            obs, _, _, _, step_info = env.step(action)
            success |= bool(step_info['is_success'])
            if success:
                break
            if (step+1) % 25 == 0:
                print(f'{label}: {step+1}/{recipe["max_skill_steps"]}', flush=True)
        frames.append(frame(obs))
        name = label + '.mp4'
        temp = src / (label + '.tmp.mp4')
        write_video(temp, np.stack(frames), 20)
        temp.replace(src / name)
        result = dict(label=label, kind='vsa', success=success, steps=len(executed), video=name,
                      boundary_xyz_error_m=None, boundary_repeat_max_error=replay_error,
                      entry_step=entry_step, actions=executed, states=states, chunks=chunks,
                      skill_trace=local_trace(entry_step),
                      end_snapshot=env._rpc('snapshot'),
                      top_camera_transform=env.camera_transform(height=env_cfg.observation_height,
                                                                 width=env_cfg.observation_width).tolist())
        render_overlay(result, src, result['top_camera_transform'],
                       require_plan=recipe.get('video', {}).get('fullskill_overlay', True),
                       uv_enabled=recipe.get('video', {}).get('uv_overlay', True))
        results.append(result)
        publish()
        print(f'{label}: success={success}, {len(executed)} steps', flush=True)
        return result

    publish()
    for trial in range(recipe['n_trials']):
        seed = recipe['seed'] + trial
        set_seed(seed)  # Seed after loading weights; paired denoising is restored at the boundary.
        env, obs = reset_environment()
        prefix, prefix_frames = [], []
        entry_rng = None
        try:
            origin = env._last['state'][:3].copy()
            reset_policy(record['skills'], mode='external', origin=origin)
            for step in range(int(recipe['max_prefix_steps'])):
                before_rng = rng_state()
                tensor = predict(obs)
                if wrapper._cursor[0] == target:
                    entry_rng = before_rng
                    boundary = env._rpc('snapshot')
                    boundary_image = frame(obs).copy()
                    boundary_state = env._last['state'].copy()
                    print(f'seed{seed}: final skill entry at {step}; prefix actions={len(prefix)}', flush=True)
                    # Hold the same final GT code/UV in BOTH suffixes. Only the
                    # fresh branch resets the policy/processors and start caches.
                    wrapper.advance_mode = 'gt'
                    wrapper.max_skill_length = 0
                    wrapper._gt_lengths[0][target] = recipe['max_skill_steps'] + 1
                    wrapper._pending_episode_done.clear()
                    break
                prefix_frames.append(frame(obs))
                wrapper.pop_gripper_action_chunks()
                wrapper.record_executed_action(tensor)
                action = numpy_action(tensor)
                prefix.append(action.tolist())
                obs, _, _, _, step_info = env.step(action)
                if step_info['is_success']:
                    raise RuntimeError('Task succeeded before selected skill; no failed boundary to diagnose')
                if (step+1) % 25 == 0:
                    print(f'seed{seed}: prefix {step+1}/{recipe["max_prefix_steps"]}; skill {wrapper._cursor[0]}', flush=True)
            if entry_rng is None:
                raise RuntimeError('The configured Terminator did not reach the selected skill within max_prefix_steps')
            entry_step = len(prefix)
            save(src / f'seed{seed}_entry.json', dict(seed=seed, entry_step=entry_step,
                 prefix_actions=prefix, boundary=boundary, observed_state=boundary_state.tolist(),
                 grounding_origin=origin.tolist(), prefix_trace=wrapper.get_skill_trace()))
            prefix_frames.append(boundary_image)
            write_video(src / f'seed{seed}_prefix.mp4', np.stack(prefix_frames), 20)
            baseline = suffix(env, obs, f'seed{seed}_continuous', entry_step, first_action=tensor)
        finally:
            env.close()

        env, obs = reset_environment()
        try:
            for action in prefix:
                obs, _, _, _, _ = env.step(action)
            reconstructed = env._rpc('snapshot')
            # Validate object/robot configuration, velocity, controller origin,
            # solver warmstart and input images, rather than EEF pose alone.
            errors = {key: float(np.max(np.abs(np.asarray(reconstructed[key]) - np.asarray(boundary[key]))))
                      for key in ['sim_state', 'qacc_warmstart', 'body_positions', 'controller_initial_joints']}
            errors['observed_state'] = float(np.max(np.abs(env._last['state'] - boundary_state)))
            errors['image_pixel'] = int(np.max(np.abs(frame(obs).astype(int) - boundary_image.astype(int))))
            save(src / f'seed{seed}_replay_validation.json', errors)
            if any(value > 1e-7 for key, value in errors.items() if key != 'image_pixel') or errors['image_pixel']:
                raise RuntimeError(f'Actual-action replay did not reproduce the boundary: {errors}')
            held = copy.deepcopy(selected)
            held['gt_length'] = recipe['max_skill_steps'] + 1
            reset_policy([held], mode='gt', origin=origin)
            restore_rng(entry_rng)
            fresh = suffix(env, obs, f'seed{seed}_fresh_reset', 0, replay_error=errors['sim_state'])
            delta = np.asarray(baseline['actions'][0]) - np.asarray(fresh['actions'][0])
            pair = dict(seed=seed, entry_step=entry_step, continuous_success=baseline['success'],
                        reset_success=fresh['success'], replay_validation=errors,
                        first_action_max_difference=float(np.abs(delta).max()),
                        first_chunk_max_difference=float(np.max(np.abs(
                            np.asarray(baseline['chunks'][0]['actions']) - np.asarray(fresh['chunks'][0]['actions'])))),
                        prefix_video=f'seed{seed}_prefix.mp4')
            pairs.append(pair)
            publish(final=True)
            print(f'Paired result: {json.dumps(pair)}', flush=True)
        finally:
            env.close()
    publish(final=True)
    print(f'Report: {output}/index.html', flush=True)
