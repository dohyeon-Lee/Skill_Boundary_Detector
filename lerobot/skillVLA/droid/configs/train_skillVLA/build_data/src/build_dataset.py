"""Encode a selected DROID FSQ and attach calibrated SkillVLA training targets.

Controller action/state stay joint-velocity8/joint-position8. Spatial targets
use measured EEF pose6, independently of the selected FSQ representation.
Original images, calibration, frame IDs and episode IDs are never rewritten.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'lerobot/src/lerobot').is_dir())
CONFIGS = ROOT / 'lerobot/skillVLA/droid/configs'
SHARED = ROOT / 'lerobot/skillVLA/libero/configs/train_skillVLA/build_data/src'
FSQ_PLAN = runpy.run_path(str(CONFIGS / 'train_skills/FSQ/eval/src/config.py'))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def settings(config):
    requested = str(config.get('fsq_checkpoint', '')).strip()
    if not requested:
        raise ValueError('Select fsq_checkpoint in train_skillVLA/build_data/train_skillVLA_config.yaml first')
    project = Path(config.get('project_root', ROOT))
    checkpoint = Path(requested).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = project / checkpoint
    match = re.fullmatch(r'FSQ_epoch(\d+)\.pt', checkpoint.name)
    if not match:
        raise ValueError('fsq_checkpoint must name a numeric FSQ_epochXXXX.pt checkpoint')
    expected = project / config['outputs_root'] / 'FSQ' / checkpoint.parent.name / checkpoint.name
    if checkpoint.resolve() != expected.resolve():
        raise ValueError(f'FSQ checkpoint must be under configured outputs_root/FSQ: {expected}')
    plan = FSQ_PLAN['build_settings'](dict(config, models=[checkpoint.parent.name],
        checkpoint=int(match[1]), camera='exterior_1', output_name='build_preflight', device='cpu'))
    row = plan['models'][0]
    levels = ''.join(map(str, row['levels']))
    group = config.get('droid_dataset', {}).get('group', 'all')
    suffix = '' if group == 'all' else '_' + group
    name = config.get('output_name') or f'FSQ{levels}_{row["action_space"]}_{int(match[1])}_pt{suffix}'
    name = FSQ_PLAN['folder'](name)
    output = project / config['dataset_root'] / 'skillvla' / config['target_dataset'] / name
    return dict(row, output=str(output), output_name=name,
        cpus=plan['cpus'], memory=plan['memory'], time=plan['time'],
        partition=plan['partition'], qos=plan['qos'], nodelist=plan['nodelist'], exclude=plan['exclude'])


def camera_matrix(dataset, episode):
    """Embed a native, letterbox-adjusted pinhole camera in the shared projector.

    Its historical LIBERO x-mirror is canceled in the stored matrix, keeping
    DROID's actual video pixels and calibrated UV completely unchanged.
    """
    with np.load(dataset / f'meta/droid/episodes/{episode:06d}.npz', allow_pickle=False) as calibration:
        intrinsics = np.asarray(calibration['exterior_1.intrinsics'], dtype=np.float64)
        poses = np.asarray(calibration['exterior_1.camera_to_base'], dtype=np.float64)
    if poses.ndim == 2:
        poses = poses[None]
    if not np.isfinite(poses).all() or not np.allclose(poses, poses[0], atol=1e-6):
        raise ValueError(f'DROID primary exterior camera must be static within episode {episode}')
    info = json.loads((dataset / 'meta/info.json').read_text())
    shape = info['features']['observation.images.image']['shape']
    height, width = (shape[:2] if shape[-1] == 3 else shape[-2:])
    if intrinsics.ndim != 2 or intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError(f'Invalid resized DROID intrinsics: episode={episode}')
    projection = np.eye(4)
    projection[:3, :3] = intrinsics
    mirror = np.eye(4)
    mirror[0, 0], mirror[0, 2] = -1, width - 1
    return mirror @ projection @ np.linalg.inv(poses[0]), int(height), int(width)


def projection_diagnostics(xyz, matrix, height, width):
    """Unclipped native pixels and an explicit nearest-border clipping flag."""
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    projected = np.c_[xyz, np.ones(len(xyz))] @ matrix.T
    front = np.isfinite(projected).all(axis=1) & (projected[:, 2] > 1e-8)
    depth = np.where(front, projected[:, 2], 1.)
    pixels = np.c_[width - 1 - projected[:, 0]/depth, projected[:, 1]/depth]
    pixels[~front] = 0
    clipped = front & ((pixels < 0).any(axis=1) | (pixels > [width-1, height-1]).any(axis=1))
    return pixels.astype(np.float32), clipped


def build(plan, *, device='cuda'):
    import torch
    from lerobot.utils import gpu_memory_guard as guard
    torch.set_num_threads(min(int(plan['cpus']), 8))
    output = Path(plan['output'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_name('.' + output.name + '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This SkillVLA build is already running; no duplicate started') from None
        return _build(plan, device, guard)


def _build(plan, device, guard):
    import pandas as pd
    import torch
    source, output = Path(plan['dataset_dir']), Path(plan['output'])
    contract = dict(format='droid_skillvla_v1', fsq_sha256=sha(plan['model_path']),
        fsq_training_contract_sha256=sha(plan['contract_path']),
        skillset_sha256=sha(Path(plan['skillset']) / 'skillset_manifest.json'),
        source_info_sha256=sha(source / 'meta/info.json'),
        source_stats_sha256=sha(source / 'meta/stats.json'),
        action_space=plan['action_space'], levels=plan['levels'], episode_ids=plan['episode_ids'],
        controller_action='joint_velocity7_absolute_gripper1', controller_state='joint_position7_gripper_state1',
        spatial_state='observation.states.ee_state', spatial_rotation='rotvec',
        videos_reencoded=False, gripper_switch_threshold=0.5)
    def check_owner(folder):
        owner = folder / 'build_contract.json'
        if not owner.is_file() or json.loads(owner.read_text()) != contract:
            raise FileExistsError(f'Output has another or unknown owner; preserve and inspect: {folder}')
    if output.exists():
        check_owner(output)
        verify(output)
        return output
    work = output.with_name('.' + output.name + '.building')
    if work.exists():
        check_owner(work)
    work.mkdir(exist_ok=True)
    atomic_json(work / 'build_contract.json', contract)
    shared_fsq = ROOT / 'lerobot/skillVLA/libero/configs/train_skills/FSQ/src'
    sys.path.insert(0, str(shared_fsq))
    encoder = module(SHARED / 'encode_FSQ_skills.py', 'droid_skillvla_encode')
    attach = module(SHARED / 'add_skill_latents_to_dataset.py', 'droid_skillvla_attach')
    from FSQ import load_fsq_encoder
    from train_FSQ import load_skill_files
    from action_representation import prepare_fsq_actions
    latents_path = work / 'skill_latents.npz'
    if not latents_path.is_file():
        print(f'Loading FSQ {plan["model_path"]}; source skills={plan["skillset"]}', flush=True)
        if plan['action_space'] == 'delta_pose':
            adapter = runpy.run_path(str(CONFIGS / 'src/fsq_action_data.py'))
            _, states, actions, metadata = adapter['load_pose_skills'](
                Path(plan['skillset']) / 'skills', source, episode_ids=plan['episode_ids'])
            trajectories = [prepare_fsq_actions(a, s, 'delta_pose', [6]) for a, s in zip(actions, states, strict=True)]
        else:
            _, states, trajectories, metadata = load_skill_files(
                Path(plan['skillset']) / 'skills', episode_ids=plan['episode_ids'])
        if not metadata:
            raise ValueError('No selected DROID skills')
        model, cfg = load_fsq_encoder(plan['model_path'], device)
        expected_mode = 'delta_pose' if plan['action_space'] == 'delta_pose' else 'dataset'
        if (cfg.action_space != expected_mode or list(cfg.fsq_levels) != plan['levels']
                or cfg.action_dim != (7 if expected_mode == 'delta_pose' else 8)):
            raise ValueError('FSQ weights and saved action/level contract disagree')
        with torch.inference_mode():
            latents, tokens = encoder._encode_trajectory_batch(model, trajectories, device=device, batch_size=256)
        guard.mark_ready()
        arrays = dict(latents=latents, tokens=tokens,
            skill_order=np.asarray(encoder._compute_skill_orders(metadata), dtype=np.float32))
        for key in ('episode_id', 'task_id', 'skill_index', 'frame_start', 'frame_end', 'length'):
            arrays[key] = np.asarray([m[key] for m in metadata])
        temporary = latents_path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, **arrays)
        temporary.replace(latents_path)
        del model
        print(f'Encoded {len(metadata)} skills into {latents_path}', flush=True)
    guard.mark_ready()
    skill_map = attach.load_skill_map(latents_path)
    max_order = max(map(len, skill_map.values()))
    observed_length = max(fe - fs for skills in skill_map.values() for fs, fe, _ in skills)
    max_length, code_count = max(500, observed_length), int(np.prod(plan['levels']))
    pmax = int(np.ceil(max_length * .30))
    dataset = work / 'skillvla'
    dataset.mkdir(exist_ok=True)
    # Copy metadata, never hardlink writable files into the source.
    shutil.copytree(source / 'meta', dataset / 'meta', dirs_exist_ok=True)
    if not (dataset / 'videos').exists():
        (dataset / 'videos').symlink_to(os.path.relpath(source / 'videos', output / 'skillvla'), target_is_directory=True)
    info = json.loads((source / 'meta/info.json').read_text())
    if (info['fps'] != 15 or info['features']['observation.state']['shape'] != [8]
            or info['features']['action']['shape'] != [8]):
        raise ValueError('DROID SkillVLA requires the 15Hz joint-velocity8/joint-position8 view')
    states, actions_for_jitter, numeric, idle_rows, focus, cameras = {}, {}, {}, [], [], []
    idle = runpy.run_path(str(CONFIGS / 'src/training_filter.py'))
    from lerobot.policies.skillVLA.chunk_patch_target import project_top_points
    count = 0
    source_files = sorted((source / 'data').rglob('*.parquet'))
    print(f'Annotating {len(source_files)} source parquet files; only selected episodes train', flush=True)
    for file_number, file in enumerate(source_files, 1):
        frame = pd.read_parquet(file)
        parts = []
        for ep, rows in frame.groupby('episode_index', sort=False):
            ep = int(ep)
            if not np.array_equal(rows['frame_index'], np.arange(len(rows))):
                raise ValueError(f'Expected one complete, ordered episode per parquet: {file}')
            cols = attach.compute_skill_columns(rows, skill_map.get(ep, []), max_order, max_length, code_count, code_count + 1)
            primary = np.stack(rows['observation.state']).astype(np.float32)
            for key, value in cols.items():
                rows[key] = list(value)
            rows['skill_decoder_state'] = list(primary)
            parts.append(rows)
            if ep not in skill_map:
                continue
            eef = np.stack(rows['observation.states.ee_state']).astype(np.float32)
            commands = np.stack(rows['action']).astype(np.float32)
            if eef.shape != (len(rows), 6) or not np.isfinite(eef).all():
                raise ValueError(f'DROID EEF pose6 unavailable: episode={ep}')
            for fs, fe, _ in skill_map[ep]:
                if not 0 <= fs < fe <= len(rows):
                    raise ValueError(f'Skill outside source episode: episode={ep}, {fs}:{fe}')
            states[ep] = eef
            actions_for_jitter[ep] = commands.copy()
            actions_for_jitter[ep][:, -1] = 2 * commands[:, -1] - 1  # comparison only; controllers stay [0,1]
            for key in ('action', 'observation.state', 'observation.states.ee_state'):
                numeric.setdefault(key, []).append(np.stack(rows[key]))
            keep = idle['raw_episode_mask'](source, ep)
            if keep.shape != (len(rows),):
                raise ValueError(f'Idle frame mapping differs from episode {ep}')
            idle_rows.append((ep, keep))
            matrix, height, width = camera_matrix(source, ep)
            cameras.append((ep, matrix))
            for rank, (fs, fe, _) in enumerate(skill_map[ep]):
                start_uv, start_valid = project_top_points(eef[fs:fs+1, :3], matrix, height=height, width=width)
                endpoint = min(fe, len(rows) - 1)
                end_uv, end_valid = project_top_points(eef[endpoint:endpoint+1, :3], matrix, height=height, width=width)
                raw_pixels, clipped = projection_diagnostics(eef[endpoint:endpoint+1, :3], matrix, height, width)
                focus.append((ep, rank, fs, fe, start_uv[0], start_valid[0], end_uv[0], end_valid[0], raw_pixels[0], clipped[0]))
        target = dataset / file.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.tmp.parquet')
        pd.concat(parts).to_parquet(temporary, index=False)
        temporary.replace(target)
        count += len(frame)
        if file_number % 100 == 0:
            print(f'DROID SkillVLA annotation: {file_number}/{len(source_files)} files, {len(states)} selected episodes', flush=True)
    if set(states) != set(skill_map) or count != info['total_frames']:
        raise ValueError('Selected skill episodes or source frames are missing')
    print(f'Preparing EEF skill-start windows and gripper-safe jitter: pmax={pmax}', flush=True)
    attach.build_skill_initial_state_npz(skill_map, states, actions_for_jitter, pmax, 6,
        work / 'skill_initial_state.npz', 'half_normal', 30., True, 'none')
    np.savez_compressed(work / 'nonidle_frames.npz',
        episode_id=np.concatenate([np.full(len(m), ep, dtype=np.int32) for ep, m in idle_rows]),
        frame_index=np.concatenate([np.arange(len(m), dtype=np.int32) for _, m in idle_rows]),
        keep=np.concatenate([m for _, m in idle_rows]))
    uv = np.stack([r[6] for r in focus])
    pixels = (uv + 1) / 2 * [width - 1, height - 1]
    np.savez_compressed(work / 'skill_focus_uv.npz',
        episode_id=np.array([r[0] for r in focus], dtype=np.int32),
        skill_index=np.array([r[1] for r in focus], dtype=np.int32),
        frame_start=np.array([r[2] for r in focus], dtype=np.int32),
        frame_end=np.array([r[3] for r in focus], dtype=np.int32),
        focus_uv=uv, focus_uv_pixels=pixels, focus_uv_raw_pixels=np.stack([r[8] for r in focus]),
        focus_valid=np.array([r[7] for r in focus]), focus_clipped=np.array([r[9] for r in focus]),
        start_focus_uv=np.stack([r[4] for r in focus]),
        start_focus_valid=np.array([r[5] for r in focus]),
        camera=np.str_('exterior_1'), image_height=height, image_width=width,
        camera_episode_id=np.array([r[0] for r in cameras], dtype=np.int32),
        camera_grounded_transform=np.stack([r[1] for r in cameras]))
    stats = json.loads((source / 'meta/stats.json').read_text())
    for key, arrays in numeric.items():
        stats[key] = attach.exact_vector_stats(np.concatenate(arrays))
    stats['skill_decoder_state'] = stats['observation.state']
    atomic_json(dataset / 'meta/stats.json', stats)
    # Decode only the training exterior + wrist, retaining the third source camera on disk.
    info['features'].pop('observation.images.image2', None)
    for key, values in cols.items():
        dim = np.asarray(values).shape[1:] or (1,)
        info['features'][key] = dict(dtype=str(np.asarray(values).dtype), shape=list(dim), names=None)
    info['features']['skill_decoder_state'] = dict(dtype='float32', shape=[8], names=None)
    info.update(repo_id=f'droid/{output.name}', environment_type='droid',
        skill_fsq_levels=plan['levels'], skill_num_embeddings=code_count, skill_code_space_id=contract['fsq_sha256'],
        skill_eos_token_id=code_count, skill_pad_token_id=code_count+1, skill_vocab_size=code_count+2,
        skill_output_vocab_size=code_count+1, skill_max_order=max_order, skill_observed_max_order=max_order,
        skill_max_length=max_length, skill_observed_max_length=observed_length, skill_sequence_size=max_order+1,
        skill_pmax=pmax, skill_jitter_max_percent=30., skill_jitter_gripper_validity=True,
        skill_jitter_distribution='half_normal', skill_initial_state_path='skill_initial_state.npz',
        skill_initial_state_window=2*pmax+1, skill_focus_uv_path='skill_focus_uv.npz',
        skill_focus_uv_camera='exterior_1', skill_focus_uv_normalization='minus_one_to_one',
        skill_spatial_state_key='observation.states.ee_state', skill_gripper_action_index=7,
        skill_gripper_action_range=[0., 1.], wrist_camera_mirror_x=False, proprio_grounding='none',
        droid_nonidle_path='nonidle_frames.npz', droid_nonidle_sha256=sha(work/'nonidle_frames.npz'),
        training_excluded_episode_ids=sorted(set(range(info['total_episodes'])) - set(states)),
        training_total_episodes=len(states), training_total_frames=sum(len(v) for v in states.values()))
    atomic_json(dataset / 'meta/info.json', info)
    shutil.copy2(plan['model_path'], work / 'FSQ.pt')
    # Auxiliary resolver uses the same token artifact under this historical name.
    shutil.copy2(latents_path, work / 'skill_code_freq_source.npz')
    marker = dict(verified=True, episode_count=len(states), skill_count=len(focus), total_frames=count,
        fsq_sha256=contract['fsq_sha256'], contract_sha256=sha(work / 'build_contract.json'),
        artifacts={str(p.relative_to(work)): sha(p) for p in
            (work/'skill_latents.npz', work/'skill_initial_state.npz', work/'skill_focus_uv.npz',
             work/'nonidle_frames.npz', work/'FSQ.pt', dataset/'meta/info.json', dataset/'meta/stats.json')})
    atomic_json(work / 'build_verified.json', marker)
    work.replace(output)
    verify(output)
    print(f'DROID SkillVLA build verified: {output} ({len(states)} episodes, {len(focus)} skills)', flush=True)
    return output


def verify(output):
    output = Path(output)
    saved = json.loads((output / 'build_verified.json').read_text())
    if not saved.get('verified') or saved['contract_sha256'] != sha(output / 'build_contract.json'):
        raise ValueError('Invalid DROID SkillVLA completion contract')
    for name, digest in saved['artifacts'].items():
        if sha(output / name) != digest:
            raise ValueError(f'DROID SkillVLA artifact changed: {output / name}')
    if not (output/'skillvla/videos').is_dir():
        raise FileNotFoundError('Source video link is unavailable')
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    plan = settings(FSQ_PLAN['load_config'](args.config))
    if args.run:
        from types import SimpleNamespace
        from lerobot.utils import gpu_memory_guard as guard
        if args.device == 'cuda':
            exit_code = guard.supervise_if_needed(SimpleNamespace(policy=SimpleNamespace(device='cuda')))
            if exit_code is not None:
                raise SystemExit(exit_code)
            guard.apply_budget()
        build(plan, device=args.device)
    else:
        print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
