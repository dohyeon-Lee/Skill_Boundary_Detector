#!/usr/bin/env python3
"""Download Coffee D0 and create a storage-light, LIBERO-compatible LeRobot v3 view."""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from download import download_one, local_path, session, sha256, source_tree

HERE = Path(__file__).resolve().parent
ROOT = next(parent for parent in HERE.parents if (parent / 'lerobot/src/lerobot').is_dir())
FORMAT = 'sbd_mimicgen_libero/v1'
IMAGE_MAP = {'observation.images.agentview': 'observation.images.image',
             'observation.images.wrist': 'observation.images.wrist_image'}
VECTOR_DIMS = {'observation.state': 8, 'observation.states.ee_state': 6,
               'observation.states.joint_state': 7, 'observation.states.gripper_state': 2,
               'action': 7}
DEFAULT_KEYS = ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index']


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def load_settings(path):
    spec = importlib.util.spec_from_file_location('mimicgen_global',
        ROOT / 'lerobot/skillVLA/libero/configs/src/global_config_loader.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # An inherited ABC/LIBERO launch snapshot must never redirect Coffee storage.
    global_path = ROOT / 'lerobot/skillVLA/mimicgen/configs/global_config.yaml'
    cfg = {**module.load_global_config(global_path), **yaml.safe_load(Path(path).read_text())}
    project = Path(cfg.get('project_root', ROOT)).expanduser().resolve()
    dataset_root = Path(cfg['dataset_root']).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = project / dataset_root
    for key in ('raw_name', 'output_name'):
        value = Path(cfg[key])
        if value.is_absolute() or '..' in value.parts or str(value) in ('', '.'):
            raise ValueError(f'{key} must name a directory under dataset_root')
    cfg['raw_path'] = str(dataset_root / cfg['raw_name'])
    cfg['output_path'] = str(dataset_root / cfg['output_name'])
    if cfg['raw_path'] == cfg['output_path']:
        raise ValueError('Raw and converted paths must differ')
    reference = Path(cfg['schema_reference']).expanduser()
    cfg['schema_reference'] = str(reference if reference.is_absolute() else project / reference)
    if not re.fullmatch(r'[a-f0-9]{40}', str(cfg['source_revision'])):
        raise ValueError('source_revision must be an immutable 40-character commit')
    if cfg['max_episodes'] != 'all' and (type(cfg['max_episodes']) is not int or cfg['max_episodes'] < 1):
        raise ValueError('max_episodes must be all or a positive integer')
    if cfg['gripper_action'] != 'signed' or cfg['image_transform'] != 'identity':
        raise ValueError('Coffee D0 uses signed actions and preserved source image pixels')
    if cfg['video_storage'] not in ('hardlink', 'copy'):
        raise ValueError('video_storage must be hardlink or copy')
    if type(cfg['download_retries']) is not int or cfg['download_retries'] < 0 or cfg['download_timeout'] <= 0:
        raise ValueError('Invalid download retry/timeout setting')
    for key in ('prepare_camera_geometry', 'download_use_proxy'):
        if type(cfg[key]) is not bool:
            raise ValueError(f'{key} must be true or false')
    return cfg


def download(cfg):
    raw = Path(cfg['raw_path'])
    raw.mkdir(parents=True, exist_ok=True)
    marker = raw / 'source_manifest.json'
    if marker.exists():
        old = json.loads(marker.read_text())
        if (old['repo'], old['revision']) != (cfg['source_repo'], cfg['source_revision']):
            raise ValueError('Raw directory belongs to another revision; choose a different raw_name')
    client = session(cfg['download_use_proxy'])
    try:
        files = source_tree(client, cfg['source_repo'], cfg['source_revision'], cfg['download_timeout'])
        files = [r for r in files if r['path'].startswith(('meta/', 'data/', 'videos/', 'eval_env_meta/'))
                 or r['path'] == 'README.md']
        print(f"Coffee D0 download: {sum(r['size'] for r in files) / 2**20:.1f} MiB, {len(files)} files", flush=True)
        # Write identity before the first transfer, so interrupted downloads cannot mix revisions.
        write_json(marker, {'repo': cfg['source_repo'], 'revision': cfg['source_revision'], 'files': files})
        for row in files:
            download_one(client, cfg['source_repo'], cfg['source_revision'], row, raw,
                         cfg['download_retries'], cfg['download_timeout'])
    finally:
        client.close()


def vector(values, dim, name):
    array = np.stack(values).astype(np.float32)
    if array.ndim != 2 or array.shape[1] != dim or not np.isfinite(array).all():
        raise ValueError(f'{name}: expected finite Nx{dim}, got {array.shape}')
    return array


def converted_vectors(frame):
    eef = vector(frame['observation.eef'], 7, 'EEF')
    quat = eef[:, 3:]
    if not np.allclose(np.linalg.norm(quat, axis=1), 1, atol=1e-3):
        raise ValueError('Source EEF quaternions are not normalized xyzw quaternions')
    # Match LiberoProcessorStep / robosuite.quat2axisangle exactly. SciPy's
    # as_rotvec canonicalizes to <= pi, flipping nearly all Coffee orientations
    # (whose recorded w is negative) relative to the pretrained LIBERO inputs.
    w = np.clip(quat[:, 3], -1, 1)
    den = np.sqrt(np.maximum(1 - w * w, 0))
    axis_angle = np.zeros((len(quat), 3), dtype=np.float32)
    valid = den > 1e-10
    axis_angle[valid] = quat[valid, :3] * (2 * np.arccos(w[valid]) / den[valid])[:, None]
    ee = np.concatenate([eef[:, :3], axis_angle], axis=1).astype(np.float32)
    grip = vector(frame['observation.gripper'], 6, 'gripper')
    if np.any(np.abs(grip[:, 2:]) > 1e-7):
        raise ValueError('Unexpected non-padding gripper dimensions; do not silently truncate')
    action = vector(frame['action'], 7, 'action')
    if np.abs(action).max() > 1.00001:
        raise ValueError('Coffee OSC actions must be normalized to [-1,1]')
    return {'observation.state': np.concatenate([ee, grip[:, :2]], axis=1),
            'observation.states.ee_state': ee,
            'observation.states.gripper_state': grip[:, :2],
            'observation.states.joint_state': vector(frame['observation.joint'], 7, 'joint'),
            'action': action}


def vector_stats(array):
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    out = {name: function(array, axis=0).tolist() for name, function in
           [('min', np.min), ('max', np.max), ('mean', np.mean), ('std', np.std)]}
    out['count'] = [len(array)]
    for q in (1, 10, 50, 90, 99):
        out[f'q{q:02d}'] = np.percentile(array, q, axis=0).tolist()
    return out


def rename_images(name):
    for source, target in IMAGE_MAP.items():
        name = name.replace(source, target)
    return name


def nested_list(value):
    if isinstance(value, (np.ndarray, list, tuple)):
        return [nested_list(item) for item in value]
    return value


def read_parquets(paths):
    if not paths:
        raise FileNotFoundError('No source parquet files')
    return pd.concat([pd.read_parquet(path) for path in sorted(paths)], ignore_index=True)


def link_or_copy(source, target, mode):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != source.stat().st_size:
            raise ValueError(f'Conflicting file in partial build: {target}')
        return
    if mode == 'hardlink':
        try:
            os.link(source, target)
            return
        except OSError as error:
            import errno
            if error.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP):
                raise
    shutil.copy2(source, target)


def load_reference(cfg, source_info):
    reference = json.loads((Path(cfg['schema_reference']) / 'meta/info.json').read_text())
    if reference['codebase_version'] != 'v3.0' or reference['fps'] != source_info['fps'] or reference['fps'] != 20:
        raise ValueError('Expected LIBERO LeRobot v3 at 20 Hz; no temporal resampling is performed')
    selected = list(VECTOR_DIMS) + list(IMAGE_MAP.values()) + DEFAULT_KEYS
    features = {key: deepcopy(reference['features'][key]) for key in selected}
    for key, dim in VECTOR_DIMS.items():
        if features[key]['shape'] != [dim] or features[key]['dtype'] != 'float32':
            raise ValueError(f'Incompatible LIBERO reference feature {key}')
    for source, target in IMAGE_MAP.items():
        source_feature = source_info['features'][source]
        if source_feature['shape'] != features[target]['shape'] or source_feature['info'] != features[target]['info']:
            raise ValueError(f'Video encoding/shape differs from LIBERO reference: {source}')
    return features


def validate_output(path, features):
    info = json.loads((path / 'meta/info.json').read_text())
    if info['features'] != features:
        raise ValueError('Converted schema differs from reference')
    frames = read_parquets(list((path / 'data').glob('*/*.parquet')))
    episodes = read_parquets(list((path / 'meta/episodes').glob('*/*.parquet')))
    if len(frames) != info['total_frames'] or len(episodes) != info['total_episodes']:
        raise ValueError('Converted frame/episode counts disagree')
    if episodes['length'].sum() != len(frames):
        raise ValueError('Episode lengths disagree with frame count')
    for key, dim in VECTOR_DIMS.items():
        vector(frames[key], dim, key)
    for _, ep in episodes.iterrows():
        start, stop = int(ep['dataset_from_index']), int(ep['dataset_to_index'])
        subset = frames.iloc[start:stop]
        if not len(subset) or len(subset) != ep['length'] or not (subset.episode_index == ep.episode_index).all():
            raise ValueError('Episode frame mapping is invalid')
        if not np.array_equal(subset.frame_index, np.arange(len(subset))):
            raise ValueError('Non-contiguous episode frames')
        for key in IMAGE_MAP.values():
            video = info['video_path'].format(video_key=key, chunk_index=int(ep[f'videos/{key}/chunk_index']),
                                            file_index=int(ep[f'videos/{key}/file_index']))
            if not local_path(path, video).is_file():
                raise FileNotFoundError(video)
    return info


def convert(cfg):
    raw, output = Path(cfg['raw_path']), Path(cfg['output_path'])
    source_info = json.loads((raw / 'meta/info.json').read_text())
    source_manifest = json.loads((raw / 'source_manifest.json').read_text())
    if (source_manifest['repo'], source_manifest['revision']) != (cfg['source_repo'], cfg['source_revision']):
        raise ValueError('Raw source manifest does not match configured revision')
    if source_info['codebase_version'] != 'v3.0' or source_info['total_tasks'] != 1:
        raise ValueError('This adapter supports single-task Coffee D0 LeRobot v3 only')
    features = load_reference(cfg, source_info)
    request = {k: cfg[k] for k in ['source_repo', 'source_revision', 'max_episodes', 'image_transform',
                                   'gripper_action', 'prepare_camera_geometry']}
    request['features'] = features
    fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    if output.exists():
        marker = output / 'conversion_manifest.json'
        if not marker.is_file() or json.loads(marker.read_text()).get('request_sha256') != fingerprint:
            raise FileExistsError(f'Output belongs to another/incomplete conversion; choose a new output_name: {output}')
        validate_output(output, features)
        print(f'Reuse completed dataset: {output}', flush=True)
        return
    work = output.with_name('.' + output.name + '.building')
    work.mkdir(parents=True, exist_ok=True)
    pending = work / 'conversion_request.json'
    if pending.exists() and json.loads(pending.read_text())['request_sha256'] != fingerprint:
        raise FileExistsError(f'Partial build belongs to a different request: {work}')
    write_json(pending, {'request_sha256': fingerprint, **request})
    source_frames = read_parquets(list((raw / 'data').glob('*/*.parquet')))
    source_eps = read_parquets(list((raw / 'meta/episodes').glob('*/*.parquet'))).sort_values('episode_index')
    if len(source_frames) != source_info['total_frames'] or len(source_eps) != source_info['total_episodes']:
        raise ValueError('Incomplete raw source; run download first')
    if cfg['max_episodes'] != 'all':
        source_eps = source_eps.iloc[:cfg['max_episodes']]
    chosen = source_eps.episode_index.astype(int).tolist()
    if not chosen:
        raise ValueError('No episodes selected')
    frame = source_frames[source_frames.episode_index.isin(chosen)].sort_values(['episode_index', 'frame_index']).copy()
    vectors = converted_vectors(frame)
    result = frame[DEFAULT_KEYS].reset_index(drop=True)
    mapping = {old: new for new, old in enumerate(chosen)}
    result['episode_index'] = result.episode_index.map(mapping).astype(np.int64)
    result['index'] = np.arange(len(result), dtype=np.int64)
    for key, values in vectors.items():
        result[key] = list(values)
    (work / 'data/chunk-000').mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(result, preserve_index=False), work / 'data/chunk-000/file-000.parquet')
    episode_rows, image_stats = [], []
    cursor = 0
    for rank, (_, original) in enumerate(source_eps.iterrows()):
        ep = {rename_images(key): value for key, value in original.to_dict().items() if not key.startswith('stats/')}
        length = int(original['length'])
        subset = result.iloc[cursor:cursor + length]
        if len(subset) != length or not (subset.episode_index == rank).all():
            raise ValueError(f'Source episode metadata disagrees with frames: {original.episode_index}')
        ep.update(episode_index=rank, dataset_from_index=cursor, dataset_to_index=cursor + length,
                  **{'data/chunk_index': 0, 'data/file_index': 0})
        # Keep source video offsets intact: videos are reused, not concatenated again.
        stats = {}
        for source, target in IMAGE_MAP.items():
            stats[target] = {}
            prefix = f'stats/{source}/'
            for key, value in original.items():
                if key.startswith(prefix):
                    name = key[len(prefix):]
                    stats[target][name] = np.asarray(nested_list(value), dtype=np.float64)
                    ep[f'stats/{target}/{name}'] = value
        image_stats.append(stats)
        for key in [*VECTOR_DIMS, *DEFAULT_KEYS]:
            array = np.stack(subset[key]) if key in VECTOR_DIMS else subset[key].to_numpy()
            for name, value in vector_stats(array).items():
                ep[f'stats/{key}/{name}'] = value
        episode_rows.append(ep)
        cursor += length
    (work / 'meta/episodes/chunk-000').mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(episode_rows), preserve_index=False),
                   work / 'meta/episodes/chunk-000/file-000.parquet')
    shutil.copy2(raw / 'meta/tasks.parquet', work / 'meta/tasks.parquet')
    from lerobot.datasets.compute_stats import aggregate_stats
    aggregated_images = aggregate_stats(image_stats)
    stats = {key: {name: np.asarray(value).tolist() for name, value in values.items()}
             for key, values in aggregated_images.items()}
    for key in [*VECTOR_DIMS, *DEFAULT_KEYS]:
        stats[key] = vector_stats(np.stack(result[key]) if key in VECTOR_DIMS else result[key].to_numpy())
    write_json(work / 'meta/stats.json', stats)
    for path in sorted((raw / 'videos').glob('*/*/*.mp4')):
        relative = Path(rename_images(str(path.relative_to(raw))))
        link_or_copy(path, work / relative, cfg['video_storage'])
    (work / 'eval_env_meta').mkdir(exist_ok=True)
    shutil.copy2(raw / 'eval_env_meta/demo.hdf5', work / 'eval_env_meta/demo.hdf5')
    info = deepcopy(source_info)
    info.update(features=features, robot_type='franka', total_episodes=len(chosen), total_frames=len(result),
                splits={'train': f'0:{len(chosen)}'},
                mimicgen={'env_name': 'Coffee_D0', 'action_space': 'normalized_osc_delta',
                          'gripper': '-1=open,+1=close', 'eef_orientation': 'axis_angle_radians',
                          'image_transform': 'identity', 'exact_demo_reset_available': False})
    write_json(work / 'meta/info.json', info)
    if cfg['prepare_camera_geometry']:
        python = ROOT / '.venv_mimicgen/bin/python'
        if not python.is_file():
            raise FileNotFoundError('Run bash setup_env.sh --mimicgen-only to prepare camera geometry')
        sample_indices = np.linspace(0, len(frame) - 1, min(32, len(frame)), dtype=int)
        validation_frames = work / 'meta/geometry_validation_frames.npz'
        np.savez(validation_frames, joint=np.stack(frame['observation.joint'])[sample_indices],
                 eef=np.stack(frame['observation.eef'])[sample_indices])
        subprocess.run([str(python), str(HERE / 'camera_geometry.py'), '--env-meta',
                        str(work / 'eval_env_meta/demo.hdf5'), '--output', str(work / 'meta'),
                        '--validation-frames', str(validation_frames)], check=True)
    validate_output(work, features)
    write_json(work / 'conversion_manifest.json', {'format': FORMAT, 'request_sha256': fingerprint,
        'source_repo': cfg['source_repo'], 'source_revision': cfg['source_revision'],
        'source_episode_ids': chosen, 'episodes': len(chosen), 'frames': len(result),
        'schema_reference': cfg['schema_reference'], 'action_transform': 'identity_signed_osc',
        'state_transform': 'EEF xyz + LIBERO_quat2axisangle_xyzw_no_sign_canonicalization + first_two_finger_qpos',
        'video_transform': 'identity', 'video_timestamps': 'preserved_source_offsets',
        'camera_geometry': 'reconstructed_from_pinned_env_not_recorded_demo_calibration',
        'image_projection_alignment_verified': False,
        'source_manifest_sha256': sha256(raw / 'source_manifest.json')})
    pending.unlink()
    os.replace(work, output)
    print(f'Converted: {len(chosen)} episodes, {len(result)} frames -> {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE.parent / 'dataset_config.yaml')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--download-only', action='store_true')
    mode.add_argument('--convert-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    cfg = load_settings(args.config)
    print(f"Raw: {cfg['raw_path']}\nOutput: {cfg['output_path']}\nCPU only; existing LIBERO checkpoints and datasets are untouched.", flush=True)
    if args.dry_run:
        return
    output = Path(cfg['output_path'])
    output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse concurrent writers; includes raw files shared by download-only runs.
    raw = Path(cfg['raw_path'])
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.convert_only:
            download(cfg)
        if not args.download_only:
            convert(cfg)


if __name__ == '__main__':
    main()
