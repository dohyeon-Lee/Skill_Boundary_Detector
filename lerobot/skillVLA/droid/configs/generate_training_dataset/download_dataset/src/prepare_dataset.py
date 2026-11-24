#!/usr/bin/env python3
"""Convert existing raw DROID episodes to LeRobot v3, without downloading data.

Cartesian/joint action representations and measured EEF poses are retained. Camera geometry
is stored per episode in meta/droid, with original frame indices in the parquet.
Source files are read-only. Only converted, compressed videos are written.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil

import av
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from camera_geometry import intrinsics, pose6_to_matrix, pose6_to_rotvec, resize_geometry

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'lerobot/src/lerobot').is_dir())
FORMAT = 'skillvla_droid/v1'
CAMERAS = {'exterior_1': ('ext1_cam_serial', 'image'),
           'wrist': ('wrist_cam_serial', 'wrist_image'),
           'exterior_2': ('ext2_cam_serial', 'image2')}
ACTION_DIMS = {'cartesian_velocity': 6, 'cartesian_position': 6,
               'joint_velocity': 7, 'joint_position': 7,
               'gripper_position': 1, 'gripper_velocity': 1}


class EmptyEpisodeError(ValueError):
    """No control samples remain after the configured inactive-frame filter."""


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def child(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts or str(relative) in ('', '.'):
        raise ValueError(f'Expected a relative path below {root}: {relative}')
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f'Path escapes root: {relative}')
    return resolved


def load_settings(path):
    spec = importlib.util.spec_from_file_location('_droid_global', ROOT / 'lerobot/skillVLA/droid/configs/src/global_config_loader.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    global_path = ROOT / 'lerobot/skillVLA/droid/configs/global_config.yaml'
    cfg = {**module.load_global_config(global_path, resolve_dataset=False), **yaml.safe_load(Path(path).read_text())}
    registry = runpy.run_path(str(ROOT / 'lerobot/skillVLA/droid/configs/src/dataset_registry.py'))
    cfg = registry['configure'](cfg)
    if 'output_name' not in cfg:
        cfg['output_name'] = cfg['source_dataset']
    # Old frozen requests omitted this key and used anisotropic scaling.
    # New requests explicitly choose letterbox in dataset_config.yaml.
    cfg.setdefault('image_transform', 'stretch')
    if cfg['image_transform'] not in ('letterbox', 'stretch'):
        raise ValueError('image_transform must be letterbox or stretch')
    project = Path(cfg.get('project_root', ROOT)).expanduser().resolve()
    if cfg.get('recording_timestamps_root'):
        timestamp_root = Path(cfg['recording_timestamps_root']).expanduser()
        cfg['recording_timestamps_root'] = str((timestamp_root if timestamp_root.is_absolute()
                                              else project / timestamp_root).resolve())
    for key in ('dataset_root', 'raw_root', 'calibration_root'):
        value = Path(cfg[key]).expanduser()
        cfg[key] = str((value if value.is_absolute() else project / value).resolve())
    cfg['output_path'] = str(child(cfg['dataset_root'], cfg['output_name']))
    output, raw = Path(cfg['output_path']), Path(cfg['raw_root'])
    if output.is_relative_to(raw) or raw.is_relative_to(output):
        raise ValueError('Output and source trees must be separate')
    if cfg.get('annotations_path'):
        value = Path(cfg['annotations_path']).expanduser()
        cfg['annotations_path'] = str(value if value.is_absolute() else project / value)
    if cfg['action_mode'] not in ('cartesian_velocity', 'cartesian_position', 'joint_velocity', 'joint_position'):
        raise ValueError('Unknown action_mode')
    if cfg['missing_calibration'] not in ('skip', 'error') or cfg['external_extrinsics'] not in ('calibrated', 'recorded'):
        raise ValueError('Invalid camera calibration policy')
    if cfg['video_tail_policy'] not in ('drop_last_if_missing_one', 'error'):
        raise ValueError('Invalid video_tail_policy')
    if cfg.get('empty_episode_policy', 'error') not in ('skip', 'error'):
        raise ValueError('empty_episode_policy must be skip or error')
    if not cfg['cameras'] or len(set(cfg['cameras'])) != len(cfg['cameras']) or set(cfg['cameras']) - CAMERAS.keys():
        raise ValueError('cameras must contain distinct exterior_1/wrist/exterior_2 names')
    if cfg['fps'] != 15:
        raise ValueError('DROID control is 15 Hz; changing the rate requires explicit temporal resampling')
    for key in ('workers', 'encoder_threads'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if len(cfg['image_size']) != 2 or any(type(x) is not int or x < 2 or x % 2 for x in cfg['image_size']):
        raise ValueError('image_size needs positive even [height,width] for yuv420p')
    if type(cfg['video_crf']) is not int or not 0 <= cfg['video_crf'] <= 51:
        raise ValueError('video_crf must be 0..51')
    if cfg['max_episodes'] != 'all' and (type(cfg['max_episodes']) is not int or cfg['max_episodes'] < 1):
        raise ValueError('max_episodes must be all or a positive integer')
    if not isinstance(cfg['episode_ids'], list) or len(set(cfg['episode_ids'])) != len(cfg['episode_ids']):
        raise ValueError('episode_ids must be a list of distinct original UUIDs')
    for key in ('success_only', 'remove_skipped_steps'):
        if type(cfg[key]) is not bool:
            raise ValueError(f'{key} must be true or false')
    return cfg


def discover(cfg):
    raw = Path(cfg['raw_root'])
    manifest = raw / 'download_manifest.jsonl'
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if len({row['episode_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate UUIDs in raw manifest')
    requested = set(cfg['episode_ids'])
    unknown = requested - {row['episode_id'] for row in rows}
    if unknown:
        raise ValueError(f'Unknown episode_ids: {sorted(unknown)}')
    calibration = Path(cfg['calibration_root'])
    inputs = {'manifest': manifest, 'intrinsics': calibration / 'intrinsics.json'}
    if cfg.get('recording_timestamps_root'):
        inputs['recording_timestamps'] = Path(cfg['recording_timestamps_root']) / 'index.json'
    camera_k = json.loads(inputs['intrinsics'].read_text())
    camera_t = {}
    if cfg['external_extrinsics'] == 'calibrated':
        inputs['extrinsics'] = calibration / 'cam2base_extrinsic_superset.json'
        camera_t = json.loads(inputs['extrinsics'].read_text())
    annotations = {}
    if cfg.get('annotations_path'):
        inputs['annotations'] = Path(cfg['annotations_path'])
        annotations = json.loads(inputs['annotations'].read_text())
    selected, excluded = [], []
    for row in rows:
        uuid = row['episode_id']
        if requested and uuid not in requested:
            continue
        directory = child(raw, row['relative_path'])
        metadata_path = directory / f'metadata_{uuid}.json'
        metadata = json.loads(metadata_path.read_text())
        if metadata['uuid'] != uuid:
            raise ValueError(f'Metadata UUID differs: {metadata_path}')
        if cfg['success_only'] and not metadata.get('success', False):
            excluded.append({'episode_id': uuid, 'reason': 'not_success'})
            continue
        try:
            geometry = {}
            for camera in cfg['cameras']:
                serial = str(metadata[CAMERAS[camera][0]])
                entry = camera_k.get(uuid, {}).get(serial)
                if entry is None:
                    raise ValueError(f'{camera}/{serial}: missing intrinsics')
                intrinsics(entry, int(entry['width']), int(entry['height']), cfg['image_size'], cfg['image_transform'])
                value = {'serial': serial, 'intrinsics': entry}
                if camera != 'wrist' and cfg['external_extrinsics'] == 'calibrated':
                    calibrated = camera_t.get(uuid, {})
                    if serial not in calibrated:
                        raise ValueError(f'{camera}/{serial}: missing calibrated extrinsics')
                    if calibrated.get('relative_path') != row['relative_path']:
                        raise ValueError('Calibration points to another source trajectory')
                    pose6_to_matrix(calibrated[serial])
                    value['calibrated_pose'] = calibrated[serial]
                    value['provenance'] = {key: calibrated.get(f'{serial}_{key}')
                                           for key in ('source', 'metric_type', 'quality_metric')}
                else:
                    value['provenance'] = {'source': 'recorded_per_frame'}
                geometry[camera] = value
        except (ValueError, KeyError) as error:
            if cfg['missing_calibration'] == 'error':
                raise ValueError(f'{uuid}: {error}') from error
            excluded.append({'episode_id': uuid, 'reason': str(error)})
            continue
        # Validate local manifest sizes without copying or writing into the raw root.
        signature = []
        for item in row['files']:
            prefix = f"robotics/droid_raw/1.0.1/{row['relative_path']}/"
            if not item['name'].startswith(prefix):
                raise ValueError(f'Manifest file outside episode: {item["name"]}')
            file = child(directory, item['name'][len(prefix):])
            stat = file.stat()
            if stat.st_size != item['size']:
                raise ValueError(f'Incomplete source: {file}')
            signature.append({'path': str(file), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
        language = annotations.get(uuid, {})
        if isinstance(language, str):
            language = {'language_instruction1': language}
        labels = [language[k] for k in ('language_instruction1', 'language_instruction2', 'language_instruction3')
                  if language.get(k)]
        task = labels[0] if labels else str(metadata.get('current_task') or 'Unspecified DROID task')
        selected.append({'episode_id': uuid, 'directory': str(directory), 'metadata': metadata,
                         'geometry': geometry, 'task': task, 'language_annotations': labels,
                         'language_source': 'official_annotations' if labels else 'raw_current_task',
                         'source_files': signature})
        if cfg['max_episodes'] != 'all' and len(selected) >= cfg['max_episodes']:
            break
    if not selected:
        raise ValueError('No eligible episodes after selection/calibration filtering')
    hashes = {key: {'path': str(path), 'sha256': sha256(path)} for key, path in inputs.items()}
    return selected, excluded, hashes


def video_info(path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = stream.frames or sum(1 for _ in container.decode(stream))
        return {'frames': frames, 'width': stream.width, 'height': stream.height,
                'fps': float(stream.average_rate) if stream.average_rate else None}


def numeric(h5, path, dim, length):
    values = np.asarray(h5[path], dtype=np.float32)
    if dim == 1 and values.shape == (length,):
        values = values[:, None]
    if values.shape != (length, dim) or not np.isfinite(values).all():
        raise ValueError(f'{path}: expected finite ({length},{dim}), got {values.shape}')
    return values


def vectors(h5, cfg):
    length = len(h5['action/gripper_position'])
    ee_euler = numeric(h5, 'observation/robot_state/cartesian_position', 6, length)
    ee = pose6_to_rotvec(ee_euler)
    joints = numeric(h5, 'observation/robot_state/joint_positions', 7, length)
    grip = numeric(h5, 'observation/robot_state/gripper_position', 1, length)
    raw_actions = {name: numeric(h5, f'action/{name}', dim, length) for name, dim in ACTION_DIMS.items()}
    grip_command = raw_actions['gripper_position']
    if np.any((grip_command < -1e-5) | (grip_command > 1.00001)):
        raise ValueError('DROID gripper_position action must be absolute 0=open/1=closed')
    primary = raw_actions[cfg['action_mode']]
    if cfg['action_mode'] == 'cartesian_position':
        primary = pose6_to_rotvec(primary)
    state = joints if cfg['action_mode'].startswith('joint_') else ee
    result = {'observation.state': np.concatenate([state, grip], axis=1),
              'observation.states.ee_state': ee, 'observation.states.ee_state_euler': ee_euler,
              'observation.states.joint_state': joints, 'observation.states.gripper_state': grip,
              'action': np.concatenate([primary, grip_command], axis=1)}
    result.update({f'source.action.{name}': array for name, array in raw_actions.items()})
    result['source.control_timestamp_ms'] = np.asarray(h5['observation/timestamp/control/step_start'], dtype=np.int64)
    if result['source.control_timestamp_ms'].shape != (length,):
        raise ValueError('Control timestamp length differs from robot states')
    for target, path in [('source.skip_action', 'observation/timestamp/skip_action'),
                         ('source.movement_enabled', 'observation/controller_info/movement_enabled')]:
        value = np.asarray(h5[path], dtype=bool) if path in h5 else None
        if value is None or value.shape != (length,):
            raise ValueError(f'Missing/misaligned control flag: {path}')
        result[target] = value
    return result


def vector_stats(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    stats = {name: func(array, axis=0).tolist() for name, func in
             [('min', np.min), ('max', np.max), ('mean', np.mean), ('std', np.std)]}
    stats['count'] = [len(array)]
    for q in (1, 10, 50, 90, 99):
        stats[f'q{q:02d}'] = np.percentile(array, q, axis=0).tolist()
    return stats


def camera_frame_map(h5, row, rank, camera, video_length, cfg):
    """Join independently recovered recording timestamps to control observations."""
    length = len(h5['action/gripper_position'])
    serial = row['geometry'][camera]['serial']
    root = cfg.get('recording_timestamps_root')
    index = json.loads((Path(root) / 'index.json').read_text()) if root else {}
    episode = index.get('episodes', {}).get(row['episode_id'])
    path = child(root, episode) / f'{serial}.json' if episode is not None else None
    if path is None:
        if video_length not in (length, length - 1):
            raise ValueError(f'Unexpected raw video/H5 alignment: video={video_length}, H5={length}')
        mapping = np.arange(length, dtype=np.int64)
        mapping[video_length:] = -1
        return mapping, {'alignment': 'legacy_control_index'}
    relative = path.relative_to(root).as_posix()
    if index['files'].get(relative) != sha256(path):
        raise ValueError(f'Recording timestamp evidence changed: {path}')
    record = json.loads(path.read_text())
    source = Path(row['directory']) / 'trajectory.h5'
    if (record['episode_id'] != row['episode_id'] or record['serial'] != serial
            or record['video_frames'] != video_length or record['h5_rows'] != length
            or record['source_h5_size'] != source.stat().st_size
            or record['source_h5_mtime_ns'] != source.stat().st_mtime_ns):
        raise ValueError(f'Recording timestamp evidence belongs to another source: {path}')
    times = np.asarray(record['timestamps_ms'], dtype=np.int64)
    if times.shape != (video_length,) or not (np.diff(times) > 0).all():
        raise ValueError(f'Non-monotonic recording timestamps: {path}')
    captured = np.asarray(h5[f'observation/timestamp/cameras/{serial}_frame_received'])
    if captured.shape != (length,) or (np.diff(captured) < 0).any():
        raise ValueError(f'Non-monotonic camera observations: {serial}')
    positions = {int(timestamp): frame for frame, timestamp in enumerate(times)}
    mapping = np.asarray([positions.get(int(timestamp), -1) for timestamp in captured], dtype=np.int64)
    if not np.array_equal(mapping, np.asarray(record['h5_to_video_frame'])):
        raise ValueError(f'Recording timestamp mapping differs: {path}')
    return mapping, {'alignment': 'exact_recording_timestamp_ms', 'timestamp_evidence': str(path),
                     'timestamp_evidence_sha256': index['files'][relative],
                     'source_svo_url': record['source_svo_url'],
                     'source_svo_sha256': record['source_svo_sha256']}


def transcode(source, target, indices, cfg):
    """Encode selected video frames; repeated camera observations reuse a frame."""
    if len(indices) and ((np.diff(indices) < 0).any() or indices[0] < 0):
        raise ValueError('Video frame map must be nonnegative and nondecreasing')
    height, width = cfg['image_size']
    seen, written = 0, 0
    sums = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    low, high = np.ones(3), np.zeros(3)
    with av.open(str(source)) as reader, av.open(str(target), 'w') as writer:
        input_stream = reader.streams.video[0]
        input_stream.codec_context.thread_count = cfg['encoder_threads']
        resized_h, resized_w, top, left, _ = resize_geometry(input_stream.width, input_stream.height,
                                                            cfg['image_size'], cfg['image_transform'])
        stream = writer.add_stream('libx264', rate=cfg['fps'])
        stream.width, stream.height, stream.pix_fmt = width, height, 'yuv420p'
        stream.codec_context.thread_count = cfg['encoder_threads']
        stream.options = {'crf': str(cfg['video_crf']), 'preset': 'veryfast'}
        for original_index, frame in enumerate(reader.decode(input_stream)):
            seen += 1
            if written == len(indices) or original_index != indices[written]:
                continue
            resized = frame.reformat(width=resized_w, height=resized_h, format='rgb24').to_ndarray()
            if (resized_h, resized_w) == (height, width):
                rgb = resized
            else:
                rgb = np.zeros((height, width, 3), dtype=np.uint8)
                rgb[top:top+resized_h, left:left+resized_w] = resized
            pixels = rgb.reshape(-1, 3).astype(np.float64) / 255
            while written < len(indices) and original_index == indices[written]:
                sums += pixels.sum(axis=0)
                squares += np.square(pixels).sum(axis=0)
                low = np.minimum(low, pixels.min(axis=0))
                high = np.maximum(high, pixels.max(axis=0))
                output_frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
                output_frame.pts = written
                for packet in stream.encode(output_frame):
                    writer.mux(packet)
                written += 1
        for packet in stream.encode():
            writer.mux(packet)
    if written != len(indices):
        raise ValueError(f'Video ended before selected control samples: {source}')
    mean = sums / (written * width * height)
    stats = {'min': low, 'max': high, 'mean': mean,
             'std': np.sqrt(np.maximum(squares / (written * width * height) - mean**2, 0))}
    stats = {key: value.reshape(3, 1, 1).tolist() for key, value in stats.items()}
    stats['count'] = [written]
    return stats, seen


def convert_episode(row, rank, cfg, cache):
    directory = cache / f'{rank:06d}'
    marker = directory / 'done.json'
    if marker.is_file():
        done = json.loads(marker.read_text())
        if done['episode_id'] != row['episode_id']:
            raise ValueError('Partial cache belongs to another episode')
        if all((directory / name).is_file() and (directory / name).stat().st_size == size
               for name, size in done['files'].items()):
            return done
        raise ValueError(f'Completed episode cache is damaged; preserve and inspect: {directory}')
    # No completion marker means only this converter's unfinished episode files.
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    source = Path(row['directory'])
    videos, probes = {}, {}
    for camera, geometry in row['geometry'].items():
        videos[camera] = source / 'recordings/MP4' / f"{geometry['serial']}.mp4"
        probes[camera] = video_info(videos[camera])
    camera_payload, provenance, frame_maps = {}, {}, {}
    with h5py.File(source / 'trajectory.h5', 'r') as h5:
        arrays = vectors(h5, cfg)
        length = len(arrays['action'])
        alignment = {}
        for camera, probe in probes.items():
            frame_maps[camera], alignment[camera] = camera_frame_map(h5, row, rank, camera, probe['frames'], cfg)
        available = np.logical_and.reduce([mapping >= 0 for mapping in frame_maps.values()])
        missing = np.flatnonzero(~available).tolist()
        if missing and (cfg['video_tail_policy'] != 'drop_last_if_missing_one' or missing != [length - 1]):
            raise ValueError(f'Unexpected raw video/H5 alignment: unavailable control rows={missing}')
        video_length = int(available.sum())
        indices = np.arange(video_length, dtype=np.int64)
        if cfg['remove_skipped_steps']:
            keep = arrays['source.movement_enabled'][indices] & ~arrays['source.skip_action'][indices]
            indices = indices[keep]
        if len(indices) == 0:
            raise EmptyEpisodeError(f'No active frames: {row["episode_id"]}')
        if len(indices) < 2:
            raise ValueError(f'Not enough active frames: {row["episode_id"]}')
        for camera, geometry in row['geometry'].items():
            serial = geometry['serial']
            probe = probes[camera]
            original_k, scaled_k, transform = intrinsics(geometry['intrinsics'], probe['width'], probe['height'], cfg['image_size'], cfg['image_transform'])
            raw_pose = numeric(h5, f'observation/camera_extrinsics/{serial}_left', 6, length)[indices]
            used_pose = np.broadcast_to(geometry['calibrated_pose'], raw_pose.shape) if 'calibrated_pose' in geometry else raw_pose
            camera_payload.update({f'{camera}.intrinsics_original': original_k,
                                   f'{camera}.intrinsics': scaled_k,
                                   f'{camera}.pixel_transform': transform,
                                   f'{camera}.camera_to_base': pose6_to_matrix(used_pose),
                                   f'{camera}.recorded_pose_xyz_euler': raw_pose})
            timestamps = h5['observation/timestamp/cameras']
            for key in timestamps:
                if key.startswith(serial + '_'):
                    values = np.asarray(timestamps[key])
                    if values.shape != (length,):
                        raise ValueError(f'Camera timestamp length differs: {key}')
                    camera_payload[f'{camera}.{key[len(serial)+1:]}_ms'] = values[indices]
            camera_payload[f'{camera}.source_video_frame_index'] = frame_maps[camera][indices]
            provenance[camera] = {'serial': serial, **geometry['provenance'], 'source_video': str(videos[camera]),
                                  **probe, **alignment[camera]}
        arrays = {key: value[indices] for key, value in arrays.items()}
    arrays['source.frame_index'] = indices
    arrays['timestamp'] = np.arange(len(indices), dtype=np.float32) / cfg['fps']
    arrays['frame_index'] = np.arange(len(indices), dtype=np.int64)
    arrays['episode_index'] = np.full(len(indices), rank, dtype=np.int64)
    # Global index/task_index are filled during final assembly in manifest order.
    table = {key: value.tolist() if value.ndim == 1 or value.shape[1] != 1 else value[:, 0].tolist()
             for key, value in arrays.items()}
    pq.write_table(pa.Table.from_pydict(table), directory / 'frames.parquet')
    camera_payload['source_frame_index'] = indices
    np.savez_compressed(directory / 'cameras.npz', **camera_payload)
    stats = {key: vector_stats(value) for key, value in arrays.items()}
    for camera in cfg['cameras']:
        feature = f'observation.images.{CAMERAS[camera][1]}'
        stats[feature], decoded = transcode(videos[camera], directory / f'{camera}.mp4', frame_maps[camera][indices], cfg)
        if decoded != probes[camera]['frames']:
            raise ValueError(f'Actual decoded frame count differs from video header: {videos[camera]}')
    write_json(directory / 'episode.json', {**row, 'camera_provenance': provenance,
               'original_control_steps': length, 'dropped_tail_steps': length - video_length,
               'removed_inactive_steps': video_length - len(indices),
               'image_transform': cfg['image_transform'],
               'camera_geometry': 'camera_to_base; metres; RGB pinhole; recorded wrist already in base frame',
               'image_projection_alignment_verified': False})
    for item in row['source_files']:
        stat = Path(item['path']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (item['size'], item['mtime_ns']):
            raise ValueError(f'Source changed during conversion: {item["path"]}')
    files = {p.name: p.stat().st_size for p in directory.iterdir() if p.is_file()}
    done = {'episode_id': row['episode_id'], 'length': len(indices), 'files': files, 'stats': stats}
    write_json(marker, done)
    return done


def features(cfg):
    from lerobot.datasets.utils import DEFAULT_FEATURES
    result = deepcopy(DEFAULT_FEATURES)
    action_dim = ACTION_DIMS[cfg['action_mode']] + 1
    dims = {'observation.state': 8 if cfg['action_mode'].startswith('joint_') else 7,
            'observation.states.ee_state': 6, 'observation.states.ee_state_euler': 6,
            'observation.states.joint_state': 7, 'observation.states.gripper_state': 1,
            'action': action_dim}
    dims.update({f'source.action.{key}': dim for key, dim in ACTION_DIMS.items()})
    result.update({key: {'dtype': 'float32', 'shape': [dim], 'names': None} for key, dim in dims.items()})
    for key in ('source.frame_index', 'source.control_timestamp_ms'):
        result[key] = {'dtype': 'int64', 'shape': [1], 'names': None}
    for key in ('source.skip_action', 'source.movement_enabled'):
        result[key] = {'dtype': 'bool', 'shape': [1], 'names': None}
    height, width = cfg['image_size']
    for camera in cfg['cameras']:
        result[f'observation.images.{CAMERAS[camera][1]}'] = {
            'dtype': 'video', 'shape': [height, width, 3], 'names': ['height', 'width', 'channels'],
            'info': {'video.height': height, 'video.width': width, 'video.codec': 'h264',
                     'video.pix_fmt': 'yuv420p', 'video.is_depth_map': False, 'video.fps': cfg['fps'],
                     'video.channels': 3, 'has_audio': False}}
    return result


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not os.path.samefile(source, target):
            raise ValueError(f'Conflicting partial output: {target}')
    else:
        os.link(source, target)


def validate_output(path, decode=False):
    path = Path(path)
    info = json.loads((path / 'meta/info.json').read_text())
    episodes = pd.read_parquet(path / 'meta/episodes/chunk-000/file-000.parquet')
    if len(episodes) != info['total_episodes'] or int(episodes.length.sum()) != info['total_frames']:
        raise ValueError('Episode/frame counts differ from metadata')
    uuids = []
    for rank, row in episodes.iterrows():
        if int(row.episode_index) != rank:
            raise ValueError('Episode order is not contiguous')
        chunk, file = int(row['data/chunk_index']), int(row['data/file_index'])
        frame = pd.read_parquet(path / info['data_path'].format(chunk_index=chunk, file_index=file))
        length = int(row.length)
        if len(frame) != length or not np.array_equal(frame.frame_index, np.arange(length)):
            raise ValueError('Episode frame map differs')
        if not (frame.episode_index == rank).all() or not np.array_equal(frame['index'], np.arange(int(row.dataset_from_index), int(row.dataset_to_index))):
            raise ValueError('Global episode/frame indices differ')
        if not np.allclose(frame.timestamp, np.arange(length) / info['fps'], atol=1e-5):
            raise ValueError('Video and parquet timing differ')
        with np.load(path / f'meta/droid/episodes/{rank:06d}.npz', allow_pickle=False) as camera:
            if not np.array_equal(camera['source_frame_index'], frame['source.frame_index']):
                raise ValueError('Camera geometry is not aligned to parquet')
            for name in info['droid']['cameras']:
                if camera[f'{name}.camera_to_base'].shape != (length, 4, 4):
                    raise ValueError('Missing dynamic camera transforms')
        for key, feature in info['features'].items():
            if feature['dtype'] == 'video':
                video = path / info['video_path'].format(video_key=key, chunk_index=chunk, file_index=file)
                probe = video_info(video)
                if probe['frames'] != length or probe['fps'] != info['fps']:
                    raise ValueError(f'Video length/rate differs: {video}')
                if [probe['height'], probe['width'], 3] != feature['shape']:
                    raise ValueError(f'Video resolution differs: {video}')
                if decode:
                    with av.open(str(video)) as container:
                        if sum(1 for _ in container.decode(video=0)) != length:
                            raise ValueError(f'Video decode failed: {video}')
            else:
                array = np.stack(frame[key]) if feature['shape'] != [1] else frame[key].to_numpy()
                if not np.isfinite(array).all():
                    raise ValueError(f'Invalid numeric data: {key}')
        uuids.append(json.loads((path / f'meta/droid/episodes/{rank:06d}.json').read_text())['episode_id'])
    if len(set(uuids)) != len(uuids):
        raise ValueError('Duplicate source episodes')
    return info


def convert(cfg, selected, excluded, hashes):
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.feature_utils import create_empty_dataset_info
    output = Path(cfg['output_path'])
    request = {key: value for key, value in cfg.items() if key in (
        'raw_root', 'calibration_root', 'annotations_path', 'max_episodes', 'episode_ids', 'success_only',
        'missing_calibration', 'cameras', 'image_size', 'image_transform', 'fps', 'remove_skipped_steps', 'video_tail_policy',
        'action_mode', 'external_extrinsics', 'video_crf', 'recording_timestamps_root', 'empty_episode_policy')}
    request.update(format=FORMAT, inputs=hashes, episodes=selected,
                   converter_sha256=sha256(HERE / 'prepare_dataset.py'),
                   camera_geometry_sha256=sha256(HERE / 'camera_geometry.py'))
    fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    if output.exists():
        marker = output / 'conversion_manifest.json'
        if not marker.is_file() or json.loads(marker.read_text()).get('request_sha256') != fingerprint:
            raise FileExistsError(f'Output belongs to another conversion; choose a new output_name: {output}')
        validate_output(output)
        print(f'Reuse verified completed dataset: {output}', flush=True)
        return output
    work = output.with_name('.' + output.name + '.building')
    work.mkdir(parents=True, exist_ok=True)
    pending = work / 'conversion_request.json'
    if pending.exists() and json.loads(pending.read_text())['request_sha256'] != fingerprint:
        raise FileExistsError(f'Partial build belongs to a different request: {work}')
    write_json(pending, {'request_sha256': fingerprint, 'format': FORMAT})
    cache = work / '.episodes'
    excluded = list(excluded)
    with ThreadPoolExecutor(max_workers=cfg['workers']) as pool:
        futures = [pool.submit(convert_episode, row, rank, cfg, cache) for rank, row in enumerate(selected)]
        done = []
        for rank, future in enumerate(futures):
            try:
                marker = future.result()
            except EmptyEpisodeError:
                if cfg.get('empty_episode_policy', 'error') != 'skip':
                    raise
                excluded.append({'episode_id': selected[rank]['episode_id'],
                                 'cache_rank': rank, 'reason': 'no_active_frames'})
                print(f'Episode {rank + 1}/{len(selected)} excluded: no active frames', flush=True)
                continue
            done.append((rank, selected[rank], marker))
            print(f'Episode {rank + 1}/{len(selected)} ready: {marker["length"]} frames', flush=True)
    if not done:
        raise ValueError('No usable episodes remain after inactive-frame filtering')
    schema = features(cfg)
    info = create_empty_dataset_info(codebase_version='v3.0', fps=cfg['fps'], robot_type='franka', features=schema, use_videos=True)
    tasks = list(dict.fromkeys(row['task'] for _, row, _ in done))
    task_indices = {task: index for index, task in enumerate(tasks)}
    episode_rows, stats_list, cursor = [], [], 0
    for rank, (cache_rank, source, marker) in enumerate(done):
        directory = cache / f'{cache_rank:06d}'
        frame = pd.read_parquet(directory / 'frames.parquet')
        length = len(frame)
        frame['index'] = np.arange(cursor, cursor + length, dtype=np.int64)
        frame['episode_index'] = np.full(length, rank, dtype=np.int64)
        frame['task_index'] = np.full(length, task_indices[source['task']], dtype=np.int64)
        chunk, file = divmod(rank, 1000)
        destination = work / info['data_path'].format(chunk_index=chunk, file_index=file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), destination)
        stats = deepcopy(marker['stats'])
        stats.update({key: vector_stats(frame[key].to_numpy()) for key in ('index', 'episode_index', 'task_index')})
        episode = {'episode_index': rank, 'length': length, 'tasks': [source['task']],
                   'dataset_from_index': cursor, 'dataset_to_index': cursor + length,
                   'data/chunk_index': chunk, 'data/file_index': file,
                   'meta/episodes/chunk_index': 0, 'meta/episodes/file_index': 0}
        for camera in cfg['cameras']:
            key = f'observation.images.{CAMERAS[camera][1]}'
            destination = work / info['video_path'].format(video_key=key, chunk_index=chunk, file_index=file)
            link(directory / f'{camera}.mp4', destination)
            episode.update({f'videos/{key}/chunk_index': chunk, f'videos/{key}/file_index': file,
                            f'videos/{key}/from_timestamp': 0., f'videos/{key}/to_timestamp': length / cfg['fps']})
        for key, values in stats.items():
            for name, value in values.items():
                episode[f'stats/{key}/{name}'] = value
        link(directory / 'cameras.npz', work / f'meta/droid/episodes/{rank:06d}.npz')
        link(directory / 'episode.json', work / f'meta/droid/episodes/{rank:06d}.json')
        episode_rows.append(episode)
        stats_list.append({key: {name: np.asarray(value) for name, value in values.items()} for key, values in stats.items()})
        cursor += length
    meta = work / 'meta/episodes/chunk-000/file-000.parquet'
    meta.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(episode_rows), preserve_index=False), meta)
    pd.DataFrame({'task_index': range(len(tasks))}, index=pd.Index(tasks, name='task')).to_parquet(work / 'meta/tasks.parquet')
    merged = aggregate_stats(stats_list)
    write_json(work / 'meta/stats.json', {key: {name: value.tolist() for name, value in values.items()} for key, values in merged.items()})
    info.update(total_episodes=len(done), total_frames=cursor, total_tasks=len(tasks), splits={'train': f'0:{len(done)}'},
                droid={'cameras': cfg['cameras'], 'action_mode': cfg['action_mode'],
                       'gripper': 'absolute_position_0_open_1_closed', 'eef_pose': 'XYZ_metres_rotation_vector_radians',
                       'raw_eef_pose': 'XYZ_metres_Euler_xyz_radians',
                       'geometry_path': 'meta/droid/episodes/{episode_index:06d}.npz',
                       'provenance_path': 'meta/droid/episodes/{episode_index:06d}.json',
                       'image_transform': cfg['image_transform'],
                       'video_transform': f'{cfg["image_transform"]}_RGB; dense active samples at 15Hz',
                       'source_frame_mapping': 'source.frame_index', 'image_projection_alignment_verified': False})
    write_json(work / 'meta/info.json', info)
    validate_output(work)
    write_json(work / 'conversion_manifest.json', {'format': FORMAT, 'request_sha256': fingerprint,
               'source_inputs': hashes, 'source_episode_ids': [row['episode_id'] for _, row, _ in done],
               'episodes': len(done), 'frames': cursor, 'excluded': excluded,
               'action_mode': cfg['action_mode'], 'original_action_fields_preserved': list(ACTION_DIMS),
               'image_transform': cfg['image_transform'],
               'camera_transform': 'camera_to_base', 'source_modified': False})
    os.replace(work, output)
    # Publish first so an interruption during cleanup leaves a verified dataset.
    # Final compressed videos/geometry are hardlinks, so this never removes them.
    shutil.rmtree(output / '.episodes')
    (output / 'conversion_request.json').unlink()
    print(f'Converted {len(done)} episodes / {cursor} frames -> {output}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE.parent / 'dataset_config.yaml')
    parser.add_argument('--workers', type=int, help='Override YAML CPU episode workers')
    parser.add_argument('--dry-run', action='store_true', help='Inspect selection/calibration without writing a dataset')
    parser.add_argument('--verify', action='store_true', help='Fully decode and verify the configured completed output')
    args = parser.parse_args()
    cfg = load_settings(args.config)
    if args.workers is not None:
        if args.workers < 1:
            parser.error('--workers must be positive')
        cfg['workers'] = args.workers
    if args.verify:
        info = validate_output(cfg['output_path'], decode=True)
        print(f"Verified {info['total_episodes']} episodes / {info['total_frames']} frames")
        return
    selected, excluded, hashes = discover(cfg)
    print(f"Eligible selection: {len(selected)} episodes; excluded: {len(excluded)}\nOutput: {cfg['output_path']}\nCPU workers: {cfg['workers']}; action: {cfg['action_mode']}; source files remain read-only", flush=True)
    if args.dry_run:
        return
    output = Path(cfg['output_path'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_name('.' + output.name + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        convert(cfg, selected, excluded, hashes)


if __name__ == '__main__':
    main()
