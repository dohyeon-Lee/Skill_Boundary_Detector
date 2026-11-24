"""Create joint-velocity or EEF-target DROID views without re-encoding video."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)  # Do not modify an inode hardlinked to source metadata.


def _joined_stats(stats, left, right):
    a, b = stats[left], stats[right]
    if a['count'] != b['count']:
        raise ValueError('DROID action/state statistic counts disagree')
    return {key: (list(a[key]) if key == 'count' else list(a[key]) + list(b[key])) for key in a}


def prepare_joint_view(source, destination, *, fps=15):
    return prepare_view(source, destination, fps=fps, mode='joint_velocity')


def prepare_view(source, destination, *, fps=15, mode='joint_velocity'):
    """Joint velocity view, or absolute EEF targets for runtime pose conversion.

    Full trajectories, all three cameras, EEF poses and camera calibration stay
    available. Reads only a completed source; never attaches to .building.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    if mode not in {'joint_velocity', 'delta_pose'}:
        raise ValueError('DROID DP action space must be joint_velocity|delta_pose')

    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError('A DROID training view must not overwrite its conversion source')
    marker = source / 'conversion_manifest.json'
    if not marker.is_file():
        raise FileNotFoundError(f'DROID conversion is not complete: {marker}')
    info = json.loads((source / 'meta/info.json').read_text())
    if info['fps'] != fps:
        raise ValueError(f'DROID source fps={info["fps"]} differs from configured fps={fps}; no silent resampling')
    identity = hashlib.sha256(b''.join((source / name).read_bytes() for name in
        ('conversion_manifest.json', 'meta/info.json', 'meta/stats.json'))).hexdigest()
    contract = {'version': 1, 'source': str(source), 'source_sha256': identity,
                'fps': fps, 'action': 'joint_velocity7_gripper_absolute1',
                'state': 'joint_position7_gripper_state1', 'videos_reencoded': False}
    dimension = 8
    if mode == 'delta_pose':
        contract.update(action='absolute_eef_pose6_gripper_absolute1', state='eef_pose6_gripper_state1')
        dimension = 7
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_name('.' + destination.name + '.view.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            saved = destination / 'training_view_manifest.json'
            if not saved.is_file() or json.loads(saved.read_text()) != contract:
                raise FileExistsError(f'Training view belongs to another source/contract: {destination}')
            view_info = json.loads((destination / 'meta/info.json').read_text())
            if view_info['features']['action']['shape'] != [dimension] or not (destination / 'videos').is_dir():
                raise ValueError(f'Training view is incomplete: {destination}')
            return destination
        work = destination.with_name('.' + destination.name + '.view-building')
        request = work / 'view_request.json'
        if work.exists():
            if not request.is_file() or json.loads(request.read_text()) != contract:
                raise FileExistsError(f'Unknown partial view; preserve and inspect: {work}')
        work.mkdir(exist_ok=True)
        atomic_json(request, contract)
        def link_missing(src, dst):
            if not Path(dst).exists():
                os.link(src, dst)
            return dst
        shutil.copytree(source / 'meta', work / 'meta', copy_function=link_missing, dirs_exist_ok=True)
        if not (work / 'videos').exists():
            (work / 'videos').symlink_to(source / 'videos', target_is_directory=True)
        files = sorted((source / 'data').rglob('*.parquet'))
        if not files:
            raise ValueError('DROID conversion has no data parquet')
        count = 0
        if mode == 'delta_pose':
            from scipy.spatial.transform import Rotation
            from lerobot.datasets.compute_stats import RunningQuantileStats
            action_stats, state_stats = RunningQuantileStats(), RunningQuantileStats()
        for path in files:
            table = pq.read_table(path)
            def array(key, dimension):
                value = np.asarray(table[key].to_pylist(), dtype=np.float32).reshape(-1, dimension)
                if len(value) != len(table) or not np.isfinite(value).all():
                    raise ValueError(f'Invalid DROID column {key}: {path}')
                return value
            if mode == 'joint_velocity':
                action = np.c_[array('source.action.joint_velocity', 7), array('source.action.gripper_position', 1)]
                state = np.c_[array('observation.states.joint_state', 7), array('observation.states.gripper_state', 1)]
            else:
                target = array('source.action.cartesian_position', 6)
                action = np.c_[target[:, :3], Rotation.from_euler('xyz', target[:, 3:]).as_rotvec(),
                               array('source.action.gripper_position', 1)].astype(np.float32)
                state = np.c_[array('observation.states.ee_state', 6), array('observation.states.gripper_state', 1)]
                action_stats.update(action)
                state_stats.update(state)
            for key, value in (('action', action), ('observation.state', state)):
                table = table.set_column(table.schema.get_field_index(key), key, pa.array(value.tolist(), type=pa.list_(pa.float32())))
            target = work / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.parquet.tmp')
            pq.write_table(table, temporary)
            temporary.replace(target)
            count += len(table)
        if count != info['total_frames']:
            raise ValueError('DROID view frame count differs from source metadata')
        stats = json.loads((source / 'meta/stats.json').read_text())
        if mode == 'joint_velocity':
            stats['action'] = _joined_stats(stats, 'source.action.joint_velocity', 'source.action.gripper_position')
            stats['observation.state'] = _joined_stats(stats, 'observation.states.joint_state', 'observation.states.gripper_state')
            action_names = [f'joint_velocity_{i}' for i in range(7)]
            state_names = [f'joint_position_{i}' for i in range(7)]
        else:
            stats['action'] = {key: value.tolist() for key, value in action_stats.get_statistics().items()}
            stats['observation.state'] = {key: value.tolist() for key, value in state_stats.get_statistics().items()}
            action_names = state_names = ['x', 'y', 'z', 'rotvec_x', 'rotvec_y', 'rotvec_z']
        info['features']['action'] = {'dtype': 'float32', 'shape': [dimension], 'names': action_names + ['gripper_position']}
        info['features']['observation.state'] = {'dtype': 'float32', 'shape': [dimension], 'names': state_names + ['gripper_state']}
        info['droid_training_contract'] = contract
        atomic_json(work / 'meta/info.json', info)
        atomic_json(work / 'meta/stats.json', stats)
        atomic_json(work / 'training_view_manifest.json', contract)
        work.replace(destination)
        (destination / 'view_request.json').unlink()
    return destination
