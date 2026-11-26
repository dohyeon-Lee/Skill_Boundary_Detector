"""Reuse SBD boundaries while reading DROID EEF targets for pose-space FSQ.

Source skill NPZs and dataset actions are never rewritten. Cartesian target
Euler angles become rotation vectors exactly as in DROID's DP pose view.
"""
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np


def load_pose_skills(skills_dir, dataset, *, episode_ids=None):
    import pandas as pd
    from scipy.spatial.transform import Rotation

    skills_dir, dataset = Path(skills_dir), Path(dataset)
    allowed = None if episode_ids is None else set(episode_ids)
    if allowed is not None and (not allowed or len(allowed) != len(episode_ids)):
        raise ValueError('FSQ pose selection requires unique nonempty episode IDs')
    records, by_episode = [], {}
    for path in sorted(skills_dir.rglob('*.npz')):
        with np.load(path, allow_pickle=False) as d:
            ep = int(d['episode_id'])
            if allowed is not None and ep not in allowed:
                continue
            start, end = int(d['frame_start']), int(d['frame_end'])
            action = d['actions'].astype(np.float32)
            if start < 0 or end <= start or end-start != len(action):
                raise ValueError(f'Invalid FSQ source boundaries: {path}')
            metadata = dict(file=path.name, episode_id=ep,
                task_id=int(d['task_id']) if 'task_id' in d else -1,
                skill_index=int(d['skill_index']), frame_start=start, frame_end=end, length=len(action))
        row = len(records)
        records.append((metadata, path))
        by_episode.setdefault(ep, []).append(row)
    if not records:
        raise ValueError('FSQ pose selection contains no skills')
    states, actions = [None]*len(records), [None]*len(records)
    seen = set()
    columns = ['episode_index', 'frame_index', 'action', 'source.action.cartesian_position',
        'source.action.gripper_position', 'observation.states.ee_state', 'observation.states.gripper_state']
    for path in sorted((dataset / 'data').rglob('*.parquet')):
        # Skip unrelated parquet row groups before reading their numeric data.
        ids = pd.read_parquet(path, columns=['episode_index']).episode_index.unique()
        if not any(int(ep) in by_episode for ep in ids):
            continue
        rows = pd.read_parquet(path, columns=columns)
        for ep, episode in rows.groupby('episode_index', sort=False):
            ep = int(ep)
            if ep not in by_episode:
                continue
            if ep in seen:
                raise ValueError('FSQ pose input requires complete per-episode parquet files')
            seen.add(ep)
            episode = episode.sort_values('frame_index')
            if episode.frame_index.tolist() != list(range(len(episode))):
                raise ValueError(f'Incomplete DROID episode {ep}')
            target = np.stack(episode['source.action.cartesian_position']).astype(np.float32)
            pose_actions = np.c_[target[:, :3], Rotation.from_euler('xyz', target[:, 3:]).as_rotvec(),
                np.asarray(episode['source.action.gripper_position'].tolist()).reshape(-1, 1)].astype(np.float32)
            pose_states = np.c_[np.stack(episode['observation.states.ee_state']),
                np.asarray(episode['observation.states.gripper_state'].tolist()).reshape(-1, 1)].astype(np.float32)
            recorded = np.stack(episode.action).astype(np.float32)
            if pose_actions.shape != (len(episode), 7) or pose_states.shape != pose_actions.shape:
                raise ValueError(f'Invalid DROID EEF pose fields in episode {ep}')
            if not np.isfinite(pose_actions).all() or not np.isfinite(pose_states).all():
                raise ValueError(f'Nonfinite DROID EEF pose in episode {ep}')
            for row in by_episode[ep]:
                metadata, source = records[row]
                start, end = metadata['frame_start'], metadata['frame_end']
                if end > len(episode):
                    raise ValueError(f'Skill exceeds DROID episode {ep}: {source}')
                with np.load(source, allow_pickle=False) as data:
                    original = data['actions']
                reference = pose_actions if original.shape[1] == 7 else recorded
                if original.shape != reference[start:end].shape or not np.allclose(original, reference[start:end], atol=1e-5, rtol=1e-5):
                    raise ValueError(f'SBD skill and DROID source actions disagree: {source}')
                states[row] = pose_states[start:end].copy()
                actions[row] = pose_actions[start:end].copy()
    if seen != set(by_episode):
        raise ValueError(f'DROID pose source episodes missing: {sorted(set(by_episode)-seen)}')
    # The shared action-sequence preset encodes actions; retain aligned EEF
    # state trajectories for anchors, augmentation and optional conditioning.
    print(f'[DROID FSQ] Reused {len(records)} skill boundaries; loaded 7D EEF targets', flush=True)
    return states, states, actions, [item[0] for item in records]


def write_stats(output, dataset, skillset, states, actions, metadata):
    """Independent pose statistics over the selected clean skills."""
    from lerobot.processor.eef_delta_pose import to_eef_delta_pose_actions_numpy

    dataset, skillset, output = Path(dataset), Path(skillset), Path(output)
    state = np.concatenate(states)
    delta = np.concatenate([to_eef_delta_pose_actions_numpy(a, s[0])
                            for a, s in zip(actions, states, strict=True)])
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    contract = dict(schema='droid_fsq_delta_pose_stats/v1',
        source_skillset=str(skillset.resolve()), source_dataset=str(dataset.resolve()),
        source_view_sha256=digest(dataset / 'training_view_manifest.json'),
        skillset_manifest_sha256=digest(skillset / 'skillset_manifest.json'),
        episode_ids=sorted({m['episode_id'] for m in metadata}),
        convention='first_target_minus_current_measured_pose_then_consecutive_targets_base_frame_SO3',
        gripper='absolute_axis6', skill_count=len(metadata))
    if output.is_file():
        saved = json.loads(output.read_text())
        if saved.get('contract') != contract:
            raise ValueError(f'Existing FSQ pose statistics belong to another source: {output}')
    document = {'contract': contract}
    for key, values in [('observation.state', state), ('action', delta)]:
        q01, q99 = np.quantile(values, [.01, .99], axis=0)
        document[key] = dict(q01=q01.tolist(), q99=q99.tolist())
    runpy.run_path(str(Path(__file__).with_name('dataset_view.py')))['atomic_json'](output, document)
    return output
