"""Selected-corpus normalization for DROID consecutive EEF pose DP targets."""
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np

VIEW = runpy.run_path(str(Path(__file__).with_name('dataset_view.py')))


def prepare(dataset, output, *, horizon, episode_ids=None):
    import pandas as pd
    from lerobot.datasets.compute_stats import RunningQuantileStats
    from lerobot.processor.eef_delta_pose import to_eef_delta_pose_actions_numpy

    dataset, output = Path(dataset), Path(output)
    marker = dataset / 'training_view_manifest.json'
    view = json.loads(marker.read_text())
    if view.get('action') != 'absolute_eef_pose6_gripper_absolute1':
        raise ValueError('DROID delta-pose statistics require an absolute EEF-target view')
    if type(horizon) is not int or horizon < 1:
        raise ValueError('DP normalization horizon must be a positive integer')
    allowed = None if episode_ids is None else set(episode_ids)
    if allowed is not None and (not allowed or len(allowed) != len(episode_ids)):
        raise ValueError('DP statistics need unique nonempty selected episode IDs')
    contract = dict(schema='droid_eef_delta_pose_stats/v1', pose_input='eef_pose',
        dataset=str(dataset.resolve()), view_sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),
        chunk_size=horizon, episode_ids=None if allowed is None else sorted(allowed),
        convention='first_target_minus_current_measured_pose_then_consecutive_targets_base_frame_SO3',
        gripper='absolute_0_open_1_closed', drop_n_last_frames=horizon-1)
    if output.is_file():
        saved = json.loads(output.read_text())
        if {key: saved.get(key) for key in contract} != contract:
            raise ValueError(f'Existing DROID action stats use a different corpus/horizon: {output}')
        return output
    stats = RunningQuantileStats()
    seen, used, anchors = set(), [], 0
    for path in sorted((dataset / 'data').rglob('*.parquet')):
        rows = pd.read_parquet(path, columns=['episode_index', 'frame_index', 'action', 'observation.state'])
        for ep, episode in rows.groupby('episode_index', sort=True):
            ep = int(ep)
            if allowed is not None and ep not in allowed:
                continue
            if ep in seen:
                raise ValueError('DROID pose stats require complete per-episode parquet files')
            seen.add(ep)
            episode = episode.sort_values('frame_index')
            if list(episode.frame_index) != list(range(len(episode))):
                raise ValueError(f'DROID pose stats: incomplete episode {ep}')
            count = len(episode) - horizon + 1
            if count < 1:
                continue
            actions = np.stack(episode.action).astype(np.float64)
            states = np.stack(episode['observation.state']).astype(np.float64)
            if actions.shape[1] != 7 or states.shape[1] != 7:
                raise ValueError('DROID EEF targets and anchors must both be 7D')
            # Exactly the distribution of all unpadded training chunks. Compute
            # each action-to-action delta once, then account for every offset.
            stats.update(to_eef_delta_pose_actions_numpy(actions[:count], states[:count]))
            if horizon > 1:
                consecutive = to_eef_delta_pose_actions_numpy(actions[1:], actions[:-1])
                for offset in range(1, horizon):
                    stats.update(consecutive[offset-1:offset-1+count])
            used.append(ep)
            anchors += count
    if allowed is not None and seen != allowed:
        raise ValueError('DROID statistics are missing selected episodes')
    if not anchors:
        raise ValueError('No selected DROID episodes are long enough for the DP horizon')
    statistics = {key: value.tolist() for key, value in stats.get_statistics().items()}
    if not all(np.isfinite(v).all() for v in statistics.values()):
        raise ValueError('DROID delta-pose normalization produced non-finite statistics')
    VIEW['atomic_json'](output, dict(contract, action=statistics, used_episode_ids=used,
                                   anchors=anchors, action_samples=anchors*horizon))
    return output
