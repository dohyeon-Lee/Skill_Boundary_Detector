"""Read-only DROID training adapters for the common stage switches."""
from pathlib import Path
import json
import runpy

import numpy as np

_core = runpy.run_path(str(Path(__file__).with_name('idle_filter.py')))
TrainingIdleFilter = _core['TrainingIdleFilter']


def raw_episode_mask(root, episode):
    """Compute on original H5 commands, then map to converted frame indices."""
    import h5py
    root = Path(root)
    provenance = json.loads((root / f'meta/droid/episodes/{int(episode):06d}.json').read_text())
    source = Path(provenance['directory']) / 'trajectory.h5'
    if not source.is_file():
        raise FileNotFoundError(f'Exact DROID idle filtering needs original commands: {source}')
    with h5py.File(source, 'r') as stream:
        mask = _core['nonidle_start_mask'](np.asarray(stream['action/joint_velocity']))
    geometry = root / f'meta/droid/episodes/{int(episode):06d}.npz'
    with np.load(geometry) as value:
        indices = _core['_frame_indices'](value['source_frame_index'], len(mask))
    return mask[indices]


def dataset_mask(dataset, config, stage):
    selector = TrainingIdleFilter(config, stage)
    if not selector.enabled:
        return np.ones(len(dataset), dtype=bool)
    table = dataset.hf_dataset
    columns = ['episode_index', 'frame_index']
    if stage.endswith('_TERM'):
        columns += ['skill_ds', 'skill_de']
    rows = table.select_columns(columns).with_format('numpy')[:]
    keep = np.zeros(len(dataset), dtype=bool)
    for episode in np.unique(rows['episode_index']):
        selected = np.flatnonzero(rows['episode_index'] == episode)
        mask = raw_episode_mask(dataset.root, int(episode))
        keep[selected] = mask[rows['frame_index'][selected].astype(np.int64)]
    if stage.endswith('_TERM'):
        keep |= _core['termination_protection_mask'](
            rows['skill_ds'], rows['skill_de'],
            post_end_steps=int(getattr(dataset, '_terminator_post_end_steps', 0)))
    return keep


def apply_frame_sampler(dataset, settings, sampler, *, grouped=False):
    """Called after the common trainer has selected tail/transition sampling."""
    stage = settings['stage']
    selector = TrainingIdleFilter(settings, stage)
    if not selector.enabled:
        return sampler
    if grouped:
        raise ValueError('DROID idle filtering must be applied to final observations inside grouped/recurrent sampling')
    mask = dataset_mask(dataset, settings, stage)
    if sampler is not None and not hasattr(sampler, 'indices'):
        raise ValueError('DROID idle filtering requires explicit candidate indices from the existing sampler')
    candidates = None if sampler is None else sampler.indices
    weights = getattr(sampler, 'weights', None)
    if hasattr(weights, 'numpy'):
        weights = weights.numpy()
    result = selector.sampler(mask, candidates=candidates, weights=weights)
    print(f'DROID idle filter {stage}: {len(result)} eligible starts / {len(dataset)} frames', flush=True)
    return result


def fsq_skill_mask(root, metadata, config):
    """Keep all trajectories available for validation/encoding; select training IDs only."""
    selector = TrainingIdleFilter(config, 'FSQ')
    keep = np.ones(len(metadata), dtype=bool)
    if not selector.enabled:
        return keep
    by_episode = {}
    for index, item in enumerate(metadata):
        by_episode.setdefault(int(item['episode_id']), []).append((index, item))
    for episode, skills in by_episode.items():
        mask = raw_episode_mask(root, episode)
        for index, item in skills:
            start, end = int(item['frame_start']), int(item['frame_end'])
            if not 0 <= start < end <= len(mask):
                raise ValueError(f'Skill exceeds DROID episode: {item}')
            keep[index] = mask[start:end].any()
    return keep
