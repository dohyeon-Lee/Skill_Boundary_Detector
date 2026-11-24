"""Immutable UUID groups, resolved to the current dataset's intact episode IDs.

This is independent of idle/start filtering. Never filter frames or renumber
episodes: DP, SBD, FSQ and future PT/FT adapters share this selection contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'lerobot/src/lerobot').is_dir())
BUILTIN_GROUPS = {'without_deformable': dict(
    manifest=str(Path(__file__).resolve().parents[1] / 'dataset_groups/without_deformable.v1.json'),
    sha256='65f678841ea09ba090f2ae080b06887520c799ebcb032e41d1219958305c2923')}


def definition(config):
    selected = config.get('droid_dataset')
    name = selected.get('group', 'all') if isinstance(selected, dict) else config.get('droid_dataset_group', 'all')
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('droid_dataset.group must be all or a defined group name')
    if name == 'all':
        return None
    groups = config.get('droid_dataset_groups', BUILTIN_GROUPS)
    if not isinstance(groups, dict) or name not in groups:
        raise ValueError(f'Unknown DROID dataset group: {name}')
    value = groups[name]
    if not isinstance(value, dict) or not value.get('manifest') or not value.get('sha256'):
        raise ValueError(f'DROID group {name} requires manifest and sha256')
    path = Path(value['manifest']).expanduser()
    if not path.is_absolute():
        path = Path(config.get('project_root', ROOT)) / path
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != value['sha256']:
        raise ValueError(f'DROID group manifest SHA256 mismatch: {path}')
    document = json.loads(content)
    ids = document.get('source_episode_ids')
    if (document.get('group') != name or not isinstance(ids, list) or not ids
            or any(not isinstance(v, str) or not v for v in ids) or len(ids) != len(set(ids))):
        raise ValueError(f'Invalid DROID UUID group manifest: {path}')
    return dict(name=name, path=path.resolve(), sha256=digest, source_episode_ids=ids)


def suffix(config):
    group = definition(config)
    return '_' + group['name'] if group else ''


def tagged(value, config):
    tag = suffix(config)
    return value if not tag or value.endswith(tag) else value + tag


def episode_mapping(dataset):
    """Use preserved UUID provenance, including derived SkillVLA metadata."""
    import pandas as pd
    dataset = Path(dataset)
    parts = sorted((dataset / 'meta/episodes').rglob('*.parquet'))
    if not parts:
        raise FileNotFoundError(f'DROID episode metadata missing: {dataset}/meta/episodes')
    ids = sorted(int(v) for p in parts for v in pd.read_parquet(p, columns=['episode_index']).episode_index)
    if not ids or len(ids) != len(set(ids)) or any(v < 0 for v in ids):
        raise ValueError('DROID dataset has empty, duplicate or negative episode IDs')
    # Reading a complete conversion manifest avoids thousands of tiny JSON reads.
    marker = dataset / 'conversion_manifest.json'
    if not marker.is_file() and (dataset / 'training_view_manifest.json').is_file():
        view = json.loads((dataset / 'training_view_manifest.json').read_text())
        marker = Path(view['source']) / 'conversion_manifest.json'
    source_ids = json.loads(marker.read_text()).get('source_episode_ids') if marker.is_file() else None
    if source_ids is not None and ids == list(range(len(source_ids))):
        mapping = dict(enumerate(source_ids))
    else:
        mapping = {ep: json.loads((dataset / f'meta/droid/episodes/{ep:06d}.json').read_text())['episode_id']
                   for ep in ids}
    if any(not isinstance(v, str) or not v for v in mapping.values()) or len(set(mapping.values())) != len(mapping):
        raise ValueError('DROID episode UUID provenance is empty or duplicated')
    return mapping


def select(config, dataset):
    """Return None for legacy/all, otherwise a reproducible read-time selection."""
    group = definition(config)
    if group is None:
        return None
    mapping = episode_mapping(dataset)
    allowed = set(group['source_episode_ids'])
    ids = [ep for ep, uuid in mapping.items() if uuid in allowed]
    if not ids:
        raise ValueError(f'DROID group {group["name"]} selects no episodes in {dataset}')
    pairs = [(ep, mapping[ep]) for ep in ids]
    return dict(group=group['name'], group_manifest_sha256=group['sha256'],
                selection_sha256=hashlib.sha256(json.dumps(pairs, separators=(',', ':')).encode()).hexdigest(),
                total_episodes=len(mapping), selected_episodes=len(ids),
                episode_ids=ids, source_episode_ids=[mapping[ep] for ep in ids])


def selected_episodes(config, dataset, episodes=None):
    """Common DP / FSQ / PT_VSA,PRED,TERM / FT_VSA,PRED,TERM / build API."""
    selection = select(config, dataset)
    if selection is None:
        return episodes
    if episodes is None:
        return selection['episode_ids']
    allowed = set(selection['episode_ids'])
    result = [ep for ep in episodes if ep in allowed]
    if not result:
        raise ValueError('DROID group and requested episode subset have no intersection')
    return result


def write_selection(path, selection):
    """Refuse accidental cache/resume mixing, without mutating source metadata."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != selection:
            raise ValueError(f'DROID selection changed; choose a new output: {path}')
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete file without replacing another submitter's contract.
    import os
    import tempfile
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(selection, stream, indent=2)
            stream.write('\n')
        try:
            os.link(temporary, path)
        except FileExistsError:
            if json.loads(path.read_text()) != selection:
                raise ValueError(f'Conflicting DROID selection: {path}')
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def freeze(global_path):
    """Bundle the selected UUID file next to a submit-time global snapshot."""
    import shutil
    import yaml
    path = Path(global_path)
    config = yaml.safe_load(path.read_text())
    if config.get('workflow_domain') != 'droid':
        return
    # Relative manifest paths in an existing bundle refer to that bundle.
    selected = config.get('droid_dataset')
    name = selected.get('group', 'all') if isinstance(selected, dict) else config.get('droid_dataset_group', 'all')
    entry = config.get('droid_dataset_groups', {}).get(name)
    if entry and not Path(entry['manifest']).is_absolute():
        local = path.parent / entry['manifest']
        if local.is_file():
            config['droid_dataset_groups'][name]['manifest'] = str(local)
    group = definition(config)
    if group is None:
        return
    destination = path.parent / ('dataset_group_' + group['sha256'] + '.json')
    if group['path'] != destination.resolve():
        shutil.copyfile(group['path'], destination)
    config.setdefault('droid_dataset_groups', {})[name] = dict(
        manifest=str(destination.resolve()), sha256=group['sha256'])
    path.write_text(yaml.safe_dump(config, sort_keys=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
