#!/usr/bin/env python3
"""ABC entry point with explicit DP episode-subset forwarding."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

CONFIG_SRC = Path(__file__).resolve().parents[2] / 'src'
_config_spec = importlib.util.spec_from_file_location(
    'abc_skill_dataset_config', CONFIG_SRC / 'train_skills_config.py')
_config_module = importlib.util.module_from_spec(_config_spec)
_config_spec.loader.exec_module(_config_module)
load_episode_selection = _config_module.load_episode_selection


def load_shared_builder():
    path = next(parent / 'libero/configs/train_skills/build_data/src/build_skill_dataset.py'
                for parent in Path(__file__).resolve().parents
                if (parent / 'libero/configs/train_skills/build_data/src/build_skill_dataset.py').is_file())
    spec = importlib.util.spec_from_file_location('abc_shared_skill_builder', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def selection_contract(payload: dict) -> dict:
    return {
        'fingerprint': payload['fingerprint'],
        'episode_count': payload['episode_count'],
        'episode_ids_sha256': hashlib.sha256(
            json.dumps(sorted(payload['episodes']), separators=(',', ':')).encode()).hexdigest(),
    }


def verify_selected_output(output: Path, payload: dict) -> None:
    import re
    expected = set(payload['episodes'])
    contract = json.loads((output / 'skillset_manifest.json').read_text())
    if contract.get('episode_selection') != selection_contract(payload):
        raise ValueError('ABC output episode-selection provenance mismatch.')
    for folder, pattern in [('curves', '*.npz'), ('skills', '*.done')]:
        paths = list((output / folder).rglob(pattern))
        ids = [int(re.match(r'ep(\d+)', path.name).group(1)) for path in paths]
        if len(ids) != len(expected) or set(ids) != expected:
            raise ValueError(f'ABC {folder} do not exactly cover selected episodes: '
                             f'{len(ids)}/{len(expected)}')


def run(shared, args) -> None:
    tasks = [value.strip() for value in args.source_tasks.split(',') if value.strip()]
    payload = load_episode_selection(Path(args.dataset_dir), tasks, args.episode_selection_manifest)
    output = Path(args.output_dir)
    manifest_path = output / 'skillset_manifest.json'
    contract = selection_contract(payload) if payload else None
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text()).get('episode_selection')
        if existing != contract:
            raise ValueError('Existing skillset has a different ABC episode selection; refusing reuse.')
    shard_path = getattr(args, 'episode_ids_path', '')
    assigned = None
    if shard_path:
        ids = json.loads(Path(shard_path).read_text())['episode_ids']
        assigned = set(int(ep) for ep in ids)
        if not assigned or len(assigned) != len(ids):
            raise ValueError('ABC episode shard must contain unique episode IDs.')
        if payload and not assigned.issubset(payload['episodes']):
            raise ValueError('ABC worker shard contains episodes outside the DP selection.')
    if payload or assigned is not None:
        selected = set(payload['episodes']) if payload else None
        load_meta = shared.load_episodes_meta
        iter_episodes = shared._iter_state_action_episodes
        make_manifest = shared._skillset_manifest
        get_pca = shared.get_or_fit_action_pca

        def selected_meta(dataset_dir):
            rows = load_meta(dataset_dir)
            if selected is not None:
                rows = rows[rows['episode_index'].isin(selected)]
            elif assigned is not None:
                rows = rows[rows['source_task'].isin(tasks)]
            if selected is not None and set(rows['episode_index']) != selected:
                raise ValueError('ABC metadata does not contain every selected episode.')
            if assigned is not None:
                rows = rows[rows['episode_index'].isin(assigned)]
                if set(rows['episode_index']) != assigned:
                    raise ValueError('ABC worker shard contains missing or out-of-task episodes.')
            return rows.reset_index(drop=True)

        def selected_episodes(dataset_dir, source_tasks=()):
            from lerobot.datasets.abc_native_io import iter_abc_native_state_action
            yield from iter_abc_native_state_action(
                dataset_dir, episode_ids=selected, source_tasks=tasks)

        def selected_manifest(*a, **kw):
            manifest = make_manifest(*a, **kw)
            manifest['source_tasks'] = tasks
            if contract:
                manifest['episode_selection'] = contract
            return manifest

        def selected_pca(path, metadata, fit):
            metadata['source_tasks'] = tasks
            if contract:
                metadata['episode_selection'] = contract
            return get_pca(path, metadata, fit)

        shared.load_episodes_meta = selected_meta
        shared._iter_state_action_episodes = selected_episodes
        shared._skillset_manifest = selected_manifest
        shared.get_or_fit_action_pca = selected_pca
        if payload:
            print(f"ABC DP episode selection: {len(selected)} episodes, {payload['fingerprint']}", flush=True)
        if assigned is not None:
            print(f"ABC worker episode shard: {len(assigned)} episodes", flush=True)
        original_source_tasks = args.source_tasks
        try:
            if assigned is not None:
                # The shared validator checks that every named task has rows.
                # Worker execution uses only its present tasks; PCA and immutable
                # output provenance above always retain the complete selection.
                rows = selected_meta(Path(args.dataset_dir))
                args.source_tasks = ','.join(task for task in tasks
                                            if task in set(rows['source_task']))
            shared.main(args)
        finally:
            args.source_tasks = original_source_tasks
            shared.load_episodes_meta = load_meta
            shared._iter_state_action_episodes = iter_episodes
            shared._skillset_manifest = make_manifest
            shared.get_or_fit_action_pca = get_pca
    else:
        shared.main(args)


def main():
    shared = load_shared_builder()

    @dataclass
    class Args(shared.Args):
        episode_selection_manifest: str = ''
        episode_ids_path: str = ''

    run(shared, shared.tyro.cli(Args))


if __name__ == '__main__':
    main()
