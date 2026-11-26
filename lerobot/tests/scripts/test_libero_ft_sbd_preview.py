"""Guard episode identity and saved-boundary selection in the FT comparison."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[2] / 'skillVLA/libero/configs/train_skillVLA/FT/src/visualize_sbd.py'
spec = importlib.util.spec_from_file_location('libero_ft_sbd_preview', PATH)
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


def metadata(dataset, ids, info):
    directory = dataset / 'meta/episodes'
    directory.mkdir(parents=True)
    pd.DataFrame({'episode_index': ids, 'length': [100] * len(ids)}).to_parquet(directory / 'file.parquet')
    (dataset / 'meta/info.json').write_text(json.dumps(info))


def test_subset_comparison_uses_original_recording_not_local_id(tmp_path):
    full, subset = tmp_path / 'full', tmp_path / 'subset'
    metadata(full, [0, 1, 12, 27], {})
    metadata(subset, [0, 1], {'source_dataset': 'full', 'source_episode_indices': [12, 27]})
    assert preview.original_episode_ids(subset) == {0: ('full', 12), 1: ('full', 27)}
    # Same local ID zero refers to different recordings in these two datasets.
    assert preview.original_episode_ids(full)[0] != preview.original_episode_ids(subset)[0]


def test_subset_mapping_rejects_ambiguous_origins(tmp_path):
    metadata(tmp_path, [0, 1], {'source_dataset': 'parent', 'source_episode_indices': [7, 7]})
    with pytest.raises(ValueError, match='Invalid source_episode_indices'):
        preview.original_episode_ids(tmp_path)


def test_preview_rejects_recut_boundaries_instead_of_displaying_them(tmp_path):
    curves = tmp_path / 'curves'
    curves.mkdir()
    np.savez(curves / 'ep0000007.npz', boundaries=[0, 40, 100], mean_val=.2)
    bounds = {key: np.array(value) for key, value in {
        'episode_id': [7, 7], 'task_id': [3, 3], 'skill_index': [1, 0],
        'frame_start': [40, 0], 'frame_end': [100, 40], 'tokens': [5, 8],
    }.items()}
    build = {'bounds': bounds, 'skillset': tmp_path, 'run': tmp_path,
             'source': 'source', 'metadata': pd.DataFrame({'length': [100]}, index=[7])}
    assert preview.saved_episode(build, 7)['skills'] == [(0, 40, 8), (40, 100, 5)]
    np.savez(curves / 'ep0000007.npz', boundaries=[0, 45, 100], mean_val=.2)
    with pytest.raises(ValueError, match='differ from FT labels'):
        preview.saved_episode(build, 7)


def test_preview_does_not_confuse_pt_reference_and_own_threshold_caches(tmp_path):
    source = tmp_path / 'full'
    contract = {'segmentation': {'mode': 'gaussian_x0hat', 'dataset_name': 'full',
                               'detector': {'boundary_threshold_mode': 'global_mean', 'initial_mask_frames': 0}}}
    for name, own in [('globalref', False), ('globalmean', True)]:
        directory = source / '_work' / name / 'skillset'
        directory.mkdir(parents=True)
        threshold = directory / 'global_boundary_threshold.json' if own else tmp_path / 'pt/global_boundary_threshold.json'
        manifest = {'mode': 'gaussian_x0hat', 'dataset_name': 'full',
                    'policy_path': '/outputs/DP/model/checkpoints/010000/pretrained_model',
                    'detector': {'boundary_threshold_mode': 'global_mean', 'global_threshold_path': str(threshold)}}
        (directory / 'skillset_manifest.json').write_text(json.dumps(manifest))
    reference, _ = preview.find_skillset(source / 'FSQ333_ft_grounded', contract)
    own, _ = preview.find_skillset(source / 'FSQ333_ft_own_grounded', contract)
    assert reference.parent.name == 'globalref'
    assert own.parent.name == 'globalmean'
