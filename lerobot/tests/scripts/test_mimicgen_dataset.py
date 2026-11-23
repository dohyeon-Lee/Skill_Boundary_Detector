import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

SRC = Path(__file__).resolve().parents[2] / 'skillVLA/mimicgen/configs/generate_training_dataset/download_dataset/src'


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, SRC / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


download = load('mimicgen_download', 'download.py')
# prepare_dataset is a command-line module with a sibling import.
sys.path.insert(0, str(SRC))
try:
    prepare = load('mimicgen_prepare', 'prepare_dataset.py')
finally:
    sys.path.pop(0)


def test_pose_is_radians_xyzw_and_signed_actions_are_preserved():
    quat = Rotation.from_rotvec([0, 0, np.pi / 2]).as_quat()
    frame = pd.DataFrame({'observation.eef': [[1, 2, 3, *quat]],
                          'observation.gripper': [[.02, -.02, 0, 0, 0, 0]],
                          'observation.joint': [np.zeros(7)], 'action': [[0, 0, 0, 0, 0, 0, -1]]})
    converted = prepare.converted_vectors(frame)
    np.testing.assert_allclose(converted['observation.state'][0], [1, 2, 3, 0, 0, np.pi / 2, .02, -.02], atol=1e-6)
    np.testing.assert_array_equal(converted['action'][0], frame.action.iloc[0])
    frame.at[0, 'observation.gripper'] = [0, 0, 1, 0, 0, 0]
    with pytest.raises(ValueError, match='non-padding'):
        prepare.converted_vectors(frame)


def test_negative_w_matches_actual_libero_processor_without_shortest_angle_flip():
    import torch
    from lerobot.processor.env_processor import LiberoProcessorStep
    q = Rotation.from_rotvec([3.3, .1, -.2]).as_quat()
    assert q[3] < 0
    frame = pd.DataFrame({'observation.eef': [[1, 2, 3, *q]],
                          'observation.gripper': [[0] * 6],
                          'observation.joint': [[0] * 7], 'action': [[0] * 7]})
    actual = prepare.converted_vectors(frame)['observation.states.ee_state'][0, 3:]
    expected = LiberoProcessorStep()._quat2axisangle(torch.tensor(q[None], dtype=torch.float32)).numpy()[0]
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert np.linalg.norm(actual) > np.pi


@pytest.fixture
def fixture_dataset(tmp_path):
    raw, ref = tmp_path / 'raw', tmp_path / 'reference'
    features = {k: {'dtype': 'float32', 'shape': [dim], 'names': None} for k, dim in prepare.VECTOR_DIMS.items()}
    for key in prepare.DEFAULT_KEYS:
        features[key] = {'dtype': 'float32' if key == 'timestamp' else 'int64', 'shape': [1], 'names': None}
    for key in prepare.IMAGE_MAP.values():
        features[key] = {'dtype': 'video', 'shape': [256, 256, 3], 'names': ['height', 'width', 'rgb'],
                         'info': {'video.codec': 'av1', 'video.fps': 20}}
    info = {'codebase_version': 'v3.0', 'fps': 20, 'features': features,
            'total_tasks': 1, 'total_episodes': 2, 'total_frames': 6,
            'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
            'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'}
    prepare.write_json(ref / 'meta/info.json', info)
    source_info = {**info, 'features': {source: features[target] for source, target in prepare.IMAGE_MAP.items()}}
    prepare.write_json(raw / 'meta/info.json', source_info)
    prepare.write_json(raw / 'source_manifest.json', {'repo': 'owner/coffee', 'revision': 'a' * 40})
    frames, episodes = [], []
    for ep in range(2):
        metadata = {'episode_index': ep, 'length': 3, 'dataset_from_index': ep * 3,
                    'dataset_to_index': ep * 3 + 3, 'tasks': ['make coffee'],
                    'data/chunk_index': 0, 'data/file_index': 0,
                    'meta/episodes/chunk_index': 0, 'meta/episodes/file_index': 0}
        for source in prepare.IMAGE_MAP:
            for name, value in [('chunk_index', 0), ('file_index', 0), ('from_timestamp', ep * 0.15),
                                ('to_timestamp', (ep + 1) * .15)]:
                metadata[f'videos/{source}/{name}'] = value
            for name, value in [('min', 0), ('max', 1), ('mean', .5), ('std', .1)]:
                metadata[f'stats/{source}/{name}'] = np.full((3, 1, 1), value).tolist()
            metadata[f'stats/{source}/count'] = [3]
            path = raw / 'videos' / source / 'chunk-000/file-000.mp4'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'test-video')
        episodes.append(metadata)
        for t in range(3):
            frames.append({'episode_index': ep, 'index': ep * 3 + t, 'frame_index': t, 'timestamp': t / 20,
                           'task_index': 0, 'observation.state': [0, 0],
                           'observation.eef': [ep, t, .3, 0, 0, 0, 1],
                           'observation.gripper': [.01, -.01, 0, 0, 0, 0],
                           'observation.joint': [0] * 7, 'action': [0, 0, 0, 0, 0, 0, (-1) ** t]})
    for name, rows in [('data/chunk-000/file-000.parquet', frames),
                        ('meta/episodes/chunk-000/file-000.parquet', episodes)]:
        path = raw / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
    pd.DataFrame({'task_index': [0]}, index=pd.Index(['make coffee'], name='task')).to_parquet(raw / 'meta/tasks.parquet')
    (raw / 'eval_env_meta').mkdir()
    (raw / 'eval_env_meta/demo.hdf5').write_bytes(b'fixture')
    return {'raw_path': str(raw), 'output_path': str(tmp_path / 'coffee'), 'schema_reference': str(ref),
            'source_repo': 'owner/coffee', 'source_revision': 'a' * 40, 'max_episodes': 1,
            'image_transform': 'identity', 'gripper_action': 'signed', 'video_storage': 'hardlink',
            'prepare_camera_geometry': False}


def test_conversion_reindexes_metadata_preserves_video_offsets_and_recomputes_stats(fixture_dataset):
    cfg = fixture_dataset
    prepare.convert(cfg)
    output = Path(cfg['output_path'])
    frame = pd.read_parquet(output / 'data/chunk-000/file-000.parquet')
    ep = pd.read_parquet(output / 'meta/episodes/chunk-000/file-000.parquet')
    assert len(frame) == 3 and len(ep) == 1
    assert len(frame['observation.state'].iloc[0]) == 8
    assert 'observation.eef' not in frame
    assert ep['videos/observation.images.image/to_timestamp'].iloc[0] == .15
    stats = json.loads((output / 'meta/stats.json').read_text())
    assert len(stats['observation.state']['q99']) == 8
    np.testing.assert_allclose(stats['observation.state']['mean'][:3], [0, 1, .3])
    assert (output / 'videos/observation.images.image/chunk-000/file-000.mp4').stat().st_ino == (
        Path(cfg['raw_path']) / 'videos/observation.images.agentview/chunk-000/file-000.mp4').stat().st_ino
    prepare.convert(cfg)  # Same request reuses completed output.
    with pytest.raises(FileExistsError):
        prepare.convert({**cfg, 'max_episodes': 'all'})


class Response:
    def __init__(self, data, status=200, headers=None):
        self.data, self.status_code, self.headers = data, status, headers or {}
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def raise_for_status(self): pass
    def iter_content(self, size): yield self.data


@pytest.mark.parametrize('honor_range', [False, True])
def test_downloader_never_appends_full_response_to_partial(tmp_path, honor_range):
    data = b'coffee fixture download'
    (tmp_path / 'data.part').write_bytes(data[:4])
    class Client:
        def get(self, url, **kwargs):
            assert kwargs['headers']['Range'].startswith('bytes=4-')
            return Response(data[4:], 206, {'Content-Range': f'bytes 4-{len(data)-1}/{len(data)}'}) if honor_range else Response(data)
    row = {'path': 'data', 'size': len(data), 'lfs': {'oid': hashlib.sha256(data).hexdigest()}}
    download.download_one(Client(), 'owner/coffee', 'a'*40, row, tmp_path, 0, 1)
    assert (tmp_path / 'data').read_bytes() == data


def test_unsafe_source_paths_are_rejected(tmp_path):
    with pytest.raises(ValueError, match='Unsafe'):
        download.local_path(tmp_path, '../other-model')
