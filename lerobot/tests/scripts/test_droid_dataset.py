"""DROID-specific action/geometry/video alignment and resumable conversion."""
import importlib.util
import json
from pathlib import Path
import sys

import av
import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'lerobot/skillVLA/droid/configs/generate_training_dataset/download_dataset'
sys.path.insert(0, str(BASE / 'src'))
try:
    spec = importlib.util.spec_from_file_location('droid_prepare_test', BASE / 'src/prepare_dataset.py')
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    from camera_geometry import intrinsics, pose6_to_matrix, project, resize_geometry
finally:
    sys.path.pop(0)


def make_video(path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=60)
        stream.width, stream.height, stream.pix_fmt = 32, 24, 'yuv420p'
        stream.codec_context.thread_count = 1
        for i in range(frames):
            rgb = np.full((24, 32, 3), 20 + 25 * i, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.mark.parametrize(('captured', 'recorded', 'expected'), [
    ([1000, 1067, 1067, 1134, 1201], [1000, 1067, 1134], [0, 1, 1, 2, -1]),
    ([1000, 1067, 1134, 1201], [933, 1000, 1067, 1134], [1, 2, 3, -1]),
])
def test_recording_timestamps_restore_duplicates_and_extra_prefix(tmp_path, captured, recorded, expected):
    source = tmp_path / 'source'; source.mkdir()
    evidence = tmp_path / 'timestamps'; (evidence / '000000').mkdir(parents=True)
    h5path = source / 'trajectory.h5'
    with h5py.File(h5path, 'w') as h5:
        h5['action/gripper_position'] = np.zeros(len(captured))
        h5['observation/timestamp/cameras/20_frame_received'] = captured
    record = {'episode_id': 'test', 'serial': '20', 'h5_rows': len(captured),
              'video_frames': len(recorded), 'timestamps_ms': recorded, 'h5_to_video_frame': expected,
              'source_h5_size': h5path.stat().st_size, 'source_h5_mtime_ns': h5path.stat().st_mtime_ns,
              'source_svo_url': 'https://example.com/source.svo', 'source_svo_sha256': 'verified'}
    path = evidence / '000000/20.json'; path.write_text(json.dumps(record))
    (evidence / 'index.json').write_text(json.dumps({'episodes': {'test': '000000'},
                                                    'files': {'000000/20.json': prepare.sha256(path)}}))
    row = {'episode_id': 'test', 'directory': str(source), 'geometry': {'wrist': {'serial': '20'}}}
    cfg = {'recording_timestamps_root': str(evidence)}
    with h5py.File(h5path, 'r') as h5:
        mapping, provenance = prepare.camera_frame_map(h5, row, 0, 'wrist', len(recorded), cfg)
        assert mapping.tolist() == expected
        assert provenance['alignment'] == 'exact_recording_timestamp_ms'
        path.write_text('{}')
        with pytest.raises(ValueError, match='evidence changed'):
            prepare.camera_frame_map(h5, row, 0, 'wrist', len(recorded), cfg)


def test_transcode_reuses_camera_frame_for_repeated_observation(tmp_path):
    source = tmp_path / 'source.mp4'; make_video(source, 3)
    output = tmp_path / 'aligned.mp4'
    stats, decoded = prepare.transcode(source, output, np.array([0, 0, 2]),
        {'image_size': [24, 32], 'image_transform': 'letterbox', 'fps': 15,
         'encoder_threads': 1, 'video_crf': 20})
    with av.open(str(output)) as video:
        means = [frame.to_ndarray(format='rgb24').mean() for frame in video.decode(video=0)]
    assert decoded == 3 and stats['count'] == [3] and len(means) == 3
    assert abs(means[0] - means[1]) < 2 and means[2] - means[0] > 30


@pytest.fixture
def source(tmp_path):
    raw, calibration = tmp_path / 'raw', tmp_path / 'calibration'
    raw.mkdir(); calibration.mkdir()
    rows, entries, transforms = [], {}, {}
    for rank in range(2):
        uuid, relative = f'LAB+user+2026-01-0{rank+1}', f'LAB/success/day/episode_{rank}'
        episode = raw / relative
        episode.mkdir(parents=True)
        metadata = {'uuid': uuid, 'success': True, 'current_task': f'Move object {rank}',
                    'wrist_cam_serial': '10', 'ext1_cam_serial': '20', 'ext2_cam_serial': '30'}
        (episode / f'metadata_{uuid}.json').write_text(json.dumps(metadata))
        with h5py.File(episode / 'trajectory.h5', 'w') as h5:
            n = 8
            ee = np.zeros((n, 6)); ee[:, 0] = np.arange(n) / 100; ee[:, 2] = 1; ee[:, 5] = .1
            h5['observation/robot_state/cartesian_position'] = ee
            h5['observation/robot_state/gripper_position'] = np.linspace(0, .5, n)
            h5['observation/robot_state/joint_positions'] = np.ones((n, 7)) * .3
            for name, dim in prepare.ACTION_DIMS.items():
                data = np.ones((n, dim)) * .2
                if dim == 1:
                    data = data[:, 0]
                h5[f'action/{name}'] = data
            h5['observation/timestamp/control/step_start'] = np.arange(n) * 67 + 1700000000000
            active = np.ones(n, dtype=bool); active[2] = False
            h5['observation/controller_info/movement_enabled'] = active
            h5['observation/timestamp/skip_action'] = ~active
            for serial in ('10', '20', '30'):
                poses = np.zeros((n, 6)); poses[:, 0] = np.arange(n) / 10
                h5[f'observation/camera_extrinsics/{serial}_left'] = poses
                h5[f'observation/timestamp/cameras/{serial}_frame_received'] = np.arange(n) * 67
        for serial in ('10', '20', '30'):
            make_video(episode / f'recordings/MP4/{serial}.mp4', 7)
        rows.append({'episode_id': uuid, 'relative_path': relative, 'files': [
            {'name': f'robotics/droid_raw/1.0.1/{relative}/{p.relative_to(episode)}', 'size': p.stat().st_size}
            for p in sorted(episode.rglob('*')) if p.is_file()]})
        entries[uuid] = {serial: {'cameraMatrix': [20, 16, 21, 12], 'width': 32, 'height': 24, 'distCoeffs': [0]*12}
                         for serial in ('10', '20', '30')}
        transforms[uuid] = {'relative_path': relative, '20': [0]*6, '30': [0]*6,
                            '20_source': 'Pred', '30_source': 'GT'}
    (raw / 'download_manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (calibration / 'intrinsics.json').write_text(json.dumps(entries))
    (calibration / 'cam2base_extrinsic_superset.json').write_text(json.dumps(transforms))
    cfg = yaml.safe_load((BASE / 'dataset_config.yaml').read_text())
    cfg.update(raw_root=str(raw), calibration_root=str(calibration), dataset_root=str(tmp_path / 'datasets'),
               image_size=[24, 32], workers=2, output_name='converted', recording_timestamps_root='')
    path = tmp_path / 'config.yaml'; path.write_text(yaml.safe_dump(cfg))
    return prepare.load_settings(path)


def test_projection_and_intrinsics():
    entry = {'cameraMatrix': [100, 40, 120, 30], 'width': 80, 'height': 60, 'distCoeffs': [0]*12}
    original, scaled, pixel_transform = intrinsics(entry, 80, 60, [30, 40])
    np.testing.assert_equal(original, [[100, 0, 40], [0, 120, 30], [0, 0, 1]])
    camera = pose6_to_matrix([1, 0, 0, 0, 0, 0])
    uv, depth = project(np.array([[1, 0, 2], [1.2, .1, 2]]), camera, original)
    np.testing.assert_allclose(uv, [[40, 30], [50, 36]])
    np.testing.assert_allclose(depth, 2)
    resized, _ = project(np.array([[1.2, .1, 2]]), camera, scaled)
    np.testing.assert_allclose(resized, [[24.75, 17.75]])
    assert pixel_transform[0, 2] == -.25
    with pytest.raises(ValueError, match='distortion'):
        intrinsics({**entry, 'distCoeffs': [.1]}, 80, 60, [30, 40])


def test_letterbox_keeps_aspect_ratio_and_projects_into_padded_pixels():
    entry = {'cameraMatrix': [1000, 640, 1000, 360], 'width': 1280, 'height': 720, 'distCoeffs': [0]*12}
    height, width, top, left, transform = resize_geometry(1280, 720, [256, 256])
    assert (height, width, top, left) == (144, 256, 56, 0)
    assert transform[0, 0] == transform[1, 1] == .2
    _, matrix, _ = intrinsics(entry, 1280, 720, [256, 256])
    np.testing.assert_allclose(matrix, [[200, 0, 127.6], [0, 200, 127.6], [0, 0, 1]])
    uv, _ = project(np.array([[0., 0., 1.], [.1, .2, 1.]]), np.eye(4), matrix)
    np.testing.assert_allclose(uv, [[127.6, 127.6], [147.6, 167.6]])
    assert resize_geometry(640, 479, [256, 256])[:4] == (191, 256, 32, 0)


def test_encoded_letterbox_has_black_bars_and_matching_camera_matrix(source, tmp_path):
    cfg = {**source, 'image_size': [32, 32], 'image_transform': 'letterbox'}
    selected, _, _ = prepare.discover(cfg)
    cache = tmp_path / 'letterbox'
    prepare.convert_episode(selected[0], 0, cfg, cache)
    directory = cache / '000000'
    with av.open(str(directory / 'exterior_1.mp4')) as video:
        rgb = next(video.decode(video=0)).to_ndarray(format='rgb24')
    assert rgb.shape == (32, 32, 3)
    assert rgb[:2].max() < 8 and rgb[-2:].max() < 8
    assert abs(float(rgb[8:24].mean()) - 20) < 8
    with np.load(directory / 'cameras.npz', allow_pickle=False) as geometry:
        np.testing.assert_allclose(geometry['exterior_1.intrinsics'], [[20, 0, 16], [0, 21, 16], [0, 0, 1]])
        np.testing.assert_allclose(geometry['exterior_1.pixel_transform'], [[1, 0, 0], [0, 1, 4], [0, 0, 1]])
    provenance = json.loads((directory / 'episode.json').read_text())
    assert provenance['image_transform'] == 'letterbox'


def test_old_frozen_configuration_keeps_explicit_legacy_geometry(source, tmp_path):
    cfg = {**source}
    cfg.pop('image_transform')
    path = tmp_path / 'legacy.yaml'
    path.write_text(yaml.safe_dump(cfg))
    assert prepare.load_settings(path)['image_transform'] == 'stretch'


@pytest.mark.parametrize('action_mode,dimension', [('cartesian_velocity', 7), ('cartesian_position', 7),
                                                 ('joint_velocity', 8), ('joint_position', 8)])
def test_conversion_reader_alignment_and_source_preservation(source, action_mode, dimension):
    cfg = {**source, 'action_mode': action_mode}
    raw = Path(cfg['raw_root'])
    before = {str(p): prepare.sha256(p) for p in raw.rglob('*') if p.is_file()}
    selected, excluded, hashes = prepare.discover(cfg)
    assert not excluded
    output = prepare.convert(cfg, selected, excluded, hashes)
    info = prepare.validate_output(output, decode=True)
    assert info['total_episodes'] == 2 and info['total_frames'] == 12
    frame = pd.read_parquet(output / 'data/chunk-000/file-000.parquet')
    np.testing.assert_array_equal(frame['source.frame_index'].to_numpy(), [0, 1, 3, 4, 5, 6])
    np.testing.assert_allclose(np.stack(frame['observation.states.ee_state'])[:, 0], [0, .01, .03, .04, .05, .06])
    assert np.stack(frame.action).shape == (6, dimension)
    assert np.stack(frame['source.action.joint_velocity']).shape == (6, 7)
    assert 'source.action.cartesian_position' in frame
    with np.load(output / 'meta/droid/episodes/000000.npz') as geometry:
        np.testing.assert_allclose(geometry['wrist.camera_to_base'][:, 0, 3], [0, .1, .3, .4, .5, .6])
        np.testing.assert_equal(geometry['exterior_1.camera_to_base'][:, :3, 3], np.zeros((6, 3)))
    provenance = json.loads((output / 'meta/droid/episodes/000000.json').read_text())
    assert provenance['camera_provenance']['exterior_1']['source'] == 'Pred'
    assert provenance['dropped_tail_steps'] == 1
    assert before == {str(p): prepare.sha256(p) for p in raw.rglob('*') if p.is_file()}
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(repo_id='local/droid', root=output, video_backend='pyav')
    assert len(dataset) == 12
    sample = dataset[2]
    assert sample['action'].shape == (dimension,)
    assert sample['observation.images.image'].shape == (3, 24, 32)
    assert int(sample['source.frame_index']) == 3
    assert abs(float(sample['observation.images.image'].mean()) - 95/255) < .04
    before = (output / 'data/chunk-000/file-000.parquet').stat().st_mtime_ns
    prepare.convert(cfg, selected, excluded, hashes)
    assert (output / 'data/chunk-000/file-000.parquet').stat().st_mtime_ns == before
    with pytest.raises(FileExistsError, match='another conversion'):
        prepare.convert({**cfg, 'remove_skipped_steps': False}, selected, excluded, hashes)


def test_joint_mode_and_resume(source, tmp_path):
    cfg = {**source, 'action_mode': 'joint_velocity'}
    selected, _, _ = prepare.discover(cfg)
    cache = tmp_path / 'cache'
    done = prepare.convert_episode(selected[0], 0, cfg, cache)
    frame = pd.read_parquet(cache / '000000/frames.parquet')
    assert np.stack(frame.action).shape == (6, 8)
    np.testing.assert_allclose(np.stack(frame['observation.state'])[:, :7], .3)
    stamp = (cache / '000000/exterior_1.mp4').stat().st_mtime_ns
    assert prepare.convert_episode(selected[0], 0, cfg, cache)['length'] == done['length']
    assert (cache / '000000/exterior_1.mp4').stat().st_mtime_ns == stamp
    # Completed caches are never silently erased if damaged.
    (cache / '000000/exterior_1.mp4').write_bytes(b'damaged')
    with pytest.raises(ValueError, match='damaged'):
        prepare.convert_episode(selected[0], 0, cfg, cache)


@pytest.mark.parametrize('policy', ['error', 'skip'])
def test_empty_episode_policy_preserves_cache_and_compacts_indices(source, policy):
    cfg = {**source, 'empty_episode_policy': policy}
    selected, _, _ = prepare.discover(cfg)
    with h5py.File(Path(selected[0]['directory']) / 'trajectory.h5', 'r+') as h5:
        h5['observation/controller_info/movement_enabled'][:] = False
        h5['observation/timestamp/skip_action'][:] = True
    selected, excluded, hashes = prepare.discover(cfg)
    raw = Path(cfg['raw_root'])
    before = {str(p): prepare.sha256(p) for p in raw.rglob('*') if p.is_file()}
    output = Path(cfg['output_path'])
    cache = output.with_name('.' + output.name + '.building') / '.episodes'
    prepare.convert_episode(selected[1], 1, cfg, cache)
    video_stat = (cache / '000001/exterior_1.mp4').stat()
    if policy == 'error':
        with pytest.raises(prepare.EmptyEpisodeError, match='No active frames'):
            prepare.convert(cfg, selected, excluded, hashes)
        assert not output.exists()
        assert (cache / '000001/exterior_1.mp4').stat().st_mtime_ns == video_stat.st_mtime_ns
    else:
        prepare.convert(cfg, selected, excluded, hashes)
        info = prepare.validate_output(output, decode=True)
        assert info['total_episodes'] == 1 and info['total_frames'] == 6
        frame = pd.read_parquet(output / 'data/chunk-000/file-000.parquet')
        assert (frame.episode_index == 0).all() and (frame.task_index == 0).all()
        metadata = pd.read_parquet(output / 'meta/episodes/chunk-000/file-000.parquet')
        assert metadata.iloc[0]['stats/episode_index/max'] == [0]
        manifest = json.loads((output / 'conversion_manifest.json').read_text())
        assert manifest['source_episode_ids'] == [selected[1]['episode_id']]
        assert manifest['excluded'] == [{'episode_id': selected[0]['episode_id'],
                                         'cache_rank': 0, 'reason': 'no_active_frames'}]
        target = output / info['video_path'].format(video_key='observation.images.image', chunk_index=0, file_index=0)
        assert target.stat().st_ino == video_stat.st_ino
        assert target.stat().st_mtime_ns == video_stat.st_mtime_ns
    assert before == {str(p): prepare.sha256(p) for p in raw.rglob('*') if p.is_file()}


def test_missing_calibration_and_large_video_gap(source, tmp_path):
    cfg = source
    path = Path(cfg['calibration_root']) / 'intrinsics.json'
    entries = json.loads(path.read_text()); first = next(iter(entries)); del entries[first]['10']
    path.write_text(json.dumps(entries))
    selected, excluded, _ = prepare.discover(cfg)
    assert len(selected) == 1 and 'missing intrinsics' in excluded[0]['reason']
    with pytest.raises(ValueError, match='missing intrinsics'):
        prepare.discover({**cfg, 'missing_calibration': 'error'})
    directory = Path(selected[0]['directory'])
    for serial in ('10', '20', '30'):
        make_video(directory / f'recordings/MP4/{serial}.mp4', 5)
    with pytest.raises(ValueError, match='alignment'):
        prepare.convert_episode(selected[0], 0, cfg, tmp_path / 'cache')


def test_unknown_episode_and_unsafe_paths(source):
    with pytest.raises(ValueError, match='Unknown episode_ids'):
        prepare.discover({**source, 'episode_ids': ['unknown']})
    with pytest.raises(ValueError, match='relative path'):
        prepare.child(source['dataset_root'], '../raw')
