import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.utils.eval_edit_metadata import episode_edit_metadata


def test_exact_transition_timing_differs_from_sampled_video_and_keeps_reentries():
    trace = [dict(batch_index=0, skill_index=i, codebook_token=7, episode_timestep=step)
             for i, step in enumerate([0, 3, 9, 12])]
    trace.append(dict(batch_index=1, skill_index=0, codebook_token=99, episode_timestep=1))
    metadata = episode_edit_metadata(
        trace, batch_index=0, episode_index=2, episode_steps=12, sim_fps=20,
        success=True, seed=1002, task_description='task', video_frame_stride=4,
        video_fps=10, annotated_video='eval_episode_2.mp4',
    )
    assert [s['end_step_exclusive'] for s in metadata['skills']] == [3, 9, 12]
    assert [t['raw_frame'] for t in metadata['transitions']] == [3, 9]
    assert [t['sim_time_s'] for t in metadata['transitions']] == [0.15, 0.45]
    assert [t['annotated_frame'] for t in metadata['transitions']] == [1, 3]
    assert [t['annotated_time_s'] for t in metadata['transitions']] == [0.1, 0.3]
    assert metadata['annotated_video']['frame_count'] == 3
    assert len(metadata['skills']) == 3  # Identical tokens still have distinct occurrences.


@pytest.mark.parametrize('collect_html', [False, True])
def test_evaluator_saves_native_frames_before_overlays_and_metadata_for_unfilmed_episodes(
    monkeypatch, tmp_path, collect_html,
):
    import torch
    import lerobot.scripts.lerobot_skillvla_eval as m
    import lerobot.utils.eval_edit_metadata as editing

    class Env:
        num_envs = 1
        metadata = {'render_fps': 20}
        value = 0

        def __init__(self):
            self.envs = [self]
            self.unwrapped = self

        def render(self):
            return np.full((16, 32, 3), self.value, dtype=np.uint8)

    class Policy:
        config = SimpleNamespace()

        def eval(self):
            pass

        def get_skill_trace(self):
            return [dict(batch_index=0, skill_index=i, codebook_token=i, episode_timestep=s)
                    for i, s in enumerate([0, 3, 5])]

    def rollout(**kwargs):
        env = kwargs['env']
        for value in [0, 1, 2, 3, 4, 5, 255]:  # Last render is an auto-reset observation.
            env.value = value
            if kwargs['render_callback']:
                kwargs['render_callback'](env)
        return dict(done=torch.tensor([[False] * 5 + [True]]),
                    reward=torch.zeros((1, 6)), success=torch.tensor([[False] * 5 + [True]]))

    written = {}

    def write_video(path, frames, fps):
        written[str(path)] = (frames.copy(), fps)
        Path(path).write_bytes(b'encoded')

    def overlay(frames, *args, **kwargs):
        frames[:] = 99  # Even an in-place overlay must not contaminate saved raw frames.
        return frames

    monkeypatch.setattr(m, 'PreTrainedPolicy', Policy)
    monkeypatch.setattr(m.gym.vector, 'SyncVectorEnv', Env)
    monkeypatch.setattr(m, 'rollout', rollout)
    monkeypatch.setattr(m, 'write_video', write_video)
    monkeypatch.setattr(editing, 'write_simulator_video', write_video)
    monkeypatch.setattr(m, '_annotate_eval_video', overlay)
    for name in ['_goal_overlay_pixels', '_video_goal_values', '_fullskill_overlay_pixels']:
        monkeypatch.setattr(m, name, lambda *a, **k: (None, None))
    monkeypatch.setattr(m, '_bottleneck_overlay_pixels', lambda *a, **k: None)
    info = m.eval_policy(
        Env(), Policy(), None, None, None, None, n_episodes=2,
        max_episodes_rendered=1, video_frame_stride=4, video_fps=10,
        raw_videos=True, record_skill_transitions=True, collect_skill_html=collect_html,
        videos_dir=tmp_path, start_seed=1000,
    )
    raw, fps = next(value for path, value in written.items() if '/raw/' in path)
    assert raw.shape == (6, 16, 32, 3)
    np.testing.assert_array_equal(raw[:, 0, 0, 0], np.arange(6))
    assert fps == 20
    assert len(info['raw_video_paths']) == 1
    assert len(info['episode_metadata_paths']) == 2
    first, second = [json.loads(Path(p).read_text()) for p in info['episode_metadata_paths']]
    assert first['raw_video']['frame_count'] == 6
    assert first['raw_video']['overlays'] is False
    assert first['annotated_video']['frame_count'] == 2
    assert [t['step'] for t in first['transitions']] == [3, 5]
    assert second['raw_video'] is None and second['annotated_video'] is None
    assert [first['seed'], second['seed']] == [1000, 1001]


def test_native_encoder_keeps_non_macroblock_dimensions_and_all_frames(tmp_path):
    import imageio.v2 as imageio
    from lerobot.utils.eval_edit_metadata import write_simulator_video

    path = tmp_path / 'raw.mp4'
    frames = np.stack([np.full((18, 34, 3), i * 30, dtype=np.uint8) for i in range(6)])
    write_simulator_video(path, frames, 20)
    reader = imageio.get_reader(path)
    try:
        assert reader.get_meta_data()['fps'] == 20
        decoded = np.stack(list(reader.iter_data()))
    finally:
        reader.close()
    assert decoded.shape == frames.shape
    np.testing.assert_allclose(decoded.astype(float), frames.astype(float), atol=3)


def test_cache_requires_raw_recordings_and_valid_episode_metadata(tmp_path):
    source = Path(__file__).parents[2] / 'skillVLA/libero/configs/train_skillVLA/PT/eval/src/run_eval.py'
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_panel_artifacts_complete')
    namespace = {'Path': Path, 'json': json}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    complete = namespace['_panel_artifacts_complete']
    cfg = SimpleNamespace(eval=SimpleNamespace(n_episodes=1, max_videos_per_task=1,
                          skill_html=False, raw_videos=True, record_skill_transitions=True))
    folder = tmp_path / 'videos/libero_10_0'
    (folder / 'raw').mkdir(parents=True)
    (folder / 'episode_metadata').mkdir()
    raw = folder / 'raw/eval_episode_0.mp4'
    raw.write_bytes(b'raw')
    (folder / 'eval_episode_0.mp4').write_bytes(b'annotated')
    metadata = dict(schema_version=1, episode_index=0, episode_steps=6, skills=[{}],
                    raw_video=dict(frame_count=6, overlays=False))
    path = folder / 'episode_metadata/eval_episode_0.json'
    path.write_text(json.dumps(metadata))
    assert complete(tmp_path, {'libero_10_0'}, cfg)
    raw.unlink()
    assert not complete(tmp_path, {'libero_10_0'}, cfg)
    raw.write_bytes(b'raw')
    metadata['raw_video']['frame_count'] = 5
    path.write_text(json.dumps(metadata))
    assert not complete(tmp_path, {'libero_10_0'}, cfg)
