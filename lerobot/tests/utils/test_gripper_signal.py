import numpy as np
import pytest

from lerobot.utils.gripper_signal import gripper_player_html, gripper_signal


def test_continuous_gripper_and_full_future_survive_early_replanning():
    actions = np.zeros((8, 7))
    actions[:, 6] = [-1, -.1, 0, .1, .49, .9, 1.2, -.2]
    signal = gripper_signal(
        actions, sim_fps=20, video_fps=10, frame_stride=4,
        chunks=[{'start_step': 0, 'values': [-1, -.1, .2, .3, .4, .5]},
                {'start_step': 3, 'values': [.1, .49, .9, 1.2, -.2, .8]}],
    )
    assert signal['values'] == actions[:, 6].tolist()  # No sign, threshold, or clipping.
    assert signal['threshold'] == 0
    assert signal['chunks'][0]['executed_steps'] == 3
    assert signal['chunks'][0]['values'][-1] == .5  # Replaced future remains visible.
    assert signal['chunks'][0]['executed_values'] == [-1, -.1, 0]
    assert signal['chunks'][1]['executed_steps'] == 5
    assert len(signal['chunks'][1]['values']) == 6  # Future past episode end remains visible.
    assert signal['video_frames_per_second'] * signal['sim_steps_per_video_frame'] == 40


@pytest.mark.parametrize('starts', [[3, 2], [0, 0], [-1, 3], [0, 8]])
def test_rejects_ambiguous_chunk_frame_alignment(starts):
    with pytest.raises(ValueError):
        gripper_signal(np.zeros((8, 7)), sim_fps=20, video_fps=10, frame_stride=4,
                       chunks=[{'start_step': s, 'values': [0, 1]} for s in starts])


def test_video_source_is_escaped_and_chunk_plot_is_available():
    markup = gripper_player_html('video"name.mp4')
    assert 'video&quot;name.mp4' in markup
    assert 'gripper-chunk-chart' in markup and 'gripper-follow' in markup
