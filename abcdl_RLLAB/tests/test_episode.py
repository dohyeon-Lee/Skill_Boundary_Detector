import numpy as np
import pytest

from abcdl.episode import CameraStream, Episode, EpisodeMeta, StateLayout


def test_state_layout_dims():
    assert StateLayout.YAM.state_dim == 14
    assert StateLayout.YAM.action_dim == 14


def _toy_episode(T=5):
    cam = CameraStream(frames=np.zeros((T, 4, 4, 3), np.uint8),
                       timestamps=np.arange(T, dtype=np.int64), width=4, height=4, codec="raw")
    meta = EpisodeMeta(task="t", fps=30.0, cameras=["top"],
                       camera_resolutions={"top": (4, 4)}, camera_codecs={"top": "raw"})
    return Episode(states=np.zeros((T, 14)), actions=np.zeros((T, 14)),
                   timestamps=np.arange(T, dtype=np.int64), cameras={"top": cam}, meta=meta)


def test_episode_num_steps_and_validate():
    ep = _toy_episode(7)
    assert ep.num_steps == 7
    ep.validate()  # no raise


def test_episode_validate_rejects_length_mismatch():
    ep = _toy_episode(5)
    ep.actions = np.zeros((4, 14))
    with pytest.raises(ValueError):
        ep.validate()


def test_episode_validate_rejects_missing_camera():
    ep = _toy_episode(5)
    ep.meta.cameras.append("missing_camera")
    with pytest.raises(ValueError):
        ep.validate()
