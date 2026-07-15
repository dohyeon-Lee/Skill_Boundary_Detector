import shutil

import numpy as np
import pytest

from abcdl.episode import CameraStream, Episode, EpisodeMeta
from abcdl.format.reader import read_abcdl
from abcdl.format.writer import write_abcdl

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _episode(T=33, h=16, w=16):
    rng = np.random.default_rng(1)
    cams = {}
    res = {}
    codecs = {}
    for name in ("top", "left_wrist"):
        cams[name] = CameraStream(
            frames=rng.integers(0, 255, (T, h, w, 3), dtype=np.uint8),
            timestamps=np.arange(T, dtype=np.int64), width=w, height=h, codec="raw")
        res[name] = (w, h)
        codecs[name] = "h264"
    meta = EpisodeMeta(task="demo", fps=30.0, cameras=["top", "left_wrist"],
                       camera_resolutions=res, camera_codecs=codecs,
                       alignment="fixed_clock_30hz_causal", t0_ns=0, tick_ns=33_333_333)
    states = rng.standard_normal((T, 14))
    actions = rng.standard_normal((T, 14))
    return Episode(states, actions, np.arange(T, dtype=np.int64), cams, meta)


def test_roundtrip_states_exact_and_frames_aligned(tmp_path):
    ep = _episode()
    write_abcdl(ep, str(tmp_path))
    back = read_abcdl(str(tmp_path))
    assert back.num_steps == ep.num_steps
    np.testing.assert_array_equal(back.states, ep.states)        # bin is lossless
    np.testing.assert_array_equal(back.actions, ep.actions)
    assert set(back.cameras) == {"top", "left_wrist"}
    # frame count preserved per camera
    for name in ep.meta.cameras:
        assert len(back.cameras[name].frames) == ep.num_steps
        fr = back.cameras[name].frames
        assert fr.dtype == np.uint8
        assert fr.shape == (ep.num_steps, 16, 16, 3)
