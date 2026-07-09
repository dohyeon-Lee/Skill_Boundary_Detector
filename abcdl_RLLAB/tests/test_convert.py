import shutil

import numpy as np
import pytest

from abcdl.convert.mcap_abcdl import abcdl_format_name, mcap_to_abcdl
from abcdl.format.reader import read_abcdl
from abcdl.mcap.reader import read_mcap

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_abcdl_format_name():
    assert abcdl_format_name() == "abcdl_224"
    assert abcdl_format_name(224) == "abcdl_224"
    assert abcdl_format_name(256) == "abcdl_256"


def test_abcdl_to_mcap_smoke(tmp_path):
    import numpy as np
    from abcdl.episode import CameraStream, Episode, EpisodeMeta
    from abcdl.format.writer import write_abcdl
    from abcdl.convert.mcap_abcdl import abcdl_to_mcap
    from abcdl.mcap.reader import read_mcap

    T, h, w = 5, 16, 16
    rng = np.random.default_rng(3)
    cams, res, codecs = {}, {}, {}
    for name in ("top", "left_wrist"):
        cams[name] = CameraStream(frames=rng.integers(0, 255, (T, h, w, 3), dtype=np.uint8),
                                  timestamps=np.arange(T, dtype=np.int64), width=w, height=h, codec="raw")
        res[name] = (w, h); codecs[name] = "h264"
    meta = EpisodeMeta(task="smoke", fps=30.0, cameras=["top", "left_wrist"],
                       camera_resolutions=res, camera_codecs=codecs,
                       alignment="fixed_clock_30hz_causal", t0_ns=0, tick_ns=33_333_333)
    states = rng.standard_normal((T, 14)); actions = rng.standard_normal((T, 14))
    ep = Episode(states, actions, np.arange(T, dtype=np.int64), cams, meta)

    abcdl_dir = tmp_path / "abcdl_ep"; write_abcdl(ep, str(abcdl_dir))
    out_mcap = str(tmp_path / "out.mcap"); abcdl_to_mcap(str(abcdl_dir), out_mcap)
    back = read_mcap(out_mcap)
    assert back.meta.task == "smoke"
    assert back.states.shape == (T, 14)
    np.testing.assert_allclose(back.states, ep.states, atol=1e-9)
    assert set(back.cameras) == {"top", "left_wrist"}


def test_mcap_to_abcdl_fixed_clock(sample_mcap, tmp_path):
    out = tmp_path / "ep"
    mcap_to_abcdl(str(sample_mcap), str(out))
    ep = read_abcdl(str(out))
    assert ep.meta.alignment == "fixed_clock_30hz_causal"
    assert abs(ep.meta.fps - 30.0) < 1e-6
    # 14-D state preserved; frame count equals state count
    assert ep.states.shape[1] == 14
    for name in ep.meta.cameras:
        assert len(ep.cameras[name].frames) == ep.num_steps
        # default size is 224x224, recorded in metadata and the decoded frames
        assert ep.meta.camera_resolutions[name] == (224, 224)
        assert ep.cameras[name].frames.shape[1:3] == (224, 224)


def test_mcap_to_abcdl_custom_size(sample_mcap, tmp_path):
    out = tmp_path / "ep96"
    mcap_to_abcdl(str(sample_mcap), str(out), size=96)
    ep = read_abcdl(str(out))
    assert ep.states.shape[1] == 14
    for name in ep.meta.cameras:
        assert ep.meta.camera_resolutions[name] == (96, 96)
        assert ep.cameras[name].frames.shape[1:3] == (96, 96)
        assert len(ep.cameras[name].frames) == ep.num_steps
