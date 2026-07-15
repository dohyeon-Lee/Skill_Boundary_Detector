import shutil

import numpy as np
import pytest

from abcdl.format.encode import encode_strict_h264, probe_frame_count

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="ffmpeg/ffprobe not installed")


def test_encode_frame_count_matches(tmp_path):
    n, h, w = 45, 16, 16
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, size=(n, h, w, 3), dtype=np.uint8)
    out = str(tmp_path / "v.mp4")
    encode_strict_h264(frames, out)
    assert probe_frame_count(out) == n


def test_encode_rejects_bad_input(tmp_path):
    out = str(tmp_path / "v.mp4")
    with pytest.raises(ValueError):
        encode_strict_h264(np.zeros((5, 8, 8), dtype=np.uint8), out)        # ndim 3
    with pytest.raises(ValueError):
        encode_strict_h264(np.zeros((5, 8, 8, 3), dtype=np.float32), out)   # wrong dtype
