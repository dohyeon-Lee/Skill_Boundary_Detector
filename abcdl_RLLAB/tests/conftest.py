import os
from pathlib import Path

import numpy as np
import pytest

_DEFAULT_SAMPLE = Path(
    "~/.cache/huggingface/hub/datasets--XDOF--ABC-130k/snapshots/"
    "071311db1ac281848714bff024f9c6f944837c40/data/val/open_the_umbrella/"
    "episode_ebcbf9d1-d42b-4ef1-b3be-cad29b11edf8/episode.mcap"
).expanduser()


@pytest.fixture
def sample_mcap() -> Path:
    path = Path(os.environ.get("ABCDL_SAMPLE_MCAP", _DEFAULT_SAMPLE)).expanduser()
    if not path.exists():
        pytest.skip(f"sample mcap not found at {path}")
    return path


@pytest.fixture
def tmp_abcdl_episode(tmp_path):
    from abcdl.episode import CameraStream, Episode, EpisodeMeta
    from abcdl.format.writer import write_abcdl

    T, h, w = 6, 16, 16
    rng = np.random.default_rng(7)
    cams, res, codecs = {}, {}, {}
    for name in ("top", "left_wrist"):
        cams[name] = CameraStream(frames=rng.integers(0, 255, (T, h, w, 3), dtype=np.uint8),
                                  timestamps=np.arange(T, dtype=np.int64), width=w, height=h, codec="raw")
        res[name] = (w, h); codecs[name] = "h264"
    meta = EpisodeMeta(task="demo task", fps=30.0, cameras=["top", "left_wrist"],
                       camera_resolutions=res, camera_codecs=codecs,
                       alignment="fixed_clock_30hz_causal", t0_ns=0, tick_ns=33_333_333)
    ep = Episode(rng.standard_normal((T, 14)), rng.standard_normal((T, 14)),
                 np.arange(T, dtype=np.int64), cams, meta)
    out = tmp_path / "abcdl_ep"
    write_abcdl(ep, str(out))
    return str(out)
