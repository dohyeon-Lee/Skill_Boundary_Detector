import shutil

import numpy as np
import pytest

from abcdl.writer import EpisodeWriter
from abcdl.format.reader import read_abcdl

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_episode_writer_abcdl(tmp_path):
    w = EpisodeWriter(str(tmp_path / "ep"), formats=("abcdl",), fps=30,
                      cameras=["top", "left_wrist"])
    rng = np.random.default_rng(0)
    T = 5
    for i in range(T):
        w.add_frame(i * 33_333_333, rng.standard_normal(14), rng.standard_normal(14),
                    {"top": rng.integers(0, 255, (16, 16, 3), np.uint8),
                     "left_wrist": rng.integers(0, 255, (16, 16, 3), np.uint8)})
    paths = w.save(task="t")
    ep = read_abcdl(paths["abcdl"])
    assert ep.num_steps == T
    assert ep.states.shape == (T, 14)
    assert set(ep.cameras) == {"top", "left_wrist"}


def test_episode_writer_mcap_encodes_decoded_frames(tmp_path):
    from abcdl.mcap.reader import read_mcap
    w = EpisodeWriter(str(tmp_path / "ep2"), formats=("mcap",), fps=30, cameras=["top"])
    rng = np.random.default_rng(1)
    for i in range(5):
        w.add_frame(i * 33_333_333, rng.standard_normal(14), rng.standard_normal(14),
                    {"top": rng.integers(0, 255, (16, 16, 3), np.uint8)})
    paths = w.save(task="t2")
    back = read_mcap(paths["mcap"])
    assert back.meta.task == "t2"
    assert back.states.shape == (5, 14)
    assert "top" in back.cameras
    assert back.cameras["top"].codec == "h264"
