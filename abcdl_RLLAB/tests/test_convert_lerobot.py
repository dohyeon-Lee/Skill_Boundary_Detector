"""Round-trip test: abcdl -> LeRobotDataset -> abcdl."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

lerobot = pytest.importorskip("lerobot")
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_abcdl_to_lerobot_roundtrip(tmp_path, tmp_abcdl_episode):
    from abcdl.convert.lerobot import abcdl_to_lerobot, lerobot_to_abcdl
    from abcdl.format.reader import read_abcdl

    root = tmp_path / "ds"; root.mkdir()
    shutil.copytree(tmp_abcdl_episode, root / "episode_0000")

    lr_root = abcdl_to_lerobot(str(root), "test/yam", lerobot_root=str(tmp_path / "lr"))
    back_dirs = lerobot_to_abcdl(lr_root, str(tmp_path / "back"))
    assert len(back_dirs) == 1
    src = read_abcdl(str(root / "episode_0000"))
    back = read_abcdl(back_dirs[0])
    assert back.num_steps == src.num_steps
    np.testing.assert_allclose(back.states, src.states, atol=1e-4)  # parquet float32
