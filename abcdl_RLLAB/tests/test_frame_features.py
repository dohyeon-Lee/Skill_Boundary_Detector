import shutil

import numpy as np
import pytest

from abcdl.rewards import compute_frame_features


def test_rewards_sparse():
    ff = compute_frame_features(5, success=True, mode="sparse", discount=0.9)
    assert list(ff["success"]) == [0, 0, 0, 0, 1]
    assert list(ff["reward"]) == [0, 0, 0, 0, 1]
    # mc_return = discounted return-to-go of a terminal +1
    np.testing.assert_allclose(ff["mc_return"], [0.9**4, 0.9**3, 0.9**2, 0.9, 1.0], rtol=1e-6)


def test_rewards_step_and_fail():
    ff = compute_frame_features(4, success=True, mode="step", discount=1.0)
    assert list(ff["reward"]) == [-1, -1, -1, 0]    # -1/step, 0 at success terminal
    assert list(ff["success"]) == [0, 0, 0, 1]
    np.testing.assert_allclose(ff["mc_return"], [-3, -2, -1, 0])  # undiscounted sum-to-go

    fail = compute_frame_features(4, success=False, mode="step", discount=1.0)
    assert list(fail["reward"]) == [-1, -1, -1, -1]
    assert list(fail["success"]) == [0, 0, 0, 0]


def test_rewards_bad_mode():
    with pytest.raises(ValueError):
        compute_frame_features(3, True, mode="nope")


pytestmark_ff = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytestmark_ff
def test_frame_features_roundtrip(tmp_path, tmp_abcdl_episode):
    """write_abcdl persists frame_features.npz; AbcdlDataset surfaces them per item."""
    import json

    from abcdl.dataset import AbcdlDataset
    from abcdl.format.reader import read_abcdl
    from abcdl.format.writer import write_abcdl

    src = read_abcdl(tmp_abcdl_episode)  # T=6 fixture episode
    T = src.num_steps
    src.frame_features = compute_frame_features(T, success=True, mode="sparse", discount=0.95)
    out = tmp_path / "ds" / "episode_0000"
    write_abcdl(src, str(out))
    # metadata records the keys
    meta = json.load(open(out / "episode_metadata.json"))
    assert set(meta["frame_feature_keys"]) == {"success", "reward", "mc_return"}

    ds = AbcdlDataset(str(tmp_path / "ds"))
    assert "reward" in ds.meta.features and "mc_return" in ds.meta.features
    last = ds[T - 1]   # terminal frame of the (single) success episode
    assert float(last["success"]) == 1.0 and float(last["reward"]) == 1.0
    first = ds[0]
    assert float(first["reward"]) == 0.0
    np.testing.assert_allclose(float(first["mc_return"]), 0.95 ** (T - 1), rtol=1e-5)
