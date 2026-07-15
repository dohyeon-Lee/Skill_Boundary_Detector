import numpy as np

from abcdl.mcap.reader import read_mcap
from abcdl.mcap.writer import write_mcap


def test_mcap_state_roundtrip(sample_mcap, tmp_path):
    ep = read_mcap(str(sample_mcap))
    out = str(tmp_path / "episode.mcap")
    write_mcap(ep, out)
    back = read_mcap(out)
    assert back.meta.task == ep.meta.task
    assert back.states.shape == ep.states.shape
    np.testing.assert_allclose(back.states, ep.states, rtol=0, atol=1e-9)
    assert set(back.cameras) == set(ep.cameras)
