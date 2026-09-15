from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lerobot.policies.skillVLA.dataset_skillVLA import _FocusUVStore


def _write_focus(path: Path) -> None:
    np.savez_compressed(
        path,
        episode_id=np.asarray([8, 8, 9], dtype=np.int32),
        skill_index=np.asarray([0, 1, 0], dtype=np.int32),
        frame_start=np.asarray([0, 45, 0], dtype=np.int32),
        focus_uv=np.asarray([[-0.5, 0.25], [0.75, -0.25], [0.0, 0.0]], dtype=np.float32),
        focus_uv_pixels=np.asarray([[64, 159], [223, 96], [128, 128]], dtype=np.int32),
        focus_valid=np.asarray([True, True, False]),
        focus_clipped=np.asarray([False, False, True]),
    )


def test_focus_store_selects_jittered_skill_rank(tmp_path: Path) -> None:
    path = tmp_path / "skill_focus_uv.npz"
    _write_focus(path)
    store = _FocusUVStore(str(path))

    uv, pixels, valid, clipped = store.target(8, 1, 45)

    np.testing.assert_allclose(uv, [0.75, -0.25])
    np.testing.assert_array_equal(pixels, [223, 96])
    assert valid is True
    assert clipped is False


def test_focus_store_rejects_wrong_ifs_alignment(tmp_path: Path) -> None:
    path = tmp_path / "skill_focus_uv.npz"
    _write_focus(path)
    store = _FocusUVStore(str(path))

    with pytest.raises(ValueError, match="Focus/IFS mismatch"):
        store.target(8, 1, 46)
