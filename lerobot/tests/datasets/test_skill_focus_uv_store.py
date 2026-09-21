from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.policies.skillVLA.dataset_skillVLA import _FocusUVStore


def _write_focus(path: Path) -> None:
    np.savez_compressed(
        path,
        episode_id=np.asarray([8, 8, 9], dtype=np.int32),
        skill_index=np.asarray([0, 1, 0], dtype=np.int32),
        frame_start=np.asarray([0, 45, 0], dtype=np.int32),
        frame_end=np.asarray([45, 80, 4], dtype=np.int32),
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


def test_end_xyz_cache_reads_grounded_state_at_canonical_endpoint(tmp_path: Path) -> None:
    path = tmp_path / "skill_focus_uv.npz"
    _write_focus(path)
    store = _FocusUVStore(str(path))
    states = np.zeros((84, 8), dtype=np.float32)
    states[45, :3] = [0.1, 0.2, 0.3]
    states[79, :3] = [0.4, 0.5, 0.6]
    states[83, :3] = [-0.1, -0.2, -0.3]

    class _States:
        def select_columns(self, columns):
            assert columns == ["observation.state"]
            return self

        def with_format(self, format_name):
            assert format_name == "numpy"
            return self

        def __getitem__(self, index):
            assert isinstance(index, slice)
            return {"observation.state": states}

    dataset = SimpleNamespace(
        hf_dataset=_States(),
        episodes=[8, 9],
        reader=SimpleNamespace(_absolute_to_relative_idx=None),
        meta=SimpleNamespace(episodes={
            8: {"length": 80, "dataset_from_index": 0},
            9: {"length": 4, "dataset_from_index": 80},
        }),
    )
    store.cache_end_xyz(dataset)
    np.testing.assert_allclose(store.target_xyz(8, 0, 0)[0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(store.target_xyz(8, 1, 45)[0], [0.4, 0.5, 0.6])
    np.testing.assert_allclose(store.target_state(8, 1, 45)[0][3:], np.zeros(5))
    np.testing.assert_allclose(store.target_xyz(9, 0, 0)[0], [-0.1, -0.2, -0.3])
    assert store.target_xyz(8, 1, 45)[1] is True


def test_end_xyz_cache_handles_subset_episode_row_mapping(tmp_path: Path) -> None:
    path = tmp_path / "skill_focus_uv.npz"
    _write_focus(path)
    store = _FocusUVStore(str(path))
    states = np.zeros((4, 8), dtype=np.float32)
    states[3, :3] = [-0.1, -0.2, -0.3]

    class _States:
        def select_columns(self, columns):
            assert columns == ["observation.state"]
            return self

        def with_format(self, format_name):
            assert format_name == "numpy"
            return self

        def __getitem__(self, index):
            assert isinstance(index, slice)
            return {"observation.state": states}

    dataset = SimpleNamespace(
        hf_dataset=_States(),
        episodes=[9],
        reader=SimpleNamespace(_absolute_to_relative_idx={80 + index: index for index in range(4)}),
        meta=SimpleNamespace(episodes={9: {"length": 4, "dataset_from_index": 80}}),
    )
    store.cache_end_xyz(dataset)
    np.testing.assert_allclose(store.target_xyz(9, 0, 0)[0], [-0.1, -0.2, -0.3])
