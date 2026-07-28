from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from build_training_dataset import (  # noqa: E402
    load_libero_scene_map,
    make_output_name,
    select_episodes,
    write_subset_libero_scene_map,
)


def _episode_frame() -> pd.DataFrame:
    # Legacy language task 0 contains two distinct original LIBERO scene-tasks.
    return pd.DataFrame(
        {
            "episode_index": list(range(9)),
            "task_index": [0] * 9,
        }
    )


def test_auto_output_name_marks_only_scene_aware_halves():
    assert make_output_name(
        "libero_90_full_full",
        "full",
        "firsthalf",
        scene_aware_halves=True,
    ) == "libero_90_full_firsthalf_scene"
    assert make_output_name(
        "libero_90_full_full",
        "full",
        "firsthalf",
        scene_aware_halves=False,
    ) == "libero_90_full_firsthalf"
    assert make_output_name(
        "langgap_56_full_full",
        "40to55",
        "firsthalf",
        scene_aware_halves=False,
    ) == "langgap_56_40to55_firsthalf"


def test_scene_aware_halves_split_each_original_libero_task():
    groups = {
        **{episode: "scene_a_demo.hdf5" for episode in range(4)},
        **{episode: "scene_b_demo.hdf5" for episode in range(4, 9)},
    }

    first = select_episodes(
        _episode_frame(),
        [0],
        "firsthalf",
        False,
        half_group_by_episode=groups,
    )
    last = select_episodes(
        _episode_frame(),
        [0],
        "lasthalf",
        False,
        half_group_by_episode=groups,
    )

    assert first == [0, 1, 4, 5]
    assert last == [2, 3, 6, 7, 8]
    assert set(first).isdisjoint(last)
    assert sorted(first + last) == list(range(9))
    assert {groups[episode] for episode in first} == set(groups.values())
    assert {groups[episode] for episode in last} == set(groups.values())


def test_legacy_half_split_still_uses_language_task_index():
    assert select_episodes(_episode_frame(), [0], "firsthalf", False) == [0, 1, 2, 3]
    assert select_episodes(_episode_frame(), [0], "lasthalf", False) == [4, 5, 6, 7, 8]


def test_scene_aware_split_rejects_incomplete_oracle_map():
    with pytest.raises(ValueError, match="does not cover every candidate episode"):
        select_episodes(
            _episode_frame(),
            [0],
            "firsthalf",
            False,
            half_group_by_episode={0: "scene_a_demo.hdf5"},
        )


def test_load_libero_scene_map(tmp_path: Path):
    path = tmp_path / "eval_init_states.npz"
    np.savez(
        path,
        episode_index=np.asarray([7, 9], dtype=np.int32),
        scene_file=np.asarray(["scene_a_demo.hdf5", "scene_b_demo.hdf5"]),
    )

    assert load_libero_scene_map(path) == {
        7: "scene_a_demo.hdf5",
        9: "scene_b_demo.hdf5",
    }


def test_write_subset_libero_scene_map_reindexes_episodes(tmp_path: Path):
    source = tmp_path / "full.npz"
    destination = tmp_path / "subset" / "eval_init_states.npz"
    np.savez(
        source,
        episode_index=np.asarray([7, 9, 11], dtype=np.int32),
        init_states=np.asarray([[70], [90], [110]], dtype=object),
        scene_file=np.asarray(["a", "b", "c"]),
        demo=np.asarray(["demo_7", "demo_9", "demo_11"]),
    )

    write_subset_libero_scene_map(source, destination, [11, 7])

    with np.load(destination, allow_pickle=True) as payload:
        assert payload["episode_index"].tolist() == [0, 1]
        assert payload["source_episode_index"].tolist() == [11, 7]
        assert payload["scene_file"].tolist() == ["c", "a"]
        assert payload["demo"].tolist() == ["demo_11", "demo_7"]
        assert np.asarray(payload["init_states"], dtype=int).reshape(-1).tolist() == [110, 70]
