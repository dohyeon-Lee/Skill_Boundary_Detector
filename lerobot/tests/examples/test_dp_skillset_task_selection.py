from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


LIBERO_EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "libero"
sys.path.insert(0, str(LIBERO_EXAMPLES))

from dp_skillset_eval import select_stage1_suite_episodes  # noqa: E402


def test_suite_selection_uses_exact_scene_instead_of_shared_language(tmp_path: Path):
    dataset_dir = tmp_path / "libero_90_full_full"
    exact_map = (
        tmp_path
        / "skillvla_dataset"
        / dataset_dir.name
        / "eval_init_states.npz"
    )
    exact_map.parent.mkdir(parents=True)
    np.savez(
        exact_map,
        episode_index=np.asarray([10, 11, 20, 21], dtype=np.int32),
        scene_file=np.asarray(
            [
                "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo.hdf5",
                "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo.hdf5",
                "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_demo.hdf5",
                "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_demo.hdf5",
            ]
        ),
    )
    # These episodes share a language-derived dataset task ID. Suite task zero
    # must still select only the exact KITCHEN_SCENE10 demonstrations.
    ep_task = {10: 70, 11: 70, 20: 70, 21: 70}

    assert select_stage1_suite_episodes(
        ep_task,
        [0],
        suite_name="libero_90",
        dataset_dir=dataset_dir,
        n_episodes=10,
    ) == [(0, [20, 21])]
