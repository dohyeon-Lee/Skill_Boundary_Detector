"""Boundary evaluation keeps scene-exact task selection after the legacy module's removal."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "skillVLA/libero/configs/train_skills/DP/eval/src/evaluate.py"
SPEC = importlib.util.spec_from_file_location("boundary_episode_selection", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def skillset(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)
    dataset = tmp_path / "dataset_filtered/libero_90_full_full"
    dataset.mkdir(parents=True)
    root = tmp_path / "skillset"
    (root / "skills").mkdir(parents=True)
    # Both scenes have the same dataset task ID/language, but suite task 0
    # refers only to KITCHEN_SCENE10. Filename indexing requires no array I/O.
    for episode, task in [(10, 70), (11, 70), (20, 70), (21, 70), (30, 80)]:
        (root / "skills" / f"ep{episode:07d}_task{task:03d}_skill000.npz").touch()
    (root / "skillset_manifest.json").write_text(json.dumps({
        "dataset_dir": "/old-server/checkout/dataset_filtered/libero_90_full_full"}))
    bank = dataset.parent / "skillvla_dataset/libero_90_full_full/eval_init_states.npz"
    bank.parent.mkdir(parents=True)
    np.savez(bank, episode_index=np.array([10, 11, 20, 21]), scene_file=np.array([
        "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo.hdf5",
        "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo.hdf5",
        "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_demo.hdf5",
        "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_demo.hdf5",
    ]))
    return root, bank


def test_suite_ids_relocate_manifest_and_select_only_exact_scene(skillset):
    root, _ = skillset
    assert MODULE._selected_episode_ids(
        root, task_ids=[0], task_id_space="suite", target_task="libero_90", n_episodes=1,
    ) == [20]
    assert MODULE._selected_task_episode_groups({
        "task_ids": [0], "task_id_space": "suite", "target_task": "libero_90", "n_episodes": 2,
    }, root, [21]) == {0: [21]}


def test_dataset_ids_keep_requested_task_order_and_per_task_limit(skillset):
    root, _ = skillset
    assert MODULE._selected_episode_ids(
        root, task_ids=[80, 70], task_id_space="dataset", target_task="", n_episodes=2,
    ) == [30, 10, 11]


def test_missing_exact_map_fails_instead_of_selecting_another_scene(skillset):
    root, bank = skillset
    bank.unlink()
    with pytest.raises(FileNotFoundError, match="episode-exact map"):
        MODULE._selected_episode_ids(
            root, task_ids=[0], task_id_space="suite", target_task="libero_90", n_episodes=20,
        )


@pytest.mark.parametrize("space,count", [("unknown", 20), ("dataset", 0)])
def test_invalid_selection_fails_clearly(skillset, space, count):
    root, _ = skillset
    with pytest.raises(ValueError):
        MODULE._selected_episode_ids(root, task_ids=[0], task_id_space=space, target_task="libero_90", n_episodes=count)


def test_coffee_partial_build_selects_manifest_episodes_without_existing_skills(tmp_path):
    # A fresh Coffee run has no segmented skills to index yet. The same selected
    # episodes must reach both threshold calibration and report task grouping.
    (tmp_path / "skillset_manifest.json").write_text(json.dumps({
        "evaluation_source": "mimicgen_subset",
        "evaluation_episodes": [
            {"episode_index": ep, "task_index": 0, "selection_task_id": 0, "length": 100}
            for ep in range(5)
        ],
    }))
    settings = {"task_ids": [0], "task_id_space": "dataset", "target_task": "Coffee_D0", "n_episodes": 5,
                "threshold_calibration": {"source": "selected", "expected_episodes": 5}}
    selected = MODULE._selected_episode_ids(
        tmp_path, **{key: settings[key] for key in ["task_ids", "task_id_space", "target_task", "n_episodes"]})
    assert selected == list(range(5))
    assert MODULE._threshold_calibration_episode_ids(settings, tmp_path, selected) == selected
    assert MODULE._selected_task_episode_groups(settings, tmp_path, selected) == {0: selected}
