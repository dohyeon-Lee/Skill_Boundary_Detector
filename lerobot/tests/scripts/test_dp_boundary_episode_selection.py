"""Boundary evaluation keeps scene-exact task selection after the legacy module's removal."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "examples/libero/configs/train_skills/skill_eval/skill_boundary_detection/src/evaluate.py"
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
