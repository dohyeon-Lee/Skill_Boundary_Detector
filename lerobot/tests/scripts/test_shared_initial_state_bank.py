import json

import numpy as np
import pytest

from lerobot.utils.initial_state_bank import inspect_bank, load_bank, require_predictor_only


def make_bank(tmp_path):
    path = tmp_path / "bank.npz"
    poses = {"table_main": {"pos": [1, 2, 3], "quat": [1, 0, 0, 0]}}
    np.savez(path, metadata=json.dumps({"version": 1, "suite": "libero_10", "task_ids": [0],
                                      "episodes": 2, "random_range": 2., "seed": 1000}),
             episode_index=[1, 0], scene_file=["task_a_demo.hdf5"] * 2,
             init_states=[[7., 8.], [3., 4.]], model_body_poses=[json.dumps(poses)] * 2)
    return path, poses


def test_shared_bank_reads_sorted_states_and_fixture_poses(tmp_path):
    path, poses = make_bank(tmp_path)
    identity = inspect_bank(path, suite="libero_10", task_ids=[0], episodes=2)
    rows = load_bank(identity)["task_a"]
    np.testing.assert_array_equal(rows[0]["state"], [3., 4.])
    assert rows[0]["model_body_poses"] == poses
    assert identity["random_range"] == 2.


@pytest.mark.parametrize("suite,tasks,episodes", [("libero_90", [0], 2), ("libero_10", [1], 2), ("libero_10", [0], 3)])
def test_wrong_scene_coverage_is_rejected(tmp_path, suite, tasks, episodes):
    path, _ = make_bank(tmp_path)
    with pytest.raises(ValueError, match="initial_state_bank"):
        inspect_bank(path, suite=suite, task_ids=tasks, episodes=episodes)


def test_bank_changed_after_resolution_is_rejected(tmp_path):
    path, _ = make_bank(tmp_path)
    identity = inspect_bank(path, suite="libero_10", task_ids=[0], episodes=2)
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        load_bank(identity)


def predictor_spec():
    return {"label": "uv_vsa", "skill_source": "external", "advance_mode": "external",
            "needs_focus_uv": True, "focus_source": "predictor", "pose_source": "predictor"}


def test_uv_predictor_without_xyz_requirement_is_allowed():
    require_predictor_only([predictor_spec()])
    require_predictor_only([{**predictor_spec(), "needs_end_pose": True}])


@pytest.mark.parametrize("override", [
    {"skill_source": "gt"}, {"advance_mode": "gt"}, {"focus_source": "gt"},
    {"needs_end_pose": True, "pose_source": "gt"},
    {"terminator_needs_goal_xyz": True, "pose_source": "gt"},
    {"latent_source": "oracle"}, {"predictor_diagnostics": True}, {"task0_intervention": {"mode": "baseline"}},
])
def test_gt_oracle_dependencies_cannot_leak_into_shared_random_scenes(override):
    with pytest.raises(ValueError, match="predictor-only"):
        require_predictor_only([{**predictor_spec(), **override}])
