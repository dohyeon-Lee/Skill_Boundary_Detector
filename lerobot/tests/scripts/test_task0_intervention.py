"""No GPU/policy/simulator required for the intervention state-machine tests."""

import copy
import json

import numpy as np
import pytest

from lerobot.utils.task0_intervention import (
    MODES, PROTOCOL, Task0Intervention, load_reference, reference_identity, summarize_interventions,
)


def reference():
    return {"protocol": PROTOCOL, "task_id": 0, "skill_code_space_id": "space", "container_object": "basket_1",
            "first_goal": ["in", "alphabet_soup_1", "basket_1"],
            "second_goal": ["in", "tomato_sauce_1", "basket_1"],
            "phases": [{"skill_index": i, "skill_code": code, "anchor": anchor,
                        "eef_end_local_xyz": [.1, .2, 0.]}
                       for i, code, anchor in [(3, 9, "tomato_sauce_1"), (4, 23, "basket_1")]]}


def scene(*, first=False, second=False, grasp=False):
    return {"task_id": 0, "predicates": [
        {"state": reference()["first_goal"], "value": first},
        {"state": reference()["second_goal"], "value": second}],
        "grasps": {"alphabet_soup_1": grasp},
        "objects": {name: {"position": [0., 0., 1.], "rotation": np.eye(3).tolist()}
                    for name in ["tomato_sauce_1", "basket_1"]},
        "camera_transform": np.eye(4).tolist(), "width": 101, "height": 51}


def test_reference_validation_and_signature(tmp_path):
    path = tmp_path / "ref.json"
    path.write_text(json.dumps(reference()))
    assert load_reference(path, code_space="space", vocabulary=27) == reference()
    before = reference_identity(path)
    with pytest.raises(ValueError, match="code-space"):
        load_reference(path, code_space="wrong")
    with pytest.raises(ValueError, match="FSQ code"):
        load_reference(path, vocabulary=3)
    changed = reference()
    changed["phases"][0]["eef_end_local_xyz"][0] = .3
    path.write_text(json.dumps(changed))
    assert reference_identity(path) != before


def test_trigger_requires_stable_goal_and_released_object_no_fixed_timer():
    engine = Task0Intervention(reference(), mode="skill_uv", stable_steps=3)
    engine.observe([scene(first=True, grasp=True)], 199)
    assert engine.propose(4, [.5, .5], 199) is None
    for step, complete in [(200, True), (201, False), (202, True), (203, True)]:
        engine.observe([scene(first=complete)], step)
        assert engine.propose(4, [.5, .5], step) is None
    engine.observe([scene(first=True)], 204)
    assert engine.trigger_step == 204
    assert engine.events == []  # Observation never forces a learned boundary.


@pytest.mark.parametrize("mode", sorted(MODES))
def test_factorial_conditions_and_retargeted_eef_not_object_center(mode):
    engine = Task0Intervention(reference(), mode=mode, stable_steps=1)
    snapshot = scene(first=True)
    snapshot["objects"]["tomato_sauce_1"]["rotation"] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    snapshot["objects"]["tomato_sauce_1"]["position"] = [.4, .5, 1.]
    unchanged = copy.deepcopy(snapshot)
    engine.observe([snapshot], 10)
    proposal = engine.propose(4, [.5, .25], 15)
    np.testing.assert_allclose(proposal["reference_eef_world"], [.2, .6, 1.])
    assert proposal["reference_eef_world"] != snapshot["objects"]["tomato_sauce_1"]["position"]
    assert proposal["applied_skill"] == (9 if mode in {"skill_only", "skill_uv"} else 4)
    assert proposal["applied_uv"] == (proposal["reference_uv"] if mode in {"uv_only", "skill_uv"} else [.5, .25])
    assert snapshot == unchanged
    engine.commit(proposal)
    assert engine.phase_index == 0 and engine.report()["first_boundary_delay"] == 5
    assert engine.report()["intervened"] == (mode != "baseline")


def test_no_phase_advance_on_rejection_and_last_phase_is_held():
    engine = Task0Intervention(reference(), mode="skill_uv", stable_steps=1)
    engine.observe([scene(first=True)], 10)
    first = engine.propose(4, [.5, .5], 10)
    engine.reject(first)
    assert engine.phase_index == -1 and engine.report()["accepted_boundaries"] == 0
    assert engine.propose(4, [.5, .5], 11)["phase_index"] == 0
    engine.commit(first)
    engine.commit(engine.propose(5, [.5, .5], 20))
    assert engine.propose(5, [.5, .5], 30)["phase_index"] == 1
    engine.observe([scene(first=True, second=True)], 31)
    assert engine.propose(5, [.5, .5], 31) is None
    engine.reset()
    assert engine.trigger_step is None and engine.events == [] and engine.rejections == []


@pytest.mark.parametrize("bad", ["batch", "task", "predicate", "contact"])
def test_missing_privileged_inputs_fail_closed(bad):
    engine = Task0Intervention(reference(), mode="uv_only")
    snapshot = scene()
    snapshots = [snapshot]
    if bad == "batch": snapshots *= 2
    if bad == "task": snapshot["task_id"] = 1
    if bad == "predicate": snapshot["predicates"] = []
    if bad == "contact": snapshot["grasps"] = {}
    with pytest.raises(ValueError):
        engine.observe(snapshots, 0)


def test_summary_reports_coverage_not_just_successful_triggered_subset():
    episodes = [{"triggered": False, "intervened": False, "success": False},
                {"triggered": True, "intervened": True, "success": True}]
    assert summarize_interventions(episodes) == {
        "episodes": 2, "triggered": 1, "intervened": 1,
        "triggered_success_pct": 100., "intervened_success_pct": 100.,
        "interpretation": "Report full-task SR AND intervention coverage; untriggered episodes are not second-can trials."}
