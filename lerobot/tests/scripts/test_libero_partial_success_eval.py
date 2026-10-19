from __future__ import annotations

from copy import deepcopy

import pytest

from lerobot.scripts.lerobot_skillvla_eval import (
    _aggregate_goal_predicate_episodes,
    _summarize_goal_predicate_episode,
)


def _snapshot(pot_1: bool, pot_2: bool, *, pot_2_z: float = 0.0) -> dict:
    return {
        "predicates": [
            {
                "index": 0,
                "name": "On",
                "arguments": ["moka_pot_1", "stove_region"],
                "label": "On(moka_pot_1, stove_region)",
                "value": pot_1,
            },
            {
                "index": 1,
                "name": "On",
                "arguments": ["moka_pot_2", "stove_region"],
                "label": "On(moka_pot_2, stove_region)",
                "value": pot_2,
            },
            {
                "index": 2,
                "name": "Turnon",
                "arguments": ["stove"],
                "label": "Turnon(stove)",
                "value": True,
            },
        ],
        "object_positions": {
            "moka_pot_1": [0.0, 0.0, 0.0],
            "moka_pot_2": [0.0, 0.0, pot_2_z],
        },
    }


def test_task8_partial_success_excludes_initially_true_stove_goal() -> None:
    initial = _snapshot(False, False)
    steps = [
        _snapshot(False, False),
        _snapshot(True, False),
        _snapshot(True, False),
        _snapshot(True, False),
        _snapshot(True, False),
        _snapshot(True, False),
        _snapshot(True, False, pot_2_z=0.04),
        _snapshot(True, True, pot_2_z=0.06),
    ]
    summary = _summarize_goal_predicate_episode(
        initial,
        steps,
        runtime_trace=[
            {"batch_index": 0, "skill_index": 0, "episode_timestep": 0},
            {"batch_index": 0, "skill_index": 1, "episode_timestep": 3},
            {"batch_index": 0, "skill_index": 2, "episode_timestep": 7},
        ],
        gt_trace=[
            {"length": 3},
            {"length": 4},
            {"length": 5},
        ],
    )

    assert summary["target_predicate_count"] == 2
    assert summary["initially_satisfied_predicate_count"] == 1
    assert summary["max_partial_success_pct"] == 100.0
    assert summary["any_target_stable"] is True
    assert summary["all_targets_simultaneous_ever"] is True
    assert summary["all_targets_simultaneous_stable"] is False
    assert summary["second_object_lift"]["object"] == "moka_pot_2"
    assert summary["second_object_lift"]["lifted_after_first_stable_goal"] is True
    assert summary["runtime_gt_transition_mae_steps"] == 0.0

    aggregate = _aggregate_goal_predicate_episodes(
        [summary, deepcopy(summary)]
    )
    assert aggregate["any_target_stable_pct"] == 100.0
    assert aggregate["all_targets_simultaneous_ever_pct"] == 100.0
    assert aggregate["second_object_lift_after_first_target_pct"] == 100.0


def _book_snapshot(*, contact=False, z=0.0, placed=False):
    return {
        "predicates": [{"index": 0, "name": "In", "arguments": ["book", "caddy_back"],
                        "label": "In(book, caddy_back)", "value": placed}],
        "object_positions": {"book": [0, 0, z]}, "object_grasps": {"book": contact},
    }


def test_book_grasp_requires_both_contact_and_lift_then_counts_actual_insertion():
    steps = [_book_snapshot(contact=True, z=.04) for _ in range(5)]
    steps.append(_book_snapshot(z=.02, placed=True))
    summary = _summarize_goal_predicate_episode(_book_snapshot(), steps)
    m = summary["single_object_manipulation"]
    assert m["supported"] and m["grasp_success"]
    assert m["first_grasp_lift_step"] == 4
    assert m["placement_ever"] and m["placement_final"] and m["placement_after_grasp"]
    assert m["released_at_final_placement"]
    # Official success ends immediately: a one-frame BDDL placement still counts.
    assert not summary["all_targets_simultaneous_stable"]
    aggregate = _aggregate_goal_predicate_episodes([summary])["single_object_manipulation"]
    assert aggregate["grasp_success_pct"] == 100
    assert aggregate["placement_given_grasp_pct"] == 100


@pytest.mark.parametrize("contact,z,count", [(False, .05, 8), (True, .01, 8), (True, .05, 4)])
def test_contact_alone_lift_alone_and_brief_contact_are_not_successful_grasps(contact, z, count):
    summary = _summarize_goal_predicate_episode(
        _book_snapshot(), [_book_snapshot(contact=contact, z=z) for _ in range(count)],
    )
    assert summary["single_object_manipulation"]["supported"]
    assert not summary["single_object_manipulation"]["grasp_success"]


def test_conditional_placement_uses_only_grasp_successes_and_unknown_is_not_a_failure():
    initial = _book_snapshot()
    grasp = [_book_snapshot(contact=True, z=.04) for _ in range(5)]
    success = _summarize_goal_predicate_episode(initial, grasp + [_book_snapshot(placed=True)])
    grasp_failed_placement = _summarize_goal_predicate_episode(initial, grasp)
    no_grasp = _summarize_goal_predicate_episode(initial, [_book_snapshot()])
    unsupported = _summarize_goal_predicate_episode(initial, [_book_snapshot(contact=None)])
    assert not unsupported["single_object_manipulation"]["supported"]
    aggregate = _aggregate_goal_predicate_episodes(
        [success, grasp_failed_placement, no_grasp, unsupported]
    )["single_object_manipulation"]
    assert aggregate["n_episodes"] == 3
    assert aggregate["n_grasp_success"] == 2
    assert aggregate["grasp_success_pct"] == pytest.approx(200 / 3)
    assert aggregate["placement_ever_pct"] == pytest.approx(100 / 3)
    assert aggregate["placement_given_grasp_pct"] == 50
    zero = _aggregate_goal_predicate_episodes([no_grasp])["single_object_manipulation"]
    assert zero["placement_given_grasp_pct"] is None


def _two_object_snapshot(*, a_contact=False, b_contact=False, a_z=0.0, b_z=0.0,
                         a_placed=False, b_placed=False):
    return {
        "predicates": [
            {"index": i, "name": "In", "arguments": [name, "basket"],
             "label": f"In({name}, basket)", "value": placed}
            for i, (name, placed) in enumerate((("a", a_placed), ("b", b_placed)))
        ],
        "object_positions": {"a": [0, 0, a_z], "b": [0, 0, b_z], "basket": [0, 0, 0]},
        "object_grasps": {"a": a_contact, "b": b_contact, "basket": False},
    }


def _two_object_episode(*, grasp_b=True, placed_b=True, contact_supported=True):
    steps = [_two_object_snapshot(a_contact=True, a_z=.05) for _ in range(5)]
    steps.extend(_two_object_snapshot(a_placed=True, b_contact=grasp_b, b_z=.05) for _ in range(5))
    steps.append(_two_object_snapshot(a_placed=True, b_placed=placed_b))
    if not contact_supported:
        for step in steps:
            step["object_grasps"]["b"] = None
    summary = _summarize_goal_predicate_episode(_two_object_snapshot(), steps)
    summary["success"] = bool(placed_b)
    return summary


def test_two_object_grasps_and_placements_are_separate_and_ignore_movable_receptacle():
    summary = _two_object_episode()
    assert not summary["single_object_manipulation"]["supported"]
    objects = summary["object_manipulations"]
    assert set(objects) == {"a", "b"}  # The basket is not a grasp target.
    assert objects["a"]["first_grasp_lift_step"] == 4
    assert objects["b"]["first_grasp_lift_step"] == 9
    assert objects["a"]["goal_predicates"] == ["In(a, basket)"]
    assert all(item["grasp_success"] and item["placement_after_grasp"] for item in objects.values())


def test_two_object_aggregation_distinguishes_missed_grasp_from_failed_placement():
    aggregate = _aggregate_goal_predicate_episodes([
        _two_object_episode(),
        _two_object_episode(placed_b=False),
        _two_object_episode(grasp_b=False, placed_b=False),
        _two_object_episode(contact_supported=False),
    ])
    a, b = (aggregate["object_manipulations"][name] for name in ("a", "b"))
    assert a["grasp_success_pct"] == 100 and a["placement_given_grasp_pct"] == 100
    assert b["n_episodes"] == 3 and b["n_unsupported_episodes"] == 1
    assert b["grasp_success_pct"] == pytest.approx(200 / 3)
    assert b["placement_given_grasp_pct"] == 50
    multi = aggregate["multi_object_manipulation"]
    assert multi["n_episodes"] == 3 and multi["n_all_grasp_success"] == 2
    assert multi["all_objects_grasp_success_pct"] == pytest.approx(200 / 3)
    assert multi["n_task_outcomes_given_all_grasps"] == 2
    assert multi["n_task_success_and_all_grasps"] == 1
    assert multi["task_success_given_all_grasps_pct"] == 50


def test_multi_object_no_grasps_and_missing_official_outcome_are_not_zero_conditionals():
    no_grasps = _summarize_goal_predicate_episode(
        _two_object_snapshot(), [_two_object_snapshot() for _ in range(5)]
    )
    multi = _aggregate_goal_predicate_episodes([no_grasps])["multi_object_manipulation"]
    assert multi["all_objects_grasp_success_pct"] == 0
    assert multi["task_success_given_all_grasps_pct"] is None
    success = _two_object_episode()
    del success["success"]
    multi = _aggregate_goal_predicate_episodes([success])["multi_object_manipulation"]
    assert multi["all_objects_grasp_success_pct"] == 100
    assert multi["task_success_given_all_grasps_pct"] is None
