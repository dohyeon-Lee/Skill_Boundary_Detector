from __future__ import annotations

from copy import deepcopy

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
