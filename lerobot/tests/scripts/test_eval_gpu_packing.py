from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/eval_gpu_packing.py"
)
SPEC = importlib.util.spec_from_file_location("eval_gpu_packing", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_ten_tasks_pack_as_four_three_three_on_three_gpus() -> None:
    plan = MODULE.plan_item_chunks(list(range(10)), 3, 4)

    assert plan["physical_gpu_count"] == 3
    assert plan["logical_worker_count"] == 10
    assert [len(group) for group in plan["groups"]] == [4, 3, 3]
    assert [
        task
        for group in plan["groups"]
        for worker in group
        for task in worker["items"]
    ] == list(range(10))


def test_large_workload_respects_four_worker_cap() -> None:
    plan = MODULE.plan_worker_indices(100, 3, 4)

    assert plan == {
        "physical_gpu_count": 3,
        "logical_worker_count": 12,
        "groups": [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
    }


def test_work_shortage_avoids_empty_gpus_and_workers() -> None:
    plan = MODULE.plan_worker_indices(2, 10, 4)

    assert plan["physical_gpu_count"] == 1
    assert plan["logical_worker_count"] == 2
    assert plan["groups"] == [[0, 1]]


@pytest.mark.parametrize("value", [0, 5])
def test_worker_cap_must_be_between_one_and_four(value: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 4"):
        MODULE.plan_worker_indices(10, 2, value)


def test_item_chunks_remain_balanced_when_tasks_outnumber_workers() -> None:
    plan = MODULE.plan_item_chunks(list(range(25)), 2, 4)
    sizes = [
        len(worker["items"])
        for group in plan["groups"]
        for worker in group
    ]

    assert plan["logical_worker_count"] == 8
    assert max(sizes) - min(sizes) == 1
    assert sum(sizes) == 25


def test_panel_item_plan_covers_cartesian_grid_once() -> None:
    plan = MODULE.plan_panel_item_chunks(list(range(10)), 2, 2, 4)

    assert plan["physical_gpu_count"] == 2
    assert plan["logical_worker_count"] == 8
    assert [len(group) for group in plan["groups"]] == [4, 4]
    covered = [
        (item, panel)
        for group in plan["groups"]
        for worker in group
        for item in worker["items"]
        for panel in worker["panels"]
    ]
    assert sorted(covered) == [
        (item, panel) for item in range(10) for panel in range(2)
    ]


def test_panel_item_plan_uses_only_needed_gpus() -> None:
    plan = MODULE.plan_panel_item_chunks([0, 1], 2, 10, 4)

    assert plan["physical_gpu_count"] == 1
    assert plan["logical_worker_count"] == 4
    assert len(plan["groups"][0]) == 4


def test_panel_item_plan_groups_panels_when_worker_slots_are_scarce() -> None:
    plan = MODULE.plan_panel_item_chunks([7], 10, 2, 4)

    assert plan["physical_gpu_count"] == 2
    assert plan["logical_worker_count"] == 8
    covered_panels = [
        panel
        for group in plan["groups"]
        for worker in group
        for panel in worker["panels"]
    ]
    assert sorted(covered_panels) == list(range(10))
