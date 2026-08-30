#!/usr/bin/env python3
"""Plan independent evaluation workers onto a bounded number of GPUs."""

from __future__ import annotations

import argparse
import json
import math
import shlex
from collections.abc import Sequence
from typing import Any

MAX_SUPPORTED_WORKERS_PER_GPU = 4


def _validate(unit_count: int, requested_gpus: int, max_workers_per_gpu: int) -> None:
    if unit_count <= 0:
        raise ValueError("unit_count must be positive.")
    if requested_gpus <= 0:
        raise ValueError("eval_num_gpus must be positive.")
    if not 1 <= max_workers_per_gpu <= MAX_SUPPORTED_WORKERS_PER_GPU:
        raise ValueError(
            "eval_max_workers_per_gpu must be between 1 and "
            f"{MAX_SUPPORTED_WORKERS_PER_GPU}."
        )


def balanced_chunks(items: Sequence[int], chunk_count: int) -> list[list[int]]:
    if not items or not 1 <= chunk_count <= len(items):
        raise ValueError("chunk_count must be between 1 and len(items).")
    base, extra = divmod(len(items), chunk_count)
    chunks: list[list[int]] = []
    start = 0
    for index in range(chunk_count):
        end = start + base + int(index < extra)
        chunks.append([int(value) for value in items[start:end]])
        start = end
    return chunks


def plan_worker_indices(
    unit_count: int,
    requested_gpus: int,
    max_workers_per_gpu: int,
) -> dict[str, Any]:
    """Return balanced logical-worker groups for physical one-GPU jobs."""
    _validate(unit_count, requested_gpus, max_workers_per_gpu)
    # ``requested_gpus`` is a ceiling, not a target to reserve unconditionally.
    # If four workers fit on one GPU, ten units need only three GPUs even when
    # the YAML permits ten. This avoids longer queue waits and wasted QoS quota.
    useful_gpu_count = math.ceil(unit_count / max_workers_per_gpu)
    gpu_count = min(requested_gpus, useful_gpu_count)
    worker_count = min(unit_count, gpu_count * max_workers_per_gpu)
    groups = balanced_chunks(list(range(worker_count)), gpu_count)
    return {
        "physical_gpu_count": gpu_count,
        "logical_worker_count": worker_count,
        "groups": groups,
    }


def plan_item_chunks(
    items: Sequence[int],
    requested_gpus: int,
    max_workers_per_gpu: int,
) -> dict[str, Any]:
    """Split items into balanced chunks, then pack those chunks onto GPUs."""
    if len(set(items)) != len(items):
        raise ValueError("items must be unique.")
    base_plan = plan_worker_indices(len(items), requested_gpus, max_workers_per_gpu)
    chunks = balanced_chunks(items, base_plan["logical_worker_count"])
    workers = [
        {
            "worker_index": index,
            "items": chunk,
            "tag": f"w{index:03d}_t{chunk[0]}-{chunk[-1]}",
        }
        for index, chunk in enumerate(chunks)
    ]
    return {
        **base_plan,
        "groups": [
            [workers[worker_index] for worker_index in group]
            for group in base_plan["groups"]
        ],
    }


def plan_panel_item_chunks(
    items: Sequence[int],
    panel_count: int,
    requested_gpus: int,
    max_workers_per_gpu: int,
) -> dict[str, Any]:
    """Pack a complete ``item x panel`` grid without losing panel identity.

    A logical worker may consume several items for one panel.  When there are
    fewer worker slots than panels, it instead consumes every item for a small
    panel group.  Consequently each Cartesian-grid unit is evaluated exactly
    once while callers can still express a worker with ``ITEMS`` and
    ``PANEL_INDICES`` environment variables.
    """
    if len(set(items)) != len(items):
        raise ValueError("items must be unique.")
    if not items:
        raise ValueError("items must be non-empty.")
    if panel_count <= 0:
        raise ValueError("panel_count must be positive.")

    unit_count = len(items) * panel_count
    base_plan = plan_worker_indices(
        unit_count, requested_gpus, max_workers_per_gpu
    )
    worker_count = int(base_plan["logical_worker_count"])
    workers: list[dict[str, Any]] = []

    if worker_count >= panel_count:
        # Give every panel at least one worker, then distribute the remaining
        # slots evenly. worker_count <= item_count * panel_count guarantees no
        # panel receives more workers than it has items.
        base, extra = divmod(worker_count, panel_count)
        worker_index = 0
        for panel_index in range(panel_count):
            panel_workers = base + int(panel_index < extra)
            for item_chunk in balanced_chunks(items, panel_workers):
                workers.append(
                    {
                        "worker_index": worker_index,
                        "items": item_chunk,
                        "panels": [panel_index],
                    }
                )
                worker_index += 1
    else:
        # Preserve the complete comparison even when there are fewer logical
        # workers than panels: one process walks a small panel group.
        for worker_index, panel_chunk in enumerate(
            balanced_chunks(list(range(panel_count)), worker_count)
        ):
            workers.append(
                {
                    "worker_index": worker_index,
                    "items": [int(value) for value in items],
                    "panels": panel_chunk,
                }
            )

    for worker in workers:
        item_values = worker["items"]
        panel_values = worker["panels"]
        item_tag = (
            f"t{item_values[0]}"
            if len(item_values) == 1
            else f"t{item_values[0]}-{item_values[-1]}"
        )
        panel_tag = (
            f"p{panel_values[0]:02d}"
            if len(panel_values) == 1
            else f"p{panel_values[0]:02d}-{panel_values[-1]:02d}"
        )
        worker["tag"] = f"w{worker['worker_index']:03d}_{item_tag}_{panel_tag}"
        worker["unit_count"] = len(item_values) * len(panel_values)

    # Longest-processing-time placement keeps sequential work balanced while
    # enforcing the hard per-GPU process cap.
    gpu_count = int(base_plan["physical_gpu_count"])
    groups: list[list[dict[str, Any]]] = [[] for _ in range(gpu_count)]
    group_loads = [0] * gpu_count
    for worker in sorted(
        workers, key=lambda row: (-int(row["unit_count"]), row["worker_index"])
    ):
        candidates = [
            index
            for index, group in enumerate(groups)
            if len(group) < max_workers_per_gpu
        ]
        target = min(
            candidates,
            key=lambda index: (group_loads[index], len(groups[index]), index),
        )
        groups[target].append(worker)
        group_loads[target] += int(worker["unit_count"])

    return {
        **base_plan,
        "groups": groups,
        "max_units_per_worker": max(int(row["unit_count"]) for row in workers),
        "max_units_per_gpu": max(group_loads),
    }


def _shell_exports(plan: dict[str, Any]) -> str:
    groups_json = json.dumps(plan["groups"], separators=(",", ":"))
    lines = [
        f"export EVAL_PHYSICAL_GPU_COUNT={plan['physical_gpu_count']}",
        f"export EVAL_LOGICAL_WORKER_COUNT={plan['logical_worker_count']}",
        f"export EVAL_GPU_GROUPS_JSON={shlex.quote(groups_json)}",
    ]
    for key, shell_name in (
        ("max_units_per_worker", "EVAL_MAX_UNITS_PER_WORKER"),
        ("max_units_per_gpu", "EVAL_MAX_UNITS_PER_GPU"),
    ):
        if key in plan:
            lines.append(f"export {shell_name}={int(plan[key])}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--unit-count", type=int)
    source.add_argument("--items-json")
    parser.add_argument("--panel-count", type=int)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--max-workers-per-gpu", type=int, required=True)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()

    if args.items_json is not None:
        items = json.loads(args.items_json)
        if not isinstance(items, list) or not items:
            raise ValueError("--items-json must contain a non-empty JSON list.")
        parsed_items = [int(value) for value in items]
        if args.panel_count is not None:
            plan = plan_panel_item_chunks(
                parsed_items,
                args.panel_count,
                args.gpus,
                args.max_workers_per_gpu,
            )
        else:
            plan = plan_item_chunks(
                parsed_items, args.gpus, args.max_workers_per_gpu
            )
    else:
        if args.panel_count is not None:
            raise ValueError("--panel-count requires --items-json.")
        plan = plan_worker_indices(
            args.unit_count, args.gpus, args.max_workers_per_gpu
        )
    if args.shell:
        print(_shell_exports(plan))
    else:
        print(json.dumps(plan, separators=(",", ":")))


if __name__ == "__main__":
    main()
