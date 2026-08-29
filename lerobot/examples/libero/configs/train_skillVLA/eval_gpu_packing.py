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


def _shell_exports(plan: dict[str, Any]) -> str:
    groups_json = json.dumps(plan["groups"], separators=(",", ":"))
    return "\n".join(
        [
            f"export EVAL_PHYSICAL_GPU_COUNT={plan['physical_gpu_count']}",
            f"export EVAL_LOGICAL_WORKER_COUNT={plan['logical_worker_count']}",
            f"export EVAL_GPU_GROUPS_JSON={shlex.quote(groups_json)}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--unit-count", type=int)
    source.add_argument("--items-json")
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--max-workers-per-gpu", type=int, required=True)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()

    if args.items_json is not None:
        items = json.loads(args.items_json)
        if not isinstance(items, list) or not items:
            raise ValueError("--items-json must contain a non-empty JSON list.")
        plan = plan_item_chunks(
            [int(value) for value in items], args.gpus, args.max_workers_per_gpu
        )
    else:
        plan = plan_worker_indices(
            args.unit_count, args.gpus, args.max_workers_per_gpu
        )
    if args.shell:
        print(_shell_exports(plan))
    else:
        print(json.dumps(plan, separators=(",", ":")))


if __name__ == "__main__":
    main()
