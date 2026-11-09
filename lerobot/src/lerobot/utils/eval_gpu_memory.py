"""Budget shared-GPU evaluation workers before loading any CUDA models."""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess

GIB = 1024**3


def budget(free, total, workers):
    if not 0 < free <= total or workers < 1:
        raise ValueError("Invalid GPU memory reading or evaluation worker count")
    reserve = 16 * GIB
    per_worker = min(30 * GIB, int((free - reserve) // workers))
    if per_worker < 8 * GIB:
        raise ValueError("Not enough free VRAM for evaluation workers with 16 GiB headroom; no workers started")
    return {"budget_bytes": per_worker, "reserve_bytes": reserve, "workers": workers,
            "free_before_bytes": free, "total_bytes": total}


def query(environment, runner=subprocess.check_output):
    text = runner(["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.free",
                   "--format=csv,noheader,nounits"], text=True, timeout=8)
    rows = [[part.strip() for part in line.split(",")] for line in text.splitlines() if line.strip()]
    visible = environment.get("CUDA_VISIBLE_DEVICES", "")
    if visible == "-1" or "," in visible or visible.startswith("MIG-"):
        raise ValueError("Shared evaluation budget requires one visible, non-MIG GPU")
    if len(rows) > 1:
        selector = visible if visible.startswith("GPU-") else environment.get("SLURM_JOB_GPUS") or visible
        rows = [row for row in rows if len(row) == 4 and selector in {row[0], row[1]}]
    if len(rows) != 1 or len(rows[0]) != 4:
        raise ValueError("Cannot identify one assigned evaluation GPU")
    _, uuid, total, free = rows[0]
    total, free = float(total), float(free)
    if not math.isfinite(total) or not math.isfinite(free):
        raise ValueError("Invalid GPU memory reading")
    return uuid, int(total * 1024**2), int(free * 1024**2)


def apply_limit(torch_module=None, environment=None):
    environment = os.environ if environment is None else environment
    value = environment.get("SBD_EVAL_GPU_BUDGET_BYTES")
    if not value:
        return None
    if torch_module is None:
        import torch as torch_module
    if not torch_module.cuda.is_available():
        raise RuntimeError("Cannot enforce evaluation VRAM budget without CUDA")
    device = torch_module.cuda.current_device()
    total = torch_module.cuda.get_device_properties(device).total_memory
    limit = int(value)
    if not 0 < limit <= total:
        raise ValueError("Invalid per-worker evaluation VRAM limit")
    previous = torch_module.cuda.memory.get_per_process_memory_fraction(device)
    fraction = min(previous, limit / total)
    torch_module.cuda.set_per_process_memory_fraction(fraction, device)
    result = {"pid": os.getpid(), "allocator_limit_gib": fraction * total / GIB,
              "device": device, "total_gib": total / GIB}
    print("Evaluation VRAM limit: " + json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    uuid, total, free = query(os.environ)
    settings = budget(free, total, args.workers)
    if args.shell:
        for key, value in {"SBD_EVAL_GPU_BUDGET_BYTES": settings["budget_bytes"],
                           "SBD_EVAL_GPU_UUID": uuid}.items():
            print("export " + key + "=" + shlex.quote(str(value)))
    else:
        print(json.dumps({"uuid": uuid, **settings}))


if __name__ == "__main__":
    main()
