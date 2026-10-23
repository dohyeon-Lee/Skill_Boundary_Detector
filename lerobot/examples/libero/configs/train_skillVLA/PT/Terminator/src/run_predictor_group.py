#!/usr/bin/env python3
"""Supervise independent Predictor or Terminator trainers sharing one GPU allocation.

No DDP, weight sharing, batch splitting, CUDA device reassignment, or extra
Slurm submissions. The parent batch script stages the dataset once. Each
worker reuses the snapshotted YAML and the normal per-model resume path.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path


def _stop_groups(processes: list[subprocess.Popen], timeout: float) -> None:
    """Stop trainers AND their dataloader/launcher descendants, including stragglers."""
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    while any(process.poll() is None for process in processes) and time.monotonic() < deadline:
        time.sleep(0.05)
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_group(
    runs: list[dict],
    command: list[str],
    log_dir: Path,
    *,
    poll_interval: float = 0.2,
    stop_timeout: float = 10.0,
    component: str = "predictor",
) -> int:
    if component not in {"predictor", "terminator"}:
        raise ValueError("Group component must be predictor or terminator.")
    title = component.capitalize()
    selector = "mode" if component == "predictor" else "architecture"
    if not runs or not command:
        raise ValueError(f"A {title} group needs at least one run and a worker command.")
    modes = [run[selector] for run in runs]
    if any(not isinstance(mode, str) or re.fullmatch(r"[A-Za-z0-9_]+", mode) is None for mode in modes):
        raise ValueError(f"Invalid {title} selector in group manifest.")
    if len(set(modes)) != len(modes) or len({run["output_dir"] for run in runs}) != len(runs):
        raise ValueError(f"{title} group selectors and output directories must be distinct.")

    log_dir.mkdir(parents=True, exist_ok=True)
    # Record provenance outside the model output directories: the trainer must
    # create those itself, or resume from a genuine last checkpoint.
    (log_dir / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")
    received_signal = 0

    def request_stop(signum, _frame):
        nonlocal received_signal
        received_signal = signum

    old_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    active: dict[subprocess.Popen, str] = {}
    results: dict[str, int] = {}
    try:
        with ExitStack() as files:
            for run in runs:
                if received_signal:
                    break
                env = os.environ.copy()
                env["SKILL_AUX_GROUP_WORKER"] = "1"
                env.pop("SKILL_AUX_PREDICTOR_MODE_OVERRIDE", None)
                env.pop("SKILL_AUX_TERMINATOR_ARCHITECTURE_OVERRIDE", None)
                key = ("SKILL_AUX_PREDICTOR_MODE_OVERRIDE" if component == "predictor"
                       else "SKILL_AUX_TERMINATOR_ARCHITECTURE_OVERRIDE")
                mode = run[selector]
                env[key] = mode
                # An explicit outer W&B run ID/name must not merge two independent
                # trainers. Resume IDs are loaded from each model's own checkpoint.
                env.pop("WANDB_RUN_ID", None)
                env.pop("WANDB_NAME", None)
                stdout = files.enter_context((log_dir / f"{mode}.out").open("a"))
                stderr = files.enter_context((log_dir / f"{mode}.err").open("a"))
                process = subprocess.Popen(
                    command, env=env, stdout=stdout, stderr=stderr, start_new_session=True
                )
                active[process] = mode
                print(
                    f"START {mode} pid={process.pid} "
                    f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'unset')} "
                    f"output={run['output_dir']} logs={log_dir / mode}.{{out,err}}",
                    flush=True,
                )
            while active and not received_signal:
                for process, mode in list(active.items()):
                    status = process.poll()
                    if status is None:
                        continue
                    results[mode] = status
                    print(f"{'DONE' if status == 0 else 'FAILED'} {mode} exit={status}", flush=True)
                    # A failed trainer can leave dataloader children behind; do
                    # not let those consume memory while the healthy run finishes.
                    _stop_groups([process], timeout=0)
                    del active[process]
                if active:
                    time.sleep(poll_interval)
            if received_signal:
                print(f"Stopping ALL {title} workers (signal {received_signal}).", flush=True)
                return 128 + received_signal
            # One OOM/failure is visible in the allocation's exit status but does
            # not throw away progress in other independent models.
            return 1 if any(results.values()) else 0
    finally:
        try:
            _stop_groups(list(active), timeout=stop_timeout)
            for process, mode in active.items():
                results[mode] = process.returncode
            (log_dir / "exit_codes.json").write_text(json.dumps(results, indent=2) + "\n")
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, help="Resolved run manifest as JSON.")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--component", choices=("predictor", "terminator"), default="predictor")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    raise SystemExit(run_group(json.loads(args.runs), command, args.log_dir, component=args.component))


if __name__ == "__main__":
    main()
