#!/usr/bin/env python3
"""Supervise independent Predictor, Terminator or VSA trainers on one GPU allocation.

No DDP, weight sharing, batch splitting, CUDA device reassignment, or extra
Slurm submissions. The parent batch script stages the dataset once. Each
worker reuses the snapshotted YAML and the normal per-model resume path.
The worker shell launcher is copied too: Bash resumes reading it after the
long-running trainer exits, so in-place repository edits must not reach it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import tempfile
import time
import uuid
from contextlib import ExitStack
from datetime import datetime, timezone
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
    if component not in {"predictor", "terminator", "vsa"}:
        raise ValueError("Group component must be predictor, terminator or vsa.")
    title = "VSA" if component == "vsa" else component.capitalize()
    selector = "mode" if component == "predictor" else "architecture"
    if not runs or not command:
        raise ValueError(f"A {title} group needs at least one run and a worker command.")
    modes = [run[selector] for run in runs]
    if any(not isinstance(mode, str) or re.fullmatch(r"[A-Za-z0-9_]+", mode) is None for mode in modes):
        raise ValueError(f"Invalid {title} selector in group manifest.")
    if len(set(modes)) != len(modes) or len({run["output_dir"] for run in runs}) != len(runs):
        raise ValueError(f"{title} group selectors and output directories must be distinct.")
    if component == "vsa" and any(
        not isinstance(run.get("config"), str) or not Path(run["config"]).is_file()
        for run in runs
    ):
        raise ValueError("Each VSA worker needs its own existing config snapshot.")

    log_dir.mkdir(parents=True, exist_ok=True)
    worker_command = list(command)
    worker_source_dir = None
    if (len(command) >= 2 and Path(command[0]).name == "bash"
            and Path(command[1]).name == "train.sbatch"):
        source = Path(command[1]).resolve()
        # Slurm freezes the parent script, but the group used to start child
        # Bash processes against the mutable repository file. Keep a unique
        # on-disk copy for this launch, never overwrite an existing live copy.
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=log_dir, prefix="worker_train_", suffix=".sbatch", delete=False,
        ) as snapshot:
            snapshot.write(source.read_bytes())
            worker_command[1] = str(Path(snapshot.name).resolve())
        worker_source_dir = str(source.parent)
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
    events = {}

    def record_event(mode, run=None, status=None):
        # The persistent-shell supervisor supplies a shared private journal.
        # Outside that supervisor, leave the existing group runner unchanged.
        if not os.environ.get("SBD_TRAINING_EVENTS_DIR"):
            return
        if run is not None:
            events[mode] = {"run_id": uuid.uuid4().hex, "name": mode,
                "parent_run_id": os.environ.get("SBD_TRAINING_PARENT_ID", ""),
                "job_id": os.environ.get("SLURM_JOB_ID", ""), "work_dir": str(Path.cwd()),
                "state": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat(),
                "outputs": [run["output_dir"]], "output": str(log_dir / f"{mode}.out"),
                "error": str(log_dir / f"{mode}.err")}
        if mode not in events:
            return
        event = events[mode]
        if status is not None:
            event.update(state="COMPLETED" if status == 0 else "FAILED", exit_code=status,
                         ended_at=datetime.now(timezone.utc).isoformat())
        folder = Path(os.environ["SBD_TRAINING_EVENTS_DIR"])
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        target = folder / (event["run_id"] + ".json")
        if target.exists():
            try:
                resource = json.loads(target.read_text()).get("resource_status_path")
                if resource:
                    event["resource_status_path"] = resource
            except (OSError, ValueError):
                pass
        temporary = target.with_suffix(".tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(event, stream)
        temporary.replace(target)
    try:
        with ExitStack() as files:
            for run in runs:
                if received_signal:
                    break
                env = os.environ.copy()
                env["STAGE1_GROUP_WORKER" if component == "vsa" else "SKILL_AUX_GROUP_WORKER"] = "1"
                if worker_source_dir is not None:
                    # BASH_SOURCE now points into the log directory; config
                    # helpers still live beside the original launcher.
                    env.setdefault("STAGE1_SRC_DIR" if component == "vsa" else "SKILL_AUX_SRC_DIR",
                                   worker_source_dir)
                env.pop("SKILL_AUX_PREDICTOR_MODE_OVERRIDE", None)
                env.pop("SKILL_AUX_TERMINATOR_ARCHITECTURE_OVERRIDE", None)
                key = {"predictor": "SKILL_AUX_PREDICTOR_MODE_OVERRIDE",
                       "terminator": "SKILL_AUX_TERMINATOR_ARCHITECTURE_OVERRIDE",
                       "vsa": "STAGE1_ARCHITECTURE_OVERRIDE"}[component]
                mode = run[selector]
                env[key] = mode
                if component == "vsa":
                    env["STAGE1_TRAIN_CONFIG"] = str(Path(run["config"]).resolve())
                # An explicit outer W&B run ID/name must not merge two independent
                # trainers. Resume IDs are loaded from each model's own checkpoint.
                env.pop("WANDB_RUN_ID", None)
                env.pop("WANDB_NAME", None)
                stdout = files.enter_context((log_dir / f"{mode}.out").open("a"))
                stderr = files.enter_context((log_dir / f"{mode}.err").open("a"))
                record_event(mode, run=run)
                env["SBD_GPU_GUARD_ORDER_NS"] = str(time.time_ns())
                if mode in events:
                    env["SBD_TRAINING_PARENT_ID"] = events[mode]["run_id"]
                    env["SBD_TRAINING_STATUS_PATH"] = str(Path(env["SBD_TRAINING_EVENTS_DIR"]) / (events[mode]["run_id"] + ".json"))
                try:
                    process = subprocess.Popen(
                        worker_command, env=env, stdout=stdout, stderr=stderr, start_new_session=True
                    )
                except OSError:
                    record_event(mode, status=1)
                    raise
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
                    record_event(mode, status=status)
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
                record_event(mode, status=process.returncode)
            (log_dir / "exit_codes.json").write_text(json.dumps(results, indent=2) + "\n")
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, help="Resolved run manifest as JSON.")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--component", choices=("predictor", "terminator", "vsa"), default="predictor")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    raise SystemExit(run_group(json.loads(args.runs), command, args.log_dir, component=args.component))


if __name__ == "__main__":
    main()
