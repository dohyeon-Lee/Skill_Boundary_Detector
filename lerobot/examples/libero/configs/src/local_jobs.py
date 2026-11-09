#!/usr/bin/env python3
"""Detached, single-machine batch/array runner used only by scheduler: local.

No Slurm daemon or training packages required. Environment variables are
inherited by the detached supervisor, never serialized to the job records.
"""
from __future__ import annotations

import contextlib
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[5]


def directory() -> Path:
    return Path(os.environ.get("SBD_LOCAL_JOBS_DIR", ROOT / ".cache/local_jobs")).resolve()


def write(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


@contextlib.contextmanager
def locked(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def fields(base: Path) -> dict[str, str]:
    try:
        return dict(line.split("=", 1) for line in base.with_suffix(".job").read_text().splitlines() if "=" in line)
    except FileNotFoundError:
        return {}


def alive(base: Path) -> bool:
    if base.with_suffix(".exit").exists():
        return False
    try:
        pid = int(base.with_suffix(".pid").read_text())
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return False
        marker = base.with_suffix(".pidstart")
        return not marker.exists() or marker.read_text() == stat[19]
    except (OSError, ValueError, IndexError):
        return False


def state(base: Path) -> str:
    if alive(base):
        return "pending" if base.with_suffix(".pending").exists() else "running"
    try:
        code = base.with_suffix(".exit").read_text().strip()
    except FileNotFoundError:
        return "stopped"
    return {"0": "done", "dependency": "cancelled(dep)", "143": "stopped"}.get(code, f"failed({code})")


def array_indices(spec: str) -> tuple[list[int], int | None]:
    values, separator, limit = spec.partition("%")
    if separator and (not limit.isdigit() or int(limit) < 1):
        raise ValueError("array concurrency must be a positive integer")
    indices = []
    for part in values.split(","):
        match = re.fullmatch(r"(\d+)(?:-(\d+)(?::(\d+))?)?", part)
        if not match:
            raise ValueError(f"unsupported array specification: {spec}")
        start = int(match[1])
        end = int(match[2]) if match[2] else start
        step = int(match[3] or 1)
        if end < start or step < 1 or (end - start) // step > 100000:
            raise ValueError("invalid or excessively large array range")
        indices.extend(range(start, end + 1, step))
    indices = list(dict.fromkeys(indices))
    return indices, int(limit) if separator else None


def dependencies(value: str, folder: Path) -> list[tuple[str, str]]:
    result = []
    if not value:
        return result
    for clause in value.split(","):
        mode, *ids = clause.split(":")
        if mode not in ("afterok", "afterany") or not ids:
            raise ValueError("only afterok:<id>[:<id>] and afterany dependencies are supported locally")
        for identifier in ids:
            if not identifier.isdigit() or not (folder / f"{identifier}.job").exists():
                raise ValueError(f"dependency {identifier} is not a local job")
            result.append((mode, identifier))
    return result


def gpu_pool(want: int) -> list[str]:
    if not want:
        return []
    visible = os.environ.get("SBD_GPUS", os.environ.get("CUDA_VISIBLE_DEVICES"))
    if visible is None:
        try:
            visible = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True, timeout=15,
                stderr=subprocess.DEVNULL,
            ).strip().replace("\n", ",")
        except (OSError, subprocess.SubprocessError):
            raise ValueError("nvidia-smi failed; cannot discover GPUs (set SBD_GPUS=0,1,... to choose)") from None
    pool = list(dict.fromkeys(item.strip() for item in visible.split(",") if item.strip()))
    if any(item in ("-1", "none", "void") for item in pool) or len(pool) < want:
        raise ValueError(f"job needs {want} GPU(s), but only {len(pool) if '-1' not in pool else 0} are available")
    return pool


def parse(args: list[str]) -> dict:
    options = {}
    script = None
    script_args = []
    parsable = False
    for position, arg in enumerate(args):
        if not arg.startswith("-"):
            script = str(Path(arg).resolve())
            script_args = args[position + 1:]
            break
        key, _, value = arg.partition("=")
        if key == "--parsable":
            parsable = True
        elif key in ("--array", "--dependency", "--wrap", "--job-name", "--output", "--error", "--gres", "--cpus-per-task", "--export"):
            if "=" not in arg:
                raise ValueError(f"use {key}=VALUE")
            options[key[2:]] = value
    if script:
        if "wrap" in options or not Path(script).is_file():
            raise ValueError(f"batch script not found or combined with --wrap: {script}")
        for line in Path(script).read_text().splitlines():
            if line.startswith("#SBATCH "):
                for item in shlex.split(line[len("#SBATCH "):], comments=True):
                    key, separator, value = item.partition("=")
                    if separator:
                        options.setdefault(key.removeprefix("--"), value)
        command = ["bash", script, *script_args]
    elif "wrap" in options:
        command = ["bash", "-c", options["wrap"]]
    else:
        raise ValueError("batch script or --wrap=COMMAND is required")
    name = options.get("job-name") or (Path(script).name if script else "wrap")
    if any(char in name for char in "\r\n/"):
        raise ValueError("invalid job name")
    indices, throttle = array_indices(options["array"]) if "array" in options else ([0], 1)
    gres = options.get("gres", "")
    want = int(gres.rsplit(":", 1)[-1]) if gres.startswith("gpu") and gres.rsplit(":", 1)[-1].isdigit() else int(gres.startswith("gpu"))
    pool = gpu_pool(want)
    cpus = int(options.get("cpus-per-task") or 1)
    if cpus < 1:
        raise ValueError("cpus-per-task must be positive")
    return dict(name=name, script=script or "--wrap", command=command, cwd=str(Path.cwd()),
                output=options.get("output", "slurm-%j.out"), error=options.get("error"),
                want=want, pool=pool, cpus=cpus, indices=indices, array="array" in options,
                concurrency=throttle or (max(1, len(pool) // want) if want else 1),
                dependency=options.get("dependency", ""), export=options.get("export", "ALL"), parsable=parsable)


def exported_environment(spec: str) -> dict[str, str]:
    parts = spec.split(",")
    result = dict(os.environ) if "ALL" in parts else {
        key: os.environ[key] for key in ("PATH", "HOME", "USER") if key in os.environ
    }
    for item in parts:
        if item in ("ALL", "NONE", "NIL"):
            continue
        key, separator, value = item.partition("=")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError("invalid --export variable name")
        if separator:
            result[key] = value
        elif key in os.environ:
            result[key] = os.environ[key]
    return result


def log_path(pattern: str, job: dict, index: int | None) -> Path:
    task_id = f"{job['id']}_{index}" if job["array"] and index is not None else job["id"]
    result = pattern.replace("%x", job["name"]).replace("%j", task_id).replace("%A", job["id"]).replace("%a", str(index or 0))
    return (Path(job["cwd"]) / result).resolve()


def save_record(base: Path, job: dict, allocations: dict) -> None:
    gpus = list(dict.fromkeys(gpu for devices in allocations.values() for gpu in devices))
    record = dict(name=job["name"], script=job["script"], cwd=job["cwd"], gpus=",".join(gpus),
                  dependency=job["dependency"], output=str(job["log"]), error=str(job["error_log"]), started=job["started"])
    if job.get("eval_output"):
        record["eval_output"] = job["eval_output"]
    write(base.with_suffix(".job"), "".join(f"{key}={value}\n" for key, value in record.items()))
    write(base.with_suffix(".alloc.json"), json.dumps(allocations))


def held_gpus(folder: Path, own: Path) -> set[str]:
    held = set()
    for record in folder.glob("*.job"):
        base = record.with_suffix("")
        if base == own or not alive(base):
            continue
        held.update(filter(None, fields(base).get("gpus", "").split(",")))
    return held


def group_alive(identifier: int) -> bool:
    """Zombie descendants hold no GPU and cannot be killed; ignore them."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(stat[2]) == identifier and stat[0] != "Z":
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def stop_process(process: subprocess.Popen) -> None:
    # Each payload owns its session. Give Python trainers time to stop their
    # own worker sessions before escalating, even if their Bash parent exited.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + 5
    while group_alive(process.pid) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.05)
    if group_alive(process.pid):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def submit(args: list[str]) -> int:
    job = parse(args)
    folder = directory()
    # --export assignments are applied in memory, never written into metadata.
    environment = exported_environment(job.pop("export"))
    environment["SBD_LOCAL_JOBS_DIR"] = str(folder)
    with locked(folder):
        dependencies(job["dependency"], folder)
        identifier = str(time.time_ns())
        while (folder / f"{identifier}.job").exists():
            identifier = str(int(identifier) + 1)
        job.update(id=identifier, started=datetime.now().astimezone().isoformat(timespec="seconds"))
        if os.environ.get("EVAL_OUT_DIR") and "/eval/" in job["script"]:
            job["eval_output"] = os.environ["EVAL_OUT_DIR"]
        base = folder / identifier
        job["log"] = str(log_path(job["output"], job, None))
        if job["array"]:
            job["log"] += ".controller"
        job["error_log"] = str(log_path(job["error"] or job["output"], job, None))
        save_record(base, job, {})
        write(base.with_suffix(".json"), json.dumps(job))
        base.with_suffix(".pending").touch()
        Path(job["log"]).parent.mkdir(parents=True, exist_ok=True)
        with Path(job["log"]).open("ab") as log:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "run", str(base)],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, close_fds=True, start_new_session=True,
                env=environment,
            )
        write(base.with_suffix(".pid"), str(process.pid))
        try:
            start = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
            write(base.with_suffix(".pidstart"), start)
        except (OSError, IndexError):
            pass
    # Give the first scheduling pass time to claim GPUs or publish a pending state.
    deadline = time.monotonic() + 5
    while not base.with_suffix(".ready").exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    if job["parsable"]:
        print(identifier)
    else:
        print(f"Submitted local job {identifier} (pid {process.pid}, {state(base)})")
        print(f"  log : {job['log']}")
        print("  list/stop: bash lerobot/examples/libero/configs/src/submit_job.sh list | stop <id>")
    return 0


def run(base: Path) -> int:
    job = json.loads(base.with_suffix(".json").read_text())
    folder = base.parent
    pending = list(job["indices"])
    processes = {}
    allocations = {}
    codes = []
    stopping = False
    result = "1"

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        deps = dependencies(job["dependency"], folder)
        while deps and not stopping:
            waiting = False
            for mode, identifier in deps:
                dependency = folder / identifier
                if alive(dependency):
                    waiting = True
                elif mode == "afterok" and state(dependency) != "done":
                    print(f"submit_job: dependency {identifier} did not finish successfully; not starting.", flush=True)
                    result = "dependency"
                    return 0
            if not waiting:
                break
            base.with_suffix(".ready").touch()
            time.sleep(float(os.environ.get("SBD_LOCAL_POLL", "0.1")))
        while (pending or processes) and not stopping:
            for index, process in list(processes.items()):
                code = process.poll()
                if code is not None:
                    stop_process(process)  # Reap leftover workers before releasing their GPU.
                    codes.append(code)
                    write(folder / f"{job['id']}_{index}.task-exit", str(code))
                    del processes[index]
                    del allocations[str(index)]
            with locked(folder):
                free = [gpu for gpu in job["pool"] if gpu not in held_gpus(folder, base)
                        and all(gpu not in devices for devices in allocations.values())]
                while pending and len(processes) < job["concurrency"] and len(free) >= job["want"]:
                    index = pending.pop(0)
                    devices = free[:job["want"]]
                    del free[:job["want"]]
                    env = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}
                    env.update(SBD_LOCAL_JOB_ID=job["id"], SBD_SCHEDULER="local", SLURM_SUBMIT_DIR=job["cwd"],
                               SLURM_CPUS_PER_TASK=str(job["cpus"]))
                    env.pop("SBD_SRUN_SESSION", None)
                    env.pop("SBD_SRUN_EXISTING_STEP", None)
                    if devices:
                        env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
                    else:
                        env["CUDA_VISIBLE_DEVICES"] = ""
                    if job["array"]:
                        env.update(SLURM_ARRAY_TASK_ID=str(index), SLURM_ARRAY_JOB_ID=job["id"],
                                   SLURM_ARRAY_TASK_COUNT=str(len(job["indices"])),
                                   SLURM_ARRAY_TASK_MIN=str(min(job["indices"])), SLURM_ARRAY_TASK_MAX=str(max(job["indices"])))
                    output = log_path(job["output"], job, index)
                    error = log_path(job["error"] or job["output"], job, index)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    error.parent.mkdir(parents=True, exist_ok=True)
                    with output.open("ab") as out:
                        if output == error:
                            processes[index] = subprocess.Popen(job["command"], cwd=job["cwd"], env=env, stdout=out, stderr=out, close_fds=True, start_new_session=True)
                        else:
                            with error.open("ab") as err:
                                processes[index] = subprocess.Popen(job["command"], cwd=job["cwd"], env=env, stdout=out, stderr=err, close_fds=True, start_new_session=True)
                    allocations[str(index)] = devices
                save_record(base, job, allocations)
            if processes:
                base.with_suffix(".pending").unlink(missing_ok=True)
            elif pending:
                base.with_suffix(".pending").touch()
            base.with_suffix(".ready").touch()
            if pending or processes:
                time.sleep(float(os.environ.get("SBD_LOCAL_POLL", "0.1")))
        result = "143" if stopping else str(next((abs(code) for code in codes if code), 0))
    finally:
        for process in processes.values():
            stop_process(process)
        with locked(folder):
            save_record(base, job, {})
            write(base.with_suffix(".exit"), result)
            base.with_suffix(".pending").unlink(missing_ok=True)
            base.with_suffix(".ready").touch()
    return 0


def main(args: list[str]) -> int:
    action = args[0] if args else "list"
    folder = directory()
    if action == "submit":
        return submit(args[1:])
    if action == "run":
        return run(Path(args[1]))
    if action == "list":
        print(f"{'ID':<20} {'STATE':<15} {'GPUS':<6} {'STARTED':<26} NAME")
        for record in sorted(folder.glob("*.job")):
            base = record.with_suffix("")
            info = fields(base)
            print(f"{base.name:<20} {state(base):<15} {info.get('gpus', ''):<6} {info.get('started', ''):<26} {info.get('name', '')}  {info.get('output', '')}")
        return 0
    if action in ("active", "stop"):
        if len(args) != 2 or not args[1].isdigit():
            raise ValueError(f"usage: {action} <numeric local job id>")
        base = folder / args[1]
        if not base.with_suffix(".job").is_file():
            raise ValueError(f"No local job {args[1]}.")
        if action == "active":
            return 0 if alive(base) else 1
        if not alive(base):
            print(f"Local job {args[1]} is not running ({state(base)}).")
            return 0
        pid = int(base.with_suffix(".pid").read_text())
        # Only signal the detached job's own session; never the caller's group.
        if os.getpgid(pid) != pid:
            raise ValueError("Local job no longer owns its recorded process group.")
        os.killpg(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to local job {args[1]} (process group {pid}).")
        return 0
    raise ValueError("usage: local_jobs.py [submit ARGS | list | active ID | stop ID]")


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"submit_job: {error}", file=sys.stderr)
        raise SystemExit(2)
