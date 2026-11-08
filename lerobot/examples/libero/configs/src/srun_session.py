#!/usr/bin/env python3
"""Launch a persistent Slurm shell in a uniquely named, detached tmux session."""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import uuid
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
SELF = Path(__file__).resolve()
BATCH_ONLY = {"--array", "--dependency", "--parsable", "--wrap", "--wait", "--hold"}
ALIASES = {"-J": "--job-name", "-p": "--partition", "-q": "--qos", "-w": "--nodelist",
           "-x": "--exclude", "-c": "--cpus-per-task", "-t": "--time", "-o": "--output",
           "-e": "--error", "-N": "--nodes", "-n": "--ntasks", "-a": "--array"}
FLAGS = {"--requeue", "--no-requeue", "--exclusive", "--oversubscribe", "--no-kill"}
SELECTORS = ("STAGE1_TRAIN_CONFIG", "STAGE1_ARCHITECTURE_OVERRIDE", "STAGE1_GROUP_WORKER_SCRIPT",
             "STAGE1_GROUP_RUNNER", "SKILL_AUX_TRAIN_CONFIG", "SKILL_AUX_PREDICTOR_MODE_OVERRIDE",
             "SKILL_AUX_TERMINATOR_ARCHITECTURE_OVERRIDE", "STAGE1_EVAL_CONFIG",
             "STAGE1_EVAL_CONFIG_RESOLVER", "STAGE1_EVAL_DIR", "STAGE1_EVAL_WORK_DIR")


def existing_allocation(identifier):
    if not isinstance(identifier, str) or re.fullmatch(r"[1-9][0-9]*", identifier) is None:
        raise ValueError("srun_existing requires a positive numeric srun_job_id.")
    text = subprocess.check_output(["scontrol", "show", "job", "-o", identifier], text=True, timeout=15)
    info = dict(part.split("=", 1) for part in text.split() if "=" in part)
    if info.get("JobId") != identifier or not info.get("UserId", "").endswith(f"({os.getuid()})"):
        raise ValueError("The requested allocation does not belong to the current user.")
    if info.get("JobState") != "RUNNING":
        raise ValueError("srun_existing requires a RUNNING allocation; no new allocation was requested.")
    gpus = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", info.get("AllocTRES", info.get("TRES", "")))
    if info.get("NumNodes") != "1" or not gpus or gpus[1] != "1":
        raise ValueError("srun_existing currently requires a one-node, one-GPU allocation.")
    return info


def parse_options(tokens):
    options = {}
    index = 0
    while index < len(tokens) and tokens[index].startswith("-"):
        token = tokens[index]
        if token == "--":
            index += 1
            break
        key, sep, value = token.partition("=")
        key = ALIASES.get(key, key)
        if not sep and key not in FLAGS and key not in {"--parsable", "--wait", "--hold"}:
            index += 1
            if index >= len(tokens):
                raise ValueError(f"Missing value for {key}")
            value = tokens[index]
        options[key] = value if sep or value else None
        index += 1
    return options, tokens[index:]


def request(arguments):
    options, command = parse_options(list(arguments))
    if not command:
        raise ValueError("A training batch script is required.")
    script = Path(command[0]).resolve()
    if not script.is_file():
        raise ValueError(f"Batch script not found: {script}")
    defaults = {}
    for line in script.read_text().splitlines():
        if line.startswith("#SBATCH "):
            parsed, _ = parse_options(shlex.split(line[len("#SBATCH "):], comments=True))
            defaults.update(parsed)
    options = {**defaults, **options}
    return options, [str(script), *command[1:]]


def private_json(path, data):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
    temporary.replace(path)


def tmux_command():
    command = ["tmux"]
    if os.environ.get("SBD_TMUX_SOCKET"):
        command += ["-L", os.environ["SBD_TMUX_SOCKET"]]
    return command


def submit(arguments, existing_job_id=None, reserve_only=False):
    if reserve_only:
        _, command = parse_options(list(arguments))
        if command:
            raise ValueError("reserve accepts allocation options only, without a training script.")
        root = Path(os.environ.get("SBD_SRUN_SESSIONS_DIR", REPO / ".cache/srun_sessions")).resolve()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        script = root / "reserve.sbatch"
        script.write_text("#!/usr/bin/env bash\nexit 0\n")
        arguments = [*arguments, str(script)]
    explicit, _ = parse_options(list(arguments))
    if set(explicit) & BATCH_ONLY:
        if existing_job_id is not None:
            raise ValueError("srun_existing does not submit arrays/dependencies/parsable jobs; no new allocation was requested.")
        print("srun mode: arrays/dependencies/numeric job-ID requests use sbatch.", file=sys.stderr)
        return subprocess.call(["sbatch", *arguments])
    options, command = request(arguments)
    if set(options) & BATCH_ONLY:
        if existing_job_id is not None:
            raise ValueError("srun_existing requires a single step, not a batch workflow.")
        print("srun mode: arrays/dependencies/numeric job-ID requests use sbatch.", file=sys.stderr)
        return subprocess.call(["sbatch", *arguments])
    if options.get("--nodes", "1") != "1" or options.get("--ntasks", "1") != "1":
        raise ValueError("Persistent srun mode supports one node and one launcher task.")
    allocation = existing_allocation(existing_job_id) if existing_job_id is not None else None
    if allocation:
        if str(options.get("--gres", "gpu:1")).rsplit(":", 1)[-1] != "1":
            raise ValueError("An additional step must request one shared GPU.")
        if int(options.get("--cpus-per-task", "1")) > int(allocation["NumCPUs"]):
            raise ValueError(f"Requested CPUs exceed allocation {existing_job_id}'s {allocation['NumCPUs']} CPUs.")
    reuse = (existing_job_id is None and os.environ.get("SBD_SRUN_SESSION") == "1"
             and bool(os.environ.get("SLURM_JOB_ID")))
    if not reuse:
        for tool in ("tmux", "srun"):
            if not shutil.which(tool):
                raise ValueError(f"srun mode requires {tool}.")
    name = str(options.get("--job-name") or Path(command[0]).stem)
    label = re.sub(r"[^A-Za-z0-9_-]", "-", name)[:40] or "train"
    unique = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:10]
    session = f"srun-{label}-{unique}"
    root = Path(os.environ.get("SBD_SRUN_SESSIONS_DIR", REPO / ".cache/srun_sessions")).resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory = root / session
    directory.mkdir(mode=0o700)
    path = directory / "launch.json"
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"TMUX", "TMUX_PANE"} and not key.startswith("SLURM_")
                   and key != "CUDA_VISIBLE_DEVICES"}
    data = {"name": name, "session": session, "cwd": str(Path.cwd()), "command": command,
            "environment": environment, "options": options}
    if reserve_only:
        data["reserve_only"] = True
    if allocation:
        data.update(existing_job_id=existing_job_id, allocation_submit_time=allocation["SubmitTime"])
    private_json(path, data)
    if reuse:
        print(f"Reusing Slurm allocation {os.environ['SLURM_JOB_ID']}; no new job requested.", flush=True)
        return run_training(path)

    rc = directory / "allocated.bashrc"
    retry = shlex.join([sys.executable, str(SELF), "run", str(path)])
    release = "Close this extra step: exit (main allocation stays)" if allocation else "Release GPU: exit"
    rc.write_text(
        'if [ -f "$HOME/.bashrc" ]; then source "$HOME/.bashrc"; fi\n'
        'export SBD_SRUN_SESSION=1\n'
        f'export SBD_SRUN_EXISTING_STEP={1 if allocation else 0}\n'
        f'export SLURM_SUBMIT_DIR={shlex.quote(data["cwd"])}\n'
        f'cd -- {shlex.quote(data["cwd"])}\n'
        f'sbd_retry() {{ {retry}; }}\n'
        'if sbd_retry; then sbd_status=0; else sbd_status=$?; fi\n' +
        ('echo "GPU allocation is ready. No training was started."\n' if reserve_only else
         'printf "\\nTraining exited (%s). GPU allocation is still held.\\n" "$sbd_status"\n') +
        f'echo "Retry frozen settings: sbd_retry | New settings: ./submit_train.sh | {release}"\n'
        'unset ' + ' '.join(SELECTORS) + '\n'
    )
    # argv and environment remain private on disk, never interpolated into shell source.
    launch = shlex.join([sys.executable, str(SELF), "connect", str(path)])
    tmux = tmux_command()
    try:
        # The tmux pane owns connect/srun itself, not an outer interactive shell.
        # Training may exit while allocated.bashrc keeps the GPU for retries;
        # only the end of the allocation/extra step closes this tmux session.
        subprocess.run([*tmux, "new-session", "-d", "-s", session, "-c", data["cwd"],
                        "exec " + launch, ";", "set-option", "-w", "-t", session,
                        "remain-on-exit", "off"], check=True)
    except Exception:
        subprocess.run([*tmux, "kill-session", "-t", session], check=False)
        raise
    print(f"Started srun session: {session}")
    attach = "switch-client" if os.environ.get("TMUX") else "attach"
    print("Attach: " + shlex.join([*tmux, attach, "-t", session]))
    print(f"Record: {path}")
    return 0


def connect(path):
    data = json.loads(path.read_text())
    environment = os.environ.copy()
    environment.update(data["environment"])
    for key in list(environment):
        if key.startswith("SLURM_") or key == "CUDA_VISIBLE_DEVICES":
            environment.pop(key)
    options = {key: value for key, value in data["options"].items()
               if key not in {"--output", "--error", "--requeue", "--no-requeue", "--chdir"}}
    options.update({"--nodes": "1", "--ntasks": "1", "--kill-on-bad-exit": "0"})
    if data.get("existing_job_id"):
        allocation = existing_allocation(data["existing_job_id"])
        if allocation.get("SubmitTime") != data["allocation_submit_time"]:
            raise ValueError("Allocation identity changed since submission; no step was started.")
        for key in ("--partition", "--qos", "--nodelist", "--exclude", "--account", "--reservation",
                    "--constraint", "--time", "--exclusive", "--oversubscribe"):
            options.pop(key, None)
        options.update({"--jobid": data["existing_job_id"], "--overlap": None, "--gres": "gpu:1"})
    args = ["srun", *(key if value is None else f"{key}={value}" for key, value in options.items()),
            "--chdir=" + data["cwd"], "--pty", "bash", "--noprofile", "--rcfile",
            str(path.parent / "allocated.bashrc"), "-i"]
    action = "Adding step to allocation " + data["existing_job_id"] if data.get("existing_job_id") else "Requesting Slurm allocation"
    print(action + " in " + data["session"], flush=True)
    code = subprocess.call(args, cwd=data["cwd"], env=environment)
    private_json(path.parent / "allocation_exit.json", {"exit_code": code})
    message = "additional step closed; main allocation unchanged" if data.get("existing_job_id") else "this session no longer holds the allocation"
    print(f"srun exited ({code}); {message}.", flush=True)
    return code


def run_training(path):
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Training must run inside the allocated Slurm shell.")
    data = json.loads(path.read_text())
    environment = os.environ.copy()
    environment.update(data["environment"])
    # Slurm assigns visible devices on the compute node; never restore the submit host's mapping.
    for key, value in os.environ.items():
        if key.startswith("SLURM_") or key in {"CUDA_VISIBLE_DEVICES", "TMUX", "TMUX_PANE"}:
            environment[key] = value
    environment.update(SBD_SRUN_SESSION="1", SLURM_SUBMIT_DIR=data["cwd"])
    if data.get("existing_job_id"):
        environment["SBD_SRUN_EXISTING_STEP"] = "1"
    identifier = environment["SLURM_JOB_ID"]
    if data.get("reserve_only"):
        private_json(path.parent / "allocation_status.json", {
            "job_id": identifier, "node": environment.get("SLURMD_NODENAME", ""),
            "state": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat(),
            "reserve_only": True})
        return 0
    substitutions = {"%j": identifier, "%A": identifier, "%x": data["name"],
                     "%u": environment.get("USER", "user"), "%N": environment.get("SLURMD_NODENAME", "node")}
    def log_path(option, suffix):
        text = data["options"].get(option) or f"logs/{data['name']}_%j.{suffix}"
        for key, value in substitutions.items():
            text = text.replace(key, value)
        target = Path(data["cwd"]) / text
        target.parent.mkdir(parents=True, exist_ok=True)
        return target
    output, error = log_path("--output", "out"), log_path("--error", "err")
    if environment.get("SBD_SRUN_EXISTING_STEP") == "1":
        step = environment.get("SLURM_STEP_ID", "unknown")
        if re.fullmatch(r"[0-9]+", step) is None:
            raise ValueError("An additional srun step requires a numeric SLURM_STEP_ID.")
        output = output.with_name(output.stem + f".step{step}" + output.suffix)
        error = error.with_name(error.stem + f".step{step}" + error.suffix)
    event_root = Path(environment.get("SBD_TRAINING_EVENTS_DIR") or
                      (Path(environment["SBD_SRUN_SESSIONS_DIR"]).parent / "training_events"
                       if environment.get("SBD_SRUN_SESSIONS_DIR") else REPO / ".cache/training_events"))
    event_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    run_id = uuid.uuid4().hex
    environment.update(SBD_TRAINING_EVENTS_DIR=str(event_root), SBD_TRAINING_PARENT_ID=run_id)
    record = {"run_id": run_id, "name": data["name"], "work_dir": data["cwd"],
              "job_id": identifier, "node": environment.get("SLURMD_NODENAME", ""),
              "state": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat(),
              "output": str(output), "error": str(error)}
    private_json(path.parent / "training_status.json", record)
    private_json(event_root / (run_id + ".json"), record)
    print(f"Slurm {identifier}: starting training; logs={output}, {error}", flush=True)
    def copy_stream(source, target, log):
        log_ok, console_ok = True, True
        while chunk := os.read(source.fileno(), 65536):
            if log_ok:
                try:
                    log.write(chunk)
                    log.flush()
                except OSError as error:
                    log_ok = False
                    print(f"Cannot write training log {log.name}: {error}; continuing to drain output.", file=sys.stderr, flush=True)
            if console_ok:
                try:
                    target.buffer.write(chunk)
                    target.buffer.flush()
                except OSError:
                    console_ok = False
    code = 1
    try:
        with ExitStack() as files:
            logs = [files.enter_context(target.open("ab", buffering=0)) for target in (output, error)]
            process = subprocess.Popen(["bash", *data["command"]], cwd=data["cwd"], env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            threads = [threading.Thread(target=copy_stream, args=args, daemon=True) for args in (
                (process.stdout, sys.stdout, logs[0]), (process.stderr, sys.stderr, logs[1]))]
            for thread in threads:
                thread.start()
            try:
                code = process.wait()
            except KeyboardInterrupt:
                process.wait()
                code = 130
            for thread in threads:
                thread.join()
    finally:
        record.update(state="COMPLETED" if code == 0 else "FAILED", exit_code=code,
                      ended_at=datetime.now(timezone.utc).isoformat())
        private_json(path.parent / "training_status.json", record)
        private_json(event_root / (run_id + ".json"), record)
    print(f"Training exited ({code}); Slurm allocation {identifier} remains available.", flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("submit", "reserve", "submit-existing", "connect", "run"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.mode == "submit":
            code = submit(args.arguments)
        elif args.mode == "reserve":
            code = submit(args.arguments, reserve_only=True)
        elif args.mode == "submit-existing":
            if not args.arguments:
                raise ValueError("srun_existing requires srun_job_id.")
            code = submit(args.arguments[1:], existing_job_id=args.arguments[0])
        else:
            path = Path(args.arguments[0])
            code = connect(path) if args.mode == "connect" else run_training(path)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, f"srun session: {error}\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
