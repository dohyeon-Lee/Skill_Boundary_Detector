"""Stop specified live jobs only after their target checkpoint has finished saving."""
from __future__ import annotations

import argparse
import fcntl
import getpass
import json
import logging
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "lerobot/src")]
from tools.slurm_monitor.slurm import Slurm
from lerobot.utils.checkpoint_retention import _is_training_checkpoint


def checkpoint_ready(job, validate=_is_training_checkpoint):
    root = Path(job["run_dir"]) / "checkpoints"
    target = root / f"{job['step']:06d}"
    last = root / "last"
    # Native training updates `last` AFTER model, processors, optimizer, RNG
    # and scheduler saves return. Directory existence alone is insufficient.
    try:
        if not last.is_symlink():
            return False
        completed = last.resolve(strict=True)
        if completed.parent != root.resolve() or not completed.name.isdigit():
            return False
        if int(completed.name) < job["step"]:
            return False
        return validate(target)
    except OSError:
        return False


def tick(data, slurm, ready=checkpoint_ready):
    active = [job for job in data["jobs"] if job.get("status", "waiting") in {"waiting", "cancel_requested"}]
    if not active:
        return False
    rows = slurm.run(["squeue", "--user", data["user"],
                      "--array", "--noheader", "--format=%i|%u|%T|%V"])
    queue = {parts[0]: parts for line in rows.splitlines() if len(parts := line.strip().split("|")) == 4}
    for job in active:
        row = queue.get(job["job_id"])
        if row is None:
            job["status"] = "stopped_after_checkpoint" if job.get("status") == "cancel_requested" else "left_queue"
            logging.info("Job %s: %s", job["job_id"], job["status"])
        elif row[1] != data["user"] or row[3] != job["submitted_at"]:
            job["status"] = "identity_changed_no_action"
            logging.error("Job %s identity changed; refusing cancellation", job["job_id"])
        elif row[2] == "RUNNING" and job.get("status", "waiting") == "waiting":
            try:
                if ready(job):
                    # Reuses Slurm Desk's fresh owner/SubmitTime verification.
                    slurm.cancel(job)
                    job["status"] = "cancel_requested"
                    job["checkpoint"] = str(Path(job["run_dir"]) / "checkpoints" / f"{job['step']:06d}")
                    job["cancel_requested_at"] = time.time()
                    job.pop("last_error", None)
                    logging.info("Job %s: verified step %d checkpoint; scancel requested", job["job_id"], job["step"])
            except (OSError, RuntimeError, ValueError) as error:
                job["last_error"] = str(error)
                logging.exception("Job %s: check/cancel failed; will retry", job["job_id"])
    return any(job.get("status", "waiting") in {"waiting", "cancel_requested"} for job in data["jobs"])


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=15)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with args.state.with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = json.loads(args.state.read_text())
        if data["user"] != getpass.getuser():
            raise ValueError("Only the current user's jobs can be monitored")
        for job in data["jobs"]:
            if not job["job_id"].isdigit() or type(job["step"]) is not int or job["step"] <= 0:
                raise ValueError("Each job needs a numeric job ID and positive target step")
        slurm = Slurm(data["user"])
        logging.info("Armed: %s; interval=%ss", [(j["job_id"], j["step"]) for j in data["jobs"]], args.interval)
        while True:
            try:
                pending = tick(data, slurm)
                data["last_checked_at"] = time.time()
                data.pop("last_error", None)
                save(args.state, data)
                if not pending:
                    logging.info("All watched jobs finished; exiting")
                    return
            except (OSError, RuntimeError, ValueError) as error:
                data["last_error"] = str(error)
                save(args.state, data)
                logging.exception("Monitor check failed; no unverified cancellation")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
