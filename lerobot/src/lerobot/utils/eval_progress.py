"""Publish the non-secret evaluation plan for RunPod and Slurm desks."""

import json
import logging
import os
import re
from pathlib import Path
import tempfile
import time


def plan(environment, execution_id):
    models = json.loads(environment["MODELS_JSON"])
    groups = json.loads(environment["EVAL_GPU_GROUPS_JSON"])
    work = Path(environment["STAGE1_EVAL_WORK_DIR"])
    return {
        "schema": 1,
        "job_id": (environment.get("SBD_LOCAL_JOB_ID") or environment.get("SBD_TRAINING_PARENT_ID")
                   or environment.get("SLURM_ARRAY_JOB_ID") or environment["SLURM_JOB_ID"]),
        "output": environment["EVAL_OUT_DIR"],
        "suite": environment["TARGET_TASK"],
        "episodes_per_task": int(environment["N_EPISODES"]),
        "batch_size": int(environment["EVAL_BATCH_SIZE"]),
        "episode_records": environment.get("SBD_EVAL_EPISODE_PROGRESS") == "1",
        "panels": [{"index": i, "label": model["label"]} for i, model in enumerate(models)],
        "workers": [
            {"tag": worker["tag"], "tasks": worker["items"],
             "panels": worker.get("panels", list(range(len(models)))),
             "log": str(work / "logs" / f"{execution_id}_{group_index}_{worker['tag']}.err")}
            for group_index, group in enumerate(groups) for worker in group
        ],
    }


def publish(environment, execution_id):
    value = plan(environment, execution_id)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value["job_id"]):
        raise ValueError("Invalid evaluation progress identity")
    if environment.get("SBD_LOCAL_JOB_ID"):
        folder = Path(environment.get("SBD_LOCAL_JOBS_DIR", ".cache/local_jobs"))
    else:
        folder = Path(environment["PROJECT_ROOT"]) / ".cache/eval_progress"
    folder.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".eval-plan-", dir=folder)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream)
        Path(name).replace(folder / (value["job_id"] + ".eval.json"))
    finally:
        Path(name).unlink(missing_ok=True)


def record_task(directory, successes, total):
    """Publish completed outcomes only, without depending on tqdm or video output."""
    if directory is None:
        return
    if total < 1 or len(successes) > total or not all(type(item) is bool for item in successes):
        raise ValueError("Invalid completed evaluation episodes")
    folder = Path(directory)
    name = None
    try:
        folder.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".episode-progress-", dir=folder)
        with os.fdopen(fd, "w") as stream:
            json.dump({"schema": 1, "successes": successes, "total_episodes": total,
                       "updated_at": time.time()}, stream)
        Path(name).replace(folder / "eval_progress.json")
    except OSError as error:
        # Reporting never changes the rollout or its success calculation.
        logging.getLogger(__name__).warning("Could not publish evaluation progress: %s", error)
    finally:
        if name:
            try:
                Path(name).unlink(missing_ok=True)
            except OSError:
                pass


if __name__ == "__main__":
    import sys
    publish(os.environ, sys.argv[1])
