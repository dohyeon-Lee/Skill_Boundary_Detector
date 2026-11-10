"""Publish the non-secret evaluation plan for the source-server RunPod Desk."""

import json
import os
from pathlib import Path
import tempfile


def plan(environment, execution_id):
    models = json.loads(environment["MODELS_JSON"])
    groups = json.loads(environment["EVAL_GPU_GROUPS_JSON"])
    work = Path(environment["STAGE1_EVAL_WORK_DIR"])
    return {
        "schema": 1,
        "job_id": environment["SBD_LOCAL_JOB_ID"],
        "output": environment["EVAL_OUT_DIR"],
        "suite": environment["TARGET_TASK"],
        "episodes_per_task": int(environment["N_EPISODES"]),
        "batch_size": int(environment["EVAL_BATCH_SIZE"]),
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
    folder = Path(environment.get("SBD_LOCAL_JOBS_DIR", ".cache/local_jobs"))
    folder.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".eval-plan-", dir=folder)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream)
        Path(name).replace(folder / (value["job_id"] + ".eval.json"))
    finally:
        Path(name).unlink(missing_ok=True)


if __name__ == "__main__":
    import sys
    publish(os.environ, sys.argv[1])
