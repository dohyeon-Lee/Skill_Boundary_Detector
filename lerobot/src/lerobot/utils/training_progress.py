"""Small, run-scoped live counters, independent of loss logging/checkpoint saves."""
import json
import logging
import os
from pathlib import Path
import time

_last = {}


def publish_training_progress(output, step, total):
    status = os.environ.get("SBD_TRAINING_STATUS_PATH")
    run_id = os.environ.get("SBD_TRAINING_PARENT_ID")
    if not status or not run_id:
        return
    now = time.monotonic()
    if step > 5 and step != total and now - _last.get(status, -10) < 2:
        return
    events = os.environ.get("SBD_TRAINING_EVENTS_DIR")
    path = (Path(events) / "progress" / f"{run_id}.json" if events
            else Path(status).with_name("training_progress.json"))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps({"run_id": run_id, "output": str(Path(output).absolute()),
                                         "step": step, "total": total, "updated_at": time.time()}))
        temporary.replace(path)
        _last[status] = now
    except OSError as error:
        logging.warning("Could not publish training progress: %s", error)
        temporary.unlink(missing_ok=True)
