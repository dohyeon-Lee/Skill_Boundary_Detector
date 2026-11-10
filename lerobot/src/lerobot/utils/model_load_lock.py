"""Serialize temporary model-loading allocations without serializing rollouts."""

from contextlib import contextmanager
import fcntl
import gc
import logging
import os


@contextmanager
def serialized_model_load():
    """Use the launcher's node-local lock, shared only by this GPU's workers.

    Closing the descriptor also releases the lock if a worker raises or dies.
    Without a launcher-provided path, direct/single-worker runs stay unlocked.
    """
    path = os.environ.get("LEROBOT_MODEL_LOAD_LOCK")
    if not path:
        yield
        return
    log = logging.getLogger(__name__)
    with open(path, "a") as lock:
        log.info("Waiting for model loading slot: %s", path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            log.info("Loading model; peer workers may continue evaluating")
            yield
        finally:
            # Release temporary checkpoint/constructor cycles before the next loader.
            gc.collect()
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

