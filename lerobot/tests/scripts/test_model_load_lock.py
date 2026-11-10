"""Check actual process exclusion and parallel evaluation without CUDA."""

import fcntl
import multiprocessing
import os

import pytest

from lerobot.utils.model_load_lock import serialized_model_load


def _worker(path, attempting, loaded, release_load, evaluating, release_eval):
    os.environ["LEROBOT_MODEL_LOAD_LOCK"] = str(path)
    attempting.set()
    with serialized_model_load():
        loaded.set()
        if not release_load.wait(10):
            raise TimeoutError("test never released model loading")
    evaluating.set()
    if not release_eval.wait(10):
        raise TimeoutError("test never released evaluation")


def _start(ctx, path):
    signals = [ctx.Event() for _ in range(5)]
    process = ctx.Process(target=_worker, args=(path, *signals))
    process.start()
    assert signals[0].wait(5)
    return process, signals


def _cleanup(*processes):
    for process in processes:
        if process.is_alive():
            process.kill()
        process.join(5)


def test_loads_serialize_but_evaluations_overlap(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    path = tmp_path / "gpu0.lock"
    first, a = _start(ctx, path)
    second = None
    try:
        assert a[1].wait(5)
        second, b = _start(ctx, path)
        assert not b[1].wait(0.15), "two workers loaded simultaneously"
        a[2].set()
        assert a[3].wait(5)
        assert b[1].wait(5), "loading waited for the previous evaluation to finish"
        b[2].set()
        assert b[3].wait(5)
        assert first.is_alive() and second.is_alive()
        a[4].set()
        b[4].set()
        first.join(5)
        second.join(5)
        assert first.exitcode == second.exitcode == 0
    finally:
        _cleanup(*[p for p in (first, second) if p is not None])


def test_killed_loader_does_not_block_other_workers(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    path = tmp_path / "gpu0.lock"
    first, a = _start(ctx, path)
    second = None
    try:
        assert a[1].wait(5)
        second, b = _start(ctx, path)
        assert not b[1].wait(0.15)
        first.kill()
        first.join(5)
        assert b[1].wait(5), "dead worker left its lock held"
        b[2].set()
        b[4].set()
        second.join(5)
        assert second.exitcode == 0
    finally:
        _cleanup(*[p for p in (first, second) if p is not None])


def test_exception_releases_lock_and_checkpoint_load_can_reenter(tmp_path, monkeypatch):
    path = tmp_path / "gpu0.lock"
    monkeypatch.setenv("LEROBOT_MODEL_LOAD_LOCK", str(path))
    with pytest.raises(RuntimeError, match="bad checkpoint"):
        with serialized_model_load():
            raise RuntimeError("bad checkpoint")
    with path.open("a") as probe:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(probe, fcntl.LOCK_UN)
    for _ in range(2):
        with serialized_model_load():
            pass


def test_direct_run_needs_no_lock_configuration(monkeypatch):
    monkeypatch.delenv("LEROBOT_MODEL_LOAD_LOCK", raising=False)
    with serialized_model_load():
        pass
