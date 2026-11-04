import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

PATH = Path(__file__).resolve().parents[1] / "stop_after_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("stop_checkpoint_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def job(tmp_path, identifier="68149"):
    return {"job_id": identifier, "submitted_at": "2026-11-03T00:58:10",
            "run_dir": str(tmp_path / identifier), "step": 100000}


@pytest.mark.parametrize("last,valid,expected", [(90000, True, False), (100000, False, False),
                                                 (100000, True, True), (110000, True, True)])
def test_partial_checkpoint_never_triggers_stop(tmp_path, last, valid, expected):
    spec = job(tmp_path)
    root = Path(spec["run_dir"]) / "checkpoints"
    (root / "100000").mkdir(parents=True)
    (root / f"{last:06d}").mkdir(exist_ok=True)
    (root / "last").symlink_to(f"{last:06d}")
    validate = Mock(return_value=valid)
    assert MODULE.checkpoint_ready(spec, validate) is expected
    if last < 100000:
        validate.assert_not_called()
    else:
        validate.assert_called_once_with(root / "100000")


def test_only_ready_job_is_cancelled_and_request_is_not_repeated(tmp_path):
    first, second = job(tmp_path), job(tmp_path, "68341")
    data = {"user": "dohyeon", "jobs": [first, second]}
    queue = "\n".join(f"{j['job_id']}|dohyeon|RUNNING|{j['submitted_at']}" for j in data["jobs"])
    slurm = SimpleNamespace(run=Mock(return_value=queue), cancel=Mock())
    ready = lambda j: j["job_id"] == "68149"
    assert MODULE.tick(data, slurm, ready)
    slurm.cancel.assert_called_once_with(first)
    assert first["status"] == "cancel_requested" and "status" not in second
    MODULE.tick(data, slurm, ready)
    slurm.cancel.assert_called_once()
    slurm.run.return_value = ""
    assert not MODULE.tick(data, slurm, ready)
    assert first["status"] == "stopped_after_checkpoint" and second["status"] == "left_queue"


@pytest.mark.parametrize("owner,submitted,state", [("other", "old", "RUNNING"),
                                                    ("dohyeon", "new", "RUNNING"),
                                                    ("dohyeon", "old", "PENDING")])
def test_changed_identity_or_nonrunning_job_cannot_be_cancelled(tmp_path, owner, submitted, state):
    spec = job(tmp_path)
    spec["submitted_at"] = "old"
    slurm = SimpleNamespace(run=Mock(return_value=f"68149|{owner}|{state}|{submitted}"), cancel=Mock())
    MODULE.tick({"user": "dohyeon", "jobs": [spec]}, slurm, lambda j: True)
    slurm.cancel.assert_not_called()
