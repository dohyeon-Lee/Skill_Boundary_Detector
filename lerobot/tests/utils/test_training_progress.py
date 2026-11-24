import json
from pathlib import Path

from lerobot.utils.training_progress import publish_training_progress


def test_live_counter_is_scoped_to_run_and_not_loss_logging(tmp_path, monkeypatch):
    status = tmp_path / "training_status.json"
    monkeypatch.setenv("SBD_TRAINING_STATUS_PATH", str(status))
    monkeypatch.setenv("SBD_TRAINING_PARENT_ID", "current")
    monkeypatch.delenv("SBD_TRAINING_EVENTS_DIR", raising=False)
    output = tmp_path / "output"
    publish_training_progress(output, 1, 20000)
    data = json.loads(status.with_name("training_progress.json").read_text())
    assert data["step"] == 1 and data["total"] == 20000
    assert data["run_id"] == "current" and data["output"] == str(output)
    publish_training_progress(output, 20000, 20000)
    assert json.loads(status.with_name("training_progress.json").read_text())["step"] == 20000


def test_unmanaged_training_does_not_write_files(tmp_path, monkeypatch):
    monkeypatch.delenv("SBD_TRAINING_STATUS_PATH", raising=False)
    publish_training_progress(tmp_path, 1, 20)
    assert not list(tmp_path.iterdir())


def test_event_counters_are_separate_per_run(tmp_path, monkeypatch):
    monkeypatch.setenv("SBD_TRAINING_STATUS_PATH", str(tmp_path / "session/status.json"))
    monkeypatch.setenv("SBD_TRAINING_EVENTS_DIR", str(tmp_path / "events"))
    for run in ("one", "two"):
        monkeypatch.setenv("SBD_TRAINING_PARENT_ID", run)
        publish_training_progress(tmp_path / run, 1, 20)
    assert json.loads((tmp_path / "events/progress/one.json").read_text())["run_id"] == "one"
    assert json.loads((tmp_path / "events/progress/two.json").read_text())["run_id"] == "two"
