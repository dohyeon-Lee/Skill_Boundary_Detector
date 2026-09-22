"""hf_checkpoints.py: finding finished checkpoints, Hub cleanup planning and the watch loop (fake Hub)."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

import pytest

CK = runpy.run_path(str(Path(__file__).resolve().parents[2] / "examples/libero/configs/src/hf_checkpoints.py"))


def _run(root: Path, rel: str, steps: list[str], last: str | None) -> Path:
    run = root / rel
    for step in steps:
        (run / "checkpoints" / step / "pretrained_model").mkdir(parents=True)
        (run / "checkpoints" / step / "pretrained_model" / "model.safetensors").write_bytes(b"w")
        (run / "checkpoints" / step / "training_state").mkdir()
    if last:
        os.symlink(last, run / "checkpoints" / "last")
    return run


def _tree(tmp_path: Path) -> Path:
    project = tmp_path / "repo"
    _run(project / "outputs_filtered", "pi05_PT/run_a", ["010000", "020000", "030000"], last="020000")  # 030000 still saving
    _run(project / "outputs_filtered", "skillVLA_stage1/VSA/run_b", ["005000"], last="005000")
    _run(project / "outputs_calvin", "pi05_PT/run_c", ["001000"], last="001000")
    _run(tmp_path / "volume", "run_start", ["050000"], last="050000")
    os.symlink(tmp_path / "volume/run_start", project / "outputs_filtered/pi05_PT/run_start")      # start checkpoint link
    (project / "outputs_filtered/pi05_PT/run_a/wandb").mkdir()
    (project / "models").mkdir()
    return project


def test_only_complete_steps_of_our_own_runs_are_found(tmp_path: Path) -> None:
    project = _tree(tmp_path)
    assert CK["output_roots"](project) == ["outputs_calvin", "outputs_filtered"]
    found = CK["local_steps"](project, CK["output_roots"](project))
    assert sorted(found) == [
        "outputs_calvin/pi05_PT/run_c/checkpoints/001000",
        "outputs_filtered/pi05_PT/run_a/checkpoints/010000",
        "outputs_filtered/pi05_PT/run_a/checkpoints/020000",
        "outputs_filtered/skillVLA_stage1/VSA/run_b/checkpoints/005000",
    ]
    assert list(CK["local_steps"](project, ["outputs_calvin"])) == ["outputs_calvin/pi05_PT/run_c/checkpoints/001000"]


def test_cleanup_keeps_the_latest_n_and_protected_steps() -> None:
    run = "outputs_filtered/pi05_PT/run_a/checkpoints/"
    steps = {run + step for step in ("010000", "020000", "050000", "060000", "070000", "100000", "110000")}
    doomed = CK["plan_cleanup"](steps, keep=2, protect={50000, 100000})
    assert doomed == [run + "010000", run + "020000", run + "060000", run + "070000"]
    assert CK["plan_cleanup"](steps, keep=0, protect=set()) == []


class FakeHub:
    def __init__(self, files=()):
        self.files = set(files)
        self.calls = []

    def create_repo(self, repo, **kwargs):
        self.calls.append(("create_repo", repo, kwargs.get("private")))

    def list_repo_files(self, repo, repo_type):
        return sorted(self.files)

    def upload_folder(self, repo_id, folder_path, repo_type, path_in_repo=None, commit_message=""):
        prefix = f"{path_in_repo}/" if path_in_repo else ""
        for file in Path(folder_path).rglob("*"):
            if file.is_file():
                self.files.add(prefix + file.relative_to(folder_path).as_posix())
        self.calls.append(("upload", path_in_repo or "(card)"))

    def delete_folder(self, path_in_repo, repo_id, repo_type, commit_message=""):
        self.files = {path for path in self.files if not path.startswith(path_in_repo + "/")}
        self.calls.append(("delete", path_in_repo))

    def super_squash_history(self, repo_id, repo_type):
        self.calls.append(("squash",))


def test_watch_uploads_new_steps_once_writes_the_card_and_cleans_up(tmp_path: Path) -> None:
    project = _tree(tmp_path)
    hub = FakeHub({"outputs_filtered/pi05_PT/run_a/checkpoints/010000/pretrained_model/model.safetensors"})
    CK["watch"](project, "me/ckpts", once=True, keep=1, protect=set(), assume_yes=True, api=hub)
    uploads = [call[1] for call in hub.calls if call[0] == "upload"]
    assert uploads[0] == "(card)" and {"README.md", "NOTICE", "LICENSE.md", "LICENSE-DINOv3.md"} <= hub.files
    assert sorted(uploads[1:]) == [
        "outputs_calvin/pi05_PT/run_c/checkpoints/001000",
        "outputs_filtered/pi05_PT/run_a/checkpoints/020000",        # 010000 already there; 030000 incomplete
        "outputs_filtered/skillVLA_stage1/VSA/run_b/checkpoints/005000",
    ]
    assert ("create_repo", "me/ckpts", False) in hub.calls                      # public
    assert [call for call in hub.calls if call[0] == "delete"] == [("delete", "outputs_filtered/pi05_PT/run_a/checkpoints/010000")]
    assert ("squash",) not in hub.calls
    assert not any("wandb" in path or "/last/" in path for path in hub.files)

    hub.calls.clear()
    CK["watch"](project, "me/ckpts", once=True, keep=1, assume_yes=True, api=hub)          # nothing new
    assert [call for call in hub.calls if call[0] in {"upload", "delete"}] == []


def test_watch_dry_run_and_decline_touch_nothing(tmp_path: Path) -> None:
    project = _tree(tmp_path)
    hub = FakeHub()
    CK["watch"](project, "me/ckpts", dry_run=True, keep=1, api=hub)
    CK["watch"](project, "me/ckpts", once=True, api=hub, ask=lambda prompt: "n")
    assert [call for call in hub.calls if call[0] in {"upload", "delete", "squash"}] == [("upload", "(card)")]


def test_steps_deleted_for_age_are_not_uploaded_again_after_a_restart() -> None:
    run = "outputs_filtered/pi05_PT/run_a/checkpoints/"
    local = {run + "010000", run + "020000", run + "030000", run + "050000"}
    remote = {run + "030000", run + "050000"}             # 010000/020000 were deleted earlier by --keep
    upload, delete, prune = CK["plan_sync"](local, remote, keep=1, protect={30000})
    assert upload == [] and delete == []                  # 050000 = latest, 030000 protected
    assert prune == [run + "010000", run + "020000"]      # the local disk keeps the same steps as the Hub
    upload, delete, prune = CK["plan_sync"](local | {run + "060000"}, remote, keep=1, protect={30000})
    assert upload == [run + "060000"] and delete == [run + "050000"]
    assert prune == [run + "010000", run + "020000", run + "050000"]
    assert CK["plan_sync"](local, set(), keep=0, protect=set())[0] == sorted(local)   # keep off: upload all


def test_local_prune_follows_the_same_window_and_never_touches_the_newest_or_linked_steps(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    run_a = _run(project / "outputs_filtered", "pi05_PT/run_a",
                 ["010000", "020000", "050000", "060000", "070000", "080000"], last="070000")   # 080000 still saving
    _run(tmp_path / "volume", "run_start", ["010000"], last="010000")
    os.symlink(tmp_path / "volume/run_start", project / "outputs_filtered/pi05_PT/run_start")
    (project / "models").mkdir()
    steps = lambda: sorted(path.name for path in (run_a / "checkpoints").iterdir() if path.name != "last")  # noqa: E731

    hub = FakeHub()
    CK["watch"](project, "me/ckpts", once=True, keep=2, protect={50000}, prune_local=False, assume_yes=True, api=hub)
    assert steps() == ["010000", "020000", "050000", "060000", "070000", "080000"]      # off: local untouched

    CK["watch"](project, "me/ckpts", once=True, keep=2, protect={50000}, prune_local=True, api=hub,
                ask=lambda prompt: "n")
    assert len(steps()) == 6                                                            # declined: nothing removed

    CK["watch"](project, "me/ckpts", once=True, keep=2, protect={50000}, prune_local=True, assume_yes=True, api=hub)
    assert steps() == ["050000", "060000", "070000", "080000"]      # protected + latest 2 + the one being saved
    assert (tmp_path / "volume/run_start/checkpoints/010000").is_dir()                   # linked start checkpoint kept
    assert (run_a / "checkpoints/last").resolve().name == "070000"


def test_pulled_checkpoints_without_last_are_never_uploaded_or_pruned(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    trained = _run(project / "outputs_filtered", "skillVLA_stage1/VSA/new_run",
                   ["010000", "020000", "030000", "040000"], last="040000")
    pulled = _run(project / "outputs_filtered", "skillVLA_stage1/VSA/pulled_run",
                  ["010000", "020000", "030000", "040000"], last=None)          # hf_sync pull: no `last`
    (project / "models").mkdir()
    hub = FakeHub()
    CK["watch"](project, "me/ckpts", once=True, keep=1, prune_local=True, assume_yes=True, api=hub)
    assert sorted(path.name for path in (trained / "checkpoints").iterdir()) == ["040000", "last"]
    assert sorted(path.name for path in (pulled / "checkpoints").iterdir()) == ["010000", "020000", "030000", "040000"]
    assert not any("pulled_run" in path for path in hub.files)


def test_hub_deletes_only_touch_runs_this_machine_trains() -> None:
    mine = "outputs_filtered/pi05_PT/mine/checkpoints/"
    other = "outputs_filtered/pi05_PT/other_pod/checkpoints/"
    local = {mine + step for step in ("010000", "020000")}
    remote = {mine + "010000"} | {other + step for step in ("010000", "020000", "030000", "040000")}
    upload, delete, prune = CK["plan_sync"](local, remote, keep=1, protect=set())
    assert upload == [mine + "020000"] and delete == [mine + "010000"] and prune == [mine + "010000"]


def test_pull_marks_steps_and_points_last_at_the_newest(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    rel = "outputs_filtered/skillVLA_stage1/VSA/run"
    pulled = _run(project, rel, ["050000", "100000"], last=None)
    CK["mark_pulled"](project, {f"{rel}/checkpoints/050000", f"{rel}/checkpoints/100000"})
    assert os.readlink(pulled / "checkpoints/last") == "100000"              # like LeRobot: relative
    assert (pulled / "checkpoints/.hf_pulled").read_text() == "050000\n100000\n"
    (pulled / "checkpoints/150000").mkdir()                                  # a later pull of a newer step
    CK["mark_pulled"](project, {f"{rel}/checkpoints/150000"})
    assert os.readlink(pulled / "checkpoints/last") == "150000"

    trained = _run(project, "outputs_filtered/pi05_PT/trained", ["030000"], last="030000")
    (trained / "checkpoints/010000").mkdir()
    CK["mark_pulled"](project, {"outputs_filtered/pi05_PT/trained/checkpoints/010000"})
    assert os.readlink(trained / "checkpoints/last") == "030000"             # its own run: untouched


def test_resumed_pulled_run_manages_only_the_new_steps(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    rel = "outputs_filtered/pi05_PT/run"
    run = _run(project, rel, ["050000", "100000"], last=None)
    (project / "models").mkdir()
    CK["mark_pulled"](project, {f"{rel}/checkpoints/050000", f"{rel}/checkpoints/100000"})
    hub = FakeHub({f"{rel}/checkpoints/{step}/pretrained_model/model.safetensors"
                   for step in ("030000", "050000", "100000")})
    CK["watch"](project, "me/ckpts", once=True, keep=1, prune_local=True, assume_yes=True, api=hub)
    assert [call for call in hub.calls if call[0] in {"upload", "delete"}] == [("upload", "(card)")]   # pulled only

    for step in ("110000", "120000"):                                        # training resumed here
        (run / "checkpoints" / step / "pretrained_model").mkdir(parents=True)
        (run / "checkpoints" / step / "pretrained_model/model.safetensors").write_bytes(b"w")
    (run / "checkpoints/last").unlink()
    os.symlink("120000", run / "checkpoints/last")
    hub.calls.clear()
    CK["watch"](project, "me/ckpts", once=True, keep=1, prune_local=True, assume_yes=True, api=hub)
    assert sorted(call[1] for call in hub.calls if call[0] == "upload") == [f"{rel}/checkpoints/120000"]
    assert [call[1] for call in hub.calls if call[0] == "delete"] == [f"{rel}/checkpoints/030000"]
    assert sorted(path.name for path in (run / "checkpoints").iterdir() if path.name[0] != ".") == [
        "050000", "100000", "120000", "last"]                               # 110000 pruned, pulled kept


def test_first_scan_asks_about_hub_deletes_too(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    _run(project / "outputs_filtered", "pi05_PT/run", ["010000", "020000"], last="020000")
    (project / "models").mkdir()
    hub = FakeHub({f"outputs_filtered/pi05_PT/run/checkpoints/{step}/pretrained_model/model.safetensors"
                   for step in ("010000", "020000")})
    questions = []
    CK["watch"](project, "me/ckpts", once=True, keep=1, api=hub, ask=lambda q: questions.append(q) or "n")
    assert len(questions) == 1 and "Hugging Face 체크포인트 1개 지우기" in questions[0]
    assert not [call for call in hub.calls if call[0] == "delete"]


def test_pod_done_after_training_stops_or_reaches_the_target_step() -> None:
    run = "outputs_filtered/pi05_PT/run/checkpoints/"
    found = {run + step: Path(step) for step in ("090000", "100000")}
    stopped = {"recorded": 1, "running": 0, "pending": 0}
    running = {"recorded": 1, "running": 1, "pending": 0}
    done = CK["pod_done"]
    assert done({}, set(), 100000, {"recorded": 0, "running": 0, "pending": 0}) is None       # nothing started
    assert "stopped" in done(found, {run + "100000"}, 100000, stopped)                       # finished
    assert done(found, {run + "090000"}, 100000, stopped) is None                            # newest not uploaded
    assert "stopped" in done({}, set(), 100000, stopped)                                      # failed before a checkpoint
    assert "100000" in done(found, {run + "100000"}, 100000, running)                        # target reached
    assert done(found, {run + "100000"}, 0, running) is None                                  # target check off
    assert done(found, {run + "100000"}, 150000, running) is None                             # not there yet
    assert done(found, {run + "100000"}, 100000, {"recorded": 2, "running": 1, "pending": 1}) is None
    assert "12 h" in done(found, {run + "100000"}, 0, running, elapsed=12 * 3600, done_time=12 * 3600)   # time limit
    assert done(found, {run + "090000"}, 0, running, elapsed=13 * 3600, done_time=12 * 3600) is None  # upload first
    assert done(found, {run + "100000"}, 0, running, elapsed=3600, done_time=12 * 3600) is None


def test_done_time_formats() -> None:
    parse = CK["parse_duration"]
    assert parse("48:00:00") == parse("2-00:00:00") == parse("2d") == parse("48h") == 48 * 3600
    assert parse("90m") == 5400 and parse("1.5h") == 5400 and parse("30s") == 30
    for bad in ("12", "12:00", "0h", "soon"):
        with pytest.raises(ValueError, match="done-time"):
            parse(bad)


def test_done_time_counts_from_the_first_training_job(tmp_path: Path, monkeypatch) -> None:
    project = _finished_pod(tmp_path, monkeypatch)
    (tmp_path / "jobs/7.job").write_text((tmp_path / "jobs/7.job").read_text() + "started=2026-09-20T09:00:00+09:00\n")
    (tmp_path / "jobs/8.job").write_text("name=late\nstarted=2026-09-21T09:00:00+09:00\n")
    assert CK["training_started"](project) == CK["datetime"].fromisoformat("2026-09-20T09:00:00+09:00").timestamp()
    hub, terminated = FakeHub(), []                                     # job 7 still running, past 1 h
    CK["watch"](project, "me/ckpts", interval=0, assume_yes=True, api=hub, done_time=3600, log_repo="me/private",
                jobs_state=lambda: {"recorded": 1, "running": 1, "pending": 0},
                terminate=lambda: terminated.append(True))
    assert terminated == [True]


def test_local_jobs_counts_running_waiting_and_finished(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SBD_LOCAL_JOBS_DIR", str(tmp_path))
    for job_id, pid in (("1", os.getpid()), ("2", os.getpid()), ("3", 2**22 + 12345)):
        (tmp_path / f"{job_id}.job").write_text("name=x\n")
        (tmp_path / f"{job_id}.pid").write_text(f"{pid}\n")
    (tmp_path / "2.pending").touch()
    assert CK["local_jobs"](tmp_path) == {"recorded": 3, "running": 1, "pending": 1}


def _finished_pod(tmp_path: Path, monkeypatch) -> Path:
    project = tmp_path / "repo"
    _run(project / "outputs_filtered", "pi05_PT/run", ["050000", "100000"], last="100000")
    (project / "models").mkdir()
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/pi05_PT_7.out").write_text("step 100000\n")
    (tmp_path / "logs/pi05_PT_7.err").write_text("Traceback ...\n")
    (jobs / "7.job").write_text(f"name=pi05_PT\noutput={tmp_path}/logs/pi05_PT_7.out\nerror={tmp_path}/logs/pi05_PT_7.err\n")
    (jobs / "7.exit").write_text("1\n")
    monkeypatch.setenv("SBD_LOCAL_JOBS_DIR", str(jobs))
    monkeypatch.setenv("RUNPOD_POD_ID", "pod123")
    return project


def test_watch_uploads_logs_then_terminates_on_the_second_scan(tmp_path: Path, monkeypatch) -> None:
    project = _finished_pod(tmp_path, monkeypatch)
    hub, terminated = FakeHub(), []
    CK["watch"](project, "me/ckpts", interval=0, keep=3, assume_yes=True, api=hub, done_step=100000,
                log_repo="me/private", terminate=lambda: terminated.append(True))
    assert terminated == [True]
    logs = sorted(path for path in hub.files if path.startswith("runpod_logs/"))
    assert [path.rsplit("/", 1)[1] for path in logs] == ["jobs.tsv", "pi05_PT_7.err", "pi05_PT_7.out"]
    assert logs[0].split("/")[1].endswith("_pod123")
    assert "outputs_filtered/pi05_PT/run/checkpoints/100000/pretrained_model/model.safetensors" in hub.files


def test_failed_log_uploads_do_not_keep_the_pod_up_forever(tmp_path: Path, monkeypatch) -> None:
    project = _finished_pod(tmp_path, monkeypatch)

    class NoLogs(FakeHub):
        def upload_folder(self, repo_id, folder_path, repo_type, path_in_repo=None, commit_message=""):
            if repo_type == "dataset":
                self.calls.append(("log upload failed",))
                raise OSError("network down")
            super().upload_folder(repo_id, folder_path, repo_type, path_in_repo, commit_message)

    hub, terminated = NoLogs(), []
    CK["watch"](project, "me/ckpts", interval=0, keep=3, assume_yes=True, api=hub, done_step=100000,
                log_repo="me/private", terminate=lambda: terminated.append(True))
    assert terminated == [True] and hub.calls.count(("log upload failed",)) == 3


def test_terminate_needs_a_runpod_pod(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    with pytest.raises(SystemExit, match="RunPod"):
        CK["watch"](tmp_path, "me/ckpts", once=True, api=FakeHub(), done_step=100000)
