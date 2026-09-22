"""hf_checkpoints.py: finding finished checkpoints, Hub cleanup planning and the watch loop (fake Hub)."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

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
