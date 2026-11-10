from pathlib import Path
from types import SimpleNamespace
import base64
import json
import subprocess
import sys

import huggingface_hub
import pytest

from tools.runpod import finish_eval as finish


@pytest.mark.parametrize("complete", [False, True])
def test_boundary_backup_inventory_requires_complete_reports(tmp_path, complete):
    out = tmp_path / "lerobot/examples/libero/configs/train_skills/skill_eval/skill_boundary_detection/outputs/run"
    out.mkdir(parents=True)
    jobs = tmp_path / ".cache/local_jobs"
    jobs.mkdir(parents=True)
    (jobs / "123.json").write_text(json.dumps({"eval_output": str(out)}))
    (jobs / "123.exit").write_text("0")
    (out / "evaluation_complete.json").write_text(json.dumps({"kind": "skill_boundary_detection"}))
    for name in ("summary.json", "threshold_statistics.json"):
        (out / name).write_text("{}")
    if complete:
        (out / "index.html").write_text("report")
    args = base64.b64encode(json.dumps({"root": str(tmp_path), "job": "123", "hashes": True}).encode()).decode()
    process = subprocess.run([sys.executable, "-c", finish.PROBE, args], capture_output=True, text=True)
    if complete:
        assert process.returncode == 0, process.stderr
        inventory = json.loads(process.stdout)["files"]
        assert set(inventory) == {"index.html", "summary.json", "threshold_statistics.json", "evaluation_complete.json"}
    else:
        assert process.returncode != 0
        assert "Boundary reports incomplete" in process.stderr


def test_changed_result_and_symlink_are_rejected(tmp_path):
    file = tmp_path / "metric.json"
    file.write_text('{"success": 1}')
    inventory = {file.name: finish.hashes(file)}
    finish.validate_local(tmp_path, inventory)
    file.write_text('{"success": 0}')
    with pytest.raises(ValueError, match="differs"):
        finish.validate_local(tmp_path, inventory)
    file.unlink()
    file.symlink_to(__file__)
    with pytest.raises(ValueError, match="symlink"):
        finish.validate_local(tmp_path, inventory)


@pytest.mark.parametrize("failure", [None, "copy", "private", "publish", "verify", "changed_after_upload"])
def test_stop_requires_complete_verified_backup(tmp_path, monkeypatch, failure):
    events = []
    monkeypatch.setattr(finish, "ROOT", tmp_path)
    inventory = {"metrics.json": {"size": 1, "sha256": "sha", "git_sha1": "git"}}
    result = {"exit": "0", "active_jobs": [], "files": inventory,
              "output": "/workspace/project/PT/eval/outputs/run"}
    calls = 0

    def probe(*args, **kwargs):
        nonlocal calls
        calls += 1
        if failure == "changed_after_upload" and calls == 3:
            return {**result, "active_jobs": ["another_job"]}
        return result

    monkeypatch.setattr(finish, "probe", probe)
    monkeypatch.setattr(finish, "ssh_command", lambda *a: ["ssh", "root@example"])

    class Client:
        def pod(self, identifier):
            return {"id": identifier, "status": "RUNNING"}

        def stop(self, identifier):
            events.append("stop")

    monkeypatch.setattr(finish, "RunPodAPI", Client)
    monkeypatch.setattr(finish.subprocess, "run", lambda *a, **k: events.append("copy"))

    def validate(*args):
        events.append("validate")
        if failure == "copy":
            raise ValueError("Bad copy")

    monkeypatch.setattr(finish, "validate_local", validate)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(
        repo_info=lambda *a, **k: SimpleNamespace(private=failure == "private")))

    class Desk:
        checkpoint_repo = "public/results"
        def __init__(self, *a):
            pass

        def publish(self, path, **kwargs):
            assert kwargs == {"eval_results": True, "runpod_id": "pod123"}
            events.append("publish")
            if failure == "publish":
                raise ValueError("Upload failed")

    monkeypatch.setattr(finish, "HFDesk", Desk)

    def verify(*a, **k):
        events.append("verify")
        if failure == "verify":
            raise ValueError("HF hash mismatch")
        return "verified-revision"

    monkeypatch.setattr(finish, "verify_hf", verify)

    def restore(*a, **k):
        events.append("restore")
        if failure == "restore":
            raise ValueError("HF restore failed")
        return tmp_path / "outputs/run"

    monkeypatch.setattr(finish, "restore_from_hf", restore)

    class EndIteration(Exception):
        pass

    def stop_loop(*a):
        raise EndIteration

    monkeypatch.setattr(finish.time, "sleep", stop_loop)
    args = SimpleNamespace(pod="pod123", job="123", repo="public/results", interval=30,
                           identity=Path("key"), desk_url="http://127.0.0.1:8765/hf",
                           remote_root="/workspace/project", restore_to_project=True)
    with pytest.raises(EndIteration):
        finish.run(args)
    if failure:
        assert "stop" not in events
        if failure == "private":
            assert "publish" not in events
    else:
        assert events == ["copy", "validate", "publish", "verify", "stop"]


@pytest.mark.parametrize("failure", [False, True])
def test_verified_results_restore_after_pod_stops_so_download_does_not_extend_gpu_billing(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(finish, "ROOT", tmp_path)
    state_dir = tmp_path / ".cache/runpod_finish"
    state_dir.mkdir(parents=True)
    state = {"pod": "pod123", "job": "123", "repo": "public/results", "verified_revision": "pinned",
             "hf_path": "prefix", "files": {"metric.json": {}}, "phase": "stopping"}
    (state_dir / "pod123.json").write_text(json.dumps(state))
    events = []

    class Client:
        def pod(self, identifier):
            events.append("confirmed_stopped")
            return {"id": identifier, "status": "EXITED"}

    monkeypatch.setattr(finish, "RunPodAPI", Client)

    def restore(*args):
        assert args[:4] == ("public/results", "prefix", "pinned", state["files"])
        events.append("restore")
        if failure:
            raise ValueError("Temporary restore failure")
        return tmp_path / "outputs/result"

    monkeypatch.setattr(finish, "restore_from_hf", restore)
    args = SimpleNamespace(pod="pod123", job="123", repo="public/results", interval=30, restore_to_project=True)
    if failure:
        monkeypatch.setattr(finish.time, "sleep", lambda _: (_ for _ in ()).throw(InterruptedError()))
        with pytest.raises(InterruptedError):
            finish.run(args)
    else:
        finish.run(args)
        assert json.loads((state_dir / "pod123.json").read_text())["local_result"] == str(tmp_path / "outputs/result")
    assert events == ["confirmed_stopped", "restore"]


@pytest.mark.parametrize("corrupt", [False, True])
def test_hf_restore_pins_revision_and_publishes_only_verified_outputs(tmp_path, monkeypatch, corrupt):
    monkeypatch.setattr(finish, "ROOT", tmp_path)
    prefix = "lerobot/examples/libero/configs/train_skills/skill_eval/skill_boundary_detection/outputs/run"
    source = tmp_path / "metric.json"
    source.write_bytes(b"verified")
    inventory = {source.name: finish.hashes(source)}
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        target = kwargs["local_dir"] / prefix
        target.mkdir(parents=True)
        (target / source.name).write_bytes(b"changed" if corrupt else b"verified")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    if corrupt:
        with pytest.raises(ValueError, match="differs"):
            finish.restore_from_hf("public/results", prefix, "pinned", inventory, tmp_path / "cache")
        assert not (tmp_path / prefix).exists()
    else:
        restored = finish.restore_from_hf("public/results", prefix, "pinned", inventory, tmp_path / "cache")
        finish.validate_local(restored, inventory)
    assert calls[0]["revision"] == "pinned"
    assert calls[0]["allow_patterns"] == [prefix + "/*"]


def test_hf_archive_backup_verifies_original_inventory_and_restores_from_pinned_revision(tmp_path, monkeypatch):
    from tools.server_sync import eval_archive
    monkeypatch.setattr(finish, "ROOT", tmp_path)
    original = tmp_path / "original"
    original.mkdir()
    (original / "metric.json").write_text('{"success": 1}')
    archive_dir = tmp_path / "archive"
    manifest = eval_archive.pack(original, archive_dir)
    prefix = "lerobot/examples/libero/configs/train_skills/skill_eval/skill_boundary_detection/outputs/run"

    class Hub:
        def repo_info(self, *a, **kw):
            return SimpleNamespace(private=False, sha="pinned")

        def list_repo_tree(self, *a, **kw):
            assert kw["revision"] == "pinned"
            return [SimpleNamespace(path=prefix + "/" + name, size=(archive_dir / name).stat().st_size,
                                    blob_id=eval_archive.file_hashes(archive_dir / name)["git_sha1"], lfs=None)
                    for name in (eval_archive.ARCHIVE, eval_archive.MANIFEST)]

    monkeypatch.setattr(huggingface_hub, "HfApi", Hub)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: archive_dir / eval_archive.MANIFEST)
    assert finish.verify_hf("public/results", prefix, manifest["files"]) == "pinned"

    def download(**kw):
        import shutil
        assert kw["revision"] == "pinned"
        target = kw["local_dir"] / prefix
        target.mkdir(parents=True)
        for name in (eval_archive.ARCHIVE, eval_archive.MANIFEST):
            shutil.copyfile(archive_dir / name, target / name)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    restored = finish.restore_from_hf("public/results", prefix, "pinned", manifest["files"], tmp_path / "cache")
    finish.validate_local(restored, manifest["files"])
    with pytest.raises(ValueError, match="inventory differs"):
        finish.verify_hf("public/results", prefix, {})


@pytest.mark.parametrize("name,kind", [("../escape", "file"), ("metric.json", "symlink")])
def test_result_archive_rejects_unsafe_members(tmp_path, name, kind):
    import io
    import tarfile
    from tools.server_sync.eval_archive import unpack
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        entry = tarfile.TarInfo(name)
        if kind == "symlink":
            entry.type = tarfile.SYMTYPE
            entry.linkname = "/etc/passwd"
            stream.addfile(entry)
        else:
            entry.size = 1
            stream.addfile(entry, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        unpack(archive, tmp_path / "output", {"metric.json": {"size": 1}})
    assert not (tmp_path / "escape").exists()
