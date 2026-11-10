from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

from tools.runpod import finish_eval as finish


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

    class EndIteration(Exception):
        pass

    def stop_loop(*a):
        raise EndIteration

    monkeypatch.setattr(finish.time, "sleep", stop_loop)
    args = SimpleNamespace(pod="pod123", job="123", repo="public/results", interval=30,
                           identity=Path("key"), desk_url="http://127.0.0.1:8765/hf",
                           remote_root="/workspace/project")
    with pytest.raises(EndIteration):
        finish.run(args)
    if failure:
        assert "stop" not in events
        if failure == "private":
            assert "publish" not in events
    else:
        assert events == ["copy", "validate", "publish", "verify", "stop"]
