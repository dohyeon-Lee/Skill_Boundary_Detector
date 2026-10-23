"""HF UI planning, HTTP boundary, and subprocess lifecycle; no Hub writes or downloads."""

import http.client
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huggingface_hub.hf_api import RepoFile, RepoFolder
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
from httpx import Request, Response
from hf_server import HFBackend, HFHTTPServer, HFTransfers, SCRIPT
from hf_delete import checked_paths, delete_selected, preview_delete


class FakeHub:
    def __init__(self):
        self.calls = []
        self.exists = True

    def list_repo_tree(self, repo, **kwargs):
        self.calls.append((repo, kwargs))
        if not kwargs.get("path_in_repo"):
            return iter([RepoFolder(path="dataset_filtered", oid="a"),
                         RepoFolder(path="outputs_filtered", oid="b")])
        prefix = kwargs["path_in_repo"]
        return (RepoFile(path=f"{prefix}/file-{i:04}.parquet", size=i, oid=str(i)) for i in range(405))

    def get_paths_info(self, repo, paths, **kwargs):
        self.calls.append((repo, paths, kwargs))
        return [RepoFolder(path=paths[0], oid="a")] if self.exists else []


class DeleteHub:
    """An immutable revision plus a movable branch; records writes in memory only."""
    def __init__(self):
        self.revision = "a" * 40
        self.head = self.revision
        self.files = {"dataset_filtered/a/file.txt": 3, "dataset_filtered/a/nested/weight.bin": 5,
                      "dataset_filtered/ab/keep.txt": 7, "outputs_filtered/run/checkpoints/050000/model.bin": 11,
                      "README.md": 13}
        self.commits = []

    def repo_info(self, repo, **kwargs):
        return SimpleNamespace(sha=self.head)

    def get_paths_info(self, repo, paths, **kwargs):
        assert kwargs["revision"] == self.revision
        entries = []
        for path in paths:
            if path in self.files:
                entries.append(RepoFile(path=path, size=self.files[path], oid="a"))
            elif any(file.startswith(path + "/") for file in self.files):
                entries.append(RepoFolder(path=path, oid="a"))
        return entries

    def list_repo_tree(self, repo, **kwargs):
        assert kwargs["revision"] == self.revision
        assert kwargs["recursive"] is True
        return (RepoFile(path=path, size=size, oid="a") for path, size in self.files.items()
                if path.startswith(kwargs["path_in_repo"] + "/"))

    def create_commit(self, **kwargs):
        self.commits.append(kwargs)
        if kwargs["parent_commit"] != self.head:
            raise ValueError("parent commit changed")
        return SimpleNamespace(commit_url="https://huggingface.co/me/data/commit/deleted")


class HFWorkspace(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hf-ui-test-")
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.data = self.project / "dataset_filtered/data"
        self.data.mkdir(parents=True)
        (self.data / "example.txt").write_text("tiny offline fixture")
        self.api = FakeHub()
        self.backend = HFBackend({"project_root": str(self.project), "hf_dataset_repo": "me/data",
                                  "hf_checkpoint_repo": "me/checkpoints", "server": "test"}, api=self.api)

    def plan(self, route="dataset-push", paths=None, options=None, dry_run=False):
        from hf_server import ROUTES
        source, destination, _ = ROUTES[route]
        return self.backend.plan(source, destination, paths or ["dataset_filtered/data"], options, dry_run)

    def wait_done(self, manager):
        until = time.monotonic() + 8
        while time.monotonic() < until:
            jobs = manager.snapshot()
            if jobs and all(job["status"] not in {"queued", "running", "cancelling"} for job in jobs):
                return jobs
            time.sleep(.02)
        self.fail("jobs did not finish")


class HFTest(HFWorkspace):
    def test_browse_roots_and_paginated_hub_iterator(self):
        self.assertEqual([entry["path"] for entry in self.backend.browse("local-datasets")["entries"]], ["dataset_filtered"])
        self.assertEqual([entry["path"] for entry in self.backend.browse("hub-datasets")["entries"]], ["dataset_filtered"])
        self.assertEqual([entry["path"] for entry in self.backend.browse("hub-checkpoints")["entries"]], ["outputs_filtered"])
        page = self.backend.browse("hub-datasets", "dataset_filtered", 200)
        self.assertEqual(len(page["entries"]), 200)
        self.assertEqual(page["entries"][0]["name"], "file-0200.parquet")
        self.assertTrue(page["has_more"])
        last = self.backend.browse("hub-datasets", "dataset_filtered", 400)
        self.assertEqual(len(last["entries"]), 5)
        self.assertFalse(last["has_more"])
        filtered = self.backend.browse("hub-datasets", "dataset_filtered", search="0404")
        self.assertEqual(len(filtered["entries"]), 1)

    def test_missing_hub_ancestors_fall_back_to_root_and_reset_filters(self):
        calls = []

        def tree(repo, **kwargs):
            path = kwargs["path_in_repo"]
            calls.append(path)
            # The SDK can raise only when its iterator is consumed.
            if path:
                raise EntryNotFoundError("Entry Not Found: " + path)
            yield RepoFolder(path="outputs_kept", oid="a")

        with patch.object(self.api, "list_repo_tree", side_effect=tree):
            listing = self.backend.browse("hub-checkpoints", "outputs_filtered/skillVLA_stage1/VSA", 200, "old-search")
        self.assertEqual(calls, ["outputs_filtered/skillVLA_stage1/VSA", "outputs_filtered/skillVLA_stage1", "outputs_filtered", None])
        self.assertEqual(listing["path"], "")
        self.assertEqual(listing["offset"], 0)
        self.assertEqual(listing["recovered_from"], "outputs_filtered/skillVLA_stage1/VSA")
        self.assertEqual([entry["path"] for entry in listing["entries"]], ["outputs_kept"])

    def test_missing_hub_path_stops_at_first_surviving_ancestor(self):
        calls = []

        def tree(repo, **kwargs):
            path = kwargs["path_in_repo"]
            calls.append(path)
            if path != "outputs_filtered":
                raise EntryNotFoundError("Entry Not Found")
            yield RepoFolder(path="outputs_filtered/other_run", oid="a")

        with patch.object(self.api, "list_repo_tree", side_effect=tree):
            listing = self.backend.browse("hub-checkpoints", "outputs_filtered/removed/run")
        self.assertEqual(listing["path"], "outputs_filtered")
        self.assertEqual(len(calls), 3)

    def test_repository_errors_do_not_trigger_parent_fallback(self):
        denied = RepositoryNotFoundError("Repository Not Found", response=Response(401,
                                        request=Request("GET", "https://huggingface.co/api/models/missing")))
        with patch.object(self.api, "list_repo_tree", side_effect=denied) as listing:
            with self.assertRaisesRegex(ValueError, "권한"):
                self.backend.browse("hub-checkpoints", "outputs_filtered/removed/run")
            self.assertEqual(listing.call_count, 1)
        with patch.object(self.api, "list_repo_tree", side_effect=EntryNotFoundError("Entry Not Found")) as listing:
            with self.assertRaisesRegex(ValueError, "루트 목록"):
                self.backend.browse("hub-checkpoints", "")
            self.assertEqual(listing.call_count, 1)

    def test_storage_root_symlinks_work_but_nested_escape_fails(self):
        volume = self.project / "volume"
        (volume / "dataset_linked").mkdir(parents=True)
        (volume / "dataset_linked/file.txt").write_text("data")
        (self.project / "dataset_linked").symlink_to(volume / "dataset_linked", target_is_directory=True)
        self.assertIn("dataset_linked", [item["path"] for item in self.backend.browse("local-datasets")["entries"]])
        self.assertEqual(len(self.backend.browse("local-datasets", "dataset_linked")["entries"]), 1)
        (self.data / "escape").symlink_to(self.project, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.plan(paths=["dataset_filtered/data/escape"])
        for bad in ["../data", "/tmp/file", "models/pi05_base", "dataset_filtered/../escape", "dataset_filtered//data"]:
            with self.assertRaises(ValueError):
                self.plan(paths=[bad])

    def test_plans_collapse_duplicates_and_do_not_launch_or_create(self):
        before = sorted(str(path) for path in self.project.rglob("*"))
        with patch("server.subprocess.Popen") as launch:
            plan = self.plan(paths=["dataset_filtered/data/example.txt", "dataset_filtered/data"])
            self.assertEqual(plan["collapsed"], 1)
            self.assertEqual(plan["items"][0]["destination"], "https://huggingface.co/datasets/me/data/dataset_filtered/data")
            launch.assert_not_called()
        self.assertEqual(before, sorted(str(path) for path in self.project.rglob("*")))
        self.api.exists = False
        with self.assertRaisesRegex(ValueError, "Hub에 항목이 없습니다"):
            self.plan("dataset-pull")

    def test_all_commands_use_existing_cli_and_preserve_paths(self):
        self.backend.server = "runpod"
        for route, paths, expected in [
            ("dataset-push", ["dataset_filtered/data"], ["push", "dataset_filtered/data"]),
            ("dataset-pull", ["dataset_filtered/data"], ["pull", "dataset_filtered/data"]),
            ("models-pull", ["pi05_base"], ["pull", "--models", "pi05_base"]),
            ("checkpoints-pull", ["outputs_filtered/run/checkpoints/050000"], ["pull", "--checkpoints", "outputs_filtered/run/checkpoints/050000"]),
        ]:
            command = self.backend.commands(self.plan(route, paths, dry_run=True)["items"][0])
            self.assertEqual(command[:3], [sys.executable, "-u", str(SCRIPT)])
            self.assertEqual(command[3:3+len(expected)], expected)
            self.assertIn("--dry-run", command)
            self.assertIn("--yes", command)
            self.assertEqual(command[command.index("--server")+1], "runpod")
            self.assertNotIn("HF_TOKEN", " ".join(command))

    def test_watch_defaults_and_explicit_retention_preview(self):
        item = self.plan("watch")["items"][0]
        command = self.backend.commands(item)
        self.assertIn("--no-prune-local", command)
        self.assertNotIn("--done-step", command)
        self.assertNotIn("--done-time", command)
        options = {"keep": 3, "protect": "050000,100000", "interval": 60, "root": "outputs_filtered",
                   "prune_local": True, "squash": True, "once": False}
        plan = self.plan("watch", options=options, dry_run=True)
        self.assertIn("로컬", plan["notice"])
        self.assertIn("되돌릴 수 없습니다", plan["notice"])
        command = self.backend.commands(plan["items"][0])
        self.assertIn("--prune-local", command)
        self.assertIn("--once", command)
        for invalid in [{"interval": 0}, {"keep": -1}, {"keep": True}, {"protect": "100,boom"},
                        {"root": "outputs_filtered/run"}, {"root": "dataset_filtered"}, {"prune_local": "false"}]:
            with self.assertRaises(ValueError):
                self.plan("watch", options=invalid)

    def test_dry_run_executes_existing_cli_without_transferring(self):
        manager = HFTransfers(self.backend)
        self.addCleanup(manager.close)
        # The real CLI uses this tiny isolated config, without contacting the Hub.
        config_dir = self.project / "loader"
        config_dir.mkdir()
        loader = config_dir / "global_config_loader.py"
        loader.write_text("def load_global_config(*args): return " + repr({"project_root": str(self.project), "server": "test"}))
        script = config_dir / "hf_sync.py"
        script.write_text(SCRIPT.read_text())
        (config_dir / "global_config.yaml").write_text("{}")
        with patch("hf_server.SCRIPT", script):
            manager.submit(self.plan(dry_run=True)["items"])
            job = self.wait_done(manager)[0]
        self.assertEqual(job["status"], "completed", job)
        self.assertIn("[dry-run] upload_folder", "\n".join(job["logs"]))
        self.assertFalse((self.project / ".cache").exists())

    def test_worker_failure_and_cancel_are_not_success(self):
        manager = HFTransfers(self.backend, concurrency=1)
        self.addCleanup(manager.close)
        with patch.object(self.backend, "commands", return_value=[sys.executable, "-c", "raise SystemExit(23)"]):
            manager.submit(self.plan()["items"])
            failed = self.wait_done(manager)[0]
        self.assertEqual(failed["status"], "failed")
        self.assertIn("Hugging Face 종료 코드 23", failed["detail"])
        with patch.object(self.backend, "commands", return_value=[sys.executable, "-c", "import time; time.sleep(30)"]):
            identifier = manager.submit(self.plan()["items"])[0]
            until = time.monotonic()+3
            while manager.jobs[identifier]["process"] is None and time.monotonic()<until:
                time.sleep(.02)
            manager.cancel(identifier)
            jobs = self.wait_done(manager)
        self.assertEqual(next(job for job in jobs if job["id"] == identifier)["status"], "cancelled")

    def test_checkpoint_downloads_serialize_shared_metadata(self):
        manager = HFTransfers(self.backend, concurrency=2)
        self.addCleanup(manager.close)
        active = 0
        peak = 0
        lock = threading.Lock()

        def execute(_self, _job):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(.05)
            with lock:
                active -= 1

        with patch("server.Transfers._execute", execute):
            manager.submit(self.plan("checkpoints-pull", ["outputs_filtered/run/checkpoints/050000",
                                                         "outputs_filtered/run/checkpoints/100000"])["items"])
            self.wait_done(manager)
        self.assertEqual(peak, 1)

    def test_delete_preview_and_exact_files_at_reviewed_revision(self):
        api = DeleteHub()
        self.backend.api = api
        plan = self.plan("dataset-delete", ["dataset_filtered/a", "dataset_filtered/a/file.txt"])
        self.assertEqual(plan["collapsed"], 1)
        item = plan["items"][0]
        self.assertEqual(item["file_count"], 2)
        self.assertEqual(item["size"], 8)
        self.assertEqual(item["paths"], ["dataset_filtered/a"])
        self.assertEqual(api.commits, [], "preview must not modify the Hub")
        command = self.backend.commands(item)
        self.assertTrue(command[2].endswith("hf_delete.py"))
        self.assertIn(api.revision, command)
        self.assertNotIn("--yes", command)
        delete_selected(api, item["repo"], item["repo_type"], item["paths"], item["revision"])
        self.assertEqual(len(api.commits), 1)
        commit = api.commits[0]
        self.assertEqual(commit["parent_commit"], api.revision)
        self.assertEqual(commit["revision"], "main")
        self.assertEqual([op.path_in_repo for op in commit["operations"]],
                         ["dataset_filtered/a/file.txt", "dataset_filtered/a/nested/weight.bin"])
        self.assertTrue(all(op.is_folder is False for op in commit["operations"]))
        self.assertNotIn("dataset_filtered/ab/keep.txt", [op.path_in_repo for op in commit["operations"]])

    def test_delete_dry_run_and_changed_branch_never_expand_scope(self):
        api = DeleteHub()
        plan = preview_delete(api, "me/data", "dataset", ["dataset_filtered/a"])
        delete_selected(api, "me/data", "dataset", plan["paths"], plan["revision"], dry_run=True)
        self.assertEqual(api.commits, [])
        api.head = "b" * 40
        with self.assertRaisesRegex(ValueError, "parent commit changed"):
            delete_selected(api, "me/data", "dataset", plan["paths"], plan["revision"])
        self.assertEqual(len(api.commits), 1, "changed revisions must fail without retrying a broader delete")
        for paths, kind in [([""], "dataset"), (["../outside"], "dataset"), (["README.md"], "dataset"),
                            (["dataset_filtered/a/*"], "dataset"), (["dataset_filtered/a"], "model"),
                            (["models/pi05_base"], "model")]:
            with self.assertRaises(ValueError):
                checked_paths(paths, kind)
        self.backend.api = DeleteHub()
        self.assertEqual(self.plan("checkpoints-delete", ["outputs_filtered/run/checkpoints/050000"])["items"][0]["file_count"], 1)

    def test_delete_confirmation_is_bound_to_scope_and_expires(self):
        self.backend.api = DeleteHub()
        plan = self.plan("dataset-delete", ["dataset_filtered/a"])
        body = {"source": "hub-datasets", "destination": "hub-datasets", "paths": ["dataset_filtered/a"],
                "plan_id": plan["plan_id"]}
        with self.assertRaisesRegex(ValueError, "확인란"):
            self.backend.confirmed_delete(body)
        body["confirm_delete"] = True
        self.assertEqual(self.backend.confirmed_delete(body)["items"][0]["revision"], "a"*40)
        with self.assertRaisesRegex(ValueError, "대상과 요청이 다릅니다"):
            self.backend.confirmed_delete({**body, "paths": ["dataset_filtered"]})
        with self.assertRaisesRegex(ValueError, "대상과 요청이 다릅니다"):
            self.backend.confirmed_delete({**body, "dry_run": True})
        self.backend.delete_previews[plan["plan_id"]]["expires"] = 0
        with self.assertRaisesRegex(ValueError, "만료"):
            self.backend.confirmed_delete(body)


class HFHTTPTest(HFWorkspace):
    def setUp(self):
        super().setUp()
        self.manager = HFTransfers(self.backend)
        self.server = HFHTTPServer(("127.0.0.1", 0), self.backend, self.manager)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.manager.close()
        self.thread.join(timeout=2)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        connection.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers or {})
        response = connection.getresponse()
        status, data = response.status, response.read()
        connection.close()
        return status, data

    def test_assets_and_session_boundary_and_preview(self):
        for path in ("/", "/app.js", "/style.css", "/hf.css", "/api/jobs"):
            self.assertEqual(self.request("GET", path)[0], 200, path)
        self.assertIn(b"Hugging Face Desk", self.request("GET", "/")[1])
        with patch("hf_server.get_token", return_value="hf_secret_not_for_browser"):
            _, body = self.request("GET", "/api/config")
            self.assertTrue(json.loads(body)["authenticated"])
            self.assertNotIn(b"hf_secret_not_for_browser", body)
        request = {"source": "local-datasets", "destination": "hub-datasets", "paths": ["dataset_filtered/data"]}
        self.assertEqual(self.request("POST", "/api/plan", request)[0], 403)
        headers = {"X-Sync-Token": self.server.token, "Content-Type": "application/json"}
        status, body = self.request("POST", "/api/plan", request, headers)
        self.assertEqual(status, 200, body)
        self.assertEqual(self.manager.snapshot(), [])
        self.assertEqual(self.request("GET", "/api/config", headers={"Host": "attacker.test"})[0], 403)
        self.assertEqual(self.request("POST", "/api/transfers", request,
            {**headers, "Origin": "https://attacker.test"})[0], 403)
        status, body = self.request("POST", "/api/transfers", {**request, "paths": ["../outside"]}, headers)
        self.assertEqual(status, 400, body)

    def test_delete_requires_review_and_confirmation_and_cannot_replay(self):
        self.backend.api = DeleteHub()
        headers = {"X-Sync-Token": self.server.token, "Content-Type": "application/json"}
        request = {"source": "hub-datasets", "destination": "hub-datasets", "paths": ["dataset_filtered/a"]}
        self.assertEqual(self.request("POST", "/api/transfers", request, headers)[0], 400)
        status, body = self.request("POST", "/api/plan", request, headers)
        self.assertEqual(status, 200, body)
        request["plan_id"] = json.loads(body)["plan_id"]
        self.assertEqual(self.request("POST", "/api/transfers", request, headers)[0], 400)
        self.assertEqual(self.manager.snapshot(), [])
        with patch.object(self.backend, "commands", return_value=[sys.executable, "-c", "print('mock Hub delete')"]):
            status, body = self.request("POST", "/api/transfers", {**request, "confirm_delete": True}, headers)
            self.assertEqual(status, 202, body)
            job = self.wait_done(self.manager)[0]
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["paths"], ["dataset_filtered/a"])
        self.assertEqual(self.request("POST", "/api/transfers", {**request, "confirm_delete": True}, headers)[0], 400)
        self.assertEqual(self.backend.api.commits, [], "HTTP test worker is mocked; no real Hub deletion")


if __name__ == "__main__":
    unittest.main()
