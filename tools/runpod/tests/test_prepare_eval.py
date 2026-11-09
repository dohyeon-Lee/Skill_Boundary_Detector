"""Offline publication/consumption checks; never allocate a Pod or upload real data."""
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
from huggingface_hub.errors import HfHubHTTPError, RemoteEntryNotFoundError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_eval as workflow
from hf_desk_client import HFDesk, local_url


class Hub:
    def __init__(self, root):
        self.root, self.files, self.history, self.head = root, {}, {}, 0
        self.private = True
        self.checkpoint_private = False
        self.commit()

    def commit(self):
        self.head += 1
        self.revision = f"{self.head:040x}"
        self.history[self.revision] = dict(self.files)

    def repo_info(self, *args, **kwargs):
        return SimpleNamespace(sha=self.revision, private=self.private if kwargs.get("repo_type") == "dataset" else self.checkpoint_private)

    def download(self, repo, filename, revision, **kwargs):
        content = self.history[revision].get(filename)
        if content is None:
            raise RemoteEntryNotFoundError("missing", response=httpx.Response(404,
                request=httpx.Request("GET", "https://example.test/fixture")))
        directory = kwargs.get("local_dir") or self.root / "hub-cache"
        path = directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def get_paths_info(self, repo, paths, **kwargs):
        files = self.history[kwargs["revision"]]
        return [SimpleNamespace(path=name, size=len(files[name]), blob_id="unused",
                               lfs=SimpleNamespace(sha256=workflow.digest(files[name])))
                for name in paths if name in files]


class Desk:
    def __init__(self, base, hub):
        self.base, self.hub, self.calls = base, hub, []
        self.fail = False
        self.checkpoint_repo = "me/checkpoints"
        self.checkpoint_calls = []

    def wait_pending(self, prefix):
        pass

    def publish(self, name, *, checkpoint=False):
        self.calls.append(name)
        if checkpoint:
            self.checkpoint_calls.append(name)
        if self.fail:
            raise ValueError("upload failed")
        selected = self.base / name / "pretrained_model" if checkpoint else self.base / name
        for path in selected.rglob("*") if selected.is_dir() else [selected]:
            if path.is_file():
                self.hub.files[path.relative_to(self.base).as_posix()] = path.read_bytes()
        self.hub.commit()
        return ["fixture-job"]


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.pod = self.root / "server", self.root / "pod"
        self.source.mkdir()
        self.pod.mkdir()
        self.config = self.source / workflow.EVALS["ft"]
        self.config.parent.mkdir(parents=True)
        self.config.write_text("n_episodes: 3\n")
        self.settings = {"hf_dataset_repo": "me/private", "hf_checkpoint_repo": "me/checkpoints", "n_episodes": 3, "dataset_root": "dataset_filtered"}
        self.asset = self.source / "outputs_filtered/run/checkpoints/050000/pretrained_model/model.safetensors"
        self.asset.parent.mkdir(parents=True)
        self.asset.write_bytes(b"completed numeric checkpoint")
        self.resolved = {"models_json": json.dumps([{"policy_path": str(self.asset.parent)}])}
        self.hub = Hub(self.root)
        self.desk = Desk(self.source, self.hub)
        self.patches = contextlib.ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch.object(workflow, "HfApi", return_value=self.hub))
        self.patches.enter_context(patch.object(workflow, "configuration", return_value=(self.resolved, self.settings)))
        self.patches.enter_context(patch.object(workflow, "connect", return_value=self.desk))
        self.patches.enter_context(patch.object(workflow, "downloaded", side_effect=self.hub.download))
        def matching(api, root, files, dataset_repo, checkpoint_repo, private_revision):
            return {item["path"]: {"repo": checkpoint_repo if item["path"].startswith("outputs_") else dataset_repo,
                "repo_type": "model" if item["path"].startswith("outputs_") else "dataset",
                "revision": self.hub.revision, "filename": item["path"]}
                for item in files if self.hub.files.get(item["path"]) == (root / item["path"]).read_bytes()}
        self.matches = self.patches.enter_context(patch.object(workflow, "refs_for_files", side_effect=matching))
        loader = lambda *args: self.settings
        self.patches.enter_context(patch.object(workflow.runpy, "run_path", return_value={"load_global_config": loader}))
        self.patches.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def prepare(self, **kwargs):
        workflow.prepare(self.source, self.config, "ft", **kwargs)

    def consume(self, **kwargs):
        workflow.consume(self.pod, self.pod / workflow.EVALS["ft"], "ft", **kwargs)

    def test_missing_files_published_through_desk_then_downloaded_and_reused(self):
        self.prepare()
        self.assertEqual(len(self.desk.calls), 2)
        self.assertEqual(self.desk.checkpoint_calls, [self.asset.parent.parent.relative_to(self.source).as_posix()])
        self.assertEqual(self.desk.calls[1], "dataset_filtered/eval_configs/ft.json")
        with patch.object(workflow.subprocess, "run") as run:
            self.consume(download_only=True)
            run.assert_not_called()
        destination = self.pod / self.asset.relative_to(self.source)
        self.assertEqual(destination.read_bytes(), self.asset.read_bytes())
        with patch.object(workflow, "downloaded", wraps=self.hub.download) as download:
            self.consume(download_only=True)
            # Warm run fetches only the small manifest, no payload.
            self.assertEqual(download.call_count, 1)
        self.prepare()
        self.assertEqual(len(self.desk.calls), 2)
        self.assertEqual(self.matches.call_count, 2)

    def test_default_consumption_launches_existing_ft_with_frozen_yaml(self):
        self.prepare()
        with patch.object(workflow.subprocess, "run") as run:
            self.consume()
        arguments = run.call_args
        self.assertEqual(arguments.args[0], ["bash", str(self.pod / workflow.EVALS["ft"].parent / "submit_eval.sh")])
        snapshot = Path(arguments.kwargs["env"]["STAGE1_EVAL_CONFIG"])
        self.assertIn(".hf_configs", snapshot.parts)
        self.assertEqual(workflow.yaml.safe_load(snapshot.read_text())["n_episodes"], 3)

    def test_existing_public_checkpoint_reused_without_payload_upload(self):
        self.hub.files[self.asset.relative_to(self.source).as_posix()] = self.asset.read_bytes()
        self.hub.commit()
        self.prepare()
        self.assertEqual(self.desk.checkpoint_calls, [])
        self.assertEqual(self.desk.calls, ["dataset_filtered/eval_configs/ft.json"])
        self.consume(download_only=True)
        self.assertEqual((self.pod / self.asset.relative_to(self.source)).read_bytes(), self.asset.read_bytes())

    def test_pt_uses_its_own_pointer_and_existing_pt_launcher(self):
        config = self.source / workflow.EVALS["pt"]
        config.parent.mkdir(parents=True)
        config.write_text("n_episodes: 3\n")
        workflow.prepare(self.source, config, "pt")
        self.assertIn("dataset_filtered/eval_configs/pt.json", self.hub.files)
        with patch.object(workflow.subprocess, "run") as run:
            workflow.consume(self.pod, self.pod / workflow.EVALS["pt"], "pt")
        self.assertEqual(run.call_args.args[0][1], str(self.pod / workflow.EVALS["pt"].parent / "submit_eval.sh"))
        self.assertEqual(run.call_args.kwargs["env"]["SBD_EVAL_HF_READY"], "1")

    def test_failed_checkpoint_upload_keeps_old_ready_manifest(self):
        self.prepare()
        old = self.hub.files["dataset_filtered/eval_configs/ft.json"]
        self.asset.write_bytes(b"different checkpoint")
        self.desk.fail = True
        with self.assertRaisesRegex(ValueError, "upload failed"):
            self.prepare()
        self.assertEqual(self.hub.files["dataset_filtered/eval_configs/ft.json"], old)

    def test_dataset_repo_must_remain_private(self):
        self.hub.private = False
        with self.assertRaisesRegex(ValueError, "비공개"):
            self.prepare()
        self.assertEqual(self.desk.calls, [])

    def test_checkpoint_repo_must_be_public(self):
        self.hub.checkpoint_private = True
        with self.assertRaisesRegex(ValueError, "공개 체크포인트"):
            self.prepare()
        self.assertEqual(self.desk.calls, [])

    def test_data_is_uploaded_at_original_dataset_path(self):
        source = self.source / 'dataset_filtered/libero/data.hdf5'
        source.parent.mkdir(parents=True)
        source.write_bytes(b'data fixture')
        self.resolved['models_json'] = json.dumps([{'policy_path': str(self.asset.parent), 'raw_dataset_dir': str(source.parent)}])
        self.prepare()
        self.assertIn('dataset_filtered/libero/data.hdf5', self.desk.calls)
        self.assertFalse(any('dataset_eval_hf' in path for path in self.hub.files))
        self.consume(download_only=True)
        self.assertEqual((self.pod / source.relative_to(self.source)).read_bytes(), b'data fixture')

    def test_private_bundle_checkpoint_reference_is_rejected(self):
        self.prepare()
        manifest = json.loads(self.hub.files['dataset_filtered/eval_configs/ft.json'])
        manifest['files'][0]['hf']['repo_type'] = 'dataset'
        with self.assertRaisesRegex(ValueError, '공개 저장소'):
            workflow.validate_manifest(manifest, self.config.relative_to(self.source).as_posix())

    def test_existing_different_checkpoint_never_overwritten(self):
        self.prepare()
        target = self.pod / self.asset.relative_to(self.source)
        target.parent.mkdir(parents=True)
        target.write_bytes(b"existing other evaluation")
        with self.assertRaisesRegex(ValueError, "로컬 파일"):
            self.consume(download_only=True)
        self.assertEqual(target.read_bytes(), b"existing other evaluation")

    def test_tampered_manifest_stops_before_downloading_weights(self):
        self.prepare()
        name = "dataset_filtered/eval_configs/ft.json"
        pinned = self.hub.history[self.hub.revision]
        manifest = json.loads(pinned[name])
        manifest["recipe"]["settings"]["n_episodes"] = 200
        pinned[name] = workflow.encoded(manifest)
        with self.assertRaisesRegex(ValueError, "해시"):
            self.consume(download_only=True)
        self.assertFalse((self.pod / "outputs_filtered").exists())

    def test_dry_run_does_not_connect_desk_or_publish(self):
        with patch.object(workflow, "connect") as connect:
            self.prepare(dry_run=True)
        connect.assert_not_called()
        self.assertEqual(self.desk.calls, [])

    def test_local_stat_change_invalidates_hash_cache(self):
        cache = workflow.HashCache(self.root / "hashes.json")
        original = cache.hash(self.asset)
        self.asset.write_bytes(b"changed input")
        self.assertNotEqual(cache.hash(self.asset), original)

    def test_path_traversal_and_code_destination_rejected(self):
        for path in ("../secret", "/etc/passwd", "outputs_filtered/../run.py", "tools/runpod/prepare_eval.py", ".git/config"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                workflow.asset_path(path)

    def test_stale_same_name_remote_file_is_not_reused(self):
        item = {"size": 3, "sha256": workflow.digest(b"new"), "git_sha1": "new"}
        entry = SimpleNamespace(size=3, lfs=SimpleNamespace(sha256=workflow.digest(b"old")))
        self.assertFalse(workflow.same(entry, item))

    def test_retry_transient_only(self):
        with patch.object(workflow.time, "sleep"), patch.object(workflow, "hf_hub_download",
                side_effect=[httpx.RemoteProtocolError("reset"), "/tmp/done"]) as download:
            self.assertEqual(workflow.retry(lambda: download()), "/tmp/done")
            self.assertEqual(download.call_count, 2)
        transient = HfHubHTTPError("busy", response=httpx.Response(503,
            request=httpx.Request("GET", "https://example.test")))
        with patch.object(workflow.time, "sleep"), patch.object(workflow, "hf_hub_download",
                side_effect=[transient, "/tmp/done"]) as download:
            self.assertEqual(workflow.retry(lambda: download()), "/tmp/done")
        forbidden = httpx.HTTPStatusError("forbidden", request=httpx.Request("GET", "https://example.test"), response=httpx.Response(403))
        with patch.object(workflow.time, "sleep") as sleep:
            with self.assertRaises(httpx.HTTPStatusError):
                workflow.retry(lambda: (_ for _ in ()).throw(forbidden))
            sleep.assert_not_called()


class DeskHTTPTests(unittest.TestCase):
    def test_input_uploads_fill_two_slots_and_refill_without_waiting_for_both(self):
        lock = threading.Lock()
        first_started, second_started, third_started = (threading.Event() for _ in range(3))
        release_first, release_second = threading.Event(), threading.Event()
        active = 0
        peak = 0
        calls = []
        def publish(path, *, checkpoint=False):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                calls.append((path, checkpoint))
            try:
                if path == 'a':
                    first_started.set()
                    assert release_first.wait(5)
                elif path == 'b':
                    second_started.set()
                    assert release_second.wait(5)
                else:
                    third_started.set()
            finally:
                with lock:
                    active -= 1
        desk = SimpleNamespace(publish=publish)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(workflow.publish_inputs, desk, {'a', 'b', 'c'}, [])
            try:
                self.assertTrue(first_started.wait(2))
                self.assertTrue(second_started.wait(2))
                self.assertFalse(third_started.is_set())
                release_second.set()
                self.assertTrue(third_started.wait(2))
                self.assertFalse(release_first.is_set())
            finally:
                release_first.set()
                release_second.set()
            future.result(timeout=5)
        self.assertEqual(peak, 2)
        self.assertCountEqual(calls, [('a', True), ('b', True), ('c', True)])

    def test_checkpoint_upload_requires_capability_and_uses_public_route(self):
        desk = object.__new__(HFDesk)
        desk.config = {}
        with self.assertRaisesRegex(ValueError, '재시작'):
            desk.publish('outputs_filtered/run/checkpoints/050000', checkpoint=True)
        desk.config = {'checkpoint_pretrained_only': True}
        desk.jobs = Mock(return_value=[])
        desk.request = Mock(return_value={'ids': ['public-job']})
        desk.wait = Mock(return_value=['public-job'])
        desk.publish('outputs_filtered/run/checkpoints/050000', checkpoint=True)
        self.assertEqual(desk.request.call_args.args, ('/api/transfers', {
            'source': 'local-checkpoints', 'destination': 'hub-checkpoints',
            'paths': ['outputs_filtered/run/checkpoints/050000'], 'operation': 'checkpoints-push',
            'dry_run': False, 'options': {'pretrained_only': True}}))

    def test_uses_desk_api_and_preserves_existing_jobs(self):
        jobs = [{"id": "existing", "route": "dataset-push", "path": "dataset_other/x", "status": "completed"}]
        requests = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass

                def reply(self, value):
                    body = json.dumps(value).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def do_GET(self):
                    self.reply({"jobs": jobs} if self.path.endswith("/jobs") else {"token": "fixture-session", "endpoints": [
                        {"id": "local-checkpoints", "base": str(root)},
                        {"id": "local-datasets", "base": str(root)},
                        {"id": "hub-datasets", "base": "https://huggingface.co/datasets/me/private"}]})

                def do_POST(self):
                    if self.headers.get("X-Sync-Token") != "fixture-session":
                        raise AssertionError("missing CSRF header")
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    requests.append(body)
                    jobs.append({"id": "new-job", "route": "dataset-push", "path": body["paths"][0], "status": "completed"})
                    self.reply({"ids": ["new-job"]})

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                desk = HFDesk(f"http://127.0.0.1:{server.server_port}/hf", root, "me/private")
                with contextlib.redirect_stdout(io.StringIO()):
                    desk.publish("dataset_filtered/eval_configs/ft.json")
                self.assertEqual(requests[0], {"source": "local-datasets", "destination": "hub-datasets",
                    "operation": "dataset-push", "paths": ["dataset_filtered/eval_configs/ft.json"], "dry_run": False})
                self.assertEqual(jobs[0]["id"], "existing")
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_desk_address_must_be_loopback(self):
        for url in ("https://127.0.0.1:8765", "http://example.test", "http://user@localhost:8765", "http://localhost:8765/other"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                local_url(url)


if __name__ == "__main__":
    unittest.main()
