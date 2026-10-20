"""All copies/deletions are confined to TemporaryDirectory; no real SSH or model I/O."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import archive_remote as protocol
import checkpoint_archive as archive
from transport import Endpoint


class LocalRemote:
    """Run the actual remote verification/commit protocol against a fresh local tree."""
    def __init__(self, base):
        self.endpoint = Endpoint("scratch2", "test archive", str(base), "user@archive.test")
        self.failure = None
        self.after_upload = None

    def request(self, action, relative, manifest):
        if self.failure:
            raise self.failure
        return protocol.process({"base": self.endpoint.base, "relative": relative,
                                 "action": action, "manifest": manifest})

    def upload(self, source, stage):
        if self.failure:
            raise self.failure
        shutil.copytree(source, stage, dirs_exist_ok=True)
        if self.after_upload:
            self.after_upload(source)


class ArchiveTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="checkpoint-archive-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source, self.destination = self.root / "source", self.root / "destination"
        self.source.mkdir()
        self.destination.mkdir()
        self.relative = Path("outputs_filtered/skillVLA_FT/Predictor/test_run")
        self.run = self.source / self.relative
        self.remote = LocalRemote(self.destination)

    def save(self, step, keep=2, enqueue=True):
        checkpoint = self.run / "checkpoints" / f"{step:06d}"
        checkpoint.mkdir(parents=True)
        for name in archive.REQUIRED | {"pretrained_model/model.safetensors", "training_state/scheduler_state.json"}:
            path = checkpoint / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes((name + str(step)).encode())
        (checkpoint / "training_state/training_step.json").write_text(json.dumps({"step": step}))
        last = checkpoint.parent / "last"
        if last.is_symlink():
            last.unlink()
        last.symlink_to(checkpoint.name)
        if enqueue:
            archive.register(checkpoint, keep, "scratch2", repo=self.source,
                             endpoint=self.remote.endpoint, start=False)
        return checkpoint

    def live_steps(self):
        return {int(path.name) for path in (self.run / "checkpoints").iterdir() if path.name.isdigit()}

    def remote_checkpoint(self, step):
        return self.destination / self.relative / "checkpoints" / f"{step:06d}"

    def test_full_requested_sweep_remotely_latest_two_locally(self):
        steps = [0, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000]
        for step in steps:
            checkpoint = self.save(step)
            original = protocol.file_manifest(checkpoint)
            self.assertEqual(archive.run_pass(self.run, self.remote), 0)
            self.assertEqual(protocol.file_manifest(self.remote_checkpoint(step)), original)
            self.assertTrue((self.remote_checkpoint(step) / protocol.RECEIPT).is_file())
        self.assertEqual(self.live_steps(), {40000, 50000})
        self.assertEqual((self.run / "checkpoints/last").resolve(), self.run / "checkpoints/050000")
        self.assertEqual((self.remote_checkpoint(50000).parent / "last").resolve(), self.remote_checkpoint(50000))
        rows = archive.entries(self.run / archive.STATE)
        self.assertEqual([row["status"] for _, row in rows], ["deleted"] * 8 + ["verified"] * 2)

    def test_vsa_1k_to_10k_sweep_archives_all_ten_and_keeps_latest_one_locally(self):
        self.relative = Path("outputs_filtered/skillVLA_FT/VSA/bspline3_ft")
        self.run = self.source / self.relative
        steps = range(1000, 10001, 1000)
        for step in steps:
            checkpoint = self.save(step, keep=1)
            original = protocol.file_manifest(checkpoint)
            self.assertEqual(archive.run_pass(self.run, self.remote), 0)
            self.assertEqual(protocol.file_manifest(self.remote_checkpoint(step)), original)
        self.assertEqual(self.live_steps(), {10000})
        self.assertTrue(all(self.remote_checkpoint(step).is_dir() for step in steps))
        self.assertEqual((self.run / "checkpoints/last").resolve(), self.run / "checkpoints/010000")
        self.assertEqual((self.remote_checkpoint(10000).parent / "last").resolve(), self.remote_checkpoint(10000))
        self.assertTrue((self.remote_checkpoint(1000) / "training_state/optimizer_state.safetensors").is_file())
        self.assertTrue((self.remote_checkpoint(1000) / "training_state/rng_state.safetensors").is_file())
        rows = archive.entries(self.run / archive.STATE)
        self.assertEqual([row["status"] for _, row in rows], ["deleted"] * 9 + ["verified"])

    def test_vsa_keep_one_never_prunes_failed_transfers(self):
        self.relative = Path("outputs_filtered/skillVLA_FT/VSA/bspline3_ft")
        self.run = self.source / self.relative
        for step in (1000, 2000):
            self.save(step, keep=1)
        self.remote.failure = RuntimeError("connection lost")
        self.assertEqual(archive.run_pass(self.run, self.remote), 2)
        self.assertEqual(self.live_steps(), {1000, 2000})
        self.remote = LocalRemote(self.destination)
        self.assertEqual(archive.run_pass(self.run, self.remote, now=float("inf")), 0)
        self.assertEqual(self.live_steps(), {2000})
        self.assertTrue(all(self.remote_checkpoint(step).is_dir() for step in (1000, 2000)))

    def test_network_failure_keeps_all_and_restart_recovers_durable_queue(self):
        for step in [0, 500, 1000, 2000]:
            self.save(step)
        self.remote.failure = RuntimeError("connection lost")
        self.assertEqual(archive.run_pass(self.run, self.remote), 4)
        self.assertEqual(self.live_steps(), {0, 500, 1000, 2000})
        self.remote = LocalRemote(self.destination)  # No in-memory transfer state survives.
        self.assertEqual(archive.run_pass(self.run, self.remote, now=float("inf")), 0)
        self.assertEqual(self.live_steps(), {1000, 2000})
        self.assertTrue(all(self.remote_checkpoint(step).is_dir() for step in [0, 500, 1000, 2000]))

    def test_keep_all_still_uploads_every_save(self):
        for step in [0, 500, 1000]:
            self.save(step, keep=-1)
        self.assertEqual(archive.run_pass(self.run, self.remote), 0)
        self.assertEqual(self.live_steps(), {0, 500, 1000})

    def test_unregistered_partial_checkpoint_is_never_uploaded_or_deleted(self):
        self.save(0, enqueue=False)
        for step in [500, 1000, 2000]:
            self.save(step)
        self.assertEqual(archive.run_pass(self.run, self.remote), 0)
        self.assertEqual(self.live_steps(), {0, 1000, 2000})
        self.assertFalse(self.remote_checkpoint(0).exists())

    def test_remote_corruption_after_upload_prevents_deletion_and_retries_cleanup(self):
        self.save(0)
        archive.run_pass(self.run, self.remote)
        (self.remote_checkpoint(0) / "pretrained_model/model.safetensors").write_bytes(b"corrupt")
        self.save(500)
        self.save(1000)
        self.assertGreater(archive.run_pass(self.run, self.remote), 0)
        self.assertEqual(self.live_steps(), {0, 500, 1000})

    def test_missing_remote_copy_prevents_deletion(self):
        self.save(0)
        archive.run_pass(self.run, self.remote)
        shutil.rmtree(self.remote_checkpoint(0))
        self.save(500)
        self.save(1000)
        self.assertGreater(archive.run_pass(self.run, self.remote), 0)
        self.assertIn(0, self.live_steps())

    def test_existing_different_remote_checkpoint_is_never_overwritten(self):
        self.save(0)
        final = self.remote_checkpoint(0)
        final.mkdir(parents=True)
        (final / "unrelated.txt").write_text("preserve")
        self.assertEqual(archive.run_pass(self.run, self.remote), 1)
        self.assertEqual((final / "unrelated.txt").read_text(), "preserve")
        self.assertIn(0, self.live_steps())

    def test_interrupted_upload_not_published_and_retry_resumes(self):
        self.save(0)
        real_upload = self.remote.upload

        def interrupted(source, stage):
            (Path(stage) / "partial").write_text("unfinished")
            raise RuntimeError("transfer interrupted")

        self.remote.upload = interrupted
        self.assertEqual(archive.run_pass(self.run, self.remote), 1)
        self.assertFalse(self.remote_checkpoint(0).exists())
        self.assertIn(0, self.live_steps())
        # A real resumed rsync is not allowed to publish unexpected extra files.
        self.remote.upload = real_upload
        self.assertEqual(archive.run_pass(self.run, self.remote, now=float("inf")), 1)
        for stage in self.remote_checkpoint(0).parent.glob(".archive-000000-*"):
            (stage / "partial").unlink()
        self.assertEqual(archive.run_pass(self.run, self.remote, now=float("inf")), 0)

    def test_source_mutation_during_upload_never_authorizes_deletion(self):
        checkpoint = self.save(0)
        self.remote.after_upload = lambda source: (source / "pretrained_model/model.safetensors").write_bytes(b"new")
        self.assertEqual(archive.run_pass(self.run, self.remote), 1)
        self.assertTrue(checkpoint.exists())
        self.assertEqual(archive.run_pass(self.run, self.remote, now=float("inf")), 1)

    def test_source_mutation_before_worker_starts_is_not_archived(self):
        checkpoint = self.save(0)
        (checkpoint / "pretrained_model/model.safetensors").write_bytes(b"rewritten")
        self.assertEqual(archive.run_pass(self.run, self.remote), 1)
        self.assertFalse(self.remote_checkpoint(0).exists())
        self.assertTrue(checkpoint.is_dir())

    def test_incomplete_save_and_duplicate_registration_are_rejected(self):
        checkpoint = self.save(0, enqueue=False)
        (checkpoint / "training_state/rng_state.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            archive.register(checkpoint, 2, "scratch2", repo=self.source, endpoint=self.remote.endpoint, start=False)
        checkpoint = self.save(500)
        with self.assertRaisesRegex(ValueError, "already registered"):
            archive.register(checkpoint, 2, "scratch2", repo=self.source, endpoint=self.remote.endpoint, start=False)

    def test_symlink_checkpoint_file_is_never_deleted(self):
        checkpoint = self.save(0)
        target = self.root / "outside"
        target.write_text("preserve")
        weights = checkpoint / "pretrained_model/model.safetensors"
        weights.unlink()
        weights.symlink_to(target)
        self.save(500)
        self.save(1000)
        self.assertGreater(archive.run_pass(self.run, self.remote), 0)
        self.assertTrue(checkpoint.exists())
        self.assertEqual(target.read_text(), "preserve")

    def test_remote_paths_cannot_escape_or_follow_symlinks(self):
        for relative in ["../outside/checkpoints/000000", "/outputs/a/checkpoints/000000",
                         "outputs_filtered/../a/checkpoints/000000", "outputs_filtered/a/checkpoints/last"]:
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                protocol.checkpoint_path(str(self.destination), relative)
        (self.destination / "outputs_filtered").symlink_to(self.root)
        with self.assertRaises(ValueError):
            protocol.checkpoint_path(str(self.destination), str(self.relative / "checkpoints/000000"))

    def test_last_symlink_is_always_retained(self):
        first = self.save(0)
        self.save(500)
        self.save(1000)
        last = self.run / "checkpoints/last"
        last.unlink()
        last.symlink_to(first.name)
        self.assertEqual(archive.run_pass(self.run, self.remote), 0)
        self.assertEqual(self.live_steps(), {0, 500, 1000})

    def test_destination_cannot_change_after_registration(self):
        self.save(0)
        second = self.save(500, enqueue=False)
        other = Endpoint("scratch2", "other", str(self.root / "other"), "user@archive.test")
        with self.assertRaisesRegex(ValueError, "destination/run changed"):
            archive.register(second, 2, "scratch2", repo=self.source, endpoint=other, start=False)

    def test_same_shared_storage_is_rejected_even_from_compute_node_hostname(self):
        checkpoint = self.save(0, enqueue=False)
        same = Endpoint("scratch2", "Yonsei shared storage", str(self.source), "user@other-login")
        with patch("checkpoint_archive.socket.gethostname", return_value="node26"):
            with self.assertRaisesRegex(ValueError, "workspace paths are identical"):
                archive.register(checkpoint, 2, "scratch2", repo=self.source, endpoint=same, start=False)
        self.assertTrue(checkpoint.is_dir())
        self.assertFalse((self.run / archive.STATE).exists())

    def test_recovered_queue_cannot_prune_a_same_storage_destination(self):
        for step in [0, 500, 1000]:
            self.save(step)
        path = self.run / archive.STATE / "config.json"
        config = archive.read_json(path)
        config["endpoint"]["base"] = str(self.source)
        protocol.atomic_json(path, config)
        with self.assertRaisesRegex(ValueError, "workspace paths are identical"):
            archive.run_pass(self.run, self.remote)
        self.assertEqual(self.live_steps(), {0, 500, 1000})

    def test_remote_protocol_runs_in_separate_python_without_installed_packages(self):
        checkpoint = self.save(0)
        manifest = protocol.file_manifest(checkpoint)
        request = {"base": str(self.destination), "relative": str(self.relative / "checkpoints/000000"),
                   "action": "prepare", "manifest": manifest}
        result = subprocess.run([sys.executable, "-S", str(TOOLS / "archive_remote.py")],
                                input=json.dumps(request), capture_output=True, text=True, check=True)
        response = json.loads(result.stdout)
        self.assertFalse(response["complete"])
        self.assertEqual(response["sha256"], protocol.manifest_id(manifest))

    @unittest.skipUnless(shutil.which("rsync"), "rsync not installed")
    def test_actual_rsync_staging_and_commit_preserves_full_training_state(self):
        def upload(source, stage):
            subprocess.run(archive.rsync_options() + ["--", str(source) + "/", stage + "/"],
                           check=True, stdout=subprocess.DEVNULL)
        self.remote.upload = upload
        for step in [0, 500, 1000]:
            self.save(step)
        self.assertEqual(archive.run_pass(self.run, self.remote), 0)
        self.assertEqual(self.live_steps(), {500, 1000})
        for step in [0, 500, 1000]:
            self.assertTrue((self.remote_checkpoint(step) / "training_state/rng_state.safetensors").is_file())

    def test_worker_lock_prevents_two_workers_for_one_run(self):
        self.save(0)
        with archive.locked(self.run / archive.STATE / "worker.lock"):
            with patch("checkpoint_archive.run_pass") as run:
                archive.worker(self.run)
                run.assert_not_called()

    def test_remote_last_never_regresses_when_an_old_transfer_retries(self):
        zero = self.save(0)
        later = self.save(1000)
        manifest = protocol.file_manifest(later)
        relative = later.relative_to(self.source).as_posix()
        prepared = self.remote.request("prepare", relative, manifest)
        self.remote.upload(later, prepared["stage"])
        self.remote.request("commit", relative, manifest)
        archive.run_pass(self.run, self.remote)
        self.assertTrue(zero.exists())
        self.assertEqual((self.remote_checkpoint(1000).parent / "last").resolve(), self.remote_checkpoint(1000))

    def test_remote_last_real_directory_is_not_removed(self):
        self.save(0)
        last = self.remote_checkpoint(0).parent / "last"
        last.mkdir(parents=True)
        (last / "user-data").write_text("preserve")
        self.assertEqual(archive.run_pass(self.run, self.remote), 1)
        self.assertEqual((last / "user-data").read_text(), "preserve")
        self.assertIn(0, self.live_steps())

    def test_ssh_transport_is_noninteractive_and_uses_existing_endpoint_options(self):
        remote = archive.Remote(self.remote.endpoint)
        digest = protocol.manifest_id({"file": {"size": 1, "sha256": "x"}})
        with patch("checkpoint_archive.subprocess.run", return_value=subprocess.CompletedProcess([], 0, json.dumps({"sha256": digest}), "")) as run:
            remote.request("verify", str(self.relative / "checkpoints/000000"), {"file": {"size": 1, "sha256": "x"}})
        self.assertIn("BatchMode=yes", run.call_args.args[0])
        self.assertNotIn("shell", run.call_args.kwargs)

    def test_invalid_keep_rejected(self):
        for keep in [0, -2, True, 2.5, "2"]:
            with self.subTest(keep=keep), self.assertRaises(ValueError):
                archive.validate_keep(keep)


class TrainerIntegrationTest(unittest.TestCase):
    def test_disabled_hook_is_noop_and_enabled_hook_only_registers_finished_save(self):
        root = TOOLS.parents[1]
        source = root / "lerobot/src/lerobot/utils/checkpoint_archive.py"
        spec = importlib.util.spec_from_file_location("train_archive_bridge", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.object(module.subprocess, "run") as run:
            module.queue_checkpoint_archive(SimpleNamespace(), Path("/unused"))
            run.assert_not_called()
            cfg = SimpleNamespace(checkpoint_archive="scratch2", keep_checkpoints=2)
            module.queue_checkpoint_archive(cfg, Path("/test/checkpoints/000500"))
            args = run.call_args.args[0]
            self.assertIn("register", args)
            self.assertIn("--keep", args)
            self.assertIn("scratch2", args)

    def test_real_trainer_hooks_follow_finished_save_and_last_update(self):
        root = TOOLS.parents[1]
        tree = ast.parse((root / "lerobot/src/lerobot/scripts/lerobot_train.py").read_text())
        hooks = []
        for node in ast.walk(tree):
            for _, value in ast.iter_fields(node):
                if not isinstance(value, list):
                    continue
                for index, statement in enumerate(value):
                    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
                        continue
                    call = statement.value
                    if isinstance(call.func, ast.Name) and call.func.id == "update_last_checkpoint":
                        next_call = value[index + 1].value
                        self.assertEqual(next_call.func.id, "queue_checkpoint_archive")
                        hooks.append(call)
        self.assertEqual(len(hooks), 2)  # step 0 and ordinary saves
        text = (root / "lerobot/src/lerobot/scripts/lerobot_train.py").read_text()
        self.assertLess(text.index("accelerator.end_training()"), text.index("finish_checkpoint_archive(cfg)"))


if __name__ == "__main__":
    unittest.main()
