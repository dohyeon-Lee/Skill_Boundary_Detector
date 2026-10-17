"""Durable, opt-in checkpoint offload. Delete locally ONLY after remote SHA256 verification.

Training registers finished saves; a separate CPU worker performs network I/O.
Nothing scans or deletes unregistered checkpoints. No UI server is required.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import json
import logging
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time

from archive_remote import RECEIPT, atomic_json, checkpoint_path, file_manifest, manifest_id, verify
from transport import Endpoint, REPO, SSH_OPTIONS, configured_endpoints, rsync_options

log = logging.getLogger("checkpoint_archive")
STATE = ".checkpoint_archive"
REQUIRED = {"pretrained_model/config.json", "pretrained_model/train_config.json",
            "training_state/training_step.json", "training_state/optimizer_param_groups.json",
            "training_state/optimizer_state.safetensors", "training_state/rng_state.safetensors"}


@contextmanager
def locked(path, nonblocking=False):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Unsafe archive lock: " + str(path))
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_keep(keep):
    if isinstance(keep, bool) or not isinstance(keep, int) or (keep != -1 and keep < 1):
        raise ValueError("keep_checkpoints must be -1 (keep all) or a positive integer")


def validate_destination(root, endpoint):
    # Yonsei compute hosts are named nodeNN, not yonsei*. The same absolute
    # workspace path can be the SAME shared storage through a different login
    # server. Conservatively refuse identical paths even across different hosts.
    if Path(os.path.abspath(endpoint.base)) == Path(root):
        raise ValueError("Source and archive workspace paths are identical; disable checkpoint_archive on Yonsei")
    remote_host = endpoint.host.rsplit("@", 1)[-1]
    if remote_host in ("", "localhost", "127.0.0.1", "::1", socket.gethostname(), socket.getfqdn()):
        raise ValueError("Refusing checkpoint offload to the local host")


def fingerprint(directory):
    """Cheap immutable-save snapshot, without reading tensor contents on the training path."""
    result = {}
    for parent, dirs, files in os.walk(directory, followlinks=False):
        for name in dirs + files:
            if (Path(parent) / name).is_symlink():
                raise ValueError("Symlink inside checkpoint")
        for name in files:
            path = Path(parent) / name
            relative = path.relative_to(directory).as_posix()
            if relative == RECEIPT:
                continue
            if not path.is_file():
                raise ValueError("Non-regular checkpoint file")
            stat = path.stat()
            result[relative] = [stat.st_size, stat.st_mtime_ns, stat.st_ino]
    return result


def check_snapshot(source, entry):
    if fingerprint(source) != entry["snapshot"]:
        raise ValueError("Completed checkpoint changed after registration: " + entry["name"])


def state_dir(run):
    run = Path(run)
    state = run / STATE
    if run.is_symlink() or state.is_symlink():
        raise ValueError("Symlinked archive run/state directory")
    for name in ("queue", "config.json", "worker.log"):
        if (state / name).is_symlink():
            raise ValueError("Symlinked archive state entry: " + name)
    return state


def register(checkpoint, keep, destination, repo=REPO, endpoint=None, start=True):
    """Called only AFTER the trainer successfully saves optimizer/RNG and updates last."""
    validate_keep(keep)
    root = Path(repo).resolve()
    checkpoint = Path(os.path.abspath(checkpoint))
    relative = checkpoint.relative_to(root).as_posix()
    checkpoint_path(str(root), relative)
    run = checkpoint.parent.parent
    if endpoint is None:
        if socket.gethostname().startswith("yonsei"):
            raise ValueError("Checkpoint offload is for the external server, not Yonsei-to-itself")
        _, endpoints = configured_endpoints(root, {**os.environ, "SYNC_MODE": "rllab"})
        endpoint = next((ep for ep in endpoints if ep.id == destination and ep.host), None)
    if endpoint is None or destination not in ("scratch", "scratch2"):
        raise ValueError("checkpoint_archive must select scratch or scratch2 (remote Yonsei)")
    validate_destination(root, endpoint)
    # No hashes on the GPU training path; only completed-file and step checks.
    for name in REQUIRED:
        path = checkpoint / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Incomplete checkpoint; not queued: " + name)
    if not any((checkpoint / "pretrained_model").glob("*.safetensors")):
        raise ValueError("Checkpoint has no model weights")
    if read_json(checkpoint / "training_state/training_step.json")["step"] != int(checkpoint.name):
        raise ValueError("Checkpoint step does not match directory")
    state = state_dir(run)
    state.mkdir(exist_ok=True)
    (state / "queue").mkdir(exist_ok=True)
    config = {"version": 1, "root": str(root), "relative_run": run.relative_to(root).as_posix(),
              "endpoint": asdict(endpoint), "keep": keep}
    with locked(state / "state.lock"):
        if (state / "config.json").exists():
            previous = read_json(state / "config.json")
            if any(previous[key] != config[key] for key in ("root", "relative_run", "endpoint")):
                raise ValueError("Archive destination/run changed; refusing to reuse old deletion receipts")
        atomic_json(state / "config.json", config)
        item_path = state / "queue" / (checkpoint.name + ".json")
        if item_path.exists():
            raise ValueError("Checkpoint already registered; refusing to rewrite an immutable save")
        atomic_json(item_path, {"step": int(checkpoint.name), "name": checkpoint.name,
                               "status": "queued", "attempts": 0, "next_retry": 0,
                               "snapshot": fingerprint(checkpoint)})
    if start:
        start_worker(run)
    return item_path


def start_worker(run):
    state = state_dir(run)
    with (state / "worker.log").open("a") as output:
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker", "--run", str(run)],
                         stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                         start_new_session=True, close_fds=True)


class Remote:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.helper = Path(__file__).with_name("archive_remote.py").read_text()

    def request(self, action, relative, manifest):
        payload = {"action": action, "base": self.endpoint.base, "relative": relative, "manifest": manifest}
        command = ["ssh", "-T", *SSH_OPTIONS, self.endpoint.host, "python3 -c " + shlex.quote(self.helper)]
        result = subprocess.run(command, input=json.dumps(payload), capture_output=True, text=True, timeout=3600)
        try:
            response = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError("Archive SSH failed: " + result.stderr[-1500:]) from error
        if result.returncode or response.get("error"):
            raise RuntimeError(response.get("error") or result.stderr[-1500:])
        if response.get("sha256") != manifest_id(manifest):
            raise RuntimeError("Invalid remote archive acknowledgement")
        return response

    def upload(self, source, stage):
        command = rsync_options() + ["-e", shlex.join(["ssh", "-T", *SSH_OPTIONS]),
                                     "--", str(source) + "/", self.endpoint.host + ":" + stage + "/"]
        subprocess.run(command, check=True, timeout=24 * 3600, stdin=subprocess.DEVNULL)


def load_run(run):
    state = state_dir(run)
    config = read_json(state / "config.json")
    if config.get("version") != 1:
        raise ValueError("Unsupported archive state version")
    validate_keep(config["keep"])
    validate_destination(config["root"], Endpoint(**config["endpoint"]))
    expected = Path(config["root"]) / config["relative_run"]
    if Path(os.path.abspath(run)) != expected or expected.resolve() != expected:
        raise ValueError("Archive run moved or symlinked; refusing cleanup")
    return state, config


def entries(state):
    result = []
    for path in sorted((state / "queue").glob("*.json")):
        if path.is_symlink():
            raise ValueError("Unsafe archive queue entry")
        entry = read_json(path)
        if path.stem != entry["name"] or not path.stem.isdigit() or int(path.stem) != entry["step"]:
            raise ValueError("Invalid archive queue step")
        if entry["status"] not in ("queued", "verified", "deleted"):
            raise ValueError("Invalid archive queue status")
        result.append((path, entry))
    return sorted(result, key=lambda pair: pair[1]["step"])


def source_path(config, entry):
    return checkpoint_path(config["root"], config["relative_run"] + "/checkpoints/" + entry["name"])


def transfer(config, entry, remote):
    source = source_path(config, entry)
    manifest = entry.get("manifest") or file_manifest(source)
    if not REQUIRED.issubset(manifest):
        raise ValueError("Incomplete checkpoint")
    relative = source.relative_to(config["root"]).as_posix()
    prepared = remote.request("prepare", relative, manifest)
    if not prepared["complete"]:
        expected_stage = str(Path(remote.endpoint.base) / Path(relative).parent /
                             (".archive-" + source.name + "-" + manifest_id(manifest)))
        if prepared.get("stage") != expected_stage:
            raise ValueError("Unexpected remote staging path")
        remote.upload(source, prepared["stage"])
        remote.request("commit", relative, manifest)
    verify(source, manifest)  # Source was not rewritten while rsync was running.
    return manifest


def unfinished(run, state, config):
    items = entries(state)
    latest = {row["name"] for _, row in items[-config["keep"]:]} if config["keep"] != -1 else set()
    last = Path(run) / "checkpoints/last"
    if last.is_symlink():
        latest.add(last.resolve().name)
    return [row for _, row in items if row["status"] == "queued" or (
        config["keep"] != -1 and row["status"] == "verified" and row["name"] not in latest)]


def prune(run, config, state, remote):
    """Every candidate is re-hashed at BOTH ends immediately before local deletion."""
    items = entries(state)
    if config["keep"] == -1:
        return
    latest = {entry["name"] for _, entry in items[-config["keep"]:]}
    last = Path(run) / "checkpoints/last"
    if last.is_symlink():
        latest.add(last.resolve().name)
    for path, entry in items:
        if entry["name"] in latest or entry["status"] != "verified":
            continue
        source = source_path(config, entry)
        if not source.exists():
            # Crash after rmtree but before the durable receipt update.
            entry["status"] = "deleted"
            atomic_json(path, entry)
            continue
        verify(source, entry["manifest"])
        check_snapshot(source, entry)
        relative = source.relative_to(config["root"]).as_posix()
        remote.request("verify", relative, entry["manifest"])
        # Registration and deletion serialize: do not prune a newly protected last save.
        with locked(state / "state.lock"):
            fresh = entries(state)
            keep = read_json(state / "config.json")["keep"]
            if keep == -1 or entry["name"] in {row["name"] for _, row in fresh[-keep:]}:
                continue
            if last.is_symlink() and last.resolve() == source:
                continue
            source_path(config, entry)  # Recheck path/symlinks, including the run parents.
            check_snapshot(source, entry)
            shutil.rmtree(source)
            entry["status"] = "deleted"
            atomic_json(path, entry)
        log.info("Archived and pruned local checkpoint %s", source)


def run_pass(run, remote=None, now=None):
    state, config = load_run(run)
    remote = remote or Remote(Endpoint(**config["endpoint"]))
    now = time.time() if now is None else now
    for path, entry in entries(state):
        if entry["status"] in ("verified", "deleted") or entry.get("next_retry", 0) > now:
            continue
        try:
            # Persist the immutable manifest before any remote writes, including failed uploads.
            source = source_path(config, entry)
            check_snapshot(source, entry)
            manifest = file_manifest(source)
            check_snapshot(source, entry)
            if entry.get("manifest") is not None and entry["manifest"] != manifest:
                raise ValueError("Registered checkpoint changed after hashing")
            entry["manifest"] = manifest
            atomic_json(path, entry)
            log.info("Uploading checkpoint %s", source)
            transfer(config, entry, remote)
            entry.update(status="verified", next_retry=0, error="")
            log.info("SHA256 verified checkpoint %s", source)
        except Exception as error:
            attempts = entry.get("attempts", 0) + 1
            entry.update(status="queued", attempts=attempts, error=str(error),
                         next_retry=time.time() + min(300, 30 * 2 ** min(attempts - 1, 4)))
            log.warning("Checkpoint kept locally; will retry: %s: %s", entry["name"], error)
        atomic_json(path, entry)
    try:
        prune(run, config, state, remote)
    except Exception:
        log.exception("Cleanup skipped; local checkpoints retained")
    return len(unfinished(run, state, config))


def worker(run):
    state, _ = load_run(run)
    try:
        with locked(state / "worker.lock", nonblocking=True):
            while run_pass(run):
                time.sleep(10)
    except BlockingIOError:
        return  # Another process for this run already owns the durable queue.


def drain(run):
    state, config = load_run(run)
    while unfinished(run, state, config):
        start_worker(run)  # Also recovers a worker killed by a previous Slurm allocation.
        log.info("Waiting for checkpoint transfers; failures remain local. Log: %s", state / "worker.log")
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    enqueue = commands.add_parser("register")
    enqueue.add_argument("--checkpoint", required=True, type=Path)
    enqueue.add_argument("--keep", required=True, type=int)
    enqueue.add_argument("--destination", choices=("scratch", "scratch2"), required=True)
    for name in ("worker", "drain", "status"):
        commands.add_parser(name).add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "register":
        register(args.checkpoint, args.keep, args.destination)
    elif args.command == "worker":
        worker(args.run)
    elif args.command == "drain":
        drain(args.run)
    else:
        state, config = load_run(args.run)
        print(json.dumps({"config": config, "checkpoints": [row for _, row in entries(state)]}, indent=2))


if __name__ == "__main__":
    main()
