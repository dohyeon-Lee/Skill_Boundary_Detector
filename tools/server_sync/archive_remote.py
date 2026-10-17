"""Portable Python 3.6+ archive verifier, also executed over SSH via stdin JSON.

Never deletes remote data or replaces an existing, different checkpoint.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import uuid

RECEIPT = ".sbd-checkpoint-archive.json"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".archive-write-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint_path(base, relative):
    root = Path(base)
    parts = PurePosixPath(relative).parts
    if (not root.is_absolute() or root == Path("/") or root.resolve() != root
            or not parts or str(PurePosixPath(relative)) != relative
            or any(part in (".", "..", "/") for part in parts)
            or len(parts) < 4 or not (parts[0] == "outputs" or parts[0].startswith("outputs_"))
            or parts[-2] != "checkpoints" or not re.fullmatch(r"\d{6,}", parts[-1])):
        raise ValueError("Unsafe checkpoint archive path: " + relative)
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Symlink in checkpoint archive path: " + str(current))
    return current


def file_manifest(directory):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Not a checkpoint directory: " + str(directory))
    result = {}
    for parent, dirs, files in os.walk(str(directory), followlinks=False):
        for name in dirs + files:
            if (Path(parent) / name).is_symlink():
                raise ValueError("Symlinks are not supported inside archived checkpoints")
        for name in files:
            path = Path(parent) / name
            relative = path.relative_to(directory).as_posix()
            if relative == RECEIPT:
                continue
            before = path.stat()
            if not path.is_file():
                raise ValueError("Non-regular checkpoint file: " + relative)
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise ValueError("Checkpoint changed while hashing: " + relative)
            result[relative] = {"size": after.st_size, "sha256": digest.hexdigest()}
    if not result:
        raise ValueError("Empty checkpoint")
    return result


def manifest_id(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def verify(directory, manifest):
    if file_manifest(directory) != manifest:
        raise ValueError("Checkpoint SHA256/file-list verification failed: " + str(directory))


def update_last(final):
    last = final.parent / "last"
    if last.is_symlink():
        target = os.readlink(str(last))
        if not re.fullmatch(r"\d{6,}", target):
            raise ValueError("Refusing to replace an unrelated remote last symlink")
        if int(target) > int(final.name) and last.exists():
            return
    elif last.exists():
        raise ValueError("Refusing to replace a real remote last directory/file")
    temporary = final.parent / (".archive-last-" + uuid.uuid4().hex)
    try:
        temporary.symlink_to(final.name)
        os.replace(str(temporary), str(last))
    finally:
        if temporary.is_symlink():
            temporary.unlink()


def process(request):
    final = checkpoint_path(request["base"], request["relative"])
    manifest = request["manifest"]
    identity = manifest_id(manifest)
    stage = final.parent / (".archive-" + final.name + "-" + identity)
    if stage.is_symlink():
        raise ValueError("Unsafe archive staging path")
    action = request["action"]
    if action not in ("prepare", "commit", "verify"):
        raise ValueError("Unknown archive operation")
    final.parent.mkdir(parents=True, exist_ok=True)
    lock_path = final.parent / ".archive.lock"
    if lock_path.is_symlink():
        raise ValueError("Unsafe archive lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if final.exists():
            # An exact pre-existing copy is safe, including copies made by the UI.
            # A different copy is a conflict, never permission to overwrite it.
            verify(final, manifest)
            if action != "verify":
                atomic_json(final / RECEIPT, {"sha256": identity, "manifest": manifest})
                update_last(final)
            return {"complete": True, "sha256": identity}
        if action == "verify":
            raise ValueError("Archived checkpoint is missing: " + str(final))
        if action == "prepare":
            stage.mkdir(exist_ok=True)
            return {"complete": False, "stage": str(stage), "sha256": identity}
        verify(stage, manifest)
        atomic_json(stage / RECEIPT, {"sha256": identity, "manifest": manifest})
        os.rename(str(stage), str(final))
        update_last(final)
        return {"complete": True, "sha256": identity}


if __name__ == "__main__":
    try:
        print(json.dumps(process(json.load(sys.stdin))))
    except Exception as error:
        print(json.dumps({"error": str(error)}))
        sys.exit(1)
