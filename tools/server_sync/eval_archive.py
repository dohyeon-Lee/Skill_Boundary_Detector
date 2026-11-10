"""Byte-verified result archives, keeping large evals below Hub file-count limits."""
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

ARCHIVE = "results.tar.gz"
MANIFEST = "archive_manifest.json"


class HashingReader:
    def __init__(self, stream, size):
        self.stream, self.size = stream, size
        self.sha, self.git = hashlib.sha256(), hashlib.sha1()
        self.git.update(b"blob " + str(size).encode() + b"\0")
        self.count = 0

    def read(self, count=-1):
        data = self.stream.read(count)
        self.sha.update(data)
        self.git.update(data)
        self.count += len(data)
        return data

    def result(self):
        if self.count != self.size:
            raise ValueError("Result changed while archiving")
        return {"size": self.size, "sha256": self.sha.hexdigest(), "git_sha1": self.git.hexdigest()}


def file_hashes(path):
    with path.open("rb") as stream:
        reader = HashingReader(stream, path.stat().st_size)
        while reader.read(8 * 1024 * 1024):
            pass
    return reader.result()


def pack(folder, destination):
    folder, destination = Path(folder), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / ".results.tmp"
    files = {}
    with temporary.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as archive:
            for path in sorted(folder.rglob("*")):
                if path.is_symlink():
                    raise ValueError("Result symlinks cannot be archived")
                name = path.relative_to(folder).as_posix()
                info = archive.gettarinfo(str(path), arcname=name)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                if path.is_dir():
                    archive.addfile(info)
                elif path.is_file():
                    before = path.stat()
                    with path.open("rb") as stream:
                        reader = HashingReader(stream, info.size)
                        archive.addfile(info, reader)
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                        raise ValueError("Result changed while archiving")
                    files[name] = reader.result()
                else:
                    raise ValueError("Only regular result files and directories can be archived")
    if not files:
        raise ValueError("Empty result archive")
    temporary.replace(destination / ARCHIVE)
    payload = {"schema": 1, "kind": "eval_result_archive", "files": files,
               "archive": {"filename": ARCHIVE, **file_hashes(destination / ARCHIVE)}}
    manifest = destination / ".manifest.tmp"
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n")
    manifest.replace(destination / MANIFEST)
    return payload


def validate_manifest(payload, expected):
    if (payload.get("schema") != 1 or payload.get("kind") != "eval_result_archive"
            or payload.get("files") != expected or payload.get("archive", {}).get("filename") != ARCHIVE):
        raise ValueError("HF archive inventory differs from the Pod")
    return payload["archive"]


def unpack(archive_path, destination, expected):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    seen = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            if (relative.is_absolute() or ".." in relative.parts or not relative.parts
                    or "\\" in member.name or relative.as_posix() != member.name):
                raise ValueError("Unsafe result archive path")
            target = destination / member.name
            if not target.resolve().is_relative_to(destination.resolve()) or target.is_symlink():
                raise ValueError("Result archive escapes its destination")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError("Unexpected result archive member")
            if member.size != expected[member.name]["size"]:
                raise ValueError("Result archive member size mismatch")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("wb") as stream:
                reader = HashingReader(source, member.size)
                for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                    stream.write(chunk)
            if reader.result() != expected[member.name]:
                raise ValueError("Result archive member hash mismatch")
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError("Incomplete result archive")
