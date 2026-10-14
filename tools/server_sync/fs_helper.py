"""Filesystem probe, also sent through SSH (compatible with Python 3.6+)."""

import fnmatch
import json
import os
import sys


ROOT_PATTERNS = (
    "dataset", "dataset_*", "libero_dataset", "libero_small_dataset",
    "libero_10_old", "libero_original_dataset", "libero_dataset_prev", "models",
    "outputs", "outputs_*", "outputs_DP", "outputs_FSQ", "VLA_outputs",
    "VLA_single_outputs", "DP_outputs", "FSQ_outputs", "pi05_outputs", "skillVLA_outputs",
)


def relative_path(value, allow_root=False):
    if not isinstance(value, str) or len(value) > 4096 or "\x00" in value:
        raise ValueError("Invalid relative path")
    if not value and allow_root:
        return ""
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("Absolute paths and parent traversal are not allowed")
    if not any(fnmatch.fnmatchcase(parts[0], pattern) for pattern in ROOT_PATTERNS):
        raise ValueError("Only dataset/model/output roots from sync_server.sh are allowed")
    return "/".join(parts)


def safe_path(base, relative):
    base = os.path.abspath(os.path.expanduser(base))
    if base == os.path.sep:
        raise ValueError("A workspace directory is required, not the filesystem root")
    root = os.path.realpath(base)
    current = root
    for component in relative.split("/") if relative else ():
        current = os.path.join(current, component)
        if os.path.islink(current):
            raise ValueError("Symlink paths cannot be selected; choose the numeric checkpoint instead")
    if os.path.commonpath((root, os.path.realpath(current))) != root:
        raise ValueError("Path escapes the configured workspace")
    return current


def probe(base, relative, operation="list", offset=0, limit=200, search=""):
    relative = relative_path(relative, allow_root=operation == "list")
    path = safe_path(base, relative)
    if operation in ("stat", "mkdir"):
        if operation == "mkdir":
            os.makedirs(path, exist_ok=True)
        exists = os.path.exists(path)
        return {"path": relative, "real_path": path, "exists": exists,
                "kind": "directory" if os.path.isdir(path) else ("file" if os.path.isfile(path) else "special"),
                "size": os.path.getsize(path) if exists and os.path.isfile(path) else None}
    if operation != "list":
        raise ValueError("Unknown probe operation")
    if not os.path.isdir(path):
        raise ValueError("Folder does not exist: " + path)
    entries = []
    with os.scandir(path) as children:
        for child in children:
            if not relative and (not child.is_dir(follow_symlinks=False) or not any(
                fnmatch.fnmatchcase(child.name, pattern) for pattern in ROOT_PATTERNS
            )):
                continue
            if search.casefold() not in child.name.casefold():
                continue
            try:
                kind = "symlink" if child.is_symlink() else (
                    "directory" if child.is_dir(follow_symlinks=False) else ("file" if child.is_file(follow_symlinks=False) else "special")
                )
                info = child.stat(follow_symlinks=False)
                entries.append({"name": child.name,
                    "path": relative + "/" + child.name if relative else child.name,
                    "kind": kind, "size": info.st_size if kind == "file" else None})
            except FileNotFoundError:
                continue
    entries.sort(key=lambda entry: (entry["kind"] != "directory", entry["name"].casefold()))
    offset, limit = max(0, int(offset)), max(1, min(500, int(limit)))
    return {"path": relative, "entries": entries[offset:offset + limit],
            "total": len(entries), "offset": offset, "limit": limit}


if __name__ == "__main__":
    try:
        print(json.dumps(probe(*json.loads(sys.argv[1])), ensure_ascii=False))
    except (ValueError, OSError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False))
        sys.exit(1)
