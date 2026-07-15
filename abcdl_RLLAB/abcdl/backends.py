"""Uniform file access for local / HTTP URIs (so readers can stream). S3 out of scope."""

from __future__ import annotations

import contextlib
import os
from typing import Optional


def _is_remote(uri: str) -> bool:
    return "://" in uri and not uri.startswith("file://")


def _fsspec_open(uri: str, mode: str):
    import fsspec
    return fsspec.open(uri, mode)


@contextlib.contextmanager
def open_file(uri: str, mode: str = "rb"):
    if not _is_remote(uri):
        path = uri[len("file://"):] if uri.startswith("file://") else uri
        with open(path, mode) as f:
            yield f
        return
    with _fsspec_open(uri, mode) as f:
        yield f


def local_path_for(uri: str, cache_dir: Optional[str] = None) -> str:
    if not _is_remote(uri):
        return uri[len("file://"):] if uri.startswith("file://") else uri
    import fsspec

    dest_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "abcdl")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(uri.rstrip("/")))
    fs, _, paths = fsspec.get_fs_token_paths(uri)
    fs.get(paths[0], dest)
    return dest
