"""Pinned, size/hash-checked HF download with resumable partial files."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import time

import requests


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def session(use_proxy):
    client = requests.Session()
    client.trust_env = use_proxy
    from huggingface_hub import get_token
    if token := get_token():
        client.headers['Authorization'] = 'Bearer ' + token
    return client


def source_tree(client, repo, revision, timeout):
    url = f'https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true&limit=1000'
    files = []
    while url:
        response = client.get(url, timeout=(15, timeout))
        response.raise_for_status()
        files.extend(row for row in response.json() if row['type'] == 'file')
        url = response.links.get('next', {}).get('url')
    return files


def local_path(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe source path: {name!r}')
    result = Path(root) / str(path)
    if not result.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f'Source path leaves raw directory: {name!r}')
    return result


def valid(path, row):
    if not path.is_file() or path.stat().st_size != row['size']:
        return False
    expected = row.get('lfs', {}).get('oid')
    return not expected or sha256(path) == expected


def download_one(client, repo, revision, row, root, retries, timeout):
    path = local_path(root, row['path'])
    if valid(path, row):
        print(f"Reuse: {row['path']}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.part')
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{row['path']}?download=true"
    failures = 0
    integrity_failures = 0
    reported = -1
    while True:
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > row['size']:
            partial.unlink()
            offset = 0
        try:
            if offset == row['size'] and valid(partial, row):
                os.replace(partial, path)
                return
            if offset == row['size']:
                partial.unlink()
                offset = 0
            # Bounded ranges avoid long-lived CDN flows that this lab network
            # repeatedly resets. Each finished range is immediately reusable.
            end = min(row['size'] - 1, offset + 1024 * 1024 - 1)
            headers = {'Range': f'bytes={offset}-{end}'}
            with client.get(url, headers=headers, stream=True, timeout=(15, timeout)) as response:
                response.raise_for_status()
                ranged = response.status_code == 206
                if ranged:
                    if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Server returned the wrong resume range')
                else:
                    offset = 0  # A server ignoring Range is safely restarted, never appended.
                if reported < 0 or offset - reported >= 32 * 1024 * 1024 or failures:
                    print(f"Download: {row['path']} ({offset / 2**20:.1f}/{row['size'] / 2**20:.1f} MiB)", flush=True)
                    reported = offset
                with partial.open('ab' if offset else 'wb') as stream:
                    for block in response.iter_content(64 * 1024):
                        stream.write(block)
            failures = 0
            if ranged and partial.stat().st_size < row['size']:
                if partial.stat().st_size <= offset:
                    raise ValueError('Empty resume response')
                continue
            if not valid(partial, row):
                if partial.stat().st_size >= row['size']:
                    partial.unlink()
                integrity_failures += 1
                if integrity_failures > retries:
                    raise RuntimeError(f"Repeated checksum mismatch: {path}")
                raise ValueError(f"Size/checksum mismatch: {row['path']}")
            os.replace(partial, path)
            return
        except (requests.RequestException, ValueError) as error:
            if failures == retries:
                raise RuntimeError(f"Download failed; partial retained: {path}") from error
            delay = min(2 ** failures, 16)
            failures += 1
            print(f'Retry {failures}/{retries} in {delay}s: {type(error).__name__}', flush=True)
            time.sleep(delay)
