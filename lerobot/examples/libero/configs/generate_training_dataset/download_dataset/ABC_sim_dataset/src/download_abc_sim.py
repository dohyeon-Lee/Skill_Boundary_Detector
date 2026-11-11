#!/usr/bin/env python3
"""Download and verify selected tasks from the official ABC Sim release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ABC_sim_dataset_config import (
    DEFAULT_CONFIG_PATH,
    dataset_root,
    load_config,
    selected_subsets,
)

MANIFEST_FORMAT = "abc_sim_dataset_manifest/v1"
SOURCE_MARKER = ".abc_sim_source.json"
CHUNK_SIZE = 8 * 1024 * 1024


def fetch_manifest(url: str, cache_path: Path, *, refresh: bool = False) -> dict[str, Any]:
    if refresh or not cache_path.is_file():
        request = urllib.request.Request(url, headers={"User-Agent": "SBD-ABC-Sim/1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(cache_path)
    manifest = json.loads(cache_path.read_text(encoding="utf-8"))
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(
            f"unexpected ABC Sim manifest format: {manifest.get('format')!r}"
        )
    if not isinstance(manifest.get("tasks"), dict) or not manifest["tasks"]:
        raise ValueError("ABC Sim manifest has no tasks")
    return manifest


def configured_tasks(specs: dict[str, dict[str, Any]]) -> list[str]:
    result: list[str] = []
    for subset_name, subset in specs.items():
        entries = subset.get("tasks") or []
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{subset_name}: tasks must be a non-empty list")
        for entry in entries:
            name = str(entry.get("task", "") if isinstance(entry, dict) else entry).strip()
            if not name:
                raise ValueError(f"{subset_name}: each task entry needs a task name")
            if name not in result:
                result.append(name)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def task_installed(task_dir: Path, task: dict[str, Any]) -> bool:
    marker_path = task_dir / SOURCE_MARKER
    if not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if marker.get("sha256") != task.get("sha256"):
        return False
    return all(
        (task_dir / f"{split}_sim").is_dir()
        for split, count in task.get("episodes", {}).items()
        if int(count) > 0
    )


def _download_with_resume(uri: str, destination: Path, expected_bytes: int, retries: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > expected_bytes:
        destination.unlink()
    attempt = 0
    while True:
        start = destination.stat().st_size if destination.exists() else 0
        if start == expected_bytes:
            return
        if start > expected_bytes:
            destination.unlink()
            start = 0
        headers = {"User-Agent": "SBD-ABC-Sim/1"}
        if start:
            headers["Range"] = f"bytes={start}-"
        request = urllib.request.Request(uri, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", 200)
                append = start > 0 and status == 206
                mode = "ab" if append else "wb"
                if start and not append:
                    start = 0
                written = start
                next_report = written + 1024**3
                with destination.open(mode) as stream:
                    while chunk := response.read(CHUNK_SIZE):
                        stream.write(chunk)
                        written += len(chunk)
                        if written >= next_report:
                            print(
                                f"    {written / 1024**3:.1f}/{expected_bytes / 1024**3:.1f} GiB",
                                flush=True,
                            )
                            next_report += 1024**3
        except (OSError, urllib.error.URLError) as exc:
            attempt += 1
            if attempt > retries:
                raise RuntimeError(f"download failed after {retries} retries: {uri}") from exc
            delay = min(60, 2 ** min(attempt, 5))
            print(f"    retry {attempt}/{retries} in {delay}s: {exc}", flush=True)
            time.sleep(delay)
            continue
        actual = destination.stat().st_size
        if actual > expected_bytes:
            destination.unlink()
            raise RuntimeError(f"server returned {actual} bytes, expected {expected_bytes}: {uri}")
        if actual == expected_bytes:
            return
        attempt += 1
        if attempt > retries:
            raise RuntimeError(f"short download after {retries} retries: {actual}/{expected_bytes}")


def _validate_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    for member in members:
        path = Path(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe tar member: {member.name!r}")
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"unsupported tar member type: {member.name!r}")
    return members


def _episode_count(root: Path, split: str) -> int:
    return sum(1 for _ in (root / f"{split}_sim").glob("episode_*/states_actions.bin"))


def install_task(
    name: str,
    task: dict[str, Any],
    *,
    raw_root: Path,
    download_root: Path,
    retries: int,
    keep_tar: bool,
    force: bool,
    manifest_generated: str,
) -> str:
    destination = raw_root / name
    if not force and task_installed(destination, task):
        return "skip"

    expected_bytes = int(task["bytes"])
    partial = download_root / f"{name}.tar.part"
    archive_path = download_root / f"{name}.tar"
    if force:
        partial.unlink(missing_ok=True)
        archive_path.unlink(missing_ok=True)
    if not archive_path.is_file() or archive_path.stat().st_size != expected_bytes:
        if archive_path.exists():
            archive_path.unlink()
        print(f"  [{name}] download {expected_bytes / 1024**3:.2f} GiB", flush=True)
        _download_with_resume(str(task["uri"]), partial, expected_bytes, retries)
        partial.replace(archive_path)
    actual_sha = _sha256(archive_path)
    if actual_sha != str(task["sha256"]):
        archive_path.unlink(missing_ok=True)
        raise ValueError(f"{name}: SHA256 mismatch: {actual_sha} != {task['sha256']}")

    temporary = raw_root / f".{name}.extracting"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        print(f"  [{name}] extract + validate", flush=True)
        with tarfile.open(archive_path, "r:*") as archive:
            members = _validate_members(archive)
            archive.extractall(temporary, members=members, filter="data")
        counts = {split: _episode_count(temporary, split) for split in ("train", "val")}
        expected = {key: int(value) for key, value in task["episodes"].items()}
        if counts != expected:
            raise ValueError(f"{name}: extracted episode counts {counts} != manifest {expected}")
        marker = {
            "format": MANIFEST_FORMAT,
            "manifest_generated": manifest_generated,
            "task": name,
            "uri": task["uri"],
            "bytes": expected_bytes,
            "sha256": task["sha256"],
            "episodes": expected,
            "frames": int(task["frames"]),
            "hours": float(task["hours"]),
        }
        (temporary / SOURCE_MARKER).write_text(json.dumps(marker, indent=2) + "\n")
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    if not keep_tar:
        archive_path.unlink(missing_ok=True)
    return "done"


def print_task_list(manifest: dict[str, Any]) -> None:
    print(f"ABC Sim tasks ({len(manifest['tasks'])}, manifest {manifest.get('generated', '?')}):")
    total_bytes = 0
    total_hours = 0.0
    for name, task in manifest["tasks"].items():
        episodes = task["episodes"]
        total_bytes += int(task["bytes"])
        total_hours += float(task["hours"])
        print(
            f"  {name:48s} {task['bytes'] / 1024**3:6.2f} GiB  "
            f"{episodes['train']:5d} train  {episodes['val']:4d} val  {task['hours']:6.2f} h"
        )
    print(f"  TOTAL{'':43s} {total_bytes / 1024**3:6.2f} GiB  {total_hours:.2f} h")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--only", default=os.environ.get("ABC_SIM_ONLY", ""))
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--refresh-manifest", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    root = dataset_root(config)
    manifest = fetch_manifest(
        str(config["abc_sim_manifest_url"]),
        root / "_sim_manifest.json",
        refresh=args.refresh_manifest,
    )
    if args.list_tasks:
        print_task_list(manifest)
        return

    specs = selected_subsets(config, args.only)
    if not specs:
        raise SystemExit("no abc_sim_subsets configured")
    wanted = configured_tasks(specs)
    missing = sorted(set(wanted) - set(manifest["tasks"]))
    if missing:
        raise SystemExit(f"tasks absent from official manifest: {missing}")
    bytes_total = sum(int(manifest["tasks"][name]["bytes"]) for name in wanted)
    print(
        f"ABC Sim download plan: {len(wanted)} task(s), {bytes_total / 1024**3:.2f} GiB -> "
        f"{root / '_sim_raw'}"
    )
    for name in wanted:
        task = manifest["tasks"][name]
        print(
            f"  {name}: {task['bytes'] / 1024**3:.2f} GiB, "
            f"{sum(task['episodes'].values())} episodes"
        )
    if args.dry_run:
        return

    raw_root = root / "_sim_raw"
    download_root = root / "_sim_downloads"
    raw_root.mkdir(parents=True, exist_ok=True)
    download_root.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for name in wanted:
        try:
            result = install_task(
                name,
                manifest["tasks"][name],
                raw_root=raw_root,
                download_root=download_root,
                retries=int(config.get("download_retries", 20)),
                keep_tar=bool(config.get("keep_downloaded_tars", False)),
                force=args.force,
                manifest_generated=str(manifest.get("generated", "")),
            )
            print(f"  [{name}] {result}", flush=True)
        except Exception as exc:  # one failed task must not hide completed downloads
            failures.append(name)
            print(f"  [{name}] FAILED: {exc}", flush=True)
    if failures:
        raise SystemExit(f"ABC Sim download failed for: {failures}")
    print("ABC Sim raw episode data ready.")


if __name__ == "__main__":
    main()
