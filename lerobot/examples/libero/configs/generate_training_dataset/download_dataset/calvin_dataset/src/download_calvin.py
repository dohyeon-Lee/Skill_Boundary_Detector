#!/usr/bin/env python3
"""Download, verify, and extract official raw CALVIN play datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from calvin_dataset_config import (
    DEFAULT_CONFIG_PATH,
    as_bool,
    calvin_raw_root,
    load_config,
    selected_variants,
    variants,
)


MARKER_NAME = ".calvin_download_complete.json"
MARKER_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mismatch_path(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = path.with_name(f"{path.name}.sha256-mismatch-{stamp}")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.sha256-mismatch-{stamp}-{suffix}")
        suffix += 1
    return candidate


def _quarantine_bad_archive(path: Path) -> Path:
    destination = _mismatch_path(path)
    path.rename(destination)
    return destination


def _validate_layout(extracted_root: Path) -> None:
    training = extracted_root / "training"
    validation = extracted_root / "validation"
    annotations = training / "lang_annotations" / "auto_lang_ann.npy"
    missing: list[str] = []
    if not training.is_dir():
        missing.append("training/")
    if not validation.is_dir():
        missing.append("validation/")
    if training.is_dir() and next(training.glob("episode_*.npz"), None) is None:
        missing.append("training/episode_*.npz")
    if validation.is_dir() and next(validation.glob("episode_*.npz"), None) is None:
        missing.append("validation/episode_*.npz")
    if not annotations.is_file():
        missing.append("training/lang_annotations/auto_lang_ann.npy")
    if missing:
        raise RuntimeError(
            f"extracted CALVIN layout is incomplete in {extracted_root}: missing {', '.join(missing)}"
        )


def _expected_marker(variant: str, spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": MARKER_SCHEMA_VERSION,
        "variant": variant,
        "archive": str(spec["archive"]),
        "archive_sha256": str(spec["sha256"]),
        "extracted_dir": str(spec["extracted_dir"]),
    }


def _is_complete(extracted_root: Path, variant: str, spec: dict[str, Any]) -> bool:
    marker_path = extracted_root / MARKER_NAME
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        expected = _expected_marker(variant, spec)
        if any(marker.get(key) != value for key, value in expected.items()):
            return False
        _validate_layout(extracted_root)
    except (OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError):
        return False
    return True


def _write_marker(extracted_root: Path, variant: str, spec: dict[str, Any]) -> None:
    marker = {
        **_expected_marker(variant, spec),
        "url": str(spec["url"]),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    marker_path = extracted_root / MARKER_NAME
    temporary = marker_path.with_name(f"{marker_path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(marker_path)


def _validate_zip_members(archive: Path, expected_dir: str) -> None:
    expected_prefix = f"{expected_dir}/"
    has_expected_member = False
    try:
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                name = member.filename
                if "\x00" in name:
                    raise RuntimeError(f"archive contains a NUL byte in a member name: {name!r}")
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts:
                    raise RuntimeError(f"archive contains an unsafe path: {name!r}")
                unix_mode = member.external_attr >> 16
                if unix_mode and stat.S_ISLNK(unix_mode):
                    raise RuntimeError(f"archive contains a symbolic link: {name!r}")
                if name == expected_dir or name.startswith(expected_prefix):
                    has_expected_member = True
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"invalid ZIP archive: {archive}") from exc
    if not has_expected_member:
        raise RuntimeError(f"archive {archive} does not contain expected directory {expected_dir}/")


def _download_command(config: dict[str, Any], url: str, partial: Path) -> list[str]:
    retries = max(1, int(config.get("calvin_download_retries", 20)))
    connect_timeout = max(1, int(config.get("calvin_connect_timeout_seconds", 30)))
    stall_timeout = max(1, int(config.get("calvin_stall_timeout_seconds", 120)))
    if shutil.which("curl"):
        return [
            "curl",
            "--fail",
            "--location",
            "--continue-at",
            "-",
            "--retry",
            str(retries),
            "--retry-delay",
            "5",
            "--retry-connrefused",
            "--connect-timeout",
            str(connect_timeout),
            "--speed-limit",
            "1024",
            "--speed-time",
            str(stall_timeout),
            "--output",
            str(partial),
            url,
        ]
    if shutil.which("wget"):
        return [
            "wget",
            "--continue",
            f"--tries={retries}",
            f"--connect-timeout={connect_timeout}",
            f"--read-timeout={stall_timeout}",
            f"--output-document={partial}",
            url,
        ]
    raise RuntimeError("neither curl nor wget is installed; one is required for CALVIN download")


def _obtain_verified_archive(config: dict[str, Any], spec: dict[str, Any], archive: Path) -> None:
    expected_sha = str(spec["sha256"])
    partial = archive.with_name(f"{archive.name}.part")

    if archive.is_file():
        print(f"  verifying existing archive: {archive}", flush=True)
        if _sha256(archive) == expected_sha:
            print("  archive SHA-256: OK", flush=True)
            return
        quarantined = _quarantine_bad_archive(archive)
        print(f"  existing archive checksum mismatch; kept as {quarantined}", flush=True)

    print(f"  downloading: {spec['url']}", flush=True)
    if partial.exists():
        print(f"  resuming partial file: {partial} ({partial.stat().st_size:,} bytes)", flush=True)
    subprocess.run(_download_command(config, str(spec["url"]), partial), check=True)

    print(f"  verifying downloaded archive: {partial}", flush=True)
    actual_sha = _sha256(partial)
    if actual_sha != expected_sha:
        quarantined = _quarantine_bad_archive(partial)
        raise RuntimeError(
            "downloaded archive SHA-256 mismatch\n"
            f"  expected: {expected_sha}\n"
            f"  actual  : {actual_sha}\n"
            f"  kept as : {quarantined}"
        )
    partial.replace(archive)
    print("  archive SHA-256: OK", flush=True)


def _extract_archive(archive: Path, raw_root: Path, expected_dir: str) -> None:
    if not shutil.which("unzip"):
        raise RuntimeError("unzip is required to extract the official CALVIN archive")
    print("  checking ZIP member paths", flush=True)
    _validate_zip_members(archive, expected_dir)
    print(f"  extracting into: {raw_root}", flush=True)
    subprocess.run(["unzip", "-q", "-o", str(archive), "-d", str(raw_root)], check=True)


def download_variant(
    config: dict[str, Any],
    variant: str,
    spec: dict[str, Any],
    *,
    dry_run: bool = False,
) -> None:
    raw_root = calvin_raw_root(config)
    archive = raw_root / "archives" / str(spec["archive"])
    extracted_root = raw_root / str(spec["extracted_dir"])
    approximate_size = spec.get("approximate_size_gb", "?")

    print(f"\n[{variant}] {spec['archive']} (~{approximate_size} GB)", flush=True)
    print(f"  archive : {archive}", flush=True)
    print(f"  data    : {extracted_root}", flush=True)
    if dry_run:
        print("  dry-run: no files changed", flush=True)
        return

    raw_root.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    if _is_complete(extracted_root, variant, spec):
        print("  already downloaded, verified, and extracted; skipping", flush=True)
        return

    free_gb = shutil.disk_usage(raw_root).free / 1_000_000_000
    print(f"  free space at staging root: {free_gb:.1f} GB", flush=True)
    _obtain_verified_archive(config, spec, archive)
    _extract_archive(archive, raw_root, str(spec["extracted_dir"]))
    _validate_layout(extracted_root)
    _write_marker(extracted_root, variant, spec)
    print("  extracted CALVIN layout: OK", flush=True)

    if not as_bool(config.get("calvin_keep_archives", True)):
        archive.unlink()
        print(f"  removed verified archive to save space: {archive}", flush=True)


def _print_variants(config: dict[str, Any]) -> None:
    print("variant  size(GB)  train envs  eval env  archive")
    for name, spec in variants(config).items():
        train_envs = "+".join(str(item) for item in spec.get("train_environments", [])) or "?"
        print(
            f"{name:<8} {str(spec.get('approximate_size_gb', '?')):<9} "
            f"{train_envs:<11} {str(spec.get('eval_environment', '?')):<9} {spec['archive']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--only",
        default=os.environ.get("CALVIN_ONLY", ""),
        help="comma/space-separated variant override (also CALVIN_ONLY)",
    )
    parser.add_argument("--dry-run", action="store_true", help="show selected paths without changing files")
    parser.add_argument("--list-variants", action="store_true", help="list configured official variants and exit")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        available = variants(config)
        if args.list_variants:
            _print_variants(config)
            return
        chosen = selected_variants(config, args.only)
        print(f"CALVIN variants: {', '.join(chosen)}")
        print(f"CALVIN staging root: {calvin_raw_root(config)}")
        for name in chosen:
            download_variant(config, name, available[name], dry_run=args.dry_run)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
