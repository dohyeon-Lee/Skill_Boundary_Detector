"""Bound reusable venv archives without touching upload staging or run history."""
from __future__ import annotations

import fcntl
from pathlib import Path
import re
import sys


def prune_venv_archives(root: Path, current: Path) -> list[Path]:
    # Archive readers take the same lock (see stage_node_local_venv). Queued
    # launches can safely fall back to the shared venv if an old archive is gone.
    archives = sorted(
        (p for p in root.glob("venv-*.tar.zst")
         if not p.is_symlink() and re.fullmatch(r"venv-[0-9a-f]{16}\.tar\.zst", p.name)),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    previous = [p for p in archives if p != current][:1]
    keep = {current, *previous}
    removed = []
    for archive in archives:
        if archive in keep:
            continue
        with Path(str(archive) + ".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            archive.unlink(missing_ok=True)
            for suffix in (".size", ".unpacked_size"):
                Path(str(archive) + suffix).unlink(missing_ok=True)
            removed.append(archive)
    return removed


if __name__ == "__main__":
    prune_venv_archives(Path(sys.argv[1]), Path(sys.argv[2]))
