"""Retain full upstream checkpoints: newest N plus explicitly protected updates."""
from __future__ import annotations

import logging
from pathlib import Path
import shutil

from molmoact2_config import is_training_checkpoint


def prune_checkpoints(run: Path, keep: int, protected_steps) -> list[Path]:
    if keep == -1:
        return []
    if keep < 1:
        raise ValueError("keep_checkpoints must be -1 or positive")
    root = run / "checkpoints"
    # Never follow symlinks (including last) or touch foreign / partial checkpoints.
    paths = sorted((p for p in root.iterdir() if p.name.isdigit()
                    and not p.is_symlink() and p.is_dir() and is_training_checkpoint(p)),
                   key=lambda p: int(p.name))
    retained = set(paths[-keep:])
    protected = set(protected_steps)
    removed = []
    for path in paths:
        if path in retained or int(path.name) in protected:
            continue
        if path.parent.resolve() != root.resolve():
            raise ValueError(f"Refusing to prune outside checkpoint directory: {path}")
        logging.info("Pruning full Molmo checkpoint %s", path)
        shutil.rmtree(path)
        removed.append(path)
    return removed
