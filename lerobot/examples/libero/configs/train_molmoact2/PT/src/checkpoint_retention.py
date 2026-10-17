"""Protect selected native Molmo training checkpoints without editing upstream code.

Only the regular sharded checkpoint cleanup is wrapped. Saving (including model,
optimizer and RNG state), distributed deletion barriers and training are native.
"""
from __future__ import annotations

from functools import wraps
import logging
from pathlib import Path
import re


log = logging.getLogger(__name__)


def removal_indices(checkpoints, save_folder, keep: int, protected_steps) -> list[int]:
    """Old registered checkpoints outside (latest N UNION protected steps).

    Do not scan disk or touch other runs. Resolve every registered path before
    permitting any deletion, including paths restored from a trainer checkpoint.
    """
    if keep <= 0:
        return []
    root = Path(save_folder).resolve()
    steps = []
    for checkpoint in checkpoints:
        path = Path(checkpoint)
        match = re.fullmatch(r"step(\d+)", path.name)
        if (match is None or path.is_symlink() or path.resolve().parent != root):
            raise ValueError(f"Refusing checkpoint cleanup outside native run checkpoints: {checkpoint}")
        steps.append(int(match[1]))
    latest = set(sorted(set(steps))[-keep:])
    protected = set(protected_steps)
    # Remove backwards so list indices remain valid. Native removal mutates the
    # trainer list and performs filesystem-rank-0 I/O between distributed barriers.
    return [index for index in reversed(range(len(steps)))
            if steps[index] not in latest and steps[index] not in protected]


def protected_removal(original, protected_steps):
    protected = frozenset(protected_steps)

    @wraps(original)
    def remove_checkpoints(self, current_checkpoints, num_checkpoints_to_keep):
        # Preserve upstream behavior for ephemeral, LoRA and model-only saves.
        if not protected or current_checkpoints is not self.checkpoints:
            return original(self, current_checkpoints, num_checkpoints_to_keep)
        indices = removal_indices(current_checkpoints, self.cfg.save_folder,
                                  num_checkpoints_to_keep, protected)
        for index in indices:
            log.info("Removing old checkpoint %s (protected steps: %s)",
                     current_checkpoints[index], sorted(protected))
            self._remove_sharded_checkpoint(index, current_checkpoints)

    remove_checkpoints._sbd_original_remove = original
    return remove_checkpoints


def install(protected_steps) -> None:
    """Called on every training rank before the original trainer is constructed."""
    from olmo.train.trainer import Trainer

    original = getattr(Trainer.remove_checkpoints, "_sbd_original_remove", Trainer.remove_checkpoints)
    # Installing a new policy never stacks wrappers or retains an old protection list.
    Trainer.remove_checkpoints = protected_removal(original, protected_steps) if protected_steps else original
