"""Opt-in VSA retention: recent resumable saves plus explicitly protected steps."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil

from safetensors import SafetensorError, safe_open


def validate_checkpoint_retention(keep, protected) -> list[int]:
    if isinstance(keep, bool) or not isinstance(keep, int) or (keep != -1 and keep < 1):
        raise ValueError("keep_checkpoints must be -1 or a positive integer.")
    if not isinstance(protected, list) or any(
        isinstance(step, bool) or not isinstance(step, int) or step < 0 for step in protected
    ):
        raise ValueError("protect_checkpoints must be a list of non-negative integers.")
    return sorted(set(protected))


def validate_retention_mode(keep, protected, archive, policy_type) -> list[int]:
    protected = validate_checkpoint_retention(keep, protected)
    if protected and (archive or policy_type != "skill_expert"):
        raise ValueError("protect_checkpoints is supported by local VSA retention without checkpoint_archive.")
    if keep != -1 and not archive and policy_type != "skill_expert":
        raise ValueError("Local checkpoint pruning requires checkpoint_archive outside VSA training.")
    return protected


def _valid_tensors(path: Path, *, allow_empty: bool = False) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    try:
        # Header/bounds verification only; never allocate/load the model tensors.
        with safe_open(path, framework="pt", device="cpu") as tensors:
            return allow_empty or bool(list(tensors.keys()))
    except (OSError, ValueError, SafetensorError):
        return False


def _is_training_checkpoint(path: Path) -> bool:
    """A partial save cannot replace a known-good recovery checkpoint."""
    if path.is_symlink() or not path.name.isdigit():
        return False
    pretrained, state = path / "pretrained_model", path / "training_state"
    if pretrained.is_symlink() or state.is_symlink():
        return False
    try:
        policy = json.loads((pretrained / "config.json").read_text())
        config = json.loads((pretrained / "train_config.json").read_text())
        if policy.get("type") != "skill_expert":
            return False
        if json.loads((state / "training_step.json").read_text())["step"] != int(path.name):
            return False
        json.loads((state / "optimizer_param_groups.json").read_text())
        if config.get("scheduler") is not None:
            json.loads((state / "scheduler_state.json").read_text())
        for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
            processor = json.loads((pretrained / name).read_text())
            for step in processor.get("steps", []):
                filename = step.get("state_file")
                if filename and (Path(filename).name != filename or not _valid_tensors(pretrained / filename)):
                    return False
        index = pretrained / "model.safetensors.index.json"
        shards = (set(json.loads(index.read_text())["weight_map"].values())
                  if index.is_file() else {"model.safetensors"})
        return bool(shards) and all(
            Path(name).name == name and _valid_tensors(pretrained / name) for name in shards
        ) and _valid_tensors(state / "optimizer_state.safetensors", allow_empty=int(path.name) == 0) \
            and _valid_tensors(state / "rng_state.safetensors")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def prune_local_checkpoints(cfg, completed_checkpoint: Path) -> list[Path]:
    """Call on rank zero, after a successful native save and `last` update.

    Local retention never runs on archived jobs. All paths and files are
    checked before deleting anything; partial/foreign saves are left alone.
    """
    keep = getattr(cfg, "keep_checkpoints", -1)
    if keep == -1 or getattr(cfg, "checkpoint_archive", ""):
        return []
    protected = validate_retention_mode(
        keep, getattr(cfg, "protect_checkpoints", []), "", cfg.policy.type,
    )
    root = Path(cfg.output_dir) / "checkpoints"
    if root.is_symlink():
        raise ValueError(f"Refusing to prune a symlinked checkpoint directory: {root}")
    current = Path(completed_checkpoint)
    if current.parent.resolve() != root.resolve():
        raise ValueError(f"Unexpected VSA checkpoint directory: {current}")
    last = root / "last"
    if not _is_training_checkpoint(current) or not last.is_symlink() or last.resolve() != current.resolve():
        logging.warning("Skipping VSA retention: new checkpoint or last link is incomplete: %s", current)
        return []
    paths = sorted(
        (path for path in root.iterdir() if path.is_dir() and _is_training_checkpoint(path)),
        key=lambda path: int(path.name),
    )
    retained = set(paths[-keep:]) | {current}
    removed = []
    for path in paths:
        if path in retained or int(path.name) in protected:
            continue
        logging.info("Pruning VSA checkpoint %s (latest %d plus protected %s)", path, keep, protected)
        try:
            shutil.rmtree(path)
        except OSError:
            logging.exception("Could not remove old VSA checkpoint %s; keeping the new saved checkpoint", path)
        else:
            removed.append(path)
    return removed
