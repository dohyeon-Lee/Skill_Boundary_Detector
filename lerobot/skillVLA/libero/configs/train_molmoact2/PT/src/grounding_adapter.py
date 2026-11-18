"""Reuse pi05's grounding and exact global stats without importing its LeRobot."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _shared_grounding():
    name = "sbd_shared_proprio_grounding"
    if name not in sys.modules:
        project = Path(__file__).resolve().parents[7]
        spec = importlib.util.spec_from_file_location(
            name, project / "lerobot/src/lerobot/datasets/proprio_grounding.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class GroundedMolmoDataset(_shared_grounding().EpisodeStartXYZGroundedDataset):
    """Importable class so spawned data workers can unpickle this view."""


def ground_dataset(dataset, mode):
    if dataset is None or mode == "none":
        return dataset
    if mode != "episode_start_xyz":
        raise ValueError(f"Unsupported proprio grounding mode {mode!r}")
    return GroundedMolmoDataset(dataset)
