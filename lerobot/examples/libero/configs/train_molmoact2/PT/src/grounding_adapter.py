"""Apply pi05's existing episode-XYZ transform before original Molmo normalization.

The hooks are installed only in a MolmoAct2 training process and its data workers.
No raw parquet, stats.json, or third-party source is rewritten.
"""
from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from olmo.data import data_loader, lerobot_wrapper
from launch_scripts.lerobot_utils import stats as mixture_stats


def _shared_grounding():
    name = "sbd_shared_proprio_grounding"
    if name not in sys.modules:
        project = Path(__file__).resolve().parents[7]
        path = project / "lerobot/src/lerobot/datasets/proprio_grounding.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def artifact_for_metadata(meta):
    return _shared_grounding().get_episode_start_xyz_artifact(SimpleNamespace(root=meta.root, meta=meta))


class GroundedMetadata(LeRobotDatasetMetadata):
    """Provide GLOBAL grounded state statistics to Molmo's original stats collector."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        artifact = artifact_for_metadata(self)
        self.stats = {**self.stats, "observation.state": artifact.state_stats}


_BaseDataset = lerobot_wrapper._MolmoLeRobotDataset
_base_worker_init = data_loader._init_fn


class GroundedMolmoDataset(_BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        artifact = artifact_for_metadata(self.meta)
        self._sbd_references = artifact.references
        self.meta = copy.copy(self.meta)
        self.meta.stats = {**self.meta.stats, "observation.state": artifact.state_stats}

    def __getitem__(self, index):
        item = super().__getitem__(index)
        episode = int(item["episode_index"].item())
        return {**item, "observation.state": _shared_grounding().ground_state_xyz(
            item["observation.state"], self._sbd_references[episode],
        )}


def worker_init(worker_id):
    _base_worker_init(worker_id)
    # Upstream wrappers drop and lazily reopen their dataset when spawned.
    # Reinstall before the first fetch so those reopened datasets are also grounded.
    install(os.environ.get("SBD_MOLMOACT2_GROUNDING", "none"))


def install(mode: str) -> None:
    if mode == "none":
        return
    if mode != "episode_start_xyz":
        raise ValueError(f"Unsupported proprio grounding mode {mode!r}.")
    os.environ["SBD_MOLMOACT2_GROUNDING"] = mode
    lerobot_wrapper._MolmoLeRobotDataset = GroundedMolmoDataset
    mixture_stats.LeRobotDatasetMetadata = GroundedMetadata
    data_loader._init_fn = worker_init
