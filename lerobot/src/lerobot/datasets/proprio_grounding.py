"""Episode-relative proprioceptive-state grounding for offline datasets."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.utils.constants import OBS_STATE


logger = logging.getLogger(__name__)

PROPRIO_GROUNDING_NONE = "none"
PROPRIO_GROUNDING_EPISODE_START_XYZ = "episode_start_xyz"
SUPPORTED_PROPRIO_GROUNDING = (
    PROPRIO_GROUNDING_NONE,
    PROPRIO_GROUNDING_EPISODE_START_XYZ,
)

# v2: state statistics are GLOBAL exact over every grounded frame (incl. quantiles).
_CACHE_SCHEMA_VERSION = 2
_CACHE_FILENAME = "episode_start_xyz_grounding_v2.npz"

# Same contract as generate_training_dataset/**/ensure_quantile_stats.py (fast mode) and the
# SkillVLA builder's exact_vector_stats: all frames pooled, float64, population std, np.quantile
# with its default linear interpolation. Per-episode quantiles are never averaged, because that
# pulls q01/q99 toward the centre.
_QUANTILES = (("q01", 0.01), ("q10", 0.10), ("q50", 0.50), ("q90", 0.90), ("q99", 0.99))


def global_exact_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    """Global exact statistics of an (N, D) array of non-video frames."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(f"Stats input must be a non-empty (N, D) array, got {array.shape}.")
    stats = {
        "min": array.min(axis=0),
        "max": array.max(axis=0),
        "mean": array.mean(axis=0),
        "std": array.std(axis=0),
        "count": np.array([array.shape[0]], dtype=np.int64),
    }
    for name, quantile in _QUANTILES:
        stats[name] = np.quantile(array, quantile, axis=0)
    return stats


def normalize_proprio_grounding(value: str | None) -> str:
    mode = str(value or PROPRIO_GROUNDING_NONE).strip().lower().replace("-", "_")
    mode = {"off": PROPRIO_GROUNDING_NONE, "false": PROPRIO_GROUNDING_NONE}.get(mode, mode)
    if mode not in SUPPORTED_PROPRIO_GROUNDING:
        raise ValueError(f"proprio_grounding must be none|episode_start_xyz, got {value!r}.")
    return mode


def ground_state_xyz(
    state: np.ndarray | torch.Tensor,
    reference_xyz: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Subtract one episode's raw first-frame XYZ from arbitrary state rows."""
    if state.shape[-1] < 3:
        raise ValueError(f"Proprio state must have at least three dimensions, got {state.shape}.")
    if isinstance(state, torch.Tensor):
        grounded = state.clone()
        reference = torch.as_tensor(reference_xyz, dtype=state.dtype, device=state.device)
        grounded[..., :3] -= reference
        return grounded
    grounded = np.asarray(state).copy()
    grounded[..., :3] -= np.asarray(reference_xyz, dtype=grounded.dtype)
    return grounded


@dataclass(frozen=True)
class EpisodeStartXYZArtifact:
    references: dict[int, np.ndarray]
    state_stats: dict[str, np.ndarray]


def _dataset_signature(dataset) -> str:
    digest = hashlib.sha256()
    digest.update(str(dataset.meta.total_frames).encode())
    digest.update(str(dataset.meta.total_episodes).encode())
    for name, value in sorted((dataset.meta.stats or {}).get(OBS_STATE, {}).items()):
        digest.update(name.encode())
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def _data_parquet_paths(root: Path) -> list[Path]:
    paths = sorted((root / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet data found under {root / 'data'}.")
    return paths


def _read_references_and_grounded_states(dataset) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Read every state row once: per-episode first-frame XYZ and all grounded states."""
    frames = pd.concat(
        [
            pd.read_parquet(path, columns=["episode_index", "frame_index", OBS_STATE])
            for path in _data_parquet_paths(Path(dataset.root))
        ],
        ignore_index=True,
    )
    episode_ids = frames["episode_index"].to_numpy(dtype=np.int64)
    states = np.stack(frames[OBS_STATE].to_numpy()).astype(np.float64)
    if states.ndim != 2 or states.shape[1] < 3:
        raise ValueError(f"{OBS_STATE} must have shape (N, D>=3), got {states.shape}.")

    references: dict[int, np.ndarray] = {}
    start_rows = np.flatnonzero(frames["frame_index"].to_numpy() == 0)
    for row in start_rows:
        episode_id = int(episode_ids[row])
        if episode_id in references:
            raise ValueError(f"Episode {episode_id} has more than one frame_index=0 row.")
        references[episode_id] = states[row, :3].astype(np.float32)

    expected_ids = {int(value) for value in dataset.meta.episodes["episode_index"]}
    missing = sorted(expected_ids.difference(references))
    extra = sorted(set(references).difference(expected_ids))
    if missing or extra:
        raise ValueError(
            "Episode-start grounding reference mismatch: "
            f"missing={missing[:10]}, extra={extra[:10]}."
        )

    # Subtract the same float32 reference the dataset view subtracts at load time.
    reference_rows = np.stack([references[int(episode_id)] for episode_id in episode_ids])
    grounded = states.copy()
    grounded[:, :3] -= reference_rows.astype(np.float64)
    return references, grounded


def _load_artifact(path: Path, signature: str) -> EpisodeStartXYZArtifact | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            if int(payload["schema_version"]) != _CACHE_SCHEMA_VERSION:
                return None
            if str(payload["dataset_signature"]) != signature:
                return None
            episode_ids = payload["episode_ids"].astype(np.int64)
            reference_xyz = payload["reference_xyz"].astype(np.float32)
            stat_names = [str(value) for value in payload["stat_names"]]
            references = {
                int(episode_id): reference.copy()
                for episode_id, reference in zip(episode_ids, reference_xyz, strict=True)
            }
            state_stats = {name: payload[f"stat_{name}"].copy() for name in stat_names}
        return EpisodeStartXYZArtifact(references, state_stats)
    except (KeyError, OSError, ValueError):
        return None


def _save_artifact(
    path: Path,
    signature: str,
    artifact: EpisodeStartXYZArtifact,
) -> None:
    episode_ids = np.asarray(sorted(artifact.references), dtype=np.int64)
    reference_xyz = np.stack(
        [artifact.references[int(episode_id)] for episode_id in episode_ids]
    ).astype(np.float32)
    stat_names = sorted(artifact.state_stats)
    payload = {
        "schema_version": np.array(_CACHE_SCHEMA_VERSION, dtype=np.int16),
        "dataset_signature": np.array(signature),
        "episode_ids": episode_ids,
        "reference_xyz": reference_xyz,
        "stat_names": np.asarray(stat_names),
        **{f"stat_{name}": artifact.state_stats[name] for name in stat_names},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.savez(file, **payload)
    temporary.replace(path)


def get_episode_start_xyz_artifact(dataset) -> EpisodeStartXYZArtifact:
    """Load or build the episode-reference and grounded-statistics sidecar."""
    cache_path = Path(dataset.root) / "meta" / _CACHE_FILENAME
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    signature = _dataset_signature(dataset)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        artifact = _load_artifact(cache_path, signature)
        if artifact is not None:
            logger.info("Loaded episode-start proprio grounding cache: %s", cache_path)
            return artifact
        references, grounded_states = _read_references_and_grounded_states(dataset)
        artifact = EpisodeStartXYZArtifact(
            references=references,
            state_stats=global_exact_stats(grounded_states),
        )
        _save_artifact(cache_path, signature, artifact)
        logger.info("Created episode-start proprio grounding cache: %s", cache_path)
        return artifact


class EpisodeStartXYZGroundedDataset(torch.utils.data.Dataset):
    """Dataset view whose observation.state is relative to its episode start."""

    def __init__(self, dataset):
        self.dataset = dataset
        artifact = get_episode_start_xyz_artifact(dataset)
        self.references = artifact.references
        all_stats = dict(dataset.meta.stats or {})
        all_stats[OBS_STATE] = artifact.state_stats
        # Expose grounded normalization statistics without mutating the raw
        # dataset view, which may still be used elsewhere in the same process.
        self.meta = copy.copy(dataset.meta)
        self.meta.stats = all_stats

    def __len__(self) -> int:
        return len(self.dataset)

    def _ground_item(self, item: dict) -> dict:
        episode_id = int(torch.as_tensor(item["episode_index"]).item())
        if episode_id not in self.references:
            raise KeyError(f"Missing episode-start XYZ for episode {episode_id}.")
        grounded = dict(item)
        grounded[OBS_STATE] = ground_state_xyz(item[OBS_STATE], self.references[episode_id])
        return grounded

    def __getitem__(self, index: int) -> dict:
        return self._ground_item(self.dataset[index])

    def get_raw_item(self, index: int) -> dict:
        return self._ground_item(self.dataset.get_raw_item(index))

    def __getattr__(self, name: str):
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)
