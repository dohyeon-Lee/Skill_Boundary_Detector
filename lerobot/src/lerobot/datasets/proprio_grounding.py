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
import pyarrow.parquet as pq
import torch

from lerobot.datasets.compute_stats import aggregate_feature_stats
from lerobot.utils.constants import OBS_STATE


logger = logging.getLogger(__name__)

PROPRIO_GROUNDING_NONE = "none"
PROPRIO_GROUNDING_EPISODE_START_XYZ = "episode_start_xyz"
SUPPORTED_PROPRIO_GROUNDING = (
    PROPRIO_GROUNDING_NONE,
    PROPRIO_GROUNDING_EPISODE_START_XYZ,
)

_CACHE_SCHEMA_VERSION = 1
_CACHE_FILENAME = "episode_start_xyz_grounding_v1.npz"


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


def _episode_metadata_parquet_paths(root: Path) -> list[Path]:
    paths = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    legacy_path = root / "meta" / "episodes.parquet"
    if not paths and legacy_path.is_file():
        paths = [legacy_path]
    if not paths:
        raise FileNotFoundError(
            f"No episode metadata parquet found under {root / 'meta' / 'episodes'}."
        )
    return paths


def _read_episode_state_stats(dataset) -> pd.DataFrame:
    """Read full per-episode state stats omitted by the runtime metadata view.

    ``LeRobotDatasetMetadata.episodes`` is a Hugging Face Dataset from which
    ``load_episodes`` deliberately removes every ``stats/*`` column. The source
    parquet files retain those columns and are small (one row per episode), so
    read only the state statistics needed for the grounded global normalizer.
    """
    paths = _episode_metadata_parquet_paths(Path(dataset.root))
    prefix = f"stats/{OBS_STATE}/"

    # Keep optional quantiles only when every shard contains them. The five
    # basic statistics are mandatory and sufficient for Diffusion MIN_MAX.
    columns_per_path = [set(pq.ParquetFile(path).schema_arrow.names) for path in paths]
    common_columns = set.intersection(*columns_per_path)
    if "episode_index" not in common_columns:
        raise ValueError("Episode metadata parquet lacks episode_index.")
    stat_columns = sorted(column for column in common_columns if column.startswith(prefix))
    required = {f"{prefix}{name}" for name in ("min", "max", "mean", "std", "count")}
    if not required.issubset(stat_columns):
        missing = sorted(required.difference(stat_columns))
        raise ValueError(
            "Episode metadata parquet lacks state statistics required for grounding: "
            f"{missing}."
        )

    columns = ["episode_index", *stat_columns]
    frame = pd.concat(
        [pd.read_parquet(path, columns=columns) for path in paths],
        ignore_index=True,
    )
    episode_ids = frame["episode_index"].astype(np.int64)
    if episode_ids.duplicated().any():
        duplicates = sorted(episode_ids[episode_ids.duplicated()].unique().tolist())
        raise ValueError(f"Duplicate episode metadata rows: {duplicates[:10]}.")

    expected_ids = {int(value) for value in dataset.meta.episodes["episode_index"]}
    actual_ids = set(episode_ids.tolist())
    missing = sorted(int(value) for value in expected_ids.difference(actual_ids))
    extra = sorted(int(value) for value in actual_ids.difference(expected_ids))
    if missing or extra:
        raise ValueError(
            "Episode state-stat metadata mismatch: "
            f"missing={missing[:10]}, extra={extra[:10]}."
        )
    return frame.sort_values("episode_index").reset_index(drop=True)


def _read_episode_start_references(dataset) -> dict[int, np.ndarray]:
    references: dict[int, np.ndarray] = {}
    for path in _data_parquet_paths(Path(dataset.root)):
        frame = pd.read_parquet(
            path,
            columns=["episode_index", "frame_index", OBS_STATE],
        )
        starts = frame[frame["frame_index"] == 0]
        for episode_id, state in zip(
            starts["episode_index"], starts[OBS_STATE], strict=True
        ):
            episode_id = int(episode_id)
            reference = np.asarray(state, dtype=np.float32)
            if reference.ndim != 1 or len(reference) < 3:
                raise ValueError(
                    f"Episode {episode_id} has invalid {OBS_STATE} shape {reference.shape}."
                )
            if episode_id in references:
                raise ValueError(f"Episode {episode_id} has more than one frame_index=0 row.")
            references[episode_id] = reference[:3].copy()

    expected_ids = {int(value) for value in dataset.meta.episodes["episode_index"]}
    missing = sorted(expected_ids.difference(references))
    extra = sorted(set(references).difference(expected_ids))
    if missing or extra:
        raise ValueError(
            "Episode-start grounding reference mismatch: "
            f"missing={missing[:10]}, extra={extra[:10]}."
        )
    return references


def _grounded_state_stats(dataset, references: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    prefix = f"stats/{OBS_STATE}/"
    episode_metadata = _read_episode_state_stats(dataset)
    columns = [column for column in episode_metadata.columns if column.startswith(prefix)]

    episode_stats: list[dict[str, np.ndarray]] = []
    for _, row in episode_metadata.iterrows():
        episode_id = int(row["episode_index"])
        reference = references[episode_id]
        stats: dict[str, np.ndarray] = {}
        for column in columns:
            name = column.removeprefix(prefix)
            value = np.asarray(row[column]).copy()
            if name != "count":
                if value.ndim != 1 or len(value) < 3:
                    raise ValueError(
                        f"Episode {episode_id} state stat {name!r} has invalid shape {value.shape}."
                    )
                if name != "std":
                    value[:3] -= reference.astype(value.dtype, copy=False)
            stats[name] = value
        episode_stats.append(stats)
    return aggregate_feature_stats(episode_stats)


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
        references = _read_episode_start_references(dataset)
        artifact = EpisodeStartXYZArtifact(
            references=references,
            state_stats=_grounded_state_stats(dataset, references),
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
