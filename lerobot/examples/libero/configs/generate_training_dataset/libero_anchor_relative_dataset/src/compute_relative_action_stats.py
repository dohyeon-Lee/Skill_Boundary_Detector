#!/usr/bin/env python3
"""Compute SO(3)-correct anchor-relative action statistics for derived LIBERO data.

The derived dataset stores absolute EEF commands.  At training time a sampled
chunk is represented relative to its observation anchor::

    rel_k(t) = relative(action[t + k], observation.state[t])

Position uses subtraction, orientation uses
``Log(R_target @ R_anchor.T)``, and the gripper command remains absolute.  This
script pools every valid ``(t, k)`` pair up to ``chunk_size`` and writes a
sidecar ``meta/relative_action_stats.json``.  It never changes the dataset or
its ordinary ``meta/stats.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_LEROBOT_SRC = Path(__file__).resolve().parents[6] / "src"
sys.path.insert(0, str(_LEROBOT_SRC))
from lerobot.processor.eef_relative_action_processor import to_eef_relative_actions  # noqa: E402

QUANTILE_SAMPLE_CAP = 20_000_000


def load_state_action(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = sorted((dataset_dir / "data").rglob("file-*.parquet")) or sorted(
        (dataset_dir / "data").rglob("episode_*.parquet")
    )
    if not files:
        raise ValueError(f"No parquet data under {dataset_dir / 'data'}")
    columns = ["observation.state", "action", "episode_index", "frame_index"]
    frame = pd.concat(
        [pd.read_parquet(path, columns=columns) for path in files], ignore_index=True
    ).sort_values(["episode_index", "frame_index"], kind="stable")
    states = np.stack(frame["observation.state"].to_numpy()).astype(np.float64)
    actions = np.stack(frame["action"].to_numpy()).astype(np.float64)
    return states, actions, frame["episode_index"].to_numpy()


def episode_slices(episode_index: np.ndarray) -> list[slice]:
    if len(episode_index) == 0:
        return []
    starts = np.flatnonzero(np.r_[True, episode_index[1:] != episode_index[:-1]])
    ends = np.r_[starts[1:], len(episode_index)]
    return [slice(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def compute_relative_stats(
    states: np.ndarray,
    actions: np.ndarray,
    episode_index: np.ndarray,
    chunk_size: int,
) -> tuple[dict[str, np.ndarray], int, int]:
    """Return exact moments/extrema and sampled quantiles over all valid chunks."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if states.ndim != 2 or states.shape[1] != 8:
        raise ValueError(f"Expected observation.state shape (N,8), got {states.shape}")
    if actions.ndim != 2 or actions.shape != (len(states), 7):
        raise ValueError(f"Expected action shape (N,7), got {actions.shape}")
    if len(episode_index) != len(states):
        raise ValueError("episode_index length does not match state/action rows")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("State/action contains NaN or Inf")

    slices = episode_slices(episode_index)
    pool_size = sum(
        max(0, (episode.stop - episode.start) - offset) for episode in slices for offset in range(chunk_size)
    )
    if pool_size == 0:
        raise ValueError("No valid (anchor, offset) pairs")
    stride = max(1, int(np.ceil(pool_size / QUANTILE_SAMPLE_CAP)))

    action_dim = actions.shape[1]
    minimum = np.full(action_dim, np.inf, dtype=np.float64)
    maximum = np.full(action_dim, -np.inf, dtype=np.float64)
    total = np.zeros(action_dim, dtype=np.float64)
    total_squared = np.zeros(action_dim, dtype=np.float64)
    count = 0
    quantile_chunks: list[np.ndarray] = []

    for episode_number, episode in enumerate(slices):
        episode_actions = torch.from_numpy(actions[episode])
        episode_states = torch.from_numpy(states[episode])
        length = len(episode_actions)
        for offset in range(min(chunk_size, length)):
            relative = to_eef_relative_actions(
                episode_actions[offset:], episode_states[: length - offset]
            ).numpy()
            minimum = np.minimum(minimum, relative.min(axis=0))
            maximum = np.maximum(maximum, relative.max(axis=0))
            total += relative.sum(axis=0)
            total_squared += np.square(relative).sum(axis=0)
            count += len(relative)
            phase = (episode_number * 31 + offset) % stride
            sampled = relative[phase::stride]
            if len(sampled):
                quantile_chunks.append(sampled.astype(np.float32))

    pooled = np.concatenate(quantile_chunks, axis=0)
    mean = total / count
    variance = np.maximum(total_squared / count - np.square(mean), 0.0)
    q01, q99 = np.quantile(pooled, [0.01, 0.99], axis=0)
    stats = {
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "std": np.sqrt(variance),
        "q01": q01.astype(np.float64),
        "q99": q99.astype(np.float64),
    }
    return stats, stride, count


def validate_action_contract(dataset_dir: Path) -> dict[str, object]:
    path = dataset_dir / "meta" / "action_contract.json"
    if not path.is_file():
        raise ValueError(f"Missing derived-dataset action contract: {path}")
    contract = json.loads(path.read_text())
    expected = {
        "storage_representation": "absolute_eef_command",
        "model_representation": "eef_anchor_relative_so3",
        "rotation_representation": "axis_angle_rotation_vector",
        "rotation_composition": "left_world",
    }
    mismatches = {
        key: (contract.get(key), value) for key, value in expected.items() if contract.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Unsupported action_contract.json values: {mismatches}")
    return contract


def compute_and_write(
    dataset_dir: Path,
    *,
    chunk_size: int,
    overwrite: bool = False,
) -> Path:
    """Compute the sidecar if needed and return its path."""
    dataset_dir = Path(dataset_dir)
    contract = validate_action_contract(dataset_dir)
    output = dataset_dir / "meta" / "relative_action_stats.json"
    if output.exists() and not overwrite:
        previous = json.loads(output.read_text())
        if (
            previous.get("representation") == "eef_anchor_relative_so3"
            and int(previous.get("chunk_size", -1)) == chunk_size
        ):
            print(f"[relative-stats] up-to-date (chunk={chunk_size}): {output}")
            return output

    states, actions, episode_index = load_state_action(dataset_dir)
    stats, stride, pair_count = compute_relative_stats(states, actions, episode_index, chunk_size)
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    action_names = (info.get("features", {}).get("action") or {}).get("names")
    payload = {
        "schema_version": 1,
        "representation": "eef_anchor_relative_so3",
        "storage_representation": "absolute_eef_command",
        "rotation_representation": "axis_angle_rotation_vector",
        "rotation_composition": "left_world",
        "gripper": "absolute_-1_or_+1",
        "chunk_size": chunk_size,
        "action_names": action_names,
        "quantile_anchor_stride": stride,
        "num_frames": int(len(actions)),
        "num_anchor_offset_pairs": int(pair_count),
        "osc_position_scale": float(contract["osc_position_scale"]),
        "osc_rotation_scale": float(contract["osc_rotation_scale"]),
        "action": {key: np.asarray(value, dtype=np.float64).tolist() for key, value in stats.items()},
    }
    output.write_text(json.dumps(payload, indent=2))
    print(
        f"[relative-stats] {dataset_dir.name}: {len(actions)} frames, "
        f"{pair_count} anchor/offset pairs, chunk={chunk_size}"
    )
    print(f"[relative-stats] wrote {output} (meta/stats.json untouched)")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    compute_and_write(
        args.root / args.dataset,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
