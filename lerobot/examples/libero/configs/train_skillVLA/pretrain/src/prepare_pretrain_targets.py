#!/usr/bin/env python3
"""Fit FAST on full variable-length skill trajectories and write a compact token target pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _parse_ranges(spec: str) -> tuple[list[tuple[int, int]], np.ndarray]:
    ranges = []
    indices = []
    for raw in spec.split(","):
        start, end = (int(value) for value in raw.strip().split(":"))
        if start < 0 or end <= start:
            raise ValueError(f"Invalid encoded dimension range: {raw!r}")
        ranges.append((start, end))
        indices.extend(range(start, end))
    return ranges, np.asarray(indices, dtype=np.int64)


def _normalize_actions(actions: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    denom = np.maximum(q99 - q01, 1e-8)
    clipped = np.clip(actions, q01, q99)
    return (2.0 * (clipped - q01) / denom - 1.0).astype(np.float32)


def load_skill_trajectories(
    repo_id: str,
    dataset_root: Path,
    transition_pack: Path,
    encoded_dims: str,
) -> tuple[list[np.ndarray], dict[str, np.ndarray], list[tuple[int, int]], np.ndarray, np.ndarray]:
    dataset = LeRobotDataset(repo_id=repo_id, root=str(dataset_root), video_keys_to_load=[])
    frames = dataset.hf_dataset.with_format("numpy")
    transitions = np.load(str(transition_pack))
    required = ("episode_id", "frame_start", "frame_end", "skill_code")
    missing = [key for key in required if key not in transitions]
    if missing:
        raise ValueError(f"Transition pack lacks provenance fields {missing}: {transition_pack}")

    provenance = {key: np.asarray(transitions[key], dtype=np.int64) for key in required}
    count = len(provenance["skill_code"])
    if any(len(values) != count for values in provenance.values()):
        raise ValueError(f"Transition pack provenance lengths disagree: {transition_pack}")

    ranges, dimension_indices = _parse_ranges(encoded_dims)
    action_stats = dataset.meta.stats["action"]
    q01 = np.asarray(action_stats["q01"], dtype=np.float32)[dimension_indices]
    q99 = np.asarray(action_stats["q99"], dtype=np.float32)[dimension_indices]

    trajectories: list[np.ndarray] = []
    lengths = np.empty(count, dtype=np.int32)
    for index in range(count):
        episode = int(provenance["episode_id"][index])
        frame_start = int(provenance["frame_start"][index])
        frame_end = int(provenance["frame_end"][index])
        episode_meta = dataset.meta.episodes[episode]
        absolute_start = int(episode_meta["dataset_from_index"]) + frame_start
        absolute_end = int(episode_meta["dataset_from_index"]) + frame_end
        if absolute_end < absolute_start:
            raise ValueError(
                f"Invalid segment {index}: episode={episode}, frames={frame_start}:{frame_end}"
            )
        raw = np.asarray(frames[absolute_start : absolute_end + 1]["action"], dtype=np.float32)
        if raw.ndim != 2 or raw.shape[0] != frame_end - frame_start + 1:
            raise ValueError(f"Action trajectory shape mismatch at segment {index}: {raw.shape}")
        selected = raw[:, dimension_indices]
        trajectories.append(_normalize_actions(selected, q01, q99))
        lengths[index] = raw.shape[0]
        if (index + 1) % 2000 == 0:
            print(f"  loaded {index + 1}/{count} skill trajectories")

    return trajectories, provenance, ranges, lengths, np.stack([q01, q99])


def fit_or_load_tokenizer(
    tokenizer_dir: Path,
    trajectories: list[np.ndarray],
    vocab_size: int,
    scale: float,
):
    from transformers import AutoProcessor  # noqa: PLC0415

    if tokenizer_dir.is_dir():
        print(f"Loading existing FAST tokenizer: {tokenizer_dir}")
        return AutoProcessor.from_pretrained(str(tokenizer_dir), trust_remote_code=True), False

    print("Loading lerobot/fast-action-tokenizer source and fitting variable-length trajectories...")
    base = AutoProcessor.from_pretrained("lerobot/fast-action-tokenizer", trust_remote_code=True)
    tokenizer = base.fit(
        trajectories,
        scale=scale,
        vocab_size=vocab_size,
        action_dim=trajectories[0].shape[1],
    )
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)
    return tokenizer, True


def encode_targets(tokenizer, trajectories: list[np.ndarray], vocab_size: int) -> tuple[np.ndarray, np.ndarray]:
    sequences = []
    offsets = [0]
    for index, trajectory in enumerate(trajectories):
        encoded = tokenizer(trajectory)
        sequence = np.asarray(encoded[0], dtype=np.int64).reshape(-1)
        if sequence.size and (int(sequence.min()) < 0 or int(sequence.max()) >= vocab_size):
            raise ValueError(
                f"FAST tokenizer emitted IDs outside [0,{vocab_size}): "
                f"segment={index}, range=[{sequence.min()},{sequence.max()}]"
            )
        sequences.append(sequence)
        offsets.append(offsets[-1] + len(sequence))
        if (index + 1) % 2000 == 0:
            print(f"  encoded {index + 1}/{len(trajectories)} skill trajectories")
    flat = np.concatenate(sequences).astype(np.uint16, copy=False) if sequences else np.empty(0, np.uint16)
    return flat, np.asarray(offsets, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--transition_pack", type=Path, required=True)
    parser.add_argument("--tokenizer_dir", type=Path, required=True)
    parser.add_argument("--target_pack", type=Path, required=True)
    parser.add_argument("--encoded_dims", default="0:7")
    parser.add_argument("--vocab_size", type=int, default=1024)
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--max_fast_tokens", type=int, required=True)
    args = parser.parse_args()

    trajectories, provenance, ranges, lengths, quantiles = load_skill_trajectories(
        args.repo_id,
        args.dataset_root,
        args.transition_pack,
        args.encoded_dims,
    )
    if not trajectories:
        raise ValueError("No skill trajectories found.")
    tokenizer, fitted = fit_or_load_tokenizer(
        args.tokenizer_dir,
        trajectories,
        args.vocab_size,
        args.scale,
    )
    tokens, offsets = encode_targets(tokenizer, trajectories, args.vocab_size)
    token_lengths = np.diff(offsets)
    token_length_max = int(token_lengths.max())
    if token_length_max > args.max_fast_tokens:
        raise ValueError(
            f"This skillset needs {token_length_max} FAST tokens, but max_fast_tokens="
            f"{args.max_fast_tokens}. Increase tokenizers.max_fast_tokens in pretrain_config.yaml. "
            "Targets are never truncated."
        )

    metadata = {
        "repo_id": args.repo_id,
        "vocab_size": args.vocab_size,
        "scale": args.scale,
        "encoded_dims": args.encoded_dims,
        "encoded_dim_ranges": ranges,
        "total_encoded_dims": int(trajectories[0].shape[1]),
        "normalization_mode": "QUANTILES",
        "variable_horizon": True,
        "num_training_trajectories": len(trajectories),
        "trajectory_length_min": int(lengths.min()),
        "trajectory_length_max": int(lengths.max()),
        "fast_token_length_mean": float(token_lengths.mean()),
        "fast_token_length_p99": float(np.percentile(token_lengths, 99)),
        "fast_token_length_max": token_length_max,
    }
    (args.tokenizer_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    args.target_pack.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.target_pack.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        fast_tokens=tokens,
        fast_token_offsets=offsets,
        trajectory_length=lengths,
        skill_code=provenance["skill_code"],
        episode_id=provenance["episode_id"],
        frame_start=provenance["frame_start"],
        frame_end=provenance["frame_end"],
        action_q01=quantiles[0],
        action_q99=quantiles[1],
        encoded_dims=np.str_(args.encoded_dims),
        tokenizer_name=np.str_(args.tokenizer_dir.name),
        vocab_size=np.int64(args.vocab_size),
        schema_version=np.int64(1),
    )
    temporary.replace(args.target_pack)
    print(
        f"FAST tokenizer {'fitted' if fitted else 'reused'}; target pack={args.target_pack} "
        f"({args.target_pack.stat().st_size / 1e6:.2f} MB)"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
