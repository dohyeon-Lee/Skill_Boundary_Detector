#!/usr/bin/env python3
"""Encode original FSQ skill NPZs into an episode-addressable eval cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_HERE = Path(__file__).resolve()
_LIBERO_EXAMPLES = _HERE.parents[4]
sys.path.insert(0, str(_LIBERO_EXAMPLES))

from FSQ import load_fsq_encoder  # noqa: E402
from train_FSQ import load_skill_files  # noqa: E402


def _fingerprint(model_path: Path, skills_dir: Path) -> dict[str, object]:
    stat = model_path.stat()
    return {
        "model_path": str(model_path.resolve()),
        "model_size": int(stat.st_size),
        "model_mtime_ns": int(stat.st_mtime_ns),
        "skills_dir": str(skills_dir.resolve()),
    }


def _cache_matches(path: Path, fingerprint: dict[str, object]) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            stored = json.loads(str(np.asarray(data["source_fingerprint"]).item()))
            required = {
                "tokens", "episode_id", "task_id", "skill_index", "frame_start", "frame_end", "length"
            }
            return stored == fingerprint and required.issubset(data.files)
    except Exception:
        return False


def prepare(model_path: Path, skills_dir: Path, output_path: Path, device: str) -> Path:
    fingerprint = _fingerprint(model_path, skills_dir)
    if _cache_matches(output_path, fingerprint):
        print(f"[FSQ eval] cache current: {output_path}")
        return output_path

    segments, _, _, metadata = load_skill_files(skills_dir)
    actual_device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
    encoder, _ = load_fsq_encoder(model_path, actual_device)
    latents, tokens = [], []
    for segment in tqdm(segments, desc="Encoding FSQ eval skills"):
        latents.append(encoder.encode_numpy(segment, device=actual_device))
        tokens.append(encoder.encode_index(segment, device=actual_device))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "latents": np.stack(latents).astype(np.float32),
        "tokens": np.asarray(tokens, dtype=np.int32),
        "source_fingerprint": np.asarray(json.dumps(fingerprint, sort_keys=True)),
    }
    for key in ("episode_id", "task_id", "skill_index", "frame_start", "frame_end", "length"):
        payload[key] = np.asarray([item[key] for item in metadata], dtype=np.int64)

    tmp = output_path.with_suffix(output_path.suffix + ".tmp.npz")
    np.savez(tmp, **payload)
    tmp.replace(output_path)
    print(f"[FSQ eval] wrote {len(tokens)} encoded skills -> {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--skills_dir", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    prepare(args.model_path, args.skills_dir, args.output_path, args.device)


if __name__ == "__main__":
    main()
