#!/usr/bin/env python3
"""Build CPU-only unimodal-null corrected GMM overlap-entropy caches."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def _regularized_cholesky(cloud: np.ndarray) -> np.ndarray:
    covariance = np.cov(np.asarray(cloud, dtype=np.float64), rowvar=False)
    dimension = int(covariance.shape[0])
    scale = max(float(np.trace(covariance)) / max(1, dimension), 1e-8)
    identity = np.eye(dimension, dtype=np.float64)
    for multiplier in (1e-6, 1e-5, 1e-4, 1e-3):
        try:
            return np.linalg.cholesky(covariance + identity * scale * multiplier)
        except np.linalg.LinAlgError:
            continue
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, scale * 1e-3)
    return vectors @ np.diag(np.sqrt(values))


def _build_episode(job: tuple) -> tuple[int, str]:
    (
        episode_id,
        raw_path,
        real_path,
        out_path,
        bootstraps,
        seed,
        components,
        n_init,
        max_iter,
    ) = job
    out_path = Path(out_path)
    if out_path.is_file():
        with np.load(out_path, allow_pickle=False) as payload:
            if {
                "gmm_null_corrected_overlap_entropy",
                "selected_k",
            }.issubset(payload.files):
                return int(episode_id), "cached"

    from sbd_multimodality import gmm_overlap_entropy_curve

    with np.load(raw_path, allow_pickle=False) as payload:
        descriptors = np.asarray(payload["descriptors"], dtype=np.float64)
    with np.load(real_path, allow_pickle=False) as payload:
        original_scores = {
            key: np.asarray(payload[key]).copy()
            for key in payload.files
            if key != "metadata_json"
        }
        real = np.asarray(
            original_scores["gmm_overlap_entropy"], dtype=np.float64
        )
    if len(real) != len(descriptors):
        raise ValueError(
            f"Episode {episode_id}: {len(real)} real scores for "
            f"{len(descriptors)} descriptor anchors"
        )

    samples, dimension = descriptors.shape[1:]
    # Common random numbers make differences between adjacent anchors reflect
    # covariance geometry instead of independent Monte Carlo seed noise.
    standard_noise = np.random.default_rng(int(seed)).standard_normal(
        (int(bootstraps), int(samples), int(dimension))
    )
    null_mean = np.empty(len(descriptors), dtype=np.float32)
    for anchor, cloud in enumerate(descriptors):
        center = cloud.mean(axis=0)
        transform = _regularized_cholesky(cloud)
        null_clouds = center[None, None, :] + standard_noise @ transform.T
        null_scores = gmm_overlap_entropy_curve(
            null_clouds,
            components=int(components),
            covariance="diag",
            n_init=int(n_init),
            max_iter=int(max_iter),
            min_effective_samples=0.0,
            weighted=True,
        )
        null_mean[anchor] = float(np.mean(null_scores))
    corrected = np.maximum(real - null_mean, 0.0).astype(np.float32)
    metadata = {
        "schema": 1,
        "metric": "gmm_null_corrected_overlap_entropy",
        "bootstraps": int(bootstraps),
        "seed": int(seed),
        "common_random_numbers": True,
        "null_distribution": "empirical_mean_and_full_covariance_gaussian",
        "gmm": {
            "components": int(components),
            "covariance": "diag",
            "n_init": int(n_init),
            "max_iter": int(max_iter),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(f".{out_path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        **original_scores,
        gmm_overlap_entropy_null_mean=null_mean,
        gmm_null_corrected_overlap_entropy=corrected,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    os.replace(temporary, out_path)
    return int(episode_id), "built"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--real-score-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstraps", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--components", type=int, default=5)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=300)
    args = parser.parse_args()
    if args.bootstraps < 1 or args.workers < 1:
        raise ValueError("--bootstraps and --workers must be positive")

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    real_dir = Path(args.real_score_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    jobs = []
    for raw_path in sorted(raw_dir.glob("ep*.npz")):
        episode_id = int(raw_path.stem.removeprefix("ep"))
        real_path = real_dir / raw_path.name
        if not real_path.is_file():
            raise FileNotFoundError(real_path)
        jobs.append(
            (
                episode_id,
                str(raw_path),
                str(real_path),
                str(out_dir / raw_path.name),
                int(args.bootstraps),
                int(args.seed),
                int(args.components),
                int(args.n_init),
                int(args.max_iter),
            )
        )
    if not jobs:
        raise FileNotFoundError(f"No ep*.npz caches under {raw_dir}")

    completed = 0
    with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = {executor.submit(_build_episode, job): job[0] for job in jobs}
        for future in as_completed(futures):
            episode_id, status = future.result()
            completed += 1
            print(
                f"[null overlap {completed}/{len(jobs)}] "
                f"ep{episode_id:07d} {status}",
                flush=True,
            )
    print(f"Null-corrected overlap caches: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
