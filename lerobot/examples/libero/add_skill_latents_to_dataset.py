"""
Skill latent vector를 LeRobot 데이터셋에 추가하는 스크립트.

skill_vae_latents_epoch*.npz에서 (episode_id, frame_start, frame_end, latent)를 읽어
해당 프레임 범위의 모든 프레임에 동일한 latent를 할당한다.
스킬 범위 밖의 프레임은 zero vector로 채운다.

결과 데이터셋은 기존 데이터셋을 복사한 뒤 observation.states.skill 피처를 추가한다.

Usage:
    python examples/libero/add_skill_latents_to_dataset.py \
        --src_dataset_dir /scratch/mdorazi/Skill_Boundary_Detector/libero_dataset/libero_10 \
        --dst_dataset_dir /scratch/mdorazi/Skill_Boundary_Detector/libero_dataset/libero_10_skill \
        --latents_path /scratch/mdorazi/Skill_Boundary_Detector/outputs/vae_sweep_epoch/vae_lat8_hid128/skill_vae_latents_epoch0100.npz \
        --dst_repo_id mdorazi/libero_10_skill
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
from tqdm import tqdm


@dataclass
class Args:
    src_dataset_dir: str
    """원본 LeRobot 데이터셋 경로"""
    dst_dataset_dir: str
    """출력 데이터셋 경로"""
    latents_path: str
    """skill_vae_latents_epoch*.npz 경로"""
    dst_repo_id: str = "mdorazi/libero_skill"
    """출력 데이터셋 repo_id (info.json에 기록)"""
    zero_outside_skill: bool = True
    """스킬 범위 밖 프레임을 zero vector로 채울지 여부 (False면 nearest skill latent 사용)"""


def build_frame_latent_map(
    latents_data: dict,
    latent_dim: int,
    zero_outside: bool,
) -> dict[int, np.ndarray]:
    """episode_id별로 (total_frames,) → latent 매핑 딕셔너리 생성.

    Returns:
        {episode_id: np.ndarray of shape (max_frame, latent_dim)}
    """
    episode_ids  = latents_data["episode_id"]
    frame_starts = latents_data["frame_start"]
    frame_ends   = latents_data["frame_end"]
    latents      = latents_data["latents"]

    # episode별로 그룹핑
    ep_map: dict[int, list[tuple[int, int, np.ndarray]]] = {}
    for ep_id, fs, fe, lat in zip(episode_ids, frame_starts, frame_ends, latents):
        ep_map.setdefault(int(ep_id), []).append((int(fs), int(fe), lat))

    return ep_map


def get_latent_for_frame(
    frame_idx: int,
    skills: list[tuple[int, int, np.ndarray]],
    latent_dim: int,
    zero_outside: bool,
) -> np.ndarray:
    """주어진 프레임에 해당하는 latent 반환."""
    for fs, fe, lat in skills:
        if fs <= frame_idx < fe:
            return lat.astype(np.float32)

    if zero_outside:
        return np.zeros(latent_dim, dtype=np.float32)

    # nearest skill latent
    best_lat = skills[0][2]
    best_dist = float("inf")
    for fs, fe, lat in skills:
        mid = (fs + fe) / 2
        dist = abs(frame_idx - mid)
        if dist < best_dist:
            best_dist = dist
            best_lat = lat
    return best_lat.astype(np.float32)


def main(args: Args) -> None:
    src_dir = Path(args.src_dataset_dir)
    dst_dir = Path(args.dst_dataset_dir)
    latents_path = Path(args.latents_path)

    # ── Load latents ───────────────────────────────────────────────
    print(f"Loading latents from {latents_path} ...")
    raw = np.load(str(latents_path))
    latent_dim = raw["latents"].shape[1]
    ep_skill_map = build_frame_latent_map(raw, latent_dim, args.zero_outside_skill)
    print(f"  latent_dim={latent_dim}, episodes with skills={len(ep_skill_map)}")

    # ── Copy dataset ───────────────────────────────────────────────
    if dst_dir.exists():
        print(f"Removing existing {dst_dir} ...")
        shutil.rmtree(dst_dir)
    print(f"Copying dataset {src_dir} → {dst_dir} ...")
    shutil.copytree(src_dir, dst_dir)

    # ── Update info.json ───────────────────────────────────────────
    info_path = dst_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["repo_id"] = args.dst_repo_id
    info["features"]["observation.states.skill"] = {
        "dtype": "float32",
        "shape": [latent_dim],
        "names": [f"skill_z{i}" for i in range(latent_dim)],
    }
    info_path.write_text(json.dumps(info, indent=2))
    print(f"  Updated info.json (added observation.states.skill dim={latent_dim})")

    # ── Process each parquet file ──────────────────────────────────
    data_files = sorted((dst_dir / "data").rglob("*.parquet"))
    print(f"Processing {len(data_files)} parquet files ...")

    for parquet_path in tqdm(data_files):
        df = pd.read_parquet(parquet_path)

        skill_latents = []
        for _, row in df.iterrows():
            ep_id = int(row["episode_index"])
            frame_idx = int(row["frame_index"])

            if ep_id in ep_skill_map:
                lat = get_latent_for_frame(
                    frame_idx,
                    ep_skill_map[ep_id],
                    latent_dim,
                    args.zero_outside_skill,
                )
            else:
                lat = np.zeros(latent_dim, dtype=np.float32)

            skill_latents.append(lat)

        df["observation.states.skill"] = skill_latents
        df.to_parquet(parquet_path, index=False)

    # ── Update stats.json ──────────────────────────────────────────
    stats_path = dst_dir / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        # 모든 latent 모아서 통계 계산
        all_latents = []
        for parquet_path in sorted((dst_dir / "data").rglob("*.parquet")):
            df = pd.read_parquet(parquet_path)
            all_latents.extend(df["observation.states.skill"].tolist())
        all_latents = np.array(all_latents)
        stats["observation.states.skill"] = {
            "min":  all_latents.min(axis=0).tolist(),
            "max":  all_latents.max(axis=0).tolist(),
            "mean": all_latents.mean(axis=0).tolist(),
            "std":  all_latents.std(axis=0).tolist(),
        }
        stats_path.write_text(json.dumps(stats, indent=2))
        print("  Updated stats.json")

    print(f"\n완료! 출력 데이터셋: {dst_dir}")
    print(f"  observation.states.skill (dim={latent_dim}) 추가됨")


if __name__ == "__main__":
    main(tyro.cli(Args))
