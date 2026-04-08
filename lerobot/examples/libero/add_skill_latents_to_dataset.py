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


def build_frame_latent_map(
    latents_data: dict,
    latent_dim: int,
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
    zero_outside: bool = True,
) -> np.ndarray:
    """주어진 프레임에 해당하는 latent 반환. 스킬 구간 밖 gap 프레임은 zero."""
    for fs, fe, lat in skills:
        if fs <= frame_idx < fe:
            return lat.astype(np.float32)
    return np.zeros(latent_dim, dtype=np.float32)


def main(args: Args) -> None:
    src_dir = Path(args.src_dataset_dir)
    dst_dir = Path(args.dst_dataset_dir)
    latents_path = Path(args.latents_path)

    # ── Load latents ───────────────────────────────────────────────
    print(f"Loading latents from {latents_path} ...")
    raw = np.load(str(latents_path))
    latent_dim = raw["latents"].shape[1]
    ep_skill_map = build_frame_latent_map(raw, latent_dim)
    valid_ep_ids = set(ep_skill_map.keys())
    print(f"  latent_dim={latent_dim}, episodes with skills={len(valid_ep_ids)}")

    # ── Copy dataset ───────────────────────────────────────────────
    if dst_dir.exists():
        print(f"Removing existing {dst_dir} ...")
        shutil.rmtree(dst_dir)
    print(f"Copying dataset {src_dir} → {dst_dir} ...")
    shutil.copytree(src_dir, dst_dir)

    # ── Process data parquet files: remove no-skill episodes, add latents ──
    data_files = sorted((dst_dir / "data").rglob("*.parquet"))
    print(f"Processing {len(data_files)} data parquet files ...")

    removed_ep_ids: set[int] = set()
    n_frames_removed = 0

    for parquet_path in tqdm(data_files):
        df = pd.read_parquet(parquet_path)

        # 스킬 latent가 없는 에피소드 제거
        missing = set(df["episode_index"].unique().tolist()) - valid_ep_ids
        if missing:
            removed_ep_ids |= missing
            before = len(df)
            df = df[df["episode_index"].isin(valid_ep_ids)].reset_index(drop=True)
            n_frames_removed += before - len(df)

        if df.empty:
            df.to_parquet(parquet_path, index=False)
            continue

        # 남은 프레임에 skill latent 할당
        # (스킬 구간 안: 해당 latent / 구간 밖 gap: zero)
        skill_latents = [
            get_latent_for_frame(
                int(row["frame_index"]),
                ep_skill_map[int(row["episode_index"])],
                latent_dim,
                zero_outside=True,  # gap 프레임은 항상 zero
            )
            for _, row in df.iterrows()
        ]
        df["observation.states.skill"] = skill_latents
        df.to_parquet(parquet_path, index=False)

    print(f"  Removed {len(removed_ep_ids)} episodes (no skill latent): {sorted(removed_ep_ids)}")
    print(f"  Removed {n_frames_removed} frames total")

    # ── Remove from meta/episodes parquet ─────────────────────────
    ep_meta_files = sorted((dst_dir / "meta" / "episodes").rglob("*.parquet"))
    for ep_meta_path in ep_meta_files:
        ep_df = pd.read_parquet(ep_meta_path)
        ep_df = ep_df[ep_df["episode_index"].isin(valid_ep_ids)].reset_index(drop=True)
        ep_df.to_parquet(ep_meta_path, index=False)

    # ── Update info.json ───────────────────────────────────────────
    info_path = dst_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["repo_id"] = args.dst_repo_id
    info["total_episodes"] = info.get("total_episodes", 0) - len(removed_ep_ids)
    info["total_frames"] = info.get("total_frames", 0) - n_frames_removed
    info["features"]["observation.states.skill"] = {
        "dtype": "float32",
        "shape": [latent_dim],
        "names": [f"skill_z{i}" for i in range(latent_dim)],
    }
    info_path.write_text(json.dumps(info, indent=2))
    print(f"  Updated info.json (episodes={info['total_episodes']}, frames={info['total_frames']})")

    # ── Update stats.json ──────────────────────────────────────────
    stats_path = dst_dir / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        all_latents = []
        for parquet_path in sorted((dst_dir / "data").rglob("*.parquet")):
            df = pd.read_parquet(parquet_path)
            if "observation.states.skill" in df.columns:
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
    print(f"  제거된 에피소드: {len(removed_ep_ids)}개 / 제거된 프레임: {n_frames_removed}개")


if __name__ == "__main__":
    main(tyro.cli(Args))
