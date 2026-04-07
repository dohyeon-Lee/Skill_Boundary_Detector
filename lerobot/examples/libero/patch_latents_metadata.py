"""
기존 skill_vae_latents_epoch*.npz 파일에 메타데이터를 추가하는 패치 스크립트.

latents 배열은 train_vae.py가 skills_dir의 npz 파일을 sorted+rglob으로 로드한 후
min_skill_len 필터를 적용한 순서와 동일하게 저장되어 있으므로,
같은 방식으로 메타데이터를 불러와서 npz에 덮어씌운다.

Usage:
    python examples/libero/patch_latents_metadata.py \
        --sweep_dir /scratch/mdorazi/Skill_Boundary_Detector/outputs/vae_sweep_epoch \
        --skills_dir /scratch/mdorazi/Skill_Boundary_Detector/outputs/replay_libero10 \
        --min_skill_len 10
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro


@dataclass
class Args:
    sweep_dir: str = ""
    """vae_sweep_epoch 디렉토리 (하위에 vae_lat*/가 있는 곳)."""
    skills_dir: str = ""
    """replay_demo.py --save_skills 로 생성된 skill npz 파일들이 있는 디렉토리."""
    min_skill_len: int = 10
    """train_vae.py 와 동일한 필터 기준."""
    dry_run: bool = False
    """True 이면 실제로 저장하지 않고 검증만 한다."""


def load_metadata(skills_dir: Path, min_skill_len: int) -> list[dict]:
    npz_files = sorted(skills_dir.rglob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"skills_dir에 npz 파일이 없습니다: {skills_dir}")

    meta = []
    for f in npz_files:
        d = np.load(str(f))
        length = int(len(d["actions"]))
        if length < min_skill_len:
            continue
        meta.append({
            "episode_id":  int(d["episode_id"]),
            "skill_index": int(d["skill_index"]),
            "frame_start": int(d["frame_start"]),
            "frame_end":   int(d["frame_end"]),
            "length":      length,
        })
    return meta


def patch_exp_dir(exp_dir: Path, meta: list[dict], dry_run: bool) -> None:
    npz_files = sorted(exp_dir.glob("skill_vae_latents_epoch*.npz"))
    if not npz_files:
        print(f"  [skip] npz 없음: {exp_dir}")
        return

    for npz_path in npz_files:
        d = np.load(str(npz_path))

        if "episode_id" in d.files:
            print(f"  [skip] 이미 메타데이터 있음: {npz_path.name}")
            continue

        latents = d["latents"]
        if len(latents) != len(meta):
            print(
                f"  [ERROR] {npz_path.name}: latents 수({len(latents)}) != "
                f"메타데이터 수({len(meta)}) → 스킵"
            )
            continue

        if dry_run:
            print(f"  [dry-run] 패치 예정: {npz_path.name}  ({len(latents)}개)")
            continue

        np.savez(
            str(npz_path),
            latents=latents,
            episode_id=np.array([m["episode_id"]  for m in meta]),
            skill_index=np.array([m["skill_index"] for m in meta]),
            frame_start=np.array([m["frame_start"] for m in meta]),
            frame_end=np.array([m["frame_end"]   for m in meta]),
            length=np.array([m["length"]      for m in meta]),
        )
        print(f"  [ok] 패치 완료: {npz_path.name}")


def main(args: Args) -> None:
    sweep_dir  = Path(args.sweep_dir)
    skills_dir = Path(args.skills_dir)

    print(f"메타데이터 로드 중: {skills_dir}  (min_len={args.min_skill_len}) ...")
    meta = load_metadata(skills_dir, args.min_skill_len)
    print(f"  → {len(meta)}개 세그먼트")

    exp_dirs = sorted(sweep_dir.glob("vae_*"))
    if not exp_dirs:
        # sweep_dir 자체가 실험 디렉토리인 경우
        exp_dirs = [sweep_dir]

    for exp_dir in exp_dirs:
        print(f"\n[{exp_dir.name}]")
        patch_exp_dir(exp_dir, meta, args.dry_run)

    print("\n완료.")


if __name__ == "__main__":
    main(tyro.cli(Args))
