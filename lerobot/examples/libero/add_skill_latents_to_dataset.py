"""
SkillVLA용 컬럼을 LeRobot 데이터셋에 추가하는 스크립트.

VQAE encoder로 뽑은 skill tokens (.npz) 를 원본 데이터셋에 추가한다.

추가되는 컬럼:
  skill_latent        : int32   현재 스킬의 codebook token index
  skill_latent_prev   : int32   이전 스킬의 token index
  skill_boundary      : int8    새 스킬의 첫 프레임에서만 1 (에피소드 첫 스킬 포함)
  skill_start_state   : (state_dim,) float32  현재 스킬 시작 시점의 proprioceptive state
  skill_frame_index   : int32   스킬 내 0-based step 위치
  skill_progress      : float32 현재 스킬 진행률 [0, 1]

npz는 VQAE로 저장된 것이어야 한다 ("tokens" key 또는 1D "latents" key).

마지막 스킬 이후 남은 프레임은 마지막 스킬의 token으로 채워지고
skill_frame_index는 마지막 스킬 시작점 기준으로 이어서 카운팅된다.

스킬 latent가 없는 에피소드는 제거하지 않고 모든 컬럼을 0으로 채운다.

Usage:
    python examples/libero/add_skill_latents_to_dataset.py \
        --src_dataset_dir /data2/dohyeon/SBD/libero_dataset/libero_90 \
        --dst_dataset_dir /data2/dohyeon/SBD/libero_dataset/libero_90_skillvla_vqae \
        --latents_path    /data2/dohyeon/SBD/outputs/libero_90_skillset_latents/libero_90_lat64_vq512/spline_vqae_latents_epoch10000.npz \
        --window 1
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
    """VQAE encoder 결과 .npz (keys: episode_id, frame_start, frame_end, tokens)"""
    dst_repo_id: str = "dohyeon/libero_90_skillvla_vqae"
    window: int = 1
    """skill_boundary를 skill 시작 시점부터 몇 프레임 동안 1로 둘지"""


# ── latents .npz 파싱 ─────────────────────────────────────────────────────────

def load_skill_map(npz_path: Path) -> dict[int, list]:
    """
    Returns skill_map: episode_id → [(frame_start, frame_end, token_int), ...]
    """
    raw = np.load(str(npz_path))
    if "tokens" in raw:
        tokens = raw["tokens"]
    elif raw["latents"].ndim == 1:
        tokens = raw["latents"].astype(np.int32)
    else:
        raise ValueError(
            f"{npz_path} contains float latent vectors, not VQAE tokens. "
            "Re-encode with a VQAE checkpoint to get integer tokens."
        )

    skill_map: dict[int, list] = {}
    for ep_id, fs, fe, tok in zip(raw["episode_id"], raw["frame_start"], raw["frame_end"], tokens):
        skill_map.setdefault(int(ep_id), []).append((int(fs), int(fe), int(tok)))
    for ep_id in skill_map:
        skill_map[ep_id].sort(key=lambda x: x[0])
    return skill_map


# ── episode별 컬럼 계산 ───────────────────────────────────────────────────────

def compute_skill_columns(
    ep_df: pd.DataFrame,
    skills: list[tuple[int, int, int]],
    boundary_window: int = 1,
) -> dict[str, np.ndarray]:
    """
    ep_df : frame_index 순으로 정렬된 한 에피소드의 DataFrame
    skills: [(frame_start, frame_end, token), ...] sorted by frame_start

    Returns dict of arrays, each length len(ep_df).
    """
    n         = len(ep_df)
    frames    = ep_df["frame_index"].values
    states    = np.stack(ep_df["observation.state"].values)
    state_dim = states.shape[1]

    z_arr           = np.zeros(n, dtype=np.int32)
    z_prev_arr      = np.zeros(n, dtype=np.int32)
    f_b_arr         = np.zeros(n, dtype=np.int8)
    start_state_arr = np.zeros((n, state_dim), dtype=np.float32)
    frame_idx_arr   = np.full(n, -1, dtype=np.int32)
    progress_arr    = np.zeros(n, dtype=np.float32)

    frame_to_row = {int(f): i for i, f in enumerate(frames)}

    for fs, fe, tok in skills:
        mask = (frames >= fs) & (frames < fe)
        if not mask.any():
            continue
        z_arr[mask]         = tok
        frame_idx_arr[mask] = frames[mask] - fs
        skill_len = max(1, fe - fs - 1)
        progress_arr[mask] = np.clip((frames[mask] - fs) / skill_len, 0.0, 1.0)
        if fs in frame_to_row:
            start_state_arr[mask] = states[frame_to_row[fs]]
        boundary_mask = (frames >= fs) & (frames < fs + boundary_window)
        f_b_arr[boundary_mask] = 1

    # 마지막 스킬 이후 남은 프레임: 마지막 스킬 token으로 채움
    if skills:
        last_fs, last_fe, last_tok = skills[-1]
        leftover = frame_idx_arr == -1
        if leftover.any():
            z_arr[leftover]         = last_tok
            frame_idx_arr[leftover] = frames[leftover] - last_fs
            last_len = max(1, last_fe - last_fs - 1)
            progress_arr[leftover] = np.clip(
                (frames[leftover] - last_fs) / last_len, 0.0, 1.0
            )
            if last_fs in frame_to_row:
                start_state_arr[leftover] = states[frame_to_row[last_fs]]

    # z_prev: 한 스킬 이전 스킬 token
    prev_tok = 0
    for fs, fe, tok in skills:
        mask = (frames >= fs) & (frames < fe)
        z_prev_arr[mask] = prev_tok
        prev_tok = tok
    if skills:
        last_fs, last_fe, _ = skills[-1]
        leftover = frame_idx_arr >= (last_fe - last_fs)
        z_prev_arr[leftover] = prev_tok

    return {
        "skill_latent":      z_arr.tolist(),
        "skill_latent_prev": z_prev_arr.tolist(),
        "skill_boundary":    f_b_arr,
        "skill_start_state": list(start_state_arr),
        "skill_frame_index": frame_idx_arr,
        "skill_progress":    progress_arr,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: Args) -> None:
    src_dir = Path(args.src_dataset_dir)
    dst_dir = Path(args.dst_dataset_dir)

    print(f"Loading skill tokens from {args.latents_path} ...")
    skill_map = load_skill_map(Path(args.latents_path))
    print(f"  episodes={len(skill_map)}")
    boundary_window = max(1, int(args.window))
    print(f"  boundary_window={boundary_window}")

    if dst_dir.exists():
        print(f"Removing existing {dst_dir} ...")
        shutil.rmtree(dst_dir)
    print(f"Copying {src_dir} → {dst_dir} ...")
    shutil.copytree(src_dir, dst_dir)

    data_files = sorted((dst_dir / "data").rglob("*.parquet"))
    print(f"Processing {len(data_files)} parquet files ...")

    n_zero_fill_eps: int = 0

    for parquet_path in tqdm(data_files):
        df = pd.read_parquet(parquet_path)
        if df.empty:
            df.to_parquet(parquet_path, index=False)
            continue

        missing = set(df["episode_index"].unique()) - set(skill_map.keys())
        if missing:
            n_zero_fill_eps += len(missing)

        df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

        col_buffers: dict[str, list] = {
            "skill_latent":      [],
            "skill_latent_prev": [],
            "skill_boundary":    [],
            "skill_start_state": [],
            "skill_frame_index": [],
            "skill_progress":    [],
        }

        for ep_id, ep_df in df.groupby("episode_index"):
            ep_df  = ep_df.sort_values("frame_index")
            skills = skill_map.get(int(ep_id), [])
            cols   = compute_skill_columns(ep_df, skills, boundary_window=boundary_window)
            for k in col_buffers:
                col_buffers[k].extend(cols[k] if isinstance(cols[k], list) else cols[k].tolist())

        for k, vals in col_buffers.items():
            df[k] = vals

        df.to_parquet(parquet_path, index=False)

    print(f"  Zero-filled {n_zero_fill_eps} episodes without skill tokens (kept in dataset)")

    # ── Update info.json ──────────────────────────────────────────────────────
    info_path = dst_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["repo_id"] = args.dst_repo_id

    state_dim = None
    for pf in sorted((dst_dir / "data").rglob("*.parquet")):
        sample = pd.read_parquet(pf)
        if "skill_start_state" in sample.columns:
            state_dim = len(sample["skill_start_state"].iloc[0])
            break

    info["features"].update({
        "skill_latent": {"dtype": "int32", "shape": [1], "names": ["skill_token"]},
        "skill_latent_prev": {"dtype": "int32", "shape": [1], "names": ["skill_token_prev"]},
        "skill_boundary": {"dtype": "int8", "shape": [1], "names": ["skill_boundary"]},
        "skill_start_state": {
            "dtype": "float32", "shape": [state_dim],
            "names": [f"s0_{i}" for i in range(state_dim)],
        } if state_dim else {},
        "skill_frame_index": {"dtype": "int32", "shape": [1], "names": ["skill_frame_index"]},
        "skill_progress":    {"dtype": "float32", "shape": [1], "names": ["skill_progress"]},
    })
    info_path.write_text(json.dumps(info, indent=2))

    print(f"\n완료: {dst_dir}")
    print(f"  추가된 컬럼: skill_latent, skill_latent_prev, skill_boundary, "
          f"skill_start_state, skill_frame_index, skill_progress")
    print(f"  episodes={info['total_episodes']}, frames={info['total_frames']}")


if __name__ == "__main__":
    main(tyro.cli(Args))
