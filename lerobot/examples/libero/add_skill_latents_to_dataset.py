"""
SkillVLA용 컬럼을 LeRobot 데이터셋에 추가하는 스크립트.

FSQ encoder로 뽑은 skill tokens (.npz) 를 원본 데이터셋에 추가한다.
선택적으로 precompute_dino_features.py가 만든 frozen visual backbone
feature cache를 frame별 column으로 붙일 수 있다. 이 visual column은 VLA
학습 속도를 높이기 위한 train-time cache이며, eval/sim에서는 raw image를
모델 안의 같은 frozen visual encoder로 통과시키는 경로를 유지한다.

추가되는 컬럼:
  skill_index          : int32   현재 스킬이 skill_sequence에서 차지하는 index (BOS=0, 첫 실제 스킬=1)
  skill_sequence       : (max_order+2,) int32  [BOS, episode skill tokens..., EOS, PAD...]
  skill_length_sequence: (max_order+2,) int32  [0, episode skill lengths..., 0, 0...]
  skill_sequence_mask  : (max_order+2,) int8   padding이 아닌 sequence 위치
  skill_sequence_len   : int32   BOS/EOS를 포함한 sequence 길이
  skill_ds             : int32   현재 스킬 시작점으로부터의 distance, 시작 프레임에서 0
  skill_de             : int32   현재 스킬 종료점까지 남은 distance, 종료 프레임에서 0
  skill_boundary       : int8    현재 스킬 종료 프레임에서만 1
  skill_max_order      : int32   데이터셋에서 허용하는 최대 real skill 개수
  skill_max_length     : int32   FSQ와 공유하는 skill max length hyperparameter
  skill_decoder_state  : (7,) float32  FSQ decoder state = eef 6D + previous gripper action
  observation.dino.image: (F,) float32  optional precomputed visual feature

skill_sequence는 FSQ scalar token index (0 ~ prod(fsq_levels)-1) 를 저장한다.
특수 토큰은 scalar index 공간 바로 위에 배치된다:
  EOS = num_embeddings + 0
  BOS = num_embeddings + 1  ← modeling_skillVLA.py 인퍼런스와 반드시 일치해야 함
  PAD = num_embeddings + 2

num_embeddings는 --fsq_levels로 자동 계산하거나 --num_embeddings로 직접 지정한다.
--fsq_levels [3,3,3] → num_embeddings=27, --fsq_levels [5,5,5] → num_embeddings=125.

마지막 스킬 이후 남은 프레임은 마지막 스킬 index로 채워지고 ds는 마지막
스킬 시작점 기준으로 이어서 카운팅된다.

스킬 latent가 없는 에피소드는 제거하지 않고 모든 컬럼을 0으로 채운다.

Usage (FSQ 333):
    python examples/libero/add_skill_latents_to_dataset.py \
        --src_dataset_dir /data2/dohyeon/SBD/libero_dataset/libero_90 \
        --dst_dataset_dir /data2/dohyeon/SBD/libero_dataset/libero_90_data/libero_90_skillvla \
        --dst_repo_id dohyeon/libero_90_skillvla \
        --latents_path /data2/dohyeon/SBD/outputs/libero_90_skillset_FSQ/skill_latents.npz \
        --fsq_levels 3 3 3 \
        --max_order 20 \
        --max_length 200

Usage (FSQ 555):
    python examples/libero/add_skill_latents_to_dataset.py \
        --src_dataset_dir /data2/dohyeon/SBD/libero_dataset/libero_90 \
        --dst_dataset_dir /data2/dohyeon/SBD/libero_dataset/libero_90_data/libero_90_skillvla \
        --dst_repo_id dohyeon/libero_90_skillvla \
        --latents_path /data2/dohyeon/SBD/outputs/libero_90_skillset_FSQ/skill_latents.npz \
        --fsq_levels 5 5 5 \
        --max_order 20 \
        --max_length 200
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
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
    """FSQ encoder 결과 .npz (keys: episode_id, frame_start, frame_end, tokens)"""
    dst_repo_id: str = "dohyeon/libero_90_skillvla"
    max_order: int = 0
    """Maximum number of real skills per episode. 0 infers it from latents_path."""
    max_length: int = 200
    """Skill max length shared with FSQ max_length."""
    fsq_levels: list[int] = field(default_factory=list)
    """FSQ levels (e.g. 3 3 3 or 5 5 5). num_embeddings = prod(fsq_levels). --num_embeddings보다 우선."""
    num_embeddings: int = 0
    """FSQ codebook size = prod(fsq_levels). EOS=K, BOS=K+1, PAD=K+2. fsq_levels 미지정 시 직접 입력."""
    dino_features_path: str = ""
    """Optional precomputed visual feature npz from precompute_dino_features.py."""
    dino_column: str = "observation.dino.image"
    """Dataset column name for per-frame visual features."""


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
            f"{npz_path} contains float latent vectors, not FSQ tokens. "
            "train_FSQ.py 출력 npz의 'tokens' key를 확인하세요."
        )

    skill_map: dict[int, list] = {}
    for ep_id, fs, fe, tok in zip(raw["episode_id"], raw["frame_start"], raw["frame_end"], tokens):
        skill_map.setdefault(int(ep_id), []).append((int(fs), int(fe), int(tok)))
    for ep_id in skill_map:
        skill_map[ep_id].sort(key=lambda x: x[0])
    return skill_map


def load_dino_feature_map(features_path: Path) -> tuple[dict[int, dict[int, np.ndarray]], int, dict]:
    """Return episode_id -> frame_index -> visual feature."""
    raw = np.load(str(features_path), allow_pickle=False)
    required = {"features", "offsets", "episode_id", "frame_start", "length"}
    missing = required - set(raw.files)
    if missing:
        raise ValueError(f"{features_path} is missing keys: {sorted(missing)}")

    features = raw["features"].astype(np.float32)
    offsets = raw["offsets"].astype(np.int64)
    feature_dim = int(features.shape[-1])
    fmap: dict[int, dict[int, np.ndarray]] = {}

    for i, (ep, fs, length) in enumerate(zip(raw["episode_id"], raw["frame_start"], raw["length"])):
        start, end = int(offsets[i]), int(offsets[i + 1])
        clip = features[start:end]
        if len(clip) != int(length):
            raise ValueError(f"Feature length mismatch at skill index {i}: {len(clip)} != {int(length)}")
        ep_map = fmap.setdefault(int(ep), {})
        for j, feat in enumerate(clip):
            ep_map[int(fs) + j] = feat

    meta = {
        "path": str(features_path),
        "image_key": str(raw["image_key"]) if "image_key" in raw.files else "",
        "image_model_name": str(raw["image_model_name"]) if "image_model_name" in raw.files else "",
        "feature_dim": feature_dim,
    }
    return fmap, feature_dim, meta


# ── episode별 컬럼 계산 ───────────────────────────────────────────────────────

def compute_skill_columns(
    ep_df: pd.DataFrame,
    skills: list[tuple[int, int, int]],
    max_order: int = 1,
    max_length: int = 200,
    eos_token_id: int = 512,
    bos_token_id: int = 513,
    pad_token_id: int = 514,
) -> dict[str, np.ndarray]:
    """
    ep_df : frame_index 순으로 정렬된 한 에피소드의 DataFrame
    skills: [(frame_start, frame_end, token), ...] sorted by frame_start

    Returns dict of arrays, each length len(ep_df).
    """
    n = len(ep_df)
    frames = ep_df["frame_index"].values.astype(np.int64)
    max_seq_len = max_order + 2  # BOS + real skills + EOS

    skill_index_arr = np.zeros(n, dtype=np.int32)
    ds_arr = np.zeros(n, dtype=np.int32)
    de_arr = np.zeros(n, dtype=np.int32)
    boundary_arr = np.zeros(n, dtype=np.int8)
    max_order_arr = np.full(n, max_order, dtype=np.int32)
    max_length_arr = np.full(n, max_length, dtype=np.int32)

    seq_tokens = np.full((max_seq_len,), pad_token_id, dtype=np.int32)
    seq_lengths = np.zeros((max_seq_len,), dtype=np.int32)
    seq_mask = np.zeros((max_seq_len,), dtype=np.int8)
    seq_tokens[0] = bos_token_id
    seq_mask[0] = 1

    n_real = len(skills)
    if n_real > max_order:
        raise ValueError(f"Episode has {n_real} skills, but max_order={max_order}.")

    if skills:
        tokens = np.array([tok for _, _, tok in skills], dtype=np.int32)
        lengths = np.array([max(0, fe - fs) for fs, fe, _ in skills], dtype=np.int32)
        seq_tokens[1:1 + n_real] = tokens
        seq_lengths[1:1 + n_real] = lengths
        seq_mask[1:1 + n_real] = 1

    eos_index = 1 + n_real
    seq_tokens[eos_index] = eos_token_id
    seq_mask[eos_index] = 1
    seq_len = n_real + 2
    seq_len_arr = np.full(n, seq_len, dtype=np.int32)
    seq_arr = np.repeat(seq_tokens[None, :], n, axis=0)
    seq_length_arr = np.repeat(seq_lengths[None, :], n, axis=0)
    seq_mask_arr = np.repeat(seq_mask[None, :], n, axis=0)

    for skill_rank, (fs, fe, tok) in enumerate(skills):
        mask = (frames >= fs) & (frames < fe)
        if not mask.any():
            continue
        seq_idx = skill_rank + 1  # index 0 is BOS
        skill_index_arr[mask] = seq_idx
        ds_arr[mask] = frames[mask] - fs
        de_arr[mask] = np.maximum((fe - 1) - frames[mask], 0)
        boundary_arr[mask & (frames == fe - 1)] = 1

    # 마지막 스킬 이후 남은 프레임: 마지막 스킬 index로 채움
    if skills:
        last_fs, last_fe, _ = skills[-1]
        leftover = (skill_index_arr == 0) & (frames >= last_fe)
        if leftover.any():
            seq_idx = len(skills)
            skill_index_arr[leftover] = seq_idx
            ds_arr[leftover] = frames[leftover] - last_fs
            de_arr[leftover] = np.maximum((last_fe - 1) - frames[leftover], 0)

    return {
        "skill_index": skill_index_arr,
        "skill_sequence": list(seq_arr),
        "skill_length_sequence": list(seq_length_arr),
        "skill_sequence_mask": list(seq_mask_arr),
        "skill_sequence_len": seq_len_arr,
        "skill_ds": ds_arr,
        "skill_de": de_arr,
        "skill_boundary": boundary_arr,
        "skill_max_order": max_order_arr,
        "skill_max_length": max_length_arr,
    }


def compute_skill_decoder_state(
    ep_df: pd.DataFrame,
    skill_ds: np.ndarray,
    *,
    state_column: str = "observation.state",
    action_column: str = "action",
    eef_state_dim: int = 6,
    gripper_action_dim: int = -1,
) -> list[np.ndarray]:
    """Build the exact state input used by the FSQ single-step decoder.

    FSQ training used EEF pose plus the previous gripper command. The first
    frame of each skill receives 0 for the previous gripper to avoid leaking
    the current target action.
    """
    if state_column not in ep_df.columns:
        raise KeyError(f"Missing required column: {state_column}")
    if action_column not in ep_df.columns:
        raise KeyError(f"Missing required column: {action_column}")

    states = np.stack(ep_df[state_column].to_numpy()).astype(np.float32)
    actions = np.stack(ep_df[action_column].to_numpy()).astype(np.float32)
    if states.shape[-1] < eef_state_dim:
        raise ValueError(f"{state_column} has dim={states.shape[-1]}, need at least {eef_state_dim}")
    if actions.shape[-1] == 0:
        raise ValueError(f"{action_column} is empty")

    grip_idx = (actions.shape[-1] + gripper_action_dim) % actions.shape[-1]
    prev_gripper = np.zeros((len(ep_df), 1), dtype=np.float32)
    if len(ep_df) > 1:
        prev_gripper[1:, 0] = actions[:-1, grip_idx]

    # FSQ decoder states are built per skill segment, so reset the previous
    # gripper command at every skill start.
    prev_gripper[np.asarray(skill_ds) == 0, 0] = 0.0

    decoder_state = np.concatenate([states[:, :eef_state_dim], prev_gripper], axis=-1)
    return [row.astype(np.float32) for row in decoder_state]


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: Args) -> None:
    src_dir = Path(args.src_dataset_dir)
    dst_dir = Path(args.dst_dataset_dir)

    # num_embeddings 결정: fsq_levels 우선, 없으면 num_embeddings 직접 사용
    if args.fsq_levels:
        num_embeddings = math.prod(args.fsq_levels)
        print(f"FSQ levels={args.fsq_levels}  →  num_embeddings={num_embeddings}")
    elif args.num_embeddings > 0:
        num_embeddings = args.num_embeddings
        print(f"num_embeddings={num_embeddings} (직접 지정)")
    else:
        raise ValueError(
            "--fsq_levels (e.g. 3 3 3) 또는 --num_embeddings 를 지정해야 합니다."
        )

    print(f"Loading skill tokens from {args.latents_path} ...")
    skill_map = load_skill_map(Path(args.latents_path))
    print(f"  episodes={len(skill_map)}")
    eos_token_id = num_embeddings
    bos_token_id = num_embeddings + 1
    pad_token_id = num_embeddings + 2
    skill_output_vocab_size = num_embeddings + 1  # FSQ tokens + EOS
    skill_vocab_size = num_embeddings + 3         # FSQ tokens + EOS/BOS/PAD
    observed_max_order = max((len(v) for v in skill_map.values()), default=0)
    observed_max_length = max((max((fe - fs for fs, fe, _ in v), default=0) for v in skill_map.values()), default=0)
    max_order = int(args.max_order) if int(args.max_order) > 0 else observed_max_order
    max_length = int(args.max_length)
    if observed_max_order > max_order:
        raise ValueError(
            f"--max_order={max_order} is smaller than observed max skill count {observed_max_order}."
        )
    if observed_max_length > max_length:
        raise ValueError(
            f"--max_length={max_length} is smaller than observed max skill length {observed_max_length}. "
            "Use the same max_length used for FSQ training."
        )
    max_seq_len = max_order + 2
    print(f"  observed_max_order={observed_max_order}, max_order={max_order}")
    print(f"  observed_max_length={observed_max_length}, max_length={max_length}")
    print(f"  max_skill_sequence_len={max_seq_len} (max_order + BOS/EOS)")
    print(
        f"  eos_token_id={eos_token_id}, bos_token_id={bos_token_id}, "
        f"pad_token_id={pad_token_id}, skill_vocab_size={skill_vocab_size}, "
        f"skill_output_vocab_size={skill_output_vocab_size}"
    )
    dino_map = None
    dino_dim = None
    dino_meta = None
    if args.dino_features_path:
        print(f"Loading DINO features from {args.dino_features_path} ...")
        dino_map, dino_dim, dino_meta = load_dino_feature_map(Path(args.dino_features_path))
        print(f"  dino episodes={len(dino_map)}, dim={dino_dim}, column={args.dino_column}")

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
            "skill_index": [],
            "skill_sequence": [],
            "skill_length_sequence": [],
            "skill_sequence_mask": [],
            "skill_sequence_len": [],
            "skill_ds": [],
            "skill_de": [],
            "skill_boundary": [],
            "skill_max_order": [],
            "skill_max_length": [],
            "skill_decoder_state": [],
        }
        dino_values: list | None = [] if dino_map is not None else None

        for ep_id, ep_df in df.groupby("episode_index"):
            ep_df  = ep_df.sort_values("frame_index")
            skills = skill_map.get(int(ep_id), [])
            cols   = compute_skill_columns(
                ep_df,
                skills,
                max_order=max_order,
                max_length=max_length,
                eos_token_id=eos_token_id,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
            )
            cols["skill_decoder_state"] = compute_skill_decoder_state(ep_df, cols["skill_ds"])
            for k in col_buffers:
                col_buffers[k].extend(cols[k] if isinstance(cols[k], list) else cols[k].tolist())
            if dino_values is not None:
                ep_features = dino_map.get(int(ep_id), {})
                zero = np.zeros((dino_dim,), dtype=np.float32)
                for frame in ep_df["frame_index"].values:
                    dino_values.append(ep_features.get(int(frame), zero))

        for k, vals in col_buffers.items():
            df[k] = vals
        if dino_values is not None:
            df[args.dino_column] = list(dino_values)

        df.to_parquet(parquet_path, index=False)

    print(f"  Zero-filled {n_zero_fill_eps} episodes without skill tokens (kept in dataset)")

    # ── Update info.json ──────────────────────────────────────────────────────
    info_path = dst_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["repo_id"] = args.dst_repo_id
    info["skill_num_embeddings"] = num_embeddings
    info["skill_fsq_levels"] = list(args.fsq_levels) if args.fsq_levels else []
    info["skill_eos_token_id"] = eos_token_id
    info["skill_bos_token_id"] = bos_token_id
    info["skill_pad_token_id"] = pad_token_id
    info["skill_vocab_size"] = skill_vocab_size
    info["skill_output_vocab_size"] = skill_output_vocab_size
    info["skill_max_order"] = max_order
    info["skill_observed_max_order"] = observed_max_order
    info["skill_max_length"] = max_length
    info["skill_observed_max_length"] = observed_max_length
    info["skill_sequence_size"] = max_seq_len
    if dino_meta is not None:
        info["dino_features_path"] = dino_meta["path"]
        info["dino_image_key"] = dino_meta["image_key"]
        info["dino_model_name"] = dino_meta["image_model_name"]
        info["dino_feature_dim"] = dino_meta["feature_dim"]

    info["features"].update({
        "skill_index": {"dtype": "int32", "shape": [1], "names": ["skill_index"]},
        "skill_sequence": {
            "dtype": "int32",
            "shape": [max_seq_len],
            "names": [f"skill_{i}" for i in range(max_seq_len)],
        },
        "skill_length_sequence": {
            "dtype": "int32",
            "shape": [max_seq_len],
            "names": [f"skill_len_{i}" for i in range(max_seq_len)],
        },
        "skill_sequence_mask": {
            "dtype": "int8",
            "shape": [max_seq_len],
            "names": [f"skill_mask_{i}" for i in range(max_seq_len)],
        },
        "skill_sequence_len": {"dtype": "int32", "shape": [1], "names": ["skill_sequence_len"]},
        "skill_ds": {"dtype": "int32", "shape": [1], "names": ["skill_ds"]},
        "skill_de": {"dtype": "int32", "shape": [1], "names": ["skill_de"]},
        "skill_boundary": {"dtype": "int8", "shape": [1], "names": ["skill_boundary"]},
        "skill_max_order": {"dtype": "int32", "shape": [1], "names": ["skill_max_order"]},
        "skill_max_length": {"dtype": "int32", "shape": [1], "names": ["skill_max_length"]},
        "skill_decoder_state": {
            "dtype": "float32",
            "shape": [7],
            "names": ["eef_x", "eef_y", "eef_z", "eef_rx", "eef_ry", "eef_rz", "prev_gripper"],
        },
    })
    if dino_dim is not None:
        info["features"][args.dino_column] = {
            "dtype": "float32",
            "shape": [dino_dim],
            "names": [f"dino_{i}" for i in range(dino_dim)],
        }
    info_path.write_text(json.dumps(info, indent=2))

    print(f"\n완료: {dst_dir}")
    print(f"  추가된 컬럼: skill_index, skill_sequence, skill_length_sequence, "
          f"skill_sequence_mask, skill_sequence_len, skill_ds, skill_de, "
          f"skill_boundary, skill_max_order, skill_max_length, skill_decoder_state")
    if dino_dim is not None:
        print(f"  추가된 DINO 컬럼: {args.dino_column} ({dino_dim} dims)")
    print(f"  episodes={info['total_episodes']}, frames={info['total_frames']}")


if __name__ == "__main__":
    main(tyro.cli(Args))
