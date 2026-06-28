"""
Stage-2 (SkillVLA) 데이터셋 빌더 — 원본 LeRobot 데이터셋에 스킬 컬럼을 추가하고,
스킬 시작 시점 randomization용 "skill-initial-state" npz를 따로 만든다.

이미지는 원본 video(mp4) 그대로 두고(프레임 참조), parquet에는 스킬 메타 컬럼만 붙인다.
VLM/action expert는 학습·추론 모두 raw 이미지를 자기 인코더로 on-the-fly 통과시키므로
precompute된 DINO/visual feature 캐시는 만들지 않는다.

────────────────────────────────────────────────────────────────────────────
parquet에 추가되는 per-frame 컬럼  (max_seq_len = max_order + 1, BOS 없음)
────────────────────────────────────────────────────────────────────────────
  skill_index          : int32              현재 프레임이 속한 스킬의 0-based index (첫 실제 스킬 = 0)
  skill_sequence (SS)  : (max_seq_len,) int32   [skill0..skill_{N-1}, EOS, PAD...]  에피소드 스킬 코드열
  skill_length_sequence: (max_seq_len,) int32   [len0..len_{N-1}, 0, 0...]          스킬별 프레임 길이
  skill_initial_frame  : (max_seq_len,) int32   [fs0..fs_{N-1}, -1, -1...]  (IFS) 스킬별 "시작 프레임 index"
                                                 → VLM 시작 이미지 디코딩 + ISS npz 교차검증 키
  skill_sequence_mask  : (max_seq_len,) int8    real skill/EOS = 1, PAD = 0
  skill_sequence_len   : int32              real skill 개수 + EOS (= N + 1)
  skill_ds             : int32              현재 스킬 시작으로부터의 거리 (시작 프레임 = 0)
  skill_de             : int32              현재 스킬 끝까지 남은 거리 (끝 프레임 = 0)
  skill_boundary       : int8               현재 스킬 마지막 프레임에서만 1
  skill_max_order      : int32              허용 최대 real skill 개수
  skill_max_length     : int32              FSQ와 공유하는 skill max length
  skill_decoder_state  : (state_dim,) float32   FSQ decoder/terminator용 state (= observation.state 전체)

skill_sequence 토큰 = FSQ scalar code(0 ~ prod(fsq_levels)-1). 특수 토큰은 그 바로 위:
  EOS = num_embeddings(K),  PAD = K + 1     (BOS 제거 — Stage-2 VLM이 scene에서 skill을 직접 예측하므로 불필요)

────────────────────────────────────────────────────────────────────────────
별도 산출물: skill-initial-state npz  ({iss_npz_path})   ← Stage-2 transition randomization용
────────────────────────────────────────────────────────────────────────────
  episode_id  : (total_skills,) int32        각 스킬이 속한 episode_index (parquet과 같은 id 규약)
  frame_start : (total_skills,) int32        각 스킬 시작 프레임 (= 그 스킬의 IFS, 교차검증용)
  offsets     : (n_episodes+1,) int64        episode → 스킬 슬라이스 (offsets[i]:offsets[i+1])
  iss_windows : (total_skills, 2*pmax+1, state_dim) float32
                스킬 시작 ±pmax 프레임의 observation.state 윈도우 (에피 경계 clamp).
                중앙 index [pmax] = 실제 시작 프레임의 state.

학습 로더는 매 스텝 p~half-normal[0,pmax]를 뽑아 스킬 시작을 jitter한다(early/late/else):
  이미지 = IFS[k'] ± p 프레임을 video에서 디코딩,  state = iss_windows[k'][pmax ± p].
  (이 스크립트는 데이터만 준비하고, jitter/디코딩은 로더가 수행.)

마지막 스킬 이후 남은 프레임은 마지막 스킬 index로 채워지고 ds는 이어서 카운팅된다.
스킬 latent가 없는 에피소드는 제거하지 않고 모든 컬럼을 0으로 채운다.

Usage:
    python examples/libero/add_skill_latents_to_dataset.py \
        --src_dataset_dir .../libero_90 \
        --dst_dataset_dir .../libero_90_skillvla \
        --dst_repo_id skillvla/libero_90 \
        --latents_path .../skill_latents.npz \
        --iss_npz_path .../skill_initial_state.npz \
        --fsq_levels 8 6 5 --max_order 20 --max_length 200 --pmax 10
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
    """FSQ codebook size = prod(fsq_levels). EOS=K, PAD=K+1 (no BOS). fsq_levels 미지정 시 직접 입력."""
    pmax: int = 10
    """Stage-2 transition randomization half-window (steps). ISS는 스킬 시작 ±pmax state를 저장."""
    iss_npz_path: str = ""
    """skill-initial-state npz 출력 경로. 비우면 dst_dataset_dir 옆 skill_initial_state.npz."""
    ess_npz_path: str = ""
    """skill-FINAL-state npz 출력 경로(Stage-1 connector의 종료 프레임 state). 비우면 dst_dataset_dir
    옆 skill_final_state.npz. ISS와 동일하게 ±pmax window라 transition randomization 대비."""
    state_column: str = "observation.state"
    """ISS/decoder-state로 쓸 로봇 state 컬럼."""


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


# ── episode별 컬럼 계산 ───────────────────────────────────────────────────────

def compute_skill_columns(
    ep_df: pd.DataFrame,
    skills: list[tuple[int, int, int]],
    max_order: int = 1,
    max_length: int = 200,
    eos_token_id: int = 512,
    pad_token_id: int = 513,
) -> dict[str, np.ndarray]:
    """
    ep_df : frame_index 순으로 정렬된 한 에피소드의 DataFrame
    skills: [(frame_start, frame_end, token), ...] sorted by frame_start

    Returns dict of arrays, each length len(ep_df). (BOS 없음 — skill_index 0-based,
    skill_sequence = [skill0..skill_{N-1}, EOS, PAD...], IFS는 스킬별 시작 프레임 index.)
    """
    n = len(ep_df)
    frames = ep_df["frame_index"].values.astype(np.int64)
    max_seq_len = max_order + 1  # real skills + EOS (no BOS)

    skill_index_arr = np.zeros(n, dtype=np.int32)   # default 0 = 첫 실제 스킬 (pre-skill 프레임도 여기로)
    assigned = np.zeros(n, dtype=bool)
    ds_arr = np.zeros(n, dtype=np.int32)
    de_arr = np.zeros(n, dtype=np.int32)
    boundary_arr = np.zeros(n, dtype=np.int8)
    max_order_arr = np.full(n, max_order, dtype=np.int32)
    max_length_arr = np.full(n, max_length, dtype=np.int32)

    seq_tokens = np.full((max_seq_len,), pad_token_id, dtype=np.int32)
    seq_lengths = np.zeros((max_seq_len,), dtype=np.int32)
    seq_mask = np.zeros((max_seq_len,), dtype=np.int8)
    seq_initial_frame = np.full((max_seq_len,), -1, dtype=np.int32)  # IFS: 스킬 시작 프레임 (EOS/PAD = -1)

    n_real = len(skills)
    if n_real > max_order:
        raise ValueError(f"Episode has {n_real} skills, but max_order={max_order}.")

    if skills:
        seq_tokens[:n_real] = np.array([tok for _, _, tok in skills], dtype=np.int32)
        seq_lengths[:n_real] = np.array([max(0, fe - fs) for fs, fe, _ in skills], dtype=np.int32)
        seq_initial_frame[:n_real] = np.array([fs for fs, _, _ in skills], dtype=np.int32)
        seq_mask[:n_real] = 1

    eos_index = n_real
    seq_tokens[eos_index] = eos_token_id
    seq_mask[eos_index] = 1
    seq_len = n_real + 1
    seq_len_arr = np.full(n, seq_len, dtype=np.int32)
    seq_arr = np.repeat(seq_tokens[None, :], n, axis=0)
    seq_length_arr = np.repeat(seq_lengths[None, :], n, axis=0)
    seq_mask_arr = np.repeat(seq_mask[None, :], n, axis=0)
    seq_if_arr = np.repeat(seq_initial_frame[None, :], n, axis=0)

    for skill_rank, (fs, fe, tok) in enumerate(skills):
        mask = (frames >= fs) & (frames < fe)
        if not mask.any():
            continue
        skill_index_arr[mask] = skill_rank   # 0-based (no BOS)
        ds_arr[mask] = frames[mask] - fs
        de_arr[mask] = np.maximum((fe - 1) - frames[mask], 0)
        boundary_arr[mask & (frames == fe - 1)] = 1
        assigned |= mask

    # 마지막 스킬 이후 남은 프레임: 마지막 스킬 index로 채우고 ds 이어서 카운팅 (de=0)
    if skills:
        last_fs, last_fe, _ = skills[-1]
        leftover = (~assigned) & (frames >= last_fe)
        if leftover.any():
            skill_index_arr[leftover] = n_real - 1
            ds_arr[leftover] = frames[leftover] - last_fs
            de_arr[leftover] = 0
            assigned |= leftover
    # 첫 스킬 이전 프레임(있다면)은 skill 0, ds=0 기본값으로 둠 (드묾)

    return {
        "skill_index": skill_index_arr,
        "skill_sequence": list(seq_arr),
        "skill_length_sequence": list(seq_length_arr),
        "skill_initial_frame": list(seq_if_arr),
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
    """State input used by the FSQ decoder/terminator = the FULL raw observation.state
    (ee pose + gripper STATE), matching train_FSQ._make_decoder_state.

    (The previous FSQ used ee pose + the previous gripper *command* — a 7-dim state.
    The current FSQ uses the observed gripper STATE that is already part of
    observation.state, so no action-derived gripper is needed and there is no target
    leak. action_column / eef_state_dim / gripper_action_dim are kept for signature
    compatibility but unused.)
    """
    if state_column not in ep_df.columns:
        raise KeyError(f"Missing required column: {state_column}")
    states = np.stack(ep_df[state_column].to_numpy()).astype(np.float32)
    return [row.astype(np.float32) for row in states]


# ── skill-initial-state npz (Stage-2 transition randomization) ───────────────────

def build_skill_initial_state_npz(
    skill_map: dict[int, list],
    ep_states: dict[int, np.ndarray],
    pmax: int,
    state_dim: int,
    out_path: Path,
) -> int:
    """각 스킬의 시작 ±pmax 프레임 observation.state 윈도우를 flat npz로 저장.

    skill_latents.npz와 같은 규약(per-skill flat + episode_id 키)이라, 로더가 episode_index로
    그룹핑 + frame_start 정렬(=skill_index 순서)해서 [episode][k]로 인덱싱한다. frame_start는
    parquet의 skill_initial_frame(IFS)과 교차검증용.
    """
    win = 2 * pmax + 1
    episode_ids: list[int] = []
    frame_starts: list[int] = []
    windows: list[np.ndarray] = []
    for ep_id in sorted(skill_map.keys()):
        states = ep_states.get(int(ep_id))
        for fs, _fe, _tok in skill_map[ep_id]:  # sorted by frame_start
            if states is None or len(states) == 0:
                w = np.zeros((win, state_dim), dtype=np.float32)
            else:
                idx = np.clip(np.arange(fs - pmax, fs + pmax + 1), 0, len(states) - 1)
                w = states[idx].astype(np.float32)  # (win, state_dim), 경계 clamp; 중앙 [pmax]=시작
            episode_ids.append(int(ep_id))
            frame_starts.append(int(fs))
            windows.append(w)
    iss = np.stack(windows) if windows else np.zeros((0, win, state_dim), dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        episode_id=np.asarray(episode_ids, dtype=np.int32),
        frame_start=np.asarray(frame_starts, dtype=np.int32),
        iss_windows=iss,
        pmax=np.int32(pmax),
        state_dim=np.int32(state_dim),
    )
    return len(windows)


def build_skill_final_state_npz(
    skill_map: dict[int, list],
    ep_states: dict[int, np.ndarray],
    pmax: int,
    state_dim: int,
    out_path: Path,
) -> int:
    """각 스킬의 끝(마지막 프레임 = fe-1) ±pmax observation.state 윈도우를 flat npz로 저장 — Stage-1
    connector의 종료 프레임 state 입력.

    ISS(build_skill_initial_state_npz)와 완전히 동일한 규약(per-skill flat + episode_id 키, 중앙
    [pmax] = 끝 프레임). window로 저장하므로 Stage-1 transition randomization을 켜도 재빌드 없이
    ess_index=pmax+offset 로 경계 jitter를 인덱싱할 수 있다(현재 로더는 중앙만 사용). frame_end는
    parquet의 IFS[k]+length-1 과 교차검증용.
    """
    win = 2 * pmax + 1
    episode_ids: list[int] = []
    frame_ends: list[int] = []
    windows: list[np.ndarray] = []
    for ep_id in sorted(skill_map.keys()):
        states = ep_states.get(int(ep_id))
        for _fs, fe, _tok in skill_map[ep_id]:  # sorted by frame_start
            end = fe - 1                        # last frame of the skill ([fs, fe) → fe-1)
            if states is None or len(states) == 0:
                w = np.zeros((win, state_dim), dtype=np.float32)
            else:
                idx = np.clip(np.arange(end - pmax, end + pmax + 1), 0, len(states) - 1)
                w = states[idx].astype(np.float32)  # (win, state_dim), 경계 clamp; 중앙 [pmax]=끝
            episode_ids.append(int(ep_id))
            frame_ends.append(int(end))
            windows.append(w)
    ess = np.stack(windows) if windows else np.zeros((0, win, state_dim), dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        episode_id=np.asarray(episode_ids, dtype=np.int32),
        frame_end=np.asarray(frame_ends, dtype=np.int32),
        ess_windows=ess,
        pmax=np.int32(pmax),
        state_dim=np.int32(state_dim),
    )
    return len(windows)


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
    eos_token_id = num_embeddings        # EOS = K
    pad_token_id = num_embeddings + 1    # PAD = K+1  (BOS 제거)
    skill_output_vocab_size = num_embeddings + 1  # FSQ tokens + EOS
    skill_vocab_size = num_embeddings + 2         # FSQ tokens + EOS/PAD (no BOS)
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
    max_seq_len = max_order + 1   # real skills + EOS (no BOS)
    print(f"  observed_max_order={observed_max_order}, max_order={max_order}")
    print(f"  observed_max_length={observed_max_length}, max_length={max_length}")
    print(f"  max_skill_sequence_len={max_seq_len} (real skills + EOS, no BOS)")
    print(
        f"  eos_token_id={eos_token_id}, pad_token_id={pad_token_id}, "
        f"skill_vocab_size={skill_vocab_size}, skill_output_vocab_size={skill_output_vocab_size}"
    )
    pmax = int(args.pmax)
    iss_npz_path = Path(args.iss_npz_path) if args.iss_npz_path else dst_dir.parent / "skill_initial_state.npz"
    ep_states_map: dict[int, np.ndarray] = {}   # episode_index → (ep_len, state_dim), ISS 윈도우용
    state_dim: int | None = None
    print(f"  pmax={pmax}  →  ISS window={2 * pmax + 1}, npz={iss_npz_path}")

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
            "skill_initial_frame": [],
            "skill_sequence_mask": [],
            "skill_sequence_len": [],
            "skill_ds": [],
            "skill_de": [],
            "skill_boundary": [],
            "skill_max_order": [],
            "skill_max_length": [],
            "skill_decoder_state": [],
        }

        for ep_id, ep_df in df.groupby("episode_index"):
            ep_df  = ep_df.sort_values("frame_index")
            skills = skill_map.get(int(ep_id), [])
            cols   = compute_skill_columns(
                ep_df,
                skills,
                max_order=max_order,
                max_length=max_length,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
            cols["skill_decoder_state"] = compute_skill_decoder_state(ep_df, cols["skill_ds"])
            for k in col_buffers:
                col_buffers[k].extend(cols[k] if isinstance(cols[k], list) else cols[k].tolist())
            # ISS 윈도우용: 이 에피소드의 frame-정렬 state (frame_index 0-based 가정)
            ep_states = np.stack(ep_df[args.state_column].to_numpy()).astype(np.float32)
            ep_states_map[int(ep_id)] = ep_states
            if state_dim is None:
                state_dim = int(ep_states.shape[1])

        for k, vals in col_buffers.items():
            df[k] = vals

        df.to_parquet(parquet_path, index=False)

    print(f"  Zero-filled {n_zero_fill_eps} episodes without skill tokens (kept in dataset)")

    # ── skill-initial-state npz (Stage-2 randomization) ──
    if state_dim is None:
        state_dim = 0
    n_iss = build_skill_initial_state_npz(skill_map, ep_states_map, pmax, state_dim, iss_npz_path)
    print(f"  Wrote ISS npz: {iss_npz_path}  (skills={n_iss}, window={2 * pmax + 1}, state_dim={state_dim})")

    # ── skill-FINAL-state npz (Stage-1 connector end-frame state; ±pmax window → randomization-ready) ──
    ess_npz_path = Path(args.ess_npz_path) if args.ess_npz_path else dst_dir.parent / "skill_final_state.npz"
    n_ess = build_skill_final_state_npz(skill_map, ep_states_map, pmax, state_dim, ess_npz_path)
    print(f"  Wrote ESS npz: {ess_npz_path}  (skills={n_ess}, window={2 * pmax + 1}, state_dim={state_dim})")

    # ── Update info.json ──────────────────────────────────────────────────────
    info_path = dst_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["repo_id"] = args.dst_repo_id
    info["skill_num_embeddings"] = num_embeddings
    info["skill_fsq_levels"] = list(args.fsq_levels) if args.fsq_levels else []
    info["skill_eos_token_id"] = eos_token_id
    info["skill_pad_token_id"] = pad_token_id          # BOS 제거됨
    info["skill_vocab_size"] = skill_vocab_size
    info["skill_output_vocab_size"] = skill_output_vocab_size
    info["skill_max_order"] = max_order
    info["skill_observed_max_order"] = observed_max_order
    info["skill_max_length"] = max_length
    info["skill_observed_max_length"] = observed_max_length
    info["skill_sequence_size"] = max_seq_len
    info["skill_pmax"] = pmax                          # ISS window 반폭 (= transition randomization)
    info["skill_initial_state_path"] = str(iss_npz_path)
    info["skill_initial_state_window"] = 2 * pmax + 1
    info["skill_final_state_path"] = str(ess_npz_path)        # Stage-1 connector end-frame state
    info["skill_final_state_window"] = 2 * pmax + 1

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
        "skill_initial_frame": {
            "dtype": "int32",
            "shape": [max_seq_len],
            "names": [f"skill_if_{i}" for i in range(max_seq_len)],
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
            "shape": [state_dim],
            "names": [f"decoder_state_{i}" for i in range(state_dim)],
        },
    })
    info_path.write_text(json.dumps(info, indent=2))

    print(f"\n완료: {dst_dir}")
    print(f"  추가된 컬럼: skill_index, skill_sequence, skill_length_sequence, skill_initial_frame, "
          f"skill_sequence_mask, skill_sequence_len, skill_ds, skill_de, "
          f"skill_boundary, skill_max_order, skill_max_length, skill_decoder_state")
    print(f"  ISS npz: {iss_npz_path}  (window={2 * pmax + 1}, state_dim={state_dim})")
    print(f"  episodes={info['total_episodes']}, frames={info['total_frames']}")


if __name__ == "__main__":
    main(tyro.cli(Args))
