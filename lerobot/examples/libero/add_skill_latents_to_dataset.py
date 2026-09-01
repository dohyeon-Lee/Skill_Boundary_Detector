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

학습 로더는 설정된 half-normal/uniform 분포에서 p를 뽑아 스킬 시작을 jitter한다(early/late/else):
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


PROPRIO_GROUNDING_NONE = "none"
PROPRIO_GROUNDING_EPISODE_START_XYZ = "episode_start_xyz"
PROPRIO_GROUNDING_MODES = {
    PROPRIO_GROUNDING_NONE,
    PROPRIO_GROUNDING_EPISODE_START_XYZ,
}
STAT_QUANTILES = (
    ("q01", 0.01),
    ("q10", 0.10),
    ("q50", 0.50),
    ("q90", 0.90),
    ("q99", 0.99),
)


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
    """ISS storage half-window; must equal the maximum directional pmax."""
    early_start_pmax: int = -1
    late_start_pmax: int = -1
    early_end_pmax: int = -1
    late_end_pmax: int = -1
    """Directional transition-jitter windows; -1 falls back to pmax."""
    jitter_distribution: str = "half_normal"
    """Transition jitter distribution: half_normal or uniform."""
    iss_npz_path: str = ""
    """skill-initial-state npz 출력 경로. 비우면 dst_dataset_dir 옆 skill_initial_state.npz."""
    state_column: str = "observation.state"
    """ISS/decoder-state로 쓸 로봇 state 컬럼."""
    proprio_grounding: str = PROPRIO_GROUNDING_NONE
    """none or episode_start_xyz. Applied in raw units before state stats."""
    rotation_outlier_threshold: float = 0.0
    """If positive, exclude episodes containing |action[3:6]| above this value from every SkillVLA loader."""


def normalize_proprio_grounding(value: str) -> str:
    mode = str(value or PROPRIO_GROUNDING_NONE).strip().lower().replace("-", "_")
    aliases = {"off": PROPRIO_GROUNDING_NONE, "false": PROPRIO_GROUNDING_NONE}
    mode = aliases.get(mode, mode)
    if mode not in PROPRIO_GROUNDING_MODES:
        raise ValueError(
            "proprio_grounding must be none|episode_start_xyz, "
            f"got {value!r}."
        )
    return mode


def ground_episode_state_xyz(
    states: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Ground raw EEF xyz to the first available frame of one episode."""
    values = np.asarray(states, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 3:
        raise ValueError(
            "Episode proprio state must have shape (T,D), T>0, D>=3; "
            f"got {values.shape}."
        )
    reference = values[0, :3].copy()
    if mode == PROPRIO_GROUNDING_NONE:
        return values.copy(), reference
    if mode != PROPRIO_GROUNDING_EPISODE_START_XYZ:
        raise ValueError(f"Unsupported proprio grounding mode: {mode!r}.")
    grounded = values.copy()
    grounded[:, :3] -= reference
    return grounded, reference


def exact_vector_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    """Compute the global non-video stats contract used by LeRobot normalizers."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(f"Stats input must be a non-empty (N,D) array, got {array.shape}.")
    result: dict[str, list[float] | list[int]] = {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [int(array.shape[0])],
    }
    for key, quantile in STAT_QUANTILES:
        result[key] = np.quantile(array, quantile, axis=0).tolist()
    return result


def find_rotation_outlier_episodes(
    dataset_dir: Path,
    threshold: float,
) -> set[int]:
    """Find whole episodes containing a saturated rotation action.

    Detection happens on the immutable source parquets before SkillVLA columns
    are added. Episode ids are preserved in the copied dataset, so the same ids
    can be stored as a stable training-exclusion contract without rebuilding
    segmentation or FSQ latents.
    """
    if threshold <= 0.0:
        return set()
    excluded: set[int] = set()
    parquet_paths = sorted((dataset_dir / "data").rglob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No source parquet files under {dataset_dir / 'data'}.")
    for parquet_path in parquet_paths:
        frame = pd.read_parquet(parquet_path, columns=["episode_index", "action"])
        if frame.empty:
            continue
        actions = np.stack(frame["action"].to_numpy()).astype(np.float32)
        if actions.ndim != 2 or actions.shape[1] < 6:
            raise ValueError(
                "Rotation-outlier exclusion requires action vectors with at least "
                f"6 dimensions, got {actions.shape} in {parquet_path}."
            )
        mask = np.max(np.abs(actions[:, 3:6]), axis=1) > threshold
        excluded.update(int(value) for value in frame.loc[mask, "episode_index"].unique())
    return excluded


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
    jitter_distribution: str,
    directional_pmaxes: dict[str, int],
    proprio_grounding: str = PROPRIO_GROUNDING_NONE,
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
        early_start_pmax=np.int32(directional_pmaxes["early_start"]),
        late_start_pmax=np.int32(directional_pmaxes["late_start"]),
        early_end_pmax=np.int32(directional_pmaxes["early_end"]),
        late_end_pmax=np.int32(directional_pmaxes["late_end"]),
        jitter_distribution=np.str_(jitter_distribution),
        proprio_grounding=np.str_(proprio_grounding),
        state_dim=np.int32(state_dim),
    )
    return len(windows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: Args) -> None:
    src_dir = Path(args.src_dataset_dir)
    dst_dir = Path(args.dst_dataset_dir)
    proprio_grounding = normalize_proprio_grounding(args.proprio_grounding)
    rotation_outlier_threshold = float(args.rotation_outlier_threshold)
    if rotation_outlier_threshold < 0.0:
        raise ValueError(
            "rotation_outlier_threshold must be non-negative, got "
            f"{rotation_outlier_threshold}."
        )
    excluded_episode_ids = find_rotation_outlier_episodes(
        src_dir, rotation_outlier_threshold
    )
    if rotation_outlier_threshold > 0.0:
        print(
            "Rotation-outlier exclusion: "
            f"threshold={rotation_outlier_threshold:g}, "
            f"episodes={sorted(excluded_episode_ids)}"
        )

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
    if excluded_episode_ids:
        skill_map = {
            episode_id: skills
            for episode_id, skills in skill_map.items()
            if episode_id not in excluded_episode_ids
        }
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
    directional_pmaxes = {
        name: (
            pmax
            if int(getattr(args, f"{name}_pmax")) < 0
            else int(getattr(args, f"{name}_pmax"))
        )
        for name in ("early_start", "late_start", "early_end", "late_end")
    }
    if pmax < 0 or any(value < 0 for value in directional_pmaxes.values()):
        raise ValueError(
            f"Transition-jitter pmax values must be non-negative: "
            f"storage={pmax}, directional={directional_pmaxes}."
        )
    expected_storage_pmax = max(directional_pmaxes.values())
    if pmax != expected_storage_pmax:
        raise ValueError(
            "--pmax must equal the largest directional pmax so the ISS window "
            f"covers every draw: pmax={pmax}, directional={directional_pmaxes}."
        )
    jitter_distribution = str(args.jitter_distribution).strip().lower().replace("-", "_").replace(" ", "_")
    if jitter_distribution not in {"half_normal", "uniform"}:
        raise ValueError(
            "--jitter_distribution must be half_normal|uniform, "
            f"got {args.jitter_distribution!r}."
        )
    iss_npz_path = Path(args.iss_npz_path) if args.iss_npz_path else dst_dir.parent / "skill_initial_state.npz"
    ep_states_map: dict[int, np.ndarray] = {}   # episode_index → (ep_len, state_dim), ISS 윈도우용
    state_dim: int | None = None
    print(
        f"  pmax={pmax}, directional_pmaxes={directional_pmaxes}, "
        f"jitter_distribution={jitter_distribution}  "
        f"→  ISS window={2 * pmax + 1}, npz={iss_npz_path}"
    )
    print(f"  proprio_grounding={proprio_grounding}")

    if dst_dir.exists():
        print(f"Removing existing {dst_dir} ...")
        shutil.rmtree(dst_dir)
    print(f"Copying {src_dir} → {dst_dir} ...")
    shutil.copytree(src_dir, dst_dir)

    data_files = sorted((dst_dir / "data").rglob("*.parquet"))
    print(f"Processing {len(data_files)} parquet files ...")

    n_zero_fill_eps: int = 0
    training_actions: list[np.ndarray] = []
    training_states: list[np.ndarray] = []
    training_ee_states: list[np.ndarray] = []

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
        grounded_state_rows: list[np.ndarray] = []
        grounded_ee_rows: list[np.ndarray] = []
        has_ee_state = "observation.states.ee_state" in df.columns

        for ep_id, ep_df in df.groupby("episode_index"):
            ep_df = ep_df.sort_values("frame_index").copy()
            episode_id = int(ep_id)
            excluded_from_training = episode_id in excluded_episode_ids
            raw_states = np.stack(ep_df[args.state_column].to_numpy()).astype(np.float32)
            ep_states, episode_start_xyz = ground_episode_state_xyz(
                raw_states, proprio_grounding
            )
            ep_df[args.state_column] = list(ep_states)
            grounded_state_rows.extend(ep_states)
            if not excluded_from_training:
                training_states.append(ep_states)
                training_actions.append(
                    np.stack(ep_df["action"].to_numpy()).astype(np.float32)
                )
            if has_ee_state:
                ee_states = np.stack(
                    ep_df["observation.states.ee_state"].to_numpy()
                ).astype(np.float32)
                if ee_states.ndim != 2 or ee_states.shape[1] < 3:
                    raise ValueError(
                        "observation.states.ee_state must have shape (T,D), D>=3; "
                        f"got {ee_states.shape}."
                    )
                ee_states = ee_states.copy()
                if proprio_grounding == PROPRIO_GROUNDING_EPISODE_START_XYZ:
                    ee_states[:, :3] -= episode_start_xyz
                ep_df["observation.states.ee_state"] = list(ee_states)
                grounded_ee_rows.extend(ee_states)
                if not excluded_from_training:
                    training_ee_states.append(ee_states)
            skills = skill_map.get(episode_id, [])
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
            if not excluded_from_training:
                ep_states_map[episode_id] = ep_states
            if state_dim is None:
                state_dim = int(ep_states.shape[1])

        for k, vals in col_buffers.items():
            df[k] = vals
        df[args.state_column] = grounded_state_rows
        if has_ee_state:
            df["observation.states.ee_state"] = grounded_ee_rows

        df.to_parquet(parquet_path, index=False)

    print(f"  Zero-filled {n_zero_fill_eps} episodes without skill tokens (kept in dataset)")

    # ── skill-initial-state npz (Stage-2 randomization) ──
    if state_dim is None:
        state_dim = 0
    n_iss = build_skill_initial_state_npz(
        skill_map,
        ep_states_map,
        pmax,
        state_dim,
        iss_npz_path,
        jitter_distribution,
        directional_pmaxes,
        proprio_grounding,
    )
    print(f"  Wrote ISS npz: {iss_npz_path}  (skills={n_iss}, window={2 * pmax + 1}, state_dim={state_dim})")

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
    info["skill_jitter_early_start_pmax"] = directional_pmaxes["early_start"]
    info["skill_jitter_late_start_pmax"] = directional_pmaxes["late_start"]
    info["skill_jitter_early_end_pmax"] = directional_pmaxes["early_end"]
    info["skill_jitter_late_end_pmax"] = directional_pmaxes["late_end"]
    info["skill_jitter_distribution"] = jitter_distribution
    info["skill_initial_state_path"] = str(iss_npz_path)
    info["skill_initial_state_window"] = 2 * pmax + 1
    info["proprio_grounding"] = proprio_grounding
    info["proprio_grounding_reference"] = (
        "episode_first_frame_observation.state[:3]"
        if proprio_grounding == PROPRIO_GROUNDING_EPISODE_START_XYZ
        else "none"
    )
    info["proprio_grounding_xyz_indices"] = [0, 1, 2]
    info["training_excluded_episode_ids"] = sorted(excluded_episode_ids)
    info["training_exclusion_contract"] = {
        "type": "action_rotation_abs_threshold",
        "rotation_indices": [3, 4, 5],
        "threshold": rotation_outlier_threshold,
        "enabled": rotation_outlier_threshold > 0.0,
    }
    info["training_total_episodes"] = int(info["total_episodes"]) - len(
        excluded_episode_ids
    )
    info["training_total_frames"] = int(
        sum(values.shape[0] for values in training_states)
    )

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

    if (
        proprio_grounding == PROPRIO_GROUNDING_EPISODE_START_XYZ
        or excluded_episode_ids
    ):
        # Grounding changes state values; exclusion changes the population used
        # by training. In either case write exact stats for the effective
        # training subset while leaving image statistics untouched.
        if not training_states or not training_actions:
            raise RuntimeError("The effective SkillVLA training subset is empty.")
        all_states = np.concatenate(training_states, axis=0)
        stats_path = dst_dir / "meta" / "stats.json"
        stats = json.loads(stats_path.read_text()) if stats_path.is_file() else {}
        state_stats = exact_vector_stats(all_states)
        stats[args.state_column] = state_stats
        stats["skill_decoder_state"] = state_stats
        stats["action"] = exact_vector_stats(
            np.concatenate(training_actions, axis=0)
        )
        if training_ee_states:
            stats["observation.states.ee_state"] = exact_vector_stats(
                np.concatenate(training_ee_states, axis=0)
            )
        stats_path.write_text(json.dumps(stats, indent=2))
        print(
            "  Recomputed effective-training stats: action, "
            f"{args.state_column}, skill_decoder_state"
            + (", observation.states.ee_state" if training_ee_states else "")
        )

    print(f"\n완료: {dst_dir}")
    print(f"  추가된 컬럼: skill_index, skill_sequence, skill_length_sequence, skill_initial_frame, "
          f"skill_sequence_mask, skill_sequence_len, skill_ds, skill_de, "
          f"skill_boundary, skill_max_order, skill_max_length, skill_decoder_state")
    print(f"  ISS npz: {iss_npz_path}  (window={2 * pmax + 1}, state_dim={state_dim})")
    print(f"  episodes={info['total_episodes']}, frames={info['total_frames']}")


if __name__ == "__main__":
    main(tyro.cli(Args))
