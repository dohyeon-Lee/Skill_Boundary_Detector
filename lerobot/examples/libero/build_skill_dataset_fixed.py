"""
build_skill_dataset_fixed.py — libero 데모를 fixed-length chunk로 스킬 분할 후 .npz 저장.

build_skill_dataset.py와 동일한 출력 포맷이지만,
VF/policy 없이 단순히 고정 길이로 episode를 자름.
잔여 프레임 처리:
  - 잔여 ≤ chunk_len/2 : 마지막 스킬에 붙임 (해당 스킬이 chunk_len보다 길어짐)
  - 잔여 > chunk_len/2 : 별도 스킬로 저장 (해당 스킬은 chunk_len보다 짧음)

Output layout:
  output_dir/skills/task{task_id:02d}/ep{ep_id:05d}_task{task_id:02d}_skill{si:02d}.npz
    └── actions, states, episode_id, task_id, skill_index, frame_start, frame_end

Usage:
  python examples/libero/build_skill_dataset_fixed.py \
    --dataset_dir .../libero_90 \
    --output_dir  .../outputs/skill_dataset_fixed \
    --chunk_len 40
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tyro

sys.path.insert(0, str(Path(__file__).parent))
from skill_divider import load_data, load_episodes_meta


# ── Args ──────────────────────────────────────────────────────────────────────

@dataclass
class Args:
    dataset_dir: str = "/data2/dohyeon/SBD/libero_dataset/libero_90"
    output_dir: str = "/data2/dohyeon/SBD/outputs/skill_dataset_fixed"
    chunk_len: int = 40
    """각 스킬 세그먼트의 기준 프레임 수."""
    task_ids: list[int] | None = None
    """처리할 task index 목록. None이면 전체 task 처리."""
    resume: bool = True
    """True면 이미 저장된 episode skip."""


# ── Core ──────────────────────────────────────────────────────────────────────

def _save_skill(task_dir: Path, ep_id: int, task_id: int, si: int,
                actions: np.ndarray, states: np.ndarray,
                frame_start: int, frame_end: int) -> str:
    fname = task_dir / f"ep{ep_id:05d}_task{task_id:02d}_skill{si:02d}.npz"
    np.savez(
        str(fname),
        actions=actions.astype(np.float32),
        states=states.astype(np.float32),
        episode_id=np.array(ep_id),
        task_id=np.array(task_id),
        skill_index=np.array(si),
        frame_start=np.array(frame_start),
        frame_end=np.array(frame_end),
    )
    return str(fname)


def _save_fixed_skills(skills_dir: Path, ep_id: int, task_id: int,
                        gt_actions: np.ndarray, states: np.ndarray,
                        chunk_len: int) -> list[str]:
    n_frames  = len(gt_actions)
    n_chunks  = n_frames // chunk_len
    remainder = n_frames % chunk_len
    half      = chunk_len // 2

    # 에피소드 전체가 chunk_len 미만인 경우
    if n_chunks == 0:
        if remainder > half:
            # chunk_len/2 초과 → 단독 스킬로 저장
            task_dir = skills_dir / f"task{task_id:02d}"
            task_dir.mkdir(exist_ok=True)
            return [_save_skill(task_dir, ep_id, task_id, 0,
                                gt_actions, states, 0, n_frames)]
        return []

    task_dir = skills_dir / f"task{task_id:02d}"
    task_dir.mkdir(exist_ok=True)

    saved = []
    for si in range(n_chunks):
        s = si * chunk_len
        e = s + chunk_len
        saved.append(_save_skill(task_dir, ep_id, task_id, si,
                                 gt_actions[s:e], states[s:e], s, e))

    # 잔여 프레임 처리
    if remainder > 0:
        tail_s = n_chunks * chunk_len
        if remainder <= half:
            # 마지막 스킬에 붙여서 덮어씀
            prev_s = tail_s - chunk_len
            _save_skill(task_dir, ep_id, task_id, n_chunks - 1,
                        gt_actions[prev_s:], states[prev_s:],
                        prev_s, n_frames)
            # saved[-1] 경로는 그대로 (fname 동일), 내용만 확장됨
        else:
            # 별도 스킬로 저장
            saved.append(_save_skill(task_dir, ep_id, task_id, n_chunks,
                                     gt_actions[tail_s:], states[tail_s:],
                                     tail_s, n_frames))

    return saved


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: Args) -> None:
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    skills_dir = output_dir / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    print(f"chunk_len = {args.chunk_len}")
    print("Loading metadata...")
    episodes_meta = load_episodes_meta(dataset_dir)
    tasks_meta = pd.read_parquet(dataset_dir / "meta" / "tasks.parquet").reset_index()

    task_ids = args.task_ids if args.task_ids is not None else sorted(tasks_meta["task_index"].tolist())
    print(f"Tasks to process: {len(task_ids)}")

    n_processed = 0
    n_saved = 0
    n_skipped_resume = 0
    n_skipped_short = 0
    total_skills = 0

    for task_id in task_ids:
        task_row = tasks_meta[tasks_meta["task_index"] == task_id]
        if task_row.empty:
            print(f"  [warn] task_id {task_id} not found, skipping.")
            continue

        target_lang = task_row.iloc[0]["task"]
        ep_of_task = episodes_meta[episodes_meta["tasks"].apply(
            lambda t: target_lang in (
                [str(x) for x in t] if isinstance(t, (list, np.ndarray)) else [str(t)]
            )
        )]
        episode_ids = ep_of_task["episode_index"].tolist()
        print(f"\nTask {task_id}: '{target_lang}' — {len(episode_ids)} episodes")

        ep_data = load_data(dataset_dir, episode_ids=episode_ids)

        for ep_id in episode_ids:
            task_dir = skills_dir / f"task{task_id:02d}"
            if args.resume and any(task_dir.glob(f"ep{ep_id:05d}_task{task_id:02d}_skill*.npz")):
                existing = list(task_dir.glob(f"ep{ep_id:05d}_task{task_id:02d}_skill*.npz"))
                n_skipped_resume += 1
                n_processed += 1
                total_skills += len(existing)
                print(f"  [skip] ep{ep_id:05d} already done ({len(existing)} skills)")
                continue

            ep_df = ep_data[ep_data["episode_index"] == ep_id].reset_index(drop=True)
            if len(ep_df) == 0:
                continue

            gt_actions = np.stack(ep_df["action"].values)
            states_arr = np.stack(ep_df["observation.state"].values)

            saved = _save_fixed_skills(skills_dir, ep_id, task_id,
                                       gt_actions, states_arr, args.chunk_len)

            n_processed += 1
            if saved:
                n_saved += 1
                total_skills += len(saved)
                print(f"  ep{ep_id:05d}: {len(ep_df)} frames → {len(saved)} skills")
            else:
                n_skipped_short += 1
                print(f"  ep{ep_id:05d}: {len(ep_df)} frames < chunk_len={args.chunk_len}, skipped")

    print(f"\n{'=' * 60}")
    print(f"Done.")
    print(f"  chunk_len       : {args.chunk_len}")
    print(f"  total processed : {n_processed}")
    print(f"  saved           : {n_saved}")
    print(f"  skipped/resume  : {n_skipped_resume}")
    print(f"  skipped/short   : {n_skipped_short}")
    print(f"  total skills    : {total_skills}")
    print(f"  skills dir      : {skills_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
