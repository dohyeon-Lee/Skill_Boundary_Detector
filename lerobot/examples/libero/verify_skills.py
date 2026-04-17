"""
verify_skills.py — skill_divider 결과와 build_skill_dataset 결과 일치 여부 검증.

비교 항목:
  1. 스킬 개수   : skill_divider mp4 수  vs  npz 파일 수
  2. 경계 프레임 : results.json div_cos_peak_ts  vs  npz frame_start
  3. 세그먼트 길이: mp4 프레임 수  vs  npz (frame_end - frame_start)
  4. 액션 데이터 : 원본 parquet GT  vs  npz actions (max abs diff)

Usage:
  python examples/libero/verify_skills.py \
    --task_id 0 \
    --episode_id 0 \
    --skill_divider_dir .../outputs/skill_divider/task0 \
    --skill_dataset_dir .../outputs/skill_dataset/skills/task00 \
    --dataset_dir .../libero_dataset/libero_90
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio
import numpy as np
import pandas as pd
import tyro

sys.path.insert(0, str(Path(__file__).parent))
from skill_divider import load_data, load_episodes_meta


@dataclass
class Args:
    task_id: int = 0
    episode_id: int = 0
    skill_divider_dir: str = ""
    """skill_divider 출력 디렉토리 (e.g. .../skill_divider/task0)"""
    skill_dataset_dir: str = ""
    """build_skill_dataset 출력 디렉토리 (e.g. .../skill_dataset/skills/task00)"""
    dataset_dir: str = ""
    """libero_90 원본 데이터셋 디렉토리 (액션 비교용)"""
    action_tol: float = 1e-4
    """액션 일치 판정 허용 오차 (max abs diff)"""


PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
WARN = "\033[93m! WARN\033[0m"


def _check(label: str, ok: bool, detail: str = "") -> bool:
    status = PASS if ok else FAIL
    print(f"  {status}  {label}" + (f"  — {detail}" if detail else ""))
    return ok


def main(args: Args) -> None:
    ep_id = args.episode_id
    task_id = args.task_id
    divider_dir = Path(args.skill_divider_dir)
    dataset_dir = Path(args.skill_dataset_dir)

    print(f"\n{'=' * 60}")
    print(f"Verifying  task={task_id}  episode={ep_id:05d}")
    print(f"  skill_divider : {divider_dir}")
    print(f"  skill_dataset : {dataset_dir}")
    print(f"{'=' * 60}\n")

    all_pass = True

    # ── 1. results.json 로드 ──────────────────────────────────────────────────
    results_json = divider_dir / "results.json"
    if not results_json.exists():
        print(f"  {FAIL}  results.json not found: {results_json}")
        return

    with open(results_json) as f:
        results = json.load(f)
    ep_result = next((r for r in results if r["episode_id"] == ep_id), None)
    if ep_result is None:
        print(f"  {FAIL}  episode {ep_id} not found in results.json")
        return

    peak_ts = ep_result["div_cos_peak_ts"]
    n_frames = ep_result["n_frames"]
    divider_boundaries = sorted(set([0] + [int(p) for p in peak_ts] + [n_frames]))
    n_skills_divider = len(divider_boundaries) - 1
    print(f"[skill_divider]  boundaries: {divider_boundaries}  ({n_skills_divider} skills)")

    # ── 2. npz 로드 ───────────────────────────────────────────────────────────
    npz_files = sorted(dataset_dir.glob(f"ep{ep_id:05d}_task{task_id:02d}_skill*.npz"))
    if not npz_files:
        print(f"  {FAIL}  No npz files found for ep{ep_id:05d} in {dataset_dir}")
        return

    npz_data = [np.load(str(f)) for f in npz_files]
    npz_starts = [int(d["frame_start"]) for d in npz_data]
    npz_ends   = [int(d["frame_end"])   for d in npz_data]
    n_skills_dataset = len(npz_files)
    print(f"[skill_dataset]  boundaries: {sorted(set(npz_starts + npz_ends))}  ({n_skills_dataset} skills)")

    print()

    # ── Check 1: 스킬 개수 ─────────────────────────────────────────────────────
    ok = n_skills_divider == n_skills_dataset
    all_pass &= _check(
        f"스킬 개수",
        ok,
        f"skill_divider={n_skills_divider}  skill_dataset={n_skills_dataset}",
    )

    # ── Check 2: 경계 프레임 ───────────────────────────────────────────────────
    divider_inner = divider_boundaries[1:-1]  # 0과 n_frames 제외
    dataset_inner = sorted(npz_starts[1:])    # skill_index 0 제외
    ok = divider_inner == dataset_inner
    all_pass &= _check(
        f"경계 프레임 일치",
        ok,
        f"skill_divider={divider_inner}  skill_dataset={dataset_inner}",
    )

    # ── Check 3: 세그먼트 길이 (mp4 프레임 수 vs npz) ─────────────────────────
    mp4_files = sorted(divider_dir.glob(f"ep{ep_id:05d}_skill*.mp4"))
    if mp4_files:
        mp4_lengths = []
        for mp4 in mp4_files:
            try:
                r = imageio.get_reader(str(mp4))
                mp4_lengths.append(r.count_frames())
                r.close()
            except Exception as e:
                mp4_lengths.append(None)
                print(f"  {WARN}  mp4 read error {mp4.name}: {e}")

        npz_lengths = [e - s for s, e in zip(npz_starts, npz_ends)]

        if len(mp4_lengths) == len(npz_lengths):
            mismatches = [(i, mp4_lengths[i], npz_lengths[i])
                          for i in range(len(mp4_lengths))
                          if mp4_lengths[i] is not None and mp4_lengths[i] != npz_lengths[i]]
            ok = len(mismatches) == 0
            detail = "all match" if ok else f"mismatch at skill(s) {[m[0] for m in mismatches]}: mp4={[m[1] for m in mismatches]} npz={[m[2] for m in mismatches]}"
        else:
            ok = False
            detail = f"mp4 count={len(mp4_files)} vs npz count={len(npz_files)}"
        all_pass &= _check("세그먼트 길이 (mp4 vs npz)", ok, detail)
    else:
        print(f"  {WARN}  mp4 files not found, skipping segment length check")

    # ── Check 4: 액션 데이터 (parquet GT vs npz) ──────────────────────────────
    if args.dataset_dir:
        try:
            ep_data = load_data(Path(args.dataset_dir), episode_ids=[ep_id])
            ep_df = ep_data[ep_data["episode_index"] == ep_id].reset_index(drop=True)
            gt_actions = np.stack(ep_df["action"].values)

            max_diffs = []
            for i, (d, s, e) in enumerate(zip(npz_data, npz_starts, npz_ends)):
                gt_seg = gt_actions[s:e].astype(np.float32)
                npz_seg = d["actions"]
                if gt_seg.shape != npz_seg.shape:
                    max_diffs.append((i, None, f"shape mismatch gt={gt_seg.shape} npz={npz_seg.shape}"))
                else:
                    diff = float(np.max(np.abs(gt_seg - npz_seg)))
                    max_diffs.append((i, diff, f"max_abs_diff={diff:.2e}"))

            bad = [(i, info) for i, diff, info in max_diffs
                   if diff is None or diff > args.action_tol]
            ok = len(bad) == 0
            summary = "all match" if ok else f"mismatch at skill(s): {[b[1] for b in bad]}"
            all_pass &= _check(f"액션 데이터 (tol={args.action_tol:.0e})", ok, summary)
            if not ok:
                for i, diff, info in max_diffs:
                    print(f"      skill{i:02d}: {info}")
        except Exception as e:
            print(f"  {WARN}  액션 비교 실패: {e}")
    else:
        print(f"  {WARN}  --dataset_dir 없음, 액션 비교 skip")

    # ── 최종 결과 ─────────────────────────────────────────────────────────────
    print()
    if all_pass:
        print(f"  \033[92m{'=' * 20} ALL PASS {'=' * 20}\033[0m")
    else:
        print(f"  \033[91m{'=' * 20} SOME CHECKS FAILED {'=' * 20}\033[0m")


if __name__ == "__main__":
    main(tyro.cli(Args))
