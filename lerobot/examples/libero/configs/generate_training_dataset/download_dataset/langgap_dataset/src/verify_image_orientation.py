#!/usr/bin/env python3
# Inputs:
#   LangGap staging : {langgap_root}/_hf/{set}
#   reference       : canonical local dataset sharing >=1 task string (auto-scан or --reference)
# Outputs:
#   {staging}/.orientation/verdict.json           (converter의 flip=auto가 읽음)
#   {staging}/.orientation/{camera}_compare.png   (사람 눈 확인용 side-by-side)
"""Decide whether LangGap frames need flipping to match the local image convention.

LangGap의 HF 변환은 raw robosuite 프레임에 [::-1](수직 플립)만 적용했고, 로컬 canonical
데이터셋은 원본 LIBERO HDF5에 [::-1, ::-1](180°)를 적용했다. 소스가 달라 결과 방향이
같을 수도/다를 수도 있으므로, 두 데이터셋이 공유하는 공식 task의 첫 프레임을 네 가지
방향 변형(none/h/w/hw)으로 비교해 MSE 최소 변형을 판정한다. 장면 배치가 에피소드마다
달라도 고정 배경(테이블/캐비닛)이 지배적이라 방향 판별에는 충분하다.

verdict가 확신 기준(best*margin < second)을 못 넘으면 "unknown" — flip=auto 빌드는
중단되고, PNG를 눈으로 확인한 뒤 config에 flip을 명시하라는 안내가 나온다.

Usage:
  python src/verify_image_orientation.py --set langgap_56_full_full
  python src/verify_image_orientation.py --set langgap_6_smoke --reference dataset_filtered/libero_90_full_full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from langgap_dataset_config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    langgap_root,
    load_config,
    project_root,
)

VARIANTS = {
    "none": lambda a: a,
    "h": lambda a: a[::-1],
    "w": lambda a: a[:, ::-1],
    "hw": lambda a: a[::-1, ::-1],
}
CAMERA_PAIRS = {
    # verdict key: (langgap feature, reference feature)
    "image": ("observation.images.image", "observation.images.image"),
    "wrist": ("observation.images.image2", "observation.images.wrist_image"),
}
CONFIDENCE_MARGIN = 1.3  # best*margin < second_best 여야 확신


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--set", required=True, help="Staging set name under {langgap_root}/_hf")
    parser.add_argument("--reference", type=Path, default=None,
                        help="Reference dataset dir (default: yaml orientation_reference, "
                             "blank -> scan {langgap_root} for a dataset sharing a task string)")
    return parser.parse_args()


def tasks_of(dataset_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(dataset_dir / "meta" / "tasks.parquet")


def candidate_references(cfg: dict, staging: Path, explicit: Path | None) -> list[Path]:
    if explicit is not None:
        ref = explicit if explicit.is_absolute() else project_root(cfg) / explicit
        return [ref]
    configured = str(cfg.get("orientation_reference", "") or "").strip()
    if configured:
        ref = Path(configured)
        return [ref if ref.is_absolute() else project_root(cfg) / ref]
    root = langgap_root(cfg)
    out = []
    for meta in sorted(root.glob("*/meta/tasks.parquet")):
        d = meta.parent.parent
        if staging.parent in d.parents or d == staging:  # skip _hf staging entries
            continue
        out.append(d)
    return out


def first_frame_of_task(dataset_dir: Path, repo_id: str, task: str, feature: str) -> np.ndarray:
    """Decode the first frame of the first episode of `task` for `feature` (HWC uint8)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=repo_id, root=dataset_dir)
    task_index = int(tasks_of(dataset_dir).loc[task].task_index)
    cols = ds.hf_dataset.select_columns(["task_index"])
    idx = next(i for i in range(len(ds)) if int(cols[i]["task_index"]) == task_index)
    frame = ds[idx][feature]  # float32 CHW in [0,1]
    arr = (frame.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return arr


def to_score_space(arr: np.ndarray) -> np.ndarray:
    img = Image.fromarray(arr).convert("L").resize((128, 128), resample=Image.BILINEAR)
    out = np.asarray(img, dtype=np.float64)
    return (out - out.mean()) / (out.std() + 1e-6)


def judge(lang: np.ndarray, ref: np.ndarray) -> tuple[str, dict[str, float], bool]:
    ref_s = to_score_space(ref)
    scores = {name: float(((to_score_space(fn(lang).copy()) - ref_s) ** 2).mean())
              for name, fn in VARIANTS.items()}
    ordered = sorted(scores, key=scores.get)
    best, second = ordered[0], ordered[1]
    confident = scores[best] * CONFIDENCE_MARGIN < scores[second]
    return best, scores, confident


def save_compare_png(out_path: Path, ref: np.ndarray, lang: np.ndarray, verdict: str) -> None:
    tiles = [("reference", ref)] + [(f"langgap:{n}{' *' if n == verdict else ''}", fn(lang).copy())
                                    for n, fn in VARIANTS.items()]
    h = max(t.shape[0] for _, t in tiles) + 16
    canvas = Image.new("RGB", (sum(t.shape[1] + 4 for _, t in tiles), h), (30, 30, 30))
    x = 0
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    for label, tile in tiles:
        canvas.paste(Image.fromarray(tile), (x, 16))
        draw.text((x + 2, 2), label, fill=(255, 255, 0))
        x += tile.shape[1] + 4
    canvas.save(out_path)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    staging = langgap_root(cfg) / "_hf" / args.set
    if not (staging / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Staging set not found (download first): {staging}")

    sys.path.insert(0, str(project_root(cfg) / "lerobot" / "src"))

    lang_tasks = set(tasks_of(staging).index)
    reference, common = None, None
    for cand in candidate_references(cfg, staging, args.reference):
        if not (cand / "meta" / "tasks.parquet").exists():
            continue
        shared = sorted(lang_tasks & set(tasks_of(cand).index))
        if shared:
            reference, common = cand, shared[0]
            break

    out_dir = staging / ".orientation"
    out_dir.mkdir(exist_ok=True)
    verdict: dict[str, object] = {"set": args.set}

    if reference is None:
        print("[warn] no reference dataset shares a task string with this set — verdict=unknown")
        print("       (langgap_6_smoke처럼 확장 task만 있는 세트는 정상. PNG 눈확인 후 flip을 명시하세요.)")
        verdict.update({"image": "unknown", "wrist": "unknown", "confident": False,
                        "reference": None, "task": None})
        # 그래도 눈확인용으로 LangGap 첫 프레임의 네 방향 변형은 덤프한다.
        any_task = sorted(lang_tasks)[0]
        for key, (lang_feat, _) in CAMERA_PAIRS.items():
            lang = first_frame_of_task(staging, f"YC11Hou/{args.set}", any_task, lang_feat)
            save_compare_png(out_dir / f"{key}_compare.png", lang, lang, "none")
    else:
        print(f"reference : {reference}")
        print(f"common task: {common!r}")
        verdict.update({"reference": str(reference), "task": common, "confident": True})
        for key, (lang_feat, ref_feat) in CAMERA_PAIRS.items():
            lang = first_frame_of_task(staging, f"YC11Hou/{args.set}", common, lang_feat)
            ref = first_frame_of_task(reference, f"dohyeon/{reference.name}", common, ref_feat)
            best, scores, confident = judge(lang, ref)
            verdict[key] = best if confident else "unknown"
            verdict["confident"] = bool(verdict["confident"]) and confident
            save_compare_png(out_dir / f"{key}_compare.png", ref, lang, best)
            pretty = ", ".join(f"{n}={scores[n]:.3f}" for n in sorted(scores, key=scores.get))
            print(f"  {key:5s}: best={best} confident={confident} ({pretty})")

    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(f"verdict -> {out_dir / 'verdict.json'}")
    print(f"PNG     -> {out_dir}/*_compare.png  (반드시 한 번 눈으로 확인!)")


if __name__ == "__main__":
    main()
