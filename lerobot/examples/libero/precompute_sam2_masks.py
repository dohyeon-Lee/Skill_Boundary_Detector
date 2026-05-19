"""
SAM2 비디오 트래킹으로 스킬별 changed/green 물체 패치 마스크(pool_size×pool_size, 2채널)를 NPZ로 저장.

출력: output_dir/task{N}/ep{X:05d}_skill{Y:03d}.npz
  patch_masks  : (T, pool_size, pool_size, 2)  bool  — ch0=changed, ch1=green
  frame_indices: (T,)                           int32 — 에피소드 내 절대 프레임 인덱스

에피소드 단위 병렬 처리: 정렬된 에피소드 목록을 worker_id / num_workers 로 분할.
"""

from __future__ import annotations

import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tyro
from PIL import Image as PILImage
from tqdm import tqdm


@dataclass
class Args:
    skillset_dir: str
    """build_skill_dataset.py 출력 디렉토리 (*_skillset/skills)"""
    dataset_dir: str
    """원본 LeRobot 데이터셋 경로"""
    output_dir: str
    """NPZ 저장 루트 디렉토리"""
    sam2_checkpoint: str
    """SAM2 체크포인트 .pt 경로"""
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    image_key: str = "observation.images.image"
    device: str = "cuda"
    task_ids: list[int] | None = None
    max_episodes_per_task: int = 0
    # AMG
    points_per_side: int = 50
    max_mask_area_ratio: float = 0.35
    min_mask_area_ratio: float = 0.001
    min_stability_score: float = 0.85
    # Gripper
    gripper_point: tuple[int, int] = (128, 45)
    gripper_iou_threshold: float = 0.3
    # Changed / green
    min_centroid_dist: float = 20.0
    min_visible_ratio: float = 0.05
    changed_top_crop_ratio: float = 0.3
    green_top_crop_ratio: float = 0.3
    changed_prop_back: int = 2
    min_object_area_ratio: float = 0.002
    """changed 검출 시 start 마스크 면적이 이미지 대비 이 비율 미만이면 무시."""
    # Patch mask
    image_size: int = 224
    patch_size: int = 16
    patch_hit_threshold: float = 0.1
    pool_size: int = 8
    # Parallel
    num_workers: int = 1
    worker_id: int = 0


# ── SAM2 ──────────────────────────────────────────────────────────────────────

def build_sam2_predictors(checkpoint: str, config: str, points_per_side: int, device: str):
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model    = build_sam2(config, checkpoint, device=device)
    amg      = SAM2AutomaticMaskGenerator(model, points_per_side=points_per_side)
    img_pred = SAM2ImagePredictor(model)
    vid_pred = build_sam2_video_predictor(config, checkpoint, device=device)
    return amg, img_pred, vid_pred


def _iou(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = (m1 & m2).sum()
    union = (m1 | m2).sum()
    return float(inter / union) if union > 0 else 0.0


def _coverage(seg: np.ndarray, zone: np.ndarray) -> float:
    area = seg.sum()
    return float((seg & zone).sum() / area) if area > 0 else 0.0


def segment_gripper(img_pred, frame: np.ndarray, point_xy: tuple[int, int]):
    """그리퍼/팔 마스크 반환. point_xy[0] < 0 이면 (None, None)."""
    if point_xy[0] < 0:
        return None, None
    with torch.inference_mode():
        img_pred.set_image(frame)
        masks, _, _ = img_pred.predict(
            point_coords=np.array([[point_xy[0], point_xy[1]]], dtype=np.float32),
            point_labels=np.array([1]),
            multimask_output=True,
        )
    masks = [m.astype(bool) for m in masks]
    areas = [m.sum() for m in masks]
    return masks[int(np.argmin(areas))], masks[int(np.argmax(areas))]


def run_amg(amg, frame: np.ndarray, max_area: float, min_area: float, min_score: float,
            gripper_mask, arm_mask, iou_thr: float) -> np.ndarray:
    """첫 프레임 AMG로 오브젝트 마스크 생성. 그리퍼/팔 겹침 필터 적용."""
    H, W  = frame.shape[:2]
    total = H * W
    filtered = []
    for r in amg.generate(frame):
        if r["stability_score"] < min_score:
            continue
        if not (min_area <= r["area"] / total <= max_area):
            continue
        seg = r["segmentation"]
        if gripper_mask is not None and _iou(seg, gripper_mask) >= iou_thr:
            continue
        if arm_mask is not None and _coverage(seg, arm_mask) >= 0.5:
            continue
        filtered.append(seg)
    return np.stack(filtered).astype(bool) if filtered else np.zeros((0, H, W), dtype=bool)


def track_needed_frames(vid_pred, ep_frames: np.ndarray, init_masks: np.ndarray,
                        needed_frames: set[int], device: str) -> dict[int, dict[int, np.ndarray]]:
    """
    SAM2 video predictor로 needed_frames의 마스크만 수집.
    propagate_in_video는 전체 프레임을 순회하므로 메모리는 스킬 범위 내 프레임만 유지.
    obj_id 0 = 그리퍼.
    """
    result: dict[int, dict[int, np.ndarray]] = {}
    if len(init_masks) == 0:
        return result
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for t, frame in enumerate(ep_frames):
            PILImage.fromarray(frame).save(str(tmpdir / f"{t:05d}.jpg"), quality=95)
        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            state = vid_pred.init_state(video_path=str(tmpdir))
            vid_pred.reset_state(state)
            for obj_id, mask in enumerate(init_masks):
                vid_pred.add_new_mask(state, frame_idx=0, obj_id=obj_id, mask=mask)
            for out_t, out_ids, out_logits in vid_pred.propagate_in_video(state):
                if out_t in needed_frames:
                    result[out_t] = {
                        int(oid): (out_logits[i] > 0.0).squeeze(0).cpu().numpy()
                        for i, oid in enumerate(out_ids)
                    }
    return result


# ── Changed / Green 검출 ──────────────────────────────────────────────────────

def _centroid(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return np.array([ys.mean(), xs.mean()])


def compute_changed(start_obj: dict, end_obj: dict,
                    min_centroid_dist: float, min_visible_ratio: float,
                    top_crop_ratio: float, exempt_ids: set[int],
                    min_object_area_ratio: float = 0.0) -> set[int]:
    """
    start→end 프레임 사이 centroid 이동이 min_centroid_dist 이상인 오브젝트 반환.
    top_crop_ratio: 시작 프레임 centroid가 상단 비율 안에 있으면 제외 (exempt_ids는 예외).
    """
    changed: set[int] = set()
    for obj_id in set(start_obj) | set(end_obj):
        if obj_id == 0:  # gripper
            continue
        s_mask = start_obj.get(obj_id)
        e_mask = end_obj.get(obj_id)
        if s_mask is None or not s_mask.any():
            continue
        if e_mask is None or not e_mask.any():
            continue
        if top_crop_ratio > 0.0 and obj_id not in exempt_ids:
            c = _centroid(s_mask)
            if c is not None and c[0] < s_mask.shape[0] * top_crop_ratio:
                continue
        s_area, e_area = int(s_mask.sum()), int(e_mask.sum())
        if min_object_area_ratio > 0.0:
            total = s_mask.shape[0] * s_mask.shape[1]
            if s_area / total < min_object_area_ratio:
                continue
        if min_visible_ratio > 0.0 and min(s_area, e_area) / max(s_area, e_area) < min_visible_ratio:
            continue
        c_s = _centroid(s_mask)
        c_e = _centroid(e_mask)
        if c_s is not None and c_e is not None:
            if np.linalg.norm(c_e - c_s) > min_centroid_dist:
                changed.add(obj_id)
    return changed


def find_green(changed_ids: set[int], end_obj: dict, top_crop_ratio: float) -> set[int]:
    """각 changed 오브젝트에 가장 가까운 non-changed 오브젝트를 green으로 반환."""
    green: set[int] = set()
    for cid in changed_ids:
        c_mask = end_obj.get(cid)
        if c_mask is None or not c_mask.any():
            continue
        c_changed = _centroid(c_mask)
        if c_changed is None:
            continue
        min_dist, closest = float("inf"), None
        for obj_id, mask in end_obj.items():
            if obj_id == 0 or obj_id in changed_ids:
                continue
            if mask is None or not mask.any():
                continue
            if top_crop_ratio > 0.0:
                c = _centroid(mask)
                if c is not None and c[0] < mask.shape[0] * top_crop_ratio:
                    continue
            c_obj = _centroid(mask)
            if c_obj is None:
                continue
            d = float(np.linalg.norm(c_obj - c_changed))
            if d < min_dist:
                min_dist, closest = d, obj_id
        if closest is not None:
            green.add(closest)
    return green


def _get_groups(indices: list[int]) -> list[list[int]]:
    """연속된 정수 목록을 그룹으로 묶음. 예: [1,2,4,5] → [[1,2],[4,5]]"""
    if not indices:
        return []
    groups, cur = [], [indices[0]]
    for idx in indices[1:]:
        if idx == cur[-1] + 1:
            cur.append(idx)
        else:
            groups.append(cur)
            cur = [idx]
    groups.append(cur)
    return groups


# ── 패치 마스크 계산 ──────────────────────────────────────────────────────────

def mask_to_pooled(mask: np.ndarray, image_size: int, patch_size: int,
                   pool_size: int, hit_threshold: float) -> np.ndarray:
    """
    (H, W) bool → (pool_size, pool_size) bool.
    마스크를 image_size로 리사이즈 → patch_size 단위 ratio 계산 →
    hit_threshold 적용 → adaptive max pool로 pool_size 출력.
    """
    n = image_size // patch_size
    resized = np.array(
        PILImage.fromarray(mask.astype(np.uint8) * 255).resize(
            (image_size, image_size), PILImage.NEAREST
        )
    ).astype(np.float32) / 255.0
    ratio = resized.reshape(n, patch_size, n, patch_size).mean(axis=(1, 3))   # (n, n)
    hits  = (ratio >= hit_threshold).astype(np.float32)
    x     = torch.from_numpy(hits).unsqueeze(0).unsqueeze(0)                   # (1,1,n,n)
    return F.adaptive_max_pool2d(x, (pool_size, pool_size)).squeeze().numpy() > 0.5


# ── 비디오 / 에피소드 유틸 ───────────────────────────────────────────────────

def load_episodes_meta(dataset_dir: Path) -> pd.DataFrame:
    ep_dir = dataset_dir / "meta" / "episodes"
    dfs = [pd.read_parquet(str(f)) for f in sorted(ep_dir.rglob("*.parquet"))]
    return pd.concat(dfs, ignore_index=True).set_index("episode_index")


def get_video_path(dataset_dir: Path, meta: pd.DataFrame, ep_id: int, image_key: str) -> Path:
    row   = meta.loc[ep_id]
    chunk = int(row[f"videos/{image_key}/chunk_index"])
    fidx  = int(row[f"videos/{image_key}/file_index"])
    return dataset_dir / "videos" / image_key / f"chunk-{chunk:03d}" / f"file-{fidx:03d}.mp4"


def read_episode_frames(vpath: Path, from_ts: float, to_ts: float, length: int) -> np.ndarray:
    from torchvision.io import read_video
    frames, _, _ = read_video(str(vpath), start_pts=from_ts, end_pts=to_ts - 0.001,
                               pts_unit="sec", output_format="THWC")
    arr = frames.numpy().astype(np.uint8)[..., :3]
    if len(arr) > length:
        arr = arr[len(arr) - length:]
    return arr


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: Args) -> None:
    skillset_dir = Path(args.skillset_dir)
    dataset_dir  = Path(args.dataset_dir)
    output_dir   = Path(args.output_dir)

    print(f"[sam2_masks] skillset  : {skillset_dir}")
    print(f"[sam2_masks] dataset   : {dataset_dir}")
    print(f"[sam2_masks] output    : {output_dir}")
    print(f"[sam2_masks] worker    : {args.worker_id} / {args.num_workers}")

    meta = load_episodes_meta(dataset_dir)

    # 스킬 NPZ 수집 → 에피소드별 그룹화
    task_dirs = sorted(skillset_dir.glob("task*"))
    if args.task_ids is not None:
        task_dirs = [d for d in task_dirs if int(d.name.replace("task", "")) in args.task_ids]

    ep_to_skills: dict[int, list] = defaultdict(list)
    for td in task_dirs:
        npz_list = sorted(td.glob("*.npz"))
        if args.max_episodes_per_task > 0:
            seen: set[int] = set()
            filtered = []
            for f in npz_list:
                d = np.load(str(f))
                ep_id = int(d["episode_id"])
                seen.add(ep_id)
                if len(seen) <= args.max_episodes_per_task:
                    filtered.append(d)
            npz_data = filtered
        else:
            npz_data = [np.load(str(f)) for f in npz_list]

        for d in npz_data:
            ep_to_skills[int(d["episode_id"])].append({
                "frame_start": int(d["frame_start"]),
                "frame_end":   int(d["frame_end"]),
                "task_id":     int(d["task_id"])     if "task_id"     in d else -1,
                "skill_index": int(d["skill_index"]) if "skill_index" in d else -1,
            })

    all_ep_ids = sorted(ep_to_skills.keys())
    my_ep_ids  = all_ep_ids[args.worker_id::args.num_workers]
    print(f"[sam2_masks] episodes total={len(all_ep_ids)}, this worker={len(my_ep_ids)}")

    print("[sam2_masks] Building SAM2 ...")
    amg, img_pred, vid_pred = build_sam2_predictors(
        args.sam2_checkpoint, args.sam2_config, args.points_per_side, args.device,
    )

    def _fetch_frames(ep_id: int):
        """Load episode frames for prefetching (returns None on missing episode)."""
        try:
            vpath = get_video_path(dataset_dir, meta, ep_id, args.image_key)
        except KeyError:
            return None
        row = meta.loc[ep_id]
        from_ts   = float(row[f"videos/{args.image_key}/from_timestamp"])
        to_ts     = float(row[f"videos/{args.image_key}/to_timestamp"])
        ep_length = int(row["length"])
        return read_episode_frames(vpath, from_ts, to_ts, ep_length)

    progress = tqdm(total=len(my_ep_ids), desc="episodes")
    # Prefetch next episode's frames in background while GPU processes current episode.
    with ThreadPoolExecutor(max_workers=1) as executor:
        prefetch = executor.submit(_fetch_frames, my_ep_ids[0]) if my_ep_ids else None

        for i, ep_id in enumerate(my_ep_ids):
            ep_frames = prefetch.result()

            # Start prefetching next episode while GPU processes current
            if i + 1 < len(my_ep_ids):
                prefetch = executor.submit(_fetch_frames, my_ep_ids[i + 1])

            skills = ep_to_skills[ep_id]

            # 이미 모든 출력이 존재하면 건너뜀 (재실행 지원)
            all_exist = all(
                (output_dir / f"task{s['task_id']}" / f"ep{ep_id:05d}_skill{s['skill_index']:03d}.npz").exists()
                for s in skills
            )
            if all_exist:
                tqdm.write(f"  [skip] ep {ep_id}: all outputs exist")
                progress.update(1)
                continue

            if ep_frames is None:
                tqdm.write(f"  [warn] ep {ep_id} not in meta, skip")
                progress.update(1)
                continue

            T    = len(ep_frames)
            H, W = ep_frames[0].shape[:2]

            # 필요한 프레임 집합: 스킬 경계(changed 검출용) + 스킬 내 전 프레임(마스크 저장용)
            needed_frames: set[int] = set()
            for s in skills:
                fi_s = min(s["frame_start"], T - 1)
                fi_e = min(s["frame_end"] - 1, T - 1)
                needed_frames.add(fi_s)
                needed_frames.add(fi_e)
                for fi in range(s["frame_start"], min(s["frame_end"], T)):
                    needed_frames.add(fi)

            # SAM2: 그리퍼 분리 → AMG → 비디오 트래킹
            gripper_mask, arm_mask = segment_gripper(img_pred, ep_frames[0], args.gripper_point)

            amg_masks = run_amg(
                amg, ep_frames[0],
                max_area=args.max_mask_area_ratio, min_area=args.min_mask_area_ratio,
                min_score=args.min_stability_score,
                gripper_mask=gripper_mask, arm_mask=arm_mask, iou_thr=args.gripper_iou_threshold,
            )

            if gripper_mask is not None:
                init_masks = (np.concatenate([[gripper_mask], amg_masks])
                              if len(amg_masks) > 0 else gripper_mask[None])
            else:
                init_masks = amg_masks

            tracked = track_needed_frames(vid_pred, ep_frames, init_masks, needed_frames, args.device)

            # Pre-pass: 스킬 순서대로 changed 검출 (이전 changed는 exempt로 전달)
            skill_changed_map: dict[int, set[int]] = {}
            cumulative_changed: set[int] = set()
            for s in sorted(skills, key=lambda x: x["skill_index"]):
                fi_s = min(s["frame_start"], T - 1)
                fi_e = min(s["frame_end"] - 1, T - 1)
                detected = compute_changed(
                    tracked.get(fi_s, {}), tracked.get(fi_e, {}),
                    min_centroid_dist=args.min_centroid_dist,
                    min_visible_ratio=args.min_visible_ratio,
                    top_crop_ratio=args.changed_top_crop_ratio,
                    exempt_ids=cumulative_changed,
                    min_object_area_ratio=args.min_object_area_ratio,
                )
                skill_changed_map[s["skill_index"]] = detected
                cumulative_changed |= detected

            # Movement group 구성
            changed_indices = sorted(k for k, v in skill_changed_map.items() if v)
            groups          = _get_groups(changed_indices)
            skill_idx_to_s  = {s["skill_index"]: s for s in skills}

            # 각 group의 마지막 스킬 end frame에서 green 탐색
            group_green: dict[tuple[int, int], set[int]] = {}
            for group in groups:
                key    = (group[0], group[-1])
                last_s = skill_idx_to_s.get(group[-1])
                if last_s is None:
                    group_green[key] = set()
                    continue
                fi_e = min(last_s["frame_end"] - 1, T - 1)
                orig_changed: set[int] = set()
                for si in group:
                    orig_changed |= skill_changed_map[si]
                group_green[key] = find_green(
                    orig_changed, tracked.get(fi_e, {}),
                    top_crop_ratio=args.green_top_crop_ratio,
                )

            # 범위 기반 전파: [first - bp, last] (fwd 없음)
            bp = args.changed_prop_back
            propagated_changed: dict[int, set[int]] = {}
            propagated_green:   dict[int, set[int]] = {}
            for s in skills:
                si = s["skill_index"]
                ch: set[int] = set()
                gr: set[int] = set()
                for group in groups:
                    first, last = group[0], group[-1]
                    if first - bp <= si <= last:
                        for g in group:
                            ch |= skill_changed_map[g]
                        gr |= group_green[(first, last)]
                propagated_changed[si] = ch
                propagated_green[si]   = gr

            # 스킬별 NPZ 저장
            for s in skills:
                si          = s["skill_index"]
                changed_ids = propagated_changed[si]
                green_ids   = propagated_green[si]

                frames_in_skill = list(range(s["frame_start"], min(s["frame_end"], T)))
                TT = len(frames_in_skill)
                patch_masks = np.zeros((TT, args.pool_size, args.pool_size, 2), dtype=bool)

                for ti, fi in enumerate(frames_in_skill):
                    obj_data = tracked.get(fi, {})

                    ch_mask = np.zeros((H, W), dtype=bool)
                    for oid in changed_ids:
                        m = obj_data.get(oid)
                        if m is not None and m.any():
                            ch_mask |= m

                    gr_mask = np.zeros((H, W), dtype=bool)
                    for oid in green_ids:
                        m = obj_data.get(oid)
                        if m is not None and m.any():
                            gr_mask |= m

                    patch_masks[ti, :, :, 0] = mask_to_pooled(
                        ch_mask, args.image_size, args.patch_size, args.pool_size, args.patch_hit_threshold,
                    )
                    patch_masks[ti, :, :, 1] = mask_to_pooled(
                        gr_mask, args.image_size, args.patch_size, args.pool_size, args.patch_hit_threshold,
                    )

                task_out = output_dir / f"task{s['task_id']}"
                task_out.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    str(task_out / f"ep{ep_id:05d}_skill{si:03d}.npz"),
                    patch_masks=patch_masks,
                    frame_indices=np.array(frames_in_skill, dtype=np.int32),
                )

            tqdm.write(f"  [done] ep {ep_id}: {len(skills)} skills → {output_dir}")
            progress.update(1)

    print("[sam2_masks] Done.")


if __name__ == "__main__":
    main(tyro.cli(Args))
