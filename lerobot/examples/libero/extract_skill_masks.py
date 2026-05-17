"""
에피소드 단위로 SAM2 Video Predictor tracking → 스킬 경계 프레임에서 마스크 추출.

1. 에피소드 전체 프레임을 SAM2 Video Predictor로 tracking (일관된 object ID)
2. 각 스킬의 frame_start / frame_end-1 에서 마스크 슬라이싱
3. start|end 나란히 wandb 시각화 (같은 색 = 같은 물체)

출력: output_dir/task00/ep00000_task00_skill00.npz
  - start_masks  : (N, H, W) bool
  - end_masks    : (N, H, W) bool
  - start_image  : (H, W, 3) uint8
  - end_image    : (H, W, 3) uint8
  - obj_ids      : (N,) int
  - scores       : (N,) float  (AMG frame 0 기준)
  - episode_id, task_id, skill_index, frame_start, frame_end

Usage:
    python examples/libero/extract_skill_masks.py \
        --skillset_dir  .../libero_90_skillset/skills \
        --dataset_dir   .../libero_90 \
        --sam2_checkpoint .../sam2.1_hiera_large.pt \
        --sam2_config   configs/sam2.1/sam2.1_hiera_l.yaml \
        --output_dir    .../libero_90_skillset_masks \
        --wandb_project image_analysis
"""

from __future__ import annotations

import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tyro
from tqdm import tqdm


@dataclass
class Args:
    skillset_dir: str
    """build_skill_dataset.py 출력 디렉토리 (*_skillset/skills)"""
    dataset_dir: str
    """원본 LeRobot 데이터셋 경로 (비디오 참조용)"""
    sam2_checkpoint: str
    """SAM2 체크포인트 .pt 경로"""
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    output_dir: str = ""
    image_key: str = "observation.images.image"
    device: str = "cuda"
    task_ids: list[int] | None = None
    max_episodes_per_task: int = 0
    """task당 처리할 최대 에피소드 수. 0이면 전체."""
    # AMG 파라미터 (첫 프레임 object 검출용)
    points_per_side: int = 32
    max_mask_area_ratio: float = 0.30
    min_mask_area_ratio: float = 0.001
    min_stability_score: float = 0.85
    # gripper (EEF) point prompt — obj_id=0 고정, 빨간색
    gripper_point: tuple[int, int] = (128, 60)
    """initial position의 gripper 픽셀 좌표 (x, y). -1이면 비활성화."""
    gripper_iou_threshold: float = 0.3
    """gripper 마스크와 이 IoU 이상 겹치는 AMG 마스크는 제거."""
    min_centroid_dist: float = 5.0
    """centroid 이동 px 이상이면 changed object로 판정."""
    min_visible_ratio: float = 0.4
    """end_mask 면적 / start_mask 면적 < 이 값이면 부분 가려짐으로 간주하여 changed 판정 skip."""
    # Depth Anything V2
    depth_checkpoint: str = ""
    """Depth Anything V2 HF 모델 경로. 비어있으면 2D 픽셀 거리만 사용."""
    depth_bg_margin: float = 30.0
    """평균 depth보다 이 % 초과하는 object는 배경으로 간주, approach ranking 제외. 예: 30 → 평균+30%."""
    approach_tie_eps: float = 0.05
    """approach score 차이가 이 값 이하면 동점으로 간주, end frame 거리가 가까운 쪽을 더 높은 순위로."""
    min_approach_area_ratio: float = 0.005
    """approach ranking 대상 object의 최소 마스크 면적 비율 (이미지 대비). 이보다 작으면 제외."""
    depth_weight: float = 0.3
    """3D 거리 계산 시 Z(depth) 기여 가중치. 1.0=완전 3D, 0.0=순수 2D 픽셀 거리."""
    top_crop_ratio: float = 0.2
    """이미지 상단 이 비율 안에 centroid가 있는 object는 approach에서 제외. 0.0이면 비활성화."""
    changed_prop_back: int = 2
    """changed object를 이전 방향으로 몇 스킬까지 전파할지."""
    changed_prop_fwd: int = 1
    """changed object를 이후 방향으로 몇 스킬까지 전파할지."""
    approach_score_relative: bool = True
    """True: score=(dist_start-dist_end)/dist_start (비율, 0~1). False: score=dist_start-dist_end (절댓값 px)."""
    # wandb
    wandb_project: str = ""
    wandb_run_name: str = ""
    wandb_log_every: int = 1


# ── 비디오 유틸 ───────────────────────────────────────────────────────────────

def _load_episodes_meta(dataset_dir: Path) -> pd.DataFrame:
    ep_dir = dataset_dir / "meta" / "episodes"
    dfs = [pd.read_parquet(str(f)) for f in sorted(ep_dir.rglob("*.parquet"))]
    return pd.concat(dfs, ignore_index=True).set_index("episode_index")


def _video_path(dataset_dir: Path, episodes_meta: pd.DataFrame, ep_id: int, image_key: str) -> Path:
    row = episodes_meta.loc[ep_id]
    chunk_idx = int(row[f"videos/{image_key}/chunk_index"])
    file_idx = int(row[f"videos/{image_key}/file_index"])
    return dataset_dir / "videos" / image_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"


def _read_episode_frames(video_path: Path, from_ts: float, to_ts: float, length: int) -> np.ndarray:
    """
    에피소드 구간(from_ts ~ to_ts)만 읽어서 (T, H, W, 3) uint8로 반환.
    H.264 키프레임 정렬로 이전 에피소드 프레임이 앞에 붙을 수 있으므로
    length 기준으로 뒤에서 정확히 자른다.
    """
    from torchvision.io import read_video
    # to_ts는 다음 에피소드의 첫 프레임 타임스탬프 (exclusive).
    # 현재 에피소드 마지막 프레임은 to_ts - 1/fps = to_ts - 0.05s에 있음.
    # → end_pts = to_ts - 0.001 로 다음 에피소드 첫 프레임을 제외.
    frames, _, _ = read_video(
        str(video_path),
        start_pts=from_ts,
        end_pts=to_ts - 0.001,
        pts_unit="sec",
        output_format="THWC",
    )
    arr = frames.numpy().astype(np.uint8)[..., :3]
    # H.264 키프레임 정렬로 앞에 이전 에피소드 프레임이 붙을 수 있으므로 뒤에서 length장 취함.
    if len(arr) > length:
        arr = arr[len(arr) - length:]
    return arr


# ── SAM2 ─────────────────────────────────────────────────────────────────────

def build_predictors(args: Args):
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam2_model = build_sam2(args.sam2_config, args.sam2_checkpoint, device=args.device)
    amg = SAM2AutomaticMaskGenerator(model=sam2_model, points_per_side=args.points_per_side)
    img_predictor = SAM2ImagePredictor(sam2_model)
    video_predictor = build_sam2_video_predictor(
        args.sam2_config, args.sam2_checkpoint, device=args.device
    )
    return amg, img_predictor, video_predictor


def build_depth_predictor(checkpoint: str, device: str):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    processor = AutoImageProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(checkpoint, local_files_only=True)
    model.eval().to(device)
    return processor, model


def estimate_depth(processor, model, image: np.ndarray) -> np.ndarray:
    """
    image: (H, W, 3) uint8
    반환: (H, W) float32 raw disparity — 정규화 전 원본값. 여러 프레임을 비교할 때는
    normalize_depth_pair()로 함께 정규화해야 일관성이 유지됨.
    """
    import torch
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(image)
    inputs = processor(images=pil_img, return_tensors="pt").to(model.device)
    with torch.no_grad():
        depth = model(**inputs).predicted_depth   # (1, H', W')
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=image.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy().astype(np.float32)
    return depth


def normalize_depth_pair(
    d1: np.ndarray, d2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    두 프레임의 depth를 공동 min/max로 함께 정규화 → [0,1], 클수록 멀다.
    정지한 물체의 depth 값이 두 프레임 간 일관되게 유지됨.
    Depth Anything V2는 disparity 형식(클수록 가까움)이므로 반전 포함.
    """
    global_min = min(d1.min(), d2.min())
    global_max = max(d1.max(), d2.max())
    if global_max > global_min:
        d1 = (d1 - global_min) / (global_max - global_min)
        d2 = (d2 - global_min) / (global_max - global_min)
    # disparity → depth: 반전 (0=가까움, 1=멀다)
    return 1.0 - d1, 1.0 - d2


def _iou(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = (m1 & m2).sum()
    union = (m1 | m2).sum()
    return float(inter / union) if union > 0 else 0.0


def _coverage(seg: np.ndarray, zone: np.ndarray) -> float:
    """seg 마스크 중 zone 안에 포함된 픽셀 비율."""
    area = seg.sum()
    return float((seg & zone).sum() / area) if area > 0 else 0.0


def get_gripper_and_arm_masks(
    img_predictor, image: np.ndarray, point_xy: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """
    같은 포인트로 multimask_output=True → fine/medium/coarse 3가지 마스크.
    면적 기준으로:
      gripper_mask = 가장 작은 마스크 (정밀 그리퍼, obj_id=0 추적용)
      arm_mask     = 가장 큰 마스크 (팔 전체, AMG exclusion zone용)
    """
    with torch.inference_mode():
        img_predictor.set_image(image)
        masks, _, _ = img_predictor.predict(
            point_coords=np.array([[point_xy[0], point_xy[1]]], dtype=np.float32),
            point_labels=np.array([1]),
            multimask_output=True,
        )
    masks = [m.astype(bool) for m in masks]
    areas = [int(m.sum()) for m in masks]
    gripper_mask = masks[int(np.argmin(areas))]
    arm_mask = masks[int(np.argmax(areas))]
    return gripper_mask, arm_mask


def run_amg(amg, image: np.ndarray, max_area_ratio: float, min_area_ratio: float, min_score: float,
            gripper_mask: np.ndarray | None = None, arm_mask: np.ndarray | None = None,
            iou_threshold: float = 0.3, arm_coverage_threshold: float = 0.5):
    H, W = image.shape[:2]
    total = H * W
    results = amg.generate(image)
    filtered = []
    for r in results:
        if r["stability_score"] < min_score:
            continue
        if not (min_area_ratio <= r["area"] / total <= max_area_ratio):
            continue
        seg = r["segmentation"]
        # gripper 마스크와 많이 겹치면 제거
        if gripper_mask is not None and _iou(seg, gripper_mask) >= iou_threshold:
            continue
        # 로봇팔 zone에 절반 이상 포함된 마스크 제거 (arm 파편 방지)
        if arm_mask is not None and _coverage(seg, arm_mask) >= arm_coverage_threshold:
            continue
        filtered.append(r)
    if not filtered:
        return np.zeros((0, H, W), dtype=bool), np.zeros(0, dtype=np.float32)
    masks = np.stack([r["segmentation"] for r in filtered])
    scores = np.array([r["stability_score"] for r in filtered], dtype=np.float32)
    return masks, scores


def track_episode(
    video_predictor,
    ep_frames: np.ndarray,
    init_masks: np.ndarray,
    target_frame_indices: set[int],
    device: str,
) -> dict[int, dict[int, np.ndarray]]:
    """
    에피소드 전체를 tracking.
    ep_frames      : (T, H, W, 3)
    init_masks     : (N, H, W) bool  — frame 0의 AMG 마스크
    target_frame_indices: 마스크를 저장할 프레임 인덱스 집합
    반환: {frame_idx: {obj_id: mask(H,W)}}
    """
    from PIL import Image as PILImage

    result: dict[int, dict[int, np.ndarray]] = {}

    if len(init_masks) == 0:
        return result

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for t, frame in enumerate(ep_frames):
            PILImage.fromarray(frame).save(str(tmpdir / f"{t:05d}.jpg"), quality=95)

        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            state = video_predictor.init_state(video_path=str(tmpdir))
            video_predictor.reset_state(state)

            for obj_id, mask in enumerate(init_masks):
                video_predictor.add_new_mask(state, frame_idx=0, obj_id=obj_id, mask=mask)

            for out_t, out_obj_ids, out_logits in video_predictor.propagate_in_video(state):
                if out_t in target_frame_indices:
                    result[out_t] = {
                        int(oid): (out_logits[i] > 0.0).squeeze(0).cpu().numpy()
                        for i, oid in enumerate(out_obj_ids)
                    }

    return result


# ── 시각화 ───────────────────────────────────────────────────────────────────

# obj_id=0 → 빨간색 (gripper 고정), 나머지는 순서대로
_GRIPPER_COLOR = (255, 30, 30)
_OBJ_COLORS = [
    (60, 180,  60),  (60, 100, 220), (220, 160,  40),
    (160, 60, 200),  (40, 200, 200), (220, 100, 160), (120, 200,  80),
    (200, 120,  40),  (80, 120, 200), (200, 200,  60),  (60, 200, 160),
]


def _obj_color(obj_id: int) -> tuple[int, int, int]:
    if obj_id == 0:
        return _GRIPPER_COLOR
    return _OBJ_COLORS[(obj_id - 1) % len(_OBJ_COLORS)]


def visualize_masks(image: np.ndarray, obj_mask_dict: dict[int, np.ndarray]) -> np.ndarray:
    """obj_id별 고정 색상으로 overlay. obj_id=0(gripper)은 항상 빨간색."""
    vis = image.copy().astype(np.float32)
    # gripper(0)를 마지막에 그려서 항상 위에 표시
    for obj_id in sorted(obj_mask_dict.keys(), key=lambda x: (x == 0, x)):
        mask = obj_mask_dict[obj_id]
        if not mask.any():
            continue
        color = _obj_color(obj_id)
        alpha = 0.75 if obj_id == 0 else 0.65
        for c in range(3):
            vis[..., c] = np.where(mask, vis[..., c] * (1 - alpha) + color[c] * alpha, vis[..., c])
    return np.clip(vis, 0, 255).astype(np.uint8)


def _centroid(mask: np.ndarray) -> np.ndarray | None:
    """마스크의 픽셀 centroid (y, x) 반환. 비어있으면 None."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return np.array([ys.mean(), xs.mean()])


def compute_changed_objects(
    start_obj_masks: dict[int, np.ndarray],
    end_obj_masks: dict[int, np.ndarray],
    gripper_obj_id: int = 0,
    min_centroid_dist: float = 5.0,
    min_visible_ratio: float = 0.4,
    top_crop_ratio: float = 0.0,
) -> set[int]:
    """
    start → end 사이에 위치가 변한 object의 obj_id 집합을 반환.
    gripper_obj_id는 항상 제외.

    변화 판정 기준:
      - 양쪽 모두 non-empty 마스크가 있어야 판정
      - start centroid Y < H * top_crop_ratio → 상단 영역 제외
      - end_area / start_area < min_visible_ratio → 부분 가려짐으로 간주 → skip
      - centroid 이동거리 > min_centroid_dist (px) → changed

    반환: 변화한 obj_id 집합
    """
    all_ids = set(start_obj_masks) | set(end_obj_masks)
    changed: set[int] = set()

    for obj_id in all_ids:
        if obj_id == gripper_obj_id:
            continue

        s_mask = start_obj_masks.get(obj_id)
        e_mask = end_obj_masks.get(obj_id)

        if s_mask is None or not s_mask.any():
            continue
        if e_mask is None or not e_mask.any():
            continue

        # 상단 영역 제외: start centroid Y < H * top_crop_ratio
        if top_crop_ratio > 0.0:
            c_top = _centroid(s_mask)
            if c_top is not None and c_top[0] < s_mask.shape[0] * top_crop_ratio:
                continue

        # 부분 가려짐: 양방향 체크
        s_area, e_area = s_mask.sum(), e_mask.sum()
        if min_visible_ratio > 0.0 and min(s_area, e_area) / max(s_area, e_area) < min_visible_ratio:
            continue

        c_s = _centroid(s_mask)
        c_e = _centroid(e_mask)
        if c_s is not None and c_e is not None:
            if np.linalg.norm(c_e - c_s) > min_centroid_dist:
                changed.add(obj_id)

    return changed


_APPROACH_COLORS = [
    (0,   0,  180),   # 1등: 진한 파랑
    (60, 120, 255),   # 2등: 파랑
    (100, 200, 255),  # 3등: 하늘색
]
_DEFAULT_OBJ_COLOR = (255, 210, 0)   # 나머지: 노랑
_GREEN_COLOR = (30, 200, 80)          # 이동 종착 물체: 초록


def find_closest_to_changed(
    changed_ids: set[int],
    end_obj_masks: dict[int, np.ndarray],
    gripper_obj_id: int = 0,
    top_crop_ratio: float = 0.0,
) -> set[int]:
    """
    changed_ids 각 물체의 end frame centroid에서 가장 가까운 다른 물체를 반환.
    gripper, changed 물체 자신, 상단 영역 물체는 제외.
    """
    green: set[int] = set()
    for changed_id in changed_ids:
        c_mask = end_obj_masks.get(changed_id)
        if c_mask is None or not c_mask.any():
            continue
        c_changed = _centroid(c_mask)
        if c_changed is None:
            continue

        min_dist = float("inf")
        closest_id = None
        for obj_id, mask in end_obj_masks.items():
            if obj_id == gripper_obj_id or obj_id in changed_ids:
                continue
            if mask is None or not mask.any():
                continue
            if top_crop_ratio > 0.0:
                c_top = _centroid(mask)
                if c_top is not None and c_top[0] < mask.shape[0] * top_crop_ratio:
                    continue
            c_obj = _centroid(mask)
            if c_obj is None:
                continue
            dist = float(np.linalg.norm(c_obj - c_changed))
            if dist < min_dist:
                min_dist = dist
                closest_id = obj_id

        if closest_id is not None:
            green.add(closest_id)
    return green


def _centroid_3d(
    mask: np.ndarray,
    depth_map: np.ndarray | None,
    depth_weight: float = 0.5,
) -> np.ndarray | None:
    """
    픽셀 XY + depth를 별도 차원으로 붙인 의사-3D 좌표 반환.
    Z = depth[cy, cx] * depth_weight * W  → 픽셀 단위로 스케일 맞춤.
    depth_weight=0 이면 순수 2D 픽셀 거리, 1.0 이면 Z가 이미지 너비만큼 기여.
    depth_map이 없으면 2D 픽셀 좌표만 반환.
    """
    c = _centroid(mask)
    if c is None:
        return None
    if depth_map is not None:
        H, W = depth_map.shape
        cy_i = int(np.clip(c[0], 0, H - 1))
        cx_i = int(np.clip(c[1], 0, W - 1))
        Z = float(depth_map[cy_i, cx_i]) * depth_weight * W
        return np.array([c[1], c[0], Z])   # (x_px, y_px, depth_px)
    return np.array([c[1], c[0]])


def compute_approach_ranking(
    start_obj_masks: dict[int, np.ndarray],
    end_obj_masks: dict[int, np.ndarray],
    gripper_obj_id: int = 0,
    top_k: int = 3,
    start_depth: np.ndarray | None = None,
    end_depth: np.ndarray | None = None,
    depth_bg_margin: float = 30.0,
    min_centroid_dist: float = 0.0,
    approach_tie_eps: float = 0.05,
    min_area_ratio: float = 0.005,
    depth_weight: float = 0.3,
    top_crop_ratio: float = 0.2,
    score_relative: bool = True,
) -> tuple[list[tuple], dict[str, int]]:
    """
    반환: (ranking, filter_stats)
      ranking   : [(obj_id, score, dist_end, depth_centroid, depth_mean), ...] 내림차순
      filter_stats: 각 필터별 제외된 object 수 (캡션 표시용)
    """
    filter_stats: dict[str, int] = {
        "no_gripper": 0, "empty": 0, "small": 0, "top": 0, "bg": 0, "moved": 0, "neg": 0,
    }

    g_s_mask = start_obj_masks.get(gripper_obj_id)
    g_e_mask = end_obj_masks.get(gripper_obj_id)
    g_start = _centroid_3d(g_s_mask, start_depth, depth_weight) if g_s_mask is not None else None
    g_end   = _centroid_3d(g_e_mask, end_depth,   depth_weight) if g_e_mask is not None else None

    if g_start is None or g_end is None:
        filter_stats["no_gripper"] = 1
        return [], filter_stats

    all_ids = set(start_obj_masks) | set(end_obj_masks)
    non_gripper = [oid for oid in all_ids if oid != gripper_obj_id]

    # start 프레임 기준 평균 depth 계산 → 배경 필터 cutoff
    cutoff = None
    if start_depth is not None and non_gripper:
        obj_depths = []
        for obj_id in non_gripper:
            m = start_obj_masks.get(obj_id)
            if m is None or not m.any():
                continue
            c = _centroid(m)
            if c is None:
                continue
            cy_i = int(np.clip(c[0], 0, start_depth.shape[0] - 1))
            cx_i = int(np.clip(c[1], 0, start_depth.shape[1] - 1))
            obj_depths.append(start_depth[cy_i, cx_i])
        if obj_depths:
            cutoff = float(np.mean(obj_depths)) * (1.0 + depth_bg_margin / 100.0)

    scores: list[tuple] = []
    for obj_id in non_gripper:
        s_mask = start_obj_masks.get(obj_id)
        e_mask = end_obj_masks.get(obj_id)
        if s_mask is None or not s_mask.any() or e_mask is None or not e_mask.any():
            filter_stats["empty"] += 1
            continue

        # 마스크 크기 필터
        img_area = s_mask.size
        if s_mask.sum() / img_area < min_area_ratio:
            filter_stats["small"] += 1
            continue

        # 상단 영역 제외: start 기준 centroid Y < H * top_crop_ratio
        if top_crop_ratio > 0.0:
            H_img = s_mask.shape[0]
            c_top = _centroid(s_mask)
            if c_top is not None and c_top[0] < H_img * top_crop_ratio:
                filter_stats["top"] += 1
                continue

        # 배경 제외
        if cutoff is not None:
            c = _centroid(s_mask)
            if c is not None:
                cy_i = int(np.clip(c[0], 0, start_depth.shape[0] - 1))
                cx_i = int(np.clip(c[1], 0, start_depth.shape[1] - 1))
                if start_depth[cy_i, cx_i] > cutoff:
                    filter_stats["bg"] += 1
                    continue

        # 이미 이동된 물체 제외
        if min_centroid_dist > 0.0:
            c_s2d = _centroid(s_mask)
            c_e2d = _centroid(e_mask)
            if c_s2d is not None and c_e2d is not None:
                if np.linalg.norm(c_e2d - c_s2d) > min_centroid_dist:
                    filter_stats["moved"] += 1
                    continue

        c_s = _centroid_3d(s_mask, start_depth, depth_weight)
        c_e = _centroid_3d(e_mask, end_depth,   depth_weight)
        if c_s is None or c_e is None:
            filter_stats["empty"] += 1
            continue
        dist_start = float(np.linalg.norm(c_s - g_start))
        dist_end   = float(np.linalg.norm(c_e - g_end))

        # end 프레임 기준 weighted Z component (depth × depth_weight × W, px 단위)
        if end_depth is not None:
            ce2d = _centroid(e_mask)
            cy_i = int(np.clip(ce2d[0], 0, end_depth.shape[0] - 1))
            cx_i = int(np.clip(ce2d[1], 0, end_depth.shape[1] - 1))
            W_px = float(end_depth.shape[1])
            depth_centroid = float(end_depth[cy_i, cx_i]) * depth_weight * W_px
            depth_mean     = float(end_depth[e_mask].mean()) * depth_weight * W_px
        else:
            depth_centroid = float("nan")
            depth_mean     = float("nan")

        if score_relative:
            score = (dist_start - dist_end) / max(dist_start, 1.0)  # 0~1 비율
        else:
            score = dist_start - dist_end  # 절댓값 px
        if score <= 0:
            filter_stats["neg"] += 1
        scores.append((obj_id, score, dist_end, depth_centroid, depth_mean))

    # 1차: approach score를 epsilon 단위로 버킷팅 후 내림차순
    # 2차: 같은 버킷 안에서 end frame 거리 오름차순 (gripper에 더 가까운 쪽 우선)
    import math
    scores.sort(key=lambda x: (-math.floor(x[1] / approach_tie_eps), x[2]))
    return (
        [(obj_id, score, dist_end, depth_centroid, depth_mean)
         for obj_id, score, dist_end, depth_centroid, depth_mean in scores],
        filter_stats,
    )


def make_approach_combined(
    start_image: np.ndarray,
    end_image: np.ndarray,
    start_obj_masks: dict[int, np.ndarray],
    end_obj_masks: dict[int, np.ndarray],
    approach_ranking: list[tuple[int, float]],
    gripper_obj_id: int = 0,
) -> np.ndarray:
    """
    approach 상위 1/2/3등: 진한파랑/파랑/하늘색, 나머지 object: 노랑, gripper: 빨강.
    start | end 나란히.
    """
    positive = [(obj_id, score) for obj_id, score, *_ in approach_ranking if score > 0]
    rank_color = {
        obj_id: _APPROACH_COLORS[i]
        for i, (obj_id, _) in enumerate(positive[:3])
    }

    def _vis(image, obj_masks):
        vis = image.copy().astype(np.float32)
        for obj_id in sorted(obj_masks.keys(), key=lambda x: (x == 0, x)):
            mask = obj_masks[obj_id]
            if mask is None or not mask.any():
                continue
            if obj_id == gripper_obj_id:
                color, alpha = _GRIPPER_COLOR, 0.75
            elif obj_id in rank_color:
                color, alpha = rank_color[obj_id], 0.75
            else:
                color, alpha = _DEFAULT_OBJ_COLOR, 0.55
            for c in range(3):
                vis[..., c] = np.where(mask, vis[..., c] * (1 - alpha) + color[c] * alpha, vis[..., c])
        return np.clip(vis, 0, 255).astype(np.uint8)

    vis_s = _vis(start_image, start_obj_masks)
    vis_e = _vis(end_image, end_obj_masks)
    H = vis_s.shape[0]
    divider = np.full((H, 6, 3), 255, dtype=np.uint8)
    return np.concatenate([vis_s, divider, vis_e], axis=1)


def _depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    """depth (H, W) float [0,1] → RGB (H, W, 3) uint8, plasma colormap."""
    from matplotlib import cm
    return (cm.plasma(depth)[:, :, :3] * 255).astype(np.uint8)


def _depth_diff_to_rgb(diff: np.ndarray) -> np.ndarray:
    """
    depth difference (H, W) → RGB, RdBu colormap.
    양수(빨강)=end에서 더 멀어짐, 음수(파랑)=end에서 더 가까워짐.
    정지 물체는 흰색(0) 에 가까워야 함.
    """
    from matplotlib import cm
    clipped = np.clip(diff, -1.0, 1.0)
    normalized = (clipped + 1.0) / 2.0  # [-1,1] → [0,1]
    return (cm.RdBu_r(normalized)[:, :, :3] * 255).astype(np.uint8)


def make_depth_combined(
    start_image: np.ndarray,
    end_image: np.ndarray,
    start_depth: np.ndarray,
    end_depth: np.ndarray,
) -> np.ndarray:
    """
    행1: start | end 원본 이미지
    행2: start | end depth colormap (plasma, 정규화 후)
    행3: end-start difference (RdBu, 파랑=가까워짐, 빨강=멀어짐, 흰색=정지)
    """
    H, W = start_image.shape[:2]
    divider_v = np.full((H, 6, 3), 200, dtype=np.uint8)
    divider_h = np.full((6, W * 2 + 6, 3), 200, dtype=np.uint8)

    top  = np.concatenate([start_image,            divider_v, end_image],                  axis=1)
    mid  = np.concatenate([_depth_to_rgb(start_depth), divider_v, _depth_to_rgb(end_depth)], axis=1)

    diff = end_depth - start_depth
    diff_rgb = _depth_diff_to_rgb(diff)
    # difference는 단일 패널로 전체 폭에 표시 (start쪽 배치, end쪽은 빈칸)
    blank = np.full((H, W + 6, 3), 240, dtype=np.uint8)
    bot  = np.concatenate([diff_rgb, blank], axis=1)

    return np.concatenate([top, divider_h, mid, divider_h, bot], axis=0)


def make_combined(
    start_image: np.ndarray,
    end_image: np.ndarray,
    start_obj_masks: dict[int, np.ndarray],
    end_obj_masks: dict[int, np.ndarray],
    changed_ids: set[int] | None = None,
    green_ids: set[int] | None = None,
) -> np.ndarray:
    """
    start | end 나란히.
    changed_ids → 검은색, green_ids → 초록색 (검은색 우선), gripper → 빨간색 유지.
    """
    def _vis(image, obj_masks):
        vis = image.copy().astype(np.float32)
        for obj_id in sorted(obj_masks.keys(), key=lambda x: (x == 0, x)):
            mask = obj_masks[obj_id]
            if mask is None or not mask.any():
                continue
            if changed_ids is not None and obj_id in changed_ids:
                color, alpha = (0, 0, 0), 0.75          # 검은색
            elif green_ids is not None and obj_id in green_ids:
                color, alpha = _GREEN_COLOR, 0.75        # 초록색
            else:
                color = _obj_color(obj_id)
                alpha = 0.75 if obj_id == 0 else 0.65
            for c in range(3):
                vis[..., c] = np.where(mask, vis[..., c] * (1 - alpha) + color[c] * alpha, vis[..., c])
        return np.clip(vis, 0, 255).astype(np.uint8)

    vis_s = _vis(start_image, start_obj_masks)
    vis_e = _vis(end_image, end_obj_masks)
    H = vis_s.shape[0]
    divider = np.full((H, 6, 3), 255, dtype=np.uint8)
    return np.concatenate([vis_s, divider, vis_e], axis=1)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args: Args) -> None:
    skillset_dir = Path(args.skillset_dir)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else skillset_dir.parent / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract_skill_masks] skillset : {skillset_dir}")
    print(f"[extract_skill_masks] dataset  : {dataset_dir}")
    print(f"[extract_skill_masks] output   : {output_dir}")

    episodes_meta = _load_episodes_meta(dataset_dir)

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or None,
            config=vars(args),
        )

    print("[extract_skill_masks] Building SAM2 predictors ...")
    amg, img_predictor, video_predictor = build_predictors(args)
    use_gripper = args.gripper_point[0] >= 0

    depth_processor, depth_model = None, None
    if args.depth_checkpoint:
        print(f"[extract_skill_masks] Loading Depth Anything V2 from {args.depth_checkpoint} ...")
        depth_processor, depth_model = build_depth_predictor(args.depth_checkpoint, args.device)

    # skill npz 목록 수집
    task_dirs = sorted(skillset_dir.glob("task*"))
    if args.task_ids is not None:
        task_dirs = [d for d in task_dirs if int(d.name.replace("task", "")) in args.task_ids]

    npz_files = []
    for td in task_dirs:
        task_npzs = sorted(td.glob("*.npz"))
        if args.max_episodes_per_task > 0:
            seen_eps: set[int] = set()
            filtered = []
            for f in task_npzs:
                ep_id = int(np.load(str(f))["episode_id"])
                seen_eps.add(ep_id)
                if len(seen_eps) <= args.max_episodes_per_task:
                    filtered.append(f)
            task_npzs = filtered
        npz_files.extend(task_npzs)
    print(f"[extract_skill_masks] Total skill files: {len(npz_files)}")

    # 에피소드별로 묶기
    ep_to_skills: dict[int, list] = defaultdict(list)
    for npz_path in npz_files:
        d = np.load(str(npz_path))
        ep_to_skills[int(d["episode_id"])].append({
            "path": npz_path,
            "frame_start": int(d["frame_start"]),
            "frame_end":   int(d["frame_end"]),
            "task_id":     int(d["task_id"]) if "task_id" in d else -1,
            "skill_index": int(d["skill_index"]) if "skill_index" in d else -1,
        })

    n_done = 0
    for ep_id, skills in tqdm(ep_to_skills.items(), desc="episodes"):
        # 이미 전부 처리된 에피소드 skip
        all_done = all(
            (output_dir / s["path"].parent.name / f"{s['path'].stem}.npz").exists()
            for s in skills
        )
        if all_done:
            n_done += len(skills)
            continue

        try:
            video_path = _video_path(dataset_dir, episodes_meta, ep_id, args.image_key)
        except KeyError:
            tqdm.write(f"  [warn] episode {ep_id} not in meta, skip")
            continue

        # 에피소드 구간만 로드 (chunk 파일에 여러 에피소드가 이어붙여져 있음)
        row = episodes_meta.loc[ep_id]
        from_ts = float(row[f"videos/{args.image_key}/from_timestamp"])
        to_ts   = float(row[f"videos/{args.image_key}/to_timestamp"])
        ep_length = int(row["length"])
        ep_frames = _read_episode_frames(video_path, from_ts, to_ts, ep_length)
        T = len(ep_frames)

        # tracking할 target 프레임 (각 스킬의 start, end-1)
        target_indices: set[int] = set()
        for s in skills:
            target_indices.add(min(s["frame_start"], T - 1))
            target_indices.add(min(s["frame_end"] - 1, T - 1))

        # gripper 마스크 (obj_id=0 고정) + 팔 전체 exclusion mask
        gripper_mask = None
        arm_mask = None
        if use_gripper:
            gripper_mask, arm_mask = get_gripper_and_arm_masks(
                img_predictor, ep_frames[0], args.gripper_point
            )

        # 첫 프레임 AMG (gripper/팔과 겹치는 마스크 제거)
        amg_masks, amg_scores = run_amg(
            amg, ep_frames[0],
            max_area_ratio=args.max_mask_area_ratio,
            min_area_ratio=args.min_mask_area_ratio,
            min_score=args.min_stability_score,
            gripper_mask=gripper_mask,
            arm_mask=arm_mask,
            iou_threshold=args.gripper_iou_threshold,
        )

        # obj_id=0: gripper, obj_id=1~N: AMG objects
        if gripper_mask is not None:
            init_masks = np.concatenate([[gripper_mask], amg_masks], axis=0) if len(amg_masks) > 0 else gripper_mask[None]
            scores = np.concatenate([[1.0], amg_scores]) if len(amg_scores) > 0 else np.array([1.0])
        else:
            init_masks, scores = amg_masks, amg_scores
        obj_ids = np.arange(len(init_masks), dtype=np.int32)

        # 에피소드 전체 tracking
        tracked = track_episode(video_predictor, ep_frames, init_masks, target_indices, args.device)

        # pre-pass: 모든 스킬의 changed_ids 수집
        skill_changed_map: dict[int, set[int]] = {}
        for s in skills:
            fi_s = min(s["frame_start"], T - 1)
            fi_e = min(s["frame_end"] - 1, T - 1)
            skill_changed_map[s["skill_index"]] = compute_changed_objects(
                tracked.get(fi_s, {}), tracked.get(fi_e, {}),
                min_centroid_dist=args.min_centroid_dist,
                min_visible_ratio=args.min_visible_ratio,
                top_crop_ratio=args.top_crop_ratio,
            )

        # 연속된 changed 스킬을 movement group으로 묶기
        changed_skill_indices = sorted(k for k, v in skill_changed_map.items() if v)

        def _get_groups(indices: list[int]) -> list[list[int]]:
            if not indices:
                return []
            groups: list[list[int]] = []
            current = [indices[0]]
            for idx in indices[1:]:
                if idx == current[-1] + 1:
                    current.append(idx)
                else:
                    groups.append(current)
                    current = [idx]
            groups.append(current)
            return groups

        movement_groups = _get_groups(changed_skill_indices)
        skill_idx_to_s = {s["skill_index"]: s for s in skills}
        bp = args.changed_prop_back
        fp = args.changed_prop_fwd

        # 각 movement group의 마지막 스킬 end frame에서 초록 물체 계산
        group_green: dict[tuple[int, int], set[int]] = {}
        for group in movement_groups:
            last_s = skill_idx_to_s.get(group[-1])
            if last_s is None:
                group_green[(group[0], group[-1])] = set()
                continue
            fi_e = min(last_s["frame_end"] - 1, T - 1)
            orig_changed: set[int] = set()
            for si in group:
                orig_changed |= skill_changed_map[si]
            group_green[(group[0], group[-1])] = find_closest_to_changed(
                orig_changed, tracked.get(fi_e, {}),
                top_crop_ratio=args.top_crop_ratio,
            )

        # 범위 기반 전파: movement group [first, last]에 대해
        # [first - back_prop, last] 안의 모든 스킬에 union (fwd 전파 없음)
        propagated_changed: dict[int, set[int]] = {}
        propagated_green: dict[int, set[int]] = {}
        for s in skills:
            skill_idx = s["skill_index"]
            ch_ids: set[int] = set()
            gr_ids: set[int] = set()
            for group in movement_groups:
                first, last = group[0], group[-1]
                if first - bp <= skill_idx <= last:
                    for si in group:
                        ch_ids |= skill_changed_map[si]
                    gr_ids |= group_green[(first, last)]
            propagated_changed[skill_idx] = ch_ids
            propagated_green[skill_idx] = gr_ids

        wandb_log: dict = {}

        for s in skills:
            out_path = output_dir / s["path"].parent.name / f"{s['path'].stem}.npz"
            if out_path.exists():
                n_done += 1
                continue

            fi_start = min(s["frame_start"], T - 1)
            fi_end = min(s["frame_end"] - 1, T - 1)
            start_obj = tracked.get(fi_start, {})
            end_obj   = tracked.get(fi_end, {})

            # (N, H, W) 배열로 변환
            H, W = ep_frames.shape[1:3]
            n_obj = len(obj_ids)
            start_masks = np.stack([start_obj.get(oid, np.zeros((H, W), dtype=bool)) for oid in obj_ids])
            end_masks   = np.stack([end_obj.get(oid,   np.zeros((H, W), dtype=bool)) for oid in obj_ids])

            # pre-pass에서 계산한 propagated 값 사용 (NPZ + 시각화 동일)
            changed_ids = propagated_changed.get(s["skill_index"], set())
            green_ids   = propagated_green.get(s["skill_index"], set())

            out_path.parent.mkdir(exist_ok=True)
            np.savez_compressed(str(out_path),
                start_masks=start_masks,
                end_masks=end_masks,
                start_image=ep_frames[fi_start],
                end_image=ep_frames[fi_end],
                obj_ids=obj_ids,
                scores=scores,
                changed_obj_ids=np.array(sorted(changed_ids), dtype=np.int32),
                green_obj_ids=np.array(sorted(green_ids), dtype=np.int32),
                episode_id=np.array(ep_id),
                task_id=np.array(s["task_id"]),
                skill_index=np.array(s["skill_index"]),
                frame_start=np.array(s["frame_start"]),
                frame_end=np.array(s["frame_end"]),
            )

            if wandb_run is not None and n_done % args.wandb_log_every == 0:
                import wandb
                task_id = s["task_id"]
                stem = s["path"].stem
                base_caption = (
                    f"ep={ep_id} skill={s['skill_index']} "
                    f"f{s['frame_start']}→{s['frame_end']} n_obj={n_obj}"
                )

                combined = make_combined(
                    ep_frames[fi_start], ep_frames[fi_end], start_obj, end_obj,
                    changed_ids=changed_ids,
                    green_ids=green_ids,
                )
                raw_changed = skill_changed_map.get(s["skill_index"], set())
                propagated_mark = " (propagated)" if changed_ids != raw_changed else ""
                wandb_log[f"changed/task{task_id:02d}/{stem}"] = wandb.Image(
                    combined, caption=base_caption + f" changed={len(changed_ids)} green={len(green_ids)}{propagated_mark}"
                )

                start_depth, end_depth = None, None
                if depth_model is not None:
                    start_depth_raw = estimate_depth(depth_processor, depth_model, ep_frames[fi_start])
                    end_depth_raw   = estimate_depth(depth_processor, depth_model, ep_frames[fi_end])

                    # raw stats 출력 — 모델이 프레임별로 얼마나 다른 값을 뽑는지 확인
                    s_min, s_max, s_mean = start_depth_raw.min(), start_depth_raw.max(), start_depth_raw.mean()
                    e_min, e_max, e_mean = end_depth_raw.min(),   end_depth_raw.max(),   end_depth_raw.mean()
                    tqdm.write(
                        f"  depth raw  start=[{s_min:.3f},{s_max:.3f},μ={s_mean:.3f}]"
                        f"  end=[{e_min:.3f},{e_max:.3f},μ={e_mean:.3f}]"
                        f"  Δmean={e_mean-s_mean:+.3f}"
                    )

                    start_depth, end_depth = normalize_depth_pair(start_depth_raw, end_depth_raw)

                    diff_abs_mean = float(np.abs(end_depth - start_depth).mean())
                    depth_caption = (
                        base_caption
                        + f" | raw_mean s={s_mean:.3f} e={e_mean:.3f} Δ={e_mean-s_mean:+.3f}"
                        + f" | diff_abs_mean={diff_abs_mean:.4f}"
                        + " | row1=RGB row2=depth(plasma) row3=diff(RdBu 파랑=가까워짐)"
                    )
                    depth_vis = make_depth_combined(
                        ep_frames[fi_start], ep_frames[fi_end], start_depth, end_depth
                    )
                    wandb_log[f"depth/task{task_id:02d}/{stem}"] = wandb.Image(
                        depth_vis, caption=depth_caption
                    )

                approach_ranking, fstats = compute_approach_ranking(
                    start_obj, end_obj,
                    start_depth=start_depth,
                    end_depth=end_depth,
                    depth_bg_margin=args.depth_bg_margin,
                    min_centroid_dist=args.min_centroid_dist,
                    approach_tie_eps=args.approach_tie_eps,
                    min_area_ratio=args.min_approach_area_ratio,
                    depth_weight=args.depth_weight,
                    top_crop_ratio=args.top_crop_ratio,
                    score_relative=args.approach_score_relative,
                )
                approach_img = make_approach_combined(
                    ep_frames[fi_start], ep_frames[fi_end], start_obj, end_obj,
                    approach_ranking=approach_ranking,
                )
                n_blue = sum(1 for _, sc, *_ in approach_ranking[:3] if sc > 0)
                sc_fmt = ".3f" if args.approach_score_relative else ".1f"
                sc_unit = "" if args.approach_score_relative else "px"
                top3_str = " ".join(
                    f"obj{oid}:Δ{sc:{sc_fmt}}{sc_unit},d{de:.1f}px,Z={dc:.1f}px(μ{dm:.1f}px)"
                    for oid, sc, de, dc, dm in approach_ranking[:3]
                )
                filter_str = (
                    f"filtered: empty={fstats['empty']} small={fstats['small']}"
                    f" top={fstats['top']} bg={fstats['bg']} moved={fstats['moved']} neg={fstats['neg']}"
                    f" → blue={n_blue}/3"
                )
                if fstats.get("no_gripper"):
                    filter_str = "NO GRIPPER MASK"
                wandb_log[f"approach/task{task_id:02d}/{stem}"] = wandb.Image(
                    approach_img,
                    caption=base_caption + f" top3=[{top3_str}] | {filter_str}",
                )

            n_done += 1

        if wandb_run is not None and wandb_log:
            wandb_run.log({"masks": wandb_log, "n_skills": n_done})

    if wandb_run is not None:
        wandb_run.finish()
    print(f"[extract_skill_masks] Done. Processed {n_done} skills → {output_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
