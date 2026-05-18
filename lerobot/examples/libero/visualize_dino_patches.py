"""
SAM2 마스크 추출 + DINOv3 패치 피처 시각화 (통합 스크립트).

에피소드 단위 SAM2 Video Predictor tracking → 스킬 경계 changed/green 물체 검출
→ DINOv3 패치 토큰 추출 → 4x4 / 8x8 / 14x14 그리드 + PCA 시각화 → wandb 로깅.

스킬당 패널 구성 (start | end):
  Row 0: 원본 이미지 + changed(검은색) / green(초록색) 마스크 오버레이
  Row 1: 4x4 풀링 패치 그리드
  Row 2: 8x8 풀링 패치 그리드
  Row 3: 14x14 원본 패치 그리드 (dinov3-vits16, patch_size=16)
  Row 4: PCA(3-component) 패치 피처 (에피소드 단위 fit)
"""

from __future__ import annotations

import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
    # Changed / green 검출
    min_centroid_dist: float = 20.0
    min_visible_ratio: float = 0.1
    changed_top_crop_ratio: float = 0.15
    """changed(빨강) 물체 검출 시 상단 제외 비율."""
    green_top_crop_ratio: float = 0.3
    """green(초록) 물체 검출 시 상단 제외 비율."""
    changed_prop_back: int = 2
    min_object_area_ratio: float = 0.002
    """changed 검출 시 start 마스크 면적이 이미지 대비 이 비율 미만이면 무시."""
    # DINOv3
    dino_model: str = "/data2/dohyeon/SBD/models/dinov3-vits16"
    image_size: int = 224
    patch_size: int = 16
    """dinov3-vits16: patch_size=16 → 224/16=14 patches per dim"""
    patch_hit_threshold: float = 0.1
    """패치 내 마스크 픽셀 비율이 이 값 이상이어야 hit 판정. 0.0이면 any()와 동일."""
    # wandb
    wandb_project: str = ""
    wandb_run_name: str = ""
    wandb_log_every: int = 1


_CHANGED_COLOR       = np.array([30,  30,  30],  dtype=np.float32)  # row0 마스크: 검은색
_CHANGED_PATCH_COLOR = np.array([220,  30,  30],  dtype=np.float32)  # 패치 그리드: 빨간색
_GREEN_COLOR         = np.array([30,  200,  80],  dtype=np.float32)  # 초록색
_OVERLAP_PATCH_COLOR = np.array([30,   60, 220],  dtype=np.float32)  # 겹침: 파란색
_GRIPPER_COLOR = (255, 30, 30)
_OBJ_COLORS = [
    (60, 180,  60),  (60, 100, 220), (220, 160,  40),
    (160, 60, 200),  (40, 200, 200), (220, 100, 160),
]
_GRID_COLOR = np.array([180, 180, 180], dtype=np.uint8)


def _obj_color(obj_id: int):
    if obj_id == 0:
        return _GRIPPER_COLOR
    return _OBJ_COLORS[(obj_id - 1) % len(_OBJ_COLORS)]


# ── 비디오 / 에피소드 유틸 ────────────────────────────────────────────────────

def _load_episodes_meta(dataset_dir: Path) -> pd.DataFrame:
    ep_dir = dataset_dir / "meta" / "episodes"
    dfs = [pd.read_parquet(str(f)) for f in sorted(ep_dir.rglob("*.parquet"))]
    return pd.concat(dfs, ignore_index=True).set_index("episode_index")


def _video_path(dataset_dir, episodes_meta, ep_id, image_key) -> Path:
    row = episodes_meta.loc[ep_id]
    chunk_idx = int(row[f"videos/{image_key}/chunk_index"])
    file_idx  = int(row[f"videos/{image_key}/file_index"])
    return dataset_dir / "videos" / image_key / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"


def _read_episode_frames(video_path, from_ts, to_ts, length) -> np.ndarray:
    from torchvision.io import read_video
    frames, _, _ = read_video(str(video_path), start_pts=from_ts, end_pts=to_ts - 0.001,
                               pts_unit="sec", output_format="THWC")
    arr = frames.numpy().astype(np.uint8)[..., :3]
    if len(arr) > length:
        arr = arr[len(arr) - length:]
    return arr


# ── SAM2 ─────────────────────────────────────────────────────────────────────

def build_sam2(args: Args):
    from sam2.build_sam import build_sam2 as _build_sam2, build_sam2_video_predictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = _build_sam2(args.sam2_config, args.sam2_checkpoint, device=args.device)
    amg   = SAM2AutomaticMaskGenerator(model=model, points_per_side=args.points_per_side)
    img_pred  = SAM2ImagePredictor(model)
    vid_pred  = build_sam2_video_predictor(args.sam2_config, args.sam2_checkpoint, device=args.device)
    return amg, img_pred, vid_pred


def _iou(m1, m2):
    inter = (m1 & m2).sum(); union = (m1 | m2).sum()
    return float(inter / union) if union > 0 else 0.0


def _coverage(seg, zone):
    area = seg.sum()
    return float((seg & zone).sum() / area) if area > 0 else 0.0


def get_gripper_arm_masks(img_pred, image, point_xy):
    with torch.inference_mode():
        img_pred.set_image(image)
        masks, _, _ = img_pred.predict(
            point_coords=np.array([[point_xy[0], point_xy[1]]], dtype=np.float32),
            point_labels=np.array([1]), multimask_output=True,
        )
    masks = [m.astype(bool) for m in masks]
    areas = [int(m.sum()) for m in masks]
    return masks[int(np.argmin(areas))], masks[int(np.argmax(areas))]


def run_amg(amg, image, max_area, min_area, min_score,
            gripper_mask=None, arm_mask=None, iou_thr=0.3, arm_cov_thr=0.5):
    H, W = image.shape[:2]; total = H * W
    results = amg.generate(image)
    filtered = []
    for r in results:
        if r["stability_score"] < min_score: continue
        if not (min_area <= r["area"] / total <= max_area): continue
        seg = r["segmentation"]
        if gripper_mask is not None and _iou(seg, gripper_mask) >= iou_thr: continue
        if arm_mask    is not None and _coverage(seg, arm_mask) >= arm_cov_thr: continue
        filtered.append(r)
    if not filtered:
        return np.zeros((0, H, W), dtype=bool), np.zeros(0, dtype=np.float32)
    return (np.stack([r["segmentation"] for r in filtered]),
            np.array([r["stability_score"] for r in filtered], dtype=np.float32))


def track_episode(vid_pred, ep_frames, init_masks, target_indices, device):
    from PIL import Image as PILImage
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
                if out_t in target_indices:
                    result[out_t] = {
                        int(oid): (out_logits[i] > 0.0).squeeze(0).cpu().numpy()
                        for i, oid in enumerate(out_ids)
                    }
    return result


# ── Changed / Green 검출 ──────────────────────────────────────────────────────

def _centroid(mask):
    ys, xs = np.where(mask)
    if len(ys) == 0: return None
    return np.array([ys.mean(), xs.mean()])


def compute_changed_objects(start_obj, end_obj, gripper_id=0,
                            min_centroid_dist=5.0, min_visible_ratio=0.4,
                            top_crop_ratio=0.0, min_object_area_ratio=0.0,
                            exempt_ids: set[int] | None = None) -> set[int]:
    """
    exempt_ids: 이전 스킬에서 이미 changed로 검출된 obj_id 집합.
    top_crop_ratio 필터를 적용하되, exempt_ids에 속한 물체는 상단에 있어도 제외하지 않음.
    """
    changed: set[int] = set()
    for obj_id in set(start_obj) | set(end_obj):
        if obj_id == gripper_id: continue
        s_mask = start_obj.get(obj_id); e_mask = end_obj.get(obj_id)
        if s_mask is None or not s_mask.any(): continue
        if e_mask is None or not e_mask.any(): continue
        if top_crop_ratio > 0.0 and (exempt_ids is None or obj_id not in exempt_ids):
            c = _centroid(s_mask)
            if c is not None and c[0] < s_mask.shape[0] * top_crop_ratio: continue
        s_area, e_area = s_mask.sum(), e_mask.sum()
        if min_object_area_ratio > 0.0:
            total = s_mask.shape[0] * s_mask.shape[1]
            if s_area / total < min_object_area_ratio: continue
        if min_visible_ratio > 0.0 and min(s_area, e_area) / max(s_area, e_area) < min_visible_ratio:
            continue
        c_s = _centroid(s_mask); c_e = _centroid(e_mask)
        if c_s is not None and c_e is not None:
            if np.linalg.norm(c_e - c_s) > min_centroid_dist:
                changed.add(obj_id)
    return changed


def find_closest_to_changed(changed_ids, end_obj, gripper_id=0, top_crop_ratio=0.0) -> set[int]:
    green: set[int] = set()
    for cid in changed_ids:
        c_mask = end_obj.get(cid)
        if c_mask is None or not c_mask.any(): continue
        c_changed = _centroid(c_mask)
        if c_changed is None: continue
        min_dist, closest = float("inf"), None
        for obj_id, mask in end_obj.items():
            if obj_id == gripper_id or obj_id in changed_ids: continue
            if mask is None or not mask.any(): continue
            if top_crop_ratio > 0.0:
                c = _centroid(mask)
                if c is not None and c[0] < mask.shape[0] * top_crop_ratio: continue
            c_obj = _centroid(mask)
            if c_obj is None: continue
            d = float(np.linalg.norm(c_obj - c_changed))
            if d < min_dist: min_dist, closest = d, obj_id
        if closest is not None: green.add(closest)
    return green


# ── DINOv3 ───────────────────────────────────────────────────────────────────

def load_dino(model_name: str, device: str):
    from transformers import AutoImageProcessor, AutoModel
    proc  = AutoImageProcessor.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True)
    model.eval().to(device)
    for p in model.parameters(): p.requires_grad_(False)
    mean = torch.tensor(getattr(proc, "image_mean", [0.485, 0.456, 0.406]),
                        dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(getattr(proc, "image_std",  [0.229, 0.224, 0.225]),
                        dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return model, mean, std


@torch.no_grad()
def extract_patches(model, mean, std, image: np.ndarray,
                    image_size: int, patch_size: int, device: str) -> np.ndarray:
    """
    (H, W, 3) uint8 → (n_h*n_w, dim) float32 patch tokens.
    CLS + register token 제외: last_hidden[-n_patches:] 사용.
    dinov3-vits16은 register token 4개 포함 → last_hidden = [CLS, reg×4, patch×196].
    """
    n_patches = (image_size // patch_size) ** 2
    x = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    x = (x - mean) / std
    last_hidden = model(pixel_values=x).last_hidden_state   # (1, 1+N_reg+n_patches, dim)
    return last_hidden[0, -n_patches:].cpu().numpy()        # (n_h*n_w, dim)


# ── 패치 히트 계산 ────────────────────────────────────────────────────────────

def mask_to_patch_hits(mask: np.ndarray, image_size: int, patch_size: int,
                       hit_threshold: float = 0.1) -> np.ndarray:
    """
    (H, W) bool → (n_h, n_w) bool.
    hit_threshold: 패치 내 마스크 픽셀 비율이 이 값 이상이어야 hit으로 판정.
    any() (=0.0) 대신 threshold를 쓰면 경계선 1px 접촉으로 인한 과도한 spread 방지.
    """
    from PIL import Image as PILImage
    n = image_size // patch_size
    m = np.array(
        PILImage.fromarray(mask.astype(np.uint8) * 255).resize(
            (image_size, image_size), PILImage.NEAREST)
    ).astype(np.float32) / 255.0
    ratio = m.reshape(n, patch_size, n, patch_size).mean(axis=(1, 3))  # (n_h, n_w)
    return ratio >= hit_threshold


def pool_hits_adaptive(hits: np.ndarray, out_size: int) -> np.ndarray:
    """(n_h, n_w) bool → (out_size, out_size) bool via adaptive max pool (비정수 배수 OK)."""
    x = torch.from_numpy(hits.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return F.adaptive_max_pool2d(x, (out_size, out_size)).squeeze().numpy() > 0.5


# ── 시각화 함수 ──────────────────────────────────────────────────────────────

def resize_to(image: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image as PILImage
    return np.array(PILImage.fromarray(image).resize((size, size), PILImage.BILINEAR))


def resize_mask_to(mask: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image as PILImage
    return np.array(
        PILImage.fromarray(mask.astype(np.uint8) * 255).resize((size, size), PILImage.NEAREST)
    ).astype(bool)


def overlay_masks(image: np.ndarray, changed_mask: np.ndarray,
                  green_mask: np.ndarray, out_size: int) -> np.ndarray:
    """이미지와 마스크를 out_size로 리사이즈 후 검은색/초록색 오버레이."""
    img = resize_to(image, out_size).astype(np.float32)
    ch  = resize_mask_to(changed_mask, out_size)
    gr  = resize_mask_to(green_mask,   out_size)
    for mask, color, alpha in [(ch, _CHANGED_PATCH_COLOR, 0.75), (gr, _GREEN_COLOR, 0.75)]:
        if mask.any():
            for c in range(3):
                img[..., c] = np.where(mask, img[..., c] * (1-alpha) + color[c]*alpha, img[..., c])
    return np.clip(img, 0, 255).astype(np.uint8)


def make_patch_grid_vis(image: np.ndarray, changed_hits: np.ndarray, green_hits: np.ndarray,
                        image_size: int, grid_size: int, cell_alpha: float = 0.55) -> np.ndarray:
    """image를 image_size로 리사이즈 + grid_size×grid_size 셀 강조 + 격자선."""
    img  = resize_to(image, image_size).astype(np.float32)
    cell = image_size // grid_size
    for gi in range(grid_size):
        for gj in range(grid_size):
            y0, y1 = gi * cell, (gi + 1) * cell
            x0, x1 = gj * cell, (gj + 1) * cell
            ch = bool(changed_hits[gi, gj])
            gr = bool(green_hits[gi, gj])
            if   ch and gr: color = _OVERLAP_PATCH_COLOR   # 겹침: 파란색
            elif ch:        color = _CHANGED_PATCH_COLOR   # changed만: 빨간색
            elif gr:        color = _GREEN_COLOR            # green만: 초록색
            else: continue
            for c in range(3):
                img[y0:y1, x0:x1, c] = img[y0:y1, x0:x1, c] * (1-cell_alpha) + color[c]*cell_alpha
    vis = np.clip(img, 0, 255).astype(np.uint8)
    for i in range(0, image_size, cell): vis[i:i+1, :] = _GRID_COLOR
    for j in range(0, image_size, cell): vis[:, j:j+1] = _GRID_COLOR
    return vis


def pca_to_rgb(patches: np.ndarray, pca, n_h: int, n_w: int, image_size: int) -> np.ndarray:
    from PIL import Image as PILImage
    t = pca.transform(patches)
    t = (t - pca.global_min) / np.clip(pca.global_max - pca.global_min, 1e-6, None)
    grid = np.clip(t, 0, 1).reshape(n_h, n_w, 3)
    return np.array(
        PILImage.fromarray((grid * 255).astype(np.uint8)).resize((image_size, image_size), PILImage.NEAREST)
    )


def hstack_pair(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    div = np.full((left.shape[0], 4, 3), 255, dtype=np.uint8)
    return np.concatenate([left, div, right], axis=1)


def vstack_rows(rows: list[np.ndarray]) -> np.ndarray:
    div = np.full((3, rows[0].shape[1], 3), 160, dtype=np.uint8)
    out = rows[0]
    for r in rows[1:]:
        out = np.concatenate([out, div, r], axis=0)
    return out


def make_skill_panel(
    start_image: np.ndarray, end_image: np.ndarray,
    start_obj: dict[int, np.ndarray], end_obj: dict[int, np.ndarray],
    changed_ids: set[int], green_ids: set[int],
    start_patches: np.ndarray, end_patches: np.ndarray,
    pca, image_size: int, patch_size: int, patch_hit_threshold: float = 0.1,
) -> np.ndarray:
    n_h = n_w = image_size // patch_size   # 14

    # 합쳐진 마스크 (원본 해상도)
    H, W = start_image.shape[:2]
    s_ch = np.zeros((H, W), dtype=bool)
    s_gr = np.zeros((H, W), dtype=bool)
    e_ch = np.zeros((H, W), dtype=bool)
    e_gr = np.zeros((H, W), dtype=bool)
    for oid in changed_ids:
        if oid in start_obj and start_obj[oid].any(): s_ch |= start_obj[oid]
        if oid in end_obj   and end_obj[oid].any():   e_ch |= end_obj[oid]
    for oid in green_ids:
        if oid in start_obj and start_obj[oid].any(): s_gr |= start_obj[oid]
        if oid in end_obj   and end_obj[oid].any():   e_gr |= end_obj[oid]

    # Row 0: 마스크 오버레이 (image_size 리사이즈 포함)
    row0 = hstack_pair(
        overlay_masks(start_image, s_ch, s_gr, image_size),
        overlay_masks(end_image,   e_ch, e_gr, image_size),
    )

    # 패치 히트: fine (14×14)
    s_ch_f = mask_to_patch_hits(s_ch, image_size, patch_size, patch_hit_threshold)
    s_gr_f = mask_to_patch_hits(s_gr, image_size, patch_size, patch_hit_threshold)
    e_ch_f = mask_to_patch_hits(e_ch, image_size, patch_size, patch_hit_threshold)
    e_gr_f = mask_to_patch_hits(e_gr, image_size, patch_size, patch_hit_threshold)

    # Row 1: 4×4  Row 2: 8×8  Row 3: 14×14
    row1 = hstack_pair(
        make_patch_grid_vis(start_image, pool_hits_adaptive(s_ch_f, 4), pool_hits_adaptive(s_gr_f, 4), image_size, 4),
        make_patch_grid_vis(end_image,   pool_hits_adaptive(e_ch_f, 4), pool_hits_adaptive(e_gr_f, 4), image_size, 4),
    )
    row2 = hstack_pair(
        make_patch_grid_vis(start_image, pool_hits_adaptive(s_ch_f, 8), pool_hits_adaptive(s_gr_f, 8), image_size, 8),
        make_patch_grid_vis(end_image,   pool_hits_adaptive(e_ch_f, 8), pool_hits_adaptive(e_gr_f, 8), image_size, 8),
    )
    row3 = hstack_pair(
        make_patch_grid_vis(start_image, s_ch_f, s_gr_f, image_size, n_h),
        make_patch_grid_vis(end_image,   e_ch_f, e_gr_f, image_size, n_h),
    )
    # Row 4: PCA
    row4 = hstack_pair(
        pca_to_rgb(start_patches, pca, n_h, n_w, image_size),
        pca_to_rgb(end_patches,   pca, n_h, n_w, image_size),
    )

    return vstack_rows([row0, row1, row2, row3, row4])


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args: Args) -> None:
    from sklearn.decomposition import PCA

    skillset_dir = Path(args.skillset_dir)
    dataset_dir  = Path(args.dataset_dir)

    print(f"[vis_dino] skillset : {skillset_dir}")
    print(f"[vis_dino] dataset  : {dataset_dir}")
    print(f"[vis_dino] DINOv3   : {args.dino_model}")

    episodes_meta = _load_episodes_meta(dataset_dir)

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name or None,
                               config=vars(args))

    print("[vis_dino] Building SAM2 ...")
    amg, img_pred, vid_pred = build_sam2(args)
    use_gripper = args.gripper_point[0] >= 0

    print("[vis_dino] Loading DINOv3 ...")
    dino_model, dino_mean, dino_std = load_dino(args.dino_model, args.device)

    # NPZ 목록 수집
    task_dirs = sorted(skillset_dir.glob("task*"))
    if args.task_ids is not None:
        task_dirs = [d for d in task_dirs if int(d.name.replace("task", "")) in args.task_ids]

    npz_files = []
    for td in task_dirs:
        task_npzs = sorted(td.glob("*.npz"))
        if args.max_episodes_per_task > 0:
            seen: set[int] = set()
            filtered = []
            for f in task_npzs:
                ep_id = int(np.load(str(f))["episode_id"])
                seen.add(ep_id)
                if len(seen) <= args.max_episodes_per_task:
                    filtered.append(f)
            task_npzs = filtered
        npz_files.extend(task_npzs)
    print(f"[vis_dino] Total skill files: {len(npz_files)}")

    # 에피소드별로 묶기
    ep_to_skills: dict[int, list] = defaultdict(list)
    for npz_path in npz_files:
        d = np.load(str(npz_path))
        ep_to_skills[int(d["episode_id"])].append({
            "path":        npz_path,
            "frame_start": int(d["frame_start"]),
            "frame_end":   int(d["frame_end"]),
            "task_id":     int(d["task_id"]) if "task_id" in d else -1,
            "skill_index": int(d["skill_index"]) if "skill_index" in d else -1,
        })

    n_done = 0
    for ep_id, skills in tqdm(ep_to_skills.items(), desc="episodes"):
        try:
            video_path = _video_path(dataset_dir, episodes_meta, ep_id, args.image_key)
        except KeyError:
            tqdm.write(f"  [warn] episode {ep_id} not in meta, skip")
            continue

        row = episodes_meta.loc[ep_id]
        from_ts   = float(row[f"videos/{args.image_key}/from_timestamp"])
        to_ts     = float(row[f"videos/{args.image_key}/to_timestamp"])
        ep_length = int(row["length"])
        ep_frames = _read_episode_frames(video_path, from_ts, to_ts, ep_length)
        T = len(ep_frames)

        target_indices: set[int] = set()
        for s in skills:
            target_indices.add(min(s["frame_start"], T - 1))
            target_indices.add(min(s["frame_end"] - 1, T - 1))

        gripper_mask = arm_mask = None
        if use_gripper:
            gripper_mask, arm_mask = get_gripper_arm_masks(img_pred, ep_frames[0], args.gripper_point)

        amg_masks, _ = run_amg(
            amg, ep_frames[0],
            max_area=args.max_mask_area_ratio, min_area=args.min_mask_area_ratio,
            min_score=args.min_stability_score,
            gripper_mask=gripper_mask, arm_mask=arm_mask, iou_thr=args.gripper_iou_threshold,
        )

        if gripper_mask is not None:
            init_masks = np.concatenate([[gripper_mask], amg_masks], axis=0) if len(amg_masks) > 0 else gripper_mask[None]
        else:
            init_masks = amg_masks

        tracked = track_episode(vid_pred, ep_frames, init_masks, target_indices, args.device)

        # pre-pass: changed/green 검출 + 범위 기반 전파
        # 스킬 순서대로 처리 — 이전 스킬의 changed IDs를 exempt로 전달
        skill_changed_map: dict[int, set[int]] = {}
        cumulative_changed: set[int] = set()
        for s in sorted(skills, key=lambda x: x["skill_index"]):
            fi_s = min(s["frame_start"], T - 1)
            fi_e = min(s["frame_end"] - 1, T - 1)
            detected = compute_changed_objects(
                tracked.get(fi_s, {}), tracked.get(fi_e, {}),
                min_centroid_dist=args.min_centroid_dist,
                min_visible_ratio=args.min_visible_ratio,
                top_crop_ratio=args.changed_top_crop_ratio,
                min_object_area_ratio=args.min_object_area_ratio,
                exempt_ids=cumulative_changed,
            )
            skill_changed_map[s["skill_index"]] = detected
            cumulative_changed |= detected

        # movement group 구성
        changed_indices = sorted(k for k, v in skill_changed_map.items() if v)

        def _get_groups(indices):
            if not indices: return []
            groups, cur = [], [indices[0]]
            for idx in indices[1:]:
                if idx == cur[-1] + 1: cur.append(idx)
                else: groups.append(cur); cur = [idx]
            groups.append(cur)
            return groups

        movement_groups = _get_groups(changed_indices)
        skill_idx_to_s  = {s["skill_index"]: s for s in skills}
        bp = args.changed_prop_back

        # 각 group의 마지막 스킬 end frame에서 green 계산
        group_green: dict[tuple[int, int], set[int]] = {}
        for group in movement_groups:
            last_s = skill_idx_to_s.get(group[-1])
            if last_s is None:
                group_green[(group[0], group[-1])] = set()
                continue
            fi_e = min(last_s["frame_end"] - 1, T - 1)
            orig_changed: set[int] = set()
            for si in group: orig_changed |= skill_changed_map[si]
            group_green[(group[0], group[-1])] = find_closest_to_changed(
                orig_changed, tracked.get(fi_e, {}), top_crop_ratio=args.green_top_crop_ratio,
            )

        # 범위 기반 전파: [first - bp, last]
        propagated_changed: dict[int, set[int]] = {}
        propagated_green:   dict[int, set[int]] = {}
        for s in skills:
            si = s["skill_index"]
            ch: set[int] = set(); gr: set[int] = set()
            for group in movement_groups:
                first, last = group[0], group[-1]
                if first - bp <= si <= last:
                    for g in group: ch |= skill_changed_map[g]
                    gr |= group_green[(first, last)]
            propagated_changed[si] = ch
            propagated_green[si]   = gr

        # DINOv3 패치 추출 (에피소드 전체 → PCA fit)
        all_patches: list[np.ndarray] = []
        for s in skills:
            fi_s = min(s["frame_start"], T - 1)
            fi_e = min(s["frame_end"] - 1, T - 1)
            sp = extract_patches(dino_model, dino_mean, dino_std,
                                 ep_frames[fi_s], args.image_size, args.patch_size, args.device)
            ep = extract_patches(dino_model, dino_mean, dino_std,
                                 ep_frames[fi_e], args.image_size, args.patch_size, args.device)
            s["start_patches"] = sp
            s["end_patches"]   = ep
            all_patches.extend([sp, ep])

        patch_matrix = np.concatenate(all_patches, axis=0)
        pca = PCA(n_components=3)
        pca.fit(patch_matrix)
        t_all = pca.transform(patch_matrix)
        pca.global_min = t_all.min(axis=0)
        pca.global_max = t_all.max(axis=0)

        # 스킬별 패널 생성 → 에피소드 단위로 가로 합치기
        ep_panels: list[np.ndarray] = []
        skill_captions: list[str] = []
        task_id = skills[0]["task_id"]

        for s in sorted(skills, key=lambda x: x["skill_index"]):
            fi_s = min(s["frame_start"], T - 1)
            fi_e = min(s["frame_end"] - 1, T - 1)
            start_obj   = tracked.get(fi_s, {})
            end_obj     = tracked.get(fi_e, {})
            changed_ids = propagated_changed.get(s["skill_index"], set())
            green_ids   = propagated_green.get(s["skill_index"], set())

            panel = make_skill_panel(
                ep_frames[fi_s], ep_frames[fi_e],
                start_obj, end_obj,
                changed_ids, green_ids,
                s["start_patches"], s["end_patches"],
                pca, args.image_size, args.patch_size, args.patch_hit_threshold,
            )

            # 스킬 헤더 바 (상단 20px, 스킬 인덱스별 색상)
            header_h = 20
            hue_step = max(1, 360 // max(len(skills), 1))
            hue = (s["skill_index"] * hue_step) % 360
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(hue / 360, 0.7, 0.9)
            header_color = (int(r * 255), int(g * 255), int(b * 255))
            header = np.full((header_h, panel.shape[1], 3), header_color, dtype=np.uint8)
            panel = np.concatenate([header, panel], axis=0)

            ep_panels.append(panel)

            raw_changed = skill_changed_map.get(s["skill_index"], set())
            prop_mark = "(p)" if changed_ids != raw_changed else ""
            skill_captions.append(
                f"sk{s['skill_index']}{prop_mark} f{s['frame_start']}→{s['frame_end']} "
                f"ch={sorted(changed_ids)} gr={sorted(green_ids)}"
            )
            n_done += 1

        # 스킬 패널 가로 이어붙이기 (스킬 간 구분선 8px)
        if ep_panels and wandb_run is not None:
            import wandb
            skill_div = np.full((ep_panels[0].shape[0], 8, 3), 40, dtype=np.uint8)
            ep_image = ep_panels[0]
            for p in ep_panels[1:]:
                ep_image = np.concatenate([ep_image, skill_div, p], axis=1)

            caption = (
                f"ep={ep_id} task={task_id} n_skills={len(skills)} | "
                f"row0=mask row1=4x4 row2=8x8 row3=14x14 row4=PCA(ep-fit) | "
                + " || ".join(skill_captions)
            )
            wandb_log = {
                f"dino_patches/task{task_id:02d}/ep{ep_id:05d}": wandb.Image(ep_image, caption=caption)
            }
            wandb_run.log({"patches": wandb_log, "n_skills": n_done})

    if wandb_run is not None:
        wandb_run.finish()
    print(f"[vis_dino] Done. Processed {n_done} skills.")


if __name__ == "__main__":
    main(tyro.cli(Args))
