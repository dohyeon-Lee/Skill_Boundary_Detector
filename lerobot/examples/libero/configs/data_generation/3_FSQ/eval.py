"""
접근 데이터셋 : libero_90, libero_90_for_FSQ
FSQ eval — task/episode 단위 스킬 시각화.

Section 1 (Row 1): skill1_initial | skill1_final | skill2_initial | skill2_final | ...
Section 2 (Row 2): 동일 순서로 8×8 DINO 패치 PCA + SAM2 마스크 오버레이 (init/final)
Section 3 (Grid) : 각 스킬별로 n개 랜덤 샘플 프레임의 패치 시각화 (세로=스킬, 가로=프레임)

패치 색상 (SAM2 마스크 우선):
  changed only  → 빨강  (220, 0, 0)
  green   only  → 초록  (0, 200, 0)
  둘 다         → 파랑  (0, 0, 255)
  없음          → DINO PCA 색상 (dino_tokens_path 미지정 시 회색)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--homedir",          default="/data2/dohyeon")
    p.add_argument("--projdir",          default="/SBD")
    p.add_argument("--dataset",          default="libero_90")
    p.add_argument("--dataset_root",     default="libero_dataset")
    p.add_argument("--image_key",        default="observation.images.image")
    p.add_argument("--task_id",          type=int, required=True)
    p.add_argument("--episode_id",       type=int, required=True)
    p.add_argument("--image_size",       type=int, default=128,
                   help="Pixel size of each skill image panel (sections 1 & 2).")
    p.add_argument("--n_samples",        type=int, default=10,
                   help="Number of random frames to sample per skill for section 3.")
    p.add_argument("--dino_tokens_path", default="",
                   help="Path to merged DINO tokens npz. Auto-resolved if empty.")
    p.add_argument("--visual_backbone",  default="dinov3_vits16")
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--output_dir",       default="")
    return p.parse_args()


# ── Dataset / video utils ─────────────────────────────────────────────────────

def load_episodes_meta(dataset_dir: Path):
    import pandas as pd
    files = sorted((dataset_dir / "meta" / "episodes").rglob("file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {dataset_dir / 'meta' / 'episodes'}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def read_episode_frames(dataset_dir: Path, meta_df, ep_id: int, image_key: str) -> np.ndarray:
    row     = meta_df[meta_df["episode_index"] == ep_id].iloc[0]
    chunk   = int(row[f"videos/{image_key}/chunk_index"])
    fidx    = int(row[f"videos/{image_key}/file_index"])
    from_ts = float(row[f"videos/{image_key}/from_timestamp"])
    to_ts   = float(row[f"videos/{image_key}/to_timestamp"])
    ep_len  = int(row["length"])
    vpath   = dataset_dir / "videos" / image_key / f"chunk-{chunk:03d}" / f"file-{fidx:03d}.mp4"

    from torchvision.io import read_video
    frames, _, _ = read_video(
        str(vpath), start_pts=from_ts, end_pts=to_ts - 0.001,
        pts_unit="sec", output_format="THWC",
    )
    arr = frames.numpy().astype(np.uint8)[..., :3]
    if len(arr) > ep_len:
        arr = arr[len(arr) - ep_len:]
    return arr


# ── Skillset utils ────────────────────────────────────────────────────────────

def load_episode_skills(skills_dir: Path, task_id: int, episode_id: int) -> list[dict]:
    candidates = [skills_dir / f"task{task_id:02d}", skills_dir / f"task{task_id}"]
    task_dir   = next((d for d in candidates if d.exists()), None)
    if task_dir is None:
        raise FileNotFoundError(
            f"No task dir for task_id={task_id} in {skills_dir}\n"
            f"Tried: {[str(d) for d in candidates]}"
        )
    files = sorted(task_dir.glob(f"ep{episode_id:05d}_*.npz"))
    if not files:
        raise FileNotFoundError(f"No skill files for ep {episode_id} in {task_dir}")
    skills = []
    for path in files:
        d = np.load(str(path))
        skills.append({
            "skill_index": int(d["skill_index"]),
            "frame_start": int(d["frame_start"]),
            "frame_end":   int(d["frame_end"]),
            "task_id":     int(d["task_id"]) if "task_id" in d else task_id,
        })
    skills.sort(key=lambda x: x["frame_start"])
    return skills


# ── SAM2 mask utils ───────────────────────────────────────────────────────────

def load_all_sam2_masks(sam2_dir: Path, task_id: int, ep_id: int,
                        skill_index: int) -> np.ndarray | None:
    """Returns (T, 8, 8, 2) bool or None."""
    path = sam2_dir / f"task{task_id}" / f"ep{ep_id:05d}_skill{skill_index:03d}.npz"
    if not path.exists():
        return None
    d  = np.load(str(path))
    pm = d["patch_masks"]
    return pm.astype(bool) if len(pm) > 0 else None


# ── DINO token utils ──────────────────────────────────────────────────────────

def load_all_dino_patches(tokens_path: Path, ep_id: int,
                          skill_index: int) -> np.ndarray | None:
    """Returns (T, 64, feat_dim) float32 or None."""
    if tokens_path is None or not tokens_path.exists():
        return None
    d        = np.load(str(tokens_path))
    ep_ids   = d["episode_id"]
    sk_ids   = d["skill_index"]
    offsets  = d["offsets"]
    features = d["features"]   # (N_total, n_tokens, feat_dim) float16

    idx = np.where((ep_ids == ep_id) & (sk_ids == skill_index))[0]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    skill_feats = features[offsets[i]:offsets[i + 1]].astype(np.float32)  # (T, 65, feat_dim)
    if len(skill_feats) == 0:
        return None
    return skill_feats[:, 1:, :]   # skip CLS → (T, 64, feat_dim)


# ── PCA coloring ──────────────────────────────────────────────────────────────

def pca_to_rgb(patches: np.ndarray) -> np.ndarray:
    """(64, feat_dim) float32 → (8, 8, 3) uint8 via SVD-based PCA."""
    X = patches.copy().astype(np.float32)
    X -= X.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        proj = X @ Vt[:3].T   # (64, 3)
    except np.linalg.LinAlgError:
        proj = np.zeros((64, 3), dtype=np.float32)
    for c in range(3):
        lo, hi = float(proj[:, c].min()), float(proj[:, c].max())
        proj[:, c] = (proj[:, c] - lo) / (hi - lo) if hi > lo else np.full(64, 0.5)
    return (proj * 255).clip(0, 255).astype(np.uint8).reshape(8, 8, 3)


# ── Patch image construction ──────────────────────────────────────────────────

_RED   = (220,   0,   0)
_GREEN = (  0, 200,   0)
_BLUE  = (  0,   0, 255)
_GRAY  = (180, 180, 180)


def make_patch_image(patches_64: np.ndarray | None,
                     mask_8x8_2: np.ndarray | None,
                     cell_size: int) -> Image.Image:
    """
    patches_64 : (64, feat_dim) float32 or None
    mask_8x8_2 : (8, 8, 2) bool  ch0=changed ch1=green  or None
    Returns PIL image of size (8*cell_size, 8*cell_size).
    """
    if patches_64 is not None:
        base_rgb = pca_to_rgb(patches_64)
    else:
        base_rgb = np.full((8, 8, 3), _GRAY, dtype=np.uint8)

    img_arr = np.repeat(np.repeat(base_rgb, cell_size, axis=0), cell_size, axis=1).copy()

    if mask_8x8_2 is not None:
        changed = mask_8x8_2[..., 0]
        green   = mask_8x8_2[..., 1]
        both    = changed & green
        for r in range(8):
            for c in range(8):
                rs, re = r * cell_size, (r + 1) * cell_size
                cs, ce = c * cell_size, (c + 1) * cell_size
                if both[r, c]:
                    img_arr[rs:re, cs:ce] = _BLUE
                elif changed[r, c]:
                    img_arr[rs:re, cs:ce] = _RED
                elif green[r, c]:
                    img_arr[rs:re, cs:ce] = _GREEN

    result = Image.fromarray(img_arr)
    draw   = ImageDraw.Draw(result)
    total  = 8 * cell_size
    for i in range(1, 8):
        pos = i * cell_size
        draw.line([(pos, 0), (pos, total - 1)], fill=(50, 50, 50), width=1)
        draw.line([(0, pos), (total - 1, pos)], fill=(50, 50, 50), width=1)
    return result


# ── Layout constants ──────────────────────────────────────────────────────────

SEP_W    = 4
LABEL_H  = 16
BG       = (30,  30,  30)
SEP_COL  = (60,  60,  60)
TXT_COL  = (230, 230, 230)


def _label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str) -> None:
    draw.text((x + 3, y + 2), text, fill=TXT_COL)


# ── Section 1 & 2: init/final images + patch grids ───────────────────────────

def build_sections_12(
    ep_frames: np.ndarray,
    skills: list[dict],
    sam2_dir: Path,
    ep_id: int,
    tokens_path: Path | None,
    image_size: int,
    cell_size: int,
) -> Image.Image:
    patch_size = 8 * cell_size
    panel_w    = max(image_size, patch_size)
    n_panels   = len(skills) * 2
    T          = len(ep_frames)

    total_w = n_panels * panel_w + (n_panels + 1) * SEP_W
    # Section 1 (image row) + Section 2 (patch row), each with label strip
    sec_h   = LABEL_H + panel_w
    total_h = 2 * sec_h + SEP_W   # horizontal sep between sections

    canvas = Image.new("RGB", (total_w, total_h), BG)
    draw   = ImageDraw.Draw(canvas)

    # vertical separators
    for k in range(n_panels + 1):
        sx = k * (panel_w + SEP_W)
        draw.rectangle([sx, 0, sx + SEP_W - 1, total_h - 1], fill=SEP_COL)

    # horizontal sep between sec 1 and sec 2
    draw.rectangle([0, sec_h, total_w - 1, sec_h + SEP_W - 1], fill=SEP_COL)

    for si, skill in enumerate(skills):
        fs      = min(int(skill["frame_start"]), T - 1)
        fe      = max(0, min(int(skill["frame_end"]) - 1, T - 1))
        sk_idx  = skill["skill_index"]
        task_id = skill["task_id"]

        all_masks  = load_all_sam2_masks(sam2_dir, task_id, ep_id, sk_idx)
        all_dino   = load_all_dino_patches(tokens_path, ep_id, sk_idx)

        for col, (abs_fi, t_rel, phase) in enumerate([
            (fs, 0,  "init"),
            (fe, -1, "final"),
        ]):
            panel_idx = si * 2 + col
            px        = panel_idx * (panel_w + SEP_W) + SEP_W

            # ── Section 1: RGB image ──
            img = Image.fromarray(ep_frames[abs_fi]).resize((image_size, image_size), Image.BILINEAR)
            if image_size < panel_w:
                bg = Image.new("RGB", (panel_w, panel_w), BG)
                bg.paste(img, ((panel_w - image_size) // 2, (panel_w - image_size) // 2))
                img = bg
            canvas.paste(img, (px, LABEL_H))
            _label(draw, px, 0, f"s{sk_idx} {phase} f{abs_fi}")

            # ── Section 2: patch PCA + SAM2 ──
            patches  = all_dino[t_rel]   if all_dino  is not None else None
            mask     = all_masks[t_rel]  if all_masks is not None else None
            pimg     = make_patch_image(patches, mask, cell_size)
            if patch_size < panel_w:
                bg = Image.new("RGB", (panel_w, panel_w), BG)
                bg.paste(pimg, ((panel_w - patch_size) // 2, (panel_w - patch_size) // 2))
                pimg = bg
            canvas.paste(pimg, (px, sec_h + SEP_W + LABEL_H))
            _label(draw, px, sec_h + SEP_W, f"s{sk_idx} {phase} patches")

    return canvas


# ── Section 3: random frame grid (rows=skills, cols=sampled frames) ──────────

def build_section3(
    skills: list[dict],
    sam2_dir: Path,
    ep_id: int,
    tokens_path: Path | None,
    n_samples: int,
    cell_size: int,
    rng: np.random.Generator,
) -> Image.Image:
    patch_size = 8 * cell_size
    panel_w    = patch_size

    n_skills = len(skills)
    total_w  = n_samples * panel_w + (n_samples + 1) * SEP_W
    row_h    = LABEL_H + panel_w
    total_h  = n_skills * row_h + (n_skills + 1) * SEP_W

    canvas = Image.new("RGB", (total_w, total_h), BG)
    draw   = ImageDraw.Draw(canvas)

    # horizontal separators (between skill rows)
    for k in range(n_skills + 1):
        sy = k * (row_h + SEP_W)
        draw.rectangle([0, sy, total_w - 1, sy + SEP_W - 1], fill=SEP_COL)

    # vertical separators (between frame columns)
    for k in range(n_samples + 1):
        sx = k * (panel_w + SEP_W)
        draw.rectangle([sx, 0, sx + SEP_W - 1, total_h - 1], fill=SEP_COL)

    for si, skill in enumerate(skills):
        sk_idx    = skill["skill_index"]
        task_id   = skill["task_id"]
        fs        = int(skill["frame_start"])
        fe        = int(skill["frame_end"])
        sk_len    = fe - fs        # number of frames in skill

        all_masks = load_all_sam2_masks(sam2_dir, task_id, ep_id, sk_idx)
        all_dino  = load_all_dino_patches(tokens_path, ep_id, sk_idx)
        actual_T  = sk_len if all_dino is None else len(all_dino)

        # Sample n indices within [0, actual_T), sorted ascending
        n = min(n_samples, actual_T)
        sampled_t = sorted(rng.choice(actual_T, size=n, replace=False).tolist())

        row_y = si * (row_h + SEP_W) + SEP_W

        for col, t_rel in enumerate(sampled_t):
            px = col * (panel_w + SEP_W) + SEP_W

            patches = all_dino[t_rel]   if all_dino  is not None else None
            mask    = all_masks[t_rel]  if all_masks is not None else None
            pimg    = make_patch_image(patches, mask, cell_size)
            canvas.paste(pimg, (px, row_y + LABEL_H))
            _label(draw, px, row_y, f"s{sk_idx} t={fs + t_rel}")

        # Fill remaining columns if actual_T < n_samples
        for col in range(n, n_samples):
            px = col * (panel_w + SEP_W) + SEP_W
            draw.rectangle([px, row_y, px + panel_w - 1, row_y + row_h - 1], fill=(20, 20, 20))

    return canvas


# ── Legend ────────────────────────────────────────────────────────────────────

def make_legend(width: int) -> Image.Image:
    h   = 20
    img = Image.new("RGB", (width, h), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    items = [("changed", _RED), ("green", _GREEN), ("both", _BLUE)]
    x = 8
    for label, color in items:
        draw.rectangle([x, 4, x + 10, 14], fill=color)
        draw.text((x + 14, 3), label, fill=TXT_COL)
        x += 80
    return img


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    root        = Path(args.homedir + args.projdir)
    dataset_dir = root / args.dataset_root / args.dataset
    fsq_dir     = root / args.dataset_root / f"{args.dataset}_data" / f"{args.dataset}_for_FSQ"
    skills_dir  = fsq_dir / f"{args.dataset}_skillset" / "skills"
    sam2_dir    = fsq_dir / "sam2_masks"

    tokens_path: Path | None = Path(args.dino_tokens_path) if args.dino_tokens_path else \
                               fsq_dir / f"{args.visual_backbone}_tokens.npz"
    if tokens_path is not None and not tokens_path.exists():
        print(f"[eval] DINO tokens not found at {tokens_path} — patches will be gray")
        tokens_path = None

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "image"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] dataset   : {dataset_dir}")
    print(f"[eval] skills    : {skills_dir}")
    print(f"[eval] sam2      : {sam2_dir}")
    print(f"[eval] tokens    : {tokens_path or '(none)'}")

    meta   = load_episodes_meta(dataset_dir)
    skills = load_episode_skills(skills_dir, args.task_id, args.episode_id)
    print(f"[eval] episode={args.episode_id}  task={args.task_id}  skills={len(skills)}")
    for s in skills:
        print(f"       skill {s['skill_index']:2d}: frames {s['frame_start']}–{s['frame_end'] - 1}")

    frames    = read_episode_frames(dataset_dir, meta, args.episode_id, args.image_key)
    cell_size = max(1, args.image_size // 8)

    print(f"[eval] decoded {len(frames)} frames")
    print(f"[eval] building sections 1+2 ...")
    sec12 = build_sections_12(
        ep_frames   = frames,
        skills      = skills,
        sam2_dir    = sam2_dir,
        ep_id       = args.episode_id,
        tokens_path = tokens_path,
        image_size  = args.image_size,
        cell_size   = cell_size,
    )

    print(f"[eval] building section 3 (n_samples={args.n_samples}) ...")
    sec3 = build_section3(
        skills      = skills,
        sam2_dir    = sam2_dir,
        ep_id       = args.episode_id,
        tokens_path = tokens_path,
        n_samples   = args.n_samples,
        cell_size   = cell_size,
        rng         = rng,
    )

    # Stack vertically: sec12 / legend / sec3
    legend     = make_legend(max(sec12.width, sec3.width))
    total_w    = max(sec12.width, sec3.width, legend.width)
    total_h    = sec12.height + SEP_W + legend.height + SEP_W + sec3.height

    out = Image.new("RGB", (total_w, total_h), BG)
    out.paste(sec12,   (0, 0))
    out.paste(legend,  (0, sec12.height + SEP_W))
    out.paste(sec3,    (0, sec12.height + SEP_W + legend.height + SEP_W))

    out_path = output_dir / f"fsq_task{args.task_id:02d}_ep{args.episode_id:05d}.png"
    out.save(str(out_path))
    print(f"[eval] saved → {out_path}")


if __name__ == "__main__":
    main()
