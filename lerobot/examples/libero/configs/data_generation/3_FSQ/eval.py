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

  python eval.py --task_id 1 --num_episodes 1 
  --dino_tokens_path ""
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CONFIG_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONFIG_DIR))

from pipeline_config import load_config  # noqa: E402


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    cfg = load_config()
    p = argparse.ArgumentParser()
    p.add_argument("--homedir",          default=cfg.homedir)
    p.add_argument("--projdir",          default=cfg.projdir)
    p.add_argument("--dataset",          default=cfg.data)
    p.add_argument("--dataset_root",     default=cfg.datadir)
    p.add_argument("--image_key",        default=cfg.image_key)
    p.add_argument("--task_id",          type=int, required=True)
    p.add_argument("--num_episodes",     type=int, default=2,
                   help="Number of episodes to visualize (taken in order from skills dir).")
    p.add_argument("--episode_offset",   type=int, default=0,
                   help="Skip this many episodes before starting.")
    p.add_argument("--image_size",       type=int, default=128,
                   help="Pixel size of each skill image panel (sections 1 & 2).")
    p.add_argument("--n_samples",        type=int, default=10,
                   help="Number of random frames to sample per skill for section 3.")
    p.add_argument("--dino_tokens_path", default="",
                   help="Path to merged DINO tokens npz. Auto-resolved if empty.")
    p.add_argument("--visual_backbone",  default=cfg.visual_backbone)
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

def _get_task_dir(skills_dir: Path, task_id: int) -> Path:
    candidates = [skills_dir / f"task{task_id:02d}", skills_dir / f"task{task_id}"]
    task_dir   = next((d for d in candidates if d.exists()), None)
    if task_dir is None:
        raise FileNotFoundError(
            f"No task dir for task_id={task_id} in {skills_dir}\n"
            f"Tried: {[str(d) for d in candidates]}"
        )
    return task_dir


def list_task_episode_ids(skills_dir: Path, task_id: int) -> list[int]:
    task_dir = _get_task_dir(skills_dir, task_id)
    return sorted({int(f.name[2:7]) for f in task_dir.glob("ep*.npz")})


def load_episode_skills(skills_dir: Path, task_id: int, episode_id: int) -> list[dict]:
    task_dir = _get_task_dir(skills_dir, task_id)
    files    = sorted(task_dir.glob(f"ep{episode_id:05d}_*.npz"))
    if not files:
        available = list_task_episode_ids(skills_dir, task_id)
        raise FileNotFoundError(
            f"No skill files for ep {episode_id} in {task_dir}\n"
            f"Available episode IDs: {available}"
        )
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


# ── SAM2 patch flags utils (merged patch_flags.npz) ──────────────────────────

def load_flags_cache(flags_path: Path | None) -> dict | None:
    """Load patch_flags.npz once. Returns dict with arrays, or None."""
    if flags_path is None or not flags_path.exists():
        return None
    print(f"[eval] loading patch_flags ...")
    d = np.load(str(flags_path))
    cache = {
        "patch_flags": d["patch_flags"],   # (N_total, 64, 2) uint8
        "offsets":     d["offsets"],        # (n_skills+1,)
        "episode_id":  d["episode_id"],
        "skill_index": d["skill_index"],
        "missing":     d["missing"],
    }
    n_skills = len(cache["episode_id"])
    print(f"[eval] patch_flags loaded: {n_skills} skills, "
          f"missing={int(cache['missing'].sum())}")
    return cache


def load_all_sam2_masks(flags_cache: dict | None, ep_id: int,
                        skill_index: int) -> np.ndarray | None:
    """Returns (T, 8, 8, 2) bool or None."""
    if flags_cache is None:
        return None
    ep_ids  = flags_cache["episode_id"]
    sk_ids  = flags_cache["skill_index"]
    offsets = flags_cache["offsets"]
    flags   = flags_cache["patch_flags"]
    missing = flags_cache["missing"]

    idx = np.where((ep_ids == ep_id) & (sk_ids == skill_index))[0]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    if missing[i]:
        return None
    flat = flags[offsets[i]:offsets[i + 1]]   # (T, 64, 2) uint8
    if len(flat) == 0:
        return None
    return flat.reshape(len(flat), 8, 8, 2).astype(bool)


# ── DINO token utils ──────────────────────────────────────────────────────────

def load_tokens_cache(tokens_path: Path | None) -> dict | None:
    """Load the tokens npz once. Returns dict with arrays, or None."""
    if tokens_path is None or not tokens_path.exists():
        return None
    print(f"[eval] loading DINO tokens (this may take a moment) ...")
    d = np.load(str(tokens_path))
    cache = {
        "features":   d["features"],     # (N_total, n_tokens, feat_dim) float16
        "episode_id": d["episode_id"],
        "skill_index": d["skill_index"],
        "offsets":    d["offsets"],
    }
    print(f"[eval] tokens loaded: features{cache['features'].shape}")
    return cache


def load_all_dino_patches(tokens_cache: dict | None, ep_id: int,
                          skill_index: int) -> np.ndarray | None:
    """Returns (T, 64, feat_dim) float32 or None."""
    if tokens_cache is None:
        return None
    ep_ids   = tokens_cache["episode_id"]
    sk_ids   = tokens_cache["skill_index"]
    offsets  = tokens_cache["offsets"]
    features = tokens_cache["features"]

    idx = np.where((ep_ids == ep_id) & (sk_ids == skill_index))[0]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    skill_feats = features[offsets[i]:offsets[i + 1]].astype(np.float32)  # (T, 65, feat_dim)
    if len(skill_feats) == 0:
        return None
    return skill_feats[:, 1:, :]   # skip CLS → (T, 64, feat_dim)


# ── PCA coloring ──────────────────────────────────────────────────────────────

def build_episode_pca(tokens_cache: dict | None, ep_id: int,
                      skills: list[dict]) -> dict | None:
    """
    Fit PCA on all patches from every skill/frame of the episode.
    Returns dict(Vt, mean, proj_min, proj_max) for consistent coloring,
    or None if tokens are unavailable.
    Uses covariance-matrix eigenvectors (fast when n_samples >> feat_dim).
    """
    if tokens_cache is None:
        return None

    all_patches = []
    for skill in skills:
        patches = load_all_dino_patches(tokens_cache, ep_id, skill["skill_index"])
        if patches is not None:
            all_patches.append(patches.reshape(-1, patches.shape[-1]))

    if not all_patches:
        return None

    X = np.concatenate(all_patches, axis=0).astype(np.float32)  # (N, feat_dim)
    mean = X.mean(axis=0)
    X -= mean

    # Covariance eigenvectors — O(feat_dim^3), much faster than full SVD on large N
    cov = (X.T @ X) / len(X)                   # (feat_dim, feat_dim)
    _, vecs = np.linalg.eigh(cov)               # ascending eigenvalues
    Vt = vecs[:, ::-1][:, :3].T.astype(np.float32)  # (3, feat_dim), top-3 components

    proj = X @ Vt.T                             # (N, 3)
    return {
        "Vt":       Vt,
        "mean":     mean,
        "proj_min": proj.min(axis=0),
        "proj_max": proj.max(axis=0),
    }


def pca_to_rgb(patches: np.ndarray, ep_pca: dict | None) -> np.ndarray:
    """(64, feat_dim) float32 → (8, 8, 3) uint8.
    Uses episode-level PCA axes if provided, otherwise per-frame fallback."""
    X = patches.astype(np.float32)
    if ep_pca is not None:
        X = X - ep_pca["mean"]
        proj = X @ ep_pca["Vt"].T                          # (64, 3)
        lo, hi = ep_pca["proj_min"], ep_pca["proj_max"]
        scale = np.where(hi > lo, hi - lo, 1.0)
        proj = (proj - lo) / scale
    else:
        X -= X.mean(axis=0)
        try:
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            proj = X @ Vt[:3].T
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
                     cell_size: int,
                     ep_pca: dict | None = None) -> Image.Image:
    """
    patches_64 : (64, feat_dim) float32 or None
    mask_8x8_2 : (8, 8, 2) bool  ch0=changed ch1=green  or None
    Returns PIL image of size (8*cell_size, 8*cell_size).
    """
    if patches_64 is not None:
        base_rgb = pca_to_rgb(patches_64, ep_pca)
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
BG            = (30,  30,  30)
SEP_COL       = (60,  60,  60)   # within-skill separator (init | final)
SKILL_SEP_COL = (200, 200, 200)  # between-skill separator
TXT_COL       = (230, 230, 230)


def _label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str) -> None:
    draw.text((x + 3, y + 2), text, fill=TXT_COL)


# ── Section 1 & 2: init/final images + patch grids ───────────────────────────

def build_sections_12(
    ep_frames: np.ndarray,
    skills: list[dict],
    flags_cache: dict | None,
    ep_id: int,
    tokens_cache: dict | None,
    ep_pca: dict | None,
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

    # vertical separators: even k = skill boundary (bright), odd k = init|final (dim)
    for k in range(n_panels + 1):
        sx    = k * (panel_w + SEP_W)
        color = SKILL_SEP_COL if k % 2 == 0 else SEP_COL
        draw.rectangle([sx, 0, sx + SEP_W - 1, total_h - 1], fill=color)

    # horizontal sep between sec 1 and sec 2
    draw.rectangle([0, sec_h, total_w - 1, sec_h + SEP_W - 1], fill=SEP_COL)

    for si, skill in enumerate(skills):
        fs      = min(int(skill["frame_start"]), T - 1)
        fe      = max(0, min(int(skill["frame_end"]) - 1, T - 1))
        sk_idx  = skill["skill_index"]
        task_id = skill["task_id"]

        all_masks  = load_all_sam2_masks(flags_cache, ep_id, sk_idx)
        all_dino   = load_all_dino_patches(tokens_cache, ep_id, sk_idx)

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
            pimg     = make_patch_image(patches, mask, cell_size, ep_pca)
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
    flags_cache: dict | None,
    ep_id: int,
    tokens_cache: dict | None,
    ep_pca: dict | None,
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

        all_masks = load_all_sam2_masks(flags_cache, ep_id, sk_idx)
        all_dino  = load_all_dino_patches(tokens_cache, ep_id, sk_idx)
        actual_T  = sk_len if all_dino is None else len(all_dino)

        # Sample n indices within [0, actual_T), sorted ascending
        n = min(n_samples, actual_T)
        sampled_t = sorted(rng.choice(actual_T, size=n, replace=False).tolist())

        row_y = si * (row_h + SEP_W) + SEP_W

        for col, t_rel in enumerate(sampled_t):
            px = col * (panel_w + SEP_W) + SEP_W

            patches = all_dino[t_rel]   if all_dino  is not None else None
            mask    = all_masks[t_rel]  if all_masks is not None else None
            pimg    = make_patch_image(patches, mask, cell_size, ep_pca)
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
    flags_path  = fsq_dir / "patch_flags.npz"

    tokens_path: Path | None = Path(args.dino_tokens_path) if args.dino_tokens_path else \
                               fsq_dir / f"{args.visual_backbone}_tokens.npz"
    if tokens_path is not None and not tokens_path.exists():
        print(f"[eval] DINO tokens not found at {tokens_path} — patches will be gray")
        tokens_path = None

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "image"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] dataset   : {dataset_dir}")
    print(f"[eval] skills    : {skills_dir}")
    print(f"[eval] flags     : {flags_path}")
    print(f"[eval] tokens    : {tokens_path or '(none)'}")

    flags_cache  = load_flags_cache(flags_path)
    tokens_cache = load_tokens_cache(tokens_path)

    all_ep_ids = list_task_episode_ids(skills_dir, args.task_id)
    ep_ids     = all_ep_ids[args.episode_offset : args.episode_offset + args.num_episodes]
    print(f"[eval] task={args.task_id}  total_episodes={len(all_ep_ids)}  selected={ep_ids}")

    meta      = load_episodes_meta(dataset_dir)
    cell_size = max(1, args.image_size // 8)

    for ep_id in ep_ids:
        skills = load_episode_skills(skills_dir, args.task_id, ep_id)
        print(f"[eval] ep={ep_id}  skills={len(skills)}")
        for s in skills:
            print(f"       skill {s['skill_index']:2d}: frames {s['frame_start']}–{s['frame_end'] - 1}")

        frames = read_episode_frames(dataset_dir, meta, ep_id, args.image_key)
        print(f"       decoded {len(frames)} frames")

        ep_pca = build_episode_pca(tokens_cache, ep_id, skills)
        print(f"       episode PCA {'fitted' if ep_pca else 'skipped (no tokens)'}")

        sec12 = build_sections_12(
            ep_frames=frames, skills=skills, flags_cache=flags_cache, ep_id=ep_id,
            tokens_cache=tokens_cache, ep_pca=ep_pca,
            image_size=args.image_size, cell_size=cell_size,
        )
        sec3 = build_section3(
            skills=skills, flags_cache=flags_cache, ep_id=ep_id,
            tokens_cache=tokens_cache, ep_pca=ep_pca,
            n_samples=args.n_samples, cell_size=cell_size, rng=rng,
        )
        legend  = make_legend(max(sec12.width, sec3.width))
        total_w = max(sec12.width, sec3.width, legend.width)
        total_h = sec12.height + SEP_W + legend.height + SEP_W + sec3.height

        out = Image.new("RGB", (total_w, total_h), BG)
        out.paste(sec12,  (0, 0))
        out.paste(legend, (0, sec12.height + SEP_W))
        out.paste(sec3,   (0, sec12.height + SEP_W + legend.height + SEP_W))

        out_path = output_dir / f"fsq_task{args.task_id:02d}_ep{ep_id:05d}.png"
        out.save(str(out_path))
        print(f"       saved → {out_path}")


if __name__ == "__main__":
    main()
