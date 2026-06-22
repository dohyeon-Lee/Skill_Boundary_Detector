#!/usr/bin/env python3
"""Side-by-side eval-video comparison.

Take 2-4 Stage-1 eval OUTPUT folders (the ones under stage1_eval/outputs/, e.g.
``..._np_state_c10_010000_adv-terminator``) and, for every SAME task + SAME episode that all of
them share, glue their rollout videos HORIZONTALLY into one clip with each panel labelled on top.
Lets you eyeball skill-MOTION differences (state vs state_cum vs state_skill ...) on identical
init states.

Matching key (per eval folder):  videos/{target_task}_{task_id}/eval_episode_{ep}.mp4
→ a combined clip is written for each (task_id, ep) present in ALL inputs.

No system ffmpeg / drawtext needed: frames are read with imageio (bundled ffmpeg), resized with
cv2, labelled with PIL, hstacked with numpy, written back with imageio (libx264). Shorter clips are
padded by freezing their last frame so every panel runs to the longest one.

Pick the runs to compare in video_compare_config.yaml (folders/labels/tasks/episodes/height/fps),
then just:  ./run.sh        (wrapper that finds the project venv and runs this with that yaml)

Any CLI flag or positional folder overrides the yaml, e.g. one-off from this dir:
    ./run.sh --tasks 0 --episodes 0
    ./run.sh np_state_c10 np_state_skill_c10        # ad-hoc folders, ignore the yaml's list
Run with the PROJECT venv (imageio/cv2/PIL — {project_root}/.venv, NOT lerobot/.venv); run.sh handles that.

Each arg may be a full folder name, or any unique substring of one (e.g. ``state_cum0.5r2_c10``).
Options:
    --labels a,b,c[,d]   override the auto-derived panel labels (default = the part of each folder
                         name that differs across the inputs)
    --height N           panel height in px (default 256; width auto from aspect)
    --tasks  0,1,2       only these task_ids (default: all shared)
    --episodes 0,1       only these episode indices (default: all shared)
    --fps N              output fps (default: source fps)
    --out DIR            output dir (default: video_compare/outputs/{label1_vs_label2_...})
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
EVAL_OUTPUTS = _HERE.parent / "outputs"          # stage1_eval/outputs/
VIDEO_RE = re.compile(r"eval_episode_(\d+)\.mp4$")


# ── folder + label helpers ───────────────────────────────────────────────────
def resolve_folder(arg: str, backbone: str = "") -> Path:
    """Resolve a CLI/yaml arg to an eval output folder: exact name, else substring match. When a
    substring matches several (e.g. 'sadaln_c10' ⊂ both '..._sadaln_c10...' and '..._np_sadaln_c10...'),
    prefer the UNIQUELY SHORTEST name — the tightest match, i.e. the one without an extra segment like
    'np_'. Only errors if the shortest is itself tied.

    When ``backbone`` is 'dino' or 'siglip', restrict to folders whose POLICY vision tag matches —
    the ``_dino_`` / ``_siglip_`` token (the `{dino|siglip}_{freeze|unfreeze}` policy backbone), NOT the
    run-tag's ``_dino8_`` (the FSQ tokenizer, present in EVERY folder). So a short substring like
    'np_state_c10' selects the dino vs siglip variant you asked for instead of silently picking one."""
    bb = f" (backbone={backbone})" if backbone else ""
    cand = [p for p in EVAL_OUTPUTS.glob("*") if p.is_dir()]
    if backbone:
        cand = [p for p in cand if f"_{backbone}_" in p.name]
    exact = EVAL_OUTPUTS / arg
    if exact.is_dir() and (not backbone or f"_{backbone}_" in exact.name):
        return exact
    hits = sorted((p for p in cand if arg in p.name), key=lambda p: len(p.name))
    if not hits:
        sys.exit(f"[error] no eval folder under {EVAL_OUTPUTS} matches {arg!r}{bb}")
    if len(hits) > 1 and len(hits[0].name) == len(hits[1].name):
        names = "\n  ".join(p.name for p in hits)
        sys.exit(f"[error] {arg!r}{bb} is ambiguous, matches:\n  {names}\n"
                 f"  → add more of the name (e.g. a preceding token) to disambiguate.")
    if len(hits) > 1:
        print(f"[note] {arg!r}{bb} matched {len(hits)}; using shortest: {hits[0].name}")
    return hits[0]


def split_backbone(arg: str, default_bb: str) -> tuple[str, str]:
    """A folder entry may be prefixed ``dino:`` / ``siglip:`` to force THAT panel's policy backbone,
    overriding the global ``backbone`` — so dino and siglip variants can sit side-by-side in ONE compare
    (e.g. 'siglip:np_state_c10_012500' next to 'dino:np_state_c10_012500'). Returns (backbone, bare_arg)."""
    for bb in ("dino", "siglip"):
        if arg.startswith(bb + ":"):
            return bb, arg[len(bb) + 1:]
    return default_bb, arg


def short_labels(names: list[str]) -> list[str]:
    """Strip the common LEADING + TRAILING underscore-tokens shared by all folder names → the
    distinguishing middle. Token-based (cuts only on '_' boundaries) and backs off the strip whenever
    a name has no unique token of its own (e.g. `state` ⊂ `state_skill`), so no label comes out empty."""
    if len(names) == 1:
        return names
    toks = [n.split("_") for n in names]
    shortest = min(len(t) for t in toks)
    p = 0
    while p < shortest and all(t[p] == toks[0][p] for t in toks):
        p += 1
    s = 0
    while s < shortest - p and all(t[-1 - s] == toks[0][-1 - s] for t in toks):
        s += 1
    def mids(p, s):
        return ["_".join(t[p: len(t) - s]) for t in toks]
    while p > 0 and any(m == "" for m in mids(p, s)):   # back off prefix first (keeps the mode token)
        p -= 1
    while s > 0 and any(m == "" for m in mids(p, s)):
        s -= 1
    return [m or "base" for m in mids(p, s)]


def load_font(px: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/google-droid-sans-fonts/DroidSans-Bold.ttf",
        "/usr/share/fonts/google-droid-sans-fonts/DroidSans.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
        "/usr/share/fonts/urw-base35/NimbusSans-Bold.otf",
    ]
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, px)
    found = list(Path("/usr/share/fonts").rglob("*.ttf"))
    if found:
        return ImageFont.truetype(str(found[0]), px)
    return ImageFont.load_default()


# ── frame ops ────────────────────────────────────────────────────────────────
def even(x: int) -> int:
    return x - (x % 2)


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    rdr = imageio.get_reader(str(path))
    fps = float(rdr.get_meta_data().get("fps", 10) or 10)
    frames = [np.asarray(f)[:, :, :3] for f in rdr]
    rdr.close()
    return frames, fps


def label_bar(width: int, bar_h: int, text: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    img = Image.new("RGB", (width, bar_h), (20, 20, 20))
    d = ImageDraw.Draw(img)
    # shrink text to fit the panel width
    f = font
    while d.textlength(text, font=f) > width - 8 and f.size > 8:
        f = ImageFont.truetype(f.path, f.size - 1) if hasattr(f, "path") else f
        if not hasattr(f, "path"):
            break
    tw = d.textlength(text, font=f)
    d.text(((width - tw) / 2, max(0, (bar_h - f.size) / 2 - 1)), text, fill=(245, 245, 245), font=f)
    return np.asarray(img)


def make_panel(frame: np.ndarray, H: int, bar: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    Wp = even(max(2, round(w * H / h)))
    resized = cv2.resize(frame, (Wp, H), interpolation=cv2.INTER_AREA)
    if bar.shape[1] != Wp:                       # bar was built per-panel at this width already
        bar = cv2.resize(bar, (Wp, bar.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.vstack([bar, resized])


# ── main ─────────────────────────────────────────────────────────────────────
def _as_list(v) -> list[str] | None:
    """Accept a comma-string ('0,1'), a yaml list ([0,1]), or None → list[str] or None."""
    if v is None or v == "":
        return None
    if isinstance(v, str):
        items = [x.strip() for x in v.split(",") if x.strip()]
    else:
        items = [str(x).strip() for x in v]
    return items or None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Glue same task/episode eval videos side by side. Reads video_compare_config.yaml "
                    "by default; any CLI flag (and positional folders) overrides the yaml.")
    ap.add_argument("folders", nargs="*", help="eval folders (override the yaml); name or unique substring")
    ap.add_argument("--config", default=str(_HERE / "video_compare_config.yaml"), help="config yaml")
    ap.add_argument("--backbone", default=None, help="restrict folder matching to dino|siglip policy variants")
    ap.add_argument("--labels", default=None, help="comma-sep panel labels (override yaml; default auto)")
    ap.add_argument("--height", type=int, default=None, help="panel height px")
    ap.add_argument("--tasks", default=None, help="comma-sep task_ids")
    ap.add_argument("--episodes", default=None, help="comma-sep episode idxs")
    ap.add_argument("--fps", type=float, default=None, help="output fps (default: source fps)")
    ap.add_argument("--out", default=None, help="output dir")
    args = ap.parse_args()

    cfg: dict = {}
    cfg_path = Path(args.config)
    if cfg_path.exists():
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    elif not args.folders:
        sys.exit(f"[error] no folders given and config not found: {cfg_path}")

    # CLI overrides yaml; yaml provides the defaults.
    raw_folders = args.folders or _as_list(cfg.get("folders")) or []
    if len(raw_folders) < 2:
        sys.exit("[error] need >=2 folders (positional args, or 'folders:' in the config yaml)")
    height = args.height or cfg.get("height") or 256
    tasks = _as_list(args.tasks) if args.tasks is not None else _as_list(cfg.get("tasks"))
    episodes = _as_list(args.episodes) if args.episodes is not None else _as_list(cfg.get("episodes"))
    fps = args.fps if args.fps is not None else float(cfg.get("fps") or 0.0)
    out = args.out if args.out is not None else (cfg.get("out") or "")
    backbone = (args.backbone if args.backbone is not None else (cfg.get("backbone") or "")).strip().lower()
    if backbone not in ("", "dino", "siglip"):
        sys.exit(f"[error] backbone must be 'dino', 'siglip' or '' (got {backbone!r})")

    # Each entry may carry a per-folder 'dino:'/'siglip:' prefix (overrides the global backbone). When
    # BOTH backbones appear among the entries → GRID layout: one ROW per backbone (dino on top, siglip
    # below), each row's columns = its entries in order, rows stacked VERTICALLY. Otherwise a single
    # horizontal row (original behavior).
    entries = []                                   # (backbone, resolved folder), in yaml order
    for a in raw_folders:
        bb, bare = split_backbone(str(a), backbone)
        entries.append((bb, resolve_folder(bare, bb)))
    present = [b for b in ("dino", "siglip", "") if any(bb == b for bb, _ in entries)]
    grid = len(present) >= 2
    rows = ([(b, [f for bb, f in entries if bb == b]) for b in present]
            if grid else [("", [f for _, f in entries])])
    ncols = len(rows[0][1])
    if grid and any(len(fs) != ncols for _, fs in rows):
        sys.exit("[error] grid mode needs the SAME number of panels per backbone row; got "
                 + ", ".join(f"{b or 'any'}:{len(fs)}" for b, fs in rows))

    # labels are PER COLUMN (shared across rows in grid mode → dino/siglip column labels may overlap).
    lab_override = _as_list(args.labels) or _as_list(cfg.get("labels"))
    labels = lab_override or short_labels([f.name for f in rows[0][1]])
    if len(labels) != ncols:
        sys.exit(f"[error] labels has {len(labels)} entries but {ncols} column(s) given")
    print(f"Comparing (grid {len(rows)}×{ncols}):" if grid else f"Comparing ({ncols} panels):")
    for bb, fs in rows:
        for ci, f in enumerate(fs):
            print(f"  [{(bb + '/') if bb else ''}{labels[ci]}]  {f.name}")

    # shared (task_id, episode) keys across ALL folders
    def keys_of(folder: Path) -> set[tuple[str, str]]:
        ks = set()
        vroot = folder / "videos"
        for taskdir in vroot.glob("*"):
            if not taskdir.is_dir():
                continue
            tid = taskdir.name.rsplit("_", 1)[-1]            # libero_90_3 → "3"
            for mp4 in taskdir.glob("eval_episode_*.mp4"):
                m = VIDEO_RE.search(mp4.name)
                if m:
                    ks.add((tid, m.group(1)))
        return ks

    all_folders = [f for _, fs in rows for f in fs]
    shared = set.intersection(*(keys_of(f) for f in all_folders))
    task_filter = set(tasks) if tasks else None
    ep_filter = set(episodes) if episodes else None
    keys = sorted(
        (t, e) for (t, e) in shared
        if (task_filter is None or t in task_filter) and (ep_filter is None or e in ep_filter)
    )
    if not keys:
        sys.exit("[error] no shared (task, episode) videos across the given folders")

    # default out-dir carries the layout/backbone so different selections with the SAME labels don't clobber.
    tag = "grid__" if grid else (f"{backbone}__" if backbone else "")
    out_dir = Path(out) if out else _HERE / "outputs" / (tag + "_vs_".join(labels))
    out_dir.mkdir(parents=True, exist_ok=True)
    bar_h = even(max(20, height // 9))
    font = load_font(int(bar_h * 0.62))
    H = even(height)

    def pad_w(img, W):                                       # right-pad black so rows of unequal width vstack
        if img.shape[1] >= W:
            return img
        return np.hstack([img, np.zeros((img.shape[0], W - img.shape[1], 3), dtype=img.dtype)])

    print(f"\n{len(keys)} clips → {out_dir}")
    for tid, ep in keys:
        # locate each folder's mp4 (task dir name carries the full target_task prefix), grouped by row
        rows_fls, out_fps, miss = [], None, False
        for _bb, fs in rows:
            fls = []
            for folder in fs:
                taskdir = next((d for d in (folder / "videos").glob(f"*_{tid}") if d.is_dir()), None)
                mp4 = taskdir / f"eval_episode_{ep}.mp4" if taskdir else None
                if not mp4 or not mp4.exists():
                    miss = True
                    break
                frames, src_fps = read_video(mp4)
                fls.append(frames)
                out_fps = out_fps or (fps or src_fps)
            if miss:
                break
            rows_fls.append(fls)
        if miss:
            continue

        n = max(len(fl) for fls in rows_fls for fl in fls)
        # per-(row,col) label bars: column label shared across rows; leftmost panel of each row carries
        # the backbone tag (so dino vs siglip rows are identifiable while column labels overlap).
        rows_bars = []
        for (bb, _), fls in zip(rows, rows_fls):
            bars = []
            for ci, fl in enumerate(fls):
                h0, w0 = fl[0].shape[:2]
                Wp = even(max(2, round(w0 * H / h0)))
                lab = f"{bb} | {labels[ci]}" if (grid and ci == 0 and bb) else labels[ci]
                bars.append(label_bar(Wp, bar_h, lab, font))
            rows_bars.append(bars)

        writer = imageio.get_writer(
            str(out_dir / f"task{tid}_ep{ep}.mp4"), fps=out_fps,
            codec="libx264", quality=8, macro_block_size=None,
        )
        for i in range(n):
            row_imgs = []
            for fls, bars in zip(rows_fls, rows_bars):
                cols = []
                for fl, bar in zip(fls, bars):
                    fr = fl[i] if i < len(fl) else fl[-1]    # freeze last frame to pad
                    cols.append(make_panel(fr, H, bar))
                row_imgs.append(np.hstack(cols))
            if len(row_imgs) > 1:
                maxw = max(r.shape[1] for r in row_imgs)
                row_imgs = [pad_w(r, maxw) for r in row_imgs]
            frame = np.vstack(row_imgs) if len(row_imgs) > 1 else row_imgs[0]
            if frame.shape[1] % 2:                           # yuv420p needs even width
                frame = frame[:, :-1]
            if frame.shape[0] % 2:                           # ...and even height (vstacked rows may be odd)
                frame = frame[:-1, :]
            writer.append_data(frame)
        writer.close()
        npan = sum(len(fls) for fls in rows_fls)
        print(f"  task{tid}_ep{ep}.mp4  ({n} frames, {npan} panels, {len(rows)} row(s))")

    print("\nDone.")


if __name__ == "__main__":
    main()
