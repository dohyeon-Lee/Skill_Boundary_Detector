"""Visualize CLIPSeg text-conditioned grounding on skill START frames.

Give a task range + episode count; for the matching skills this reads each skill's
START frame (same episode-relative → file mapping as the FSQ pipeline), runs CLIPSeg
with the episode language instruction (raw, as-is) and saves a side-by-side PNG per
skill: [start frame | CLIPSeg heat | overlay] into ./outputs/.

Paths (skillset / raw dataset) default to the sibling train_skills_config.yaml, so
normally you only pass --task_ids and --n_episodes.

Usage:
    python eval_CLIPseg.py --task_ids 0-4 --n_episodes 2
    python eval_CLIPseg.py --task_ids 0,3,7 --n_episodes 1 --max_skills 30
    python eval_CLIPseg.py --task_ids 0 --n_episodes 1 --prompt "the book"   # test a phrase
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

# ── locate the libero scripts + the train_skills config helpers ──
_HERE = Path(__file__).resolve()
_LIBERO_DIR = _HERE.parents[3]              # .../lerobot/examples/libero
_TRAIN_SKILLS_SRC = _HERE.parents[1] / "src"
sys.path.insert(0, str(_LIBERO_DIR))
sys.path.insert(0, str(_TRAIN_SKILLS_SRC))

from precompute_dino_features import (  # noqa: E402  (shared video / meta / frame helpers)
    _episode_language_map,
    _load_episodes_meta,
    _read_file_start_frames,
    _resolve_image_key,
    _task_language_map,
    _video_path,
    load_skill_metadata,
)
from train_skills_config import load_config, train_settings  # noqa: E402

DEFAULT_CONFIG = _HERE.parents[1] / "train_skills_config.yaml"


def _normalize_underscore_cli_flags() -> None:
    """Allow both --task_ids and tyro's default --task-ids style."""
    aliases = {
        "--task_ids": "--task-ids",
        "--n_episodes": "--n-episodes",
        "--max_skills": "--max-skills",
        "--output_dir": "--output-dir",
        "--overlay_alpha": "--overlay-alpha",
        "--skills_dir": "--skills-dir",
        "--dataset_dir": "--dataset-dir",
        "--clipseg_checkpoint": "--clipseg-checkpoint",
        "--image_key": "--image-key",
        "--batch_size": "--batch-size",
    }
    sys.argv = [aliases.get(arg, arg) for arg in sys.argv]


@dataclass
class Args:
    task_ids: str = ""
    """Task range/list to visualize: '0-4' | '0,3,7' | '' = all tasks."""
    n_episodes: int = 2
    """Episodes per task to include (the first N episode_ids of each task)."""
    max_skills: int = 0
    """Cap on total skills visualized (0 = no cap)."""
    config: Path = DEFAULT_CONFIG
    output_dir: str = ""
    """PNG output dir (default: this file's outputs/)."""
    prompt: str = ""
    """Override the text prompt for ALL skills (e.g. 'the book'); empty = episode instruction."""
    overlay_alpha: float = 0.5
    device: str = "cuda"
    # path overrides (default: read from train_skills_config.yaml)
    skills_dir: str = ""
    dataset_dir: str = ""
    clipseg_checkpoint: str = "/data2/dohyeon/SBD/models/clipseg-rd64-refined"
    image_key: str = "observation.images.image"
    batch_size: int = 64


def _parse_task_ids(spec: str) -> set[int] | None:
    spec = spec.strip()
    if not spec:
        return None  # all
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def _clipseg_maps(ckpt, frames, texts, device, batch_size):
    """Per-image CLIPSeg text-conditioned segmentation probability map (H, W) in [0,1]."""
    from PIL import Image  # noqa: PLC0415
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor  # noqa: PLC0415
    proc = CLIPSegProcessor.from_pretrained(ckpt)
    model = CLIPSegForImageSegmentation.from_pretrained(ckpt).to(device).eval()
    out = []
    for b0 in range(0, len(frames), batch_size):
        fb = [Image.fromarray(f) for f in frames[b0:b0 + batch_size]]
        tb = [t or "object" for t in texts[b0:b0 + batch_size]]
        inputs = proc(text=tb, images=fb, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits          # (B, H, W) or (H, W) for B==1
        if logits.ndim == 2:
            logits = logits[None]
        masks = torch.sigmoid(logits).float().cpu().numpy()
        out.extend(list(masks))
    return out


def main(args: Args) -> None:
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import textwrap  # noqa: PLC0415

    settings = train_settings(load_config(args.config))
    skills_dir = Path(args.skills_dir) if args.skills_dir else Path(settings["skillset_dir"]) / "skills"
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path(settings["raw_dataset_dir"])
    out_dir = Path(args.output_dir) if args.output_dir else _HERE.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    meta = load_skill_metadata(skills_dir)
    episodes_meta = _load_episodes_meta(dataset_dir)
    image_key = _resolve_image_key(episodes_meta, args.image_key)
    ep_lang = _episode_language_map(episodes_meta)
    task_lang = _task_language_map(dataset_dir)

    def language_for(m: dict) -> str:
        return ep_lang.get(int(m["episode_id"])) or task_lang.get(int(m["task_id"]), "") or ""

    # ── select skills: tasks in range, first n_episodes episodes per task ──
    want_tasks = _parse_task_ids(args.task_ids)
    by_task_ep: dict[int, dict[int, list[int]]] = {}
    for i, m in enumerate(meta):
        t = int(m["task_id"])
        if want_tasks is not None and t not in want_tasks:
            continue
        by_task_ep.setdefault(t, {}).setdefault(int(m["episode_id"]), []).append(i)

    selected: list[int] = []
    for t in sorted(by_task_ep):
        for ep in sorted(by_task_ep[t])[: args.n_episodes]:
            selected.extend(by_task_ep[t][ep])
    selected.sort()
    if args.max_skills > 0:
        selected = selected[: args.max_skills]
    if not selected:
        raise SystemExit(f"No skills matched task_ids={args.task_ids!r} in {skills_dir}")
    print(f"[eval] {len(selected)} skills selected "
          f"(tasks={sorted(by_task_ep)}, n_episodes={args.n_episodes})")

    # ── start frame per selected skill (same correct mapping as the FSQ pipeline) ──
    import json  # noqa: PLC0415
    from collections import defaultdict  # noqa: PLC0415
    fps = float(json.loads((dataset_dir / "meta" / "info.json").read_text())["fps"])
    from_ts_col = f"videos/{image_key}/from_timestamp"
    ep_from = {int(r["episode_index"]): float(r[from_ts_col]) for _, r in episodes_meta.iterrows()}
    by_file: dict[Path, list[tuple[int, int]]] = defaultdict(list)
    for i in selected:
        ep = int(meta[i]["episode_id"])
        path = _video_path(dataset_dir, episodes_meta, ep, image_key)
        abs_frame = int(round(ep_from[ep] * fps)) + int(meta[i]["frame_start"])
        by_file[path].append((i, abs_frame))
    frames: dict[int, np.ndarray] = {}
    for path, targets in by_file.items():
        for si, fr in _read_file_start_frames(path, targets, fps):
            frames[si] = fr
    print(f"[eval] decoded {len(frames)} start frames from {len(by_file)} video files")

    # ── CLIPSeg heatmaps (raw episode instruction unless --prompt overrides) ──
    order = [i for i in selected if i in frames]
    imgs = [frames[i] for i in order]
    texts = [args.prompt or language_for(meta[i]) for i in order]
    print(f"[eval] computing CLIPSeg maps for {len(order)} skills ...", flush=True)
    maps = _clipseg_maps(args.clipseg_checkpoint, imgs, texts, device, args.batch_size)

    # ── render: original | CLIPSeg heat | overlay ──
    for k, i in enumerate(order):
        img = frames[i]
        m = meta[i]
        h2d = maps[k]
        vmin, vmax = float(h2d.min()), float(h2d.max())
        H, W = img.shape[:2]
        lang = texts[k]  # the prompt actually fed to CLIPSeg
        ids = (f"task{int(m['task_id'])} · ep{int(m['episode_id'])} · "
               f"skill{int(m['skill_index'])} · frame{int(m['frame_start'])}")

        fig, ax = plt.subplots(1, 3, figsize=(13, 4.9))
        ax[0].imshow(img); ax[0].set_title(ids, fontsize=9); ax[0].axis("off")
        im = ax[1].imshow(h2d, cmap="jet", vmin=vmin, vmax=vmax, interpolation="bilinear")
        ax[1].set_title(f"CLIPSeg heat [{vmin:.3f},{vmax:.3f}]", fontsize=9); ax[1].axis("off")
        fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
        ax[2].imshow(img)
        ax[2].imshow(h2d, cmap="jet", vmin=vmin, vmax=vmax, alpha=args.overlay_alpha,
                     extent=[0, W, H, 0], interpolation="bilinear")
        ax[2].set_title(f"overlay (α={args.overlay_alpha})", fontsize=9); ax[2].axis("off")
        headline = "\n".join(textwrap.wrap(lang, width=110)) if lang else "(no language)"
        n_lines = headline.count("\n") + 1
        fig.suptitle(headline, fontsize=14, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95 - 0.04 * (n_lines - 1)))
        name = f"task{int(m['task_id']):02d}_ep{int(m['episode_id']):05d}_skill{int(m['skill_index']):02d}.png"
        fig.savefig(out_dir / name, dpi=110)
        plt.close(fig)

    print(f"[eval] saved {len(order)} PNGs → {out_dir}")


if __name__ == "__main__":
    _normalize_underscore_cli_flags()
    main(tyro.cli(Args))
