"""Visualize SAM3 concept (text) grounding on skill START frames.

SAM3 does Promptable Concept Segmentation: a short object phrase ("book", "basket")
→ instance masks of that concept. This matches the "feed only the object, not the full
instruction" idea. For each matching skill this reads the START frame (same mapping as
the FSQ pipeline), runs SAM3 with an OBJECT phrase, and saves a side-by-side PNG:
[start frame | SAM3 heat | overlay] into ./outputs/.

Object phrase per skill defaults to the noun extracted from the instruction
(--object_index 0 = the grasp target, 1 = destination, ...); override all with --prompt.

Usage:
    python eval_SAM3.py --task_ids 0-4 --n_episodes 2                  # grasp-object noun
    python eval_SAM3.py --task_ids 0 --n_episodes 1 --object_index 1   # destination noun
    python eval_SAM3.py --task_ids 0 --n_episodes 1 --prompt "book"    # fixed phrase
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

_HERE = Path(__file__).resolve()
_LIBERO_DIR = _HERE.parents[3]
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
        "--object_index": "--object-index",
        "--mask_threshold": "--mask-threshold",
        "--overlay_alpha": "--overlay-alpha",
        "--skills_dir": "--skills-dir",
        "--dataset_dir": "--dataset-dir",
        "--sam3_checkpoint": "--sam3-checkpoint",
        "--image_key": "--image-key",
    }
    sys.argv = [aliases.get(arg, arg) for arg in sys.argv]

# object noun phrases = "the/a/an <words>" up to a connective/stopword (zero-shot, no training)
_STOP = r"(?:\s+(?:and|on|onto|in|into|to|of|it|that|so|then|next|from|with|up|off|under)\b|[.,]|$)"


def _object_phrases(instruction: str) -> list[str]:
    out: list[str] = []
    for m in re.finditer(r"\b(?:the|a|an)\s+(.+?)" + _STOP, instruction):
        p = m.group(1).strip()
        if p and p not in out:
            out.append(p)
    return out


@dataclass
class Args:
    task_ids: str = ""
    """Task range/list: '0-4' | '0,3,7' | '' = all."""
    n_episodes: int = 2
    max_skills: int = 0
    config: Path = DEFAULT_CONFIG
    output_dir: str = ""
    prompt: str = ""
    """Override the object phrase for ALL skills (e.g. 'book'); empty = extract from instruction."""
    object_index: int = 0
    """Which extracted object noun to use (0 = grasp target, 1 = destination, ...)."""
    threshold: float = 0.3
    """SAM3 instance score threshold."""
    mask_threshold: float = 0.5
    overlay_alpha: float = 0.5
    device: str = "cuda"
    skills_dir: str = ""
    dataset_dir: str = ""
    sam3_checkpoint: str = "/data2/dohyeon/SBD/models/sam3"
    image_key: str = "observation.images.image"


def _parse_task_ids(spec: str) -> set[int] | None:
    spec = spec.strip()
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def main(args: Args) -> None:
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import textwrap  # noqa: PLC0415
    from transformers import Sam3Model, Sam3Processor  # noqa: PLC0415

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

    def instruction_for(m: dict) -> str:
        return ep_lang.get(int(m["episode_id"])) or task_lang.get(int(m["task_id"]), "") or ""

    def object_for(m: dict) -> str:
        if args.prompt:
            return args.prompt
        objs = _object_phrases(instruction_for(m))
        if not objs:
            return "object"
        return objs[min(args.object_index, len(objs) - 1)]

    # ── select skills ──
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
    print(f"[eval] {len(selected)} skills selected (tasks={sorted(by_task_ep)}, n_episodes={args.n_episodes})")

    # ── start frames (same correct mapping as the FSQ pipeline) ──
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

    # ── SAM3 concept segmentation (per skill: object phrase) ──
    from PIL import Image  # noqa: PLC0415
    proc = Sam3Processor.from_pretrained(args.sam3_checkpoint)
    model = Sam3Model.from_pretrained(args.sam3_checkpoint).to(device).eval()

    order = [i for i in selected if i in frames]
    print(f"[eval] running SAM3 for {len(order)} skills ...", flush=True)
    for k, i in enumerate(order):
        img = frames[i]
        m = meta[i]
        H, W = img.shape[:2]
        phrase = object_for(m)
        inputs = proc(images=Image.fromarray(img), text=phrase, return_tensors="pt")
        inputs = {kk: vv.to(device) for kk, vv in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        res = proc.post_process_instance_segmentation(
            outputs, threshold=args.threshold, mask_threshold=args.mask_threshold,
            target_sizes=[(H, W)],
        )[0]
        masks = res["masks"]
        scores = res["scores"]
        n_inst = 0 if masks is None else len(masks)
        heat = np.zeros((H, W), dtype=np.float32)
        if n_inst:
            mk = masks.float().cpu().numpy()                    # (N, H, W)
            sc = scores.float().cpu().numpy()                   # (N,)
            heat = (mk * sc[:, None, None]).max(0)              # soft: per-pixel best instance score
        top = float(scores.max()) if n_inst else 0.0

        ids = (f"task{int(m['task_id'])} · ep{int(m['episode_id'])} · "
               f"skill{int(m['skill_index'])} · frame{int(m['frame_start'])}")
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.9))
        ax[0].imshow(img); ax[0].set_title(ids, fontsize=9); ax[0].axis("off")
        im = ax[1].imshow(heat, cmap="jet", vmin=0.0, vmax=1.0)
        ax[1].set_title(f"SAM3 '{phrase}'  ({n_inst} inst, top {top:.2f})", fontsize=9); ax[1].axis("off")
        fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
        ax[2].imshow(img)
        ax[2].imshow(heat, cmap="jet", vmin=0.0, vmax=1.0, alpha=args.overlay_alpha,
                     extent=[0, W, H, 0])
        ax[2].set_title(f"overlay (α={args.overlay_alpha})", fontsize=9); ax[2].axis("off")
        head = f"'{phrase}'   ⟵   {instruction_for(m)}"
        headline = "\n".join(textwrap.wrap(head, width=110))
        n_lines = headline.count("\n") + 1
        fig.suptitle(headline, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95 - 0.04 * (n_lines - 1)))
        name = f"task{int(m['task_id']):02d}_ep{int(m['episode_id']):05d}_skill{int(m['skill_index']):02d}.png"
        fig.savefig(out_dir / name, dpi=110)
        plt.close(fig)
        if (k + 1) % 10 == 0 or k + 1 == len(order):
            print(f"[eval] {k + 1}/{len(order)}", flush=True)

    print(f"[eval] saved {len(order)} PNGs → {out_dir}")


if __name__ == "__main__":
    _normalize_underscore_cli_flags()
    main(tyro.cli(Args))
