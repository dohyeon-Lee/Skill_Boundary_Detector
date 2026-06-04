"""Visualize saved SAM2 patch flags on skill START frames.

This reads the FSQ input dataset from train_skills_config.yaml, selects skills by
task range and episode count, and saves PNGs showing:

  [start frame | SAM2 object proposals on start frame | saved patch flag overlay]

The patch flags come from FSQ_inputs/patch_flags.npz by default. The SAM2 object
proposal panel re-runs SAM2 AMG on the skill start image for visualization only;
it does not change saved FSQ data.

Usage:
    python eval_SAM2_flags.py --task_ids 0-4 --n_episodes 2
    python eval_SAM2_flags.py --task_ids 0,3,7 --n_episodes 1 --max_skills 30
    python eval_SAM2_flags.py --task_ids 0 --n_episodes 1 --skip_sam2_objects
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
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

from precompute_dino_features import (  # noqa: E402
    _episode_language_map,
    _load_episodes_meta,
    _read_file_start_frames,
    _resolve_image_key,
    _task_language_map,
    _video_path,
    load_skill_metadata,
)
from precompute_sam2_masks import run_amg, segment_gripper  # noqa: E402
from train_skills_config import load_config, train_settings  # noqa: E402

DEFAULT_CONFIG = _HERE.parents[1] / "train_skills_config.yaml"


def _normalize_underscore_cli_flags() -> None:
    """Allow both --task_ids and tyro's default --task-ids style."""
    aliases = {
        "--task_ids": "--task-ids",
        "--n_episodes": "--n-episodes",
        "--max_skills": "--max-skills",
        "--output_dir": "--output-dir",
        "--skills_dir": "--skills-dir",
        "--dataset_dir": "--dataset-dir",
        "--patch_flags_path": "--patch-flags-path",
        "--image_key": "--image-key",
        "--patch_grid": "--patch-grid",
        "--overlay_alpha": "--overlay-alpha",
        "--skip_sam2_objects": "--skip-sam2-objects",
        "--sam2_checkpoint": "--sam2-checkpoint",
        "--sam2_config": "--sam2-config",
        "--points_per_side": "--points-per-side",
        "--max_mask_area_ratio": "--max-mask-area-ratio",
        "--min_mask_area_ratio": "--min-mask-area-ratio",
        "--min_stability_score": "--min-stability-score",
        "--gripper_point": "--gripper-point",
        "--gripper_iou_threshold": "--gripper-iou-threshold",
    }
    sys.argv = [aliases.get(arg, arg) for arg in sys.argv]


@dataclass
class Args:
    task_ids: str = ""
    """Task range/list to visualize: '0-4' | '0,3,7' | '' = all tasks."""
    n_episodes: int = 2
    """Episodes per task to include (first N episode_ids of each task)."""
    max_skills: int = 0
    """Cap on total skills visualized (0 = no cap)."""
    config: Path = DEFAULT_CONFIG
    output_dir: str = ""
    """PNG output dir (default: this file's outputs/sam2_flags)."""
    skills_dir: str = ""
    dataset_dir: str = ""
    patch_flags_path: str = ""
    """Merged patch_flags.npz or per-skill sam2_masks dir. Empty = config default."""
    image_key: str = "observation.images.image"
    patch_grid: int = 0
    """Patch grid size. 0 = infer from config / patch_flags."""
    overlay_alpha: float = 0.48
    device: str = "cuda"

    # SAM2 object proposal visualization on the start frame.
    skip_sam2_objects: bool = False
    sam2_checkpoint: str = ""
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    points_per_side: int = 50
    max_mask_area_ratio: float = 0.35
    min_mask_area_ratio: float = 0.001
    min_stability_score: float = 0.85
    gripper_point: tuple[int, int] = (128, 45)
    gripper_iou_threshold: float = 0.3


def _parse_task_ids(spec: str) -> set[int] | None:
    spec = spec.strip()
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def _mask_path(root: Path, meta: dict) -> Path:
    return (
        root
        / f"task{int(meta['task_id'])}"
        / f"ep{int(meta['episode_id']):05d}_skill{int(meta['skill_index']):03d}.npz"
    )


def _load_start_flags(flags_path: Path, metadata: list[dict], n_patches: int) -> list[np.ndarray]:
    """Return one (n_patches, 2) start-frame flag array per skill."""
    if flags_path.is_file():
        data = np.load(str(flags_path))
        flags = data["patch_flags"].astype(np.float32)
        offsets = data["offsets"].astype(np.int64)
        if len(offsets) != len(metadata) + 1:
            raise ValueError(f"{flags_path} offsets count does not match skill metadata.")
        out = []
        for i, meta in enumerate(metadata):
            expected_len = int(meta["length"])
            clip = flags[offsets[i]:offsets[i + 1]]
            if len(clip) != expected_len:
                raise ValueError(
                    f"{flags_path} length mismatch at skill {i}: got {len(clip)}, expected {expected_len}"
                )
            out.append(clip[0] if len(clip) else np.zeros((n_patches, 2), dtype=np.float32))
        return out

    if flags_path.is_dir():
        out = []
        for meta in metadata:
            path = _mask_path(flags_path, meta)
            if path.exists():
                data = np.load(str(path))
                patch_masks = data["patch_masks"]
                if len(patch_masks):
                    out.append(patch_masks[0].astype(np.float32).reshape(-1, 2))
                    continue
            out.append(np.zeros((n_patches, 2), dtype=np.float32))
        return out

    raise FileNotFoundError(f"Patch flags path not found: {flags_path}")


def _flag_overlay_image(frame: np.ndarray, flags: np.ndarray, grid: int, alpha: float) -> np.ndarray:
    from PIL import Image  # noqa: PLC0415

    h, w = frame.shape[:2]
    f = np.asarray(flags, dtype=np.float32).reshape(grid, grid, 2)
    rgba = np.zeros((grid, grid, 4), dtype=np.float32)
    rgba[..., 0] = np.clip(f[..., 0], 0.0, 1.0)
    rgba[..., 1] = np.clip(f[..., 1], 0.0, 1.0)
    rgba[..., 3] = alpha * (f.max(axis=-1) > 0)
    return np.asarray(Image.fromarray((rgba * 255).astype(np.uint8)).resize((w, h), Image.NEAREST))


def _draw_grid(ax, grid: int, h: int, w: int) -> None:
    for x in np.linspace(0, w, grid + 1):
        ax.axvline(x, color="white", lw=0.35, alpha=0.45)
    for y in np.linspace(0, h, grid + 1):
        ax.axhline(y, color="white", lw=0.35, alpha=0.45)


def _build_sam2_amg(checkpoint: Path, config: str, points_per_side: int, device: str):
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # noqa: PLC0415
    from sam2.build_sam import build_sam2  # noqa: PLC0415
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: PLC0415

    model = build_sam2(config, str(checkpoint), device=device)
    amg = SAM2AutomaticMaskGenerator(model, points_per_side=points_per_side)
    img_pred = SAM2ImagePredictor(model)
    return amg, img_pred


def _sam2_object_overlay(
    frame: np.ndarray,
    amg,
    img_pred,
    args: Args,
) -> tuple[np.ndarray, int]:
    masks = None
    gripper_mask, arm_mask = segment_gripper(img_pred, frame, args.gripper_point)
    with torch.inference_mode():
        masks = run_amg(
            amg,
            frame,
            max_area=args.max_mask_area_ratio,
            min_area=args.min_mask_area_ratio,
            min_score=args.min_stability_score,
            gripper_mask=gripper_mask,
            arm_mask=arm_mask,
            iou_thr=args.gripper_iou_threshold,
        )

    h, w = frame.shape[:2]
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    if len(masks) == 0:
        return overlay, 0
    rng = np.random.default_rng(0)
    colors = rng.uniform(0.15, 1.0, size=(len(masks), 3)).astype(np.float32)
    for mask, color in zip(masks, colors, strict=False):
        overlay[mask, :3] = 0.55 * overlay[mask, :3] + 0.45 * color
        overlay[mask, 3] = np.maximum(overlay[mask, 3], 0.48)
    return overlay, len(masks)


def main(args: Args) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import textwrap  # noqa: PLC0415

    settings = train_settings(load_config(args.config))
    skills_dir = Path(args.skills_dir) if args.skills_dir else Path(settings["skillset_dir"]) / "skills"
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path(settings["raw_dataset_dir"])
    flags_path = Path(args.patch_flags_path) if args.patch_flags_path else Path(settings["sam2_flags_path"])
    out_dir = Path(args.output_dir) if args.output_dir else _HERE.parent / "outputs" / "sam2_flags"
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_skill_metadata(skills_dir)
    episodes_meta = _load_episodes_meta(dataset_dir)
    image_key = _resolve_image_key(episodes_meta, args.image_key)
    ep_lang = _episode_language_map(episodes_meta)
    task_lang = _task_language_map(dataset_dir)

    patch_grid = args.patch_grid or int(settings["dino_patch_grid"])
    n_patches = patch_grid * patch_grid
    start_flags = _load_start_flags(flags_path, metadata, n_patches)
    if any(f.shape != (n_patches, 2) for f in start_flags):
        bad = next(f.shape for f in start_flags if f.shape != (n_patches, 2))
        raise ValueError(f"Expected start flags {(n_patches, 2)}, got {bad}")

    def language_for(meta: dict) -> str:
        return ep_lang.get(int(meta["episode_id"])) or task_lang.get(int(meta["task_id"]), "") or ""

    want_tasks = _parse_task_ids(args.task_ids)
    by_task_ep: dict[int, dict[int, list[int]]] = {}
    for i, meta in enumerate(metadata):
        task_id = int(meta["task_id"])
        if want_tasks is not None and task_id not in want_tasks:
            continue
        by_task_ep.setdefault(task_id, {}).setdefault(int(meta["episode_id"]), []).append(i)

    selected: list[int] = []
    for task_id in sorted(by_task_ep):
        for ep_id in sorted(by_task_ep[task_id])[: args.n_episodes]:
            selected.extend(by_task_ep[task_id][ep_id])
    selected.sort()
    if args.max_skills > 0:
        selected = selected[: args.max_skills]
    if not selected:
        raise SystemExit(f"No skills matched task_ids={args.task_ids!r} in {skills_dir}")
    print(f"[eval] selected {len(selected)} skills from tasks={sorted(by_task_ep)}")

    fps = float(json.loads((dataset_dir / "meta" / "info.json").read_text())["fps"])
    from_ts_col = f"videos/{image_key}/from_timestamp"
    ep_from = {int(row["episode_index"]): float(row[from_ts_col]) for _, row in episodes_meta.iterrows()}
    by_file: dict[Path, list[tuple[int, int]]] = defaultdict(list)
    for i in selected:
        ep_id = int(metadata[i]["episode_id"])
        path = _video_path(dataset_dir, episodes_meta, ep_id, image_key)
        abs_frame = int(round(ep_from[ep_id] * fps)) + int(metadata[i]["frame_start"])
        by_file[path].append((i, abs_frame))

    frames: dict[int, np.ndarray] = {}
    for path, targets in by_file.items():
        for skill_idx, frame in _read_file_start_frames(path, targets, fps):
            frames[skill_idx] = frame
    order = [i for i in selected if i in frames]
    print(f"[eval] decoded {len(order)} start frames from {len(by_file)} video files")

    amg = img_pred = None
    device = args.device if torch.cuda.is_available() else "cpu"
    if not args.skip_sam2_objects:
        ckpt = Path(args.sam2_checkpoint) if args.sam2_checkpoint else Path(settings["sam2_checkpoint"])
        if not ckpt.exists():
            print(f"[eval] SAM2 checkpoint not found, object proposal panel will be empty: {ckpt}")
        else:
            print(f"[eval] loading SAM2 object proposal model: {ckpt}")
            cwd = Path.cwd()
            try:
                os.chdir(str(settings["lerobot_root"]))
                amg, img_pred = _build_sam2_amg(ckpt, args.sam2_config, args.points_per_side, device)
            finally:
                os.chdir(str(cwd))

    for k, i in enumerate(order):
        frame = frames[i]
        meta = metadata[i]
        h, w = frame.shape[:2]
        flags = start_flags[i]
        n_changed = int(flags[:, 0].sum())
        n_green = int(flags[:, 1].sum())
        flag_rgba = _flag_overlay_image(frame, flags, patch_grid, args.overlay_alpha)

        obj_overlay = np.zeros((h, w, 4), dtype=np.float32)
        n_objects = 0
        if amg is not None and img_pred is not None:
            obj_overlay, n_objects = _sam2_object_overlay(frame, amg, img_pred, args)

        ids = (
            f"task{int(meta['task_id'])} | ep{int(meta['episode_id'])} | "
            f"skill{int(meta['skill_index'])} | start frame {int(meta['frame_start'])}"
        )
        lang = language_for(meta)

        fig, ax = plt.subplots(1, 3, figsize=(14, 5.1))
        ax[0].imshow(frame)
        ax[0].set_title(ids, fontsize=9)
        ax[0].axis("off")

        ax[1].imshow(frame)
        if n_objects:
            ax[1].imshow(obj_overlay)
        ax[1].set_title(f"SAM2 object proposals ({n_objects})", fontsize=9)
        ax[1].axis("off")

        ax[2].imshow(frame)
        ax[2].imshow(flag_rgba)
        _draw_grid(ax[2], patch_grid, h, w)
        ax[2].set_title(f"start patch flags: changed={n_changed}, green={n_green}", fontsize=9)
        ax[2].axis("off")

        legend = "red = changed | green = green | yellow = both"
        headline = f"{legend}\n{lang}" if lang else legend
        headline = "\n".join(textwrap.wrap(headline, width=120))
        n_lines = headline.count("\n") + 1
        fig.suptitle(headline, fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94 - 0.035 * (n_lines - 1)))

        name = f"task{int(meta['task_id']):02d}_ep{int(meta['episode_id']):05d}_skill{int(meta['skill_index']):02d}.png"
        fig.savefig(out_dir / name, dpi=120)
        plt.close(fig)
        if (k + 1) % 10 == 0 or k + 1 == len(order):
            print(f"[eval] {k + 1}/{len(order)}")

    print(f"[eval] saved {len(order)} PNGs -> {out_dir}")


if __name__ == "__main__":
    _normalize_underscore_cli_flags()
    main(tyro.cli(Args))
