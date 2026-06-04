"""Visualize SAM3 grounding for robot arm vs end-effector on skill START frames.

This is an inspection script for testing whether text-prompted SAM3 can replace
some hand-built SAM2 filters. It selects episodes/tasks from the FSQ skillset,
reads each selected skill's start image, runs SAM3 with two prompts, and saves:

  [start frame | arm heat | eef heat | combined overlay]

Usage:
    python eval_SAM3_robot_eef.py --task_ids 0 --n_episodes 1
    python eval_SAM3_robot_eef.py --task_ids 0-3 --episode_ids 0,1,2
    python eval_SAM3_robot_eef.py --task_ids 0 --n_episodes 1 --max_skills 20
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
from train_skills_config import load_config, train_settings  # noqa: E402

DEFAULT_CONFIG = _HERE.parents[1] / "train_skills_config.yaml"


def _normalize_underscore_cli_flags() -> None:
    """Allow both --task_ids and tyro's default --task-ids style."""
    aliases = {
        "--task_ids": "--task-ids",
        "--episode_ids": "--episode-ids",
        "--n_episodes": "--n-episodes",
        "--max_skills": "--max-skills",
        "--output_dir": "--output-dir",
        "--skills_dir": "--skills-dir",
        "--dataset_dir": "--dataset-dir",
        "--image_key": "--image-key",
        "--sam3_checkpoint": "--sam3-checkpoint",
        "--arm_prompt": "--arm-prompt",
        "--eef_prompt": "--eef-prompt",
        "--mask_threshold": "--mask-threshold",
        "--overlay_alpha": "--overlay-alpha",
        "--batch_size": "--batch-size",
    }
    sys.argv = [aliases.get(arg, arg) for arg in sys.argv]


@dataclass
class Args:
    task_ids: str = ""
    """Task range/list: '0-4' | '0,3,7' | '' = all tasks."""
    episode_ids: str = ""
    """Global episode_index range/list: '0-4' | '0,3,7' | '' = first n_episodes per task."""
    n_episodes: int = 1
    """Episodes per task when --episode_ids is empty."""
    max_skills: int = 0
    """Cap on total skills visualized (0 = no cap)."""
    config: Path = DEFAULT_CONFIG
    output_dir: str = ""
    """PNG output dir (default: this file's outputs/sam3_robot_eef)."""
    skills_dir: str = ""
    dataset_dir: str = ""
    image_key: str = "observation.images.image"
    sam3_checkpoint: str = "/data2/dohyeon/SBD/models/sam3"
    arm_prompt: str = "robot arm"
    eef_prompt: str = "robot gripper"
    threshold: float = 0.3
    """SAM3 instance score threshold."""
    mask_threshold: float = 0.5
    overlay_alpha: float = 0.55
    batch_size: int = 4
    device: str = "cuda"


def _parse_int_spec(spec: str) -> set[int] | None:
    spec = spec.strip()
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def _sam3_heatmaps(model, proc, frames, prompt: str, args: Args, device: str) -> tuple[list[np.ndarray], list[int], list[float]]:
    from PIL import Image  # noqa: PLC0415

    heats: list[np.ndarray] = []
    counts: list[int] = []
    top_scores: list[float] = []
    for b0 in range(0, len(frames), args.batch_size):
        batch = frames[b0:b0 + args.batch_size]
        images = [Image.fromarray(frame) for frame in batch]
        texts = [prompt] * len(images)
        inputs = proc(images=images, text=texts, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = [frame.shape[:2] for frame in batch]
        results = proc.post_process_instance_segmentation(
            outputs,
            threshold=args.threshold,
            mask_threshold=args.mask_threshold,
            target_sizes=target_sizes,
        )
        for frame, result in zip(batch, results, strict=False):
            h, w = frame.shape[:2]
            masks = result["masks"]
            scores = result["scores"]
            heat = np.zeros((h, w), dtype=np.float32)
            n_inst = 0 if masks is None else len(masks)
            if n_inst:
                mk = masks.float().cpu().numpy()
                sc = scores.float().cpu().numpy()
                heat = (mk * sc[:, None, None]).max(0)
                top_scores.append(float(sc.max()))
            else:
                top_scores.append(0.0)
            heats.append(heat)
            counts.append(n_inst)
    return heats, counts, top_scores


def _combined_rgba(arm: np.ndarray, eef: np.ndarray, alpha: float) -> np.ndarray:
    arm = np.clip(arm, 0.0, 1.0)
    eef = np.clip(eef, 0.0, 1.0)
    rgba = np.zeros((*arm.shape, 4), dtype=np.float32)
    rgba[..., 0] = arm
    rgba[..., 1] = eef
    rgba[..., 2] = 0.25 * eef
    rgba[..., 3] = alpha * np.maximum(arm, eef)
    return rgba


def main(args: Args) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import textwrap  # noqa: PLC0415
    from transformers import Sam3Model, Sam3Processor  # noqa: PLC0415

    settings = train_settings(load_config(args.config))
    skills_dir = Path(args.skills_dir) if args.skills_dir else Path(settings["skillset_dir"]) / "skills"
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path(settings["raw_dataset_dir"])
    out_dir = Path(args.output_dir) if args.output_dir else _HERE.parent / "outputs" / "sam3_robot_eef"
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_skill_metadata(skills_dir)
    episodes_meta = _load_episodes_meta(dataset_dir)
    image_key = _resolve_image_key(episodes_meta, args.image_key)
    ep_lang = _episode_language_map(episodes_meta)
    task_lang = _task_language_map(dataset_dir)

    want_tasks = _parse_int_spec(args.task_ids)
    want_eps = _parse_int_spec(args.episode_ids)
    by_task_ep: dict[int, dict[int, list[int]]] = {}
    for i, meta in enumerate(metadata):
        task_id = int(meta["task_id"])
        ep_id = int(meta["episode_id"])
        if want_tasks is not None and task_id not in want_tasks:
            continue
        if want_eps is not None and ep_id not in want_eps:
            continue
        by_task_ep.setdefault(task_id, {}).setdefault(ep_id, []).append(i)

    selected: list[int] = []
    for task_id in sorted(by_task_ep):
        ep_ids = sorted(by_task_ep[task_id])
        if want_eps is None:
            ep_ids = ep_ids[: args.n_episodes]
        for ep_id in ep_ids:
            selected.extend(by_task_ep[task_id][ep_id])
    selected.sort()
    if args.max_skills > 0:
        selected = selected[: args.max_skills]
    if not selected:
        raise SystemExit(
            f"No skills matched task_ids={args.task_ids!r}, episode_ids={args.episode_ids!r} in {skills_dir}"
        )
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

    frames_by_idx: dict[int, np.ndarray] = {}
    for path, targets in by_file.items():
        for skill_idx, frame in _read_file_start_frames(path, targets, fps):
            frames_by_idx[skill_idx] = frame
    order = [i for i in selected if i in frames_by_idx]
    frames = [frames_by_idx[i] for i in order]
    print(f"[eval] decoded {len(frames)} start frames from {len(by_file)} video files")

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[eval] loading SAM3: {args.sam3_checkpoint}")
    proc = Sam3Processor.from_pretrained(args.sam3_checkpoint)
    model = Sam3Model.from_pretrained(args.sam3_checkpoint).to(device).eval()

    print(f"[eval] running SAM3 prompt: {args.arm_prompt!r}")
    arm_maps, arm_counts, arm_scores = _sam3_heatmaps(model, proc, frames, args.arm_prompt, args, device)
    print(f"[eval] running SAM3 prompt: {args.eef_prompt!r}")
    eef_maps, eef_counts, eef_scores = _sam3_heatmaps(model, proc, frames, args.eef_prompt, args, device)

    def language_for(meta: dict) -> str:
        return ep_lang.get(int(meta["episode_id"])) or task_lang.get(int(meta["task_id"]), "") or ""

    for k, i in enumerate(order):
        frame = frames[k]
        meta = metadata[i]
        arm = arm_maps[k]
        eef = eef_maps[k]
        combo = _combined_rgba(arm, eef, args.overlay_alpha)
        lang = language_for(meta)
        ids = (
            f"task{int(meta['task_id'])} | ep{int(meta['episode_id'])} | "
            f"skill{int(meta['skill_index'])} | start frame {int(meta['frame_start'])}"
        )

        fig, ax = plt.subplots(1, 4, figsize=(17, 4.8))
        ax[0].imshow(frame)
        ax[0].set_title(ids, fontsize=9)
        ax[0].axis("off")

        im1 = ax[1].imshow(arm, cmap="Reds", vmin=0.0, vmax=1.0)
        ax[1].set_title(f"{args.arm_prompt} ({arm_counts[k]} inst, top {arm_scores[k]:.2f})", fontsize=9)
        ax[1].axis("off")
        fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

        im2 = ax[2].imshow(eef, cmap="Greens", vmin=0.0, vmax=1.0)
        ax[2].set_title(f"{args.eef_prompt} ({eef_counts[k]} inst, top {eef_scores[k]:.2f})", fontsize=9)
        ax[2].axis("off")
        fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

        ax[3].imshow(frame)
        ax[3].imshow(combo)
        ax[3].set_title("overlay: red=arm, green=eef", fontsize=9)
        ax[3].axis("off")

        headline = f"{lang}" if lang else "(no language)"
        headline = "\n".join(textwrap.wrap(headline, width=130))
        n_lines = headline.count("\n") + 1
        fig.suptitle(headline, fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94 - 0.035 * (n_lines - 1)))

        name = f"task{int(meta['task_id']):02d}_ep{int(meta['episode_id']):05d}_skill{int(meta['skill_index']):02d}.png"
        fig.savefig(out_dir / name, dpi=120)
        plt.close(fig)
        if (k + 1) % 10 == 0 or k + 1 == len(order):
            print(f"[eval] saved {k + 1}/{len(order)}")

    print(f"[eval] saved {len(order)} PNGs -> {out_dir}")


if __name__ == "__main__":
    _normalize_underscore_cli_flags()
    main(tyro.cli(Args))
