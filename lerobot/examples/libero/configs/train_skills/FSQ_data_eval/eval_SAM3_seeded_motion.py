"""Prototype SAM3-seeded object tracking and moved/occluded classification.

Per selected episode:
  1. Run SAM3 on the episode first frame with broad + object-specific prompts.
  2. Run SAM3 robot prompts on the same frame and subtract robot/eef regions.
  3. NMS the remaining instance masks and use them as SAM2 video seeds.
  4. Track those object ids through the episode with SAM2 video predictor.
  5. For each skill start/end, classify each object:

       if visible_ratio >= 0.5:
           moved if centroid_dist >= move_thr else unchanged
       elif robot_occlusion_score >= occ_thr:
           occluded
       else:
           uncertain

     Then keep at most one moved object per skill.

The visualization paints all object ids that were classified as moved at least once
in red on every selected skill start frame, even when the skill is not the exact
skill where the movement was detected. This is intentionally inspection-oriented.

Usage:
    python eval_SAM3_seeded_motion.py --task_ids 0 --n_episodes 1
    python eval_SAM3_seeded_motion.py --task_ids 0-20 --n_episodes 1 --max_skills 100
    python eval_SAM3_seeded_motion.py --task_ids 0 --episode_ids 0,1
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from PIL import Image as PILImage

_HERE = Path(__file__).resolve()
_LIBERO_DIR = _HERE.parents[3]
_TRAIN_SKILLS_SRC = _HERE.parents[1] / "src"
sys.path.insert(0, str(_LIBERO_DIR))
sys.path.insert(0, str(_TRAIN_SKILLS_SRC))

from precompute_dino_features import (  # noqa: E402
    _episode_language_map,
    _load_episodes_meta,
    _resolve_image_key,
    _task_language_map,
    _video_path,
    load_skill_metadata,
)
from precompute_sam2_masks import read_episode_frames  # noqa: E402
from train_skills_config import load_config, train_settings  # noqa: E402

DEFAULT_CONFIG = _HERE.parents[1] / "train_skills_config.yaml"


def _normalize_underscore_cli_flags() -> None:
    aliases = {
        "--task_ids": "--task-ids",
        "--episode_ids": "--episode-ids",
        "--n_episodes": "--n-episodes",
        "--max_skills": "--max-skills",
        "--output_dir": "--output-dir",
        "--skills_dir": "--skills-dir",
        "--dataset_dir": "--dataset-dir",
        "--image_key": "--image-key",
        "--sam2_checkpoint": "--sam2-checkpoint",
        "--sam2_config": "--sam2-config",
        "--sam3_checkpoint": "--sam3-checkpoint",
        "--object_prompts": "--object-prompts",
        "--extra_object_prompts": "--extra-object-prompts",
        "--use_language_objects": "--use-language-objects",
        "--arm_prompt": "--arm-prompt",
        "--eef_prompt": "--eef-prompt",
        "--exclude_eef": "--exclude-eef",
        "--sam3_threshold": "--sam3-threshold",
        "--sam3_mask_threshold": "--sam3-mask-threshold",
        "--seed_min_area_ratio": "--seed-min-area-ratio",
        "--seed_max_area_ratio": "--seed-max-area-ratio",
        "--seed_workspace_y_min_ratio": "--seed-workspace-y-min-ratio",
        "--seed_workspace_y_max_ratio": "--seed-workspace-y-max-ratio",
        "--seed_min_workspace_overlap": "--seed-min-workspace-overlap",
        "--seed_require_centroid_in_workspace": "--seed-require-centroid-in-workspace",
        "--seed_nms_iou": "--seed-nms-iou",
        "--max_seed_objects": "--max-seed-objects",
        "--visible_threshold": "--visible-threshold",
        "--move_thr": "--move-thr",
        "--occ_thr": "--occ-thr",
        "--overlay_alpha": "--overlay-alpha",
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
    """Cap on total visualized skills after episode selection (0 = no cap)."""
    config: Path = DEFAULT_CONFIG
    output_dir: str = ""
    skills_dir: str = ""
    dataset_dir: str = ""
    image_key: str = "observation.images.image"

    sam2_checkpoint: str = ""
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam3_checkpoint: str = "/data2/dohyeon/SBD/models/sam3"
    device: str = "cuda"

    object_prompts: str = "object|item|foreground object|movable object|scene object"
    """'|' separated SAM3 object prompts unioned on the episode first frame."""
    extra_object_prompts: str = ""
    """Optional '|' separated debugging prompts, kept off by default for scalability."""
    use_language_objects: bool = False
    """Append noun phrases extracted from selected episode language instructions."""
    arm_prompt: str = "robot arm"
    eef_prompt: str = "robot gripper"
    exclude_eef: bool = True

    sam3_threshold: float = 0.3
    sam3_mask_threshold: float = 0.5
    seed_min_area_ratio: float = 0.003
    seed_max_area_ratio: float = 0.45
    seed_workspace_y_min_ratio: float = 0.30
    """Only keep seed masks whose pixels overlap this lower image region."""
    seed_workspace_y_max_ratio: float = 1.00
    seed_min_workspace_overlap: float = 0.40
    """Minimum fraction of a seed mask that must lie inside the workspace ROI."""
    seed_require_centroid_in_workspace: bool = True
    """Reject first-frame seed masks whose centroid is outside the workspace ROI."""
    seed_nms_iou: float = 0.65
    max_seed_objects: int = 40

    visible_threshold: float = 0.5
    move_thr: float = 30.0
    occ_thr: float = 0.25
    overlay_alpha: float = 0.55


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


def _object_phrases(instruction: str) -> list[str]:
    import re  # noqa: PLC0415

    stop = r"(?:\s+(?:and|on|onto|in|into|to|of|it|that|so|then|next|from|with|up|off|under)\b|[.,]|$)"
    out: list[str] = []
    for match in re.finditer(r"\b(?:the|a|an)\s+(.+?)" + stop, instruction):
        phrase = match.group(1).strip()
        if phrase and phrase not in out:
            out.append(phrase)
    return out


def _iou(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter / union) if union > 0 else 0.0


def _centroid(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return np.array([ys.mean(), xs.mean()], dtype=np.float32)


def _sam3_instances(model, proc, frame: np.ndarray, prompts: list[str], args: Args, device: str) -> list[dict]:
    instances: list[dict] = []
    image = PILImage.fromarray(frame)
    h, w = frame.shape[:2]
    for prompt in prompts:
        inputs = proc(images=image, text=prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        result = proc.post_process_instance_segmentation(
            outputs,
            threshold=args.sam3_threshold,
            mask_threshold=args.sam3_mask_threshold,
            target_sizes=[(h, w)],
        )[0]
        masks = result["masks"]
        scores = result["scores"]
        if masks is None or len(masks) == 0:
            continue
        masks_np = masks.bool().cpu().numpy()
        scores_np = scores.float().cpu().numpy()
        for mask, score in zip(masks_np, scores_np, strict=False):
            instances.append({"mask": mask.astype(bool), "score": float(score), "prompt": prompt})
    return instances


def _sam3_union(model, proc, frame: np.ndarray, prompts: list[str], args: Args, device: str) -> np.ndarray:
    h, w = frame.shape[:2]
    union = np.zeros((h, w), dtype=bool)
    for inst in _sam3_instances(model, proc, frame, prompts, args, device):
        union |= inst["mask"]
    return union


def _workspace_mask(shape: tuple[int, int], args: Args) -> np.ndarray:
    h, w = shape
    y0 = int(round(h * max(0.0, min(1.0, args.seed_workspace_y_min_ratio))))
    y1 = int(round(h * max(0.0, min(1.0, args.seed_workspace_y_max_ratio))))
    if y1 <= y0:
        y0, y1 = 0, h
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, :] = True
    return mask


def _filter_and_nms_instances(
    instances: list[dict],
    robot_mask: np.ndarray,
    args: Args,
    total_area: int,
) -> tuple[list[dict], dict[str, int]]:
    workspace = _workspace_mask(robot_mask.shape, args)
    reject = {
        "empty_after_robot": 0,
        "small": 0,
        "large": 0,
        "outside_roi": 0,
        "centroid_outside_roi": 0,
        "nms": 0,
    }
    filtered: list[dict] = []
    for inst in instances:
        mask = inst["mask"] & ~robot_mask
        if not mask.any():
            reject["empty_after_robot"] += 1
            continue
        area_ratio = float(mask.sum() / total_area)
        if area_ratio < args.seed_min_area_ratio:
            reject["small"] += 1
            continue
        if area_ratio > args.seed_max_area_ratio:
            reject["large"] += 1
            continue
        workspace_overlap = float((mask & workspace).sum() / max(mask.sum(), 1))
        if workspace_overlap < args.seed_min_workspace_overlap:
            reject["outside_roi"] += 1
            continue
        if args.seed_require_centroid_in_workspace:
            c = _centroid(mask)
            cy = -1 if c is None else max(0, min(workspace.shape[0] - 1, int(round(c[0]))))
            cx = -1 if c is None else max(0, min(workspace.shape[1] - 1, int(round(c[1]))))
            if c is None or not workspace[cy, cx]:
                reject["centroid_outside_roi"] += 1
                continue
        item = dict(inst)
        item["mask"] = mask
        item["area_ratio"] = area_ratio
        item["workspace_overlap"] = workspace_overlap
        filtered.append(item)

    filtered.sort(key=lambda x: (x["score"], x["area_ratio"]), reverse=True)
    kept: list[dict] = []
    for inst in filtered:
        if all(_iou(inst["mask"], prev["mask"]) < args.seed_nms_iou for prev in kept):
            kept.append(inst)
        else:
            reject["nms"] += 1
        if len(kept) >= args.max_seed_objects:
            break
    return kept, reject


def _track_episode(vid_pred, frames: np.ndarray, init_masks: np.ndarray, needed_frames: set[int], device: str):
    tracked: dict[int, dict[int, np.ndarray]] = {}
    if len(init_masks) == 0:
        return tracked
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for t, frame in enumerate(frames):
            PILImage.fromarray(frame).save(str(tmp / f"{t:05d}.jpg"), quality=95)
        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            state = vid_pred.init_state(video_path=str(tmp))
            vid_pred.reset_state(state)
            for obj_id, mask in enumerate(init_masks, start=1):
                vid_pred.add_new_mask(state, frame_idx=0, obj_id=obj_id, mask=mask)
            for out_t, out_ids, out_logits in vid_pred.propagate_in_video(state):
                if out_t in needed_frames:
                    tracked[out_t] = {
                        int(obj_id): (out_logits[j] > 0.0).squeeze(0).cpu().numpy()
                        for j, obj_id in enumerate(out_ids)
                    }
    return tracked


def _classify_skill_objects(
    start_masks: dict[int, np.ndarray],
    end_masks: dict[int, np.ndarray],
    end_robot: np.ndarray,
    args: Args,
) -> tuple[dict[int, dict], int | None]:
    states: dict[int, dict] = {}
    moved_candidates: list[tuple[int, float]] = []
    for obj_id, start_mask in start_masks.items():
        if obj_id <= 0 or start_mask is None or not start_mask.any():
            continue
        end_mask = end_masks.get(obj_id)
        if end_mask is None:
            end_mask = np.zeros_like(start_mask, dtype=bool)
        start_area = max(float(start_mask.sum()), 1.0)
        visible_ratio = float(end_mask.sum() / start_area)
        robot_occ = float((end_robot & start_mask).sum() / start_area)
        cs = _centroid(start_mask)
        ce = _centroid(end_mask)
        centroid_dist = 0.0
        if cs is not None and ce is not None:
            centroid_dist = float(np.linalg.norm(ce - cs))

        if visible_ratio >= args.visible_threshold:
            state = "moved" if centroid_dist >= args.move_thr else "unchanged"
        elif robot_occ >= args.occ_thr:
            state = "occluded"
        else:
            state = "uncertain"

        score = centroid_dist * max(visible_ratio, 0.05)
        states[obj_id] = {
            "state": state,
            "visible_ratio": visible_ratio,
            "robot_occlusion_score": robot_occ,
            "centroid_dist": centroid_dist,
            "score": score,
        }
        if state == "moved":
            moved_candidates.append((obj_id, score))

    moved_id = None
    if moved_candidates:
        moved_id = max(moved_candidates, key=lambda x: x[1])[0]
        for obj_id, info in states.items():
            if info["state"] == "moved" and obj_id != moved_id:
                info["state"] = "unchanged_by_one_moved_constraint"
    return states, moved_id


def _get_groups(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    groups: list[list[int]] = []
    cur = [indices[0]]
    for idx in indices[1:]:
        if idx == cur[-1] + 1:
            cur.append(idx)
        else:
            groups.append(cur)
            cur = [idx]
    groups.append(cur)
    return groups


def _nearest_green_object(changed_ids: set[int], masks: dict[int, np.ndarray]) -> int | None:
    changed_centroids = []
    for obj_id in changed_ids:
        mask = masks.get(obj_id)
        if mask is None or not mask.any():
            continue
        c = _centroid(mask)
        if c is not None:
            changed_centroids.append(c)
    if not changed_centroids:
        return None

    best_id = None
    best_dist = float("inf")
    for obj_id, mask in masks.items():
        if obj_id <= 0 or obj_id in changed_ids or mask is None or not mask.any():
            continue
        c = _centroid(mask)
        if c is None:
            continue
        dist = min(float(np.linalg.norm(c - cc)) for cc in changed_centroids)
        if dist < best_dist:
            best_dist = dist
            best_id = obj_id
    return best_id


def _moved_overlay(
    frame: np.ndarray,
    masks: dict[int, np.ndarray],
    moved_ids: set[int],
    green_ids: set[int],
    alpha: float,
) -> np.ndarray:
    h, w = frame.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for obj_id, mask in sorted(masks.items()):
        if obj_id <= 0 or mask is None or not mask.any():
            continue
        if obj_id in moved_ids or obj_id in green_ids:
            continue
        rgba[mask, 0] = 0.05
        rgba[mask, 1] = 0.28
        rgba[mask, 2] = 1.0
        rgba[mask, 3] = alpha * 0.55
    for obj_id in sorted(green_ids):
        mask = masks.get(obj_id)
        if mask is None or not mask.any():
            continue
        rgba[mask, 0] = 0.05
        rgba[mask, 1] = 1.0
        rgba[mask, 2] = 0.05
        rgba[mask, 3] = alpha
    for obj_id in sorted(moved_ids):
        mask = masks.get(obj_id)
        if mask is None or not mask.any():
            continue
        rgba[mask, 0] = 1.0
        rgba[mask, 1] = 0.05
        rgba[mask, 2] = 0.02
        rgba[mask, 3] = alpha
    return rgba


def _colored_object_overlay(
    shape: tuple[int, int],
    masks_by_id: dict[int, np.ndarray],
    colors: dict[int, np.ndarray],
    alpha: float,
) -> np.ndarray:
    h, w = shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for obj_id, mask in sorted(masks_by_id.items()):
        if obj_id <= 0 or mask is None or not mask.any():
            continue
        color = colors.get(obj_id)
        if color is None:
            continue
        rgba[mask, :3] = color
        rgba[mask, 3] = alpha
    return rgba


def _instance_overlay(shape: tuple[int, int], instances: list[dict], alpha: float) -> np.ndarray:
    h, w = shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rng = np.random.default_rng(17)
    for inst in sorted(instances, key=lambda x: x["score"], reverse=True):
        mask = inst["mask"]
        if mask is None or not mask.any():
            continue
        color = rng.uniform(0.1, 1.0, size=3).astype(np.float32)
        rgba[mask, :3] = color
        rgba[mask, 3] = alpha
    return rgba


def _save_no_seed_diagnostic(
    out_dir: Path,
    frames: np.ndarray,
    skill_indices: list[int],
    metadata: list[dict],
    first_frame: np.ndarray,
    raw_instances: list[dict],
    prompts: list[str],
    lang: str,
    reason: str,
    workspace: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import textwrap  # noqa: PLC0415

    raw_overlay = _instance_overlay(first_frame.shape[:2], raw_instances, alpha=0.45)
    for idx in skill_indices:
        meta = metadata[idx]
        fs = min(int(meta["frame_start"]), len(frames) - 1)
        frame = frames[fs]
        fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.8))
        ax[0].imshow(frame)
        ax[0].set_title(
            f"task{int(meta['task_id'])} | ep{int(meta['episode_id'])} | "
            f"skill{int(meta['skill_index'])} | start frame {int(meta['frame_start'])}",
            fontsize=9,
        )
        ax[0].axis("off")
        ax[1].imshow(first_frame)
        ax[1].imshow(raw_overlay)
        ys = np.where(workspace.any(axis=1))[0]
        if len(ys):
            ax[1].axhline(int(ys.min()), color="cyan", lw=1.2, alpha=0.9)
            ax[1].axhline(int(ys.max()), color="cyan", lw=1.2, alpha=0.9)
        ax[1].set_title(f"first-frame raw SAM3 instances: {len(raw_instances)}", fontsize=9)
        ax[1].axis("off")
        ax[2].imshow(frame)
        ax[2].set_title(reason, fontsize=9)
        ax[2].axis("off")
        headline = (
            f"{reason}\n"
            f"prompts={prompts}\n"
            f"{lang or '(no language)'}"
        )
        headline = "\n".join(textwrap.wrap(headline, width=120))
        n_lines = headline.count("\n") + 1
        fig.suptitle(headline, fontsize=11, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94 - 0.035 * (n_lines - 1)))
        name = f"task{int(meta['task_id']):02d}_ep{int(meta['episode_id']):05d}_skill{int(meta['skill_index']):02d}_NO_SEED.png"
        fig.savefig(out_dir / name, dpi=120)
        plt.close(fig)


def main(args: Args) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import textwrap  # noqa: PLC0415
    from sam2.build_sam import build_sam2_video_predictor  # noqa: PLC0415
    from transformers import Sam3Model, Sam3Processor  # noqa: PLC0415

    settings = train_settings(load_config(args.config))
    skills_dir = Path(args.skills_dir) if args.skills_dir else Path(settings["skillset_dir"]) / "skills"
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path(settings["raw_dataset_dir"])
    out_dir = Path(args.output_dir) if args.output_dir else _HERE.parent / "outputs" / "sam3_seeded_motion"
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
        raise SystemExit(f"No skills matched task_ids={args.task_ids!r}, episode_ids={args.episode_ids!r}")

    selected_by_ep: dict[int, list[int]] = defaultdict(list)
    for idx in selected:
        selected_by_ep[int(metadata[idx]["episode_id"])].append(idx)
    print(f"[eval] selected {len(selected)} skills from {len(selected_by_ep)} episodes")

    device = args.device if torch.cuda.is_available() else "cpu"
    sam2_ckpt = Path(args.sam2_checkpoint) if args.sam2_checkpoint else Path(settings["sam2_checkpoint"])
    print(f"[eval] loading SAM3: {args.sam3_checkpoint}")
    sam3_proc = Sam3Processor.from_pretrained(args.sam3_checkpoint)
    sam3_model = Sam3Model.from_pretrained(args.sam3_checkpoint).to(device).eval()

    print(f"[eval] loading SAM2 video predictor: {sam2_ckpt}")
    cwd = Path.cwd()
    try:
        os.chdir(str(settings["lerobot_root"]))
        vid_pred = build_sam2_video_predictor(args.sam2_config, str(sam2_ckpt), device=device)
    finally:
        os.chdir(str(cwd))

    base_prompts = [p.strip() for p in args.object_prompts.split("|") if p.strip()]
    for prompt in [p.strip() for p in args.extra_object_prompts.split("|") if p.strip()]:
        if prompt not in base_prompts:
            base_prompts.append(prompt)
    robot_prompts = [args.arm_prompt] + ([args.eef_prompt] if args.exclude_eef else [])

    for ep_count, (ep_id, skill_indices) in enumerate(sorted(selected_by_ep.items()), start=1):
        row = episodes_meta[episodes_meta["episode_index"] == ep_id].iloc[0]
        vpath = _video_path(dataset_dir, episodes_meta, ep_id, image_key)
        from_ts = float(row[f"videos/{image_key}/from_timestamp"])
        to_ts = float(row[f"videos/{image_key}/to_timestamp"])
        length = int(row["length"])
        frames = read_episode_frames(vpath, from_ts, to_ts, length)
        if len(frames) == 0:
            print(f"[eval] ep{ep_id}: no frames, skip")
            continue

        lang = ep_lang.get(ep_id) or ""
        prompts = list(base_prompts)
        if args.use_language_objects:
            for idx in skill_indices:
                instruction = ep_lang.get(ep_id) or task_lang.get(int(metadata[idx]["task_id"]), "")
                for phrase in _object_phrases(instruction):
                    if phrase not in prompts:
                        prompts.append(phrase)

        first_frame = frames[0]
        print(f"[eval] ep{ep_id} ({ep_count}/{len(selected_by_ep)}): SAM3 seed prompts={len(prompts)}")
        object_instances = _sam3_instances(sam3_model, sam3_proc, first_frame, prompts, args, device)
        first_robot = _sam3_union(sam3_model, sam3_proc, first_frame, robot_prompts, args, device)
        seeds, reject = _filter_and_nms_instances(
            object_instances,
            first_robot,
            args,
            first_frame.shape[0] * first_frame.shape[1],
        )
        print(
            f"[eval] ep{ep_id}: raw instances={len(object_instances)}, seeds={len(seeds)}, "
            f"reject={reject}"
        )
        if not seeds:
            reason = (
                "no seed after filtering"
                if object_instances
                else "no raw SAM3 object instance"
            )
            _save_no_seed_diagnostic(
                out_dir,
                frames,
                skill_indices,
                metadata,
                first_frame,
                object_instances,
                prompts,
                lang,
                reason,
                _workspace_mask(first_frame.shape[:2], args),
            )
            print(f"[eval] ep{ep_id}: saved no-seed diagnostics")
            continue
        rng = np.random.default_rng(ep_id)
        seed_colors = {
            obj_id: rng.uniform(0.1, 1.0, size=3).astype(np.float32)
            for obj_id in range(1, len(seeds) + 1)
        }
        first_seed_masks = {obj_id: seed["mask"] for obj_id, seed in enumerate(seeds, start=1)}

        needed_frames: set[int] = set()
        for idx in skill_indices:
            fs = min(int(metadata[idx]["frame_start"]), len(frames) - 1)
            fe = min(int(metadata[idx]["frame_end"]) - 1, len(frames) - 1)
            needed_frames.add(fs)
            needed_frames.add(fe)

        init_masks = np.stack([seed["mask"] for seed in seeds]).astype(bool)
        tracked = _track_episode(vid_pred, frames, init_masks, needed_frames, device)

        end_frames = sorted({min(int(metadata[idx]["frame_end"]) - 1, len(frames) - 1) for idx in skill_indices})
        end_robot: dict[int, np.ndarray] = {}
        for fi in end_frames:
            end_robot[fi] = _sam3_union(sam3_model, sam3_proc, frames[fi], robot_prompts, args, device)

        per_skill_state: dict[int, dict] = {}
        moved_by_skill_index: dict[int, int] = {}
        moved_ids_for_episode: set[int] = set()
        for idx in skill_indices:
            fs = min(int(metadata[idx]["frame_start"]), len(frames) - 1)
            fe = min(int(metadata[idx]["frame_end"]) - 1, len(frames) - 1)
            states, moved_id = _classify_skill_objects(
                tracked.get(fs, {}),
                tracked.get(fe, {}),
                end_robot.get(fe, np.zeros(frames[fe].shape[:2], dtype=bool)),
                args,
            )
            per_skill_state[idx] = states
            if moved_id is not None:
                moved_ids_for_episode.add(moved_id)
                moved_by_skill_index[int(metadata[idx]["skill_index"])] = moved_id

        skill_index_to_meta_idx = {int(metadata[idx]["skill_index"]): idx for idx in skill_indices}
        green_ids_for_episode: set[int] = set()
        for group in _get_groups(sorted(moved_by_skill_index)):
            changed_ids = {moved_by_skill_index[skill_index] for skill_index in group}
            last_idx = skill_index_to_meta_idx[group[-1]]
            last_end = min(int(metadata[last_idx]["frame_end"]) - 1, len(frames) - 1)
            green_id = _nearest_green_object(changed_ids, tracked.get(last_end, {}))
            if green_id is not None:
                green_ids_for_episode.add(green_id)

        print(
            f"[eval] ep{ep_id}: moved object ids={sorted(moved_ids_for_episode)}, "
            f"green object ids={sorted(green_ids_for_episode)}"
        )
        for idx in skill_indices:
            meta = metadata[idx]
            fs = min(int(meta["frame_start"]), len(frames) - 1)
            frame = frames[fs]
            masks = tracked.get(fs, {})
            overlay = _moved_overlay(
                frame,
                masks,
                moved_ids_for_episode,
                green_ids_for_episode,
                args.overlay_alpha,
            )
            states = per_skill_state.get(idx, {})
            counts = defaultdict(int)
            for info in states.values():
                counts[info["state"]] += 1

            ids = (
                f"task{int(meta['task_id'])} | ep{ep_id} | skill{int(meta['skill_index'])} | "
                f"start frame {int(meta['frame_start'])}"
            )
            summary = (
                f"seeds={len(seeds)} | moved_ids={sorted(moved_ids_for_episode)} | "
                f"green_ids={sorted(green_ids_for_episode)} | "
                f"moved={counts['moved']} occluded={counts['occluded']} uncertain={counts['uncertain']}"
            )
            fig, ax = plt.subplots(1, 4, figsize=(17.5, 4.8))
            ax[0].imshow(frame)
            ax[0].set_title(ids, fontsize=9)
            ax[0].axis("off")

            ax[1].imshow(frame)
            ax[1].imshow(overlay)
            ax[1].set_title("tracked objects: moved=red, green=nearest, others=blue", fontsize=9)
            ax[1].axis("off")

            first_seed_overlay = _colored_object_overlay(
                first_frame.shape[:2],
                first_seed_masks,
                seed_colors,
                alpha=0.45,
            )
            ax[2].imshow(first_frame)
            ax[2].imshow(first_seed_overlay)
            workspace = _workspace_mask(first_frame.shape[:2], args)
            ys = np.where(workspace.any(axis=1))[0]
            if len(ys):
                ax[2].axhline(int(ys.min()), color="cyan", lw=1.2, alpha=0.9)
                ax[2].axhline(int(ys.max()), color="cyan", lw=1.2, alpha=0.9)
            ax[2].set_title("episode first-frame SAM3 seeds", fontsize=9)
            ax[2].axis("off")

            current_seed_overlay = _colored_object_overlay(
                frame.shape[:2],
                masks,
                seed_colors,
                alpha=0.35,
            )
            ax[3].imshow(frame)
            ax[3].imshow(current_seed_overlay)
            ax[3].set_title("same seed ids at this skill start", fontsize=9)
            ax[3].axis("off")

            headline = "\n".join(textwrap.wrap(f"{summary}\n{lang or '(no language)'}", width=120))
            n_lines = headline.count("\n") + 1
            fig.suptitle(headline, fontsize=12, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.94 - 0.035 * (n_lines - 1)))
            name = f"task{int(meta['task_id']):02d}_ep{ep_id:05d}_skill{int(meta['skill_index']):02d}.png"
            fig.savefig(out_dir / name, dpi=120)
            plt.close(fig)

    print(f"[eval] saved PNGs -> {out_dir}")


if __name__ == "__main__":
    _normalize_underscore_cli_flags()
    main(tyro.cli(Args))
