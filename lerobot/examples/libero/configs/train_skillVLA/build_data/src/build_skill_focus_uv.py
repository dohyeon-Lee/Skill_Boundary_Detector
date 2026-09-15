#!/usr/bin/env python3
"""Build canonical start/end focus targets for every SkillVLA skill.

The output follows the flat row order of ``skill_latents.npz``.  A training
sample that transition-jitters from skill k to k' can therefore select the
same k' in this artifact without inventing a second randomization decision.
The historical ``focus_*`` arrays remain the endpoint target; explicit
``start_focus_*`` arrays retain the first skill's otherwise unrecoverable start.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from lerobot.policies.skillVLA.focus_projection import (
    camera_transform_from_recorded_xml,
    project_eef,
)

_HERE = Path(__file__).resolve().parent
_SKILL_EVAL_SRC = _HERE.parents[1] / "stage1_skill_eval" / "src"
sys.path.insert(0, str(_SKILL_EVAL_SRC))

VIDEO_KEY = "observation.images.image"
CURRENT_SCHEMA_VERSION = 2
CURRENT_ARRAYS = {
    "focus_uv",
    "focus_uv_pixels",
    "focus_uv_raw_pixels",
    "focus_valid",
    "focus_clipped",
    "start_focus_uv",
    "start_focus_uv_pixels",
    "start_focus_uv_raw_pixels",
    "start_focus_valid",
    "start_focus_clipped",
}


def artifact_is_current(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as artifact:
            return (
                int(np.asarray(artifact["schema_version"]).item())
                >= CURRENT_SCHEMA_VERSION
                and CURRENT_ARRAYS.issubset(artifact.files)
            )
    except (KeyError, OSError, ValueError):
        return False


def _skill_evaluation_dataset_cls():
    # The evaluator imports video and simulation dependencies. Keep it out of
    # the lightweight --check-current path used by submission scripts.
    from skill_data import SkillEvaluationDataset

    return SkillEvaluationDataset


def _image_size(info: dict, video_key: str = VIDEO_KEY) -> tuple[int, int]:
    feature = (info.get("features") or {}).get(video_key)
    if not isinstance(feature, dict):
        raise ValueError(f"Dataset info has no video feature {video_key!r}.")
    video_info = feature.get("info") or {}
    height = int(video_info.get("video.height", 0) or 0)
    width = int(video_info.get("video.width", 0) or 0)
    if height > 0 and width > 0:
        return height, width
    shape = [int(value) for value in feature.get("shape", [])]
    names = list(feature.get("names") or [])
    if len(shape) == 3 and names and "height" in names and "width" in names:
        return shape[names.index("height")], shape[names.index("width")]
    raise ValueError(
        f"Cannot infer image height/width for {video_key!r}: {feature!r}."
    )


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def build(args: argparse.Namespace) -> None:
    dataset_info_path = args.skill_dataset_dir / "meta" / "info.json"
    if not dataset_info_path.is_file():
        raise FileNotFoundError(f"SkillVLA dataset info not found: {dataset_info_path}")
    info = json.loads(dataset_info_path.read_text())
    height, width = _image_size(info)

    dataset = _skill_evaluation_dataset_cls()(
        skill_dataset_dir=args.skill_dataset_dir,
        skill_latents_path=args.skill_latents_path,
        eval_init_states_path=args.eval_init_states_path,
        original_dataset_dir=args.original_dataset_dir,
        suite_name=args.suite,
    )
    with np.load(args.skill_latents_path, allow_pickle=False) as source:
        required = {
            "episode_id",
            "task_id",
            "skill_index",
            "frame_start",
            "frame_end",
        }
        missing = sorted(required - set(source.files))
        if missing:
            raise ValueError(f"skill_latents.npz is missing {missing}.")
        identity = {
            name: np.asarray(source[name]).copy()
            for name in sorted(required)
        }

    sizes = {name: len(value) for name, value in identity.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"Skill identity arrays have inconsistent lengths: {sizes}")
    count = next(iter(sizes.values()), 0)
    focus_uv = np.empty((count, 2), dtype=np.float32)
    focus_uv_pixels = np.empty((count, 2), dtype=np.int32)
    focus_uv_raw_pixels = np.empty((count, 2), dtype=np.float32)
    focus_valid = np.empty(count, dtype=np.bool_)
    focus_clipped = np.empty(count, dtype=np.bool_)
    start_focus_uv = np.empty((count, 2), dtype=np.float32)
    start_focus_uv_pixels = np.empty((count, 2), dtype=np.int32)
    start_focus_uv_raw_pixels = np.empty((count, 2), dtype=np.float32)
    start_focus_valid = np.empty(count, dtype=np.bool_)
    start_focus_clipped = np.empty(count, dtype=np.bool_)

    rows_by_episode: dict[int, list[int]] = {}
    for index, episode_id in enumerate(identity["episode_id"]):
        rows_by_episode.setdefault(int(episode_id), []).append(index)

    processed = 0
    for episode_id in sorted(rows_by_episode):
        aligned = dataset.load_aligned_episode(episode_id)
        if not aligned.model_xml:
            raise ValueError(
                "focus_uv requires recorded fixed-camera XML; "
                f"episode {episode_id} has none."
            )
        transform = camera_transform_from_recorded_xml(
            aligned.model_xml,
            camera_name=args.camera,
            height=height,
            width=width,
        )
        episode_length = len(aligned.filtered_states)
        for index in rows_by_episode[episode_id]:
            start_frame = int(identity["frame_start"][index])
            endpoint_frame = min(int(identity["frame_end"][index]), episode_length - 1)
            if not 0 <= start_frame < episode_length or endpoint_frame < start_frame:
                raise ValueError(
                    f"Episode {episode_id} has invalid skill frames "
                    f"[{start_frame}, {endpoint_frame}] for length={episode_length}."
                )
            if dataset.proprio_grounding not in {"none", "episode_start_xyz"}:
                raise ValueError(
                    "Unsupported proprio_grounding while building focus_uv: "
                    f"{dataset.proprio_grounding!r}."
                )
            xyz = np.asarray(
                aligned.filtered_states[[start_frame, endpoint_frame], :3],
                dtype=np.float64,
            ).copy()
            if dataset.proprio_grounding == "episode_start_xyz":
                xyz += np.asarray(aligned.episode_start_xyz, dtype=np.float64)[None]
            start_projection = project_eef(
                xyz[0], transform, height=height, width=width
            )
            end_projection = project_eef(
                xyz[1], transform, height=height, width=width
            )
            start_focus_uv[index] = start_projection.normalized_xy
            start_focus_uv_pixels[index] = start_projection.pixel_xy
            start_focus_uv_raw_pixels[index] = start_projection.raw_xy
            start_focus_valid[index] = start_projection.valid
            start_focus_clipped[index] = start_projection.clipped
            focus_uv[index] = end_projection.normalized_xy
            focus_uv_pixels[index] = end_projection.pixel_xy
            focus_uv_raw_pixels[index] = end_projection.raw_xy
            focus_valid[index] = end_projection.valid
            focus_clipped[index] = end_projection.clipped
            processed += 1
        if processed % 500 == 0 or processed == count:
            print(f"focus_uv: projected {processed}/{count} skills", flush=True)

    _atomic_npz(
        args.output,
        schema_version=np.asarray(CURRENT_SCHEMA_VERSION, dtype=np.int32),
        camera=np.asarray(args.camera),
        image_height=np.asarray(height, dtype=np.int32),
        image_width=np.asarray(width, dtype=np.int32),
        startpoint_convention=np.asarray("frame_start"),
        endpoint_convention=np.asarray("min(frame_end, episode_length - 1)"),
        **identity,
        focus_uv=focus_uv,
        focus_uv_pixels=focus_uv_pixels,
        focus_uv_raw_pixels=focus_uv_raw_pixels,
        focus_valid=focus_valid,
        focus_clipped=focus_clipped,
        start_focus_uv=start_focus_uv,
        start_focus_uv_pixels=start_focus_uv_pixels,
        start_focus_uv_raw_pixels=start_focus_uv_raw_pixels,
        start_focus_valid=start_focus_valid,
        start_focus_clipped=start_focus_clipped,
    )

    info.update(
        {
            "skill_focus_uv_path": str(args.output.resolve()),
            "skill_focus_uv_schema_version": 2,
            "skill_focus_uv_camera": args.camera,
            "skill_focus_uv_normalization": "minus_one_to_one",
            "skill_focus_uv_start": "frame_start",
            "skill_focus_uv_endpoint": "min(frame_end, episode_length - 1)",
            "skill_focus_uv_image_size": [height, width],
        }
    )
    _atomic_json(dataset_info_path, info)
    start_invalid = int((~start_focus_valid).sum())
    end_invalid = int((~focus_valid).sum())
    print(
        f"Wrote {count} start/end focus targets to {args.output} "
        f"(outside-image clipped: start={start_invalid}, end={end_invalid})."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-current", type=Path)
    parser.add_argument("--skill-dataset-dir", type=Path)
    parser.add_argument("--skill-latents-path", type=Path)
    parser.add_argument("--eval-init-states-path", type=Path)
    parser.add_argument("--original-dataset-dir", type=Path)
    parser.add_argument("--suite")
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.check_current is not None:
        raise SystemExit(0 if artifact_is_current(args.check_current) else 1)
    missing = [
        option
        for option in (
            "skill_dataset_dir",
            "skill_latents_path",
            "eval_init_states_path",
            "original_dataset_dir",
            "suite",
            "output",
        )
        if getattr(args, option) is None
    ]
    if missing:
        parser.error(
            "build mode requires "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    build(args)


if __name__ == "__main__":
    main()
