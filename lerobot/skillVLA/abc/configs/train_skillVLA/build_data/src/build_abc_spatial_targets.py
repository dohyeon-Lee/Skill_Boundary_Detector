#!/usr/bin/env python3
"""Build exact bimanual XYZ/UV targets for an ABC-native SkillVLA dataset.

The artifact is deliberately independent from the compact skill annotations.
It can therefore be added to an already completed SkillVLA build without
re-encoding FSQ skills or rebuilding ISS.  Per-episode shards make the costly
MuJoCo FK pass resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mujoco
import numpy as np


SCHEMA_VERSION = 1
SCHEMA_NAME = "abc_bimanual_skill_spatial/v1"
PROJECTION_NAMES = ("top_left", "top_right", "left_wrist", "right_wrist")
CAMERA_NAMES = ("top", "left", "right")
SITE_NAMES = ("left_grasp_site", "right_grasp_site")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_npz(path: Path, *, compressed: bool = True, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = np.savez_compressed if compressed else np.savez
        writer(handle, **arrays)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _kinematics_only_xml(scene_xml: str) -> str:
    root = ET.fromstring(scene_xml)
    for parent in root.iter():
        for child in list(parent):
            if child.tag in {"geom", "contact", "sensor"}:
                parent.remove(child)
    asset = root.find("asset")
    if asset is not None:
        root.remove(asset)
    return ET.tostring(root, encoding="unicode")


def _compile_model(scene_path: Path, assets_dir: Path) -> mujoco.MjModel:
    scene_xml = scene_path.read_text(encoding="utf-8")
    scene_xml = re.sub(
        r'(?<=\")/[^\"]*?/models/assets(?=[/\"])', str(assets_dir), scene_xml
    )
    scene_xml = _kinematics_only_xml(scene_xml)
    previous = Path.cwd()
    os.chdir(assets_dir.parent)
    try:
        return mujoco.MjModel.from_xml_string(scene_xml)
    finally:
        os.chdir(previous)


def _projection_matrix(
    position: np.ndarray,
    rotation: np.ndarray,
    *,
    fovy: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Return P where P@[xyz,1] gives [u*depth,v*depth,depth]."""
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    world_to_camera = np.empty((3, 4), dtype=np.float64)
    world_to_camera[:, :3] = rotation.T
    world_to_camera[:, 3] = -rotation.T @ np.asarray(position, dtype=np.float64)
    focal = 0.5 * height / math.tan(math.radians(float(fovy)) / 2.0)
    camera_to_pixel = np.asarray(
        [
            [focal, 0.0, -width / 2.0],
            [0.0, -focal, -height / 2.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    return (camera_to_pixel @ world_to_camera).astype(np.float32)


def _project(points: np.ndarray, matrices: np.ndarray, width: int, height: int):
    points = np.asarray(points, dtype=np.float32)
    matrices = np.asarray(matrices, dtype=np.float32)
    homogeneous = np.concatenate(
        [points, np.ones((*points.shape[:-1], 1), dtype=np.float32)], axis=-1
    )
    projected = np.einsum("...ij,...j->...i", matrices, homogeneous)
    depth = projected[..., 2]
    valid = np.isfinite(projected).all(axis=-1) & (depth > 1e-8)
    raw = np.full((*points.shape[:-1], 2), np.nan, dtype=np.float32)
    np.divide(
        projected[..., :2],
        depth[..., None],
        out=raw,
        where=valid[..., None],
    )
    clipped = valid & (
        (raw[..., 0] < 0)
        | (raw[..., 0] > width - 1)
        | (raw[..., 1] < 0)
        | (raw[..., 1] > height - 1)
    )
    bounded = raw.copy()
    bounded[..., 0] = np.clip(bounded[..., 0], 0, width - 1)
    bounded[..., 1] = np.clip(bounded[..., 1], 0, height - 1)
    normalized = np.full_like(bounded, np.nan)
    normalized[..., 0] = 2.0 * bounded[..., 0] / max(width - 1, 1) - 1.0
    normalized[..., 1] = 2.0 * bounded[..., 1] / max(height - 1, 1) - 1.0
    pixels = np.full(raw.shape, -1, dtype=np.int16)
    finite = np.isfinite(bounded).all(axis=-1)
    pixels[finite] = np.rint(bounded[finite]).astype(np.int16)
    return normalized, pixels, raw, valid, clipped


def _shard_current(path: Path, episode_id: int, frame_length: int, skill_rows: np.ndarray) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as shard:
            return (
                int(shard["schema_version"]) == SCHEMA_VERSION
                and int(shard["episode_id"]) == episode_id
                and int(shard["episode_length"]) == frame_length
                and np.array_equal(shard["skill_rows"], skill_rows)
            )
    except (KeyError, OSError, ValueError):
        return False


def _build_episode(job: dict) -> str:
    episode_id = int(job["episode_id"])
    episode_dir = Path(job["episode_dir"])
    shard_path = Path(job["shard_path"])
    skill_rows = np.asarray(job["skill_rows"], dtype=np.int32)
    starts = np.asarray(job["frame_start"], dtype=np.int32)
    ends = np.asarray(job["frame_end"], dtype=np.int32)
    frame_skill_rank = np.asarray(job["frame_skill_rank"], dtype=np.int32)
    length = len(frame_skill_rank)
    if _shard_current(shard_path, episode_id, length, skill_rows):
        return str(shard_path)

    metadata = json.loads((episode_dir / "episode_metadata.json").read_text())
    width, height = int(metadata["image_width"]), int(metadata["image_height"])
    if not {"top", "left", "right"}.issubset(metadata.get("cameras", [])):
        raise ValueError(f"Episode {episode_id} lacks top/left/right cameras")
    scene_qpos = np.load(episode_dir / "scene_qpos.npy", mmap_mode="r")
    if len(scene_qpos) != length:
        raise ValueError(
            f"Episode {episode_id}: qpos={len(scene_qpos)}, annotations={length}"
        )
    model = _compile_model(
        episode_dir / "scene_assembled.xml", Path(job["assets_dir"])
    )
    if scene_qpos.shape[1] != model.nq:
        raise ValueError(
            f"Episode {episode_id}: scene nq={scene_qpos.shape[1]}, model nq={model.nq}"
        )
    camera_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        for name in CAMERA_NAMES
    ]
    site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in SITE_NAMES
    ]
    if min(camera_ids + site_ids) < 0:
        raise ValueError(f"Episode {episode_id} lacks required cameras or grasp sites")
    fovys = [float(model.cam_fovy[index]) for index in camera_ids]

    skill_count = len(starts)
    start_xyz = np.full((skill_count, 2, 3), np.nan, dtype=np.float32)
    end_xyz = np.full((skill_count, 2, 3), np.nan, dtype=np.float32)
    wrist_projection = np.empty((length, 2, 3, 4), dtype=np.float32)
    wanted_start = {int(frame): rank for rank, frame in enumerate(starts)}
    endpoint_frames = np.minimum(ends, length - 1)
    wanted_end = {int(frame): [] for frame in endpoint_frames}
    for rank, frame in enumerate(endpoint_frames):
        wanted_end[int(frame)].append(rank)

    data = mujoco.MjData(model)
    top_projection = None
    for frame_index, qpos in enumerate(scene_qpos):
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        site_xyz = np.stack([np.asarray(data.site_xpos[index]) for index in site_ids])
        if frame_index in wanted_start:
            start_xyz[wanted_start[frame_index]] = site_xyz
        for rank in wanted_end.get(frame_index, []):
            end_xyz[rank] = site_xyz
        for camera_slot, camera_id in enumerate(camera_ids):
            matrix = _projection_matrix(
                data.cam_xpos[camera_id],
                np.asarray(data.cam_xmat[camera_id]).reshape(3, 3),
                fovy=fovys[camera_slot],
                width=width,
                height=height,
            )
            if camera_slot == 0:
                if top_projection is None:
                    top_projection = matrix
            else:
                wrist_projection[frame_index, camera_slot - 1] = matrix
    if top_projection is None or not np.isfinite(start_xyz).all() or not np.isfinite(end_xyz).all():
        raise ValueError(f"Episode {episode_id} produced incomplete spatial targets")
    if frame_skill_rank.min(initial=0) < 0 or frame_skill_rank.max(initial=0) >= skill_count:
        raise ValueError(f"Episode {episode_id} has invalid frame skill ranks")

    targets = end_xyz[frame_skill_rank]
    top = np.broadcast_to(top_projection, (length, 3, 4))
    matrices = np.stack(
        [top, top, wrist_projection[:, 0], wrist_projection[:, 1]], axis=1
    )
    points = np.stack(
        [targets[:, 0], targets[:, 1], targets[:, 0], targets[:, 1]], axis=1
    )
    focus_uv, focus_pixels, focus_raw, focus_valid, focus_clipped = _project(
        points, matrices, width, height
    )
    _atomic_npz(
        shard_path,
        compressed=True,
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int32),
        episode_id=np.asarray(episode_id, dtype=np.int32),
        episode_length=np.asarray(length, dtype=np.int32),
        image_height=np.asarray(height, dtype=np.int32),
        image_width=np.asarray(width, dtype=np.int32),
        skill_rows=skill_rows,
        start_xyz=start_xyz,
        end_xyz=end_xyz,
        top_projection=top_projection,
        wrist_projection=wrist_projection,
        focus_uv=focus_uv,
        focus_uv_pixels=focus_pixels,
        focus_uv_raw_pixels=focus_raw,
        focus_valid=focus_valid,
        focus_clipped=focus_clipped,
    )
    return str(shard_path)


def artifact_is_current(
    path: Path, skill_latents_path: Path, annotations_path: Path
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as artifact:
            return (
                int(artifact["schema_version"]) >= SCHEMA_VERSION
                and str(artifact["schema_name"]) == SCHEMA_NAME
                and str(artifact["skill_latents_sha256"]) == _sha256(skill_latents_path)
                and str(artifact["skill_annotations_sha256"]) == _sha256(annotations_path)
                and {
                    "skill_start_xyz",
                    "skill_end_xyz",
                    "top_projection",
                    "wrist_projection",
                    "focus_uv",
                    "focus_valid",
                }.issubset(artifact.files)
            )
    except (KeyError, OSError, ValueError):
        return False


def build(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    latents_path = args.skill_latents_path.resolve()
    annotations_path = args.annotations_path.resolve()
    if artifact_is_current(output, latents_path, annotations_path):
        print(f"ABC spatial target artifact already current: {output}")
        return

    with np.load(latents_path, allow_pickle=False) as source:
        required = {"episode_id", "skill_index", "frame_start", "frame_end"}
        missing = sorted(required - set(source.files))
        if missing:
            raise ValueError(f"{latents_path} is missing {missing}")
        identity = {key: np.asarray(source[key]).copy() for key in required}
    with np.load(annotations_path, allow_pickle=False) as source:
        episode_ids = np.asarray(source["episode_id"], dtype=np.int32)
        episode_lengths = np.asarray(source["episode_length"], dtype=np.int32)
        offsets = np.asarray(source["offsets"], dtype=np.int64)
        frame_skill_index = np.asarray(source["skill_index"], dtype=np.int32)

    marker_path = args.skill_dataset_dir.resolve() / "native_dataset.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    raw_root = Path(marker["raw_root"]).expanduser()
    if not raw_root.is_absolute():
        raw_root = (marker_path.parent / raw_root).resolve()
    sources = marker["sources"]
    assets_dir = args.assets_dir.resolve()
    if not assets_dir.is_dir():
        raise FileNotFoundError(f"ABC MuJoCo assets not found: {assets_dir}")

    rows_by_episode: dict[int, list[int]] = {}
    for row, episode_id in enumerate(identity["episode_id"]):
        rows_by_episode.setdefault(int(episode_id), []).append(row)
    shard_dir = args.shard_dir.resolve()
    shard_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for episode_position, (episode_id, length) in enumerate(
        zip(episode_ids, episode_lengths, strict=True)
    ):
        episode_id = int(episode_id)
        rows = np.asarray(rows_by_episode.get(episode_id, []), dtype=np.int32)
        if not len(rows):
            raise ValueError(f"Episode {episode_id} has no skill rows")
        start, stop = int(offsets[episode_position]), int(offsets[episode_position + 1])
        jobs.append(
            {
                "episode_id": episode_id,
                "episode_dir": str(raw_root / sources[episode_id]["path"]),
                "assets_dir": str(assets_dir),
                "shard_path": str(shard_dir / f"episode_{episode_id:06d}.npz"),
                "skill_rows": rows,
                "frame_start": identity["frame_start"][rows],
                "frame_end": identity["frame_end"][rows],
                "frame_skill_rank": frame_skill_index[start:stop],
            }
        )

    workers = max(1, min(int(args.workers), len(jobs)))
    print(
        f"ABC spatial targets: {len(jobs)} episodes, {len(frame_skill_index)} frames, "
        f"{len(identity['episode_id'])} skills, {workers} workers",
        flush=True,
    )
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_build_episode, job): job for job in jobs}
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 25 == 0 or completed == len(jobs):
                print(f"  FK/projection {completed}/{len(jobs)} episodes", flush=True)

    skill_count = len(identity["episode_id"])
    skill_start_xyz = np.empty((skill_count, 2, 3), dtype=np.float32)
    skill_end_xyz = np.empty((skill_count, 2, 3), dtype=np.float32)
    top_projection = []
    wrist_projection = []
    focus_uv = []
    focus_uv_pixels = []
    focus_uv_raw_pixels = []
    focus_valid = []
    focus_clipped = []
    image_sizes = set()
    for job in jobs:
        with np.load(job["shard_path"], allow_pickle=False) as shard:
            rows = np.asarray(shard["skill_rows"], dtype=np.int32)
            skill_start_xyz[rows] = shard["start_xyz"]
            skill_end_xyz[rows] = shard["end_xyz"]
            top_projection.append(np.asarray(shard["top_projection"], dtype=np.float32))
            wrist_projection.append(np.asarray(shard["wrist_projection"], dtype=np.float32))
            focus_uv.append(np.asarray(shard["focus_uv"], dtype=np.float32))
            focus_uv_pixels.append(np.asarray(shard["focus_uv_pixels"], dtype=np.int16))
            focus_uv_raw_pixels.append(
                np.asarray(shard["focus_uv_raw_pixels"], dtype=np.float32)
            )
            focus_valid.append(np.asarray(shard["focus_valid"], dtype=np.bool_))
            focus_clipped.append(np.asarray(shard["focus_clipped"], dtype=np.bool_))
            image_sizes.add((int(shard["image_height"]), int(shard["image_width"])))
    if len(image_sizes) != 1:
        raise ValueError(f"ABC spatial targets require one image size; got {image_sizes}")
    height, width = image_sizes.pop()

    _atomic_npz(
        output,
        compressed=True,
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int32),
        schema_name=np.asarray(SCHEMA_NAME),
        skill_latents_sha256=np.asarray(_sha256(latents_path)),
        skill_annotations_sha256=np.asarray(_sha256(annotations_path)),
        projection_names=np.asarray(PROJECTION_NAMES),
        camera_names=np.asarray(CAMERA_NAMES),
        arm_names=np.asarray(("left", "right")),
        xyz_frame=np.asarray("MuJoCo world; left/right grasp sites"),
        startpoint_convention=np.asarray("frame_start"),
        endpoint_convention=np.asarray("min(frame_end, episode_length - 1)"),
        uv_normalization=np.asarray("minus_one_to_one; clipped to image border"),
        image_height=np.asarray(height, dtype=np.int32),
        image_width=np.asarray(width, dtype=np.int32),
        episode_id=episode_ids,
        episode_length=episode_lengths,
        frame_offsets=offsets,
        skill_episode_id=np.asarray(identity["episode_id"], dtype=np.int32),
        skill_index=np.asarray(identity["skill_index"], dtype=np.int32),
        frame_start=np.asarray(identity["frame_start"], dtype=np.int32),
        frame_end=np.asarray(identity["frame_end"], dtype=np.int32),
        skill_start_xyz=skill_start_xyz,
        skill_end_xyz=skill_end_xyz,
        top_projection=np.stack(top_projection),
        wrist_projection=np.concatenate(wrist_projection),
        focus_uv=np.concatenate(focus_uv),
        focus_uv_pixels=np.concatenate(focus_uv_pixels),
        focus_uv_raw_pixels=np.concatenate(focus_uv_raw_pixels),
        focus_valid=np.concatenate(focus_valid),
        focus_clipped=np.concatenate(focus_clipped),
    )

    info_path = args.skill_dataset_dir.resolve() / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info.update(
        {
            "skill_spatial_targets_path": str(output),
            "skill_spatial_targets_schema": SCHEMA_NAME,
            "skill_spatial_xyz_frame": "mujoco_world_grasp_sites",
            "skill_spatial_arms": ["left", "right"],
            "skill_focus_uv_views": list(PROJECTION_NAMES),
            "skill_focus_uv_normalization": "minus_one_to_one",
            "skill_focus_uv_image_size": [height, width],
            "skill_focus_uv_outside_policy": "clip_to_nearest_image_point",
        }
    )
    _atomic_json(info_path, info)

    manifest_path = output.parent / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "spatial_targets_path": str(output),
            "spatial_targets_sha256": _sha256(output),
            "spatial_targets_schema": SCHEMA_NAME,
            "spatial_target_skills": skill_count,
            "spatial_target_frames": len(frame_skill_index),
            "spatial_target_views": list(PROJECTION_NAMES),
        }
    )
    _atomic_json(manifest_path, manifest)
    if not args.keep_shards:
        shutil.rmtree(shard_dir)
    valid = np.concatenate(focus_valid)
    clipped = np.concatenate(focus_clipped)
    print(
        f"DONE: {output} ({output.stat().st_size / 2**20:.1f} MiB; "
        f"valid={valid.mean(axis=0).round(4).tolist()}, "
        f"clipped={clipped.mean(axis=0).round(4).tolist()})",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-current", type=Path)
    parser.add_argument("--skill-dataset-dir", type=Path)
    parser.add_argument("--skill-latents-path", type=Path)
    parser.add_argument("--annotations-path", type=Path)
    parser.add_argument("--assets-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shard-dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--keep-shards", action="store_true")
    args = parser.parse_args()
    if args.check_current is not None:
        if args.skill_latents_path is None or args.annotations_path is None:
            parser.error("--check-current requires --skill-latents-path and --annotations-path")
        raise SystemExit(
            0
            if artifact_is_current(
                args.check_current.resolve(),
                args.skill_latents_path.resolve(),
                args.annotations_path.resolve(),
            )
            else 1
        )
    required = (
        "skill_dataset_dir",
        "skill_latents_path",
        "annotations_path",
        "assets_dir",
        "output",
        "shard_dir",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("build mode is missing: " + ", ".join(missing))
    return args


if __name__ == "__main__":
    build(parse_args())
