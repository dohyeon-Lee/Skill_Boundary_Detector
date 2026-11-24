"""Where the skill-end goal lands in the evaluation video, for the wrist and agent views.

A rollout that misses its goal looks the same as one that never knew where the goal was. Drawing
the goal the policy was actually conditioned on -- GT or predicted -- tells the two apart at a
glance, which is the whole point of this module.

The geometry is not new. The wrist projection is the one the alignment target uses
(``wrist_patch_target.project_into_wrist``, verified frame by frame against the offline probe);
the agent view uses the same pinhole convention as ``build_skill_focus_uv``. What this module adds
is the bookkeeping that the two cameras need DIFFERENT coordinates:

* the wrist camera rides the gripper, so only ``goal - eef`` matters and the episode grounding
  cancels -- grounded values go straight in;
* the agent view is bolted to the world, so the grounding must be added back first.

Getting that backwards puts the dot somewhere plausible but wrong, which is worse than no dot.

Camera parameters come from the live simulation rather than a parsed XML: the evaluator restores
init states into an already-built env, so ``sim.model`` is authoritative and cannot fail to parse.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from lerobot.policies.skill_expert.wrist_patch_target import WristCamera, project_into_wrist

AGENT_CAMERA = "agentview"
# Rendered LIBERO frames are mirrored in x relative to the raw projection (LiberoEnv.render and the
# env preprocessor both flip, leaving exactly build_skill_focus_uv's stored orientation).
MIRRORED_IN_X = True


@dataclass(frozen=True)
class GoalMarks:
    """Goal pixels for a run of frames, and which of them are actually on screen."""

    pixels: np.ndarray  # [frames, 2], NaN where not visible
    visible: np.ndarray  # [frames] bool

    def __len__(self) -> int:
        return int(self.visible.shape[0])


def values_from_trace(
    trace, field: str, dimensions: int, *, batch_index: int,
    n_video_frames: int, video_frame_stride: int, fallback: str | None = None,
) -> np.ndarray | None:
    """Sample chunk goals at video times, without leaking later goals backwards."""
    samples = []
    for record in trace:
        if int(record.get("batch_index", 0)) != batch_index:
            continue
        points = [record, *record.get("goal_points", [])]
        for point in points:
            if point is not record and field not in point and (fallback is None or fallback not in point):
                continue
            value = point.get(field)
            if value is None and fallback is not None:
                value = point.get(fallback)
            row = np.full(dimensions, np.nan)
            if value is not None:
                array = np.asarray(value, dtype=np.float64).reshape(-1)
                if len(array) >= dimensions and np.isfinite(array[:dimensions]).all():
                    row = array[:dimensions]
            samples.append((int(point.get("episode_timestep", 0)), row))
    if not samples:
        return None
    samples.sort(key=lambda item: item[0])
    result = np.full((n_video_frames, dimensions), np.nan)
    index = 0
    for frame_index in range(n_video_frames):
        timestep = frame_index * max(1, int(video_frame_stride))
        while index < len(samples) and samples[index][0] <= timestep:
            index += 1
        if index:
            result[frame_index] = samples[index - 1][1]
    return result if np.isfinite(result).any() else None


def bottleneck_from_trace(
    trace, *, batch_index: int, n_video_frames: int, video_frame_stride: int,
) -> dict:
    """Hold each action chunk's auxiliaries, resetting at every skill occurrence."""
    records = sorted((r for r in trace if int(r.get("batch_index", 0)) == batch_index),
                     key=lambda r: int(r.get("episode_timestep", 0)))
    result = {}
    for record_index, record in enumerate(records):
        start = int(record.get("episode_timestep", 0))
        end = (int(records[record_index + 1].get("episode_timestep", 0))
               if record_index + 1 < len(records) else float("inf"))
        points = sorted(record.get("bottleneck_points", []), key=lambda p: int(p["episode_timestep"]))
        index = -1
        for frame in range(n_video_frames):
            step = frame * max(1, int(video_frame_stride))
            if not start <= step < end:
                continue
            while index + 1 < len(points) and int(points[index + 1]["episode_timestep"]) <= step:
                index += 1
            if index < 0:
                continue
            for tower, values in points[index].get("towers", {}).items():
                result.setdefault(tower, {}).setdefault(
                    "input_timestep", np.full(n_video_frames, np.nan)
                )[frame] = int(points[index]["episode_timestep"])
                for field, dimensions in (("xyz_world", 3), ("top_patch_box_uv", 4),
                                          ("wrist_patch_box_uv", 4)):
                    value = values.get(field)
                    if value is None or np.asarray(value).shape != (dimensions,):
                        continue
                    timeline = result.setdefault(tower, {}).setdefault(
                        field, np.full((n_video_frames, dimensions), np.nan))
                    timeline[frame] = value
    return result


def marks_from_uv(uv: np.ndarray, *, height: int, width: int) -> np.ndarray:
    """Stored UV already has the rendered Top image orientation; never mirror again."""
    uv = np.asarray(uv, dtype=np.float64)
    pixels = (uv.clip(-1, 1) + 1) * np.array([width - 1, height - 1]) / 2
    pixels[~np.isfinite(uv).all(-1)] = np.nan
    return pixels


def fullskill_from_trace(
    trace, *, batch_index: int, n_video_frames: int, video_frame_stride: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Sample the first and latest plan of each occurrence, never across a skill boundary."""
    records = sorted(
        (record for record in trace if int(record.get("batch_index", 0)) == batch_index),
        key=lambda record: int(record.get("episode_timestep", 0)),
    )
    sample_plan = next((point["xyz_world"] for record in records
                        for point in record.get("fullskill_plans", [])), None)
    if sample_plan is None:
        return None, None
    shape = np.asarray(sample_plan).shape
    if len(shape) != 2 or shape[0] < 2 or shape[1] != 3:
        raise ValueError("Fullskill plans must have shape [plan_points, 3].")
    initial = np.full((n_video_frames, *shape), np.nan)
    current = np.full_like(initial, np.nan)
    for record_index, record in enumerate(records):
        start = int(record.get("episode_timestep", 0))
        end = (int(records[record_index + 1].get("episode_timestep", 0))
               if record_index + 1 < len(records) else float("inf"))
        points = sorted(record.get("fullskill_plans", []), key=lambda point: int(point["episode_timestep"]))
        point_index = -1
        for frame_index in range(n_video_frames):
            timestep = frame_index * max(1, int(video_frame_stride))
            if not start <= timestep < end:
                continue
            while point_index + 1 < len(points) and int(points[point_index + 1]["episode_timestep"]) <= timestep:
                point_index += 1
            if point_index >= 0:
                initial[frame_index] = points[0]["xyz_world"]
                current[frame_index] = points[point_index]["xyz_world"]
    return initial, current


@lru_cache(maxsize=16)
def dataset_gripper_reference(dataset_dir: str, episode_index: int = 0) -> dict:
    """Measure open and object-constrained grasp widths from stable GT commands.

    A held object prevents fully closed joints. Use the demonstrated grasp width,
    never the midpoint of the PT joint quantiles, to color Coffee's state readout.
    This reference is only for visualization and never changes policy inputs.
    """
    from pathlib import Path
    import pyarrow as pa
    import pyarrow.parquet as pq

    tables = [pq.read_table(path, filters=[("episode_index", "=", episode_index)],
                           columns=["frame_index", "action", "observation.state"])
              for path in sorted((Path(dataset_dir) / "data").glob("**/*.parquet"))]
    table = pa.concat_tables([part for part in tables if len(part)]).sort_by("frame_index")
    actions = np.asarray(table["action"].to_pylist())
    states = np.asarray(table["observation.state"].to_pylist())
    if states.shape[1] != 8:
        raise ValueError("Gripper visualization calibration expects two finger qpos in state[6:8].")
    width = np.abs(states[:, 6:8]).sum(-1)
    closed_command = actions[:, -1] > 0
    age = np.zeros(len(actions), dtype=int)
    for index in range(1, len(actions)):
        age[index] = age[index - 1] + 1 if closed_command[index] == closed_command[index - 1] else 0
    stable = age >= 8  # exclude the physical closing/opening transient
    opened = width[stable & ~closed_command]
    grasped = width[stable & closed_command]
    if not len(opened) or not len(grasped):
        raise ValueError("Reference episode needs stable open and close-command states.")
    open_width, grasp_width = float(np.median(opened)), float(np.median(grasped))
    if open_width - grasp_width <= 1e-4:
        raise ValueError("GT open and grasp widths cannot be distinguished.")
    return dict(open_width_m=open_width, grasp_width_m=grasp_width,
                episode_index=episode_index, stable_command_steps=8)


def fullskill_gripper_from_trace(
    trace, *, batch_index: int, n_video_frames: int, video_frame_stride: int,
    gripper_reference: dict | None = None,
) -> np.ndarray | None:
    """Latest plan's command or finger opening; missing historical values stay unknown."""
    records = sorted(
        (record for record in trace if int(record.get("batch_index", 0)) == batch_index),
        key=lambda record: int(record.get("episode_timestep", 0)),
    )
    sample = next((point["gripper_open_fraction"] for record in records
                   for point in record.get("fullskill_plans", [])
                   if "gripper_open_fraction" in point), None)
    if sample is None:
        return None
    result = np.full((n_video_frames, len(sample)), np.nan)
    for record_index, record in enumerate(records):
        start = int(record.get("episode_timestep", 0))
        end = (int(records[record_index + 1].get("episode_timestep", 0))
               if record_index + 1 < len(records) else float("inf"))
        points = sorted(record.get("fullskill_plans", []), key=lambda point: int(point["episode_timestep"]))
        point_index = -1
        for frame_index in range(n_video_frames):
            timestep = frame_index * max(1, int(video_frame_stride))
            if not start <= timestep < end:
                continue
            while point_index + 1 < len(points) and int(points[point_index + 1]["episode_timestep"]) <= timestep:
                point_index += 1
            if point_index >= 0:
                point = points[point_index]
                if "gripper_action" in point:
                    result[frame_index] = np.clip((1 - np.asarray(point["gripper_action"])) / 2, 0, 1)
                elif gripper_reference is not None and "gripper_qpos" in point:
                    width = np.abs(point["gripper_qpos"]).sum(-1)
                    result[frame_index] = np.clip(
                        (width - gripper_reference["grasp_width_m"]) /
                        (gripper_reference["open_width_m"] - gripper_reference["grasp_width_m"]), 0, 1,
                    )
                elif gripper_reference is None and "gripper_open_fraction" in point:
                    result[frame_index] = point["gripper_open_fraction"]
    return result


def draw_fullskill(image, initial, current, *, gripper_open_fraction=None) -> None:
    """Draw only the latest plan, blue=open/red=closed; unknown state is gray."""
    from PIL import ImageDraw  # noqa: PLC0415

    draw = ImageDraw.Draw(image)
    pixels = current if current is not None else initial
    if pixels is None:
        return
    points = np.asarray(pixels)
    opening = (np.full(len(points), np.nan) if gripper_open_fraction is None
               else np.asarray(gripper_open_fraction))
    if opening.shape != (len(points),):
        raise ValueError("Fullskill gripper values must match the trajectory waypoints.")
    colors = [(160, 170, 185) if not np.isfinite(value) else
              (45, 125, 255) if value > .5 else (245, 65, 65) for value in opening]
    for index, (start, end) in enumerate(zip(points[:-1], points[1:], strict=True)):
        if np.isfinite(start).all() and np.isfinite(end).all():
            draw.line([tuple(start), tuple(end)], fill=colors[index], width=3)
    for (x, y), color in zip(points, colors, strict=True):
        if np.isfinite((x, y)).all():
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)


def _finish(
    columns: np.ndarray,
    rows: np.ndarray,
    depth: np.ndarray,
    *,
    height: int,
    width: int,
    mirror: bool = True,
) -> GoalMarks:
    """``mirror`` is False when the caller's projection already applied it."""
    if mirror and MIRRORED_IN_X:
        columns = (width - 1) - columns
    visible = (
        np.isfinite(depth)
        & (depth > 1e-8)
        & np.isfinite(columns)
        & np.isfinite(rows)
        & (columns >= 0.0)
        & (columns <= width - 1.0)
        & (rows >= 0.0)
        & (rows <= height - 1.0)
    )
    pixels = np.stack([columns, rows], axis=-1)
    pixels[~visible] = np.nan
    return GoalMarks(pixels=pixels, visible=visible)


def agent_view_transform(sim, *, height: int, width: int, camera: str = AGENT_CAMERA) -> np.ndarray:
    """``intrinsic @ world_to_camera`` for a world-fixed camera, read from the live model."""
    index = int(sim.model.camera_name2id(camera))
    position = np.asarray(sim.model.cam_pos[index], dtype=np.float64)
    quaternion = np.asarray(sim.model.cam_quat[index], dtype=np.float64)
    fovy = float(sim.model.cam_fovy[index])

    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    # MuJoCo cameras look down their own -z with +y up; the renderer flips both.
    camera_to_world = rotation @ np.diag([1.0, -1.0, -1.0])
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = camera_to_world.T
    world_to_camera[:3, 3] = -camera_to_world.T @ position

    focal = 0.5 * height / np.tan(np.deg2rad(fovy) / 2.0)
    intrinsic = np.eye(4, dtype=np.float64)
    intrinsic[:3, :3] = [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]
    return intrinsic @ world_to_camera


def marks_in_agent_view(
    transform: np.ndarray,
    goals: np.ndarray,
    grounding: np.ndarray | None,
    *,
    height: int,
    width: int,
) -> GoalMarks:
    """Project GROUNDED goals ``[frames, 3]`` into the world-fixed camera.

    ``grounding`` is the episode-start xyz this run subtracted; it has to go back on, because the
    camera does not move with the episode. None means the run was not grounded.
    """
    world = np.asarray(goals, dtype=np.float64)[:, :3].copy()
    if grounding is not None:
        world += np.asarray(grounding, dtype=np.float64).reshape(-1)[:3]
    points = np.concatenate([world, np.ones((world.shape[0], 1))], axis=1)
    projected = points @ np.asarray(transform, dtype=np.float64).T
    depth = projected[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        columns = projected[:, 0] / depth
        rows = projected[:, 1] / depth
    return _finish(columns, rows, depth, height=height, width=width)


def marks_in_wrist_view(
    states: np.ndarray, goals: np.ndarray, *, height: int, width: int, mirror_x: bool = True
) -> GoalMarks:
    """Project goals into the gripper camera from each frame's EEF pose.

    ``states`` is ``[frames, >=6]`` of ``observation.state`` (xyz + axis-angle). States and goals
    must share a frame; grounded or world does not matter, because the camera rides the same body
    the goal is measured against. Reuses the projection the alignment target is built with.
    """
    states = np.asarray(states, dtype=np.float64)
    if states.shape[1] < 6:
        raise ValueError(f"states need xyz + axis-angle, got {states.shape[1]} columns.")
    pixels, depth = project_into_wrist(
        torch.as_tensor(states[:, :3], dtype=torch.float32),
        torch.as_tensor(states[:, 3:6], dtype=torch.float32),
        torch.as_tensor(np.asarray(goals, dtype=np.float64)[:, :3], dtype=torch.float32),
        WristCamera(mirror_x=mirror_x),
        height=height,
        width=width,
    )
    columns, rows = pixels[:, 0].numpy().astype(np.float64), pixels[:, 1].numpy().astype(np.float64)
    # WristCamera.mirror_x already put these in the rendered orientation; mirroring again would
    # silently undo it and leave the dot mirrored about the image centre.
    return _finish(
        columns, rows, depth.numpy().astype(np.float64), height=height, width=width, mirror=False
    )


def draw_goal(
    image, pixel, *, color: tuple[int, int, int], scale: float = 1.0,
    marker: str = "cross",
) -> None:
    """Draw one crosshair-and-ring marker in place on a PIL image; NaN pixels draw nothing.

    The same shape the wrist probe uses, so an eval frame and a probe frame read alike.
    """
    if pixel is None or not np.all(np.isfinite(pixel)):
        return
    from PIL import ImageDraw  # noqa: PLC0415 - only needed when something is drawn

    draw = ImageDraw.Draw(image)
    x, y = float(pixel[0]) * scale, float(pixel[1]) * scale
    arm, ring = 9.0 * scale, 4.0 * scale
    if marker == "circle":
        ring = 7.0 * scale
        draw.ellipse([(x - ring, y - ring), (x + ring, y + ring)], outline=color, width=2)
        draw.ellipse([(x - 1, y - 1), (x + 1, y + 1)], fill=color)
        return
    draw.line([(x - arm, y), (x + arm, y)], fill=color, width=2)
    draw.line([(x, y - arm), (x, y + arm)], fill=color, width=2)
    draw.ellipse([(x - ring, y - ring), (x + ring, y + ring)], outline=color, width=2)
