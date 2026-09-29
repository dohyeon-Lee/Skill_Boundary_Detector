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
    states: np.ndarray, goals: np.ndarray, *, height: int, width: int
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
        WristCamera(),
        height=height,
        width=width,
    )
    columns, rows = pixels[:, 0].numpy().astype(np.float64), pixels[:, 1].numpy().astype(np.float64)
    # WristCamera.mirror_x already put these in the rendered orientation; mirroring again would
    # silently undo it and leave the dot mirrored about the image centre.
    return _finish(
        columns, rows, depth.numpy().astype(np.float64), height=height, width=width, mirror=False
    )


def draw_goal(image, pixel, *, color: tuple[int, int, int], scale: float = 1.0) -> None:
    """Draw one crosshair-and-ring marker in place on a PIL image; NaN pixels draw nothing.

    The same shape the wrist probe uses, so an eval frame and a probe frame read alike.
    """
    if pixel is None or not np.all(np.isfinite(pixel)):
        return
    from PIL import ImageDraw  # noqa: PLC0415 - only needed when something is drawn

    draw = ImageDraw.Draw(image)
    x, y = float(pixel[0]) * scale, float(pixel[1]) * scale
    arm, ring = 9.0 * scale, 4.0 * scale
    draw.line([(x - arm, y), (x + arm, y)], fill=color, width=2)
    draw.line([(x, y - arm), (x, y + arm)], fill=color, width=2)
    draw.ellipse([(x - ring, y - ring), (x + ring, y + ring)], outline=color, width=2)
