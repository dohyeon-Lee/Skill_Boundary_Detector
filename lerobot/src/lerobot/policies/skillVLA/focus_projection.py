"""Project a robot EEF point into a fixed MuJoCo / robosuite camera image."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProjectedFocus:
    """Continuous/clipped image coordinates and their normalized target."""

    raw_xy: tuple[float, float]
    pixel_xy: tuple[int, int]
    normalized_xy: tuple[float, float]
    valid: bool
    clipped: bool


def _quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError(f"Camera quaternion must be four finite values, got {q}.")
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise ValueError("Camera quaternion has zero norm.")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def camera_transform_from_recorded_xml(
    model_xml: str, *, camera_name: str, height: int, width: int
) -> np.ndarray:
    """Match robosuite.get_camera_transform_matrix without compiling MuJoCo."""
    root = ET.fromstring(model_xml)
    matches: list[tuple[ET.Element, ET.Element]] = []
    for parent in root.iter():
        for child in parent:
            if child.tag == "camera" and child.get("name") == camera_name:
                matches.append((parent, child))
    if len(matches) != 1:
        raise ValueError(
            f"Recorded XML must contain exactly one camera {camera_name!r}; found {len(matches)}."
        )
    parent, camera = matches[0]
    if parent.tag != "worldbody":
        raise ValueError(
            f"Camera {camera_name!r} is attached to <{parent.tag}>; "
            "only fixed world cameras are supported."
        )
    if camera.get("pos") is None or camera.get("quat") is None:
        raise ValueError(
            f"Camera {camera_name!r} must explicitly record pos and quat."
        )
    position = np.fromstring(camera.get("pos", ""), sep=" ", dtype=np.float64)
    quaternion = np.fromstring(camera.get("quat", ""), sep=" ", dtype=np.float64)
    if position.shape != (3,):
        raise ValueError(f"Camera position must have three values, got {position}.")
    camera_rotation = _quat_wxyz_to_matrix(quaternion)

    # MuJoCo camera convention -> projected image convention used by robosuite.
    axis_correction = np.diag([1.0, -1.0, -1.0])
    camera_to_world_rotation = camera_rotation @ axis_correction
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = camera_to_world_rotation.T
    world_to_camera[:3, 3] = -camera_to_world_rotation.T @ position

    fovy = float(camera.get("fovy", 45.0))
    focal = 0.5 * height / np.tan(np.deg2rad(fovy) / 2.0)
    intrinsic = np.eye(4, dtype=np.float64)
    intrinsic[:3, :3] = np.asarray(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]
    )
    return intrinsic @ world_to_camera


def project_eef(
    xyz: np.ndarray, transform: np.ndarray, *, height: int, width: int
) -> ProjectedFocus:
    """Project EEF world xyz into the orientation stored by LIBERO videos."""
    homogeneous = np.concatenate(
        [np.asarray(xyz, dtype=np.float64), np.ones(1, dtype=np.float64)]
    )
    projected = np.asarray(transform, dtype=np.float64) @ homogeneous
    depth = float(projected[2])
    if not np.isfinite(depth) or depth <= 1e-8:
        raise ValueError(f"EEF point {xyz.tolist()} is behind the preview camera.")
    column = float(projected[0] / depth)
    row = float(projected[1] / depth)
    if not np.isfinite(column) or not np.isfinite(row):
        raise ValueError(f"EEF projection is non-finite for point {xyz.tolist()}.")

    # Stored LIBERO top images use the same display orientation as LiberoEnv.render.
    raw_x = float(width - 1 - column)
    raw_y = float(row)
    clipped_x = float(np.clip(raw_x, 0.0, width - 1.0))
    clipped_y = float(np.clip(raw_y, 0.0, height - 1.0))
    valid = 0.0 <= raw_x <= width - 1.0 and 0.0 <= raw_y <= height - 1.0
    norm_x = 2.0 * clipped_x / max(width - 1, 1) - 1.0
    norm_y = 2.0 * clipped_y / max(height - 1, 1) - 1.0
    return ProjectedFocus(
        raw_xy=(raw_x, raw_y),
        pixel_xy=(int(round(clipped_x)), int(round(clipped_y))),
        normalized_xy=(norm_x, norm_y),
        valid=valid,
        clipped=not valid,
    )
