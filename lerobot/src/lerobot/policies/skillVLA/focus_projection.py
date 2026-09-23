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


@dataclass(frozen=True)
class HandCamera:
    """A camera rigidly attached to the gripper, relative to the frame the EEF pose is recorded in.

    LIBERO's ``observation.state`` mixes two frames: the position is the grip site, the rotation is
    the hand body (they differ by 90 deg about z). ``rotation_frame`` therefore names the frame the
    recorded rotation belongs to, and both fields below are expressed in it.
    """

    rotation: np.ndarray            # 3x3, rotation frame -> MuJoCo camera frame
    position: np.ndarray            # 3, camera origin relative to the EEF site, in the rotation frame
    fovy: float


def _local_transform(element: ET.Element) -> np.ndarray:
    """The element's own pos/quat as a 4x4 (MuJoCo defaults: origin, identity)."""
    for attribute in ("euler", "axisangle", "xyaxes", "zaxis"):
        if element.get(attribute) is not None:
            raise ValueError(
                f"<{element.tag} name={element.get('name')!r}> uses {attribute}=; "
                "only pos/quat are supported."
            )
    transform = np.eye(4, dtype=np.float64)
    position = np.fromstring(element.get("pos", "0 0 0"), sep=" ", dtype=np.float64)
    if position.shape != (3,):
        raise ValueError(f"<{element.tag}> pos must have three values, got {position}.")
    quaternion = element.get("quat")
    if quaternion is not None:
        transform[:3, :3] = _quat_wxyz_to_matrix(
            np.fromstring(quaternion, sep=" ", dtype=np.float64)
        )
    transform[:3, 3] = position
    return transform


def _chain_to(parents: dict, element: ET.Element) -> list[ET.Element]:
    chain = [element]
    while element in parents:
        element = parents[element]
        chain.append(element)
    return list(reversed(chain))


def hand_camera_from_recorded_xml(
    model_xml: str, *, camera_name: str, eef_site_name: str, rotation_frame: str | None = None
) -> HandCamera:
    """Constant EEF->camera transform of a gripper-mounted camera (LIBERO: ``eye_in_hand``).

    The camera, the EEF site and ``rotation_frame`` (a body, default: the site itself) hang off the
    same hand body with no joint between them, so their relative pose is a model constant: walking
    the static ``pos``/``quat`` chain of each gives the transform without compiling MuJoCo. Unlike a
    world camera the pose is not usable on its own; combine it with the per-frame EEF pose through
    ``camera_transform_from_hand_pose``.
    """
    root = ET.fromstring(model_xml)
    parents = {child: parent for parent in root.iter() for child in parent}
    wanted = {"camera": (camera_name, "camera"), "site": (eef_site_name, "site")}
    if rotation_frame is not None:
        wanted["rotation"] = (rotation_frame, "body")
    found = {}
    for key, (name, tag) in wanted.items():
        elements = [e for e in root.iter(tag) if e.get("name") == name]
        if len(elements) != 1:
            raise ValueError(f"Recorded XML needs exactly one {tag} {name!r}; found {len(elements)}.")
        found[key] = elements[0]
    camera, site = found["camera"], found["site"]
    chains = {key: _chain_to(parents, element) for key, element in found.items()}
    camera_chain, site_chain = chains["camera"], chains["site"]
    shared = 0
    while all(shared < len(chain) for chain in chains.values()) and all(
        chain[shared] is camera_chain[shared] for chain in chains.values()
    ):
        shared += 1
    for chain in chains.values():
        for element in chain[shared:]:
            if any(child.tag in {"joint", "freejoint"} for child in element):
                raise ValueError(
                    f"<{element.tag} name={element.get('name')!r}> has a joint between the EEF site "
                    f"and {camera_name!r}; their relative pose is not a constant."
                )

    def compose(chain: list[ET.Element]) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        for element in chain[shared:]:
            transform = transform @ _local_transform(element)
        return transform

    to_camera, to_site = compose(camera_chain), compose(site_chain)
    to_rotation = compose(chains["rotation"]) if rotation_frame is not None else to_site
    return HandCamera(
        rotation=to_rotation[:3, :3].T @ to_camera[:3, :3],
        position=to_rotation[:3, :3].T @ (to_camera[:3, 3] - to_site[:3, 3]),
        fovy=float(camera.get("fovy", 45.0)),
    )


def camera_transform_from_hand_pose(
    hand_camera: HandCamera,
    eef_position: np.ndarray,
    eef_rotation: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """``camera_transform_from_recorded_xml`` for one frame of a gripper-mounted camera.

    ``eef_position`` is the world position of the EEF site and ``eef_rotation`` the world rotation
    of ``hand_camera``'s rotation frame; camera pose = that pose o the constant offset.
    """
    eef_rotation = np.asarray(eef_rotation, dtype=np.float64)
    if eef_rotation.shape != (3, 3):
        raise ValueError(f"eef_rotation must be 3x3, got {eef_rotation.shape}.")
    position = np.asarray(eef_position, dtype=np.float64) + eef_rotation @ hand_camera.position
    camera_rotation = eef_rotation @ hand_camera.rotation

    axis_correction = np.diag([1.0, -1.0, -1.0])
    camera_to_world_rotation = camera_rotation @ axis_correction
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = camera_to_world_rotation.T
    world_to_camera[:3, 3] = -camera_to_world_rotation.T @ position

    focal = 0.5 * height / np.tan(np.deg2rad(hand_camera.fovy) / 2.0)
    intrinsic = np.eye(4, dtype=np.float64)
    intrinsic[:3, :3] = np.asarray(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]
    )
    return intrinsic @ world_to_camera


def rotation_from_axis_angle(axis_angle: np.ndarray) -> np.ndarray:
    """Rodrigues rotation of a 3-vector whose norm is the angle (robosuite EEF orientation)."""
    vector = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = vector / angle
    cross = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return (
        np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)
    )


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
