"""Recorded DROID camera poses, explicit pixel transforms, and pinhole projection.

Pose6 convention follows droid.misc.transformations: XYZ translation in metres,
SciPy extrinsic/lowercase 'xyz' Euler angles in radians. T_base_camera maps
camera coordinates into the robot base frame; projection uses its inverse.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def pose6_to_matrix(pose):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape[-1:] != (6,) or not np.isfinite(pose).all():
        raise ValueError('Expected finite (...,6) DROID XYZ/Euler pose')
    shape = pose.shape[:-1]
    matrix = np.broadcast_to(np.eye(4), (*shape, 4, 4)).copy()
    matrix[..., :3, :3] = Rotation.from_euler('xyz', pose[..., 3:].reshape(-1, 3)).as_matrix().reshape(*shape, 3, 3)
    matrix[..., :3, 3] = pose[..., :3]
    return matrix


def pose6_to_rotvec(pose):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 6 or not np.isfinite(pose).all():
        raise ValueError('Expected finite Nx6 DROID XYZ/Euler pose')
    return np.concatenate([pose[:, :3], Rotation.from_euler('xyz', pose[:, 3:]).as_rotvec()], axis=1).astype(np.float32)


def resize_geometry(source_width, source_height, output_size, mode='letterbox'):
    """Return resized H/W, top/left padding and the exact pixel-centre map.

    Match OpenPI's integer truncation and centred black padding. Intrinsics and
    RGB encoding must use this same discrete resize, including odd padding.
    """
    height, width = output_size
    if min(source_width, source_height, height, width) <= 0:
        raise ValueError('Image dimensions must be positive')
    if mode == 'letterbox':
        ratio = max(source_width / width, source_height / height)
        resized_height = max(1, int(source_height / ratio))
        resized_width = max(1, int(source_width / ratio))
    elif mode == 'stretch':
        resized_height, resized_width = height, width
    else:
        raise ValueError('image_transform must be letterbox or stretch')
    top, left = (height - resized_height) // 2, (width - resized_width) // 2
    sx, sy = resized_width / source_width, resized_height / source_height
    transform = np.array([[sx, 0, left + (sx-1)/2],
                          [0, sy, top + (sy-1)/2], [0, 0, 1]], dtype=np.float64)
    return resized_height, resized_width, top, left, transform


def intrinsics(entry, source_width, source_height, output_size, mode='letterbox'):
    """KarlP DROID cameraMatrix packs [fx,cx,fy,cy], NOT [fx,fy,cx,cy]."""
    values = np.asarray(entry['cameraMatrix'], dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all() or values[0] <= 0 or values[2] <= 0:
        raise ValueError('Expected positive focal lengths in DROID [fx,cx,fy,cy]')
    if (int(entry['width']), int(entry['height'])) != (source_width, source_height):
        raise ValueError('Calibration resolution differs from source video; refusing implicit crop/stereo assumptions')
    distortion = np.asarray(entry.get('distCoeffs', []), dtype=np.float64)
    if not np.isfinite(distortion).all() or np.any(np.abs(distortion) > 1e-10):
        raise ValueError('Nonzero distortion needs explicit undistortion before pinhole UV projection')
    fx, cx, fy, cy = values
    original = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    *_, pixel_transform = resize_geometry(source_width, source_height, output_size, mode)
    return original, pixel_transform @ original, pixel_transform


def project(points_base, camera_to_base, matrix):
    points_base = np.asarray(points_base, dtype=np.float64)
    transform = np.asarray(camera_to_base, dtype=np.float64)
    homogeneous = np.concatenate([points_base, np.ones((*points_base.shape[:-1], 1))], axis=-1)
    points_camera = np.einsum('...ij,...j->...i', np.linalg.inv(transform), homogeneous)[..., :3]
    pixels = np.einsum('...ij,...j->...i', matrix, points_camera)
    depth = points_camera[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = pixels[..., :2] / pixels[..., 2:3]
    return uv, depth
