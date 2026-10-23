"""Top-camera goal coordinates for Terminators (metre noise precedes projection)."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


TOP_CAMERA = "terminator_top_camera_projection"


def normalized_top_camera_projection(transform, *, height: int, width: int) -> np.ndarray:
    """Grounded XYZ -> homogeneous centered UV, including the stored horizontal mirror."""
    if min(height, width) < 2:
        raise ValueError("Top-camera image dimensions must be >=2.")
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Top camera needs a finite 4x4 grounded projection matrix.")
    return np.stack((matrix[2] - 2 * matrix[0] / (width - 1),
                     2 * matrix[1] / (height - 1) - matrix[2], matrix[2])).astype(np.float32)


def project_goal_uv(xyz: Tensor, projection: Tensor) -> tuple[Tensor, Tensor]:
    """Project already-perturbed XYZ; never sample noise in image coordinates.

    Projection is in float32 even under AMP. Out-of-view coordinates follow the
    existing focus-UV artifact's clipping convention. Behind-camera/nonfinite
    rows get a finite neutral token, never a division-by-zero or XYZ fallback.
    """
    if xyz.ndim != 2 or xyz.shape[-1] != 3 or projection.shape != (xyz.shape[0], 3, 4):
        raise ValueError("UV projection expects XYZ (B,3) and camera matrices (B,3,4).")
    with torch.autocast(device_type=xyz.device.type, enabled=False):
        homogeneous = torch.cat((xyz.float(), torch.ones_like(xyz[..., :1]).float()), dim=-1)
        projected = torch.bmm(projection.float(), homogeneous.unsqueeze(-1)).squeeze(-1)
        depth = projected[:, 2:3]
        finite = torch.isfinite(projected).all(-1) & (depth[:, 0] > 1e-8)
        uv = projected[:, :2] / torch.where(finite[:, None], depth, torch.ones_like(depth))
        in_view = finite & (uv.abs() <= 1).all(-1)
        uv = torch.where(finite[:, None], uv.clamp(-1, 1), torch.zeros_like(uv))
    return uv, in_view
