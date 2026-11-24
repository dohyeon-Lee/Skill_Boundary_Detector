"""Fixed normalized-time targets using the original FSQ spline convention.

Interpolate continuous robot axes cubically and gripper axes linearly.
These are sampled trajectory VALUES, not B-spline control coefficients. Keep
episode grounding unchanged; do not subtract the selected skill's first state.

The legacy LIBERO representation is 8-D with grippers at axes 6 and 7. ABC is
14-D bimanual joint proprio with grippers at axes 6 and 13.  Callers may pass
the gripper axes explicitly; omitting them preserves the LIBERO convention.
"""

import numpy as np
from scipy.interpolate import make_interp_spline


def resample_skill_proprio(
    trajectory,
    tokens: int = 30,
    *,
    gripper_axes: tuple[int, ...] | None = None,
) -> np.ndarray:
    values = np.asarray(trajectory, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] == 0 or len(values) == 0:
        raise ValueError(f"Expected a non-empty (time, dim) proprio trajectory, got {values.shape}.")
    if tokens < 2 or not np.isfinite(values).all():
        raise ValueError("Spline targets need >=2 tokens and finite proprio values.")
    if len(values) == 1:
        return np.repeat(values, tokens, axis=0)
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, tokens)
    if gripper_axes is None:
        gripper_axes = (6, 7) if values.shape[1] == 8 else ()
    invalid = [axis for axis in gripper_axes if axis < 0 or axis >= values.shape[1]]
    if invalid:
        raise ValueError(f"gripper_axes {invalid} are outside proprio dimension {values.shape[1]}.")
    gripper_axes_set = set(gripper_axes)
    result = np.empty((tokens, values.shape[1]), dtype=np.float32)
    for axis in range(values.shape[1]):
        degree = min(1 if axis in gripper_axes_set else 3, len(values) - 1)
        result[:, axis] = make_interp_spline(source, values[:, axis], k=degree)(target)
    return result


def resample_skill_with_gripper_action(trajectory, gripper_action, tokens: int = 30) -> np.ndarray:
    """Pose6 spline + aligned gripper command, held until its next source frame.

    Commands use the dataset's action scale (-1=open, +1=close). Do not apply
    finger-position quantiles or interpolate between discrete commands.
    """
    states = np.asarray(trajectory, dtype=np.float32)
    commands = np.asarray(gripper_action, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 8 or commands.shape != (len(states),):
        raise ValueError("Mixed Bspline targets require aligned (time, 8) states and (time,) commands.")
    if not np.isfinite(commands).all() or np.any(np.abs(commands) > 1.00001):
        raise ValueError("Gripper commands must be finite raw actions in [-1, 1].")
    pose = resample_skill_proprio(states[:, :6], tokens, gripper_axes=())
    # Integer arithmetic avoids round-off advancing/delaying an exact frame boundary.
    indices = np.arange(tokens, dtype=np.int64) * (len(states) - 1) // (tokens - 1)
    return np.concatenate((pose, commands[indices, None]), axis=-1)
