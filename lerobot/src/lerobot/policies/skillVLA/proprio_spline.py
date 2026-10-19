"""Fixed normalized-time targets using the original FSQ spline convention.

Interpolate the six pose axes cubically and the two gripper axes linearly.
These are sampled trajectory VALUES, not B-spline control coefficients. Keep
episode grounding unchanged; do not subtract the selected skill's first state.
"""

import numpy as np
from scipy.interpolate import make_interp_spline


def resample_skill_proprio(trajectory, tokens: int = 30) -> np.ndarray:
    values = np.asarray(trajectory, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 8 or len(values) == 0:
        raise ValueError(f"Expected a non-empty (time, 8) proprio trajectory, got {values.shape}.")
    if tokens < 2 or not np.isfinite(values).all():
        raise ValueError("Spline targets need >=2 tokens and finite proprio values.")
    if len(values) == 1:
        return np.repeat(values, tokens, axis=0)
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, tokens)
    result = np.empty((tokens, 8), dtype=np.float32)
    for axis in range(8):
        degree = min(1 if axis >= 6 else 3, len(values) - 1)
        result[:, axis] = make_interp_spline(source, values[:, axis], k=degree)(target)
    return result
