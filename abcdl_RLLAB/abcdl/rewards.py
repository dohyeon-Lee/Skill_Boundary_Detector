"""Per-frame success / reward / Monte-Carlo return for an episode.

These are stored as abcdl ``frame_features`` and surfaced as dataset item keys, so a
recorded teleop episode carries the RL signals a policy/value learner needs.
"""

from __future__ import annotations

import numpy as np

REWARD_MODES = ("sparse", "step")


def compute_frame_features(num_steps: int, success: bool, *, mode: str = "sparse",
                           discount: float = 0.99) -> dict:
    """Return ``{"success", "reward", "mc_return"}`` per-frame arrays of length *num_steps*.

    - ``success``  : terminal success flag — ``[0, …, 0, 1]`` for a successful episode,
      all-zero for a failed one (1 only on the last frame).
    - ``reward``   :
        * ``"sparse"`` (default): 0 everywhere, ``+1`` on the last frame iff *success*.
        * ``"step"``  : ``-1`` per step, ``0`` on the last frame iff *success* (reach the
          goal → stop paying the step penalty; a failed episode pays ``-1`` throughout).
    - ``mc_return``: discounted return-to-go ``G[t] = r[t] + γ·G[t+1]`` with *discount* γ.
    """
    if mode not in REWARD_MODES:
        raise ValueError(f"reward mode must be one of {REWARD_MODES}, got {mode!r}")
    T = int(num_steps)
    success_flag = np.zeros(T, np.float32)
    if T > 0 and success:
        success_flag[-1] = 1.0

    if mode == "sparse":
        reward = np.zeros(T, np.float32)
        if T > 0 and success:
            reward[-1] = 1.0
    else:  # "step"
        reward = -np.ones(T, np.float32)
        if T > 0 and success:
            reward[-1] = 0.0

    mc_return = np.zeros(T, np.float32)
    acc = 0.0
    for t in range(T - 1, -1, -1):
        acc = float(reward[t]) + discount * acc
        mc_return[t] = acc

    return {"success": success_flag, "reward": reward, "mc_return": mc_return}
