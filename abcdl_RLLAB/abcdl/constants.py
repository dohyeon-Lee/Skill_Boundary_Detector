"""Fixed-clock and encoding constants for the abcdl format (paper Appendix B.1)."""

from __future__ import annotations

FPS = 30
TICK_NS = 33_333_333  # int(1e9 / 30)
TIMESCALE = 15_360
TICKS_PER_FRAME = 512  # TIMESCALE // FPS

# YAM bimanual default state/action layout: [L_arm(6), L_grip(1), R_arm(6), R_grip(1)].
YAM_ARM_DOF = 6
YAM_GRIPPER_DOF = 1
YAM_STATE_DIM = 2 * (YAM_ARM_DOF + YAM_GRIPPER_DOF)  # 14
YAM_ACTION_DIM = YAM_STATE_DIM
