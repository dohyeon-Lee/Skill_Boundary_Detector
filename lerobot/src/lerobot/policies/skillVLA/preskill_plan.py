"""Observation contract for non-recursive, detached previous-skill plan memory.

States/endpoints are episode-grounded raw proprio, except STATE_NORMALIZED,
which uses the policy's saved observation.state processor statistics.
"""

IMAGE = "preskill_image"
WRIST_IMAGE = "preskill_wrist_image"
STATE = "preskill_state"
STATE_NORMALIZED = "preskill_state_normalized"
START_STATE = "preskill_start_state"
END_STATE = "preskill_end_state"
FOCUS_UV = "preskill_focus_uv"
FOCUS_VALID = "preskill_focus_valid"
CODE = "preskill_code"
VALID = "preskill_valid"
FRAME = "preskill_frame"
CACHE_ID = "preskill_cache_id"
KEYS = (IMAGE, WRIST_IMAGE, STATE, STATE_NORMALIZED, START_STATE, END_STATE,
        CODE, VALID, FRAME, CACHE_ID, FOCUS_UV, FOCUS_VALID)


def previous_observation_frame(skill_rank, causal_start_frame):
    """One frame before the SAME causal jitter draw; never cross episode BOS."""
    return causal_start_frame - 1 if skill_rank > 0 and causal_start_frame > 0 else -1
