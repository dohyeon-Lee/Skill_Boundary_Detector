"""GT end supervision for the skill selected by VSA transition randomization."""
import numpy as np
import torch


DISTANCE = "vlm_termination_distance"
TERMINAL = "vlm_termination_terminal"
PAST_CODES = "vlm_termination_past_codes"
PAST_VALID = "vlm_termination_past_valid"
PAST_UV = "vlm_termination_past_uv"
PAST_TERMINAL = "vlm_termination_past_terminal"
PAST_DISTANCE = "vlm_termination_past_distance"
KEYS = (DISTANCE, TERMINAL, PAST_CODES, PAST_VALID, PAST_UV, PAST_TERMINAL, PAST_DISTANCE)


def termination_conditions(frame, selected, starts, lengths, codes, steps, *, episode_length):
    """Signed distance and extra GT conditions, without jittering ends.

    A late-held skill is positive from its GT endpoint through endpoint+k.
    Older holds outside that window receive no loss, rather than a false
    negative. Keep the GT current skill even when VSA jitter selected another
    one. Extra conditions exclude the selected code just as Term1 excludes
    conflicting same-code positives. No future frame is decoded.
    """
    starts, lengths, codes = map(np.asarray, (starts, lengths, codes))
    end = int(starts[selected]) + int(lengths[selected]) - 1
    indices = np.full(steps + 2, -1, dtype=np.int64)
    count = 0
    for rank in range(len(starts) - 1, -1, -1):
        if starts[rank] < 0 or lengths[rank] <= 0 or int(codes[rank]) == int(codes[selected]):
            continue
        endpoint = int(starts[rank]) + int(lengths[rank]) - 1
        distance = frame - endpoint
        # Include the canonical current skill when the primary VSA condition
        # is an early next / late previous skill. Otherwise a late-held A
        # would lose the ordinary Term1 supervision of current skill B.
        is_current = int(starts[rank]) <= frame <= endpoint
        if is_current or 0 <= distance <= steps:
            if count == len(indices):
                break
            indices[count] = rank
            count += 1
    return end - frame, end == frame == episode_length - 1, indices


def termination_targets(distance, *, sigma, post_end_steps):
    """Gaussian pre-end target (Term1 default sigma=1), then k hard positives."""
    distance = distance.float()
    before = distance.clamp_min(0)
    target = (torch.exp(-before.square() / (2 * sigma**2))
              if sigma > 0 else (distance <= 0).float())
    return target, distance >= -post_end_steps
