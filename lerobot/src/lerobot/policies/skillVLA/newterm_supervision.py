"""Frame-preserving post-end supervision for standalone Newterm training."""

import numpy as np
import torch


POST_END_CODES = "terminator_post_end_codes"
POST_END_VALID = "terminator_post_end_valid"
POST_END_START_STATE = "terminator_post_end_start_state"
POST_END_GOAL_XYZ = "terminator_post_end_goal_xyz"
POST_END_GOAL_VALID = "terminator_post_end_goal_valid"
HISTORY_VALID = "terminator_history_valid"
POST_END_HISTORY_VALID = "terminator_post_end_history_valid"


def history_valid_mask(frame, skill_start, length, *, reset_on_skill=True):
    """Validity for a right-aligned one-frame-stride history, not repeated padding."""
    if length <= 0 or skill_start < 0 or frame < skill_start:
        raise ValueError("History requires a positive length and a reached skill start.")
    start = int(skill_start) if reset_on_skill else 0
    return torch.arange(int(frame) - length + 1, int(frame) + 1) >= start


def post_end_conditions(
    frame, skill_index, starts, codes, steps, *, terminal_repeat=False, return_skill_indices=False
):
    """Return all past skills whose inclusive endpoints are within k frames.

    Fixed k slots allow ordinary collation. Exclude past conditions with the
    current code, retaining the common Newterm supervision rule. Optional
    occurrence indices select the correct past start/goal for Newterm4--7.
    """
    result = np.zeros(steps, dtype=np.int64)
    valid = np.zeros(steps, dtype=np.bool_)
    skill_indices = np.full(steps, -1, dtype=np.int64)
    if terminal_repeat or skill_index == 0:
        output = (torch.from_numpy(result), torch.from_numpy(valid))
        return (*output, skill_indices) if return_skill_indices else output
    count = 0
    for previous in range(skill_index - 1, -1, -1):
        end = int(starts[previous + 1]) - 1
        distance = int(frame) - end
        if distance > steps:
            break
        if distance >= 1 and int(codes[previous]) != int(codes[skill_index]):
            result[count] = int(codes[previous])
            valid[count] = True
            skill_indices[count] = previous
            count += 1
    output = (torch.from_numpy(result), torch.from_numpy(valid))
    return (*output, skill_indices) if return_skill_indices else output


class NewTermFrameSampler:
    """Shuffle every real frame once plus k copies of each episode's last frame.

    Interior frames are never duplicated. A tagged endpoint index prevents
    repeated terminal positives from also repeating earlier-skill losses.
    HF row indices remain local, including when episodes have been filtered.
    """

    def __init__(self, dataset, steps, *, shuffle=True):
        if steps <= 0:
            raise ValueError("Newterm sampling requires post_end_steps > 0.")
        episodes = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
        if not len(episodes):
            raise ValueError("Newterm dataset has no frames.")
        endpoints = np.flatnonzero(np.r_[episodes[1:] != episodes[:-1], True])
        self.indices = list(range(len(episodes)))
        self.indices.extend((int(index), True) for index in endpoints for _ in range(steps))
        self.shuffle = shuffle

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        order = torch.randperm(len(self)) if self.shuffle else range(len(self))
        for index in order:
            yield self.indices[int(index)]
