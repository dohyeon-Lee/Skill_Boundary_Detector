from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.skillVLA.newterm_supervision import NewTermFrameSampler, post_end_conditions


@pytest.mark.parametrize("frame,expected", [(20, []), (21, [4]), (30, [4]), (31, [])])
def test_post_end_window_uses_inclusive_endpoint(frame, expected):
    codes, valid = post_end_conditions(frame, int(frame >= 21), [0, 21], [4, 7], 10)
    assert codes[valid].tolist() == expected


def test_short_skills_keep_all_past_conditions_without_future_or_episode_leak():
    codes, valid = post_end_conditions(25, 2, [0, 21, 24], [4, 7, 8], 10)
    assert codes[valid].tolist() == [7, 4]
    _, first_valid = post_end_conditions(0, 0, [0, 21], [4, 7], 10)
    assert not first_valid.any()
    _, repeat_valid = post_end_conditions(25, 2, [0, 21, 24], [4, 7, 8], 10, terminal_repeat=True)
    assert not repeat_valid.any()


def test_recurrent_code_excludes_only_conflicting_past_condition():
    codes, valid = post_end_conditions(25, 2, [0, 21, 24], [4, 7, 4], 10)
    assert codes[valid].tolist() == [7]


def test_past_start_indices_follow_conditions_and_ignore_eos_padding():
    codes, valid, indices = post_end_conditions(
        25, 2, [0, 21, 24, -1], [4, 7, 8, 27], 10, return_skill_indices=True
    )
    assert codes[valid].tolist() == [7, 4]
    assert indices[valid.numpy()].tolist() == [1, 0]
    assert (indices[~valid.numpy()] == -1).all()


def test_only_episode_final_frames_are_repeated_in_sampler():
    # Noncontiguous selected episode IDs; indices must stay HF-local.
    dataset = SimpleNamespace(hf_dataset={"episode_index": [3, 3, 3, 8, 8]})
    indices = list(NewTermFrameSampler(dataset, 10, shuffle=False))
    assert indices[:5] == list(range(5))
    assert indices[5:] == [(2, True)] * 10 + [(4, True)] * 10
    assert len(indices) == 25
    torch.manual_seed(3)
    shuffled = list(NewTermFrameSampler(dataset, 10))
    assert sorted(map(str, shuffled)) == sorted(map(str, indices))
