from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.skillVLA.newterm_supervision import NewTermFrameSampler, post_end_conditions, history_valid_mask


@pytest.mark.parametrize("frame,start,count", [(0, 0, 1), (1, 0, 2), (35, 35, 1), (36, 35, 2), (60, 35, 20)])
def test_masked_history_only_includes_real_current_skill_frames(frame, start, count):
    mask = history_valid_mask(frame, start, 20)
    assert mask.tolist() == [False] * (20 - count) + [True] * count


def test_post_end_history_uses_past_skill_start_instead_of_current_start():
    current = history_valid_mask(25, 24, 20)
    past = history_valid_mask(25, 21, 20)
    assert current.sum() == 2
    assert past.sum() == 5
    assert history_valid_mask(25, 24, 20, reset_on_skill=False).all()


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


def test_joint_past_predictor_images_use_canonical_occurrence_anchors_only():
    from lerobot.policies.skillVLA.dataset_skillVLA import _post_end_predictor_images, CAM_3RD, CAM_WRIST
    import numpy as np
    calls = []
    def query(timestamps, episode):
        calls.append((timestamps, episode))
        return {key: torch.full((3, 2, 2), timestamps[key][0]) for key in timestamps}
    reader = SimpleNamespace(_query_videos=query, _image_transforms=None)
    templates = {key: torch.zeros(3, 2, 2) for key in (CAM_3RD, CAM_WRIST)}
    images = _post_end_predictor_images(reader, 7, [0, 21, 24], np.array([1, 0, -1]), 10, templates)
    assert [call[0][CAM_3RD] for call in calls] == [[2.1], [0.0]]
    assert all(call[1] == 7 for call in calls)
    assert images[CAM_3RD].shape == (3, 3, 2, 2)
    assert images[CAM_3RD][0].mean() == pytest.approx(2.1)
    assert not images[CAM_3RD][2].any()


def test_newterm_sampler_cannot_change_joint_vsa_predictor_sampling():
    import ast
    from pathlib import Path
    script = Path(__file__).resolve().parents[2] / "src/lerobot/scripts/lerobot_train.py"
    tree = ast.parse(script.read_text())
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and "terminator_post_end_steps" in ast.unparse(node.test))
    condition = compile(ast.Expression(branch.test), str(script), "eval")
    for joint, expected in ((False, True), (True, False)):
        cfg = SimpleNamespace(policy=SimpleNamespace(terminator_post_end_steps=10, newtask_joint_enabled=joint))
        assert eval(condition, {"cfg": cfg}) is expected
