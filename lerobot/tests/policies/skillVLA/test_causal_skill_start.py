from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.skillVLA.dataset_skillVLA import SkillVLADataset
from lerobot.policies.skillVLA.skill_jitter import causal_skill_start


@pytest.mark.parametrize("gt,offset,current,length,expected", [
    (100, 10, 105, 200, (105, 5)),
    (100, 10, 120, 200, (110, 10)),
    (100, -10, 105, 200, (90, -10)),
    (0, -10, 0, 200, (0, 0)),
    (110, -10, 105, 200, (100, -10)),
    (10, 20, 15, 200, (15, 5)),
    (190, 20, 199, 200, (199, 9)),
])
def test_start_observation_is_causal_and_episode_bounded(gt, offset, current, length, expected):
    assert causal_skill_start(gt, offset, current, length) == expected


@pytest.mark.parametrize("current,length", [(-1, 10), (10, 10), (0, 0)])
def test_causal_start_rejects_invalid_episode_coordinates(current, length):
    with pytest.raises(ValueError):
        causal_skill_start(0, 0, current, length)


def _fake_dataset(monkeypatch, frame=15):
    dataset = object.__new__(SkillVLADataset)
    dataset.meta = SimpleNamespace(fps=20, episodes={0: {"length": 60}})
    dataset._terminator_post_end_steps = 0
    dataset._include_terminator_start_inputs = False
    dataset._include_terminator_predictor_inputs = False
    dataset._include_terminator_goal_target = False
    dataset._terminator_history_mask_padding = False
    dataset._include_predictor_start_inputs = True
    dataset._include_skill_end_xyz_target = False
    dataset._include_skill_end_state_target = True
    dataset._include_canonical_skill_actions = True
    dataset._include_chunk_end_state_target = False
    dataset._storage_pmax = 20
    dataset._pmax = 20
    dataset._foveated_vision = SimpleNamespace(enabled=False, randomization_enabled=False)
    row = {"frame_index": torch.tensor(frame), "episode_index": torch.tensor(0),
           "skill_index": torch.tensor(1), "skill_ds": torch.tensor(frame - 10),
           "skill_de": torch.tensor(39 - frame), "skill_sequence_len": torch.tensor(3),
           "skill_sequence": torch.tensor([2, 3, 27]),
           "skill_initial_frame": torch.tensor([0, 10, -1]),
           "skill_length_sequence": torch.tensor([10, 30, 0]),
           "action": torch.zeros(7), "observation.state": torch.full((8,), float(frame))}
    rows = [deepcopy(row) for _ in range(60)]
    monkeypatch.setattr(LeRobotDataset, "hf_dataset", property(lambda self: rows))
    monkeypatch.setattr(LeRobotDataset, "__getitem__", lambda self, index: deepcopy(row))
    requested = []

    def query_videos(timestamps, episode):
        assert episode == 0
        requested.append(timestamps)
        return {camera: torch.full((3, 2, 2), round(times[0] * 20))
                for camera, times in timestamps.items()}

    dataset._ensure_reader = lambda: SimpleNamespace(_query_videos=query_videos, _image_transforms=None)
    dataset._iss = SimpleNamespace(pmax=20, state=lambda ep, rank, index, gt:
                                  np.full(8, max(0, gt + index - 20), dtype=np.float32))
    dataset._focus_uv = SimpleNamespace(
        target_state=lambda ep, rank, gt: (np.full(8, 999, dtype=np.float32), True),
        target=lambda ep, rank, gt: (np.zeros(2, np.float32), np.zeros(2, np.int32), True, False))
    canonical_starts = []

    def canonical(**kwargs):
        canonical_starts.append(kwargs["skill_start_frame"])
        return torch.ones(40, 7), torch.zeros(40, dtype=torch.bool)

    dataset._canonical_skill_actions = canonical
    return dataset, requested, canonical_starts


def test_dataset_clips_images_and_start_xyz_together_without_changing_targets(monkeypatch):
    dataset, requested, canonical_starts = _fake_dataset(monkeypatch)
    # Sampler picks the current skill, starting +10 after its GT start at 10.
    item = dataset[(15, -1, False, 1, 10)]
    assert item["skill_start_requested_frame"].item() == 20
    assert item["skill_start_frame"].item() == 15
    assert all(value == [15 / 20] for value in requested[0].values())
    assert item["skill_start_image"].eq(15).all()
    assert item["skill_start_wrist_image"].eq(15).all()
    assert item["skill_start_state"].eq(15).all()
    assert item["observation.state"].eq(15).all()
    assert item["skill_code"].item() == 3
    assert item["skill_previous_code"].item() == 2
    assert item["skill_end_state"].eq(999).all()
    assert canonical_starts == [10]  # Canonical trajectory is NOT re-anchored.
    assert item["skill_effective_de"].item() == 24
    torch.testing.assert_close(item["skill_start_focus_uv"], item["skill_focus_uv"])
    assert item["skill_start_focus_uv"].data_ptr() != item["skill_focus_uv"].data_ptr()
    assert item["skill_start_focus_valid"].item()


def test_clipping_start_does_not_change_the_sampled_late_transition_end(monkeypatch):
    dataset, _, canonical_starts = _fake_dataset(monkeypatch)
    # A deliberately short previous skill: requested previous start 0+20 is
    # future, but its delayed end is still governed by the original +20 draw.
    item = dataset[(15, -1, False, 0, 20)]
    assert item["skill_start_frame"].item() == 15
    assert item["skill_start_state"].eq(15).all()
    assert item["skill_code"].item() == 2
    assert item["skill_previous_code"].item() == -1
    assert item["skill_effective_de"].item() == 20 - 1 - 5
    assert canonical_starts == [0]


def test_causal_frame_audit_fields_survive_the_policy_processor_roundtrip():
    from lerobot.policies.skill_expert.processor_skill_expert import (
        skill_expert_batch_to_transition, skill_expert_transition_to_batch,
    )

    batch = {"observation.state": torch.zeros(1, 8),
             "skill_previous_code": torch.tensor([-1]),
             "skill_start_frame": torch.tensor([15]),
             "skill_start_cache_id": torch.tensor([2]),
             "skill_start_focus_uv": torch.tensor([[.2, -.3]]),
             "skill_start_focus_valid": torch.tensor([True]),
             "skill_start_requested_frame": torch.tensor([20])}
    restored = skill_expert_transition_to_batch(skill_expert_batch_to_transition(batch))
    for key in batch:
        torch.testing.assert_close(restored[key], batch[key])


@pytest.mark.parametrize("skill_index,offset,expected", [(1, -5, 2), (1, 8, 2), (0, 20, -1)])
def test_previous_code_follows_jittered_occurrence_not_current_frame(
    monkeypatch, skill_index, offset, expected,
):
    dataset, _, _ = _fake_dataset(monkeypatch)
    item = dataset[(15, -1, False, skill_index, offset)]
    assert item["skill_previous_code"].item() == expected
    assert item["skill_code"].item() == [2, 3][skill_index]
