import numpy as np

from lerobot.policies.skillVLA.same_skill_batch_sampler import SameSkillDifferentTaskBatchSampler


def test_partner_filter_uses_post_jitter_code_and_progress() -> None:
    sampler = object.__new__(SameSkillDifferentTaskBatchSampler)
    sampler.progress_candidates = 8
    sampler.progress_temperature = 0.1
    sampler._skill = np.asarray([5, 5, 5])
    sampler._task = np.asarray([0, 1, 2])
    sampler._progress = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    sampler._skill_index = np.asarray([0, 0, 0])
    sampler._ds = np.asarray([9, 9, 9])
    sampler._de = np.asarray([0, 0, 0])
    sampler._sequence_len = np.asarray([3, 3, 3])
    sampler._skill_sequence = np.asarray([[5, 7, 0], [5, 7, 0], [5, 8, 0]])
    sampler._skill_starts = np.asarray([[0, 10, -1], [0, 10, -1], [0, 10, -1]])
    sampler._skill_lengths = np.asarray([[10, 10, 0], [10, 10, 0], [10, 10, 0]])
    sampler._pools = {
        5: {
            0: (np.asarray([0]), np.asarray([1.0], dtype=np.float32)),
            1: (np.asarray([1]), np.asarray([1.0], dtype=np.float32)),
            2: (np.asarray([2]), np.asarray([1.0], dtype=np.float32)),
        }
    }

    matched = sampler._sample_partner(
        anchor=0,
        draw=(1, True, 1),
        rng=np.random.default_rng(0),
        used={0},
    )

    assert matched == (1, (1, -1), (1, -1))
