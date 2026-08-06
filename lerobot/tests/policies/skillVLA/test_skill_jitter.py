from types import SimpleNamespace

import numpy as np
import torch

from lerobot.policies.skillVLA import dataset_transitions, skill_jitter
from lerobot.policies.skillVLA.dataset_skillVLA import SKILL_START_STATE
from lerobot.policies.skillVLA.dataset_transitions import SkillTransitionDataset
from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy


class _FixedRng:
    def __init__(self, p: float, random_values=(0.0,)) -> None:
        self.p = p
        self.random_values = iter(random_values)

    def normal(self, *_args):
        return self.p

    def random(self):
        return next(self.random_values)


class _UniformRng(_FixedRng):
    def __init__(self, p: int, random_values=(0.0,)) -> None:
        super().__init__(float(p), random_values)
        self.uniform_p = p

    def integers(self, low: int, high: int) -> int:
        assert low == 0
        assert self.uniform_p < high
        return self.uniform_p


def test_frame_jitter_uses_exact_early_and_late_boundary_frames() -> None:
    # p=1 early: previous skill's final frame (de=0) is the first early-transition frame.
    assert skill_jitter.choose_jitter(
        k=0, ds=9, de=0, seq_len=3, pmax=2, rng=_FixedRng(1.0)) == (1, -1)
    # One frame farther away (de=1) must remain on the current skill.
    assert skill_jitter.choose_jitter(
        k=0, ds=8, de=1, seq_len=3, pmax=2, rng=_FixedRng(1.0, (0.0,))) == (0, 1)

    # p=1 late: current skill's first frame (ds=0) still uses the previous skill.
    assert skill_jitter.choose_jitter(
        k=1, ds=0, de=9, seq_len=3, pmax=2, rng=_FixedRng(1.0, (0.0,))) == (0, 1)
    # At ds=1 the delayed boundary has already switched to the current skill.
    assert skill_jitter.choose_jitter(
        k=1, ds=1, de=8, seq_len=3, pmax=2, rng=_FixedRng(1.0, (0.0,))) == (1, 1)


def test_torch_jitter_matches_boundary_convention(monkeypatch) -> None:
    monkeypatch.setattr(torch, "randn", lambda shape, device=None: torch.ones(shape, device=device))
    monkeypatch.setattr(torch, "rand", lambda shape, device=None: torch.zeros(shape, device=device))
    k = torch.tensor([0, 1])
    k_prime, offset = skill_jitter.choose_jitter_torch(
        k, torch.tensor([9, 0]), torch.tensor([0, 9]), torch.tensor([3, 3]), pmax=2)

    torch.testing.assert_close(k_prime, torch.tensor([1, 0]))
    torch.testing.assert_close(offset, torch.tensor([-1, 1]))


def test_uniform_jitter_uses_full_configured_magnitude() -> None:
    assert skill_jitter.sample_p(
        pmax=20, rng=_UniformRng(20), distribution="uniform") == 20
    assert skill_jitter.choose_jitter(
        k=0, ds=4, de=19, seq_len=3, pmax=20,
        rng=_UniformRng(20), distribution="uniform",
    ) == (1, -20)


def test_uniform_torch_jitter_uses_randint(monkeypatch) -> None:
    monkeypatch.setattr(
        torch, "randint", lambda low, high, shape, device=None: torch.full(shape, high - 1, device=device))
    monkeypatch.setattr(torch, "rand", lambda shape, device=None: torch.zeros(shape, device=device))
    k_prime, offset = skill_jitter.choose_jitter_torch(
        torch.tensor([0]), torch.tensor([4]), torch.tensor([19]), torch.tensor([3]),
        pmax=20, distribution="uniform",
    )
    torch.testing.assert_close(k_prime, torch.tensor([1]))
    torch.testing.assert_close(offset, torch.tensor([-20]))


def test_reusable_jitter_draw_resolves_against_each_samples_metadata() -> None:
    draw = (2, True, -1)

    assert skill_jitter.apply_jitter_draw(k=0, ds=8, de=1, seq_len=3, draw=draw) == (1, -2)
    assert skill_jitter.apply_jitter_draw(k=0, ds=4, de=5, seq_len=3, draw=draw) == (0, -2)


def test_effective_skill_end_tracks_early_and_late_virtual_boundaries() -> None:
    starts = np.array([0, 10, 20])
    lengths = np.array([10, 10, 10])

    # The next skill starts two frames early. At frame 9 its unchanged end is
    # frame 19, so offsets 0..10 are assigned to the jittered next skill.
    assert skill_jitter.effective_jittered_skill_de(
        k=0,
        k_prime=1,
        ds=9,
        de=0,
        skill_initial_frames=starts,
        skill_lengths=lengths,
        offset=-2,
    ) == 10

    # A two-frame delayed transition keeps the previous skill active at the
    # new skill's ds=0 and ds=1 frames respectively.
    assert skill_jitter.effective_jittered_skill_de(
        k=1,
        k_prime=0,
        ds=0,
        de=9,
        skill_initial_frames=starts,
        skill_lengths=lengths,
        offset=2,
    ) == 1
    assert skill_jitter.effective_jittered_skill_de(
        k=1,
        k_prime=0,
        ds=1,
        de=8,
        skill_initial_frames=starts,
        skill_lengths=lengths,
        offset=-2,
    ) == 0

    assert skill_jitter.effective_jittered_skill_de(
        k=1,
        k_prime=1,
        ds=4,
        de=5,
        skill_initial_frames=starts,
        skill_lengths=lengths,
        offset=-2,
    ) == 5


def test_sample_jitter_draw_uses_configured_distribution() -> None:
    draw = skill_jitter.sample_jitter_draw(
        pmax=20,
        rng=_UniformRng(20, random_values=(0.0, 0.9)),
        distribution="uniform",
    )

    assert draw == (20, True, -1)


def test_stage1_jitters_action_skill_code(monkeypatch) -> None:
    stub = SimpleNamespace(
        training=True,
        config=SimpleNamespace(
            transition_jitter_pmax=10,
            transition_jitter_distribution="uniform",
            skill_vocab_size=8,
        ),
    )
    batch = {
        "skill_sequence": torch.tensor([[2, 5, 8]]),
        "skill_index": torch.tensor([0]),
        "skill_sequence_len": torch.tensor([3]),
        "skill_ds": torch.tensor([9]),
        "skill_de": torch.tensor([0]),
    }
    monkeypatch.setattr(
        skill_jitter,
        "choose_jitter_torch",
        lambda k, ds, de, seq_len, pmax, distribution: (
            k + 1, torch.full_like(k, -1)) if distribution == "uniform" else None,
    )

    jittered = SkillExpertPolicy._skill_code(stub, batch)

    torch.testing.assert_close(jittered, torch.tensor([5]))


def test_stage1_reuses_skillvla_dataset_jitter_code() -> None:
    stub = SimpleNamespace(
        training=True,
        config=SimpleNamespace(skill_vocab_size=8),
    )
    batch = {
        "skill_code": torch.tensor([5]),
        "skill_sequence": torch.tensor([[2, 5, 7]]),
        "skill_index": torch.tensor([0]),
    }

    code = SkillExpertPolicy._skill_code(stub, batch)

    torch.testing.assert_close(code, torch.tensor([5]))
    torch.testing.assert_close(
        stub._last_transition_jitter_fraction, torch.tensor(1.0)
    )


def test_transition_dataset_uses_state_from_same_half_normal_offset(monkeypatch) -> None:
    pack = SimpleNamespace(
        pmax=1,
        jitter_distribution="half_normal",
        skill_code=np.array([4]),
        task_index=np.array([0]),
        tasks=["move object"],
        start_state=np.array([[[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]], dtype=np.float32),
        image=lambda cam, seg, win_idx: torch.full((3, 2, 2), float(win_idx)),
    )
    ds = object.__new__(SkillTransitionDataset)
    ds._packs = [pack]
    ds._cum = np.array([1])
    ds._state_dim = 2
    ds._act_shape = (1, 2)
    ds._transition_randomization = True
    monkeypatch.setattr(
        dataset_transitions,
        "sample_offset",
        lambda pmax, distribution: 1 if distribution == "half_normal" else 0,
    )

    item = SkillTransitionDataset.__getitem__(ds, 0)

    torch.testing.assert_close(item[SKILL_START_STATE], torch.tensor([30.0, 31.0]))
