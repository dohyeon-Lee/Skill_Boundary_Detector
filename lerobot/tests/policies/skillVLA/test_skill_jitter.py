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


def test_stage1_jitters_action_code_but_keeps_terminator_true_code(monkeypatch) -> None:
    stub = SimpleNamespace(
        training=True,
        config=SimpleNamespace(transition_jitter_pmax=10, skill_vocab_size=8),
    )
    stub._true_skill_code = lambda batch: SkillExpertPolicy._true_skill_code(stub, batch)
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
        lambda k, ds, de, seq_len, pmax: (k + 1, torch.full_like(k, -1)),
    )

    jittered = SkillExpertPolicy._skill_code(stub, batch)
    true = SkillExpertPolicy._true_skill_code(stub, batch)

    torch.testing.assert_close(jittered, torch.tensor([5]))
    torch.testing.assert_close(true, torch.tensor([2]))


def test_transition_dataset_uses_state_from_same_half_normal_offset(monkeypatch) -> None:
    pack = SimpleNamespace(
        pmax=1,
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
    monkeypatch.setattr(dataset_transitions, "sample_offset", lambda pmax: 1)

    item = SkillTransitionDataset.__getitem__(ds, 0)

    torch.testing.assert_close(item[SKILL_START_STATE], torch.tensor([30.0, 31.0]))
