"""The training-only wrist patch-alignment head and its masked cross-entropy."""

from __future__ import annotations

import math

import pytest
import torch

from lerobot.policies.skill_expert.wrist_patch_alignment import (
    WristPatchAlignmentHead,
    wrist_patch_alignment_loss,
)
from lerobot.policies.skill_expert.wrist_patch_target import soft_targets

GRID = 14
PATCHES = GRID * GRID
QUERY_WIDTH, PATCH_WIDTH, TOKENS = 256, 384, 100


def _head() -> WristPatchAlignmentHead:
    torch.manual_seed(0)
    return WristPatchAlignmentHead(QUERY_WIDTH, PATCH_WIDTH)


def test_the_head_scores_every_patch_and_stays_float32() -> None:
    head = _head()
    query = torch.randn(4, TOKENS, QUERY_WIDTH, dtype=torch.bfloat16)
    patches = torch.randn(4, PATCHES, PATCH_WIDTH, dtype=torch.bfloat16)
    logits = head(query, patches)
    assert logits.shape == (4, PATCHES)
    assert logits.dtype == torch.float32                      # the readout runs in full precision

    with pytest.raises(ValueError, match="batches differ"):
        head(query, patches[:2])
    with pytest.raises(ValueError, match=r"\[batch, tokens, width\]"):
        head(query[:, 0], patches)


def test_the_head_follows_the_patch_it_is_pointed_at() -> None:
    """A query aligned with one patch's feature must score that patch highest."""
    head = _head()
    patches = torch.randn(2, PATCHES, PATCH_WIDTH)
    query = torch.randn(2, TOKENS, QUERY_WIDTH)
    logits = head(query, patches)
    chosen = int(logits[0].argmax())
    patches[0, chosen] *= 8.0                                  # make that patch dominate
    assert int(head(query, patches)[0].argmax()) == chosen


def test_the_loss_drops_masked_frames_and_starts_near_one() -> None:
    label = torch.tensor([10, 57, 3, 120])
    valid = torch.tensor([True, False, True, False])
    logits = torch.zeros(4, PATCHES)                           # a chance-level head

    loss, metrics = wrist_patch_alignment_loss(logits, label, valid, grid=GRID, sigma=0.7)
    assert loss == pytest.approx(1.0, abs=1e-5)                # ln(196) normalization
    assert metrics["wrist_patch/valid_fraction"] == pytest.approx(0.5)

    # The dropped frames cannot move the loss, whatever their logits say.
    noisy = logits.clone()
    noisy[1] = torch.randn(PATCHES) * 10.0
    noisy[3] = torch.randn(PATCHES) * 10.0
    again, _ = wrist_patch_alignment_loss(noisy, label, valid, grid=GRID, sigma=0.7)
    assert float(again) == pytest.approx(float(loss), abs=1e-6)


def test_a_confident_correct_head_beats_a_confident_wrong_one() -> None:
    label = torch.tensor([57])
    valid = torch.tensor([True])
    right = torch.full((1, PATCHES), -10.0)
    right[0, 57] = 10.0
    wrong = torch.full((1, PATCHES), -10.0)
    wrong[0, 0] = 10.0

    good, good_metrics = wrist_patch_alignment_loss(right, label, valid, grid=GRID, sigma=0.7)
    bad, bad_metrics = wrist_patch_alignment_loss(wrong, label, valid, grid=GRID, sigma=0.7)
    assert float(good) < float(bad)
    assert good_metrics["wrist_patch/cell_accuracy"] == 1.0
    assert good_metrics["wrist_patch/cell_distance"] == 0.0
    assert bad_metrics["wrist_patch/cell_accuracy"] == 0.0
    assert bad_metrics["wrist_patch/cell_distance"] == pytest.approx(4.0)   # (4,1) -> (0,0)

    neighbour = torch.full((1, PATCHES), -10.0)
    neighbour[0, 58] = 10.0
    _, neighbour_metrics = wrist_patch_alignment_loss(neighbour, label, valid, grid=GRID, sigma=0.7)
    assert neighbour_metrics["wrist_patch/within_one_cell"] == 1.0
    assert neighbour_metrics["wrist_patch/cell_accuracy"] == 0.0


def test_a_batch_with_nothing_to_learn_from_contributes_no_gradient() -> None:
    head = _head()
    logits = head(torch.randn(3, TOKENS, QUERY_WIDTH), torch.randn(3, PATCHES, PATCH_WIDTH))
    loss, metrics = wrist_patch_alignment_loss(
        logits, torch.full((3,), -1), torch.zeros(3, dtype=torch.bool), grid=GRID, sigma=0.7
    )
    assert float(loss.detach()) == 0.0
    assert metrics["wrist_patch/valid_fraction"] == 0.0
    loss.backward()                                            # the head still sees a gradient path
    assert head.query_proj.weight.grad is not None
    assert torch.count_nonzero(head.query_proj.weight.grad) == 0

    with pytest.raises(ValueError, match="Expected 196 logits"):
        wrist_patch_alignment_loss(
            logits[:, :100], torch.zeros(3, dtype=torch.long), torch.ones(3, dtype=torch.bool),
            grid=GRID, sigma=0.7,
        )


def test_the_loss_is_lowest_when_the_head_matches_the_soft_target() -> None:
    """Smoothing sets the floor: predicting the blurred cell beats both vagueness and overconfidence."""
    label = torch.tensor([57])
    valid = torch.tensor([True])
    # The target as logits; distant cells underflow to zero, so clamp before taking the log.
    matched = soft_targets(label, grid=GRID, sigma=0.7).clamp_min(1e-30).log()
    overconfident = torch.full((1, PATCHES), -20.0)
    overconfident[0, 57] = 20.0

    best, _ = wrist_patch_alignment_loss(matched, label, valid, grid=GRID, sigma=0.7)
    chance, _ = wrist_patch_alignment_loss(torch.zeros(1, PATCHES), label, valid, grid=GRID, sigma=0.7)
    peaked, _ = wrist_patch_alignment_loss(overconfident, label, valid, grid=GRID, sigma=0.7)
    assert 0.0 < float(best) < float(chance) < float(peaked)
    assert float(chance) == pytest.approx(1.0, abs=1e-5)
    # The floor is the smoothed target's own entropy, so the aim stays "roughly here", not "exactly here".
    entropy = -(soft_targets(label, grid=GRID, sigma=0.7) * matched).sum()
    assert float(best) == pytest.approx(float(entropy) / math.log(PATCHES), abs=1e-5)

    # sigma=0 asks for the exact cell, and a confident head then gets nearly zero loss.
    sharp, _ = wrist_patch_alignment_loss(overconfident, label, valid, grid=GRID, sigma=0.0)
    assert float(sharp) < 1e-6
