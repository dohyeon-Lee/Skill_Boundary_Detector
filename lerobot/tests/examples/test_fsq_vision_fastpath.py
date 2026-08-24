from __future__ import annotations

import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "lerobot/examples/libero"))

import FSQ as fsq_module  # noqa: E402
from FSQ_original import FSQOriginalConfig  # noqa: E402
from FSQ import (  # noqa: E402
    BSQ,
    BoundaryAugmentationContext,
    DtypeAlignedRMSNorm,
    FSQStateRNNTerminator,
    FSQTrajectoryDataset,
    FSQQueryTerminator,
    FSQWristOnlyQueryTerminator,
    SplineFSQAE,
    SplineFSQAEConfig,
    absorb_z_head_calibration_,
    bsq_pair_joint_overlaps,
    bsq_joint_soft_assignments,
    bsq_js_pair_loss,
    build_adjacent_skill_indices,
    build_boundary_augmentation_contexts,
    calibrate_fsq_z_head_,
    episode_grouped_train_val_ids,
    fsq_entropy_terms,
    fsq_js_pair_loss,
    fsq_lr_factor,
    fsq_overlap_pair_loss,
    fsq_pair_joint_overlaps,
    fsq_pair_weight_at_epoch,
    fsq_reconstruction_loss,
    sample_boundary_augmented_segment,
)


class _FixedJitterRng:
    def __init__(self, random_values: list[float], normal_value: float):
        self._random_values = iter(random_values)
        self._normal_value = normal_value

    def random(self) -> float:
        return next(self._random_values)

    def normal(self, *_args) -> float:
        return self._normal_value


def test_z_head_calibration_is_folded_into_linear_parameters_once() -> None:
    torch.manual_seed(7)
    head = nn.Linear(5, 3)
    hidden = torch.randn(128, 5) * 2.0 + 1.5
    before = head(hidden).detach()
    mean = before.mean(dim=0)
    scale = before.std(dim=0, correction=0)

    absorb_z_head_calibration_(head, mean, scale, gain=0.8)
    after = head(hidden).detach()

    torch.testing.assert_close(after, 0.8 * (before - mean) / scale)
    torch.testing.assert_close(after.mean(dim=0), torch.zeros(3), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        after.std(dim=0, correction=0), torch.full((3,), 0.8), atol=1e-6, rtol=0
    )


def test_pair_weight_schedule_has_reconstruction_warmup_and_linear_ramp() -> None:
    schedule = lambda epoch: fsq_pair_weight_at_epoch(  # noqa: E731
        0.01, epoch, warmup_epochs=50, ramp_epochs=50
    )

    assert schedule(1) == 0.0
    assert schedule(50) == 0.0
    assert schedule(51) == pytest.approx(0.0002)
    assert schedule(75) == pytest.approx(0.005)
    assert schedule(100) == pytest.approx(0.01)
    assert schedule(500) == pytest.approx(0.01)
    assert fsq_pair_weight_at_epoch(
        0.01,
        1,
        warmup_epochs=50,
        ramp_epochs=50,
        enabled=False,
    ) == pytest.approx(0.01)


def test_bsq5_uses_32_symmetric_sign_codes() -> None:
    quantizer = BSQ(5)
    z = torch.tensor(
        [
            [-3.0, -2.0, -1.0, -4.0, -5.0],
            [3.0, 2.0, 1.0, 4.0, 5.0],
            [3.0, -2.0, 1.0, -4.0, 5.0],
        ]
    )

    z_q, indices = quantizer(z)

    assert quantizer.codebook_size == 32
    assert indices.tolist() == [0, 31, 21]
    torch.testing.assert_close(
        quantizer.normalized(z_q),
        quantizer.code_to_normalized(indices),
    )
    torch.testing.assert_close(z_q.norm(dim=-1), torch.ones(3))


def test_bsq_calibration_summary_follows_quantizer_device() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quantizer = BSQ(5).to(device)
    # Calibration intentionally retains the complete latent table on CPU even
    # when the model/quantizer is on CUDA.
    latents = torch.randn(128, 5, device="cpu")

    summary = fsq_module._fsq_code_summary_from_latents(quantizer, latents)

    assert 1 <= summary["active_entries"] <= 32
    assert 0.0 < summary["utilization_pct"] <= 100.0
    assert 0.0 < summary["dominant_code_pct"] <= 100.0


def test_boundary_metrics_keep_any_axis_and_per_axis_rates_separate() -> None:
    margins = torch.tensor(
        [
            [0.00, 0.50],
            [0.25, 0.05],
            [0.50, 0.50],
            [0.50, 0.00],
        ]
    )

    metrics = fsq_module._boundary_margin_metrics(margins)

    assert metrics["near_boundary_pct"] == pytest.approx(75.0)
    assert metrics["per_axis_near_boundary_pct"] == pytest.approx(37.5)
    assert metrics["axis_0_near_boundary_pct"] == pytest.approx(25.0)
    assert metrics["axis_1_near_boundary_pct"] == pytest.approx(50.0)
    assert metrics["boundary_margin_mean_pct"] == pytest.approx(27.5)
    assert metrics["per_axis_boundary_margin_mean_pct"] == pytest.approx(57.5)


def test_assignment_stability_reports_per_axis_bit_flips() -> None:
    previous = torch.tensor([0, 0, 7, 2])
    current = torch.tensor([1, 2, 3, 2])

    metrics = fsq_module._code_assignment_stability(
        previous,
        current,
        codebook_size=8,
        levels=[2, 2, 2],
    )

    assert metrics["change_pct"] == pytest.approx(75.0)
    assert metrics["per_axis_change_pct"] == pytest.approx(25.0)
    assert metrics["axis_0_change_pct"] == pytest.approx(25.0)
    assert metrics["axis_1_change_pct"] == pytest.approx(25.0)
    assert metrics["axis_2_change_pct"] == pytest.approx(25.0)


def test_bsq_js_is_zero_for_equal_diffuse_distributions_without_confidence_pressure() -> None:
    clean = torch.tensor([[0.0, 0.6, -0.8, 0.0, 0.0]], requires_grad=True)
    same = clean.detach().clone().requires_grad_(True)
    different = torch.tensor([[0.6, -0.8, 0.0, 0.0, 0.0]])

    same_js = bsq_js_pair_loss(clean, same, inv_temperature=5.0)
    different_js = bsq_js_pair_loss(clean, different, inv_temperature=5.0)

    assert same_js.item() == pytest.approx(0.0, abs=1e-8)
    assert different_js > same_js
    same_js.backward()
    assert clean.grad is not None
    assert same.grad is not None
    distribution = bsq_joint_soft_assignments(clean.detach(), inv_temperature=5.0)
    assert distribution.shape == (1, 32)
    torch.testing.assert_close(distribution.sum(dim=-1), torch.ones(1))


def test_overlap_pair_loss_prefers_matching_soft_fsq_codes_and_backpropagates() -> None:
    clean = torch.tensor([[0.15, -0.20]], requires_grad=True)
    matching = torch.tensor([[0.15, -0.20]], requires_grad=True)
    different = torch.tensor([[1.0, 1.0]])

    matching_loss, matching_overlap = fsq_overlap_pair_loss(
        clean, matching, [3, 3], 5.0
    )
    different_loss, different_overlap = fsq_overlap_pair_loss(
        clean, different, [3, 3], 5.0
    )

    assert matching_loss < different_loss
    assert matching_overlap > different_overlap
    matching_loss.backward()
    assert torch.isfinite(clean.grad).all()
    assert torch.isfinite(matching.grad).all()


def test_js_pair_loss_matches_diffuse_distributions_without_sharpening() -> None:
    # Both views sit on the same FSQ boundaries. JS is already satisfied,
    # whereas overlap still has a confidence/sharpening penalty.
    clean = torch.tensor([[0.5, -0.5]], requires_grad=True)
    matching = torch.tensor([[0.5, -0.5]], requires_grad=True)
    different = torch.tensor([[-0.5, 0.5]], requires_grad=True)

    matching_js = fsq_js_pair_loss(clean, matching, [3, 3], 5.0)
    different_js = fsq_js_pair_loss(clean, different, [3, 3], 5.0)
    matching_overlap, _ = fsq_overlap_pair_loss(
        clean, matching, [3, 3], 5.0
    )

    torch.testing.assert_close(matching_js, torch.zeros_like(matching_js), atol=1e-7, rtol=0)
    assert matching_overlap > 0
    assert different_js > matching_js
    different_js.backward()
    assert torch.isfinite(clean.grad).all()
    assert torch.isfinite(different.grad).all()


@pytest.mark.parametrize("quantizer", ["fsq", "bsq"])
def test_linear_contrastive_overlap_uses_symmetric_positive_and_negative_terms(
    quantizer: str,
) -> None:
    dimensions = 2 if quantizer == "fsq" else 3
    clean = torch.zeros(2, dimensions, requires_grad=True)
    positive = torch.tensor(
        [[0.0] * dimensions, [0.4] * dimensions], requires_grad=True
    )
    negative = torch.tensor(
        [[1.0] * dimensions, [0.0] * dimensions], requires_grad=True
    )
    config = SplineFSQAEConfig(
        action_dim=1,
        max_action_dim=1,
        chunk_size=1,
        samples_per_skill=1,
        action_loss_weight=0.0,
        progress_loss_weight=0.0,
        end_loss_weight=0.0,
        pair_loss="contrastive",
        pair_weight=1.0,
        pair_inv_temperature=5.0,
        quantizer=quantizer,
        fsq_levels=[3, 3],
        bsq_code_dim=3,
    )
    output = {
        "actions": torch.zeros(2, 1, 1),
        "progress": torch.zeros(2),
        "term_logits": torch.zeros(2),
        "u_cont": clean,
        "augmented_u_cont": positive,
        "negative_u_cont": negative,
        "indices": torch.tensor([0, 1]),
        "augmented_indices": torch.tensor([0, 2]),
        "negative_indices": torch.tensor([3, 1]),
    }
    batch = {
        "ctrl": torch.zeros(2, 1, 1),
        "actions": torch.zeros(2, 1, 1, 1),
        "progress": torch.zeros(2, 1),
        "termination": torch.zeros(2, 1),
        # The second row represents a true singleton episode. Its placeholder
        # negative is masked, while its positive consistency remains active.
        "negative_valid": torch.tensor([True, False]),
    }
    if quantizer == "fsq":
        positive_overlap = fsq_pair_joint_overlaps(clean, positive, [3, 3], 5.0)
        negative_overlap = fsq_pair_joint_overlaps(clean, negative, [3, 3], 5.0)
    else:
        positive_overlap = bsq_pair_joint_overlaps(clean, positive, 5.0)
        negative_overlap = bsq_pair_joint_overlaps(clean, negative, 5.0)
    expected = torch.stack(
        (
            0.5 * ((1.0 - positive_overlap[0]) + negative_overlap[0]),
            1.0 - positive_overlap[1],
        )
    ).mean()

    loss, metrics = fsq_reconstruction_loss(output, batch, config)

    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(metrics["pair_loss"], expected)
    torch.testing.assert_close(
        metrics["pair_positive_linear_loss"], (1.0 - positive_overlap).mean()
    )
    torch.testing.assert_close(
        metrics["pair_negative_joint_overlap"], negative_overlap[0]
    )
    torch.testing.assert_close(
        metrics["pair_negative_valid_fraction"], torch.tensor(0.5)
    )
    assert 0.0 <= loss.item() <= 1.0
    loss.backward()
    assert torch.isfinite(clean.grad).all()
    assert torch.isfinite(positive.grad).all()
    assert torch.isfinite(negative.grad).all()


def test_entropy_conf_ceiling_is_an_exact_per_sample_hinge() -> None:
    config = SplineFSQAEConfig(
        action_dim=1,
        max_action_dim=1,
        chunk_size=1,
        samples_per_skill=1,
        action_loss_weight=0.0,
        progress_loss_weight=0.0,
        end_loss_weight=0.0,
        fsq_entropy=True,
        entropy_conf_weight=1.0,
        entropy_conf_ceiling=0.1,
        entropy_div_weight=0.0,
        entropy_inv_temperature=10.0,
        fsq_levels=[3],
    )
    bounded = torch.tensor([[0.0], [0.4]], requires_grad=True)
    output = {
        "actions": torch.zeros(2, 1, 1),
        "progress": torch.zeros(2),
        "term_logits": torch.zeros(2),
        "u_cont": bounded,
    }
    batch = {
        "ctrl": torch.zeros(2, 1, 1),
        "actions": torch.zeros(2, 1, 1, 1),
        "progress": torch.zeros(2, 1),
        "termination": torch.zeros(2, 1),
    }

    raw_entropy, _ = fsq_entropy_terms(bounded, [3], 10.0, joint_dataset=True)
    loss, metrics = fsq_reconstruction_loss(output, batch, config)

    torch.testing.assert_close(metrics["entropy_sample"], raw_entropy)
    torch.testing.assert_close(loss, metrics["entropy_conf_loss"])
    torch.testing.assert_close(
        metrics["entropy_conf_active_fraction"], torch.tensor(0.5)
    )
    assert metrics["entropy_conf_ceiling_nats"].item() == pytest.approx(
        0.1 * np.log(3)
    )
    loss.backward()
    torch.testing.assert_close(bounded.grad[0], torch.zeros(1), atol=0, rtol=0)
    assert bounded.grad[1].abs().item() > 0

    config.entropy_conf_ceiling = 0.0
    legacy_bounded = torch.tensor([[0.0], [0.4]])
    legacy_output = {**output, "u_cont": legacy_bounded}
    legacy_raw_entropy, _ = fsq_entropy_terms(
        legacy_bounded, [3], 10.0, joint_dataset=True
    )
    legacy_loss, legacy_metrics = fsq_reconstruction_loss(
        legacy_output, batch, config
    )
    torch.testing.assert_close(legacy_metrics["entropy_conf_loss"], legacy_raw_entropy)
    torch.testing.assert_close(legacy_loss, legacy_raw_entropy)


def test_shuffle_delta_metrics_use_probabilities_and_only_different_codes() -> None:
    config = SplineFSQAEConfig(
        action_dim=1,
        max_action_dim=1,
        chunk_size=1,
        samples_per_skill=2,
        terminator_only=True,
        terminator_progress=True,
        terminator_termination=True,
    )
    output = {
        "actions": torch.zeros(4, 1, 1),
        "progress": torch.tensor([0.1, 0.2, 0.7, 0.8]),
        "term_logits": torch.tensor([-2.0, -1.0, 1.0, 2.0]),
        "skill_shuffle_progress": torch.tensor([0.7, 0.8, 0.1, 0.2]),
        "skill_shuffle_term_logits": torch.tensor([1.0, 2.0, -2.0, -1.0]),
        "skill_shuffle_valid": torch.tensor([True, False]),
    }
    batch = {
        "ctrl": torch.zeros(2, 1, 1),
        "actions": torch.zeros(2, 2, 1, 1),
        "progress": torch.zeros(2, 2),
        "termination": torch.zeros(2, 2),
    }

    _, metrics = fsq_reconstruction_loss(output, batch, config)

    assert metrics["skill_shuffle_progress_delta"].item() == pytest.approx(0.6)
    expected_end = (
        (torch.sigmoid(torch.tensor(-2.0)) - torch.sigmoid(torch.tensor(1.0))).abs()
        + (torch.sigmoid(torch.tensor(-1.0)) - torch.sigmoid(torch.tensor(2.0))).abs()
    ) / 2
    torch.testing.assert_close(
        metrics["skill_shuffle_end_probability_delta"], expected_end
    )
    assert metrics["skill_shuffle_valid_fraction"].item() == pytest.approx(0.5)


def test_boundary_augmentation_moves_only_one_boundary_with_adjacent_context() -> None:
    trajectory = np.arange(30, dtype=np.float32)[:, None]
    segments = [trajectory[:10], trajectory[10:20], trajectory[20:]]
    metadata = [
        {
            "episode_id": 0,
            "task_id": 0,
            "skill_index": i,
            "frame_start": 10 * i,
            "frame_end": 10 * (i + 1),
        }
        for i in range(3)
    ]
    context = build_boundary_augmentation_contexts(segments, metadata, pmax=10)[1]
    assert isinstance(context, BoundaryAugmentationContext)
    assert (context.start, context.end, len(context.trajectory)) == (10, 20, 30)

    # Select start and draw offset=-4: prepend four frames, leaving the end fixed.
    start_aug, boundary, offset = sample_boundary_augmented_segment(
        context,
        pmax=10,
        min_length=1,
        distribution="half_normal",
        rng=_FixedJitterRng([0.1, 0.1], normal_value=4.0),
    )
    assert (boundary, offset) == (0, -4)
    np.testing.assert_array_equal(start_aug[:, 0], np.arange(6, 20))

    # Select end and draw offset=+4: append four frames, leaving the start fixed.
    end_aug, boundary, offset = sample_boundary_augmented_segment(
        context,
        pmax=10,
        min_length=1,
        distribution="half_normal",
        rng=_FixedJitterRng([0.9, 0.9], normal_value=4.0),
    )
    assert (boundary, offset) == (1, 4)
    np.testing.assert_array_equal(end_aug[:, 0], np.arange(10, 24))


def test_adjacent_skill_indices_use_only_contiguous_in_episode_neighbours() -> None:
    metadata = [
        {
            "task_id": 0,
            "episode_id": 10,
            "skill_index": i,
            "frame_start": 4 * i,
            "frame_end": 4 * (i + 1),
        }
        for i in range(3)
    ]
    metadata.append(
        {
            "task_id": 0,
            "episode_id": 11,
            "skill_index": 0,
            "frame_start": 0,
            "frame_end": 4,
        }
    )

    assert build_adjacent_skill_indices(metadata) == [(1,), (0, 2), (1,), ()]


def test_episode_grouped_split_never_separates_adjacent_negatives() -> None:
    metadata = []
    for episode_id, count in enumerate((3, 2, 1, 4)):
        for skill_index in range(count):
            metadata.append(
                {
                    "task_id": 0,
                    "episode_id": episode_id,
                    "skill_index": skill_index,
                }
            )

    train_ids, val_ids = episode_grouped_train_val_ids(metadata, target_val_size=3)

    assert train_ids and val_ids
    train_episodes = {metadata[i]["episode_id"] for i in train_ids}
    val_episodes = {metadata[i]["episode_id"] for i in val_ids}
    assert train_episodes.isdisjoint(val_episodes)
    assert len(val_ids) >= 3


def test_contrastive_dataset_samples_one_full_adjacent_skill_and_masks_singletons(
    monkeypatch,
) -> None:
    segments = [
        np.full((4, 8), fill_value=float(i), dtype=np.float32)
        for i in range(4)
    ]
    metadata = [
        {
            "task_id": 0,
            "episode_id": 0,
            "skill_index": i,
            "frame_start": 4 * i,
            "frame_end": 4 * (i + 1),
        }
        for i in range(3)
    ] + [
        {
            "task_id": 0,
            "episode_id": 1,
            "skill_index": 0,
            "frame_start": 0,
            "frame_end": 4,
        }
    ]
    config = _state_rnn_config(
        reconstructor_arch="oneshot",
        pair_loss="contrastive",
        boundary_aug_pmax=1,
        length_min=1.0,
        length_max=6.0,
    )
    contexts = build_boundary_augmentation_contexts(segments, metadata, pmax=1)
    neighbours = build_adjacent_skill_indices(metadata)
    actions = [np.zeros((4, 7), dtype=np.float32) for _ in segments]
    monkeypatch.setattr(
        fsq_module,
        "sample_boundary_augmented_segment",
        lambda selected, **_kwargs: (
            selected.trajectory[selected.start : selected.end].copy(),
            0,
            0,
        ),
    )
    dataset = FSQTrajectoryDataset(
        segments=segments,
        states=segments,
        actions=actions,
        metadata=metadata,
        raw_dataset_dir="unused",
        cfg=config,
        training=True,
        boundary_contexts=contexts,
        adjacent_skill_indices=neighbours,
    )

    monkeypatch.setattr(fsq_module.np.random, "random", lambda: 0.1)
    center_left = dataset[1]
    assert center_left["negative_trajectory_index"].item() == 0
    torch.testing.assert_close(center_left["negative_ctrl"], torch.from_numpy(dataset.ctrl[0]))
    assert center_left["negative_valid"].item() is True

    monkeypatch.setattr(fsq_module.np.random, "random", lambda: 0.9)
    center_right = dataset[1]
    assert center_right["negative_trajectory_index"].item() == 2
    torch.testing.assert_close(center_right["negative_ctrl"], torch.from_numpy(dataset.ctrl[2]))

    singleton = dataset[3]
    assert singleton["negative_trajectory_index"].item() == 3
    assert singleton["negative_valid"].item() is False


@pytest.mark.parametrize(
    ("boundary_draw", "direction_draw", "expected_boundary", "expected_offset", "expected"),
    [
        (0.1, 0.1, 0, -2, np.arange(8, 20)),
        (0.1, 0.9, 0, 3, np.arange(13, 20)),
        (0.9, 0.1, 1, -4, np.arange(10, 16)),
        (0.9, 0.9, 1, 5, np.arange(10, 25)),
    ],
)
def test_boundary_augmentation_uses_distinct_directional_windows(
    boundary_draw: float,
    direction_draw: float,
    expected_boundary: int,
    expected_offset: int,
    expected: np.ndarray,
) -> None:
    context = BoundaryAugmentationContext(
        trajectory=np.arange(30, dtype=np.float32)[:, None],
        start=10,
        end=20,
    )

    augmented, boundary, offset = sample_boundary_augmented_segment(
        context,
        pmax=0,
        early_start_pmax=2,
        late_start_pmax=3,
        early_end_pmax=4,
        late_end_pmax=5,
        min_length=1,
        distribution="half_normal",
        rng=_FixedJitterRng([boundary_draw, direction_draw], normal_value=100.0),
    )

    assert (boundary, offset) == (expected_boundary, expected_offset)
    np.testing.assert_array_equal(augmented[:, 0], expected)


def test_zero_directional_window_disables_only_that_direction() -> None:
    context = BoundaryAugmentationContext(
        trajectory=np.arange(30, dtype=np.float32)[:, None],
        start=10,
        end=20,
    )

    augmented, boundary, offset = sample_boundary_augmented_segment(
        context,
        pmax=0,
        early_start_pmax=0,
        late_start_pmax=3,
        early_end_pmax=0,
        late_end_pmax=0,
        min_length=1,
        distribution="half_normal",
        rng=_FixedJitterRng([], normal_value=100.0),
    )

    assert (boundary, offset) == (0, 3)
    np.testing.assert_array_equal(augmented[:, 0], np.arange(13, 20))


def test_pair_dataset_refits_augmented_spline_only_during_training(monkeypatch) -> None:
    states = np.arange(12 * 8, dtype=np.float32).reshape(12, 8) / 100.0
    segments = [states[:4], states[4:8], states[8:]]
    metadata = [
        {
            "episode_id": 0,
            "task_id": 0,
            "skill_index": i,
            "frame_start": 4 * i,
            "frame_end": 4 * (i + 1),
        }
        for i in range(3)
    ]
    context = build_boundary_augmentation_contexts(segments, metadata, pmax=2)[1]
    config = _state_rnn_config(
        reconstructor_arch="oneshot",
        pair_loss="overlap",
        boundary_aug_pmax=2,
        length_min=1.0,
        length_max=6.0,
    )
    actions = np.zeros((4, 7), dtype=np.float32)
    monkeypatch.setattr(
        fsq_module,
        "sample_boundary_augmented_segment",
        lambda selected, **_kwargs: (
            selected.trajectory[selected.start - 2 : selected.end].copy(),
            0,
            -2,
        ),
    )

    training = FSQTrajectoryDataset(
        segments=[segments[1]],
        states=[segments[1]],
        actions=[actions],
        metadata=[metadata[1]],
        raw_dataset_dir="unused",
        cfg=config,
        training=True,
        boundary_contexts=[context],
    )[0]
    validation = FSQTrajectoryDataset(
        segments=[segments[1]],
        states=[segments[1]],
        actions=[actions],
        metadata=[metadata[1]],
        raw_dataset_dir="unused",
        cfg=config,
        training=False,
    )[0]

    assert training["length"].item() == 4
    assert training["augmented_length"].item() == 6
    assert training["augmentation_boundary"].item() == 0
    assert training["augmentation_offset"].item() == -2
    assert training["augmented_ctrl"].shape == training["ctrl"].shape
    assert "augmented_ctrl" not in validation


def _state_rnn_config(**overrides) -> SplineFSQAEConfig:
    state_dim = 8
    action_dim = 7
    values = dict(
        action_dim=action_dim,
        enc_dim=state_dim,
        state_dim=state_dim,
        n_control=6,
        spline_degree=3,
        encoder_input_mode="raw_state",
        hidden_dim=32,
        num_layers=1,
        fsq_levels=[3, 3, 3],
        max_state_dim=state_dim,
        max_action_dim=action_dim,
        chunk_size=1,
        samples_per_skill=2,
        length_min=1.0,
        length_max=6.0,
        terminator_input_space="state",
        terminator_model="rnn",
        terminator_progress=False,
        terminator_termination=True,
        state_rnn_terminator=True,
        terminator_termination_only=True,
        encoder_min=np.full(state_dim, -1.0, dtype=np.float32),
        encoder_max=np.full(state_dim, 1.0, dtype=np.float32),
        reconstructor_output_mode="raw_state",
        reconstructor_min=np.full(state_dim, -1.0, dtype=np.float32),
        reconstructor_max=np.full(state_dim, 1.0, dtype=np.float32),
        state_min=np.full(state_dim, -1.0, dtype=np.float32),
        state_max=np.full(state_dim, 1.0, dtype=np.float32),
        state_q01=np.full(state_dim, -1.0, dtype=np.float32),
        state_q99=np.full(state_dim, 1.0, dtype=np.float32),
        action_q01=np.full(action_dim, -1.0, dtype=np.float32),
        action_q99=np.full(action_dim, 1.0, dtype=np.float32),
    )
    values.update(overrides)
    return SplineFSQAEConfig(**values)


def test_fsq_z_head_calibration_uses_clean_cached_trajectories() -> None:
    rng = np.random.default_rng(11)
    segments = [rng.normal(size=(4, 8)).astype(np.float32) for _ in range(12)]
    actions = [np.zeros((4, 7), dtype=np.float32) for _ in segments]
    metadata = [
        {"episode_id": i, "task_id": 0, "skill_index": 0, "frame_start": 0}
        for i in range(len(segments))
    ]
    config = _state_rnn_config(
        reconstructor_arch="oneshot",
        samples_per_skill=1,
        init_calibration=True,
        init_calibration_gain=0.8,
    )
    dataset = FSQTrajectoryDataset(
        segments=segments,
        states=segments,
        actions=actions,
        metadata=metadata,
        raw_dataset_dir="unused",
        cfg=config,
        training=False,
    )
    model = SplineFSQAE(config)

    metrics = calibrate_fsq_z_head_(
        model,
        dataset,
        torch.device("cpu"),
        batch_size=4,
        gain=0.8,
    )

    assert metrics["init_calibration/samples"] == 12
    for axis in range(3):
        assert metrics[f"init_calibration/post_mean_axis_{axis}"] == pytest.approx(
            0.0, abs=1e-5
        )
        assert metrics[f"init_calibration/post_std_axis_{axis}"] == pytest.approx(
            0.8, abs=1e-5
        )


@pytest.mark.parametrize("quantizer", ["fsq", "bsq"])
def test_full_model_encodes_adjacent_negative_for_contrastive_mode(
    quantizer: str,
) -> None:
    config = _state_rnn_config(
        quantizer=quantizer,
        bsq_code_dim=5,
        reconstructor_arch="oneshot",
        samples_per_skill=1,
        pair_loss="contrastive",
        boundary_aug_pmax=1,
        state_rnn_terminator=False,
        terminator_model="default",
        terminator_termination_only=False,
        reconstructor_only=True,
    )
    model = SplineFSQAE(config).eval()
    ctrl = torch.randn(2, config.n_control, config.enc_dim)
    output = model(
        ctrl=ctrl,
        lengths=torch.tensor([4, 4]),
        start_state=torch.zeros(2, config.max_state_dim),
        raw_state=torch.zeros(2, config.state_dim),
        progress_target=torch.zeros(2),
        third=None,
        wrist=None,
        samples_per_skill=1,
        augmented_ctrl=ctrl + 0.1,
        augmented_lengths=torch.tensor([4, 4]),
        negative_ctrl=torch.flip(ctrl, dims=(0,)),
        negative_lengths=torch.tensor([4, 4]),
    )

    assert output["augmented_u_cont"].shape == output["u_cont"].shape
    assert output["negative_u_cont"].shape == output["u_cont"].shape
    assert output["negative_indices"].shape == output["indices"].shape


def test_state_rnn_model_builds_without_a_vision_encoder(monkeypatch) -> None:
    monkeypatch.setattr(
        fsq_module,
        "_load_dino_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("state-RNN FSQ must not load DINO")
        ),
    )

    model = SplineFSQAE(_state_rnn_config(reconstructor_arch="oneshot"))

    assert isinstance(model.terminator, FSQStateRNNTerminator)
    assert not hasattr(model.terminator, "vision_encoder")


def test_state_rnn_dataset_caches_full_skill_and_skips_images(monkeypatch) -> None:
    config = _state_rnn_config(reconstructor_arch="oneshot", end_target_sigma=0.0)
    states = np.arange(4 * 8, dtype=np.float32).reshape(4, 8)
    actions = np.zeros((4, 7), dtype=np.float32)
    dataset = FSQTrajectoryDataset(
        segments=[states],
        states=[states],
        actions=[actions],
        metadata=[{"episode_id": 0, "skill_index": 0, "frame_start": 0}],
        raw_dataset_dir="unused",
        cfg=config,
        training=True,
    )
    monkeypatch.setattr(
        dataset,
        "_sample_images",
        lambda *args: (_ for _ in ()).throw(AssertionError("images must stay unused")),
    )
    monkeypatch.setattr(
        fsq_module.np.random,
        "choice",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("oneshot+RNN must not sample a timestep")
        ),
    )

    item = dataset[0]

    assert dataset.samples_per_skill == 1
    torch.testing.assert_close(item["sample_index"], torch.tensor([0]))
    assert "third" not in item and "wrist" not in item
    assert item["terminator_state_sequence"].shape == (6, 8)
    torch.testing.assert_close(item["terminator_state_sequence"][-4:], torch.from_numpy(states))
    torch.testing.assert_close(
        item["terminator_progress"][:4],
        torch.tensor([0.0, 1 / 3, 2 / 3, 1.0]),
    )
    torch.testing.assert_close(
        item["terminator_termination"],
        torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    )


def test_spline_encoder_input_and_reconstruction_output_are_independent() -> None:
    config = _state_rnn_config(
        reconstructor_arch="oneshot",
        encoder_input_mode="raw_state",
        reconstructor_output_mode="zero_grounded",
    )
    states = np.arange(4 * 8, dtype=np.float32).reshape(4, 8)
    actions = np.zeros((4, 7), dtype=np.float32)
    dataset = FSQTrajectoryDataset(
        segments=[states],
        states=[states],
        actions=[actions],
        metadata=[{"episode_id": 0, "skill_index": 0, "frame_start": 0}],
        raw_dataset_dir="unused",
        cfg=config,
        training=False,
    )

    item = dataset[0]
    encoder_ctrl = item["ctrl"].numpy()
    target_ctrl = item["reconstructor_ctrl"].numpy()

    assert np.abs(encoder_ctrl[:, :3].mean(0)).max() > 1.0
    np.testing.assert_allclose(target_ctrl[:, :3].mean(0), np.zeros(3), atol=1e-5)
    np.testing.assert_allclose(target_ctrl[:, 3:], encoder_ctrl[:, 3:], atol=1e-5)


def test_state_rnn_loss_supervises_all_valid_steps_and_masks_padding() -> None:
    config = _state_rnn_config(
        terminator_only=True,
        action_loss_weight=0.0,
        progress_loss_weight=0.0,
        end_loss_weight=1.0,
    )
    output = {
        "actions": torch.zeros(2, 1, 7),
        "progress": torch.zeros(1, 4),
        # Invalid padding is deliberately very wrong; it must not affect loss.
        "term_logits": torch.tensor([[0.0, 0.0, 100.0, 100.0]]),
    }
    batch = {
        "ctrl": torch.zeros(1, 6, 8),
        "length": torch.tensor([2]),
        "actions": torch.zeros(1, 2, 1, 7),
        "progress": torch.zeros(1, 2),
        "termination": torch.zeros(1, 2),
        "terminator_progress": torch.zeros(1, 4),
        "terminator_termination": torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
    }

    loss, metrics = fsq_reconstruction_loss(output, batch, config)

    torch.testing.assert_close(loss, torch.tensor(np.log(2.0), dtype=torch.float32))
    torch.testing.assert_close(metrics["termination"], loss)
    torch.testing.assert_close(metrics["terminator_mean_valid_length"], torch.tensor(2.0))


def _sampling_only_dataset() -> FSQTrajectoryDataset:
    dataset = FSQTrajectoryDataset.__new__(FSQTrajectoryDataset)
    dataset.samples_per_skill = 5
    dataset.training = True
    dataset.cfg = SimpleNamespace(end_target_sigma=1.0)
    return dataset


def test_training_samples_are_uniform_over_the_full_skill(monkeypatch) -> None:
    dataset = _sampling_only_dataset()

    def full_skill_choice(high, *, size, replace):
        assert high == 10
        assert size == 5
        assert replace is False
        return np.asarray([8, 0, 6, 2, 4])

    monkeypatch.setattr(fsq_module.np.random, "choice", full_skill_choice)

    sample = dataset._sample_indices(10)

    np.testing.assert_array_equal(sample, [0, 2, 4, 6, 8])


def test_validation_samples_are_a_deterministic_linspace() -> None:
    dataset = _sampling_only_dataset()
    dataset.training = False

    np.testing.assert_array_equal(dataset._sample_indices(10), [0, 2, 4, 7, 9])


def test_reconstruction_action_loss_is_plain_sample_mean() -> None:
    config = SplineFSQAEConfig(
        action_dim=1,
        max_action_dim=1,
        chunk_size=1,
        samples_per_skill=2,
        action_loss_weight=1.0,
        progress_loss_weight=0.0,
        end_loss_weight=0.0,
    )
    output = {
        "actions": torch.tensor([[[0.0]], [[2.0]]]),
        "progress": torch.zeros(2),
        "term_logits": torch.zeros(2),
    }
    batch = {
        "ctrl": torch.zeros(1, 1, 1),
        "actions": torch.zeros(1, 2, 1, 1),
        "progress": torch.zeros(1, 2),
        "termination": torch.zeros(1, 2),
    }

    loss, metrics = fsq_reconstruction_loss(output, batch, config)

    torch.testing.assert_close(loss, torch.tensor(2.0))
    torch.testing.assert_close(metrics["action"], torch.tensor(2.0))
    assert "action_objective" not in metrics


def test_zero_scheduled_pair_weight_keeps_overlap_diagnostic_but_not_objective() -> None:
    config = SplineFSQAEConfig(
        action_dim=1,
        max_action_dim=1,
        chunk_size=1,
        samples_per_skill=1,
        action_loss_weight=1.0,
        progress_loss_weight=0.0,
        end_loss_weight=0.0,
        pair_loss="overlap",
        pair_weight=0.1,
        fsq_levels=[3],
    )
    output = {
        "actions": torch.tensor([[[1.0]]]),
        "progress": torch.zeros(1),
        "term_logits": torch.zeros(1),
        "u_cont": torch.tensor([[0.0]]),
        "augmented_u_cont": torch.tensor([[1.0]]),
        "indices": torch.tensor([1]),
        "augmented_indices": torch.tensor([2]),
    }
    batch = {
        "ctrl": torch.zeros(1, 1, 1),
        "actions": torch.zeros(1, 1, 1, 1),
        "progress": torch.zeros(1, 1),
        "termination": torch.zeros(1, 1),
    }

    loss, metrics = fsq_reconstruction_loss(
        output, batch, config, pair_weight=0.0
    )

    torch.testing.assert_close(loss, torch.tensor(1.0))
    assert metrics["pair_overlap_loss"] > 0
    torch.testing.assert_close(metrics["pair_weight"], torch.tensor(0.0))
    torch.testing.assert_close(metrics["pair_weighted_loss"], torch.tensor(0.0))


def test_js_pair_loss_is_selected_as_the_weighted_objective() -> None:
    config = SplineFSQAEConfig(
        action_dim=1,
        max_action_dim=1,
        chunk_size=1,
        samples_per_skill=1,
        action_loss_weight=1.0,
        progress_loss_weight=0.0,
        end_loss_weight=0.0,
        pair_loss="js",
        pair_weight=0.2,
        fsq_levels=[3],
    )
    output = {
        "actions": torch.tensor([[[1.0]]]),
        "progress": torch.zeros(1),
        "term_logits": torch.zeros(1),
        "u_cont": torch.tensor([[0.0]]),
        "augmented_u_cont": torch.tensor([[1.0]]),
        "indices": torch.tensor([1]),
        "augmented_indices": torch.tensor([2]),
    }
    batch = {
        "ctrl": torch.zeros(1, 1, 1),
        "actions": torch.zeros(1, 1, 1, 1),
        "progress": torch.zeros(1, 1),
        "termination": torch.zeros(1, 1),
    }

    expected_js = fsq_js_pair_loss(
        output["u_cont"], output["augmented_u_cont"], [3], 5.0
    )
    loss, metrics = fsq_reconstruction_loss(output, batch, config)

    torch.testing.assert_close(loss, 1.0 + 0.2 * expected_js)
    torch.testing.assert_close(metrics["pair_loss"], expected_js)
    torch.testing.assert_close(metrics["pair_js_loss"], expected_js)
    assert metrics["pair_overlap_loss"] > 0


def test_fsq_lr_schedule_supports_cosine_and_constant() -> None:
    assert fsq_lr_factor("constant", epoch=0, epochs=500) == 1.0
    assert fsq_lr_factor("constant", epoch=499, epochs=500) == 1.0
    assert fsq_lr_factor("cosine", epoch=0, epochs=500) == 1.0
    assert fsq_lr_factor("cosine", epoch=499, epochs=500) == 0.01


class _CountingDino(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = 0

    def forward(self, images: torch.Tensor) -> SimpleNamespace:
        self.calls += 1
        pooled = images.mean(dim=(-2, -1))
        token = torch.cat([pooled, pooled.mean(dim=-1, keepdim=True)], dim=-1)
        # CLS, one register token, then two patch tokens.
        hidden = torch.stack([token, token + 10.0, token + 20.0, token + 30.0], dim=1)
        return SimpleNamespace(last_hidden_state=hidden)


class _CountingResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = 0

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        spatial = torch.nn.functional.adaptive_avg_pool2d(
            images.mean(dim=1, keepdim=True), (7, 7)
        )
        return spatial.expand(-1, 512, -1, -1)


def _terminator_frontend(
    terminator_cls=FSQQueryTerminator,
) -> FSQQueryTerminator:
    # Avoid loading an external DINO checkpoint: these tests exercise only the
    # preprocessing/shared-forward/projection fast path.
    module = terminator_cls.__new__(terminator_cls)
    nn.Module.__init__(module)
    module.vision_backbone = "dino"
    module.freeze_vision_encoder = True
    module.dino = _CountingDino()
    module.siglip = None
    module.n_register = 1
    module.vision_image_size = 4
    module.register_buffer("_img_mean", torch.zeros(1, 3, 1, 1), persistent=False)
    module.register_buffer("_img_std", torch.ones(1, 3, 1, 1), persistent=False)
    module.image_proj = nn.Linear(4, 5, bias=False)
    return module


def test_resnet18_frontend_keeps_spatial_tokens_and_shares_one_camera_call(
    monkeypatch,
) -> None:
    tower = _CountingResNet()
    monkeypatch.setattr(
        fsq_module,
        "_build_resnet18_vision_tower",
        lambda: tower,
    )
    module = FSQQueryTerminator(
        state_dim=8,
        fsq_levels=[3, 3, 3],
        hidden_dim=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        arch="small",
        vision_backbone="resnet",
        freeze_vision_encoder=True,
        dino_model_path="unused",
        dino_image_size=224,
        siglip_image_size=224,
        resnet_image_size=224,
        skill_cond_mode="token",
        state_min=np.zeros(8, dtype=np.float32),
        state_max=np.ones(8, dtype=np.float32),
    )
    module.train()

    third = torch.rand(2, 3, 64, 64)
    wrist = torch.rand(2, 3, 64, 64)
    tokens = module._prepare_image_tokens(third, wrist)

    assert tokens.shape == (2, 98, 32)  # 49 third + 49 wrist spatial tokens
    assert tower.calls == 1
    assert tower.training is False
    assert all(not parameter.requires_grad for parameter in tower.parameters())


def test_fusion_terminator_uses_skill_token_readout_and_reuses_vision_for_shuffle(
    monkeypatch,
) -> None:
    tower = _CountingResNet()
    monkeypatch.setattr(
        fsq_module,
        "_build_resnet18_vision_tower",
        lambda: tower,
    )
    module = FSQQueryTerminator(
        state_dim=8,
        fsq_levels=[3, 3, 3],
        hidden_dim=32,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        arch="fusion",
        vision_backbone="resnet",
        freeze_vision_encoder=True,
        dino_model_path="unused",
        dino_image_size=224,
        siglip_image_size=224,
        resnet_image_size=224,
        # Deliberately accepted but unused: fusion has one uniform token route.
        skill_cond_mode="broadcast",
        state_min=np.zeros(8, dtype=np.float32),
        state_max=np.ones(8, dtype=np.float32),
    )
    assert not hasattr(module, "progress_query")
    assert not any(
        isinstance(child, fsq_module.ConditionalRMSNorm)
        for child in module.modules()
    )
    assert all(
        isinstance(layer, fsq_module.MultimodalFusionLayer)
        for layer in module.layers
    )

    z_norm = torch.tensor(
        [[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]],
        requires_grad=True,
    )
    raw_state = torch.rand(2, 8)
    third = torch.rand(2, 3, 64, 64)
    wrist = torch.rand(2, 3, 64, 64)
    progress, logits, shuffled_progress, shuffled_logits = (
        module.forward_with_skill_shuffle(
            z_norm,
            z_norm.flip(0),
            raw_state,
            third,
            wrist,
        )
    )

    assert progress.shape == logits.shape == (2,)
    assert shuffled_progress.shape == shuffled_logits.shape == (2,)
    assert tower.calls == 1
    assert not torch.allclose(logits, shuffled_logits)
    (progress.sum() + logits.sum()).backward()
    assert z_norm.grad is not None
    assert z_norm.grad.abs().sum() > 0


def test_different_code_shuffle_sources_avoids_same_code_when_possible() -> None:
    sources, valid = fsq_module.different_code_shuffle_sources(
        torch.tensor([2, 2, 5, 5])
    )
    codes = torch.tensor([2, 2, 5, 5])

    assert valid.tolist() == [True, True, True, True]
    assert torch.all(codes[sources] != codes)

    collapsed_sources, collapsed_valid = fsq_module.different_code_shuffle_sources(
        torch.tensor([3, 3])
    )
    assert collapsed_sources.tolist() == [0, 1]
    assert collapsed_valid.tolist() == [False, False]


def test_bsq_fusion_full_model_emits_shuffle_diagnostics_with_one_vision_call(
    monkeypatch,
) -> None:
    tower = _CountingResNet()
    monkeypatch.setattr(
        fsq_module,
        "_build_resnet18_vision_tower",
        lambda: tower,
    )
    config = _state_rnn_config(
        quantizer="bsq",
        bsq_code_dim=5,
        terminator_arch="fusion",
        terminator_input_space="both",
        terminator_model="default",
        terminator_progress=True,
        terminator_termination=True,
        state_rnn_terminator=False,
        terminator_termination_only=False,
        terminator_only=True,
        vision_backbone="resnet",
        freeze_vision_encoder=True,
        resnet_image_size=224,
        image_encoder_layers=1,
        image_encoder_heads=4,
    )
    model = SplineFSQAE(config).eval()
    bsize, samples = 2, 2
    output = model(
        ctrl=torch.randn(bsize, config.n_control, config.enc_dim),
        lengths=torch.tensor([4, 5]),
        start_pose=None,
        start_state=torch.randn(bsize * samples, config.max_state_dim),
        raw_state=torch.randn(bsize * samples, config.state_dim),
        progress_target=torch.rand(bsize * samples),
        third=torch.rand(bsize * samples, 3, 64, 64),
        wrist=torch.rand(bsize * samples, 3, 64, 64),
        samples_per_skill=samples,
        compute_skill_shuffle=True,
    )

    assert model.fsq.code_dim == 5
    assert output["progress"].shape == (bsize * samples,)
    assert output["skill_shuffle_progress"].shape == (bsize * samples,)
    assert output["skill_shuffle_valid"].shape == (bsize,)
    assert tower.calls == 1


def test_top_and_wrist_share_one_dino_call_without_changing_token_order() -> None:
    module = _terminator_frontend().eval()
    third = torch.linspace(0.0, 1.0, 2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
    wrist = torch.flip(third, dims=(-1,))

    # Historical reference: one shared tower called separately for each camera.
    third_features = module._image_features(third)
    wrist_features = module._image_features(wrist)
    expected = torch.cat(
        [
            module.image_proj(third_features.to(module.image_proj.weight.dtype)),
            module.image_proj(wrist_features.to(module.image_proj.weight.dtype)),
        ],
        dim=1,
    )
    assert module.dino.calls == 2

    module.dino.calls = 0
    actual = module._prepare_image_tokens(third, wrist)

    assert module.dino.calls == 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_wrist_only_frontend_encodes_only_wrist_tokens() -> None:
    module = _terminator_frontend(FSQWristOnlyQueryTerminator).eval()
    wrist = torch.linspace(0.0, 1.0, 2 * 3 * 4 * 4).reshape(2, 3, 4, 4)

    expected = module.image_proj(
        module._image_features(wrist).to(module.image_proj.weight.dtype)
    )
    assert module.dino.calls == 1

    module.dino.calls = 0
    actual = module._prepare_wrist_tokens(wrist)

    assert module.dino.calls == 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_uint8_and_zero_one_float_image_contracts_match() -> None:
    module = _terminator_frontend().eval()
    uint8_image = torch.arange(48, dtype=torch.uint8).reshape(1, 3, 4, 4)

    integer_input = module._preprocess_image(uint8_image)
    float_input = module._preprocess_image(uint8_image.float() / 255.0)

    torch.testing.assert_close(integer_input, float_input, rtol=0, atol=0)


def test_dtype_aligned_rmsnorm_avoids_mixed_dtype_fallback_warning() -> None:
    norm = DtypeAlignedRMSNorm(8)
    values = torch.randn(2, 4, 8, dtype=torch.bfloat16)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output = norm(values)

    assert output.dtype == values.dtype
    assert not any("Mismatch dtype between input and weight" in str(item.message) for item in caught)
    assert norm.weight.dtype == torch.float32
    assert set(norm.state_dict()) == {"weight"}


def test_image_only_builder_uses_fsq_config_but_no_fsq_model_weights(
    monkeypatch,
) -> None:
    config = SplineFSQAEConfig(
        vision_backbone="dino",
        dino_model_path="pretrained-dino",
    )

    class _FreshImageTerminator(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.anchor = nn.Parameter(torch.zeros(()))

    monkeypatch.setattr(
        fsq_module.torch,
        "load",
        lambda *args, **kwargs: {"cfg": config},
    )
    monkeypatch.setattr(
        fsq_module,
        "FSQImageOnlyQueryTerminator",
        _FreshImageTerminator,
    )
    monkeypatch.setattr(
        fsq_module,
        "load_fsq_terminator",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("image-only builder must not load FSQ terminator weights")
        ),
    )

    terminator, loaded_config = fsq_module.build_fsq_image_only_terminator(
        "FSQ.pt"
    )

    assert loaded_config is config
    assert terminator.kwargs["dino_model_path"] == "pretrained-dino"
    assert terminator.training is False


def test_trainable_terminator_accepts_fsq_original_as_fresh_contract(
    monkeypatch,
) -> None:
    state_min = np.arange(8, dtype=np.float32)
    state_max = state_min + 10.0
    config = FSQOriginalConfig(
        enc_dim=8,
        hidden_dim=256,
        fsq_levels=[3, 3, 3],
        num_layers=3,
        num_heads=4,
        encoder_input_mode="raw_state",
        encoder_min=state_min,
        encoder_max=state_max,
    )

    class _FreshStateImageTerminator(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.anchor = nn.Parameter(torch.zeros(()))

    monkeypatch.setattr(
        fsq_module.torch,
        "load",
        lambda *args, **kwargs: {
            "cfg": config,
            "model_state": {"encoder.unused": torch.ones(())},
        },
    )
    monkeypatch.setattr(
        fsq_module,
        "FSQQueryTerminator",
        _FreshStateImageTerminator,
    )

    terminator, loaded_config = fsq_module.build_trainable_fsq_terminator(
        "FSQ.pt"
    )

    assert loaded_config is config
    assert terminator.kwargs["state_dim"] == 8
    assert terminator.kwargs["fsq_levels"] == [3, 3, 3]
    assert terminator.kwargs["n_layers"] == 3
    assert terminator.kwargs["n_heads"] == 4
    assert terminator.kwargs["skill_cond_mode"] == "broadcast"
    np.testing.assert_array_equal(terminator.kwargs["state_min"], state_min)
    np.testing.assert_array_equal(terminator.kwargs["state_max"], state_max)
    assert terminator.training is False


def test_trainable_terminator_keeps_v3_warm_start(monkeypatch) -> None:
    config = SplineFSQAEConfig()

    class _WarmStartedTerminator(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.anchor = nn.Parameter(torch.zeros(()))

    monkeypatch.setattr(
        fsq_module.torch,
        "load",
        lambda *args, **kwargs: {
            "cfg": config,
            "model_state": {"terminator.anchor": torch.tensor(3.0)},
        },
    )
    monkeypatch.setattr(
        fsq_module,
        "FSQQueryTerminator",
        _WarmStartedTerminator,
    )

    terminator, loaded_config = fsq_module.build_trainable_fsq_terminator(
        "FSQ.pt"
    )

    assert loaded_config is config
    assert terminator.anchor.item() == 3.0
    assert terminator.training is False


def test_trainable_terminator_initializes_fresh_from_reconstructor_only_v3(
    monkeypatch,
) -> None:
    config = SplineFSQAEConfig(reconstructor_only=True)

    class _FreshStateImageTerminator(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.anchor = nn.Parameter(torch.zeros(()))

    monkeypatch.setattr(
        fsq_module.torch,
        "load",
        lambda *args, **kwargs: {
            "cfg": config,
            "model_state": {
                "encoder.unused": torch.ones(()),
                "reconstructor.unused": torch.ones(()),
            },
        },
    )
    monkeypatch.setattr(
        fsq_module,
        "FSQQueryTerminator",
        _FreshStateImageTerminator,
    )

    terminator, loaded_config = fsq_module.build_trainable_fsq_terminator(
        "FSQ.pt"
    )

    assert loaded_config is config
    assert terminator.anchor.item() == 0.0
    assert terminator.kwargs["arch"] == config.terminator_arch
    assert terminator.kwargs["vision_backbone"] == config.vision_backbone
    assert terminator.training is False


def test_pristine_fsq_terminator_still_rejects_fsq_original(monkeypatch) -> None:
    config = FSQOriginalConfig(
        encoder_min=np.zeros(8, dtype=np.float32),
        encoder_max=np.ones(8, dtype=np.float32),
    )
    monkeypatch.setattr(
        fsq_module.torch,
        "load",
        lambda *args, **kwargs: {"cfg": config, "model_state": {}},
    )

    with pytest.raises(ValueError, match="Legacy FSQ checkpoint is unsupported"):
        fsq_module.load_fsq_terminator("FSQ.pt")


def test_wrist_only_builder_uses_fsq_config_but_no_fsq_model_weights(
    monkeypatch,
) -> None:
    config = SplineFSQAEConfig(
        vision_backbone="dino",
        dino_model_path="pretrained-dino",
    )

    class _FreshWristTerminator(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.anchor = nn.Parameter(torch.zeros(()))

    monkeypatch.setattr(
        fsq_module.torch,
        "load",
        lambda *args, **kwargs: {"cfg": config},
    )
    monkeypatch.setattr(
        fsq_module,
        "FSQWristOnlyQueryTerminator",
        _FreshWristTerminator,
    )
    monkeypatch.setattr(
        fsq_module,
        "load_fsq_terminator",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("wrist-only builder must not load FSQ terminator weights")
        ),
    )

    terminator, loaded_config = fsq_module.build_fsq_wrist_only_terminator(
        "FSQ.pt"
    )

    assert loaded_config is config
    assert terminator.kwargs["dino_model_path"] == "pretrained-dino"
    assert terminator.training is False
