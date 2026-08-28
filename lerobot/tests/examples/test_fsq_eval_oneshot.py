"""fsq_eval support for the oneshot (control-point) reconstructor."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "lerobot/examples/libero"))
sys.path.insert(0, str(_ROOT / "lerobot/src"))

import fsq_eval as FE  # noqa: E402
from FSQ import (  # noqa: E402
    BSQ,
    N_GRIPPER_DIMS,
    SplineFSQAE,
    SplineFSQAEConfig,
    START_GROUNDED_CONVENTION,
    load_fsq_model,
)


def _config(**overrides) -> SplineFSQAEConfig:
    enc_dim = 8
    base = dict(
        enc_dim=enc_dim,
        n_control=6,
        spline_degree=3,
        action_dim=7,
        max_action_dim=7,
        state_dim=enc_dim,
        max_state_dim=enc_dim,
        chunk_size=4,
        hidden_dim=32,
        num_layers=1,
        fsq_levels=[3, 3, 3],
        encoder_input_mode="raw_state",
        encoder_min=np.full(enc_dim, -1.0, dtype=np.float32),
        encoder_max=np.full(enc_dim, 1.0, dtype=np.float32),
        reconstructor_output_mode="raw_state",
        reconstructor_min=np.full(enc_dim, -1.0, dtype=np.float32),
        reconstructor_max=np.full(enc_dim, 1.0, dtype=np.float32),
        state_min=np.full(enc_dim, -1.0, dtype=np.float32),
        state_max=np.full(enc_dim, 1.0, dtype=np.float32),
        state_q01=np.full(enc_dim, -1.0, dtype=np.float32),
        state_q99=np.full(enc_dim, 1.0, dtype=np.float32),
        action_q01=np.full(7, -1.0, dtype=np.float32),
        action_q99=np.full(7, 1.0, dtype=np.float32),
    )
    base.update(overrides)
    return SplineFSQAEConfig(**base)


def _model(**overrides) -> SplineFSQAE:
    return SplineFSQAE(_config(**overrides)).eval()


def test_joint_model_accepts_only_matching_start_grounded_convention() -> None:
    overrides = dict(
        encoder_input_mode="start_grounded",
        encoder_grounding_convention=START_GROUNDED_CONVENTION,
        reconstructor_arch="oneshot",
        reconstructor_output_mode="start_grounded",
        reconstructor_start_state=False,
        reconstructor_only=True,
    )
    model = _model(**overrides)
    assert model.encoder.encoder_input_mode == "start_grounded"
    assert model.cfg.reconstructor_output_mode == "start_grounded"

    overrides["encoder_grounding_convention"] = "trajectory_mean_xyz_v1"
    with pytest.raises(ValueError, match="does not match encoder_input_mode"):
        _model(**overrides)


def test_joint_model_can_replace_fsq_grid_with_bsq5() -> None:
    model = _model(
        quantizer="bsq",
        bsq_code_dim=5,
        reconstructor_arch="oneshot",
        reconstructor_start_state=False,
        reconstructor_only=True,
    )

    assert isinstance(model.fsq, BSQ)
    assert model.encoder.z_head.out_features == 5
    assert model.reconstructor.mlp[0].net[0].in_features == 5
    assert model.cfg.fsq_levels == [2, 2, 2, 2, 2]
    assert model.fsq.codebook_size == 32


def test_joint_bsq_checkpoint_round_trip(tmp_path: Path) -> None:
    model = _model(
        quantizer="bsq",
        bsq_code_dim=5,
        reconstructor_arch="oneshot",
        reconstructor_start_state=False,
        reconstructor_only=True,
    )
    path = tmp_path / "BSQ_epoch0001.pt"
    torch.save({"cfg": model.cfg, "model_state": model.state_dict()}, path)

    restored, restored_cfg = load_fsq_model(path)

    assert isinstance(restored.fsq, BSQ)
    assert restored_cfg.quantizer == "bsq"
    assert restored_cfg.bsq_code_dim == 5
    assert restored.encoder.z_head.out_features == 5


def test_is_oneshot_reads_the_checkpoint_arch() -> None:
    assert FE.is_oneshot(_config(reconstructor_arch="oneshot"))
    assert not FE.is_oneshot(_config())
    # Checkpoints predating the flag are chunk models.
    assert not FE.is_oneshot(_config().__class__(**{}))


@pytest.mark.parametrize("start_state", [False, True])
def test_sample_control_points_returns_one_grid_per_skill(start_state: bool) -> None:
    """The grid is per skill, not per timestep -- that is the whole point of oneshot."""
    model = _model(reconstructor_arch="oneshot", reconstructor_start_state=start_state)
    z_q, _ = model.fsq(torch.zeros(3, 3))
    states = torch.zeros(3, model.cfg.state_dim)

    ctrl = model.sample_control_points(z_q, states)

    assert ctrl.shape == (3, model.cfg.n_control, model.cfg.enc_dim)
    assert torch.isfinite(ctrl).all()


def test_sample_control_points_uses_reconstructor_output_statistics() -> None:
    model = _model(
        reconstructor_arch="oneshot",
        reconstructor_start_state=False,
        reconstructor_output_mode="zero_grounded",
        reconstructor_min=np.full(8, 10.0, dtype=np.float32),
        reconstructor_max=np.full(8, 20.0, dtype=np.float32),
    )
    for parameter in model.reconstructor.parameters():
        torch.nn.init.zeros_(parameter)
    z_q, _ = model.fsq(torch.zeros(1, 3))

    ctrl = model.sample_control_points(z_q)

    torch.testing.assert_close(ctrl, torch.full_like(ctrl, 15.0))


def test_sample_control_points_restores_weighted_gripper_state_axes() -> None:
    model = _model(
        autoencoder_mode="raw",
        action_gripper_weight=0.25,
        reconstructor_arch="oneshot",
        reconstructor_start_state=False,
        reconstructor_output_mode="raw_state",
        reconstructor_min=np.zeros(8, dtype=np.float32),
        reconstructor_max=np.full(8, 10.0, dtype=np.float32),
        reconstructor_only=True,
    )

    class _FixedWeightedControlPoints(torch.nn.Module):
        state_dim = 0

        def forward(self, z_norm, start_state=None):
            values = torch.zeros(
                z_norm.shape[0], model.cfg.n_control, model.cfg.enc_dim
            )
            values[..., -N_GRIPPER_DIMS:] = 0.5
            return values, None

    model.reconstructor = _FixedWeightedControlPoints()
    z_q, _ = model.fsq(torch.zeros(1, 3))
    ctrl = model.sample_control_points(z_q)

    torch.testing.assert_close(ctrl[..., :-2], torch.full_like(ctrl[..., :-2], 5.0))
    torch.testing.assert_close(ctrl[..., -2:], torch.full_like(ctrl[..., -2:], 10.0))


def test_sample_control_points_needs_start_states_when_the_decoder_takes_them() -> None:
    model = _model(reconstructor_arch="oneshot", reconstructor_start_state=True)
    z_q, _ = model.fsq(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="start-state"):
        model.sample_control_points(z_q, None)


def test_oneshot_start_state_adaln_keeps_code_and_context_paths_separate() -> None:
    model = _model(
        autoencoder_mode="raw",
        reconstructor_arch="oneshot",
        reconstructor_start_state=True,
        reconstructor_start_state_conditioning="adaln",
        reconstructor_only=True,
    )
    decoder = model.reconstructor
    assert decoder.mlp is None
    assert decoder.z_proj.net[0].in_features == len(model.cfg.fsq_levels)
    assert decoder.state_proj[0].in_features == model.cfg.state_dim

    z_q, _ = model.fsq(torch.zeros(2, 3))
    starts = torch.tensor(
        [[-1.0] * model.cfg.state_dim, [1.0] * model.cfg.state_dim]
    )
    ctrl, _ = decoder(model.fsq.normalized(z_q), start_state=starts)
    ctrl.square().mean().backward()

    modulation = decoder.output_adaln[-1]
    assert modulation.weight.grad is not None
    assert torch.count_nonzero(modulation.weight.grad) > 0


def test_sample_control_points_rejects_the_chunk_reconstructor() -> None:
    model = _model()
    z_q, _ = model.fsq(torch.zeros(2, 3))
    with pytest.raises(RuntimeError, match="oneshot"):
        model.sample_control_points(z_q, torch.zeros(2, model.cfg.state_dim))


def test_traj_dim_groups_puts_gripper_state_in_the_trailing_dims() -> None:
    """An encoder trajectory keeps N_GRIPPER_DIMS gripper dims, unlike an action."""
    assert FE._traj_dim_groups(8) == {
        "xyz": [0, 1, 2],
        "rpy": [3, 4, 5],
        "gripper": list(range(8 - N_GRIPPER_DIMS, 8)),
    }


def test_ctrl_and_chunk_metrics_agree_on_their_keys() -> None:
    """Downstream charts, panels and aggregation read one metric schema."""
    T, dim = 5, 8
    traj = np.zeros((T, dim), dtype=np.float32)
    progress = np.linspace(0.0, 1.0, T, dtype=np.float32)
    term = np.zeros(T, dtype=np.float32)
    term[-1] = 1.0

    ctrl = FE.ctrl_skill_metrics(
        (np.zeros((6, dim), np.float32), traj), progress, term, traj, T, 0.5,
        FE._traj_dim_groups(dim),
    )
    chunk = FE.skill_metrics(
        np.zeros((T, 4, 7), np.float32), progress, term, np.zeros((T, 7), np.float32),
        T, 0.5, FE._dim_groups(7),
    )

    assert set(ctrl) == set(chunk)
    assert ctrl["chunk_mse"] == 0.0
    assert ctrl["timing"] == 0 and ctrl["prog_err"] == 0.0


def test_ctrl_metrics_score_the_decoded_trajectory_not_the_control_points() -> None:
    T, dim = 4, 8
    gt = np.zeros((T, dim), dtype=np.float32)
    traj = np.full((T, dim), 2.0, dtype=np.float32)
    # Control points are deliberately far off: only the trajectory is scored.
    recon = (np.full((6, dim), 99.0, np.float32), traj)
    progress = np.linspace(0.0, 1.0, T, dtype=np.float32)
    term = np.zeros(T, dtype=np.float32)

    metrics = FE.ctrl_skill_metrics(recon, progress, term, gt, T, 0.5, FE._traj_dim_groups(dim))

    assert metrics["chunk_mse"] == pytest.approx(4.0)
    assert metrics["mse_xyz"] == pytest.approx(4.0)
