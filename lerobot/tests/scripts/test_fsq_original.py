from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/libero"))
sys.path.insert(0, str(ROOT / "src"))

from FSQ import (  # noqa: E402
    encoder_grounding_position,
    prepare_encoder_trajectory,
    spline_encode,
)
from FSQ_original import (  # noqa: E402
    BSQ,
    FSQOriginalConfig,
    FSQOriginalDataset,
    SplineFSQOriginalAE,
    bsq_entropy_terms,
    coverage_loss,
    fsq_entropy_terms,
    fsq_original_loss,
    spline_decode,
)

ENC_DIM = 8
N_CONTROL = 12


def _segments(count: int = 6) -> tuple[list[np.ndarray], list[dict]]:
    rng = np.random.default_rng(0)
    segments, metadata = [], []
    for i in range(count):
        length = int(rng.integers(12, 40))
        t = np.linspace(0.0, 1.0, length)[:, None]
        # Low-frequency harmonics only, so N_CONTROL spline points can represent them.
        harmonics = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        pose = np.sin(2 * np.pi * (t + i / 5.0) * harmonics[None]).astype(np.float32)
        gripper = np.tile((t > 0.5).astype(np.float32), (1, 2)) * 0.04
        segments.append(np.concatenate([pose, gripper], axis=1).astype(np.float32))
        metadata.append({"episode_id": i, "skill_index": 0, "length": length})
    return segments, metadata


def _config(segments: list[np.ndarray], mode: str = "optimal") -> FSQOriginalConfig:
    prep = np.concatenate([prepare_encoder_trajectory(s, mode) for s in segments])
    lengths = [len(s) for s in segments]
    start_min = start_max = None
    if mode == "optimal":
        starts = np.stack([encoder_grounding_position(s) for s in segments])
        start_min, start_max = starts.min(0), starts.max(0)
    return FSQOriginalConfig(
        enc_dim=ENC_DIM,
        n_control=N_CONTROL,
        hidden_dim=32,
        fsq_levels=[3, 3, 3],
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        decoder_layers=2,
        encoder_input_mode=mode,
        length_min=float(min(lengths)),
        length_max=float(max(lengths)),
        encoder_min=prep.min(0),
        encoder_max=prep.max(0),
        encoder_start_min=start_min,
        encoder_start_max=start_max,
        device="cpu",
    )


def test_spline_decode_round_trips_encoder_input() -> None:
    segments, _ = _segments()
    ctrl, length = spline_encode(segments[0], N_CONTROL, 3, input_mode="zero_grounded")
    recon = spline_decode(ctrl, length, 3)
    target = prepare_encoder_trajectory(segments[0], "zero_grounded")
    assert recon.shape == target.shape
    # XYZ is centered around the skill mean; rotation and gripper stay absolute.
    assert np.abs(recon[:, :3].mean(0)).max() < 0.1
    assert float(np.abs(recon - target).mean()) < 0.1


def test_grounding_centers_only_xyz_and_optimal_uses_that_mean() -> None:
    trajectory = np.asarray(
        [
            [1.0, 3.0, 5.0, 0.1, 0.2, 0.3, 0.01, 0.02],
            [3.0, 5.0, 7.0, 0.4, 0.5, 0.6, 0.03, 0.04],
            [5.0, 7.0, 9.0, 0.7, 0.8, 0.9, 0.05, 0.06],
        ],
        dtype=np.float32,
    )

    grounded = prepare_encoder_trajectory(trajectory, "zero_grounded")
    optimal = prepare_encoder_trajectory(trajectory, "optimal")

    np.testing.assert_allclose(grounded[:, :3].mean(0), np.zeros(3), atol=1e-6)
    np.testing.assert_array_equal(grounded[:, 3:], trajectory[:, 3:])
    np.testing.assert_array_equal(optimal, grounded)
    np.testing.assert_allclose(
        encoder_grounding_position(trajectory), trajectory[:, :3].mean(0)
    )
    assert encoder_grounding_position(trajectory).shape == (3,)


def test_one_shot_forward_reconstructs_input_shapes() -> None:
    segments, metadata = _segments()
    cfg = _config(segments)
    dataset = FSQOriginalDataset(segments, metadata, cfg)
    model = SplineFSQOriginalAE(cfg)
    batch = {
        "ctrl": torch.stack([dataset[i]["ctrl"] for i in range(4)]),
        "length": torch.stack([dataset[i]["length"] for i in range(4)]),
        "length_target": torch.stack([dataset[i]["length_target"] for i in range(4)]),
        "start_pose": torch.stack([dataset[i]["start_pose"] for i in range(4)]),
    }
    output = model(batch["ctrl"], batch["length"], batch["start_pose"])
    assert output["ctrl_hat"].shape == batch["ctrl"].shape
    assert output["length_hat"].shape == (4,)
    loss, metrics = fsq_original_loss(output, batch, cfg)
    loss.backward()
    assert set(metrics) == {"loss", "ctrl", "length", "ctrl_pose", "ctrl_gripper"}


def test_reconstruct_numpy_matches_input_convention() -> None:
    segments, metadata = _segments()
    cfg = _config(segments)
    model = SplineFSQOriginalAE(cfg)
    recon_true_len = model.reconstruct_numpy(segments[0], use_true_length=True)
    assert recon_true_len.shape == segments[0].shape
    recon_pred_len = model.reconstruct_numpy(segments[0], use_true_length=False)
    assert recon_pred_len.ndim == 2 and recon_pred_len.shape[1] == ENC_DIM
    index = model.encode_index(segments[0])
    decoded = model.decode_index_numpy(index)
    assert 0 <= index < model.fsq.codebook_size
    assert decoded.ndim == 2 and decoded.shape[1] == ENC_DIM


def test_reconstruct_length_false_drops_length_head_and_loss() -> None:
    segments, metadata = _segments()
    cfg = _config(segments)
    cfg.reconstruct_length = False
    dataset = FSQOriginalDataset(segments, metadata, cfg)
    model = SplineFSQOriginalAE(cfg)
    assert model.decoder.length_head is None
    batch = {
        "ctrl": torch.stack([dataset[i]["ctrl"] for i in range(3)]),
        "length": torch.stack([dataset[i]["length"] for i in range(3)]),
        "length_target": torch.stack([dataset[i]["length_target"] for i in range(3)]),
        "start_pose": torch.stack([dataset[i]["start_pose"] for i in range(3)]),
    }
    output = model(batch["ctrl"], batch["length"], batch["start_pose"])
    assert output["length_hat"] is None
    loss, metrics = fsq_original_loss(output, batch, cfg)
    loss.backward()
    assert metrics["length"].item() == 0.0
    # Decoding still works with an explicit length, and refuses without one.
    recon = model.reconstruct_numpy(segments[0], use_true_length=True)
    assert recon.shape == segments[0].shape
    with pytest.raises(ValueError, match="reconstruct_length=False"):
        model.reconstruct_numpy(segments[0], use_true_length=False)
    index = model.encode_index(segments[0])
    assert model.decode_index_numpy(index, length=20).shape == (20, ENC_DIM)
    with pytest.raises(ValueError, match="reconstruct_length=False"):
        model.decode_index_numpy(index)


def _actions_for(segments: list[np.ndarray]) -> list[np.ndarray]:
    rng = np.random.default_rng(1)
    return [rng.normal(size=(len(s), 7)).astype(np.float32) for s in segments]


def _rnn_config(segments: list[np.ndarray], actions: list[np.ndarray]) -> FSQOriginalConfig:
    cfg = _config(segments)
    cfg.decoder_arch = "rnn"
    cfg.action_dim = 7
    all_actions = np.concatenate(actions)
    cfg.action_q01 = np.quantile(all_actions, 0.01, axis=0).astype(np.float32)
    cfg.action_q99 = np.quantile(all_actions, 0.99, axis=0).astype(np.float32)
    return cfg


def test_rnn_arch_forward_loss_and_padding_mask() -> None:
    segments, metadata = _segments()
    actions = _actions_for(segments)
    cfg = _rnn_config(segments, actions)
    dataset = FSQOriginalDataset(segments, metadata, cfg, actions=actions)
    model = SplineFSQOriginalAE(cfg)
    batch = {
        key: torch.stack([dataset[i][key] for i in range(3)])
        for key in ("ctrl", "length", "length_target", "start_pose", "actions_norm")
    }
    max_len = int(batch["length"].max())
    output = model(batch["ctrl"], batch["length"], batch["start_pose"], unroll_steps=max_len)
    assert output["actions_hat"].shape == (3, max_len, 7)
    assert output["term_logits"].shape == (3, max_len)
    loss, metrics = fsq_original_loss(output, batch, cfg)
    loss.backward()
    assert set(metrics) == {"loss", "action", "termination"}
    # Steps past each skill's length are masked: extending the unroll must not
    # change the loss value.
    with torch.no_grad():
        longer = model(batch["ctrl"], batch["length"], batch["start_pose"], unroll_steps=max_len + 9)
        loss_longer, _ = fsq_original_loss(longer, batch, cfg)
    assert abs(float(loss_longer) - float(loss)) < 1e-5


def test_rnn_rollout_and_oneshot_guards() -> None:
    segments, metadata = _segments()
    actions = _actions_for(segments)
    cfg = _rnn_config(segments, actions)
    model = SplineFSQOriginalAE(cfg)
    rolled, terminated = model.rollout_actions_numpy(segments[0], max_steps=25)
    assert rolled.ndim == 2 and rolled.shape[1] == 7 and 1 <= len(rolled) <= 25
    assert isinstance(terminated, bool)
    index = model.encode_index(segments[0])
    rolled_idx, _ = model.rollout_index_numpy(index, max_steps=10)
    assert rolled_idx.shape[1] == 7 and len(rolled_idx) <= 10
    with pytest.raises(ValueError, match="oneshot-only"):
        model.decode_index_numpy(index, length=10)
    with pytest.raises(ValueError, match="oneshot-only"):
        model.reconstruct_numpy(segments[0])
    oneshot = SplineFSQOriginalAE(_config(segments))
    with pytest.raises(ValueError, match="decoder_arch='rnn'"):
        oneshot.rollout_actions_numpy(segments[0])


def test_bsq_quantizer_contract() -> None:
    torch.manual_seed(0)
    bsq = BSQ(code_dim=4, inv_temperature=10.0)
    z = torch.randn(16, 4)
    z_q, index = bsq(z)
    assert bsq.codebook_size == 16
    assert z_q.shape == (16, 4) and index.shape == (16,)
    assert int(index.min()) >= 0 and int(index.max()) < 16
    # Quantized points are sphere corners; normalized() maps them to ±1 bits.
    torch.testing.assert_close(z_q.norm(dim=-1), torch.ones(16), atol=1e-5, rtol=0)
    norm = bsq.normalized(z_q)
    torch.testing.assert_close(norm.abs(), torch.ones_like(norm), atol=1e-5, rtol=0)
    # code_to_normalized round-trips the forward index's bit pattern.
    torch.testing.assert_close(bsq.code_to_normalized(index), norm)
    # Margins: 0 on a bit boundary, 0.5 at a corner coordinate.
    margins = bsq.boundary_margin(z)
    assert margins.shape == (16, 4)
    assert float(margins.min()) >= 0.0 and float(margins.max()) <= 0.5
    corner = torch.ones(1, 4)
    torch.testing.assert_close(bsq.boundary_margin(corner), torch.full((1, 4), 0.5))


def test_bsq_joint_entropy_sees_antipodal_collapse() -> None:
    # Antipodal pair: half the batch at corner c, half at its complement ~c.
    # Every bit marginal is exactly 50/50, so the factorized dataset entropy is
    # maximal (L*ln2) — blind to the collapse. The exact joint entropy sees a
    # two-code distribution (ln2) and penalizes it.
    code_dim = 5
    corner = torch.tensor([[1.0, -1.0, 1.0, 1.0, -1.0]]) / math.sqrt(code_dim)
    u = torch.cat([corner.expand(8, -1), (-corner).expand(8, -1)])
    _, dataset_factorized = bsq_entropy_terms(u, 10.0, joint_dataset=False)
    _, dataset_joint = bsq_entropy_terms(u, 10.0, joint_dataset=True)
    assert abs(float(dataset_factorized) - code_dim * math.log(2)) < 1e-3
    assert abs(float(dataset_joint) - math.log(2)) < 1e-2
    # A balanced spread over many codes must score HIGHER joint entropy.
    torch.manual_seed(0)
    spread = torch.nn.functional.normalize(torch.randn(64, code_dim), dim=-1)
    _, dataset_joint_spread = bsq_entropy_terms(spread, 10.0, joint_dataset=True)
    assert float(dataset_joint_spread) > float(dataset_joint) + 1.0
    # Sample entropy is exact either way and must be identical.
    sample_a, _ = bsq_entropy_terms(u, 10.0, joint_dataset=False)
    sample_b, _ = bsq_entropy_terms(u, 10.0, joint_dataset=True)
    torch.testing.assert_close(sample_a, sample_b)
    # Gradient flows through the joint term.
    u_grad = u.clone().requires_grad_(True)
    _, h = bsq_entropy_terms(u_grad, 10.0, joint_dataset=True)
    h.backward()
    assert u_grad.grad is not None and torch.isfinite(u_grad.grad).all()


def test_fsq_entropy_terms_confidence_and_collapse_detection() -> None:
    levels = [3, 3, 3]
    # All samples exactly on bin centers -> confident: sample entropy ~ 0.
    centered = torch.tensor([[0.0, 1.0, -1.0], [1.0, 0.0, 0.0]])
    sample_c, _ = fsq_entropy_terms(centered, levels, 10.0)
    assert float(sample_c) < 0.01
    # A sample on a rounding boundary (0.5) is maximally uncertain in that dim.
    boundary = torch.tensor([[0.5, 0.0, 0.0]])
    sample_b, _ = fsq_entropy_terms(boundary, levels, 10.0)
    assert float(sample_b) > 0.5 * math.log(2)
    # FSQ analog of the antipodal collapse: half at (-1,-1,-1), half at (1,1,1).
    # Per-dim marginals look spread (ln2 each) but the joint has TWO codes (ln2).
    anti = torch.cat([
        torch.full((8, 3), -1.0), torch.full((8, 3), 1.0)
    ])
    _, dataset_fact = fsq_entropy_terms(anti, levels, 10.0, joint_dataset=False)
    _, dataset_joint = fsq_entropy_terms(anti, levels, 10.0, joint_dataset=True)
    assert abs(float(dataset_fact) - 3 * math.log(2)) < 1e-2
    assert abs(float(dataset_joint) - math.log(2)) < 1e-2
    # Gradient flows through both terms.
    grad_in = anti.clone().requires_grad_(True)
    s, d = fsq_entropy_terms(grad_in, levels, 10.0, joint_dataset=True)
    (s + d).backward()
    assert grad_in.grad is not None and torch.isfinite(grad_in.grad).all()


def test_coverage_loss_revives_dead_codes_without_flattening() -> None:
    floor = 1.0 / 32  # 0.031 — below every living code's 5% share
    # All 8 codes above the floor, HEAVILY skewed (one holds 65%): pressure must
    # be exactly zero — coverage never touches living codes' shares.
    skewed = torch.tensor([[0.65, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]]).repeat(4, 1)
    assert float(coverage_loss(skewed, floor)) == 0.0
    # Two dead codes -> positive loss proportional to shortfall.
    dead = torch.tensor([[0.5, 0.5, 0.0, 0.0]]).repeat(4, 1)
    assert float(coverage_loss(dead, floor)) > 0.0
    # Gradient flows toward raising the dead codes' mass.
    q = torch.tensor([[0.6, 0.399, 0.001, 0.0]]).repeat(4, 1).requires_grad_(True)
    coverage_loss(q, floor).backward()
    assert q.grad is not None and float(q.grad[0, 2]) < 0 and float(q.grad[0, 3]) < 0
    assert float(q.grad[0, 0]) == 0.0  # 살아있는 대형 클러스터는 무압력


def test_coverage_wires_into_loss() -> None:
    segments, metadata = _segments()
    cfg = _config(segments)
    cfg.reconstruct_length = False
    cfg.fsq_entropy = True
    cfg.bsq_entropy_cov_weight = 0.1
    dataset = FSQOriginalDataset(segments, metadata, cfg)
    batch = {
        key: torch.stack([dataset[i][key] for i in range(3)])
        for key in ("ctrl", "length", "length_target", "start_pose")
    }
    model = SplineFSQOriginalAE(cfg)
    output = model(batch["ctrl"], batch["length"], batch["start_pose"])
    loss, metrics = fsq_original_loss(output, batch, cfg)
    loss.backward()
    assert "coverage" in metrics and float(metrics["coverage"]) >= 0.0
    # Off by default.
    cfg.bsq_entropy_cov_weight = 0.0
    model2 = SplineFSQOriginalAE(cfg)
    _, metrics2 = fsq_original_loss(
        model2(batch["ctrl"], batch["length"], batch["start_pose"]), batch, cfg
    )
    assert "coverage" not in metrics2
    # fsq quantizer + cov without the entropy machinery is a config error.
    cfg.bsq_entropy_cov_weight = 0.1
    cfg.fsq_entropy = False
    with pytest.raises(ValueError, match="fsq_entropy=True"):
        SplineFSQOriginalAE(cfg)


def test_fsq_entropy_probe_wires_into_loss() -> None:
    segments, metadata = _segments()
    cfg = _config(segments)
    cfg.reconstruct_length = False
    dataset = FSQOriginalDataset(segments, metadata, cfg)
    batch = {
        key: torch.stack([dataset[i][key] for i in range(3)])
        for key in ("ctrl", "length", "length_target", "start_pose")
    }
    # Default off: no entropy metrics, u_cont absent.
    model = SplineFSQOriginalAE(cfg)
    output = model(batch["ctrl"], batch["length"], batch["start_pose"])
    assert output["u_cont"] is None
    _, metrics = fsq_original_loss(output, batch, cfg)
    assert "entropy_sample" not in metrics
    # Probe on: entropy terms appear and train.
    cfg.fsq_entropy = True
    cfg.bsq_entropy_joint = True
    model = SplineFSQOriginalAE(cfg)
    output = model(batch["ctrl"], batch["length"], batch["start_pose"])
    assert output["u_cont"] is not None and output["u_cont"].shape == (3, 3)
    loss, metrics = fsq_original_loss(output, batch, cfg)
    loss.backward()
    assert "entropy_sample" in metrics and "entropy_dataset" in metrics


def test_bsq_joint_entropy_guard_rejects_large_code_dim() -> None:
    segments, _ = _segments()
    cfg = _config(segments)
    cfg.quantizer = "bsq"
    cfg.bsq_code_dim = 16
    cfg.bsq_entropy_joint = True
    with pytest.raises(ValueError, match="too large"):
        SplineFSQOriginalAE(cfg)


def test_bsq_model_forward_and_entropy_loss() -> None:
    segments, metadata = _segments()
    cfg = _config(segments)
    cfg.quantizer = "bsq"
    cfg.bsq_code_dim = 4
    cfg.reconstruct_length = False
    model = SplineFSQOriginalAE(cfg)
    assert isinstance(model.fsq, BSQ)
    assert model.encoder.z_head.out_features == 4
    assert model.decoder.mlp[0].net[0].in_features == 4
    dataset = FSQOriginalDataset(segments, metadata, cfg)
    batch = {
        key: torch.stack([dataset[i][key] for i in range(3)])
        for key in ("ctrl", "length", "length_target", "start_pose")
    }
    output = model(batch["ctrl"], batch["length"], batch["start_pose"])
    assert output["u_cont"] is not None and output["u_cont"].shape == (3, 4)
    loss, metrics = fsq_original_loss(output, batch, cfg)
    loss.backward()
    assert "entropy_sample" in metrics and "entropy_dataset" in metrics
    assert metrics["entropy_sample"].item() >= 0.0
    # Confidence-only setup must remain expressible (dataset term off).
    cfg.bsq_entropy_div_weight = 0.0
    loss2, metrics2 = fsq_original_loss(
        model(batch["ctrl"], batch["length"], batch["start_pose"]), batch, cfg
    )
    assert "entropy_sample" in metrics2
    # Encoding APIs stay usable with the BSQ head.
    index = model.encode_index(segments[0])
    assert 0 <= index < 16


def test_action_seq_encoder_forward_and_pad_invariance() -> None:
    segments, metadata = _segments()
    actions = _actions_for(segments)
    cfg = _rnn_config(segments, actions)
    cfg.encoder_arch = "action_seq"
    model = SplineFSQOriginalAE(cfg)
    dataset = FSQOriginalDataset(segments, metadata, cfg, actions=actions)
    batch = {
        key: torch.stack([dataset[i][key] for i in range(3)])
        for key in ("ctrl", "length", "length_target", "start_pose", "actions_norm")
    }
    output = model(
        batch["ctrl"], batch["length"], batch["start_pose"],
        unroll_steps=int(batch["length"].max()),
        action_seq=batch["actions_norm"],
    )
    assert output["actions_hat"].shape[0] == 3
    loss, metrics = fsq_original_loss(output, batch, cfg)
    loss.backward()
    assert "action" in metrics and "termination" in metrics
    # Pad invariance: garbage in the padded region must not change z.
    lengths = batch["length"][:1]
    steps = int(lengths.max())
    acts = batch["actions_norm"][:1, :steps]
    with torch.no_grad():
        z_a = model.encoder.encode_continuous(acts, lengths)
        padded = torch.cat([acts, torch.randn(1, 7, acts.shape[-1])], dim=1)
        z_b = model.encoder.encode_continuous(padded, lengths)
    torch.testing.assert_close(z_a, z_b, atol=1e-5, rtol=1e-4)
    # Numpy encode path + guards.
    index = model.encode_actions_index(actions[0])
    assert 0 <= index < model.fsq.codebook_size
    rolled, _ = model.rollout_from_actions_numpy(actions[0], max_steps=20)
    assert rolled.shape[1] == 7 and len(rolled) <= 20
    with pytest.raises(ValueError, match="encode_actions_numpy"):
        model.encode_numpy(segments[0])
    with pytest.raises(ValueError, match="rollout_from_actions_numpy"):
        model.rollout_actions_numpy(segments[0])


def test_action_seq_encoder_composes_with_bsq() -> None:
    segments, metadata = _segments()
    actions = _actions_for(segments)
    cfg = _rnn_config(segments, actions)
    cfg.encoder_arch = "action_seq"
    cfg.quantizer = "bsq"
    cfg.bsq_code_dim = 4
    cfg.bsq_entropy_joint = True
    model = SplineFSQOriginalAE(cfg)
    # The BSQ swap must land on the action-seq encoder's head/quantizer.
    assert isinstance(model.fsq, BSQ)
    assert model.encoder.z_head.out_features == 4
    dataset = FSQOriginalDataset(segments, metadata, cfg, actions=actions)
    batch = {
        key: torch.stack([dataset[i][key] for i in range(3)])
        for key in ("ctrl", "length", "length_target", "start_pose", "actions_norm")
    }
    output = model(
        batch["ctrl"], batch["length"], batch["start_pose"],
        unroll_steps=int(batch["length"].max()),
        action_seq=batch["actions_norm"],
    )
    assert output["u_cont"] is not None and output["u_cont"].shape == (3, 4)
    loss, metrics = fsq_original_loss(output, batch, cfg)
    loss.backward()
    assert "entropy_sample" in metrics and "entropy_dataset" in metrics
    index = model.encode_actions_index(actions[0])
    assert 0 <= index < 16
    rolled, _ = model.rollout_from_actions_numpy(actions[0], max_steps=15)
    assert rolled.shape[1] == 7 and len(rolled) <= 15


def test_action_seq_requires_rnn_decoder() -> None:
    segments, metadata = _segments()
    actions = _actions_for(segments)
    cfg = _rnn_config(segments, actions)
    cfg.encoder_arch = "action_seq"
    cfg.decoder_arch = "oneshot"
    with pytest.raises(ValueError, match="requires decoder_arch='rnn'"):
        SplineFSQOriginalAE(cfg)


def test_length_free_encoder_ignores_duration() -> None:
    segments, metadata = _segments()
    cfg = _config(segments)
    cfg.encoder_length_token = False
    model = SplineFSQOriginalAE(cfg)
    # No length projection; token count = n_control + start token only.
    assert model.encoder.enc_len_proj is None
    assert model.encoder.enc_traj_pool.n_tokens == N_CONTROL + 1
    assert not any("enc_len_proj" in key for key in model.state_dict())
    dataset = FSQOriginalDataset(segments, metadata, cfg)
    ctrl = torch.stack([dataset[i]["ctrl"] for i in range(2)])
    start = torch.stack([dataset[i]["start_pose"] for i in range(2)])
    short = torch.tensor([5, 5], dtype=torch.long)
    long = torch.tensor([500, 500], dtype=torch.long)
    with torch.no_grad():
        z_short = model.encoder.encode_continuous(ctrl, short, start)
        z_long = model.encoder.encode_continuous(ctrl, long, start)
    # Duration must be invisible to z: same ctrl -> same encoding for any length.
    torch.testing.assert_close(z_short, z_long)
    # Full round trip still works (rollout/reconstruct APIs accept lengths).
    recon = model.reconstruct_numpy(segments[0], use_true_length=True)
    assert recon.shape == segments[0].shape


def test_rnn_requires_action_stats() -> None:
    segments, _ = _segments()
    cfg = _config(segments)
    cfg.decoder_arch = "rnn"
    with pytest.raises(ValueError, match="action_q01"):
        SplineFSQOriginalAE(cfg)


def test_optimal_mode_requires_start_stats() -> None:
    segments, _ = _segments()
    cfg = _config(segments, mode="optimal")
    cfg.encoder_start_min = None
    with pytest.raises(ValueError, match="encoder_start_min"):
        SplineFSQOriginalAE(cfg)


def test_dataset_exposes_codebook_diagnostic_layout() -> None:
    segments, metadata = _segments()
    cfg = _config(segments)
    dataset = FSQOriginalDataset(segments, metadata, cfg)
    model = SplineFSQOriginalAE(cfg)
    # _collect_code_assignments (reused from FSQ.py) needs these attributes.
    assert len(dataset.ctrl) == len(dataset.lengths) == len(segments)
    assert dataset.start_poses is not None and len(dataset.start_poses) == len(segments)
    assert dataset.start_poses[0].shape == (3,)
    assert model.encoder.enc_start_proj.in_features == 3
    assert dataset.ctrl[0].shape == (N_CONTROL, ENC_DIM)
