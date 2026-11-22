"""Delta action targets must agree in training, jittered pairs, and encoding."""

import sys
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[2] / "skillVLA/libero/configs/train_skills/FSQ/src"
sys.path.insert(0, str(SRC))

import FSQ as fsq  # noqa: E402
from action_representation import (  # noqa: E402
    pack_action_state_context,
    prepare_fsq_actions,
    unpack_delta_joint_context,
)


def test_joint_delta_round_trip_preserves_both_grippers_and_raw_data():
    actions = np.arange(56, dtype=np.float32).reshape(4, 14) / 10
    states = np.full((4, 14), 2, dtype=np.float32)
    original = actions.copy()
    delta = prepare_fsq_actions(actions, states, "delta_joint", [6, 13])
    restored = np.cumsum(delta, axis=0) + states[0]
    restored[:, [6, 13]] = delta[:, [6, 13]]
    np.testing.assert_allclose(restored, actions, atol=1e-6)
    np.testing.assert_array_equal(delta[:, [6, 13]], actions[:, [6, 13]])
    np.testing.assert_array_equal(actions, original)
    np.testing.assert_array_equal(prepare_fsq_actions(actions, states, "absolute_joint"), original)


def test_shifted_boundary_reanchors_first_delta_to_its_current_state():
    actions = np.tile(np.arange(6, dtype=np.float32)[:, None], (1, 14))
    states = actions + 10
    packed = pack_action_state_context(actions, states)
    shifted = unpack_delta_joint_context(packed[2:5], 14, [6, 13])
    np.testing.assert_array_equal(shifted[0, :6], -10)
    np.testing.assert_array_equal(shifted[1:, :6], 1)
    np.testing.assert_array_equal(shifted[:, [6, 13]], actions[2:5, [6, 13]])


def test_delta_training_pairs_loss_and_saved_encoder_share_one_convention(tmp_path, monkeypatch):
    cfg = fsq.SplineFSQAEConfig(
        action_space="delta_joint", action_dim=14, enc_dim=14, state_dim=14,
        encoder_arch="action_seq", reconstructor_arch="action_seq_transformer",
        autoencoder_mode="norm_action", reconstructor_only=True,
        reconstructor_start_state=False, encoder_input_mode="zero_grounded",
        reconstructor_output_mode="zero_grounded", terminator_progress=False,
        terminator_termination=False, hidden_dim=32, num_layers=1, fsq_levels=[3]*4,
        max_action_dim=14, max_state_dim=14, chunk_size=1, samples_per_skill=1,
        n_control=6, length_min=1, length_max=6, pair_loss="contrastive",
        boundary_aug_pmax=1, action_gripper_indices=[6, 13], action_gripper_weight=.1,
        encoder_min=np.full(14, -20), encoder_max=np.full(14, 20),
        reconstructor_min=np.full(14, -20), reconstructor_max=np.full(14, 20),
        state_min=np.full(14, -20), state_max=np.full(14, 20),
        state_q01=np.full(14, -20), state_q99=np.full(14, 20),
        action_q01=np.full(14, -20), action_q99=np.full(14, 20),
    )
    states = [np.full((4, 14), 10+i, dtype=np.float32) for i in range(2)]
    actions = [np.tile(np.arange(4, dtype=np.float32)[:, None]+i, (1, 14)) for i in range(2)]
    metadata = [dict(episode_id=0, task_id=0, skill_index=i, frame_start=i*4, frame_end=(i+1)*4) for i in range(2)]
    contexts = fsq.build_boundary_augmentation_contexts(
        [pack_action_state_context(a, s) for a, s in zip(actions, states)], metadata, pmax=1,
    )
    monkeypatch.setattr(fsq, "sample_boundary_augmented_segment", lambda context, **kwargs:
                        (context.trajectory[context.start+1:context.end].copy(), 0, 1))
    dataset = fsq.FSQTrajectoryDataset(states, states, actions, metadata, "unused", cfg,
        training=True, boundary_contexts=contexts,
        adjacent_skill_indices=fsq.build_adjacent_skill_indices(metadata))
    item = dataset[0]
    expected = prepare_fsq_actions(actions[0][1:], states[0][1:], "delta_joint", [6, 13])
    np.testing.assert_allclose(item["augmented_action_seq"][:3],
        fsq.normalize_action_sequence(expected, cfg.action_q01, cfg.action_q99,
            gripper_weight=.1, gripper_indices=[6, 13], clip=True))
    model = fsq.SplineFSQAE(cfg)
    items = [item, dataset[1]]
    batch = {key: torch.stack([x[key] for x in items]) for key in item}
    output = model(ctrl=batch["ctrl"], lengths=batch["length"],
        start_state=batch["start_state"].reshape(2, 14), raw_state=batch["raw_state"].reshape(2, 14),
        progress_target=batch["progress"].reshape(2), third=None, wrist=None, samples_per_skill=1,
        action_seq=batch["encoder_action_seq"], augmented_action_seq=batch["augmented_action_seq"],
        augmented_lengths=batch["augmented_length"], negative_action_seq=batch["negative_action_seq"],
        negative_lengths=batch["negative_length"])
    loss, _ = fsq.fsq_reconstruction_loss(output, batch, cfg)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.encoder.action_proj.weight.grad is not None
    path = tmp_path / "delta.pt"
    torch.save({"cfg": cfg, "model_state": model.state_dict()}, path)
    encoder, saved = fsq.load_fsq_encoder(path, "cpu")
    assert saved.action_space == "delta_joint"
    target = prepare_fsq_actions(actions[0], states[0], "delta_joint", [6, 13])
    np.testing.assert_allclose(encoder._prepare_actions_numpy(target), item["encoder_action_seq"][:4])
