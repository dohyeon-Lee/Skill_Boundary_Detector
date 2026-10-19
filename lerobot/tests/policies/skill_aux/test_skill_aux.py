from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.skill_aux.configuration_skill_aux import SkillAuxConfig
from lerobot.policies.skill_aux import modeling_skill_aux as skill_aux_module
from lerobot.policies.skill_expert import modeling_skill_predictor as predictor_module
from lerobot.policies.skill_expert import modeling_utils as modeling_utils_module
from lerobot.policies.skill_expert.processor_skill_expert import (
    skill_expert_batch_to_transition,
    skill_expert_transition_to_batch,
)
from lerobot.policies.skill_aux.modeling_state_terminator import (
    StateSkillMLPTerminator,
    StateSkillRNNTerminator,
)


class _DummyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))


class _DummyTerminator(nn.Module):
    def __init__(
        self,
        *,
        context_mode="proprio",
        camera_mode="both",
        goal_xyz=False,
        start_proprio=False,
        chunk_end_pose_mode="off",
        chunk_end_state_dim=8,
    ):
        super().__init__()
        self.progress = nn.Parameter(torch.tensor(0.0))
        self.end = nn.Parameter(torch.tensor(0.0))
        self.vision_encoder = _DummyVision()
        self.state_dim = 8 if goal_xyz or chunk_end_pose_mode != "off" else 2
        self.context_mode = context_mode
        self.camera_mode = camera_mode
        self.goal_xyz = bool(goal_xyz)
        self.start_proprio = bool(start_proprio)
        self.last_goal_xyz = None
        self.last_start_state = None
        self.chunk_end_pose_mode = chunk_end_pose_mode
        self.chunk_end_prediction = (
            nn.Parameter(torch.zeros(chunk_end_state_dim))
            if chunk_end_pose_mode != "off"
            else None
        )
        self._last_chunk_end_prediction = None
        self.freeze_vision_encoder = False

    def forward(
        self, z_q, state, image, wrist_image, *, goal_xyz=None, start_state=None
    ):
        del state, image, wrist_image
        self.last_goal_xyz = goal_xyz
        self.last_start_state = start_state
        batch_size = z_q.shape[0]
        self._last_chunk_end_prediction = (
            self.chunk_end_prediction.unsqueeze(0).expand(batch_size, -1)
            if self.chunk_end_prediction is not None and self.training
            else None
        )
        return (
            self.progress.sigmoid().expand(batch_size),
            self.end.expand(batch_size),
        )

    def normalize_previous_action(self, action):
        return action[..., : self.state_dim]

    def take_training_chunk_end_state_prediction(self):
        prediction = self._last_chunk_end_prediction
        self._last_chunk_end_prediction = None
        if prediction is None:
            raise RuntimeError("No chunk-end prediction is available.")
        return prediction


class _DummyImageOnlyTerminator(_DummyTerminator):
    def forward(self, z_q, image, wrist_image):
        del image, wrist_image
        batch_size = z_q.shape[0]
        return (
            self.progress.sigmoid().expand(batch_size),
            self.end.expand(batch_size),
        )


class _DummyWristOnlyTerminator(_DummyTerminator):
    def forward(self, z_q, wrist_image):
        del wrist_image
        batch_size = z_q.shape[0]
        return (
            self.progress.sigmoid().expand(batch_size),
            self.end.expand(batch_size),
        )


def test_trainable_fsq_wrapper_forwards_progress_detach_backbone(monkeypatch):
    captured = {}
    fake_fsq = ModuleType("FSQ")

    def build_trainable(path, **kwargs):
        captured.update(kwargs)
        return object(), SimpleNamespace()

    fake_fsq.build_trainable_fsq_terminator = build_trainable
    monkeypatch.setitem(sys.modules, "FSQ", fake_fsq)

    modeling_utils_module.build_trainable_fsq_terminator(
        "FSQ.pt",
        progress_detach_backbone=True,
        proprio_conditioning="tokens_delta_adarms",
        proprio_noise_magnitude=0.02,
        proprio_noise_exclude_last_n=2,
        chunk_end_pose_mode="learned_token",
        chunk_end_state_dim=8,
    )

    assert captured["progress_detach_backbone"] is True
    assert captured["proprio_conditioning"] == "tokens_delta_adarms"
    assert captured["proprio_noise_magnitude"] == pytest.approx(0.02)
    assert captured["proprio_noise_exclude_last_n"] == 2
    assert captured["chunk_end_pose_mode"] == "learned_token"
    assert captured["chunk_end_state_dim"] == 8


def test_chunk_end_terminator_requests_action_horizon_for_target_indexing():
    config = SkillAuxConfig(
        train_terminator=True,
        train_skill_predictor=False,
        fsq_path="dummy.pt",
        terminator_context="proprio",
        terminator_arch="fusion",
        terminator_cameras="top",
        terminator_goal_xyz=False,
        terminator_chunk_end_pose_mode="learned_token",
        terminator_chunk_end_state_horizon=10,
        terminator_chunk_end_state_q01=[0.0] * 8,
        terminator_chunk_end_state_q99=[1.0] * 8,
        max_state_dim=8,
        max_action_dim=7,
        dtype="float32",
        device="cpu",
    )

    assert config.action_delta_indices == list(range(10))


def test_terminator_start_randomization_accepts_start_anchor_only_mode():
    config = SkillAuxConfig(
        train_terminator=True,
        train_skill_predictor=False,
        fsq_path="dummy.pt",
        terminator_context="proprio",
        terminator_arch="fusion",
        terminator_cameras="top",
        terminator_start_proprio=True,
        terminator_start_randomization=True,
        terminator_start_randomization_early_frames=15,
        terminator_start_randomization_late_frames=10,
        terminator_start_randomization_distribution="uniform",
        terminator_start_randomization_probability=0.5,
        terminator_start_randomization_shift_current_observation=False,
        dtype="float32",
        device="cpu",
    )

    assert config.terminator_start_randomization is True
    assert config.terminator_start_randomization_probability == pytest.approx(0.5)
    assert config.terminator_start_randomization_shift_current_observation is False


def test_terminator_current_only_proprio_noise_does_not_require_start_proprio():
    config = SkillAuxConfig(
        train_terminator=True,
        train_skill_predictor=False,
        fsq_path="dummy.pt",
        terminator_context="proprio",
        terminator_arch="fusion",
        terminator_cameras="top",
        terminator_start_proprio=False,
        terminator_proprio_conditioning="tokens",
        terminator_proprio_noise_magnitude=0.02,
        terminator_proprio_noise_distribution="uniform",
        terminator_proprio_noise_exclude_last_n=2,
        terminator_proprio_noise_clamp=True,
        dtype="float32",
        device="cpu",
    )

    assert config.terminator_start_proprio is False
    assert config.terminator_proprio_noise_magnitude == pytest.approx(0.02)


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("inf")])
def test_terminator_start_randomization_probability_is_validated(probability):
    with pytest.raises(ValueError, match="probability"):
        SkillAuxConfig(
            train_terminator=True,
            train_skill_predictor=False,
            fsq_path="dummy.pt",
            terminator_context="proprio",
            terminator_arch="fusion",
            terminator_cameras="top",
            terminator_start_proprio=True,
            terminator_start_randomization=True,
            terminator_start_randomization_probability=probability,
            dtype="float32",
            device="cpu",
        )


class _DummyPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.reader = nn.Linear(1, 2)
        self.head = nn.Linear(2, config.skill_vocab_size)
        self.vlm = nn.Linear(1, 1)
        self.vlm.requires_grad_(False)
        self.lora_layer_count = 0
        self._train_vlm = False
        self.focus_uv_bias = (
            nn.Parameter(torch.zeros(2))
            if config.skill_predictor_focus_uv_enabled
            else None
        )
        self.end_state_bias = (
            nn.Parameter(torch.zeros(
                3 if config.skill_predictor_end_state_mode == "xyz"
                else config.skill_predictor_end_state_dim
            ))
            if config.skill_predictor_end_state_mode != "off"
            else None
        )

    def set_train_vlm_base(self, enabled):
        self._train_vlm = bool(enabled)
        self.vlm.requires_grad_(self._train_vlm)

    def vlm_parameters(self):
        return list(self.vlm.parameters()) if self._train_vlm else []

    def reader_head_parameters(self):
        parameters = [*self.reader.parameters(), *self.head.parameters()]
        if self.focus_uv_bias is not None:
            parameters.append(self.focus_uv_bias)
        if self.end_state_bias is not None:
            parameters.append(self.end_state_bias)
        return parameters

    def lora_parameters(self):
        return []

    def auxiliary_parameters(self):
        return [*self.reader_head_parameters(), *self.vlm_parameters()]

    def gradient_checkpointing_enable(self):
        return None

    def loss(self, images, language_tokens, language_mask, target):
        del images, language_tokens, language_mask
        hidden = self.reader(torch.ones(target.shape[0], 1, device=target.device))
        logits = self.head(hidden)
        loss = nn.functional.cross_entropy(logits, target)
        accuracy = (logits.argmax(dim=-1) == target).float().mean().item()
        return loss, accuracy

    def loss_with_focus_uv(
        self, images, language_tokens, language_mask, target, focus_uv, focus_valid
    ):
        skill_loss, accuracy = self.loss(
            images, language_tokens, language_mask, target
        )
        predicted = self.focus_uv_bias.unsqueeze(0).expand_as(focus_uv)
        valid = focus_valid.bool()
        uv_loss = nn.functional.smooth_l1_loss(predicted[valid], focus_uv[valid])
        total = skill_loss + 0.25 * uv_loss
        return total, {
            "skill_loss": float(skill_loss.detach()),
            "skill_accuracy": accuracy,
            "focus_uv_loss": float(uv_loss.detach()),
            "focus_uv_mae": float(
                (predicted[valid] - focus_uv[valid]).abs().mean().detach()
            ),
            "focus_uv_valid_fraction": float(valid.float().mean()),
            "total_loss": float(total.detach()),
        }

    def loss_with_end_state(
        self,
        images,
        language_tokens,
        language_mask,
        target,
        end_state,
        end_state_valid,
        *,
        predicted_skill_probability=0.0,
    ):
        self.last_predicted_skill_probability = predicted_skill_probability
        skill_loss, accuracy = self.loss(images, language_tokens, language_mask, target)
        predicted = self.end_state_bias.unsqueeze(0).expand(target.shape[0], -1)
        state_target = end_state[:, :predicted.shape[-1]]
        valid = end_state_valid.bool()
        state_loss = nn.functional.smooth_l1_loss(predicted[valid], state_target[valid])
        total = skill_loss + state_loss
        mae = (predicted[valid] - state_target[valid]).abs().mean()
        xyz_loss = nn.functional.smooth_l1_loss(
            predicted[valid, :3], state_target[valid, :3]
        )
        xyz_mae = (predicted[valid, :3] - state_target[valid, :3]).abs().mean()
        metrics = {
            "skill_loss": float(skill_loss.detach()),
            "skill_accuracy": accuracy,
            "end_state_loss": float(state_loss.detach()),
            "end_state_mae": float(mae.detach()),
            "end_xyz_loss": float(xyz_loss.detach()),
            "end_xyz_mae": float(xyz_mae.detach()),
            "end_state_valid_fraction": float(valid.float().mean()),
            "end_state_predicted_skill_probability": predicted_skill_probability,
            "end_state_predicted_skill_fraction": predicted_skill_probability,
            "end_state_condition_skill_accuracy": 1.0,
            "total_loss": float(total.detach()),
        }
        if predicted.shape[-1] > 3:
            metrics["end_rest_loss"] = float(nn.functional.smooth_l1_loss(
                predicted[valid, 3:], state_target[valid, 3:]
            ).detach())
            metrics["end_rest_mae"] = float(
                (predicted[valid, 3:] - state_target[valid, 3:]).abs().mean().detach()
            )
        return total, metrics


def _config(
    *,
    terminator: bool,
    predictor: bool,
    image_terminator: bool = False,
    wrist_terminator: bool = False,
    state_terminator: bool = False,
    state_rnn_terminator: bool = False,
    state_termination_only: bool = True,
    state_rnn_termination_only: bool = True,
    state_balance_positive_negative: bool = True,
    state_rnn_balance_positive_negative: bool = True,
    state_rnn_full_skill_sequence: bool = True,
    terminator_proprio_history: bool = False,
) -> SkillAuxConfig:
    return SkillAuxConfig(
        train_terminator=terminator,
        train_image_only_terminator=image_terminator,
        train_wrist_only_terminator=wrist_terminator,
        train_state_only_terminator=state_terminator,
        train_state_rnn_terminator=state_rnn_terminator,
        terminator_context="proprio" if terminator_proprio_history else "prev_action",
        terminator_proprio_history=terminator_proprio_history,
        terminator_history_length=4,
        state_only_terminator_termination_only=state_termination_only,
        state_rnn_terminator_termination_only=state_rnn_termination_only,
        state_only_terminator_balance_positive_negative=(
            state_balance_positive_negative
        ),
        state_rnn_terminator_balance_positive_negative=(
            state_rnn_balance_positive_negative
        ),
        state_rnn_terminator_full_skill_sequence=state_rnn_full_skill_sequence,
        state_rnn_terminator_sequence_length=4,
        train_skill_predictor=predictor,
        fsq_path="dummy.pt",
        skill_predictor_lora=False,
        skill_predictor_detach_vlm=True,
        dtype="float32",
        device="cpu",
        max_state_dim=2,
    )


def _batch() -> dict:
    batch_size = 2
    image = torch.zeros(batch_size, 3, 4, 4)
    return {
        "skill_code_true": torch.tensor([0, 1]),
        "skill_sequence": torch.tensor([[0, 1], [0, 1]]),
        "skill_index": torch.tensor([0, 1]),
        "skill_code": torch.tensor([0, 1]),
        # Endpoint-anchored samples: the recurrent state trainer expands each
        # full valid prefix and supervises every timestep.
        "skill_ds": torch.tensor([2, 3]),
        "skill_de": torch.tensor([0, 0]),
        "skill_decoder_state": torch.zeros(batch_size, 2),
        "skill_previous_action": torch.zeros(batch_size, 2),
        "skill_previous_action_bos": torch.zeros(batch_size, dtype=torch.bool),
        "observation.state": torch.zeros(batch_size, 2),
        "observation.images.image": image,
        "observation.images.wrist_image": image,
        "skill_start_image": image,
        "skill_start_wrist_image": image,
        "skill_focus_uv": torch.tensor([[0.25, -0.5], [-0.25, 0.5]]),
        "skill_focus_valid": torch.tensor([True, True]),
        "skill_end_state": torch.tensor([[0.1] * 8, [0.2] * 8]),
        "skill_end_state_valid": torch.tensor([True, True]),
        "observation.language.tokens": torch.ones(batch_size, 3, dtype=torch.long),
        "observation.language.attention_mask": torch.ones(
            batch_size, 3, dtype=torch.bool
        ),
    }


def test_predictor_spatial_targets_survive_preprocessor_round_trip() -> None:
    batch = _batch()
    batch["skill_end_xyz"] = batch["skill_end_state"][:, :3]
    batch["skill_end_xyz_valid"] = batch["skill_end_state_valid"]
    restored = skill_expert_transition_to_batch(skill_expert_batch_to_transition(batch))
    for key in ("skill_end_state", "skill_end_state_valid", "skill_end_xyz", "skill_end_xyz_valid"):
        torch.testing.assert_close(restored[key], batch[key])


@pytest.mark.parametrize("has_post_end", [True, False])
@pytest.mark.parametrize("with_start", [True, False])
@pytest.mark.parametrize("with_goal", [True, False])
@pytest.mark.parametrize("with_mask", [False, True])
def test_newterm_condition_losses_share_vision_and_original_batch_denominator(monkeypatch, has_post_end, with_start, with_goal, with_mask):
    from lerobot.policies.skillVLA.newterm_supervision import (
        POST_END_CODES, POST_END_VALID, POST_END_START_STATE, POST_END_GOAL_XYZ, POST_END_GOAL_VALID,
        HISTORY_VALID, POST_END_HISTORY_VALID,
    )
    if with_mask and with_start:
        pytest.skip("Start proprio and history are mutually exclusive.")

    class Term(_DummyTerminator):
        def __init__(self):
            super().__init__(context_mode="proprio", camera_mode="top", goal_xyz=with_goal)
            self.vision_calls = 0
            self.calls = []
            self.proprio_history = not with_start

        def _prepare_image_tokens(self, third, wrist):
            self.vision_calls += 1
            return third.mean(dim=(2, 3)).unsqueeze(1)

        def _forward_from_image_tokens(self, z, context, images, *, start_state=None, goal_xyz=None, history_mask=None):
            logits = self.end + z.sum(dim=-1) * 0.25
            self.calls.append((z.detach(), context.detach(), images.detach(), logits, start_state, goal_xyz, history_mask))
            return torch.zeros_like(logits), logits

    monkeypatch.setattr(skill_aux_module, "build_trainable_fsq_terminator", lambda *a, **kw: Term())
    policy = skill_aux_module.SkillAuxPolicy(SkillAuxConfig(
        fsq_path="dummy.pt", train_terminator=True, train_skill_predictor=False,
        terminator_context="proprio", terminator_cameras="top",
        terminator_proprio_history=not with_start, terminator_start_proprio=with_start,
        terminator_history_mask_padding=with_mask, terminator_history_reset_on_skill=with_mask,
        terminator_goal_xyz=with_goal, terminator_goal_noise_max_m=0.01 if with_goal else 0.0,
        terminator_termination_only=True,
        terminator_post_end_steps=10, terminator_end_target_sigma=1.0,
        terminator_end_pos_weight=2.0, dtype="float32", device="cpu", max_state_dim=2,
    ))
    batch = _batch()
    dim = 8 if with_goal else 2
    batch["skill_decoder_state"] = torch.arange(40 * dim).float().reshape(2, 20, dim)
    batch["observation.state"] = batch["skill_decoder_state"].clone()
    batch["skill_ds"] = torch.tensor([1, 5])
    batch["skill_de"] = torch.tensor([5, 2])
    batch[POST_END_CODES] = torch.zeros(2, 10, dtype=torch.long)
    batch[POST_END_CODES][0, 0] = 1
    batch[POST_END_VALID] = torch.zeros(2, 10, dtype=torch.bool)
    batch[POST_END_VALID][0, 0] = has_post_end
    if with_mask:
        batch[HISTORY_VALID] = torch.zeros(2, 20, dtype=torch.bool)
        batch[HISTORY_VALID][:, -2:] = True
        batch[POST_END_HISTORY_VALID] = torch.zeros(2, 10, 20, dtype=torch.bool)
        batch[POST_END_HISTORY_VALID][0, 0, -5:] = True
    if with_start:
        batch["skill_start_state"] = torch.full((2, dim), 0.1)
        batch[POST_END_START_STATE] = torch.zeros(2, 10, dim)
        batch[POST_END_START_STATE][0, 0] = torch.full((dim,), 0.9)
    if with_goal:
        batch[POST_END_GOAL_XYZ] = torch.zeros(2, 10, 3)
        batch[POST_END_GOAL_XYZ][0, 0] = torch.tensor([0.4, 0.5, 0.6])
        batch[POST_END_GOAL_VALID] = batch[POST_END_VALID].clone()
    restored = skill_expert_transition_to_batch(skill_expert_batch_to_transition(batch))
    torch.testing.assert_close(restored[POST_END_VALID], batch[POST_END_VALID])
    # The real preprocessor snapshots raw observation.state before normalization.
    restored["skill_decoder_state"] = restored["observation.state"].clone()
    loss, metrics = policy._terminator_objective(restored)
    term = policy.model.fsq_term_train
    assert term.vision_calls == 1
    if with_mask:
        torch.testing.assert_close(term.calls[0][6], batch[HISTORY_VALID])
    if with_start:
        torch.testing.assert_close(term.calls[0][4], batch["skill_start_state"])
    if with_goal:
        assert ((term.calls[0][5] - batch["skill_end_state"][:, :3]).abs() <= 0.010001).all()
    target = torch.exp(-batch["skill_de"].float().square() / 2)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        term.calls[0][3], target, pos_weight=torch.tensor(2.0)
    )
    if has_post_end:
        if with_mask:
            torch.testing.assert_close(term.calls[1][6], batch[POST_END_HISTORY_VALID][:1, 0])
            assert not torch.equal(term.calls[1][6], term.calls[0][6][:1])
        torch.testing.assert_close(term.calls[1][1], term.calls[0][1][:1])
        torch.testing.assert_close(term.calls[1][2], term.calls[0][2][:1])
        if with_start:
            torch.testing.assert_close(term.calls[1][4], batch[POST_END_START_STATE][:1, 0])
            assert not torch.equal(term.calls[1][4], term.calls[0][4][:1])
        if with_goal:
            expected_goal = batch[POST_END_GOAL_XYZ][:1, 0]
            assert ((term.calls[1][5] - expected_goal).abs() <= 0.010001).all()
            assert not torch.equal(term.calls[1][5], term.calls[0][5][:1])
        extra = torch.nn.functional.binary_cross_entropy_with_logits(
            term.calls[1][3], torch.ones(1), pos_weight=torch.tensor(2.0), reduction="sum"
        ) / 2
        expected = expected + extra
        assert metrics["terminator/post_end_loss"] == pytest.approx(extra.item())
    else:
        assert len(term.calls) == 1
        assert metrics["terminator/post_end_loss"] == 0
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(term.end.grad)


@pytest.fixture(autouse=True)
def _mock_auxiliary_builders(monkeypatch):
    monkeypatch.setattr(
        skill_aux_module,
        "build_trainable_fsq_terminator",
        lambda path, **kwargs: _DummyTerminator(
            context_mode=kwargs.get("context") or "proprio",
            camera_mode=kwargs.get("cameras") or "both",
            goal_xyz=kwargs.get("goal_xyz", False),
            start_proprio=kwargs.get("start_proprio", False),
            chunk_end_pose_mode=kwargs.get("chunk_end_pose_mode", "off"),
            chunk_end_state_dim=kwargs.get("chunk_end_state_dim", 8),
        ),
    )
    monkeypatch.setattr(
        skill_aux_module,
        "build_fsq_image_only_terminator",
        lambda path, **kwargs: _DummyImageOnlyTerminator(),
    )
    monkeypatch.setattr(
        skill_aux_module,
        "build_fsq_wrist_only_terminator",
        lambda path, **kwargs: _DummyWristOnlyTerminator(),
    )
    monkeypatch.setattr(skill_aux_module, "FrozenVLMSkillPredictor", _DummyPredictor)


@pytest.mark.parametrize(
    (
        "terminator",
        "image_terminator",
        "wrist_terminator",
        "predictor",
        "expected_groups",
    ),
    [
        (True, False, False, False, {"terminator"}),
        (False, True, False, False, {"image_terminator"}),
        (False, False, True, False, {"wrist_terminator"}),
        (False, False, False, True, {"skill_predictor_reader_head"}),
    ],
)
def test_independent_training_switches(
    terminator,
    image_terminator,
    wrist_terminator,
    predictor,
    expected_groups,
):
    policy = skill_aux_module.SkillAuxPolicy(
        _config(
            terminator=terminator,
            image_terminator=image_terminator,
            wrist_terminator=wrist_terminator,
            predictor=predictor,
        )
    )
    loss, metrics = policy(_batch())
    assert loss.requires_grad
    assert {group["group_name"] for group in policy.get_optim_params()} == expected_groups
    assert any(key.startswith("terminator/") for key in metrics) is terminator
    assert (
        any(key.startswith("image_terminator/") for key in metrics)
        is image_terminator
    )
    assert (
        any(key.startswith("wrist_terminator/") for key in metrics)
        is wrist_terminator
    )
    assert any(key.startswith("skill_predictor/") for key in metrics) is predictor


def test_chunk_end_terminator_loss_ignores_invalid_targets():
    config = SkillAuxConfig(
        train_terminator=True,
        train_skill_predictor=False,
        fsq_path="dummy.pt",
        terminator_context="proprio",
        terminator_arch="fusion",
        terminator_cameras="top",
        terminator_goal_xyz=False,
        terminator_chunk_end_pose_mode="learned_token",
        terminator_chunk_end_state_loss_weight=0.3,
        terminator_chunk_end_state_horizon=10,
        terminator_chunk_end_state_q01=[0.0] * 8,
        terminator_chunk_end_state_q99=[1.0] * 8,
        max_state_dim=8,
        max_action_dim=7,
        skill_predictor_lora=False,
        skill_predictor_detach_vlm=True,
        dtype="float32",
        device="cpu",
    )
    policy = skill_aux_module.SkillAuxPolicy(config)
    batch = _batch()
    batch["skill_decoder_state"] = torch.zeros(2, 8)
    batch["observation.state"] = torch.zeros(2, 8)
    batch["skill_chunk_end_state"] = torch.tensor(
        [[1.0] * 8, [1000.0] * 8]
    )
    batch["skill_chunk_end_state_valid"] = torch.tensor([True, False])

    loss, metrics = policy(batch)

    assert loss.requires_grad
    assert metrics["terminator/chunk_end_state_loss"] == pytest.approx(1.0)
    assert metrics["terminator/chunk_end_state_weighted"] == pytest.approx(0.3)
    assert metrics["terminator/chunk_end_state_valid_fraction"] == pytest.approx(0.5)


def test_goal_xyz_noise_is_bounded_and_disabled_at_eval() -> None:
    config = _config(terminator=True, predictor=False)
    config.terminator_context = "proprio"
    config.terminator_goal_xyz = True
    config.terminator_goal_noise_max_m = 0.01
    policy = skill_aux_module.SkillAuxPolicy(config)
    batch = _batch()
    batch["skill_decoder_state"] = torch.zeros(2, 8)
    clean = batch["skill_end_state"][:, :3]

    policy.train()
    policy._terminator_objective(batch)
    noisy = policy.model.fsq_term_train.last_goal_xyz
    assert noisy is not None
    assert torch.all((noisy - clean).abs() <= 0.01 + 1e-7)

    policy.eval()
    policy._terminator_objective(batch)
    torch.testing.assert_close(policy.model.fsq_term_train.last_goal_xyz, clean)


def test_start_proprio_is_forwarded_to_terminator() -> None:
    config = _config(terminator=True, predictor=False)
    config.terminator_context = "proprio"
    config.terminator_start_proprio = True
    policy = skill_aux_module.SkillAuxPolicy(config)
    batch = _batch()
    batch["skill_decoder_state"] = torch.zeros(2, 8)
    batch["observation.state"] = torch.zeros(2, 8)
    batch["skill_start_state"] = torch.tensor(
        [[0.1] * 8, [0.2] * 8], dtype=torch.float32
    )

    policy._terminator_objective(batch)

    torch.testing.assert_close(
        policy.model.fsq_term_train.last_start_state,
        batch["skill_start_state"][
            ..., : policy.model.fsq_term_train.state_dim
        ],
    )


def test_legacy_joint_config_remains_loadable_but_disables_transition_sampling():
    config = _config(terminator=True, predictor=True)

    assert config.train_skill_predictor is True
    assert config.train_terminator is True
    assert config.predictor_transition_sampling is False


def test_full_vlm_predictor_gets_a_separate_optimizer_group():
    config = _config(terminator=False, predictor=True)
    config.skill_predictor_freeze_vlm = False
    config.skill_predictor_detach_vlm = False
    policy = skill_aux_module.SkillAuxPolicy(config)

    assert {group["group_name"] for group in policy.get_optim_params()} == {
        "skill_predictor_reader_head",
        "skill_predictor_vlm",
    }
    assert all(
        parameter.requires_grad
        for parameter in policy.model.skill_predictor.vlm.parameters()
    )


def test_focus_uv_objective_is_reported_separately():
    config = _config(terminator=False, predictor=True)
    config.skill_predictor_focus_uv_enabled = True
    policy = skill_aux_module.SkillAuxPolicy(config)

    loss, metrics = policy(_batch())

    assert loss.requires_grad
    assert "skill_predictor/focus_uv_loss" in metrics
    assert "skill_predictor/focus_uv_mae" in metrics
    assert metrics["skill_predictor/focus_uv_valid_fraction"] == 1.0


@pytest.mark.parametrize("mode", ["xyz", "full_state"])
def test_end_state_objective_is_reported_separately(mode):
    config = _config(terminator=False, predictor=True)
    config.skill_predictor_end_state_mode = mode
    config.skill_predictor_end_state_dim = 8
    policy = skill_aux_module.SkillAuxPolicy(config)

    loss, metrics = policy(_batch())

    assert loss.requires_grad
    assert "skill_predictor/end_state_loss" in metrics
    assert "skill_predictor/end_xyz_loss" in metrics
    assert "skill_predictor/end_xyz_mae" in metrics
    assert metrics["skill_predictor/end_state_valid_fraction"] == 1.0
    assert policy.model.skill_predictor.end_state_bias.shape[0] == (3 if mode == "xyz" else 8)
    assert ("skill_predictor/end_rest_loss" in metrics) is (mode == "full_state")
    assert ("skill_predictor/end_rest_mae" in metrics) is (mode == "full_state")


def test_end_state_skill_condition_schedule_reaches_expected_probabilities():
    config = _config(terminator=False, predictor=True)
    config.skill_predictor_end_state_mode = "xyz"
    config.skill_predictor_end_state_skill_source = "scheduled"
    config.skill_predictor_end_state_schedule_start_step = 10
    config.skill_predictor_end_state_schedule_end_step = 30
    policy = skill_aux_module.SkillAuxPolicy(config)

    for step, expected in ((0, 0.0), (10, 0.0), (20, 0.5), (30, 1.0), (50, 1.0)):
        policy.set_training_step(step)
        _, metrics = policy(_batch())
        assert metrics[
            "skill_predictor/end_state_predicted_skill_probability"
        ] == pytest.approx(expected)
        assert policy.model.skill_predictor.last_predicted_skill_probability == pytest.approx(
            expected
        )


@pytest.mark.parametrize("mode,output_dim", [("xyz", 3), ("full_state", 8)])
@pytest.mark.parametrize("predictor_mode", ["mode1", "mode2", "mode1_preskill", "mode2_preskill"])
def test_real_predictor_end_state_head_shape_and_gradient(
    monkeypatch, tmp_path, mode, output_dim, predictor_mode,
):
    from safetensors.torch import save_file
    from lerobot.policies.skill_expert.modeling_skill_expert import (
        _load_complete_predictor_parameters,
    )

    class _TinyVLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Linear(1, 1)
            self.language_model.config = SimpleNamespace(hidden_size=16)

    monkeypatch.setattr(predictor_module, "build_paligemma_model", lambda *args, **kwargs: _TinyVLM())
    config = _config(terminator=False, predictor=True)
    config.skill_predictor_end_state_mode = mode
    config.skill_predictor_mode = predictor_mode
    history_kwargs = (
        {"previous_skill_code": torch.tensor([-1, 0])}
        if predictor_mode.endswith("_preskill") else {}
    )
    config.skill_predictor_end_state_dim = 8
    predictor = predictor_module.FrozenVLMSkillPredictor(config)
    for parameter in predictor.reader_head_parameters():
        parameter.requires_grad_(True)
    memory = torch.randn(2, 5, 16)
    key_ignore = torch.zeros(2, 5, dtype=torch.bool)

    prediction = predictor.predict_end_state_from_memory(memory, key_ignore, torch.tensor([0, 1]))
    assert (predictor.end_state_skill_projection is None) == predictor_mode.startswith("mode2")
    if predictor_mode.startswith("mode2"):
        torch.testing.assert_close(
            prediction,
            predictor.predict_end_state_from_memory(memory, key_ignore, torch.tensor([8, 9])),
        )
        torch.testing.assert_close(
            prediction, predictor.predict_end_state_from_memory(memory, key_ignore, None)
        )

    save_file(
        {f"model.skill_predictor.{key}": value.contiguous() for key, value in predictor.state_dict().items()},
        tmp_path / "model.safetensors",
    )
    restored = predictor_module.FrozenVLMSkillPredictor(config)
    assert _load_complete_predictor_parameters(restored, tmp_path) == len(predictor.state_dict())
    torch.testing.assert_close(
        prediction, restored.predict_end_state_from_memory(memory, key_ignore, torch.tensor([0, 1]))
    )
    if predictor_mode.endswith("_preskill"):
        from dataclasses import replace
        from lerobot.policies.skill_expert.modeling_skill_expert import (
            SkillExpertPolicy, _load_learned_predictor_parameters,
        )

        config.save_pretrained(tmp_path)
        # A predictor-free VSA must adopt the checkpoint's preskill structure,
        # including BOS, instead of imposing its unused mode1 defaults.
        target = object.__new__(SkillExpertPolicy)
        nn.Module.__init__(target)
        target.config = replace(config, skill_predictor_mode="mode1")
        target.model = nn.Module()
        target.model.anchor = nn.Linear(1, 1)
        target.model.skill_predictor = None
        target.load_external_skill_predictor(tmp_path)
        attached = target.model.skill_predictor
        assert attached.config.skill_predictor_mode == predictor_mode
        torch.testing.assert_close(attached.previous_skill_bos, predictor.previous_skill_bos)
        assert not attached.previous_skill_bos.requires_grad
        assert _load_learned_predictor_parameters(attached, tmp_path) > 0

    assert prediction.shape == (2, output_dim)
    prediction.sum().backward()
    assert predictor.end_state_head[-1].weight.grad is not None
    monkeypatch.setattr(predictor, "reader_memory", lambda *args: (memory, key_ignore))
    original_end_state_prediction = predictor.predict_end_state_from_memory
    conditioning = {}

    def _capture_conditioning(memory, key_ignore, skill_code):
        conditioning["skill_code"] = None if skill_code is None else skill_code.detach().clone()
        return original_end_state_prediction(memory, key_ignore, skill_code)

    monkeypatch.setattr(
        predictor,
        "predict_end_state_from_memory",
        _capture_conditioning,
    )
    monkeypatch.setattr(
        predictor.head,
        "decode",
        lambda hidden: torch.tensor([2, 3], device=hidden.device),
    )
    loss, metrics = predictor.loss_with_end_state(
        [], torch.zeros(2, 1, dtype=torch.long), torch.ones(2, 1, dtype=torch.bool),
        torch.tensor([0, 1]), torch.randn(2, 8), torch.tensor([True, False]),
        predicted_skill_probability=1.0,
        **history_kwargs,
    )
    assert loss.requires_grad
    assert metrics["end_state_valid_fraction"] == 0.5
    assert metrics["end_state_loss"] >= 0.0
    assert metrics["end_xyz_loss"] >= 0.0
    if predictor_mode.startswith("mode1"):
        torch.testing.assert_close(conditioning["skill_code"], torch.tensor([2, 3]))
    else:
        assert conditioning["skill_code"] is None
    assert metrics["end_state_predicted_skill_fraction"] == float(predictor_mode.startswith("mode1"))
    assert metrics["end_state_skill_conditioned"] == float(predictor_mode.startswith("mode1"))
    assert metrics["end_state_condition_skill_accuracy"] == 0.0
    assert ("end_rest_loss" in metrics) is (mode == "full_state")
    if mode == "full_state":
        assert metrics["end_state_loss"] == pytest.approx(
            (3 * metrics["end_xyz_loss"] + 5 * metrics["end_rest_loss"]) / 8
        )


@pytest.mark.parametrize("mode", ["mode1_preskill", "mode2_preskill"])
def test_preskill_bos_and_projection_train_and_runtime_requires_explicit_history(monkeypatch, mode):
    class TinyVLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Linear(1, 1)
            self.language_model.config = SimpleNamespace(hidden_size=16)

    monkeypatch.setattr(predictor_module, "build_paligemma_model", lambda *a, **kw: TinyVLM())
    config = _config(terminator=False, predictor=True)
    config.skill_predictor_mode = mode
    config.skill_predictor_end_state_mode = "xyz"
    predictor = predictor_module.FrozenVLMSkillPredictor(config)
    for parameter in predictor.reader_head_parameters():
        parameter.requires_grad_(True)
    assert torch.count_nonzero(predictor.previous_skill_bos) == 0
    assert any(p is predictor.previous_skill_bos for p in predictor.reader_head_parameters())
    memory = torch.randn(2, 5, 16)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    monkeypatch.setattr(predictor, "reader_memory", lambda *a: (memory, mask))
    with pytest.raises(ValueError, match="requires skill_previous_code"):
        predictor.skill_reader_hidden(memory, mask)
    for bad in (torch.tensor([-2, 0]), torch.tensor([27, 0]), torch.tensor([0.0, 1.0])):
        with pytest.raises(ValueError, match="skill_previous_code"):
            predictor.skill_reader_hidden(memory, mask, previous_skill_code=bad)

    conditions = []
    hook = predictor.reader.register_forward_pre_hook(
        lambda module, args, kwargs: conditions.append(kwargs["probe_condition"].detach().clone()),
        with_kwargs=True,
    )
    previous = torch.tensor([-1, 0])  # code zero is a real skill, NOT BOS
    predictor.skill_reader_hidden(memory, mask, previous_skill_code=previous)
    expected = predictor.previous_skill_projection(predictor._skill_code_coordinates(torch.tensor([0]), memory.dtype))
    torch.testing.assert_close(conditions[-1][0], predictor.previous_skill_bos)
    torch.testing.assert_close(conditions[-1][1], expected[0])
    hook.remove()

    policy = object.__new__(skill_aux_module.SkillAuxPolicy)
    nn.Module.__init__(policy)
    policy.config = config
    policy._training_step = 0
    policy.model = nn.Module()
    policy.model.skill_predictor = predictor
    batch = _batch()
    batch["skill_previous_code"] = previous
    loss, metrics = policy._skill_predictor_objective(batch)
    loss.backward()
    assert predictor.previous_skill_bos.grad is not None
    assert predictor.previous_skill_bos.grad.abs().sum() > 0
    assert predictor.previous_skill_projection[0].weight.grad.abs().sum() > 0
    assert metrics["skill_predictor/previous_skill_enabled"] == 1.0
    assert metrics["skill_predictor/previous_skill_bos_fraction"] == 0.5

    hidden_a = predictor.skill_reader_hidden(memory, mask, previous_skill_code=previous)
    hidden_b = predictor.skill_reader_hidden(memory, mask, previous_skill_code=torch.tensor([1, 2]))
    assert not torch.allclose(hidden_a, hidden_b)
    # Joint proprio and previous skill are additive, separate adapters.
    config.skill_predictor_start_proprio = True
    config.skill_predictor_start_proprio_dim = 8
    with_proprio = predictor_module.FrozenVLMSkillPredictor(config)
    with_proprio.skill_reader_hidden(memory, mask, torch.zeros(2, 8), previous)


def test_joint_start_proprio_is_zero_init_and_xyz_skill_condition_can_detach(monkeypatch):
    class _TinyVLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Linear(1, 1)
            self.language_model.config = SimpleNamespace(hidden_size=16)

    monkeypatch.setattr(
        predictor_module, "build_paligemma_model", lambda *args, **kwargs: _TinyVLM()
    )
    config = _config(terminator=False, predictor=True)
    config.skill_predictor_end_state_mode = "xyz"
    config.skill_predictor_start_proprio = True
    config.skill_predictor_start_proprio_dim = 8
    predictor = predictor_module.FrozenVLMSkillPredictor(config)
    predictor.reader_head_parameters()
    predictor.reader.requires_grad_(True)
    predictor.head.requires_grad_(True)
    predictor.end_state_reader.requires_grad_(True)
    predictor.end_state_skill_projection.requires_grad_(True)
    predictor.end_state_head.requires_grad_(True)
    predictor.start_proprio_skill_projection.requires_grad_(True)
    predictor.start_proprio_end_state_projection.requires_grad_(True)

    memory = torch.randn(2, 5, 16)
    key_ignore = torch.zeros(2, 5, dtype=torch.bool)
    start = torch.randn(2, 8)
    zero_condition = predictor.start_proprio_condition(
        start, branch="skill", dtype=memory.dtype
    )
    torch.testing.assert_close(zero_condition, torch.zeros_like(zero_condition))
    torch.testing.assert_close(
        predictor.reader(memory, key_ignore, probe_condition=zero_condition),
        predictor.reader(memory, key_ignore),
    )

    coordinates = torch.zeros(2, 3, requires_grad=True)
    _, _, ste = predictor.head.quantize_coordinates(coordinates)
    xyz = predictor.predict_end_state_from_memory(
        memory, key_ignore, ste.detach(), start
    )
    xyz.sum().backward()
    assert coordinates.grad is None
    assert predictor.end_state_head[-1].weight.grad is not None


def test_policy_accepts_generic_factory_dataset_metadata():
    policy = skill_aux_module.SkillAuxPolicy(
        _config(terminator=True, predictor=False),
        dataset_stats={"action": {}},
        dataset_meta=object(),
    )

    assert policy.model.fsq_term_train is not None


def test_component_specific_ft_loads_complete_weights_from_its_file(
    tmp_path,
):
    predictor_source = skill_aux_module.SkillAuxPolicy(
        _config(terminator=False, predictor=True)
    )
    with torch.no_grad():
        predictor_source.model.skill_predictor.reader.weight.fill_(7.0)
    predictor_dir = tmp_path / "predictor_pt"
    predictor_source.save_pretrained(predictor_dir)

    terminator_source = skill_aux_module.SkillAuxPolicy(
        _config(terminator=True, predictor=False)
    )
    with torch.no_grad():
        terminator_source.model.fsq_term_train.end.fill_(4.0)
    terminator_dir = tmp_path / "terminator_pt"
    terminator_source.save_pretrained(terminator_dir)

    predictor_config = _config(terminator=False, predictor=True)
    predictor_config.skill_predictor_checkpoint_path = str(predictor_dir)
    predictor_target = skill_aux_module.SkillAuxPolicy(predictor_config)

    terminator_config = _config(terminator=True, predictor=False)
    terminator_config.terminator_checkpoint_path = str(terminator_dir)
    terminator_target = skill_aux_module.SkillAuxPolicy(terminator_config)

    assert torch.all(
        predictor_target.model.skill_predictor.reader.weight
        == torch.full_like(
            predictor_target.model.skill_predictor.reader.weight, 7.0
        )
    )
    assert terminator_target.model.fsq_term_train.end.item() == 4.0


def test_cli_off_end_state_mode_survives_yaml_parsing():
    import draccus

    # draccus reads CLI values as YAML, where a bare `off` is the boolean False.
    config = draccus.parse(config_class=SkillAuxConfig, args=["--skill_predictor_end_state_mode=off"])
    assert config.skill_predictor_end_state_mode == "off"


def test_both_disabled_is_rejected():
    with pytest.raises(ValueError, match="terminator.train"):
        _config(terminator=False, predictor=False)


def test_auxiliary_grad_groups_are_parameter_disjoint():
    policy = skill_aux_module.SkillAuxPolicy(
        _config(
            terminator=True,
            image_terminator=True,
            wrist_terminator=True,
            predictor=False,
        )
    )
    groups = policy.isolated_main_optimizer_grad_groups()
    assert set(groups) == {
        "terminator",
        "image_terminator",
        "wrist_terminator",
    }
    assert {id(parameter) for parameter in groups["image_terminator"]}.isdisjoint(
        {id(parameter) for parameter in groups["terminator"]}
    )
    assert {id(parameter) for parameter in groups["wrist_terminator"]}.isdisjoint(
        {id(parameter) for parameter in groups["image_terminator"]}
    )


def test_terminator_derives_current_skill_without_predictor_dataset_fields():
    policy = skill_aux_module.SkillAuxPolicy(_config(terminator=True, predictor=False))
    batch = _batch()
    del batch["skill_code_true"]
    loss, metrics = policy(batch)
    assert loss.requires_grad
    assert "terminator/loss" in metrics


def test_context_free_top_terminator_needs_neither_state_nor_wrist():
    config = _config(terminator=True, predictor=False)
    config.terminator_context = "none"
    config.terminator_cameras = "top"
    policy = skill_aux_module.SkillAuxPolicy(config)
    batch = _batch()
    del batch["skill_decoder_state"]
    del batch["observation.images.wrist_image"]

    loss, metrics = policy(batch)

    assert loss.requires_grad
    assert "terminator/loss" in metrics


def test_image_only_terminator_ignores_state_and_randomized_predictor_fields():
    policy = skill_aux_module.SkillAuxPolicy(
        _config(terminator=False, image_terminator=True, predictor=False)
    )
    batch = _batch()
    for key in (
        "skill_decoder_state",
        "skill_code_true",
        "skill_code",
        "skill_start_image",
        "skill_start_wrist_image",
    ):
        del batch[key]
    loss, metrics = policy(batch)
    assert loss.requires_grad
    assert "image_terminator/loss" in metrics


def test_wrist_only_terminator_uses_no_state_top_or_predictor_inputs():
    policy = skill_aux_module.SkillAuxPolicy(
        _config(
            terminator=False,
            image_terminator=False,
            wrist_terminator=True,
            predictor=False,
        )
    )
    batch = _batch()
    for key in (
        "skill_decoder_state",
        "observation.images.image",
        "skill_code_true",
        "skill_code",
        "skill_start_image",
        "skill_start_wrist_image",
    ):
        del batch[key]

    loss, metrics = policy(batch)

    assert loss.requires_grad
    assert "wrist_terminator/loss" in metrics
    assert any(
        key.startswith("model.fsq_wrist_term_train.")
        for key in policy.state_dict()
    )


@pytest.mark.parametrize(
    ("state_terminator", "state_rnn_terminator", "expected_group", "metric"),
    [
        (True, False, "state_terminator", "state_terminator/loss"),
        (False, True, "state_rnn_terminator", "state_rnn_terminator/loss"),
    ],
)
def test_state_terminators_train_without_any_image_inputs(
    state_terminator,
    state_rnn_terminator,
    expected_group,
    metric,
):
    policy = skill_aux_module.SkillAuxPolicy(
        _config(
            terminator=False,
            predictor=False,
            state_terminator=state_terminator,
            state_rnn_terminator=state_rnn_terminator,
        )
    )
    batch = _batch()
    batch["observation.state"] = torch.randn(2, 4, 2)
    for key in (
        "skill_decoder_state",
        "observation.images.image",
        "observation.images.wrist_image",
        "skill_start_image",
        "skill_start_wrist_image",
    ):
        del batch[key]

    loss, metrics = policy(batch)

    assert loss.requires_grad
    assert metric in metrics
    assert {group["group_name"] for group in policy.get_optim_params()} == {
        expected_group
    }
    assert any(
        key.startswith(f"model.fsq_{expected_group.replace('terminator', 'term')}_train.")
        for key in policy.state_dict()
    )


def test_state_rnn_sequence_matches_online_hidden_rollout() -> None:
    torch.manual_seed(7)
    model = StateSkillRNNTerminator(
        state_dim=2,
        skill_dim=3,
        input_dim=8,
        hidden_dim=8,
        num_layers=1,
    ).eval()
    states = torch.randn(3, 5, 2)
    skills = torch.randn(3, 3)

    sequence_logits, sequence_hidden = model.forward_sequence(skills, states)
    hidden = None
    for index in range(states.shape[1]):
        step_logits, hidden = model.step(skills, states[:, index], hidden)

    torch.testing.assert_close(step_logits, sequence_logits)
    torch.testing.assert_close(hidden, sequence_hidden)


def test_state_rnn_all_step_outputs_match_online_hidden_rollout() -> None:
    torch.manual_seed(17)
    model = StateSkillRNNTerminator(
        state_dim=2,
        skill_dim=3,
        input_dim=8,
        hidden_dim=8,
        num_layers=1,
        termination_only=False,
    ).eval()
    states = torch.randn(3, 5, 2)
    skills = torch.randn(3, 3)

    sequence_progress, sequence_logits, sequence_hidden = model.forward_all_outputs(
        skills,
        states,
    )
    hidden = None
    online_progress = []
    online_logits = []
    for index in range(states.shape[1]):
        progress, logits, hidden = model.step_outputs(
            skills,
            states[:, index],
            hidden,
        )
        online_progress.append(progress)
        online_logits.append(logits)

    torch.testing.assert_close(sequence_progress, torch.stack(online_progress, dim=1))
    torch.testing.assert_close(sequence_logits, torch.stack(online_logits, dim=1))
    torch.testing.assert_close(sequence_hidden, hidden)


def test_state_rnn_loss_supervises_every_valid_timestep() -> None:
    policy = skill_aux_module.SkillAuxPolicy(
        _config(
            terminator=False,
            predictor=False,
            state_rnn_terminator=True,
        )
    )
    batch = _batch()
    batch["observation.state"] = torch.randn(2, 4, 2)

    loss, metrics = policy._state_rnn_terminator_objective(batch)
    loss.backward()

    assert metrics["state_rnn_terminator/all_step_supervision"] == 1.0
    assert metrics["state_rnn_terminator/mean_valid_length"] == pytest.approx(3.5)
    assert metrics["state_rnn_terminator/positive_fraction"] > 0.0
    assert policy.model.fsq_state_rnn_term_train.rnn.weight_ih_l0.grad is not None


def test_state_sequence_loss_can_use_ordinary_or_balanced_bce() -> None:
    logits = torch.tensor([[-3.0, -2.0, -1.0, 0.0]])
    target = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    valid = torch.ones_like(target, dtype=torch.bool)
    common = {
        "prefix": "state_rnn_terminator",
        "progress_prediction": torch.zeros_like(target),
        "termination_logits": logits,
        "progress_target": torch.zeros_like(target),
        "termination_target": target,
        "valid": valid,
        "positive_weight": 1.0,
        "termination_only": True,
    }

    ordinary, ordinary_metrics = (
        skill_aux_module.SkillAuxPolicy._state_sequence_loss_and_metrics(
            **common,
            balance_positive_negative=False,
        )
    )
    balanced, balanced_metrics = (
        skill_aux_module.SkillAuxPolicy._state_sequence_loss_and_metrics(
            **common,
            balance_positive_negative=True,
        )
    )
    element_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )

    torch.testing.assert_close(ordinary, element_loss.mean())
    torch.testing.assert_close(
        balanced,
        (element_loss[:, :3].mean() + element_loss[:, 3:].mean()) / 2,
    )
    assert ordinary_metrics["state_rnn_terminator/balanced_positive_negative"] == 0.0
    assert balanced_metrics["state_rnn_terminator/balanced_positive_negative"] == 1.0


def test_state_rnn_uses_vanilla_rnn_and_saves_checkpoint(tmp_path) -> None:
    config = _config(
        terminator=False,
        predictor=False,
        state_rnn_terminator=True,
    )
    policy = skill_aux_module.SkillAuxPolicy(config)
    recurrent = policy.model.fsq_state_rnn_term_train

    assert recurrent is not None
    assert isinstance(recurrent.rnn, nn.RNN)
    assert recurrent.rnn.mode == "RNN_TANH"
    assert not hasattr(recurrent, "gru")

    policy.save_pretrained(tmp_path)
    restored = skill_aux_module.SkillAuxPolicy.from_pretrained(
        tmp_path,
        config=config,
        strict=True,
    )
    torch.testing.assert_close(
        list(policy.parameters()),
        list(restored.parameters()),
        rtol=0,
        atol=0,
    )


def test_state_rnn_ignores_history_before_current_skill_start() -> None:
    torch.manual_seed(11)
    model = StateSkillRNNTerminator(
        state_dim=2,
        skill_dim=3,
        input_dim=8,
        hidden_dim=8,
    ).eval()
    states = torch.randn(2, 6, 2)
    skills = torch.randn(2, 3)
    lengths = torch.tensor([2, 4])
    changed_prefix = states.clone()
    changed_prefix[0, :-2] += 1000
    changed_prefix[1, :-4] -= 1000

    logits, _ = model.forward_sequence(skills, states, lengths=lengths)
    changed_logits, _ = model.forward_sequence(
        skills,
        changed_prefix,
        lengths=lengths,
    )

    torch.testing.assert_close(changed_logits, logits)


def test_small_state_mlp_output_contract() -> None:
    model = StateSkillMLPTerminator(
        state_dim=8,
        skill_dim=3,
        hidden_dim=16,
        num_layers=2,
    )
    assert model(torch.randn(4, 3), torch.randn(4, 8)).shape == (4,)


@pytest.mark.parametrize("recurrent", [False, True])
def test_state_terminator_optional_progress_head(recurrent: bool) -> None:
    policy = skill_aux_module.SkillAuxPolicy(
        _config(
            terminator=False,
            predictor=False,
            state_terminator=not recurrent,
            state_rnn_terminator=recurrent,
            state_termination_only=False,
            state_rnn_termination_only=False,
        )
    )
    batch = _batch()
    batch["observation.state"] = torch.randn(2, 4, 2)
    prefix = "state_rnn_terminator" if recurrent else "state_terminator"
    module = (
        policy.model.fsq_state_rnn_term_train
        if recurrent
        else policy.model.fsq_state_term_train
    )

    loss, metrics = policy(batch)
    loss.backward()

    assert metrics[f"{prefix}/termination_only"] == 0.0
    assert f"{prefix}/progress_loss" in metrics
    assert f"{prefix}/progress_mae" in metrics
    assert hasattr(module, "progress_head")
    assert all(
        parameter.grad is not None for parameter in module.progress_head.parameters()
    )


@pytest.mark.parametrize("recurrent", [False, True])
def test_state_termination_only_keeps_original_parameter_set(recurrent: bool) -> None:
    module = (
        StateSkillRNNTerminator(
            state_dim=2,
            skill_dim=3,
            input_dim=8,
            hidden_dim=8,
            termination_only=True,
        )
        if recurrent
        else StateSkillMLPTerminator(
            state_dim=2,
            skill_dim=3,
            hidden_dim=8,
            termination_only=True,
        )
    )

    assert not hasattr(module, "progress_head")
    assert all("progress_head" not in key for key in module.state_dict())


def test_recurrent_policy_predict_accepts_and_returns_hidden() -> None:
    policy = skill_aux_module.SkillAuxPolicy(
        _config(
            terminator=False,
            predictor=False,
            state_rnn_terminator=True,
        )
    )
    codes = torch.tensor([0, 1])
    states = torch.randn(2, 2)

    probability, hidden = policy.state_rnn_terminator_predict(codes, states)
    next_probability, next_hidden = policy.state_rnn_terminator_predict(
        codes,
        states,
        hidden,
    )

    assert probability.shape == next_probability.shape == (2,)
    assert hidden.shape == next_hidden.shape == (1, 2, 64)


def test_recurrent_config_requests_only_proprio_history() -> None:
    config = _config(
        terminator=False,
        predictor=False,
        state_rnn_terminator=True,
    )
    metadata = type(
        "Metadata",
        (),
        {
            "fps": 10,
            "features": {
                "observation.state": {},
                "observation.images.image": {},
                "observation.images.wrist_image": {},
            },
        },
    )()

    delta_timestamps = resolve_delta_timestamps(config, metadata)

    assert config.observation_delta_indices == [-3, -2, -1, 0]
    assert config.state_only_auxiliary is True
    assert set(delta_timestamps) == {"observation.state"}
    assert delta_timestamps["observation.state"] == [-0.3, -0.2, -0.1, 0.0]


def test_fusion_transformer_requests_only_proprio_history() -> None:
    config = _config(
        terminator=True,
        predictor=False,
        terminator_proprio_history=True,
    )
    metadata = type(
        "Metadata",
        (),
        {
            "fps": 10,
            "features": {
                "observation.state": {},
                "observation.images.image": {},
                "observation.images.wrist_image": {},
            },
        },
    )()

    delta_timestamps = resolve_delta_timestamps(config, metadata)

    assert config.observation_delta_indices == [-3, -2, -1, 0]
    assert set(delta_timestamps) == {"observation.state"}
    assert delta_timestamps["observation.state"] == [-0.3, -0.2, -0.1, 0.0]


def test_recurrent_rolling_window_disables_endpoint_sampling() -> None:
    config = _config(
        terminator=False,
        predictor=False,
        state_rnn_terminator=True,
        state_rnn_full_skill_sequence=False,
    )

    assert config.observation_delta_indices == [-3, -2, -1, 0]
    assert config.state_full_skill_supervision is False


def test_state_mlp_and_rnn_objectives_have_isolated_gradients() -> None:
    policy = skill_aux_module.SkillAuxPolicy(
        _config(
            terminator=False,
            predictor=False,
            state_terminator=True,
            state_rnn_terminator=True,
        )
    )
    batch = _batch()
    batch["observation.state"] = torch.randn(2, 4, 2)
    mlp_parameters = list(policy.model.fsq_state_term_train.parameters())
    rnn_parameters = list(policy.model.fsq_state_rnn_term_train.parameters())

    mlp_loss, _ = policy._state_only_terminator_objective(batch)
    mlp_loss.backward()

    assert any(parameter.grad is not None for parameter in mlp_parameters)
    assert all(parameter.grad is None for parameter in rnn_parameters)
    policy.zero_grad(set_to_none=True)

    rnn_loss, _ = policy._state_rnn_terminator_objective(batch)
    rnn_loss.backward()

    assert all(parameter.grad is None for parameter in mlp_parameters)
    assert any(parameter.grad is not None for parameter in rnn_parameters)


@pytest.mark.parametrize("mode", ["mode1", "mode2", "mode1_preskill", "mode2_preskill"])
def test_predictor_architecture_roundtrip_and_removed_sampling(tmp_path, mode):
    from dataclasses import replace
    from lerobot.configs.policies import PreTrainedConfig

    config = replace(
        _config(terminator=False, predictor=True),
        skill_predictor_mode=mode,
        skill_predictor_end_state_mode="xyz",
    )
    config.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.skill_predictor_mode == mode
    assert restored.skill_predictor_sampling_mode == "mode1"
    with pytest.raises(ValueError, match="removed"):
        replace(config, skill_predictor_sampling_mode="mode2")
    if mode.startswith("mode2"):
        policy = object.__new__(skill_aux_module.SkillAuxPolicy)
        nn.Module.__init__(policy)
        policy.config = config
        config.skill_predictor_end_state_skill_source = "predicted"
        assert policy._end_state_predicted_skill_probability() == 0.0


def test_the_shared_processor_factory_accepts_an_auxiliary_config() -> None:
    """skill_aux reuses the skill_expert pipeline, so that factory must not read Stage-1-only fields."""
    from lerobot.policies.skill_expert.processor_skill_expert import (
        make_skill_expert_pre_post_processors,
    )
    from lerobot.policies.skillVLA.processor_skillVLA import (
        SkillVLAPreserveRawStateProcessorStep,
    )

    config = _config(terminator=False, predictor=True)
    assert not hasattr(config, "trains_wrist_patch_alignment")   # only Stage-1 owns the patch head

    preprocessor, _ = make_skill_expert_pre_post_processors(config, dataset_stats=None)
    kept = any(
        isinstance(step, SkillVLAPreserveRawStateProcessorStep) for step in preprocessor.steps
    )
    assert kept is bool(config.train_terminator)
