from __future__ import annotations

import pytest
import torch
from torch import nn

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.skill_aux.configuration_skill_aux import SkillAuxConfig
from lerobot.policies.skill_aux import modeling_skill_aux as skill_aux_module
from lerobot.policies.skill_aux.modeling_state_terminator import (
    StateSkillMLPTerminator,
    StateSkillRNNTerminator,
)


class _DummyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))


class _DummyTerminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.progress = nn.Parameter(torch.tensor(0.0))
        self.end = nn.Parameter(torch.tensor(0.0))
        self.vision_encoder = _DummyVision()
        self.state_dim = 2
        self.freeze_vision_encoder = False

    def forward(self, z_q, state, image, wrist_image):
        del state, image, wrist_image
        batch_size = z_q.shape[0]
        return (
            self.progress.sigmoid().expand(batch_size),
            self.end.expand(batch_size),
        )


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


class _DummyPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.reader = nn.Linear(1, 2)
        self.head = nn.Linear(2, config.skill_vocab_size)
        self.vlm = nn.Linear(1, 1)
        self.vlm.requires_grad_(False)
        self.lora_layer_count = 0

    def reader_head_parameters(self):
        return [*self.reader.parameters(), *self.head.parameters()]

    def lora_parameters(self):
        return []

    def auxiliary_parameters(self):
        return self.reader_head_parameters()

    def gradient_checkpointing_enable(self):
        return None

    def loss(self, images, language_tokens, language_mask, target):
        del images, language_tokens, language_mask
        hidden = self.reader(torch.ones(target.shape[0], 1, device=target.device))
        logits = self.head(hidden)
        loss = nn.functional.cross_entropy(logits, target)
        accuracy = (logits.argmax(dim=-1) == target).float().mean().item()
        return loss, accuracy


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
) -> SkillAuxConfig:
    return SkillAuxConfig(
        train_terminator=terminator,
        train_image_only_terminator=image_terminator,
        train_wrist_only_terminator=wrist_terminator,
        train_state_only_terminator=state_terminator,
        train_state_rnn_terminator=state_rnn_terminator,
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
        "observation.state": torch.zeros(batch_size, 2),
        "observation.images.image": image,
        "observation.images.wrist_image": image,
        "skill_start_image": image,
        "skill_start_wrist_image": image,
        "observation.language.tokens": torch.ones(batch_size, 3, dtype=torch.long),
        "observation.language.attention_mask": torch.ones(
            batch_size, 3, dtype=torch.bool
        ),
    }


@pytest.fixture(autouse=True)
def _mock_auxiliary_builders(monkeypatch):
    monkeypatch.setattr(
        skill_aux_module,
        "build_trainable_fsq_terminator",
        lambda path, **kwargs: _DummyTerminator(),
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
        (
            True,
            True,
            True,
            True,
            {
                "terminator",
                "image_terminator",
                "wrist_terminator",
                "skill_predictor_reader_head",
            },
        ),
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


def test_both_disabled_is_rejected():
    with pytest.raises(ValueError, match="terminator.train"):
        _config(terminator=False, predictor=False)


def test_auxiliary_grad_groups_are_parameter_disjoint():
    policy = skill_aux_module.SkillAuxPolicy(
        _config(
            terminator=True,
            image_terminator=True,
            wrist_terminator=True,
            predictor=True,
        )
    )
    groups = policy.isolated_main_optimizer_grad_groups()
    assert set(groups) == {
        "terminator",
        "image_terminator",
        "wrist_terminator",
        "skill_predictor",
    }
    assert {id(parameter) for parameter in groups["terminator"]}.isdisjoint(
        {id(parameter) for parameter in groups["skill_predictor"]}
    )
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
