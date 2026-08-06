from __future__ import annotations

import pytest
import torch
from torch import nn

from lerobot.policies.skill_aux.configuration_skill_aux import SkillAuxConfig
from lerobot.policies.skill_aux import modeling_skill_aux as skill_aux_module
from lerobot.policies.skill_expert.processor_skill_expert import (
    SKILL_BATCH_KEYS,
    make_skill_expert_pre_post_processors,
    skill_expert_batch_to_transition,
    skill_expert_transition_to_batch,
)
from lerobot.policies.skillVLA.processor_skillVLA import (
    SkillVLAPreserveRawStateProcessorStep,
)
from lerobot.types import TransitionKey


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


class _DummyStartComparisonTerminator(_DummyTerminator):
    def forward(self, z_q, state, start_image, image, wrist_image):
        del state, start_image, image, wrist_image
        batch_size = z_q.shape[0]
        return (
            self.progress.sigmoid().expand(batch_size),
            self.end.expand(batch_size),
        )


class _DummyStartComparisonImageOnlyTerminator(_DummyTerminator):
    def forward(self, z_q, start_image, image, wrist_image):
        del start_image, image, wrist_image
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
    start_comparison_terminator: bool = False,
    start_comparison_image_only_terminator: bool = False,
    endpoint_oversampling: bool = False,
) -> SkillAuxConfig:
    return SkillAuxConfig(
        train_terminator=terminator,
        train_image_only_terminator=image_terminator,
        train_wrist_only_terminator=wrist_terminator,
        train_start_comparison_terminator=start_comparison_terminator,
        train_start_comparison_image_only_terminator=(
            start_comparison_image_only_terminator
        ),
        train_skill_predictor=predictor,
        terminator_endpoint_oversampling_enabled=endpoint_oversampling,
        terminator_endpoint_exact_end_fraction=0.5,
        terminator_endpoint_near_end_fraction=0.5,
        terminator_endpoint_near_end_max_distance=2,
        fsq_path="dummy.pt",
        skill_predictor_lora=False,
        skill_predictor_detach_vlm=True,
        dtype="float32",
        device="cpu",
    )


def _batch() -> dict:
    batch_size = 2
    image = torch.zeros(batch_size, 3, 4, 4)
    return {
        "skill_code_true": torch.tensor([0, 1]),
        "skill_sequence": torch.tensor([[0, 1], [0, 1]]),
        "skill_index": torch.tensor([0, 1]),
        "skill_code": torch.tensor([0, 1]),
        "skill_ds": torch.tensor([0, 2]),
        "skill_de": torch.tensor([2, 0]),
        "skill_decoder_state": torch.zeros(batch_size, 2),
        "observation.images.image": image,
        "observation.images.wrist_image": image,
        "skill_start_image": image,
        "skill_start_wrist_image": image,
        "terminator_start_image": image,
        "observation.language.tokens": torch.ones(batch_size, 3, dtype=torch.long),
        "observation.language.attention_mask": torch.ones(
            batch_size, 3, dtype=torch.bool
        ),
    }


@pytest.fixture(autouse=True)
def _mock_auxiliary_builders(monkeypatch):
    monkeypatch.setattr(skill_aux_module, "build_fsq_terminator", lambda path: _DummyTerminator())
    monkeypatch.setattr(
        skill_aux_module,
        "build_fsq_image_only_terminator",
        lambda path: _DummyImageOnlyTerminator(),
    )
    monkeypatch.setattr(
        skill_aux_module,
        "build_fsq_wrist_only_terminator",
        lambda path: _DummyWristOnlyTerminator(),
    )
    monkeypatch.setattr(
        skill_aux_module,
        "build_fsq_start_comparison_terminator",
        lambda path: _DummyStartComparisonTerminator(),
    )
    monkeypatch.setattr(
        skill_aux_module,
        "build_fsq_start_comparison_image_only_terminator",
        lambda path: _DummyStartComparisonImageOnlyTerminator(),
    )
    monkeypatch.setattr(skill_aux_module, "FrozenVLMSkillPredictor", _DummyPredictor)


@pytest.mark.parametrize(
    (
        "terminator",
        "image_terminator",
        "wrist_terminator",
        "start_comparison_terminator",
        "start_comparison_image_only_terminator",
        "predictor",
        "expected_groups",
    ),
    [
        (True, False, False, False, False, False, {"terminator"}),
        (False, True, False, False, False, False, {"image_terminator"}),
        (False, False, True, False, False, False, {"wrist_terminator"}),
        (False, False, False, True, False, False, {"start_comparison_terminator"}),
        (
            False,
            False,
            False,
            False,
            True,
            False,
            {"start_comparison_image_only_terminator"},
        ),
        (False, False, False, False, False, True, {"skill_predictor_reader_head"}),
        (
            True,
            True,
            True,
            True,
            True,
            True,
            {
                "terminator",
                "image_terminator",
                "wrist_terminator",
                "start_comparison_terminator",
                "start_comparison_image_only_terminator",
                "skill_predictor_reader_head",
            },
        ),
    ],
)
def test_independent_training_switches(
    terminator,
    image_terminator,
    wrist_terminator,
    start_comparison_terminator,
    start_comparison_image_only_terminator,
    predictor,
    expected_groups,
):
    policy = skill_aux_module.SkillAuxPolicy(
        _config(
            terminator=terminator,
            image_terminator=image_terminator,
            wrist_terminator=wrist_terminator,
            start_comparison_terminator=start_comparison_terminator,
            start_comparison_image_only_terminator=(
                start_comparison_image_only_terminator
            ),
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
    assert (
        any(key.startswith("start_comparison_terminator/") for key in metrics)
        is start_comparison_terminator
    )
    assert (
        any(
            key.startswith("start_comparison_image_only_terminator/")
            for key in metrics
        )
        is start_comparison_image_only_terminator
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
            start_comparison_terminator=True,
            start_comparison_image_only_terminator=True,
            predictor=True,
        )
    )
    groups = policy.isolated_main_optimizer_grad_groups()
    assert set(groups) == {
        "terminator",
        "image_terminator",
        "wrist_terminator",
        "start_comparison_terminator",
        "start_comparison_image_only_terminator",
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
    assert {
        id(parameter) for parameter in groups["start_comparison_terminator"]
    }.isdisjoint({id(parameter) for parameter in groups["terminator"]})
    assert {
        id(parameter)
        for parameter in groups["start_comparison_image_only_terminator"]
    }.isdisjoint(
        {id(parameter) for parameter in groups["start_comparison_terminator"]}
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


def test_start_comparison_image_only_terminator_uses_no_state() -> None:
    policy = skill_aux_module.SkillAuxPolicy(
        _config(
            terminator=False,
            start_comparison_image_only_terminator=True,
            predictor=False,
        )
    )
    batch = _batch()
    del batch["skill_decoder_state"]

    loss, metrics = policy(batch)

    assert loss.requires_grad
    assert "start_comparison_image_only_terminator/loss" in metrics
    assert any(
        key.startswith("model.fsq_start_comparison_image_term_train.")
        for key in policy.state_dict()
    )


def test_start_comparison_dataset_key_survives_batch_conversion() -> None:
    assert "terminator_start_image" in SKILL_BATCH_KEYS
    image = torch.zeros(2, 3, 4, 4)

    state = torch.randn(2, 8)
    transition = skill_expert_batch_to_transition(
        {
            "observation.state": state,
            "observation.images.image": image,
            "terminator_start_image": image + 1,
        }
    )
    transition = SkillVLAPreserveRawStateProcessorStep()(transition)
    restored = skill_expert_transition_to_batch(transition)

    complementary = transition[TransitionKey.COMPLEMENTARY_DATA]
    torch.testing.assert_close(complementary["terminator_start_image"], image + 1)
    torch.testing.assert_close(restored["terminator_start_image"], image + 1)
    torch.testing.assert_close(restored["skill_decoder_state"], state)


def test_full_start_comparison_preserves_raw_state() -> None:
    config = _config(
        terminator=False,
        start_comparison_terminator=True,
        predictor=False,
    )
    config.validate_features()

    preprocessor, _ = make_skill_expert_pre_post_processors(config)

    assert any(
        isinstance(step, SkillVLAPreserveRawStateProcessorStep)
        for step in preprocessor.steps
    )


def test_endpoint_batch_composition_metrics_are_reported():
    policy = skill_aux_module.SkillAuxPolicy(
        _config(
            terminator=True,
            predictor=False,
            endpoint_oversampling=True,
        )
    )

    _, metrics = policy(_batch())

    assert metrics["batch_sampling/exact_end_fraction"] == 0.5
    assert metrics["batch_sampling/near_end_fraction"] == 0.5
    assert metrics["batch_sampling/outside_end_window_fraction"] == 0.0
    assert metrics["batch_sampling/exact_end_target_fraction"] == 0.5
    assert metrics["batch_sampling/near_end_target_fraction"] == 0.5


def test_endpoint_oversampling_requires_a_terminator_objective():
    with pytest.raises(ValueError, match="endpoint oversampling requires"):
        _config(
            terminator=False,
            image_terminator=False,
            predictor=True,
            endpoint_oversampling=True,
        )
