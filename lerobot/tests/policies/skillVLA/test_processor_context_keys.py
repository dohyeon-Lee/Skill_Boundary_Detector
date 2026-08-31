import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.skill_expert.processor_skill_expert import (
    SkillExpertNormalizerProcessorStep,
    skill_expert_batch_to_transition,
    skill_expert_transition_to_batch,
)
from lerobot.policies.skillVLA.dataset_skillVLA import (
    SKILL_CANONICAL_ACTION_IS_PAD,
    SKILL_CANONICAL_ACTION_LENGTH,
    SKILL_CANONICAL_ACTIONS,
    SKILL_PREVIOUS_ACTION,
    SKILL_PREVIOUS_ACTION_BOS,
)
from lerobot.policies.skillVLA.processor_skillVLA import (
    skill_vla_batch_to_transition,
    skill_vla_transition_to_batch,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION


def test_prev_action_context_survives_training_preprocessors() -> None:
    batch = {
        SKILL_PREVIOUS_ACTION: torch.randn(2, 7),
        SKILL_PREVIOUS_ACTION_BOS: torch.tensor([True, False]),
    }

    converters = (
        (skill_expert_batch_to_transition, skill_expert_transition_to_batch),
        (skill_vla_batch_to_transition, skill_vla_transition_to_batch),
    )
    for to_transition, to_batch in converters:
        restored = to_batch(to_transition(batch))
        torch.testing.assert_close(
            restored[SKILL_PREVIOUS_ACTION], batch[SKILL_PREVIOUS_ACTION]
        )
        torch.testing.assert_close(
            restored[SKILL_PREVIOUS_ACTION_BOS], batch[SKILL_PREVIOUS_ACTION_BOS]
        )


def test_arch0_skill_canonical_target_survives_and_shares_action_normalization() -> None:
    batch = {
        ACTION: torch.tensor([[[0.0, 10.0], [5.0, 5.0]]]),
        SKILL_CANONICAL_ACTIONS: torch.tensor(
            [[[0.0, 10.0], [5.0, 5.0], [10.0, 0.0]]]
        ),
        SKILL_CANONICAL_ACTION_IS_PAD: torch.tensor([[False, False, True]]),
        SKILL_CANONICAL_ACTION_LENGTH: torch.tensor([2]),
    }
    transition = skill_expert_batch_to_transition(batch)
    normalizer = SkillExpertNormalizerProcessorStep(
        features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))
        },
        norm_map={FeatureType.ACTION: NormalizationMode.QUANTILES},
        stats={
            ACTION: {
                "q01": torch.tensor([0.0, 0.0]),
                "q99": torch.tensor([10.0, 10.0]),
            }
        },
    )

    normalized = normalizer(transition)
    restored = skill_expert_transition_to_batch(normalized)

    torch.testing.assert_close(
        normalized[TransitionKey.ACTION],
        torch.tensor([[[-1.0, 1.0], [0.0, 0.0]]]),
    )
    torch.testing.assert_close(
        restored[SKILL_CANONICAL_ACTIONS],
        torch.tensor([[[-1.0, 1.0], [0.0, 0.0], [1.0, -1.0]]]),
    )
    torch.testing.assert_close(
        restored[SKILL_CANONICAL_ACTION_IS_PAD],
        batch[SKILL_CANONICAL_ACTION_IS_PAD],
    )
