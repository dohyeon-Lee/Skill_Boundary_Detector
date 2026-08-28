import torch

from lerobot.policies.skill_expert.processor_skill_expert import (
    skill_expert_batch_to_transition,
    skill_expert_transition_to_batch,
)
from lerobot.policies.skillVLA.dataset_skillVLA import (
    SKILL_PREVIOUS_ACTION,
    SKILL_PREVIOUS_ACTION_BOS,
)
from lerobot.policies.skillVLA.processor_skillVLA import (
    skill_vla_batch_to_transition,
    skill_vla_transition_to_batch,
)


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
