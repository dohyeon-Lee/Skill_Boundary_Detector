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
    SKILL_START_STATE,
    SKILL_START_STATE_NORMALIZED,
    TERMINATOR_START_STATE,
)
from lerobot.policies.skillVLA.processor_skillVLA import (
    TERMINATOR_PREDICTOR_TASK,
    SkillVLAPrepareStateTokenizerProcessorStep,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION


def test_prev_action_context_survives_training_preprocessors() -> None:
    batch = {
        SKILL_PREVIOUS_ACTION: torch.randn(2, 7),
        SKILL_PREVIOUS_ACTION_BOS: torch.tensor([True, False]),
    }

    restored = skill_expert_transition_to_batch(
        skill_expert_batch_to_transition(batch)
    )
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


def test_joint_predictor_gets_normalized_start_proprio_without_changing_raw_goal() -> None:
    batch = {SKILL_START_STATE: torch.tensor([[0.0, 5.0, 10.0]])}
    normalizer = SkillExpertNormalizerProcessorStep(
        features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,))
        },
        norm_map={FeatureType.STATE: NormalizationMode.QUANTILES},
        stats={
            "observation.state": {
                "q01": torch.tensor([0.0, 0.0, 0.0]),
                "q99": torch.tensor([10.0, 10.0, 10.0]),
            }
        },
    )
    restored = skill_expert_transition_to_batch(
        normalizer(skill_expert_batch_to_transition(batch))
    )
    torch.testing.assert_close(restored[SKILL_START_STATE], batch[SKILL_START_STATE])
    torch.testing.assert_close(
        restored[SKILL_START_STATE_NORMALIZED],
        torch.tensor([[-1.0, 0.0, 1.0]]),
    )


def test_joint_terminator_gets_a_separate_predictor_prompt() -> None:
    step = SkillVLAPrepareStateTokenizerProcessorStep(
        state_q01=[0.0, 0.0, 0.0],
        state_q99=[10.0, 10.0, 10.0],
    )
    transition = {
        TransitionKey.OBSERVATION: {
            "observation.state": torch.zeros(1, 3),
        },
        TransitionKey.COMPLEMENTARY_DATA: {
            "task": ["pick_object"],
            SKILL_START_STATE: torch.zeros(1, 3),
            TERMINATOR_START_STATE: torch.full((1, 3), 10.0),
        },
    }
    processed = step(transition)
    complementary = processed[TransitionKey.COMPLEMENTARY_DATA]
    assert complementary["task"][0].startswith("Task: pick object")
    assert complementary[TERMINATOR_PREDICTOR_TASK][0].startswith(
        "Task: pick object"
    )
    assert complementary["task"] != complementary[TERMINATOR_PREDICTOR_TASK]


def test_joint_past_terminator_anchors_keep_raw_and_normalized_states_and_own_prompts():
    from lerobot.policies.skillVLA.newterm_supervision import (
        POST_END_START_STATE, POST_END_START_STATE_NORMALIZED,
        POST_END_PREDICTOR_START_IMAGE, POST_END_PREDICTOR_TASK, TERMINAL_FRAME,
    )
    raw = torch.tensor([[[0., 0., 0.], [10., 10., 10.]]])
    batch = {"task": ["pick"], SKILL_START_STATE: torch.full((1, 3), 5.),
             POST_END_START_STATE: raw, POST_END_PREDICTOR_START_IMAGE: torch.zeros(1, 2, 3, 4, 4),
             TERMINAL_FRAME: torch.tensor([True])}
    normalizer = SkillExpertNormalizerProcessorStep(
        features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,))},
        norm_map={FeatureType.STATE: NormalizationMode.QUANTILES},
        stats={"observation.state": {"q01": torch.zeros(3), "q99": torch.full((3,), 10.)}},
    )
    transition = normalizer(skill_expert_batch_to_transition(batch))
    step = SkillVLAPrepareStateTokenizerProcessorStep(state_q01=[0]*3, state_q99=[10]*3)
    restored = skill_expert_transition_to_batch(step(transition))
    torch.testing.assert_close(restored[POST_END_START_STATE], raw)
    torch.testing.assert_close(restored[POST_END_START_STATE_NORMALIZED], torch.tensor([[[-1.]*3, [1.]*3]]))
    assert len(restored[POST_END_PREDICTOR_TASK]) == 2
    assert "State: 0 0 0" in restored[POST_END_PREDICTOR_TASK][0]
    assert "State: 255 255 255" in restored[POST_END_PREDICTOR_TASK][1]
    assert restored[TERMINAL_FRAME].tolist() == [True]


def test_past_prompt_tokenizer_preserves_main_and_current_terminator_tokens_and_serializes_keys():
    from lerobot.policies.skillVLA.processor_skillVLA import (
        SkillVLATerminatorPredictorTokenizerProcessorStep, TERMINATOR_PREDICTOR_LANGUAGE_TOKENS,
    )
    from lerobot.policies.skillVLA.newterm_supervision import (
        POST_END_PREDICTOR_TASK, POST_END_PREDICTOR_LANGUAGE_TOKENS, POST_END_PREDICTOR_LANGUAGE_ATTENTION_MASK,
    )
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            size = (len(texts), kwargs["max_length"])
            return {"input_ids": torch.full(size, 7), "attention_mask": torch.ones(size)}

    step = SkillVLATerminatorPredictorTokenizerProcessorStep(
        tokenizer=Tokenizer(), max_length=6, task_key=POST_END_PREDICTOR_TASK,
        output_tokens_key=POST_END_PREDICTOR_LANGUAGE_TOKENS,
        output_mask_key=POST_END_PREDICTOR_LANGUAGE_ATTENTION_MASK,
    )
    transition = {
        TransitionKey.OBSERVATION: {
            OBS_LANGUAGE_TOKENS: torch.ones(2, 6, dtype=torch.long),
            OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 6, dtype=torch.bool),
            TERMINATOR_PREDICTOR_LANGUAGE_TOKENS: torch.full((2, 6), 3),
        },
        TransitionKey.COMPLEMENTARY_DATA: {POST_END_PREDICTOR_TASK: ["a", "b", "c", "d"]},
    }
    processed = step(transition)[TransitionKey.OBSERVATION]
    assert processed[POST_END_PREDICTOR_LANGUAGE_TOKENS].shape == (4, 6)
    assert processed[POST_END_PREDICTOR_LANGUAGE_TOKENS].eq(7).all()
    assert processed[OBS_LANGUAGE_TOKENS].eq(1).all()
    assert processed[TERMINATOR_PREDICTOR_LANGUAGE_TOKENS].eq(3).all()
    restored = SkillVLATerminatorPredictorTokenizerProcessorStep(tokenizer=Tokenizer(), **step.get_config())
    assert restored.output_tokens_key == POST_END_PREDICTOR_LANGUAGE_TOKENS
    # The same saved Joint processor is valid at rollout with no auxiliary prompts.
    inference = {TransitionKey.OBSERVATION: {OBS_LANGUAGE_TOKENS: torch.ones(1, 6)},
                 TransitionKey.COMPLEMENTARY_DATA: {"task": ["pick"]}}
    assert step(inference)[TransitionKey.OBSERVATION][OBS_LANGUAGE_TOKENS].eq(1).all()
