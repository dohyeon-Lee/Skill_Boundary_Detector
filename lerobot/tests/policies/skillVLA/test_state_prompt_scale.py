"""The predictor's state prompt must arrive on the scale the predictor was trained with."""

from __future__ import annotations

import numpy as np

from lerobot.policies.skillVLA.processor_skillVLA import (
    SkillVLAPrepareStateTokenizerProcessorStep,
)

POLICY_Q01 = [-0.16, -0.2828, -0.2586]     # what the FT policy's normalizer applies
POLICY_Q99 = [0.288, 0.3184, 0.0853]
PRED_Q01 = [-0.147, -0.3013, -0.268]       # what the FT predictor was trained against
PRED_Q99 = [0.2685, 0.2919, 0.1482]


def _normalize(raw, low, high):
    low, high = np.asarray(low), np.asarray(high)
    return 2.0 * (np.asarray(raw) - low) / (high - low) - 1.0


def _step(**fields):
    return SkillVLAPrepareStateTokenizerProcessorStep(**fields)


def test_a_state_normalized_by_the_policy_is_converted_to_the_predictors_scale() -> None:
    raw = np.array([[0.05, -0.10, 0.02], [-0.12, 0.25, 0.12]])
    incoming = _normalize(raw, POLICY_Q01, POLICY_Q99).astype(np.float32)

    converted = _step(
        state_q01=PRED_Q01, state_q99=PRED_Q99,
        incoming_q01=POLICY_Q01, incoming_q99=POLICY_Q99,
    )._rescale_to_predictor(incoming)

    np.testing.assert_allclose(converted, _normalize(raw, PRED_Q01, PRED_Q99), atol=1e-5)
    # The shift is real, not cosmetic: z differs most because its upper quantile differs most.
    assert abs(float(converted[1, 2] - incoming[1, 2])) > 0.1


def test_matching_scales_leave_the_prompt_untouched() -> None:
    """Every run whose policy and predictor share a dataset must behave exactly as before."""
    values = np.array([[0.2, -0.4, 0.6]], dtype=np.float32)

    same = _step(
        state_q01=POLICY_Q01, state_q99=POLICY_Q99,
        incoming_q01=POLICY_Q01, incoming_q99=POLICY_Q99,
    )._rescale_to_predictor(values)
    np.testing.assert_array_equal(same, values)


def test_an_unknown_scale_is_left_alone() -> None:
    """Without both scales the step cannot convert, and must not guess."""
    values = np.array([[0.2, -0.4, 0.6]], dtype=np.float32)

    for fields in (
        {},
        {"state_q01": PRED_Q01, "state_q99": PRED_Q99},                   # no incoming scale
        {"incoming_q01": POLICY_Q01, "incoming_q99": POLICY_Q99},         # no target scale
    ):
        np.testing.assert_array_equal(_step(**fields)._rescale_to_predictor(values), values)


def test_the_scales_survive_a_checkpoint_round_trip() -> None:
    """get_config persists them, so a reloaded pipeline still converts."""
    config = _step(
        state_q01=PRED_Q01, state_q99=PRED_Q99,
        incoming_q01=POLICY_Q01, incoming_q99=POLICY_Q99,
    ).get_config()

    np.testing.assert_allclose(config["state_q01"][:3], PRED_Q01, atol=1e-6)
    np.testing.assert_allclose(config["incoming_q99"][:3], POLICY_Q99, atol=1e-6)
