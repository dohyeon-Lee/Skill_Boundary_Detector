import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/stage1_eval/oracle_matching"
)
sys.path.insert(0, str(_SRC))

from build_langgap_init_states import (  # noqa: E402
    image_signature,
    rank_candidates,
    resolve_task_specs,
)


def _suite(prefix: str, count: int):
    return SimpleNamespace(
        tasks=[
            SimpleNamespace(name=f"{prefix}_{index}", language=f"{prefix} language {index}")
            for index in range(count)
        ]
    )


def test_langgap_task_mapping_uses_official_suite_and_collected_ext_subset() -> None:
    suites = {
        "libero_10": _suite("ten", 10),
        "libero_goal": _suite("goal", 10),
        "libero_object": _suite("object", 10),
        "libero_spatial": _suite("spatial", 10),
        "langgap_ext": _suite("ext", 59),
    }
    task_languages = {
        0: suites["libero_10"].tasks[4].language,
        10: suites["libero_goal"].tasks[8].language,
        20: suites["libero_object"].tasks[9].language,
        30: suites["libero_spatial"].tasks[6].language,
        40: suites["langgap_ext"].tasks[0].language,
        53: suites["langgap_ext"].tasks[45].language,
    }

    specs = resolve_task_specs(task_languages, suites)

    assert (specs[0].suite_name, specs[0].suite_task_id) == ("libero_10", 4)
    assert (specs[30].suite_name, specs[30].suite_task_id) == ("libero_spatial", 6)
    assert (specs[40].suite_name, specs[40].suite_task_id) == ("langgap_ext", 0)
    assert (specs[53].suite_name, specs[53].suite_task_id) == ("langgap_ext", 45)


def test_candidate_ranking_uses_state_and_image_and_reports_margin() -> None:
    candidate_states = np.zeros((3, 8), dtype=np.float32)
    candidate_states[1, 0] = 0.01
    candidate_states[2, 0] = 0.04
    episode_state = candidate_states[1].copy()
    candidate_images = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    candidate_images[1] = 80
    candidate_images[2] = 160

    match = rank_candidates(
        episode_state,
        candidate_images[1],
        None,
        candidate_states,
        candidate_images,
        None,
        state_weight=1.0,
        image_weight=4.0,
        wrist_weight=0.0,
        max_state_score=1.0,
        max_image_mae=0.18,
        min_score_margin=0.01,
    )

    assert match.init_index == 1
    assert match.confident is True
    assert match.margin > 0.01


def test_candidate_ranking_rejects_an_ambiguous_exact_tie() -> None:
    states = np.zeros((2, 8), dtype=np.float32)
    images = np.zeros((2, 4, 4, 3), dtype=np.uint8)

    match = rank_candidates(
        states[0],
        images[0],
        None,
        states,
        images,
        None,
        state_weight=1.0,
        image_weight=1.0,
        wrist_weight=0.0,
        max_state_score=1.0,
        max_image_mae=0.18,
        min_score_margin=0.01,
    )

    assert match.confident is False
    assert "margin" in match.reason


def test_image_signature_normalizes_chw_float_and_hwc_uint8() -> None:
    hwc = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    chw = np.moveaxis(hwc.astype(np.float32) / 255.0, -1, 0)

    assert np.array_equal(image_signature(hwc, 8), image_signature(chw, 8))
