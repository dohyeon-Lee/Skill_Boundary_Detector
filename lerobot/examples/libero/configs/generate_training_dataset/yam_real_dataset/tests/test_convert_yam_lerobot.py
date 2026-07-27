from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from convert_yam_lerobot import (  # noqa: E402
    canonical_camera_map,
    output_features,
    recorder_position_indices,
    resize_with_pad,
    select_episode_rows,
)


def recorder_state_feature():
    names = []
    for arm in ("left", "right"):
        for field in ("pos", "vel", "eff"):
            names.extend([f"{arm}.{field}.{index}" for index in range(7)])
    return {"dtype": "float32", "shape": [42], "names": names}


def test_position_indices_follow_recorder_names_not_first_fourteen_values():
    assert recorder_position_indices(recorder_state_feature()) == [*range(7), *range(21, 28)]


def test_camera_map_uses_pi05_canonical_names():
    assert canonical_camera_map(None) == {
        "observation.images.agentview": "observation.images.top",
        "observation.images.wrist_left": "observation.images.left_wrist",
        "observation.images.wrist_right": "observation.images.right_wrist",
    }


def test_episode_filter_keeps_successes_without_reindex_assumptions():
    rows = [
        {"episode_index": 5, "length": 20},
        {"episode_index": 8, "length": 4},
        {"episode_index": 11, "length": 30},
    ]
    outcomes = {
        5: {"outcome": "success"},
        8: {"outcome": "success"},
        11: {"outcome": "fail"},
    }
    selected, skipped = select_episode_rows(
        rows,
        outcomes,
        include_outcomes={"success"},
        require_outcomes=True,
        min_frames=10,
        include_indices=set(),
        exclude_indices=set(),
        max_episodes=None,
    )

    assert [row["episode_index"] for row in selected] == [5]
    assert skipped == {"too_short": 1, "outcome=fail": 1}


def test_output_features_are_14d_and_three_equal_sized_cameras():
    camera_map = canonical_camera_map(None)
    source_info = {"features": {key: {"dtype": "video", "shape": [480, 640, 3]} for key in camera_map}}
    features = output_features(source_info, camera_map, image_size=256)

    assert features["observation.state"]["shape"] == (14,)
    assert features["action"]["shape"] == (14,)
    for target in camera_map.values():
        assert features[target]["shape"] == (256, 256, 3)


def test_resize_with_pad_preserves_content_and_adds_letterbox():
    image = np.full((4, 8, 3), 255, dtype=np.uint8)
    output = resize_with_pad(image, 8)

    assert output.shape == (8, 8, 3)
    assert np.all(output[2:6] == 255)
    assert np.all(output[:2] == 0)
    assert np.all(output[6:] == 0)


def test_missing_outcome_fails_when_filter_is_strict():
    with pytest.raises(ValueError, match="no outcomes"):
        select_episode_rows(
            [{"episode_index": 1, "length": 20}],
            {},
            include_outcomes={"success"},
            require_outcomes=True,
            min_frames=1,
            include_indices=set(),
            exclude_indices=set(),
            max_episodes=None,
        )
