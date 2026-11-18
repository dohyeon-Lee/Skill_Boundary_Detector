import json
import sys
from pathlib import Path

import numpy as np


_BUILD_SRC = (
    Path(__file__).resolve().parents[2]
    / "skillVLA/libero/configs/train_skills/build_data/src"
)
sys.path.insert(0, str(_BUILD_SRC))

from compute_global_boundary_threshold import main


def test_global_mean_reducer_applies_threshold_scale(
    tmp_path: Path, monkeypatch
) -> None:
    curves = tmp_path / "curves"
    curves.mkdir()
    np.savez(curves / "ep0000000.npz", episode_id=0, sg_vals=[0.0, 1.0])
    np.savez(curves / "ep0000001.npz", episode_id=1, sg_vals=[2.0, 10.0])
    output = tmp_path / "threshold.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute_global_boundary_threshold.py",
            "--curves_dir",
            str(curves),
            "--output_path",
            str(output),
            "--expected_episodes",
            "2",
            "--threshold_scale",
            "0.8",
        ],
    )

    main()
    payload = json.loads(output.read_text())

    expected_mean = np.mean([0.0, 1.0, 2.0, 10.0])
    assert payload["boundary_threshold_scale"] == 0.8
    assert payload["global_threshold"] == expected_mean * 0.8
    assert payload["global_mean"] == expected_mean


def test_global_mean_reducer_accounts_for_too_short_episode(
    tmp_path: Path, monkeypatch
) -> None:
    curves = tmp_path / "curves"
    curves.mkdir()
    np.savez(curves / "ep0000000.npz", episode_id=0, sg_vals=[1.0, 3.0])
    (curves / "ep0000001.too_short.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reason": "episode_shorter_than_dp_future_horizon",
                "episode_id": 1,
                "task_id": 7,
                "n_frames": 6,
                "future_horizon": 36,
            }
        )
    )
    output = tmp_path / "threshold.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute_global_boundary_threshold.py",
            "--curves_dir",
            str(curves),
            "--output_path",
            str(output),
            "--expected_episodes",
            "2",
        ],
    )

    main()
    payload = json.loads(output.read_text())

    assert payload["global_mean"] == 2.0
    assert payload["n_episodes"] == 1
    assert payload["n_skipped_too_short"] == 1


def test_global_mean_reducer_excludes_terminal_masked_curve_tail(
    tmp_path: Path, monkeypatch
) -> None:
    curves = tmp_path / "curves"
    curves.mkdir()
    np.savez(
        curves / "ep0000000.npz",
        episode_id=0,
        sg_vals=[1.0, 3.0, 1000.0],
        threshold_valid_count=2,
    )
    output = tmp_path / "threshold.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute_global_boundary_threshold.py",
            "--curves_dir",
            str(curves),
            "--output_path",
            str(output),
            "--expected_episodes",
            "1",
        ],
    )

    main()

    payload = json.loads(output.read_text())
    assert payload["global_mean"] == 2.0
    assert payload["n_replan_points"] == 2
