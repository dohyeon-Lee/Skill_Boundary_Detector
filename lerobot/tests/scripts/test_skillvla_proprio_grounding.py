import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


_LIBERO_EXAMPLES = Path(__file__).resolve().parents[2] / "examples/libero"
sys.path.insert(0, str(_LIBERO_EXAMPLES))

from add_skill_latents_to_dataset import Args, main  # noqa: E402


def _state(x: float, y: float, z: float, tail: float) -> np.ndarray:
    return np.asarray([x, y, z, tail, tail + 1, tail + 2, tail + 3, tail + 4], dtype=np.float32)


def test_skillvla_builder_grounds_each_episode_and_recomputes_stats(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "data/chunk-000").mkdir(parents=True)
    (source / "meta").mkdir()
    frame = pd.DataFrame(
        {
            "episode_index": [0, 0, 1, 1],
            "frame_index": [0, 1, 0, 1],
            "observation.state": [
                _state(-0.20, 0.00, 1.17, 10),
                _state(-0.10, 0.05, 1.07, 20),
                _state(-0.05, 0.00, 0.68, 30),
                _state(-0.07, -0.10, 0.78, 40),
            ],
            "observation.states.ee_state": [
                value[:6].copy()
                for value in [
                    _state(-0.20, 0.00, 1.17, 10),
                    _state(-0.10, 0.05, 1.07, 20),
                    _state(-0.05, 0.00, 0.68, 30),
                    _state(-0.07, -0.10, 0.78, 40),
                ]
            ],
            "action": [np.zeros(7, dtype=np.float32) for _ in range(4)],
        }
    )
    frame.to_parquet(source / "data/chunk-000/file-000.parquet", index=False)
    (source / "meta/info.json").write_text(
        json.dumps(
            {
                "repo_id": "test/source",
                "total_episodes": 2,
                "total_frames": 4,
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [8]},
                    "observation.states.ee_state": {"dtype": "float32", "shape": [6]},
                    "action": {"dtype": "float32", "shape": [7]},
                },
            }
        )
    )
    original_action_stats = {"mean": [0.0] * 7, "count": [4]}
    (source / "meta/stats.json").write_text(
        json.dumps(
            {
                "observation.state": {"mean": [999.0] * 8, "count": [4]},
                "observation.states.ee_state": {"mean": [999.0] * 6, "count": [4]},
                "action": original_action_stats,
            }
        )
    )
    latents = tmp_path / "latents.npz"
    np.savez(
        latents,
        episode_id=np.asarray([0, 1], dtype=np.int32),
        frame_start=np.asarray([0, 0], dtype=np.int32),
        frame_end=np.asarray([2, 2], dtype=np.int32),
        tokens=np.asarray([1, 2], dtype=np.int32),
    )
    iss = tmp_path / "skill_initial_state.npz"

    main(
        Args(
            src_dataset_dir=str(source),
            dst_dataset_dir=str(destination),
            latents_path=str(latents),
            dst_repo_id="test/grounded",
            fsq_levels=[3, 3, 3],
            max_order=1,
            max_length=2,
            pmax=0,
            early_start_pmax=0,
            late_start_pmax=0,
            early_end_pmax=0,
            late_end_pmax=0,
            iss_npz_path=str(iss),
            proprio_grounding="episode_start_xyz",
        )
    )

    built = pd.read_parquet(destination / "data/chunk-000/file-000.parquet")
    states = np.stack(built["observation.state"].to_numpy())
    np.testing.assert_allclose(
        states[:, :3],
        [[0, 0, 0], [0.10, 0.05, -0.10], [0, 0, 0], [-0.02, -0.10, 0.10]],
        atol=1e-6,
    )
    np.testing.assert_allclose(states[:, 3:], [row[3:] for row in frame["observation.state"]])
    np.testing.assert_allclose(np.stack(built["skill_decoder_state"]), states)

    info = json.loads((destination / "meta/info.json").read_text())
    assert info["proprio_grounding"] == "episode_start_xyz"
    stats = json.loads((destination / "meta/stats.json").read_text())
    np.testing.assert_allclose(stats["observation.state"]["mean"][:3], states[:, :3].mean(0))
    assert stats["observation.state"]["count"] == [4]
    assert stats["action"] == original_action_stats
    with np.load(iss) as payload:
        assert str(payload["proprio_grounding"]) == "episode_start_xyz"
        np.testing.assert_allclose(payload["iss_windows"][:, 0, :3], 0.0)
