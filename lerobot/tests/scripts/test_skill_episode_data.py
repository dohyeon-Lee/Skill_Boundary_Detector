from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/build_data/src"
)
sys.path.insert(0, str(_SRC))
import skill_episode_data  # noqa: E402


def test_langgap_dataset_builds_episode_replay_plan(tmp_path: Path) -> None:
    import pandas as pd

    dataset_dir = tmp_path / "skill_dataset"
    (dataset_dir / "meta" / "episodes").mkdir(parents=True)
    (dataset_dir / "data" / "chunk-000").mkdir(parents=True)
    pd.DataFrame(
        {
            "episode_index": [9],
            "data/chunk_index": [0],
            "data/file_index": [0],
            "length": [3],
            "tasks": [["task"]],
        }
    ).to_parquet(dataset_dir / "meta" / "episodes" / "chunk.parquet")
    pd.DataFrame(
        {"task_index": [0]},
        index=pd.Index(["task"], name="task"),
    ).to_parquet(dataset_dir / "meta" / "tasks.parquet")
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)
    states = np.arange(24, dtype=np.float32).reshape(3, 8)
    pd.DataFrame(
        {
            "episode_index": [9, 9, 9],
            "frame_index": [0, 1, 2],
            "action": list(actions),
            "observation.state": list(states),
        }
    ).to_parquet(dataset_dir / "data" / "chunk-000" / "file-000.parquet")
    latents_path = tmp_path / "skill_latents.npz"
    np.savez(
        latents_path,
        tokens=np.asarray([2]),
        episode_id=np.asarray([9]),
        skill_index=np.asarray([0]),
        frame_start=np.asarray([2]),
        frame_end=np.asarray([3]),
    )
    init_state = np.asarray([8.0, 9.0, 10.0], dtype=np.float64)
    exact_path = tmp_path / "eval_init_states.npz"
    np.savez(
        exact_path,
        episode_index=np.asarray([9]),
        dataset_task_id=np.asarray([0]),
        init_states=np.asarray([init_state]),
        suite_name=np.asarray(["langgap_ext"]),
        suite_task_id=np.asarray([5]),
        init_state_index=np.asarray([17]),
    )

    dataset = skill_episode_data.SkillEvaluationDataset(
        skill_dataset_dir=dataset_dir,
        skill_latents_path=latents_path,
        eval_init_states_path=exact_path,
        original_dataset_dir="",
        suite_name="langgap_ext",
    )
    aligned = dataset.load_aligned_episode(9)
    state, replay, init_index = aligned.restoration_at(2)

    assert dataset.uses_langgap_replay is True
    assert aligned.requires_episode_replay is True
    assert aligned.source.task_id == 0
    assert aligned.source.env_task_id == 5
    dataset.task_descriptions[0] = "wrong task-ID lookup"
    assert dataset.episode_task_description(9) == "task"
    assert init_index == 17
    np.testing.assert_array_equal(state, init_state)
    np.testing.assert_array_equal(replay, actions[:2])
    np.testing.assert_array_equal(aligned.episode_start_xyz, states[0, :3])


def test_grounded_langgap_uses_raw_episode_start_xyz(tmp_path: Path) -> None:
    import json
    import pandas as pd

    skill_dir = tmp_path / "skill_dataset"
    raw_dir = tmp_path / "raw_dataset"
    for dataset_dir in (skill_dir, raw_dir):
        (dataset_dir / "meta" / "episodes").mkdir(parents=True)
        (dataset_dir / "data" / "chunk-000").mkdir(parents=True)
        pd.DataFrame(
            {
                "episode_index": [9],
                "data/chunk_index": [0],
                "data/file_index": [0],
                "length": [3],
                "tasks": [["task"]],
            }
        ).to_parquet(dataset_dir / "meta" / "episodes" / "chunk.parquet")
        pd.DataFrame(
            {"task_index": [0]},
            index=pd.Index(["task"], name="task"),
        ).to_parquet(dataset_dir / "meta" / "tasks.parquet")

    raw_states = np.asarray(
        [
            [-0.2, 0.01, 1.15, 3.14, 0.0, 0.0, 0.04, -0.04],
            [-0.1, 0.03, 1.05, 3.14, 0.0, 0.0, 0.04, -0.04],
            [0.0, 0.05, 0.95, 3.14, 0.0, 0.0, 0.04, -0.04],
        ],
        dtype=np.float32,
    )
    grounded_states = raw_states.copy()
    grounded_states[:, :3] -= raw_states[0, :3]
    actions = np.zeros((3, 7), dtype=np.float32)
    for dataset_dir, states in (
        (skill_dir, grounded_states),
        (raw_dir, raw_states),
    ):
        pd.DataFrame(
            {
                "episode_index": [9, 9, 9],
                "frame_index": [0, 1, 2],
                "action": list(actions),
                "observation.state": list(states),
            }
        ).to_parquet(dataset_dir / "data" / "chunk-000" / "file-000.parquet")
    (skill_dir / "meta" / "info.json").write_text(
        json.dumps({"proprio_grounding": "episode_start_xyz"})
    )

    latents_path = tmp_path / "skill_latents.npz"
    np.savez(
        latents_path,
        tokens=np.asarray([2]),
        episode_id=np.asarray([9]),
        skill_index=np.asarray([0]),
        frame_start=np.asarray([2]),
        frame_end=np.asarray([3]),
    )
    init_state = np.asarray([8.0, 9.0, 10.0], dtype=np.float64)
    exact_path = tmp_path / "eval_init_states.npz"
    np.savez(
        exact_path,
        episode_index=np.asarray([9]),
        dataset_task_id=np.asarray([0]),
        init_states=np.asarray([init_state]),
        suite_name=np.asarray(["langgap_ext"]),
        suite_task_id=np.asarray([5]),
        init_state_index=np.asarray([17]),
    )

    dataset = skill_episode_data.SkillEvaluationDataset(
        skill_dataset_dir=skill_dir,
        skill_latents_path=latents_path,
        eval_init_states_path=exact_path,
        original_dataset_dir=None,
        suite_name="langgap_ext",
        raw_dataset_dir=raw_dir,
    )
    aligned = dataset.load_aligned_episode(9)

    np.testing.assert_allclose(aligned.filtered_states, grounded_states)
    np.testing.assert_allclose(aligned.episode_start_xyz, raw_states[0, :3])


def test_grounded_langgap_rejects_missing_raw_dataset(tmp_path: Path) -> None:
    import json
    import pandas as pd

    dataset_dir = tmp_path / "skill_dataset"
    (dataset_dir / "meta" / "episodes").mkdir(parents=True)
    (dataset_dir / "data" / "chunk-000").mkdir(parents=True)
    pd.DataFrame(
        {
            "episode_index": [9],
            "data/chunk_index": [0],
            "data/file_index": [0],
            "length": [1],
            "tasks": [["task"]],
        }
    ).to_parquet(dataset_dir / "meta" / "episodes" / "chunk.parquet")
    pd.DataFrame(
        {"task_index": [0]}, index=pd.Index(["task"], name="task")
    ).to_parquet(dataset_dir / "meta" / "tasks.parquet")
    pd.DataFrame(
        {
            "episode_index": [9],
            "frame_index": [0],
            "action": [np.zeros(7, dtype=np.float32)],
            "observation.state": [np.zeros(8, dtype=np.float32)],
        }
    ).to_parquet(dataset_dir / "data" / "chunk-000" / "file-000.parquet")
    (dataset_dir / "meta" / "info.json").write_text(
        json.dumps({"proprio_grounding": "episode_start_xyz"})
    )
    latents_path = tmp_path / "skill_latents.npz"
    np.savez(
        latents_path,
        tokens=np.asarray([2]),
        episode_id=np.asarray([9]),
        skill_index=np.asarray([0]),
        frame_start=np.asarray([0]),
        frame_end=np.asarray([1]),
    )
    exact_path = tmp_path / "eval_init_states.npz"
    np.savez(
        exact_path,
        episode_index=np.asarray([9]),
        dataset_task_id=np.asarray([0]),
        init_states=np.asarray([[8.0, 9.0, 10.0]]),
        suite_name=np.asarray(["langgap_ext"]),
        suite_task_id=np.asarray([5]),
        init_state_index=np.asarray([17]),
    )
    dataset = skill_episode_data.SkillEvaluationDataset(
        skill_dataset_dir=dataset_dir,
        skill_latents_path=latents_path,
        eval_init_states_path=exact_path,
        original_dataset_dir=None,
        suite_name="langgap_ext",
    )

    with pytest.raises(RuntimeError, match="raw source dataset"):
        dataset.load_aligned_episode(9)

