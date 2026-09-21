from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from datasets import Dataset

from lerobot.datasets.compute_stats import get_feature_stats
from lerobot.datasets.proprio_grounding import (
    EpisodeStartXYZGroundedDataset,
    ground_state_xyz,
)
from lerobot.utils.constants import OBS_STATE


class _StubDataset:
    def __init__(self, root, episode_states):
        self.root = root
        rows = []
        episode_rows = []
        all_states = []
        self.items = []
        offset = 0
        for episode_id, states in enumerate(episode_states):
            states = np.asarray(states, dtype=np.float32)
            all_states.append(states)
            for frame_id, state in enumerate(states):
                rows.append(
                    {
                        "episode_index": episode_id,
                        "frame_index": frame_id,
                        OBS_STATE: state,
                    }
                )
            stats = get_feature_stats(states, axis=0, keepdims=False)
            episode_rows.append(
                {
                    "episode_index": episode_id,
                    "dataset_from_index": offset,
                    "dataset_to_index": offset + len(states),
                    **{
                        f"stats/{OBS_STATE}/{name}": np.asarray(value, dtype=np.float64)
                        for name, value in stats.items()
                    },
                }
            )
            # Mimic a delta-timestamp state window returned by LeRobotDataset.
            self.items.append(
                {
                    "episode_index": torch.tensor(episode_id),
                    OBS_STATE: torch.from_numpy(states.copy()),
                }
            )
            offset += len(states)

        data_path = root / "data" / "chunk-000"
        data_path.mkdir(parents=True)
        pd.DataFrame(rows).to_parquet(data_path / "file-000.parquet", index=False)

        metadata_path = root / "meta" / "episodes" / "chunk-000"
        metadata_path.mkdir(parents=True)
        pd.DataFrame(episode_rows).to_parquet(
            metadata_path / "file-000.parquet", index=False
        )

        all_states = np.concatenate(all_states, axis=0)
        # Match LeRobotDatasetMetadata.episodes: a Hugging Face Dataset whose
        # load path deliberately strips all stats/* columns.
        runtime_episode_rows = [
            {
                "episode_index": row["episode_index"],
                "dataset_from_index": row["dataset_from_index"],
                "dataset_to_index": row["dataset_to_index"],
            }
            for row in episode_rows
        ]
        self.meta = SimpleNamespace(
            total_frames=len(all_states),
            total_episodes=len(episode_states),
            episodes=Dataset.from_list(runtime_episode_rows),
            stats={OBS_STATE: get_feature_stats(all_states, axis=0, keepdims=False)},
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def get_raw_item(self, index):
        return self.items[index]


def test_ground_state_xyz_changes_only_the_first_three_coordinates():
    states = np.array([[10.0, 20.0, 30.0, 4.0], [11.0, 18.0, 35.0, 5.0]])

    grounded = ground_state_xyz(states, np.array([10.0, 20.0, 30.0]))

    np.testing.assert_allclose(grounded, [[0.0, 0.0, 0.0, 4.0], [1.0, -2.0, 5.0, 5.0]])
    np.testing.assert_allclose(states[0], [10.0, 20.0, 30.0, 4.0])


def test_episode_grounded_dataset_uses_each_episode_reference_and_grounded_stats(tmp_path):
    episodes = [
        [[10.0, 20.0, 30.0, 4.0], [11.0, 18.0, 35.0, 5.0]],
        [[100.0, 200.0, 300.0, 6.0], [99.0, 202.0, 297.0, 7.0]],
    ]
    dataset = _StubDataset(tmp_path, episodes)
    raw_stats = dataset.meta.stats
    assert not hasattr(dataset.meta.episodes, "columns")
    assert all(not name.startswith("stats/") for name in dataset.meta.episodes.column_names)

    grounded_dataset = EpisodeStartXYZGroundedDataset(dataset)

    expected = np.array(
        [
            [0.0, 0.0, 0.0, 4.0],
            [1.0, -2.0, 5.0, 5.0],
            [0.0, 0.0, 0.0, 6.0],
            [-1.0, 2.0, -3.0, 7.0],
        ],
        dtype=np.float32,
    )
    expected_stats = get_feature_stats(expected, axis=0, keepdims=False)

    torch.testing.assert_close(
        grounded_dataset[0][OBS_STATE], torch.from_numpy(expected[:2])
    )
    torch.testing.assert_close(
        grounded_dataset[1][OBS_STATE], torch.from_numpy(expected[2:])
    )
    for name in ("min", "max", "mean", "std", "count"):
        np.testing.assert_allclose(
            grounded_dataset.meta.stats[OBS_STATE][name],
            expected_stats[name],
            rtol=1e-6,
            atol=1e-6,
        )

    assert dataset.meta.stats is raw_stats
    assert (tmp_path / "meta" / "episode_start_xyz_grounding_v2.npz").is_file()


def test_grounded_stats_match_the_dataset_generation_contract(tmp_path):
    """Grounded stats must equal generate_training_dataset's GLOBAL exact stats of the grounded frames."""
    import importlib.util
    import json
    import sys
    from pathlib import Path

    script_dir = (
        Path(__file__).resolve().parents[2]
        / "examples/libero/configs/generate_training_dataset/download_dataset/filtered_dataset/src"
    )
    sys.path.insert(0, str(script_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "ensure_quantile_stats", script_dir / "ensure_quantile_stats.py"
        )
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
    finally:
        sys.path.remove(str(script_dir))

    rng = np.random.default_rng(0)
    # Short, differently-offset episodes: exactly the case where averaging per-episode
    # quantiles would narrow q01/q99.
    episodes = [
        (rng.normal(size=(length, 8)) + rng.normal(scale=5.0, size=(1, 8))).astype(np.float32)
        for length in (7, 23, 11, 40, 5)
    ]
    dataset = _StubDataset(tmp_path / "raw", episodes)
    grounded_stats = EpisodeStartXYZGroundedDataset(dataset).meta.stats[OBS_STATE]

    # Materialize the grounded frames as a dataset and run the generation script on it.
    grounded_root = tmp_path / "grounded"
    (grounded_root / "data").mkdir(parents=True)
    (grounded_root / "meta").mkdir()
    rows = []
    for states in episodes:
        states = states.astype(np.float64)
        states[:, :3] -= states[0, :3].astype(np.float32).astype(np.float64)
        rows.extend({OBS_STATE: row} for row in states)
    pd.DataFrame(rows).to_parquet(grounded_root / "data" / "file-000.parquet", index=False)
    reference.fast_recompute(grounded_root, video_keys=set())
    expected = json.loads((grounded_root / "meta" / "stats.json").read_text())[OBS_STATE]

    assert set(grounded_stats) == set(expected)
    for name, value in expected.items():
        np.testing.assert_allclose(grounded_stats[name], value, rtol=0, atol=1e-12, err_msg=name)

    # The rejected alternative (count-weighted mean of per-episode quantiles) is visibly narrower.
    per_episode_q99 = np.average(
        [np.quantile(e[:, 3], 0.99) for e in episodes], weights=[len(e) for e in episodes]
    )
    assert per_episode_q99 < grounded_stats["q99"][3] - 0.5


def test_single_state_items_are_grounded_like_state_windows(tmp_path):
    """pi05 reads ONE current state per item (no delta window), unlike Diffusion's history."""
    dataset = _StubDataset(tmp_path, [[[10.0, 20.0, 30.0, 4.0], [11.0, 18.0, 35.0, 5.0]]])
    dataset.items = [
        {"episode_index": torch.tensor(0), OBS_STATE: torch.tensor([11.0, 18.0, 35.0, 5.0])}
    ]
    grounded = EpisodeStartXYZGroundedDataset(dataset)
    torch.testing.assert_close(grounded[0][OBS_STATE], torch.tensor([1.0, -2.0, 5.0, 5.0]))


def test_state_history_cache_is_skipped_without_a_state_delta_window():
    """Regression: pi05 has no observation.state delta window; caching it must not be attempted."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / "src/lerobot/datasets/factory.py").read_text()
    guard = 'if OBS_STATE in (getattr(dataset, "delta_timestamps", None) or {}):'
    # The grounding branch's cache call sits directly under the guard.
    guarded = source[source.index(guard):]
    assert guarded.splitlines()[1].strip() == "dataset.cache_delta_columns([OBS_STATE])"
