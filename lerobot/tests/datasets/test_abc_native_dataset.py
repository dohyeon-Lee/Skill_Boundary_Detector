import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.abc_native_dataset import ABCNativeDataset
from lerobot.datasets.abc_native_io import (
    abc_native_episode_frame_table,
    abc_native_task_indices,
    iter_abc_native_state_action,
    load_abc_native_state_action,
)
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy
from lerobot.policies.pi05.configuration_pi05 import PI05Config


def _load_relative_stats_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/abc/configs/train_skills/DP/src/prepare_native_action_stats.py"
    )
    spec = importlib.util.spec_from_file_location("abc_native_relative_stats", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_native_dataset(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "abc_native"
    raw_root = tmp_path / "raw"
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    raw_root.mkdir()

    features = {
        "observation.state": {"dtype": "float32", "shape": [2], "names": ["s0", "s1"]},
        "action": {"dtype": "float32", "shape": [2], "names": ["a0", "a1"]},
        **{
            f"observation.images.{camera}": {
                "dtype": "video",
                "shape": [2, 4, 3],
                "names": ["height", "width", "channel"],
            }
            for camera in ("top", "left_wrist", "right_wrist")
        },
    }
    info = {
        "codebase_version": "v3.0",
        "storage_format": "sbd_abc_sim_native/v1",
        "robot_type": "yam",
        "total_episodes": 2,
        "total_frames": 8,
        "total_tasks": 2,
        "fps": 10,
        "splits": {"all": "0:2"},
        "features": features,
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "stats.json").write_text(json.dumps({}))
    pd.DataFrame(
        {"task_index": [0, 1]},
        index=pd.Index(["first prompt", "second prompt"], name="task"),
    ).to_parquet(root / "meta" / "tasks.parquet")

    layout = [
        {"source": "top", "camera": "top", "row_start": 0, "row_end": 2},
        {"source": "left", "camera": "left_wrist", "row_start": 2, "row_end": 4},
        {"source": "right", "camera": "right_wrist", "row_start": 4, "row_end": 6},
    ]
    rows = []
    for episode_index in range(2):
        episode_dir = raw_root / f"episode_{episode_index}"
        episode_dir.mkdir()
        values = np.arange(16, dtype="<f8").reshape(4, 4) + episode_index * 100
        values.tofile(episode_dir / "states_actions.bin")
        (episode_dir / "combined_camera-images-rgb.mp4").touch()
        rows.append(
            {
                "episode_index": episode_index,
                "length": 4,
                "dataset_from_index": episode_index * 4,
                "dataset_to_index": (episode_index + 1) * 4,
                "tasks": ["first prompt", "second prompt"],
                "task_indices": [0, 1],
                "source_task": f"synthetic_{episode_index}",
                "source_split": "train",
                "source_episode": f"episode_{episode_index}",
                "source_path": f"episode_{episode_index}",
                "states_actions_path": f"episode_{episode_index}/states_actions.bin",
                "video_path": f"episode_{episode_index}/combined_camera-images-rgb.mp4",
                "prompt_segments_json": json.dumps(
                    [{"frame": 0, "task_index": 0}, {"frame": 2, "task_index": 1}]
                ),
                "camera_layout_json": json.dumps(layout),
            }
        )
    pd.DataFrame(rows).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    (root / "native_dataset.json").write_text(
        json.dumps(
            {
                "format": "sbd_abc_sim_native/v1",
                "raw_root": "../raw",
                "fps": 10,
                "source_image_size": [4, 2],
                "image_preprocessing": "raw_camera_crop",
                "state_dim": 2,
                "action_dim": 2,
                "cameras": ["top", "left_wrist", "right_wrist"],
            }
        )
    )
    return root, raw_root


@pytest.fixture
def fake_video_decoder(monkeypatch):
    def decode(_path, timestamps, _tolerance, _backend, *, decoder_num_threads=None):
        del decoder_num_threads
        frames = []
        for timestamp in timestamps:
            frame = torch.empty(3, 6, 4)
            frame[:, 0:2] = 0.2 + timestamp
            frame[:, 2:4] = 0.5 + timestamp
            frame[:, 4:6] = 0.8 + timestamp
            frames.append(frame)
        return torch.stack(frames)

    monkeypatch.setattr("lerobot.datasets.abc_native_dataset.decode_video_frames", decode)


def test_complete_action_chunks_match_abc_sampling(tmp_path: Path, fake_video_decoder) -> None:
    root, _ = _write_native_dataset(tmp_path)
    dataset = ABCNativeDataset(
        "abc_native",
        root,
        delta_timestamps={"action": [0.0, 0.1, 0.2]},
    )

    # Each four-frame episode has only two complete three-action starts.
    assert len(dataset) == 4
    item = dataset[1]
    assert item["episode_index"].item() == 0
    assert item["frame_index"].item() == 1
    assert item["index"].item() == 1
    torch.testing.assert_close(
        item["action"],
        torch.tensor([[6.0, 7.0], [10.0, 11.0], [14.0, 15.0]]),
    )
    assert not item["action_is_pad"].any()
    assert item["observation.images.top"].shape == (3, 2, 4)
    torch.testing.assert_close(
        item["observation.images.top"],
        torch.full((3, 2, 4), 0.3),
    )


def test_native_episode_io_supports_sbd_without_frame_parquet(tmp_path: Path) -> None:
    root, _ = _write_native_dataset(tmp_path)
    episodes = pd.read_parquet(
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )

    states, actions = load_abc_native_state_action(root, episodes.iloc[0])
    assert states.shape == actions.shape == (4, 2)
    np.testing.assert_array_equal(states[0], [0.0, 1.0])
    np.testing.assert_array_equal(actions[0], [2.0, 3.0])

    selected = list(iter_abc_native_state_action(root, source_tasks=["synthetic_1"]))
    assert [episode for episode, _actions, _states in selected] == [1]
    np.testing.assert_array_equal(selected[0][1][0], [102.0, 103.0])

    table = abc_native_episode_frame_table(root, [0])
    assert table["episode_index"].unique().tolist() == [0]
    assert table["task_index"].tolist() == [0, 0, 1, 1]
    assert table["action"].iloc[0].shape == (2,)
    task_indices, n_episodes = abc_native_task_indices(root, ["synthetic_0"])
    assert task_indices == [0, 1]
    assert n_episodes == 1


def test_native_anchor_relative_stats_are_task_scoped_and_keep_excluded_dim_absolute(
    tmp_path: Path,
) -> None:
    root, _ = _write_native_dataset(tmp_path)
    module = _load_relative_stats_module()

    output = module.prepare(
        root,
        ["synthetic_0"],
        chunk_size=2,
        exclude_tokens=["a1"],
        action_mode="anchor_relative",
    )
    payload = json.loads(output.read_text())

    assert payload["tasks"] == ["synthetic_0"]
    assert payload["episode_count"] == 1
    assert payload["chunk_size"] == 2
    assert payload["mask"] == [True, False]
    assert payload["action"]["count"] == [7]
    np.testing.assert_allclose(payload["action"]["mean"], [26 / 7, 69 / 7])

    # The fingerprinted cache is reused instead of generating another file.
    assert module.prepare(root, ["synthetic_0"], 2, ["a1"], "anchor_relative") == output


def test_native_delta_joint_stats_use_consecutive_targets(tmp_path: Path) -> None:
    root, _ = _write_native_dataset(tmp_path)
    module = _load_relative_stats_module()

    output = module.prepare(
        root,
        ["synthetic_0"],
        chunk_size=2,
        exclude_tokens=["a1"],
        action_mode="delta_joint",
    )
    payload = json.loads(output.read_text())

    assert payload["action_mode"] == "delta_joint"
    assert payload["mask"] == [True, False]
    assert payload["action"]["count"] == [7]
    # Arm: four first-step (action-state)=2 and three consecutive deltas=4.
    # Excluded a1/gripper remains the original absolute action at every offset.
    np.testing.assert_allclose(payload["action"]["mean"], [20 / 7, 69 / 7])


def test_optional_lerobot_padding_and_mid_episode_prompt(tmp_path: Path, fake_video_decoder) -> None:
    root, _ = _write_native_dataset(tmp_path)
    dataset = ABCNativeDataset(
        "abc_native",
        root,
        episodes=[1],
        delta_timestamps={"action": [0.0, 0.1, 0.2]},
        video_keys_to_load=[],
        require_complete_action_chunks=False,
    )

    assert len(dataset) == 4
    item = dataset[3]
    assert item["episode_index"].item() == 1
    assert item["index"].item() == 7
    assert item["task_index"].item() == 1
    assert item["task"] == "second prompt"
    torch.testing.assert_close(item["action"], torch.tensor([[114.0, 115.0]] * 3))
    assert item["action_is_pad"].tolist() == [False, True, True]
    assert not any(key.startswith("observation.images") for key in item)


def test_source_task_filter_selects_whole_episodes(
    tmp_path: Path, fake_video_decoder, monkeypatch
) -> None:
    root, _ = _write_native_dataset(tmp_path)
    dataset = ABCNativeDataset(
        "abc_native",
        root,
        tasks=["synthetic_1"],
        delta_timestamps={"action": [0.0, 0.1, 0.2]},
        video_keys_to_load=[],
    )

    assert dataset.tasks == ["synthetic_1"]
    assert dataset.episodes == [1]
    assert dataset.num_episodes == 1
    assert len(dataset) == 2
    assert dataset[0]["episode_index"].item() == 1
    np.testing.assert_allclose(dataset.meta.stats["observation.state"]["mean"], [106.0, 107.0])
    np.testing.assert_allclose(dataset.meta.stats["action"]["mean"], [108.0, 109.0])
    values = np.arange(16, dtype=np.float32).reshape(4, 4) + 100
    np.testing.assert_allclose(
        dataset.meta.stats["observation.state"]["q01"],
        np.quantile(values[:, :2], 0.01, axis=0),
    )
    np.testing.assert_allclose(
        dataset.meta.stats["action"]["q99"],
        np.quantile(values[:, 2:], 0.99, axis=0),
    )
    assert dataset.selection_stats_path is not None
    assert dataset.selection_stats_path.is_file()
    cache_payload = json.loads(dataset.selection_stats_path.read_text())
    assert cache_payload["tasks"] == ["synthetic_1"]
    assert cache_payload["episode_count"] == 1
    assert cache_payload["frame_count"] == 4

    def fail_if_recomputed(_self):
        raise AssertionError("selection stats cache was not reused")

    monkeypatch.setattr(ABCNativeDataset, "_compute_selection_stats", fail_if_recomputed)
    cached = ABCNativeDataset(
        "abc_native",
        root,
        tasks=["synthetic_1"],
        video_keys_to_load=[],
    )
    np.testing.assert_allclose(cached.meta.stats["action"]["mean"], [108.0, 109.0])

    with pytest.raises(ValueError, match="Unknown ABC native source task"):
        ABCNativeDataset("abc_native", root, tasks=["missing"], video_keys_to_load=[])

    with pytest.raises(ValueError, match="no episodes in common"):
        ABCNativeDataset(
            "abc_native", root, episodes=[0], tasks=["synthetic_1"], video_keys_to_load=[]
        )


def test_source_paths_cannot_escape_raw_root(tmp_path: Path, fake_video_decoder) -> None:
    root, _ = _write_native_dataset(tmp_path)
    episodes_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes = pd.read_parquet(episodes_path)
    episodes.loc[0, "states_actions_path"] = "../outside.bin"
    episodes.to_parquet(episodes_path)
    dataset = ABCNativeDataset("abc_native", root, episodes=[0], video_keys_to_load=[])

    with pytest.raises(ValueError, match="escapes raw_root"):
        dataset[0]


def test_factory_selects_native_reader_and_pi05_horizon(tmp_path: Path, fake_video_decoder) -> None:
    root, _ = _write_native_dataset(tmp_path)
    config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="abc_native", root=str(root), episodes=[0]),
        policy=PI05Config(chunk_size=3, n_action_steps=3, device="cpu"),
    )

    dataset = make_dataset(config)

    assert isinstance(dataset, ABCNativeDataset)
    assert dataset.delta_indices == {"action": [0, 1, 2]}
    assert len(dataset) == 2
    assert dataset[0]["action"].shape == (3, 2)
    for camera in dataset.meta.camera_keys:
        assert set(dataset.meta.stats[camera]) == {"mean", "std"}


def test_factory_forwards_abc_source_task_filter(tmp_path: Path, fake_video_decoder) -> None:
    root, _ = _write_native_dataset(tmp_path)
    config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="abc_native", root=str(root), tasks=["synthetic_1"]),
        policy=PI05Config(chunk_size=3, n_action_steps=3, device="cpu"),
    )

    dataset = make_dataset(config)

    assert isinstance(dataset, ABCNativeDataset)
    assert dataset.tasks == ["synthetic_1"]
    assert dataset.episodes == [1]


def test_factory_rejects_libero_xyz_grounding_for_abc(tmp_path: Path) -> None:
    root, _ = _write_native_dataset(tmp_path)
    config = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id="abc_native",
            root=str(root),
            episodes=[0],
            use_imagenet_stats=False,
        ),
        policy=PI05Config(
            chunk_size=3,
            n_action_steps=3,
            device="cpu",
            proprio_grounding="episode_start_xyz",
        ),
    )

    with pytest.raises(ValueError, match="joint positions, not EEF XYZ"):
        make_dataset(config)


def test_policy_feature_schema_uses_renamed_camera_keys(tmp_path: Path, monkeypatch) -> None:
    root, _ = _write_native_dataset(tmp_path)
    metadata = LeRobotDatasetMetadata("abc_native", root=root)
    rename_map = {
        "observation.images.top": "observation.images.base_0_rgb",
        "observation.images.left_wrist": "observation.images.left_wrist_0_rgb",
        "observation.images.right_wrist": "observation.images.right_wrist_0_rgb",
    }

    class StubPolicy(torch.nn.Module):
        def __init__(self, config, **_kwargs):
            super().__init__()
            self.config = config

    monkeypatch.setattr("lerobot.policies.factory.get_policy_class", lambda _type: StubPolicy)
    config = PI05Config(device="cpu")
    policy = make_policy(config, ds_meta=metadata, rename_map=rename_map)

    assert set(policy.config.input_features) == {
        "observation.state",
        "observation.images.base_0_rgb",
        "observation.images.left_wrist_0_rgb",
        "observation.images.right_wrist_0_rgb",
    }
    assert "observation.images.top" not in policy.config.input_features
