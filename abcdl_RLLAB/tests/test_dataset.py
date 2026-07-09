import shutil

import numpy as np
import pytest
import torch

from abcdl.dataset import AbcdlDataset

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _make_root(tmp_path, tmp_abcdl_episode):
    # two episodes under a root
    root = tmp_path / "ds"
    root.mkdir()
    for k in range(2):
        shutil.copytree(tmp_abcdl_episode, root / f"episode_{k:04d}")
    return str(root)


def test_dataset_len_and_item(tmp_path, tmp_abcdl_episode):
    root = _make_root(tmp_path, tmp_abcdl_episode)
    ds = AbcdlDataset(root)
    assert ds.num_episodes == 2
    assert ds.num_frames == len(ds) == 12  # 6 frames x 2
    item = ds[0]
    assert item["observation.state"].shape == (14,)
    assert item["action"].shape == (14,)
    assert item["observation.images.top"].shape[0] == 3  # CHW
    assert 0.0 <= float(item["observation.images.top"].max()) <= 1.0
    assert item["task"] == "demo task"
    assert int(item["episode_index"]) in (0, 1)


def test_dataset_action_chunk(tmp_path, tmp_abcdl_episode):
    root = _make_root(tmp_path, tmp_abcdl_episode)
    ds = AbcdlDataset(root, delta_timestamps={"action": [0.0, 1 / 30, 2 / 30]})
    item = ds[0]
    assert item["action"].shape == (3, 14)


def test_meta_features(tmp_path, tmp_abcdl_episode):
    root = _make_root(tmp_path, tmp_abcdl_episode)
    ds = AbcdlDataset(root)
    assert ds.meta.fps == 30.0
    assert set(ds.meta.camera_keys) == {"top", "left_wrist"}
    assert "observation.state" in ds.meta.features
    assert ds.meta.features["observation.state"]["shape"] == (14,)


def test_openpi_surface(tmp_path, tmp_abcdl_episode):
    """openpi compatibility: tasks dict, task_index, episode subset, flexible
    delta_timestamps keyed on the action_sequence_key (e.g. 'actions')."""
    root = _make_root(tmp_path, tmp_abcdl_episode)

    ds = AbcdlDataset(root)
    # tasks is a {task_index: task_string} dict (openpi _lerobot_tasks_to_dict form)
    assert isinstance(ds.meta.tasks, dict)
    assert set(ds.meta.tasks.values()) == {"demo task"}
    item = ds[0]
    assert int(item["task_index"]) == 0 and item["task"] == "demo task"

    # episode subset (like LeRobotDataset(episodes=...))
    one = AbcdlDataset(root, episodes=[0])
    assert one.num_episodes == 1 and len(one) == 6

    # openpi requests delta on action_sequence_keys (default "actions")
    ds2 = AbcdlDataset(root, delta_timestamps={"actions": [0.0, 1 / 30, 2 / 30]})
    assert ds2[0]["actions"].shape == (3, 14)


def test_streaming_from_hub():
    """Stream an abcdl dataset from the Hub via HTTP range reads (no full download).

    Network-gated: skips when offline / the public demo repo is unreachable.
    """
    try:
        ds = AbcdlDataset("jellyho/abcdl_demo", fmt="abcdl_224", streaming=True)
        item = ds[ds.num_frames // 2]
    except Exception as e:  # noqa: BLE001 - any network/HF error -> skip, not fail
        pytest.skip(f"streaming demo repo unreachable: {type(e).__name__}: {e}")
    assert ds.num_episodes > 0 and len(ds) > 0
    assert item["observation.state"].ndim == 1
    assert item["observation.images." + ds.meta.camera_keys[0]].shape[0] == 3  # CHW


def test_dataloader_multiworker_fork_safe(tmp_path, tmp_abcdl_episode):
    """torchcodec decoders are not fork-safe; the dataset must reset its decoder
    cache per worker process so a num_workers>0 DataLoader does not crash."""
    from torch.utils.data import DataLoader

    root = _make_root(tmp_path, tmp_abcdl_episode)
    ds = AbcdlDataset(root)
    _ = ds[0]  # populate the parent-process decoder cache before forking workers
    dl = DataLoader(ds, batch_size=4, num_workers=2, shuffle=True)
    seen = 0
    for batch in dl:
        assert batch["observation.state"].shape == (min(4, len(ds)), 14)
        assert batch["observation.images.top"].shape[1] == 3  # (B, C, H, W)
        seen += batch["observation.state"].shape[0]
    assert seen == len(ds)
