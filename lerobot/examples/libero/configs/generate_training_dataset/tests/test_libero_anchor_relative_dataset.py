import sys
from pathlib import Path

import numpy as np
import torch

from lerobot.processor.eef_relative_action_processor import (
    osc_actions_to_absolute_eef,
    to_eef_relative_actions,
)

WORKFLOW_SRC = Path(__file__).resolve().parent.parent / "libero_anchor_relative_dataset" / "src"
sys.path.insert(0, str(WORKFLOW_SRC))

from build_libero_anchor_relative_dataset import (  # noqa: E402
    aggregate_shards,
    balanced_contiguous_shard_bounds,
    build_one,
    checkpoint_output_path,
    derive_aligned_absolute_targets,
    resolve_builder_vcodec,
    safe_output_path,
)
from compute_relative_action_stats import compute_relative_stats  # noqa: E402


def test_builder_shifts_action_to_its_pre_action_observation():
    states = np.array(
        [
            [0.30, 0.10, 1.00, 0.10, -0.20, 2.90, 0.01, -0.01],
            [0.31, 0.08, 1.01, 0.12, -0.18, 2.92, 0.01, -0.01],
            [0.32, 0.06, 1.02, 0.14, -0.16, 2.94, 0.01, -0.01],
        ],
        dtype=np.float32,
    )
    actions = np.array(
        [
            [-0.9, -0.9, -0.9, -0.9, -0.9, -0.9, 1.0],
            [0.4, -0.5, 0.2, 0.2, -0.3, 0.1, -1.0],
            [0.1, 0.2, -0.3, -0.2, 0.1, 0.4, 1.0],
        ],
        dtype=np.float32,
    )

    kept_states, targets = derive_aligned_absolute_targets(
        states,
        actions,
        position_scale=0.05,
        rotation_scale=0.5,
    )
    expected = osc_actions_to_absolute_eef(
        torch.from_numpy(actions[1:]),
        torch.from_numpy(states[:-1]),
    ).numpy()
    assert np.array_equal(kept_states, states[:-1])
    assert np.allclose(targets, expected, atol=2e-6)
    assert not np.allclose(targets[0, :3], states[0, :3] + 0.05 * actions[0, :3])


def test_relative_stats_pool_each_future_target_against_one_anchor():
    states = np.array(
        [
            [0.30, 0.10, 1.00, 0.10, -0.20, 2.90, 0.0, 0.0],
            [0.31, 0.08, 1.01, 0.12, -0.18, 2.92, 0.0, 0.0],
            [0.32, 0.06, 1.02, 0.14, -0.16, 2.94, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    osc = torch.tensor(
        [
            [0.4, -0.5, 0.2, 0.2, -0.3, 0.1, -1.0],
            [0.1, 0.2, -0.3, -0.2, 0.1, 0.4, 1.0],
            [-0.2, 0.1, 0.1, 0.3, 0.2, -0.1, -1.0],
        ],
        dtype=torch.float64,
    )
    actions = osc_actions_to_absolute_eef(osc, torch.from_numpy(states)).numpy()
    episode_index = np.zeros(3, dtype=np.int64)

    stats, stride, count = compute_relative_stats(states, actions, episode_index, chunk_size=2)
    offset_zero = to_eef_relative_actions(torch.from_numpy(actions), torch.from_numpy(states)).numpy()
    offset_one = to_eef_relative_actions(torch.from_numpy(actions[1:]), torch.from_numpy(states[:-1])).numpy()
    expected = np.concatenate([offset_zero, offset_one], axis=0)
    assert stride == 1
    assert count == 5
    assert np.allclose(stats["mean"], expected.mean(axis=0), atol=1e-10)
    assert np.allclose(stats["min"], expected.min(axis=0), atol=1e-10)
    assert np.allclose(stats["max"], expected.max(axis=0), atol=1e-10)


def test_overwrite_target_cannot_alias_source_or_escape_root(tmp_path):
    with np.testing.assert_raises_regex(ValueError, "one folder name"):
        safe_output_path(tmp_path, "../source")
    with np.testing.assert_raises_regex(ValueError, "must be separate"):
        build_one(
            source=tmp_path,
            output=tmp_path,
            output_name="same",
            cfg={},
            spec={},
            overwrite=True,
            max_episodes=1,
        )


def test_balanced_shards_are_contiguous_nonempty_and_cover_all_episodes():
    lengths = [101, 11, 101, 11, 51]
    bounds = balanced_contiguous_shard_bounds(lengths, 3)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == len(lengths)
    assert all(left < right for left, right in bounds)
    assert all(bounds[index][1] == bounds[index + 1][0] for index in range(len(bounds) - 1))
    frame_totals = [sum(length - 1 for length in lengths[left:right]) for left, right in bounds]
    assert max(frame_totals) <= 2 * min(frame_totals)


def test_portable_h264_uses_nvenc_only_after_real_probe(monkeypatch):
    monkeypatch.setattr(
        "build_libero_anchor_relative_dataset.probe_h264_nvenc",
        lambda: (True, "probe succeeded"),
    )
    assert resolve_builder_vcodec("portable_h264") == ("h264_nvenc", "probe succeeded")

    monkeypatch.setattr(
        "build_libero_anchor_relative_dataset.probe_h264_nvenc",
        lambda: (False, "no compatible GPU"),
    )
    assert resolve_builder_vcodec("portable_h264") == ("h264", "no compatible GPU")


def test_checkpoint_path_is_confined_to_private_intermediate_root(tmp_path):
    path = checkpoint_output_path(tmp_path, "libero_rel", 2, 7)
    assert path == (
        tmp_path
        / "_libero_anchor_relative_checkpoints"
        / "libero_rel"
        / "array-002"
        / "checkpoint-0007"
    )
    with np.testing.assert_raises_regex(ValueError, "one folder name"):
        checkpoint_output_path(tmp_path, "../escape", 0, 0)


def test_incomplete_final_is_preserved_until_all_replacement_shards_exist(tmp_path):
    output = tmp_path / "libero_rel"
    output.mkdir()
    sentinel = output / "partial-data"
    sentinel.write_text("keep until replacement is ready")

    with np.testing.assert_raises_regex(FileNotFoundError, "Shard 0/1 is not complete"):
        aggregate_shards(
            root=tmp_path,
            output_name="libero_rel",
            cfg={"convert_replace_incomplete_output": True},
            spec={},
            num_shards=1,
            overwrite=False,
            skip_stats=False,
        )

    assert sentinel.read_text() == "keep until replacement is ready"
