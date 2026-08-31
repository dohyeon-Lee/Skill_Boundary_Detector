from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest


CALVIN_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/generate_training_dataset/download_dataset/calvin_dataset/src"
)
sys.path.insert(0, str(CALVIN_SRC))

from calvin_dataset_config import (  # noqa: E402
    calvin_raw_root,
    conversion_settings,
    load_config,
    selected_variants,
    variants,
)
from convert_calvin_to_lerobot import (  # noqa: E402
    _copy_source_tree,
    conversion_units,
    make_features,
    policy_action,
    policy_state,
    subtract_intervals,
)
from download_calvin import (  # noqa: E402
    MARKER_NAME,
    _is_complete,
    _validate_zip_members,
    download_variant,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_archive(path: Path, extracted_dir: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr(f"{extracted_dir}/training/episode_0000000.npz", b"train")
        bundle.writestr(
            f"{extracted_dir}/training/lang_annotations/auto_lang_ann.npy",
            b"annotations",
        )
        bundle.writestr(f"{extracted_dir}/validation/episode_0000001.npz", b"validation")


def test_default_config_exposes_all_official_variants_and_pins_raw_root() -> None:
    config = load_config()
    configured = variants(config)

    assert list(configured) == ["debug", "D", "ABC", "ABCD"]
    assert selected_variants(config) == ["D"]
    assert calvin_raw_root(config) == (
        Path(config["project_root"]) / config["calvin_dataset_root"] / "_calvin_raw"
    ).resolve()
    assert configured["D"]["archive"] == "task_D_D.zip"
    assert configured["D"]["sha256"] == (
        "45efc2fb24a09a50ab3ed6cdc7637604ee857d3ba1bab23d63925c2d71e79d4f"
    )


def test_local_verified_archive_extracts_and_then_skips(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    extracted_dir = "fake_calvin"
    raw_root = tmp_path / "data" / "raw"
    archive = raw_root / "archives" / "fake.zip"
    _fake_archive(archive, extracted_dir)
    spec = {
        "archive": archive.name,
        "extracted_dir": extracted_dir,
        "url": archive.as_uri(),
        "sha256": _sha256(archive),
        "approximate_size_gb": 0,
    }
    config = {
        "project_root": str(tmp_path),
        "dataset_root": "data",
        "calvin_raw_subdir": "raw",
        "calvin_keep_archives": True,
    }

    download_variant(config, "fake", spec)
    extracted_root = raw_root / extracted_dir
    assert (extracted_root / MARKER_NAME).is_file()
    assert _is_complete(extracted_root, "fake", spec)

    download_variant(config, "fake", spec)
    assert "already downloaded, verified, and extracted; skipping" in capsys.readouterr().out


def test_zip_path_traversal_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("fake_calvin/training/episode_0000000.npz", b"train")
        bundle.writestr("../outside.txt", b"unsafe")

    with pytest.raises(RuntimeError, match="unsafe path"):
        _validate_zip_members(archive, "fake_calvin")


def test_completion_marker_is_invalidated_by_changed_checksum(tmp_path: Path) -> None:
    extracted_dir = "fake_calvin"
    raw_root = tmp_path / "data" / "raw"
    archive = raw_root / "archives" / "fake.zip"
    _fake_archive(archive, extracted_dir)
    spec = {
        "archive": archive.name,
        "extracted_dir": extracted_dir,
        "url": archive.as_uri(),
        "sha256": _sha256(archive),
        "approximate_size_gb": 0,
    }
    config = {
        "project_root": str(tmp_path),
        "dataset_root": "data",
        "calvin_raw_subdir": "raw",
        "calvin_keep_archives": True,
    }
    download_variant(config, "fake", spec)

    changed_spec = {**spec, "sha256": "0" * 64}
    assert not _is_complete(raw_root / extracted_dir, "fake", changed_spec)


def test_default_conversion_uses_relative_action_full_robot_state_and_global_output() -> None:
    config = load_config()

    settings = conversion_settings(config)

    assert "calvin_convert_output_root" not in config
    assert "calvin_convert_output_name" not in config
    assert settings["calvin_policy_action"] == "relative"
    assert settings["calvin_policy_state"] == "robot_obs"
    assert settings["calvin_convert_mode"] == "annotated"
    assert settings["calvin_task_split"] == "all"
    assert settings["calvin_heldout_tasks"] == []
    assert settings["calvin_convert_source_dir"] == (
        Path(config["project_root"])
        / config["calvin_dataset_root"]
        / "_calvin_raw"
        / "calvin_debug_dataset"
        / "training"
    ).resolve()
    assert settings["calvin_convert_output_dir"] == (
        Path(config["project_root"]) / config["dataset_root"] / "calvin_debug_full_full"
    ).resolve()


def test_validation_output_name_is_automatic_and_does_not_collide() -> None:
    config = load_config()
    config["calvin_convert_split"] = "validation"

    settings = conversion_settings(config)

    assert settings["calvin_convert_output_name"] == "calvin_debug_validation_full_full"
    assert settings["calvin_convert_output_dir"] == (
        Path(config["project_root"])
        / config["dataset_root"]
        / "calvin_debug_validation_full_full"
    ).resolve()


@pytest.mark.parametrize(
    ("mode", "task_split", "expected_name"),
    [
        ("annotated", "all", "calvin_debug_full_full"),
        ("annotated", "pretrain", "calvin_debug_pretrain_full_full"),
        ("annotated", "heldout", "calvin_debug_heldout_full_full"),
        ("play", "all", "calvin_debug_play_full_full"),
        ("play", "pretrain", "calvin_debug_play_pretrain_full_full"),
    ],
)
def test_conversion_mode_and_task_split_determine_output_name(
    mode: str, task_split: str, expected_name: str
) -> None:
    config = load_config()
    config["calvin_convert_mode"] = mode
    config["calvin_task_split"] = task_split
    config["calvin_heldout_tasks"] = ["task_b"] if task_split != "all" else []

    settings = conversion_settings(config)

    assert settings["calvin_convert_output_name"] == expected_name


def test_task_split_rejects_missing_heldout_tasks_and_play_heldout() -> None:
    config = load_config()
    config["calvin_task_split"] = "pretrain"
    config["calvin_heldout_tasks"] = []
    with pytest.raises(ValueError, match="requires non-empty"):
        conversion_settings(config)

    config["calvin_convert_mode"] = "play"
    config["calvin_task_split"] = "heldout"
    config["calvin_heldout_tasks"] = ["task_b"]
    with pytest.raises(ValueError, match="does not support"):
        conversion_settings(config)


def test_play_pretrain_removes_exact_heldout_intervals_without_bridging() -> None:
    annotation = {
        "annotations": np.asarray(["train", "held out", "held out overlap"]),
        "task_ids": np.asarray(["task_a", "task_b", "task_b"]),
        "embeddings": np.zeros((3, 4), dtype=np.float32),
        "intervals": np.asarray([[2, 3], [5, 7], [7, 8]], dtype=np.int64),
    }

    units, removed = conversion_units(
        annotation,
        recordings=[(0, 10), (20, 22)],
        mode="play",
        task_split="pretrain",
        heldout_tasks=["task_b"],
    )

    assert removed == [(5, 8)]
    assert [(unit["start"], unit["end"]) for unit in units] == [
        (0, 4),
        (9, 10),
        (20, 22),
    ]
    assert all(unit["task_id"] == "play" and unit["language"] == "" for unit in units)


def test_annotated_task_split_is_task_disjoint() -> None:
    annotation = {
        "annotations": np.asarray(["train a", "held out", "train c"]),
        "task_ids": np.asarray(["task_a", "task_b", "task_c"]),
        "embeddings": np.zeros((3, 4), dtype=np.float32),
        "intervals": np.asarray([[0, 1], [2, 3], [4, 5]], dtype=np.int64),
    }

    pretrain, _ = conversion_units(
        annotation, [(0, 5)], "annotated", "pretrain", ["task_b"]
    )
    heldout, _ = conversion_units(
        annotation, [(0, 5)], "annotated", "heldout", ["task_b"]
    )

    assert [unit["task_id"] for unit in pretrain] == ["task_a", "task_c"]
    assert [unit["task_id"] for unit in heldout] == ["task_b"]


def test_interval_subtraction_clips_to_each_recording() -> None:
    assert subtract_intervals([(10, 19), (30, 39)], [(5, 12), (17, 32)]) == [
        (0, 13, 16, 10, 19),
        (1, 33, 39, 30, 39),
    ]


def test_policy_action_and_state_presets_keep_reprojection_sources() -> None:
    actions = np.arange(7, dtype=np.float64)
    rel_actions = -actions
    robot_obs = np.arange(15, dtype=np.float64)
    timestep = {"actions": actions, "rel_actions": rel_actions}

    np.testing.assert_array_equal(policy_action(timestep, "relative"), rel_actions)
    np.testing.assert_array_equal(policy_action(timestep, "absolute"), actions)
    np.testing.assert_array_equal(policy_state(robot_obs, "robot_obs"), robot_obs)
    np.testing.assert_array_equal(
        policy_state(robot_obs, "tcp_pose_gripper"), robot_obs[[0, 1, 2, 3, 4, 5, 6, 14]]
    )
    np.testing.assert_array_equal(policy_state(robot_obs, "joint_gripper"), robot_obs[6:15])


def test_converter_features_expose_only_canonical_policy_inputs() -> None:
    from lerobot.datasets.feature_utils import dataset_to_policy_features

    timestep = {
        "rgb_static": np.zeros((200, 200, 3), dtype=np.uint8),
        "rgb_gripper": np.zeros((84, 84, 3), dtype=np.uint8),
        "actions": np.zeros(7, dtype=np.float64),
        "rel_actions": np.zeros(7, dtype=np.float64),
        "robot_obs": np.zeros(15, dtype=np.float64),
        "scene_obs": np.zeros(24, dtype=np.float64),
    }

    features = make_features(timestep, 224, "relative", "robot_obs", 384)
    policy_features = dataset_to_policy_features(features)

    assert set(policy_features) == {
        "observation.images.image",
        "observation.images.wrist_image",
        "observation.state",
        "action",
    }
    assert features["observation.state"]["shape"] == (15,)
    assert features["action"]["shape"] == (7,)
    assert {"calvin.actions", "calvin.rel_actions", "calvin.robot_obs"} <= features.keys()


def test_hardlink_preservation_keeps_exact_raw_files(tmp_path: Path) -> None:
    source = tmp_path / "source" / "training"
    source.mkdir(parents=True)
    raw_file = source / "episode_0000000.npz"
    raw_file.write_bytes(b"exact raw timestep")
    output = tmp_path / "converted"
    output.mkdir()

    result = _copy_source_tree(source, output, "hardlink")
    retained = output / "calvin_source" / "training" / raw_file.name

    assert retained.read_bytes() == raw_file.read_bytes()
    assert retained.stat().st_ino == raw_file.stat().st_ino
    assert result["linked"] == 1
    assert result["copied"] == 0
