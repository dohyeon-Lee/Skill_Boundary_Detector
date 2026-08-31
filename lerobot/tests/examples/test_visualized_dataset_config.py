from __future__ import annotations

import sys
from pathlib import Path

import pytest


CONFIG_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/generate_training_dataset/visualized_dataset/src"
)
VISUALIZER_DIR = CONFIG_SRC.parent
sys.path.insert(0, str(CONFIG_SRC))
sys.path.insert(0, str(VISUALIZER_DIR))

from visualized_dataset_config import (  # noqa: E402
    VISUALIZED_DIR,
    dataset_settings,
    load_config,
    reject_cli_arguments,
    visualization_settings,
)
from visualize_training_dataset import select_samples  # noqa: E402


def test_default_yaml_is_shared_and_uses_global_roots() -> None:
    config = load_config()
    dataset = dataset_settings(config)
    visualization = visualization_settings(config)

    assert dataset.dataset == config["dataset"]
    assert dataset.dataset_root == Path(config["project_root"]) / config["dataset_root"]
    assert visualization.task == str(config["visualize"]["task"])
    assert visualization.samples == int(config["visualize"]["samples"])


def test_relative_root_and_output_are_resolved_from_documented_bases(tmp_path: Path) -> None:
    config = {
        "project_root": str(tmp_path),
        "dataset_root": "global_data",
        "dataset_root_override": "local_data",
        "dataset": "demo",
        "visualize": {
            "task": 2,
            "samples": 3,
            "sampling": "random",
            "seed": 7,
            "cameras": ["observation.images.image"],
            "output": "reports/demo.html",
            "list_tasks_only": False,
            "force": False,
            "crf": 20,
            "preset": "fast",
            "ffmpeg": "ffmpeg",
        },
    }

    dataset = dataset_settings(config)
    visualization = visualization_settings(config)
    assert dataset.dataset_root == (tmp_path / "local_data").resolve()
    assert dataset.dataset_dir == (tmp_path / "local_data/demo").resolve()
    assert visualization.task == "2"
    assert visualization.output == (VISUALIZED_DIR / "reports/demo.html").resolve()


def test_invalid_visualization_values_fail_before_dataset_access(tmp_path: Path) -> None:
    config = {
        "project_root": str(tmp_path),
        "dataset_root": "data",
        "dataset": "demo",
        "visualize": {"task": "all", "samples": 0},
    }
    with pytest.raises(ValueError, match="samples must be positive"):
        visualization_settings(config)


def test_cli_arguments_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["inspect_training_dataset.py", "--dataset", "other"])
    with pytest.raises(SystemExit, match="CLI arguments are not used"):
        reject_cli_arguments()


def test_samples_is_an_upper_bound_when_task_has_fewer_episodes() -> None:
    import pandas as pd

    episodes = pd.DataFrame(
        {
            "episode_index": range(50),
            "length": [20] * 50,
            "_resolved_task_index": [0] * 50,
        }
    )

    selected = select_samples(
        dataset_dir=Path("unused"),
        episodes=episodes,
        info={},
        task_indexes=[0],
        sample_count=100,
        sampling="first",
        seed=0,
        camera_keys=[],
    )

    assert len(selected) == 50
    assert [sample.episode_index for sample in selected] == list(range(50))
