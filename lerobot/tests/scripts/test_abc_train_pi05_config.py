from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
ABC_CONFIGS = ROOT / "lerobot/examples/abc/configs"
TRAIN_MODULE = ABC_CONFIGS / "train_pi05/PT/src/train_pi05_config.py"
LOADER_MODULE = ABC_CONFIGS / "src/global_config_loader.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TRAIN = _load_module("abc_train_pi05_config", TRAIN_MODULE)
GLOBAL = _load_module("abc_global_config_loader", LOADER_MODULE)


def _config(project_root: Path, **overrides):
    tokenizer = project_root / "models/tokenizer"
    tokenizer.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (tokenizer / name).write_text("{}", encoding="utf-8")
    config = {
        "project_root": str(project_root),
        "dataset_root": "dataset_ABC",
        "outputs_root": "outputs_ABC",
        "pi05_tokenizer": "models/tokenizer",
        "pt_dataset_root": "dataset_ABC",
        "pt_dataset": "abc_sim_all",
    }
    config.update(overrides)
    return config


def test_abc_global_config_is_independent_and_keeps_yonsei_profile() -> None:
    config = GLOBAL.load_global_config(ABC_CONFIGS / "global_config.yaml", server="yonsei")
    assert config["dataset_root"] == "dataset_ABC"
    assert config["outputs_root"] == "outputs_ABC"
    assert config["train_qos"] == "base_qos"
    assert "suma_rtx4090" in config["train_partition"]
    assert "node05" in config["train_exclude_nodes"]
    assert config["abc_sim_task_groups"]["group_ABCVLA_task"] == [
        "ball_tray_balancing",
        "conveyor_pick",
        "count_into_opaque_box",
        "grab_clutter",
        "lego_blocks_sorting",
        "nuts_bolts_sorting",
        "put_relative",
        "sim_pouring_beads",
        "sim_inhand_transfer_the_item_to_other_side",
        "sim_load_the_plates_into_the_dish_rack",
        "sim_put_the_plastic_bottles_in_the_bin",
        "sim_sweep_away_paper_scraps_from_the_table",
    ]
    assert config["abc_sim_task_groups"]["group_ABCVLA_easy_task"] == [
        "sim_inhand_transfer_the_item_to_other_side",
        "sim_put_the_plastic_bottles_in_the_bin",
        "sim_load_the_plates_into_the_dish_rack",
        "put_relative",
        "grab_clutter",
    ]
    assert config["abc_sim_task_groups"]["group_multitask"] == [
        "put_relative",
        "grab_clutter",
    ]


def test_abc_native_pi05_defaults_and_rejects_xyz_grounding(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset_ABC/abc_sim_all"
    dataset.mkdir(parents=True)
    (dataset / "native_dataset.json").write_text(
        json.dumps({"format": "sbd_abc_sim_native/v1"}), encoding="utf-8"
    )

    settings = TRAIN.build_settings(_config(tmp_path))

    assert settings["pt_is_abc_native"] is True
    assert settings["pt_chunk_size"] == 30
    assert settings["pt_n_action_steps"] == 15
    assert settings["pt_use_imagenet_stats"] is False
    assert settings["pt_proprio_grounding"] == "none"
    assert settings["pt_rename_map"] == TRAIN.ABC_PI05_RENAME_MAP
    assert settings["pt_output_dir"].is_relative_to(tmp_path / "outputs_ABC")

    with pytest.raises(ValueError, match="YAM joint positions"):
        TRAIN.build_settings(_config(tmp_path, pt_proprio_grounding=True))


def test_abc_pi05_can_filter_official_source_tasks(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset_ABC/abc_sim_all"
    dataset.mkdir(parents=True)
    (dataset / "native_dataset.json").write_text(
        json.dumps(
            {
                "format": "sbd_abc_sim_native/v1",
                "task_sha256": {
                    "sim_pouring_beads": "a",
                    "sim_inhand_transfer_the_item_to_other_side": "b",
                },
            }
        ),
        encoding="utf-8",
    )

    settings = TRAIN.build_settings(
        _config(
            tmp_path,
            pt_tasks=[
                "sim_pouring_beads",
                "sim_inhand_transfer_the_item_to_other_side",
            ],
            pt_exp="selected",
        )
    )

    assert settings["pt_tasks"] == [
        "sim_pouring_beads",
        "sim_inhand_transfer_the_item_to_other_side",
    ]
    assert settings["pt_resolved_tasks"] == [
        "sim_pouring_beads",
        "sim_inhand_transfer_the_item_to_other_side",
    ]
    assert settings["pt_task_filter"] is True
    assert settings["pt_task_count"] == 2
    assert settings["pt_tasks_label"].startswith("sim_pouring_beads,")
    assert "_tasks2-" in settings["pt_run_name"]

    single = TRAIN.build_settings(
        _config(tmp_path, pt_tasks="sim_pouring_beads", pt_exp="selected")
    )
    assert single["pt_run_name"].endswith("_task-sim_pouring_beads")

    with pytest.raises(ValueError, match="Unknown pt_tasks"):
        TRAIN.build_settings(_config(tmp_path, pt_tasks=["missing"]))


def test_abc_pi05_expands_shared_named_task_group(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "dataset_ABC/abc_sim_all"
    dataset.mkdir(parents=True)
    group_tasks = ["sim_pouring_beads", "sim_inhand_transfer_the_item_to_other_side"]
    (dataset / "native_dataset.json").write_text(
        json.dumps(
            {
                "format": "sbd_abc_sim_native/v1",
                "task_sha256": {task: str(index) for index, task in enumerate(group_tasks)},
            }
        ),
        encoding="utf-8",
    )

    settings = TRAIN.build_settings(
        _config(
            tmp_path,
            pt_tasks="group_ABCVLA_task",
            abc_sim_task_groups={"group_ABCVLA_task": group_tasks},
        )
    )

    assert settings["pt_task_selectors"] == ["group_ABCVLA_task"]
    assert settings["pt_task_groups"] == ["group_ABCVLA_task"]
    assert settings["pt_tasks"] == ["group_ABCVLA_task"]
    assert settings["pt_resolved_tasks"] == group_tasks
    assert settings["pt_tasks_label"] == "group_ABCVLA_task"
    assert settings["pt_resolved_tasks_label"] == ",".join(group_tasks)
    assert settings["pt_run_name"].endswith("_group_ABCVLA_task")

    # submit_train.sh exports its resolved shell settings into Slurm; the job's
    # second resolver pass must retain the group identity and output path.
    monkeypatch.setenv("PT_TASKS", json.dumps(settings["pt_tasks"]))
    rerun = TRAIN.build_settings(
        _config(
            tmp_path,
            pt_tasks="all",
            abc_sim_task_groups={"group_ABCVLA_task": group_tasks},
        )
    )
    assert rerun["pt_run_name"] == settings["pt_run_name"]
    assert rerun["pt_resolved_tasks"] == group_tasks
    monkeypatch.delenv("PT_TASKS")

    with pytest.raises(ValueError, match="Unknown ABC Sim task group"):
        TRAIN.build_settings(
            _config(
                tmp_path,
                pt_tasks="group_missing",
                abc_sim_task_groups={"group_ABCVLA_task": group_tasks},
            )
        )


def test_shipped_pi05_config_targets_abc_sim_native_dataset() -> None:
    config = yaml.safe_load((ABC_CONFIGS / "train_pi05/PT/pi05_PT_config.yaml").read_text())
    assert config["pt_dataset_root"] == "dataset_ABC"
    assert config["pt_dataset"] == "abc_sim_all"
    assert config["pt_tasks"]
    assert config["pt_chunk_size"] == 30
    assert config["pt_n_action_steps"] == 15
