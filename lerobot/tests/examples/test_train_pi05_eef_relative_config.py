import importlib.util
import json
from pathlib import Path

import pytest


CONFIG_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_pi05/src/train_pi05_config.py"
)
SPEC = importlib.util.spec_from_file_location("train_pi05_config_eef_test", CONFIG_MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
CONFIG_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONFIG_MODULE)
build_settings = CONFIG_MODULE.build_settings


def _base_config(tmp_path: Path) -> dict[str, object]:
    tokenizer = tmp_path / "models" / "tokenizer"
    tokenizer.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (tokenizer / name).write_text("{}")

    return {
        "project_root": str(tmp_path),
        "dataset_root": "dataset",
        "outputs_root": "outputs",
        "pi05_tokenizer": "models/tokenizer",
        "pi_base": "models/pi05_base",
        "pt_dataset": "libero_90_full_full_rel",
        "pt_batch_size": 2,
        "pt_chunk_size": 10,
        "pt_exp": "",
    }


def _write_contract(
    tmp_path: Path,
    *,
    position_scale: float = 0.03,
    rotation_scale: float = 0.7,
    stats_position_scale: float | None = None,
    stats_chunk_size: int = 50,
) -> None:
    meta = tmp_path / "dataset" / "libero_90_full_full_rel" / "meta"
    meta.mkdir(parents=True)
    (meta / "action_contract.json").write_text(
        json.dumps(
            {
                "storage_representation": "absolute_eef_command",
                "model_representation": "eef_anchor_relative_so3",
                "rotation_representation": "axis_angle_rotation_vector",
                "rotation_composition": "left_world",
                "osc_position_scale": position_scale,
                "osc_rotation_scale": rotation_scale,
            }
        )
    )
    (meta / "relative_action_stats.json").write_text(
        json.dumps(
            {
                "representation": "eef_anchor_relative_so3",
                "storage_representation": "absolute_eef_command",
                "rotation_representation": "axis_angle_rotation_vector",
                "rotation_composition": "left_world",
                "chunk_size": stats_chunk_size,
                "osc_position_scale": (
                    position_scale if stats_position_scale is None else stats_position_scale
                ),
                "osc_rotation_scale": rotation_scale,
            }
        )
    )


def test_pi05_eef_relative_is_opt_in_and_loads_dataset_contract(tmp_path: Path):
    config = _base_config(tmp_path)
    ordinary = build_settings(config)
    assert ordinary["pt_eef_relative"] is False
    assert not ordinary["pt_run_name"].endswith("_eefrel")

    _write_contract(tmp_path)
    config["pt_eef_relative"] = True
    relative = build_settings(config)
    assert relative["pt_eef_relative"] is True
    assert relative["pt_eef_position_scale"] == 0.03
    assert relative["pt_eef_rotation_scale"] == 0.7
    assert relative["pt_eef_chunk_size"] == 10
    assert relative["pt_run_name"].endswith("_eefrel")
    assert Path(relative["pt_eef_relative_stats_path"]).name == "relative_action_stats.json"


def test_pi05_eef_relative_rejects_manual_scales(tmp_path: Path):
    config = _base_config(tmp_path)
    _write_contract(tmp_path)
    config.update(pt_eef_relative=True, pt_eef_position_scale=0.03)
    with pytest.raises(ValueError, match="Remove manual EEF scale settings"):
        build_settings(config)


def test_pi05_eef_relative_rejects_short_stats_chunk(tmp_path: Path):
    config = _base_config(tmp_path)
    _write_contract(tmp_path, stats_chunk_size=9)
    config["pt_eef_relative"] = True
    with pytest.raises(ValueError, match="chunk_size=9 < PI0.5 chunk_size=10"):
        build_settings(config)


def test_pi05_eef_relative_rejects_inconsistent_scales(tmp_path: Path):
    config = _base_config(tmp_path)
    _write_contract(tmp_path, stats_position_scale=0.04)
    config["pt_eef_relative"] = True
    with pytest.raises(ValueError, match="osc_position_scale mismatch"):
        build_settings(config)


def test_pi05_warmup_constant_schedule_is_opt_in_and_tagged(tmp_path: Path):
    config = _base_config(tmp_path)
    config.update(
        pt_lr_mode="warmup_constant",
        pt_warmup_steps=123,
        pt_decay_steps=456,
        pt_decay_lr=1e-6,
    )
    settings = build_settings(config)
    assert settings["pt_lr_mode"] == "warmup_constant"
    assert settings["pt_warmup_steps"] == 123
    assert settings["pt_decay_steps"] == 456
    assert settings["pt_decay_lr"] == 1e-6
    assert settings["pt_run_name"].endswith("_constlr")

    config["pt_lr_mode"] = "constant"
    with pytest.raises(ValueError, match="warmup_constant"):
        build_settings(config)
