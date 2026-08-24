import json
import sys
from pathlib import Path

import pytest

CONFIG_SRC = Path(__file__).resolve().parents[2] / "examples/libero/configs/train_skills/src"
sys.path.insert(0, str(CONFIG_SRC))

from train_skills_config import train_settings  # noqa: E402


def _config(tmp_path: Path) -> dict[str, object]:
    return {
        "project_root": str(tmp_path),
        "dataset_root": "dataset",
        "outputs_root": "outputs",
        "target_dataset": "libero_demo_rel",
        "dp_n_obs_steps": 4,
        "dp_horizon": 8,
    }


def _write_eef_contract(
    tmp_path: Path,
    *,
    position_scale: float = 0.03,
    rotation_scale: float = 0.7,
    stats_position_scale: float | None = None,
) -> None:
    meta = tmp_path / "dataset" / "libero_demo_rel" / "meta"
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
                "osc_position_scale": (
                    position_scale if stats_position_scale is None else stats_position_scale
                ),
                "osc_rotation_scale": rotation_scale,
            }
        )
    )


def test_eef_relative_dp_is_opt_in_and_uses_single_live_anchor_action(tmp_path: Path):
    ordinary = train_settings(_config(tmp_path))
    assert ordinary["dp_eef_relative"] is False
    assert ordinary["dp_n_action_steps"] == 5
    assert not ordinary["dp_policy"].endswith("_eefrel")

    config = _config(tmp_path)
    config["dp_eef_relative"] = True
    _write_eef_contract(tmp_path)
    eef_relative = train_settings(config)
    assert eef_relative["dp_eef_relative"] is True
    assert eef_relative["dp_relative"] is False
    assert eef_relative["dp_n_action_steps"] == 1
    assert eef_relative["dp_policy"].endswith("_eefrel")
    assert eef_relative["dp_eef_position_scale"] == 0.03
    assert eef_relative["dp_eef_rotation_scale"] == 0.7


def test_eef_relative_dp_rejects_incompatible_modes(tmp_path: Path):
    config = _config(tmp_path)
    config.update(dp_eef_relative=True, dp_relative=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        train_settings(config)

    config = _config(tmp_path)
    config.update(dp_eef_relative=True, dp_n_action_steps=2)
    with pytest.raises(ValueError, match="dp_n_action_steps=1"):
        train_settings(config)


def test_eef_relative_dp_requires_dataset_scales_and_rejects_duplicates(tmp_path: Path):
    config = _config(tmp_path)
    config["dp_eef_relative"] = True
    with pytest.raises(FileNotFoundError, match="dataset contract and stats"):
        train_settings(config)

    _write_eef_contract(tmp_path)
    config["dp_eef_position_scale"] = 0.03
    with pytest.raises(ValueError, match="Remove manual EEF scale settings"):
        train_settings(config)


def test_eef_relative_dp_rejects_inconsistent_dataset_scales(tmp_path: Path):
    config = _config(tmp_path)
    config["dp_eef_relative"] = True
    _write_eef_contract(tmp_path, stats_position_scale=0.04)
    with pytest.raises(ValueError, match="osc_position_scale mismatch"):
        train_settings(config)
