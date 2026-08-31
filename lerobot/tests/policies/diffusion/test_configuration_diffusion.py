import json

import pytest

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


LEGACY_EEF_FIELDS = {
    "use_eef_relative_actions": False,
    "eef_relative_stats_path": None,
    "eef_position_scale": 0.05,
    "eef_rotation_scale": 0.5,
}


def _write_config_with_legacy_eef_fields(tmp_path, *, enabled: bool):
    DiffusionConfig(device="cpu")._save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    serialized = json.loads(config_path.read_text())
    serialized.update(LEGACY_EEF_FIELDS)
    serialized["use_eef_relative_actions"] = enabled
    config_path.write_text(json.dumps(serialized))


def test_inactive_legacy_eef_fields_are_ignored_when_loading_checkpoint(tmp_path):
    _write_config_with_legacy_eef_fields(tmp_path, enabled=False)

    loaded = PreTrainedConfig.from_pretrained(tmp_path)

    assert isinstance(loaded, DiffusionConfig)
    for key in LEGACY_EEF_FIELDS:
        assert not hasattr(loaded, key)


def test_enabled_legacy_eef_mode_is_not_silently_ignored(tmp_path):
    _write_config_with_legacy_eef_fields(tmp_path, enabled=True)

    with pytest.raises(ValueError, match="removed EEF-relative action mode"):
        PreTrainedConfig.from_pretrained(tmp_path)


def test_new_diffusion_configs_do_not_serialize_removed_eef_fields(tmp_path):
    DiffusionConfig(device="cpu")._save_pretrained(tmp_path)

    serialized = json.loads((tmp_path / "config.json").read_text())

    assert LEGACY_EEF_FIELDS.keys().isdisjoint(serialized)
