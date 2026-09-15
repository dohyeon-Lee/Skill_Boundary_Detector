import json

import pytest
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionConditionalUnet1d


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


def test_default_action_sequence_starts_at_current_observation():
    config = DiffusionConfig(
        device="cpu",
        n_obs_steps=4,
        horizon=8,
        n_action_steps=5,
    )

    assert config.action_sequence_mode == "future_only"
    assert config.action_delta_indices == list(range(8))
    assert config.action_execution_start_index == 0
    assert config.drop_n_last_frames == 7


def test_future_only_action_indices_are_independent_of_observation_history():
    config = DiffusionConfig(
        device="cpu",
        n_obs_steps=20,
        horizon=24,
        n_action_steps=5,
        action_sequence_mode="future_only",
    )

    assert config.observation_delta_indices == list(range(-19, 1))
    assert config.action_delta_indices == list(range(24))
    assert config.action_execution_start_index == 0
    assert config.drop_n_last_frames == 23


def test_history_reconstruction_derives_aligned_past_and_future_ranges():
    config = DiffusionConfig(
        device="cpu",
        n_obs_steps=20,
        horizon=43,
        future_action_horizon=24,
        n_action_steps=24,
        action_sequence_mode="history_reconstruction",
    )

    assert config.observation_delta_indices == list(range(-19, 1))
    assert config.action_delta_indices == list(range(-19, 24))
    assert config.action_execution_start_index == 19
    assert config.action_prediction_horizon == 24
    assert config.drop_n_last_frames == 23
    assert config.padded_horizon == 48


def test_history_reconstruction_unet_pads_and_crops_non_multiple_horizon():
    config = DiffusionConfig(
        device="cpu",
        state_only=True,
        n_obs_steps=20,
        horizon=43,
        future_action_horizon=24,
        n_action_steps=24,
        action_sequence_mode="history_reconstruction",
        down_dims=(8, 16, 32),
        n_groups=4,
        diffusion_step_embed_dim=8,
        kernel_size=3,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )
    unet = DiffusionConditionalUnet1d(config, global_cond_dim=4)

    output = unet(
        torch.randn(2, 43, 3),
        torch.tensor([1, 2]),
        global_cond=torch.randn(2, 4),
    )

    assert output.shape == (2, 43, 3)


def test_future_only_accepts_execution_horizon_larger_than_old_limit():
    config = DiffusionConfig(
        device="cpu",
        n_obs_steps=20,
        horizon=24,
        n_action_steps=24,
        action_sequence_mode="future_only",
        drop_n_last_frames=23,
    )

    assert config.n_action_steps == config.horizon
    assert config.drop_n_last_frames == 23


def test_future_only_mode_survives_checkpoint_config_roundtrip(tmp_path):
    config = DiffusionConfig(
        device="cpu",
        n_obs_steps=20,
        horizon=24,
        n_action_steps=5,
        action_sequence_mode="future_only",
        drop_n_last_frames=23,
    )
    config._save_pretrained(tmp_path)

    loaded = PreTrainedConfig.from_pretrained(tmp_path)

    assert isinstance(loaded, DiffusionConfig)
    assert loaded.action_sequence_mode == "future_only"
    assert loaded.action_delta_indices == list(range(24))
    assert loaded.action_execution_start_index == 0
    assert loaded.drop_n_last_frames == 23


def test_history_reconstruction_mode_survives_checkpoint_config_roundtrip(tmp_path):
    config = DiffusionConfig(
        device="cpu",
        n_obs_steps=20,
        horizon=43,
        future_action_horizon=24,
        n_action_steps=24,
        action_sequence_mode="history_reconstruction",
    )
    config._save_pretrained(tmp_path)

    loaded = PreTrainedConfig.from_pretrained(tmp_path)

    assert isinstance(loaded, DiffusionConfig)
    assert loaded.action_sequence_mode == "history_reconstruction"
    assert loaded.action_delta_indices == list(range(-19, 24))
    assert loaded.action_execution_start_index == 19
    assert loaded.action_prediction_horizon == 24
    assert loaded.padded_horizon == 48


def test_removed_observation_aligned_mode_is_rejected():
    with pytest.raises(ValueError, match="action_sequence_mode must be one of"):
        DiffusionConfig(device="cpu", action_sequence_mode="observation_aligned")


def test_action_sequence_mode_rejects_out_of_range_execution_length():
    with pytest.raises(ValueError, match=r"n_action_steps must be in \[1, 24\]"):
        DiffusionConfig(
            device="cpu",
            n_obs_steps=20,
            horizon=24,
            n_action_steps=25,
            action_sequence_mode="future_only",
        )


def test_episode_start_xyz_grounding_is_checkpointed(tmp_path):
    config = DiffusionConfig(device="cpu", proprio_grounding="episode-start-xyz")
    config._save_pretrained(tmp_path)

    loaded = PreTrainedConfig.from_pretrained(tmp_path)

    assert loaded.proprio_grounding == "episode_start_xyz"


def test_unknown_proprio_grounding_is_rejected():
    with pytest.raises(ValueError, match="proprio_grounding must be"):
        DiffusionConfig(device="cpu", proprio_grounding="per_task")


def test_episode_grounding_rejects_relative_action_mode():
    with pytest.raises(ValueError, match="cannot be combined with use_relative_actions"):
        DiffusionConfig(
            device="cpu",
            proprio_grounding="episode_start_xyz",
            use_relative_actions=True,
            relative_stats_path="unused.json",
        )
