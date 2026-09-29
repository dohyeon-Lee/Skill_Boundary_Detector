import json

import pytest
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    DiffusionModel,
)


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


def test_checkpoint_without_history_encoder_fields_keeps_flat_behavior(tmp_path):
    DiffusionConfig(device="cpu")._save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    serialized = json.loads(config_path.read_text())
    for key in (
        "history_encoder",
        "history_encoder_dim",
        "history_gru_layers",
        "history_transformer_layers",
        "history_transformer_heads",
    ):
        serialized.pop(key)
    config_path.write_text(json.dumps(serialized))

    loaded = PreTrainedConfig.from_pretrained(tmp_path)

    assert isinstance(loaded, DiffusionConfig)
    assert loaded.history_encoder == "flat"
    assert loaded.history_encoder_dim == 128


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


def test_action_history_conditioning_requests_strict_past_and_future_target():
    config = DiffusionConfig(
        device="cpu",
        state_only=True,
        history_conditioning="action",
        n_obs_steps=20,
        horizon=24,
        n_action_steps=24,
        action_sequence_mode="future_only",
        proprio_grounding="episode_start_xyz",
    )

    assert config.observation_delta_indices == [0]
    assert config.action_delta_indices == list(range(-20, 0)) + list(range(24))
    assert config.action_execution_start_index == 0
    assert config.action_prediction_horizon == 24
    assert config.proprio_grounding == "none"


def test_action_history_conditioning_rejects_visual_or_reconstruction_modes():
    with pytest.raises(ValueError, match="requires state_only=true"):
        DiffusionConfig(device="cpu", history_conditioning="action")

    with pytest.raises(ValueError, match="requires action_sequence_mode='future_only'"):
        DiffusionConfig(
            device="cpu",
            state_only=True,
            history_conditioning="action",
            n_obs_steps=4,
            horizon=7,
            future_action_horizon=4,
            n_action_steps=4,
            action_sequence_mode="history_reconstruction",
        )


def test_action_history_model_splits_condition_prefix_from_future_target():
    config = DiffusionConfig(
        device="cpu",
        state_only=True,
        history_conditioning="action",
        n_obs_steps=4,
        horizon=8,
        n_action_steps=4,
        down_dims=(8, 16),
        n_groups=4,
        diffusion_step_embed_dim=8,
        kernel_size=3,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,))
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))
        },
    )
    model = DiffusionModel(config)
    batch = {
        "observation.state": torch.randn(2, 1, 7),
        "action": torch.randn(2, 12, 7),
        "action_is_pad": torch.zeros(2, 12, dtype=torch.bool),
    }

    loss = model.compute_loss(batch)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    "history_encoder",
    ["gru", "transformer", "transformer_cls"],
)
def test_temporal_history_encoder_conditions_diffusion_with_fixed_width(history_encoder):
    config = DiffusionConfig(
        device="cpu",
        state_only=True,
        history_encoder=history_encoder,
        history_encoder_dim=16,
        history_transformer_heads=4,
        n_obs_steps=4,
        horizon=8,
        n_action_steps=4,
        down_dims=(8, 16),
        n_groups=4,
        diffusion_step_embed_dim=8,
        kernel_size=3,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,))
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))
        },
    )
    model = DiffusionModel(config)
    state_history = torch.randn(2, 4, 3)
    conditioning = model._prepare_global_conditioning(
        {"observation.state": state_history}
    )

    assert conditioning.shape == (2, 16)
    assert not torch.allclose(
        conditioning,
        model._prepare_global_conditioning(
            {"observation.state": state_history.flip(1)}
        ),
    )

    loss = model.compute_loss(
        {
            "observation.state": state_history,
            "action": torch.randn(2, 8, 2),
            "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
        }
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_transformer_cls_uses_a_separate_summary_token():
    config = DiffusionConfig(
        device="cpu",
        state_only=True,
        history_encoder="transformer_cls",
        history_encoder_dim=16,
        history_transformer_heads=4,
        n_obs_steps=4,
        horizon=8,
        n_action_steps=4,
        down_dims=(8, 16),
        n_groups=4,
        diffusion_step_embed_dim=8,
        kernel_size=3,
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,))
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))
        },
    )
    history_encoder = DiffusionModel(config).history_encoder

    assert history_encoder is not None
    assert history_encoder.cls_token.shape == (1, 1, 16)
    assert history_encoder.position_embedding.shape == (1, 5, 16)


@pytest.mark.parametrize("history_encoder", ["transformer", "transformer_cls"])
def test_history_encoder_config_roundtrip(tmp_path, history_encoder):
    config = DiffusionConfig(
        device="cpu",
        history_encoder=history_encoder,
        history_encoder_dim=64,
    )
    config._save_pretrained(tmp_path)

    loaded = PreTrainedConfig.from_pretrained(tmp_path)

    assert isinstance(loaded, DiffusionConfig)
    assert loaded.history_encoder == history_encoder
    assert loaded.history_encoder_dim == 64


@pytest.mark.parametrize("history_encoder", ["transformer", "transformer_cls"])
def test_invalid_history_encoder_settings_are_rejected(history_encoder):
    with pytest.raises(ValueError, match="history_encoder must be one of"):
        DiffusionConfig(device="cpu", history_encoder="rnn")

    with pytest.raises(ValueError, match="must be divisible"):
        DiffusionConfig(
            device="cpu",
            history_encoder=history_encoder,
            history_encoder_dim=15,
            history_transformer_heads=4,
        )


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
