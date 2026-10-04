from types import SimpleNamespace

import pytest

import lerobot.datasets.factory as dataset_factory
import lerobot.policies.skillVLA.dataset_skillVLA as skill_dataset_module


@pytest.mark.parametrize(
    ("shift_current", "expected"),
    [
        (True, (5, 102, 12, 3)),
        (False, (7, 100, 10, 5)),
    ],
)
def test_terminator_start_randomization_plan_preserves_selected_contract(
    shift_current: bool, expected: tuple[int, int, int, int]
) -> None:
    plan = skill_dataset_module._terminator_start_randomization_plan(
        item_index=100,
        frame_index=10,
        distance_from_start=7,
        distance_to_end=5,
        offset=2,
        shift_current_observation=shift_current,
    )

    assert plan == expected


def test_stage2_skill_only_dataset_includes_canonical_actions(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeSkillVLADataset:
        def __init__(self, *args, **kwargs):
            del args
            captured.update(kwargs)
            self.meta = SimpleNamespace(camera_keys=[], stats={})

    monkeypatch.setattr(
        dataset_factory,
        "LeRobotDatasetMetadata",
        lambda *args, **kwargs: SimpleNamespace(features={}, camera_keys=[]),
    )
    monkeypatch.setattr(
        skill_dataset_module, "SkillVLADataset", FakeSkillVLADataset
    )
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            repo_id="local/test",
            root="/tmp/test",
            revision=None,
            image_transforms=SimpleNamespace(enable=False),
            streaming=False,
            episodes=None,
            video_backend="pyav",
            use_imagenet_stats=False,
        ),
        policy=SimpleNamespace(
            type="skill_vla_stage2",
            # SkillVLA datasets have already materialized this transform. The
            # generic factory must not apply the DP-only runtime wrapper again.
            proprio_grounding="episode_start_xyz",
            dsbc_latent_predictor_enabled=True,
            dsbc_latent_supervision="skill_only",
            skill_flow_enabled=False,
            skill_flow_target="canonical",
            skill_flow_max_length=0,
            transition_jitter_pmax=10,
            transition_jitter_early_start_pmax=10,
            transition_jitter_late_start_pmax=5,
            transition_jitter_early_end_pmax=10,
            transition_jitter_late_end_pmax=5,
            foveated_vision_enabled=True,
            foveation_randomization_enabled=True,
            foveation_mode="crop",
            foveation_crop_size=128,
            foveation_output_size=224,
            foveation_inner_box_enabled=True,
            foveation_inner_box_mode="blur",
            foveation_inner_box_size=32,
            foveation_inner_box_line_width=3,
            foveation_shape="square",
            foveation_sharp_size=80,
            foveation_feather=12,
            foveation_peripheral_blur_radius=7.0,
            foveation_color_enabled=True,
            foveation_brightness_min=0.7,
            foveation_brightness_max=1.3,
            foveation_contrast_min=0.8,
            foveation_contrast_max=1.2,
            foveation_saturation_min=0.9,
            foveation_saturation_max=1.1,
            foveation_hue_min=-0.2,
            foveation_hue_max=0.2,
            foveation_crop_enabled=True,
            foveation_crop_offset_min_px=-16,
            foveation_crop_offset_max_px=16,
            foveation_inner_box_offset_min_px=-4,
            foveation_inner_box_offset_max_px=4,
            foveation_input_blur_enabled=True,
            foveation_input_blur_min_radius=0.5,
            foveation_input_blur_max_radius=3.0,
            use_dino_features=False,
            state_only=False,
            state_only_auxiliary=False,
        ),
        tolerance_s=1e-4,
    )

    dataset = dataset_factory.make_dataset(cfg)

    assert isinstance(dataset, FakeSkillVLADataset)
    assert captured["include_canonical_skill_actions"] is True
    assert captured["canonical_skill_action_max_length"] is None
    assert captured["foveated_vision_config"] == {
        "enabled": True,
        "randomization_enabled": True,
        "mode": "crop",
        "crop_size": 128,
        "output_size": 224,
        "inner_box_enabled": True,
        "inner_box_mode": "blur",
        "inner_box_size": 32,
        "inner_box_line_width": 3,
        "shape": "square",
        "sharp_size": 80,
        "feather": 12,
        "peripheral_mode": "blur",
        "peripheral_blur_radius": 7.0,
        "color_enabled": True,
        "brightness": (0.7, 1.3),
        "contrast": (0.8, 1.2),
        "saturation": (0.9, 1.1),
        "hue": (-0.2, 0.2),
        "crop_enabled": True,
        "crop_offset_px": (-16, 16),
        "inner_box_offset_px": (-4, 4),
        "input_blur_enabled": True,
        "input_blur_radius": (0.5, 3.0),
    }


@pytest.mark.parametrize(
    ("sampling_mode", "expected_video_keys"),
    [("mode1", []), ("mode2", None)],
)
def test_auxiliary_predictor_loads_mode_specific_videos(
    monkeypatch, sampling_mode, expected_video_keys
) -> None:
    captured: dict[str, object] = {}

    class FakeSkillVLADataset:
        def __init__(self, *args, **kwargs):
            del args
            captured.update(kwargs)
            self.meta = SimpleNamespace(camera_keys=[], stats={})

    monkeypatch.setattr(
        dataset_factory,
        "LeRobotDatasetMetadata",
        lambda *args, **kwargs: SimpleNamespace(features={}, camera_keys=[]),
    )
    monkeypatch.setattr(
        skill_dataset_module, "SkillVLADataset", FakeSkillVLADataset
    )
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            repo_id="local/test",
            root="/tmp/test",
            revision=None,
            image_transforms=SimpleNamespace(enable=False),
            streaming=False,
            episodes=None,
            video_backend="pyav",
            use_imagenet_stats=False,
        ),
        policy=SimpleNamespace(
            type="skill_aux",
            train_skill_predictor=True,
            train_terminator=False,
            train_state_rnn_terminator=False,
            predictor_transition_sampling=True,
            skill_predictor_sampling_mode=sampling_mode,
            use_dino_features=False,
            state_only=False,
            state_only_auxiliary=False,
            reward_delta_indices=None,
            action_delta_indices=None,
            observation_delta_indices=None,
        ),
        tolerance_s=1e-4,
    )

    dataset = dataset_factory.make_dataset(cfg)

    assert isinstance(dataset, FakeSkillVLADataset)
    assert captured["include_predictor_start_inputs"] is True
    assert captured["video_keys_to_load"] == expected_video_keys


def test_term14_forwards_coherent_start_randomization_to_dataset(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeSkillVLADataset:
        def __init__(self, *args, **kwargs):
            del args
            captured.update(kwargs)
            self.meta = SimpleNamespace(camera_keys=[], stats={})

    monkeypatch.setattr(
        dataset_factory,
        "LeRobotDatasetMetadata",
        lambda *args, **kwargs: SimpleNamespace(features={}, camera_keys=[]),
    )
    monkeypatch.setattr(
        skill_dataset_module, "SkillVLADataset", FakeSkillVLADataset
    )
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            repo_id="local/test",
            root="/tmp/test",
            revision=None,
            image_transforms=SimpleNamespace(enable=False),
            streaming=False,
            episodes=None,
            video_backend="pyav",
            use_imagenet_stats=False,
        ),
        policy=SimpleNamespace(
            type="skill_aux",
            train_skill_predictor=False,
            train_terminator=True,
            train_state_rnn_terminator=False,
            predictor_transition_sampling=False,
            terminator_start_proprio=True,
            terminator_start_randomization=True,
            terminator_start_randomization_early_frames=15,
            terminator_start_randomization_late_frames=10,
            terminator_start_randomization_distribution="half_normal",
            terminator_start_randomization_shift_current_observation=True,
            terminator_goal_xyz=False,
            terminator_chunk_end_pose_mode="off",
            use_dino_features=False,
            state_only=False,
            state_only_auxiliary=False,
            reward_delta_indices=None,
            action_delta_indices=None,
            observation_delta_indices=None,
        ),
        tolerance_s=1e-4,
    )

    dataset = dataset_factory.make_dataset(cfg)

    assert isinstance(dataset, FakeSkillVLADataset)
    assert captured["include_predictor_start_inputs"] is False
    assert captured["include_terminator_start_inputs"] is True
    assert captured["terminator_start_randomization"] is True
    assert captured["terminator_start_randomization_early_frames"] == 15
    assert captured["terminator_start_randomization_late_frames"] == 10
    assert captured["terminator_start_randomization_distribution"] == "half_normal"
    assert captured[
        "terminator_start_randomization_shift_current_observation"
    ] is True
