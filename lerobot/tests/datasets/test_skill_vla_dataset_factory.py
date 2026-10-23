from types import SimpleNamespace

import pytest

import lerobot.datasets.factory as dataset_factory
import lerobot.policies.skillVLA.dataset_skillVLA as skill_dataset_module
from lerobot.policies.skill_expert.configuration_skill_expert import (
    DIRECT_PATCH_ARCH_LABELS,
    DIRECT_PATCH_REVISION,
    FULLSKILL_1_REVISION,
    FULLSKILL_2_REVISION,
    LAYERWISE_COND_BOTTLENECK_ARCHITECTURE,
    LAYERWISE_COND_BOTTLENECK_CROSS_ATTENTION,
    RESIDUAL_PATCH_REVISION,
    WINDOW_PATCH_ARCH_LABELS,
    SkillExpertConfig,
    BSPLINE_REVISIONS,
)


@pytest.mark.parametrize("current", [False, True])
@pytest.mark.parametrize("ft", [False, True])
@pytest.mark.parametrize("skill_loss", [False, True])
@pytest.mark.parametrize("ablation", ["", "_preskill", "_none"])
@pytest.mark.parametrize("uv", [False, True, "diffaux"])
def test_bspline_factory_keeps_start_xyz_but_skips_unused_camera_decode(monkeypatch, current, ft, skill_loss, ablation, uv):
    if ablation and not current:
        pytest.skip("Memory ablations are current-observation models only")
    captured = {}
    class CapturingDataset:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            self.meta = SimpleNamespace(camera_keys=[], stats={})
    monkeypatch.setattr(dataset_factory, "LeRobotDatasetMetadata",
                        lambda *a, **k: SimpleNamespace(features={}, camera_keys=[]))
    monkeypatch.setattr(skill_dataset_module, "SkillVLADataset", CapturingDataset)
    label = "fullskill_bspline_3_lastlayer_lastvis" + ("_current" if current else "") + ablation
    if uv:
        label = "uv_" + label
    if uv == "diffaux":
        label += "_diffaux"
    policy = SkillExpertConfig(
        architecture_label=label, architecture=LAYERWISE_COND_BOTTLENECK_ARCHITECTURE,
        architecture_revision=BSPLINE_REVISIONS[label],
        vision_conditioning_mode=LAYERWISE_COND_BOTTLENECK_CROSS_ATTENTION,
        visual_bottleneck_tokens=100, visual_bridge_last_n_layers=9,
        chunk_end_state_dim=8, chunk_end_state_q01=[-1.] * 8, chunk_end_state_q99=[1.] * 8,
        skill_flow_enabled=False, skill_flow_max_length=0, newtask_ft_enabled=ft, device="cpu",
        newtask_ft_freeze_expert=ft, newtask_ft_skill_flow_loss=skill_loss,
    )
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(repo_id="local/test", root="/tmp/test", revision=None,
            image_transforms=SimpleNamespace(enable=False), streaming=False,
            episodes=None, video_backend="pyav", use_imagenet_stats=False),
        policy=policy, tolerance_s=1e-4,
    )
    dataset_factory.make_dataset(cfg)
    assert policy.uses_current_bspline_vision is current
    assert captured["decode_predictor_start_images"] is (not current)
    assert captured["include_skill_end_state_target"]
    assert captured.get("include_predictor_start_inputs", True)
    assert captured["bspline_skill_state_tokens"] == (0 if ft and not skill_loss else 30)
    assert not captured["include_canonical_skill_actions"]
    assert captured["include_preskill_plan_inputs"] is (ablation == "_preskill")


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


def test_terminator_start_randomization_probability_gate(monkeypatch) -> None:
    def fail_if_sampled() -> float:
        raise AssertionError("p=0 and p=1 must not consume an extra RNG draw")

    monkeypatch.setattr(skill_dataset_module.np.random, "random", fail_if_sampled)
    assert not skill_dataset_module._should_randomize_terminator_start(True, 0.0)
    assert skill_dataset_module._should_randomize_terminator_start(True, 1.0)
    assert not skill_dataset_module._should_randomize_terminator_start(False, 0.5)

    monkeypatch.setattr(skill_dataset_module.np.random, "random", lambda: 0.49)
    assert skill_dataset_module._should_randomize_terminator_start(True, 0.5)
    monkeypatch.setattr(skill_dataset_module.np.random, "random", lambda: 0.5)
    assert not skill_dataset_module._should_randomize_terminator_start(True, 0.5)


@pytest.mark.parametrize("motion_window", [False, True])
def test_skill_only_dataset_uses_canonical_or_fixed_window_targets(monkeypatch, motion_window) -> None:
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
            type="skill_expert" if motion_window else "skill_vla_stage2",
            uses_motion_window=motion_window,
            chunk_size=10,
            # SkillVLA datasets have already materialized this transform. The
            # generic factory must not apply the DP-only runtime wrapper again.
            proprio_grounding="episode_start_xyz",
            dsbc_latent_predictor_enabled=True,
            dsbc_latent_supervision="skill_only",
            skill_flow_enabled=motion_window,
            skill_flow_target="residual_window" if motion_window else "canonical",
            skill_flow_max_length=40 if motion_window else 0,
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
    assert captured["include_canonical_skill_actions"] is (not motion_window)
    assert captured["canonical_skill_action_max_length"] == (40 if motion_window else None)
    assert captured["residual_action_horizon"] == (10 if motion_window else 0)
    assert captured["include_skill_end_state_target"] is motion_window
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


@pytest.mark.parametrize("label", (*WINDOW_PATCH_ARCH_LABELS, "fullskill_1", "fullskill_2"))
@pytest.mark.parametrize("ft", [False, True])
def test_all_window_patch_datasets_request_endpoint_state(monkeypatch, label, ft):
    """Exercise the real policy contract through the factory, not a hand-filled batch."""
    captured = {}

    class CapturingDataset:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            self.meta = SimpleNamespace(camera_keys=[], stats={})

    monkeypatch.setattr(dataset_factory, "LeRobotDatasetMetadata",
                        lambda *args, **kwargs: SimpleNamespace(features={}, camera_keys=[]))
    monkeypatch.setattr(skill_dataset_module, "SkillVLADataset", CapturingDataset)
    policy = SkillExpertConfig(
        architecture_label=label,
        architecture=LAYERWISE_COND_BOTTLENECK_ARCHITECTURE,
        architecture_revision=(FULLSKILL_1_REVISION if label == "fullskill_1" else
                              FULLSKILL_2_REVISION if label == "fullskill_2" else
                              DIRECT_PATCH_REVISION if label in DIRECT_PATCH_ARCH_LABELS else RESIDUAL_PATCH_REVISION),
        vision_conditioning_mode=LAYERWISE_COND_BOTTLENECK_CROSS_ATTENTION,
        visual_bottleneck_tokens=100,
        chunk_size=10, n_action_steps=5,
        skill_flow_enabled=True, skill_flow_target="canonical" if label.startswith("fullskill_") else "residual_window",
        skill_flow_state_conditioned=not label.startswith("fullskill_"), skill_flow_max_length=120 if label.startswith("fullskill_") else 40,
        newtask_ft_enabled=ft, device="cpu",
    )
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(repo_id="local/test", root="/tmp/test", revision=None,
            image_transforms=SimpleNamespace(enable=False), streaming=False,
            episodes=None, video_backend="pyav", use_imagenet_stats=False),
        policy=policy, tolerance_s=1e-4,
    )
    dataset_factory.make_dataset(cfg)
    # These flags control production of skill_end_state AND its validity flag.
    # Both are needed even by variant 6: Cond-Gemma and patch supervision still
    # use the goal, despite the Expert's direct-goal ablation.
    assert captured["include_skill_end_state_target"] is True
    # SkillVLADataset defaults to predictor-start inputs for action policies.
    assert captured.get("include_predictor_start_inputs", True) is True
    assert captured["residual_action_horizon"] == (0 if label.startswith("fullskill_") else 10)
    assert captured["include_chunk_end_patch_target"] == (label == "fullskill_1")
    assert captured["include_canonical_skill_actions"] == (label.startswith("fullskill_") and not ft)
    assert captured["canonical_skill_action_max_length"] == (120 if label.startswith("fullskill_") else 40)
    if label == "fullskill_2":
        assert not captured["include_chunk_end_state_target"]
        assert not captured["include_chunk_end_patch_target"]


@pytest.mark.parametrize(
    "predictor_mode", [
        "mode1", "mode2", "mode3_preskill", "mode3_preskill_wristX", "mode3_preskill_wristX_proprioX",
        "mode3_preskill_proprioX",
    ],
)
def test_both_predictor_architectures_query_jittered_start_videos(
    monkeypatch, predictor_mode
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
            skill_predictor_mode=predictor_mode,
            skill_predictor_sampling_mode="mode1",
            transition_jitter_pmax=9,
            transition_jitter_early_start_pmax=9,
            transition_jitter_late_start_pmax=4,
            transition_jitter_early_end_pmax=8,
            transition_jitter_late_end_pmax=3,
            transition_jitter_distribution="uniform",
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
    assert captured["predictor_wrist_input"] == ("wristX" not in predictor_mode)
    assert captured["video_keys_to_load"] == []
    assert captured["jitter_pmax"] == 9
    assert captured["jitter_early_start_pmax"] == 9
    assert captured["jitter_late_start_pmax"] == 4
    assert captured["jitter_early_end_pmax"] == 8
    assert captured["jitter_late_end_pmax"] == 3
    assert captured["jitter_distribution"] == "uniform"


@pytest.mark.parametrize("goal_uv", [False, True])
def test_term14_forwards_coherent_start_randomization_to_dataset(monkeypatch, goal_uv) -> None:
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
            terminator_start_randomization_probability=0.5,
            terminator_start_randomization_shift_current_observation=True,
            terminator_goal_xyz=False,
            terminator_goal_uv=goal_uv,
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
    assert captured["include_terminator_goal_target"] is goal_uv
    assert captured["terminator_goal_uv"] is goal_uv
    assert captured["terminator_start_randomization"] is True
    assert captured["terminator_start_randomization_early_frames"] == 15
    assert captured["terminator_start_randomization_late_frames"] == 10
    assert captured["terminator_start_randomization_distribution"] == "half_normal"
    assert captured["terminator_start_randomization_probability"] == pytest.approx(0.5)
    assert captured[
        "terminator_start_randomization_shift_current_observation"
    ] is True
