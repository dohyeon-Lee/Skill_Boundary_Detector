from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.skill_expert.configuration_skill_expert import (
    COND_GEMMA_ARCHITECTURE,
    COND_GEMMA_ARCHITECTURE_REVISION,
    FIXED_BOTTLENECK_CROSS_ATTENTION,
    FIXED_VISUAL_BOTTLENECK_ARCHITECTURE,
    FIXED_VISUAL_BOTTLENECK_REVISION,
    INTERLEAVED_CROSS_ATTENTION,
    LAYERWISE_COND_BOTTLENECK_ARCHITECTURE,
    LAYERWISE_COND_BOTTLENECK_CROSS_ATTENTION,
    LAYERWISE_COND_BOTTLENECK_CORE_EXIT_REVISION,
    LAYERWISE_COND_BOTTLENECK_LATENT_UV_REVISION,
    LAYERWISE_COND_BOTTLENECK_LATENT_XYZ_REVISION,
    LAYERWISE_COND_BOTTLENECK_UV_COND_XYZ_COND_TERMINATION_REVISION,
    LAYERWISE_COND_BOTTLENECK_UV_COND_XYZ_REVISION,
    LAYERWISE_COND_BOTTLENECK_UV_COND_XYZ_TERMINATION_REVISION,
    LAYERWISE_COND_BOTTLENECK_XYZ_COND_UV_EXPERT_END_POSE_REVISION,
    LAYERWISE_COND_BOTTLENECK_WRIST_EXPERT_SKILL_DELTA_BRIDGE_PROPRIO_REVISION,
    LAYERWISE_COND_BOTTLENECK_WRIST_EXPERT_SKILL_DELTA_ALIGN_REVISION,
    LAYERWISE_COND_BOTTLENECK_WRIST_EXPERT_SKILL_DELTA_BRIDGE_PROPRIO_ALIGN_REVISION,
    LAYERWISE_COND_BOTTLENECK_WRIST_EXPERT_SKILL_START_END_BRIDGE_PROPRIO_ALIGN_REVISION,
    LAYERWISE_COND_BOTTLENECK_WRIST_EXPERT_SKILL_START_END_BRIDGE_PROPRIO_REVISION,
    LAYERWISE_COND_BOTTLENECK_WRIST_EXPERT_SKILL_DELTA_REVISION,
    LAYERWISE_COND_BOTTLENECK_XYZ_SKILL_COND_UV_EXPERT_END_POSE_REVISION,
    LAYERWISE_COND_BOTTLENECK_XYZ_SKILL_COND_UV_EXPERT_SKILL_DELTA_REVISION,
    LAYERWISE_COND_BOTTLENECK_XYZ_SKILL_COND_UV_REVISION,
    LAYERWISE_COND_BOTTLENECK_XYZ_COND_UV_REVISION,
    LAYERWISE_COND_BOTTLENECK_WRIST_SKILL_END_POSE_COND_TERMINATION_REVISION,
    LAYERWISE_COND_BOTTLENECK_WRIST_SKILL_END_POSE_REVISION,
    LAYERWISE_COND_BOTTLENECK_WRIST_SKILL_END_POSE_TERMINATION_REVISION,
    LAYERWISE_COND_BOTTLENECK_WRIST_END_POSE_COND_TERMINATION_REVISION,
    LAYERWISE_COND_BOTTLENECK_WRIST_END_POSE_REVISION,
    LAYERWISE_COND_BOTTLENECK_WRIST_COND_SKILL_END_POSE_EXPERT_END_POSE_REVISION,
    LAYERWISE_COND_BOTTLENECK_WRIST_COND_SKILL_END_POSE_EXPERT_END_POSE_TERMINATION_REVISION,
    LAYERWISE_COND_BOTTLENECK_WRIST_COND_SKILL_END_POSE_EXPERT_SKILL_REVISION,
    LAYERWISE_COND_BOTTLENECK_WRIST_COND_SKILL_END_POSE_EXPERT_SKILL_TERMINATION_REVISION,
    LAYERWISE_COND_BOTTLENECK_REVISION,
    LAYERWISE_COND_BOTTLENECK_UV_REVISION,
    LATE_VISUAL_BOTTLENECK_REVISION,
    SUPPORTED_ARCHITECTURE_LABELS,
    SkillExpertConfig,
)
from lerobot.policies.skill_expert.fixed_visual_bottleneck import (
    FixedVisualBottleneckSkillExpert,
    LateVisualBottleneckSkillExpert,
)
from lerobot.policies.skill_expert.layerwise_cond_bottleneck import (
    BottleneckUVAlignedCoreExitLayerwiseCondBottleneckSkillExpert,
    BottleneckXYZAlignedCoreExitLayerwiseCondBottleneckSkillExpert,
    CoreExitLayerwiseCondBottleneckSkillExpert,
    LayerwiseCondBottleneckSkillExpert,
    UVAlignedCoreExitLayerwiseCondBottleneckSkillExpert,
    UVConditionedBottleneckXYZSkillExpert,
    UVConditionedBottleneckXYZTerminationSkillExpert,
    XYZConditionedBottleneckUVExpertEndPoseSkillExpert,
    XYZConditionedBottleneckUVSkillExpert,
    WristSkillDeltaGoalBridgeProprioSkillExpert,
    WristSkillDeltaGoalSkillExpert,
    WristSkillDeltaGoalAlignSkillExpert,
    WristSkillDeltaGoalBridgeProprioAlignSkillExpert,
    WristSkillStartEndGoalBridgeProprioAlignSkillExpert,
    WristSkillStartEndGoalBridgeProprioSkillExpert,
    XYZSkillConditionedBottleneckUVExpertEndPoseSkillExpert,
    XYZSkillConditionedBottleneckUVExpertSkillDeltaSkillExpert,
    XYZSkillConditionedBottleneckUVSkillExpert,
    WristSkillEndPoseLayerwiseCondBottleneckSkillExpert,
    WristSkillEndPoseTerminationSkillExpert,
    WristEndPoseLayerwiseCondBottleneckSkillExpert,
    WristEndPoseTerminationSkillExpert,
    WristCondSkillEndPoseLayerwiseCondBottleneckSkillExpert,
    WristCondSkillEndPoseTerminationSkillExpert,
    WristCondSkillEndPoseExpertSkillLayerwiseCondBottleneckSkillExpert,
    WristCondSkillEndPoseExpertSkillTerminationSkillExpert,
)
from lerobot.policies.skill_expert.cond_gemma import CondGemmaSkillExpert
from lerobot.policies.skill_expert.modeling_skill_expert import (
    SkillExpertPolicy,
    _allowed_pi05_missing_key,
    _map_pi05_key,
)


class _RecordingAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.memories: list[torch.Tensor] = []

    def forward(self, query, key, value, *, need_weights):
        assert need_weights is False
        self.memories.append(key.detach().clone())
        pooled = value.mean(dim=1, keepdim=True)
        return pooled.expand(-1, query.shape[1], -1), None


def _skill_config(label: str) -> SkillExpertConfig:
    if label not in SUPPORTED_ARCHITECTURE_LABELS:
        raise AssertionError(label)
    is_arch1 = label == "arch1" or label.startswith("arch1_")
    is_arch2 = label == "arch2" or label.startswith("arch2_")
    is_arch3 = label.startswith("arch3")
    is_arch4 = label.startswith("arch4")
    is_arch5 = label.startswith("arch5")
    is_arch6 = label.startswith("arch6")
    is_arch7 = label.startswith("arch7")
    is_arch8_1 = label.startswith("arch8_1")
    is_arch8_2 = label.startswith("arch8_2")
    is_arch9_1 = label.startswith("arch9_1")
    is_arch9_2 = label.startswith("arch9_2")
    is_arch10_1 = label.startswith("arch10_1")
    is_arch10_2 = label.startswith("arch10_2")
    is_arch11_1 = label.startswith("arch11_1")
    is_arch11_2 = label.startswith("arch11_2")
    is_arch12_1 = label.startswith("arch12_1")
    is_arch12_2 = label.startswith("arch12_2")
    is_arch18 = label.startswith("arch18")
    is_arch17 = label.startswith("arch17")
    is_arch16 = label.startswith("arch16")
    is_align = label.startswith(("arch16_align", "arch17_align", "arch18_align"))
    is_arch15 = label.startswith("arch15")
    is_arch14 = label.startswith("arch14")
    is_arch19 = label.startswith("arch19")
    is_arch20 = label.startswith("arch20")
    is_arch13 = label.startswith(("arch13", "arch14", "arch15", "arch19", "arch20"))
    is_layerwise = is_arch3 or is_arch4 or is_arch5 or is_arch6 or is_arch7 or is_arch8_1 or is_arch8_2 or is_arch9_1 or is_arch9_2 or is_arch10_1 or is_arch10_2 or is_arch11_1 or is_arch11_2 or is_arch12_1 or is_arch12_2 or is_arch13 or is_arch16 or is_arch17 or is_arch18
    is_visual_bottleneck = is_arch1 or is_arch2
    kwargs = {
        "architecture": (
            LAYERWISE_COND_BOTTLENECK_ARCHITECTURE
            if is_layerwise
            else (
                FIXED_VISUAL_BOTTLENECK_ARCHITECTURE
                if is_visual_bottleneck
                else COND_GEMMA_ARCHITECTURE
            )
        ),
        "architecture_label": label,
        "architecture_revision": (
            LAYERWISE_COND_BOTTLENECK_LATENT_XYZ_REVISION
            if is_arch7
            else (
                LAYERWISE_COND_BOTTLENECK_LATENT_UV_REVISION
                if is_arch6
                else (
                    LAYERWISE_COND_BOTTLENECK_UV_REVISION
                    if is_arch5
                    else (
                        LAYERWISE_COND_BOTTLENECK_CORE_EXIT_REVISION
                        if is_arch4
                        else (
                            LAYERWISE_COND_BOTTLENECK_REVISION
                            if is_arch3
                            else (
                                LATE_VISUAL_BOTTLENECK_REVISION
                                if is_arch2
                                else (
                                    FIXED_VISUAL_BOTTLENECK_REVISION
                                    if is_arch1
                                    else COND_GEMMA_ARCHITECTURE_REVISION
                                )
                            )
                        )
                    )
                )
            )
        ),
        "vision_conditioning_mode": (
            LAYERWISE_COND_BOTTLENECK_CROSS_ATTENTION
            if is_layerwise
            else (
                FIXED_BOTTLENECK_CROSS_ATTENTION
                if is_visual_bottleneck
                else INTERLEAVED_CROSS_ATTENTION
            )
        ),
    }
    if label.endswith("_skill"):
        kwargs.update(
            skill_flow_enabled=True,
            skill_flow_target="canonical",
            skill_flow_max_length=120,
        )
    elif label.endswith("_skill_chunk"):
        kwargs.update(
            skill_flow_enabled=True,
            skill_flow_target="extended_chunk",
            skill_flow_max_length=30,
            skill_flow_chunk_multiplier=3,
        )
    if is_arch8_1:
        kwargs["architecture_revision"] = LAYERWISE_COND_BOTTLENECK_UV_COND_XYZ_REVISION
    if is_arch8_2:
        kwargs["architecture_revision"] = LAYERWISE_COND_BOTTLENECK_UV_COND_XYZ_COND_TERMINATION_REVISION
    if is_arch9_1:
        kwargs["architecture_revision"] = LAYERWISE_COND_BOTTLENECK_WRIST_SKILL_END_POSE_REVISION
    if is_arch9_2:
        kwargs["architecture_revision"] = LAYERWISE_COND_BOTTLENECK_WRIST_SKILL_END_POSE_COND_TERMINATION_REVISION
    if is_arch10_1:
        kwargs["architecture_revision"] = LAYERWISE_COND_BOTTLENECK_WRIST_END_POSE_REVISION
    if is_arch10_2:
        kwargs["architecture_revision"] = LAYERWISE_COND_BOTTLENECK_WRIST_END_POSE_COND_TERMINATION_REVISION
    if is_arch11_1:
        kwargs["architecture_revision"] = LAYERWISE_COND_BOTTLENECK_WRIST_COND_SKILL_END_POSE_EXPERT_END_POSE_REVISION
    if is_arch11_2:
        kwargs["architecture_revision"] = LAYERWISE_COND_BOTTLENECK_WRIST_COND_SKILL_END_POSE_EXPERT_END_POSE_TERMINATION_REVISION
    if is_arch12_1:
        kwargs["architecture_revision"] = LAYERWISE_COND_BOTTLENECK_WRIST_COND_SKILL_END_POSE_EXPERT_SKILL_REVISION
    if is_arch12_2:
        kwargs["architecture_revision"] = LAYERWISE_COND_BOTTLENECK_WRIST_COND_SKILL_END_POSE_EXPERT_SKILL_TERMINATION_REVISION
    if is_arch13:
        kwargs["architecture_revision"] = LAYERWISE_COND_BOTTLENECK_XYZ_COND_UV_REVISION
    if is_arch14:
        kwargs["architecture_revision"] = LAYERWISE_COND_BOTTLENECK_XYZ_COND_UV_EXPERT_END_POSE_REVISION
    if is_arch15:
        kwargs["architecture_revision"] = LAYERWISE_COND_BOTTLENECK_XYZ_SKILL_COND_UV_EXPERT_END_POSE_REVISION
    if is_arch16:
        kwargs["architecture_revision"] = LAYERWISE_COND_BOTTLENECK_WRIST_EXPERT_SKILL_DELTA_REVISION
    if is_arch17:
        kwargs["architecture_revision"] = LAYERWISE_COND_BOTTLENECK_WRIST_EXPERT_SKILL_DELTA_BRIDGE_PROPRIO_REVISION
    if is_arch18:
        kwargs["architecture_revision"] = LAYERWISE_COND_BOTTLENECK_WRIST_EXPERT_SKILL_START_END_BRIDGE_PROPRIO_REVISION
    if is_align:
        kwargs["architecture_revision"] = (
            LAYERWISE_COND_BOTTLENECK_WRIST_EXPERT_SKILL_START_END_BRIDGE_PROPRIO_ALIGN_REVISION if is_arch18
            else LAYERWISE_COND_BOTTLENECK_WRIST_EXPERT_SKILL_DELTA_BRIDGE_PROPRIO_ALIGN_REVISION if is_arch17
            else LAYERWISE_COND_BOTTLENECK_WRIST_EXPERT_SKILL_DELTA_ALIGN_REVISION
        )
    if is_arch19:
        kwargs["architecture_revision"] = LAYERWISE_COND_BOTTLENECK_XYZ_SKILL_COND_UV_EXPERT_SKILL_DELTA_REVISION
    if is_arch20:
        kwargs["architecture_revision"] = LAYERWISE_COND_BOTTLENECK_XYZ_SKILL_COND_UV_REVISION
    return SkillExpertConfig(**kwargs)


@pytest.mark.parametrize("label", sorted(SUPPORTED_ARCHITECTURE_LABELS))
def test_only_retained_stage1_architectures_validate(label: str) -> None:
    config = _skill_config(label)

    is_arch1 = label == "arch1" or label.startswith("arch1_")
    is_arch2 = label == "arch2" or label.startswith("arch2_")
    is_arch3 = label.startswith("arch3")
    is_arch4 = label.startswith("arch4")
    is_arch5 = label.startswith("arch5")
    is_arch6 = label.startswith("arch6")
    is_arch7 = label.startswith("arch7")
    is_arch8_1 = label.startswith("arch8_1")
    is_arch8_2 = label.startswith("arch8_2")
    is_arch9_1 = label.startswith("arch9_1")
    is_arch9_2 = label.startswith("arch9_2")
    is_arch10_1 = label.startswith("arch10_1")
    is_arch10_2 = label.startswith("arch10_2")
    is_arch11_1 = label.startswith("arch11_1")
    is_arch11_2 = label.startswith("arch11_2")
    is_arch12_1 = label.startswith("arch12_1")
    is_arch12_2 = label.startswith("arch12_2")
    is_arch18 = label.startswith("arch18")
    is_arch17 = label.startswith("arch17")
    is_arch16 = label.startswith("arch16")
    is_align = label.startswith(("arch16_align", "arch17_align", "arch18_align"))
    is_arch15 = label.startswith("arch15")
    is_arch14 = label.startswith("arch14")
    is_arch19 = label.startswith("arch19")
    is_arch20 = label.startswith("arch20")
    is_arch13 = label.startswith(("arch13", "arch14", "arch15", "arch19", "arch20"))
    is_layerwise = is_arch3 or is_arch4 or is_arch5 or is_arch6 or is_arch7 or is_arch8_1 or is_arch8_2 or is_arch9_1 or is_arch9_2 or is_arch10_1 or is_arch10_2 or is_arch11_1 or is_arch11_2 or is_arch12_1 or is_arch12_2 or is_arch13 or is_arch16 or is_arch17 or is_arch18
    is_visual_bottleneck = is_arch1 or is_arch2
    assert config.architecture == (
        LAYERWISE_COND_BOTTLENECK_ARCHITECTURE
        if is_layerwise
        else (
            FIXED_VISUAL_BOTTLENECK_ARCHITECTURE
            if is_visual_bottleneck
            else COND_GEMMA_ARCHITECTURE
        )
    )
    expected_revision = COND_GEMMA_ARCHITECTURE_REVISION
    for enabled, revision in (
        (is_arch1, FIXED_VISUAL_BOTTLENECK_REVISION),
        (is_arch2, LATE_VISUAL_BOTTLENECK_REVISION),
        (is_arch3, LAYERWISE_COND_BOTTLENECK_REVISION),
        (is_arch4, LAYERWISE_COND_BOTTLENECK_CORE_EXIT_REVISION),
        (is_arch5, LAYERWISE_COND_BOTTLENECK_UV_REVISION),
        (is_arch6, LAYERWISE_COND_BOTTLENECK_LATENT_UV_REVISION),
        (is_arch7, LAYERWISE_COND_BOTTLENECK_LATENT_XYZ_REVISION),
        (is_arch8_1, LAYERWISE_COND_BOTTLENECK_UV_COND_XYZ_REVISION),
        (is_arch8_2, LAYERWISE_COND_BOTTLENECK_UV_COND_XYZ_COND_TERMINATION_REVISION),
        (is_arch9_1, LAYERWISE_COND_BOTTLENECK_WRIST_SKILL_END_POSE_REVISION),
        (is_arch9_2, LAYERWISE_COND_BOTTLENECK_WRIST_SKILL_END_POSE_COND_TERMINATION_REVISION),
        (is_arch10_1, LAYERWISE_COND_BOTTLENECK_WRIST_END_POSE_REVISION),
        (is_arch10_2, LAYERWISE_COND_BOTTLENECK_WRIST_END_POSE_COND_TERMINATION_REVISION),
        (is_arch11_1, LAYERWISE_COND_BOTTLENECK_WRIST_COND_SKILL_END_POSE_EXPERT_END_POSE_REVISION),
        (is_arch11_2, LAYERWISE_COND_BOTTLENECK_WRIST_COND_SKILL_END_POSE_EXPERT_END_POSE_TERMINATION_REVISION),
        (is_arch12_1, LAYERWISE_COND_BOTTLENECK_WRIST_COND_SKILL_END_POSE_EXPERT_SKILL_REVISION),
        (is_arch12_2, LAYERWISE_COND_BOTTLENECK_WRIST_COND_SKILL_END_POSE_EXPERT_SKILL_TERMINATION_REVISION),
        (is_arch13, LAYERWISE_COND_BOTTLENECK_XYZ_COND_UV_REVISION),
        (is_arch14, LAYERWISE_COND_BOTTLENECK_XYZ_COND_UV_EXPERT_END_POSE_REVISION),
        (is_arch15, LAYERWISE_COND_BOTTLENECK_XYZ_SKILL_COND_UV_EXPERT_END_POSE_REVISION),
        (is_arch16, LAYERWISE_COND_BOTTLENECK_WRIST_EXPERT_SKILL_DELTA_REVISION),
        (is_arch17, LAYERWISE_COND_BOTTLENECK_WRIST_EXPERT_SKILL_DELTA_BRIDGE_PROPRIO_REVISION),
        (is_arch18, LAYERWISE_COND_BOTTLENECK_WRIST_EXPERT_SKILL_START_END_BRIDGE_PROPRIO_REVISION),
        (is_arch19, LAYERWISE_COND_BOTTLENECK_XYZ_SKILL_COND_UV_EXPERT_SKILL_DELTA_REVISION),
        (is_arch20, LAYERWISE_COND_BOTTLENECK_XYZ_SKILL_COND_UV_REVISION),
        (is_align and is_arch16, LAYERWISE_COND_BOTTLENECK_WRIST_EXPERT_SKILL_DELTA_ALIGN_REVISION),
        (is_align and is_arch17, LAYERWISE_COND_BOTTLENECK_WRIST_EXPERT_SKILL_DELTA_BRIDGE_PROPRIO_ALIGN_REVISION),
        (is_align and is_arch18, LAYERWISE_COND_BOTTLENECK_WRIST_EXPERT_SKILL_START_END_BRIDGE_PROPRIO_ALIGN_REVISION),
    ):
        if enabled:
            expected_revision = revision
    assert config.architecture_revision == expected_revision
    assert config.conditioning_route == "state_cond"
    assert config.skill_flow_enabled is (
        label not in {"arch0", "arch1", "arch2", "arch3", "arch4", "arch5", "arch6", "arch7", "arch8_1", "arch8_2", "arch9_1", "arch9_2", "arch10_1", "arch10_2", "arch11_1", "arch11_2", "arch12_1", "arch12_2", "arch13", "arch14", "arch15", "arch16", "arch17", "arch18", "arch16_align", "arch17_align", "arch18_align", "arch19", "arch20"}
    )


@pytest.mark.parametrize(
    ("label", "legacy_revision"),
    [
        ("arch8_2_skill", LAYERWISE_COND_BOTTLENECK_UV_COND_XYZ_TERMINATION_REVISION),
        (
            "arch9_2_skill",
            LAYERWISE_COND_BOTTLENECK_WRIST_SKILL_END_POSE_TERMINATION_REVISION,
        ),
    ],
)
def test_arch8_2_arch9_2_legacy_termination_revisions_still_validate(
    label: str, legacy_revision: str
) -> None:
    config = _skill_config(label)
    config.architecture_revision = legacy_revision
    config.__post_init__()


def test_removed_stage1_architectures_are_rejected() -> None:
    with pytest.raises(ValueError, match="architecture_label"):
        SkillExpertConfig(architecture_label="removed_mode")


def test_architecture_and_revision_are_fixed() -> None:
    with pytest.raises(ValueError, match="requires architecture"):
        SkillExpertConfig(architecture="removed_architecture")
    with pytest.raises(ValueError, match="architecture_revision"):
        SkillExpertConfig(architecture_revision="removed_revision")
    with pytest.raises(ValueError, match="conditioning_route"):
        SkillExpertConfig(conditioning_route="visiononly_cond")


@pytest.mark.parametrize(
    ("label", "kwargs"),
    [
        ("arch0", {"skill_flow_enabled": True, "skill_flow_max_length": 30}),
        ("arch0_skill", {}),
        ("arch0_skill_chunk", {}),
        (
            "arch1",
            {
                "architecture": FIXED_VISUAL_BOTTLENECK_ARCHITECTURE,
                "architecture_revision": FIXED_VISUAL_BOTTLENECK_REVISION,
                "vision_conditioning_mode": FIXED_BOTTLENECK_CROSS_ATTENTION,
                "skill_flow_enabled": True,
                "skill_flow_max_length": 30,
            },
        ),
    ],
)
def test_label_and_auxiliary_objective_cannot_diverge(
    label: str, kwargs: dict
) -> None:
    with pytest.raises(ValueError, match="skill_flow_enabled"):
        SkillExpertConfig(architecture_label=label, **kwargs)


def test_retained_skill_objectives_have_fixed_targets() -> None:
    with pytest.raises(ValueError, match="fixed target/state contract"):
        SkillExpertConfig(
            architecture_label="arch0_skill",
            skill_flow_enabled=True,
            skill_flow_target="extended_chunk",
            skill_flow_max_length=30,
        )
    with pytest.raises(ValueError, match="fixed target/state contract"):
        SkillExpertConfig(
            architecture_label="arch0_skill_chunk",
            skill_flow_enabled=True,
            skill_flow_target="extended_chunk",
            skill_flow_state_conditioned=True,
            skill_flow_max_length=30,
            skill_flow_chunk_multiplier=3,
        )


def test_extended_chunk_controls_dataset_action_horizon() -> None:
    config = _skill_config("arch0_skill_chunk")
    assert config.action_delta_indices == list(range(30))
    assert _skill_config("arch0").action_delta_indices == list(range(10))


def test_latent_best_of_n_is_limited_to_skill_modes() -> None:
    with pytest.raises(ValueError, match="requires skill_flow_enabled"):
        SkillExpertConfig(
            architecture_label="arch0",
            skill_flow_latent_best_of_n_enabled=True,
        )
    config = SkillExpertConfig(
        architecture_label="arch0_skill",
        skill_flow_enabled=True,
        skill_flow_target="canonical",
        skill_flow_max_length=120,
        skill_flow_latent_best_of_n_enabled=True,
        skill_flow_latent_ranking_route="main",
        skill_flow_latent_fp32=True,
    )
    assert config.skill_flow_latent_dim == 2
    assert config.skill_flow_latent_distribution == "uniform_square"


def test_pi05_mapping_has_one_stage1_path() -> None:
    assert _map_pi05_key(
        "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight"
    ) == "model.gemma_expert.model.layers.0.self_attn.q_proj.weight"
    assert _map_pi05_key("action_in_proj.weight") == "model.action_in_proj.weight"
    assert _map_pi05_key("paligemma_with_expert.paligemma.model.foo") is None
    with pytest.raises(ValueError, match="Unsupported Stage-1 architecture"):
        _map_pi05_key("action_in_proj.weight", architecture="removed_architecture")


def test_pi05_missing_keys_cover_only_current_arch0_modules() -> None:
    config = _skill_config("arch0")
    assert _allowed_pi05_missing_key("model.dino.encoder.weight", config)
    assert _allowed_pi05_missing_key("model.image_proj.weight", config)
    assert _allowed_pi05_missing_key("model.state_proj.weight", config)
    assert _allowed_pi05_missing_key("model.skill_proj.weight", config)
    assert _allowed_pi05_missing_key("model.cond_encoder.layers.0.weight", config)
    assert not _allowed_pi05_missing_key("model.visual_resampler.weight", config)


def test_arch1_pi05_missing_keys_include_only_new_visual_bridge() -> None:
    config = _skill_config("arch1")
    assert _allowed_pi05_missing_key(
        "model.visual_bottleneck_attention.in_proj_weight", config
    )
    assert _allowed_pi05_missing_key(
        "model.visual_bridge_attention.in_proj_weight", config
    )
    assert _allowed_pi05_missing_key("model.visual_state_film.weight", config)
    assert _allowed_pi05_missing_key("model.visual_bridge_gates", config)


def test_arch3_pi05_missing_keys_include_cond_and_layerwise_interface() -> None:
    config = _skill_config("arch3")
    assert _allowed_pi05_missing_key(
        "model.cond_encoder.model.layers.0.self_attn.q_proj.weight", config
    )
    assert _allowed_pi05_missing_key("model.layerwise_latent_queries", config)
    assert _allowed_pi05_missing_key(
        "model.layerwise_condition_readers.0.cross_attention.q_proj_weight", config
    )
    assert _allowed_pi05_missing_key(
        "model.visual_bridge_attention.in_proj_weight", config
    )


@pytest.mark.parametrize("label", ["arch5", "arch5_skill", "arch6", "arch6_skill"])
def test_uv_aligned_architectures_allow_new_uv_head_in_pi05_warm_start(label: str) -> None:
    config = _skill_config(label)
    for key in (
        "model.focus_uv_token_norm.weight",
        "model.focus_uv_token_norm.bias",
        "model.focus_uv_token_score.weight",
        "model.focus_uv_token_score.bias",
        "model.focus_uv_head.0.weight",
        "model.focus_uv_head.0.bias",
        "model.focus_uv_head.1.weight",
        "model.focus_uv_head.1.bias",
        "model.focus_uv_head.3.weight",
        "model.focus_uv_head.3.bias",
    ):
        assert _allowed_pi05_missing_key(key, config)
    assert not _allowed_pi05_missing_key("model.focus_uv_unknown.weight", config)


def test_other_architectures_do_not_allow_uv_head_in_pi05_warm_start() -> None:
    assert not _allowed_pi05_missing_key(
        "model.focus_uv_head.1.weight", _skill_config("arch4_skill")
    )


@pytest.mark.parametrize("label", ["arch7", "arch7_skill", "arch7_skill_chunk"])
def test_arch7_allows_only_new_xyz_head_in_pi05_warm_start(label: str) -> None:
    config = _skill_config(label)
    for key in (
        "model.end_xyz_token_norm.weight",
        "model.end_xyz_token_score.weight",
        "model.end_xyz_head.1.weight",
        "model.end_xyz_head.3.bias",
    ):
        assert _allowed_pi05_missing_key(key, config)
    assert not _allowed_pi05_missing_key("model.focus_uv_head.1.weight", config)


@pytest.mark.parametrize(
    "model_class",
    [FixedVisualBottleneckSkillExpert, LateVisualBottleneckSkillExpert],
)
def test_visual_bridge_gates_stay_fp32_during_bf16_cast(model_class) -> None:
    # Exercise the dtype-preservation hook without allocating the full Gemma.
    model = model_class.__new__(model_class)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(skill_flow_latent_fp32=False)
    model.mode_latent_mlp = None
    model.mode_latent_gain = None
    model.visual_state_film = nn.Linear(4, 8)
    model.visual_bridge_gates = nn.Parameter(torch.full((18,), 0.01))

    model.to(dtype=torch.bfloat16)

    assert model.visual_bridge_gates.dtype == torch.float32
    assert model.visual_state_film.weight.dtype == torch.float32
    before = model.visual_bridge_gates.detach().clone()
    with torch.no_grad():
        model.visual_bridge_gates.sub_(2.5e-5)
    assert torch.all(
        model.visual_bridge_gates.detach() < before
    ), "an optimizer-sized update must not be rounded away"


def test_arch2_uses_only_terminal_visual_bridge_layers() -> None:
    model = LateVisualBottleneckSkillExpert.__new__(
        LateVisualBottleneckSkillExpert
    )
    nn.Module.__init__(model)
    model.config = SimpleNamespace(visual_bridge_last_n_layers=1)
    model.gemma_expert = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(num_hidden_layers=18))
    )
    model.visual_bridge_gates = nn.Parameter(torch.full((18,), 0.01))

    assert model.visual_bridge_start_layer == 17
    assert not any(model._layer_uses_visual_bridge(i) for i in range(17))
    assert model._layer_uses_visual_bridge(17)
    assert model._active_visual_bridge_gates().shape == (1,)


@pytest.mark.parametrize(
    "model_class",
    [LayerwiseCondBottleneckSkillExpert, CoreExitLayerwiseCondBottleneckSkillExpert],
)
def test_layerwise_main_path_maps_terminal_expert_layers_to_matching_cond_latents(
    model_class,
) -> None:
    model = model_class.__new__(model_class)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(visual_bridge_last_n_layers=3)
    model._gradient_checkpointing = False

    class _IdentityNorm(nn.Module):
        def forward(self, hidden, *, cond, cond_start_index=None):
            del cond, cond_start_index
            return hidden, None

    model.gemma_expert = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=18),
            norm=_IdentityNorm(),
        )
    )
    model._action_tokens = MethodType(lambda self, actions: actions, model)
    model._sequence_geometry = MethodType(
        lambda self, hidden, gemma: (
            torch.zeros(hidden.shape[0], 1, hidden.shape[1], hidden.shape[1]),
            torch.arange(hidden.shape[1])[None],
            (hidden, hidden),
        ),
        model,
    )
    seen: list[tuple[int, float | None]] = []

    def fake_layer(
        self,
        layer_index,
        hidden,
        attention_mask,
        position_ids,
        expert_condition,
        expert_skill,
        layer_latent,
        position_embeddings,
    ):
        del (
            self,
            attention_mask,
            position_ids,
            expert_condition,
            expert_skill,
            position_embeddings,
        )
        seen.append(
            (
                layer_index,
                None if layer_latent is None else float(layer_latent.mean()),
            )
        )
        return hidden

    model._expert_layer_with_latent_bridge = MethodType(fake_layer, model)
    latents = [torch.full((1, 2, 4), float(index)) for index in range(18)]
    model._run_expert_with_layerwise_latents(
        torch.zeros(1, 2, 4),
        torch.zeros(1, 4),
        torch.zeros(1, 4),
        latents,
    )

    assert [value for _, value in seen[:15]] == [None] * 15
    assert [value for _, value in seen[15:]] == [15.0, 16.0, 17.0]


def test_arch3_narrow_interface_stays_fp32_during_bf16_cast() -> None:
    model = LayerwiseCondBottleneckSkillExpert.__new__(
        LayerwiseCondBottleneckSkillExpert
    )
    nn.Module.__init__(model)
    model.config = SimpleNamespace(skill_flow_latent_fp32=False)
    model.mode_latent_mlp = None
    model.mode_latent_gain = None
    model.layerwise_latent_queries = nn.Parameter(torch.randn(3, 4))
    model.layerwise_condition_memory_norm = nn.LayerNorm(4)
    model.layerwise_condition_readers = nn.ModuleList([nn.Linear(4, 4)])
    model.visual_bridge_query_norm = nn.LayerNorm(4)
    model.visual_bridge_attention = nn.MultiheadAttention(
        4, 1, batch_first=True
    )
    model.visual_bridge_gates = nn.Parameter(torch.full((2,), 0.01))

    model.to(dtype=torch.bfloat16)

    assert model.layerwise_latent_queries.dtype == torch.float32
    assert next(model.layerwise_condition_readers.parameters()).dtype == torch.float32
    assert next(model.visual_bridge_attention.parameters()).dtype == torch.float32
    assert model.visual_bridge_gates.dtype == torch.float32


def test_arch2_skill_route_stops_before_visual_layers() -> None:
    model = LateVisualBottleneckSkillExpert.__new__(
        LateVisualBottleneckSkillExpert
    )
    nn.Module.__init__(model)
    model.config = SimpleNamespace(visual_bridge_last_n_layers=1)
    model._gradient_checkpointing = False

    class _IdentityNorm(nn.Module):
        def forward(self, hidden, *, cond, cond_start_index=None):
            del cond, cond_start_index
            return hidden, None

    fake_expert_model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=18),
        rotary_emb=lambda hidden, position_ids: (hidden, position_ids),
        norm=_IdentityNorm(),
    )
    model.gemma_expert = SimpleNamespace(model=fake_expert_model)
    visited: list[int] = []

    def fake_layer(
        self,
        layer_index,
        hidden,
        attention_mask,
        position_ids,
        expert_condition,
        expert_skill,
        visual_tokens,
        position_embeddings,
    ):
        del (
            self,
            attention_mask,
            position_ids,
            expert_condition,
            expert_skill,
            position_embeddings,
        )
        assert visual_tokens is None
        visited.append(layer_index)
        return hidden + 1

    model._expert_layer_with_visual_bridge = MethodType(fake_layer, model)
    hidden = model._skill_only_expert_hidden(
        torch.zeros(1, 2, 4),
        torch.zeros(1, 1, 2, 2),
        torch.arange(2)[None],
        torch.zeros(1, 4),
        torch.zeros(1, 4),
    )

    assert visited == list(range(17))
    assert torch.equal(hidden, torch.full((1, 2, 4), 17.0))


@pytest.mark.parametrize("last_n", [1, 4])
def test_arch4_skill_route_stops_before_visual_layers(last_n: int) -> None:
    model = CoreExitLayerwiseCondBottleneckSkillExpert.__new__(
        CoreExitLayerwiseCondBottleneckSkillExpert
    )
    nn.Module.__init__(model)
    model.config = SimpleNamespace(visual_bridge_last_n_layers=last_n)
    model._gradient_checkpointing = False

    class _IdentityNorm(nn.Module):
        def forward(self, hidden, *, cond, cond_start_index=None):
            del cond, cond_start_index
            return hidden, None

    model.gemma_expert = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=18),
            rotary_emb=lambda hidden, position_ids: (hidden, position_ids),
            norm=_IdentityNorm(),
        )
    )
    visited: list[int] = []

    def fake_layer(
        self,
        layer_index,
        hidden,
        attention_mask,
        position_ids,
        expert_condition,
        expert_skill,
        layer_latent,
        position_embeddings,
    ):
        del (
            self,
            attention_mask,
            position_ids,
            expert_condition,
            expert_skill,
            position_embeddings,
        )
        assert layer_latent is None
        visited.append(layer_index)
        return hidden + 1

    model._expert_layer_with_latent_bridge = MethodType(fake_layer, model)
    hidden = model._skill_only_expert_hidden(
        torch.zeros(1, 2, 4),
        torch.zeros(1, 1, 2, 2),
        torch.arange(2)[None],
        torch.zeros(1, 4),
        torch.zeros(1, 4),
    )

    expected_layers = 18 - last_n
    assert visited == list(range(expected_layers))
    assert torch.equal(hidden, torch.full((1, 2, 4), float(expected_layers)))


def test_arch4_rejects_empty_skill_motion_core() -> None:
    with pytest.raises(ValueError, match="visual_bridge_last_n_layers <= 17"):
        SkillExpertConfig(
            architecture=LAYERWISE_COND_BOTTLENECK_ARCHITECTURE,
            architecture_label="arch4_skill",
            architecture_revision=LAYERWISE_COND_BOTTLENECK_CORE_EXIT_REVISION,
            vision_conditioning_mode=LAYERWISE_COND_BOTTLENECK_CROSS_ATTENTION,
            visual_bridge_last_n_layers=18,
            skill_flow_enabled=True,
            skill_flow_target="canonical",
            skill_flow_max_length=120,
        )


@pytest.mark.parametrize("tokens", [4, 100])
@pytest.mark.parametrize(
    ("model_class", "hook_name"),
    [
        (UVAlignedCoreExitLayerwiseCondBottleneckSkillExpert, "_on_final_condition_hidden"),
        (BottleneckUVAlignedCoreExitLayerwiseCondBottleneckSkillExpert, "_on_final_bottleneck_latent"),
    ],
)
def test_uv_head_reads_all_tokens(model_class: type, hook_name: str, tokens: int) -> None:
    model = model_class.__new__(model_class)
    nn.Module.__init__(model)
    model.focus_uv_token_norm = nn.LayerNorm(4)
    model.focus_uv_token_score = nn.Linear(4, 1)
    model.focus_uv_head = nn.Sequential(nn.Linear(4, 2), nn.Tanh())
    model.training = True
    latent = torch.randn(2, tokens, 4, requires_grad=True)
    getattr(model, hook_name)(latent)
    predicted = model.predict_training_focus_uv()
    assert predicted.shape == (2, 2)
    assert bool(((predicted >= -1) & (predicted <= 1)).all())
    predicted.square().sum().backward()
    assert latent.grad is not None and latent.grad.abs().sum() > 0
    with pytest.raises(RuntimeError, match="preceding training"):
        model.predict_training_focus_uv()
    model.training = False
    getattr(model, hook_name)(latent)
    assert model._final_uv_tokens is None


@pytest.mark.parametrize("tokens", [4, 100])
def test_arch7_xyz_head_reads_bottleneck_tokens(tokens: int) -> None:
    model = BottleneckXYZAlignedCoreExitLayerwiseCondBottleneckSkillExpert.__new__(
        BottleneckXYZAlignedCoreExitLayerwiseCondBottleneckSkillExpert
    )
    nn.Module.__init__(model)
    model.end_xyz_token_norm = nn.LayerNorm(4)
    model.end_xyz_token_score = nn.Linear(4, 1)
    model.end_xyz_head = nn.Linear(4, 3)
    model.training = True
    latent = torch.randn(2, tokens, 4, requires_grad=True)
    model._on_final_bottleneck_latent(latent)
    predicted = model.predict_training_end_xyz()
    assert predicted.shape == (2, 3)
    predicted.square().sum().backward()
    assert latent.grad is not None and latent.grad.abs().sum() > 0
    with pytest.raises(RuntimeError, match="preceding training"):
        model.predict_training_end_xyz()
    model.training = False
    model._on_final_bottleneck_latent(latent)
    assert model._final_xyz_tokens is None


def test_arch8_1_uv_changes_only_cond_adarms_input() -> None:
    model = UVConditionedBottleneckXYZSkillExpert.__new__(
        UVConditionedBottleneckXYZSkillExpert
    )
    nn.Module.__init__(model)
    model.state_proj = nn.Linear(4, 4)
    model.action_in_proj = nn.Linear(4, 4)
    model.focus_uv_condition = nn.Linear(2, 4, bias=False)
    state = torch.randn(2, 4)
    uv = torch.tensor([[0.0, 0.0], [0.5, -0.5]])
    conditioned = model._project_condition_state(state, uv)
    assert conditioned.shape == (2, 4)
    assert torch.allclose(conditioned[0], model._project_state(state)[0])
    assert not torch.allclose(conditioned[1], model._project_state(state)[1])
    conditioned.sum().backward()
    assert model.focus_uv_condition.weight.grad is not None
    with pytest.raises(ValueError, match="requires skill-end focus UV"):
        model._project_condition_state(state)


def test_arch13_xyz_changes_cond_adarms_and_bottleneck_predicts_uv() -> None:
    model = XYZConditionedBottleneckUVSkillExpert.__new__(
        XYZConditionedBottleneckUVSkillExpert
    )
    nn.Module.__init__(model)
    model.state_proj = nn.Linear(4, 4)
    model.action_in_proj = nn.Linear(4, 4)
    model.end_xyz_condition = nn.Linear(3, 4, bias=False)
    model.focus_uv_token_norm = nn.LayerNorm(4)
    model.focus_uv_token_score = nn.Linear(4, 1)
    model.focus_uv_head = nn.Linear(4, 2)
    model.training = True

    state = torch.randn(2, 4)
    xyz = torch.tensor([[0.0, 0.0, 0.0], [0.5, -0.5, 0.25]])
    conditioned = model._project_condition_state(state, end_pose=xyz)
    assert conditioned.shape == (2, 4)
    assert torch.allclose(conditioned[0], model._project_state(state)[0])
    assert not torch.allclose(conditioned[1], model._project_state(state)[1])

    latent = torch.randn(2, 9, 4, requires_grad=True)
    model._on_final_bottleneck_latent(latent)
    predicted_uv = model.predict_training_focus_uv()
    assert predicted_uv.shape == (2, 2)
    (conditioned.square().sum() + predicted_uv.square().sum()).backward()
    assert model.end_xyz_condition.weight.grad is not None
    assert latent.grad is not None and latent.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="requires skill-end EEF XYZ"):
        model._project_condition_state(state)


def test_arch13_warm_start_allows_only_uv_head_and_xyz_condition() -> None:
    config = _skill_config("arch13_skill")
    assert _allowed_pi05_missing_key("model.focus_uv_head.3.weight", config)
    assert _allowed_pi05_missing_key("model.end_xyz_condition.2.weight", config)
    assert not _allowed_pi05_missing_key("model.end_xyz_head.3.weight", config)


def test_arch8_2_termination_head_reads_final_cond_hidden() -> None:
    model = UVConditionedBottleneckXYZTerminationSkillExpert.__new__(
        UVConditionedBottleneckXYZTerminationSkillExpert
    )
    nn.Module.__init__(model)
    model.end_xyz_token_norm = nn.LayerNorm(4)
    model.end_xyz_token_score = nn.Linear(4, 1)
    model.end_xyz_head = nn.Linear(4, 3)
    model.termination_token_norm = nn.LayerNorm(4)
    model.termination_token_score = nn.Linear(4, 1)
    model.termination_head = nn.Linear(4, 1)
    model._termination_source = "cond"
    model.training = True
    latent = torch.randn(2, 5, 4, requires_grad=True)
    cond_hidden = torch.randn(2, 7, 4, requires_grad=True)
    model._on_final_bottleneck_latent(latent)
    model._on_final_condition_hidden(cond_hidden)
    xyz = model.predict_training_end_xyz()
    logits = model.predict_training_termination_logits()
    assert xyz.shape == (2, 3)
    assert logits.shape == (2,)
    (xyz.square().sum() + logits.square().sum()).backward()
    assert latent.grad is not None and latent.grad.abs().sum() > 0
    assert cond_hidden.grad is not None and cond_hidden.grad.abs().sum() > 0
    with pytest.raises(RuntimeError, match="preceding training"):
        model.predict_training_termination_logits()
    model.training = False
    model._on_final_condition_hidden(cond_hidden.detach())
    assert model._final_termination_tokens is None
    assert model.last_termination_probability.shape == (2,)
    assert bool(((model.last_termination_probability >= 0) & (model.last_termination_probability <= 1)).all())


def test_arch8_2_warm_start_allows_only_its_new_heads() -> None:
    config = _skill_config("arch8_2_skill")
    assert _allowed_pi05_missing_key("model.termination_head.3.weight", config)
    assert _allowed_pi05_missing_key("model.end_xyz_head.1.weight", config)
    assert not _allowed_pi05_missing_key("model.gemma_expert.model.layers.0.foo", config)


def test_arch9_1_wrist_skill_and_end_pose_are_separate_conditions() -> None:
    model = WristSkillEndPoseLayerwiseCondBottleneckSkillExpert.__new__(
        WristSkillEndPoseLayerwiseCondBottleneckSkillExpert
    )
    nn.Module.__init__(model)
    model.config = SimpleNamespace(skill_end_pose_mode="pose")
    model.state_proj = nn.Linear(4, 4, bias=False)
    model.action_in_proj = nn.Linear(4, 4, bias=False)
    model.image_proj = nn.Linear(4, 4, bias=False)
    model._image_features = lambda image: image
    model.cond_skill_condition = nn.Linear(3, 4, bias=False)
    model.end_pose_condition = nn.Linear(6, 4, bias=False)
    model.register_buffer("_fsq_levels", torch.tensor([3, 3, 3]))
    model.register_buffer("_fsq_strides", torch.tensor([1, 3, 9]))
    model.register_buffer("_fsq_half", torch.ones(3))
    model._time_condition = lambda time: torch.zeros(time.shape[0], 4)
    model._mode_latent_condition = lambda latent: None

    wrist_tokens = torch.randn(2, 5, 4)
    assert model._condition_tokens([wrist_tokens]).shape == (2, 5, 4)
    with pytest.raises(ValueError, match="only the wrist"):
        model._condition_tokens([wrist_tokens, wrist_tokens])

    state = torch.randn(2, 4)
    skill = torch.tensor([0, 26])
    cond = model._project_condition_state(state, skill_code=skill)
    assert cond.shape == (2, 4)
    assert not torch.allclose(cond, model._project_state(state))
    with pytest.raises(ValueError, match="requires skill"):
        model._project_condition_state(state)

    time = torch.ones(2)
    pose = torch.randn(2, 6)
    expert_cond = model._expert_condition(time, end_pose=pose)
    assert expert_cond.shape == (2, 4)
    assert not torch.allclose(expert_cond, model._expert_condition(time))
    (cond.sum() + expert_cond.sum()).backward()
    assert model.cond_skill_condition.weight.grad is not None
    assert model.end_pose_condition.weight.grad is not None
    with pytest.raises(ValueError, match="shape"):
        model._expert_condition(time, end_pose=pose[:, :3])


def test_arch9_1_warm_start_allows_only_its_new_conditions() -> None:
    config = _skill_config("arch9_1_skill")
    assert _allowed_pi05_missing_key("model.cond_skill_condition.2.weight", config)
    assert _allowed_pi05_missing_key("model.end_pose_condition.2.weight", config)
    assert not _allowed_pi05_missing_key("model.gemma_expert.model.layers.0.foo", config)


def test_arch9_skill_only_flow_keeps_end_pose_in_expert_adarms() -> None:
    model = WristSkillEndPoseLayerwiseCondBottleneckSkillExpert.__new__(
        WristSkillEndPoseLayerwiseCondBottleneckSkillExpert
    )
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        skill_flow_enabled=True,
        skill_flow_latent_best_of_n_enabled=False,
    )
    captured = {}
    model.sample_noise = lambda shape, device: torch.zeros(shape, device=device)
    model._action_tokens = lambda value: value
    model._skill_broadcasts = lambda skill: (
        None,
        torch.zeros(skill.shape[0], 1, device=skill.device),
    )
    def _expert_condition(time, mode_latent=None, end_pose=None):
        del mode_latent
        captured["end_pose"] = end_pose
        return torch.zeros(time.shape[0], 1, device=time.device)

    model._expert_condition = _expert_condition
    model._skill_only_expert_hidden = lambda tokens, mask, positions, condition, skill: tokens
    model._action_velocity = torch.zeros_like

    actions = torch.randn(2, 4, 3)
    end_pose = torch.randn(2, 3)
    residual = model.skill_only_flow_residual(
        actions,
        torch.tensor([0, 1]),
        torch.zeros(2, 4, dtype=torch.bool),
        time=torch.full((2,), 0.5),
        end_pose=end_pose,
    )

    assert residual.shape == actions.shape
    assert captured["end_pose"] is end_pose


def test_arch9_2_termination_head_reads_final_cond_hidden() -> None:
    model = WristSkillEndPoseTerminationSkillExpert.__new__(WristSkillEndPoseTerminationSkillExpert)
    nn.Module.__init__(model)
    model.termination_token_norm = nn.LayerNorm(4)
    model.termination_token_score = nn.Linear(4, 1)
    model.termination_head = nn.Linear(4, 1)
    model._termination_source = "cond"
    model.training = True
    cond_hidden = torch.randn(2, 5, 4, requires_grad=True)
    model._on_final_condition_hidden(cond_hidden)
    logits = model.predict_training_termination_logits()
    assert logits.shape == (2,)
    logits.square().sum().backward()
    assert cond_hidden.grad is not None and cond_hidden.grad.abs().sum() > 0
    with pytest.raises(RuntimeError, match="preceding training"):
        model.predict_training_termination_logits()
    model.training = False
    model._on_final_condition_hidden(cond_hidden.detach())
    assert model.last_termination_probability.shape == (2,)


def test_arch9_2_warm_start_allows_its_condition_and_termination_heads() -> None:
    config = _skill_config("arch9_2_skill")
    assert _allowed_pi05_missing_key("model.cond_skill_condition.2.weight", config)
    assert _allowed_pi05_missing_key("model.end_pose_condition.2.weight", config)
    assert _allowed_pi05_missing_key("model.termination_head.3.weight", config)
    assert not _allowed_pi05_missing_key("model.gemma_expert.model.layers.0.foo", config)


def test_arch10_1_cond_is_proprio_only_but_expert_keeps_end_pose() -> None:
    model = WristEndPoseLayerwiseCondBottleneckSkillExpert.__new__(
        WristEndPoseLayerwiseCondBottleneckSkillExpert
    )
    nn.Module.__init__(model)
    model.config = SimpleNamespace(skill_end_pose_mode="pose")
    model.state_proj = nn.Linear(4, 4, bias=False)
    model.action_in_proj = nn.Linear(4, 4, bias=False)
    model.end_pose_condition = nn.Linear(6, 4, bias=False)
    model._time_condition = lambda time: torch.zeros(time.shape[0], 4)
    model._mode_latent_condition = lambda latent: None

    state = torch.randn(2, 4)
    skill = torch.tensor([0, 26])
    projected = model._project_condition_state(state, skill_code=skill)
    torch.testing.assert_close(projected, model._project_state(state))

    time = torch.ones(2)
    pose = torch.randn(2, 6)
    without_pose = model._expert_condition(time)
    with_pose = model._expert_condition(time, skill_code=skill, end_pose=pose)
    assert not torch.allclose(with_pose, without_pose)
    with_pose.sum().backward()
    assert model.end_pose_condition.weight.grad is not None


def test_arch10_warm_start_excludes_arch9_cond_skill_parameters() -> None:
    config = _skill_config("arch10_1_skill")
    assert _allowed_pi05_missing_key("model.end_pose_condition.2.weight", config)
    assert not _allowed_pi05_missing_key("model.cond_skill_condition.2.weight", config)
    assert not _allowed_pi05_missing_key("model.gemma_expert.model.layers.0.foo", config)


def test_arch10_2_termination_head_reads_final_cond_hidden() -> None:
    model = WristEndPoseTerminationSkillExpert.__new__(WristEndPoseTerminationSkillExpert)
    nn.Module.__init__(model)
    model.termination_token_norm = nn.LayerNorm(4)
    model.termination_token_score = nn.Linear(4, 1)
    model.termination_head = nn.Linear(4, 1)
    model.training = True
    cond_hidden = torch.randn(2, 5, 4, requires_grad=True)
    model._on_final_condition_hidden(cond_hidden)
    logits = model.predict_training_termination_logits()
    assert logits.shape == (2,)
    logits.square().sum().backward()
    assert cond_hidden.grad is not None and cond_hidden.grad.abs().sum() > 0


def _init_cond_skill_end_pose_probe(model: nn.Module) -> None:
    model.config = SimpleNamespace(skill_end_pose_mode="pose", dtype="float32")
    model.state_proj = nn.Linear(4, 4, bias=False)
    model.action_in_proj = nn.Linear(4, 4, bias=False)
    model.cond_skill_condition = nn.Linear(3, 4, bias=False)
    model.cond_end_pose_condition = nn.Linear(6, 4, bias=False)
    model.register_buffer("_fsq_levels", torch.tensor([3, 3, 3]))
    model.register_buffer("_fsq_strides", torch.tensor([1, 3, 9]))
    model.register_buffer("_fsq_half", torch.ones(3))
    model._time_condition = lambda time: torch.zeros(time.shape[0], 4)
    model._mode_latent_condition = lambda latent: None


def test_arch11_cond_and_expert_both_receive_end_pose() -> None:
    model = WristCondSkillEndPoseLayerwiseCondBottleneckSkillExpert.__new__(
        WristCondSkillEndPoseLayerwiseCondBottleneckSkillExpert
    )
    nn.Module.__init__(model)
    _init_cond_skill_end_pose_probe(model)
    model.end_pose_condition = nn.Linear(6, 4, bias=False)

    state = torch.randn(2, 4)
    skill = torch.tensor([0, 26])
    pose = torch.randn(2, 6)
    cond = model._project_condition_state(state, skill_code=skill, end_pose=pose)
    assert not torch.allclose(cond, model._project_state(state))
    expert = model._expert_condition(torch.ones(2), skill_code=skill, end_pose=pose)
    assert not torch.allclose(expert, model._expert_condition(torch.ones(2)))
    (cond.sum() + expert.sum()).backward()
    assert model.cond_skill_condition.weight.grad is not None
    assert model.cond_end_pose_condition.weight.grad is not None
    assert model.end_pose_condition.weight.grad is not None


def test_arch12_cond_receives_end_pose_but_expert_does_not() -> None:
    model = WristCondSkillEndPoseExpertSkillLayerwiseCondBottleneckSkillExpert.__new__(
        WristCondSkillEndPoseExpertSkillLayerwiseCondBottleneckSkillExpert
    )
    nn.Module.__init__(model)
    _init_cond_skill_end_pose_probe(model)

    state = torch.randn(2, 4)
    skill = torch.tensor([0, 26])
    pose = torch.randn(2, 6)
    cond = model._project_condition_state(state, skill_code=skill, end_pose=pose)
    assert not torch.allclose(cond, model._project_state(state))
    time = torch.ones(2)
    with_pose = model._expert_condition(time, skill_code=skill, end_pose=pose)
    without_pose = model._expert_condition(time, skill_code=skill)
    torch.testing.assert_close(with_pose, without_pose)
    assert not hasattr(model, "end_pose_condition")


@pytest.mark.parametrize(
    "model_class",
    [
        WristCondSkillEndPoseTerminationSkillExpert,
        WristCondSkillEndPoseExpertSkillTerminationSkillExpert,
    ],
)
def test_arch11_arch12_termination_reads_final_cond_hidden(model_class: type) -> None:
    model = model_class.__new__(model_class)
    nn.Module.__init__(model)
    model.termination_token_norm = nn.LayerNorm(4)
    model.termination_token_score = nn.Linear(4, 1)
    model.termination_head = nn.Linear(4, 1)
    model.training = True
    cond_hidden = torch.randn(2, 5, 4, requires_grad=True)
    model._on_final_condition_hidden(cond_hidden)
    logits = model.predict_training_termination_logits()
    assert logits.shape == (2,)
    logits.square().sum().backward()
    assert cond_hidden.grad is not None and cond_hidden.grad.abs().sum() > 0


def test_arch11_arch12_warm_start_allowlists_match_expert_pose_contract() -> None:
    arch11 = _skill_config("arch11_2_skill")
    arch12 = _skill_config("arch12_2_skill")
    for config in (arch11, arch12):
        assert _allowed_pi05_missing_key("model.cond_skill_condition.2.weight", config)
        assert _allowed_pi05_missing_key("model.cond_end_pose_condition.2.weight", config)
        assert _allowed_pi05_missing_key("model.termination_head.3.weight", config)
    assert _allowed_pi05_missing_key("model.end_pose_condition.2.weight", arch11)
    assert not _allowed_pi05_missing_key("model.end_pose_condition.2.weight", arch12)


@pytest.mark.parametrize(
    "label",
    [
        "arch9_1_skill", "arch9_2_skill", "arch10_1_skill", "arch10_2_skill",
        "arch11_1_skill", "arch11_2_skill", "arch12_1_skill", "arch12_2_skill",
    ],
)
def test_arch9_only_wrist_reaches_vsa_but_predictor_keeps_two_cameras(label: str) -> None:
    policy = SkillExpertPolicy.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.register_parameter("dummy", nn.Parameter(torch.zeros(())))
    policy.config = SimpleNamespace(architecture_label=label)
    top = torch.ones(1, 3, 4, 4)
    wrist = torch.zeros(1, 3, 4, 4)
    batch = {
        "observation.images.image": top,
        "observation.images.wrist_image": wrist,
    }
    vsa_images = policy._collect_images(batch)
    predictor_images = policy._collect_images(batch, for_predictor=True)
    assert len(vsa_images) == 1 and torch.equal(vsa_images[0], wrist)
    assert len(predictor_images) == 2
    assert torch.equal(predictor_images[0], top)
    assert torch.equal(predictor_images[1], wrist)


@pytest.mark.parametrize(
    ("model_class", "expected_shape", "is_bottleneck"),
    [
        (UVAlignedCoreExitLayerwiseCondBottleneckSkillExpert, (2, 3, 6), False),
        (BottleneckUVAlignedCoreExitLayerwiseCondBottleneckSkillExpert, (2, 5, 4), True),
    ],
)
def test_uv_hook_uses_architecture_specific_final_tokens(
    model_class: type, expected_shape: tuple[int, ...], is_bottleneck: bool
) -> None:
    model = model_class.__new__(model_class)
    nn.Module.__init__(model)
    model.cond_encoder = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(num_hidden_layers=2))
    )
    model.layerwise_latent_queries = nn.Parameter(torch.zeros(5, 4))
    model._gradient_checkpointing = False
    model._vsa_debug_active = False
    model._sequence_geometry = lambda hidden, cond: (None, None, None)
    model._condition_layer_with_latent = lambda index, hidden, latent, *args: (
        hidden + 1,
        latent + index + 1,
    )

    latents = model._encode_layerwise_latents(torch.zeros(2, 3, 6), torch.zeros(2, 4))

    assert len(latents) == 2
    assert model._final_uv_tokens.shape == expected_shape
    if is_bottleneck:
        assert model._final_uv_tokens is latents[-1]


@pytest.mark.parametrize("last_n", [0, 19])
def test_arch2_rejects_invalid_visual_bridge_depth(last_n: int) -> None:
    with pytest.raises(ValueError, match="visual_bridge_last_n_layers"):
        SkillExpertConfig(
            architecture=FIXED_VISUAL_BOTTLENECK_ARCHITECTURE,
            architecture_label="arch2",
            architecture_revision=LATE_VISUAL_BOTTLENECK_REVISION,
            vision_conditioning_mode=FIXED_BOTTLENECK_CROSS_ATTENTION,
            visual_bridge_last_n_layers=last_n,
        )


def test_compact_stage1_paths_keep_fp32_master_parameters() -> None:
    # The real Gemmas are intentionally omitted: this exercises the mixed
    # precision boundary and an optimizer-sized update without allocating the
    # full policy.
    model = CondGemmaSkillExpert.__new__(CondGemmaSkillExpert)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        skill_flow_latent_fp32=False,
        min_period=0.004,
        max_period=4.0,
    )
    model.width = 4
    model.gemma_expert = nn.Linear(4, 4, bias=False)
    model.image_proj = nn.Linear(4, 4)
    model.state_proj = nn.Linear(2, 4)
    model.skill_proj = nn.Linear(3, 4)
    model.action_in_proj = nn.Linear(2, 4)
    model.action_out_proj = nn.Linear(4, 2)
    model.time_mlp_in = nn.Linear(4, 4)
    model.time_mlp_out = nn.Linear(4, 4)
    model.mode_latent_mlp = None
    model.mode_latent_gain = None

    model.to(dtype=torch.bfloat16)

    assert next(model.gemma_expert.parameters()).dtype == torch.bfloat16
    for name in (
        "image_proj",
        "state_proj",
        "skill_proj",
        "action_in_proj",
        "action_out_proj",
        "time_mlp_in",
        "time_mlp_out",
    ):
        assert next(getattr(model, name).parameters()).dtype == torch.float32

    before = model.skill_proj.weight.detach().clone()
    with torch.no_grad():
        model.skill_proj.weight.sub_(2.5e-5)
    assert torch.all(model.skill_proj.weight.detach() < before)

    state = model._project_state(torch.ones(1, 2, dtype=torch.bfloat16))
    action_tokens = model._action_tokens(
        torch.ones(1, 3, 2, dtype=torch.bfloat16)
    )
    velocity = model._action_velocity(
        torch.ones(1, 3, 4, dtype=torch.bfloat16)
    )
    time_condition = model._time_condition(torch.full((1,), 0.5))
    assert state.dtype == torch.bfloat16
    assert action_tokens.dtype == torch.bfloat16
    assert velocity.dtype == torch.float32
    assert time_condition.dtype == torch.bfloat16


def test_arch1_assigns_two_bottleneck_queries_to_each_camera() -> None:
    # Exercise camera routing without allocating DINO or Gemma.
    model = FixedVisualBottleneckSkillExpert.__new__(
        FixedVisualBottleneckSkillExpert
    )
    nn.Module.__init__(model)
    width = 4
    model.action_in_proj = nn.Linear(1, 1, bias=False)
    model.image_proj = nn.Linear(width, width, bias=False)
    with torch.no_grad():
        model.image_proj.weight.copy_(torch.eye(width))
    model.visual_camera_embedding = nn.Parameter(torch.zeros(2, width))
    model.visual_bottleneck_queries = nn.Parameter(torch.zeros(4, width))
    attention = _RecordingAttention()
    model.visual_bottleneck_attention = attention
    model.visual_bottleneck_norm = nn.Identity()
    model._image_features = MethodType(lambda self, image: image, model)
    model._code_to_zq = MethodType(
        lambda self, code: torch.zeros(code.shape[0], 3), model
    )

    top = torch.ones(2, 5, width)
    wrist = torch.full((2, 7, width), 10.0)
    result = model._condition_tokens(
        [top, wrist], batch_size=2, skill_code=torch.zeros(2, dtype=torch.long)
    )

    assert len(attention.memories) == 2
    assert attention.memories[0].shape[1] == 5
    assert attention.memories[1].shape[1] == 7
    assert torch.equal(result[:, :2], torch.ones(2, 2, width))
    assert torch.equal(result[:, 2:], torch.full((2, 2, width), 10.0))

    other_skill_result = model._condition_tokens(
        [top, wrist], batch_size=2, skill_code=torch.ones(2, dtype=torch.long)
    )
    assert torch.equal(other_skill_result, result)


def test_arch1_state_film_modulates_visual_tokens() -> None:
    model = FixedVisualBottleneckSkillExpert.__new__(
        FixedVisualBottleneckSkillExpert
    )
    nn.Module.__init__(model)
    model._vsa_debug_active = False
    model.visual_state_film = nn.Linear(2, 4, bias=False)
    with torch.no_grad():
        model.visual_state_film.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                ]
            )
        )
    tokens = torch.ones(1, 4, 2)
    state = torch.tensor([[2.0, 3.0]])

    conditioned = model._apply_visual_state_film(tokens, state)

    expected = torch.tensor([[[6.0, 6.0]]]).expand(1, 4, 2)
    assert torch.equal(conditioned, expected)
    assert model._project_expert_state(state, state) is None


def test_arch1_reports_top_and_wrist_sensitivity_separately() -> None:
    model = FixedVisualBottleneckSkillExpert.__new__(
        FixedVisualBottleneckSkillExpert
    )
    nn.Module.__init__(model)
    model._vsa_debug_active = True
    model._gradient_checkpointing = False

    def predict(self, memory, noisy_actions, state, skill, time, mode_latent):
        del self, noisy_actions, time, mode_latent
        value = memory.float().mean(dim=(1, 2))
        value = value + state.float().mean(dim=1) + skill.float()
        return value[:, None, None]

    model._predict_velocity_from_condition = MethodType(predict, model)
    memory = torch.tensor(
        [
            [[1.0], [1.0], [10.0], [10.0]],
            [[2.0], [2.0], [20.0], [20.0]],
        ]
    )
    state = torch.tensor([[3.0], [4.0]])
    skill = torch.tensor([0, 1])
    noisy_actions = torch.zeros(2, 1, 1)
    time = torch.zeros(2)
    baseline = model._predict_velocity_from_condition(
        memory, noisy_actions, state, skill, time, None
    )

    stats = model._input_sensitivity_stats(
        predicted_velocity=baseline,
        condition_tokens=memory,
        noisy_actions=noisy_actions,
        state=state,
        skill_code=skill,
        time=time,
    )

    for name in (
        "top_image_shuffle",
        "wrist_image_shuffle",
        "both_images_shuffle",
        "visual_bottleneck_shuffle",
        "state_shuffle",
        "skill_shuffle",
    ):
        assert stats[f"sensitivity/{name}/relative_output_delta"] > 0.0


@pytest.mark.parametrize("label", sorted(SUPPORTED_ARCHITECTURE_LABELS))
def test_newtask_ft_is_limited_to_core_exit_architectures(label: str) -> None:
    import dataclasses

    base = _skill_config(label)
    arch_number = int(label.removeprefix("arch").split("_")[0])
    if arch_number >= 4:
        assert dataclasses.replace(base, newtask_ft_enabled=True).newtask_ft_enabled
    else:
        with pytest.raises(ValueError, match="Arch4--Arch20"):
            dataclasses.replace(base, newtask_ft_enabled=True)


def _arch14_stub(pose_mode: str = "xyz"):
    model = XYZConditionedBottleneckUVExpertEndPoseSkillExpert.__new__(
        XYZConditionedBottleneckUVExpertEndPoseSkillExpert
    )
    nn.Module.__init__(model)
    pose_dim = 3 if pose_mode == "xyz" else 6
    model.config = SimpleNamespace(
        skill_end_pose_mode=pose_mode, min_period=4e-3, max_period=4.0,
        skill_flow_latent_best_of_n_enabled=False,
    )
    model.width = 4
    model.state_proj = nn.Linear(4, 4)
    model.time_mlp_in = nn.Linear(4, 4)
    model.time_mlp_out = nn.Linear(4, 4)
    model.action_in_proj = nn.Linear(4, 4)
    model.end_xyz_condition = nn.Linear(pose_dim, 4, bias=False)
    model.end_pose_condition = nn.Linear(pose_dim, 4, bias=False)
    model.mode_latent_mlp = None
    return model, pose_dim


@pytest.mark.parametrize("pose_mode", ["xyz", "pose"])
def test_arch14_end_pose_conditions_cond_and_expert_adarms(pose_mode: str) -> None:
    model, pose_dim = _arch14_stub(pose_mode)
    state = torch.randn(2, 4)
    pose = torch.zeros(2, pose_dim)
    pose[1] = torch.linspace(-0.5, 0.5, pose_dim)
    time = torch.tensor([0.3, 0.7])

    # Cond side: Arch13's behaviour, with the pose width following the mode.
    conditioned = model._project_condition_state(state, end_pose=pose)
    assert torch.allclose(conditioned[0], model._project_state(state)[0])
    assert not torch.allclose(conditioned[1], model._project_state(state)[1])

    # Expert side: the SAME hook serves the deployed and the skill-only route.
    goal_free = model._expert_condition(time)
    goal = model._expert_condition(time, end_pose=pose)
    assert torch.allclose(goal[0], goal_free[0])          # zero pose → no shift
    assert not torch.allclose(goal[1], goal_free[1])
    (conditioned.square().sum() + goal.square().sum()).backward()
    assert model.end_xyz_condition.weight.grad is not None
    assert model.end_pose_condition.weight.grad is not None

    with pytest.raises(ValueError, match="requires the skill-end EEF pose"):
        model._project_condition_state(state)
    with pytest.raises(ValueError, match=f"batch, {pose_dim}"):
        model._expert_condition(time, end_pose=torch.zeros(2, pose_dim + 1))


def test_arch13_expert_condition_stays_goal_free() -> None:
    """Arch14's only structural difference from Arch13 is the Expert-side pose."""
    arch14, _ = _arch14_stub()
    arch13 = XYZConditionedBottleneckUVSkillExpert.__new__(XYZConditionedBottleneckUVSkillExpert)
    nn.Module.__init__(arch13)
    arch13.config, arch13.width = arch14.config, 4
    arch13.time_mlp_in, arch13.time_mlp_out = arch14.time_mlp_in, arch14.time_mlp_out
    arch13.action_in_proj = arch14.action_in_proj
    arch13.mode_latent_mlp = None
    time, pose = torch.tensor([0.3, 0.7]), torch.ones(2, 3)
    assert torch.allclose(arch13._expert_condition(time, end_pose=pose), arch13._expert_condition(time))
    assert not torch.allclose(arch14._expert_condition(time, end_pose=pose), arch14._expert_condition(time))


def test_arch14_warm_start_allows_uv_head_and_both_pose_projections() -> None:
    config = _skill_config("arch14_skill")
    for key in ("model.focus_uv_head.3.weight", "model.end_xyz_condition.2.weight", "model.end_pose_condition.0.weight"):
        assert _allowed_pi05_missing_key(key, config)
    assert not _allowed_pi05_missing_key("model.end_xyz_head.3.weight", config)
    # Arch13 has no Expert-side pose projection to leave uninitialised.
    assert not _allowed_pi05_missing_key("model.end_pose_condition.0.weight", _skill_config("arch13_skill"))


def test_arch14_pose_modes_validate_and_arch13_stays_xyz() -> None:
    import dataclasses

    assert dataclasses.replace(_skill_config("arch14_skill"), skill_end_pose_mode="pose").skill_end_pose_mode == "pose"
    with pytest.raises(ValueError, match="skill_end_pose_mode"):
        dataclasses.replace(_skill_config("arch13_skill"), skill_end_pose_mode="pose")


def test_arch15_adds_the_skill_to_cond_adarms_and_keeps_arch14_expert_side() -> None:
    arch14, _ = _arch14_stub()
    arch15 = XYZSkillConditionedBottleneckUVExpertEndPoseSkillExpert.__new__(
        XYZSkillConditionedBottleneckUVExpertEndPoseSkillExpert
    )
    nn.Module.__init__(arch15)
    arch15.config, arch15.width = arch14.config, 4
    for name in ("state_proj", "time_mlp_in", "time_mlp_out", "action_in_proj", "end_xyz_condition", "end_pose_condition"):
        setattr(arch15, name, getattr(arch14, name))
    arch15.mode_latent_mlp = None
    arch15.cond_skill_condition = nn.Linear(3, 4, bias=False)
    coordinates = {0: [0.0, 0.0, 0.0], 1: [1.0, -1.0, 0.5]}
    arch15._code_to_zq = lambda codes: torch.tensor([coordinates[int(c)] for c in codes])

    state, pose, time = torch.randn(2, 4), torch.ones(2, 3), torch.tensor([0.3, 0.7])
    skills = torch.tensor([0, 1])
    base = arch14._project_condition_state(state, end_pose=pose)
    conditioned = arch15._project_condition_state(state, skill_code=skills, end_pose=pose)
    assert torch.allclose(conditioned[0], base[0])             # zero skill coordinates → Arch14
    assert not torch.allclose(conditioned[1], base[1])         # the skill now reaches Cond
    conditioned.square().sum().backward()
    assert arch15.cond_skill_condition.weight.grad is not None

    # Expert side, and therefore the skill-only route, is exactly Arch14's.
    assert torch.allclose(arch15._expert_condition(time, end_pose=pose), arch14._expert_condition(time, end_pose=pose))
    with pytest.raises(ValueError, match="requires the skill"):
        arch15._project_condition_state(state, end_pose=pose)


def test_arch15_warm_start_allows_the_cond_skill_projection() -> None:
    config = _skill_config("arch15_skill")
    for key in ("model.focus_uv_head.3.weight", "model.end_xyz_condition.2.weight",
                "model.end_pose_condition.0.weight", "model.cond_skill_condition.0.weight"):
        assert _allowed_pi05_missing_key(key, config)
    assert not _allowed_pi05_missing_key("model.cond_skill_condition.0.weight", _skill_config("arch14_skill"))


def _arch16_stub(cls=WristSkillDeltaGoalSkillExpert):
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        skill_end_pose_mode="xyz", min_period=4e-3, max_period=4.0,
        skill_flow_latent_best_of_n_enabled=False,
    )
    model.width = 4
    model.state_proj = nn.Linear(4, 4)
    model.time_mlp_in, model.time_mlp_out = nn.Linear(4, 4), nn.Linear(4, 4)
    model.action_in_proj = nn.Linear(4, 4)
    model.end_pose_condition = nn.Linear(3, 4, bias=False)          # Expert side
    model.cond_end_pose_condition = nn.Linear(3, 4, bias=False)     # Cond side
    model.cond_skill_condition = nn.Linear(3, 4, bias=False)
    model.mode_latent_mlp = None
    model._code_to_zq = lambda codes: torch.zeros(len(codes), 3)
    return model


def test_arch16_expert_sees_only_the_skill_displacement() -> None:
    model = _arch16_stub()
    state, time, skills = torch.randn(2, 4), torch.tensor([0.3, 0.7]), torch.tensor([0, 1])
    end = torch.tensor([[0.30, 0.10, 0.20], [0.30, 0.10, 0.20]])
    start = torch.tensor([[0.10, 0.10, 0.20], [0.25, 0.05, 0.10]])
    goal = torch.cat([end, end - start], dim=1)

    # Translating the whole skill (same displacement, other place) leaves the Expert untouched...
    shifted = torch.cat([end + 5.0, end - start], dim=1)
    assert torch.allclose(model._expert_condition(time, end_pose=goal), model._expert_condition(time, end_pose=shifted))
    # ...while Cond-Gemma, which also holds the absolute proprio, does see the absolute goal.
    assert not torch.allclose(
        model._project_condition_state(state, skill_code=skills, end_pose=goal),
        model._project_condition_state(state, skill_code=skills, end_pose=shifted),
    )
    # Same end, different start -> different Expert goal (the ambiguity Arch11 cannot resolve).
    expert = model._expert_condition(time, end_pose=goal)
    assert not torch.allclose(expert[0], expert[1] - (model._expert_condition(time)[1] - model._expert_condition(time)[0]))
    with pytest.raises(ValueError, match="batch, 6"):
        model._expert_condition(time, end_pose=end)
    with pytest.raises(ValueError, match="requires the packed"):
        model._project_condition_state(state, skill_code=skills)


def test_arch17_proprio_shifts_only_the_bridge_layers() -> None:
    model = _arch16_stub(WristSkillDeltaGoalBridgeProprioSkillExpert)
    model.config.max_state_dim = 4
    model.bridge_proprio_condition = nn.Linear(4, 4, bias=False)
    model._bridge_condition_shift = None
    model._layer_uses_visual_bridge = lambda index: index >= 2       # layers 0,1 = motion core
    seen: dict[int, torch.Tensor] = {}

    def record(self, layer_index, hidden, mask, positions, condition, *rest):
        seen[layer_index] = condition.detach().clone()
        return hidden

    parent = WristSkillDeltaGoalBridgeProprioSkillExpert.__mro__[1]
    original = parent._expert_layer_with_latent_bridge
    parent._expert_layer_with_latent_bridge = record
    try:
        state = torch.randn(2, 4)
        goal = torch.cat([torch.rand(2, 3), torch.rand(2, 3)], dim=1)
        model._project_condition_state(state, skill_code=torch.tensor([0, 1]), end_pose=goal)
        condition = model._expert_condition(torch.tensor([0.3, 0.7]), end_pose=goal)
        for layer_index in range(3):
            model._expert_layer_with_latent_bridge(layer_index, torch.zeros(2, 1, 4), None, None, condition, None, None, None)
    finally:
        parent._expert_layer_with_latent_bridge = original
    shift = model.bridge_proprio_condition(state)
    assert torch.allclose(seen[0], condition) and torch.allclose(seen[1], condition)   # skill-only prefix
    assert torch.allclose(seen[2], condition + shift)                                   # bridge layer


def test_arch16_arch17_policy_builds_the_goal_from_start_and_end_state() -> None:
    from lerobot.policies.skill_expert.configuration_skill_expert import (
        SKILL_DELTA_GOAL_ARCH_PREFIXES, WRIST_ONLY_ARCH_PREFIXES,
    )

    assert SKILL_DELTA_GOAL_ARCH_PREFIXES == ("arch16", "arch17", "arch19")
    assert {"arch16", "arch17"} <= set(WRIST_ONLY_ARCH_PREFIXES)
    policy = SimpleNamespace(config=SimpleNamespace(architecture_label="arch16_skill"))
    batch = {
        "skill_end_state": torch.tensor([[0.3, 0.1, 0.2, 9.0, 9.0, 9.0, 9.0, 9.0]]),
        "skill_start_state": torch.tensor([[0.1, 0.1, 0.0, 7.0, 7.0, 7.0, 7.0, 7.0]]),
        "skill_end_state_valid": torch.tensor([True]),
    }
    goal = SkillExpertPolicy._skill_delta_goal(policy, batch, require_valid=True)
    torch.testing.assert_close(goal, torch.tensor([[0.3, 0.1, 0.2, 0.2, 0.0, 0.2]]))
    with pytest.raises(KeyError, match="skill_start_state"):
        SkillExpertPolicy._skill_delta_goal(policy, {"skill_end_state": batch["skill_end_state"]})


def test_arch16_arch17_warm_start_and_pose_mode() -> None:
    import dataclasses

    for label in ("arch16_skill", "arch17_skill"):
        config = _skill_config(label)
        for key in ("model.cond_skill_condition.0.weight", "model.cond_end_pose_condition.0.weight",
                    "model.end_pose_condition.0.weight"):
            assert _allowed_pi05_missing_key(key, config)
        assert not _allowed_pi05_missing_key("model.focus_uv_head.3.weight", config)   # no UV head
        with pytest.raises(ValueError, match="skill_end_pose_mode must be xyz"):
            dataclasses.replace(config, skill_end_pose_mode="pose")
    assert _allowed_pi05_missing_key("model.bridge_proprio_condition.0.weight", _skill_config("arch17_skill"))
    assert not _allowed_pi05_missing_key("model.bridge_proprio_condition.0.weight", _skill_config("arch16_skill"))


def test_arch18_expert_takes_start_and_end_as_separate_absolute_inputs() -> None:
    arch18 = _arch16_stub(WristSkillStartEndGoalBridgeProprioSkillExpert)
    arch18.start_pose_condition = nn.Linear(3, 4, bias=False)
    arch11 = _arch16_stub(WristSkillDeltaGoalSkillExpert.__mro__[1])          # Arch11_1
    arch11.end_pose_condition = arch18.end_pose_condition
    arch11.time_mlp_in, arch11.time_mlp_out = arch18.time_mlp_in, arch18.time_mlp_out
    time = torch.tensor([0.3, 0.7])
    end = torch.tensor([[0.30, 0.10, 0.20], [0.30, 0.10, 0.20]])
    start = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.05, 0.10]])
    goal = torch.cat([end, start], dim=1)

    condition = arch18._expert_condition(time, end_pose=goal)
    arch11_condition = arch11._expert_condition(time, end_pose=end)
    # The end goes through Arch11_1's own projection; the start adds a separate term.
    assert torch.allclose(condition[0], arch11_condition[0])                  # zero start -> Arch11
    assert torch.allclose(condition - arch11_condition, arch18.start_pose_condition(start))
    # Unlike Arch16/Arch17 the Expert goal is absolute: translating the skill changes it.
    shifted = torch.cat([end + 5.0, start + 5.0], dim=1)
    assert not torch.allclose(condition, arch18._expert_condition(time, end_pose=shifted))
    condition.square().sum().backward()
    assert arch18.start_pose_condition.weight.grad is not None
    assert arch18.end_pose_condition.weight.grad is not None


def test_arch18_policy_packs_end_then_start_and_keeps_arch17_bridge_proprio() -> None:
    from lerobot.policies.skill_expert.configuration_skill_expert import (
        SKILL_START_CONDITIONED_ARCH_PREFIXES, WRIST_ONLY_ARCH_PREFIXES,
    )
    from lerobot.policies.skill_expert.modeling_skill_expert import _NEWTASK_FT_FROZEN_SKILL_ROUTE_MODULES

    assert SKILL_START_CONDITIONED_ARCH_PREFIXES == ("arch16", "arch17", "arch19", "arch18")
    assert "arch18" in WRIST_ONLY_ARCH_PREFIXES
    assert "start_pose_condition" in _NEWTASK_FT_FROZEN_SKILL_ROUTE_MODULES
    batch = {
        "skill_end_state": torch.tensor([[0.3, 0.1, 0.2, 9.0]]),
        "skill_start_state": torch.tensor([[0.1, 0.1, 0.0, 7.0]]),
    }
    for label, expected in (("arch18_skill", [0.3, 0.1, 0.2, 0.1, 0.1, 0.0]), ("arch17_skill", [0.3, 0.1, 0.2, 0.2, 0.0, 0.2])):
        policy = SimpleNamespace(config=SimpleNamespace(architecture_label=label))
        torch.testing.assert_close(SkillExpertPolicy._skill_delta_goal(policy, batch), torch.tensor([expected]))
    assert issubclass(WristSkillStartEndGoalBridgeProprioSkillExpert, WristSkillDeltaGoalBridgeProprioSkillExpert)
    config = _skill_config("arch18_skill")
    for key in ("model.start_pose_condition.0.weight", "model.bridge_proprio_condition.0.weight",
                "model.end_pose_condition.0.weight", "model.cond_end_pose_condition.0.weight"):
        assert _allowed_pi05_missing_key(key, config)
    assert not _allowed_pi05_missing_key("model.start_pose_condition.0.weight", _skill_config("arch17_skill"))


def _arch19_stub():
    """Arch15 stub (Arch14 projections + Cond skill) typed as Arch19."""
    arch14, _ = _arch14_stub()
    arch19 = XYZSkillConditionedBottleneckUVExpertSkillDeltaSkillExpert.__new__(
        XYZSkillConditionedBottleneckUVExpertSkillDeltaSkillExpert
    )
    nn.Module.__init__(arch19)
    arch19.config, arch19.width = arch14.config, 4
    for name in ("state_proj", "time_mlp_in", "time_mlp_out", "action_in_proj", "end_xyz_condition", "end_pose_condition"):
        setattr(arch19, name, getattr(arch14, name))
    arch19.mode_latent_mlp = None
    arch19.cond_skill_condition = nn.Linear(3, 4, bias=False)
    arch19._code_to_zq = lambda codes: torch.tensor([[1.0, -1.0, 0.5]] * len(codes))
    return arch19, arch14


def test_arch19_keeps_arch15_cond_and_gives_the_expert_only_the_displacement() -> None:
    arch19, arch14 = _arch19_stub()
    state, time, skills = torch.randn(2, 4), torch.tensor([0.3, 0.7]), torch.tensor([0, 1])
    end = torch.tensor([[0.30, 0.10, 0.20], [0.30, 0.10, 0.20]])
    start = torch.tensor([[0.10, 0.10, 0.20], [0.25, 0.05, 0.10]])
    goal = torch.cat([end, end - start], dim=1)

    # Cond side == Arch15 fed the absolute skill-end xyz.
    arch15 = XYZSkillConditionedBottleneckUVExpertEndPoseSkillExpert.__new__(
        XYZSkillConditionedBottleneckUVExpertEndPoseSkillExpert
    )
    nn.Module.__init__(arch15)
    arch15.config, arch15.width = arch19.config, 4
    for name in ("state_proj", "action_in_proj", "end_xyz_condition", "cond_skill_condition", "_code_to_zq"):
        setattr(arch15, name, getattr(arch19, name))
    torch.testing.assert_close(
        arch19._project_condition_state(state, skill_code=skills, end_pose=goal),
        arch15._project_condition_state(state, skill_code=skills, end_pose=end),
    )
    # Expert side == Arch14's end-pose projection applied to the displacement, so translating the
    # whole skill leaves it unchanged while Cond still sees the new absolute goal.
    torch.testing.assert_close(arch19._expert_condition(time, end_pose=goal), arch14._expert_condition(time, end_pose=end - start))
    shifted = torch.cat([end + 5.0, end - start], dim=1)
    torch.testing.assert_close(arch19._expert_condition(time, end_pose=goal), arch19._expert_condition(time, end_pose=shifted))
    assert not torch.allclose(
        arch19._project_condition_state(state, skill_code=skills, end_pose=goal),
        arch19._project_condition_state(state, skill_code=skills, end_pose=shifted),
    )
    torch.testing.assert_close(arch19._expert_condition(time), arch14._expert_condition(time))   # goal-free call
    with pytest.raises(ValueError, match="batch, 6"):
        arch19._expert_condition(time, end_pose=end)
    with pytest.raises(ValueError, match="requires the packed"):
        arch19._project_condition_state(state, skill_code=skills)


def test_arch20_is_arch13_plus_the_cond_skill_with_a_goal_free_expert() -> None:
    arch14, _ = _arch14_stub()
    arch20 = XYZSkillConditionedBottleneckUVSkillExpert.__new__(XYZSkillConditionedBottleneckUVSkillExpert)
    nn.Module.__init__(arch20)
    arch20.config, arch20.width = arch14.config, 4
    for name in ("state_proj", "time_mlp_in", "time_mlp_out", "action_in_proj", "end_xyz_condition"):
        setattr(arch20, name, getattr(arch14, name))
    arch20.mode_latent_mlp = None
    arch20.cond_skill_condition = nn.Linear(3, 4, bias=False)
    coordinates = {0: [0.0, 0.0, 0.0], 1: [1.0, -1.0, 0.5]}
    arch20._code_to_zq = lambda codes: torch.tensor([coordinates[int(c)] for c in codes])
    state, pose, time, skills = torch.randn(2, 4), torch.ones(2, 3), torch.tensor([0.3, 0.7]), torch.tensor([0, 1])

    # Cond: Arch13 (proprio + skill-end xyz) plus the skill, like Arch15.
    base = arch14._project_condition_state(state, end_pose=pose)
    conditioned = arch20._project_condition_state(state, skill_code=skills, end_pose=pose)
    assert torch.allclose(conditioned[0], base[0]) and not torch.allclose(conditioned[1], base[1])
    # Expert: no goal at all, whatever pose the caller passes.
    torch.testing.assert_close(arch20._expert_condition(time, end_pose=pose), arch20._expert_condition(time))
    assert not hasattr(arch20, "end_pose_condition")
    with pytest.raises(ValueError, match="requires the skill"):
        arch20._project_condition_state(state, end_pose=pose)


def _align_stub(cls=WristSkillDeltaGoalAlignSkillExpert, *, width=8, bottleneck=6):
    """The alignment mixin alone, without building a whole Gemma stack."""
    from lerobot.policies.skill_expert.wrist_patch_alignment import WristPatchAlignmentHead

    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.width = width
    model.wrist_patch_align_head = WristPatchAlignmentHead(bottleneck, width, width=4)
    model._final_patch_tokens = None
    model._final_patch_query = None
    return model


def test_the_align_labels_are_their_base_architecture_with_one_extra_head() -> None:
    from lerobot.policies.skill_expert.configuration_skill_expert import (
        SKILL_START_END_GOAL_ARCH_PREFIXES, WRIST_ONLY_ARCH_PREFIXES,
    )

    for base, cls in (
        ("arch16", WristSkillDeltaGoalAlignSkillExpert),
        ("arch17", WristSkillDeltaGoalBridgeProprioAlignSkillExpert),
        ("arch18", WristSkillStartEndGoalBridgeProprioAlignSkillExpert),
    ):
        label = f"{base}_align_skill"
        config = _skill_config(label)
        # Same wrist-only contract and same goal packing as the base label...
        assert label.startswith(WRIST_ONLY_ARCH_PREFIXES) and config.skill_end_pose_mode == "xyz"
        assert config.trains_wrist_patch_alignment and not _skill_config(f"{base}_skill").trains_wrist_patch_alignment
        assert label.startswith(SKILL_START_END_GOAL_ARCH_PREFIXES) is (base == "arch18")
        # ...but its own revision, because the checkpoint carries the extra head.
        assert config.architecture_revision.endswith("_align_v1")
        assert config.architecture_revision != _skill_config(f"{base}_skill").architecture_revision
        assert _allowed_pi05_missing_key("model.wrist_patch_align_head.query_proj.weight", config)
        assert not _allowed_pi05_missing_key(
            "model.wrist_patch_align_head.query_proj.weight", _skill_config(f"{base}_skill")
        )
    assert issubclass(WristSkillStartEndGoalBridgeProprioAlignSkillExpert, WristSkillStartEndGoalBridgeProprioSkillExpert)
    assert issubclass(WristSkillDeltaGoalBridgeProprioAlignSkillExpert, WristSkillDeltaGoalBridgeProprioSkillExpert)
    assert issubclass(WristSkillDeltaGoalAlignSkillExpert, WristSkillDeltaGoalSkillExpert)


def test_the_align_head_scores_the_wrist_patches_only_while_training() -> None:
    model = _align_stub()
    # The condition sequence is [wrist CLS, 196 patches]; the query is the bottleneck latent.
    hidden = torch.randn(2, 197, 8)
    latent = torch.randn(2, 5, 6)
    model._on_final_condition_hidden(hidden)
    model._on_final_bottleneck_latent(latent)
    logits = model.predict_training_wrist_patch_logits()
    assert logits.shape == (2, 196)                       # the CLS token is not a patch
    with pytest.raises(RuntimeError, match="preceding training"):
        model.predict_training_wrist_patch_logits()       # consumed, so a stale stash cannot leak

    model.training = False                                # .eval() would need the real DINO stack
    model._on_final_condition_hidden(hidden)
    model._on_final_bottleneck_latent(latent)
    with pytest.raises(RuntimeError, match="preceding training"):
        model.predict_training_wrist_patch_logits()       # nothing is stashed at inference


def test_the_policy_turns_the_raw_pose_and_goal_into_a_patch_loss() -> None:
    from lerobot.policies.skill_expert.wrist_patch_alignment import wrist_patch_alignment_loss
    from lerobot.policies.skill_expert.wrist_patch_target import WristCamera, patch_labels

    logits = torch.zeros(2, 196)                          # a chance-level head
    policy = SimpleNamespace(
        config=SimpleNamespace(
            architecture_label="arch18_align_skill", dino_image_size=224,
            wrist_patch_align_target_sigma=0.7,
        ),
        model=SimpleNamespace(predict_training_wrist_patch_logits=lambda: logits),
    )
    # Frame 0: the goal is 25 cm down the gripper's own axis, so it is in view.
    # Frame 1: the goal IS the gripper, the one answer the head could give without looking.
    state = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.1, -0.2, 0.3, 0.4, 0.1, 0.2, 0.0, 0.0]])
    batch = {
        "skill_decoder_state": state,
        "skill_end_state": torch.stack([
            state[0, :3] + torch.tensor([0.0, 0.0, 0.25]), state[1, :3]
        ]),
    }
    loss, metrics = SkillExpertPolicy._wrist_patch_alignment_loss(policy, batch, top_k=1)
    assert metrics["wrist_patch/valid_fraction"] == pytest.approx(0.5)     # the gripper frame is dropped
    assert float(loss) == pytest.approx(1.0, abs=1e-5)                     # ln(196) normalized

    cell, _, valid = patch_labels(
        state[:, :3], state[:, 3:6], batch["skill_end_state"], WristCamera(),
        grid=14, height=224, width=224,
    )
    assert valid.tolist() == [True, False]
    expected, _ = wrist_patch_alignment_loss(logits, cell, valid, grid=14, sigma=0.7)
    assert float(loss) == pytest.approx(float(expected), abs=1e-6)

    for missing in ("skill_decoder_state", "skill_end_state"):
        with pytest.raises(KeyError, match=missing):
            SkillExpertPolicy._wrist_patch_alignment_loss(
                policy, {k: v for k, v in batch.items() if k != missing}, top_k=1
            )


def test_arch19_arch20_contracts_warm_start_and_goal_packing() -> None:
    import dataclasses

    from lerobot.policies.skill_expert.configuration_skill_expert import (
        EXPERT_END_POSE_XYZ_COND_UV_ARCH_PREFIXES, SKILL_COND_XYZ_COND_UV_ARCH_PREFIXES,
        WRIST_ONLY_ARCH_PREFIXES, XYZ_COND_UV_ARCH_PREFIXES, is_arch2_label,
    )
    from lerobot.policies.skill_expert.modeling_skill_expert import _default_architecture_revision

    for label in ("arch19", "arch20"):
        assert label in XYZ_COND_UV_ARCH_PREFIXES and label in SKILL_COND_XYZ_COND_UV_ARCH_PREFIXES
        assert label not in WRIST_ONLY_ARCH_PREFIXES and not is_arch2_label(label)
    assert "arch19" in EXPERT_END_POSE_XYZ_COND_UV_ARCH_PREFIXES
    assert "arch20" not in EXPERT_END_POSE_XYZ_COND_UV_ARCH_PREFIXES
    assert is_arch2_label("arch2_skill") and not is_arch2_label("arch20_skill")
    assert _default_architecture_revision("arch20_skill", LAYERWISE_COND_BOTTLENECK_ARCHITECTURE) == LAYERWISE_COND_BOTTLENECK_XYZ_SKILL_COND_UV_REVISION
    assert _default_architecture_revision("arch19_skill", LAYERWISE_COND_BOTTLENECK_ARCHITECTURE) == LAYERWISE_COND_BOTTLENECK_XYZ_SKILL_COND_UV_EXPERT_SKILL_DELTA_REVISION

    arch19, arch20 = _skill_config("arch19_skill"), _skill_config("arch20_skill")
    for config in (arch19, arch20):
        for key in ("model.focus_uv_head.3.weight", "model.end_xyz_condition.2.weight", "model.cond_skill_condition.0.weight"):
            assert _allowed_pi05_missing_key(key, config)
        with pytest.raises(ValueError, match="skill_end_pose_mode"):
            dataclasses.replace(config, skill_end_pose_mode="pose")
        with pytest.raises(ValueError, match="original"):
            dataclasses.replace(config, foveated_vision_enabled=True)
    assert _allowed_pi05_missing_key("model.end_pose_condition.0.weight", arch19)
    assert not _allowed_pi05_missing_key("model.end_pose_condition.0.weight", arch20)

    # Arch19 packs [end, end - start] (Arch16's goal); Arch20 keeps Arch13's plain skill-end xyz.
    batch = {
        "skill_end_state": torch.tensor([[0.3, 0.1, 0.2, 9.0]]),
        "skill_start_state": torch.tensor([[0.1, 0.1, 0.0, 7.0]]),
        "skill_end_xyz": torch.tensor([[0.3, 0.1, 0.2]]),
    }
    policy = SimpleNamespace(config=SimpleNamespace(architecture_label="arch19_skill"))
    torch.testing.assert_close(SkillExpertPolicy._skill_delta_goal(policy, batch), torch.tensor([[0.3, 0.1, 0.2, 0.2, 0.0, 0.2]]))
    policy = SimpleNamespace(config=SimpleNamespace(architecture_label="arch20_skill", skill_end_pose_mode="xyz"))
    torch.testing.assert_close(SkillExpertPolicy._xyz_cond_end_pose(policy, batch), batch["skill_end_xyz"])


def test_cli_off_end_state_mode_survives_yaml_parsing():
    import draccus

    # draccus reads CLI values as YAML, where a bare `off` is the boolean False.
    config = draccus.parse(config_class=SkillExpertConfig, args=["--skill_predictor_end_state_mode=off"])
    assert config.skill_predictor_end_state_mode == "off"
