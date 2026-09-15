import pytest

from lerobot.policies.skill_expert.configuration_skill_expert import (
    COND_GEMMA_ARCHITECTURE,
    COND_GEMMA_ARCHITECTURE_REVISION,
    FIXED_BOTTLENECK_CROSS_ATTENTION,
    FIXED_VISUAL_BOTTLENECK_ARCHITECTURE,
    FIXED_VISUAL_BOTTLENECK_REVISION,
    INTERLEAVED_CROSS_ATTENTION,
    SUPPORTED_ARCHITECTURE_LABELS,
    SkillExpertConfig,
)
from lerobot.policies.skill_expert.modeling_skill_expert import (
    _allowed_pi05_missing_key,
    _map_pi05_key,
)


def _skill_config(label: str) -> SkillExpertConfig:
    if label not in SUPPORTED_ARCHITECTURE_LABELS:
        raise AssertionError(label)
    is_arch1 = label.startswith("arch1")
    kwargs = {
        "architecture": (
            FIXED_VISUAL_BOTTLENECK_ARCHITECTURE
            if is_arch1
            else COND_GEMMA_ARCHITECTURE
        ),
        "architecture_label": label,
        "architecture_revision": (
            FIXED_VISUAL_BOTTLENECK_REVISION
            if is_arch1
            else COND_GEMMA_ARCHITECTURE_REVISION
        ),
        "vision_conditioning_mode": (
            FIXED_BOTTLENECK_CROSS_ATTENTION
            if is_arch1
            else INTERLEAVED_CROSS_ATTENTION
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
    return SkillExpertConfig(**kwargs)


@pytest.mark.parametrize("label", sorted(SUPPORTED_ARCHITECTURE_LABELS))
def test_only_retained_stage1_architectures_validate(label: str) -> None:
    config = _skill_config(label)

    is_arch1 = label.startswith("arch1")
    assert config.architecture == (
        FIXED_VISUAL_BOTTLENECK_ARCHITECTURE
        if is_arch1
        else COND_GEMMA_ARCHITECTURE
    )
    assert config.architecture_revision == (
        FIXED_VISUAL_BOTTLENECK_REVISION
        if is_arch1
        else COND_GEMMA_ARCHITECTURE_REVISION
    )
    assert config.conditioning_route == "state_cond"
    assert config.skill_flow_enabled is (label not in {"arch0", "arch1"})


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
    assert _allowed_pi05_missing_key("model.visual_bridge_gates", config)
