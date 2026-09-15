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
    SUPPORTED_ARCHITECTURE_LABELS,
    SkillExpertConfig,
)
from lerobot.policies.skill_expert.fixed_visual_bottleneck import (
    FixedVisualBottleneckSkillExpert,
)
from lerobot.policies.skill_expert.modeling_skill_expert import (
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
    assert _allowed_pi05_missing_key("model.visual_state_film.weight", config)
    assert _allowed_pi05_missing_key("model.visual_bridge_gates", config)


def test_arch1_visual_bridge_gates_stay_fp32_during_bf16_cast() -> None:
    # Exercise the dtype-preservation hook without allocating the full Gemma.
    model = FixedVisualBottleneckSkillExpert.__new__(
        FixedVisualBottleneckSkillExpert
    )
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


def test_arch1_assigns_two_bottleneck_queries_to_each_camera() -> None:
    # Exercise camera routing without allocating DINO or Gemma.
    model = FixedVisualBottleneckSkillExpert.__new__(
        FixedVisualBottleneckSkillExpert
    )
    nn.Module.__init__(model)
    width = 4
    model.action_in_proj = nn.Linear(1, 1, bias=False)
    model.image_proj = nn.Identity()
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
