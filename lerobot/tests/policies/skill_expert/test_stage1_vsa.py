import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding

from lerobot.policies.skill_expert.configuration_skill_expert import (
    COND_GEMMA_ARCHITECTURE,
    COND_GEMMA_ARCHITECTURE_REVISION,
    COND_GEMMA_DUAL_STATE_REVISION,
    COND_GEMMA_EXPERT_TOKENS_REVISION,
    COND_GEMMA_EXPERT_STATE_REVISION,
    COND_GEMMA_ISOLATED_SKILL_TOKEN_REVISION,
    COND_GEMMA_PERCEIVER_EXPERT_TOKENS_REVISION,
    COND_GEMMA_SEPARATE_DUAL_STATE_REVISION,
    COND_GEMMA_SKILL_ADARMS_REVISION,
    COND_GEMMA_SKILL_TOKEN_REVISION,
    COND_GEMMA_WRIST_DUAL_STATE_REVISION,
    COMPRESSED_VISUAL_KV_REVISION,
    COMPRESSED_VISUAL_KV_SELF_ATTENTION,
    GLOBAL_VISUAL_ADARMS,
    IN_CONTEXT_TOKENS,
    INTERLEAVED_CROSS_ATTENTION,
    UNCOMPRESSED_VISUAL_KV_REVISION,
    UNCOMPRESSED_VISUAL_KV_SELF_ATTENTION,
    VSA_ARCHITECTURE,
    VSA_ARCHITECTURE_REVISION,
    SkillExpertConfig,
)
from lerobot.policies.skill_expert.cond_gemma import (
    CondGemmaSkillExpert,
    expert_token_attention_contract,
)
from lerobot.policies.skill_expert.modeling_skill_expert import (
    SkillExpertPolicy,
    SkillExpertPytorch,
    _allowed_pi05_missing_key,
    _map_pi05_key,
)
from lerobot.policies.skill_expert.legacy_vsa_eval import (
    LegacyResidualSA18VSAActionExpert,
    LegacyVSAActionExpert,
)
from lerobot.policies.skill_expert.vsa_perceiver_crossattn import (
    CameraPerceiverResampler,
    InterleavedExpertBlock,
    VSAActionExpert,
)
from lerobot.policies.pi_gemma import (
    PiGemmaForCausalLM,
    PiGemmaRMSNorm,
    _gated_residual,
)


def _tiny_gemma_config(depth: int = 4):
    config = CONFIG_MAPPING["gemma"](
        head_dim=8,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_hidden_layers=depth,
        num_key_value_heads=1,
        vocab_size=128,
        hidden_activation="gelu_pytorch_tanh",
        attention_bias=False,
    )
    config._attn_implementation = "eager"  # noqa: SLF001
    return config


def test_token_selective_adarms_matches_expanded_mask_math_and_gradients() -> None:
    torch.manual_seed(7)
    norm = PiGemmaRMSNorm(8, cond_dim=8)
    with torch.no_grad():
        norm.dense.weight.normal_(std=0.05)
        norm.dense.bias.normal_(std=0.05)
    hidden = torch.randn(2, 6, 8, requires_grad=True)
    update = torch.randn(2, 6, 8, requires_grad=True)
    condition = torch.randn(2, 8, requires_grad=True)
    start = 3

    optimized_norm, compact_gate = norm(
        hidden, condition, cond_start_index=start
    )
    optimized = _gated_residual(hidden, update, compact_gate)

    modulation = norm.dense(condition).unsqueeze(1)
    scale, shift, gate = modulation.chunk(3, dim=-1)
    mask = torch.zeros(2, 6, 1)
    mask[:, start:] = 1
    reference_norm = (
        norm._norm(hidden) * (1 + scale * mask) + shift * mask
    )
    reference_gate = gate * mask + (1 - mask)
    reference = hidden + update * reference_gate

    assert torch.allclose(optimized_norm, reference_norm, atol=1e-6, rtol=1e-6)
    assert torch.allclose(optimized, reference, atol=1e-6, rtol=1e-6)

    optimized_grads = torch.autograd.grad(
        optimized.square().mean(),
        (hidden, update, condition, norm.dense.weight, norm.dense.bias),
        retain_graph=True,
    )
    reference_grads = torch.autograd.grad(
        reference.square().mean(),
        (hidden, update, condition, norm.dense.weight, norm.dense.bias),
    )
    for optimized_grad, reference_grad in zip(
        optimized_grads, reference_grads, strict=True
    ):
        assert torch.allclose(
            optimized_grad, reference_grad, atol=1e-6, rtol=1e-6
        )


def _position_embeddings(config, context, actions):
    hidden = torch.cat((context, actions), dim=1)
    positions = torch.arange(hidden.shape[1])[None].expand(hidden.shape[0], -1)
    return GemmaRotaryEmbedding(config)(hidden, positions)


def test_legacy_eval_expert_preserves_alternating_checkpoint_layout() -> None:
    config = _tiny_gemma_config(depth=4)
    expert = LegacyVSAActionExpert(
        config,
        include_state_in_visual_crossattn=True,
        include_skill_in_visual_crossattn=True,
    ).eval()

    assert hasattr(expert.blocks[0], "attention")
    assert not hasattr(expert.blocks[0], "self_attention")
    assert expert.blocks[0].cross_attention is False
    assert expert.blocks[1].cross_attention is True
    output = expert(
        torch.randn(2, 2, 32),
        torch.randn(2, 5, 32),
        torch.randn(2, 16, 32),
        torch.randn(2, 32),
    )
    assert output.shape == (2, 5, 32)


def test_historical_residual_eval_expert_preserves_sa18_layout() -> None:
    expert = LegacyResidualSA18VSAActionExpert(
        _tiny_gemma_config(depth=4),
        vision_conditioning_mode="residual_cross_attention",
        include_state_in_visual_crossattn=True,
        include_skill_in_visual_crossattn=True,
    ).eval()

    assert expert.blocks[0].self_attention is not None
    assert expert.blocks[1].self_attention is not None
    assert expert.blocks[1].visual_cross_attention is not None
    assert expert.blocks[1].visual_residual_gate is not None
    output = expert(
        torch.randn(2, 2, 32),
        torch.randn(2, 5, 32),
        torch.randn(2, 16, 32),
        torch.randn(2, 32),
    )
    assert output.shape == (2, 5, 32)


def test_config_defaults_to_vsa_and_cond_architecture_is_explicit() -> None:
    config = SkillExpertConfig()

    assert config.architecture == VSA_ARCHITECTURE == "vsa_perceiver_crossattn"
    assert config.vision_conditioning_mode == INTERLEAVED_CROSS_ATTENTION
    assert config.include_state_in_visual_crossattn is True
    assert config.include_skill_in_visual_crossattn is True
    assert config.action_expert_variant == "gemma_300m"
    assert config.dino_lr_scale == 0.1
    assert config.num_visual_latents_per_camera == 32
    assert config.visual_perceiver_width == 1024
    assert config.n_action_steps == 5
    assert config.vsa_debug_schedule == ()
    assert config.scheduler_mode == "cosine_decay"
    assert config.scheduler_warmup_steps == 1_000
    assert SkillExpertConfig(vsa_debug_schedule=[1, 100]).vsa_debug_schedule == (1, 100)
    assert config.conditioning_route == "state_skill_cond"
    assert config.cond_encoder_variant == "gemma_300m"
    assert config.freeze_vision_encoder is False
    cond = SkillExpertConfig(
        architecture=COND_GEMMA_ARCHITECTURE,
        architecture_label="arch0",
        architecture_revision=COND_GEMMA_ARCHITECTURE_REVISION,
        conditioning_route="state_skill_cond",
    )
    assert cond.architecture == "cond_gemma"
    assert cond.architecture_label == "arch0"
    for revision, label in (
        (COND_GEMMA_EXPERT_STATE_REVISION, "arch0_1"),
        (COND_GEMMA_DUAL_STATE_REVISION, "arch0_2"),
        (COND_GEMMA_SEPARATE_DUAL_STATE_REVISION, "arch0_2_sep"),
        (COND_GEMMA_WRIST_DUAL_STATE_REVISION, "arch0_3"),
        (COND_GEMMA_SKILL_ADARMS_REVISION, "arch0_adarms"),
        (COND_GEMMA_SKILL_TOKEN_REVISION, "arch0_token"),
        (COND_GEMMA_ISOLATED_SKILL_TOKEN_REVISION, "arch0_token_iso"),
    ):
        assert SkillExpertConfig(
            architecture=COND_GEMMA_ARCHITECTURE,
            architecture_label=label,
            architecture_revision=revision,
            conditioning_route="state_cond",
        ).architecture_label == label
    assert SkillExpertConfig(
        architecture=COND_GEMMA_ARCHITECTURE,
        architecture_label="arch1_1",
        architecture_revision=COND_GEMMA_EXPERT_TOKENS_REVISION,
    ).architecture_label == "arch1_1"
    assert SkillExpertConfig(
        architecture=COND_GEMMA_ARCHITECTURE,
        architecture_label="arch1_2",
        architecture_revision=COND_GEMMA_PERCEIVER_EXPERT_TOKENS_REVISION,
    ).architecture_label == "arch1_2"
    assert SkillExpertConfig(
        architecture_label="arch1_3",
        architecture_revision=UNCOMPRESSED_VISUAL_KV_REVISION,
        vision_conditioning_mode=UNCOMPRESSED_VISUAL_KV_SELF_ATTENTION,
        num_visual_latents_per_camera=197,
    ).architecture_label == "arch1_3"
    assert SkillExpertConfig(
        architecture_label="arch2_1",
        architecture_revision=COMPRESSED_VISUAL_KV_REVISION,
        vision_conditioning_mode=COMPRESSED_VISUAL_KV_SELF_ATTENTION,
    ).architecture_label == "arch2_1"
    assert SkillExpertConfig(
        architecture_label="arch2",
        architecture_revision=VSA_ARCHITECTURE_REVISION,
        vision_conditioning_mode=INTERLEAVED_CROSS_ATTENTION,
    ).architecture_label == "arch2"
    with pytest.raises(ValueError, match="architecture must be"):
        SkillExpertConfig(architecture="state_skill_cond")
    with pytest.raises(ValueError, match="vision_conditioning_mode must be one of"):
        SkillExpertConfig(vision_conditioning_mode="unknown")

    with pytest.raises(ValueError, match="sorted and contain no duplicates"):
        SkillExpertConfig(vsa_debug_schedule=(100, 1, 100))
    with pytest.raises(ValueError, match="scheduler_mode must be"):
        SkillExpertConfig(scheduler_mode="unknown")


def test_policy_loader_rejects_cond_checkpoint_when_vsa_is_requested(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"type": "skill_expert", "conditioning_route": "state_cond"})
    )

    with pytest.raises(ValueError, match="checkpoint architecture mismatch"):
        SkillExpertPolicy.from_pretrained(tmp_path, config=SkillExpertConfig())


def test_policy_loader_rejects_historical_vsa_checkpoint_without_eval_contract(
    tmp_path,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "type": "skill_expert",
                "architecture": "vsa_perceiver_crossattn",
            }
        )
    )

    with pytest.raises(ValueError, match="does not match the current VSA revision"):
        SkillExpertPolicy.from_pretrained(tmp_path)


def test_policy_loader_rejects_explicit_cross_mode_checkpoint_override(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "type": "skill_expert",
                "architecture": "vsa_perceiver_crossattn",
                "architecture_revision": VSA_ARCHITECTURE_REVISION,
                "vision_conditioning_mode": "in_context_tokens",
            }
        )
    )

    with pytest.raises(ValueError, match="Cross-mode checkpoint conversion"):
        SkillExpertPolicy.from_pretrained(
            tmp_path,
            config=SkillExpertConfig(
                vision_conditioning_mode="global_visual_adarms"
            ),
        )


def test_policy_loader_rejects_cross_revision_cond_checkpoint(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "type": "skill_expert",
                "architecture": COND_GEMMA_ARCHITECTURE,
                "architecture_revision": COND_GEMMA_EXPERT_TOKENS_REVISION,
                "architecture_label": "arch1_1",
                "conditioning_route": "state_skill_cond",
            }
        )
    )
    requested = SkillExpertConfig(
        architecture=COND_GEMMA_ARCHITECTURE,
        architecture_label="arch1_2",
        architecture_revision=COND_GEMMA_PERCEIVER_EXPERT_TOKENS_REVISION,
    )
    with pytest.raises(ValueError, match="architecture_revision mismatch"):
        SkillExpertPolicy.from_pretrained(tmp_path, config=requested)


def test_perceiver_shape_and_camera_parameters_are_separate() -> None:
    torch.manual_seed(0)
    top = CameraPerceiverResampler(
        24, expert_width=32, perceiver_width=16, num_latents=8
    ).eval()
    wrist = CameraPerceiverResampler(
        24, expert_width=32, perceiver_width=16, num_latents=8
    ).eval()
    image_a = torch.randn(2, 197, 24)
    image_b = torch.randn(2, 197, 24)

    top_a, wrist_b = top(image_a), wrist(image_b)
    swapped = torch.cat((top(image_b), wrist(image_a)), dim=1)

    assert top_a.shape == wrist_b.shape == (2, 8, 32)
    assert torch.cat((top_a, wrist_b), dim=1).shape == (2, 16, 32)
    assert top.latents is not wrist.latents
    assert not torch.equal(torch.cat((top_a, wrist_b), dim=1), swapped)


def test_direct_width_perceiver_has_no_channel_projection_parameters() -> None:
    resampler = CameraPerceiverResampler(
        1024, expert_width=1024, perceiver_width=1024, num_latents=32
    )

    assert isinstance(resampler.input_proj, nn.Identity)
    assert isinstance(resampler.output_proj, nn.Identity)
    assert not any("input_proj" in key or "output_proj" in key for key in resampler.state_dict())


def test_self_attention_mask_has_required_direction() -> None:
    mask = VSAActionExpert.self_attention_mask(2, 5, "cpu")

    assert mask.shape == (2, 1, 7, 7)
    assert torch.all(mask[:, :, :2, :2] == 0)
    assert torch.all(mask[:, :, :2, 2:] < -1e20)
    assert torch.all(mask[:, :, 2:, :] == 0)


def test_in_context_mask_and_continuous_position_ids() -> None:
    mask = VSAActionExpert.in_context_attention_mask(
        batch_size=2, visual_tokens=4, action_tokens=3, device="cpu"
    )

    assert mask.shape == (2, 1, 9, 9)
    assert torch.all(mask[:, :, :4, :4] == 0)
    assert torch.all(mask[:, :, :4, 4:] < -1e20)
    assert torch.all(mask[:, :, 4:6, :6] == 0)
    assert torch.all(mask[:, :, 4:6, 6:] < -1e20)
    assert torch.all(mask[:, :, 6:, :] == 0)
    valid = torch.tensor([[True, True, False, True, True]])
    positions = VSAActionExpert.position_ids_from_valid_mask(valid)
    assert positions.tolist() == [[0, 1, 1, 2, 3]]

    visual_kv_mask = VSAActionExpert.visual_kv_attention_mask(
        batch_size=2, visual_tokens=4, action_tokens=3, device="cpu"
    )
    assert visual_kv_mask.shape == (2, 1, 5, 9)
    assert torch.all(visual_kv_mask[:, :, :2, :6] == 0)
    assert torch.all(visual_kv_mask[:, :, :2, 6:] < -1e20)
    assert torch.all(visual_kv_mask[:, :, 2:, :] == 0)


def test_cond_expert_token_mask_has_fixed_three_block_contract() -> None:
    mask, positions = expert_token_attention_contract(
        batch_size=2, visual_tokens=4, action_tokens=3, device="cpu"
    )

    assert mask.shape == (2, 1, 9, 9)
    # Visual cannot read state, skill, or actions.
    assert torch.all(mask[:, :, :4, :4] == 0)
    assert torch.all(mask[:, :, :4, 4:] < -1e20)
    # State and skill are one bidirectional block and cannot read actions.
    assert torch.all(mask[:, :, 4:6, :6] == 0)
    assert torch.all(mask[:, :, 4:6, 6:] < -1e20)
    # Actions read visual + state + skill + the entire action block.
    assert torch.all(mask[:, :, 6:, :] == 0)
    assert positions[0].tolist() == list(range(9))


class _TinyDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(hidden_size=32, num_register_tokens=0)

    def forward(self, image):
        pooled = image.mean(dim=(1, 2, 3), keepdim=True).reshape(-1, 1, 1)
        hidden = (pooled * self.scale).expand(-1, 197, 32)
        return SimpleNamespace(last_hidden_state=hidden)


def _tiny_projection_gemma(*, use_adarms: bool):
    config = _tiny_gemma_config(depth=2)
    config.use_adarms = use_adarms
    config.adarms_cond_dim = 32 if use_adarms else None
    model = PiGemmaForCausalLM(config)
    model.model.embed_tokens = None
    model.lm_head = None
    return model


def _tiny_projection_gemma_pi05_heads(*, use_adarms: bool):
    config = CONFIG_MAPPING["gemma"](
        head_dim=4,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=8,
        num_hidden_layers=2,
        num_key_value_heads=1,
        vocab_size=128,
        hidden_activation="gelu_pytorch_tanh",
        attention_bias=False,
    )
    config._attn_implementation = "eager"  # noqa: SLF001
    config.use_adarms = use_adarms
    config.adarms_cond_dim = 32 if use_adarms else None
    model = PiGemmaForCausalLM(config)
    model.model.embed_tokens = None
    model.lm_head = None
    return model


@pytest.mark.parametrize(
    ("revision", "label", "visual_tokens"),
    [
        (COND_GEMMA_EXPERT_TOKENS_REVISION, "arch1_1", 394),
        (COND_GEMMA_PERCEIVER_EXPERT_TOKENS_REVISION, "arch1_2", 8),
    ],
)
def test_cond_expert_token_forward_and_cached_sampling_preserve_action_shape(
    revision: str, label: str, visual_tokens: int
) -> None:
    config = SkillExpertConfig(
        architecture=COND_GEMMA_ARCHITECTURE,
        architecture_label=label,
        architecture_revision=revision,
        max_state_dim=4,
        max_action_dim=4,
        chunk_size=3,
        n_action_steps=3,
        cumulative_xyz_loss_enabled=True,
        dino_model_path="unused",
        num_visual_latents_per_camera=4,
        visual_perceiver_width=32,
    )
    tiny_geometry = SimpleNamespace(width=32, depth=2)
    with (
        patch(
            "lerobot.policies.skill_expert.cond_gemma.get_gemma_config",
            return_value=tiny_geometry,
        ),
        patch(
            "lerobot.policies.skill_expert.cond_gemma.build_gemma",
            side_effect=lambda _variant, *, use_adarms: _tiny_projection_gemma(
                use_adarms=use_adarms
            ),
        ),
        patch(
            "lerobot.policies.skill_expert.cond_gemma.AutoModel.from_pretrained",
            return_value=_TinyDino(),
        ),
    ):
        model = CondGemmaSkillExpert(config).eval()

    images = [torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16)]
    state = torch.randn(1, 4)
    skill = torch.tensor([3])
    actions = torch.randn(1, 3, 4)
    noise = torch.randn_like(actions)
    time = torch.full((1,), 0.5)

    condition = model._condition_tokens(images)
    assert condition.shape == (1, visual_tokens, 32)
    assert model._expert_context_tokens(state, skill).shape == (1, 2, 32)
    residual = model(images, state, skill, actions, noise=noise, time=time)
    assert model._last_predicted_actions is not None
    assert model._last_predicted_actions.shape == actions.shape
    sampled = model.sample_actions(
        images, state, skill, noise=noise, num_steps=1
    )
    assert residual.shape == sampled.shape == actions.shape
    if label == "arch1_2":
        model.train()
        model.gradient_checkpointing_enable()
        checkpointed = model(
            images, state, skill, actions, noise=noise, time=time
        )
        checkpointed.square().mean().backward()
        assert model.action_in_proj.weight.grad is not None


@pytest.mark.parametrize(
    ("revision", "label"),
    [
        (COND_GEMMA_ARCHITECTURE_REVISION, "arch0"),
        (COND_GEMMA_EXPERT_STATE_REVISION, "arch0_1"),
        (COND_GEMMA_DUAL_STATE_REVISION, "arch0_2"),
        (COND_GEMMA_SEPARATE_DUAL_STATE_REVISION, "arch0_2_sep"),
        (COND_GEMMA_WRIST_DUAL_STATE_REVISION, "arch0_3"),
        (COND_GEMMA_SKILL_ADARMS_REVISION, "arch0_adarms"),
        (COND_GEMMA_SKILL_TOKEN_REVISION, "arch0_token"),
        (COND_GEMMA_ISOLATED_SKILL_TOKEN_REVISION, "arch0_token_iso"),
        (COND_GEMMA_EXPERT_TOKENS_REVISION, "arch1_1"),
        (COND_GEMMA_PERCEIVER_EXPERT_TOKENS_REVISION, "arch1_2"),
    ],
)
def test_cond_architectures_collect_scheduled_debug_and_input_influence(
    revision: str, label: str
) -> None:
    config = SkillExpertConfig(
        architecture=COND_GEMMA_ARCHITECTURE,
        architecture_label=label,
        architecture_revision=revision,
        conditioning_route=(
            "state_cond"
            if label.startswith("arch0")
            else "state_skill_cond"
        ),
        max_state_dim=4,
        max_action_dim=4,
        chunk_size=3,
        n_action_steps=3,
        dino_model_path="unused",
        num_visual_latents_per_camera=4,
        visual_perceiver_width=32,
        vsa_debug_schedule=(1,),
    )
    tiny_geometry = SimpleNamespace(width=32, depth=2)
    with (
        patch(
            "lerobot.policies.skill_expert.cond_gemma.get_gemma_config",
            return_value=tiny_geometry,
        ),
        patch(
            "lerobot.policies.skill_expert.cond_gemma.build_gemma",
            side_effect=lambda _variant, *, use_adarms: _tiny_projection_gemma_pi05_heads(
                use_adarms=use_adarms
            ),
        ),
        patch(
            "lerobot.policies.skill_expert.cond_gemma.AutoModel.from_pretrained",
            return_value=_TinyDino(),
        ),
    ):
        model = CondGemmaSkillExpert(config).train()

    model.set_training_step(1)
    residual = model(
        [torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)],
        torch.randn(2, 4),
        torch.tensor([3, 5]),
        torch.randn(2, 3, 4),
        noise=torch.randn(2, 3, 4),
        time=torch.tensor([0.3, 0.7]),
    )
    stats = model._last_vsa_debug_stats

    assert "visual/top_latents/effective_rank_fraction" in stats
    assert "activation/flow_prediction_rms" in stats
    assert "sensitivity/top_image_shuffle/relative_output_delta" in stats
    assert "sensitivity/wrist_image_shuffle/relative_output_delta" in stats
    assert "sensitivity/state_shuffle/relative_output_delta" in stats
    assert "sensitivity/skill_shuffle/relative_output_delta" in stats

    residual.square().mean().backward()
    policy = SkillExpertPolicy.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.config = config
    policy.model = model
    gradient_metrics = policy.training_debug_metrics()
    assert "vsa_debug/gradient/preclip/conditioner_grad_rms" in gradient_metrics
    assert "vsa_debug/gradient/preclip/expert_grad_rms" in gradient_metrics
    assert model._vsa_debug_active is False


def test_arch0_routes_state_to_cond_and_skill_to_expert_broadcast() -> None:
    config = SkillExpertConfig(
        architecture=COND_GEMMA_ARCHITECTURE,
        architecture_label="arch0",
        architecture_revision=COND_GEMMA_ARCHITECTURE_REVISION,
        conditioning_route="state_cond",
        max_state_dim=4,
        max_action_dim=4,
        dino_model_path="unused",
    )
    tiny_geometry = SimpleNamespace(width=32, depth=2)
    with (
        patch(
            "lerobot.policies.skill_expert.cond_gemma.get_gemma_config",
            return_value=tiny_geometry,
        ),
        patch(
            "lerobot.policies.skill_expert.cond_gemma.build_gemma",
            side_effect=lambda _variant, *, use_adarms: _tiny_projection_gemma_pi05_heads(
                use_adarms=use_adarms
            ),
        ),
        patch(
            "lerobot.policies.skill_expert.cond_gemma.AutoModel.from_pretrained",
            return_value=_TinyDino(),
        ),
    ):
        model = CondGemmaSkillExpert(config).eval()

    state_condition = model._state_condition(torch.randn(2, 4))
    condition_skill, expert_skill = model._skill_broadcasts(torch.tensor([3, 5]))

    assert state_condition.shape == (2, 32)
    assert condition_skill is None
    assert expert_skill is not None
    assert expert_skill.shape == (2, 32)


def test_arch0_adarms_replaces_skill_broadcast_with_normalized_expert_adarms() -> None:
    torch.manual_seed(19)
    config = SkillExpertConfig(
        architecture=COND_GEMMA_ARCHITECTURE,
        # Mixed case survives because architecture_label is lowercased.
        architecture_label="arch0_adaRMS",
        architecture_revision=COND_GEMMA_SKILL_ADARMS_REVISION,
        conditioning_route="state_cond",
        max_state_dim=4,
        max_action_dim=4,
        chunk_size=3,
        n_action_steps=3,
        dino_model_path="unused",
    )
    assert config.architecture_label == "arch0_adarms"
    tiny_geometry = SimpleNamespace(width=32, depth=2)
    with (
        patch(
            "lerobot.policies.skill_expert.cond_gemma.get_gemma_config",
            return_value=tiny_geometry,
        ),
        patch(
            "lerobot.policies.skill_expert.cond_gemma.build_gemma",
            side_effect=lambda _variant, *, use_adarms: _tiny_projection_gemma_pi05_heads(
                use_adarms=use_adarms
            ),
        ),
        patch(
            "lerobot.policies.skill_expert.cond_gemma.AutoModel.from_pretrained",
            return_value=_TinyDino(),
        ),
    ):
        model = CondGemmaSkillExpert(config).eval()

    state = torch.randn(2, 4)
    skill = torch.tensor([3, 5])
    time = torch.tensor([0.3, 0.7])

    # State keeps the Arch0 Cond-Gemma AdaRMS path untouched.
    assert model.uses_cond_state_adarms is True
    assert model.uses_expert_state_adarms is False
    assert model._state_condition(state).shape == (2, 32)
    # Skill leaves the layerwise broadcast on both streams.
    assert model._skill_broadcasts(skill) == (None, None)

    skill_condition = model._expert_skill_condition(skill)
    assert skill_condition.shape == (2, 32)
    # The condition is time + skill, with no state term.
    assert torch.allclose(
        model._expert_condition(time, None, skill),
        model._time_condition(time) + skill_condition,
    )
    # PiGemmaRMSNorm pins the skill term at unit RMS so it cannot swamp the
    # timestep embedding at initialization.
    assert torch.allclose(
        skill_condition.square().mean(dim=-1).sqrt(),
        torch.ones(2),
        atol=1e-5,
    )
    # FSQ333 code 13 is the grid centre, whose _code_to_zq is the zero vector.
    # Without the norm it would condition far more weakly than a corner code.
    centre = model._expert_skill_condition(torch.tensor([13]))
    corner = model._expert_skill_condition(torch.tensor([0]))
    assert torch.allclose(
        torch.cat((centre, corner)).square().mean(dim=-1).sqrt(),
        torch.ones(2),
        atol=1e-5,
    )
    assert not torch.allclose(centre, corner)
    assert not torch.allclose(skill_condition[0], skill_condition[1])

    images = [torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)]
    actions = torch.randn(2, 3, 4)
    noise = torch.randn_like(actions)
    residual = model(images, state, skill, actions, noise=noise, time=time)
    sampled = model.sample_actions(images, state, skill, noise=noise, num_steps=1)
    assert residual.shape == sampled.shape == actions.shape

    # AdaRMS dense weights are zero-init, so every AdaRMS signal -- timestep
    # included -- starts with exactly zero effect on the output. Unlike the
    # Arch0 broadcast, Arch0_adaRMS skill therefore only begins to act once
    # those denses move off zero.
    other = model.sample_actions(
        images, state, torch.tensor([7, 11]), noise=noise, num_steps=1
    )
    assert torch.allclose(sampled, other)

    with torch.no_grad():
        for module in model.gemma_expert.modules():
            if isinstance(module, PiGemmaRMSNorm) and module.dense is not None:
                module.dense.weight.normal_(std=0.05)
    trained_like = model.sample_actions(
        images, state, skill, noise=noise, num_steps=1
    )
    trained_like_other = model.sample_actions(
        images, state, torch.tensor([7, 11]), noise=noise, num_steps=1
    )
    assert not torch.allclose(trained_like, trained_like_other)

    model.train()
    model.gradient_checkpointing_enable()
    model(images, state, skill, actions, noise=noise, time=time).square().mean().backward()
    assert model.skill_proj.weight.grad is not None
    assert model.expert_skill_norm.weight.grad is not None


def test_arch0_token_makes_skill_one_expert_token_and_keeps_state_on_cond_adarms() -> None:
    torch.manual_seed(23)
    config = SkillExpertConfig(
        architecture=COND_GEMMA_ARCHITECTURE,
        architecture_label="arch0_token",
        architecture_revision=COND_GEMMA_SKILL_TOKEN_REVISION,
        conditioning_route="state_cond",
        max_state_dim=4,
        max_action_dim=4,
        chunk_size=3,
        n_action_steps=3,
        dino_model_path="unused",
    )
    tiny_geometry = SimpleNamespace(width=32, depth=2)
    with (
        patch(
            "lerobot.policies.skill_expert.cond_gemma.get_gemma_config",
            return_value=tiny_geometry,
        ),
        patch(
            "lerobot.policies.skill_expert.cond_gemma.build_gemma",
            side_effect=lambda _variant, *, use_adarms: _tiny_projection_gemma_pi05_heads(
                use_adarms=use_adarms
            ),
        ),
        patch(
            "lerobot.policies.skill_expert.cond_gemma.AutoModel.from_pretrained",
            return_value=_TinyDino(),
        ),
    ):
        model = CondGemmaSkillExpert(config).eval()

    state = torch.randn(2, 4)
    skill = torch.tensor([3, 5])

    # Skill is the only context token; state stays on the Arch0 Cond-Gemma path.
    assert model.uses_expert_context_tokens is True
    assert model.uses_expert_state_token is False
    assert model.n_expert_context_tokens == 1
    assert model.uses_cond_state_adarms is True
    assert model.cond_encoder.model.config.use_adarms is True
    assert model._state_condition(state).shape == (2, 32)
    # No skill broadcast and no skill AdaRMS on this route.
    assert model._skill_broadcasts(skill) == (None, None)
    assert model.uses_expert_skill_adarms is False
    # Arch1_1's state token machinery must not be allocated here.
    assert model.state_norm is None
    assert model.state_proj is not None

    context_tokens = model._expert_context_tokens(state, skill)
    assert context_tokens.shape == (2, 1, 32)
    # The token is RMS-normalized like Arch1_1's, and distinct per FSQ code.
    assert torch.allclose(
        context_tokens.square().mean(dim=-1).sqrt(),
        torch.ones(2, 1),
        atol=1e-5,
    )
    assert not torch.allclose(context_tokens[0], context_tokens[1])

    # [visual | skill | actions]: the skill token reads visual + itself, and the
    # action block reads everything before it.
    mask, positions = expert_token_attention_contract(1, 4, 3, "cpu", 1)
    assert mask.shape == (1, 1, 8, 8)
    visible = mask[0, 0] == 0.0
    # Visual tokens see only each other.
    assert visible[0].tolist() == [True] * 4 + [False] * 4
    # The skill token adds itself to the visual prefix, and nothing more.
    assert visible[4].tolist() == [True] * 5 + [False] * 3
    # Every action token sees visual + skill + the whole bidirectional chunk.
    for action_row in (5, 6, 7):
        assert visible[action_row].tolist() == [True] * 8
    assert positions[0].tolist() == list(range(8))

    images = [torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)]
    other_images = [torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)]
    # Unlike Arch0_token_iso, this skill token is contextualized by the scene:
    # its cached key differs once the images change.
    assert model.uses_isolated_skill_token is False
    skill_keys = [
        model._visual_context_cache(
            model._condition_tokens(frames, batch_size=2),
            context_tokens,
            model._state_condition(state),
        ).layers[1].keys[:, :, -1, :]
        for frames in (images, other_images)
    ]
    assert not torch.allclose(skill_keys[0], skill_keys[1])

    actions = torch.randn(2, 3, 4)
    noise = torch.randn_like(actions)
    time = torch.tensor([0.3, 0.7])
    residual = model(images, state, skill, actions, noise=noise, time=time)
    sampled = model.sample_actions(images, state, skill, noise=noise, num_steps=1)
    assert residual.shape == sampled.shape == actions.shape
    # An in-context token enters attention directly, so unlike the zero-init
    # AdaRMS of Arch0_adaRMS it already moves the output at initialization.
    other = model.sample_actions(
        images, state, torch.tensor([7, 11]), noise=noise, num_steps=1
    )
    assert not torch.allclose(sampled, other)

    model.train()
    model.gradient_checkpointing_enable()
    model(images, state, skill, actions, noise=noise, time=time).square().mean().backward()
    assert model.skill_proj.weight.grad is not None
    assert model.skill_norm.weight.grad is not None
    assert model.state_proj.weight.grad is not None


def test_arch0_token_iso_hides_vision_from_the_skill_token_in_both_paths() -> None:
    torch.manual_seed(29)
    config = SkillExpertConfig(
        architecture=COND_GEMMA_ARCHITECTURE,
        architecture_label="arch0_token_iso",
        architecture_revision=COND_GEMMA_ISOLATED_SKILL_TOKEN_REVISION,
        conditioning_route="state_cond",
        max_state_dim=4,
        max_action_dim=4,
        chunk_size=3,
        n_action_steps=3,
        dino_model_path="unused",
    )
    tiny_geometry = SimpleNamespace(width=32, depth=2)
    with (
        patch(
            "lerobot.policies.skill_expert.cond_gemma.get_gemma_config",
            return_value=tiny_geometry,
        ),
        patch(
            "lerobot.policies.skill_expert.cond_gemma.build_gemma",
            side_effect=lambda _variant, *, use_adarms: _tiny_projection_gemma_pi05_heads(
                use_adarms=use_adarms
            ),
        ),
        patch(
            "lerobot.policies.skill_expert.cond_gemma.AutoModel.from_pretrained",
            return_value=_TinyDino(),
        ),
    ):
        model = CondGemmaSkillExpert(config).eval()

    assert model.uses_isolated_skill_token is True
    assert model.n_expert_context_tokens == 1

    mask, _ = expert_token_attention_contract(1, 4, 3, "cpu", 1, False)
    visible = mask[0, 0] == 0.0
    # The skill token now sees only itself: no vision, and still no actions.
    assert visible[4].tolist() == [False] * 4 + [True] + [False] * 3
    # Actions keep full visibility of vision and skill.
    for action_row in (5, 6, 7):
        assert visible[action_row].tolist() == [True] * 8

    state = torch.randn(2, 4)
    skill = torch.tensor([3, 5])
    context_tokens = model._expert_context_tokens(state, skill)
    actions = torch.randn(2, 3, 4)
    noise = torch.randn_like(actions)
    time = torch.tensor([0.3, 0.7])
    images_a = [torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)]
    images_b = [torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)]

    # The cached prefix is what the action stream reads, so the skill half of it
    # must be identical under different images -- that is the isolation claim,
    # and it must hold on the inference path too.
    def skill_prefix(images):
        condition_tokens = model._condition_tokens(images, batch_size=2)
        cache = model._visual_context_cache(
            condition_tokens, context_tokens, model._state_condition(state)
        )
        # (batch, kv_heads, prefix, head_dim); the skill token is the last one.
        return [layer.keys[:, :, -1, :].clone() for layer in cache.layers]

    isolated_a = skill_prefix(images_a)
    isolated_b = skill_prefix(images_b)
    assert len(isolated_a) == 2
    for first, second in zip(isolated_a, isolated_b, strict=True):
        assert torch.allclose(first, second, atol=1e-6)

    residual = model(images_a, state, skill, actions, noise=noise, time=time)
    sampled = model.sample_actions(images_a, state, skill, noise=noise, num_steps=1)
    assert residual.shape == sampled.shape == actions.shape
    other = model.sample_actions(
        images_a, state, torch.tensor([7, 11]), noise=noise, num_steps=1
    )
    assert not torch.allclose(sampled, other)
    # Vision still reaches the actions directly, just not through the skill token.
    assert not torch.allclose(
        sampled,
        model.sample_actions(images_b, state, skill, noise=noise, num_steps=1),
    )


@pytest.mark.parametrize(
    ("revision", "label", "cond_state", "wrist_only", "separate_projection"),
    [
        (COND_GEMMA_EXPERT_STATE_REVISION, "arch0_1", False, False, False),
        (COND_GEMMA_DUAL_STATE_REVISION, "arch0_2", True, False, False),
        (
            COND_GEMMA_SEPARATE_DUAL_STATE_REVISION,
            "arch0_2_sep",
            True,
            False,
            True,
        ),
        (COND_GEMMA_WRIST_DUAL_STATE_REVISION, "arch0_3", True, True, False),
    ],
)
def test_arch0_state_location_ablations_match_training_and_cached_inference(
    revision: str,
    label: str,
    cond_state: bool,
    wrist_only: bool,
    separate_projection: bool,
) -> None:
    config = SkillExpertConfig(
        architecture=COND_GEMMA_ARCHITECTURE,
        architecture_label=label,
        architecture_revision=revision,
        conditioning_route="state_cond",
        max_state_dim=4,
        max_action_dim=4,
        chunk_size=3,
        n_action_steps=3,
        cumulative_xyz_loss_enabled=True,
        dino_model_path="unused",
    )
    tiny_geometry = SimpleNamespace(width=32, depth=2)
    with (
        patch(
            "lerobot.policies.skill_expert.cond_gemma.get_gemma_config",
            return_value=tiny_geometry,
        ),
        patch(
            "lerobot.policies.skill_expert.cond_gemma.build_gemma",
            side_effect=lambda _variant, *, use_adarms: _tiny_projection_gemma_pi05_heads(
                use_adarms=use_adarms
            ),
        ),
        patch(
            "lerobot.policies.skill_expert.cond_gemma.AutoModel.from_pretrained",
            return_value=_TinyDino(),
        ),
    ):
        model = CondGemmaSkillExpert(config).eval()

    images = [torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16)]
    state = torch.randn(1, 4)
    skill = torch.tensor([3])
    actions = torch.randn(1, 3, 4)
    noise = torch.randn_like(actions)
    time = torch.tensor([0.5])
    condition_tokens = model._condition_tokens(images)
    projected_state = model._project_state(state)
    expert_projected_state = model._project_expert_state(
        state, projected_state
    )

    assert model.cond_encoder.model.config.use_adarms is cond_state
    assert (model._state_condition(state) is not None) is cond_state
    assert (model.expert_state_proj is not None) is separate_projection
    if separate_projection:
        assert expert_projected_state is not projected_state
        assert model.expert_state_proj is not model.state_proj
        assert torch.allclose(expert_projected_state, projected_state)
    else:
        assert expert_projected_state is projected_state
    expert_condition = model._expert_condition(time, expert_projected_state)
    assert torch.allclose(
        expert_condition,
        model._time_condition(time) + expert_projected_state,
    )
    state_start_index = model._condition_state_start_index(condition_tokens)
    if wrist_only:
        assert state_start_index == 197
    else:
        assert state_start_index is None

    residual = model(images, state, skill, actions, noise=noise, time=time)
    assert model._last_predicted_actions is not None
    assert model._last_predicted_actions.shape == actions.shape
    sampled = model.sample_actions(
        images, state, skill, noise=noise, num_steps=1
    )
    assert residual.shape == sampled.shape == actions.shape
    if separate_projection:
        model.train()
        residual = model(images, state, skill, actions, noise=noise, time=time)
        residual.square().mean().backward()
        assert model.state_proj.weight.grad is not None
        assert model.expert_state_proj.weight.grad is not None
    if wrist_only:
        model.train()
        model.gradient_checkpointing_enable()
        checkpointed = model(
            images, state, skill, actions, noise=noise, time=time
        )
        checkpointed.square().mean().backward()
        assert model.action_in_proj.weight.grad is not None


@pytest.mark.parametrize(
    "mode",
    [
        UNCOMPRESSED_VISUAL_KV_SELF_ATTENTION,
        COMPRESSED_VISUAL_KV_SELF_ATTENTION,
        INTERLEAVED_CROSS_ATTENTION,
        IN_CONTEXT_TOKENS,
        GLOBAL_VISUAL_ADARMS,
    ],
)
def test_vision_modes_preserve_action_shape_and_instantiate_only_used_modules(
    mode: str,
) -> None:
    torch.manual_seed(11)
    expert = VSAActionExpert(
        _tiny_gemma_config(depth=4), vision_conditioning_mode=mode
    ).eval()
    context = torch.randn(2, 2, 32)
    actions = torch.randn(2, 5, 32)
    memory = torch.randn(2, 8, 32)
    output = expert(context, actions, memory, torch.randn(2, 32))

    assert output.shape == actions.shape
    cross_layers = [block for block in expert.blocks if block.cross_attention]
    if mode == INTERLEAVED_CROSS_ATTENTION:
        assert len(cross_layers) == 2
        assert expert.last_sequence_length == 7
    else:
        assert not cross_layers
        assert not any(
            "visual_cross_attention" in key for key in expert.state_dict()
        )
        expected_length = (
            15
            if mode
            in {
                UNCOMPRESSED_VISUAL_KV_SELF_ATTENTION,
                COMPRESSED_VISUAL_KV_SELF_ATTENTION,
                IN_CONTEXT_TOKENS,
            }
            else 7
        )
        assert expert.last_sequence_length == expected_length
    assert expert.last_position_ids[0].tolist() == list(
        range(expert.last_sequence_length)
    )


@pytest.mark.parametrize(
    "mode",
    [
        UNCOMPRESSED_VISUAL_KV_SELF_ATTENTION,
        COMPRESSED_VISUAL_KV_SELF_ATTENTION,
    ],
)
def test_visual_kv_self_attention_reuses_fixed_memory_without_new_parameters(
    mode: str,
) -> None:
    torch.manual_seed(111)
    config = _tiny_gemma_config(depth=3)
    expert = VSAActionExpert(config, vision_conditioning_mode=mode).eval()
    all_self_attention = VSAActionExpert(
        config, vision_conditioning_mode=GLOBAL_VISUAL_ADARMS
    ).eval()
    assert expert.state_dict().keys() == all_self_attention.state_dict().keys()
    assert sum(p.numel() for p in expert.parameters()) == sum(
        p.numel() for p in all_self_attention.parameters()
    )

    context = torch.randn(2, 2, 32)
    actions = torch.randn(2, 5, 32)
    memory = torch.randn(2, 8, 32, requires_grad=True)
    seen_memory = []
    query_lengths = []
    key_lengths = []
    handles = [
        block.register_forward_pre_hook(
            lambda _module, args: seen_memory.append(args[2])
        )
        for block in expert.blocks
    ]
    handles.extend(
        [
            expert.blocks[0].self_attention.q_proj.register_forward_pre_hook(
                lambda _module, args: query_lengths.append(args[0].shape[1])
            ),
            expert.blocks[0].self_attention.k_proj.register_forward_pre_hook(
                lambda _module, args: key_lengths.append(args[0].shape[1])
            ),
        ]
    )
    output = expert(context, actions, memory, torch.randn(2, 32))
    for handle in handles:
        handle.remove()
    output.square().mean().backward()

    assert output.shape == actions.shape
    assert len(seen_memory) == 3
    assert all(item is memory for item in seen_memory)
    assert query_lengths == [7]
    assert key_lengths == [15]
    assert memory.grad is not None and memory.grad.abs().sum() > 0


def test_in_context_visual_tokens_receive_gradients_without_cross_attention() -> None:
    torch.manual_seed(12)
    expert = VSAActionExpert(
        _tiny_gemma_config(depth=2), vision_conditioning_mode=IN_CONTEXT_TOKENS
    )
    context = torch.randn(2, 2, 32, requires_grad=True)
    actions = torch.randn(2, 5, 32, requires_grad=True)
    memory = torch.randn(2, 8, 32, requires_grad=True)
    output = expert(context, actions, memory, torch.randn(2, 32))
    output.square().mean().backward()

    assert memory.grad is not None and memory.grad.abs().sum() > 0
    assert all(block.visual_cross_attention is None for block in expert.blocks)


def test_in_context_sequence_order_is_visual_state_skill_then_actions() -> None:
    expert = VSAActionExpert(
        _tiny_gemma_config(depth=1), vision_conditioning_mode=IN_CONTEXT_TOKENS
    ).eval()
    context = torch.randn(2, 2, 32)
    actions = torch.randn(2, 5, 32)
    memory = torch.randn(2, 8, 32)
    captured_context = []
    handle = expert.blocks[0].self_attention_norm.register_forward_pre_hook(
        lambda _module, args: captured_context.append(args[0].detach().clone())
    )

    output = expert(context, actions, memory, torch.randn(2, 32))
    handle.remove()

    torch.testing.assert_close(captured_context[0], torch.cat((memory, context), dim=1))
    assert output.shape == actions.shape


def test_global_condition_reaches_every_action_adarms() -> None:
    expert = VSAActionExpert(
        _tiny_gemma_config(depth=2),
        vision_conditioning_mode=GLOBAL_VISUAL_ADARMS,
    ).eval()
    condition = torch.randn(2, 32)
    captured = []
    handles = []
    for block in expert.blocks:
        handles.append(
            block.self_attention_norm.register_forward_pre_hook(
                lambda _module, args: captured.append(args[2])
            )
        )
        handles.append(
            block.ffn_norm.register_forward_pre_hook(
                lambda _module, args: captured.append(args[2])
            )
        )
    handles.append(
        expert.final_norm.register_forward_pre_hook(
            lambda _module, _args, kwargs: captured.append(kwargs["cond"]),
            with_kwargs=True,
        )
    )

    output = expert(
        torch.randn(2, 2, 32),
        torch.randn(2, 5, 32),
        torch.randn(2, 8, 32),
        condition,
    )
    for handle in handles:
        handle.remove()

    assert output.shape == (2, 5, 32)
    assert len(captured) == 2 * len(expert.blocks) + 1
    for actual in captured:
        assert actual is condition


def test_context_is_invariant_to_noisy_actions_but_actions_read_context() -> None:
    torch.manual_seed(1)
    config = _tiny_gemma_config(depth=1)
    block = InterleavedExpertBlock(config, 0, cross_attention=False).eval()
    context = torch.randn(2, 2, 32)
    actions_a = torch.randn(2, 5, 32)
    actions_b = torch.randn(2, 5, 32)
    memory = torch.randn(2, 16, 32)
    time = torch.randn(2, 32)
    mask = VSAActionExpert.self_attention_mask(2, 5, "cpu")

    context_a, action_a = block(
        context,
        actions_a,
        memory,
        time,
        mask,
        _position_embeddings(config, context, actions_a),
    )
    context_b, _ = block(
        context,
        actions_b,
        memory,
        time,
        mask,
        _position_embeddings(config, context, actions_b),
    )
    changed_context = context.clone()
    changed_context[:, 0] += 3
    _, action_b = block(
        changed_context,
        actions_a,
        memory,
        time,
        mask,
        _position_embeddings(config, changed_context, actions_a),
    )

    torch.testing.assert_close(context_a, context_b, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(action_a, action_b)


def test_visual_cross_attention_updates_actions_only() -> None:
    torch.manual_seed(2)
    config = _tiny_gemma_config(depth=2)
    block = InterleavedExpertBlock(
        config,
        1,
        cross_attention=True,
        include_state_in_visual_crossattn=False,
        include_skill_in_visual_crossattn=False,
    ).eval()
    context = torch.randn(2, 2, 32)
    actions = torch.randn(2, 5, 32)
    memory_a = torch.randn(2, 16, 32)
    memory_b = memory_a + 2
    time = torch.randn(2, 32)
    mask = VSAActionExpert.self_attention_mask(2, 5, "cpu")
    positions = _position_embeddings(config, context, actions)

    context_a, action_a = block(
        context, actions, memory_a, time, mask, positions
    )
    context_b, action_b = block(
        context, actions, memory_b, time, mask, positions
    )

    # Context is excluded from this explicit action-only visual query.
    torch.testing.assert_close(context_a, context_b, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(action_a, action_b)


def test_state_visual_cross_attention_query_excludes_skill() -> None:
    torch.manual_seed(3)
    config = _tiny_gemma_config(depth=2)
    block = InterleavedExpertBlock(
        config,
        1,
        cross_attention=True,
        include_state_in_visual_crossattn=True,
        include_skill_in_visual_crossattn=False,
    ).eval()
    context = torch.randn(2, 2, 32)
    actions = torch.randn(2, 5, 32)
    memory = torch.randn(2, 16, 32)
    time = torch.randn(2, 32)
    mask = VSAActionExpert.self_attention_mask(2, 5, "cpu")
    captured_queries = []
    handle = block.visual_cross_attention.register_forward_pre_hook(
        lambda _module, args: captured_queries.append(args[0].detach().clone())
    )

    block(
        context,
        actions,
        memory,
        time,
        mask,
        _position_embeddings(config, context, actions),
    )
    handle.remove()

    assert captured_queries[0].shape[1] == actions.shape[1] + 1


def test_state_skill_visual_cross_attention_query_includes_both() -> None:
    torch.manual_seed(4)
    config = _tiny_gemma_config(depth=2)
    block = InterleavedExpertBlock(
        config,
        1,
        cross_attention=True,
        include_state_in_visual_crossattn=True,
        include_skill_in_visual_crossattn=True,
    ).eval()
    context = torch.randn(2, 2, 32)
    actions = torch.randn(2, 5, 32)
    memory = torch.randn(2, 16, 32)
    time = torch.randn(2, 32)
    mask = VSAActionExpert.self_attention_mask(2, 5, "cpu")
    captured_queries = []
    handle = block.visual_cross_attention.register_forward_pre_hook(
        lambda _module, args: captured_queries.append(args[0].detach().clone())
    )

    output_context, output_actions = block(
        context,
        actions,
        memory,
        time,
        mask,
        _position_embeddings(config, context, actions),
    )
    handle.remove()

    assert captured_queries[0].shape[1] == actions.shape[1] + 2
    assert output_context.shape == context.shape
    assert output_actions.shape == actions.shape


def test_visual_query_option_preserves_outputs_and_state_dict_contract() -> None:
    torch.manual_seed(5)
    config = _tiny_gemma_config(depth=2)
    action_only = VSAActionExpert(
        config, include_state_in_visual_crossattn=False
    ).eval()
    state_and_action = VSAActionExpert(
        config,
        include_state_in_visual_crossattn=True,
        include_skill_in_visual_crossattn=True,
    ).eval()
    context = torch.randn(2, 2, 32)
    actions = torch.randn(2, 5, 32)
    memory = torch.randn(2, 16, 32)
    time = torch.randn(2, 32)

    action_only_output = action_only(context, actions, memory, time)
    state_and_action_output = state_and_action(context, actions, memory, time)

    assert action_only_output.shape == (2, 5, 32)
    assert state_and_action_output.shape == action_only_output.shape
    assert action_only.state_dict().keys() == state_and_action.state_dict().keys()
    assert sum(p.numel() for p in action_only.parameters()) == sum(
        p.numel() for p in state_and_action.parameters()
    )
    state_and_action.load_state_dict(action_only.state_dict(), strict=True)


def test_default_mode_strict_loads_as_identical_interleaved_architecture() -> None:
    torch.manual_seed(51)
    config = _tiny_gemma_config(depth=2)
    mode_field_missing = VSAActionExpert(config).eval()
    explicit_interleaved = VSAActionExpert(
        config, vision_conditioning_mode=INTERLEAVED_CROSS_ATTENTION
    ).eval()
    explicit_interleaved.load_state_dict(mode_field_missing.state_dict(), strict=True)
    context = torch.randn(2, 2, 32)
    actions = torch.randn(2, 5, 32)
    memory = torch.randn(2, 16, 32)
    condition = torch.randn(2, 32)

    expected = mode_field_missing(context, actions, memory, condition)
    actual = explicit_interleaved(context, actions, memory, condition)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert explicit_interleaved.state_dict().keys() == mode_field_missing.state_dict().keys()


def test_cross_attention_debug_reports_attention_and_token_updates() -> None:
    torch.manual_seed(50)
    expert = VSAActionExpert(
        _tiny_gemma_config(depth=4),
        include_state_in_visual_crossattn=True,
        include_skill_in_visual_crossattn=True,
    ).eval()
    expert.debug_enabled = True

    output = expert(
        torch.randn(2, 2, 32),
        torch.randn(2, 5, 32),
        torch.randn(2, 16, 32),
        torch.randn(2, 32),
    )

    assert output.shape == (2, 5, 32)
    for layer in (1, 3):
        prefix = f"cross_layer_{layer:02d}"
        assert 0 <= expert.last_debug_stats[f"{prefix}/attention/normalized_entropy"] <= 1
        assert 1 <= expert.last_debug_stats[f"{prefix}/attention/effective_memory_tokens"] <= 16
        assert expert.last_debug_stats[f"{prefix}/attention/top_camera_mass"] + expert.last_debug_stats[
            f"{prefix}/attention/wrist_camera_mass"
        ] == pytest.approx(1.0)
        assert expert.last_debug_stats[f"{prefix}/action/applied_update_ratio"] >= 0
        assert expert.last_debug_stats[f"{prefix}/state/applied_update_ratio"] >= 0
        assert expert.last_debug_stats[f"{prefix}/skill/applied_update_ratio"] >= 0
        assert f"{prefix}/residual_gate/tanh_scale" not in expert.last_debug_stats


def test_latent_debug_detects_collapsed_tokens() -> None:
    diverse = torch.eye(8).repeat(2, 1, 1)
    collapsed = torch.ones(2, 8, 8)

    diverse_stats = SkillExpertPytorch._latent_debug_stats(diverse, "camera")
    collapsed_stats = SkillExpertPytorch._latent_debug_stats(collapsed, "camera")

    assert diverse_stats["visual/camera/effective_rank_fraction"] > collapsed_stats[
        "visual/camera/effective_rank_fraction"
    ]
    assert collapsed_stats["visual/camera/pair_cosine_abs_mean"] == pytest.approx(1.0)
    assert collapsed_stats["visual/camera/token_spread_rms"] == pytest.approx(0.0)


def test_action_diagnostics_split_flow_time_components_and_horizon() -> None:
    per_sample_error = torch.tensor([1.0, 4.0, 9.0, 16.0])
    squared_error = per_sample_error[:, None, None].expand(4, 10, 7)
    valid = torch.ones(4, 10, dtype=torch.bool)
    flow_time = torch.tensor([0.1, 0.3, 0.6, 0.9])

    metrics = SkillExpertPolicy._action_diagnostic_losses(
        squared_error, valid, flow_time
    )

    assert metrics["flow_timestep/t_0_025_loss"] == pytest.approx(1.0)
    assert metrics["flow_timestep/t_025_050_loss"] == pytest.approx(4.0)
    assert metrics["flow_timestep/t_050_075_loss"] == pytest.approx(9.0)
    assert metrics["flow_timestep/t_075_100_loss"] == pytest.approx(16.0)
    for component in ("translation", "rotation", "gripper"):
        assert metrics[f"action_component/{component}_loss"] == pytest.approx(7.5)
    for segment in ("early", "middle", "late"):
        assert metrics[f"action_horizon/{segment}_loss"] == pytest.approx(7.5)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_small_expert_forward_backward_has_all_core_gradients(batch_size: int) -> None:
    torch.manual_seed(6)
    expert = VSAActionExpert(_tiny_gemma_config())
    context = torch.randn(batch_size, 2, 32, requires_grad=True)
    actions = torch.randn(batch_size, 5, 32, requires_grad=True)
    memory = torch.randn(batch_size, 16, 32, requires_grad=True)
    time = torch.randn(batch_size, 32, requires_grad=True)
    output = expert(context, actions, memory, time)
    output.square().mean().backward()

    assert output.shape == (batch_size, 5, 32)
    assert torch.isfinite(output).all()
    assert context.grad is not None and context.grad.abs().sum() > 0
    assert actions.grad is not None and actions.grad.abs().sum() > 0
    assert memory.grad is not None and memory.grad.abs().sum() > 0
    assert expert.blocks[0].self_attention.q_proj.weight.grad is not None
    assert expert.blocks[1].self_attention is None
    assert expert.blocks[1].visual_cross_attention.q_proj.weight.grad is not None
    assert expert.blocks[0].mlp.up_proj.weight.grad is not None


class _DummyDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16, num_register_tokens=4)
        self.proj = nn.Linear(3, 16)
        self.calls = 0

    def forward(self, image):
        self.calls += 1
        pooled = image.mean(dim=(-2, -1))
        hidden = self.proj(pooled)[:, None].expand(-1, 201, -1)
        return SimpleNamespace(last_hidden_state=hidden)


class _TinyResampler(nn.Module):
    def __init__(
        self, dino_width, expert_width, perceiver_width=1024, num_latents=8
    ):
        super().__init__()
        self.proj = nn.Linear(dino_width, expert_width)
        self.num_latents = num_latents
        self.calls = 0

    def forward(self, tokens):
        self.calls += 1
        return self.proj(tokens.mean(dim=1))[:, None].expand(
            -1, self.num_latents, -1
        )


class _TinyExpert(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.self_attn = nn.Linear(1024, 1024)
        self.cross_attn = nn.Linear(1024, 1024)
        self.debug_enabled = False
        self.last_debug_stats = {}

    def gradient_checkpointing_enable(self):
        pass

    def forward(self, context, actions, visual_memory, time):
        return (
            actions
            + self.self_attn(context.mean(dim=1))[:, None]
            + self.cross_attn(visual_memory.mean(dim=1))[:, None]
            + time[:, None]
        )


class _DummyDINO1024(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(hidden_size=1024, num_register_tokens=4)
        self.calls = 0

    def forward(self, image):
        self.calls += 1
        value = image.mean(dim=(1, 2, 3), keepdim=True).reshape(-1, 1, 1)
        return SimpleNamespace(
            last_hidden_state=(value * self.scale).expand(-1, 201, 1024)
        )


def test_arch1_3_uses_uncompressed_dino_tokens_without_resamplers() -> None:
    dino = _DummyDINO1024()
    config = SkillExpertConfig(
        architecture_label="arch1_3",
        architecture_revision=UNCOMPRESSED_VISUAL_KV_REVISION,
        vision_conditioning_mode=UNCOMPRESSED_VISUAL_KV_SELF_ATTENTION,
        num_visual_latents_per_camera=197,
    )
    with (
        patch(
            "lerobot.policies.skill_expert.modeling_skill_expert.AutoModel.from_pretrained",
            return_value=dino,
        ),
        patch(
            "lerobot.policies.skill_expert.modeling_skill_expert.CameraPerceiverResampler",
            side_effect=AssertionError("Arch1_3 must not build a Perceiver"),
        ),
        patch(
            "lerobot.policies.skill_expert.modeling_skill_expert.VSAActionExpert",
            _TinyExpert,
        ),
    ):
        model = SkillExpertPytorch(config)

    memory = model.encode_visual_memory(
        [torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)]
    )
    assert memory.shape == (1, 394, 1024)
    assert model.top_resampler is None
    assert model.wrist_resampler is None
    assert dino.calls == 1


def test_global_visual_adarms_zero_init_and_second_step_gradient_path() -> None:
    dino = _DummyDINO()
    with (
        patch(
            "lerobot.policies.skill_expert.modeling_skill_expert.AutoModel.from_pretrained",
            return_value=dino,
        ),
        patch(
            "lerobot.policies.skill_expert.modeling_skill_expert.CameraPerceiverResampler",
            _TinyResampler,
        ),
        patch(
            "lerobot.policies.skill_expert.modeling_skill_expert.VSAActionExpert",
            _TinyExpert,
        ),
    ):
        model = SkillExpertPytorch(
            SkillExpertConfig(vision_conditioning_mode=GLOBAL_VISUAL_ADARMS)
        )

    projection = model.visual_condition_projection
    assert projection is not None
    assert torch.count_nonzero(projection.weight) == 0
    assert torch.count_nonzero(projection.bias) == 0
    ordered_memory = torch.cat(
        (
            torch.full((2, 32, 1024), 2.0),
            torch.full((2, 32, 1024), 5.0),
        ),
        dim=1,
    )
    pooled = model._pool_visual_memory(ordered_memory)
    assert pooled.shape == (2, 2048)
    torch.testing.assert_close(pooled[:, :1024], torch.full((2, 1024), 2.0))
    torch.testing.assert_close(pooled[:, 1024:], torch.full((2, 1024), 5.0))
    memory = torch.randn(2, 64, 1024, requires_grad=True)
    time_condition = torch.randn(2, 1024)
    first = model._action_condition(memory, time_condition)
    torch.testing.assert_close(first, time_condition)
    first.sum().backward()
    assert projection.weight.grad is not None
    assert projection.weight.grad.abs().sum() > 0
    assert memory.grad is not None and torch.count_nonzero(memory.grad) == 0

    torch.optim.SGD(projection.parameters(), lr=1e-4).step()
    projection.zero_grad(set_to_none=True)
    memory.grad = None
    second = model._action_condition(memory, time_condition)
    second.square().mean().backward()
    assert memory.grad is not None and memory.grad.abs().sum() > 0


def test_non_global_modes_have_no_visual_condition_projection_parameters() -> None:
    for mode in (INTERLEAVED_CROSS_ATTENTION, IN_CONTEXT_TOKENS):
        dino = _DummyDINO()
        with (
            patch(
                "lerobot.policies.skill_expert.modeling_skill_expert.AutoModel.from_pretrained",
                return_value=dino,
            ),
            patch(
                "lerobot.policies.skill_expert.modeling_skill_expert.CameraPerceiverResampler",
                _TinyResampler,
            ),
            patch(
                "lerobot.policies.skill_expert.modeling_skill_expert.VSAActionExpert",
                _TinyExpert,
            ),
        ):
            model = SkillExpertPytorch(
                SkillExpertConfig(vision_conditioning_mode=mode)
            )
        assert model.visual_condition_projection is None
        assert not any(
            key.startswith("visual_condition_projection")
            for key in model.state_dict()
        )


@pytest.mark.parametrize("batch_size", [1, 3])
def test_stage1_flow_and_sampling_cache_vision_once(batch_size: int) -> None:
    dino = _DummyDINO()
    with (
        patch(
            "lerobot.policies.skill_expert.modeling_skill_expert.AutoModel.from_pretrained",
            return_value=dino,
        ),
        patch(
            "lerobot.policies.skill_expert.modeling_skill_expert.CameraPerceiverResampler",
            _TinyResampler,
        ),
        patch(
            "lerobot.policies.skill_expert.modeling_skill_expert.VSAActionExpert",
            _TinyExpert,
        ),
    ):
        model = SkillExpertPytorch(SkillExpertConfig())

    images = [torch.rand(batch_size, 3, 256, 256) for _ in range(2)]
    state = torch.randn(batch_size, 32)
    skill = torch.arange(batch_size) % 27
    actions = torch.randn(batch_size, 10, 32)
    residual = model(images, state, skill, actions)
    residual.square().mean().backward()

    assert residual.shape == actions.shape
    assert torch.isfinite(residual).all()
    assert dino.calls == 1
    assert model.top_resampler.calls == model.wrist_resampler.calls == 1
    for module in (
        dino.proj,
        model.top_resampler.proj,
        model.wrist_resampler.proj,
        model.state_proj,
        model.skill_proj,
        model.expert.self_attn,
        model.expert.cross_attn,
        model.action_in_proj,
        model.action_out_proj,
    ):
        assert module.weight.grad is not None
    torch.optim.SGD(model.parameters(), lr=1e-4).step()

    dino.calls = model.top_resampler.calls = model.wrist_resampler.calls = 0
    sampled = model.sample_actions(images, state, skill, num_steps=2)
    assert sampled.shape == actions.shape
    assert torch.isfinite(sampled).all()
    assert dino.calls == 1
    assert model.top_resampler.calls == model.wrist_resampler.calls == 1


def test_stage1_scheduled_debug_collects_diversity_and_sensitivity() -> None:
    dino = _DummyDINO()
    with (
        patch(
            "lerobot.policies.skill_expert.modeling_skill_expert.AutoModel.from_pretrained",
            return_value=dino,
        ),
        patch(
            "lerobot.policies.skill_expert.modeling_skill_expert.CameraPerceiverResampler",
            _TinyResampler,
        ),
        patch(
            "lerobot.policies.skill_expert.modeling_skill_expert.VSAActionExpert",
            _TinyExpert,
        ),
    ):
        model = SkillExpertPytorch(
            SkillExpertConfig(vsa_debug_schedule=(1, 100))
        ).train()

    model.set_training_step(1)
    residual = model(
        [torch.rand(2, 3, 64, 64) for _ in range(2)],
        torch.randn(2, 32),
        torch.tensor([0, 13]),
        torch.randn(2, 10, 32),
    )
    stats = model._last_vsa_debug_stats

    assert residual.shape == (2, 10, 32)
    assert "visual/top_latents/effective_rank_fraction" in stats
    assert "visual/cross_camera/centroid_cosine" in stats
    assert "sensitivity/top_image_shuffle/relative_output_delta" in stats
    assert "sensitivity/wrist_image_shuffle/relative_output_delta" in stats
    assert "sensitivity/state_shuffle/relative_output_delta" in stats
    assert "sensitivity/skill_shuffle/relative_output_delta" in stats

    model.set_training_step(2)
    model(
        [torch.rand(2, 3, 64, 64) for _ in range(2)],
        torch.randn(2, 32),
        torch.tensor([1, 2]),
        torch.randn(2, 10, 32),
    )
    assert model._last_vsa_debug_stats == {}


def test_scheduled_debug_accepts_current_interleaved_blocks_without_residual_gate() -> None:
    class _DebugBlock(nn.Module):
        def __init__(self, *, cross_attention: bool):
            super().__init__()
            self.cross_attention = cross_attention
            self.visual_cross_attention = nn.Linear(4, 4) if cross_attention else None
            self.self_attention = None if cross_attention else nn.Linear(4, 4)
            self.mlp = nn.Linear(4, 4)

    policy = SkillExpertPolicy.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(architecture=VSA_ARCHITECTURE)
    policy.model = nn.Module()
    policy.model._vsa_debug_active = True
    policy.model.dino = nn.Linear(4, 4)
    policy.model.top_resampler = nn.Linear(4, 4)
    policy.model.wrist_resampler = nn.Linear(4, 4)
    policy.model.state_proj = nn.Linear(4, 4)
    policy.model.state_norm = nn.LayerNorm(4)
    policy.model.skill_proj = nn.Linear(4, 4)
    policy.model.skill_norm = nn.LayerNorm(4)
    policy.model.action_in_proj = nn.Linear(4, 4)
    policy.model.action_out_proj = nn.Linear(4, 4)
    policy.model.time_mlp_in = nn.Linear(4, 4)
    policy.model.time_mlp_out = nn.Linear(4, 4)
    policy.model.visual_condition_projection = None
    policy.model.expert = nn.Module()
    policy.model.expert.blocks = nn.ModuleList(
        [_DebugBlock(cross_attention=False), _DebugBlock(cross_attention=True)]
    )
    policy.model.expert.debug_enabled = True

    metrics = policy.training_debug_metrics()

    assert not any("visual_residual_gate" in key for key in metrics)
    assert "vsa_debug/gradient/preclip/expert_cross_attention_grad_rms" in metrics
    assert "vsa_debug/parameter/expert_cross_attention_rms" in metrics
    assert (
        "vsa_debug/gradient/preclip/expert_cross_attention_to_parameter_rms_ratio"
        in metrics
    )
    assert policy.model._vsa_debug_active is False
    assert policy.model.expert.debug_enabled is False


def test_optimizer_covers_every_trainable_parameter_once_and_scales_dino() -> None:
    policy = SkillExpertPolicy.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(optimizer_lr=2.5e-5, dino_lr_scale=0.1)
    policy.model = nn.Module()
    policy.model.dino = nn.Linear(4, 4)
    policy.model.body = nn.Linear(4, 4)
    policy.model.skill_predictor = nn.Linear(4, 4)
    policy.model.fsq_term_train = nn.Linear(4, 4)

    groups = policy.get_optim_params()
    grouped = [parameter for group in groups for parameter in group["params"]]
    expected = [parameter for parameter in policy.parameters() if parameter.requires_grad]

    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {id(parameter) for parameter in expected}
    assert not any(parameter.requires_grad for parameter in policy.model.skill_predictor.parameters())
    assert not any(parameter.requires_grad for parameter in policy.model.fsq_term_train.parameters())
    assert not hasattr(SkillExpertPolicy, "isolated_auxiliary_step")
    assert not hasattr(SkillExpertPolicy, "isolated_main_optimizer_grad_groups")
    dino_group = next(group for group in groups if group["group_name"] == "dino")
    assert dino_group["lr"] == pytest.approx(2.5e-6)
    assert dino_group["lr_scale"] == 0.1

    optimizer = torch.optim.AdamW(groups, lr=policy.config.optimizer_lr)
    scheduler = SkillExpertConfig().get_scheduler_preset().build(optimizer, 30_000)
    for _ in range(3):
        optimizer.step()
        scheduler.step()
        assert optimizer.param_groups[1]["lr"] == pytest.approx(
            optimizer.param_groups[0]["lr"] * 0.1
        )


def test_arch1_optimizer_applies_relative_dino_lr_scale() -> None:
    policy = SkillExpertPolicy.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        architecture=COND_GEMMA_ARCHITECTURE,
        optimizer_lr=2.5e-5,
        dino_lr_scale=0.1,
        dino_lr=None,
        freeze_vision_encoder=False,
    )
    policy.model = nn.Module()
    policy.model.dino = nn.Linear(4, 4)
    policy.model.body = nn.Linear(4, 4)
    policy.model.skill_predictor = nn.Linear(4, 4)
    policy.model.fsq_term_train = nn.Linear(4, 4)

    groups = policy.get_optim_params()
    grouped = [parameter for group in groups for parameter in group["params"]]
    expected = [parameter for parameter in policy.parameters() if parameter.requires_grad]

    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {id(parameter) for parameter in expected}
    dino_group = next(group for group in groups if group["group_name"] == "dino")
    assert dino_group["lr"] == pytest.approx(2.5e-6)
    assert dino_group["lr_scale"] == pytest.approx(0.1)

    optimizer = torch.optim.AdamW(groups, lr=policy.config.optimizer_lr)
    scheduler = SkillExpertConfig().get_scheduler_preset().build(optimizer, 30_000)
    for _ in range(3):
        optimizer.step()
        scheduler.step()
        assert optimizer.param_groups[1]["lr"] == pytest.approx(
            optimizer.param_groups[0]["lr"] * 0.1
        )


def test_pi05_vsa_initialization_mapping_is_explicit() -> None:
    assert _map_pi05_key(
        "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight"
    ) == "model.expert.blocks.0.self_attention.q_proj.weight"
    assert _map_pi05_key(
        "paligemma_with_expert.gemma_expert.model.layers.1.self_attn.q_proj.weight"
    ) is None
    assert _map_pi05_key(
        "paligemma_with_expert.gemma_expert.model.layers.1.self_attn.q_proj.weight",
        vision_conditioning_mode=COMPRESSED_VISUAL_KV_SELF_ATTENTION,
    ) == "model.expert.blocks.1.self_attention.q_proj.weight"
    assert _map_pi05_key(
        "paligemma_with_expert.gemma_expert.model.layers.1.mlp.up_proj.weight"
    ) == "model.expert.blocks.1.mlp.up_proj.weight"
    assert _map_pi05_key(
        "paligemma_with_expert.gemma_expert.model.layers.0.input_layernorm.dense.weight"
    ) == "model.expert.blocks.0.self_attention_norm.action_norm.dense.weight"
    assert _map_pi05_key("action_in_proj.weight") == "model.action_in_proj.weight"
    assert _map_pi05_key(
        "paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.up_proj.weight"
    ) is None


def test_pi05_condition_gemma_mapping_matches_skillvla_real_layout() -> None:
    assert _map_pi05_key(
        "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight",
        architecture=COND_GEMMA_ARCHITECTURE,
    ) == "model.gemma_expert.model.layers.0.self_attn.q_proj.weight"
    assert _map_pi05_key(
        "paligemma_with_expert.gemma_expert.lm_head.weight",
        architecture=COND_GEMMA_ARCHITECTURE,
    ) is None
    assert _map_pi05_key(
        "action_in_proj.weight",
        architecture=COND_GEMMA_ARCHITECTURE,
    ) == "model.action_in_proj.weight"


def test_pi05_missing_allowlist_is_mode_specific() -> None:
    interleaved = SkillExpertConfig(
        vision_conditioning_mode=INTERLEAVED_CROSS_ATTENTION
    )
    in_context = SkillExpertConfig(vision_conditioning_mode=IN_CONTEXT_TOKENS)
    global_adarms = SkillExpertConfig(
        vision_conditioning_mode=GLOBAL_VISUAL_ADARMS
    )
    cross_key = "model.expert.blocks.1.visual_cross_attention.q_proj.weight"
    global_key = "model.visual_condition_projection.weight"

    assert _allowed_pi05_missing_key("model.top_resampler.latents", interleaved)
    assert _allowed_pi05_missing_key(
        "model.expert.blocks.0.self_attention_norm.context_norm.weight",
        interleaved,
    )
    assert _allowed_pi05_missing_key(cross_key, interleaved)
    assert not _allowed_pi05_missing_key(cross_key, in_context)
    assert _allowed_pi05_missing_key(global_key, global_adarms)
    assert not _allowed_pi05_missing_key(global_key, interleaved)
    assert not _allowed_pi05_missing_key(
        "model.expert.blocks.0.self_attention.q_proj.weight", interleaved
    )
    assert not _allowed_pi05_missing_key("model.action_in_proj.weight", interleaved)
    cond_perceiver = SkillExpertConfig(
        architecture=COND_GEMMA_ARCHITECTURE,
        architecture_label="arch1_2",
        architecture_revision=COND_GEMMA_PERCEIVER_EXPERT_TOKENS_REVISION,
    )
    assert _allowed_pi05_missing_key(
        "model.context_input_norms.0.weight", cond_perceiver
    )
    assert _allowed_pi05_missing_key(
        "model.top_resampler.latents", cond_perceiver
    )


def test_skill_end_loss_mask_excludes_only_offsets_after_boundary() -> None:
    owner = SimpleNamespace(
        config=SimpleNamespace(mask_actions_after_skill_end=True)
    )
    actions = torch.zeros(2, 5, 7)
    batch = {
        "skill_de": torch.tensor([1, 3]),
        "skill_effective_de": torch.tensor([1, 3]),
        "action_is_pad": torch.tensor(
            [
                [False, False, False, False, False],
                [False, False, True, False, False],
            ]
        ),
    }

    valid = SkillExpertPolicy._valid_action_steps(owner, actions, batch)

    torch.testing.assert_close(
        valid,
        torch.tensor(
            [
                [True, True, False, False, False],
                [True, True, False, True, False],
            ]
        ),
    )


def test_disabled_skill_end_loss_mask_preserves_full_unpadded_chunk() -> None:
    owner = SimpleNamespace(
        config=SimpleNamespace(mask_actions_after_skill_end=False)
    )
    actions = torch.zeros(1, 4, 7)
    batch = {
        "skill_de": torch.tensor([0]),
        "action_is_pad": torch.tensor([[False, False, False, True]]),
    }

    valid = SkillExpertPolicy._valid_action_steps(owner, actions, batch)

    torch.testing.assert_close(valid, torch.tensor([[True, True, True, False]]))


def test_skill_end_loss_mask_requires_distance_to_end() -> None:
    owner = SimpleNamespace(
        config=SimpleNamespace(mask_actions_after_skill_end=True)
    )
    with pytest.raises(KeyError, match="skill_de"):
        SkillExpertPolicy._valid_action_steps(owner, torch.zeros(1, 4, 7), {})


def test_skill_end_loss_mask_prefers_jittered_effective_boundary() -> None:
    owner = SimpleNamespace(
        config=SimpleNamespace(
            mask_actions_after_skill_end=True,
            transition_jitter_pmax=15,
        )
    )
    actions = torch.zeros(2, 5, 7)
    batch = {
        "skill_de": torch.tensor([0, 4]),
        "skill_effective_de": torch.tensor([3, 1]),
    }

    valid = SkillExpertPolicy._valid_action_steps(owner, actions, batch)

    torch.testing.assert_close(
        valid,
        torch.tensor(
            [
                [True, True, True, True, False],
                [True, True, False, False, False],
            ]
        ),
    )


def test_cumulative_xyz_loss_uses_one_horizon_normalizer_per_sample() -> None:
    predicted = torch.ones(2, 3, 3)
    target = torch.zeros_like(predicted)
    valid = torch.tensor(
        [
            [True, True, True],
            [True, True, False],
        ]
    )

    normalized, raw, normalized_per_sample, raw_per_sample = (
        SkillExpertPolicy._cumulative_xyz_loss(predicted, target, valid)
    )

    # Prefix MSEs are [1, 4, 9]. Row 0: mean=14/3, normalized by 2.
    # Row 1 uses [1, 4]: mean=5/2, normalized by 3/2.
    torch.testing.assert_close(
        raw_per_sample, torch.tensor([14.0 / 3.0, 5.0 / 2.0])
    )
    torch.testing.assert_close(
        normalized_per_sample, torch.tensor([7.0 / 3.0, 5.0 / 3.0])
    )
    torch.testing.assert_close(raw, torch.tensor(43.0 / 12.0))
    torch.testing.assert_close(normalized, torch.tensor(2.0))
