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
    GLOBAL_VISUAL_ADARMS,
    IN_CONTEXT_TOKENS,
    RESIDUAL_CROSS_ATTENTION,
    VSA_ARCHITECTURE,
    SkillExpertConfig,
)
from lerobot.policies.skill_expert.modeling_skill_expert import (
    SkillExpertPolicy,
    SkillExpertPytorch,
    _allowed_pi05_missing_key,
    _map_pi05_key,
)
from lerobot.policies.skill_expert.legacy_vsa_eval import LegacyVSAActionExpert
from lerobot.policies.skill_expert.vsa_perceiver_crossattn import (
    VISUAL_RESIDUAL_GATE_INIT,
    CameraPerceiverResampler,
    ResidualVisualExpertBlock,
    VSAActionExpert,
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


def test_config_defaults_to_vsa_and_cond_architecture_is_explicit() -> None:
    config = SkillExpertConfig()

    assert config.architecture == VSA_ARCHITECTURE == "vsa_perceiver_crossattn"
    assert config.vision_conditioning_mode == RESIDUAL_CROSS_ATTENTION
    assert config.include_state_in_visual_crossattn is False
    assert config.include_skill_in_visual_crossattn is False
    assert config.action_expert_variant == "gemma_300m"
    assert config.dino_lr_scale == 0.1
    assert config.num_visual_latents_per_camera == 32
    assert config.n_action_steps == 5
    assert config.vsa_debug_schedule == ()
    assert SkillExpertConfig(vsa_debug_schedule=[1, 100]).vsa_debug_schedule == (1, 100)
    assert config.conditioning_route == "state_skill_cond"
    assert config.cond_encoder_variant == "gemma_300m"
    assert config.freeze_vision_encoder is False
    cond = SkillExpertConfig(
        architecture=COND_GEMMA_ARCHITECTURE,
        architecture_revision=COND_GEMMA_ARCHITECTURE_REVISION,
        conditioning_route="state_skill_cond",
    )
    assert cond.architecture == "cond_gemma"
    with pytest.raises(ValueError, match="architecture must be"):
        SkillExpertConfig(architecture="state_skill_cond")
    with pytest.raises(ValueError, match="vision_conditioning_mode must be one of"):
        SkillExpertConfig(vision_conditioning_mode="unknown")

    with pytest.raises(ValueError, match="sorted and contain no duplicates"):
        SkillExpertConfig(vsa_debug_schedule=(100, 1, 100))


def test_policy_loader_rejects_cond_checkpoint_when_vsa_is_requested(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"type": "skill_expert", "conditioning_route": "state_cond"})
    )

    with pytest.raises(ValueError, match="checkpoint architecture mismatch"):
        SkillExpertPolicy.from_pretrained(tmp_path, config=SkillExpertConfig())


def test_policy_loader_rejects_pre_residual_vsa_checkpoint(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "type": "skill_expert",
                "architecture": "vsa_perceiver_crossattn",
            }
        )
    )

    with pytest.raises(ValueError, match="predates the residual-SA18 VSA revision"):
        SkillExpertPolicy.from_pretrained(tmp_path)


def test_policy_loader_rejects_explicit_cross_mode_checkpoint_override(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "type": "skill_expert",
                "architecture": "vsa_perceiver_crossattn",
                "architecture_revision": "residual_sa18_v2",
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


@pytest.mark.parametrize(
    "mode", [RESIDUAL_CROSS_ATTENTION, IN_CONTEXT_TOKENS, GLOBAL_VISUAL_ADARMS]
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
    if mode == RESIDUAL_CROSS_ATTENTION:
        assert len(cross_layers) == 2
        assert expert.last_sequence_length == 7
    else:
        assert not cross_layers
        assert not any(
            "visual_cross_attention" in key for key in expert.state_dict()
        )
        expected_length = 15 if mode == IN_CONTEXT_TOKENS else 7
        assert expert.last_sequence_length == expected_length
    assert expert.last_position_ids[0].tolist() == list(
        range(expert.last_sequence_length)
    )


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
    block = ResidualVisualExpertBlock(config, 0, cross_attention=False).eval()
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
    block = ResidualVisualExpertBlock(config, 1, cross_attention=True).eval()
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

    # Context is excluded from the default action-only visual query, while the
    # weakly initialized gate exposes actions to vision from the first step.
    torch.testing.assert_close(context_a, context_b, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(action_a, action_b)
    assert block.visual_residual_gate.item() == pytest.approx(VISUAL_RESIDUAL_GATE_INIT)


def test_state_visual_cross_attention_query_excludes_skill() -> None:
    torch.manual_seed(3)
    config = _tiny_gemma_config(depth=2)
    block = ResidualVisualExpertBlock(
        config,
        1,
        cross_attention=True,
        include_state_in_visual_crossattn=True,
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
    block = ResidualVisualExpertBlock(
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


def test_default_mode_strict_loads_as_identical_residual_architecture() -> None:
    torch.manual_seed(51)
    config = _tiny_gemma_config(depth=2)
    mode_field_missing = VSAActionExpert(config).eval()
    explicit_residual = VSAActionExpert(
        config, vision_conditioning_mode=RESIDUAL_CROSS_ATTENTION
    ).eval()
    explicit_residual.load_state_dict(mode_field_missing.state_dict(), strict=True)
    context = torch.randn(2, 2, 32)
    actions = torch.randn(2, 5, 32)
    memory = torch.randn(2, 16, 32)
    condition = torch.randn(2, 32)

    expected = mode_field_missing(context, actions, memory, condition)
    actual = explicit_residual(context, actions, memory, condition)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert explicit_residual.state_dict().keys() == mode_field_missing.state_dict().keys()


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
        assert expert.last_debug_stats[f"{prefix}/action/applied_update_ratio"] >= 0
        assert expert.last_debug_stats[f"{prefix}/state/applied_update_ratio"] >= 0
        assert expert.last_debug_stats[f"{prefix}/skill/applied_update_ratio"] >= 0
        assert expert.last_debug_stats[
            f"{prefix}/residual_gate/tanh_scale"
        ] == pytest.approx(torch.tanh(torch.tensor(VISUAL_RESIDUAL_GATE_INIT)).item())


def test_latent_debug_detects_collapsed_tokens() -> None:
    diverse = torch.eye(8).repeat(2, 1, 1)
    collapsed = torch.ones(2, 8, 8)

    diverse_stats = SkillExpertPytorch._latent_debug_stats(diverse, "camera")
    collapsed_stats = SkillExpertPytorch._latent_debug_stats(collapsed, "camera")

    assert diverse_stats["visual/camera/effective_rank"] > collapsed_stats[
        "visual/camera/effective_rank"
    ]
    assert collapsed_stats["visual/camera/pair_cosine_mean"] == pytest.approx(1.0)
    assert collapsed_stats["visual/camera/token_spread_rms"] == pytest.approx(0.0)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_small_expert_forward_backward_has_all_core_gradients(batch_size: int) -> None:
    torch.manual_seed(6)
    expert = VSAActionExpert(_tiny_gemma_config())
    context = torch.randn(batch_size, 2, 32, requires_grad=True)
    actions = torch.randn(batch_size, 5, 32, requires_grad=True)
    memory = torch.randn(batch_size, 16, 32, requires_grad=True)
    time = torch.randn(batch_size, 32, requires_grad=True)
    for block in expert.blocks:
        if block.visual_residual_gate is not None:
            block.visual_residual_gate.data.fill_(0.25)

    output = expert(context, actions, memory, time)
    output.square().mean().backward()

    assert output.shape == (batch_size, 5, 32)
    assert torch.isfinite(output).all()
    assert context.grad is not None and context.grad.abs().sum() > 0
    assert actions.grad is not None and actions.grad.abs().sum() > 0
    assert memory.grad is not None and memory.grad.abs().sum() > 0
    assert expert.blocks[0].self_attention.q_proj.weight.grad is not None
    assert expert.blocks[1].self_attention.q_proj.weight.grad is not None
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
    def __init__(self, dino_width, expert_width, num_latents=8):
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
    for mode in (RESIDUAL_CROSS_ATTENTION, IN_CONTEXT_TOKENS):
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
    assert "visual/top_latents/effective_rank" in stats
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


def test_optimizer_covers_every_trainable_parameter_once_and_scales_dino() -> None:
    policy = SkillExpertPolicy.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        optimizer_lr=2.5e-5, dino_lr_scale=0.1, terminator_lr_scale=1.0
    )
    policy.model = nn.Module()
    policy.model.dino = nn.Linear(4, 4)
    policy.model.body = nn.Linear(4, 4)
    policy.model.fsq_term_train = None

    groups = policy.get_optim_params()
    grouped = [parameter for group in groups for parameter in group["params"]]
    expected = [parameter for parameter in policy.parameters() if parameter.requires_grad]

    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {id(parameter) for parameter in expected}
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


def test_pi05_vsa_initialization_mapping_is_explicit() -> None:
    assert _map_pi05_key(
        "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight"
    ) == "model.expert.blocks.0.self_attention.q_proj.weight"
    assert _map_pi05_key(
        "paligemma_with_expert.gemma_expert.model.layers.1.self_attn.q_proj.weight"
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
    residual = SkillExpertConfig(
        vision_conditioning_mode=RESIDUAL_CROSS_ATTENTION
    )
    in_context = SkillExpertConfig(vision_conditioning_mode=IN_CONTEXT_TOKENS)
    global_adarms = SkillExpertConfig(
        vision_conditioning_mode=GLOBAL_VISUAL_ADARMS
    )
    cross_key = "model.expert.blocks.1.visual_cross_attention.q_proj.weight"
    global_key = "model.visual_condition_projection.weight"

    assert _allowed_pi05_missing_key("model.top_resampler.latents", residual)
    assert _allowed_pi05_missing_key(
        "model.expert.blocks.0.self_attention_norm.context_norm.weight",
        residual,
    )
    assert _allowed_pi05_missing_key(cross_key, residual)
    assert not _allowed_pi05_missing_key(cross_key, in_context)
    assert _allowed_pi05_missing_key(global_key, global_adarms)
    assert not _allowed_pi05_missing_key(global_key, residual)
    assert not _allowed_pi05_missing_key(
        "model.expert.blocks.0.self_attention.q_proj.weight", residual
    )
    assert not _allowed_pi05_missing_key("model.action_in_proj.weight", residual)
