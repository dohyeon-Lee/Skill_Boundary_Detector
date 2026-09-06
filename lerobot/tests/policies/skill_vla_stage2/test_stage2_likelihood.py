import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding

from lerobot.policies.skill_expert.cond_gemma import CondGemmaSkillExpert
from lerobot.policies.skill_expert import modeling_skill_predictor
from lerobot.policies.skill_expert.modeling_skill_predictor import (
    FrozenVLMSkillPredictor,
)
from lerobot.policies.skill_expert.processor_skill_expert import (
    _needs_canonical_action_normalization,
    skill_expert_batch_to_transition,
    skill_expert_transition_to_batch,
)
from lerobot.policies.skillVLA.dataset_skillVLA import (
    SKILL_CANONICAL_ACTION_IS_PAD,
    SKILL_CANONICAL_ACTIONS,
)
from lerobot.policies.skill_vla_stage2.configuration_skill_vla_stage2 import (
    SkillVLAStage2Config,
)
from lerobot.policies.skill_vla_stage2.modeling_skill_vla_stage2 import (
    LatentExpertBlock,
    LikelihoodBlock,
    SkillVLAStage2Policy,
    SkillVLAStage2Pytorch,
    _load_pi05_base_vlm_parameters,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    STAGE2_MODE_LATENT_OVERRIDE,
    STAGE2_VLM_CACHE_ID,
)


def _config(**overrides) -> SkillVLAStage2Config:
    values = {
        "stage1_checkpoint_path": "/tmp/stage1",
        "vlm_base_path": "/tmp/pi05_base",
        "fsq_path": "/tmp/FSQ.pt",
    }
    values.update(overrides)
    return SkillVLAStage2Config(**values)


def test_stage2_config_fixes_bayesvla_contract() -> None:
    config = _config()

    assert config.type == "skill_vla_stage2"
    assert config.architecture == "cond_gemma"
    assert config.stage2_mode == "likelihood"
    assert config.likelihood_num_layers == 4
    assert config.training_skill_source == "gt"
    assert config.train_skill_predictor
    assert not config.train_terminator
    assert (
        _config(conditioning_route="state_skill_cond").conditioning_route
        == "state_skill_cond"
    )
    assert (
        _config(conditioning_route="state_skill_only_cond").conditioning_route
        == "state_skill_only_cond"
    )
    assert (
        _config(conditioning_route="skillonly_cond").conditioning_route
        == "skillonly_cond"
    )
    assert (
        _config(conditioning_route="visiononly_cond").conditioning_route
        == "visiononly_cond"
    )
    with pytest.raises(ValueError, match="fixes likelihood_num_layers=4"):
        _config(likelihood_num_layers=3)
    with pytest.raises(ValueError, match="fixed to 'flow'"):
        _config(action_loss_mode="flow_endpoint_xyz")
    with pytest.raises(ValueError, match="complete action chunk"):
        _config(mask_actions_after_skill_end=True)
    assert _config(proprio_grounding="episode_start_xyz").proprio_grounding == (
        "episode_start_xyz"
    )
    with pytest.raises(ValueError, match="skill_flow_enabled=False"):
        _config(
            architecture_label="arch0_skill",
            conditioning_route="state_cond",
            skill_flow_enabled=True,
            skill_flow_max_length=20,
        )
    with pytest.raises(ValueError, match="last.*layer_mix"):
        _config(likelihood_vlm_memory="every_layer")
    with pytest.raises(ValueError, match="gate_lr_scale"):
        _config(likelihood_gate_lr_scale=0.0)
    assert _config(likelihood_vlm_memory="layer_mix").likelihood_vlm_memory == "layer_mix"
    with pytest.raises(ValueError, match="gt.*predictor"):
        _config(training_skill_source="mixed")
    assert _config(skill_predictor_checkpoint_path=None).training_skill_source == "gt"
    with pytest.raises(ValueError, match="vlm_base_path"):
        _config(vlm_base_path="")
    with pytest.raises(ValueError, match="skill_predictor_checkpoint_path"):
        _config(
            training_skill_source="predictor",
            skill_predictor_checkpoint_path=None,
        )
    assert (
        _config(
            training_skill_source="predictor",
            skill_predictor_checkpoint_path="/tmp/predictor",
        ).training_skill_source
        == "predictor"
    )
    with pytest.raises(ValueError, match="train_skill_predictor=True"):
        _config(train_skill_predictor=False)
    with pytest.raises(ValueError, match="without a terminator"):
        _config(train_terminator=True)
    with pytest.raises(ValueError, match="cond_gemma"):
        _config(
            architecture="vsa_perceiver_crossattn",
            architecture_revision="interleaved_direct1024_v3",
        )
    dsbc = _config(
        stage2_mode="dsbc",
        dsbc_noise_output_mode="per_step",
        dsbc_frs_num_steps=8,
        dsbc_anchor_seed=17,
    )
    assert dsbc.dsbc_noise_output_mode == "per_step"
    assert dsbc.dsbc_noise_output_bound == pytest.approx(5.0)
    assert dsbc.dsbc_frs_num_steps == 8
    assert dsbc.dsbc_anchor_seed == 17
    assert dsbc.dsbc_reader == "final"
    with pytest.raises(ValueError, match="shared.*per_step"):
        _config(stage2_mode="dsbc", dsbc_noise_output_mode="full")
    with pytest.raises(ValueError, match="noise_output_bound"):
        _config(stage2_mode="dsbc", dsbc_noise_output_bound=0.0)
    with pytest.raises(ValueError, match="FRS noise"):
        _config(stage2_mode="dsbc", cumulative_xyz_loss_enabled=True)
    latent_dsbc = _config(
        stage2_mode="dsbc",
        architecture_label="arch0_skill",
        skill_flow_enabled=False,
        skill_flow_latent_best_of_n_enabled=True,
        dsbc_reader="all_layers",
        dsbc_latent_predictor_enabled=True,
        dsbc_latent_loss_weight=0.5,
        dsbc_latent_timesteps=3,
    )
    assert latent_dsbc.dsbc_reader == "all_layers"
    assert latent_dsbc.dsbc_latent_predictor_enabled
    assert latent_dsbc.dsbc_latent_predictor_mode == "skill_start"
    assert latent_dsbc.dsbc_latent_supervision == "main_chunk"
    assert latent_dsbc.dsbc_latent_loss_weight == pytest.approx(0.5)
    assert latent_dsbc.dsbc_latent_timesteps == 3
    assert (
        _config(
            stage2_mode="dsbc",
            architecture_label="arch0_skill",
            skill_flow_latent_best_of_n_enabled=True,
            dsbc_latent_predictor_enabled=True,
            dsbc_latent_predictor_mode="per-chunk-final",
        ).dsbc_latent_predictor_mode
        == "per_chunk_final"
    )
    assert (
        _config(
            stage2_mode="dsbc",
            architecture_label="arch0_skill",
            skill_flow_latent_best_of_n_enabled=True,
            dsbc_latent_predictor_enabled=True,
            dsbc_latent_predictor_mode="per-chunk-expert",
        ).dsbc_latent_predictor_mode
        == "per_chunk_expert"
    )
    with pytest.raises(ValueError, match="skill_start.*per_chunk_final.*per_chunk_expert"):
        _config(
            stage2_mode="dsbc",
            architecture_label="arch0_skill",
            skill_flow_latent_best_of_n_enabled=True,
            dsbc_latent_predictor_enabled=True,
            dsbc_latent_predictor_mode="current",
        )
    with pytest.raises(ValueError, match="latent-enabled Stage-1"):
        _config(
            stage2_mode="dsbc",
            dsbc_latent_predictor_enabled=True,
        )
    skill_only_dsbc = _config(
        stage2_mode="dsbc",
        architecture_label="arch0_skill",
        skill_flow_latent_best_of_n_enabled=True,
        dsbc_latent_predictor_enabled=True,
        dsbc_latent_supervision="skill-only",
    )
    assert skill_only_dsbc.dsbc_latent_supervision == "skill_only"
    assert _needs_canonical_action_normalization(skill_only_dsbc)
    with pytest.raises(ValueError, match="canonical arch0_skill"):
        _config(
            stage2_mode="dsbc",
            architecture_label="arch0_skill_chunk",
            skill_flow_latent_best_of_n_enabled=True,
            dsbc_latent_predictor_enabled=True,
            dsbc_latent_supervision="skill_only",
        )
    with pytest.raises(ValueError, match="only when the latent predictor"):
        _config(dsbc_latent_supervision="skill_only")


def test_stage2_skill_only_supervision_enables_canonical_normalization() -> None:
    assert _needs_canonical_action_normalization(
        SimpleNamespace(
            type="skill_vla_stage2",
            skill_flow_enabled=False,
            dsbc_latent_predictor_enabled=True,
            dsbc_latent_supervision="skill_only",
        )
    )
    assert not _needs_canonical_action_normalization(
        SimpleNamespace(
            type="skill_vla_stage2",
            skill_flow_enabled=False,
            dsbc_latent_predictor_enabled=True,
            dsbc_latent_supervision="main_chunk",
        )
    )


def test_stage2_loads_only_base_vlm_tensors(monkeypatch) -> None:
    class TinyPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.vlm = nn.Linear(2, 3)
            self.reader = nn.Linear(3, 1)

    predictor = TinyPredictor()
    expected_weight = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    expected_bias = torch.arange(3, dtype=torch.float32)

    def fake_load(*args, **kwargs):
        assert kwargs["include_predictor_vlm"] is True
        return (
            {
                "model.skill_predictor.vlm.weight": expected_weight,
                "model.skill_predictor.vlm.bias": expected_bias,
                "model.action_out_proj.weight": torch.ones(1),
            },
            True,
        )

    monkeypatch.setattr(
        "lerobot.policies.skill_vla_stage2.modeling_skill_vla_stage2."
        "_load_pretrained_state_dict",
        fake_load,
    )
    reader_before = predictor.reader.weight.detach().clone()

    loaded = _load_pi05_base_vlm_parameters(predictor, "/tmp/pi05_base")

    assert loaded == 2
    torch.testing.assert_close(predictor.vlm.weight, expected_weight)
    torch.testing.assert_close(predictor.vlm.bias, expected_bias)
    torch.testing.assert_close(predictor.reader.weight, reader_before)


def test_external_predictor_replaces_complete_vlm_and_clears_eval_cache(
    monkeypatch,
    tmp_path,
) -> None:
    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    holder = nn.Module()
    holder.skill_predictor = nn.Linear(2, 2)
    policy.model = holder
    policy._eval_vlm_cache_ids = torch.tensor([3])
    policy._eval_vlm_cache = ([torch.ones(1, 1, 1)], torch.zeros(1, 1))
    policy.config = SimpleNamespace(
        skill_vocab_size=27,
        skill_fsq_levels=[3, 3, 3],
        skill_predictor_vlm_variant="gemma_2b",
        skill_predictor_image_size=224,
        dtype="bfloat16",
    )
    source = tmp_path / "external_predictor"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "type": "skill_aux",
                "train_skill_predictor": True,
                "skill_vocab_size": 27,
                "skill_fsq_levels": [3, 3, 3],
                "skill_predictor_vlm_variant": "gemma_2b",
                "skill_predictor_image_size": 224,
            }
        )
    )
    calls = []

    class TinyExternalPredictor(nn.Linear):
        def __init__(self, _config):
            super().__init__(2, 2)

    def load_complete(predictor, path):
        calls.append((predictor, path))
        return 17

    monkeypatch.setattr(
        "lerobot.policies.skill_vla_stage2.modeling_skill_vla_stage2."
        "_load_complete_predictor_parameters",
        load_complete,
    )
    monkeypatch.setattr(
        "lerobot.policies.skill_vla_stage2.modeling_skill_vla_stage2."
        "FrozenVLMSkillPredictor",
        TinyExternalPredictor,
    )
    monkeypatch.setattr(
        "lerobot.policies.skill_vla_stage2.modeling_skill_vla_stage2."
        "_load_learned_predictor_parameters",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("eval must load the complete predictor/VLM")
        ),
    )

    policy.load_external_skill_predictor(source)

    assert len(calls) == 1
    assert calls[0][0] is holder.skill_predictor
    assert calls[0][1] == source
    assert isinstance(holder.skill_predictor, TinyExternalPredictor)
    assert not holder.skill_predictor.training
    assert not any(
        parameter.requires_grad
        for parameter in holder.skill_predictor.parameters()
    )
    assert policy._eval_vlm_cache_ids is None
    assert policy._eval_vlm_cache is None


def test_fresh_likelihood_block_is_exact_identity() -> None:
    config = CONFIG_MAPPING["gemma"](
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        num_hidden_layers=2,
        vocab_size=16,
        use_adarms=True,
        adarms_cond_dim=32,
    )
    config._attn_implementation = "eager"  # noqa: SLF001
    block = LikelihoodBlock(config, layer_index=2)
    hidden = torch.randn(2, 5, 32)
    memory = torch.randn(2, 7, 32)
    key_padding = torch.zeros(2, 7, dtype=torch.bool)
    positions = torch.arange(5)[None].expand(2, -1)
    rotary = GemmaRotaryEmbedding(config)

    output = block(
        hidden,
        memory,
        key_padding,
        torch.randn(2, 32),
        rotary(hidden, positions),
    )

    torch.testing.assert_close(output, hidden, rtol=0.0, atol=0.0)


def test_fresh_latent_expert_block_is_exact_identity() -> None:
    config = CONFIG_MAPPING["gemma"](
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        num_hidden_layers=2,
        vocab_size=16,
        use_adarms=True,
        adarms_cond_dim=32,
    )
    config._attn_implementation = "eager"  # noqa: SLF001
    block = LatentExpertBlock(config, layer_index=2)
    hidden = torch.randn(2, 4, 32)
    cond_memory = torch.randn(2, 6, 32)
    vlm_memory = torch.randn(2, 7, 32)
    cond_padding = torch.zeros(2, 6, dtype=torch.bool)
    vlm_padding = torch.zeros(2, 7, dtype=torch.bool)
    positions = torch.arange(4)[None].expand(2, -1)
    rotary = GemmaRotaryEmbedding(config)

    output = block(
        hidden,
        cond_memory,
        cond_padding,
        vlm_memory,
        vlm_padding,
        torch.randn(2, 32),
        rotary(hidden, positions),
    )

    torch.testing.assert_close(output, hidden, rtol=0.0, atol=0.0)


def test_stage2_optimizer_contains_only_injected_path_and_action_head() -> None:
    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    holder = nn.Module()
    holder.prior = nn.Linear(2, 2)
    holder.prior.requires_grad_(False)
    holder.vlm_to_expert_projection = nn.Linear(2, 2)
    holder.likelihood_blocks = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
    holder.action_out_proj = nn.Linear(2, 1)
    policy.model = holder
    policy.config = SimpleNamespace(likelihood_gate_lr_scale=1.0)

    groups = policy.get_optim_params()
    optimized = {id(parameter) for parameter in groups[0]["params"]}
    expected = {
        id(parameter)
        for module in (
            holder.vlm_to_expert_projection,
            holder.likelihood_blocks,
            holder.action_out_proj,
        )
        for parameter in module.parameters()
    }

    assert len(groups) == 1
    assert optimized == expected
    assert not optimized & {id(parameter) for parameter in holder.prior.parameters()}


def test_dsbc_freezes_stage1_head_and_optimizes_only_selector_path() -> None:
    model = SkillVLAStage2Pytorch.__new__(SkillVLAStage2Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(stage2_mode="dsbc")
    model.prior = nn.Linear(2, 2)
    model.vlm_to_expert_projection = nn.Linear(2, 2)
    model.likelihood_blocks = nn.ModuleList([nn.Linear(2, 2)])
    model.action_out_proj = nn.Linear(2, 1)
    model.noise_out_proj = nn.Linear(2, 1)
    model.likelihood_layer_mix = None

    model._freeze_stage1_prior()

    assert not any(parameter.requires_grad for parameter in model.prior.parameters())
    assert not any(
        parameter.requires_grad for parameter in model.action_out_proj.parameters()
    )
    assert all(
        parameter.requires_grad
        for module in (
            model.vlm_to_expert_projection,
            model.likelihood_blocks,
            model.noise_out_proj,
        )
        for parameter in module.parameters()
    )

    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    policy.model = model
    policy.config = SimpleNamespace(
        stage2_mode="dsbc",
        likelihood_gate_lr_scale=1.0,
    )
    optimized = {
        id(parameter)
        for parameter in policy.get_optim_params()[0]["params"]
    }
    assert optimized == {
        id(parameter)
        for module in (
            model.vlm_to_expert_projection,
            model.likelihood_blocks,
            model.noise_out_proj,
        )
        for parameter in module.parameters()
    }


def test_per_chunk_latent_expert_parameters_are_all_optimized() -> None:
    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    holder = nn.Module()
    holder.vlm_to_expert_projection = nn.Linear(2, 2)
    holder.vlm_to_expert_projection.requires_grad_(False)
    holder.likelihood_blocks = nn.ModuleList([nn.Linear(2, 2)])
    holder.action_out_proj = nn.Linear(2, 1)
    holder.action_out_proj.requires_grad_(False)
    holder.noise_out_proj = nn.Linear(2, 1)
    holder.latent_head = nn.Linear(2, 2)
    holder.latent_expert_cond_projection = nn.Linear(2, 2)
    holder.latent_expert_vlm_projection = nn.Linear(2, 2)
    holder.latent_expert_skill_projection = nn.Linear(1, 2)
    holder.latent_expert_blocks = nn.ModuleList([nn.Linear(2, 2)])
    holder.latent_expert_queries = nn.Parameter(torch.randn(1, 2, 2))
    holder.latent_expert_cond_layer_mix = nn.Parameter(torch.randn(1, 2))
    holder.latent_expert_vlm_layer_mix = nn.Parameter(torch.randn(1, 2))
    policy.model = holder
    policy.config = SimpleNamespace(
        stage2_mode="dsbc",
        likelihood_gate_lr_scale=1.0,
    )

    groups = policy.get_optim_params()
    optimized = {id(parameter) for parameter in groups[0]["params"]}
    expected_modules = (
        holder.likelihood_blocks,
        holder.noise_out_proj,
        holder.latent_head,
        holder.latent_expert_cond_projection,
        holder.latent_expert_vlm_projection,
        holder.latent_expert_skill_projection,
        holder.latent_expert_blocks,
    )
    expected = {
        id(parameter)
        for module in expected_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    expected.update(
        {
            id(holder.latent_expert_queries),
            id(holder.latent_expert_cond_layer_mix),
            id(holder.latent_expert_vlm_layer_mix),
        }
    )
    assert optimized == expected


def test_stage2_layer_mix_memories_start_near_the_last_layer() -> None:
    model = SkillVLAStage2Pytorch.__new__(SkillVLAStage2Pytorch)
    nn.Module.__init__(model)
    model.likelihood_blocks = nn.ModuleList([nn.Identity(), nn.Identity()])
    model.vlm_to_expert_projection = nn.Identity()
    model.vlm_to_expert_projection.weight = nn.Parameter(torch.zeros(1))
    mix = torch.zeros(2, 3)
    mix[:, -1] = 5.0
    model.likelihood_layer_mix = nn.Parameter(mix)
    stack = torch.stack(
        [torch.full((1, 4, 2), float(layer)) for layer in (1.0, 2.0, 3.0)], dim=1
    )

    memories = model._likelihood_memories(stack)

    assert len(memories) == 2
    weights = torch.softmax(model.likelihood_layer_mix, dim=-1)
    assert weights[0, -1].item() > 0.97
    expected = (weights[0, :, None, None] * stack[0]).sum(dim=0)
    torch.testing.assert_close(memories[0][0], expected)

    # Without the mix, every block shares one projected last-hidden memory.
    model.likelihood_layer_mix = None
    shared = model._likelihood_memories(torch.randn(1, 4, 2))
    assert len(shared) == 2 and shared[0] is shared[1]


def test_stage2_gate_lr_scale_splits_bootstrap_parameters() -> None:
    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    holder = nn.Module()
    holder.vlm_to_expert_projection = nn.Linear(2, 2)
    holder.likelihood_blocks = nn.ModuleList([nn.Linear(2, 2)])
    holder.action_out_proj = nn.Linear(2, 1)
    policy.model = holder
    policy.config = SimpleNamespace(
        likelihood_gate_lr_scale=10.0, optimizer_lr=2.5e-5
    )

    groups = policy.get_optim_params()

    assert len(groups) == 2
    assert groups[1]["lr"] == pytest.approx(2.5e-4)
    boosted = {id(parameter) for parameter in groups[1]["params"]}
    assert boosted == {
        id(parameter)
        for parameter in holder.vlm_to_expert_projection.parameters()
    }


def test_stage2_freeze_contract_violation_is_detected() -> None:
    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    holder = nn.Module()
    holder.prior = nn.Linear(2, 2)  # stays trainable -> contract violation
    holder.vlm_to_expert_projection = nn.Linear(2, 2)
    holder.likelihood_blocks = nn.ModuleList([nn.Linear(2, 2)])
    holder.action_out_proj = nn.Linear(2, 1)
    policy.model = holder

    with pytest.raises(RuntimeError, match="freeze contract"):
        policy.get_optim_params()


def test_stage2_has_no_auxiliary_training_step() -> None:
    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)

    assert policy.isolated_auxiliary_step({}, None, 1.0) == {}


def test_stage2_forward_preserves_inherited_conditioning_route_in_frozen_prior() -> None:
    captured = {}

    class _Predictor:
        @staticmethod
        def encode_base_last_hidden(images, language_tokens, language_mask):
            del language_tokens, language_mask
            captured["vlm_images"] = images
            return torch.zeros(1, 2, 2), torch.zeros(1, 2, dtype=torch.bool)

    model = SkillVLAStage2Pytorch.__new__(SkillVLAStage2Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(cumulative_xyz_loss_enabled=True)
    model.uses_expert_context_tokens = False
    model.uses_cond_state_adarms = True
    model.likelihood_layer_mix = None
    condition_state = torch.tensor([[9.0, 8.0]])
    condition_skill = torch.tensor([[7.0, 6.0]])
    expert_time = torch.tensor([[5.0, 4.0]])
    model.skill_predictor = _Predictor()
    model._likelihood_memories = lambda hidden: [hidden]

    def condition_tokens(images, batch_size=None):
        del batch_size
        captured["vsa_images"] = images
        return torch.zeros(1, 2, 2)

    model._condition_tokens = condition_tokens
    model._project_state = lambda state: condition_state
    model._project_expert_state = lambda state, shared: None
    model._expert_condition = lambda time, state=None, skill=None: expert_time
    model._skill_broadcasts = lambda code: (condition_skill, None)
    model._condition_state_start_index = lambda tokens: None

    def run_prior(
        condition,
        actions,
        routed_state,
        expert_condition,
        routed_condition_skill,
        expert_skill,
        condition_state_start_index=None,
    ):
        del condition, condition_state_start_index
        captured["state"] = routed_state
        captured["expert_condition"] = expert_condition
        captured["condition_skill"] = routed_condition_skill
        captured["expert_skill"] = expert_skill
        return actions

    model._run_joint_hidden = run_prior
    model._likelihood_velocity = (
        lambda prior, vlm, padding, expert: torch.zeros_like(prior)
    )
    actions = torch.ones(1, 3, 2)
    residual = model.forward(
        [torch.tensor([1.0])],
        [torch.tensor([2.0])],
        torch.zeros(1, 2),
        torch.tensor([0]),
        actions,
        torch.zeros(1, 1, dtype=torch.long),
        torch.ones(1, 1, dtype=torch.bool),
        noise=torch.zeros_like(actions),
        time=torch.full((1,), 0.5),
    )

    assert captured["vsa_images"][0].item() == 1.0
    assert captured["vlm_images"][0].item() == 2.0
    assert captured["state"] is condition_state
    assert captured["expert_condition"] is expert_time
    assert captured["condition_skill"] is condition_skill
    assert captured["expert_skill"] is None
    # likelihood receives only K action positions from the frozen prior.
    assert residual.shape == actions.shape
    torch.testing.assert_close(
        model._last_predicted_actions,
        torch.full_like(actions, 0.5),
    )


def test_stage2_memory_disables_skill_lora_but_predictor_memory_keeps_it(
    monkeypatch,
) -> None:
    predictor = FrozenVLMSkillPredictor.__new__(FrozenVLMSkillPredictor)
    nn.Module.__init__(predictor)
    predictor._embed_prefix = lambda images, tokens, mask: (
        torch.zeros(1, 2, 3),
        torch.ones(1, 2, dtype=torch.bool),
        torch.zeros(1, 2, dtype=torch.bool),
    )
    predictor._encode_prefix = lambda prefix, valid, all_layers: (prefix + 1.0, None)
    selected = []
    monkeypatch.setattr(
        modeling_skill_predictor,
        "set_active_adapters",
        lambda names: selected.append(set(names)),
    )
    tokens = torch.zeros(1, 1, dtype=torch.long)
    mask = torch.ones(1, 1, dtype=torch.bool)

    base_hidden, base_padding = predictor.encode_base_last_hidden([], tokens, mask)
    adapted_hidden, adapted_padding = predictor.encode_last_hidden([], tokens, mask)

    assert selected == [set(), {"skill"}]
    torch.testing.assert_close(base_hidden, adapted_hidden)
    torch.testing.assert_close(base_padding, adapted_padding)


@pytest.mark.parametrize("output_mode", ["shared", "per_step"])
def test_dsbc_selector_uses_fixed_t1_anchor_and_configurable_noise_shape(
    output_mode: str,
) -> None:
    model = SkillVLAStage2Pytorch.__new__(SkillVLAStage2Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        stage2_mode="dsbc",
        dsbc_noise_output_mode=output_mode,
    )
    model.action_in_proj = nn.Linear(2, 2, bias=False)
    model.noise_out_proj = nn.Identity()
    model.dsbc_anchor_noise = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]
    )
    hidden = torch.tensor(
        [[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]]
    )
    captured = {}
    model._condition_tokens = lambda images, batch_size=None: torch.zeros(1, 1, 2)

    def prior(condition, anchor, state, skill, time):
        del condition, state, skill
        captured["anchor"] = anchor.detach().clone()
        captured["time"] = time.detach().clone()
        return hidden, torch.zeros(1, 2)

    def vlm(start_images, tokens, mask):
        del tokens, mask
        captured["start_images"] = start_images
        return torch.zeros(1, 1, 2), torch.zeros(1, 1, dtype=torch.bool)

    model._prior_action_hidden = prior
    model._encode_likelihood_memory = vlm
    model._likelihood_memories = lambda vlm_hidden: [vlm_hidden]
    model._run_likelihood_blocks = (
        lambda prior_hidden, memories, padding, condition: prior_hidden
    )
    start_images = [torch.tensor([9.0])]

    prediction = model._dsbc_noise_prediction(
        [torch.tensor([8.0])],
        start_images,
        torch.zeros(1, 2),
        torch.tensor([7]),
        torch.zeros(1, 1, dtype=torch.long),
        torch.ones(1, 1, dtype=torch.bool),
    )

    torch.testing.assert_close(captured["anchor"], model.dsbc_anchor_noise)
    torch.testing.assert_close(captured["time"], torch.ones(1))
    assert captured["start_images"] is start_images
    expected = hidden.mean(dim=1) if output_mode == "shared" else hidden
    torch.testing.assert_close(prediction, 5.0 * torch.tanh(expected))
    assert prediction.abs().max() <= 5.0


def test_all_layer_dsbc_reader_uses_frozen_expert_stack_without_vlm_memory() -> None:
    model = SkillVLAStage2Pytorch.__new__(SkillVLAStage2Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        stage2_mode="dsbc",
        dsbc_reader="all_layers",
        dsbc_noise_output_mode="per_step",
    )
    model.action_in_proj = nn.Linear(2, 2, bias=False)
    model.noise_out_proj = nn.Identity()
    model.dsbc_anchor_noise = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    final = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])
    stack = torch.stack((final + 10.0, final + 20.0), dim=1)
    mode = torch.tensor([[0.25, -0.5]])
    captured = {}
    model._condition_tokens = lambda images, batch_size=None: torch.zeros(1, 1, 2)

    def prior_stack(condition, anchor, state, skill, time, mode_latent):
        del condition, state
        captured.update(
            anchor=anchor.detach().clone(),
            skill=skill.detach().clone(),
            time=time.detach().clone(),
            mode=mode_latent.detach().clone(),
        )
        return final, torch.zeros(1, 2), stack

    def reader(prior, layers, condition, skill, mode_latent):
        del condition
        captured.update(
            prior=prior,
            layers=layers,
            reader_skill=skill,
            reader_mode=mode_latent,
        )
        return prior

    model._prior_action_hidden_stack = prior_stack
    model._run_all_layer_frs_reader = reader
    model._encode_likelihood_memory = lambda *args: (_ for _ in ()).throw(
        AssertionError("all-layer DSBC must not encode VLM reader memory")
    )

    prediction = model._dsbc_noise_prediction(
        [torch.tensor([1.0])],
        [torch.tensor([2.0])],
        torch.zeros(1, 2),
        torch.tensor([7]),
        torch.zeros(1, 1, dtype=torch.long),
        torch.ones(1, 1, dtype=torch.bool),
        mode_latent=mode,
    )

    torch.testing.assert_close(captured["anchor"], model.dsbc_anchor_noise)
    torch.testing.assert_close(captured["time"], torch.ones(1))
    torch.testing.assert_close(captured["mode"], mode)
    assert captured["layers"] is stack
    assert captured["reader_skill"].item() == 7
    torch.testing.assert_close(prediction, 5.0 * torch.tanh(final))


def test_dsbc_training_detaches_latent_from_frs_and_noise_reader_losses() -> None:
    model = SkillVLAStage2Pytorch.__new__(SkillVLAStage2Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        max_action_dim=2,
        dsbc_latent_predictor_enabled=True,
    )
    model.real_action_dim = 2
    model.latent_source = nn.Parameter(torch.tensor([[0.2, -0.3]]))
    model.reader_source = nn.Parameter(torch.tensor(0.4))
    model.sample_noise = lambda shape, device: torch.zeros(shape, device=device)
    model._condition_tokens = lambda images, batch_size=None: torch.zeros(1, 1, 2)
    model._training_mode_latent = lambda *args, **kwargs: model.latent_source
    captured = {}

    def target(*args, mode_latent=None, **kwargs):
        del args, kwargs
        captured["target_mode_requires_grad"] = mode_latent.requires_grad
        return torch.zeros(1, 2, 2)

    def prediction(*args, mode_latent=None, **kwargs):
        del args, kwargs
        captured["reader_mode_requires_grad"] = mode_latent.requires_grad
        return model.reader_source.expand(1, 2, 2)

    def latent_residual(
        condition, state, skill, actions, predicted_noise, padding, mode_latent
    ):
        del condition, state, skill, actions, padding
        captured["latent_noise_requires_grad"] = predicted_noise.requires_grad
        return mode_latent[:, None, None, :].expand(1, 2, 2, 2)

    model._frs_target_noise = target
    model._dsbc_noise_prediction = prediction
    model._dsbc_latent_flow_residual = latent_residual

    pred, target_noise, _, latent = model.dsbc_training_pair(
        [],
        [],
        torch.zeros(1, 2),
        torch.tensor([1]),
        torch.zeros(1, 2, 2),
        torch.zeros(1, 1, dtype=torch.long),
        torch.ones(1, 1, dtype=torch.bool),
    )
    loss = (pred - target_noise).square().mean() + latent.square().mean()
    loss.backward()

    assert captured["target_mode_requires_grad"] is False
    assert captured["reader_mode_requires_grad"] is False
    assert captured["latent_noise_requires_grad"] is False
    assert model.reader_source.grad is not None
    assert model.latent_source.grad is not None


def test_dsbc_skill_only_latent_supervision_reuses_one_noise_across_times() -> None:
    model = SkillVLAStage2Pytorch.__new__(SkillVLAStage2Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(dsbc_latent_timesteps=3)
    model.real_action_dim = 2
    source = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
    model.sample_noise = lambda shape, device: source.to(device)
    model.sample_time = lambda batch_size, device: torch.full(
        (batch_size,), 0.5, device=device
    )
    observed_noise = []

    def residual(actions, skill, is_pad, *, time, noise, state, mode_latent):
        del skill, is_pad, time, state
        observed_noise.append(noise)
        return mode_latent[:, None, :].expand(-1, actions.shape[1], -1)

    model._skill_only_flow_residual = residual
    latent = torch.tensor([[0.2, -0.3]], requires_grad=True)
    result = model._dsbc_latent_skill_only_flow_residual(
        None,
        torch.tensor([4]),
        torch.zeros(1, 4, 6),
        torch.tensor([[False, False, True, True]]),
        latent,
    )

    assert result.shape == (1, 3, 4, 2)
    assert all(value is observed_noise[0] for value in observed_noise)
    result.square().mean().backward()
    assert latent.grad is not None


def test_per_chunk_final_latent_uses_zero_anchor_vsa_hidden() -> None:
    model = SkillVLAStage2Pytorch.__new__(SkillVLAStage2Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        dsbc_latent_predictor_enabled=True,
        dsbc_latent_predictor_mode="per_chunk_final",
        skill_flow_latent_dim=2,
    )
    model.action_in_proj = nn.Linear(2, 2)
    model.dsbc_anchor_noise = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]]]
    )
    model.latent_head = nn.Sequential(
        nn.Identity(), nn.Linear(2, 2, bias=False)
    )
    with torch.no_grad():
        model.latent_head[1].weight.copy_(torch.eye(2))
    model.latent_final_skill_projection = nn.Sequential(
        nn.Linear(1, 2, bias=False), nn.Identity(), nn.Identity()
    )
    with torch.no_grad():
        model.latent_final_skill_projection[0].weight.copy_(
            torch.tensor([[0.1], [0.2]])
        )
    model._code_to_zq = lambda skill: skill.float().unsqueeze(-1)
    captured = {}
    model._condition_tokens = lambda images, batch_size=None: torch.zeros(
        batch_size, 1, 2
    )

    def prior(condition, anchor, state, skill, time, mode_latent):
        del condition, state, skill
        captured["anchor"] = anchor.detach().clone()
        captured["time"] = time.detach().clone()
        captured["mode_latent"] = mode_latent.detach().clone()
        hidden = torch.tensor([[[0.2, -0.4], [0.2, -0.4]]])
        return hidden, torch.zeros(1, 2)

    model._prior_action_hidden = prior
    model._encode_latent_final_memory = lambda *args: (
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, dtype=torch.bool),
    )
    model._latent_final_memories = lambda hidden: [hidden]
    def run_reader(prior_hidden, memories, mask, condition):
        del memories, mask
        captured["reader_condition"] = condition.detach().clone()
        return prior_hidden

    model._run_latent_final_blocks = run_reader

    predicted = model._predict_per_chunk_mode_latent(
        [torch.zeros(1, 3, 2, 2)],
        [torch.zeros(1, 3, 2, 2)],
        torch.zeros(1, 2),
        torch.tensor([4]),
        torch.zeros(1, 1, dtype=torch.long),
        torch.ones(1, 1, dtype=torch.bool),
    )

    torch.testing.assert_close(captured["anchor"], model.dsbc_anchor_noise)
    torch.testing.assert_close(captured["time"], torch.ones(1))
    torch.testing.assert_close(captured["mode_latent"], torch.zeros(1, 2))
    torch.testing.assert_close(
        captured["reader_condition"], torch.tensor([[0.4, 0.8]])
    )
    torch.testing.assert_close(
        predicted, torch.tanh(torch.tensor([[0.2, -0.4]]))
    )


def test_per_chunk_expert_latent_uses_queries_cond_vlm_and_skill_without_anchor() -> None:
    model = SkillVLAStage2Pytorch.__new__(SkillVLAStage2Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        dsbc_latent_predictor_enabled=True,
        dsbc_latent_predictor_mode="per_chunk_expert",
    )
    model.action_in_proj = nn.Linear(2, 2)
    model.latent_head = nn.Sequential(
        nn.Identity(), nn.Linear(2, 2, bias=False)
    )
    with torch.no_grad():
        model.latent_head[1].weight.copy_(torch.eye(2))
    model.latent_expert_queries = nn.Parameter(torch.zeros(1, 1, 2))
    model.latent_expert_blocks = nn.ModuleList([nn.Identity()])
    model.latent_expert_skill_projection = nn.Sequential(
        nn.Linear(1, 2, bias=False), nn.Identity(), nn.Identity()
    )
    with torch.no_grad():
        model.latent_expert_skill_projection[0].weight.copy_(
            torch.tensor([[0.25], [-0.5]])
        )
    model._code_to_zq = lambda skill: skill.float().unsqueeze(-1)
    captured = {}
    model._condition_tokens = lambda images, batch_size=None: torch.full(
        (batch_size, 2, 2), 3.0
    )

    def encode_cond(condition, state, skill):
        captured["cond_input"] = condition.detach().clone()
        captured["cond_state"] = state.detach().clone()
        captured["cond_skill"] = skill.detach().clone()
        return torch.full((1, 1, 2, 2), 4.0)

    def encode_vlm(images, tokens, mask):
        del images
        captured["language"] = tokens.detach().clone()
        captured["language_mask"] = mask.detach().clone()
        return torch.full((1, 1, 3, 2), 5.0), torch.zeros(
            1, 3, dtype=torch.bool
        )

    model._encode_latent_expert_cond_stack = encode_cond
    model._latent_expert_cond_memories = lambda stack: [stack[:, 0]]
    model._encode_latent_expert_vlm_stack = encode_vlm
    model._latent_expert_vlm_memories = lambda stack: [stack[:, 0]]

    def run_expert(cond, cond_mask, vlm, vlm_mask, skill_condition):
        captured["cond_memory"] = cond[0].detach().clone()
        captured["cond_mask"] = cond_mask.detach().clone()
        captured["vlm_memory"] = vlm[0].detach().clone()
        captured["vlm_mask"] = vlm_mask.detach().clone()
        captured["skill_condition"] = skill_condition.detach().clone()
        return torch.tensor([[[0.3, -0.2]]])

    model._run_latent_expert_blocks = run_expert
    model._prior_action_hidden = lambda *args, **kwargs: pytest.fail(
        "per_chunk_expert must not run the fixed action anchor"
    )
    language = torch.tensor([[7, 8, 9]])
    language_mask = torch.ones(1, 3, dtype=torch.bool)
    predicted = model._predict_per_chunk_expert_mode_latent(
        [torch.zeros(1, 3, 2, 2)],
        [torch.ones(1, 3, 2, 2)],
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([2]),
        language,
        language_mask,
    )

    torch.testing.assert_close(captured["cond_input"], torch.full((1, 2, 2), 3.0))
    torch.testing.assert_close(captured["cond_state"], torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(captured["cond_skill"], torch.tensor([2]))
    torch.testing.assert_close(captured["language"], language)
    torch.testing.assert_close(captured["language_mask"], language_mask)
    torch.testing.assert_close(
        captured["skill_condition"], torch.tensor([[0.5, -1.0]])
    )
    assert not bool(captured["cond_mask"].any())
    torch.testing.assert_close(
        predicted, torch.tanh(torch.tensor([[0.3, -0.2]]))
    )


def test_online_frs_runs_action_to_noise_and_forces_linear_padding_path() -> None:
    model = SkillVLAStage2Pytorch.__new__(SkillVLAStage2Pytorch)
    nn.Module.__init__(model)
    model.real_action_dim = 1
    model.action_in_proj = nn.Linear(2, 2, bias=False)
    model.action_out_proj = nn.Identity()
    model._visual_context_cache = lambda condition, context, state: object()
    times = []
    noisy_states = []

    def expert_condition(time):
        times.append(time.detach().clone())
        return torch.zeros(time.shape[0], 2)

    def action_hidden(noisy, condition, skill, cache, attention, positions):
        del condition, skill, cache, attention, positions
        noisy_states.append(noisy.detach().clone())
        return noisy

    model._expert_condition = expert_condition
    model._action_hidden_with_condition_cache = action_hidden
    target = model._frs_reverse_with_expert_context_cache(
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 2),
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[2.0]]]),
        num_steps=2,
        condition_state=None,
    )

    torch.testing.assert_close(target, torch.tensor([[[2.25]]]))
    torch.testing.assert_close(torch.stack(times).flatten(), torch.tensor([0.0, 0.5]))
    torch.testing.assert_close(noisy_states[0], torch.tensor([[[1.0, 0.0]]]))
    torch.testing.assert_close(noisy_states[1], torch.tensor([[[1.5, 1.0]]]))


def test_dsbc_action_metric_is_detached_legacy_flow_residual() -> None:
    model = SkillVLAStage2Pytorch.__new__(SkillVLAStage2Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(dsbc_noise_output_mode="shared")
    model.real_action_dim = 1
    model.action_in_proj = nn.Linear(2, 2, bias=False)
    model.action_out_proj = nn.Identity()
    model.sample_time = lambda batch_size, device: torch.full(
        (batch_size,), 0.5, device=device
    )

    def prior(condition, x_t, state, skill, time):
        del condition, state, skill, time
        return x_t, torch.zeros(x_t.shape[0], 2)

    model._prior_action_hidden = prior
    residual = model._dsbc_action_flow_residual(
        torch.zeros(1, 1, 2),
        torch.zeros(1, 2),
        torch.tensor([7]),
        torch.tensor([[[1.0, 0.0], [3.0, 0.0]]]),
        torch.tensor([[5.0]], requires_grad=True),
        torch.zeros(1, 2, 1),
    )

    torch.testing.assert_close(residual, torch.tensor([[[1.0], [-2.0]]]))
    assert not residual.requires_grad


class _CaptureStage2Residual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.seen_actions = None
        self.seen_images = None
        self.seen_vlm_start_images = None

    def forward(
        self,
        images,
        vlm_start_images,
        state,
        skill_code,
        actions,
        language_tokens,
        language_mask,
    ):
        del state, skill_code, language_tokens, language_mask
        self.seen_images = images
        self.seen_vlm_start_images = vlm_start_images
        self.seen_actions = actions.detach().clone()
        self._last_predicted_actions = actions.clone()
        self._last_predicted_actions[:, :2, 0] += 1.0
        return torch.tensor(
            [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [100.0, 100.0, 100.0]]],
            device=actions.device,
        )


def test_stage2_mixes_flow_and_cumulative_xyz_like_stage1() -> None:
    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        max_action_dim=3,
        max_state_dim=2,
        output_features={ACTION: SimpleNamespace(shape=(3,))},
        conditioning_route="state_cond",
        cumulative_xyz_loss_enabled=True,
        cumulative_xyz_loss_weight=0.5,
        training_skill_source="gt",
    )
    policy.model = _CaptureStage2Residual()
    current_images = [torch.tensor([1.0])]
    start_images = [torch.tensor([2.0])]
    policy._collect_images = lambda batch: current_images
    policy._predictor_start_images = lambda batch: start_images
    policy._training_skill_code = lambda batch: torch.zeros(1, dtype=torch.long)
    policy._last_transition_jitter_fraction = torch.zeros(())
    actions = torch.tensor(
        [[[1.0, 2.0, 0.1], [3.0, 4.0, 0.2], [5.0, 6.0, 0.3]]]
    )
    batch = {
        ACTION: actions,
        OBS_STATE: torch.zeros(1, 2),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 1, dtype=torch.bool),
        "action_is_pad": torch.tensor([[False, False, True]]),
    }

    loss, metrics = policy.forward(batch)

    assert policy.model.seen_images is current_images
    assert policy.model.seen_vlm_start_images is start_images
    torch.testing.assert_close(policy.model.seen_actions, actions)
    # The fake model adds +1.0 XYZ error to the first two (valid) steps, so the
    # masked prefix cumulative squared error is [1/3, 4/3] -> raw 5/6, and the
    # (valid_steps + 1) / 2 = 1.5 horizon normalization gives 5/9.
    assert metrics["action_loss"] == pytest.approx(2.5)
    assert metrics["cumulative_xyz/raw"] == pytest.approx(5.0 / 6.0)
    assert metrics["cumulative_xyz/normalized"] == pytest.approx(5.0 / 9.0)
    assert metrics["cumulative_xyz/weighted"] == pytest.approx(5.0 / 18.0)
    assert metrics["action_flow_weight"] == 1.0
    assert metrics["action_cumulative_xyz_weight"] == 0.5
    torch.testing.assert_close(loss, torch.tensor(2.5 + 5.0 / 18.0))


class _CaptureDSBCPair(nn.Module):
    def __init__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        action_residual: torch.Tensor,
    ) -> None:
        super().__init__()
        self.real_action_dim = target.shape[-1]
        self.prediction = prediction
        self.target = target
        self.action_residual = action_residual
        self.seen = None

    def dsbc_training_pair(
        self,
        images,
        start_images,
        state,
        skill,
        actions,
        tokens,
        mask,
    ):
        del state, actions, tokens, mask
        self.seen = (images, start_images, skill.detach().clone())
        return self.prediction, self.target, self.action_residual


def _dsbc_policy(
    output_mode: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    action_residual: torch.Tensor | None = None,
):
    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        stage2_mode="dsbc",
        max_action_dim=3,
        max_state_dim=2,
        output_features={ACTION: SimpleNamespace(shape=(2,))},
        conditioning_route="state_skill_cond",
        training_skill_source="gt",
        dsbc_noise_output_mode=output_mode,
        dsbc_frs_num_steps=10,
        same_skill_batch_enabled=False,
        mask_actions_after_skill_end=False,
    )
    if action_residual is None:
        action_residual = torch.zeros_like(target)
    policy.model = _CaptureDSBCPair(prediction, target, action_residual)
    current_images = [torch.tensor([1.0])]
    start_images = [torch.tensor([2.0])]
    policy._collect_images = lambda batch: current_images
    policy._predictor_start_images = lambda batch: start_images
    policy._training_skill_code = lambda batch: torch.tensor([7])
    policy._last_transition_jitter_fraction = torch.ones(())
    return policy, current_images, start_images


def test_dsbc_shared_mode_supervises_mean_valid_frs_noise() -> None:
    target = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
    prediction = torch.tensor([[1.0, 2.0]], requires_grad=True)
    policy, current_images, start_images = _dsbc_policy(
        "shared", prediction, target
    )
    batch = {
        ACTION: torch.zeros(1, 3, 2),
        OBS_STATE: torch.zeros(1, 2),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 1, dtype=torch.bool),
        "action_is_pad": torch.tensor([[False, False, True]]),
    }

    loss, metrics = policy.forward(batch)

    torch.testing.assert_close(loss, torch.tensor(2.5))
    assert metrics["action_loss"] == pytest.approx(0.0)
    assert metrics["gt_noise_loss"] == pytest.approx(2.5)
    assert metrics["noise_loss"] == pytest.approx(2.5)
    assert metrics["dsbc/supervision_target_rms"] == pytest.approx(10.0**0.5)
    assert metrics["dsbc/frs_target_valid_rms"] == pytest.approx(11.0**0.5)
    assert metrics["dsbc/supervision_target_abs_max"] == pytest.approx(4.0)
    assert metrics["dsbc/frs_target_abs_max"] == pytest.approx(5.0)
    assert metrics["dsbc/supervision_target_outside_bound_fraction"] == 0.0
    assert metrics["dsbc/frs_target_outside_bound_fraction"] == 0.0
    assert metrics["dsbc/noise_output_bound"] == pytest.approx(5.0)
    assert metrics["regime/transition_jitter_fraction"] == pytest.approx(1.0)
    assert policy.model.seen[0] is current_images
    assert policy.model.seen[1] is start_images
    torch.testing.assert_close(policy.model.seen[2], torch.tensor([7]))


def test_dsbc_per_step_mode_masks_padded_frs_targets() -> None:
    target = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
    prediction = (target + 1.0).requires_grad_()
    action_residual = torch.tensor(
        [[[1.0, 1.0], [2.0, 2.0], [100.0, 100.0]]]
    )
    policy, _, _ = _dsbc_policy(
        "per_step", prediction, target, action_residual
    )
    batch = {
        ACTION: torch.zeros(1, 3, 2),
        OBS_STATE: torch.zeros(1, 2),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 1, dtype=torch.bool),
        "action_is_pad": torch.tensor([[False, False, True]]),
    }

    loss, metrics = policy.forward(batch)

    torch.testing.assert_close(loss, torch.tensor(1.0))
    assert metrics["action_loss"] == pytest.approx(2.5)
    assert metrics["gt_noise_loss"] == pytest.approx(1.0)
    assert metrics["noise_loss"] == pytest.approx(1.0)


def test_dsbc_skill_only_latent_loss_uses_canonical_padding_mask() -> None:
    class _CanonicalPair(nn.Module):
        real_action_dim = 2
        _last_mode_latent = None

        def dsbc_training_pair(
            self,
            images,
            start_images,
            state,
            skill,
            actions,
            tokens,
            mask,
            canonical_actions,
            canonical_action_is_pad,
        ):
            del images, start_images, state, skill, actions, tokens, mask
            assert canonical_actions.shape == (1, 4, 3)
            torch.testing.assert_close(
                canonical_action_is_pad,
                torch.tensor([[False, False, True, True]]),
            )
            prediction = torch.zeros(1, 3, 2, requires_grad=True)
            target = torch.zeros_like(prediction)
            # Valid canonical steps have residuals 1 and 3; padded steps are
            # deliberately huge and must not contribute.
            latent = torch.tensor(
                [[[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0], [100.0, 100.0]]]],
                requires_grad=True,
            )
            return prediction, target, latent.detach(), latent

    policy, _, _ = _dsbc_policy(
        "per_step",
        torch.zeros(1, 3, 2),
        torch.zeros(1, 3, 2),
    )
    policy.config.dsbc_latent_predictor_enabled = True
    policy.config.dsbc_latent_supervision = "skill_only"
    policy.config.dsbc_latent_loss_weight = 1.0
    policy.config.dsbc_latent_timesteps = 1
    policy.model = _CanonicalPair()
    batch = {
        ACTION: torch.zeros(1, 3, 2),
        OBS_STATE: torch.zeros(1, 2),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 1, dtype=torch.bool),
        "action_is_pad": torch.zeros(1, 3, dtype=torch.bool),
        SKILL_CANONICAL_ACTIONS: torch.zeros(1, 4, 2),
        SKILL_CANONICAL_ACTION_IS_PAD: torch.tensor(
            [[False, False, True, True]]
        ),
    }

    loss, metrics = policy.forward(batch)

    # Mean over two valid steps and two real action dimensions:
    # (1^2 + 1^2 + 3^2 + 3^2) / 4 = 5.
    torch.testing.assert_close(loss, torch.tensor(5.0))
    assert metrics["latent/action_loss"] == pytest.approx(5.0)
    assert metrics["latent/supervision_skill_only"] == 1.0


def test_stage2_same_skill_metrics_use_post_jitter_code_and_different_task() -> None:
    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(same_skill_batch_enabled=True)
    batch = {
        "same_skill_pair_id": torch.tensor([0, 0, -1, -1]),
        "same_skill_pair_fallback": torch.tensor([False, False, False, False]),
        "skill_code": torch.tensor([7, 7, 3, 4]),
        "task_index": torch.tensor([1, 2, 3, 4]),
        "skill_progress": torch.tensor([0.25, 0.30, 0.4, 0.8]),
    }

    metrics = policy._same_skill_batch_metrics(
        batch, torch.tensor([7, 7, 3, 4])
    )

    assert metrics["batch_sampling/constructed_fraction"] == pytest.approx(0.5)
    assert metrics["batch_sampling/effective_after_jitter_fraction"] == pytest.approx(0.5)
    assert metrics["batch_sampling/effective_conditioning_fraction"] == pytest.approx(0.5)
    assert metrics["batch_sampling/jittered_progress_gap"] == pytest.approx(0.05)


def test_stage2_processor_preserves_same_skill_sampler_metadata() -> None:
    batch = {
        "same_skill_pair_id": torch.tensor([0, 0, -1, -1]),
        "same_skill_pair_fallback": torch.tensor([False, False, False, True]),
        "skill_progress": torch.tensor([0.1, 0.2, 0.3, 0.4]),
        "skill_effective_de": torch.tensor([4, 3, 2, 1]),
    }

    restored = skill_expert_transition_to_batch(
        skill_expert_batch_to_transition(batch)
    )

    for key, expected in batch.items():
        torch.testing.assert_close(restored[key], expected)


def test_stage2_sampling_keeps_condition_routing_and_likelihood_sees_actions_only() -> None:
    class _ConditionModel:
        @staticmethod
        def forward(**kwargs):
            del kwargs
            return SimpleNamespace(past_key_values=[("k", "v")])

    class _Predictor:
        @staticmethod
        def encode_base_last_hidden(images, language_tokens, language_mask):
            del images, language_tokens, language_mask
            return torch.zeros(1, 2, 2), torch.zeros(1, 2, dtype=torch.bool)

    model = SkillVLAStage2Pytorch.__new__(SkillVLAStage2Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_inference_steps=1,
        chunk_size=3,
        max_action_dim=2,
    )
    model.uses_expert_context_tokens = False
    model.uses_cond_state_adarms = True
    model.likelihood_layer_mix = None
    model.cond_encoder = SimpleNamespace(model=_ConditionModel())
    model.skill_predictor = _Predictor()
    model._likelihood_memories = lambda hidden: [hidden]
    model._condition_tokens = lambda images, batch_size=None: torch.zeros(1, 2, 2)
    model._project_state = lambda state: torch.zeros(1, 2)
    model._project_expert_state = lambda state, shared: None
    model._expert_condition = lambda time, state=None, skill=None: torch.zeros(1, 2)
    model._skill_broadcasts = lambda code: (torch.tensor([[7.0, 8.0]]), None)
    model._condition_state_start_index = lambda tokens: None
    captured = {}

    def action_prior(
        noisy_actions,
        expert_condition,
        expert_skill,
        condition_cache,
        attention_mask,
        position_ids,
    ):
        del expert_condition, condition_cache, position_ids
        captured["expert_skill"] = expert_skill
        captured["attention"] = attention_mask
        return noisy_actions

    def likelihood(prior, vlm, padding, expert):
        del vlm, padding, expert
        captured["likelihood_shape"] = prior.shape
        return prior

    model._action_hidden_with_condition_cache = action_prior
    model._likelihood_velocity = likelihood
    noise = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    cached_vlm = (
        [torch.zeros(1, 2, 2)],
        torch.zeros(1, 2, dtype=torch.bool),
    )
    sampled = model.sample_actions(
        [],
        [],
        torch.zeros(1, 2),
        torch.tensor([0]),
        torch.zeros(1, 1, dtype=torch.long),
        torch.ones(1, 1, dtype=torch.bool),
        noise=noise,
        num_steps=1,
        vlm_memory=cached_vlm,
    )

    torch.testing.assert_close(sampled, torch.zeros_like(noise))
    assert captured["expert_skill"] is None
    assert captured["likelihood_shape"] == noise.shape
    allowed = captured["attention"][0, 0].eq(0)
    expected = torch.ones(3, 5, dtype=torch.bool)
    torch.testing.assert_close(allowed, expected)


@pytest.mark.parametrize(
    ("output_mode", "prediction"),
    [
        ("shared", torch.tensor([[1.0, 2.0]])),
        (
            "per_step",
            torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),
        ),
    ],
)
def test_dsbc_sampling_uses_selected_real_noise_and_preserves_gaussian_padding(
    monkeypatch,
    output_mode: str,
    prediction: torch.Tensor,
) -> None:
    model = SkillVLAStage2Pytorch.__new__(SkillVLAStage2Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        stage2_mode="dsbc",
        dsbc_noise_output_mode=output_mode,
        chunk_size=3,
        max_action_dim=4,
    )
    model.real_action_dim = 2
    captured = {}

    def predict_noise(*args, **kwargs):
        del args
        captured["vlm_memory"] = kwargs["vlm_memory"]
        return prediction

    model._dsbc_noise_prediction = predict_noise

    def base_sample(self, images, state, skill, noise=None, num_steps=None):
        del self, images, state, skill, num_steps
        captured["noise"] = noise.detach().clone()
        return noise

    monkeypatch.setattr(CondGemmaSkillExpert, "sample_actions", base_sample)
    reservoir = torch.tensor(
        [[[9.0, 9.0, 3.0, 4.0], [9.0, 9.0, 5.0, 6.0], [9.0, 9.0, 7.0, 8.0]]]
    )
    cached_vlm = (
        [torch.zeros(1, 2, 2)],
        torch.zeros(1, 2, dtype=torch.bool),
    )

    sampled = model.sample_actions(
        [],
        [],
        torch.zeros(1, 2),
        torch.tensor([7]),
        torch.zeros(1, 1, dtype=torch.long),
        torch.ones(1, 1, dtype=torch.bool),
        noise=reservoir,
        num_steps=10,
        vlm_memory=cached_vlm,
    )

    expected = reservoir.clone()
    expected[..., :2] = prediction[:, None] if output_mode == "shared" else prediction
    torch.testing.assert_close(captured["noise"], expected)
    torch.testing.assert_close(sampled, expected)
    assert captured["vlm_memory"] is cached_vlm


def test_eval_vlm_cache_refreshes_only_rows_whose_skill_generation_changes() -> None:
    class _CacheModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.likelihood_blocks = nn.ModuleList([nn.Identity(), nn.Identity()])
            self.calls = []

        def encode_likelihood_memories(self, images, tokens, mask):
            del tokens
            values = images[0].flatten(1).mean(dim=1)[:, None, None]
            self.calls.append(values.detach().clone())
            return [values, values + 100.0], mask.detach().clone()

    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(n_action_steps=2)
    policy.model = _CacheModel()
    policy.reset()
    tokens = torch.zeros(2, 1, dtype=torch.long)

    first = policy._cached_eval_vlm_memory(
        [torch.tensor([[[[1.0]]], [[[2.0]]]])],
        tokens,
        torch.ones(2, 1, dtype=torch.bool),
        torch.tensor([0, 0]),
    )
    second = policy._cached_eval_vlm_memory(
        [torch.tensor([[[[10.0]]], [[[20.0]]]])],
        tokens,
        torch.zeros(2, 1, dtype=torch.bool),
        torch.tensor([0, 0]),
    )
    third = policy._cached_eval_vlm_memory(
        [torch.tensor([[[[10.0]]], [[[20.0]]]])],
        tokens,
        torch.tensor([[False], [True]]),
        torch.tensor([0, 1]),
    )

    assert first is second is third
    assert [call.flatten().tolist() for call in policy.model.calls] == [
        [1.0, 2.0],
        [20.0],
    ]
    torch.testing.assert_close(third[0][0].flatten(), torch.tensor([1.0, 20.0]))
    torch.testing.assert_close(third[0][1].flatten(), torch.tensor([101.0, 120.0]))
    torch.testing.assert_close(
        third[1], torch.tensor([[True], [True]], dtype=torch.bool)
    )

    policy.reset()
    assert policy._eval_vlm_cache is None
    assert policy._eval_vlm_cache_ids is None


def test_eval_vlm_cache_is_opt_in_for_direct_policy_callers() -> None:
    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(n_action_steps=2)
    policy.model = nn.Module()
    policy.reset()

    assert (
        policy._cached_eval_vlm_memory(
            [],
            torch.zeros(1, 1, dtype=torch.long),
            torch.ones(1, 1, dtype=torch.bool),
            None,
        )
        is None
    )
    assert STAGE2_VLM_CACHE_ID == "stage2_vlm_cache_id"


def test_eval_mode_latent_is_cached_once_per_skill_generation() -> None:
    class _LatentModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = []

        def _predict_mode_latent(self, images, tokens, mask, skill):
            del tokens, mask
            image_value = images[0].flatten(1).mean(dim=1)
            self.calls.append((image_value.detach().clone(), skill.detach().clone()))
            return torch.stack((image_value, skill.float()), dim=-1)

    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        n_action_steps=2,
        skill_flow_latent_best_of_n_enabled=True,
        dsbc_latent_predictor_enabled=True,
    )
    policy.model = _LatentModel()
    policy.reset()
    tokens = torch.zeros(2, 1, dtype=torch.long)
    mask = torch.ones(2, 1, dtype=torch.bool)

    first = policy._cached_eval_mode_latent(
        [torch.tensor([[[[1.0]]], [[[2.0]]]])],
        tokens,
        mask,
        torch.tensor([3, 4]),
        torch.tensor([10, 20]),
    )
    second = policy._cached_eval_mode_latent(
        [torch.tensor([[[[11.0]]], [[[22.0]]]])],
        tokens,
        mask,
        torch.tensor([3, 4]),
        torch.tensor([10, 20]),
    )
    third = policy._cached_eval_mode_latent(
        [torch.tensor([[[[11.0]]], [[[22.0]]]])],
        tokens,
        mask,
        torch.tensor([3, 4]),
        torch.tensor([10, 21]),
    )
    fourth = policy._cached_eval_mode_latent(
        [torch.tensor([[[[33.0]]], [[[44.0]]]])],
        tokens,
        mask,
        torch.tensor([5, 4]),
        torch.tensor([10, 21]),
    )

    assert first is second is third is fourth
    observed_calls = [
        (values.tolist(), skills.tolist()) for values, skills in policy.model.calls
    ]
    assert observed_calls == [
        ([1.0, 2.0], [3, 4]),
        ([22.0], [4]),
        ([33.0], [5]),
    ]
    torch.testing.assert_close(fourth, torch.tensor([[33.0, 5.0], [22.0, 4.0]]))

    policy.reset()
    assert policy._eval_mode_latent_ids is None
    assert policy._eval_mode_latent_skill_codes is None
    assert policy._eval_mode_latent_cache is None
    assert policy._last_eval_predicted_mode_latent is None
    assert policy._last_eval_mode_latent is None


@pytest.mark.parametrize("mode", ["per_chunk_final", "per_chunk_expert"])
def test_per_chunk_latent_is_recomputed_for_each_action_chunk(mode: str) -> None:
    class _LatentModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def _predict_per_chunk_mode_latent(self, images, *args, **kwargs):
            del args, kwargs
            self.calls += 1
            value = images[0].flatten(1).mean(dim=1)
            return torch.stack((value, -value), dim=-1)

        def _predict_per_chunk_expert_mode_latent(self, images, *args, **kwargs):
            return self._predict_per_chunk_mode_latent(images, *args, **kwargs)

    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        dsbc_latent_predictor_enabled=True,
        dsbc_latent_predictor_mode=mode,
    )
    policy.model = _LatentModel()
    policy._cached_eval_latent_vlm_memory = lambda *args: None
    tokens = torch.zeros(1, 1, dtype=torch.long)
    mask = torch.ones(1, 1, dtype=torch.bool)
    common = (
        [torch.zeros(1, 3, 1, 1)],
        torch.zeros(1, 2),
        torch.tensor([3]),
        tokens,
        mask,
        torch.tensor([7]),
    )

    first = policy._eval_mode_latent(
        [torch.ones(1, 3, 1, 1)], *common
    )
    second = policy._eval_mode_latent(
        [torch.full((1, 3, 1, 1), 2.0)], *common
    )

    assert policy.model.calls == 2
    torch.testing.assert_close(first, torch.tensor([[1.0, -1.0]]))
    torch.testing.assert_close(second, torch.tensor([[2.0, -2.0]]))


def test_eval_mode_latent_override_replaces_only_finite_rows() -> None:
    class _SampleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.observed_latent = None

        def sample_actions(self, *args, mode_latent=None, **kwargs):
            del args, kwargs
            self.observed_latent = mode_latent.detach().clone()
            return torch.zeros(2, 3, 2)

    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        conditioning_route="state_cond",
        max_state_dim=4,
        stage2_mode="dsbc",
        dsbc_reader="all_layers",
        output_features={ACTION: SimpleNamespace(shape=(2,))},
    )
    policy.model = _SampleModel()
    policy._skill_code = lambda batch: batch["skill_code"]
    policy._predictor_start_images = lambda batch: []
    policy._collect_images = lambda batch: []
    policy._cached_eval_mode_latent = lambda *args: torch.tensor(
        [[0.1, 0.2], [0.3, 0.4]]
    )
    batch = {
        OBS_STATE: torch.zeros(2, 2),
        OBS_LANGUAGE_TOKENS: torch.zeros(2, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 1, dtype=torch.bool),
        "skill_code": torch.tensor([1, 2]),
        STAGE2_MODE_LATENT_OVERRIDE: torch.tensor(
            [[0.9, -0.8], [float("nan"), float("nan")]]
        ),
    }

    actions = policy.predict_action_chunk(batch)

    assert actions.shape == (2, 3, 2)
    torch.testing.assert_close(
        policy.model.observed_latent,
        torch.tensor([[0.9, -0.8], [0.3, 0.4]]),
    )
    torch.testing.assert_close(
        policy._last_eval_predicted_mode_latent,
        torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
    )
    torch.testing.assert_close(
        policy._last_eval_mode_latent,
        torch.tensor([[0.9, -0.8], [0.3, 0.4]]),
    )


def test_hindsight_latent_grid_selects_lowest_gt_flow_residual() -> None:
    class _OracleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.real_action_dim = 2
            self.working_dtype = torch.float32
            self.action_out_proj = nn.Identity()

        def _condition_tokens(self, images, *, batch_size):
            del images
            return torch.zeros(batch_size, 1, 2)

        def _dsbc_noise_prediction(self, *args, mode_latent=None, **kwargs):
            del args, kwargs
            return mode_latent[:, None].expand(-1, 2, -1)

        def _prior_action_hidden(
            self, condition, x_t, state, skill, time, mode_latent
        ):
            del condition, state, skill, time, mode_latent
            return torch.zeros_like(x_t), None

    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        stage2_mode="dsbc",
        dsbc_latent_predictor_enabled=True,
        skill_flow_latent_dim=2,
        conditioning_route="state_cond",
        max_state_dim=2,
        max_action_dim=2,
        chunk_size=2,
        dsbc_anchor_seed=0,
        dsbc_noise_output_mode="per_step",
        dsbc_latent_timesteps=2,
    )
    policy.model = _OracleModel()
    policy._skill_code = lambda batch: batch["skill_code"]
    policy._predictor_start_images = lambda batch: []
    policy._collect_images = lambda batch: []
    policy._cached_eval_mode_latent = lambda *args: torch.tensor([[0.75, 0.75]])
    batch = {
        OBS_STATE: torch.zeros(1, 2),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 1, dtype=torch.bool),
        "skill_code": torch.tensor([1]),
    }

    selected, scores = policy.select_hindsight_mode_latent(
        batch,
        torch.zeros(1, 2, 2),
        torch.ones(1, 2, dtype=torch.bool),
        grid_size=3,
        timesteps=2,
    )

    torch.testing.assert_close(selected, torch.zeros(1, 2))
    assert scores.shape == (1, 10)  # learned prediction + the 3x3 grid
    assert scores[0, 0] > scores.min()


def test_hindsight_latent_oracle_aggregates_one_code_over_skill_windows() -> None:
    class _OracleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.real_action_dim = 2
            self.working_dtype = torch.float32
            self.action_out_proj = nn.Identity()

        def _condition_tokens(self, images, *, batch_size):
            del images
            return torch.zeros(batch_size, 1, 2)

        def _dsbc_noise_prediction(self, *args, mode_latent=None, **kwargs):
            del args, kwargs
            return mode_latent[:, None].expand(-1, 2, -1)

        def _prior_action_hidden(
            self, condition, x_t, state, skill, time, mode_latent
        ):
            del condition, state, skill, time, mode_latent
            return torch.zeros_like(x_t), None

    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        stage2_mode="dsbc",
        dsbc_latent_predictor_enabled=True,
        skill_flow_latent_dim=2,
        conditioning_route="state_cond",
        max_state_dim=2,
        max_action_dim=2,
        chunk_size=2,
        dsbc_anchor_seed=0,
        dsbc_noise_output_mode="per_step",
        dsbc_latent_timesteps=2,
    )
    policy.model = _OracleModel()
    policy._skill_code = lambda batch: batch["skill_code"]
    policy._predictor_start_images = lambda batch: []
    policy._collect_images = lambda batch: []
    predicted_calls = []

    def _predicted(*args):
        predicted_calls.append(args)
        return torch.tensor([[0.75, 0.75]])

    policy._cached_eval_mode_latent = _predicted
    batch = {
        OBS_STATE: torch.zeros(2, 2),
        OBS_LANGUAGE_TOKENS: torch.zeros(2, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 1, dtype=torch.bool),
        "skill_code": torch.tensor([1, 1]),
        STAGE2_VLM_CACHE_ID: torch.tensor([4, 4]),
    }

    selected, scores = policy.select_hindsight_mode_latent(
        batch,
        torch.zeros(2, 2, 2),
        torch.tensor([[True, True], [True, False]]),
        grid_size=3,
        timesteps=2,
        aggregate_windows=True,
    )

    torch.testing.assert_close(selected, torch.zeros(1, 2))
    assert scores.shape == (1, 10)
    # One skill-start prediction, with the rollout cache deliberately bypassed
    # while vector-env members are scored independently.
    assert predicted_calls[0][-1] is None
    assert predicted_calls[0][1].shape[0] == 1
