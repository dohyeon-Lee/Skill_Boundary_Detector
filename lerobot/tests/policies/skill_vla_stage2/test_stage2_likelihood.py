from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding

from lerobot.policies.skill_expert import modeling_skill_predictor
from lerobot.policies.skill_expert.modeling_skill_predictor import (
    FrozenVLMSkillPredictor,
)
from lerobot.policies.skill_expert.processor_skill_expert import (
    skill_expert_batch_to_transition,
    skill_expert_transition_to_batch,
)
from lerobot.policies.skill_vla_stage2.configuration_skill_vla_stage2 import (
    SkillVLAStage2Config,
)
from lerobot.policies.skill_vla_stage2.modeling_skill_vla_stage2 import (
    LikelihoodBlock,
    SkillVLAStage2Policy,
    SkillVLAStage2Pytorch,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


def _config(**overrides) -> SkillVLAStage2Config:
    values = {
        "stage1_checkpoint_path": "/tmp/stage1",
        "skill_predictor_checkpoint_path": "/tmp/predictor",
        "fsq_path": "/tmp/FSQ.pt",
    }
    values.update(overrides)
    return SkillVLAStage2Config(**values)


def test_stage2_config_fixes_bayesvla_contract() -> None:
    config = _config()

    assert config.type == "skill_vla_stage2"
    assert config.architecture == "cond_gemma"
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
    with pytest.raises(ValueError, match="last.*layer_mix"):
        _config(likelihood_vlm_memory="every_layer")
    with pytest.raises(ValueError, match="gate_lr_scale"):
        _config(likelihood_gate_lr_scale=0.0)
    assert _config(likelihood_vlm_memory="layer_mix").likelihood_vlm_memory == "layer_mix"
    with pytest.raises(ValueError, match="gt.*predictor"):
        _config(training_skill_source="mixed")
    with pytest.raises(ValueError, match="skill_predictor_checkpoint_path"):
        _config(skill_predictor_checkpoint_path=None)
    with pytest.raises(ValueError, match="train_skill_predictor=True"):
        _config(train_skill_predictor=False)
    with pytest.raises(ValueError, match="without a terminator"):
        _config(train_terminator=True)
    with pytest.raises(ValueError, match="cond_gemma"):
        _config(
            architecture="vsa_perceiver_crossattn",
            architecture_revision="interleaved_direct1024_v3",
        )


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
    sampled = model.sample_actions(
        [],
        [],
        torch.zeros(1, 2),
        torch.tensor([0]),
        torch.zeros(1, 1, dtype=torch.long),
        torch.ones(1, 1, dtype=torch.bool),
        noise=noise,
        num_steps=1,
    )

    torch.testing.assert_close(sampled, torch.zeros_like(noise))
    assert captured["expert_skill"] is None
    assert captured["likelihood_shape"] == noise.shape
    allowed = captured["attention"][0, 0].eq(0)
    expected = torch.ones(3, 5, dtype=torch.bool)
    torch.testing.assert_close(allowed, expected)
