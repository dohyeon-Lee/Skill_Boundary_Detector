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
        "train_skill_predictor": True,
        "train_terminator": True,
        "fsq_path": "/tmp/FSQ.pt",
    }
    values.update(overrides)
    return SkillVLAStage2Config(**values)


def test_stage2_config_fixes_bayesvla_contract() -> None:
    config = _config()

    assert config.type == "skill_vla_stage2"
    assert config.likelihood_num_layers == 4
    assert config.training_skill_source == "gt"
    assert not config.finetune_skill_predictor
    assert not config.finetune_terminator
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
    with pytest.raises(ValueError, match="gt.*predictor"):
        _config(training_skill_source="mixed")
    with pytest.raises(ValueError, match="VLM/predictor"):
        _config(train_skill_predictor=False)


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
    holder.fsq_term_train = None
    policy.model = holder
    policy.config = SimpleNamespace(finetune_terminator=False)

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

    assert optimized == expected
    assert not optimized & {id(parameter) for parameter in holder.prior.parameters()}


def test_optional_terminator_has_separate_lr_and_clipping_group() -> None:
    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    holder = nn.Module()
    holder.vlm_to_expert_projection = nn.Linear(2, 2)
    holder.likelihood_blocks = nn.ModuleList([nn.Linear(2, 2)])
    holder.action_out_proj = nn.Linear(2, 1)
    holder.fsq_term_train = nn.Linear(2, 2)
    policy.model = holder
    policy.config = SimpleNamespace(
        finetune_terminator=True,
        optimizer_lr=2e-4,
        terminator_lr_scale=0.5,
    )

    groups = policy.get_optim_params()
    terminator_parameters = list(holder.fsq_term_train.parameters())

    assert len(groups) == 2
    assert groups[1]["lr"] == pytest.approx(1e-4)
    assert {id(parameter) for parameter in groups[1]["params"]} == {
        id(parameter) for parameter in terminator_parameters
    }
    assert policy.isolated_main_optimizer_grad_groups() == {
        "terminator": terminator_parameters
    }


def test_stage2_forward_preserves_inherited_conditioning_route_in_frozen_prior() -> None:
    class _Predictor:
        @staticmethod
        def encode_base_last_hidden(images, language_tokens, language_mask):
            del images, language_tokens, language_mask
            return torch.zeros(1, 2, 2), torch.zeros(1, 2, dtype=torch.bool)

    model = SkillVLAStage2Pytorch.__new__(SkillVLAStage2Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(action_loss_mode="flow_endpoint_xyz")
    condition_state = torch.tensor([[9.0, 8.0]])
    condition_skill = torch.tensor([[7.0, 6.0]])
    expert_time = torch.tensor([[5.0, 4.0]])
    captured = {}
    model.skill_predictor = _Predictor()
    model._condition_tokens = lambda images, batch_size=None: torch.zeros(1, 2, 2)
    model._state_condition = lambda state: condition_state
    model._expert_condition = lambda time, state=None, skill=None: expert_time
    model._skill_broadcasts = lambda code: (condition_skill, None)

    def run_prior(
        condition,
        actions,
        routed_state,
        expert_condition,
        routed_condition_skill,
        expert_skill,
    ):
        del condition
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
        [],
        torch.zeros(1, 2),
        torch.tensor([0]),
        actions,
        torch.zeros(1, 1, dtype=torch.long),
        torch.ones(1, 1, dtype=torch.bool),
        noise=torch.zeros_like(actions),
        time=torch.full((1,), 0.5),
    )

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

    def forward(
        self,
        images,
        state,
        skill_code,
        actions,
        language_tokens,
        language_mask,
    ):
        del images, state, skill_code, language_tokens, language_mask
        self.seen_actions = actions.detach().clone()
        self._last_predicted_actions = actions.clone()
        self._last_predicted_actions[:, :2, 0] += 1.0
        return torch.tensor(
            [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [100.0, 100.0, 100.0]]],
            device=actions.device,
        )


def test_stage2_can_mix_flow_and_endpoint_xyz_without_hold_targets() -> None:
    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        max_action_dim=3,
        max_state_dim=2,
        output_features={ACTION: SimpleNamespace(shape=(3,))},
        finetune_terminator=False,
        action_loss_mode="flow_endpoint_xyz",
        training_skill_source="gt",
    )
    policy.model = _CaptureStage2Residual()
    policy._collect_images = lambda batch: []
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

    torch.testing.assert_close(policy.model.seen_actions, actions)
    assert metrics["action_loss"] == pytest.approx(2.5)
    assert metrics["endpoint_xyz_loss"] == pytest.approx(4.0 / 3.0)
    assert metrics["action_flow_weight"] == 0.5
    assert metrics["action_endpoint_weight"] == 0.5
    torch.testing.assert_close(loss, torch.tensor(23.0 / 12.0))


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


def test_stage2_terminator_uses_only_random_and_failed_pair_slots() -> None:
    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(same_skill_batch_enabled=True)
    batch = {
        "same_skill_pair_id": torch.tensor([0, 0, -1, -1]),
        "skill_code_true": torch.tensor([1, 1, 2, 3]),
        "skill_ds": torch.tensor([0, 1, 2, 3]),
        "skill_de": torch.tensor([3, 2, 1, 0]),
        "skill_decoder_state": torch.arange(8).view(4, 2),
        "observation.images.image": torch.zeros(4, 3, 2, 2),
        "observation.images.wrist_image": torch.zeros(4, 3, 2, 2),
    }

    selected, fraction = policy._terminator_random_batch(batch)

    assert fraction == pytest.approx(0.5)
    assert selected["skill_code_true"].tolist() == [2, 3]
    assert selected["skill_ds"].tolist() == [2, 3]


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
    model.cond_encoder = SimpleNamespace(model=_ConditionModel())
    model.skill_predictor = _Predictor()
    model._condition_tokens = lambda images, batch_size=None: torch.zeros(1, 2, 2)
    model._state_condition = lambda state: torch.zeros(1, 2)
    model._expert_condition = lambda time, state=None, skill=None: torch.zeros(1, 2)
    model._skill_broadcasts = lambda code: (torch.tensor([[7.0, 8.0]]), None)
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
