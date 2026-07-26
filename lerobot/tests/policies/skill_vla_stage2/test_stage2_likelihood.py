from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding

from lerobot.policies.skill_vla_stage2.configuration_skill_vla_stage2 import (
    SkillVLAStage2Config,
)
from lerobot.policies.skill_vla_stage2.modeling_skill_vla_stage2 import (
    LikelihoodBlock,
    SkillVLAStage2Policy,
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
