from __future__ import annotations

import copy

import torch
from torch import nn
from transformers import GemmaConfig

from lerobot.policies.pi_gemma import PiGemmaModel
from lerobot.policies.skill_expert.modeling_skill_predictor import (
    FrozenVLMSkillPredictor,
)


def _tiny_pi_gemma() -> PiGemmaModel:
    config = GemmaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        hidden_activation="gelu_pytorch_tanh",
        attention_dropout=0.0,
        use_cache=False,
    )
    config.use_adarms = False
    config.adarms_cond_dim = None
    return PiGemmaModel(config)


class _PredictorVLM(nn.Module):
    def __init__(self, language_model: PiGemmaModel):
        super().__init__()
        self.language_model = language_model


def _all_hidden_loss(model: PiGemmaModel, inputs: torch.Tensor) -> torch.Tensor:
    output = model(
        inputs_embeds=inputs,
        use_cache=False,
        output_hidden_states=True,
    )
    return torch.stack([hidden.square().mean() for hidden in output.hidden_states]).sum()


def test_predictor_checkpointing_reaches_every_pi_gemma_layer_and_preserves_gradients() -> None:
    torch.manual_seed(7)
    reference = _tiny_pi_gemma().train()
    checkpointed = copy.deepcopy(reference).train()

    predictor = FrozenVLMSkillPredictor.__new__(FrozenVLMSkillPredictor)
    nn.Module.__init__(predictor)
    predictor.vlm = _PredictorVLM(checkpointed)
    predictor.gradient_checkpointing_enable()

    assert checkpointed.gradient_checkpointing is True
    assert all(layer.gradient_checkpointing for layer in checkpointed.layers)

    checkpoint_calls = 0
    for layer in checkpointed.layers:
        checkpoint_function = layer._gradient_checkpointing_func

        def recording_checkpoint(function, *args, _checkpoint=checkpoint_function):
            nonlocal checkpoint_calls
            checkpoint_calls += 1
            return _checkpoint(function, *args)

        layer._gradient_checkpointing_func = recording_checkpoint

    reference_input = torch.randn(2, 6, 16, requires_grad=True)
    checkpointed_input = reference_input.detach().clone().requires_grad_(True)
    reference_loss = _all_hidden_loss(reference, reference_input)
    checkpointed_loss = _all_hidden_loss(checkpointed, checkpointed_input)

    torch.testing.assert_close(checkpointed_loss, reference_loss)
    reference_loss.backward()
    checkpointed_loss.backward()

    assert checkpoint_calls == len(checkpointed.layers)
    torch.testing.assert_close(checkpointed_input.grad, reference_input.grad)
    for reference_parameter, checkpointed_parameter in zip(
        reference.parameters(), checkpointed.parameters(), strict=True
    ):
        if reference_parameter.grad is None:
            assert checkpointed_parameter.grad is None
        else:
            torch.testing.assert_close(
                checkpointed_parameter.grad,
                reference_parameter.grad,
            )
