from types import MethodType, SimpleNamespace

import torch
from transformers import GemmaConfig

from lerobot.policies.pi_gemma import PiGemmaModel
from lerobot.policies.skillVLA_stage0_pretrain.modeling_skillVLA_stage0_pretrain import (
    SkillVLAStage0PretrainPytorch,
)


def test_motor_vlm_mask_keeps_predicted_target_sequence_causal() -> None:
    model = SimpleNamespace(
        _vlm_is_causal=torch.tensor([False, False, True, True, True])
    )
    pad = torch.tensor([[True, True, True, True, False]])

    allow = SkillVLAStage0PretrainPytorch._motor_vlm_self_mask(model, pad)

    expected = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
            [True, True, True, True, False],
            [False, False, False, False, False],
        ]
    )
    torch.testing.assert_close(allow[0], expected)


def test_fast_context_choice_is_constrained_to_fast_or_stop() -> None:
    class _Head:
        def __call__(self, hidden):
            logits = torch.full((hidden.shape[0], hidden.shape[1], 8), -10.0)
            logits[..., 2] = 100.0  # Unrelated vocabulary winner must never be selected.
            logits[..., 4] = 3.0
            logits[..., 5] = 2.0
            logits[..., 7] = 4.0
            return logits

    model = SimpleNamespace(
        paligemma_with_expert=SimpleNamespace(
            paligemma=SimpleNamespace(lm_head=_Head())
        ),
        _fast_token_ids=torch.tensor([4, 5]),
        _action_stop_id=torch.tensor(7),
        _token_to_slot=torch.arange(8),
    )
    model._output_delta_logits = lambda hidden: None
    model._selected_scores = lambda base, delta, ids: base.index_select(-1, ids)
    hidden = torch.zeros(1, 3)

    first = SkillVLAStage0PretrainPytorch._next_fast_context_token(
        model, hidden, allow_stop=False
    )
    later = SkillVLAStage0PretrainPytorch._next_fast_context_token(
        model, hidden, allow_stop=True
    )

    assert first.item() == 4
    assert later.item() == 7


def test_predicted_fast_context_uses_kv_cache_without_gt_targets() -> None:
    config = GemmaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
    )
    vlm = PiGemmaModel(config)
    calls = []

    def choose(self, hidden, *, allow_stop):
        calls.append(allow_stop)
        token = self._action_stop_id if allow_stop else self._fast_token_ids[0]
        return token.expand(hidden.shape[0])

    model = SimpleNamespace(
        _vlm=vlm,
        _paligemma_tokenizer=SimpleNamespace(bos_token_id=1),
        _action_prefix_ids=torch.tensor([2, 3]),
        _action_stop_id=torch.tensor(12),
        _fast_token_ids=torch.tensor([10, 11]),
        config=SimpleNamespace(max_action_tokens=4),
    )
    model._add_input_delta = lambda embeddings, token_ids: embeddings
    model._next_fast_context_token = MethodType(choose, model)

    tokens, masks, terminated = SkillVLAStage0PretrainPytorch._predict_fast_context(
        model,
        torch.randn(2, 3, 8),
        torch.ones(2, 3, dtype=torch.bool),
        torch.tensor([[4], [5]]),
    )

    torch.testing.assert_close(tokens, torch.full((2, 1), 10))
    assert masks.all()
    assert terminated.all()
    assert calls == [False, True]
