"""Warm-starting a predictor that gains LoRA adapters it was not trained with."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from lerobot.policies.skill_expert.modeling_skill_expert import (
    _load_complete_predictor_parameters,
)

PREFIX = "model.skill_predictor."


def _checkpoint(tmp_path: Path, tensors: dict[str, torch.Tensor]) -> Path:
    path = tmp_path / "pretrained_model"
    path.mkdir(parents=True)
    save_file({PREFIX + key: value for key, value in tensors.items()}, path / "model.safetensors")
    return path


def _predictor(state: dict[str, torch.Tensor]):
    return SimpleNamespace(state_dict=lambda: state)


def test_a_plain_checkpoint_fills_the_lora_wrapped_projection(tmp_path: Path) -> None:
    """Without routing, a wrapped projection would read as missing and stay at random init."""
    weight = torch.arange(6.0).reshape(2, 3)
    source = _checkpoint(
        tmp_path, {"vlm.q_proj.weight": weight, "head.0.weight": weight.clone()}
    )
    # The model gained LoRA, so its pretrained projection now lives under ``.base.``.
    state = {
        "vlm.q_proj.base.weight": torch.zeros(2, 3),
        "vlm.q_proj.adapters.skill.lora_A.weight": torch.zeros(4, 3),
        "vlm.q_proj.adapters.skill.lora_B.weight": torch.zeros(2, 4),
        "head.0.weight": torch.zeros(2, 3),
    }

    loaded = _load_complete_predictor_parameters(
        _predictor(state), source, allowed_missing_substrings=(".adapters.",)
    )

    assert loaded == 2
    torch.testing.assert_close(state["vlm.q_proj.base.weight"], weight)   # routed, not left random
    torch.testing.assert_close(state["head.0.weight"], weight)
    # The adapter stays at its own init: zero on the B side, so it starts as the identity.
    assert torch.count_nonzero(state["vlm.q_proj.adapters.skill.lora_B.weight"]) == 0


def test_missing_adapters_are_refused_unless_they_are_allowed(tmp_path: Path) -> None:
    weight = torch.arange(6.0).reshape(2, 3)
    source = _checkpoint(tmp_path, {"vlm.q_proj.weight": weight})
    state = {
        "vlm.q_proj.base.weight": torch.zeros(2, 3),
        "vlm.q_proj.adapters.skill.lora_A.weight": torch.zeros(4, 3),
    }

    with pytest.raises(RuntimeError, match="missing="):
        _load_complete_predictor_parameters(_predictor(state), source)


def test_a_matching_checkpoint_still_loads_untouched(tmp_path: Path) -> None:
    """Routing is a no-op when the names already agree, so ordinary warm starts are unaffected."""
    tensors = {"reader.0.weight": torch.ones(2, 2), "head.0.bias": torch.full((2,), 3.0)}
    source = _checkpoint(tmp_path, tensors)
    state = {key: torch.zeros_like(value) for key, value in tensors.items()}

    assert _load_complete_predictor_parameters(_predictor(state), source) == 2
    for key, value in tensors.items():
        torch.testing.assert_close(state[key], value)
