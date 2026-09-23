"""Warm-starting a predictor that gains LoRA adapters it was not trained with."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from lerobot.policies.skill_aux.modeling_skill_aux import SkillAuxPolicy
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


def _warm_start_stub(tmp_path: Path, *, checkpoint_fields: dict, config_fields: dict):
    """The FT warm start with everything heavy stubbed out."""
    from lerobot.policies.skill_expert.modeling_skill_expert import (
        _PREDICTOR_CHECKPOINT_CONTRACT_FIELDS,
    )

    baseline = {
        "skill_vocab_size": 27,
        "skill_fsq_levels": [3, 3, 3],
        "skill_predictor_vlm_variant": "gemma_2b",
        "skill_predictor_image_size": 224,
        "skill_predictor_reader_tokens": 4,
        "skill_predictor_reader_depth": 2,
        "skill_predictor_reader_heads": 8,
        "skill_predictor_all_layers": True,
        "skill_predictor_freeze_vlm": False,
        "skill_predictor_detach_vlm": False,
        "skill_predictor_lora": False,
        "skill_predictor_lora_targets": "q,k,v,o",
        "skill_predictor_lora_rank": 8,
        "skill_predictor_lora_alpha": 16.0,
        "skill_predictor_lora_dropout": 0.0,
        "skill_predictor_deadzone_frac": 0.8,
        "skill_predictor_attend_image": True,
        "skill_predictor_attend_language": True,
        "skill_predictor_focus_uv_enabled": False,
        "skill_predictor_end_state_mode": "xyz",
        "skill_predictor_end_state_dim": 8,
        "tokenizer_max_length": 200,
    }
    assert set(baseline) == set(_PREDICTOR_CHECKPOINT_CONTRACT_FIELDS)

    weight = torch.arange(6.0).reshape(2, 3)
    path = _checkpoint(tmp_path, {"head.0.weight": weight})
    source = {
        "type": "skill_aux",
        "train_skill_predictor": True,
        "skill_code_space_id": "FSQ333_test",
        **baseline,
        **checkpoint_fields,
    }
    (path / "config.json").write_text(json.dumps(source))

    state = {"head.0.weight": torch.zeros(2, 3)}
    policy = SimpleNamespace(
        model=SimpleNamespace(skill_predictor=_predictor(state)),
        config=SimpleNamespace(
            skill_code_space_id="FSQ333_test", **{**baseline, **config_fields}
        ),
    )
    SkillAuxPolicy._load_complete_predictor_warm_start(policy, path)
    return state, weight


def test_ft_may_re_adapt_the_vlm_but_not_reshape_the_predictor(tmp_path: Path) -> None:
    """freeze/LoRA is the FT run's choice; anything that fixes a tensor shape is not."""
    # A fullvlm checkpoint warm-starting a frozen-VLM run: allowed, and the weights still load.
    state, weight = _warm_start_stub(
        tmp_path / "readapt",
        checkpoint_fields={},
        config_fields={"skill_predictor_freeze_vlm": True, "skill_predictor_detach_vlm": True},
    )
    torch.testing.assert_close(state["head.0.weight"], weight)

    # The reader's shape is a different matter: the inherited weights would stop fitting.
    with pytest.raises(ValueError, match="skill_predictor_reader_tokens"):
        _warm_start_stub(
            tmp_path / "reshape",
            checkpoint_fields={},
            config_fields={"skill_predictor_reader_tokens": 8},
        )
