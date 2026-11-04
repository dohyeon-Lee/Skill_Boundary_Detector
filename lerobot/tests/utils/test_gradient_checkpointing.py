import copy

import pytest
import torch

from lerobot.utils.gradient_checkpointing import (
    checkpoint_indices, configure_vision_checkpointing, parse_checkpointing,
)


@pytest.mark.parametrize("value,expected", [(False, set()), (0, set()), (True, set(range(18))),
                                          (18, set(range(18))), (2, {16, 17}), ([1, 18], {0, 17})])
def test_layer_selection_includes_final_expert_layer(value, expected):
    assert checkpoint_indices(value, 18) == expected


@pytest.mark.parametrize("value", [-1, [0], [1, 1], [True], 1.5, "all", [19], 19])
def test_invalid_selection_is_not_silently_clamped(value):
    with pytest.raises(ValueError):
        checkpoint_indices(value, 18)


def test_legacy_bool_and_explicit_mapping_are_distinct():
    global_flag, streams = parse_checkpointing(True)
    assert global_flag and all(value is None for value in streams.values())
    global_flag, streams = parse_checkpointing({"action_expert": 18, "skill_planner": True})
    assert not global_flag and streams == {"action_expert": 18, "skill_planner": True,
                                          "vlm": False, "vision": False}
    with pytest.raises(ValueError):
        parse_checkpointing({"expertt": True})
    with pytest.raises(ValueError):
        checkpoint_indices(18, 17)


@pytest.mark.parametrize("kind", ["siglip", "dino"])
def test_real_vision_blocks_preserve_outputs_gradients_and_checkpoint_keys(monkeypatch, kind):
    if kind == "siglip":
        from transformers import SiglipVisionConfig, SiglipVisionModel
        config = SiglipVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                                    num_attention_heads=2, image_size=8, patch_size=4)
        model = SiglipVisionModel(config).train()
        blocks = model.vision_model.encoder.layers
    else:
        from transformers import DINOv3ViTConfig, DINOv3ViTModel
        config = DINOv3ViTConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                               num_attention_heads=2, image_size=8, patch_size=4, num_register_tokens=0)
        model = DINOv3ViTModel(config).train()
        blocks = model.layer
    reference = copy.deepcopy(model)
    images = torch.randn(1, 3, 8, 8)
    torch.manual_seed(123)
    expected = reference(pixel_values=images).last_hidden_state
    expected.square().mean().backward()
    keys = set(model.state_dict())
    model.gradient_checkpointing_enable()
    configure_vision_checkpointing(model, [1, 3])
    assert not model.is_gradient_checkpointing
    calls=[]
    original=torch.utils.checkpoint.checkpoint
    def checkpoint(function, *args, **kwargs):
        calls.append(function.__self__)
        return original(function, *args, **kwargs)
    monkeypatch.setattr(torch.utils.checkpoint, 'checkpoint', checkpoint)
    torch.manual_seed(123)
    actual=model(pixel_values=images).last_hidden_state
    actual.square().mean().backward()
    assert calls == [blocks[0], blocks[2]]
    torch.testing.assert_close(actual, expected)
    assert set(model.state_dict()) == keys
    for name, parameter in model.named_parameters():
        other=dict(reference.named_parameters())[name]
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, atol=1e-5, rtol=1e-4)
    model.load_state_dict(reference.state_dict(), strict=True)
    configure_vision_checkpointing(model, False)
    calls.clear()
    model(pixel_values=images)
    assert not calls
