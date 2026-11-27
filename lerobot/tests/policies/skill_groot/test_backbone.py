"""Exercise the real native Qwen3-VL forward and checkpointing on a small CPU model."""
import pytest
import torch
from torch import nn
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.feature_extraction_utils import BatchFeature

from lerobot.policies.skill_groot.backbone import GrootVLM
from lerobot.policies.skill_groot.native import native_modules


class ProcessorFixture:
    image_processor = type("ImageConfig", (), {"merge_size": 2})()

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"][-1]["text"]

    def __call__(self, *, text, images, **kwargs):
        # Two 2x2 patch cameras, one merged token each, then condition text.
        return BatchFeature({
            "input_ids": torch.tensor([[5, 7, 6, 5, 7, 6, int(t)] for t in text]),
            "attention_mask": torch.ones(len(text), 7, dtype=torch.long),
            "pixel_values": torch.stack([x[:, :32, :32].flatten()[:1536].repeat(4, 1) for x in images]).flatten(0, 1),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images)),
        })


def backbone():
    native, _ = native_modules()
    config = Qwen3VLConfig(
        image_token_id=7, video_token_id=8, vision_start_token_id=5, vision_end_token_id=6,
        text_config=dict(vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            head_dim=16, rope_parameters={"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True}),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2,
            patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
            out_hidden_size=32, num_position_embeddings=16, deepstack_visual_indexes=[0]),
    )
    base = native.Qwen3Backbone.__new__(native.Qwen3Backbone)
    nn.Module.__init__(base)
    base.model = Qwen3VLForConditionalGeneration(config)
    base.set_trainable_parameters(tune_llm=True, tune_visual=True)
    vlm = GrootVLM.__new__(GrootVLM)
    nn.Module.__init__(vlm)
    vlm.backbone, vlm.processor = base, ProcessorFixture()
    return vlm


@pytest.mark.parametrize("checkpoint", [False, True])
def test_native_causal_layer_taps_and_camera_masks(checkpoint):
    torch.manual_seed(3)
    vlm = backbone()
    vlm.configure_checkpointing(llm=checkpoint, vision=checkpoint)
    images = [torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)]
    layers, valid, image, cameras = vlm.encode(images, ["10"])
    assert len(layers) == 2
    assert image.sum() == 2 and all(mask.sum() == 1 for mask in cameras)
    assert torch.equal(cameras[0] | cameras[1], image)
    changed, *_ = vlm.encode(images, ["11"])
    # The native image-before-condition causal mask must survive the adapter.
    torch.testing.assert_close(layers[-1][:, :6], changed[-1][:, :6], rtol=0, atol=0)
    assert not torch.equal(layers[-1][:, -1], changed[-1][:, -1])
    loss = sum(value[:, -1].square().mean() for value in layers)
    loss.backward()
    assert vlm.vision_tower.patch_embed.proj.weight.grad.abs().sum() > 0
    assert vlm.language_model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0


def test_selective_checkpointing_uses_actual_native_layer_numbers():
    vlm = backbone()
    vlm.configure_checkpointing(llm=[2], vision=1)
    assert [x.gradient_checkpointing for x in vlm.language_model.layers] == [False, True]
    assert [x.gradient_checkpointing for x in vlm.vision_tower.blocks] == [False, True]
