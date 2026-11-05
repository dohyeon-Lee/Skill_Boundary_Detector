"""Independent recomputation must preserve outputs, gradients and stream boundaries."""
import copy
import itertools
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import GemmaConfig

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05 import modeling_pi05 as module
from lerobot.policies.pi_gemma import PiGemmaForCausalLM


class TinyVisionBody(nn.Module):
    def get_image_features(self, image):
        return SimpleNamespace(pooler_output=self.multi_modal_projector(
            self.vision_tower(image.mean(dim=(-2, -1))))[:, None])


def tiny_core(monkeypatch, dtype="float32", **config_options):
    monkeypatch.setattr(module, "get_gemma_config", lambda variant: module.GemmaConfig(
        64 if variant == "gemma_2b" else 32, 2, 128, 8, 1, 8))

    def initialize(self, vlm, expert, use_adarms=None, precision="float32", **kwargs):
        nn.Module.__init__(self)
        self.freeze_vision_encoder = self.freeze_language_model = self.train_expert_only = False
        self.paligemma = nn.Module()
        self.paligemma.config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=2, hidden_size=64))
        self.paligemma.model = TinyVisionBody()
        for target, size, adaptive in [(self.paligemma.model, 64, False), (self, 32, True)]:
            config = GemmaConfig(hidden_size=size, intermediate_size=128, num_hidden_layers=2,
                num_attention_heads=8, num_key_value_heads=1, head_dim=8, vocab_size=32,
                use_adarms=adaptive, adarms_cond_dim=size if adaptive else None,
                hidden_activation="gelu_pytorch_tanh", attn_implementation="eager")
            if adaptive:
                self.gemma_expert = PiGemmaForCausalLM(config)
                self.gemma_expert.model.embed_tokens = None
            else:
                target.language_model = PiGemmaForCausalLM(config).model
        self.paligemma.model.vision_tower = nn.Linear(3, 64)
        self.paligemma.model.multi_modal_projector = nn.Linear(64, 64)
        self.to_bfloat16_for_selected_params(precision)

    monkeypatch.setattr(module.PaliGemmaWithExpertModel, "__init__", initialize)
    return module.PI05Pytorch(PI05Config(device="cpu", dtype=dtype, chunk_size=3, n_action_steps=3, **config_options))


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_frozen_expert_preserves_weights_while_action_loss_trains_vlm(monkeypatch, dtype, checkpoint):
    torch.manual_seed(42)
    model = tiny_core(monkeypatch, dtype, freeze_action_expert=True, gradient_checkpointing=checkpoint)
    model.train()
    frozen = [(parameter, parameter.detach().clone()) for part in model._action_expert_modules()
              for parameter in part.parameters()]
    assert frozen and all(not p.requires_grad for p, _ in frozen)
    assert all(not part.training for part in model._action_expert_modules())
    vlm = model.paligemma_with_expert.paligemma.model
    assert vlm.language_model.training and vlm.vision_tower.training
    trainable = [vlm.vision_tower.weight, vlm.multi_modal_projector.weight,
                 vlm.language_model.layers[0].self_attn.k_proj.weight]
    before = [p.detach().clone() for p in trainable]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    loss = model([torch.randn(2, 3, 4, 4)], [torch.ones(2, dtype=torch.bool)],
                 torch.tensor([[1, 2], [3, 4]]), torch.ones(2, 2, dtype=torch.bool),
                 torch.randn(2, 3, 32), noise=torch.randn(2, 3, 32), time=torch.tensor([.3, .7])).mean()
    loss.backward()
    assert all(p.grad is None for p, _ in frozen)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in trainable)
    optimizer.step()
    for parameter, original in frozen:
        torch.testing.assert_close(parameter, original, rtol=0, atol=0)
        assert parameter not in optimizer.state
    assert all(not torch.equal(p, old) for p, old in zip(trainable, before))
    model.eval().train()
    assert all(not part.training for part in model._action_expert_modules())


def test_expert_freeze_config_roundtrip_and_conflict(tmp_path):
    config = PI05Config(device="cpu", freeze_action_expert=True)
    config.save_pretrained(tmp_path)
    assert PreTrainedConfig.from_pretrained(tmp_path).freeze_action_expert
    path = tmp_path / "config.json"
    import json
    legacy = json.loads(path.read_text());legacy.pop("freeze_action_expert")
    path.write_text(json.dumps(legacy))
    assert not PreTrainedConfig.from_pretrained(tmp_path).freeze_action_expert
    with pytest.raises(ValueError, match="conflicts"):
        PI05Config(device="cpu", freeze_action_expert=True, train_expert_only=True)


@pytest.mark.parametrize("flags", itertools.product((False, True), repeat=3))
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_components_preserve_full_forward_backward_without_recomputing_other_streams(monkeypatch, flags, dtype):
    torch.manual_seed(24)
    baseline = tiny_core(monkeypatch, dtype)
    model = copy.deepcopy(baseline)
    expert, llm, vision = flags
    model.set_gradient_checkpointing(expert=expert, llm=llm, vision=vision)
    counts = Counter()
    for name, part in [("vision", model.paligemma_with_expert.paligemma.model.vision_tower),
                       *[("llm", l.self_attn.q_proj) for l in model.paligemma_with_expert.paligemma.model.language_model.layers],
                       *[("expert", l.self_attn.q_proj) for l in model.paligemma_with_expert.gemma_expert.model.layers]]:
        part.register_forward_pre_hook(lambda m, a, n=name: counts.update([n]))
    args = ([torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)],
            [torch.tensor([True, True]), torch.tensor([True, False])],
            torch.tensor([[1, 2, 3], [4, 5, 0]]), torch.tensor([[True, True, True], [True, True, False]]),
            torch.randn(2, 3, 32))
    noise, time = torch.randn(2, 3, 32), torch.tensor([.3, .7])
    expected = baseline(*args, noise=noise, time=time)
    actual = model(*args, noise=noise, time=time)
    tolerance = dict(rtol=.03, atol=.003) if dtype == "bfloat16" else dict(rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(actual, expected, **tolerance)
    expected.mean().backward()
    actual.mean().backward()
    for (name, parameter), (_, reference) in zip(model.named_parameters(), baseline.named_parameters()):
        assert (parameter.grad is None) == (reference.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, reference.grad, **tolerance, msg=name)
    for name, enabled in zip(("expert", "llm", "vision"), flags):
        assert (counts[name] > 2) if enabled else (counts[name] == 2), (name, counts)
    model.eval()
    with torch.no_grad():
        assert torch.isfinite(model(*args, noise=noise, time=time)).all()


def test_component_overrides_and_legacy_boolean_roundtrip(tmp_path):
    assert PI05Config(device="cpu", gradient_checkpointing=True).checkpointing_components == dict.fromkeys(
        ("expert", "llm", "vision"), True)
    cfg = PI05Config(device="cpu", gradient_checkpointing=True, gradient_checkpointing_llm=False,
                     gradient_checkpointing_vision=False)
    assert cfg.checkpointing_components == {"expert": True, "llm": False, "vision": False}
    cfg.save_pretrained(tmp_path)
    assert PreTrainedConfig.from_pretrained(tmp_path).checkpointing_components == cfg.checkpointing_components
    with pytest.raises(ValueError, match="gradient_checkpointing_llm"):
        PI05Config(device="cpu", gradient_checkpointing_llm="false")
