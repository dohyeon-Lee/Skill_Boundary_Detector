"""Vision-only FT must optimize the image stack THROUGH the frozen language model."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.skill_aux.configuration_skill_aux import SkillAuxConfig
from lerobot.policies.skill_aux.modeling_skill_aux import SkillAuxPolicy
from lerobot.policies.skill_expert import modeling_skill_predictor as model


class TinyVision(nn.Linear):
    def __init__(self):
        super().__init__(3, 16)
        self.checkpoint_enabled = False

    def gradient_checkpointing_enable(self, **kwargs):
        assert kwargs["gradient_checkpointing_kwargs"]["use_reentrant"] is False
        self.checkpoint_enabled = True


class TinyLanguage(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16, num_hidden_layers=1)
        self.embed_tokens = nn.Embedding(32, 16)
        self.q_proj = nn.Linear(16, 16)
        self.checkpoint_enabled = False

    def gradient_checkpointing_enable(self, **kwargs):
        assert kwargs["gradient_checkpointing_kwargs"]["use_reentrant"] is False
        self.checkpoint_enabled = True

    def forward(self, inputs_embeds, **kwargs):
        # Mix image/language tokens so every reader can reach the image inputs.
        hidden = inputs_embeds + inputs_embeds.mean(1, keepdim=True)
        if self.training and self.checkpoint_enabled:
            hidden = checkpoint(self.q_proj, hidden, use_reentrant=False)
        else:
            hidden = self.q_proj(hidden)
        return SimpleNamespace(last_hidden_state=hidden.tanh())


class TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = TinyVision()
        self.multi_modal_projector = nn.Linear(16, 16)
        self.language_model = TinyLanguage()

    def get_image_features(self, image):
        hidden = self.vision_tower(image.mean((-2, -1)))
        hidden = self.multi_modal_projector(hidden)
        return SimpleNamespace(pooler_output=hidden[:, None].expand(-1, 2, -1))


def _config(*, vision=False, full=False, lora=False, checkpointing=False):
    return SkillAuxConfig(
        device="cpu", dtype="float32", train_skill_predictor=True, train_terminator=False,
        skill_predictor_freeze_vlm=not full,
        skill_predictor_train_vision_encoder=vision,
        skill_predictor_detach_vlm=not (vision or full or lora),
        skill_predictor_lora=lora, skill_predictor_lora_targets="q", skill_predictor_lora_rank=2,
        skill_predictor_all_layers=False, skill_predictor_reader_depth=1,
        skill_predictor_reader_heads=2, skill_predictor_image_size=8,
        gradient_checkpointing=checkpointing,
    )


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("lora", [False, True])
def test_vision_and_projector_receive_gradients_through_frozen_language_model(
    monkeypatch, checkpointing, lora,
):
    torch.manual_seed(7)
    monkeypatch.setattr(model, "build_paligemma_model", lambda *a, **kw: TinyVLM())
    policy = SkillAuxPolicy(_config(vision=True, lora=lora, checkpointing=checkpointing)).train()
    p = policy.model.skill_predictor
    assert p.vlm.vision_tower.training
    assert p.vlm.multi_modal_projector.training
    assert p.vlm.language_model.training == (lora or checkpointing)
    groups = policy.get_optim_params()
    all_params = [parameter for group in groups for parameter in group["params"]]
    assert len(all_params) == len({id(parameter) for parameter in all_params})
    vlm_group = next(group for group in groups if group["group_name"] == "skill_predictor_vlm")
    assert {id(p) for p in vlm_group["params"]} == {
        id(p) for p in [*policy.model.skill_predictor.vlm.vision_tower.parameters(),
                        *policy.model.skill_predictor.vlm.multi_modal_projector.parameters()]
    }
    before = {name: param.detach().clone() for name, param in p.named_parameters()}
    objective, _ = p.loss(
        [torch.rand(2, 3, 8, 8)], torch.ones(2, 2, dtype=torch.long),
        torch.ones(2, 2, dtype=torch.bool), torch.tensor([0, 1]),
    )
    objective.backward()
    for name, param in p.vlm.named_parameters():
        trainable = name.startswith(("vision_tower.", "multi_modal_projector.")) or ".adapters." in name
        assert param.requires_grad == trainable, name
        if trainable:
            assert param.grad is not None, name
        else:
            assert param.grad is None, name
    assert p.vlm.vision_tower.weight.grad.abs().sum() > 0
    assert p.vlm.multi_modal_projector.weight.grad.abs().sum() > 0
    for group in groups:
        group["lr"] = .1  # use a visible test step, retaining the verified group membership
    optimizer = torch.optim.SGD(groups, lr=.1)
    optimizer.step()
    for name, param in p.named_parameters():
        if name.startswith("vlm.language_model.") and ".adapters." not in name:
            torch.testing.assert_close(param, before[name], rtol=0, atol=0)
    assert not torch.equal(p.vlm.vision_tower.weight, before["vlm.vision_tower.weight"])
    assert not torch.equal(p.vlm.multi_modal_projector.weight, before["vlm.multi_modal_projector.weight"])
    policy.eval()
    assert not p.vlm.vision_tower.training and not p.vlm.multi_modal_projector.training
    assert not p.vlm.language_model.training


@pytest.mark.parametrize("full", [False, True])
def test_legacy_all_and_none_still_freeze_the_expected_parameters(monkeypatch, full):
    monkeypatch.setattr(model, "build_paligemma_model", lambda *a, **kw: TinyVLM())
    policy = SkillAuxPolicy(_config(full=full))
    p = policy.model.skill_predictor
    assert all(param.requires_grad == full for param in p.vlm.parameters())
    assert bool(p.vlm_parameters()) == full


def test_vision_training_choice_roundtrips_and_old_configs_default_off(tmp_path):
    config = _config(vision=True)
    config.save_pretrained(tmp_path)
    loaded = PreTrainedConfig.from_pretrained(tmp_path)
    assert loaded.skill_predictor_train_vision_encoder is True
    assert loaded.skill_predictor_detach_vlm is False
    assert _config().skill_predictor_train_vision_encoder is False


@pytest.mark.parametrize("flags", [{"vision": True, "full": True}])
def test_invalid_partial_and_full_training_combination_is_rejected(flags):
    with pytest.raises(ValueError, match="Vision-only"):
        _config(**flags)


def test_external_predictor_loads_changed_vision_weights_not_just_heads(monkeypatch, tmp_path):
    from lerobot.policies.skill_expert import modeling_skill_expert as expert

    monkeypatch.setattr(model, "build_paligemma_model", lambda *a, **kw: TinyVLM())
    policy = SkillAuxPolicy(_config(vision=True))
    policy.config.save_pretrained(tmp_path)
    chosen = []
    monkeypatch.setattr(expert, "_load_complete_predictor_parameters", lambda *a: chosen.append("complete") or 1)
    monkeypatch.setattr(expert, "_load_learned_predictor_parameters", lambda *a: chosen.append("heads_only") or 1)
    expert.SkillExpertPolicy._initialize_frozen_skill_predictor(policy, tmp_path)
    assert chosen == ["complete"]
