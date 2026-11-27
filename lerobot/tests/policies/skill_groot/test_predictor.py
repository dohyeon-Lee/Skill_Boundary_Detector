"""Mode2 reader/UV training uses the GR00T VLM and survives save/warm start."""
from types import SimpleNamespace

import torch
from torch import nn

from test_model import TinyVLM
from lerobot.policies.skill_groot import predictor as implementation
from lerobot.policies.skill_aux.configuration_skill_aux import SkillAuxConfig
from lerobot.policies.skill_aux.modeling_skill_aux import SkillAuxPolicy


class PredictorVLM(TinyVLM):
    def __init__(self):
        super().__init__(nn.Identity())
        self.language_model.config = SimpleNamespace(hidden_size=16)
        self.language_model.norm = nn.LayerNorm(16)

    @property
    def vision_tower(self):
        return self.image_input

    @property
    def multi_modal_projector(self):
        return self.image_input


def test_predictor_mode2_skill_and_uv_backpropagate_and_reload(monkeypatch, tmp_path):
    monkeypatch.setattr(implementation, 'predictor_vlm', lambda cfg: PredictorVLM())
    monkeypatch.setattr(implementation.AutoTokenizer, 'from_pretrained', lambda *a, **k:
        SimpleNamespace(decode=lambda *a, **k: 'Task: put the cup in the cabinet'))
    config = SkillAuxConfig(device='cpu', dtype='float32', skillvla_backbone='groot_integrate',
        train_skill_predictor=True, train_terminator=False, skill_predictor_mode='mode2',
        skill_predictor_all_layers=True, skill_predictor_freeze_vlm=False, skill_predictor_lora=False,
        skill_predictor_focus_uv_enabled=True, skill_predictor_reader_heads=2,
        skill_fsq_levels=[3, 3, 3], skill_vocab_size=27)
    policy = SkillAuxPolicy(config)
    predictor = policy.model.skill_predictor
    assert isinstance(predictor, implementation.GrootSkillPredictor)
    images = [torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32)]
    tokens, mask = torch.ones(2, 5, dtype=torch.long), torch.ones(2, 5, dtype=torch.bool)
    loss, _ = predictor.loss_with_focus_uv(images, tokens, mask, torch.tensor([1, 4]),
        torch.tensor([[.1, -.2], [-.3, .4]]), torch.ones(2, dtype=torch.bool),
        previous_skill_code=torch.tensor([0, 1]))
    loss.backward()
    for parameter in (predictor.vlm.image_input.weight, predictor.previous_skill_projection[0].weight,
                      predictor.focus_uv_head[-2].weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    parameters = [p for group in policy.get_optim_params() for p in group['params']]
    assert len(parameters) == len({id(p) for p in parameters})
    policy.save_pretrained(tmp_path)
    restored = SkillAuxPolicy.from_pretrained(tmp_path, strict=True)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
