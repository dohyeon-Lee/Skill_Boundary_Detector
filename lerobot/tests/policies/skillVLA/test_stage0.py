from types import SimpleNamespace

import torch
from torch import nn

from lerobot.policies.pi05.lora import NamedLoRALinear
from lerobot.policies.skillVLA.modeling_skillVLA import (
    LanguageKVBridge,
    SkillVLAPytorch,
    SkillVLAPolicy,
    _skill_endpoint_weights,
)


def test_language_bridge_is_zero_initialized_with_kv_head_shape() -> None:
    bridge = LanguageKVBridge(hidden_dim=16, rank=4, key_dim=8, value_dim=8, n_kv_heads=2)

    delta_k, delta_v = bridge(torch.randn(3, 5, 16))

    assert delta_k.shape == (3, 2, 5, 4)
    assert delta_v.shape == (3, 2, 5, 4)
    assert torch.count_nonzero(delta_k) == 0
    assert torch.count_nonzero(delta_v) == 0


def test_wrong_language_prefers_same_skill_then_falls_back() -> None:
    tokens = torch.tensor([[1, 2, 0], [3, 4, 0], [5, 6, 0], [7, 8, 0]])
    masks = torch.tensor([[1, 1, 0]] * 4, dtype=torch.bool)
    skills = torch.tensor([2, 2, 3, 4])
    tasks = torch.tensor([10, 11, 12, 13])

    wrong_tokens, wrong_masks, eligible, same_skill = SkillVLAPolicy._wrong_language_batch(
        tokens, masks, skills, tasks)

    assert eligible.tolist() == [True, True, True, True]
    assert same_skill.tolist() == [True, True, False, False]
    torch.testing.assert_close(wrong_masks, masks)
    assert not torch.equal(wrong_tokens, tokens)


def test_wrong_language_skips_batch_without_a_distinct_prompt() -> None:
    tokens = torch.tensor([[1, 2], [1, 2]])
    masks = torch.ones_like(tokens, dtype=torch.bool)

    result = SkillVLAPolicy._wrong_language_batch(
        tokens, masks, torch.tensor([0, 0]), torch.tensor([4, 4]))

    assert result == (None, None, None, None)


def test_stage0_endpoint_weights_support_both_directions_and_uniform() -> None:
    valid = torch.ones(2, 1, dtype=torch.bool)
    batch = {
        "skill_ds": torch.tensor([0, 9]),
        "skill_de": torch.tensor([9, 0]),
    }

    increasing = _skill_endpoint_weights(valid, batch, start_weight=1.0, end_weight=3.0)
    decreasing = _skill_endpoint_weights(valid, batch, start_weight=3.0, end_weight=1.0)
    uniform = _skill_endpoint_weights(valid, batch, start_weight=1.0, end_weight=1.0)

    torch.testing.assert_close(increasing, torch.tensor([1.0, 3.0]))
    torch.testing.assert_close(decreasing, torch.tensor([3.0, 1.0]))
    torch.testing.assert_close(uniform, torch.ones(2))


class _Stage0TrainabilityStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            stage0_a_train_components="vlm_lora,lang_bridge,cond,expert_lora",
            stage0_b_train_components="cond,cond_vision,expert_lora",
            stage0_a_drop_vlm=False,
            stage0_b_drop_vlm=True,
        )
        self.vlm = NamedLoRALinear(nn.Linear(4, 4))
        self.vlm.add_adapter("vlm_lora", 2, 4)
        self.expert = NamedLoRALinear(nn.Linear(4, 4))
        self.expert.add_adapter("expert_lora", 2, 4)
        self.cond = nn.Linear(4, 4)
        self.cond_vision = nn.Linear(4, 4)
        self.vlm_vision = nn.Linear(4, 4)
        self.state_proj = nn.Linear(4, 4)
        self.skill_proj = nn.Linear(4, 4)
        self.skill_reader = nn.Linear(4, 4)
        self.skill_head = nn.Linear(4, 4)
        self.lang_bridges = nn.ModuleList([nn.Linear(4, 4)])
        self.paligemma_with_expert = SimpleNamespace(
            paligemma=SimpleNamespace(model=SimpleNamespace(multi_modal_projector=None)))

    _set_requires_grad = staticmethod(SkillVLAPytorch._set_requires_grad)
    _stage0_components = SkillVLAPytorch._stage0_components
    _stage0_drops_vlm = SkillVLAPytorch._stage0_drops_vlm

    def _regime_groups(self):
        return {
            "llm": [self.vlm],
            "vlm_vision": [self.vlm_vision],
            "expert": [self.expert],
            "cond": [self.cond],
            "cond_vision": [self.cond_vision],
        }


def _adapter_trainable(module: NamedLoRALinear, name: str) -> bool:
    return all(param.requires_grad for param in module.adapters[name].parameters())


def test_stage0_branch_matrix_controls_exact_gradient_scope() -> None:
    model = _Stage0TrainabilityStub()

    SkillVLAPytorch._set_stage0_trainability(model, "stage0_a")
    assert all(param.requires_grad for param in model.cond.parameters())
    assert not any(param.requires_grad for param in model.cond_vision.parameters())
    assert _adapter_trainable(model.vlm, "vlm_lora")
    assert _adapter_trainable(model.expert, "expert_lora")
    assert all(param.requires_grad for param in model.lang_bridges.parameters())
    assert not model._stage0_drops_vlm("stage0_a")

    SkillVLAPytorch._set_stage0_trainability(model, "stage0_b")
    assert all(param.requires_grad for param in model.cond.parameters())
    assert all(param.requires_grad for param in model.cond_vision.parameters())
    assert not _adapter_trainable(model.vlm, "vlm_lora")
    assert _adapter_trainable(model.expert, "expert_lora")
    assert not any(param.requires_grad for param in model.lang_bridges.parameters())
    assert model._stage0_drops_vlm("stage0_b")


def test_stage0_can_unfreeze_vlm_vision_without_unfreezing_vlm() -> None:
    model = _Stage0TrainabilityStub()
    projector = nn.Linear(4, 4)
    model.paligemma_with_expert.paligemma.model.multi_modal_projector = projector
    model.config.stage0_a_train_components += ",vlm_vision"

    SkillVLAPytorch._set_stage0_trainability(model, "stage0_a")

    assert all(param.requires_grad for param in model.vlm_vision.parameters())
    assert all(param.requires_grad for param in projector.parameters())
    assert not any(param.requires_grad for param in model.vlm.base.parameters())


def test_stage0_terminator_vision_freeze_override() -> None:
    for freeze_vision in (True, False):
        terminator = nn.Module()
        terminator.vision_encoder = nn.Linear(4, 4)
        terminator.head = nn.Linear(4, 1)
        terminator.freeze_vision_encoder = not freeze_vision

        stub = SimpleNamespace(
            config=SimpleNamespace(
                terminator_dino_model_path=None,
                terminator_freeze_vision_encoder=freeze_vision,
            ),
            _construct_fsq=lambda path, dino_model_path: terminator,
            parameters=lambda: iter([torch.nn.Parameter(torch.zeros(1))]),
        )

        SkillVLAPytorch._build_terminator_trainable(stub, "FSQ.pt")

        assert terminator.freeze_vision_encoder is freeze_vision
        assert any(param.requires_grad for param in terminator.head.parameters())
        assert all(param.requires_grad != freeze_vision for param in terminator.vision_encoder.parameters())
        assert terminator.vision_encoder.training != freeze_vision
