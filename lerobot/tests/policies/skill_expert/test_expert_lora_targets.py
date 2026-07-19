from types import SimpleNamespace

import torch
from torch import nn

from lerobot.policies.pi05.lora import NamedLoRALinear
from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy, SkillExpertPytorch


class _ExpertAdapterStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            lora_expert=True,
            lora_targets="q,mlp,action_out",
            lora_rank=2,
            lora_alpha=4.0,
            lora_dropout=0.0,
        )
        self.gemma_expert = nn.Module()
        self.gemma_expert.q_proj = nn.Linear(4, 4)
        self.gemma_expert.mlp = nn.Module()
        self.gemma_expert.mlp.gate_proj = nn.Linear(4, 8)
        self.gemma_expert.mlp.up_proj = nn.Linear(4, 8)
        self.gemma_expert.mlp.down_proj = nn.Linear(8, 4)
        self.cond_encoder = nn.Module()
        self.cond_encoder.q_proj = nn.Linear(4, 4)
        self.action_in_proj = nn.Linear(3, 4)
        self.action_out_proj = nn.Linear(4, 3)

    @property
    def _wdtype(self):
        return self.action_in_proj.weight.dtype


def test_expanded_targets_wrap_mlp_and_action_output_but_not_cond_or_action_input() -> None:
    model = _ExpertAdapterStub()

    SkillExpertPytorch._inject_expert_lora(model)

    assert isinstance(model.gemma_expert.q_proj, NamedLoRALinear)
    assert isinstance(model.gemma_expert.mlp.gate_proj, NamedLoRALinear)
    assert isinstance(model.gemma_expert.mlp.up_proj, NamedLoRALinear)
    assert isinstance(model.gemma_expert.mlp.down_proj, NamedLoRALinear)
    assert isinstance(model.action_out_proj, NamedLoRALinear)
    assert isinstance(model.action_in_proj, nn.Linear)
    assert isinstance(model.cond_encoder.q_proj, nn.Linear)


def test_frozen_regime_trains_outer_action_projection_adapter() -> None:
    model = _ExpertAdapterStub()
    SkillExpertPytorch._inject_expert_lora(model)
    model.time_mlp_in = nn.Linear(4, 4)
    model.time_mlp_out = nn.Linear(4, 4)
    model.state_proj = nn.Linear(4, 4)
    model.skill_proj = nn.Linear(4, 4)
    policy = SimpleNamespace(config=model.config, model=model)

    SkillExpertPolicy._apply_frozen_expert_lora_regime(policy)

    assert all(param.requires_grad for param in model.action_out_proj.adapters["expert"].parameters())
    assert not any(param.requires_grad for param in model.action_out_proj.base.parameters())


def test_full_finetune_regime_unfreezes_complete_fsq_action_side() -> None:
    model = _ExpertAdapterStub()
    model.time_mlp_in = nn.Linear(4, 4)
    model.time_mlp_out = nn.Linear(4, 4)
    model.state_proj = nn.Linear(4, 4)
    model.skill_proj = nn.Linear(4, 4)
    model.config = SimpleNamespace(lora_expert=False)
    action_modules = (
        model.gemma_expert,
        model.action_in_proj,
        model.action_out_proj,
        model.time_mlp_in,
        model.time_mlp_out,
        model.state_proj,
        model.skill_proj,
    )
    for module in action_modules:
        for param in module.parameters():
            param.requires_grad_(False)

    policy = SimpleNamespace(config=model.config, model=model)
    SkillExpertPolicy._apply_frozen_expert_lora_regime(policy)

    assert all(param.requires_grad for module in action_modules for param in module.parameters())


def test_full_finetune_reference_is_nonpersistent_and_matches_action_state() -> None:
    model = _ExpertAdapterStub()
    model.time_mlp_in = nn.Linear(4, 4)
    model.time_mlp_out = nn.Linear(4, 4)
    model.state_proj = nn.Linear(4, 4)
    model.skill_proj = nn.Linear(4, 4)
    model.config = SimpleNamespace(lora_expert=False)
    model._fsq_reference_buffers = {}
    prefixes = (
        "gemma_expert.", "state_proj.", "skill_proj.", "action_in_proj.",
        "action_out_proj.", "time_mlp_in.", "time_mlp_out.",
    )
    state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if key.startswith(prefixes)
    }

    SkillExpertPytorch.set_fsq_reference(model, state)

    reference_action_in = SkillExpertPytorch._fsq_reference_module_state(model, "action_in_proj.")
    torch.testing.assert_close(reference_action_in["weight"], state["action_in_proj.weight"])
    assert not any(key.startswith("_fsq_reference_") for key in model.state_dict())


def test_full_finetune_image_free_batch_anchors_to_frozen_fsq_teacher() -> None:
    class _FullFtAnchorStub:
        config = SimpleNamespace(lora_expert=False)

        def __init__(self) -> None:
            self.teacher_calls = 0

        def sample_time(self, bsize, device):
            return torch.zeros(bsize, device=device)

        def sample_noise(self, shape, device):
            return torch.ones(shape, device=device)

        def _action_prefix(self, skill_code, state):
            return None

        def _expert_cond(self, time, state, skill_code):
            return torch.zeros(state.shape[0], 1, device=state.device)

        def _skill_broadcast(self, skill_code):
            return None

        def _run_fsq_reference_expert_only(self, x_t, time, state, skill_code):
            self.teacher_calls += 1
            return torch.zeros_like(x_t)

        def _run_expert_only(self, x_t, expert_cond, action_prefix, skill_broadcast):
            return torch.full_like(x_t, 2.0)

    model = _FullFtAnchorStub()
    actions = torch.full((2, 3, 4), 0.25)
    residual, diagnostic = SkillExpertPytorch.image_free_lora_anchor(
        model, torch.zeros(2, 5), torch.zeros(2, dtype=torch.long), actions,
        noise=torch.ones_like(actions), time=torch.zeros(2),
    )

    assert model.teacher_calls == 1
    torch.testing.assert_close(residual, torch.full_like(actions, 2.0))
    torch.testing.assert_close(diagnostic, torch.full_like(actions, -1.25))
