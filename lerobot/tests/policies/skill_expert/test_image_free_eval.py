from types import SimpleNamespace

import torch

from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPytorch


class _ImageFreeSamplerStub:
    def __init__(self) -> None:
        self.config = SimpleNamespace(num_inference_steps=3, chunk_size=2, max_action_dim=3)
        self.lora_active: list[bool] = []
        self.expert_calls = 0

    def sample_noise(self, shape, device):
        return torch.zeros(shape, device=device)

    def _set_expert_lora_active(self, active: bool) -> None:
        self.lora_active.append(active)

    def _action_prefix(self, skill_code, state):
        return None

    def _expert_cond(self, time, state, skill_code):
        return time

    def _skill_broadcast(self, skill_code):
        return None

    def _run_expert_only(self, x_t, expert_cond, action_prefix, skill_broadcast):
        self.expert_calls += 1
        return torch.ones_like(x_t)


def test_image_free_sampler_uses_only_the_action_expert_path() -> None:
    sampler = _ImageFreeSamplerStub()
    state = torch.zeros(1, 4)
    skill = torch.zeros(1, dtype=torch.long)

    actions = SkillExpertPytorch.sample_actions_image_free(sampler, state, skill)

    assert sampler.lora_active == [True]
    assert sampler.expert_calls == 3
    torch.testing.assert_close(actions, -torch.ones(1, 2, 3))
