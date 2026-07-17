from types import SimpleNamespace

import torch

from lerobot.policies.skillVLA import modeling_skillVLA
from lerobot.policies.skillVLA.modeling_skillVLA import SkillVLAPytorch, _reverse_skill_endpoint_weights


class _CondSeveredStub:
    training = True

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            regime_probs_ft=None,
            cond_severed_prob=1.0,
            cond_severed_anchor_weight=0.25,
            severed_hold_target=True,
            cond_skill_source="gt",
            vsa_distill=False,
        )
        self._fsq_half = torch.ones(3)
        self.calls: list[tuple[frozenset[str], torch.Tensor]] = []

    def finalize_motion_counter(self) -> None:
        pass

    def _resolved_pt_stage(self):
        return "cond"

    def _code_to_z(self, skill_code):
        return torch.zeros(skill_code.shape[0], 3)

    def _flow_view(
        self, cond_images, start_images, lang_tokens, lang_masks, state, skill_zq,
        x_t, time, severed, severed_adapters=frozenset(),
    ):
        self.calls.append((severed_adapters, x_t.detach().clone()))
        value = 3.0 if "cond_lora" in severed_adapters else 0.0
        return torch.full_like(x_t, value), torch.zeros(1)


def test_cond_severed_anchors_to_frozen_vsa_on_hold_derived_xt() -> None:
    model = _CondSeveredStub()
    actions = torch.full((1, 2, 3), 0.8)
    hold_actions = torch.full_like(actions, 0.2)
    noise = torch.ones_like(actions)
    time = torch.tensor([0.25])

    losses, skill_hidden = SkillVLAPytorch.forward(
        model,
        cond_images=[],
        start_images=[],
        lang_tokens=torch.zeros(1, 1, dtype=torch.long),
        lang_masks=torch.ones(1, 1, dtype=torch.bool),
        state=torch.zeros(1, 4),
        skill_code=torch.zeros(1, dtype=torch.long),
        actions=actions,
        noise=noise,
        time=time,
        hold_actions=hold_actions,
    )

    expected_xt = time[:, None, None] * noise + (1 - time[:, None, None]) * hold_actions
    assert [adapters for adapters, _ in model.calls] == [frozenset(), frozenset({"cond_lora"})]
    torch.testing.assert_close(model.calls[0][1], expected_xt)
    torch.testing.assert_close(model.calls[1][1], expected_xt)
    torch.testing.assert_close(model._last_cond_anchor_raw, torch.full_like(actions, 9.0))
    torch.testing.assert_close(losses, torch.full_like(actions, 2.25))
    assert skill_hidden is None
    assert model._last_severed_hold is True


def test_severed_flow_selects_adapters_before_building_cond_tokens(monkeypatch) -> None:
    events = []
    active = set()

    def set_active(adapters):
        active.clear()
        active.update(adapters)
        events.append(("active", frozenset(active)))

    class _FlowStub:
        _has_stage1_expert_lora = True

        def _active_adapters(self, adapters=()):
            return SkillVLAPytorch._active_adapters(self, adapters)

        def _cond_tokens(self, cond_images):
            events.append(("cond", frozenset(active)))
            return torch.zeros(1, 1, 2)

        def _vsa_velocity(self, cond_tokens, x_t, time, state, skill_zq):
            return torch.zeros_like(x_t)

    monkeypatch.setattr(modeling_skillVLA, "set_active_adapters", set_active)
    stub = _FlowStub()
    SkillVLAPytorch._flow_view(
        stub,
        cond_images=[],
        start_images=[],
        lang_tokens=torch.zeros(1, 1, dtype=torch.long),
        lang_masks=torch.ones(1, 1, dtype=torch.bool),
        state=torch.zeros(1, 2),
        skill_zq=torch.zeros(1, 3),
        x_t=torch.zeros(1, 2, 2),
        time=torch.zeros(1),
        severed=True,
        severed_adapters=frozenset({"cond_lora"}),
    )

    expected = frozenset({"cond_lora", "expert_lora"})
    assert events == [("active", expected), ("cond", expected)]


def test_stage2_reverse_weight_decays_from_skill_start_to_end() -> None:
    valid = torch.tensor([[True, False, False], [True, True, True]])
    batch = {
        "skill_ds": torch.tensor([0, 2]),
        "skill_de": torch.tensor([4, 2]),
    }

    weights = _reverse_skill_endpoint_weights(valid, batch, max_weight=3.0)

    torch.testing.assert_close(weights, torch.tensor([3.0, 1.0]))
