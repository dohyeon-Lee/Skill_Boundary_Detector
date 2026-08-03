from types import SimpleNamespace

import torch
from torch import nn

from lerobot.policies.pi05.lora import NamedLoRALinear
from lerobot.policies.pi_gemma import PiGemmaRMSNorm, add_broadcast_condition
from lerobot.policies.skillVLA.modeling_skillVLA import (
    LanguageKVBridge,
    Stage0VLMResidual,
    SkillVLAPytorch,
    SkillVLAPolicy,
    _relative_language_ranking,
    _skill_endpoint_weights,
    _stage0_endpoint_xyz_loss,
)


def test_stage0_vlm_residual_starts_as_fp32_zero_correction_with_open_gate() -> None:
    residual = Stage0VLMResidual(
        expert_dim=16,
        vlm_dim=16,
        n_heads=4,
        dropout=0.0,
        alpha_min=0.1,
        alpha_max=0.2,
        init_alpha=0.15,
        zero_init_output=True,
    ).to(dtype=torch.bfloat16)

    query = torch.randn(2, 3, 16, dtype=torch.bfloat16)
    vlm = torch.randn(2, 5, 16, dtype=torch.bfloat16)
    valid = torch.ones(2, 5, dtype=torch.bool)
    blocked = torch.zeros(5, dtype=torch.bool)
    correction = residual(query, vlm, valid, blocked)

    assert residual.gate_logit.dtype == torch.float32
    torch.testing.assert_close(residual.alpha(), torch.tensor(0.15))
    assert correction.dtype == torch.float32
    assert torch.count_nonzero(correction) == 0

    (correction - 1.0).square().mean().backward()
    assert residual.attn.out_proj.weight.grad is not None
    assert torch.count_nonzero(residual.attn.out_proj.weight.grad) > 0


def test_stage0_detached_base_call_keeps_input_gradient_only() -> None:
    class Stub:
        _call_with_detached_parameters = staticmethod(SkillVLAPytorch._call_with_detached_parameters)
        _action_out = SkillVLAPytorch._action_out
        _expert_final_norm = SkillVLAPytorch._expert_final_norm

        def __init__(self) -> None:
            self._expert = SimpleNamespace(norm=PiGemmaRMSNorm(8, cond_dim=8))
            self.action_out_proj = nn.Linear(8, 3)

    model = Stub()
    hidden = torch.randn(2, 4, 8, requires_grad=True)
    condition = torch.randn(2, 8, requires_grad=True)
    normalized = model._expert_final_norm(hidden, condition, detach_parameters=True)
    output = model._action_out(normalized, detach_parameters=True)

    output.square().mean().backward()

    assert hidden.grad is not None
    assert torch.count_nonzero(hidden.grad) > 0
    assert condition.grad is None
    assert all(param.grad is None for param in model._expert.norm.parameters())
    assert all(param.grad is None for param in model.action_out_proj.parameters())


class _SplitRouteStub(nn.Module):
    _call_with_detached_parameters = staticmethod(SkillVLAPytorch._call_with_detached_parameters)
    _action_out = SkillVLAPytorch._action_out
    _expert_final_norm = SkillVLAPytorch._expert_final_norm
    _stage0_dual_flow_view = SkillVLAPytorch._stage0_dual_flow_view

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(stage0_gradient_routing="split")
        self.joint_calls = 0
        self.base = nn.Linear(8, 8)
        self.vlm = nn.Linear(8, 8)
        self._expert = SimpleNamespace(norm=PiGemmaRMSNorm(8))
        self.action_out_proj = nn.Linear(8, 3)
        self.stage0_vlm_residual = Stage0VLMResidual(
            expert_dim=8,
            vlm_dim=8,
            n_heads=2,
            dropout=0.0,
            alpha_min=0.1,
            alpha_max=0.2,
            init_alpha=0.15,
            zero_init_output=False,
        )

    @staticmethod
    def _active_adapters(_names):
        return frozenset()

    @staticmethod
    def _cond_tokens(cond_images):
        return torch.zeros(cond_images[0].shape[0], 1, 8)

    def _vlm_tokens(self, start_images, _lang_tokens, _lang_masks):
        hidden = self.vlm(start_images[0])
        valid = torch.ones(hidden.shape[:2], dtype=torch.bool)
        blocked = torch.zeros(hidden.shape[1], dtype=torch.bool)
        return hidden, valid, blocked

    @staticmethod
    def _action_in(x_t):
        return x_t

    @staticmethod
    def _action_prefix_from_z(_skill_zq):
        return None

    @staticmethod
    def _expert_cond_from_z(_time, _state, _skill_zq):
        return None

    @staticmethod
    def _skill_broadcast_from_z(_skill_zq):
        return None

    @staticmethod
    def _cond_adarms(_state):
        return None

    def _joint_forward(
        self, _cond_tokens, vlm_embeds, _vlm_pad, _vlm_xattn_block, action_tokens,
        _expert_cond, **_kwargs,
    ):
        self.joint_calls += 1
        return vlm_embeds, None, self.base(action_tokens)

    @staticmethod
    def _vlm_prefix_out(vlm_embeds, _vlm_pad, all_layers=False, *, predictor=True):
        assert not all_layers
        assert predictor is False
        return vlm_embeds


def _stage0_split_route_outputs(model: _SplitRouteStub) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = 2
    return model._stage0_dual_flow_view(
        cond_images=[torch.randn(batch_size, 1, 8)],
        start_images=[torch.randn(batch_size, 3, 8)],
        lang_tokens=torch.zeros(batch_size, 1, dtype=torch.long),
        lang_masks=torch.ones(batch_size, 1, dtype=torch.bool),
        state=torch.randn(batch_size, 4),
        skill_zq=torch.randn(batch_size, 3),
        x_t=torch.randn(batch_size, 5, 8),
        time=torch.rand(batch_size),
    )


def test_stage0_split_route_sends_conditional_gradient_only_to_vlm_residual() -> None:
    model = _SplitRouteStub()
    conditional, _ = _stage0_split_route_outputs(model)

    conditional.square().mean().backward()

    assert any(param.grad is not None for param in model.vlm.parameters())
    assert any(param.grad is not None for param in model.stage0_vlm_residual.parameters())
    assert all(param.grad is None for param in model.base.parameters())
    assert all(param.grad is None for param in model._expert.norm.parameters())
    assert all(param.grad is None for param in model.action_out_proj.parameters())


def test_stage0_split_route_sends_unconditional_gradient_only_to_base() -> None:
    model = _SplitRouteStub()
    _, unconditional = _stage0_split_route_outputs(model)

    unconditional.square().mean().backward()

    assert any(param.grad is not None for param in model.base.parameters())
    assert any(param.grad is not None for param in model._expert.norm.parameters())
    assert any(param.grad is not None for param in model.action_out_proj.parameters())
    assert all(param.grad is None for param in model.vlm.parameters())
    assert all(param.grad is None for param in model.stage0_vlm_residual.parameters())


def test_stage0_wrong_language_reuses_one_base_forward() -> None:
    model = _SplitRouteStub()
    batch_size = 2
    model._stage0_dual_flow_view(
        cond_images=[torch.randn(batch_size, 1, 8)],
        start_images=[torch.randn(batch_size, 3, 8)],
        lang_tokens=torch.zeros(batch_size, 1, dtype=torch.long),
        lang_masks=torch.ones(batch_size, 1, dtype=torch.bool),
        state=torch.randn(batch_size, 4),
        skill_zq=torch.randn(batch_size, 3),
        x_t=torch.randn(batch_size, 5, 8),
        time=torch.rand(batch_size),
        wrong_lang_tokens=torch.ones(batch_size, 1, dtype=torch.long),
        wrong_lang_masks=torch.ones(batch_size, 1, dtype=torch.bool),
    )

    assert model.joint_calls == 1
    assert model._last_stage0_wrong_velocity is not None
    assert model._last_stage0_wrong_velocity.shape == (batch_size, 5, 3)


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


def test_relative_language_ranking_uses_current_skill_steps_per_sample() -> None:
    correct = torch.tensor([
        [[1.0], [1.0], [100.0]],
        [[2.0], [2.0], [2.0]],
    ])
    wrong = torch.tensor([
        [[1.02], [1.02], [0.0]],
        [[2.0], [2.0], [2.0]],
    ])
    within_skill = torch.tensor([
        [True, True, False],
        [True, True, True],
    ])

    result = _relative_language_ranking(
        correct, wrong, within_skill, torch.tensor([True, False]), relative_margin=0.01)

    assert result is not None
    torch.testing.assert_close(result["relative_gap"], torch.tensor(0.02))
    torch.testing.assert_close(result["loss"], torch.tensor(0.0))
    torch.testing.assert_close(result["satisfied_fraction"], torch.tensor(1.0))
    torch.testing.assert_close(result["active_fraction"], torch.tensor(0.0))


def test_relative_language_ranking_hinge_is_samplewise() -> None:
    correct = torch.ones(2, 1, 1)
    wrong = torch.tensor([[[1.02]], [[1.0]]])
    result = _relative_language_ranking(
        correct,
        wrong,
        torch.ones(2, 1, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
        relative_margin=0.01,
    )

    assert result is not None
    torch.testing.assert_close(result["loss"], torch.tensor(0.005))
    torch.testing.assert_close(result["active_fraction"], torch.tensor(0.5))
    torch.testing.assert_close(result["satisfied_fraction"], torch.tensor(0.5))


def test_stage0_endpoint_xyz_loss_uses_only_valid_chunk_endpoint() -> None:
    predicted = torch.zeros(2, 3, 4, requires_grad=True)
    with torch.no_grad():
        predicted[0, 0, 0] = 1.0
        predicted[0, 1, 0] = -1.0
        predicted[1, 0, 0] = 1.0
        predicted[1, 1, 0] = 100.0
    target = torch.zeros_like(predicted)
    valid = torch.tensor([
        [True, True, True],
        [True, False, False],
    ])

    loss = _stage0_endpoint_xyz_loss(predicted, target, valid)
    loss.backward()

    # Sample 0 cancels at the endpoint; sample 1 has XYZ MSE=(1^2+0+0)/3.
    torch.testing.assert_close(loss.detach(), torch.tensor(1.0 / 6.0))
    assert predicted.grad is not None
    assert torch.count_nonzero(predicted.grad[1, 1:]) == 0
    assert torch.count_nonzero(predicted.grad[..., 3:]) == 0


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


class _Stage0FlowStub:
    training = True

    def __init__(self, b_probability: float) -> None:
        self.config = SimpleNamespace(
            regime_probs_ft=None,
            stage0_vlm_severed_prob=b_probability,
            stage0_a_drop_vlm=False,
            stage0_b_drop_vlm=True,
            cond_severed_prob=0.0,
            severed_hold_target=True,
            cond_skill_source="gt",
            stage0_wrong_language_weight=0.0,
            vsa_distill=False,
        )
        self._fsq_half = torch.ones(3)
        self.flow_calls = []

    def finalize_motion_counter(self) -> None:
        pass

    def _resolved_pt_stage(self):
        return "stage0"

    def _set_stage0_trainability(self, regime: str) -> None:
        pass

    def _stage0_drops_vlm(self, regime: str) -> bool:
        return regime == "stage0_b"

    def _code_to_z(self, skill_code):
        return torch.zeros(skill_code.shape[0], 3)

    def _flow_view(
        self, cond_images, start_images, lang_tokens, lang_masks, state, skill_zq,
        x_t, time, severed, severed_adapters=frozenset(),
    ):
        self.flow_calls.append((severed, x_t.detach().clone()))
        return torch.zeros_like(x_t), torch.zeros(1)


def test_stage0_hold_target_is_b_only() -> None:
    actions = torch.full((1, 2, 3), 0.8)
    hold_actions = torch.full_like(actions, 0.2)
    noise = torch.ones_like(actions)
    time = torch.tensor([0.25])
    expected_a_xt = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
    expected_b_xt = time[:, None, None] * noise + (1 - time[:, None, None]) * hold_actions

    for b_probability, expected_severed, expected_xt in (
        (0.0, False, expected_a_xt),
        (1.0, True, expected_b_xt),
    ):
        model = _Stage0FlowStub(b_probability)
        SkillVLAPytorch.forward(
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

        assert model.flow_calls[0][0] is expected_severed
        torch.testing.assert_close(model.flow_calls[0][1], expected_xt)
        assert model._last_severed_hold is expected_severed


class _Stage0TrainabilityStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            stage0_a_train_components="vlm_lora,lang_bridge,cond,expert_lora",
            stage0_b_train_components="cond,cond_vision,expert_lora",
            stage0_a_drop_vlm=False,
            stage0_b_drop_vlm=True,
            stage0_expert_source="fsq",
            stage0_cond_state_adarms=False,
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
    _stage0_pi05_expert = SkillVLAPytorch._stage0_pi05_expert
    _direct_expert_conditioning = SkillVLAPytorch._direct_expert_conditioning
    _cond_uses_state_adarms = SkillVLAPytorch._cond_uses_state_adarms

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


def test_stage0_can_unfreeze_vlm_base_from_branch_matrix() -> None:
    model = _Stage0TrainabilityStub()
    model.config.stage0_a_train_components += ",vlm"

    SkillVLAPytorch._set_stage0_trainability(model, "stage0_a")

    assert all(param.requires_grad for param in model.vlm.base.parameters())
    assert _adapter_trainable(model.vlm, "vlm_lora")


def test_stage0_matrix_can_unfreeze_every_component() -> None:
    model = _Stage0TrainabilityStub()
    model.config.stage0_a_train_components = (
        "vlm,cond,expert,vlm_lora,expert_lora,lang_bridge,"
        "vlm_vision,cond_vision,skill_reader,skill_head"
    )

    SkillVLAPytorch._set_stage0_trainability(model, "stage0_a")

    for module in (
        model.vlm.base,
        model.cond,
        model.expert.base,
        model.vlm_vision,
        model.cond_vision,
        model.state_proj,
        model.skill_proj,
        model.skill_reader,
        model.skill_head,
        model.lang_bridges,
    ):
        assert all(param.requires_grad for param in module.parameters())
    assert _adapter_trainable(model.vlm, "vlm_lora")
    assert _adapter_trainable(model.expert, "expert_lora")


def test_stage0_pi05_state_projection_belongs_to_cond() -> None:
    model = _Stage0TrainabilityStub()
    model.config.stage0_expert_source = "pi05_base"
    model.config.stage0_cond_state_adarms = True
    model.config.stage0_a_train_components = "cond"

    SkillVLAPytorch._set_stage0_trainability(model, "stage0_a")

    assert all(param.requires_grad for param in model.cond.parameters())
    assert all(param.requires_grad for param in model.state_proj.parameters())
    assert not any(param.requires_grad for param in model.skill_proj.parameters())
    assert not any(param.requires_grad for param in model.expert.base.parameters())


def test_stage0_pi05_expert_is_time_only_and_has_no_skill_prefix() -> None:
    class Stub:
        config = SimpleNamespace(stage0_expert_source="pi05_base")
        stage1_config = SimpleNamespace(state_cond_mode="state")
        _stage0_pi05_expert = SkillVLAPytorch._stage0_pi05_expert
        _direct_expert_conditioning = SkillVLAPytorch._direct_expert_conditioning
        _state_cond_mode = SkillVLAPytorch._state_cond_mode

        @staticmethod
        def _time_cond(time):
            return torch.full((time.shape[0], 4), 2.0)

        @staticmethod
        def _state_cond(state):
            raise AssertionError("pi05 expert must not consume state directly")

    model = Stub()
    time = torch.tensor([0.25, 0.75])
    state = torch.randn(2, 6)
    skill_zq = torch.randn(2, 3)

    cond = SkillVLAPytorch._expert_cond_from_z(model, time, state, skill_zq)
    prefix = SkillVLAPytorch._action_prefix_from_z(model, skill_zq)

    torch.testing.assert_close(cond, torch.full((2, 4), 2.0))
    assert prefix is None


def test_broadcast_mode_keeps_skill_out_of_prefix_and_adarms() -> None:
    class Stub:
        config = SimpleNamespace(stage0_expert_source="fsq")
        stage1_config = SimpleNamespace(state_cond_mode="broadcast")
        skill_proj = nn.Linear(3, 4, bias=False)
        _stage0_pi05_expert = SkillVLAPytorch._stage0_pi05_expert
        _direct_expert_conditioning = SkillVLAPytorch._direct_expert_conditioning
        _state_cond_mode = SkillVLAPytorch._state_cond_mode

        @property
        def _wdtype(self):
            return self.skill_proj.weight.dtype

        @staticmethod
        def _time_cond(time):
            return torch.full((time.shape[0], 4), 2.0)

        @staticmethod
        def _state_cond(state):
            return torch.full((state.shape[0], 4), 3.0)

    model = Stub()
    with torch.no_grad():
        model.skill_proj.weight.copy_(torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]))
    time = torch.tensor([0.25])
    state = torch.randn(1, 6)
    skill_zq = torch.tensor([[0.2, 0.4, 0.6]])

    expert_cond = SkillVLAPytorch._expert_cond_from_z(model, time, state, skill_zq)
    prefix = SkillVLAPytorch._action_prefix_from_z(model, skill_zq)
    broadcast = SkillVLAPytorch._skill_broadcast_from_z(model, skill_zq)

    torch.testing.assert_close(expert_cond, torch.full((1, 4), 5.0))
    assert prefix is None
    torch.testing.assert_close(broadcast, torch.tensor([[0.2, 0.4, 0.6, 1.2]]))


def test_broadcast_condition_is_added_to_every_token_without_mutating_input() -> None:
    hidden = torch.zeros(2, 3, 4)
    condition = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

    result = add_broadcast_condition(hidden, condition)

    torch.testing.assert_close(result, condition[:, None, :].expand(-1, 3, -1))
    torch.testing.assert_close(hidden, torch.zeros_like(hidden))


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
                _construct_fsq=lambda path: terminator,
            parameters=lambda: iter([torch.nn.Parameter(torch.zeros(1))]),
        )

        SkillVLAPytorch._build_terminator_trainable(stub, "FSQ.pt")

        assert terminator.freeze_vision_encoder is freeze_vision
        assert any(param.requires_grad for param in terminator.head.parameters())
        assert all(param.requires_grad != freeze_vision for param in terminator.vision_encoder.parameters())
        assert terminator.vision_encoder.training != freeze_vision
