from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from safetensors.torch import save_file

from lerobot.policies.pi05.modeling_pi05 import get_gemma_config
from lerobot.policies.skill_expert.configuration_skill_expert import SkillExpertConfig
from lerobot.policies.skill_expert.modeling_skill_expert import (
    SkillExpertPolicy,
    SkillExpertPytorch,
    _load_pretrained_state_dict,
    _map_pi05_key,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


def test_stage1_uses_two_matching_18_layer_transformers() -> None:
    config = SkillExpertConfig()

    assert config.cond_encoder_variant == config.action_expert_variant == "gemma_300m"
    assert get_gemma_config(config.action_expert_variant).depth == 18


def test_stage1_contract_is_dino_with_selectable_skill_route_and_full_vocabulary() -> None:
    config = SkillExpertConfig()

    assert config.vision_backbone == "dino"
    assert not config.freeze_vision_encoder
    assert config.state_cond_mode == "broadcast"
    assert config.skill_fsq_levels == [3, 3, 3]
    assert config.skill_vocab_size == 27
    assert not hasattr(config, "lora_enable")
    assert config.fsq_path is None  # optional terminator source; never an expert warm-start

    assert SkillExpertConfig(state_cond_mode="token").state_cond_mode == "token"
    with pytest.raises(ValueError, match="broadcast.*token"):
        SkillExpertConfig(state_cond_mode="state_skill")
    with pytest.raises(ValueError, match="dino_lr cannot be set"):
        SkillExpertConfig(freeze_vision_encoder=True, dino_lr=1e-5)
    with pytest.raises(ValueError, match="requires fsq_path"):
        SkillExpertConfig(train_terminator=True)
    with pytest.raises(ValueError, match="action_loss_mode"):
        SkillExpertConfig(action_loss_mode="endpoint_only")


def test_stage3a_predictor_contract_allows_only_skill_lora_vlm_gradients() -> None:
    config = SkillExpertConfig(
        train_skill_predictor=True,
        skill_predictor_all_layers=True,
        skill_predictor_detach_vlm=False,
        skill_predictor_lora=True,
        skill_predictor_deadzone_frac=0.8,
    )

    assert config.skill_predictor_lora_targets == "q,k,v,o"
    assert config.skill_predictor_lora_rank == 8
    assert config.skill_predictor_lora_alpha == 16.0
    assert config.skill_predictor_lora_lr_scale == 10.0
    with pytest.raises(ValueError, match="must be False"):
        SkillExpertConfig(
            train_skill_predictor=True,
            skill_predictor_lora=True,
            skill_predictor_detach_vlm=True,
        )
    with pytest.raises(ValueError, match="full VLM fine-tuning"):
        SkillExpertConfig(
            train_skill_predictor=True,
            skill_predictor_lora=False,
            skill_predictor_detach_vlm=False,
        )


def test_pi05_warm_start_maps_only_action_expert_motion_prior() -> None:
    assert _map_pi05_key(
        "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight"
    ) == "model.gemma_expert.model.layers.0.self_attn.q_proj.weight"
    assert _map_pi05_key("action_in_proj.weight") == "model.action_in_proj.weight"
    assert _map_pi05_key("time_mlp_out.bias") == "model.time_mlp_out.bias"
    assert _map_pi05_key(
        "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight"
    ) is None
    assert _map_pi05_key(
        "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight",
        include_predictor_vlm=True,
    ) == (
        "model.skill_predictor.vlm.language_model.layers.0.self_attn.q_proj.weight"
    )


def test_pi05_loader_does_not_materialize_vlm_weights(tmp_path) -> None:
    checkpoint_path = tmp_path / "model.safetensors"
    save_file(
        {
            "paligemma_with_expert.gemma_expert.model.norm.weight": torch.ones(2),
            "paligemma_with_expert.paligemma.model.language_model.norm.weight": torch.ones(5),
            "action_out_proj.weight": torch.ones(3, 2),
        },
        checkpoint_path,
    )

    loaded = _load_pretrained_state_dict(tmp_path, {})

    assert loaded is not None
    state_dict, is_pi05 = loaded
    assert is_pi05
    assert set(state_dict) == {
        "model.gemma_expert.model.norm.weight",
        "model.action_out_proj.weight",
    }
    assert _map_pi05_key(
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight"
    ) is None


def test_flat_skill_code_maps_to_fsq_grid_coordinate() -> None:
    model = SimpleNamespace(
        _fsq_levels=torch.tensor([3, 3, 3]),
        _fsq_strides=torch.tensor([1, 3, 9]),
        _fsq_half=torch.tensor([1.0, 1.0, 1.0]),
    )

    low_and_high = SkillExpertPytorch._code_to_zq(model, torch.tensor([0, 26]))

    torch.testing.assert_close(
        low_and_high, torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    )


def test_skill_conditioning_selects_broadcast_or_one_prefix_token() -> None:
    model = SkillExpertPytorch.__new__(SkillExpertPytorch)
    nn.Module.__init__(model)
    model.action_in_proj = nn.Linear(2, 4)
    model.skill_proj = nn.Linear(3, 4, bias=False)
    model.register_buffer("_fsq_levels", torch.tensor([3, 3, 3]))
    model.register_buffer("_fsq_strides", torch.tensor([1, 3, 9]))
    model.register_buffer("_fsq_half", torch.tensor([1.0, 1.0, 1.0]))
    code = torch.tensor([0, 26])

    model.config = SimpleNamespace(state_cond_mode="broadcast")
    token, broadcast = model._skill_conditioning(code)
    assert token is None
    assert broadcast is not None and broadcast.shape == (2, 4)

    model.config = SimpleNamespace(state_cond_mode="token")
    token, token_broadcast = model._skill_conditioning(code)
    assert token_broadcast is None
    assert token is not None and token.shape == (2, 1, 4)
    torch.testing.assert_close(token[:, 0], broadcast)


def test_token_route_uses_separate_prefix_block_and_decodes_only_actions() -> None:
    model = SkillExpertPytorch.__new__(SkillExpertPytorch)
    nn.Module.__init__(model)
    model.action_in_proj = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.action_in_proj.weight.copy_(torch.eye(2))
    model.gemma_expert = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=1),
            norm=object(),
        )
    )
    model.cond_encoder = SimpleNamespace(model=object())
    model._gradient_checkpointing = False
    captured = {}

    def fake_layer(
        layer_index,
        streams,
        attention_mask,
        position_ids,
        adarms_conditions,
        **kwargs,
    ):
        del layer_index, position_ids, adarms_conditions, kwargs
        captured["attention"] = attention_mask
        captured["action_stream"] = streams[1]
        return streams

    condition = torch.zeros(1, 2, 2)
    actions = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    skill_token = torch.tensor([[[9.0, 8.0]]])
    with (
        patch(
            "lerobot.policies.skill_expert.modeling_skill_expert.compute_layer_complete",
            side_effect=fake_layer,
        ),
        patch(
            "lerobot.policies.skill_expert.modeling_skill_expert.layernorm_forward",
            side_effect=lambda norm, hidden, cond: (hidden, None),
        ),
    ):
        hidden = model._run_joint_hidden(
            condition,
            actions,
            torch.zeros(1, 2),
            skill_broadcast=None,
            skill_token=skill_token,
        )

    # The expert receives [skill, action x 3], but only the last three action
    # positions are returned to the flow head.
    assert captured["action_stream"].shape == (1, 4, 2)
    torch.testing.assert_close(hidden, actions)
    allowed = captured["attention"][0, 0].eq(0)
    expected = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    torch.testing.assert_close(allowed, expected)


def test_token_route_cached_sampling_uses_same_prefix_mask() -> None:
    class _ConditionModel:
        def forward(self, **kwargs):
            del kwargs
            return SimpleNamespace(past_key_values=[("k", "v")])

    class _ActionModel:
        def __init__(self) -> None:
            self.attention = None
            self.inputs = None

        def forward(self, *, inputs_embeds, attention_mask, **kwargs):
            del kwargs
            self.inputs = inputs_embeds
            self.attention = attention_mask
            return SimpleNamespace(last_hidden_state=inputs_embeds)

    model = SkillExpertPytorch.__new__(SkillExpertPytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(state_cond_mode="token")
    model.action_in_proj = nn.Linear(2, 2, bias=False)
    model.action_out_proj = nn.Linear(2, 2, bias=False)
    model.skill_proj = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        model.action_in_proj.weight.copy_(torch.eye(2))
        model.action_out_proj.weight.copy_(torch.eye(2))
    model.register_buffer("_fsq_levels", torch.tensor([3, 3, 3]))
    model.register_buffer("_fsq_strides", torch.tensor([1, 3, 9]))
    model.register_buffer("_fsq_half", torch.tensor([1.0, 1.0, 1.0]))
    action_model = _ActionModel()
    model.cond_encoder = SimpleNamespace(model=_ConditionModel())
    model.gemma_expert = SimpleNamespace(model=action_model)
    model._expert_condition = lambda time, state: torch.zeros(
        state.shape[0], 2, device=state.device
    )

    noise = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    sampled = model._sample_with_condition_cache(
        torch.zeros(1, 2, 2),
        noise,
        torch.zeros(1, 2),
        torch.tensor([0]),
        num_steps=1,
    )

    # Identity fake velocity makes one Euler step map x -> 0. More
    # importantly, the flow head received only the three action positions.
    torch.testing.assert_close(sampled, torch.zeros_like(noise))
    assert action_model.inputs.shape == (1, 4, 2)
    allowed = action_model.attention[0, 0].eq(0)
    expected = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    torch.testing.assert_close(allowed, expected)


def test_action_validity_keeps_cross_skill_tail_and_masks_only_episode_padding() -> None:
    actions = torch.tensor(
        [[[1.0, 2.0, 0.1], [3.0, 4.0, 0.2], [5.0, 6.0, 0.3]]]
    )
    batch = {
        "skill_de": torch.tensor([0]),
        "action_is_pad": torch.tensor([[False, False, True]]),
    }

    valid = SkillExpertPolicy._valid_action_steps(actions, batch)

    assert valid.tolist() == [[True, True, False]]


class _CaptureActionResidual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.seen_actions = None

    def forward(self, images, state, skill_code, actions):
        del images, state, skill_code
        self.seen_actions = actions.detach().clone()
        self._last_predicted_actions = actions.clone()
        self._last_predicted_actions[:, :2, 0] += 1.0
        return torch.tensor(
            [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [100.0, 100.0, 100.0]]],
            device=actions.device,
        )


def test_stage1_forward_matches_stage0_unconditional_action_target_and_mask() -> None:
    policy = SkillExpertPolicy.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        max_action_dim=3,
        max_state_dim=2,
        output_features={ACTION: SimpleNamespace(shape=(3,))},
        train_terminator=False,
        action_loss_mode="flow",
    )
    policy.model = _CaptureActionResidual()
    policy._collect_images = lambda batch: []
    policy._skill_code = lambda batch: torch.zeros(1, dtype=torch.long)
    policy._last_transition_jitter_fraction = torch.zeros(())
    actions = torch.tensor(
        [[[1.0, 2.0, 0.1], [3.0, 4.0, 0.2], [5.0, 6.0, 0.3]]]
    )
    batch = {
        ACTION: actions,
        OBS_STATE: torch.zeros(1, 2),
        "skill_de": torch.tensor([0]),
        "action_is_pad": torch.tensor([[False, False, True]]),
    }

    loss, metrics = policy.forward(batch)

    torch.testing.assert_close(policy.model.seen_actions, actions)
    torch.testing.assert_close(loss, torch.tensor(2.5))
    assert metrics["action_loss"] == pytest.approx(2.5)
    assert metrics["loss_per_dim"] == pytest.approx([2.5, 2.5, 2.5])


def test_endpoint_xyz_loss_matches_stage0_accumulated_delta_definition() -> None:
    predicted = torch.zeros(2, 3, 4, requires_grad=True)
    with torch.no_grad():
        predicted[0, 0, 0] = 1.0
        predicted[0, 1, 0] = -1.0
        predicted[1, 0, 0] = 1.0
        predicted[1, 1, 0] = 100.0
    target = torch.zeros_like(predicted)
    valid = torch.tensor([[True, True, True], [True, False, False]])

    loss, per_sample = SkillExpertPolicy._endpoint_xyz_loss(
        predicted, target, valid
    )
    loss.backward()

    torch.testing.assert_close(per_sample, torch.tensor([0.0, 1.0 / 3.0]))
    torch.testing.assert_close(loss.detach(), torch.tensor(1.0 / 6.0))
    assert predicted.grad is not None
    assert torch.count_nonzero(predicted.grad[1, 1:]) == 0
    assert torch.count_nonzero(predicted.grad[..., 3:]) == 0


def test_stage1_can_mix_flow_and_endpoint_xyz_half_and_half() -> None:
    policy = SkillExpertPolicy.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        max_action_dim=3,
        max_state_dim=2,
        output_features={ACTION: SimpleNamespace(shape=(3,))},
        train_terminator=False,
        action_loss_mode="flow_endpoint_xyz",
    )
    policy.model = _CaptureActionResidual()
    policy._collect_images = lambda batch: []
    policy._skill_code = lambda batch: torch.zeros(1, dtype=torch.long)
    policy._last_transition_jitter_fraction = torch.zeros(())
    batch = {
        ACTION: torch.tensor(
            [[[1.0, 2.0, 0.1], [3.0, 4.0, 0.2], [5.0, 6.0, 0.3]]]
        ),
        OBS_STATE: torch.zeros(1, 2),
        "action_is_pad": torch.tensor([[False, False, True]]),
    }

    loss, metrics = policy.forward(batch)

    assert metrics["action_loss"] == pytest.approx(2.5)
    assert metrics["endpoint_xyz_loss"] == pytest.approx(4.0 / 3.0)
    assert metrics["action_flow_weight"] == 0.5
    assert metrics["action_endpoint_weight"] == 0.5
    torch.testing.assert_close(loss, torch.tensor(23.0 / 12.0))


class _FakePredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vlm = nn.Linear(2, 2, bias=False)
        self.skill_lora = nn.Linear(2, 2, bias=False)
        self.reader = nn.Linear(2, 2, bias=False)
        self.head = nn.Linear(2, 1, bias=False)
        self.lora_layer_count = 1
        self.requires_grad_(False)

    def reader_head_parameters(self) -> list[nn.Parameter]:
        return [*self.reader.parameters(), *self.head.parameters()]

    def lora_parameters(self) -> list[nn.Parameter]:
        return list(self.skill_lora.parameters())

    def auxiliary_parameters(self) -> list[nn.Parameter]:
        return [*self.reader_head_parameters(), *self.lora_parameters()]

    def loss(self, images, language_tokens, language_mask, skill_code):
        del images, language_tokens, language_mask, skill_code
        objective = (
            self.reader.weight.square().mean()
            + self.head.weight.square().mean()
            + self.skill_lora.weight.square().mean()
        )
        return objective, 0.25

    def predict(self, images, language_tokens, language_mask):
        del images, language_tokens, language_mask
        return torch.tensor([6])


class _FakeAccelerator:
    def __init__(self) -> None:
        self.clipped_ids: set[int] = set()

    @staticmethod
    def autocast():
        return nullcontext()

    @staticmethod
    def backward(loss: torch.Tensor) -> None:
        loss.backward()

    def clip_grad_norm_(self, params, max_norm):
        params = list(params)
        self.clipped_ids = {id(parameter) for parameter in params}
        return torch.nn.utils.clip_grad_norm_(params, max_norm)


def test_skill_predictor_auxiliary_step_is_parameter_and_clip_isolated() -> None:
    policy = SkillExpertPolicy.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        train_skill_predictor=True,
        skill_vocab_size=27,
        skill_predictor_weight=0.5,
        skill_predictor_lr_scale=0.25,
        skill_predictor_lora_lr_scale=10.0,
        skill_predictor_all_layers=False,
        skill_predictor_deadzone_frac=0.8,
        optimizer_lr=1e-3,
        optimizer_betas=(0.9, 0.95),
        optimizer_eps=1e-8,
        optimizer_weight_decay=0.0,
        freeze_vision_encoder=True,
        dino_lr=None,
    )
    holder = nn.Module()
    holder.main_weight = nn.Parameter(torch.tensor([3.0]))
    holder.skill_predictor = _FakePredictor()
    policy.model = holder
    auxiliary = holder.skill_predictor.auxiliary_parameters()
    main_optimizer_ids = {
        id(parameter)
        for group in policy.get_optim_params()
        for parameter in group["params"]
    }
    assert id(holder.main_weight) in main_optimizer_ids
    assert not main_optimizer_ids & {
        id(parameter) for parameter in holder.skill_predictor.parameters()
    }
    reader_before = holder.skill_predictor.reader.weight.detach().clone()
    lora_before = holder.skill_predictor.skill_lora.weight.detach().clone()
    vlm_before = holder.skill_predictor.vlm.weight.detach().clone()
    main_before = holder.main_weight.detach().clone()
    accelerator = _FakeAccelerator()
    batch = {
        "skill_start_image": torch.zeros(2, 3, 4, 4),
        "skill_start_wrist_image": torch.zeros(2, 3, 4, 4),
        "skill_code": torch.tensor([0, 1]),
        OBS_LANGUAGE_TOKENS: torch.zeros(2, 3, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 3, dtype=torch.bool),
    }

    metrics = policy.isolated_auxiliary_step(
        batch, accelerator, grad_clip_norm=1.0, current_lr=2e-4
    )

    assert not torch.equal(holder.skill_predictor.reader.weight, reader_before)
    assert not torch.equal(holder.skill_predictor.skill_lora.weight, lora_before)
    torch.testing.assert_close(holder.skill_predictor.vlm.weight, vlm_before)
    torch.testing.assert_close(holder.main_weight, main_before)
    assert accelerator.clipped_ids == {id(parameter) for parameter in auxiliary}
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in auxiliary)
    assert metrics["skill_predictor/skill_acc"] == 0.25
    assert metrics["skill_predictor/lr"] == pytest.approx(5e-5)
    assert metrics["skill_predictor/lora_lr"] == pytest.approx(2e-3)
    assert metrics["skill_predictor/lora_layers"] == 1.0


def test_runtime_skill_prediction_uses_current_images_and_language_tokens() -> None:
    policy = SkillExpertPolicy.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    holder = nn.Module()
    holder.anchor = nn.Parameter(torch.zeros(()))
    holder.skill_predictor = _FakePredictor()
    policy.model = holder
    def collect_images(batch):
        return [batch["image"]]

    policy._collect_images = collect_images
    batch = {
        "image": torch.zeros(1, 3, 4, 4),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 3, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 3, dtype=torch.bool),
    }

    prediction = policy.predict_skill_code(batch)

    assert prediction.tolist() == [6]


class _FakeTerminatorModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.main_weight = nn.Parameter(torch.tensor([3.0]))
        self.fsq_term_train = nn.Linear(1, 2, bias=False)

    def terminator_predict(self, true_code, raw_state, image, wrist_image):
        del true_code, raw_state, image, wrist_image
        output = self.fsq_term_train(torch.ones(2, 1))
        return output[:, 0], output[:, 1]


def test_terminator_loss_optimizer_and_clipping_are_vsa_disjoint() -> None:
    policy = SkillExpertPolicy.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        train_terminator=True,
        skill_vocab_size=27,
        terminator_end_target_sigma=2.0,
        terminator_end_pos_weight=1.0,
        terminator_lr_scale=0.5,
        optimizer_lr=2e-4,
        freeze_vision_encoder=True,
        dino_lr=None,
    )
    policy.model = _FakeTerminatorModel()
    batch = {
        "skill_code_true": torch.tensor([0, 1]),
        "skill_ds": torch.tensor([0, 2]),
        "skill_de": torch.tensor([2, 0]),
        "skill_decoder_state": torch.zeros(2, 3),
        "observation.images.image": torch.zeros(2, 3, 4, 4),
        "observation.images.wrist_image": torch.zeros(2, 3, 4, 4),
    }

    term_loss, progress_loss, end_loss = policy._terminator_loss(batch)
    term_loss.backward()

    assert term_loss.detach() == progress_loss.detach() + end_loss.detach()
    assert policy.model.main_weight.grad is None
    term_parameters = list(policy.model.fsq_term_train.parameters())
    assert all(parameter.grad is not None for parameter in term_parameters)
    isolated = policy.isolated_main_optimizer_grad_groups()
    assert isolated == {"terminator": term_parameters}

    groups = policy.get_optim_params()
    term_ids = {id(parameter) for parameter in term_parameters}
    term_group = next(
        group
        for group in groups
        if {id(parameter) for parameter in group["params"]} == term_ids
    )
    assert term_group["lr"] == pytest.approx(1e-4)
    base_ids = {
        id(parameter)
        for group in groups
        if group is not term_group
        for parameter in group["params"]
    }
    assert id(policy.model.main_weight) in base_ids
    assert not base_ids & term_ids
