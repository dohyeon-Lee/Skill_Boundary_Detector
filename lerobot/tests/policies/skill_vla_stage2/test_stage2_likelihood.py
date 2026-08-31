import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding

from lerobot.policies.skill_expert.cond_gemma import CondGemmaSkillExpert
from lerobot.policies.skill_expert import modeling_skill_predictor
from lerobot.policies.skill_expert.modeling_skill_predictor import (
    FrozenVLMSkillPredictor,
)
from lerobot.policies.skill_expert.processor_skill_expert import (
    skill_expert_batch_to_transition,
    skill_expert_transition_to_batch,
)
from lerobot.policies.skill_vla_stage2.configuration_skill_vla_stage2 import (
    SkillVLAStage2Config,
)
from lerobot.policies.skill_vla_stage2.modeling_skill_vla_stage2 import (
    LikelihoodBlock,
    SkillVLAStage2Policy,
    SkillVLAStage2Pytorch,
    _load_pi05_base_vlm_parameters,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    STAGE2_VLM_CACHE_ID,
)


def _config(**overrides) -> SkillVLAStage2Config:
    values = {
        "stage1_checkpoint_path": "/tmp/stage1",
        "vlm_base_path": "/tmp/pi05_base",
        "fsq_path": "/tmp/FSQ.pt",
    }
    values.update(overrides)
    return SkillVLAStage2Config(**values)


def test_stage2_config_fixes_bayesvla_contract() -> None:
    config = _config()

    assert config.type == "skill_vla_stage2"
    assert config.architecture == "cond_gemma"
    assert config.stage2_mode == "likelihood"
    assert config.likelihood_num_layers == 4
    assert config.training_skill_source == "gt"
    assert config.train_skill_predictor
    assert not config.train_terminator
    assert (
        _config(conditioning_route="state_skill_cond").conditioning_route
        == "state_skill_cond"
    )
    assert (
        _config(conditioning_route="state_skill_only_cond").conditioning_route
        == "state_skill_only_cond"
    )
    assert (
        _config(conditioning_route="skillonly_cond").conditioning_route
        == "skillonly_cond"
    )
    assert (
        _config(conditioning_route="visiononly_cond").conditioning_route
        == "visiononly_cond"
    )
    with pytest.raises(ValueError, match="fixes likelihood_num_layers=4"):
        _config(likelihood_num_layers=3)
    with pytest.raises(ValueError, match="fixed to 'flow'"):
        _config(action_loss_mode="flow_endpoint_xyz")
    with pytest.raises(ValueError, match="complete action chunk"):
        _config(mask_actions_after_skill_end=True)
    with pytest.raises(ValueError, match="last.*layer_mix"):
        _config(likelihood_vlm_memory="every_layer")
    with pytest.raises(ValueError, match="gate_lr_scale"):
        _config(likelihood_gate_lr_scale=0.0)
    assert _config(likelihood_vlm_memory="layer_mix").likelihood_vlm_memory == "layer_mix"
    with pytest.raises(ValueError, match="gt.*predictor"):
        _config(training_skill_source="mixed")
    assert _config(skill_predictor_checkpoint_path=None).training_skill_source == "gt"
    with pytest.raises(ValueError, match="vlm_base_path"):
        _config(vlm_base_path="")
    with pytest.raises(ValueError, match="skill_predictor_checkpoint_path"):
        _config(
            training_skill_source="predictor",
            skill_predictor_checkpoint_path=None,
        )
    assert (
        _config(
            training_skill_source="predictor",
            skill_predictor_checkpoint_path="/tmp/predictor",
        ).training_skill_source
        == "predictor"
    )
    with pytest.raises(ValueError, match="train_skill_predictor=True"):
        _config(train_skill_predictor=False)
    with pytest.raises(ValueError, match="without a terminator"):
        _config(train_terminator=True)
    with pytest.raises(ValueError, match="cond_gemma"):
        _config(
            architecture="vsa_perceiver_crossattn",
            architecture_revision="interleaved_direct1024_v3",
        )
    dsbc = _config(
        stage2_mode="dsbc",
        dsbc_noise_output_mode="per_step",
        dsbc_frs_num_steps=8,
        dsbc_anchor_seed=17,
    )
    assert dsbc.dsbc_noise_output_mode == "per_step"
    assert dsbc.dsbc_noise_output_bound == pytest.approx(5.0)
    assert dsbc.dsbc_frs_num_steps == 8
    assert dsbc.dsbc_anchor_seed == 17
    with pytest.raises(ValueError, match="shared.*per_step"):
        _config(stage2_mode="dsbc", dsbc_noise_output_mode="full")
    with pytest.raises(ValueError, match="noise_output_bound"):
        _config(stage2_mode="dsbc", dsbc_noise_output_bound=0.0)
    with pytest.raises(ValueError, match="FRS noise"):
        _config(stage2_mode="dsbc", cumulative_xyz_loss_enabled=True)


def test_stage2_loads_only_base_vlm_tensors(monkeypatch) -> None:
    class TinyPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.vlm = nn.Linear(2, 3)
            self.reader = nn.Linear(3, 1)

    predictor = TinyPredictor()
    expected_weight = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    expected_bias = torch.arange(3, dtype=torch.float32)

    def fake_load(*args, **kwargs):
        assert kwargs["include_predictor_vlm"] is True
        return (
            {
                "model.skill_predictor.vlm.weight": expected_weight,
                "model.skill_predictor.vlm.bias": expected_bias,
                "model.action_out_proj.weight": torch.ones(1),
            },
            True,
        )

    monkeypatch.setattr(
        "lerobot.policies.skill_vla_stage2.modeling_skill_vla_stage2."
        "_load_pretrained_state_dict",
        fake_load,
    )
    reader_before = predictor.reader.weight.detach().clone()

    loaded = _load_pi05_base_vlm_parameters(predictor, "/tmp/pi05_base")

    assert loaded == 2
    torch.testing.assert_close(predictor.vlm.weight, expected_weight)
    torch.testing.assert_close(predictor.vlm.bias, expected_bias)
    torch.testing.assert_close(predictor.reader.weight, reader_before)


def test_external_predictor_replaces_complete_vlm_and_clears_eval_cache(
    monkeypatch,
    tmp_path,
) -> None:
    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    holder = nn.Module()
    holder.skill_predictor = nn.Linear(2, 2)
    policy.model = holder
    policy._eval_vlm_cache_ids = torch.tensor([3])
    policy._eval_vlm_cache = ([torch.ones(1, 1, 1)], torch.zeros(1, 1))
    policy.config = SimpleNamespace(
        skill_vocab_size=27,
        skill_fsq_levels=[3, 3, 3],
        skill_predictor_vlm_variant="gemma_2b",
        skill_predictor_image_size=224,
        dtype="bfloat16",
    )
    source = tmp_path / "external_predictor"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "type": "skill_aux",
                "train_skill_predictor": True,
                "skill_vocab_size": 27,
                "skill_fsq_levels": [3, 3, 3],
                "skill_predictor_vlm_variant": "gemma_2b",
                "skill_predictor_image_size": 224,
            }
        )
    )
    calls = []

    class TinyExternalPredictor(nn.Linear):
        def __init__(self, _config):
            super().__init__(2, 2)

    def load_complete(predictor, path):
        calls.append((predictor, path))
        return 17

    monkeypatch.setattr(
        "lerobot.policies.skill_vla_stage2.modeling_skill_vla_stage2."
        "_load_complete_predictor_parameters",
        load_complete,
    )
    monkeypatch.setattr(
        "lerobot.policies.skill_vla_stage2.modeling_skill_vla_stage2."
        "FrozenVLMSkillPredictor",
        TinyExternalPredictor,
    )
    monkeypatch.setattr(
        "lerobot.policies.skill_vla_stage2.modeling_skill_vla_stage2."
        "_load_learned_predictor_parameters",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("eval must load the complete predictor/VLM")
        ),
    )

    policy.load_external_skill_predictor(source)

    assert len(calls) == 1
    assert calls[0][0] is holder.skill_predictor
    assert calls[0][1] == source
    assert isinstance(holder.skill_predictor, TinyExternalPredictor)
    assert not holder.skill_predictor.training
    assert not any(
        parameter.requires_grad
        for parameter in holder.skill_predictor.parameters()
    )
    assert policy._eval_vlm_cache_ids is None
    assert policy._eval_vlm_cache is None


def test_fresh_likelihood_block_is_exact_identity() -> None:
    config = CONFIG_MAPPING["gemma"](
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        num_hidden_layers=2,
        vocab_size=16,
        use_adarms=True,
        adarms_cond_dim=32,
    )
    config._attn_implementation = "eager"  # noqa: SLF001
    block = LikelihoodBlock(config, layer_index=2)
    hidden = torch.randn(2, 5, 32)
    memory = torch.randn(2, 7, 32)
    key_padding = torch.zeros(2, 7, dtype=torch.bool)
    positions = torch.arange(5)[None].expand(2, -1)
    rotary = GemmaRotaryEmbedding(config)

    output = block(
        hidden,
        memory,
        key_padding,
        torch.randn(2, 32),
        rotary(hidden, positions),
    )

    torch.testing.assert_close(output, hidden, rtol=0.0, atol=0.0)


def test_stage2_optimizer_contains_only_injected_path_and_action_head() -> None:
    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    holder = nn.Module()
    holder.prior = nn.Linear(2, 2)
    holder.prior.requires_grad_(False)
    holder.vlm_to_expert_projection = nn.Linear(2, 2)
    holder.likelihood_blocks = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
    holder.action_out_proj = nn.Linear(2, 1)
    policy.model = holder
    policy.config = SimpleNamespace(likelihood_gate_lr_scale=1.0)

    groups = policy.get_optim_params()
    optimized = {id(parameter) for parameter in groups[0]["params"]}
    expected = {
        id(parameter)
        for module in (
            holder.vlm_to_expert_projection,
            holder.likelihood_blocks,
            holder.action_out_proj,
        )
        for parameter in module.parameters()
    }

    assert len(groups) == 1
    assert optimized == expected
    assert not optimized & {id(parameter) for parameter in holder.prior.parameters()}


def test_dsbc_freezes_stage1_head_and_optimizes_only_selector_path() -> None:
    model = SkillVLAStage2Pytorch.__new__(SkillVLAStage2Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(stage2_mode="dsbc")
    model.prior = nn.Linear(2, 2)
    model.vlm_to_expert_projection = nn.Linear(2, 2)
    model.likelihood_blocks = nn.ModuleList([nn.Linear(2, 2)])
    model.action_out_proj = nn.Linear(2, 1)
    model.noise_out_proj = nn.Linear(2, 1)
    model.likelihood_layer_mix = None

    model._freeze_stage1_prior()

    assert not any(parameter.requires_grad for parameter in model.prior.parameters())
    assert not any(
        parameter.requires_grad for parameter in model.action_out_proj.parameters()
    )
    assert all(
        parameter.requires_grad
        for module in (
            model.vlm_to_expert_projection,
            model.likelihood_blocks,
            model.noise_out_proj,
        )
        for parameter in module.parameters()
    )

    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    policy.model = model
    policy.config = SimpleNamespace(
        stage2_mode="dsbc",
        likelihood_gate_lr_scale=1.0,
    )
    optimized = {
        id(parameter)
        for parameter in policy.get_optim_params()[0]["params"]
    }
    assert optimized == {
        id(parameter)
        for module in (
            model.vlm_to_expert_projection,
            model.likelihood_blocks,
            model.noise_out_proj,
        )
        for parameter in module.parameters()
    }


def test_stage2_layer_mix_memories_start_near_the_last_layer() -> None:
    model = SkillVLAStage2Pytorch.__new__(SkillVLAStage2Pytorch)
    nn.Module.__init__(model)
    model.likelihood_blocks = nn.ModuleList([nn.Identity(), nn.Identity()])
    model.vlm_to_expert_projection = nn.Identity()
    model.vlm_to_expert_projection.weight = nn.Parameter(torch.zeros(1))
    mix = torch.zeros(2, 3)
    mix[:, -1] = 5.0
    model.likelihood_layer_mix = nn.Parameter(mix)
    stack = torch.stack(
        [torch.full((1, 4, 2), float(layer)) for layer in (1.0, 2.0, 3.0)], dim=1
    )

    memories = model._likelihood_memories(stack)

    assert len(memories) == 2
    weights = torch.softmax(model.likelihood_layer_mix, dim=-1)
    assert weights[0, -1].item() > 0.97
    expected = (weights[0, :, None, None] * stack[0]).sum(dim=0)
    torch.testing.assert_close(memories[0][0], expected)

    # Without the mix, every block shares one projected last-hidden memory.
    model.likelihood_layer_mix = None
    shared = model._likelihood_memories(torch.randn(1, 4, 2))
    assert len(shared) == 2 and shared[0] is shared[1]


def test_stage2_gate_lr_scale_splits_bootstrap_parameters() -> None:
    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    holder = nn.Module()
    holder.vlm_to_expert_projection = nn.Linear(2, 2)
    holder.likelihood_blocks = nn.ModuleList([nn.Linear(2, 2)])
    holder.action_out_proj = nn.Linear(2, 1)
    policy.model = holder
    policy.config = SimpleNamespace(
        likelihood_gate_lr_scale=10.0, optimizer_lr=2.5e-5
    )

    groups = policy.get_optim_params()

    assert len(groups) == 2
    assert groups[1]["lr"] == pytest.approx(2.5e-4)
    boosted = {id(parameter) for parameter in groups[1]["params"]}
    assert boosted == {
        id(parameter)
        for parameter in holder.vlm_to_expert_projection.parameters()
    }


def test_stage2_freeze_contract_violation_is_detected() -> None:
    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    holder = nn.Module()
    holder.prior = nn.Linear(2, 2)  # stays trainable -> contract violation
    holder.vlm_to_expert_projection = nn.Linear(2, 2)
    holder.likelihood_blocks = nn.ModuleList([nn.Linear(2, 2)])
    holder.action_out_proj = nn.Linear(2, 1)
    policy.model = holder

    with pytest.raises(RuntimeError, match="freeze contract"):
        policy.get_optim_params()


def test_stage2_has_no_auxiliary_training_step() -> None:
    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)

    assert policy.isolated_auxiliary_step({}, None, 1.0) == {}


def test_stage2_forward_preserves_inherited_conditioning_route_in_frozen_prior() -> None:
    captured = {}

    class _Predictor:
        @staticmethod
        def encode_base_last_hidden(images, language_tokens, language_mask):
            del language_tokens, language_mask
            captured["vlm_images"] = images
            return torch.zeros(1, 2, 2), torch.zeros(1, 2, dtype=torch.bool)

    model = SkillVLAStage2Pytorch.__new__(SkillVLAStage2Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(cumulative_xyz_loss_enabled=True)
    model.uses_expert_context_tokens = False
    model.uses_cond_state_adarms = True
    model.likelihood_layer_mix = None
    condition_state = torch.tensor([[9.0, 8.0]])
    condition_skill = torch.tensor([[7.0, 6.0]])
    expert_time = torch.tensor([[5.0, 4.0]])
    model.skill_predictor = _Predictor()
    model._likelihood_memories = lambda hidden: [hidden]

    def condition_tokens(images, batch_size=None):
        del batch_size
        captured["vsa_images"] = images
        return torch.zeros(1, 2, 2)

    model._condition_tokens = condition_tokens
    model._project_state = lambda state: condition_state
    model._project_expert_state = lambda state, shared: None
    model._expert_condition = lambda time, state=None, skill=None: expert_time
    model._skill_broadcasts = lambda code: (condition_skill, None)
    model._condition_state_start_index = lambda tokens: None

    def run_prior(
        condition,
        actions,
        routed_state,
        expert_condition,
        routed_condition_skill,
        expert_skill,
        condition_state_start_index=None,
    ):
        del condition, condition_state_start_index
        captured["state"] = routed_state
        captured["expert_condition"] = expert_condition
        captured["condition_skill"] = routed_condition_skill
        captured["expert_skill"] = expert_skill
        return actions

    model._run_joint_hidden = run_prior
    model._likelihood_velocity = (
        lambda prior, vlm, padding, expert: torch.zeros_like(prior)
    )
    actions = torch.ones(1, 3, 2)
    residual = model.forward(
        [torch.tensor([1.0])],
        [torch.tensor([2.0])],
        torch.zeros(1, 2),
        torch.tensor([0]),
        actions,
        torch.zeros(1, 1, dtype=torch.long),
        torch.ones(1, 1, dtype=torch.bool),
        noise=torch.zeros_like(actions),
        time=torch.full((1,), 0.5),
    )

    assert captured["vsa_images"][0].item() == 1.0
    assert captured["vlm_images"][0].item() == 2.0
    assert captured["state"] is condition_state
    assert captured["expert_condition"] is expert_time
    assert captured["condition_skill"] is condition_skill
    assert captured["expert_skill"] is None
    # likelihood receives only K action positions from the frozen prior.
    assert residual.shape == actions.shape
    torch.testing.assert_close(
        model._last_predicted_actions,
        torch.full_like(actions, 0.5),
    )


def test_stage2_memory_disables_skill_lora_but_predictor_memory_keeps_it(
    monkeypatch,
) -> None:
    predictor = FrozenVLMSkillPredictor.__new__(FrozenVLMSkillPredictor)
    nn.Module.__init__(predictor)
    predictor._embed_prefix = lambda images, tokens, mask: (
        torch.zeros(1, 2, 3),
        torch.ones(1, 2, dtype=torch.bool),
        torch.zeros(1, 2, dtype=torch.bool),
    )
    predictor._encode_prefix = lambda prefix, valid, all_layers: (prefix + 1.0, None)
    selected = []
    monkeypatch.setattr(
        modeling_skill_predictor,
        "set_active_adapters",
        lambda names: selected.append(set(names)),
    )
    tokens = torch.zeros(1, 1, dtype=torch.long)
    mask = torch.ones(1, 1, dtype=torch.bool)

    base_hidden, base_padding = predictor.encode_base_last_hidden([], tokens, mask)
    adapted_hidden, adapted_padding = predictor.encode_last_hidden([], tokens, mask)

    assert selected == [set(), {"skill"}]
    torch.testing.assert_close(base_hidden, adapted_hidden)
    torch.testing.assert_close(base_padding, adapted_padding)


@pytest.mark.parametrize("output_mode", ["shared", "per_step"])
def test_dsbc_selector_uses_fixed_t1_anchor_and_configurable_noise_shape(
    output_mode: str,
) -> None:
    model = SkillVLAStage2Pytorch.__new__(SkillVLAStage2Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        stage2_mode="dsbc",
        dsbc_noise_output_mode=output_mode,
    )
    model.action_in_proj = nn.Linear(2, 2, bias=False)
    model.noise_out_proj = nn.Identity()
    model.dsbc_anchor_noise = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]
    )
    hidden = torch.tensor(
        [[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]]
    )
    captured = {}
    model._condition_tokens = lambda images, batch_size=None: torch.zeros(1, 1, 2)

    def prior(condition, anchor, state, skill, time):
        del condition, state, skill
        captured["anchor"] = anchor.detach().clone()
        captured["time"] = time.detach().clone()
        return hidden, torch.zeros(1, 2)

    def vlm(start_images, tokens, mask):
        del tokens, mask
        captured["start_images"] = start_images
        return torch.zeros(1, 1, 2), torch.zeros(1, 1, dtype=torch.bool)

    model._prior_action_hidden = prior
    model._encode_likelihood_memory = vlm
    model._likelihood_memories = lambda vlm_hidden: [vlm_hidden]
    model._run_likelihood_blocks = (
        lambda prior_hidden, memories, padding, condition: prior_hidden
    )
    start_images = [torch.tensor([9.0])]

    prediction = model._dsbc_noise_prediction(
        [torch.tensor([8.0])],
        start_images,
        torch.zeros(1, 2),
        torch.tensor([7]),
        torch.zeros(1, 1, dtype=torch.long),
        torch.ones(1, 1, dtype=torch.bool),
    )

    torch.testing.assert_close(captured["anchor"], model.dsbc_anchor_noise)
    torch.testing.assert_close(captured["time"], torch.ones(1))
    assert captured["start_images"] is start_images
    expected = hidden.mean(dim=1) if output_mode == "shared" else hidden
    torch.testing.assert_close(prediction, 5.0 * torch.tanh(expected))
    assert prediction.abs().max() <= 5.0


def test_online_frs_runs_action_to_noise_and_forces_linear_padding_path() -> None:
    model = SkillVLAStage2Pytorch.__new__(SkillVLAStage2Pytorch)
    nn.Module.__init__(model)
    model.real_action_dim = 1
    model.action_in_proj = nn.Linear(2, 2, bias=False)
    model.action_out_proj = nn.Identity()
    model._visual_context_cache = lambda condition, context, state: object()
    times = []
    noisy_states = []

    def expert_condition(time):
        times.append(time.detach().clone())
        return torch.zeros(time.shape[0], 2)

    def action_hidden(noisy, condition, skill, cache, attention, positions):
        del condition, skill, cache, attention, positions
        noisy_states.append(noisy.detach().clone())
        return noisy

    model._expert_condition = expert_condition
    model._action_hidden_with_condition_cache = action_hidden
    target = model._frs_reverse_with_expert_context_cache(
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 2),
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[2.0]]]),
        num_steps=2,
        condition_state=None,
    )

    torch.testing.assert_close(target, torch.tensor([[[2.25]]]))
    torch.testing.assert_close(torch.stack(times).flatten(), torch.tensor([0.0, 0.5]))
    torch.testing.assert_close(noisy_states[0], torch.tensor([[[1.0, 0.0]]]))
    torch.testing.assert_close(noisy_states[1], torch.tensor([[[1.5, 1.0]]]))


def test_dsbc_action_metric_is_detached_legacy_flow_residual() -> None:
    model = SkillVLAStage2Pytorch.__new__(SkillVLAStage2Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(dsbc_noise_output_mode="shared")
    model.real_action_dim = 1
    model.action_in_proj = nn.Linear(2, 2, bias=False)
    model.action_out_proj = nn.Identity()
    model.sample_time = lambda batch_size, device: torch.full(
        (batch_size,), 0.5, device=device
    )

    def prior(condition, x_t, state, skill, time):
        del condition, state, skill, time
        return x_t, torch.zeros(x_t.shape[0], 2)

    model._prior_action_hidden = prior
    residual = model._dsbc_action_flow_residual(
        torch.zeros(1, 1, 2),
        torch.zeros(1, 2),
        torch.tensor([7]),
        torch.tensor([[[1.0, 0.0], [3.0, 0.0]]]),
        torch.tensor([[5.0]], requires_grad=True),
        torch.zeros(1, 2, 1),
    )

    torch.testing.assert_close(residual, torch.tensor([[[1.0], [-2.0]]]))
    assert not residual.requires_grad


class _CaptureStage2Residual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.seen_actions = None
        self.seen_images = None
        self.seen_vlm_start_images = None

    def forward(
        self,
        images,
        vlm_start_images,
        state,
        skill_code,
        actions,
        language_tokens,
        language_mask,
    ):
        del state, skill_code, language_tokens, language_mask
        self.seen_images = images
        self.seen_vlm_start_images = vlm_start_images
        self.seen_actions = actions.detach().clone()
        self._last_predicted_actions = actions.clone()
        self._last_predicted_actions[:, :2, 0] += 1.0
        return torch.tensor(
            [[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [100.0, 100.0, 100.0]]],
            device=actions.device,
        )


def test_stage2_mixes_flow_and_cumulative_xyz_like_stage1() -> None:
    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        max_action_dim=3,
        max_state_dim=2,
        output_features={ACTION: SimpleNamespace(shape=(3,))},
        conditioning_route="state_cond",
        cumulative_xyz_loss_enabled=True,
        cumulative_xyz_loss_weight=0.5,
        training_skill_source="gt",
    )
    policy.model = _CaptureStage2Residual()
    current_images = [torch.tensor([1.0])]
    start_images = [torch.tensor([2.0])]
    policy._collect_images = lambda batch: current_images
    policy._predictor_start_images = lambda batch: start_images
    policy._training_skill_code = lambda batch: torch.zeros(1, dtype=torch.long)
    policy._last_transition_jitter_fraction = torch.zeros(())
    actions = torch.tensor(
        [[[1.0, 2.0, 0.1], [3.0, 4.0, 0.2], [5.0, 6.0, 0.3]]]
    )
    batch = {
        ACTION: actions,
        OBS_STATE: torch.zeros(1, 2),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 1, dtype=torch.bool),
        "action_is_pad": torch.tensor([[False, False, True]]),
    }

    loss, metrics = policy.forward(batch)

    assert policy.model.seen_images is current_images
    assert policy.model.seen_vlm_start_images is start_images
    torch.testing.assert_close(policy.model.seen_actions, actions)
    # The fake model adds +1.0 XYZ error to the first two (valid) steps, so the
    # masked prefix cumulative squared error is [1/3, 4/3] -> raw 5/6, and the
    # (valid_steps + 1) / 2 = 1.5 horizon normalization gives 5/9.
    assert metrics["action_loss"] == pytest.approx(2.5)
    assert metrics["cumulative_xyz/raw"] == pytest.approx(5.0 / 6.0)
    assert metrics["cumulative_xyz/normalized"] == pytest.approx(5.0 / 9.0)
    assert metrics["cumulative_xyz/weighted"] == pytest.approx(5.0 / 18.0)
    assert metrics["action_flow_weight"] == 1.0
    assert metrics["action_cumulative_xyz_weight"] == 0.5
    torch.testing.assert_close(loss, torch.tensor(2.5 + 5.0 / 18.0))


class _CaptureDSBCPair(nn.Module):
    def __init__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        action_residual: torch.Tensor,
    ) -> None:
        super().__init__()
        self.real_action_dim = target.shape[-1]
        self.prediction = prediction
        self.target = target
        self.action_residual = action_residual
        self.seen = None

    def dsbc_training_pair(
        self,
        images,
        start_images,
        state,
        skill,
        actions,
        tokens,
        mask,
    ):
        del state, actions, tokens, mask
        self.seen = (images, start_images, skill.detach().clone())
        return self.prediction, self.target, self.action_residual


def _dsbc_policy(
    output_mode: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    action_residual: torch.Tensor | None = None,
):
    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        stage2_mode="dsbc",
        max_action_dim=3,
        max_state_dim=2,
        output_features={ACTION: SimpleNamespace(shape=(2,))},
        conditioning_route="state_skill_cond",
        training_skill_source="gt",
        dsbc_noise_output_mode=output_mode,
        dsbc_frs_num_steps=10,
        same_skill_batch_enabled=False,
        mask_actions_after_skill_end=False,
    )
    if action_residual is None:
        action_residual = torch.zeros_like(target)
    policy.model = _CaptureDSBCPair(prediction, target, action_residual)
    current_images = [torch.tensor([1.0])]
    start_images = [torch.tensor([2.0])]
    policy._collect_images = lambda batch: current_images
    policy._predictor_start_images = lambda batch: start_images
    policy._training_skill_code = lambda batch: torch.tensor([7])
    policy._last_transition_jitter_fraction = torch.ones(())
    return policy, current_images, start_images


def test_dsbc_shared_mode_supervises_mean_valid_frs_noise() -> None:
    target = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
    prediction = torch.tensor([[1.0, 2.0]], requires_grad=True)
    policy, current_images, start_images = _dsbc_policy(
        "shared", prediction, target
    )
    batch = {
        ACTION: torch.zeros(1, 3, 2),
        OBS_STATE: torch.zeros(1, 2),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 1, dtype=torch.bool),
        "action_is_pad": torch.tensor([[False, False, True]]),
    }

    loss, metrics = policy.forward(batch)

    torch.testing.assert_close(loss, torch.tensor(2.5))
    assert metrics["action_loss"] == pytest.approx(0.0)
    assert metrics["gt_noise_loss"] == pytest.approx(2.5)
    assert metrics["noise_loss"] == pytest.approx(2.5)
    assert metrics["dsbc/supervision_target_rms"] == pytest.approx(10.0**0.5)
    assert metrics["dsbc/frs_target_valid_rms"] == pytest.approx(11.0**0.5)
    assert metrics["dsbc/supervision_target_abs_max"] == pytest.approx(4.0)
    assert metrics["dsbc/frs_target_abs_max"] == pytest.approx(5.0)
    assert metrics["dsbc/supervision_target_outside_bound_fraction"] == 0.0
    assert metrics["dsbc/frs_target_outside_bound_fraction"] == 0.0
    assert metrics["dsbc/noise_output_bound"] == pytest.approx(5.0)
    assert metrics["regime/transition_jitter_fraction"] == pytest.approx(1.0)
    assert policy.model.seen[0] is current_images
    assert policy.model.seen[1] is start_images
    torch.testing.assert_close(policy.model.seen[2], torch.tensor([7]))


def test_dsbc_per_step_mode_masks_padded_frs_targets() -> None:
    target = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
    prediction = (target + 1.0).requires_grad_()
    action_residual = torch.tensor(
        [[[1.0, 1.0], [2.0, 2.0], [100.0, 100.0]]]
    )
    policy, _, _ = _dsbc_policy(
        "per_step", prediction, target, action_residual
    )
    batch = {
        ACTION: torch.zeros(1, 3, 2),
        OBS_STATE: torch.zeros(1, 2),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 1, dtype=torch.bool),
        "action_is_pad": torch.tensor([[False, False, True]]),
    }

    loss, metrics = policy.forward(batch)

    torch.testing.assert_close(loss, torch.tensor(1.0))
    assert metrics["action_loss"] == pytest.approx(2.5)
    assert metrics["gt_noise_loss"] == pytest.approx(1.0)
    assert metrics["noise_loss"] == pytest.approx(1.0)


def test_stage2_same_skill_metrics_use_post_jitter_code_and_different_task() -> None:
    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(same_skill_batch_enabled=True)
    batch = {
        "same_skill_pair_id": torch.tensor([0, 0, -1, -1]),
        "same_skill_pair_fallback": torch.tensor([False, False, False, False]),
        "skill_code": torch.tensor([7, 7, 3, 4]),
        "task_index": torch.tensor([1, 2, 3, 4]),
        "skill_progress": torch.tensor([0.25, 0.30, 0.4, 0.8]),
    }

    metrics = policy._same_skill_batch_metrics(
        batch, torch.tensor([7, 7, 3, 4])
    )

    assert metrics["batch_sampling/constructed_fraction"] == pytest.approx(0.5)
    assert metrics["batch_sampling/effective_after_jitter_fraction"] == pytest.approx(0.5)
    assert metrics["batch_sampling/effective_conditioning_fraction"] == pytest.approx(0.5)
    assert metrics["batch_sampling/jittered_progress_gap"] == pytest.approx(0.05)


def test_stage2_processor_preserves_same_skill_sampler_metadata() -> None:
    batch = {
        "same_skill_pair_id": torch.tensor([0, 0, -1, -1]),
        "same_skill_pair_fallback": torch.tensor([False, False, False, True]),
        "skill_progress": torch.tensor([0.1, 0.2, 0.3, 0.4]),
        "skill_effective_de": torch.tensor([4, 3, 2, 1]),
    }

    restored = skill_expert_transition_to_batch(
        skill_expert_batch_to_transition(batch)
    )

    for key, expected in batch.items():
        torch.testing.assert_close(restored[key], expected)


def test_stage2_sampling_keeps_condition_routing_and_likelihood_sees_actions_only() -> None:
    class _ConditionModel:
        @staticmethod
        def forward(**kwargs):
            del kwargs
            return SimpleNamespace(past_key_values=[("k", "v")])

    class _Predictor:
        @staticmethod
        def encode_base_last_hidden(images, language_tokens, language_mask):
            del images, language_tokens, language_mask
            return torch.zeros(1, 2, 2), torch.zeros(1, 2, dtype=torch.bool)

    model = SkillVLAStage2Pytorch.__new__(SkillVLAStage2Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_inference_steps=1,
        chunk_size=3,
        max_action_dim=2,
    )
    model.uses_expert_context_tokens = False
    model.uses_cond_state_adarms = True
    model.likelihood_layer_mix = None
    model.cond_encoder = SimpleNamespace(model=_ConditionModel())
    model.skill_predictor = _Predictor()
    model._likelihood_memories = lambda hidden: [hidden]
    model._condition_tokens = lambda images, batch_size=None: torch.zeros(1, 2, 2)
    model._project_state = lambda state: torch.zeros(1, 2)
    model._project_expert_state = lambda state, shared: None
    model._expert_condition = lambda time, state=None, skill=None: torch.zeros(1, 2)
    model._skill_broadcasts = lambda code: (torch.tensor([[7.0, 8.0]]), None)
    model._condition_state_start_index = lambda tokens: None
    captured = {}

    def action_prior(
        noisy_actions,
        expert_condition,
        expert_skill,
        condition_cache,
        attention_mask,
        position_ids,
    ):
        del expert_condition, condition_cache, position_ids
        captured["expert_skill"] = expert_skill
        captured["attention"] = attention_mask
        return noisy_actions

    def likelihood(prior, vlm, padding, expert):
        del vlm, padding, expert
        captured["likelihood_shape"] = prior.shape
        return prior

    model._action_hidden_with_condition_cache = action_prior
    model._likelihood_velocity = likelihood
    noise = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    cached_vlm = (
        [torch.zeros(1, 2, 2)],
        torch.zeros(1, 2, dtype=torch.bool),
    )
    sampled = model.sample_actions(
        [],
        [],
        torch.zeros(1, 2),
        torch.tensor([0]),
        torch.zeros(1, 1, dtype=torch.long),
        torch.ones(1, 1, dtype=torch.bool),
        noise=noise,
        num_steps=1,
        vlm_memory=cached_vlm,
    )

    torch.testing.assert_close(sampled, torch.zeros_like(noise))
    assert captured["expert_skill"] is None
    assert captured["likelihood_shape"] == noise.shape
    allowed = captured["attention"][0, 0].eq(0)
    expected = torch.ones(3, 5, dtype=torch.bool)
    torch.testing.assert_close(allowed, expected)


@pytest.mark.parametrize(
    ("output_mode", "prediction"),
    [
        ("shared", torch.tensor([[1.0, 2.0]])),
        (
            "per_step",
            torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),
        ),
    ],
)
def test_dsbc_sampling_uses_selected_real_noise_and_preserves_gaussian_padding(
    monkeypatch,
    output_mode: str,
    prediction: torch.Tensor,
) -> None:
    model = SkillVLAStage2Pytorch.__new__(SkillVLAStage2Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        stage2_mode="dsbc",
        dsbc_noise_output_mode=output_mode,
        chunk_size=3,
        max_action_dim=4,
    )
    model.real_action_dim = 2
    captured = {}

    def predict_noise(*args, **kwargs):
        del args
        captured["vlm_memory"] = kwargs["vlm_memory"]
        return prediction

    model._dsbc_noise_prediction = predict_noise

    def base_sample(self, images, state, skill, noise=None, num_steps=None):
        del self, images, state, skill, num_steps
        captured["noise"] = noise.detach().clone()
        return noise

    monkeypatch.setattr(CondGemmaSkillExpert, "sample_actions", base_sample)
    reservoir = torch.tensor(
        [[[9.0, 9.0, 3.0, 4.0], [9.0, 9.0, 5.0, 6.0], [9.0, 9.0, 7.0, 8.0]]]
    )
    cached_vlm = (
        [torch.zeros(1, 2, 2)],
        torch.zeros(1, 2, dtype=torch.bool),
    )

    sampled = model.sample_actions(
        [],
        [],
        torch.zeros(1, 2),
        torch.tensor([7]),
        torch.zeros(1, 1, dtype=torch.long),
        torch.ones(1, 1, dtype=torch.bool),
        noise=reservoir,
        num_steps=10,
        vlm_memory=cached_vlm,
    )

    expected = reservoir.clone()
    expected[..., :2] = prediction[:, None] if output_mode == "shared" else prediction
    torch.testing.assert_close(captured["noise"], expected)
    torch.testing.assert_close(sampled, expected)
    assert captured["vlm_memory"] is cached_vlm


def test_eval_vlm_cache_refreshes_only_rows_whose_skill_generation_changes() -> None:
    class _CacheModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.likelihood_blocks = nn.ModuleList([nn.Identity(), nn.Identity()])
            self.calls = []

        def encode_likelihood_memories(self, images, tokens, mask):
            del tokens
            values = images[0].flatten(1).mean(dim=1)[:, None, None]
            self.calls.append(values.detach().clone())
            return [values, values + 100.0], mask.detach().clone()

    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(n_action_steps=2)
    policy.model = _CacheModel()
    policy.reset()
    tokens = torch.zeros(2, 1, dtype=torch.long)

    first = policy._cached_eval_vlm_memory(
        [torch.tensor([[[[1.0]]], [[[2.0]]]])],
        tokens,
        torch.ones(2, 1, dtype=torch.bool),
        torch.tensor([0, 0]),
    )
    second = policy._cached_eval_vlm_memory(
        [torch.tensor([[[[10.0]]], [[[20.0]]]])],
        tokens,
        torch.zeros(2, 1, dtype=torch.bool),
        torch.tensor([0, 0]),
    )
    third = policy._cached_eval_vlm_memory(
        [torch.tensor([[[[10.0]]], [[[20.0]]]])],
        tokens,
        torch.tensor([[False], [True]]),
        torch.tensor([0, 1]),
    )

    assert first is second is third
    assert [call.flatten().tolist() for call in policy.model.calls] == [
        [1.0, 2.0],
        [20.0],
    ]
    torch.testing.assert_close(third[0][0].flatten(), torch.tensor([1.0, 20.0]))
    torch.testing.assert_close(third[0][1].flatten(), torch.tensor([101.0, 120.0]))
    torch.testing.assert_close(
        third[1], torch.tensor([[True], [True]], dtype=torch.bool)
    )

    policy.reset()
    assert policy._eval_vlm_cache is None
    assert policy._eval_vlm_cache_ids is None


def test_eval_vlm_cache_is_opt_in_for_direct_policy_callers() -> None:
    policy = SkillVLAStage2Policy.__new__(SkillVLAStage2Policy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(n_action_steps=2)
    policy.model = nn.Module()
    policy.reset()

    assert (
        policy._cached_eval_vlm_memory(
            [],
            torch.zeros(1, 1, dtype=torch.long),
            torch.ones(1, 1, dtype=torch.bool),
            None,
        )
        is None
    )
    assert STAGE2_VLM_CACHE_ID == "stage2_vlm_cache_id"
