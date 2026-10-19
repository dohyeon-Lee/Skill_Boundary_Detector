from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from lerobot.policies.skill_expert import fullskill_bspline, fullskill_vision, modeling_utils
from lerobot.policies.skill_expert.configuration_skill_expert import BSPLINE_REVISIONS, BSPLINE_LEGACY_REVISIONS
from lerobot.policies.skill_expert.fullskill_bspline import FullskillBsplineSkillExpert
from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy, _allowed_pi05_missing_key
from lerobot.policies.skillVLA.dataset_skillVLA import SkillVLADataset
from lerobot.policies.skillVLA.proprio_spline import resample_skill_proprio
from test_stage1_vsa import _skill_config


@pytest.mark.parametrize("label", BSPLINE_REVISIONS)
@pytest.mark.parametrize("bridge_layers", [1, 9, 17])
def test_config_roundtrip_and_base_weight_loading(tmp_path, label, bridge_layers):
    from lerobot.configs.policies import PreTrainedConfig
    config = _skill_config(label, visual_bridge_last_n_layers=bridge_layers)
    config.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.uses_bspline_plan and restored.uses_symmetric_start_vision
    assert restored.visual_bridge_last_n_layers == bridge_layers
    assert not restored.skill_flow_enabled
    assert restored.bspline_tokens == 30
    for key in ("bspline_queries", "bspline_reader.q_proj_weight", "start_bridge_gate",
                "start_align_bridge_gate", "cond_skill_condition.0.weight", "start_pose_condition.0.weight"):
        assert _allowed_pi05_missing_key("model." + key, restored)
    if label.startswith(("fullskill_bspline_2_", "fullskill_bspline_3_", "fullskill_bspline_6_")):
        assert _allowed_pi05_missing_key("model.bridge_proprio_condition.0.weight", restored)


@pytest.mark.parametrize("label", BSPLINE_LEGACY_REVISIONS)
def test_legacy_bspline_checkpoints_keep_original_9_9_split(label):
    config = _skill_config(label, architecture_revision=BSPLINE_LEGACY_REVISIONS[label])
    assert config.visual_bridge_last_n_layers == 9
    with pytest.raises(ValueError, match="Legacy Bspline"):
        _skill_config(label, architecture_revision=BSPLINE_LEGACY_REVISIONS[label], visual_bridge_last_n_layers=1)


@pytest.mark.parametrize("bridge_layers", [0, 18, 19])
def test_bspline_rejects_empty_prefix_or_late_route(bridge_layers):
    with pytest.raises(ValueError, match="visual_bridge_last_n_layers"):
        _skill_config("fullskill_bspline_2_firstlayer", visual_bridge_last_n_layers=bridge_layers)


def test_preprocessor_preserves_target_and_normalizes_start_state():
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.skill_expert.processor_skill_expert import make_skill_expert_pre_post_processors
    config = _skill_config(
        "fullskill_bspline_2_firstlayer", device="cpu",
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    stats = {"observation.state": {"q01": torch.zeros(8), "q99": torch.full((8,), 2.)},
             "action": {"q01": -torch.ones(7), "q99": torch.ones(7)}}
    pre, _ = make_skill_expert_pre_post_processors(config, stats)
    batch = pre({"observation.state": torch.ones(2, 8), "action": torch.zeros(2, 10, 7),
                 "skill_start_state": torch.ones(2, 8), "skill_bspline_states": torch.ones(2, 30, 8)})
    torch.testing.assert_close(batch["skill_start_state_normalized"], torch.zeros(2, 8), atol=1e-6, rtol=0)
    torch.testing.assert_close(batch["skill_bspline_states"], torch.ones(2, 30, 8))


def tiny_model(monkeypatch, variant, backend="sdpa", checkpoint=False, bridge_layers=9,
               start_position="firstlayer"):
    monkeypatch.setattr(modeling_utils, "get_gemma_config", lambda variant: SimpleNamespace(
        width=32, depth=18, mlp_dim=64, num_heads=4, num_kv_heads=2, head_dim=8,
    ))

    def initialize(self, config):
        nn.Module.__init__(self)
        self.config, self.width = config, 32
        self.gemma_expert = modeling_utils.build_gemma("gemma_300m", use_adarms=True, attention_backend=backend)
        # Exercise attention/conditioning gradients, not only zero-init residual bypasses.
        with torch.no_grad():
            for layer in self.gemma_expert.model.layers:
                for norm in (layer.input_layernorm, layer.post_attention_layernorm):
                    nn.init.normal_(norm.dense.weight, std=.02)
                    norm.dense.bias[-32:].fill_(.5)
        self.visual_bridge_query_norm = nn.LayerNorm(32)
        self.visual_bridge_attention = nn.MultiheadAttention(32, 4, kdim=16, vdim=16, batch_first=True)
        self.visual_bridge_gates = nn.Parameter(torch.full((18,), .1))
        self.visual_align_bridge_attention = nn.MultiheadAttention(32, 4, kdim=16, vdim=16, batch_first=True)
        self.visual_align_bridge_gates = nn.Parameter(torch.full((18,), .1))
        self._gradient_checkpointing = checkpoint
        self._capture_visual_bridge_attention = False
        self._vsa_debug_active = False
        self._action_tokens = lambda x: x

    monkeypatch.setattr(fullskill_vision.BothLIT7SkillExpert, "__init__", initialize)
    config = SimpleNamespace(
        architecture_label=f"fullskill_bspline_{variant}_{start_position}", max_state_dim=32,
        visual_bridge_last_n_layers=bridge_layers, visual_bridge_heads=4, visual_bottleneck_width=16,
        visual_bridge_gate_init=.1, uses_symmetric_start_vision=True,
        bspline_tokens=30, skill_flow_checkpoint_layers=9,
    )
    model = FullskillBsplineSkillExpert(config)
    if model.direct_current_proprio:
        nn.init.normal_(model.bridge_proprio_condition[-1].weight, std=.03)
        model._bridge_condition_shift = model.bridge_proprio_condition(torch.randn(2, 32))
    model._bspline_time_condition = torch.randn(2, 32)
    return model


@pytest.mark.parametrize("variant", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("backend,dtype", [("eager", torch.float32), ("sdpa", torch.bfloat16)])
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("bridge_layers", [1, 9, 17])
@pytest.mark.parametrize("start_position", ["firstlayer", "lastlayer"])
def test_real_tiny_gemma_plan_routing_and_attached_gradients(
    monkeypatch, variant, backend, dtype, checkpoint, bridge_layers, start_position,
):
    torch.manual_seed(10)
    model = tiny_model(monkeypatch, variant, backend, checkpoint, bridge_layers, start_position)
    split = 18 - bridge_layers
    assert model.split_layer == split
    start_layer = split - 1 if start_position == "lastlayer" else 0
    assert model.start_bridge_layer == start_layer
    model.gemma_expert.to(dtype=dtype)
    start = torch.randn(2, 100, 16, requires_grad=True)
    latents = [torch.randn(2, 100, 16, requires_grad=True) for _ in range(18)]
    model.set_skill_start_latent(start)
    noise = torch.randn(2, 10, 32, dtype=dtype, requires_grad=True)
    condition = torch.randn(2, 32, dtype=dtype, requires_grad=True)
    skill = torch.randn(2, 32, dtype=dtype, requires_grad=True)
    calls, plan_calls = [], []
    original = model._expert_layer_with_latent_bridge
    def record(index, hidden, mask, pos, cond, code, latent, rotary):
        calls.append((index, code is not None, latent, cond.detach().clone()))
        return original(index, hidden, mask, pos, cond, code, latent, rotary)
    model._expert_layer_with_latent_bridge = record
    hook = model.bspline_generator.register_forward_hook(lambda *args: plan_calls.append(len(calls)))
    output = model._run_expert_with_layerwise_latents(noise, condition, skill, latents)
    prediction = model.take_bspline_prediction()
    hook.remove()
    assert output.shape == noise.shape
    assert prediction.shape == (2, 30, 8)
    assert plan_calls == [split]
    assert [index for index, _, memory, _ in calls if memory is not None] == [start_layer, *range(split, 18)]
    assert calls[start_layer][2] is start
    assert all(calls[i][2] is latents[i] for i in range(split, 18))
    assert all(has_skill == (i < split or variant == 2) for i, has_skill, _, _ in calls)
    for i in range(split, 18):
        expected = condition if variant == 2 else model._bspline_time_condition.to(condition.dtype)
        if variant in (2, 3, 6):
            expected = expected + model._bridge_condition_shift.to(expected.dtype)
        torch.testing.assert_close(calls[i][3], expected)
    # Reading hidden, not decoded proprio: action gradients reach queries but not the aux head.
    query_grad, head_grad = torch.autograd.grad(
        output.float().square().mean(), (model.bspline_queries, model.bspline_proprio_head[-1].weight),
        retain_graph=True, allow_unused=True,
    )
    assert query_grad.abs().sum() > 0 and head_grad is None
    # Later inference/debug caches must not change checkpoint recomputation.
    model.set_skill_start_latent(None)
    model._bridge_condition_shift, model._bspline_time_condition = None, None
    (output.float().square().mean() + prediction.square().mean()).backward()
    assert model.bspline_proprio_head[-1].weight.grad.abs().sum() > 0
    assert model.gemma_expert.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert start.grad.abs().sum() > 0
    assert all(latents[i].grad is None for i in range(split))
    assert all(latents[i].grad.abs().sum() > 0 for i in range(split, 18))


@pytest.mark.parametrize("variant", [2, 3, 4, 5, 6])
def test_first_and_last_positions_preserve_parameter_geometry_and_initialization(monkeypatch, variant):
    torch.manual_seed(123)
    first = tiny_model(monkeypatch, variant)
    torch.manual_seed(123)
    last = tiny_model(monkeypatch, variant, start_position="lastlayer")
    assert first.start_bridge_layer == 0 and last.start_bridge_layer == 8
    assert first.state_dict().keys() == last.state_dict().keys()
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, last.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="architecture_revision"):
        _skill_config(
            f"fullskill_bspline_{variant}_lastlayer",
            architecture_revision=BSPLINE_REVISIONS[f"fullskill_bspline_{variant}_firstlayer"],
        )


@pytest.mark.parametrize("start_position", ["firstlayer", "lastlayer"])
@pytest.mark.parametrize("variant", [4, 5, 6])
def test_goal_free_variants_remove_only_cond_xyz_from_start_and_current_vision(monkeypatch, start_position, variant):
    model = tiny_model(monkeypatch, variant, start_position=start_position)
    model.config.skill_end_pose_mode = "xyz"
    model.cond_skill_condition = nn.Linear(3, 32, bias=False)
    if variant == 4:
        model.cond_end_pose_condition = nn.Linear(3, 32, bias=False)
    else:
        assert isinstance(model.cond_end_pose_condition, nn.Identity)
        assert not any(name.startswith("cond_end_pose_condition.") for name, _ in model.named_parameters())
    model._project_state = lambda state: state
    model._code_to_zq = lambda code: code
    model._condition_tokens = lambda images, **kwargs: images[0]
    model._encode_layerwise_latents = lambda tokens, condition: [tokens + condition[:, None]]
    model._time_condition = lambda time: time[:, None].expand(-1, 32)
    model._mode_latent_condition = lambda mode: None
    model.start_pose_condition = nn.Linear(3, 32, bias=False)
    model.end_pose_condition = nn.Linear(3, 32, bias=False)
    state = torch.randn(2, 32, requires_grad=True)
    skill = torch.randn(2, 3, requires_grad=True)
    goal = torch.randn(2, 6, requires_grad=True)
    tokens = torch.randn(2, 4, 32)

    current = model._project_condition_state(state, None, skill, goal)
    held_start = model.encode_skill_start([tokens], state, skill, goal)
    changed = model._project_condition_state(state, None, skill, goal + 1)
    changed_start = model.encode_skill_start([tokens], state, skill, goal + 1)
    gradients = torch.autograd.grad(current.sum() + held_start.sum(), (state, skill, goal), allow_unused=True)
    assert all(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients[:2])
    if variant in (5, 6):
        torch.testing.assert_close(current, changed, atol=0, rtol=0)
        torch.testing.assert_close(held_start, changed_start, atol=0, rtol=0)
        assert gradients[2] is None
    else:
        assert not torch.allclose(current, changed)
        assert not torch.allclose(held_start, changed_start)
        assert gradients[2][:, :3].abs().sum() > 0

    # Removing the Cond goal must not remove either XYZ adapter in the prefix.
    time = torch.rand(2)
    early = model._expert_condition(time, end_pose=goal)
    expert_gradient = torch.autograd.grad(early.sum(), goal)[0]
    assert expert_gradient[:, :3].abs().sum() > 0
    assert expert_gradient[:, 3:].abs().sum() > 0
    late = model._action_layer_condition(model.split_layer, early)
    expected = model._time_condition(time)
    if variant == 6:
        expected = expected + model.bridge_proprio_condition(state)
    torch.testing.assert_close(late, expected)


@pytest.mark.parametrize("start_position", ["firstlayer", "lastlayer"])
@pytest.mark.parametrize("bridge_layers", [1, 9, 17])
def test_variant6_uses_live_proprio_only_in_late_expert_after_start_encoding(monkeypatch, start_position, bridge_layers):
    model = tiny_model(monkeypatch, 6, bridge_layers=bridge_layers, start_position=start_position)
    model.config.skill_end_pose_mode = "xyz"
    model.cond_skill_condition = nn.Linear(3, 32, bias=False)
    model._project_state = lambda state: state
    model._code_to_zq = lambda code: code
    model._condition_tokens = lambda images, **kwargs: images[0]
    model._encode_layerwise_latents = lambda tokens, condition: [tokens + condition[:, None]]
    model._time_condition = lambda time: time[:, None].expand(-1, 32)
    model._mode_latent_condition = lambda mode: None
    model.start_pose_condition = nn.Linear(3, 32, bias=False)
    model.end_pose_condition = nn.Linear(3, 32, bias=False)
    start_state = torch.randn(2, 32, requires_grad=True)
    current_state = torch.randn(2, 32, requires_grad=True)
    skill, goal = torch.randn(2, 3), torch.randn(2, 6, requires_grad=True)
    model.encode_skill_start([torch.randn(2, 4, 32)], start_state, skill, goal)
    model._project_condition_state(current_state, None, skill, goal)
    time = torch.rand(2)
    early = model._expert_condition(time, end_pose=goal)
    torch.testing.assert_close(model._action_layer_condition(model.split_layer - 1, early), early)
    early_proprio = torch.autograd.grad(early.sum(), (start_state, current_state), allow_unused=True)
    assert early_proprio == (None, None)
    late = model._action_layer_condition(model.split_layer, early)
    torch.testing.assert_close(late, model._time_condition(time) + model.bridge_proprio_condition(current_state))
    gradients = torch.autograd.grad(late.sum(), (start_state, current_state, goal), allow_unused=True)
    assert gradients[0] is None and gradients[2] is None
    assert gradients[1].abs().sum() > 0


@pytest.mark.parametrize("length", [1, 2, 3, 10, 248])
def test_spline_keeps_grounding_endpoints_and_linear_gripper(length):
    trajectory = np.arange(length * 8, dtype=np.float32).reshape(length, 8) * .01 + 2.
    target = resample_skill_proprio(trajectory)
    assert target.shape == (30, 8) and target.dtype == np.float32
    np.testing.assert_allclose(target[0], trajectory[0])
    np.testing.assert_allclose(target[-1], trajectory[-1])
    np.testing.assert_allclose(target[:, 6], np.linspace(trajectory[0, 6], trajectory[-1, 6], 30), atol=1e-6)


def test_dataset_spline_respects_selected_episode_mapping_and_cache():
    dataset = object.__new__(SkillVLADataset)
    dataset._bspline_skill_state_tokens, dataset._bspline_target_cache = 30, {}
    dataset.meta = SimpleNamespace(episodes={5: {"length": 4, "dataset_from_index": 20}})
    dataset.reader = SimpleNamespace(_absolute_to_relative_idx={i: i-20 for i in range(20, 24)})
    dataset._chunk_end_state_cache = torch.arange(32).reshape(4, 8).float()
    first = dataset._bspline_skill_states(episode_index=5, skill_start_frame=1, skill_length=2)
    torch.testing.assert_close(first[0], dataset._chunk_end_state_cache[1])
    torch.testing.assert_close(first[-1], dataset._chunk_end_state_cache[2])
    first.zero_()
    second = dataset._bspline_skill_states(episode_index=5, skill_start_frame=1, skill_length=2)
    assert second.sum() > 0 and len(dataset._bspline_target_cache) == 1
    with pytest.raises(ValueError, match="leaves episode"):
        dataset._bspline_skill_states(episode_index=5, skill_start_frame=3, skill_length=2)


@pytest.mark.parametrize("variant", [2, 3, 4, 5, 6])
def test_policy_uses_proprio_mse_and_never_runs_canonical_flow(variant):
    from lerobot.configs.types import FeatureType, PolicyFeature
    class TinyPlan(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_scale = nn.Parameter(torch.tensor(2.))
            self.plan_scale = nn.Parameter(torch.tensor(2.))
            self.bspline_read_gate = nn.Parameter(torch.tensor(.1))
            self._last_vsa_debug_stats = {}
        def encode_skill_start(self, images, state, code, goal):
            return state
        def set_skill_start_latent(self, value):
            self.start = value
        def forward(self, images, state, code, actions, **kwargs):
            self.plan = self.plan_scale.expand(len(actions), 30, 8)
            return torch.ones_like(actions) * self.action_scale
        def take_bspline_prediction(self):
            return self.plan
        def skill_only_flow_residual(self, *args, **kwargs):
            raise AssertionError("Bspline must not run the old canonical flow pass.")
    config = _skill_config(
        f"fullskill_bspline_{variant}_firstlayer", chunk_size=2, n_action_steps=1,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    policy = object.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.config, policy.model = config, TinyPlan()
    policy._collect_images = lambda batch: []
    policy._wrist_patch_alignment_loss = lambda *args, **kwargs: (torch.tensor(0.), {})
    batch = {
        "action": torch.zeros(1, 2, 7), "observation.state": torch.zeros(1, 8),
        "skill_code": torch.tensor([1]), "skill_start_state": torch.zeros(1, 8),
        "skill_start_state_normalized": torch.ones(1, 8),
        "skill_start_image": torch.ones(1, 3, 2, 2),
        "skill_start_wrist_image": torch.ones(1, 3, 2, 2),
        "skill_start_focus_uv": torch.zeros(1, 2), "skill_start_focus_valid": torch.tensor([True]),
        "skill_end_state": torch.ones(1, 8), "skill_end_state_valid": torch.tensor([True]),
        "skill_bspline_states": torch.ones(1, 30, 8),
    }
    loss, metrics = policy(batch)
    assert metrics["action_loss"] == 4. and metrics["bspline/loss"] == 1.
    assert "skill_flow/loss" not in metrics and loss.item() == 5.
    loss.backward()
    assert policy.model.action_scale.grad.item() != 0 and policy.model.plan_scale.grad.item() != 0
    action_loss, action_metrics = policy._vsa_forward(batch, action_only=True)
    assert action_loss.item() == 4. and "bspline/loss" not in action_metrics
