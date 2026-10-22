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
               start_position="firstlayer", current_route="", uv=False, diffaux=False):
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
        self.dino = None
        self.paligemma = None
        self.skill_predictor = None
        self.fsq_term_train = None
        self._action_tokens = lambda x: x

    monkeypatch.setattr(fullskill_vision.BothLIT7SkillExpert, "__init__", initialize)
    config = SimpleNamespace(
        architecture_label=("uv_" if uv else "") + f"fullskill_bspline_{variant}_{start_position}{current_route}" + ("_diffaux" if diffaux else ""), max_state_dim=32,
        uses_top_uv_goal=uv,
        trains_agent_end_state_prediction=diffaux,
        uses_current_bspline_vision=(variant == 3 and start_position == "lastlayer" and current_route.startswith("_lastvis_current")),
        uses_bspline_preskill_memory=current_route.endswith("_preskill"),
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


@pytest.mark.parametrize("route", ["_lastvis_current_preskill", "_lastvis_current_none"])
def test_current_memory_ablation_routes_and_gradients(monkeypatch, route):
    model = tiny_model(monkeypatch, 3, start_position="lastlayer", current_route=route)
    assert model.current_vision_route == "final_all"
    assert model.start_bridge_layer == 8
    hidden = torch.randn(2, 10, 32, requires_grad=True)
    previous = torch.randn(2, 30, 16, requires_grad=True)
    valid = torch.tensor([False, True])
    output, prediction = model._make_and_read_plan(hidden, previous, valid)
    torch.testing.assert_close(output[0], hidden[0])
    if route.endswith("_none"):
        torch.testing.assert_close(output, hidden)
    else:
        assert not torch.equal(output[1], hidden[1])
    output.square().mean().backward(retain_graph=True)
    assert hidden.grad is not None
    assert previous.grad is None
    assert model.bspline_queries.grad is None  # auxiliary only; not action-memory loss
    if route.endswith("_none"):
        assert model.bspline_read_gate.grad is None
        assert all(p.grad is None for p in model.bspline_reader.parameters())
    else:
        assert model.bspline_read_gate.grad is not None
        assert model.bspline_reader.out_proj.weight.grad is not None
    prediction.square().mean().backward()
    assert model.bspline_queries.grad is not None


def test_previous_memory_encoder_is_prefix_only_detached_deterministic(monkeypatch):
    model = tiny_model(monkeypatch, 3, start_position="lastlayer", current_route="_lastvis_current_preskill")
    model.config.chunk_size, model.config.max_action_dim = 10, 32
    model.train()
    before_shift, before_time = model._bridge_condition_shift, model._bspline_time_condition
    calls, times = [], []
    latent = torch.randn(2, 100, 16)
    model._condition_tokens = lambda images, **kw: images[0]
    model._project_condition_state = lambda state, *args: state
    model._encode_layerwise_latents = lambda tokens, state: [latent] * 18
    model._skill_broadcasts = lambda code: (None, torch.zeros(2, 32))
    def condition(time, **kw):
        times.append(time)
        return torch.zeros(2, 32)
    model._expert_condition = condition
    def layer(index, hidden, mask, positions, condition, skill, visual, rotary):
        calls.append((index, torch.is_grad_enabled(), visual))
        return hidden + .01
    model._expert_layer_with_latent_bridge = layer
    def forbidden(*args, **kwargs):
        raise AssertionError("Previous plan must never read an earlier plan")
    model._make_and_read_plan = forbidden
    state = torch.randn(2, 32, requires_grad=True)
    rng = torch.random.get_rng_state().clone()
    first = model.encode_preskill_plan([state, state], state, torch.tensor([1, 2]), torch.zeros(2, 6))
    second = model.encode_preskill_plan([state, state], state, torch.tensor([1, 2]), torch.zeros(2, 6))
    assert first.shape == (2, 30, 16) and not first.requires_grad
    torch.testing.assert_close(first, second)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert [index for index, _, _ in calls[:9]] == list(range(9))
    assert all(not grad for _, grad, _ in calls)
    assert calls[8][2] is latent and calls[0][2] is None
    assert all(torch.equal(time, torch.ones(2)) for time in times)
    assert model.training and model.bspline_generator.training
    assert model._bridge_condition_shift is before_shift and model._bspline_time_condition is before_time


@pytest.mark.parametrize("rank,start,expected", [(0, 0, -1), (0, 5, -1), (1, 0, -1),
                                                 (2, 15, 14), (2, 25, 24), (2, 8, 7)])
def test_preskill_dataset_uses_causal_boundary_and_previous_condition(rank, start, expected):
    from lerobot.policies.skillVLA import preskill_plan as memory
    dataset = SkillVLADataset.__new__(SkillVLADataset)
    reader = SimpleNamespace(
        _image_transforms=None, _absolute_to_relative_idx={index: index - 90 for index in range(100, 160)},
        _query_videos=lambda query, episode: {
            key: torch.full((3, 4, 4), next(iter(query.values()))[0]) for key in query
        },
    )
    dataset._ensure_reader = lambda: reader
    dataset.meta = SimpleNamespace(fps=1, episodes={0: {"dataset_from_index": 100}})
    dataset._chunk_end_state_cache = torch.arange(70).float()[:, None].expand(-1, 8)
    dataset._skill_end_state = SimpleNamespace(target_state=lambda *args: (np.full(8, args[1] + 100), True))
    dataset._focus_uv = SimpleNamespace(target=lambda *args: (np.array([args[1] / 10., -.4]), None, True, False))
    dataset._foveated_vision = SimpleNamespace(enabled=False, randomization_enabled=False)
    item = {"observation.state": torch.ones(8), "observation.images.image": torch.ones(3, 4, 4),
            "observation.images.wrist_image": torch.ones(3, 4, 4)}
    dataset._add_preskill_plan_inputs(item, 0, rank, start, [0, 10, 20], [2, 3, 4])
    assert item[memory.FRAME].item() == expected
    assert item[memory.VALID].item() == (expected >= 0)
    if expected >= 0:
        assert item[memory.CODE].item() == [2, 3, 4][rank - 1]
        assert item[memory.IMAGE][0, 0, 0].item() == expected
        assert item[memory.STATE][0].item() == expected + 10
        assert item[memory.START_STATE][0].item() == min([0, 10, 20][rank - 1], expected) + 10
        assert item[memory.END_STATE][0].item() == rank - 1 + 100
        torch.testing.assert_close(item[memory.FOCUS_UV], torch.tensor([(rank - 1) / 10., -.4], dtype=torch.float64))
        assert item[memory.FOCUS_VALID].item()


def test_preskill_preprocessor_preserves_raw_goal_and_normalizes_only_observation():
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.skill_expert.processor_skill_expert import make_skill_expert_pre_post_processors
    from lerobot.policies.skillVLA import preskill_plan as memory
    config = _skill_config(
        "fullskill_bspline_3_lastlayer_lastvis_current_preskill", device="cpu",
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )
    pre, _ = make_skill_expert_pre_post_processors(config, {
        "observation.state": {"q01": torch.zeros(8), "q99": torch.full((8,), 2.)},
        "action": {"q01": -torch.ones(7), "q99": torch.ones(7)},
    })
    batch = pre({"observation.state": torch.ones(2, 8), memory.STATE: torch.ones(2, 8),
                 memory.START_STATE: torch.full((2, 8), .5), memory.END_STATE: torch.ones(2, 8),
                 memory.VALID: torch.tensor([False, True]), memory.CACHE_ID: torch.tensor([0, 1])})
    torch.testing.assert_close(batch[memory.STATE_NORMALIZED], torch.zeros(2, 8), atol=1e-6, rtol=0)
    torch.testing.assert_close(batch[memory.START_STATE], torch.full((2, 8), .5))
    assert batch[memory.VALID].tolist() == [False, True]
    assert batch[memory.CACHE_ID].tolist() == [0, 1]


def test_preskill_policy_cache_updates_only_changed_occurrences_and_clears_bos():
    from lerobot.policies.skillVLA import preskill_plan as memory
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.calls = []
        def encode_preskill_plan(self, images, state, code, goal):
            self.calls.append((state.clone(), code.clone(), goal.clone()))
            return state[:, :1, None].expand(-1, 30, 256) + self.anchor
        def set_preskill_plan(self, plan, valid):
            self.plan, self.valid = plan, valid
    policy = SkillExpertPolicy.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.config = _skill_config("fullskill_bspline_3_lastlayer_lastvis_current_preskill")
    policy.model = Encoder()
    batch = {
        memory.IMAGE: torch.ones(2, 3, 4, 4), memory.WRIST_IMAGE: torch.ones(2, 3, 4, 4),
        memory.STATE_NORMALIZED: torch.tensor([[1.] * 8, [2.] * 8]),
        memory.START_STATE: torch.zeros(2, 8), memory.END_STATE: torch.ones(2, 8),
        memory.CODE: torch.tensor([5, 5]), memory.VALID: torch.tensor([False, True]),
        memory.CACHE_ID: torch.tensor([0, 1]),
    }
    policy._prepare_preskill_memory(batch, inference=True)
    assert len(policy.model.calls) == 1 and policy.model.calls[0][1].tolist() == [5]
    assert not policy.model.plan.requires_grad
    assert policy.model.plan[0].count_nonzero() == 0
    torch.testing.assert_close(policy.model.plan[1], torch.full((30, 256), 2.))
    batch[memory.STATE_NORMALIZED] += 10
    policy._prepare_preskill_memory(batch, inference=True)
    assert len(policy.model.calls) == 1  # held through all action replans
    batch[memory.CACHE_ID][0] = 1
    batch[memory.VALID][0] = True
    policy._prepare_preskill_memory(batch, inference=True)
    assert len(policy.model.calls) == 2 and policy.model.calls[-1][0].shape[0] == 1
    torch.testing.assert_close(policy.model.plan[0], torch.full((30, 256), 11.))
    torch.testing.assert_close(policy.model.plan[1], torch.full((30, 256), 2.))
    # Same FSQ code but a NEW occurrence still generates a new memory.
    batch[memory.CACHE_ID][1] = 2
    policy._prepare_preskill_memory(batch, inference=True)
    assert len(policy.model.calls) == 3
    policy.reset()
    assert policy._preskill_identity is None and policy._preskill_plan_cache is None
    assert policy.model.plan is None


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("route", ["_lastvis_current_preskill", "_lastvis_current_none"])
def test_memory_ablation_full_expert_backward_survives_cache_replacement(monkeypatch, checkpoint, route):
    model = tiny_model(monkeypatch, 3, checkpoint=checkpoint, start_position="lastlayer", current_route=route)
    model.train()
    previous = torch.randn(2, 30, 16, requires_grad=True)
    model.set_preskill_plan(previous, torch.tensor([False, True]))
    hidden = torch.randn(2, 10, 32, requires_grad=True)
    condition = torch.randn(2, 32)
    skill = torch.randn(2, 32)
    latent = torch.randn(2, 100, 16, requires_grad=True)
    output = model._run_expert_with_layerwise_latents(hidden, condition, skill, [latent] * 18)
    prediction = model.take_bspline_prediction()
    model.set_preskill_plan(None, None)
    model._bspline_time_condition = model._bridge_condition_shift = None
    (output.float().square().mean() + prediction.square().mean()).backward()
    assert hidden.grad is not None and latent.grad is not None
    assert previous.grad is None
    assert model.bspline_queries.grad is not None
    assert model.bspline_proprio_head[-1].weight.grad is not None


@pytest.mark.parametrize("start_position,route", [
    *[(position, route) for position in ("firstlayer", "lastlayer")
      for route in ("lastvis", "lastvis_first", "lastvis_last")],
    ("lastlayer", "lastvis_current"),
])
@pytest.mark.parametrize("bridge_layers", [1, 9, 17])
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("backend,dtype", [("eager", torch.float32), ("sdpa", torch.bfloat16)])
def test_lastvis_routes_only_change_current_memory(
    monkeypatch, route, start_position, bridge_layers, checkpoint, backend, dtype,
):
    torch.manual_seed(123)
    base = tiny_model(monkeypatch, 3, backend, checkpoint, bridge_layers, start_position)
    torch.manual_seed(123)
    model = tiny_model(monkeypatch, 3, backend, checkpoint, bridge_layers, start_position, "_" + route)
    assert model.state_dict().keys() == base.state_dict().keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, base.state_dict()[name], rtol=0, atol=0)
    split = 18 - bridge_layers
    expected_layers = (
        [split] if route == "lastvis_first" else [17] if route == "lastvis_last" else list(range(split, 18))
    )
    start_layer = 0 if start_position == "firstlayer" else split - 1
    assert model.start_bridge_layer == start_layer
    assert list(model.current_bridge_layers) == expected_layers
    assert model._active_visual_bridge_gates().numel() == len(expected_layers)
    model.gemma_expert.to(dtype=dtype)
    start = torch.randn(2, 100, 16, requires_grad=True)
    latents = [torch.randn(2, 100, 16, requires_grad=True) for _ in range(18)]
    shared_current = model.config.uses_current_bspline_vision
    if not shared_current:
        model.set_skill_start_latent(start)
    noise = torch.randn(2, 10, 32, dtype=dtype, requires_grad=True)
    condition = torch.randn(2, 32, dtype=dtype, requires_grad=True)
    skill = torch.randn(2, 32, dtype=dtype, requires_grad=True)
    calls, plan_calls = [], []
    original = model._expert_layer_with_latent_bridge
    def record(index, hidden, mask, pos, cond, code, latent, rotary):
        calls.append((index, code is not None, latent))
        return original(index, hidden, mask, pos, cond, code, latent, rotary)
    model._expert_layer_with_latent_bridge = record
    hook = model.bspline_generator.register_forward_hook(lambda *args: plan_calls.append(len(calls)))
    output = model._run_expert_with_layerwise_latents(noise, condition, skill, latents)
    prediction = model.take_bspline_prediction()
    hook.remove()
    assert output.shape == noise.shape and prediction.shape == (2, 30, 8)
    assert plan_calls == [split]
    assert [i for i, _, memory in calls if memory is not None] == [start_layer, *expected_layers]
    assert calls[start_layer][2] is (latents[-1] if shared_current else start)
    assert all(calls[i][2] is latents[-1] for i in expected_layers)
    assert all(has_skill == (i < split) for i, has_skill, _ in calls)
    # Checkpoint recomputation must use the captured tensors, not later caches.
    model.set_skill_start_latent(None)
    model._bridge_condition_shift, model._bspline_time_condition = None, None
    (output.float().square().mean() + prediction.square().mean()).backward()
    if shared_current:
        assert start.grad is None
    else:
        assert start.grad.abs().sum() > 0
    assert latents[-1].grad.abs().sum() > 0
    assert all(latent.grad is None for latent in latents[:-1])
    assert model.bspline_queries.grad.abs().sum() > 0
    assert model.bspline_proprio_head[-1].weight.grad.abs().sum() > 0
    assert model.gemma_expert.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0


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
@pytest.mark.parametrize("uv", [False, True])
@pytest.mark.parametrize("frozen_ft", [False, True])
def test_policy_uses_proprio_mse_and_never_runs_canonical_flow(variant, uv, frozen_ft, current=False):
    if uv and variant != 3:
        pytest.skip("Top-UV ablations use variant 3 only")
    from lerobot.configs.types import FeatureType, PolicyFeature
    class TinyPlan(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_scale = nn.Parameter(torch.tensor(2.))
            self.plan_scale = nn.Parameter(torch.tensor(2.))
            self.bspline_read_gate = nn.Parameter(torch.tensor(.1))
            self._last_vsa_debug_stats = {}
        def encode_skill_start(self, images, state, code, goal):
            if current:
                raise AssertionError("Current Bspline must not encode held start images.")
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
    label = ("fullskill_bspline_3_lastlayer_lastvis_current" if current else
             "fullskill_bspline_3_lastlayer_lastvis" if uv else f"fullskill_bspline_{variant}_firstlayer")
    config = _skill_config(
        ("uv_" if uv else "") + label,
        newtask_ft_enabled=frozen_ft, newtask_ft_freeze_expert=frozen_ft,
        newtask_ft_skill_flow_loss=True,
        chunk_size=2, n_action_steps=1,
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
        "skill_focus_uv": torch.tensor([[.2, -.4]]), "skill_focus_valid": torch.tensor([True]),
        "skill_bspline_states": torch.ones(1, 30, 8),
    }
    if current:
        for key in ("skill_start_state_normalized", "skill_start_image", "skill_start_wrist_image",
                    "skill_start_focus_uv", "skill_start_focus_valid"):
            del batch[key]
        def forbidden_start_loss(*args, **kwargs):
            raise AssertionError("Current-only encoding must not apply a second start-patch loss.")
        policy._skill_start_patch_alignment_loss = forbidden_start_loss
        policy._prepare_start_vision(batch, batch["skill_code"], None, inference=True)
    loss, metrics = policy(batch)
    assert metrics["action_loss"] == 4. and metrics["bspline/loss"] == 1.
    assert "skill_flow/loss" not in metrics and loss.item() == 5.
    loss.backward()
    assert policy.model.action_scale.grad.item() != 0 and policy.model.plan_scale.grad.item() != 0
    action_loss, action_metrics = policy._vsa_forward(batch, action_only=True)
    assert action_loss.item() == 4. and "bspline/loss" not in action_metrics


@pytest.mark.parametrize("uv", [False, True])
def test_current_policy_needs_no_start_encoding_or_start_patch_targets(uv):
    test_policy_uses_proprio_mse_and_never_runs_canonical_flow(3, current=True, uv=uv, frozen_ft=False)


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("current", [False, True])
def test_frozen_expert_and_plan_head_pass_bspline_loss_to_visual_inputs_only(monkeypatch, checkpoint, current):
    from lerobot.policies.skill_expert.modeling_skill_expert import apply_newtask_ft_freeze

    model = tiny_model(monkeypatch, 3, checkpoint=checkpoint, start_position="lastlayer",
                       current_route="_lastvis_current" if current else "_lastvis")
    model.cond_encoder = nn.Linear(16, 16)
    model.bottleneck_tokens = nn.Parameter(torch.randn(2, 100, 16))
    config = SimpleNamespace(visual_bridge_last_n_layers=9, newtask_ft_freeze_expert=True,
                             uses_bspline_plan=True)
    apply_newtask_ft_freeze(model, config)
    before = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    memory = model.cond_encoder(model.bottleneck_tokens)
    model.set_skill_start_latent(memory)
    latents = [torch.zeros(2, 100, 16) for _ in range(18)]
    if current:
        latents[-1] = memory
    model._run_expert_with_layerwise_latents(
        torch.randn(2, 10, 32), torch.zeros(2, 32), torch.zeros(2, 32), latents,
    )
    # Auxiliary supervision ONLY, without any main-action loss contribution.
    (model.take_bspline_prediction().float() - 1).square().mean().backward()
    assert model.cond_encoder.weight.grad.abs().sum() > 0
    assert model.bottleneck_tokens.grad.abs().sum() > 0
    assert model.start_bridge_attention.q_proj_weight.grad.abs().sum() > 0
    assert not model.bspline_queries.requires_grad and model.bspline_queries.grad is None
    assert all(p.grad is None for p in model.bspline_proprio_head.parameters())
    assert all(p.grad is None for p in model.gemma_expert.parameters())
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name in before:
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
