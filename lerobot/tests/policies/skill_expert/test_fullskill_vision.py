from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.skill_expert import fullskill_vision, layerwise_cond_bottleneck, modeling_utils
from lerobot.policies.skill_expert.fullskill_vision import FullskillVisSkillExpert
from lerobot.policies.skill_expert.configuration_skill_expert import (
    FULLSKILL_VIS_ARCH_LABELS, FULLSKILL_VIS_REVISIONS, FULLSKILL_VIS_LEGACY_REVISIONS,
    FULLSKILL_VIS_SPLIT_REVISIONS,
)
from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy
from lerobot.policies.skill_expert import modeling_skill_expert
from lerobot.policies.skill_expert.wrist_patch_alignment import WristPatchAlignmentHead


def tiny_model(monkeypatch, checkpoint=False, label="fullskill_vis_1_firstlayer", trained_proprio=True, revision=None, bridge_layers=1):
    def initialize(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.width = 8
        self.action_in_proj = nn.Linear(8, 8)
        self.action_out_proj = nn.Linear(8, 8)
        self.gemma_expert = nn.Module()
        self.gemma_expert.model = nn.Module()
        self.gemma_expert.model.config = SimpleNamespace(num_hidden_layers=18)
        self.gemma_expert.model.norm = nn.Identity()
        self.gemma_expert.model.rotary_emb = lambda hidden, pos: (pos, pos)
        self.visual_bridge_query_norm = nn.LayerNorm(8)
        self.visual_bridge_attention = nn.MultiheadAttention(8, 2, kdim=4, vdim=4, batch_first=True)
        self.visual_bridge_gates = nn.Parameter(torch.full((18,), .1))
        self.visual_align_bridge_attention = nn.MultiheadAttention(8, 2, kdim=4, vdim=4, batch_first=True)
        self.visual_align_bridge_gates = nn.Parameter(torch.zeros(18))
        self._gradient_checkpointing = checkpoint
        self._vsa_debug_active = False
        self._capture_visual_bridge_attention = False
        self._captured_visual_bridge_attention = {}
        self.calls = []

    monkeypatch.setattr(fullskill_vision.BothLIT7SkillExpert, "__init__", initialize)
    monkeypatch.setattr(layerwise_cond_bottleneck, "layernorm_forward", lambda norm, h, c: (norm(h), None))
    revision = revision or (
        FULLSKILL_VIS_SPLIT_REVISIONS.get(label, FULLSKILL_VIS_REVISIONS[label])
        if bridge_layers != 1 else FULLSKILL_VIS_REVISIONS[label]
    )
    config = SimpleNamespace(architecture_label=label, architecture_revision=revision,
                             uses_symmetric_start_vision=label != "fullskill_vis_1" and revision in (
                                 FULLSKILL_VIS_REVISIONS[label], FULLSKILL_VIS_SPLIT_REVISIONS.get(label)),
                             max_state_dim=8,
                             visual_bridge_last_n_layers=bridge_layers, visual_bridge_heads=2,
                             visual_bottleneck_width=4, visual_bridge_gate_init=.1,
                             skill_flow_checkpoint_layers=17)
    model = FullskillVisSkillExpert(config)
    if model.direct_current_proprio:
        # Nonzero learned adapter so the routing check isn't vacuous at zero-init.
        if trained_proprio:
            nn.init.normal_(model.bridge_proprio_condition[-1].weight)
        model.current_proprio = torch.ones(2, 8, requires_grad=True)
        model._bridge_condition_shift = model.bridge_proprio_condition(model.current_proprio)
    model._action_tokens = lambda value: value

    def layer(self, i, hidden, mask, pos, cond, skill, latent, rotary):
        self.calls.append((i, None if latent is None else latent.detach().clone()))
        self.condition_calls.append((i, cond.detach().clone()))
        if self._layer_uses_visual_bridge(i):
            hidden = hidden + self._visual_bridge_update(i, hidden, latent)
        return hidden + .01 * hidden.tanh() + .01 * cond[:, None]

    model._expert_layer_with_latent_bridge = MethodType(layer, model)
    model.condition_calls = []
    model._sequence_geometry = lambda h, m: (
        torch.zeros(h.shape[0], 1, h.shape[1], h.shape[1]),
        torch.arange(h.shape[1])[None].expand(h.shape[0], -1), (torch.zeros(1), torch.zeros(1)))
    return model


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("label", FULLSKILL_VIS_ARCH_LABELS)
@pytest.mark.parametrize("bridge_layers", [1, 9, 17])
def test_start_and_current_vision_reach_only_their_intended_layers(monkeypatch, checkpoint, label, bridge_layers):
    model = tiny_model(monkeypatch, checkpoint, label, bridge_layers=bridge_layers)
    split = 18 - bridge_layers
    start_layer = split - 1 if label.endswith("_lastlayer") or label.startswith("fullskill_vis_2_lastlayer_") else 0
    current_layers = (
        [17] if label.endswith("_lastvis_last") else
        [split] if label == "fullskill_vis_2_lastlayer_lastvis_first" else list(range(split, 18))
    )
    start = torch.randn(2, 100, 4, requires_grad=True)
    current = [torch.randn(2, 100, 4, requires_grad=True) for _ in range(18)]
    model.set_skill_start_latent(start)
    chunk = model._run_expert_with_layerwise_latents(
        torch.randn(2, 2, 8), torch.zeros(2, 8), torch.zeros(2, 8), current)
    assert [i for i, v in model.calls if v is not None] == [start_layer, *current_layers]
    torch.testing.assert_close(model.calls[start_layer][1], start.detach())
    for index in current_layers:
        source = 17 if model.current_vision_route != "layerwise" else index
        torch.testing.assert_close(model.calls[index][1], current[source].detach())
    expected_proprio_layers = list(range(split, 18)) if label.startswith("fullskill_vis_2_") else []
    assert [i for i, c in model.condition_calls if c.abs().sum() > 0] == expected_proprio_layers
    torch.testing.assert_close(model._active_visual_bridge_gates(), model.visual_bridge_gates[current_layers])
    model.calls.clear()
    model.condition_calls.clear()
    skill = model._skill_only_expert_hidden(
        torch.randn(2, 5, 8), torch.zeros(2, 1, 5, 5),
        torch.arange(5)[None].expand(2, -1), torch.zeros(2, 8), torch.zeros(2, 8))
    assert [i for i, v in model.calls if v is not None] == [start_layer]
    assert all(c.abs().sum() == 0 for _, c in model.condition_calls)
    assert len(model.calls) == split
    # Replacing the held cache before backward must NOT change checkpoint inputs.
    model.set_skill_start_latent(torch.zeros_like(start))
    model._bridge_condition_shift = None
    skill_grad = torch.autograd.grad(skill.square().sum(), (start, *current), retain_graph=True, allow_unused=True)
    assert skill_grad[0].abs().sum() > 0
    assert all(g is None for g in skill_grad[1:])
    if model.direct_current_proprio:
        proprio_grad = torch.autograd.grad(
            skill.square().sum(), model.current_proprio, retain_graph=True, allow_unused=True)[0]
        assert proprio_grad is None
        assert torch.autograd.grad(chunk.square().sum(), model.current_proprio, retain_graph=True)[0].abs().sum() > 0
    chunk_grad = torch.autograd.grad(chunk.square().sum(), (start, *current), retain_graph=True, allow_unused=True)
    assert chunk_grad[0].abs().sum() > 0
    expected_sources = set(current_layers) if model.current_vision_route == "layerwise" else {17}
    for index, gradient in enumerate(chunk_grad[1:]):
        assert (gradient is not None and gradient.abs().sum() > 0) == (index in expected_sources)
    (skill.square().mean() + chunk.square().mean()).backward()
    assert model.start_bridge_attention.in_proj_weight is None  # Separate K/V width.
    assert model.start_bridge_attention.q_proj_weight.grad.abs().sum() > 0
    assert model.visual_bridge_attention.q_proj_weight.grad.abs().sum() > 0


@pytest.mark.parametrize("label", [
    "fullskill_vis_2_lastlayer", "fullskill_vis_2_lastlayer_lastvis",
    "fullskill_vis_2_lastlayer_lastvis_last", "fullskill_vis_2_lastlayer_lastvis_first",
])
@pytest.mark.parametrize("backend,dtype", [("eager", torch.float32), ("sdpa", torch.bfloat16)])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_real_gemma_9_9_action_and_canonical_backward(monkeypatch, label, backend, dtype, checkpoint):
    from lerobot.policies.pi_gemma import layernorm_forward
    torch.manual_seed(7)
    model = tiny_model(monkeypatch, checkpoint=checkpoint, label=label, bridge_layers=9)
    monkeypatch.setattr(modeling_utils, "get_gemma_config", lambda variant: SimpleNamespace(
        width=8, depth=18, mlp_dim=16, num_heads=2, num_kv_heads=1, head_dim=4,
    ))
    model.gemma_expert = modeling_utils.build_gemma("gemma_300m", use_adarms=True, attention_backend=backend)
    # Train-like nonzero AdaRMS gates exercise attention, not just residual bypasses.
    with torch.no_grad():
        for layer in model.gemma_expert.model.layers:
            for norm in (layer.input_layernorm, layer.post_attention_layernorm):
                nn.init.normal_(norm.dense.weight, std=.02)
                norm.dense.bias[-8:].fill_(.5)
    model.gemma_expert.to(dtype=dtype)
    monkeypatch.setattr(layerwise_cond_bottleneck, "layernorm_forward", layernorm_forward)
    model._sequence_geometry = layerwise_cond_bottleneck.LayerwiseCondBottleneckSkillExpert._sequence_geometry
    model._expert_layer_with_latent_bridge = MethodType(
        layerwise_cond_bottleneck.LayerwiseCondBottleneckSkillExpert._expert_layer_with_latent_bridge, model,
    )
    start = torch.randn(2, 100, 4, requires_grad=True)
    current = [torch.randn(2, 100, 4, requires_grad=True) for _ in range(18)]
    model.set_skill_start_latent(start)
    condition, skill = [torch.randn(2, 8, dtype=dtype, requires_grad=True) for _ in range(2)]
    action = model._run_expert_with_layerwise_latents(torch.randn(2, 10, 8, dtype=dtype), condition, skill, current)
    assert action.shape == (2, 10, 8)
    hidden = torch.randn(2, 5, 8, dtype=dtype)
    mask, positions, _ = model._sequence_geometry(hidden, model.gemma_expert.model)
    canonical = model._skill_only_expert_hidden(hidden, mask, positions, condition, skill)
    assert canonical.shape == (2, 5, 8)
    # Checkpoint recomputation must not consult caches from a later forward.
    model.set_skill_start_latent(None)
    model._bridge_condition_shift = None
    (action.float().square().mean() + canonical.float().square().mean()).backward()
    assert start.grad.abs().sum() > 0
    assert model.current_proprio.grad.abs().sum() > 0
    sources = set(model.current_bridge_layers) if model.current_vision_route == "layerwise" else {17}
    for i, memory in enumerate(current):
        assert (memory.grad is not None and memory.grad.abs().sum() > 0) == (i in sources)
    assert model.gemma_expert.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert model.gemma_expert.model.layers[9].self_attn.q_proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("checkpoint", [False, True])
def test_frozen_expert_passes_skill_flow_gradients_to_cond_bottleneck_and_start_bridge(monkeypatch, checkpoint):
    from lerobot.policies.pi_gemma import layernorm_forward
    from lerobot.policies.skill_expert.modeling_skill_expert import apply_newtask_ft_freeze

    model = tiny_model(monkeypatch, checkpoint=checkpoint, label="fullskill_vis_2_lastlayer", bridge_layers=9)
    monkeypatch.setattr(modeling_utils, "get_gemma_config", lambda variant: SimpleNamespace(
        width=8, depth=18, mlp_dim=16, num_heads=2, num_kv_heads=1, head_dim=4,
    ))
    model.gemma_expert = modeling_utils.build_gemma("gemma_300m", use_adarms=True, attention_backend="sdpa")
    with torch.no_grad():
        for layer in model.gemma_expert.model.layers:
            for norm in (layer.input_layernorm, layer.post_attention_layernorm):
                nn.init.normal_(norm.dense.weight, std=.02)
                norm.dense.bias[-8:].fill_(.5)
    monkeypatch.setattr(layerwise_cond_bottleneck, "layernorm_forward", layernorm_forward)
    model._expert_layer_with_latent_bridge = MethodType(
        layerwise_cond_bottleneck.LayerwiseCondBottleneckSkillExpert._expert_layer_with_latent_bridge, model,
    )
    model._sequence_geometry = layerwise_cond_bottleneck.LayerwiseCondBottleneckSkillExpert._sequence_geometry
    model.cond_encoder = nn.Linear(4, 4)
    model.bottleneck_tokens = nn.Parameter(torch.randn(2, 100, 4))
    apply_newtask_ft_freeze(model, SimpleNamespace(visual_bridge_last_n_layers=9, newtask_ft_freeze_expert=True))
    before = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    model.set_skill_start_latent(model.cond_encoder(model.bottleneck_tokens))
    hidden = torch.randn(2, 5, 8)
    condition = torch.zeros(2, 8)
    mask, positions, _ = model._sequence_geometry(hidden, model.gemma_expert.model)
    hidden = model._skill_only_expert_hidden(hidden, mask, positions, condition, torch.zeros(2, 8))
    hidden, _ = layernorm_forward(model.gemma_expert.model.norm, hidden, condition)
    residual = model.action_out_proj(hidden) - 1
    # Full-skill residual loss ONLY: no deployed action-chunk objective.
    residual.square().mean().backward()
    assert model.cond_encoder.weight.grad.abs().sum() > 0
    assert model.bottleneck_tokens.grad.abs().sum() > 0
    assert model.start_bridge_attention.q_proj_weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.gemma_expert.parameters())
    assert all(p.grad is None for p in model.action_out_proj.parameters())
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name in before:
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("label", ["fullskill_vis_2_firstlayer", "fullskill_vis_2_lastlayer"])
def test_proprio_adapter_is_both3_zero_initialized_and_fp32(monkeypatch, label):
    model = tiny_model(monkeypatch, label=label, trained_proprio=False)
    assert torch.count_nonzero(model.bridge_proprio_condition[-1].weight) == 0
    assert torch.count_nonzero(model._bridge_condition_shift) == 0
    # tiny_model replaces the heavyweight parent initializer, so it has no
    # DINO/Cond reader stack for the parent's dtype policy to traverse.
    monkeypatch.setattr(fullskill_vision.BothLIT7SkillExpert, "_apply", nn.Module._apply)
    model.to(dtype=torch.bfloat16)
    assert all(p.dtype == torch.float32 for p in model.bridge_proprio_condition.parameters())
    assert model.start_bridge_attention.q_proj_weight.dtype == torch.float32
    assert model.start_align_bridge_attention.q_proj_weight.dtype == torch.float32
    assert model.start_align_bridge_gate.dtype == torch.float32


@pytest.mark.parametrize("label", ["fullskill_vis_2_firstlayer", "fullskill_vis_2_lastlayer"])
def test_start_encoding_cannot_replace_the_live_l18_proprio(monkeypatch, label):
    model = tiny_model(monkeypatch, label=label)
    monkeypatch.setattr(fullskill_vision.BothLIT7SkillExpert, "_project_condition_state",
                        lambda self, state, *args: state)
    model._condition_tokens = lambda images, **kwargs: images[0]
    model._encode_layerwise_latents = lambda tokens, state: [tokens + state[:, None]]
    start_state, current_state = torch.ones(2, 8), torch.full((2, 8), 7.)
    model.encode_skill_start([torch.ones(2, 100, 8)], start_state, torch.ones(2), torch.ones(2, 6))
    model._project_condition_state(current_state, None, torch.ones(2), torch.ones(2, 6))
    base_condition = torch.zeros(2, 8)
    torch.testing.assert_close(model._action_layer_condition(17, base_condition),
                               model.bridge_proprio_condition(current_state))
    for layer in range(17):
        torch.testing.assert_close(model._action_layer_condition(layer, base_condition), base_condition)


def test_unprepared_start_is_not_silently_replaced_by_current_vision(monkeypatch):
    model = tiny_model(monkeypatch)
    with pytest.raises(RuntimeError, match="skill-start observation"):
        model._run_expert_with_layerwise_latents(
            torch.ones(1, 2, 8), torch.zeros(1, 8), torch.zeros(1, 8), [torch.ones(1, 100, 4)] * 18)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.calls = []
        self._skill_start_latent = None

    def encode_skill_start(self, images, state, code, goal):
        self.calls.append((images[0].clone(), state.clone(), code.clone(), goal.clone()))
        return (images[0].mean((1, 2, 3)) + state.sum(-1))[:, None, None] * self.weight

    def set_skill_start_latent(self, value):
        self._skill_start_latent = value


@pytest.mark.parametrize("label", [label for label in FULLSKILL_VIS_ARCH_LABELS if label != "fullskill_vis_1"])
def test_start_and_current_have_identical_98_plus_2_readers_but_independent_weights(monkeypatch, label):
    model = tiny_model(monkeypatch, label=label)
    # Equalize parameter VALUES, not parameter objects: equal queries/latents
    # must then give exactly the same readout on both bridges.
    model.start_bridge_query_norm.load_state_dict(model.visual_bridge_query_norm.state_dict())
    model.start_bridge_attention.load_state_dict(model.visual_bridge_attention.state_dict())
    model.start_align_bridge_attention.load_state_dict(model.visual_align_bridge_attention.state_dict())
    with torch.no_grad():
        model.start_bridge_gate.copy_(model.visual_bridge_gates[17])
        model.start_align_bridge_gate.fill_(.2)
        model.visual_align_bridge_gates[17].fill_(.2)
    for start, current in [
        (model.start_bridge_attention, model.visual_bridge_attention),
        (model.start_align_bridge_attention, model.visual_align_bridge_attention),
    ]:
        assert all(a is not b and a.data_ptr() != b.data_ptr() for a, b in zip(start.parameters(), current.parameters()))
    lengths = []
    handles = [reader.register_forward_pre_hook(lambda reader, args: lengths.append(args[1].shape[1]))
               for reader in [model.start_bridge_attention, model.start_align_bridge_attention,
                              model.visual_bridge_attention, model.visual_align_bridge_attention]]
    hidden, latent = torch.randn(2, 3, 8), torch.randn(2, 100, 4, requires_grad=True)
    start = model._visual_bridge_update(model.start_bridge_layer, hidden, latent)
    current = model._visual_bridge_update(17, hidden, latent)
    for handle in handles:
        handle.remove()
    assert lengths == [98, 2, 98, 2]
    torch.testing.assert_close(start, current)
    start.square().sum().backward()
    assert latent.grad[:, :2].abs().sum() > 0
    assert latent.grad[:, 2:].abs().sum() > 0
    assert model.start_align_bridge_attention.q_proj_weight.grad.abs().sum() > 0
    assert model.start_align_bridge_gate.grad.abs().sum() > 0
    assert model.visual_align_bridge_attention.q_proj_weight.grad is None


@pytest.mark.parametrize("label", tuple(FULLSKILL_VIS_LEGACY_REVISIONS))
def test_legacy_revisions_keep_their_original_start_reader(monkeypatch, label):
    model = tiny_model(monkeypatch, label=label, revision=FULLSKILL_VIS_LEGACY_REVISIONS[label])
    assert not model.symmetric_start_vision
    assert not hasattr(model, "start_align_bridge_attention")
    assert not hasattr(model, "start_align_bridge_gate")


def test_start_and_current_losses_share_heads_but_consume_their_own_features_and_raw_geometry(monkeypatch):
    model = object.__new__(FullskillVisSkillExpert)
    nn.Module.__init__(model)
    model.agent_patch_align_head = WristPatchAlignmentHead(4, 8, width=8)
    model.wrist_patch_align_head = WristPatchAlignmentHead(4, 8, width=8)
    policy = object.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.model = model
    policy.config = SimpleNamespace(
        architecture_label="fullskill_vis_1_firstlayer", dino_image_size=224,
        wrist_patch_align_target_sigma=.7, trains_agent_patch_alignment=True,
    )
    states = []
    project = modeling_skill_expert.patch_labels
    def record_projection(xyz, rotation, target, camera, **kwargs):
        states.append(torch.cat((xyz, rotation), dim=-1).clone())
        return project(xyz, rotation, target, camera, **kwargs)
    monkeypatch.setattr(modeling_skill_expert, "patch_labels", record_projection)
    batch = {
        "skill_start_state": torch.zeros(1, 8),
        "skill_start_state_normalized": torch.full((1, 8), 99.),
        "skill_decoder_state": torch.tensor([[.02, .01, 0., 0., 0., 0., 0., 0.]]),
        "skill_end_state": torch.tensor([[0., 0., .25]]),
        "skill_end_state_valid": torch.tensor([True]),
        "skill_start_focus_uv": torch.tensor([[-.5, 0.]]),
        "skill_start_focus_valid": torch.tensor([True]),
        "skill_focus_uv": torch.tensor([[.5, 0.]]),
        "skill_focus_valid": torch.tensor([True]),
    }
    def populate():
        features = [torch.randn(1, 196, 8, requires_grad=True) for _ in range(2)]
        queries = torch.randn(1, 100, 4, requires_grad=True)
        model._final_agent_patch_tokens, model._final_wrist_patch_tokens = features
        model._final_dual_patch_query = queries[:, :2]
        return features, queries
    start_features, start_query = populate()
    start_loss, start_metrics = policy._skill_start_patch_alignment_loss(batch)
    assert model._final_agent_patch_tokens is model._final_wrist_patch_tokens is model._final_dual_patch_query is None
    current_features, current_query = populate()
    current_loss, _ = policy._wrist_patch_alignment_loss(batch, top_k=1)
    torch.testing.assert_close(states[0], batch["skill_start_state"][:, :6])
    torch.testing.assert_close(states[1], batch["skill_decoder_state"][:, :6])
    assert start_metrics["wrist_patch/valid_fraction"] == 1.
    head_parameters = [model.agent_patch_align_head.query_proj.weight,
                       model.wrist_patch_align_head.query_proj.weight]
    start_grad = torch.autograd.grad(start_loss, head_parameters, retain_graph=True)
    current_grad = torch.autograd.grad(current_loss, head_parameters, retain_graph=True)
    assert all(g.abs().sum() > 0 for g in [*start_grad, *current_grad])
    (0.5 * (start_loss + current_loss)).backward()
    for parameter, a, b in zip(head_parameters, start_grad, current_grad):
        torch.testing.assert_close(parameter.grad, .5 * (a + b))
    assert all(feature.grad.abs().sum() > 0 for feature in [*start_features, *current_features])
    assert start_query.grad[:, :2].abs().sum() > 0
    assert current_query.grad[:, :2].abs().sum() > 0


def test_start_patch_loss_masks_out_of_view_wrist_and_top_targets():
    logits = torch.randn(1, 196, requires_grad=True)
    policy = object.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        architecture_label="fullskill_vis_1_firstlayer", dino_image_size=224,
        wrist_patch_align_target_sigma=.7, trains_agent_patch_alignment=True,
    )
    policy.model = SimpleNamespace(predict_training_camera_patch_logits=lambda: (logits, logits))
    batch = {
        "skill_start_state": torch.zeros(1, 8),
        "skill_decoder_state": torch.ones(1, 8),
        "skill_end_state": torch.tensor([[100., 0., .25]]),
        "skill_end_state_valid": torch.tensor([True]),
        "skill_start_focus_uv": torch.tensor([[3., 0.]]),
        "skill_start_focus_valid": torch.tensor([True]),
    }
    loss, metrics = policy._skill_start_patch_alignment_loss(batch)
    assert loss.item() == 0.
    assert metrics["wrist_patch/valid_fraction"] == metrics["agent_patch/valid_fraction"] == 0.
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() == 0.


@pytest.mark.parametrize("label", FULLSKILL_VIS_ARCH_LABELS)
def test_inference_cache_holds_start_and_refreshes_same_code_occurrences_per_row(label):
    policy = object.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(architecture_label=label, max_state_dim=8, n_action_steps=2)
    policy.model = Encoder()
    policy.reset()
    batch = {"skill_start_image": torch.ones(2, 3, 2, 2),
             "skill_start_wrist_image": torch.ones(2, 3, 2, 2),
             "skill_start_state_normalized": torch.ones(2, 8),
             "skill_start_cache_id": torch.tensor([0, 0])}
    code, goal = torch.tensor([3, 3]), torch.ones(2, 6)
    policy._prepare_start_vision(batch, code, goal, inference=True)
    first = policy.model._skill_start_latent.clone()
    batch["skill_start_image"] *= 5
    batch["skill_start_state_normalized"] *= 2
    policy._prepare_start_vision(batch, code, goal, inference=True)
    assert len(policy.model.calls) == 1
    torch.testing.assert_close(policy.model._skill_start_latent, first)
    batch["skill_start_cache_id"][1] = 1
    policy._prepare_start_vision(batch, code, goal, inference=True)
    assert len(policy.model.calls) == 2
    assert policy.model.calls[-1][0].shape[0] == 1
    torch.testing.assert_close(policy.model._skill_start_latent[0], first[0])
    assert not torch.equal(policy.model._skill_start_latent[1], first[1])
    policy.reset()
    policy._prepare_start_vision(batch, code, goal, inference=True)
    assert len(policy.model.calls) == 3


@pytest.mark.parametrize("label", FULLSKILL_VIS_ARCH_LABELS)
def test_training_start_encoding_keeps_gradients_and_requires_consistent_metadata(label):
    policy = object.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(architecture_label=label, max_state_dim=8)
    policy.model = Encoder()
    batch = {"skill_start_image": torch.ones(1, 3, 2, 2),
             "skill_start_wrist_image": torch.ones(1, 3, 2, 2),
             "skill_start_state_normalized": torch.ones(1, 8)}
    policy._prepare_start_vision(batch, torch.tensor([3]), torch.ones(1, 6))
    policy.model._skill_start_latent.sum().backward()
    assert policy.model.weight.grad.abs() > 0
    del batch["skill_start_state_normalized"]
    with pytest.raises(KeyError, match="normalized skill-start"):
        policy._prepare_start_vision(batch, torch.tensor([3]), torch.ones(1, 6))
