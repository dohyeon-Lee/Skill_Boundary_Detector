"""Numerical contracts against the pinned native N1.7 expert, without GPU/2B VLM."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.skill_groot import model as adapter
from lerobot.policies.skill_groot.native import native_modules


class TinyVLM(nn.Module):
    def __init__(self, backbone, *_):
        super().__init__()
        self.backbone = backbone
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(2)])
        self.image_input = nn.Linear(3, 16)

    def configure_checkpointing(self, **kwargs):
        pass

    def encode(self, images, texts):
        self.last_texts = texts
        # Four spatial patches per camera plus one condition token.
        tokens = [self.image_input(torch.nn.functional.adaptive_avg_pool2d(x, (2, 2)).flatten(2).transpose(1, 2)) for x in images]
        tokens.append(torch.ones(len(texts), 1, 16, device=images[0].device))
        hidden, layers = torch.cat(tokens, 1), []
        for layer in self.language_model.layers:
            hidden = hidden + layer(hidden).tanh()
            layers.append(hidden)
        valid = torch.ones(hidden.shape[:2], dtype=torch.bool)
        image = valid.clone()
        image[:, -1] = False
        cameras = [torch.zeros_like(valid), torch.zeros_like(valid)]
        cameras[0][:, :4] = True
        cameras[1][:, 4:8] = True
        return layers, valid, image, cameras


def tiny(monkeypatch, variant="groot_integrate", *, freeze=False, checkpoint=False, options=None):
    torch.manual_seed(5)
    module, _ = native_modules()
    config = module.GR00TN17Config(
        backbone_embedding_dim=16, input_embedding_dim=16, hidden_size=16,
        max_action_dim=8, max_state_dim=8, max_num_embodiments=3,
        vl_self_attention_cfg={"num_layers": 0}, state_dropout_prob=0,
        diffusion_model_cfg=dict(num_layers=8, num_attention_heads=2, attention_head_dim=8,
            dropout=0., final_dropout=False, norm_type="ada_norm", output_dim=16,
            interleave_self_attention=True, positional_embeddings=None),
    )
    head = module.GR00TN17ActionHead(config)
    saved = {k: v.detach().clone() for k, v in head.state_dict().items()}
    monkeypatch.setattr(adapter, "GrootVLM", TinyVLM)
    options = options or SimpleNamespace(skillvla_backbone=variant, visual_bottleneck_width=16,
        skill_fsq_levels=[3, 3, 3], skill_plan_state_dim=7,
        vlm_termination_enabled=True, newtask_ft_enabled=freeze,
        newtask_ft_freeze_expert=freeze, gradient_checkpointing=checkpoint)
    model = adapter.GrootSkillModel(options, native=SimpleNamespace(
        backbone=nn.Identity(), action_head=head, config=config))
    for key, value in model.expert.state_dict().items():
        torch.testing.assert_close(value, saved[key], rtol=0, atol=0)
    return model


def inputs(model):
    images = [torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32)]
    code, uv = torch.tensor([1, 4]), torch.tensor([[.1, -.2], [-.3, .4]])
    banks, uv = model.observe(images, code, uv)
    state, ids = model.state_token(torch.randn(2, 8))
    return banks, uv, code, state, ids


@pytest.mark.parametrize("variant", ["groot_integrate", "groot_seperate"])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_native_action_and_plan_training_gradients(monkeypatch, variant, checkpoint):
    model = tiny(monkeypatch, variant, checkpoint=checkpoint)
    banks, uv, code, state, ids = inputs(model)
    action = model.flow_loss(torch.randn(2, 3, 7), state, ids, code, uv, banks).mean()
    plan = model.flow_loss(torch.randn(2, 5, 7), state, ids, code, uv, banks, planner=True).mean()
    # Fullskill shares the first half only. Action trains both halves.
    late = model.expert.model.transformer_blocks[-1].attn1.to_q.weight
    assert torch.autograd.grad(plan, late, allow_unused=True, retain_graph=True)[0] is None
    (action + plan).backward()
    for parameter in (late, model.vlm.image_input.weight, model.bottlenecks[0].queries,
                      model.bottlenecks[1].queries, model.plan_input.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("variant", ["groot_integrate", "groot_seperate"])
def test_expert_cross_attention_memory_contract(monkeypatch, variant):
    model = tiny(monkeypatch, variant)
    banks, uv, code, state, ids = inputs(model)
    captured = []
    def record(module, args, kwargs):
        captured.append((kwargs["encoder_hidden_states"], kwargs["attention_mask"].clone()))
    handles = [layer.attn1.register_forward_pre_hook(record, with_kwargs=True)
               for layer in model.expert.model.transformer_blocks[::2]]
    model.velocity(torch.randn(2, 3, 7), state, ids, torch.tensor([100, 200]), code, uv, banks)
    for handle in handles:
        handle.remove()
    for index, (memory, mask) in enumerate(captured):
        assert memory is banks[0 if index < 2 else 1]
        assert memory.shape == (2, 100, 16)  # raw VLM has 9 tokens
        if variant == "groot_integrate":
            assert mask.all()
        else:
            assert (mask.sum(-1) == 50).all()
            assert mask[:, 50:].all() if index % 2 == 0 else mask[:, :50].all()


def test_termination_does_not_backpropagate_to_vlm_or_expert(monkeypatch):
    model = tiny(monkeypatch)
    inputs(model)
    logits, cache = model.take_vlm_termination_outputs()
    past = model.vlm_termination_from_image_tokens(cache, None, torch.tensor([2, 3]), torch.zeros(2, 6))
    (logits.square().mean() + past.square().mean()).backward()
    assert all(p.grad is None for p in model.vlm.parameters())
    assert all(p.grad is None for p in model.expert.parameters())
    assert all(p.grad is None for p in model.bottlenecks.parameters())
    assert model.vlm_termination_bottleneck.queries.grad.abs().sum() > 0
    assert model.vlm_termination_bottleneck.head[-1].weight.grad.abs().sum() > 0


def test_separate_visual_tokens_can_learn_from_condition_tokens():
    torch.manual_seed(1)
    bottle = adapter.Bottleneck(2, 16, width=16, separate=True)
    layers = [torch.randn(1, 9, 16, requires_grad=True) for _ in range(2)]
    valid = torch.ones(1, 9, dtype=torch.bool)
    image = valid.clone()
    image[:, -1] = False
    cameras = [image.clone(), image.clone()]
    cameras[0][:, 4:] = False
    cameras[1][:, :4] = False
    visual, _ = bottle(layers, valid, image, cameras)
    visual[:, :3].square().mean().backward()
    assert layers[-1].grad[:, -1].abs().sum() > 0


def test_ft_freezes_motion_core_but_keeps_adaptation_trainable(monkeypatch):
    model = tiny(monkeypatch, freeze=True)
    model.train()
    assert not model.expert.training
    assert not any(p.requires_grad for p in model.expert.parameters())
    assert not any(p.requires_grad for p in model.expert.model.parameters())
    assert not any(p.requires_grad for p in model.expert.action_decoder.parameters())
    banks, uv, code, state, ids = inputs(model)
    loss = model.flow_loss(torch.randn(2, 3, 7), state, ids, code, uv, banks).mean()
    loss = loss + model.flow_loss(torch.randn(2, 5, 7), state, ids, code, uv, banks, planner=True).mean()
    loss.backward()
    assert all(p.grad is None for p in model.expert.model.parameters())
    assert model.vlm.image_input.weight.grad.abs().sum() > 0
    assert model.bottlenecks[0].queries.grad.abs().sum() > 0


def test_eval_camera_crop_preserves_uv_geometry():
    cache = adapter.crop_cameras([torch.rand(2, 3, 256, 256)], training=False)
    torch.testing.assert_close(cache.uv(torch.zeros(2, 2)), torch.zeros(2, 2))
    point = torch.tensor([[13 / 128 - 1, 13 / 128 - 1]] * 2)
    torch.testing.assert_close(cache.uv(point), -torch.ones(2, 2))


def policy_config():
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.configs.uv_fullskill import DUAL_BOTTLENECK_TERM_DETACH_LABEL, UV_BSPLINE_REVISIONS
    from lerobot.policies.skill_expert.configuration_skill_expert import SkillExpertConfig
    label = DUAL_BOTTLENECK_TERM_DETACH_LABEL
    return SkillExpertConfig(
        device="cpu", skillvla_backbone="groot_integrate", visual_bottleneck_width=256,
        visual_bottleneck_tokens=100,
        architecture="layerwise_cond_bottleneck", architecture_label=label,
        vision_conditioning_mode="layerwise_cond_bottleneck_cross_attention",
        architecture_revision=UV_BSPLINE_REVISIONS[label], visual_bridge_last_n_layers=9,
        skill_flow_enabled=False, skill_flow_max_length=0, vlm_termination_enabled=True,
        chunk_end_state_q01=[-1.] * 8, chunk_end_state_q99=[1.] * 8,
        skill_fsq_levels=[3, 3, 3], skill_vocab_size=27, bspline_tokens=5,
        chunk_size=3, n_action_steps=2,
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (8,)),
            "observation.images.image": PolicyFeature(FeatureType.VISUAL, (3, 32, 32)),
            "observation.images.wrist_image": PolicyFeature(FeatureType.VISUAL, (3, 32, 32))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
    )


def test_policy_loss_and_saved_checkpoint_roundtrip(monkeypatch, tmp_path):
    from lerobot.policies.skill_groot import policy as implementation
    from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy
    from lerobot.policies.skillVLA.vlm_termination_supervision import (
        DISTANCE, TERMINAL, PAST_CODES, PAST_VALID, PAST_UV, PAST_UV_VALID, PAST_TERMINAL, PAST_DISTANCE)
    config = policy_config()
    monkeypatch.setattr(implementation, "GrootSkillModel", lambda cfg: tiny(monkeypatch, options=cfg))
    policy = implementation.GrootSkillExpertPolicy(config)
    batch = {
        "observation.state": torch.randn(2, 8), "skill_decoder_state": torch.randn(2, 8),
        "observation.images.image": torch.rand(2, 3, 32, 32),
        "observation.images.wrist_image": torch.rand(2, 3, 32, 32),
        "action": torch.randn(2, 3, 7), "skill_code": torch.tensor([1, 4]),
        "skill_code_true": torch.tensor([2, 5]),  # unjittered skill must not override selected code
        "skill_focus_uv": torch.zeros(2, 2), "skill_end_state": torch.randn(2, 8),
        "skill_end_state_valid": torch.ones(2, dtype=torch.bool),
        "skill_bspline_states": torch.randn(2, 5, 8),
        DISTANCE: torch.tensor([1., -1.]), TERMINAL: torch.zeros(2, dtype=torch.bool),
        PAST_CODES: torch.tensor([[0], [1]]), PAST_VALID: torch.tensor([[True], [False]]),
        PAST_UV: torch.zeros(2, 1, 2), PAST_UV_VALID: torch.ones(2, 1, dtype=torch.bool),
        PAST_DISTANCE: torch.full((2, 1), -1.), PAST_TERMINAL: torch.zeros(2, 1, dtype=torch.bool),
    }
    torch.testing.assert_close(policy._inputs(batch)[2], batch["skill_code"])
    loss, metrics = policy(batch)
    assert torch.isfinite(loss)
    assert all(name in metrics for name in ("action_loss", "bspline/loss", "vlm_termination/loss"))
    loss.backward()
    assert policy.model.vlm.image_input.weight.grad.abs().sum() > 0
    assert policy.isolated_main_optimizer_grad_groups()["vlm_termination"]
    policy.save_pretrained(tmp_path)
    restored = SkillExpertPolicy.from_pretrained(tmp_path)
    assert isinstance(restored, implementation.GrootSkillExpertPolicy)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    restored.eval()
    assert restored.predict_action_chunk(batch).shape == (2, 3, 7)
    assert restored.predict_termination_probability(batch).shape == (2,)
