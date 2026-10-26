"""Diffaux5/6: independent start-Gemma XYZ and all three bottleneck targets."""

import pytest
import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.configs.uv_fullskill import UV_BSPLINE_DUAL_XYZ_LABELS as LABELS
from lerobot.policies.skill_expert import fullskill_bspline, modeling_skill_expert as impl
from lerobot.policies.skill_expert.fullskill_bspline import FullskillBsplineSkillExpert
from lerobot.policies.skill_expert.wrist_patch_alignment import WristPatchAlignmentHead
from test_cond_sdpa import _tiny_model
from test_fullskill_bspline import tiny_model
from test_stage1_vsa import _skill_config
from test_uv_fullskill_bspline import policy_for
from test_uv_gt_xyz import goal_batch


@pytest.mark.parametrize("label", LABELS)
def test_dual_xyz_identity_roundtrip_and_warm_start(tmp_path, label):
    config = _skill_config(label)
    assert config.uses_triple_auxiliary_prediction and config.uses_gemma_xyz_readout
    assert config.uses_start_xyz_auxiliary and config.uses_separate_vision_tower_xyz
    assert config.uses_start_predicted_expert_xyz and config.uses_gt_expert_xyz
    assert config.uses_joint_bottleneck_self_attention and not config.uses_uv_readout_token
    assert config.agent_end_state_loss_weight == .1
    assert config.vision_tower_xyz_loss_weight == 1.
    assert config.uses_expert_uv_and_xyz == ("diffaux6" in label)
    assert config.uses_current_bspline_vision == ("_current_preskill_" in label)
    config.save_pretrained(tmp_path)
    loaded = PreTrainedConfig.from_pretrained(tmp_path)
    assert loaded.architecture_revision == config.architecture_revision
    assert loaded.uses_separate_vision_tower_xyz
    for key in ("vision_tower_xyz_head.1.weight", "agent_end_state_head.1.weight", "xyz_readout_token"):
        assert impl._allowed_pi05_missing_key("model." + key, loaded)
    if config.uses_expert_uv_and_xyz:
        assert impl._allowed_pi05_missing_key("model.expert_uv_condition.0.weight", loaded)
    baseline = label.replace("diffaux5", "diffaux3").replace("diffaux6", "diffaux3")
    with pytest.raises(ValueError, match="revision"):
        _skill_config(label, architecture_revision=_skill_config(baseline).architecture_revision)


@pytest.mark.parametrize("label", LABELS)
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_both_heads_camera_scopes_joint_attention_and_gemma_only_inference(monkeypatch, label, checkpoint, dtype):
    model = tiny_model(monkeypatch, 3, uv=True, diffaux=True, architecture_label=label)
    core = _tiny_model(monkeypatch, "sdpa", checkpoint, model_class=FullskillBsplineSkillExpert)
    model.cond_encoder = core.cond_encoder.to(dtype=dtype)
    model.layerwise_condition_memory_norm = core.layerwise_condition_memory_norm
    model.layerwise_condition_readers = core.layerwise_condition_readers
    model.layerwise_latent_queries = nn.Parameter(torch.randn(100, 16))
    model._gradient_checkpointing = checkpoint
    model.agent_patch_align_head = WristPatchAlignmentHead(16, 32)
    model.wrist_patch_align_head = WristPatchAlignmentHead(16, 32)
    assert model.start_bridge_layer == 8
    assert model.current_bridge_layers == ((17,) if label.endswith("_last") else tuple(range(9, 18)))
    assert model.agent_end_state_head[-1].in_features == 16
    assert model.vision_tower_xyz_head[-1].in_features == 32
    reads, mixing, token_lengths = [], [], []
    for reader in model.layerwise_condition_readers:
        reader.cross_attention.register_forward_pre_hook(
            lambda module, args: reads.append((args[0].shape[1], args[1].shape[1])))
        reader.self_attention.register_forward_pre_hook(lambda module, args: mixing.append(args[0].shape[1]))
    model.cond_encoder.model.layers[0].self_attn.register_forward_pre_hook(
        lambda module, args: token_lengths.append(args[0].shape[1]))
    images = torch.randn(2, 10, 32, dtype=dtype, requires_grad=True)
    model._encode_layerwise_latents(images, torch.zeros(2, 32, dtype=dtype))
    assert token_lengths[0] == 11  # extra Gemma token, outside 100 bottlenecks
    assert reads == [(1, 5), (1, 5), (1, 10), (97, 10)] * 2
    assert mixing == [100, 100]
    tower_xyz = model.predict_training_gemma_xyz()
    tower_xyz.square().mean().backward(retain_graph=True)
    assert model.layerwise_latent_queries.grad is None
    assert model.xyz_readout_token.grad.abs().sum() > 0
    assert images.grad[:, :5].abs().sum() > 0 and images.grad[:, 5:].abs().sum() > 0
    top, wrist, bottleneck_xyz = model.predict_training_top_wrist_patch_and_xyz()
    assert top.shape == wrist.shape == (2, 4)
    assert tower_xyz.shape == bottleneck_xyz.shape == (2, 3)
    (top.square().mean() + wrist.square().mean() + bottleneck_xyz.square().mean()).backward()
    assert model.agent_end_state_head[-1].weight.grad.abs().sum() > 0
    assert model.vision_tower_xyz_head[-1].weight.grad.abs().sum() > 0
    assert model.layerwise_latent_queries.grad.abs().sum() > 0
    assert model._final_dual_patch_query is None and model._final_xyz_readout is None
    model.eval()
    # Poison the bottleneck predictor: inference must exclusively read Gemma.
    with torch.no_grad():
        model.agent_end_state_head[-1].bias.fill_(float("nan"))
        model._encode_layerwise_latents(images.detach(), torch.zeros(2, 32, dtype=dtype))
        assert torch.isfinite(model.predict_start_xyz()).all()
    assert model._inference_bottleneck_xyz is None


@pytest.mark.parametrize("label", LABELS)
def test_training_exact_loss_coefficients_and_start_only_tower_gradient(monkeypatch, label):
    current = "_current_preskill_" in label

    class Trainer(nn.Module):
        def __init__(self):
            super().__init__()
            self.start_tower_xyz = nn.Parameter(torch.ones(2, 3))
            self.current_tower_xyz = nn.Parameter(torch.full((2, 3), 7.))
            self.start_bottleneck_xyz = nn.Parameter(torch.full((2, 3), 3.))
            self.current_bottleneck_xyz = nn.Parameter(torch.full((2, 3), 2.))
            self.start_patch = nn.Parameter(torch.full((2, 4), 2.))
            self.current_patch = nn.Parameter(torch.ones(2, 4))
            self.action_scale = nn.Parameter(torch.tensor(2.))
            self.plan_scale = nn.Parameter(torch.tensor(2.))
            self.bspline_read_gate = nn.Parameter(torch.tensor(.1))
            self._last_vsa_debug_stats = {}
            self.head_inputs = []
            self.held_start = None

        def encode_skill_start(self, images, state, code, goal):
            self.phase = "start"
            return state

        def set_skill_start_latent(self, value):
            self.held_start = value

        def predict_training_gemma_xyz(self):
            self.head_inputs.append(self.phase)
            assert self.phase == "start"
            return self.start_tower_xyz

        def predict_training_top_wrist_patch_and_xyz(self, *, include_xyz=True):
            assert include_xyz, "Adding Gemma XYZ must not disable bottleneck XYZ."
            start = self.phase == "start"
            patch = self.start_patch if start else self.current_patch
            xyz = self.start_bottleneck_xyz if start else self.current_bottleneck_xyz
            return patch, patch, xyz

        def forward(self, images, state, code, actions, **kwargs):
            self.phase = "current"
            self.plan = self.plan_scale.expand(actions.shape[0], 30, 8)
            return self.action_scale.expand_as(actions)

        def take_bspline_prediction(self):
            return self.plan

    policy = policy_for(label)
    policy.config = _skill_config(label, chunk_size=2, n_action_steps=1,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))})
    policy.config.chunk_end_state_q01, policy.config.chunk_end_state_q99 = [0.] * 8, [2.] * 8
    policy.model = Trainer()
    monkeypatch.setattr(policy, "_prepare_preskill_memory", lambda batch: None)
    monkeypatch.setattr(policy, "_collect_images", lambda batch: [])
    monkeypatch.setattr(impl, "patch_labels", lambda *a, **kw: (
        torch.zeros(2, dtype=torch.long), None, torch.ones(2, dtype=torch.bool)))
    monkeypatch.setattr(impl, "wrist_patch_alignment_loss", lambda logits, *a, **kw: (
        logits.square().mean(), {}))
    batch = goal_batch()
    batch.update({"action": torch.zeros(2, 2, 7), "observation.state": torch.zeros(2, 8),
        "skill_code": torch.tensor([1, 2]), "skill_end_state": torch.ones(2, 8),
        "skill_decoder_state": torch.zeros(2, 8), "skill_start_state_normalized": torch.full((2, 8), .25),
        "skill_start_image": torch.ones(2, 3, 2, 2),
        "skill_start_wrist_image": torch.full((2, 3, 2, 2), 2.),
        "skill_start_focus_uv": torch.zeros(2, 2), "skill_start_focus_valid": torch.ones(2, dtype=torch.bool),
        "skill_bspline_states": torch.ones(2, 30, 8)})
    if current:
        batch["preskill_valid"] = torch.tensor([False, True])
    loss, metrics = policy(batch)
    # Action=4, Bspline=4. Current bottleneck aux=.1*(1+1+4)=.6.
    # Held-start aux averages that with .1*(4+4+9)=1.7. Tower XYZ=1 separately.
    aux = .6 if current else 1.15
    torch.testing.assert_close(loss, torch.tensor(9. + aux))
    assert metrics["patch_alignment/weighted"] == pytest.approx(aux)
    assert metrics["vision_tower_xyz_start/loss"] == 1.
    assert metrics["vision_tower_xyz_start/weight"] == 1.
    assert metrics["vision_tower_xyz_start/weighted"] == 1.
    assert metrics["agent_end_state/weight"] == .1
    assert metrics["bottleneck_xyz/weight"] == .1
    assert policy.model.head_inputs == ["start"]
    loss.backward()
    torch.testing.assert_close(policy.model.start_tower_xyz.grad, torch.full((2, 3), 1. / 3))
    assert policy.model.current_tower_xyz.grad is None
    assert policy.model.current_bottleneck_xyz.grad.abs().sum() > 0
    assert (policy.model.start_bottleneck_xyz.grad is None) is current
    assert policy.model.action_scale.grad.abs() > 0 and policy.model.plan_scale.grad.abs() > 0


@pytest.mark.parametrize("label", LABELS)
def test_expert_uv_uses_same_noisy_projection_with_whole_adapter_validity_mask(monkeypatch, label):
    policy = policy_for(label)
    calls = []
    monkeypatch.setattr(torch, "rand_like", lambda x: calls.append(True) or torch.full_like(x, .75))
    packet = policy._skill_delta_goal(goal_batch(), require_valid=True)
    assert calls == [True]
    model = tiny_model(monkeypatch, 3, uv=True, diffaux=True, architecture_label=label)
    model.start_pose_condition = nn.Linear(3, 32)
    model._time_condition = lambda t: torch.zeros(t.shape[0], 32)
    monkeypatch.setattr(fullskill_bspline._LITGoalFreeCondMixin, "_project_condition_state",
                        lambda self, state, *args: torch.zeros(state.shape[0], 32))
    seen_cond, seen_xyz, seen_uv = [], [], []
    model.cond_end_pose_condition.register_forward_pre_hook(lambda m, a: seen_cond.append(a[0]))
    model.end_pose_condition.register_forward_pre_hook(lambda m, a: seen_xyz.append(a[0]))
    if policy.config.uses_expert_uv_and_xyz:
        model.expert_uv_condition.register_forward_pre_hook(lambda m, a: seen_uv.append(a[0]))
        nn.init.normal_(model.expert_uv_condition[-1].weight, std=.1)
    model._project_condition_state(torch.zeros(2, 32), end_pose=packet)
    cond = model._expert_condition(torch.ones(2), end_pose=packet)
    torch.testing.assert_close(seen_cond[0], packet[:, :2])
    torch.testing.assert_close(seen_xyz[0], packet[:, 6:9])
    if seen_uv:
        torch.testing.assert_close(seen_uv[0], seen_cond[0])
    changed = packet.clone(); changed[:, :2] += .2
    cond2 = model._expert_condition(torch.ones(2), end_pose=changed)
    assert torch.equal(cond, cond2) is (not policy.config.uses_expert_uv_and_xyz)
    if policy.config.uses_expert_uv_and_xyz:
        packet[:, 2] = 0; changed[:, 2] = 0
        torch.testing.assert_close(model._expert_condition(torch.ones(2), end_pose=packet),
                                   model._expert_condition(torch.ones(2), end_pose=changed))
        assert torch.equal(model._action_layer_condition(9, cond), model._action_layer_condition(9, cond2))
