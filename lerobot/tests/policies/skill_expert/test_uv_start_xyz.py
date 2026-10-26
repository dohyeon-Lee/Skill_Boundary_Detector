"""Start-only XYZ supervision must preserve patch weights and Expert routing."""

from dataclasses import asdict

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.configs.uv_fullskill import UV_BSPLINE_DIFFAUX4_START_XYZ_BASE_LABELS as START_MODELS
from lerobot.policies.skill_expert import modeling_skill_expert as impl
from test_fullskill_bspline import tiny_model
from test_stage1_vsa import _skill_config
from test_uv_fullskill_bspline import policy_for
from test_uv_gt_xyz import goal_batch


@pytest.mark.parametrize("label,baseline", START_MODELS.items())
def test_start_xyz_keeps_expected_expert_vision_layers_and_distinct_revision(monkeypatch, label, baseline):
    config, original = _skill_config(label), _skill_config(baseline)
    before, after = asdict(original), asdict(config)
    assert {key for key in before if before[key] != after[key]} == {
        "architecture_label", "architecture_revision",
    }
    assert config.uses_start_xyz_auxiliary and not original.uses_start_xyz_auxiliary
    assert config.uses_gemma_xyz_readout and config.uses_gt_expert_xyz
    assert not config.uses_uv_readout_token
    with pytest.raises(ValueError, match="revision"):
        _skill_config(label, architecture_revision=original.architecture_revision)
    model = tiny_model(monkeypatch, 3, uv=True, diffaux=True, architecture_label=label)
    assert model.start_bridge_layer == 8
    assert model.current_bridge_layers == ((17,) if label.endswith("_last") else tuple(range(9, 18)))
    assert model.current_vision_route == ("final_last" if label.endswith("_last") else "final_all")


@pytest.mark.parametrize("label", START_MODELS)
def test_full_training_uses_start_xyz_only_and_preserves_patch_losses(monkeypatch, label):
    current = "_current_preskill_" in label

    class Trainer(nn.Module):
        def __init__(self):
            super().__init__()
            self.start_xyz = nn.Parameter(torch.ones(2, 3))
            self.current_xyz = nn.Parameter(torch.full((2, 3), 7.))
            self.start_patch = nn.Parameter(torch.full((2, 4), 2.))
            self.current_patch = nn.Parameter(torch.ones(2, 4))
            self.action_scale = nn.Parameter(torch.tensor(2.))
            self.plan_scale = nn.Parameter(torch.tensor(2.))
            self.bspline_read_gate = nn.Parameter(torch.tensor(.1))
            self._last_vsa_debug_stats = {}
            self.encodings, self.head_inputs = [], []
            self.held_start = None

        def encode_skill_start(self, images, state, code, goal):
            self.encodings.append((images, state, code, goal))
            self.phase = "start"
            return state

        def set_skill_start_latent(self, value):
            self.held_start = value

        def predict_training_gemma_xyz(self):
            self.head_inputs.append(self.phase)
            assert self.phase == "start"
            return self.start_xyz

        def predict_training_top_wrist_patch_and_xyz(self, *, include_xyz=True):
            assert not include_xyz, "Patch passes must never call the XYZ head."
            patch = self.start_patch if self.phase == "start" else self.current_patch
            return patch, patch, None

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
    preskill_calls = []
    monkeypatch.setattr(policy, "_prepare_preskill_memory", lambda batch: preskill_calls.append(batch))
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
    # Action MSE=4, normalized plan MSE=4; XYZ MSE=1 contributes the full .1.
    aux = .3 if current else .6
    torch.testing.assert_close(loss, torch.tensor(8. + aux))
    assert metrics["patch_alignment/weighted"] == pytest.approx(aux)
    assert metrics["start_agent_end_state/raw_loss"] == 1.
    assert metrics["start_agent_end_state/weighted"] == pytest.approx(.1)
    assert "agent_end_state/raw_loss" not in metrics
    assert ("start_agent_patch/raw_loss" in metrics) is (not current)
    assert policy.model.head_inputs == ["start"]
    assert len(policy.model.encodings) == 1 and len(preskill_calls) == 1
    images, state, code, goal = policy.model.encodings[0]
    torch.testing.assert_close(images[0], batch["skill_start_image"])
    torch.testing.assert_close(images[1], batch["skill_start_wrist_image"])
    torch.testing.assert_close(state[:, :8], batch["skill_start_state_normalized"])
    assert (policy.model.held_start is None) is current
    loss.backward()
    torch.testing.assert_close(policy.model.start_xyz.grad, torch.full((2, 3), .1 / 3))
    assert policy.model.current_xyz.grad is None
    assert policy.model.current_patch.grad.abs().sum() > 0
    assert (policy.model.start_patch.grad is None) is current
    assert policy.model.action_scale.grad.abs() > 0 and policy.model.plan_scale.grad.abs() > 0


@pytest.mark.parametrize("label", START_MODELS)
def test_readout_skips_current_xyz_head_and_releases_cache(monkeypatch, label):
    model = tiny_model(monkeypatch, 3, uv=True, diffaux=True, architecture_label=label)
    class PatchHead(nn.Module):
        def forward(self, query, patches):
            return query[:, 0, :4] + patches[:, 0, :4]
    model.agent_patch_align_head = PatchHead()
    model.wrist_patch_align_head = PatchHead()
    model._final_dual_patch_query = torch.randn(2, 2, 16, requires_grad=True)
    model._final_agent_patch_tokens = torch.randn(2, 4, 16, requires_grad=True)
    model._final_wrist_patch_tokens = torch.randn(2, 4, 16, requires_grad=True)
    model._final_xyz_readout = torch.randn(2, 32, requires_grad=True)
    calls = []
    model.agent_end_state_head.register_forward_hook(lambda *a: calls.append(True))
    top, wrist, xyz = model.predict_training_top_wrist_patch_and_xyz(include_xyz=False)
    assert xyz is None and calls == []
    assert model._final_xyz_readout is None and model._final_dual_patch_query is None
    (top.square().mean() + wrist.square().mean()).backward()
    assert model.agent_end_state_head[-1].weight.grad is None
