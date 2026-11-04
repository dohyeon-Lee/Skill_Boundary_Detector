"""wovis trains current vision once while retaining action/Fullskill/aux3 gradients."""
import pytest
import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.configs.uv_fullskill import PI05_VLM_WOVIS_LEGACY_REVISIONS as LABELS
from lerobot.configs.uv_fullskill import VANILA_PI05_VLM_BASE_LABELS
from lerobot.policies.skill_expert.modeling_skill_expert import SkillExpertPolicy
from test_pi05_vlm_vsa import tiny_vlm_model
from test_stage1_vsa import _skill_config


@pytest.mark.parametrize("label", (*LABELS, *VANILA_PI05_VLM_BASE_LABELS))
@pytest.mark.parametrize("checkpoint,dtype", [(False, torch.float32), (True, torch.bfloat16)])
def test_training_without_start_images_encodes_current_once_and_keeps_all_objectives(
    monkeypatch, label, checkpoint, dtype,
):
    model = tiny_vlm_model(monkeypatch, label, checkpoint, dtype, bridge_layers=17)
    model.config.vsa_debug_schedule = []
    model.config.vsa_debug_every = 0
    model.config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(32,))}
    assert model.config.omits_start_vision and not model.config.uses_symmetric_start_vision
    policy = SkillExpertPolicy.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.config, policy.model = model.config, model
    images = [torch.rand(2, 3, 4, 4)] * 2
    policy._collect_images = lambda batch: images
    goal = torch.tensor([[.2, -.2, 1., .1, .2, .3]] * 2)
    policy._skill_delta_goal = lambda batch, **kwargs: goal
    monkeypatch.setattr(model, "encode_skill_start", lambda *a, **k: pytest.fail("wovis encoded a start image"))
    encoded, aux_calls = [], []
    original = model._condition_tokens
    def encode(*args, **kwargs):
        encoded.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(model, "_condition_tokens", encode)
    def current_aux(batch, **kwargs):
        aux_calls.append(1)
        top, wrist, xyz = model.predict_training_top_wrist_patch_and_xyz()
        return .1 * sum(value.float().square().mean() for value in (top, wrist, xyz)), {}
    monkeypatch.setattr(policy, "_wrist_patch_alignment_loss", current_aux)
    # No start images or start-focus targets, but start XYZ and skill targets remain.
    batch = {"action": torch.randn(2, 10, 32), "observation.state": torch.zeros(2, 32),
             "skill_code": torch.tensor([1, 2]), "skill_start_state": torch.zeros(2, 8),
             "skill_start_state_normalized": torch.zeros(2, 8),
             "skill_bspline_states": torch.randn(2, 30, 8)}
    loss, metrics = policy(batch)
    assert len(encoded) == len(aux_calls) == 1
    planner = model.config.uses_independent_skill_planner
    assert ("bspline/loss" in metrics) is planner
    assert "patch_alignment/loss" in metrics
    assert not any(key.startswith("start_") for key in metrics)
    assert "patch_alignment/symmetric_start" not in metrics
    assert loss.item() == pytest.approx(
        metrics["action_loss"] + model.config.bspline_loss_weight * metrics.get("bspline/loss", 0.)
        + metrics["patch_alignment/loss"], rel=.01,
    )
    loss.backward()
    parameters = [model.action_out_proj.weight,
                      model.cond_encoder.model.layers[0].self_attn.q_proj.weight,
                      model.cond_encoder.vlm.vision_tower.weight,
                      model.agent_patch_align_head.query_proj.weight,
                      model.wrist_patch_align_head.query_proj.weight,
                      model.agent_end_state_head[-1].weight]
    if planner:
        parameters.append(model.bspline_noise_input.weight)
    for parameter in parameters:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    model.eval()
    generated = model.sample_actions(images, batch["observation.state"], batch["skill_code"],
                                     num_steps=1, end_pose=goal)
    assert generated.shape == batch["action"].shape and torch.isfinite(generated).all()


@pytest.mark.parametrize("label,revision", LABELS.items())
def test_legacy_checkpoint_preserves_its_training_contract(tmp_path, label, revision):
    config = _skill_config(label, architecture_revision=revision)
    assert not config.omits_start_vision and config.uses_symmetric_start_vision
    config.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.architecture_revision == revision and not restored.omits_start_vision
    assert _skill_config(label).omits_start_vision
