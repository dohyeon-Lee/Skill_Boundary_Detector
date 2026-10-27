"""Independent start plans: topology, gradients, flow objective and occurrence caches."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.configs.uv_fullskill import UV_BSPLINE_SKILL_NAMES
from lerobot.policies.skill_expert.independent_skill_planner import IndependentSkillPlannerExpert
from lerobot.policies.skill_expert.modeling_skill_expert import (
    SkillExpertPolicy, _allowed_pi05_missing_key, apply_newtask_ft_freeze,
)
from test_fullskill_bspline import tiny_model
from test_stage1_vsa import _skill_config
from test_uv_inference_start_xyz import batch as rollout_batch
from test_uv_inference_start_xyz import rollout_policy


LABELS = tuple(UV_BSPLINE_SKILL_NAMES)
READER_LABELS = tuple(label for label in LABELS if not label.endswith("_none"))
NONE_LABEL = "uv_bspline_skill_denoising_none"


def planner(monkeypatch, label, *, checkpoint=False, dtype=torch.float32):
    template = tiny_model(monkeypatch, 3, uv=True, diffaux=True, architecture_label=label,
                          checkpoint=checkpoint, start_position="lastlayer", current_route="_lastvis")
    config = template.config
    config.skill_planner_mode = label.removeprefix("uv_bspline_skill_").removesuffix("_none")
    config.reads_bspline_plan = not label.endswith("_none")
    config.chunk_size, config.max_action_dim, config.num_inference_steps = 10, 32, 2
    config.time_sampling_beta_alpha, config.time_sampling_beta_beta = 1.5, 1.0
    config.time_sampling_scale, config.time_sampling_offset = .999, .001
    model = IndependentSkillPlannerExpert(config)
    model.start_pose_condition = nn.Linear(3, 32)
    model._time_condition = lambda t: t[:, None].expand(-1, 32)
    model._skill_broadcasts = lambda code: (None, torch.zeros(len(code), 32))
    model.gemma_expert.to(dtype=dtype)
    model._bridge_condition_shift = torch.zeros(2, 32)
    model._bspline_time_condition = torch.zeros(2, 32)
    return model


def conditions():
    return torch.tensor([3, 4]), torch.tensor([[.1, -.2, 1., 0., 0., 0., .2, .3, .4]] * 2)


@pytest.mark.parametrize("label", LABELS)
def test_checkpoint_identity_aux3_contract_and_roundtrip(tmp_path, label):
    config = _skill_config(label)
    assert config.uses_independent_skill_planner and config.uses_triple_auxiliary_prediction
    assert config.uses_gt_expert_xyz and config.uses_start_predicted_expert_xyz
    assert config.uses_joint_bottleneck_self_attention
    assert not config.uses_current_bspline_vision and not config.uses_bspline_preskill_memory
    assert not config.uses_gemma_xyz_readout and not config.uses_separate_vision_tower_xyz
    assert not config.skill_flow_enabled  # not the old canonical action-flow objective
    assert config.reads_bspline_plan is (label != NONE_LABEL)
    config.save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.skill_planner_mode == config.skill_planner_mode


@pytest.mark.parametrize("label", READER_LABELS)
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prefix_shapes_action_read_and_attached_gradients(monkeypatch, label, checkpoint, dtype):
    torch.manual_seed(42)
    model = planner(monkeypatch, label, checkpoint=checkpoint, dtype=dtype)
    start = torch.randn(2, 100, 16, requires_grad=True)
    model.set_skill_start_latent(start)
    code, goal = conditions()
    prefix_lengths = []
    hook = model.gemma_expert.model.layers[0].self_attn.q_proj.register_forward_pre_hook(
        lambda module, args: prefix_lengths.append(args[0].shape[1]),
    )
    plan, prediction, flow_loss = model.generate_independent_plan(code, goal, target=torch.zeros(2, 30, 8))
    hook.remove()
    assert plan.shape == (2, 30, 16) and prediction.shape == (2, 30, 8)
    expected_tokens = 10 if model.planner_mode == "learnable_query" else 30
    assert prefix_lengths and set(prefix_lengths) == {expected_tokens}
    assert hasattr(model, "bspline_queries") is (model.planner_mode == "learnable_query")
    assert hasattr(model, "bspline_generator") is (model.planner_mode == "learnable_query")
    assert (flow_loss is not None) is (model.planner_mode == "denoising")
    model.set_independent_plan(plan, prediction)
    current = [torch.randn(2, 100, 16, requires_grad=True) for _ in range(18)]
    action_noise = torch.randn(2, 10, 32, dtype=dtype, requires_grad=True)
    output = model._run_expert_with_layerwise_latents(action_noise, torch.zeros(2, 32, dtype=dtype),
                                                     torch.zeros(2, 32, dtype=dtype), current)
    torch.testing.assert_close(model.take_bspline_prediction(), prediction)
    # Changing the next forward's mutable caches must not corrupt checkpoint backward.
    model.set_independent_plan(torch.zeros_like(plan), torch.zeros_like(prediction))
    model.set_skill_start_latent(None)
    output.float().square().mean().backward(retain_graph=True)
    assert action_noise.grad.abs().sum() > 0 and current[-1].grad.abs().sum() > 0
    assert start.grad.abs().sum() > 0
    assert model.gemma_expert.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    if model.planner_mode == "denoising":
        assert model.bspline_noise_input.weight.grad.abs().sum() > 0
        assert model.bspline_proprio_head[-1].weight.grad.abs().sum() > 0
    else:
        assert model.bspline_learnable_input.grad.abs().sum() > 0
        assert model.bspline_proprio_head[-1].weight.grad is None
    (flow_loss.mean() if flow_loss is not None else prediction.square().mean()).backward()
    assert model.bspline_proprio_head[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("label", READER_LABELS)
def test_plan_does_not_read_current_action_or_gt_target(monkeypatch, label):
    model = planner(monkeypatch, label).eval()
    start = torch.randn(2, 100, 16)
    model.set_skill_start_latent(start)
    code, goal = conditions()
    torch.manual_seed(5)
    a = model.generate_independent_plan(code, goal, target=torch.zeros(2, 30, 8))
    model._bridge_condition_shift = torch.randn(2, 32) * 100
    torch.manual_seed(5)
    b = model.generate_independent_plan(code, goal, target=torch.full((2, 30, 8), 100.))
    torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
    torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)
    model.set_skill_start_latent(start + 1)
    torch.manual_seed(5)
    c = model.generate_independent_plan(code, goal)
    assert not torch.allclose(a[0], c[0])


@pytest.mark.parametrize("label", READER_LABELS)
def test_inference_plan_cache_is_per_occurrence_not_per_action_or_gt(monkeypatch, label):
    policy = rollout_policy(monkeypatch, label)
    calls = []
    original = policy.model.generate_independent_plan

    def generate(code, goal, **kwargs):
        calls.append((code.clone(), goal.clone()))
        plan, prediction, loss = original(code, goal, **kwargs)
        return plan * len(calls), prediction * len(calls), loss

    policy.model.generate_independent_plan = generate
    inputs = rollout_batch()
    policy.predict_action_chunk(inputs)
    first = policy.model.plan.clone()
    for _ in range(3):
        inputs["observation.state"].add_(1)
        inputs["observation.images.image"].add_(1)
        policy.predict_action_chunk(inputs)
        torch.testing.assert_close(policy.model.plan, first)
    assert len(calls) == 1
    inputs["skill_start_cache_id"][0] += 1  # even for an identical FSQ code
    policy.predict_action_chunk(inputs)
    assert len(calls) == 2 and len(calls[-1][0]) == 1
    torch.testing.assert_close(policy.model.plan[1], first[1])
    assert not torch.equal(policy.model.plan[0], first[0])
    policy.reset()
    assert policy.model.plan is None
    policy.predict_action_chunk(inputs)
    assert len(calls) == 3


@pytest.mark.parametrize("label", LABELS)
def test_policy_uses_regression_or_flow_once_with_aux3_weights(label):
    class TrainingPlan(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_scale = nn.Parameter(torch.tensor(2.))
            self.plan_scale = nn.Parameter(torch.tensor(2.))
            self.flow_scale = nn.Parameter(torch.tensor(3.))
            self.bspline_read_gate = nn.Parameter(torch.tensor(.1))
            self._last_vsa_debug_stats = {}
            self.targets = []

        def encode_skill_start(self, images, state, code, goal):
            return torch.ones(len(code), 100, 16)

        def set_skill_start_latent(self, latent):
            self.start = latent

        def generate_independent_plan(self, code, goal, *, target=None):
            self.targets.append(target)
            if label == NONE_LABEL:
                return None, None, self.flow_scale.square().expand(len(code)) if target is not None else None
            prediction = self.plan_scale.expand(len(code), 30, 8)
            flow = self.flow_scale.square().expand(len(code)) if target is not None else None
            return self.plan_scale.expand(len(code), 30, 16), prediction, flow

        def set_independent_plan(self, plan, prediction):
            self.plan, self.prediction = plan, prediction

        def forward(self, images, state, code, actions, **kwargs):
            return torch.ones_like(actions) * self.action_scale

        def take_bspline_prediction(self):
            assert label != NONE_LABEL  # No generated prediction may be requested for `_none`.
            return self.prediction

    policy = SkillExpertPolicy.__new__(SkillExpertPolicy)
    nn.Module.__init__(policy)
    policy.config = _skill_config(label, chunk_size=2, n_action_steps=1,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))})
    policy.model = TrainingPlan()
    policy._collect_images = lambda batch: []
    policy._skill_delta_goal = lambda batch, **kwargs: conditions()[1][:1]
    policy._wrist_patch_alignment_loss = lambda *args, **kwargs: (torch.tensor(0.), {})
    policy._visual_auxiliary_loss = lambda *args, **kwargs: (torch.tensor(.3), {})
    inputs = {
        "action": torch.zeros(1, 2, 7), "observation.state": torch.zeros(1, 8),
        "skill_code": torch.tensor([1]), "skill_start_state": torch.zeros(1, 8),
        "skill_start_state_normalized": torch.ones(1, 8),
        "skill_start_image": torch.ones(1, 3, 2, 2), "skill_start_wrist_image": torch.ones(1, 3, 2, 2),
        "skill_start_focus_uv": torch.zeros(1, 2), "skill_start_focus_valid": torch.tensor([True]),
        "skill_bspline_states": torch.ones(1, 30, 8),
    }
    loss, metrics = policy(inputs)
    denoising = "_denoising" in label
    plan_loss = 9. if denoising else 1.
    assert loss.item() == pytest.approx(4. + plan_loss + .3)
    assert metrics["bspline/loss"] == plan_loss
    assert ("planner/flow_loss" in metrics) is denoising
    assert metrics["bspline/read_enabled"] == float(label != NONE_LABEL)
    assert ("bspline/mae_normalized" in metrics) is (label != NONE_LABEL)
    assert not any(key.startswith("vision_tower_xyz_start/") for key in metrics)
    torch.testing.assert_close(policy.model.targets[-1], inputs["skill_bspline_states"])
    loss.backward()
    assert policy.model.action_scale.grad.abs() > 0
    assert (policy.model.flow_scale.grad is not None) is denoising
    loss, metrics = policy._vsa_forward(inputs, action_only=True)
    assert loss.item() == 4. and "bspline/loss" not in metrics
    assert policy.model.targets[-1] is None


def test_flow_sign_and_clean_generation_use_separate_noise_from_teacher(monkeypatch):
    model = planner(monkeypatch, "uv_bspline_skill_denoising")
    model.set_skill_start_latent(torch.zeros(2, 100, 16))
    noises = [torch.zeros(2, 30, 8), torch.full((2, 30, 8), 3.)]
    model.sample_noise = lambda shape, device: noises.pop(0)
    model.sample_time = lambda batch, device: torch.full((batch,), .25)
    seen = []

    def readout(state, time, *args):
        seen.append((state.clone(), time.clone()))
        return torch.zeros(2, 30, 16), torch.ones(2, 30, 8)

    model._flow_readout = readout
    plan, clean, loss = model.generate_independent_plan(*conditions(), target=torch.full((2, 30, 8), 2.))
    torch.testing.assert_close(clean, torch.full((2, 30, 8), -1.))
    torch.testing.assert_close(seen[-1][0], torch.full((2, 30, 8), 2.25))
    assert not loss.any()  # teacher velocity = noise(3) - target(2) = 1


@pytest.mark.parametrize("label", LABELS)
@pytest.mark.parametrize("unfreeze", [False, True])
def test_ft_freeze_handles_direct_and_query_parameters(monkeypatch, label, unfreeze):
    model = planner(monkeypatch, label)
    apply_newtask_ft_freeze(model, SimpleNamespace(visual_bridge_last_n_layers=9, uses_bspline_plan=True,
                                                  newtask_ft_unfreeze_skill_layers=unfreeze))
    for name in ("bspline_queries", "bspline_learnable_input", "bspline_noise_input",
                 "bspline_plan_projection", "bspline_generator", "bspline_proprio_head"):
        item = getattr(model, name, None)
        if item is not None:
            parameters = [item] if isinstance(item, nn.Parameter) else list(item.parameters())
            assert all(parameter.requires_grad is unfreeze for parameter in parameters)


@pytest.mark.parametrize("label", LABELS)
def test_planner_weights_restore_and_pi05_warm_start_keys_are_allowed(monkeypatch, label):
    original = planner(monkeypatch, label).eval()
    restored = planner(monkeypatch, label).eval()
    restored.load_state_dict(original.state_dict(), strict=True)
    config = _skill_config(label)
    for key in original.state_dict():
        if key.startswith("bspline_"):
            assert _allowed_pi05_missing_key("model." + key, config)
    start = torch.randn(2, 100, 16)
    original.set_skill_start_latent(start)
    restored.set_skill_start_latent(start)
    torch.manual_seed(101)
    first = original.generate_independent_plan(*conditions())
    torch.manual_seed(101)
    second = restored.generate_independent_plan(*conditions())
    if label == NONE_LABEL:
        assert first == second == (None, None, None)
    else:
        torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
        torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_none_one_flow_forward_no_reader_and_separate_gradients(monkeypatch, checkpoint, dtype):
    model = planner(monkeypatch, NONE_LABEL, checkpoint=checkpoint, dtype=dtype)
    model.set_skill_start_latent(torch.randn(2, 100, 16, requires_grad=True))
    prefix_lengths = []
    hook = model.gemma_expert.model.layers[0].self_attn.q_proj.register_forward_pre_hook(
        lambda module, args: prefix_lengths.append(args[0].shape[1]),
    )
    plan, prediction, flow_loss = model.generate_independent_plan(*conditions(), target=torch.randn(2, 30, 8))
    hook.remove()
    assert prefix_lengths == [30]  # No iterative denoising or final-state re-encoding.
    assert plan is prediction is None and flow_loss.shape == (2,)
    model.set_independent_plan(plan, prediction)

    def forbidden(*args, **kwargs):
        raise AssertionError("Action must never read a fullskill plan in `_none`.")

    model.bspline_reader.forward = forbidden
    model.bspline_read_norm.forward = forbidden
    current = [torch.randn(2, 100, 16, requires_grad=True) for _ in range(18)]
    action = torch.randn(2, 10, 32, dtype=dtype, requires_grad=True)
    hidden = torch.randn(2, 10, 32, dtype=dtype)
    passed, missing = model._make_and_read_plan(hidden)
    assert passed is hidden and missing is None
    output = model._run_expert_with_layerwise_latents(
        action, torch.zeros(2, 32, dtype=dtype), torch.zeros(2, 32, dtype=dtype), current,
    )
    assert model._bspline_prediction is None
    output.float().square().mean().backward()
    assert action.grad.abs().sum() > 0 and current[-1].grad.abs().sum() > 0
    shared = model.gemma_expert.model.layers[0].self_attn.q_proj.weight
    assert shared.grad.abs().sum() > 0
    for module in (model.bspline_noise_input, model.bspline_plan_projection, model.bspline_proprio_head):
        assert all(parameter.grad is None for parameter in module.parameters())
    model.zero_grad(set_to_none=True)
    flow_loss.mean().backward()
    assert shared.grad.abs().sum() > 0
    for module in (model.bspline_noise_input, model.bspline_plan_projection, model.bspline_proprio_head):
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in module.parameters())


def test_none_skips_all_plan_work_without_supervision_and_at_rollout(monkeypatch):
    model = planner(monkeypatch, NONE_LABEL)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupervised/rollout Action must not invoke the planner.")

    model._required_start_latent = forbidden
    assert model.generate_independent_plan(*conditions()) == (None, None, None)
    policy = rollout_policy(monkeypatch, NONE_LABEL)
    policy.model.generate_independent_plan = forbidden
    inputs = rollout_batch()
    for _ in range(2):
        policy.predict_action_chunk(inputs)
        assert policy.model.plan is None and policy.model.plan_prediction is None
    inputs["skill_start_cache_id"][0] += 1
    policy.predict_action_chunk(inputs)
    assert policy._independent_plan_identity is None
    policy.reset()
    policy.predict_action_chunk(inputs)


def test_none_flow_sign_and_shape_validation(monkeypatch):
    model = planner(monkeypatch, NONE_LABEL)
    model.set_skill_start_latent(torch.zeros(2, 100, 16))
    model.sample_noise = lambda shape, device: torch.full(shape, 3., device=device)
    model.sample_time = lambda batch, device: torch.full((batch,), .25, device=device)
    seen = []

    def readout(state, time, *args):
        seen.append((state.clone(), time.clone()))
        return torch.zeros(2, 30, 16), torch.ones(2, 30, 8)

    model._flow_readout = readout
    plan, prediction, loss = model.generate_independent_plan(*conditions(), target=torch.full((2, 30, 8), 2.))
    assert plan is prediction is None and len(seen) == 1 and not loss.any()
    torch.testing.assert_close(seen[0][0], torch.full((2, 30, 8), 2.25))
    for invalid in (torch.zeros(2, 10, 8), torch.full((2, 30, 8), float("nan"))):
        with pytest.raises(ValueError, match="finite"):
            model.generate_independent_plan(*conditions(), target=invalid)
