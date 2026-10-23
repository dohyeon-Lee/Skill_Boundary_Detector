from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from lerobot.policies.skill_aux.terminator_uv import (
    TOP_CAMERA, normalized_top_camera_projection, project_goal_uv,
)
from lerobot.policies.skillVLA.chunk_patch_target import load_dataset_top_camera_transforms, project_top_points
from lerobot.policies.skillVLA.focus_projection import camera_transform_from_recorded_xml
from lerobot.policies.skill_aux.configuration_skill_aux import SkillAuxConfig
from lerobot.policies.skill_aux import modeling_skill_aux as aux
from lerobot.policies.skill_expert.processor_skill_expert import skill_expert_batch_to_transition, skill_expert_transition_to_batch
from lerobot.policies.skillVLA.newterm_supervision import (
    POST_END_CODES, POST_END_VALID, POST_END_GOAL_XYZ, POST_END_GOAL_VALID, POST_END_START_STATE,
)

XML = '<mujoco><worldbody><camera name="agentview" pos="0 0 1" quat="1 0 0 0" fovy="45"/></worldbody></mujoco>'


def test_project_uv_matches_dataset_projection_including_grounding_and_mirror():
    matrix = camera_transform_from_recorded_xml(XML, camera_name="agentview", height=256, width=256)
    offset = np.array([.1, -.05, .2])
    matrix[:, 3] += matrix[:, :3] @ offset
    points = torch.tensor([[0., 0., 0.], [.1, .05, .1], [10., 0., 0.], [0., 0., 2.], [float("nan"), 0., 0.]])
    projection = torch.from_numpy(normalized_top_camera_projection(matrix, height=256, width=256))
    uv, valid = project_goal_uv(points, projection.expand(len(points), -1, -1))
    expected, expected_valid = project_top_points(points.numpy(), matrix, height=256, width=256)
    np.testing.assert_allclose(uv[:3], expected[:3], atol=1e-6)
    np.testing.assert_array_equal(valid, expected_valid)
    assert torch.isfinite(uv).all()
    assert not uv[-2:].any()


def test_embedded_camera_loader_never_loads_all_dataset_states(tmp_path):
    path = tmp_path / "focus.npz"
    np.savez(path, image_height=256, image_width=256, camera_episode_id=[7], camera_grounded_transform=[np.eye(4)])
    matrices, h, w = load_dataset_top_camera_transforms(SimpleNamespace(episodes=[7]), path)
    np.testing.assert_array_equal(matrices[7], np.eye(4))
    assert (h, w) == (256, 256)


def test_projection_keeps_float32_geometry_under_autocast():
    xyz = torch.tensor([[.0139242, .137458, .019384]])
    matrix = camera_transform_from_recorded_xml(XML, camera_name="agentview", height=256, width=256)
    projection = torch.from_numpy(normalized_top_camera_projection(matrix, height=256, width=256))[None]
    expected, _ = project_goal_uv(xyz, projection)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual, _ = project_goal_uv(xyz, projection)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("with_start", [True, False])
@pytest.mark.parametrize("training", [True, False])
def test_uv_objective_perturbs_xyz_before_projection_for_current_and_post_end(monkeypatch, with_start, training):
    class Term(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))
            self.vision_encoder = nn.Linear(1, 1)
            self.context_mode, self.camera_mode = "proprio", "top"
            self.goal_xyz, self.goal_uv = False, True
            self.proprio_history, self.state_dim = False, 8
            self.calls, self.vision_calls = [], 0

        def _prepare_image_tokens(self, third, wrist):
            self.vision_calls += 1
            return third.mean((2, 3)).unsqueeze(1)

        def _forward_from_image_tokens(self, z, context, images, *, goal_uv, start_state=None):
            self.calls.append((goal_uv.clone(), start_state))
            logits = self.weight * (goal_uv.sum(-1) + 1)
            return torch.zeros_like(logits), logits

    monkeypatch.setattr(aux, "build_trainable_fsq_terminator", lambda *a, **kw: Term())
    policy = aux.SkillAuxPolicy(SkillAuxConfig(
        fsq_path="unused", dtype="float32", device="cpu", max_state_dim=8,
        terminator_context="proprio", terminator_cameras="top", terminator_goal_uv=True,
        terminator_current_proprio=False, terminator_start_proprio=with_start,
        terminator_goal_noise_max_m=.01, terminator_termination_only=True, terminator_post_end_steps=10,
    ))
    policy.train(training)
    matrix = camera_transform_from_recorded_xml(XML, camera_name="agentview", height=256, width=256)
    projection = torch.from_numpy(normalized_top_camera_projection(matrix, height=256, width=256)).expand(2, -1, -1)
    batch = {
        "skill_ds": torch.ones(2, dtype=torch.long), "skill_de": torch.ones(2, dtype=torch.long),
        "skill_sequence": torch.tensor([[0, 1], [0, 1]]), "skill_index": torch.zeros(2, dtype=torch.long),
        "skill_decoder_state": torch.zeros(2, 8), "observation.state": torch.zeros(2, 8),
        "observation.images.image": torch.rand(2, 3, 4, 4),
        "skill_end_state": torch.tensor([[.02, .03, .04, 0., 0., 0., 0., 0.]]).repeat(2, 1),
        "skill_end_state_valid": torch.ones(2, dtype=torch.bool), TOP_CAMERA: projection,
        POST_END_CODES: torch.zeros(2, 10, dtype=torch.long),
        POST_END_VALID: torch.zeros(2, 10, dtype=torch.bool),
        POST_END_GOAL_XYZ: torch.full((2, 10, 3), .1),
        POST_END_GOAL_VALID: torch.ones(2, 10, dtype=torch.bool),
    }
    batch[POST_END_VALID][0, 0] = True
    if with_start:
        batch["skill_start_state"] = torch.full((2, 8), .2)
        batch[POST_END_START_STATE] = torch.full((2, 10, 8), .3)
    restored = skill_expert_transition_to_batch(skill_expert_batch_to_transition(batch))
    torch.testing.assert_close(restored[TOP_CAMERA], projection)
    restored["skill_decoder_state"] = batch["skill_decoder_state"]
    draws = []
    def fixed_noise(x):
        draws.append(x.clone())
        return torch.full_like(x, .75)  # +.005 metres on each XYZ axis
    monkeypatch.setattr(torch, "rand_like", fixed_noise)
    loss, metrics = policy._terminator_objective(restored)
    term = policy.model.fsq_term_train
    noise = .005 if training else 0
    current_expected, _ = project_goal_uv(batch["skill_end_state"][:, :3] + noise, projection)
    past_expected, _ = project_goal_uv(batch[POST_END_GOAL_XYZ][:1, 0] + noise, projection[:1])
    torch.testing.assert_close(term.calls[0][0], current_expected)
    torch.testing.assert_close(term.calls[1][0], past_expected)
    assert [list(x.shape) for x in draws] == ([[2, 3], [1, 3]] if training else [])
    assert term.vision_calls == 1
    assert len(term.calls) == 2
    assert metrics["terminator/goal_uv_in_view_fraction"] == 1
    if with_start:
        torch.testing.assert_close(term.calls[0][1], batch["skill_start_state"])
        torch.testing.assert_close(term.calls[1][1], batch[POST_END_START_STATE][:1, 0])
    loss.backward()
    assert torch.isfinite(term.weight.grad)
