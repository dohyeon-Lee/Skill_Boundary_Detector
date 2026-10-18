from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pytest
import torch

from lerobot.utils.motion_plan_visualization import (
    command_path, decode_motion_plan, project_path, select_skill_samples, write_motion_plan_gallery,
)


def context():
    transform = np.eye(4)
    transform[:3, :3] = [[30, 0, 50], [0, 30, 50], [0, 0, 1]]
    return {"eef_world": np.array([0., 0., 1.]),
            "input_min": -np.ones(3), "input_max": np.ones(3),
            "output_min": -.05 * np.ones(3), "output_max": .05 * np.ones(3),
            "cameras": {"agentview": {"image": np.full((100, 100, 3), 200, dtype=np.uint8),
                                       "transform": transform}}}


def record(step=0, skill=0):
    base = np.zeros((8, 7))
    base[:, 0] = .2
    base[:, -1] = np.tile([-1., 1.], 4)
    chunk = base[2:4].copy()
    chunk[:, 1] = .4
    return {"episode_step": step, "skill_index": skill, "token": 13, "n_action_steps": 1,
            "flow_time": .1, "context": context(), "base_actions": base,
            "chunk_actions": chunk, "final_window_actions": base.copy()}


def test_controller_scaling_clipping_and_past_anchor():
    actions = np.zeros((8, 7))
    actions[:, 0] = 2  # Clipped to 1, then scaled to 0.05 m, NOT two meters.
    path = command_path(actions, context(), anchor_index=2)
    np.testing.assert_allclose(path[2], [0, 0, 1])
    np.testing.assert_allclose(path[:, 0], (np.arange(9) - 2) * .05)
    with pytest.raises(ValueError, match="Anchor"):
        command_path(actions, context(), anchor_index=10)


def test_live_camera_projection_uses_display_mirror_and_masks_bad_points():
    camera = context()["cameras"]["agentview"]
    pixels = project_path(np.array([[0, 0, 1], [1, 0, 1], [0, 0, -1], [100, 0, 1]]), camera)
    np.testing.assert_allclose(pixels[:2], [[49, 50], [19, 50]], atol=1e-5)
    assert np.isnan(pixels[2:]).all()


def test_uniform_sampling_keeps_repeated_code_occurrences_separate():
    records = [record(step=i * 5, skill=0) for i in range(10)] + [record(step=100, skill=1)]
    selected = select_skill_samples(records, 4)
    assert [item["episode_step"] for item in selected[0]] == [0, 15, 30, 45]
    assert [item["episode_step"] for item in selected[1]] == [100]


def test_decode_uses_real_processors_in_their_per_step_shape():
    base = torch.zeros(2, 8, 32)
    plan = {"base_window": base, "final_window": base + 1,
            "chunk": torch.full((2, 2, 7), 2.), "episode_step": 5,
            "skill_indices": [0, 1], "tokens": [13, 4], "flow_time": .1, "n_action_steps": 1}
    def unnormalize(actions):
        assert actions.shape == (36, 7)
        return actions * 3 + 10
    decoded = decode_motion_plan(plan, [context(), context()], unnormalize,
                                 lambda transition: {"action": transition["action"] + 1})
    assert decoded[1]["skill_index"] == 1 and decoded[1]["token"] == 4
    np.testing.assert_array_equal(decoded[0]["base_actions"], 11)
    np.testing.assert_array_equal(decoded[0]["chunk_actions"], 17)
    np.testing.assert_array_equal(decoded[0]["final_window_actions"], 14)
    assert torch.equal(base, torch.zeros_like(base))


def test_independent_gallery_preserves_both_readouts_and_exact_normalized_noise(tmp_path):
    noise = torch.arange(8 * 32).reshape(1, 8, 32).float()
    plan = {"kind": "independent_window", "base_window": torch.zeros(1, 8, 7),
            "final_window": torch.full((1, 8, 7), .1), "chunk": torch.full((1, 2, 7), .1),
            "last_step_base_window": torch.full((1, 8, 7), .09),
            "initial_noise_normalized": noise, "flow_time": 0., "last_step_flow_time": .1,
            "episode_step": 0, "skill_indices": [0], "tokens": [13], "n_action_steps": 1}
    records = decode_motion_plan(plan, [context()], lambda action: action, lambda transition: transition)
    assert records[0]["kind"] == "independent_window"
    np.testing.assert_array_equal(records[0]["initial_noise_normalized"], noise[0].numpy())
    assert records[0]["final_window_actions"].shape == (8, 7)
    np.testing.assert_allclose(records[0]["last_step_base_actions"], .09)
    page = write_motion_plan_gallery(records, tmp_path)
    assert "Independent base-only 4H" in Path(page).read_text()
    assert "EXACT SAME 4H Gaussian noise" in Path(page).read_text()
    assert "SEPARATE intermediate action states" in Path(page).read_text()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest[0]["kind"] == "independent_window"
    with np.load(tmp_path / manifest[0]["data"], allow_pickle=False) as data:
        np.testing.assert_array_equal(data["initial_noise_normalized"], noise[0].numpy())
        assert data["base_actions"].shape == data["last_step_base_actions"].shape == (8, 7)
        assert float(data["last_step_flow_time"]) == .1


def test_gallery_writes_selected_pngs_and_reusable_raw_data(tmp_path):
    records = [record(step=i * 5) for i in range(6)]
    original = deepcopy(records)
    path = write_motion_plan_gallery(records, tmp_path, samples_per_skill=4, success=False)
    assert "FAILURE" in Path(path).read_text()
    assert "NOT simulated" in Path(path).read_text() or "not simulated" in Path(path).read_text()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest) == 4
    for item in manifest:
        assert (tmp_path / item["image"]).stat().st_size > 1000
        with np.load(tmp_path / item["data"], allow_pickle=False) as saved:
            assert saved["base_actions"].shape == (8, 7)
            assert saved["chunk_actions"].shape == (2, 7)
            np.testing.assert_array_equal(saved["base_xyz"][2], context()["eef_world"])
            assert saved["agentview_image"].shape == (100, 100, 3)
    np.testing.assert_array_equal(records[0]["base_actions"], original[0]["base_actions"])


def test_canonical_gallery_uses_one_fixed_world_anchor_and_variable_gt_length(tmp_path):
    full = torch.zeros(3, 7)
    full[:, 0] = .2
    plan = {"kind": "canonical", "full_skills": [{"actions": full, "gt_length": 3, "skill_start_step": 0}],
            "chunk": torch.zeros(1, 2, 7), "episode_step": 0,
            "skill_indices": [0], "tokens": [13], "n_action_steps": 1}
    anchors = {}
    first = decode_motion_plan(plan, [context()], lambda action: action, lambda transition: transition, anchors=anchors)[0]
    current_context = context()
    current_context["eef_world"] += [.2, .1, 0]
    plan["episode_step"] = 5
    second = decode_motion_plan(plan, [current_context], lambda action: action,
                                lambda transition: transition, anchors=anchors)[0]
    np.testing.assert_array_equal(first["full_skill_start_world"], second["full_skill_start_world"])
    assert second["base_actions"].shape == (3, 7)  # Neither padded to 248 nor forced to 4H.
    assert not np.array_equal(second["full_skill_start_world"], second["context"]["eef_world"])
    path = write_motion_plan_gallery([first, second], tmp_path)
    page = Path(path).read_text()
    assert "Full skill vs current action chunk" in page
    assert "GT length is used ONLY" in page
    assert "NOT a learned/aligned phase" in page
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest) == 2 and all(item["gt_length"] == 3 for item in manifest)
    for item in manifest:
        with np.load(tmp_path / item["data"], allow_pickle=False) as data:
            np.testing.assert_array_equal(data["base_xyz"][0], context()["eef_world"])
            expected_current = context()["eef_world"] if item["episode_step"] == 0 else current_context["eef_world"]
            np.testing.assert_array_equal(data["chunk_xyz"][0], expected_current)


def test_rollout_capture_keeps_actions_and_processor_state_unchanged(monkeypatch, tmp_path):
    from lerobot.scripts import lerobot_skillvla_eval as evaluation

    monkeypatch.setattr(evaluation, "preprocess_observation", lambda observation: observation)
    monkeypatch.setattr(evaluation, "add_envs_task", lambda env, observation: observation)
    monkeypatch.setattr(evaluation, "check_env_attributes_and_types", lambda env: None)
    monkeypatch.setattr(evaluation, "inside_slurm", lambda: True)

    class Processor:
        def __init__(self, offset):
            self.calls = 0
            self.offset = offset

        def __call__(self, action):
            self.calls += 1
            if isinstance(action, dict):
                return {"action": action["action"] + self.offset}
            return action + self.offset

    class Env:
        num_envs = 1

        def reset(self, seed=None):
            self.steps, self.camera_reads = 0, 0
            return {}, {}

        def call(self, method):
            if method == "_max_episode_steps":
                return (3,)
            assert method == "get_motion_plan_context"
            self.camera_reads += 1
            return (context(),)

        def step(self, action):
            self.steps += 1
            return {}, np.ones(1), np.array([self.steps == 3]), np.array([False]), {"is_success": np.array([True])}

    class Policy(torch.nn.Module):
        def __init__(self, enabled):
            super().__init__()
            self.capture_motion_plans = enabled

        def reset(self):
            self.steps = 0
            self.pending = None

        def select_action(self, observation):
            chunk = torch.ones(1, 2, 7) * self.steps
            if self.steps % 2 == 0:
                self.pending = {"base_window": torch.zeros(1, 8, 7), "final_window": torch.ones(1, 8, 7),
                                "chunk": chunk, "episode_step": self.steps, "skill_indices": [0],
                                "tokens": [13], "flow_time": .1, "n_action_steps": 2}
            self.steps += 1
            return chunk[:, 0]

        def pop_motion_plan(self):
            pending, self.pending = self.pending, None
            return pending

    outputs = []
    for enabled in (False, True):
        env = Env()
        action_processor, env_processor = Processor(10), Processor(1)
        result = evaluation.rollout(env, Policy(enabled), lambda value: value, env_processor,
                                   lambda value: value, action_processor)
        assert env.steps == 3
        assert action_processor.calls == env_processor.calls == 3  # Diagnostic copies are isolated.
        assert env.camera_reads == (2 if enabled else 0)
        outputs.append(result)
    torch.testing.assert_close(outputs[0]["action"], outputs[1]["action"], rtol=0, atol=0)
    assert [item["episode_step"] for item in outputs[1]["motion_plan_records"]] == [0, 2]
    np.testing.assert_array_equal(outputs[1]["motion_plan_records"][0]["chunk_actions"], 11)
    # Exercise the outer eval writer too, not just geometry/rollout in isolation.
    monkeypatch.setattr(evaluation, "PreTrainedPolicy", torch.nn.Module)
    policy = Policy(True)
    policy.motion_plan_samples_per_skill = 4
    result = evaluation.eval_policy(
        Env(), policy, lambda value: value, Processor(1), lambda value: value,
        Processor(10), n_episodes=1, videos_dir=tmp_path,
    )
    assert len(result["motion_plan_paths"]) == 1
    assert Path(result["motion_plan_paths"][0]).is_file()
    assert result["per_episode"][0]["success"] is True
