"""Auxiliary video capture must be invisible to the deployed policy."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.configs.uv_fullskill import DUAL_BOTTLENECK_NAMES, UV_BSPLINE_SKILL_PI05_VLM_BASE_LABELS
from lerobot.policies.skill_expert.bottleneck_video import (
    capture_bottleneck, capture_condition, capture_latent, decode_auxiliaries,
)
from test_pi05_vlm_vsa import tiny_vlm_model
from test_dual_bottleneck import inputs


@pytest.mark.parametrize("label", [*DUAL_BOTTLENECK_NAMES, "vanila_last_pi05vlm_9_9",
                                   next(iter(UV_BSPLINE_SKILL_PI05_VLM_BASE_LABELS))])
@torch.no_grad()
def test_capture_preserves_actions_rng_and_single_vlm_pass(monkeypatch, label):
    model = tiny_vlm_model(monkeypatch, label).eval()
    images, state, skill, _, goal = inputs()
    if not model.config.omits_start_vision:
        model.set_skill_start_latent(model.encode_skill_start(images, state, skill, goal))
    keys = set(model.state_dict())
    torch.manual_seed(91)
    baseline = model.sample_actions(images, state, skill, num_steps=2, end_pose=goal)
    rng = torch.get_rng_state()
    calls = []
    hook = model.cond_encoder.vlm.vision_tower.register_forward_hook(lambda *args: calls.append(1))
    torch.manual_seed(91)
    with capture_bottleneck(model) as capture:
        actual = model.sample_actions(images, state, skill, num_steps=2, end_pose=goal)
    hook.remove()
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    assert len(calls) == 2 and set(model.state_dict()) == keys
    assert not hasattr(model, "_video_aux_capture") and "top" not in capture and "wrist" not in capture
    expected = {"upper", "lower"} if label in DUAL_BOTTLENECK_NAMES else {"upper"}
    assert set(capture["outputs"]) == expected
    for values in capture["outputs"].values():
        assert values["xyz_normalized"].shape == (2, 3)
        assert values["top_patch_box_uv"].shape == (2, 4)
        assert values["wrist_patch_box_uv"].shape == (2, 4)
        assert np.isfinite(values["xyz_normalized"]).all()
        np.testing.assert_allclose(values["top_patch_box_uv"][:, 2:] - values["top_patch_box_uv"][:, :2], 1)
    with capture_bottleneck(model, False) as disabled:
        assert disabled is None
    model.train()
    with capture_bottleneck(model) as disabled:
        assert disabled is None


def test_decode_xyz_world_once_and_skip_unknown_grounding_or_crop():
    config = SimpleNamespace(chunk_end_state_q01=[-1, -2, -3], chunk_end_state_q99=[1, 2, 3],
                             proprio_grounding="episode_start_xyz", foveated_vision_enabled=False)
    outputs = {"upper": {"xyz_normalized": np.array([[0., 1., -1.]]),
                         "top_patch_box_uv": np.array([[-1., 0., 0., 1.]])}}
    decoded = decode_auxiliaries(outputs, config, [[10., 20., 30.]])
    np.testing.assert_allclose(decoded["upper"]["xyz_world"], [[10., 22., 27.]])
    assert "xyz_world" not in decode_auxiliaries(outputs, config)["upper"]
    config.foveated_vision_enabled = True
    assert decode_auxiliaries(outputs, config) == {}


@pytest.mark.parametrize("label", DUAL_BOTTLENECK_NAMES)
@torch.no_grad()
def test_dual_diagnostic_replans_reuse_current_features_without_changing_action_state(monkeypatch, label):
    model = tiny_vlm_model(monkeypatch, label).eval()
    images, state, skill, _, goal = inputs()
    model.sample_actions(images, state, skill, num_steps=2, end_pose=goal)
    latent = model._last_lower_latent
    time = model._split_time_condition.clone()
    calls = []
    hook = model.cond_encoder.vlm.vision_tower.register_forward_hook(lambda *args: calls.append(1))
    torch.manual_seed(91)
    baseline = model.sample_actions(images, state, skill, num_steps=2, end_pose=goal)
    rng = torch.get_rng_state().clone()
    calls.clear()
    model._last_lower_latent = latent
    model._split_time_condition = time.clone()
    proprio = model._current_proprio_condition
    with torch.random.fork_rng():
        prediction = model.generate_visualization_plan(
            skill[:1], goal[:1], start_latent=latent[:1], batch_indices=[0],
        )
    assert prediction.shape == (1, 30, 8) and not calls
    assert model._last_lower_latent is latent
    assert model._current_proprio_condition is proprio
    torch.testing.assert_close(model._split_time_condition, time, rtol=0, atol=0)
    torch.manual_seed(91)
    actual = model.sample_actions(images, state, skill, num_steps=2, end_pose=goal)
    hook.remove()
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())


def test_dino_capture_discards_both_cls_tokens_and_patch_argmax():
    class PatchHead:
        def __call__(self, query, top):
            torch.testing.assert_close(top, torch.tensor([[[1.], [2.], [3.], [4.]]]))
            return torch.tensor([[0., 1., 9., 2.]])
    class WristPatchHead:
        def __call__(self, query, wrist):
            torch.testing.assert_close(wrist, torch.tensor([[[6.], [7.], [8.], [9.]]]))
            return torch.tensor([[0., 9., 1., 2.]])
    model = SimpleNamespace(training=False, _alignment_token_count=3,
                            config=SimpleNamespace(), agent_patch_align_head=PatchHead(),
                            wrist_patch_align_head=WristPatchHead())
    with capture_bottleneck(model) as capture:
        capture_condition(model, torch.arange(10.).reshape(1, 10, 1))
        capture_latent(model, torch.zeros(1, 3, 1))
    np.testing.assert_allclose(capture["outputs"]["upper"]["top_patch_box_uv"], [[-1, 0, 0, 1]])
    np.testing.assert_allclose(capture["outputs"]["upper"]["wrist_patch_box_uv"], [[0, -1, 1, 0]])
    decoded = decode_auxiliaries(capture["outputs"], SimpleNamespace())
    np.testing.assert_allclose(decoded["upper"]["wrist_patch_box_uv"], [[0, -1, 1, 0]])


def test_capture_cleans_up_after_action_error():
    model = SimpleNamespace(training=False)
    with pytest.raises(RuntimeError, match="action failed"):
        with capture_bottleneck(model):
            raise RuntimeError("action failed")
    assert not hasattr(model, "_video_aux_capture")
