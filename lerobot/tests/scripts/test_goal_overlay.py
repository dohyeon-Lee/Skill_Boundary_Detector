"""Projecting the skill-end goal onto evaluation frames (wrist and agent view)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_SRC = (
    Path(__file__).resolve().parents[2]
    / "examples/libero/configs/train_skillVLA/stage1_eval/src"
)
sys.path.insert(0, str(_SRC))

from goal_overlay import (  # noqa: E402
    agent_view_transform,
    draw_goal,
    marks_in_agent_view,
    marks_in_wrist_view,
)
from lerobot.policies.skillVLA.focus_projection import (  # noqa: E402
    camera_transform_from_recorded_xml,
)

SIZE = 256
CAMERA_POS = (0.6, 0.0, 1.35)
CAMERA_QUAT = (0.653, 0.271, 0.271, 0.653)   # a plausible agentview pose
CAMERA_FOVY = 45.0


def _sim(pos=CAMERA_POS, quat=CAMERA_QUAT, fovy=CAMERA_FOVY):
    """The few fields of a MuJoCo model the transform reads."""
    return SimpleNamespace(
        model=SimpleNamespace(
            camera_name2id=lambda name: 0 if name == "agentview" else pytest.fail(name),
            cam_pos=np.array([pos]),
            cam_quat=np.array([quat]),
            cam_fovy=np.array([fovy]),
        )
    )


def test_the_live_model_gives_the_same_camera_as_the_recorded_xml() -> None:
    """The sim-read transform must equal the XML one the focus-UV builder is validated on."""
    xml = (
        "<mujoco><worldbody>"
        f'<camera name="agentview" pos="{CAMERA_POS[0]} {CAMERA_POS[1]} {CAMERA_POS[2]}" '
        f'quat="{CAMERA_QUAT[0]} {CAMERA_QUAT[1]} {CAMERA_QUAT[2]} {CAMERA_QUAT[3]}" '
        f'fovy="{CAMERA_FOVY}"/>'
        "</worldbody></mujoco>"
    )
    expected = camera_transform_from_recorded_xml(xml, camera_name="agentview", height=SIZE, width=SIZE)

    np.testing.assert_allclose(
        agent_view_transform(_sim(), height=SIZE, width=SIZE), expected, atol=1e-9
    )


def test_the_agent_view_puts_the_grounding_back_on() -> None:
    """The world-fixed camera needs world xyz; a grounded goal without the offset lands elsewhere."""
    transform = agent_view_transform(_sim(), height=SIZE, width=SIZE)
    goal = np.array([[0.05, -0.02, 0.10]])
    grounding = np.array([0.10, 0.20, 0.90])

    grounded = marks_in_agent_view(transform, goal, None, height=SIZE, width=SIZE)
    world = marks_in_agent_view(transform, goal, grounding, height=SIZE, width=SIZE)

    assert world.visible[0]
    assert not np.allclose(world.pixels[0], grounded.pixels[0])
    # Adding the offset by hand must be identical to passing it in.
    by_hand = marks_in_agent_view(transform, goal + grounding, None, height=SIZE, width=SIZE)
    np.testing.assert_allclose(world.pixels, by_hand.pixels, atol=1e-9)


def test_the_wrist_view_ignores_the_grounding() -> None:
    """The camera rides the gripper, so shifting both the goal and the EEF changes nothing."""
    state = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1, -0.2, 0.3, 0.4, 0.1, 0.2]])
    goal = np.array([[0.0, 0.0, 0.25], [0.1, -0.2, 0.55]])
    offset = np.array([0.3, -0.4, 0.9])

    here = marks_in_wrist_view(state, goal, height=SIZE, width=SIZE)
    shifted_state = state.copy()
    shifted_state[:, :3] += offset
    shifted = marks_in_wrist_view(shifted_state, goal + offset, height=SIZE, width=SIZE)

    np.testing.assert_allclose(here.pixels, shifted.pixels, atol=1e-3)


def test_a_goal_behind_or_outside_the_wrist_camera_is_not_drawn() -> None:
    state = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 3)
    goals = np.array(
        [
            [0.0, 0.0, 0.25],    # ahead of the gripper camera
            [0.0, 0.0, -0.25],   # behind it
            [2.0, 0.0, 0.25],    # in front, far outside the frame
        ]
    )
    marks = marks_in_wrist_view(state, goals, height=SIZE, width=SIZE)

    assert marks.visible.tolist() == [True, False, False]
    assert np.all(np.isfinite(marks.pixels[0]))
    assert np.all(np.isnan(marks.pixels[1:]))     # invisible pixels are never half-drawn
    assert len(marks) == 3


def test_drawing_is_a_no_op_for_an_invisible_goal() -> None:
    from PIL import Image

    blank = Image.new("RGB", (32, 32), (0, 0, 0))
    draw_goal(blank, np.array([np.nan, np.nan]), color=(255, 0, 0))
    assert np.asarray(blank).sum() == 0

    draw_goal(blank, np.array([16.0, 16.0]), color=(255, 0, 0))
    assert np.asarray(blank).sum() > 0


def _eval_module():
    from lerobot.scripts import lerobot_skillvla_eval

    return lerobot_skillvla_eval


def test_the_marker_reaches_the_rollout_panel() -> None:
    """A goal pixel must actually change the frame, and only where it was asked for."""
    module = _eval_module()
    frames = np.zeros((2, 64, 64, 3), dtype=np.uint8)
    pixels = np.array([[32.0, 32.0], [np.nan, np.nan]])

    plain = module._annotate_eval_video(frames, success=True, task_description=None)
    marked = module._annotate_eval_video(
        frames, success=True, task_description=None, rollout_goal_pixels=pixels
    )

    assert marked.shape == plain.shape          # the overlay must not resize the video
    assert not np.array_equal(marked, plain)    # frame 0 gained a marker
    # Frame 1's goal is off screen, so only the label bar differs there, not the image body.
    body = slice(40, 64)
    np.testing.assert_array_equal(marked[1, body], plain[1, body])


def test_the_overlay_never_breaks_an_evaluation() -> None:
    """Anything missing or broken yields no marker instead of an exception."""
    module = _eval_module()

    class _Broken:
        def get_skill_trace(self):
            raise RuntimeError("no trace here")

    assert module._goal_overlay_pixels(
        _Broken(), None, batch_index=0, n_video_frames=3,
        video_frame_stride=1, height=SIZE, width=SIZE,
    ) == (None, None)

    class _NoGoals:
        def get_skill_trace(self):
            return [{"batch_index": 0, "episode_timestep": 0}]     # no end_pose recorded

    assert module._goal_overlay_pixels(
        _NoGoals(), None, batch_index=0, n_video_frames=3,
        video_frame_stride=1, height=SIZE, width=SIZE,
    ) == (None, None)


def test_the_wrist_overlay_follows_the_active_skill() -> None:
    """Each frame uses the goal of the skill that was active at its timestep."""
    module = _eval_module()

    class _Policy:
        def get_skill_trace(self):
            return [
                {"batch_index": 0, "episode_timestep": 0, "end_pose": [0.0, 0.0, 0.25]},
                {"batch_index": 0, "episode_timestep": 2, "end_pose": [0.0, 0.0, -0.25]},
            ]

        def get_eef_pose_log(self):
            return np.zeros((4, 1, 6), dtype=np.float32)

        def get_grounding_offset(self):
            return None

    agent, wrist = module._goal_overlay_pixels(
        _Policy(), None, batch_index=0, n_video_frames=4,
        video_frame_stride=1, height=SIZE, width=SIZE,
    )

    assert agent is None                      # no env, so only the wrist view is available
    assert wrist.shape == (4, 2)
    # Frames 0-1 see the goal ahead of the gripper; from frame 2 it sits behind the camera.
    assert np.all(np.isfinite(wrist[:2]))
    assert np.all(np.isnan(wrist[2:]))
