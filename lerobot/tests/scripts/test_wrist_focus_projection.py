"""Gripper-mounted (eye-in-hand) camera projection: constant EEF->camera offset + per-frame pose."""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.policies.skillVLA.focus_projection import (
    camera_transform_from_hand_pose,
    hand_camera_from_recorded_xml,
    project_eef,
    rotation_from_axis_angle,
)

# hand -> camera: 5 cm along +x; hand -> EEF site: 10 cm along +z (like the Panda grip site).
XML = """
<mujoco><worldbody>
  <body name="link"><joint name="j"/>
    <body name="right_hand" pos="0 0 0.3" quat="0.924 0 0 -0.383">
      <camera name="eye_in_hand" pos="0.05 0 0" fovy="75"/>
      <body name="gripper" pos="0 0 0.05">
        <site name="grip_site" pos="0 0 0.05"/>
      </body>
    </body>
  </body>
</worldbody></mujoco>
"""


def test_the_eef_to_camera_offset_is_read_from_the_model() -> None:
    camera = hand_camera_from_recorded_xml(XML, camera_name="eye_in_hand", eef_site_name="grip_site")
    np.testing.assert_allclose(camera.position, [0.05, 0.0, -0.1], atol=1e-12)
    np.testing.assert_allclose(camera.rotation, np.eye(3), atol=1e-12)
    assert camera.fovy == 75.0                       # the hand's own pose and joint cancel out


def test_a_point_in_front_of_the_camera_lands_in_the_image_centre() -> None:
    camera = hand_camera_from_recorded_xml(XML, camera_name="eye_in_hand", eef_site_name="grip_site")
    transform = camera_transform_from_hand_pose(
        camera, np.zeros(3), np.eye(3), height=256, width=256
    )
    # The MuJoCo camera looks along its own -z; with an identity EEF that is world -z.
    centre = project_eef(np.array([0.05, 0.0, -0.4]), transform, height=256, width=256)
    assert centre.valid and not centre.clipped
    assert centre.pixel_xy == pytest.approx((127, 128), abs=1)

    # Moving the target sideways moves the projection off centre, and behind the camera is invalid.
    right = project_eef(np.array([0.05, 0.05, -0.4]), transform, height=256, width=256)
    assert right.pixel_xy[0] != centre.pixel_xy[0] or right.pixel_xy[1] != centre.pixel_xy[1]
    with pytest.raises(ValueError, match="behind"):
        project_eef(np.array([0.05, 0.0, 0.4]), transform, height=256, width=256)


def test_the_camera_follows_the_eef_rotation() -> None:
    camera = hand_camera_from_recorded_xml(XML, camera_name="eye_in_hand", eef_site_name="grip_site")
    # Yaw the EEF by 90 deg about +z: the camera now looks along world -z still, but its offset turns.
    rotation = rotation_from_axis_angle([0.0, 0.0, np.pi / 2])
    np.testing.assert_allclose(rotation @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-9)
    transform = camera_transform_from_hand_pose(
        camera, np.zeros(3), rotation, height=256, width=256
    )
    turned = project_eef(np.array([0.0, 0.05, -0.4]), transform, height=256, width=256)
    assert turned.pixel_xy == pytest.approx((127, 128), abs=1)


def test_a_joint_between_the_site_and_the_camera_is_refused() -> None:
    xml = XML.replace('<body name="gripper" pos="0 0 0.05">',
                      '<body name="gripper" pos="0 0 0.05"><joint name="finger"/>')
    with pytest.raises(ValueError, match="joint between the EEF site"):
        hand_camera_from_recorded_xml(xml, camera_name="eye_in_hand", eef_site_name="grip_site")
