#!/usr/bin/env python3
"""Reconstruct Coffee camera geometry using the isolated simulator, without rendering."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import h5py
import numpy as np


def export(env_meta, output, validation_frames=None):
    import robosuite
    import mimicgen  # noqa: F401
    from mimicgen.envs.robosuite.coffee import Coffee_D0  # noqa: F401
    from robosuite.utils.camera_utils import get_camera_transform_matrix

    with h5py.File(env_meta) as archive:
        recorded = json.loads(archive['data'].attrs['env_args'])
    if recorded['env_name'] != 'Coffee_D0' or recorded['env_version'] != '1.4.1':
        raise ValueError('Expected Coffee_D0 robosuite 1.4.1 environment metadata')
    kwargs = dict(recorded['env_kwargs'])
    kwargs.update(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, hard_reset=False)
    np.random.seed(0)
    env = robosuite.make(env_name=recorded['env_name'], **kwargs)
    try:
        env.reset()
        robot = env.robots[0]
        sim = env.sim
        height, width = int(kwargs['camera_heights']), int(kwargs['camera_widths'])
        body = robot.robot_model.eef_name
        site = robot.gripper.important_sites['grip_site']
        position = np.asarray(sim.data.get_site_xpos(site)).copy()
        rotation = np.asarray(sim.data.get_body_xmat(body)).copy()
        camera_id = sim.model.camera_name2id('robot0_eye_in_hand')
        wrist_position = np.asarray(sim.data.cam_xpos[camera_id]).copy()
        wrist_rotation = np.asarray(sim.data.cam_xmat[camera_id]).reshape(3, 3).copy()
        calibration = {
            'schema': 'mimicgen_coffee_camera/v1', 'env_name': 'Coffee_D0',
            'width': width, 'height': height,
            'top_camera': 'agentview', 'wrist_camera': 'robot0_eye_in_hand',
            'eef_site': site, 'eef_rotation_body': body,
            'eef_quaternion_order': 'xyzw', 'state_orientation': 'axis_angle_radians',
            'top_world_to_pixel': get_camera_transform_matrix(sim, 'agentview', height, width).tolist(),
            'wrist_camera_rotation_in_eef_body': (rotation.T @ wrist_rotation).tolist(),
            'wrist_camera_position_from_eef_site_in_eef_body': (rotation.T @ (wrist_position - position)).tolist(),
            'wrist_fovy_degrees': float(sim.model.cam_fovy[camera_id]),
            'projection_convention': 'OpenCV top-left origin before source image orientation adjustment',
            'image_transform': 'identity', 'source_video_orientation_verified': False,
            'provenance': 'reconstructed from pinned assets and HF env_args; not recorded per-demo calibration',
            'versions': {key: importlib.metadata.version(key) for key in ('mimicgen', 'robosuite', 'mujoco')},
        }
        if validation_frames:
            from scipy.spatial.transform import Rotation
            samples = np.load(validation_frames, allow_pickle=False)
            errors, angles = [], []
            for joint, eef in zip(samples['joint'], samples['eef'], strict=True):
                sim.data.qpos[robot._ref_joint_pos_indexes] = joint
                sim.forward()
                errors.append(float(np.linalg.norm(sim.data.get_site_xpos(site) - eef[:3])))
                actual = Rotation.from_matrix(sim.data.get_body_xmat(body))
                expected = Rotation.from_quat(eef[3:])
                angles.append(float((actual.inv() * expected).magnitude()))
            if max(errors) > 1e-3 or max(angles) > 1e-3:
                raise ValueError(f'Source EEF pose does not match reconstructed Panda geometry: '
                                 f'{max(errors):.6f}m, {max(angles):.6f}rad')
            calibration['recorded_joint_fk_check'] = {
                'frames': len(errors), 'max_position_error_m': max(errors), 'max_rotation_error_rad': max(angles)}
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / 'mimicgen_model.xml').write_text(sim.model.get_xml())
        (output / 'mimicgen_camera.json').write_text(json.dumps(calibration, indent=2) + '\n')
        print('Coffee geometry: static top + EEF-relative wrist calibration saved (CPU, no rendering)', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-meta', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validation-frames', type=Path)
    args = parser.parse_args()
    export(args.env_meta, args.output, args.validation_frames)
