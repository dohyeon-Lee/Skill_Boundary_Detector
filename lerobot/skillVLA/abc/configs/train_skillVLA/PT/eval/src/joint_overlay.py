"""Joint-plan FK in private MuJoCo data, projected through the LIVE cameras."""
from __future__ import annotations

import math
import numpy as np
from PIL import Image, ImageDraw

ARM_COLORS = ((30, 230, 255), (255, 160, 40))
AUX_COLORS = {"upper": (255, 127, 80), "lower": (180, 120, 255)}


def denormalize_plan(prediction, q01, q99):
    prediction = np.asarray(prediction, dtype=np.float32)
    low, high = np.asarray(q01, dtype=np.float32), np.asarray(q99, dtype=np.float32)
    if prediction.ndim != 2 or prediction.shape[1] != 14 or low.shape != (14,) or high.shape != (14,):
        raise ValueError("ABC full-skill prediction and quantiles must include all 14 joint channels")
    if not all(np.isfinite(x).all() for x in (prediction, low, high)) or np.any(high < low):
        raise ValueError("non-finite plan or invalid normalization bounds")
    return (prediction + 1) * np.maximum(high - low, 1e-6) / 2 + low


class JointPlanFK:
    def __init__(self, env):
        import mujoco
        self.mj = mujoco
        self.model = env.model
        self.data = mujoco.MjData(env.model)
        self.indices = np.asarray(env._qpos_indices, dtype=int)
        if self.indices.shape != (14,):
            raise ValueError("ABC FK needs the simulator's 14D qpos mapping")
        self.grippers = list(env._gripper_indices)
        from abc_sim.env import _GRIPPER_CTRL_MAX
        self.gripper_scale = _GRIPPER_CTRL_MAX
        self.sites = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}_grasp_site")
                      for arm in ("left", "right")]
        if min(self.sites) < 0:
            raise ValueError("ABC model is missing left/right grasp sites")

    def trajectory(self, live_data, joints):
        joints = np.asarray(joints, dtype=np.float64)
        if joints.ndim != 2 or joints.shape[1] != 14 or not np.isfinite(joints).all():
            raise ValueError("FK requires finite [T,14] absolute joints in radians, grippers in [0,1]")
        # Match ABCEnv._set_qpos_from_state; objects and robot bases remain in
        # their actual scene. No mj_step, contacts, or writes to live_data.
        self.data.qpos[:] = live_data.qpos
        if self.model.nmocap:
            self.data.mocap_pos[:] = live_data.mocap_pos
            self.data.mocap_quat[:] = live_data.mocap_quat
        result = []
        for pose in joints:
            pose = pose.copy()
            pose[self.grippers] *= self.gripper_scale
            self.data.qpos[self.indices] = pose
            self.mj.mj_kinematics(self.model, self.data)
            result.append(self.data.site_xpos[self.sites].copy())
        return np.asarray(result, dtype=np.float32)


def camera_projection(model, data, camera, height, width):
    import mujoco
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if index < 0:
        raise ValueError(f"unknown ABC camera {camera}")
    rotation = data.cam_xmat[index].reshape(3, 3)
    extrinsic = np.column_stack((rotation.T, -rotation.T @ data.cam_xpos[index]))
    focal = 0.5 * height / math.tan(math.radians(float(model.cam_fovy[index])) / 2)
    intrinsic = np.array([[focal, 0, -width / 2], [0, -focal, -height / 2], [0, 0, -1]])
    return intrinsic @ extrinsic


def project_xyz(xyz, projection):
    homogeneous = np.concatenate((xyz, np.ones((*xyz.shape[:-1], 1))), axis=-1)
    projected = homogeneous @ projection.T
    valid = np.isfinite(projected).all(-1) & (projected[..., 2] > 1e-8)
    uv = np.full(projected.shape[:-1] + (2,), np.nan)
    np.divide(projected[..., :2], projected[..., 2:3], out=uv, where=valid[..., None])
    return uv, valid


def draw_auxiliaries(draw, env, camera, height, width, auxiliaries):
    """Patch rectangles and world-XYZ crosses from the actual action readout."""
    fields = {"top": (("top_left_patch_box_uv", "L"), ("top_right_patch_box_uv", "R")),
              "left": (("left_patch_box_uv", "L"),),
              "right": (("right_patch_box_uv", "R"),)}
    for tower, values in auxiliaries.items():
        color = AUX_COLORS[tower]
        prefix = "U" if tower == "upper" else "L"
        for field, arm in fields.get(camera, ()):
            box = np.asarray(values.get(field, []), dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all():
                continue
            # Same normalized UV bins as the trained ABC patch targets.
            pixels = (box.reshape(2, 2) + 1) * np.array([width - 1, height - 1]) / 2
            draw.rectangle(tuple(pixels.reshape(-1)), outline=color, width=2)
            draw.text(tuple(pixels[0]), f"{prefix}-{arm} patch", fill=color)
        xyz = np.asarray(values.get("xyz_world", []), dtype=float)
        if xyz.shape != (2, 3):
            continue
        uv, valid = project_xyz(xyz, camera_projection(env.model, env.data, camera, height, width))
        for arm, ((x, y), visible) in enumerate(zip(uv, valid, strict=True)):
            if not visible or not (0 <= x < width and 0 <= y < height):
                continue
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=color, width=2)
            draw.line((x - 7, y, x + 7, y), fill=color, width=2)
            draw.line((x, y - 7, x, y + 7), fill=color, width=2)
            draw.text((x + 6, y + 5), f"{prefix}-{'L' if arm == 0 else 'R'} XYZ", fill=color)


def draw_frame(env, images, cameras, xyz, goal_uv, label, *, auxiliaries=None, goal_valid=None):
    frames = []
    for camera in cameras:
        frame = np.moveaxis(np.asarray(images[camera], dtype=np.uint8), 0, -1).copy()
        height, width = frame.shape[:2]
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        if camera == "top" and xyz is not None:
            uv, valid = project_xyz(xyz, camera_projection(env.model, env.data, camera, height, width))
            for arm, color in enumerate(ARM_COLORS):
                # Never join over points behind the camera. PIL clips segments
                # to the image; cap extreme projections before integer conversion.
                points = np.clip(uv[:, arm], -10_000, 10_000)
                for i in range(len(points) - 1):
                    if valid[i, arm] and valid[i + 1, arm]:
                        draw.line([tuple(points[i]), tuple(points[i + 1])], fill=color, width=2)
                if len(points) and valid[-1, arm]:
                    x, y = points[-1]
                    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        if camera == "top" and goal_uv is not None:
            valid_goals = np.ones(2, dtype=bool) if goal_valid is None else goal_valid
            for uv, color, valid in zip(np.asarray(goal_uv).reshape(2, 2), ARM_COLORS, valid_goals, strict=True):
                if not valid or not np.isfinite(uv).all():
                    continue
                x, y = (uv + 1) * np.array([width - 1, height - 1]) / 2
                draw.line((x - 5, y, x + 5, y), fill=color, width=2)
                draw.line((x, y - 5, x, y + 5), fill=color, width=2)
        if auxiliaries:
            draw_auxiliaries(draw, env, camera, height, width, auxiliaries)
        frames.append(np.asarray(image))
    combined = Image.fromarray(np.concatenate(frames, axis=1))
    draw = ImageDraw.Draw(combined)
    draw.rectangle((0, 0, combined.width, 16), fill=(10, 15, 20))
    draw.text((4, 2), label, fill="white")
    return np.asarray(combined)
