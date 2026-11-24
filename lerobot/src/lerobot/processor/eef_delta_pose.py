"""Consecutive base-frame EEF deltas from XYZ/rotation-vector targets, no FK.

Input/output are 7D: XYZ, SO(3) rotation vector, absolute gripper. A chunk's
first target is anchored on the current measured pose; later targets on the
preceding target. Rotation uses composition, never Euler/rotvec subtraction.
"""
import numpy as np
import torch


def _quaternion(rotvec):
    angle = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    return torch.cat((torch.cos(angle / 2), rotvec * (0.5 * torch.sinc(angle / (2 * torch.pi)))), dim=-1)


def to_eef_delta_pose_actions(actions, state):
    if actions.ndim not in (2, 3) or actions.shape[-1] != 7 or state.shape[-1] != 7:
        raise ValueError('EEF delta-pose requires actions [B,7]|[B,T,7] and 7D anchor states')
    if actions.ndim == 2 and state.ndim == 1:
        return to_eef_delta_pose_actions(actions.unsqueeze(0), state.unsqueeze(0)).squeeze(0)
    if state.ndim != 2 or actions.shape[0] != state.shape[0]:
        raise ValueError('EEF pose actions and anchor batch dimensions differ')
    state = state.to(device=actions.device, dtype=actions.dtype)
    previous = (torch.cat((state[:, None, :], actions[:, :-1]), dim=1)
                if actions.ndim == 3 else state)
    target_q, anchor_q = _quaternion(actions[..., 3:6]), _quaternion(previous[..., 3:6])
    # q_target * inverse(q_anchor): spatial/base-frame rotation increment.
    tw, tv = target_q[..., :1], target_q[..., 1:]
    aw, av = anchor_q[..., :1], -anchor_q[..., 1:]
    delta_q = torch.cat((tw * aw - (tv * av).sum(-1, keepdim=True),
                        tw * av + aw * tv + torch.cross(tv, av, dim=-1)), dim=-1)
    delta_q = torch.where(delta_q[..., :1] < 0, -delta_q, delta_q)
    angle = 2 * torch.atan2(torch.linalg.vector_norm(delta_q[..., 1:], dim=-1, keepdim=True),
                            delta_q[..., :1])
    rotation = delta_q[..., 1:] * (2 / torch.sinc(angle / (2 * torch.pi)))
    return torch.cat((actions[..., :3] - previous[..., :3], rotation, actions[..., 6:7]), dim=-1)


def to_eef_delta_pose_actions_numpy(actions, state):
    """Vectorized CPU counterpart used for statistics and SBD target probes."""
    from scipy.spatial.transform import Rotation
    actions, state = np.asarray(actions), np.asarray(state)
    if actions.ndim not in (2, 3) or actions.shape[-1] != 7 or state.shape[-1] != 7:
        raise ValueError('EEF delta-pose requires 7D XYZ/rotvec/gripper actions and states')
    if actions.ndim == 2 and state.ndim == 1:
        return to_eef_delta_pose_actions_numpy(actions[None], state[None])[0]
    if state.ndim != 2 or actions.shape[0] != state.shape[0]:
        raise ValueError('EEF pose actions and anchor batch dimensions differ')
    previous = np.concatenate((state[:, None], actions[:, :-1]), axis=1) if actions.ndim == 3 else state
    rotations = (Rotation.from_rotvec(actions[..., 3:6].reshape(-1, 3)) *
                 Rotation.from_rotvec(previous[..., 3:6].reshape(-1, 3)).inv()).as_rotvec()
    return np.concatenate((actions[..., :3] - previous[..., :3],
                           rotations.reshape(actions.shape[:-1] + (3,)), actions[..., 6:7]), axis=-1)
