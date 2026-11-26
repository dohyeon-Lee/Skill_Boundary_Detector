"""FSQ action conventions shared by training, augmentation, and encoding."""

from __future__ import annotations

import numpy as np


def prepare_fsq_actions(actions, states, action_space="dataset", gripper_indices=(-1,)):
    """DP-style deltas: first target minus current state, then action differences.

    Grippers retain their absolute command at every timestep. Conversion must
    follow boundary augmentation so the first reference matches the new start.
    The stored raw skill actions are never changed.
    """
    actions = np.asarray(actions, dtype=np.float32)
    if action_space in {"dataset", "absolute_joint"}:
        return actions
    if action_space == "delta_pose":
        from lerobot.processor.eef_delta_pose import to_eef_delta_pose_actions_numpy

        states = np.asarray(states, dtype=np.float32)
        indices = [i + 7 if i < 0 else i for i in gripper_indices]
        if (actions.ndim != 2 or actions.shape[1] != 7 or not len(actions)
                or states.shape != actions.shape or indices != [6]):
            raise ValueError('Delta-pose FSQ requires aligned 7D XYZ/rotvec/gripper targets and anchors; gripper axis 6')
        return to_eef_delta_pose_actions_numpy(actions, states[0]).astype(np.float32)
    if action_space != "delta_joint":
        raise ValueError(f"Unknown FSQ action_space: {action_space!r}")
    states = np.asarray(states, dtype=np.float32)
    if (actions.ndim != 2 or states.ndim != 2 or not len(actions)
            or len(states) < len(actions) or states.shape[1] < actions.shape[1]):
        raise ValueError("Delta-joint FSQ requires aligned nonempty action/state sequences.")
    dim = actions.shape[1]
    indices = [index + dim if index < 0 else index for index in gripper_indices]
    if any(index < 0 or index >= dim for index in indices):
        raise ValueError(f"Gripper indices {gripper_indices} do not fit action dim {dim}.")
    joint_mask = np.ones(dim, dtype=bool)
    joint_mask[indices] = False
    result = actions.copy()
    result[0, joint_mask] -= states[0, :dim][joint_mask]
    result[1:, joint_mask] -= actions[:-1, joint_mask]
    return result


def pack_action_state_context(actions, states):
    """Keep state anchors aligned while an augmentation moves skill boundaries."""
    actions = np.asarray(actions, dtype=np.float32)
    states = np.asarray(states, dtype=np.float32)
    if actions.ndim != 2 or states.ndim != 2 or states.shape[0] < len(actions) or states.shape[1] < actions.shape[1]:
        raise ValueError("Action/state context dimensions do not match.")
    return np.concatenate([actions, states[:len(actions), :actions.shape[1]]], axis=1)


def unpack_delta_joint_context(context, action_dim, gripper_indices):
    return unpack_delta_action_context(context, action_dim, gripper_indices, 'delta_joint')


def unpack_delta_action_context(context, action_dim, gripper_indices, action_space):
    """Re-anchor either delta convention after moving the skill boundary."""
    context = np.asarray(context, dtype=np.float32)
    if context.ndim != 2 or context.shape[1] != 2 * action_dim:
        raise ValueError("Delta-action boundary context must contain actions and state anchors.")
    return prepare_fsq_actions(context[:, :action_dim], context[:, action_dim:],
                               action_space, gripper_indices)
