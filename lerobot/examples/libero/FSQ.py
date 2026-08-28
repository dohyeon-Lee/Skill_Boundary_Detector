"""Trajectory FSQ with a transformer reconstructor and configurable terminator.

The training unit is a skill trajectory, not a frame:

    B complete skill trajectories -> spline encoder once -> z_q (B, D)
    M sampled timesteps / trajectory -> reconstructor + terminator (B*M)

The action branch is deliberately image-free here. It predicts an action chunk
from the skill code, skill-start state, and per-sample progress. The historical
terminator keeps the skillVLA query contract; the optional fusion terminator
instead gives skill, state, and unpooled camera tokens full self-attention and
reads progress/end directly from the updated skill token.

Checkpoint format v3 has three explicit, independently loadable components:

    encoder.*        trajectory -> FSQ code
    reconstructor.*  (start_state, progress, z_q) -> action chunk
    terminator.*     (image, raw state, z_q) -> progress/end

There is intentionally no PI05/Gemma action expert in this FSQ variant.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from lerobot.policies.skill_aux.modeling_state_terminator import (
    StateSkillMLPTerminator,
    StateSkillRNNTerminator,
)
from lerobot.policies.skillVLA.skill_jitter import (
    normalize_jitter_distribution,
    sample_p,
)


FORMAT_VERSION = 3
ENCODER_GROUNDING_CONVENTION = "trajectory_mean_xyz_v1"
START_GROUNDED_CONVENTION = "trajectory_start_se3_v1"
N_GRIPPER_DIMS = 2
N_POSITION_DIMS = 3
N_ROTATION_DIMS = 3
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_IMAGE_MODEL = str(_REPO_ROOT / "models" / "dinov3-vits16")
_DEFAULT_PI_BASE = str(_REPO_ROOT / "models" / "pi05_base")
ACTION_SEQUENCE_RECONSTRUCTOR_ARCHS = frozenset(
    {"action_seq", "action_seq_transformer"}
)

log = logging.getLogger(__name__)


def is_action_sequence_reconstructor_arch(value: str) -> bool:
    return str(value) in ACTION_SEQUENCE_RECONSTRUCTOR_ARCHS


def scale_gripper_features(
    values: np.ndarray,
    *,
    gripper_weight: float,
    gripper_dims: int,
) -> np.ndarray:
    """Apply an exact MSE weight to trailing normalized gripper features."""
    array = np.asarray(values, dtype=np.float32)
    weight = float(gripper_weight)
    if not math.isfinite(weight) or not 0.0 < weight <= 1.0:
        raise ValueError(f"gripper_weight must be in (0, 1], got {weight}.")
    dims = int(gripper_dims)
    if dims < 1 or dims > array.shape[-1]:
        raise ValueError(
            f"gripper_dims must be in [1, {array.shape[-1]}], got {dims}."
        )
    if weight == 1.0:
        return array
    scaled = array.copy()
    scaled[..., -dims:] *= math.sqrt(weight)
    return scaled


def normalize_action_sequence(
    actions: np.ndarray,
    action_q01: np.ndarray,
    action_q99: np.ndarray,
    *,
    gripper_weight: float = 1.0,
    clip: bool = True,
) -> np.ndarray:
    """Map dataset-unit actions to the normalized action-sequence contract.

    Every action axis uses the dataset-wide q01/q99 anchors. Values outside
    that robust range are clipped because the matched decoder has a tanh
    output. The final action axis is multiplied by ``sqrt(gripper_weight)`` in
    both the encoder input and reconstruction target, so its squared-error
    contribution is weighted by exactly ``gripper_weight``.
    """
    values = np.asarray(actions, dtype=np.float32)
    lo = np.asarray(action_q01, dtype=np.float32)
    hi = np.asarray(action_q99, dtype=np.float32)
    if values.shape[-1] != lo.shape[-1] or hi.shape != lo.shape:
        raise ValueError(
            "Action/stat shapes do not match: "
            f"actions={values.shape}, q01={lo.shape}, q99={hi.shape}."
        )
    normalized = 2.0 * (values - lo) / (hi - lo + 1e-8) - 1.0
    if clip:
        normalized = np.clip(normalized, -1.0, 1.0)
    return scale_gripper_features(
        normalized,
        gripper_weight=gripper_weight,
        gripper_dims=1,
    )


# -----------------------------------------------------------------------------
# Spline trajectory encoder
# -----------------------------------------------------------------------------


def zero_ground_trajectory(trajectory: np.ndarray) -> np.ndarray:
    """Center XYZ at the skill mean; retain absolute rotation and gripper state."""
    traj = np.asarray(trajectory, dtype=np.float32).copy()
    if len(traj) == 0:
        raise ValueError("Cannot encode an empty skill trajectory.")
    if traj.shape[-1] < N_POSITION_DIMS + N_GRIPPER_DIMS:
        raise ValueError(
            f"Expected at least {N_POSITION_DIMS + N_GRIPPER_DIMS} state dimensions, "
            f"got {traj.shape[-1]}."
        )
    traj[:, :N_POSITION_DIMS] -= traj[:, :N_POSITION_DIMS].mean(axis=0, keepdims=True)
    return traj


def start_ground_trajectory(trajectory: np.ndarray) -> np.ndarray:
    """Express the 6D EEF pose relative to its first pose; retain gripper state.

    Position is rotated into the starting EEF frame and rotation is composed on
    SO(3), rather than subtracting axis-angle vector components.  For absolute
    poses ``(p_t, R_t)`` this produces ``(R_0^-1 (p_t - p_0), R_0^-1 R_t)``.
    """
    # Keep scipy optional for model construction and non-spline encoder paths.
    from scipy.spatial.transform import Rotation

    traj = np.asarray(trajectory, dtype=np.float32).copy()
    if len(traj) == 0:
        raise ValueError("Cannot encode an empty skill trajectory.")
    pose_dims = N_POSITION_DIMS + N_ROTATION_DIMS
    required_dims = pose_dims + N_GRIPPER_DIMS
    if traj.shape[-1] < required_dims:
        raise ValueError(
            f"Start-grounded pose requires at least {required_dims} state dimensions "
            f"(XYZ + rotation vector + gripper), got {traj.shape[-1]}."
        )

    absolute_positions = traj[:, :N_POSITION_DIMS].astype(np.float64, copy=True)
    absolute_rotations = Rotation.from_rotvec(
        traj[:, N_POSITION_DIMS:pose_dims].astype(np.float64, copy=False)
    )
    start_rotation_inv = absolute_rotations[0].inv()
    relative_positions = start_rotation_inv.apply(
        absolute_positions - absolute_positions[0]
    )
    relative_rotations = start_rotation_inv * absolute_rotations

    traj[:, :N_POSITION_DIMS] = relative_positions.astype(np.float32)
    traj[:, N_POSITION_DIMS:pose_dims] = relative_rotations.as_rotvec().astype(np.float32)
    # Remove tiny numerical residue and make the contract exact at t=0.
    traj[0, :pose_dims] = 0.0
    return traj


def encoder_grounding_convention(input_mode: str) -> str:
    """Return the checkpoint convention corresponding to an encoder input mode."""
    if input_mode == "start_grounded":
        return START_GROUNDED_CONVENTION
    return ENCODER_GROUNDING_CONVENTION


def encoder_grounding_position(trajectory: np.ndarray) -> np.ndarray:
    """Absolute mean XYZ used as the trajectory's zero-grounding reference."""
    traj = np.asarray(trajectory, dtype=np.float32)
    if len(traj) == 0:
        raise ValueError("Cannot encode an empty skill trajectory.")
    if traj.shape[-1] < N_POSITION_DIMS + N_GRIPPER_DIMS:
        raise ValueError(
            f"Expected at least {N_POSITION_DIMS + N_GRIPPER_DIMS} state dimensions, "
            f"got {traj.shape[-1]}."
        )
    return traj[:, :N_POSITION_DIMS].mean(axis=0, dtype=np.float32)


def encoder_start_eef_pose(trajectory: np.ndarray) -> np.ndarray:
    """Compatibility alias for the optimal encoder's 3D grounding position."""
    return encoder_grounding_position(trajectory)


def prepare_encoder_trajectory(
    trajectory: np.ndarray,
    input_mode: str,
) -> np.ndarray:
    """Apply the checkpointed FSQ-encoder input convention before spline fitting."""
    traj = np.asarray(trajectory, dtype=np.float32)
    if len(traj) == 0:
        raise ValueError("Cannot encode an empty skill trajectory.")
    if input_mode in {"zero_grounded", "optimal"}:
        return zero_ground_trajectory(traj)
    if input_mode == "start_grounded":
        return start_ground_trajectory(traj)
    if input_mode == "raw_state":
        return traj.copy()
    raise ValueError(
        "encoder_input_mode must be zero_grounded|start_grounded|raw_state|optimal, "
        f"got {input_mode!r}."
    )


def spline_encode(
    trajectory: np.ndarray,
    n_control: int,
    degree: int,
    *,
    input_mode: str = "zero_grounded",
) -> tuple[np.ndarray, int]:
    """Trajectory -> fixed control points and original skill length."""
    # SciPy is unnecessary for action_seq and state-RNN module construction.
    # Import it only when the spline codec is actually selected.
    from scipy.interpolate import make_interp_spline

    trajectory = prepare_encoder_trajectory(trajectory, input_mode)
    length, dim = trajectory.shape
    if length == 1:
        return np.repeat(trajectory, n_control, axis=0).astype(np.float32), length
    t_orig = np.linspace(0.0, 1.0, length)
    t_ctrl = np.linspace(0.0, 1.0, n_control)
    ctrl = np.zeros((n_control, dim), dtype=np.float32)
    for d in range(dim):
        k = 1 if d >= dim - N_GRIPPER_DIMS else degree
        k = min(k, length - 1)
        ctrl[:, d] = make_interp_spline(t_orig, trajectory[:, d], k=k)(t_ctrl)
    return ctrl, length


class TokenTransformerPool(nn.Module):
    """Transformer over an ordered token set followed by learned-query pooling."""

    def __init__(self, hidden_dim: int, n_tokens: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        if hidden_dim % n_heads:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by n_heads={n_heads}.")
        self.n_tokens = int(n_tokens)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pool = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.gradient_checkpointing = False
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.query, std=0.02)

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    @staticmethod
    def _broadcast_like(cond: Tensor, x: Tensor) -> Tensor:
        if cond.ndim == 2:
            cond = cond.unsqueeze(1)
        if cond.ndim != 3 or cond.shape[0] != x.shape[0] or cond.shape[-1] != x.shape[-1]:
            raise ValueError(
                "broadcast_cond must have shape (B,H) or (B,1|N,H), got "
                f"{tuple(cond.shape)} for hidden states {tuple(x.shape)}."
            )
        if cond.shape[1] not in (1, x.shape[1]):
            raise ValueError(
                "broadcast_cond token dimension must be 1 or match hidden states, got "
                f"{cond.shape[1]} and {x.shape[1]}."
            )
        return cond.to(device=x.device, dtype=x.dtype)

    @staticmethod
    def _broadcast_layer(layer: nn.TransformerEncoderLayer, x: Tensor, cond: Tensor) -> Tensor:
        if layer.norm_first:
            attn_in = layer.norm1(x) + cond
            attn, _ = layer.self_attn(attn_in, attn_in, attn_in, need_weights=False)
            x = x + layer.dropout1(attn)
            ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(x)))))
            return x + layer.dropout2(ff)
        attn_in = x + cond
        attn, _ = layer.self_attn(attn_in, attn_in, attn_in, need_weights=False)
        x = layer.norm1(x + layer.dropout1(attn))
        ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
        return layer.norm2(x + layer.dropout2(ff))

    def _checkpoint_enabled(self) -> bool:
        return bool(self.gradient_checkpointing and self.training)

    def _encoder_plain(self, x: Tensor) -> Tensor:
        if not self._checkpoint_enabled():
            return self.encoder(x)
        for layer in self.encoder.layers:
            x = torch.utils.checkpoint.checkpoint(
                layer, x, use_reentrant=False, preserve_rng_state=False)
        if self.encoder.norm is not None:
            x = self.encoder.norm(x)
        return x

    def _encoder_with_broadcast(self, x: Tensor, cond: Tensor) -> Tensor:
        """Condition each self-attention calculation without accumulating cond in the residual stream."""
        cond = self._broadcast_like(cond, x)
        for layer in self.encoder.layers:
            if self._checkpoint_enabled():
                x = torch.utils.checkpoint.checkpoint(
                    lambda hidden, layer=layer: self._broadcast_layer(layer, hidden, cond),
                    x,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                x = self._broadcast_layer(layer, x, cond)
        if self.encoder.norm is not None:
            x = self.encoder.norm(x)
        return x

    def forward(self, tokens: Tensor, broadcast_cond: Tensor | None = None) -> Tensor:
        if tokens.shape[1] != self.n_tokens:
            raise ValueError(f"Expected {self.n_tokens} tokens, got {tokens.shape[1]}.")
        x = tokens + self.pos_embed.to(tokens.dtype)
        x = self._encoder_plain(x) if broadcast_cond is None else self._encoder_with_broadcast(x, broadcast_cond)
        query = self.query.to(x.dtype).expand(x.shape[0], -1, -1)
        pooled, _ = self.pool(query, x, x, need_weights=False)
        return self.out_norm(pooled[:, 0])


class FSQ(nn.Module):
    """Finite Scalar Quantizer with a straight-through rounded grid."""

    def __init__(self, levels: list[int]):
        super().__init__()
        if not levels or any(int(x) < 2 for x in levels):
            raise ValueError(f"FSQ levels must all be >=2, got {levels}.")
        lv = torch.tensor(levels, dtype=torch.float32)
        levels_half = (lv - 1.0) / 2.0
        offset = torch.where(lv % 2 == 0, torch.full_like(lv, 0.5), torch.zeros_like(lv))
        is_binary = lv == 2
        # The even-level offset needs atanh(1) for L=2.  A zero shift keeps
        # both raw grid values {-1, 0} reachable without a singularity.
        shift_arg = torch.where(is_binary, torch.zeros_like(lv), offset / levels_half)
        shift = torch.atanh(shift_arg)
        half_width = torch.div(lv, 2, rounding_mode="floor")
        strides = torch.ones(len(levels), dtype=torch.long)
        for i in range(1, len(levels)):
            strides[i] = strides[i - 1] * int(levels[i - 1])
        self.register_buffer("levels", lv, persistent=False)
        self.register_buffer("levels_half", levels_half)
        self.register_buffer("offset", offset, persistent=False)
        self.register_buffer("shift", shift, persistent=False)
        self.register_buffer("half_width", half_width, persistent=False)
        self.register_buffer("level_min", -half_width, persistent=False)
        self.register_buffer("level_max", (lv - 1) - half_width, persistent=False)
        self.register_buffer("strides", strides)
        self.codebook_size = int(np.prod(levels))
        self.latent_dim = len(levels)

    def bound(self, z: Tensor) -> Tensor:
        """Map continuous encoder outputs onto the coordinate system rounded by FSQ."""
        half = self.levels_half.to(z.dtype)
        return torch.tanh(z + self.shift.to(z.dtype)) * half - self.offset.to(z.dtype)

    def boundary_margin(self, z: Tensor) -> Tensor:
        """Distance to the nearest rounding boundary, in unit-width FSQ-bin coordinates.

        A margin of 0 lies exactly on a decision boundary and 0.5 is the
        center of a bin.  The returned tensor retains the FSQ latent axes so
        callers can take the minimum across axes for sample-level stability.
        """
        bounded = self.bound(z)
        return (0.5 - (bounded - torch.round(bounded)).abs()).clamp_(0.0, 0.5)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        bounded = self.bound(z)
        z_int = torch.round(bounded)
        z_q = bounded + (z_int - bounded).detach()
        index = ((z_int + self.half_width.to(z.dtype)).long() * self.strides).sum(dim=-1)
        return z_q, index

    def normalized(self, z_q: Tensor) -> Tensor:
        """Raw FSQ coordinate -> Stage-1 centered grid coordinate in [-1,1]."""
        return (z_q + self.offset.to(z_q.dtype)) / self.levels_half.to(z_q.dtype)

    def code_to_normalized(self, code: Tensor) -> Tensor:
        idx = code.view(-1, 1).long()
        level_ids = torch.div(idx, self.strides[None], rounding_mode="floor") % self.levels[None].long()
        return (level_ids.float() - self.levels_half[None]) / self.levels_half[None]


class BSQ(nn.Module):
    """Binary spherical quantizer with the same runtime interface as :class:`FSQ`.

    Encoder outputs are projected onto the unit sphere, then each coordinate is
    quantized by its sign. Dimension 0 remains the fastest-changing index, so
    downstream token IDs follow the existing FSQ convention. There is
    deliberately no entropy/confidence objective here: this BSQ path trains
    with the same reconstruction and optional pair losses as FSQ.
    """

    def __init__(self, code_dim: int):
        super().__init__()
        if int(code_dim) < 2:
            raise ValueError(f"BSQ code_dim must be >= 2, got {code_dim}.")
        self.code_dim = int(code_dim)
        self.latent_dim = self.code_dim
        self.codebook_size = 2**self.code_dim
        self.register_buffer(
            "bit_weights",
            2 ** torch.arange(self.code_dim, dtype=torch.long),
            persistent=False,
        )

    def unit(self, z: Tensor) -> Tensor:
        return F.normalize(z.float(), dim=-1, eps=1e-8).to(z.dtype)

    def bound(self, z: Tensor) -> Tensor:
        """Continuous coordinate consumed by the sign quantizer and pair loss."""
        return self.unit(z)

    def boundary_margin(self, z: Tensor) -> Tensor:
        """Distance to a sign flip, mapped onto FSQ's common 0--0.5 scale."""
        u = self.unit(z.float())
        return (u.abs() * math.sqrt(self.code_dim) * 0.5).clamp(0.0, 0.5)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        u = self.unit(z)
        signs = torch.where(u >= 0, torch.ones_like(u), -torch.ones_like(u))
        corner = signs / math.sqrt(self.code_dim)
        z_q = u + (corner - u).detach()
        index = ((u >= 0).long() * self.bit_weights).sum(dim=-1)
        return z_q, index

    def normalized(self, z_q: Tensor) -> Tensor:
        return z_q * math.sqrt(self.code_dim)

    def code_to_normalized(self, code: Tensor) -> Tensor:
        idx = code.view(-1, 1).long()
        bits = torch.div(
            idx,
            self.bit_weights[None].to(idx.device),
            rounding_mode="floor",
        ) % 2
        return bits.float() * 2.0 - 1.0


class SplineFSQEncoder(nn.Module):
    """Full-trajectory spline encoder. It runs once for each trajectory in B."""

    def __init__(
        self,
        enc_dim: int,
        n_control: int,
        spline_degree: int,
        hidden_dim: int,
        fsq_levels: list[int],
        n_layers: int,
        n_heads: int,
        dropout: float,
        length_min: float,
        length_max: float,
        encoder_min: np.ndarray,
        encoder_max: np.ndarray,
        encoder_input_mode: str = "zero_grounded",
        encoder_start_min: np.ndarray | None = None,
        encoder_start_max: np.ndarray | None = None,
        gripper_weight: float = 1.0,
    ):
        super().__init__()
        if encoder_input_mode not in {
            "zero_grounded", "start_grounded", "raw_state", "optimal"
        }:
            raise ValueError(
                "encoder_input_mode must be zero_grounded|start_grounded|raw_state|optimal, "
                f"got {encoder_input_mode!r}."
            )
        self.enc_dim = int(enc_dim)
        self.n_control = int(n_control)
        self.spline_degree = int(spline_degree)
        self.encoder_input_mode = encoder_input_mode
        self.gripper_weight = float(gripper_weight)
        if N_GRIPPER_DIMS >= self.enc_dim:
            raise ValueError(
                f"Expected encoder dim > {N_GRIPPER_DIMS}, got {self.enc_dim}."
            )
        self.length_min = float(length_min)
        self.length_max = float(length_max)
        self.enc_ctrl_proj = nn.Linear(enc_dim, hidden_dim)
        self.enc_len_proj = nn.Linear(1, hidden_dim)
        self.enc_start_proj: nn.Linear | None = None
        if encoder_input_mode == "optimal":
            if encoder_start_min is None or encoder_start_max is None:
                raise ValueError("optimal encoder mode requires encoder_start_min/max.")
            start_min = torch.as_tensor(encoder_start_min, dtype=torch.float32)
            start_max = torch.as_tensor(encoder_start_max, dtype=torch.float32)
            if start_min.shape != (N_POSITION_DIMS,) or start_max.shape != (N_POSITION_DIMS,):
                raise ValueError(
                    "optimal encoder grounding statistics must each contain mean XYZ "
                    f"({N_POSITION_DIMS} values), got {tuple(start_min.shape)} and "
                    f"{tuple(start_max.shape)}."
                )
            self.enc_start_proj = nn.Linear(N_POSITION_DIMS, hidden_dim)
            # Historical checkpoint names are retained, but these buffers now
            # normalize the skill's mean XYZ rather than its start EEF pose.
            self.register_buffer("encoder_start_min", start_min)
            self.register_buffer("encoder_start_max", start_max)
        self.enc_traj_pool = TokenTransformerPool(
            hidden_dim, n_control + 1 + int(encoder_input_mode == "optimal"),
            n_layers=n_layers, n_heads=n_heads, dropout=dropout,
        )
        self.z_head = nn.Linear(hidden_dim, len(fsq_levels))
        self.fsq = FSQ(fsq_levels)
        self.register_buffer("encoder_min", torch.as_tensor(encoder_min, dtype=torch.float32))
        self.register_buffer("encoder_max", torch.as_tensor(encoder_max, dtype=torch.float32))

    def normalize_control_points(self, ctrl: Tensor) -> Tensor:
        lo = self.encoder_min.to(ctrl.device, ctrl.dtype)
        hi = self.encoder_max.to(ctrl.device, ctrl.dtype)
        normalized = 2.0 * (ctrl - lo) / (hi - lo + 1e-8) - 1.0
        if self.gripper_weight != 1.0:
            normalized = normalized.clone()
            normalized[..., -N_GRIPPER_DIMS:] *= math.sqrt(
                self.gripper_weight
            )
        return normalized

    def normalize_start_pose(self, start_pose: Tensor) -> Tensor:
        if self.enc_start_proj is None:
            raise ValueError("Only optimal encoder mode accepts a grounding position.")
        lo = self.encoder_start_min.to(start_pose.device, start_pose.dtype)
        hi = self.encoder_start_max.to(start_pose.device, start_pose.dtype)
        return 2.0 * (start_pose - lo) / (hi - lo + 1e-8) - 1.0

    def encode_continuous(
        self,
        ctrl: Tensor,
        lengths: Tensor,
        start_pose: Tensor | None = None,
        *,
        normalized: bool = True,
    ) -> Tensor:
        if not normalized:
            ctrl = self.normalize_control_points(ctrl)
            if start_pose is not None:
                start_pose = self.normalize_start_pose(start_pose)
        bsize = ctrl.shape[0]
        ctrl_tok = self.enc_ctrl_proj(ctrl)
        length_norm = (
            2.0 * (lengths.float() - self.length_min) / (self.length_max - self.length_min + 1e-8) - 1.0
        ).view(bsize, 1, 1).to(ctrl_tok.dtype)
        length_tok = self.enc_len_proj(length_norm)
        tokens = [ctrl_tok]
        if self.enc_start_proj is not None:
            if start_pose is None:
                raise ValueError("optimal encoder mode requires the mean XYZ grounding position.")
            tokens.append(self.enc_start_proj(start_pose).unsqueeze(1))
        tokens.append(length_tok)
        return self.z_head(self.enc_traj_pool(torch.cat(tokens, dim=1)))

    def forward(
        self,
        ctrl: Tensor,
        lengths: Tensor,
        start_pose: Tensor | None = None,
        *,
        normalized: bool = True,
    ) -> tuple[Tensor, Tensor]:
        z_e = self.encode_continuous(
            ctrl,
            lengths,
            start_pose,
            normalized=normalized,
        )
        return self.fsq(z_e)

    @torch.no_grad()
    def encode_numpy(self, trajectory: np.ndarray, device: str | torch.device = "cpu") -> np.ndarray:
        ctrl, length = spline_encode(
            trajectory,
            self.n_control,
            self.spline_degree,
            input_mode=self.encoder_input_mode,
        )
        ctrl_t = torch.from_numpy(ctrl).float().unsqueeze(0).to(device)
        length_t = torch.tensor([length], dtype=torch.long, device=device)
        start_t = None
        if self.enc_start_proj is not None:
            start_t = torch.from_numpy(
                encoder_grounding_position(trajectory)
            ).float().unsqueeze(0).to(device)
        z_q, _ = self(ctrl_t, length_t, start_t, normalized=False)
        return z_q[0].cpu().numpy()

    @torch.no_grad()
    def encode_index(self, trajectory: np.ndarray, device: str | torch.device = "cpu") -> int:
        ctrl, length = spline_encode(
            trajectory,
            self.n_control,
            self.spline_degree,
            input_mode=self.encoder_input_mode,
        )
        ctrl_t = torch.from_numpy(ctrl).float().unsqueeze(0).to(device)
        length_t = torch.tensor([length], dtype=torch.long, device=device)
        start_t = None
        if self.enc_start_proj is not None:
            start_t = torch.from_numpy(
                encoder_grounding_position(trajectory)
            ).float().unsqueeze(0).to(device)
        _, index = self(ctrl_t, length_t, start_t, normalized=False)
        return int(index.item())


class LengthFreeSplineFSQEncoder(SplineFSQEncoder):
    """SplineFSQEncoder without the length token (encoder_length_token=False).

    Control points live on normalized time, so with no length token the
    absolute duration reaches z only through motion shape. ``lengths`` is
    still accepted everywhere for API compatibility but never enters the
    computation. Checkpoint shapes differ from the parent (smaller pos_embed,
    no enc_len_proj), so runs can never silently cross-load.
    """

    def __init__(
        self,
        *,
        enc_dim: int,
        n_control: int,
        spline_degree: int,
        hidden_dim: int,
        fsq_levels: list[int],
        n_layers: int,
        n_heads: int,
        dropout: float,
        length_min: float,
        length_max: float,
        encoder_min: np.ndarray,
        encoder_max: np.ndarray,
        encoder_input_mode: str = "zero_grounded",
        encoder_start_min: np.ndarray | None = None,
        encoder_start_max: np.ndarray | None = None,
        gripper_weight: float = 1.0,
    ):
        super().__init__(
            enc_dim=enc_dim,
            n_control=n_control,
            spline_degree=spline_degree,
            hidden_dim=hidden_dim,
            fsq_levels=fsq_levels,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            length_min=length_min,
            length_max=length_max,
            encoder_min=encoder_min,
            encoder_max=encoder_max,
            encoder_input_mode=encoder_input_mode,
            encoder_start_min=encoder_start_min,
            encoder_start_max=encoder_start_max,
            gripper_weight=gripper_weight,
        )
        self.enc_len_proj = None
        self.enc_traj_pool = TokenTransformerPool(
            hidden_dim,
            n_control + int(encoder_input_mode == "optimal"),
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

    def encode_continuous(
        self,
        ctrl: Tensor,
        lengths: Tensor,
        start_pose: Tensor | None = None,
        *,
        normalized: bool = True,
    ) -> Tensor:
        _ = lengths  # accepted for API compatibility; duration is not an input
        if not normalized:
            ctrl = self.normalize_control_points(ctrl)
            if start_pose is not None:
                start_pose = self.normalize_start_pose(start_pose)
        tokens = [self.enc_ctrl_proj(ctrl)]
        if self.enc_start_proj is not None:
            if start_pose is None:
                raise ValueError("optimal encoder mode requires the mean XYZ grounding position.")
            tokens.append(self.enc_start_proj(start_pose).unsqueeze(1))
        return self.z_head(self.enc_traj_pool(torch.cat(tokens, dim=1)))


class ActionSeqEncoder(nn.Module):
    """Variable-length ACTION-sequence encoder (encoder_arch='action_seq').

    Consumes the caller-provided action sequence directly. The matched main
    action_seq autoencoder and FSQ-original v2 supply native raw controller
    actions; legacy main action_seq-to-chunk checkpoints retain their historical
    q01/q99 convention. There is no spline codec, grounding decision
    (delta actions carry no absolute pose), or length token (duration is
    implicit in the sequence). Sinusoidal step-index
    positions keep absolute timing visible; padding is masked in every
    attention, so z is invariant to the batch pad width. Exposes ``z_head`` /
    ``fsq`` under the same names as SplineFSQEncoder so quantizer swaps and
    codebook diagnostics reuse unchanged.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        hidden_dim: int,
        fsq_levels: list[int],
        n_layers: int,
        n_heads: int,
        dropout: float,
        raw_actions: bool = False,
        action_q01: np.ndarray | None = None,
        action_q99: np.ndarray | None = None,
        clip_normalized_actions: bool = False,
        action_gripper_weight: float = 1.0,
    ):
        super().__init__()
        if hidden_dim % 2 or hidden_dim % n_heads:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be even and divisible by n_heads={n_heads}."
            )
        self.hidden_dim = int(hidden_dim)
        self.raw_actions = bool(raw_actions)
        self.action_q01 = (
            None if action_q01 is None else np.asarray(action_q01, dtype=np.float32)
        )
        self.action_q99 = (
            None if action_q99 is None else np.asarray(action_q99, dtype=np.float32)
        )
        self.clip_normalized_actions = bool(clip_normalized_actions)
        self.action_gripper_weight = float(action_gripper_weight)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pool = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.z_head = nn.Linear(hidden_dim, len(fsq_levels))
        self.fsq = FSQ(fsq_levels)
        nn.init.trunc_normal_(self.query, std=0.02)

    @staticmethod
    def _sinusoidal_positions(steps: int, dim: int, device: torch.device) -> Tensor:
        position = torch.arange(steps, device=device, dtype=torch.float32)[:, None]
        div = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            / dim
        )
        pe = torch.zeros(steps, dim, device=device)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        return pe

    def encode_continuous(
        self,
        actions: Tensor,
        lengths: Tensor,
        start_pose: Tensor | None = None,
        *,
        normalized: bool = True,
    ) -> Tensor:
        # API parity with SplineFSQEncoder; this module never rescales actions.
        _ = start_pose, normalized
        bsize, steps, _ = actions.shape
        pad_mask = torch.arange(steps, device=actions.device)[None] >= lengths[:, None]
        x = self.action_proj(actions)
        x = x + self._sinusoidal_positions(steps, self.hidden_dim, actions.device)[None].to(x.dtype)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        query = self.query.to(x.dtype).expand(bsize, -1, -1)
        pooled, _ = self.pool(query, x, x, key_padding_mask=pad_mask, need_weights=False)
        return self.z_head(self.out_norm(pooled[:, 0]))

    def forward(
        self,
        actions: Tensor,
        lengths: Tensor,
        start_pose: Tensor | None = None,
        *,
        normalized: bool = True,
    ) -> tuple[Tensor, Tensor]:
        return self.fsq(self.encode_continuous(actions, lengths, start_pose, normalized=normalized))

    def _prepare_actions_numpy(self, actions: np.ndarray) -> Tensor:
        actions = np.asarray(actions, dtype=np.float32)
        if self.raw_actions:
            scaled = scale_gripper_features(
                actions,
                gripper_weight=self.action_gripper_weight,
                gripper_dims=1,
            )
            return torch.from_numpy(scaled)
        if self.action_q01 is None or self.action_q99 is None:
            raise ValueError(
                "This action-sequence encoder needs q01/q99 statistics for its "
                "legacy normalized-action contract."
            )
        normalized = normalize_action_sequence(
            actions,
            self.action_q01,
            self.action_q99,
            gripper_weight=self.action_gripper_weight,
            clip=self.clip_normalized_actions,
        )
        return torch.from_numpy(normalized)

    @torch.no_grad()
    def encode_actions_numpy(
        self, actions: np.ndarray, device: str | torch.device = "cpu"
    ) -> np.ndarray:
        sequence = self._prepare_actions_numpy(actions).unsqueeze(0).to(device)
        lengths = torch.tensor([sequence.shape[1]], dtype=torch.long, device=device)
        z_q, _ = self(sequence, lengths)
        return z_q[0].cpu().numpy()

    @torch.no_grad()
    def encode_actions_index(
        self, actions: np.ndarray, device: str | torch.device = "cpu"
    ) -> int:
        sequence = self._prepare_actions_numpy(actions).unsqueeze(0).to(device)
        lengths = torch.tensor([sequence.shape[1]], dtype=torch.long, device=device)
        _, index = self(sequence, lengths)
        return int(index.item())


def fsq_soft_assignments(
    bounded: Tensor,
    fsq_levels: list[int],
    inv_temperature: float,
) -> list[Tensor]:
    """Dimension-wise soft assignments on the integer-spaced FSQ grid."""
    if inv_temperature <= 0:
        raise ValueError(
            f"FSQ soft-assignment inverse temperature must be positive, got {inv_temperature}."
        )
    probs: list[Tensor] = []
    for d, level in enumerate(int(v) for v in fsq_levels):
        centers = torch.arange(level, device=bounded.device, dtype=torch.float32)
        centers = centers - level // 2
        logits = -inv_temperature * (bounded[:, d : d + 1].float() - centers[None]) ** 2
        probs.append(torch.softmax(logits, dim=-1).clamp(1e-6, 1.0))
    return probs


def fsq_joint_soft_assignments(
    bounded: Tensor,
    fsq_levels: list[int],
    inv_temperature: float,
) -> Tensor:
    """Full soft-code distribution with FSQ dim 0 as the fastest index."""
    probs = fsq_soft_assignments(bounded, fsq_levels, inv_temperature)
    joint = probs[0]
    for p in probs[1:]:
        joint = (p[:, :, None] * joint[:, None, :]).reshape(joint.shape[0], -1)
    return joint / joint.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def bsq_joint_soft_assignments(
    unit: Tensor,
    inv_temperature: float,
) -> Tensor:
    """Full soft distribution over BSQ corners with bit 0 as fastest index."""
    if inv_temperature <= 0:
        raise ValueError(
            "BSQ soft-assignment inverse temperature must be positive, "
            f"got {inv_temperature}."
        )
    p_positive = torch.sigmoid(2.0 * inv_temperature * unit.float()).clamp(
        1e-6, 1.0 - 1e-6
    )
    per_bit = torch.stack((1.0 - p_positive, p_positive), dim=-1)
    joint = per_bit[:, 0]
    for dim in range(1, per_bit.shape[1]):
        p = per_bit[:, dim]
        joint = (p[:, :, None] * joint[:, None, :]).reshape(joint.shape[0], -1)
    return joint / joint.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def bsq_pair_joint_overlaps(
    unit: Tensor,
    augmented_unit: Tensor,
    inv_temperature: float,
) -> Tensor:
    """Per-pair probability of drawing the same complete BSQ code."""
    if unit.shape != augmented_unit.shape:
        raise ValueError(
            "BSQ overlap pair shapes must match, got "
            f"{tuple(unit.shape)} and {tuple(augmented_unit.shape)}."
        )
    p = torch.sigmoid(2.0 * inv_temperature * unit.float()).clamp(1e-6, 1.0 - 1e-6)
    q = torch.sigmoid(2.0 * inv_temperature * augmented_unit.float()).clamp(
        1e-6, 1.0 - 1e-6
    )
    bit_overlap = (p * q + (1.0 - p) * (1.0 - q)).clamp_min(1e-12)
    return bit_overlap.log().sum(dim=-1).exp()


def bsq_overlap_pair_loss(
    unit: Tensor,
    augmented_unit: Tensor,
    inv_temperature: float,
) -> tuple[Tensor, Tensor]:
    """BSQ counterpart of the factorized full-code overlap objective."""
    joint_overlap = bsq_pair_joint_overlaps(
        unit, augmented_unit, inv_temperature
    )
    return -joint_overlap.clamp_min(1e-12).log().mean(), joint_overlap.mean()


def bsq_js_pair_loss(
    unit: Tensor,
    augmented_unit: Tensor,
    inv_temperature: float,
) -> Tensor:
    """JS divergence between clean/augmented full soft BSQ-code distributions."""
    if unit.shape != augmented_unit.shape:
        raise ValueError(
            "BSQ JS pair shapes must match, got "
            f"{tuple(unit.shape)} and {tuple(augmented_unit.shape)}."
        )
    p = bsq_joint_soft_assignments(unit, inv_temperature)
    q = bsq_joint_soft_assignments(augmented_unit, inv_temperature)
    midpoint = 0.5 * (p + q)
    log_midpoint = midpoint.clamp_min(1e-12).log()
    js = 0.5 * (
        (p * (p.clamp_min(1e-12).log() - log_midpoint)).sum(dim=-1)
        + (q * (q.clamp_min(1e-12).log() - log_midpoint)).sum(dim=-1)
    )
    return js.mean()


def fsq_pair_joint_overlaps(
    bounded: Tensor,
    augmented_bounded: Tensor,
    fsq_levels: list[int],
    inv_temperature: float,
) -> Tensor:
    """Per-pair probability of drawing the same complete FSQ code.

    FSQ assignments factor over scalar axes. The full-code collision
    probability therefore equals the product of dimension-wise overlaps, so
    this computes the exact probability without enumerating prod(levels) codes.
    """
    if bounded.shape != augmented_bounded.shape:
        raise ValueError(
            "FSQ overlap pair shapes must match, got "
            f"{tuple(bounded.shape)} and {tuple(augmented_bounded.shape)}."
        )
    probs = fsq_soft_assignments(bounded, fsq_levels, inv_temperature)
    augmented_probs = fsq_soft_assignments(
        augmented_bounded, fsq_levels, inv_temperature
    )
    log_joint_overlap = bounded.new_zeros(bounded.shape[0], dtype=torch.float32)
    for p, p_aug in zip(probs, augmented_probs, strict=True):
        overlap = (p * p_aug).sum(dim=-1).clamp_min(1e-12)
        log_joint_overlap = log_joint_overlap + overlap.log()
    return log_joint_overlap.exp()


def fsq_overlap_pair_loss(
    bounded: Tensor,
    augmented_bounded: Tensor,
    fsq_levels: list[int],
    inv_temperature: float,
) -> tuple[Tensor, Tensor]:
    """Negative log probability that a pair draws one joint FSQ code."""
    joint_overlap = fsq_pair_joint_overlaps(
        bounded, augmented_bounded, fsq_levels, inv_temperature
    )
    return -joint_overlap.clamp_min(1e-12).log().mean(), joint_overlap.mean()


def fsq_js_pair_loss(
    bounded: Tensor,
    augmented_bounded: Tensor,
    fsq_levels: list[int],
    inv_temperature: float,
) -> Tensor:
    """JS divergence between clean/augmented full soft FSQ-code distributions.

    Unlike overlap loss, this is exactly zero whenever both distributions are
    equal, even if they are diffuse near a quantization boundary. It therefore
    supplies pair consistency without an additional confidence/sharpening
    pressure. Natural logarithms are used, so the per-pair range is [0, ln 2].
    """
    if bounded.shape != augmented_bounded.shape:
        raise ValueError(
            "FSQ JS pair shapes must match, got "
            f"{tuple(bounded.shape)} and {tuple(augmented_bounded.shape)}."
        )
    p = fsq_joint_soft_assignments(bounded, fsq_levels, inv_temperature)
    q = fsq_joint_soft_assignments(
        augmented_bounded, fsq_levels, inv_temperature
    )
    midpoint = 0.5 * (p + q)
    log_midpoint = midpoint.clamp_min(1e-12).log()
    js = 0.5 * (
        (p * (p.clamp_min(1e-12).log() - log_midpoint)).sum(dim=-1)
        + (q * (q.clamp_min(1e-12).log() - log_midpoint)).sum(dim=-1)
    )
    return js.mean()


def fsq_entropy_statistics(
    bounded: Tensor,
    fsq_levels: list[int],
    inv_temperature: float,
    *,
    joint_dataset: bool = False,
) -> tuple[Tensor, Tensor]:
    """Per-sample confidence entropy and dataset entropy for the FSQ grid.

    ``bounded`` is FSQ.bound's continuous coordinate (grid spacing 1, centers
    at integers). Each dim gets a soft level assignment
    p ∝ softmax(-τ·(bounded - center)²), so τ is in grid-step units: the
    confidence term's gradient lives near rounding boundaries (distance 0.5)
    and vanishes at bin centers. Returns one summed entropy per sample plus the
    scalar dataset entropy, both in nats. The joint dataset mode enumerates all
    prod(levels) codes in FSQ's index order (dim 0 fastest) — the factorized
    mode is an upper bound blind to inter-dim correlations.
    """
    probs = fsq_soft_assignments(bounded, fsq_levels, inv_temperature)
    sample_entropies = sum(-(p * p.log()).sum(dim=-1) for p in probs)
    if not joint_dataset:
        dataset_entropy = bounded.new_zeros(())
        for p in probs:
            p_bar = p.mean(dim=0)
            p_bar = p_bar / p_bar.sum()
            dataset_entropy = dataset_entropy - (p_bar * p_bar.log()).sum()
        return sample_entropies, dataset_entropy
    q = probs[0]
    for p in probs[1:]:
        q = (p[:, :, None] * q[:, None, :]).reshape(q.shape[0], -1)
    q_bar = q.mean(dim=0).clamp_min(1e-12)
    q_bar = q_bar / q_bar.sum()
    dataset_entropy = -(q_bar * q_bar.log()).sum()
    return sample_entropies, dataset_entropy


def fsq_entropy_terms(
    bounded: Tensor,
    fsq_levels: list[int],
    inv_temperature: float,
    *,
    joint_dataset: bool = False,
) -> tuple[Tensor, Tensor]:
    """BSQ-style mean sample entropy and dataset entropy in nats."""
    sample_entropies, dataset_entropy = fsq_entropy_statistics(
        bounded,
        fsq_levels,
        inv_temperature,
        joint_dataset=joint_dataset,
    )
    return sample_entropies.mean(), dataset_entropy


class _MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _StartConditionedMLPBlock(nn.Module):
    """Residual MLP block with a separate zero-init start-state AdaLN path."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

    def forward(self, hidden: Tensor, condition: Tensor) -> Tensor:
        shift, scale = self.adaln(condition).chunk(2, dim=-1)
        modulated = self.norm(hidden) * (1.0 + scale) + shift
        return hidden + self.ffn(modulated)


class OneShotTrajectoryDecoder(nn.Module):
    """z_norm [+ optional start-state context] -> normalized control-point grid
    (+ normalized length), in one shot.

    Heads are linear (no tanh/sigmoid): both targets are min/max-normalized and
    spline control points can legitimately overshoot slightly outside [-1, 1].
    ``state_dim > 0`` with ``concat`` revives the original spline_vqae
    contract. ``adaln`` instead keeps z as the content input and uses the
    normalized skill-start state only for zero-init layerwise modulation.
    Default state_dim=0 keeps the pure z-only decoder.
    """

    def __init__(
        self,
        *,
        fsq_dim: int,
        enc_dim: int,
        n_control: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
        predict_length: bool = True,
        state_dim: int = 0,
        start_state_conditioning: str = "concat",
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"decoder_layers must be >= 1, got {n_layers}.")
        self.enc_dim = int(enc_dim)
        self.n_control = int(n_control)
        self.state_dim = int(state_dim)
        self.start_state_conditioning = str(start_state_conditioning).strip().lower()
        if self.start_state_conditioning not in {"concat", "adaln"}:
            raise ValueError(
                "start_state_conditioning must be concat|adaln, "
                f"got {start_state_conditioning!r}."
            )
        if self.state_dim > 0 and self.start_state_conditioning == "adaln":
            # z remains the content path.  The start state never joins z by
            # concatenation; it only controls per-channel AdaLN parameters.
            self.z_proj = _MLPBlock(fsq_dim, hidden_dim, dropout)
            self.state_proj = nn.Sequential(
                nn.Linear(self.state_dim, hidden_dim),
                nn.SiLU(),
            )
            self.conditioned_blocks = nn.ModuleList(
                [
                    _StartConditionedMLPBlock(hidden_dim, dropout)
                    for _ in range(max(n_layers - 1, 0))
                ]
            )
            self.output_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
            self.output_adaln = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim * 2),
            )
            nn.init.zeros_(self.output_adaln[-1].weight)
            nn.init.zeros_(self.output_adaln[-1].bias)
            self.mlp = None
        else:
            # Preserve the exact legacy z-only / start-state-concat module and
            # checkpoint key layout whenever adaptive conditioning is unused.
            blocks = [_MLPBlock(fsq_dim + self.state_dim, hidden_dim, dropout)]
            for _ in range(n_layers - 1):
                blocks.append(_MLPBlock(hidden_dim, hidden_dim, dropout))
            self.mlp = nn.Sequential(*blocks)
        self.ctrl_head = nn.Linear(hidden_dim, n_control * enc_dim)
        self.length_head = nn.Linear(hidden_dim, 1) if predict_length else None

    def forward(
        self, z_norm: Tensor, start_state: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        if self.state_dim > 0:
            if start_state is None:
                raise ValueError("This oneshot decoder was built with a start-state input.")
            state = start_state[:, : self.state_dim].to(z_norm.dtype)
        else:
            state = None
        if self.state_dim > 0 and self.start_state_conditioning == "adaln":
            condition = self.state_proj(state)
            hidden = self.z_proj(z_norm)
            for block in self.conditioned_blocks:
                hidden = block(hidden, condition)
            shift, scale = self.output_adaln(condition).chunk(2, dim=-1)
            hidden = self.output_norm(hidden) * (1.0 + scale) + shift
        else:
            if state is not None:
                z_norm = torch.cat([z_norm, state], dim=-1)
            hidden = self.mlp(z_norm)
        ctrl = self.ctrl_head(hidden).view(-1, self.n_control, self.enc_dim)
        length = None if self.length_head is None else self.length_head(hidden).squeeze(-1)
        return ctrl, length


class ActionSequenceRNNDecoder(nn.Module):
    """Decode one complete raw controller-action sequence from a skill code.

    The GRU receives only the quantized skill code: the code is projected to a
    constant input token at every step and also initializes the hidden state.
    Previous ground-truth or predicted actions are never fed back, so the code
    remains the sole source of trajectory information. LIBERO controller
    commands already live in [-1, 1], matching the action head's tanh range.

    ``predict_termination`` is used by FSQ-original, whose sequence decoder also
    owns the length head. The regular FSQ path keeps termination in its existing
    independently selectable terminator branch and therefore disables it here.
    """

    def __init__(
        self,
        *,
        fsq_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
        predict_termination: bool = False,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"Action-sequence decoder layers must be >= 1, got {n_layers}.")
        self.action_dim = int(action_dim)
        self.n_layers = int(n_layers)
        self.hidden_dim = int(hidden_dim)
        self.z_proj = nn.Linear(fsq_dim, hidden_dim)
        self.h0_proj = nn.Linear(fsq_dim, n_layers * hidden_dim)
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.term_head = nn.Linear(hidden_dim, 1) if predict_termination else None

    def forward(self, z_norm: Tensor, steps: int) -> tuple[Tensor, Tensor | None]:
        bsize = z_norm.shape[0]
        z_token = self.z_proj(z_norm)
        inputs = z_token.unsqueeze(1).expand(bsize, int(steps), -1)
        h0 = (
            torch.tanh(self.h0_proj(z_norm))
            .view(bsize, self.n_layers, self.hidden_dim)
            .transpose(0, 1)
            .contiguous()
        )
        hidden, _ = self.gru(inputs, h0)
        actions = torch.tanh(self.action_head(hidden))
        term_logits = None if self.term_head is None else self.term_head(hidden).squeeze(-1)
        return actions, term_logits


class _FSQConditionedCausalDecoderBlock(nn.Module):
    """Pre-norm causal Transformer block modulated by one FSQ skill code.

    The FSQ code supplies per-channel scale and shift parameters before both
    self-attention and the feed-forward network. The modulation projection is
    zero-initialized, so a new block starts as a conventional pre-norm
    Transformer while retaining the decoder's direct z-to-query path.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        n_heads: int,
        dropout: float,
        start_conditioned: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 4),
        )
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)
        self.start_adaln = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim * 4),
            )
            if start_conditioned
            else None
        )
        if self.start_adaln is not None:
            nn.init.zeros_(self.start_adaln[-1].weight)
            nn.init.zeros_(self.start_adaln[-1].bias)

    @staticmethod
    def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
        self,
        tokens: Tensor,
        z_condition: Tensor,
        causal_mask: Tensor,
        start_condition: Tensor | None = None,
    ) -> Tensor:
        modulation = self.adaln(z_condition)
        if self.start_adaln is not None:
            if start_condition is None:
                raise ValueError("This decoder block requires a start-state condition.")
            # Keep skill and spatial-context projections independent.  Their
            # scale/shift contributions meet only at the normalized hidden state.
            modulation = modulation + self.start_adaln(start_condition)
        attn_shift, attn_scale, ffn_shift, ffn_scale = modulation.chunk(4, dim=-1)
        attn_input = self._modulate(
            self.norm1(tokens), attn_shift, attn_scale
        )
        attn_output, _ = self.self_attn(
            attn_input,
            attn_input,
            attn_input,
            attn_mask=causal_mask,
            need_weights=False,
        )
        tokens = tokens + self.attn_dropout(attn_output)
        ffn_input = self._modulate(self.norm2(tokens), ffn_shift, ffn_scale)
        return tokens + self.ffn_dropout(self.ffn(ffn_input))


class CausalActionSequenceTransformerDecoder(nn.Module):
    """Decode raw actions from z with prefix-invariant causal self-attention.

    The caller chooses only how many output slots to evaluate. Every slot sees
    the same projected FSQ code plus an absolute sinusoidal timestep encoding.
    The code additionally modulates attention and feed-forward normalization in
    every decoder block. An optional absolute start state contributes through a
    separate zero-init AdaLN projection; it is never concatenated with the code.
    No ground-truth/predicted action, normalized progress, or final sequence
    length is embedded. A strict causal mask ensures that appending future
    slots cannot change an already generated prefix.
    """

    def __init__(
        self,
        *,
        fsq_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        state_dim: int = 0,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError(
                f"Action-sequence decoder layers must be >= 1, got {n_layers}."
            )
        if hidden_dim % 2 or hidden_dim % n_heads:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be even and divisible by n_heads={n_heads}."
            )
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.state_dim = int(state_dim)
        self.z_proj = nn.Linear(fsq_dim, hidden_dim)
        self.state_proj = (
            nn.Sequential(
                nn.Linear(self.state_dim, hidden_dim),
                nn.SiLU(),
            )
            if self.state_dim > 0
            else None
        )
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.blocks = nn.ModuleList(
            [
                _FSQConditionedCausalDecoderBlock(
                    hidden_dim=hidden_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    start_conditioned=self.state_dim > 0,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(
        self,
        z_norm: Tensor,
        steps: int,
        start_state: Tensor | None = None,
    ) -> tuple[Tensor, None]:
        steps = int(steps)
        if steps < 1:
            raise ValueError(f"Action-sequence decode steps must be >= 1, got {steps}.")
        positions = ActionSeqEncoder._sinusoidal_positions(
            steps, self.hidden_dim, z_norm.device
        ).to(dtype=z_norm.dtype)
        z_condition = self.z_proj(z_norm)
        start_condition = None
        if self.state_dim > 0:
            if start_state is None:
                raise ValueError(
                    "This action-sequence decoder was built with start-state conditioning."
                )
            start_condition = self.state_proj(
                start_state[:, : self.state_dim].to(z_norm.dtype)
            )
        tokens = (
            z_condition.unsqueeze(1)
            + self.query.to(dtype=z_norm.dtype)
            + positions.unsqueeze(0)
        )
        causal_mask = torch.triu(
            torch.ones(steps, steps, device=z_norm.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = tokens
        for block in self.blocks:
            hidden = block(
                hidden,
                z_condition,
                causal_mask,
                start_condition=start_condition,
            )
        hidden = self.final_norm(hidden)
        return torch.tanh(self.action_head(hidden)), None



class MotionChunkReconstructor(nn.Module):
    """Image-free transformer decoder for normalized action chunks.

    Inputs are the normalized skill code, normalized skill-start state, and GT/predicted
    progress. ``skill_cond_mode=token`` uses z as an explicit token; ``broadcast`` uses
    z to condition every pool block input without appending or repeatedly accumulating
    a residual skill vector.
    """

    def __init__(
        self,
        *,
        fsq_levels: list[int],
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        skill_cond_mode: str,
        max_state_dim: int,
        max_action_dim: int,
        chunk_size: int,
        use_start_state: bool = True,
    ):
        super().__init__()
        if skill_cond_mode not in {"token", "broadcast"}:
            raise ValueError(f"skill_cond_mode must be token|broadcast, got {skill_cond_mode!r}.")
        self.skill_cond_mode = skill_cond_mode
        self.max_state_dim = int(max_state_dim)
        self.max_action_dim = int(max_action_dim)
        self.chunk_size = int(chunk_size)
        # Probe: drop the skill-start-state token entirely — the chunk is then a
        # pure (z, progress) motion-program lookup. The forward still ACCEPTS
        # start_state for API compatibility and ignores it.
        self.use_start_state = bool(use_start_state)
        self.state_proj = nn.Linear(max_state_dim, hidden_dim) if self.use_start_state else None
        self.skill_proj = nn.Linear(len(fsq_levels), hidden_dim)
        self.progress_proj = nn.Linear(1, hidden_dim)
        n_tokens = (2 if skill_cond_mode == "token" else 1) + int(self.use_start_state)
        self.pool = TokenTransformerPool(
            hidden_dim=hidden_dim,
            n_tokens=n_tokens,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.action_head = nn.Linear(hidden_dim, chunk_size * max_action_dim)

    @property
    def working_dtype(self) -> torch.dtype:
        return self.action_head.weight.dtype

    def forward(
        self,
        start_state: Tensor,
        z_norm: Tensor,
        progress: Tensor,
    ) -> Tensor:
        bsize = z_norm.shape[0]
        skill_tok = self.skill_proj(z_norm.to(self.working_dtype))
        progress_tok = self.progress_proj(progress.reshape(bsize, 1).to(self.working_dtype))
        parts: list[Tensor] = [skill_tok] if self.skill_cond_mode == "token" else []
        if self.use_start_state:
            parts.append(self.state_proj(start_state.to(self.working_dtype)))
        parts.append(progress_tok)
        tokens = torch.stack(parts, dim=1)
        cond = None if self.skill_cond_mode == "token" else skill_tok
        hidden = self.pool(tokens, broadcast_cond=cond)
        action = self.action_head(hidden).view(bsize, self.chunk_size, self.max_action_dim)
        return torch.tanh(action).float()


def initialize_terminator_vision_from_pi05(
    terminator: "FSQQueryTerminator", pretrained: str | Path
) -> int:
    """Warm-start SigLIP from the same PI05 vision tower used by Stage-1.

    DINO initializes from ``dino_model_path`` and ResNet18 from ImageNet, so
    neither has a PI05 mapping. The projection remains task-specific and is
    learned by FSQ.
    """
    if terminator.vision_backbone != "siglip":
        return 0
    from safetensors import safe_open

    path = Path(pretrained)
    if path.is_dir():
        path = path / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"PI05 model.safetensors not found: {path}")
    prefix = "paligemma_with_expert.paligemma.model.vision_tower."
    mapped: dict[str, Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            plain = key[len("model.") :] if key.startswith("model.") else key
            if plain.startswith(prefix):
                mapped[plain[len(prefix) :]] = handle.get_tensor(key)
    missing, unexpected = terminator.siglip.load_state_dict(mapped, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"PI05 SigLIP initialization mismatch: missing={missing}, unexpected={unexpected}"
        )
    return len(mapped)


# -----------------------------------------------------------------------------
# One-way query terminator
# -----------------------------------------------------------------------------


class ConditionalRMSNorm(nn.Module):
    """RMSNorm whose scale/shift are produced from state+skill conditioning."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.modulation = nn.Linear(cond_dim, dim * 2)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        norm = x * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 1e-6)
        scale, shift = self.modulation(cond).chunk(2, dim=-1)
        while scale.ndim < x.ndim:
            scale, shift = scale.unsqueeze(1), shift.unsqueeze(1)
        return (norm * self.weight.float() * (1.0 + scale.float()) + shift.float()).to(x.dtype)


class DtypeAlignedRMSNorm(nn.RMSNorm):
    """RMSNorm that keeps its autocast input and weight on the same fused-kernel dtype.

    Parameters remain FP32 in the module/state dict and gradients still flow through
    the temporary cast.  This only avoids PyTorch's slower mixed-dtype RMSNorm path
    when the surrounding FSQ forward runs under BF16 autocast.
    """

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight
        if weight is not None and weight.dtype != x.dtype:
            weight = weight.to(dtype=x.dtype)
        return F.rms_norm(x, self.normalized_shape, weight, self.eps)


class QueryTerminatorLayer(nn.Module):
    """Images self-attend; each output query reads images+self, never the other query."""

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.image_norm1 = DtypeAlignedRMSNorm(hidden_dim)
        self.query_norm1 = ConditionalRMSNorm(hidden_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.image_norm2 = DtypeAlignedRMSNorm(hidden_dim)
        self.query_norm2 = ConditionalRMSNorm(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def _norm(self, x: Tensor, n_image: int, cond: Tensor, second: bool) -> Tensor:
        image_norm = self.image_norm2 if second else self.image_norm1
        query_norm = self.query_norm2 if second else self.query_norm1
        return torch.cat([image_norm(x[:, :n_image]), query_norm(x[:, n_image:], cond)], dim=1)

    @staticmethod
    def _add_query_broadcast(x: Tensor, n_image: int, cond: Tensor | None) -> Tensor:
        if cond is None:
            return x
        query_count = x.shape[1] - n_image
        if cond.ndim == 2:
            cond = cond.unsqueeze(1)
        if cond.ndim != 3 or cond.shape[0] != x.shape[0] or cond.shape[-1] != x.shape[-1]:
            raise ValueError(
                "query broadcast condition must have shape (B,H) or (B,1|Q,H), got "
                f"{tuple(cond.shape)} for hidden states {tuple(x.shape)}."
            )
        if cond.shape[1] == 1:
            cond = cond.expand(-1, query_count, -1)
        elif cond.shape[1] != query_count:
            raise ValueError(
                "query broadcast token dimension must be 1 or match query count, got "
                f"{cond.shape[1]} and {query_count}."
            )
        return torch.cat(
            [x[:, :n_image], x[:, n_image:] + cond.to(device=x.device, dtype=x.dtype)],
            dim=1,
        )

    def forward(
        self,
        x: Tensor,
        n_image: int,
        cond: Tensor,
        disallow: Tensor,
        skill_broadcast: Tensor | None = None,
    ) -> Tensor:
        normed = self._norm(x, n_image, cond, second=False)
        normed = self._add_query_broadcast(normed, n_image, skill_broadcast)
        attn, _ = self.attention(normed, normed, normed, attn_mask=disallow, need_weights=False)
        x = x + self.dropout(attn)
        x = x + self.dropout(self.ffn(self._norm(x, n_image, cond, second=True)))
        return x


class MultimodalFusionLayer(nn.Module):
    """Pre-norm full self-attention over skill, state, and image tokens."""

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = DtypeAlignedRMSNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = DtypeAlignedRMSNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        normed = self.norm1(x)
        attended, _ = self.attention(normed, normed, normed, need_weights=False)
        x = x + self.dropout(attended)
        return x + self.dropout(self.ffn(self.norm2(x)))


def resolve_image_model_path(name: str) -> str:
    path = Path(name)
    if not path.is_absolute() or path.exists():
        return name
    local = _REPO_ROOT / "models" / path.name
    return str(local) if local.exists() else name


def _build_siglip_vision_tower(image_size: int) -> nn.Module:
    """Build the exact SigLIP tower used by Stage-1's condition stream."""
    from transformers import SiglipVisionModel
    from transformers.models.auto import CONFIG_MAPPING

    vlm_cfg = CONFIG_MAPPING["paligemma"]()
    vision_cfg = vlm_cfg.vision_config
    vision_cfg.image_size = int(image_size)
    vision_cfg.intermediate_size = 4304
    vision_cfg.projection_dim = 2048
    vision_cfg.projector_hidden_act = "gelu_fast"
    return SiglipVisionModel(vision_cfg)


def _load_dino_model(model_path: str) -> nn.Module:
    """Keep Transformers/DINO completely out of state-only FSQ startup."""
    from transformers import AutoModel

    return AutoModel.from_pretrained(model_path)


def _build_resnet18_vision_tower() -> nn.Module:
    """Build an ImageNet-pretrained ResNet18 without global pooling or FC."""
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    return nn.Sequential(*list(model.children())[:-2])


class FSQQueryTerminator(nn.Module):
    """Live third+wrist images + context/skill conditioning -> termination.

    The visual frontend and token contract intentionally match Stage-1's
    condition stream:
    one shared DINO, SigLIP, or ResNet18 tower is applied to both cameras, no
    spatial pooling is performed, and both token sequences share one
    ``image_proj``. ResNet18's final 7x7 map becomes 49 spatial tokens at the
    default 224px input size.

    ``arch='small'`` uses a lightweight query transformer. ``arch='fusion'``
    instead represents the quantized skill, current state, and both cameras as
    ordinary tokens in one bidirectional Transformer. It has no learned output
    queries, AdaRMS, or skill broadcast: the updated skill token itself feeds
    the progress/termination heads.

    ``skill_cond_mode='token'`` keeps the historical terminator behavior:
    state+skill modulate the query tokens through AdaRMS. ``broadcast`` keeps
    state AdaRMS but adds the projected skill vector to the query hidden states
    used by every attention layer.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        fsq_levels: list[int],
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        arch: str,
        vision_backbone: str,
        freeze_vision_encoder: bool,
        dino_model_path: str,
        dino_image_size: int,
        siglip_image_size: int,
        resnet_image_size: int = 224,
        skill_cond_mode: str,
        state_min: np.ndarray,
        state_max: np.ndarray,
        context_mode: str = "proprio",
        context_gripper_weight: float = 1.0,
        termination_only: bool = False,
    ):
        super().__init__()
        if arch not in {"small", "fusion"}:
            raise ValueError(f"terminator_arch must be small|fusion, got {arch!r}.")
        if vision_backbone not in {"dino", "siglip", "resnet"}:
            raise ValueError(
                "vision_backbone must be dino|siglip|resnet, "
                f"got {vision_backbone!r}."
            )
        if hidden_dim % n_heads:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by n_heads={n_heads}.")
        if skill_cond_mode not in {"token", "broadcast"}:
            raise ValueError(f"skill_cond_mode must be token|broadcast, got {skill_cond_mode!r}.")
        self.state_dim = int(state_dim)
        self.context_mode = str(context_mode)
        self.context_gripper_weight = float(context_gripper_weight)
        if self.context_mode not in {"prev_action", "proprio"}:
            raise ValueError(
                "context_mode must be prev_action|proprio, "
                f"got {self.context_mode!r}."
            )
        self.fsq_levels = [int(x) for x in fsq_levels]
        self.arch = arch
        self.skill_cond_mode = skill_cond_mode
        # The progress query never reaches the termination query (queries cannot
        # attend to each other and images cannot attend to queries), so keeping
        # its token/head with termination_only is inert and shape-compatible.
        self.termination_only = bool(termination_only)
        self.vision_backbone = vision_backbone
        self.freeze_vision_encoder = bool(freeze_vision_encoder)
        self.dino_model_path = resolve_image_model_path(dino_model_path)
        self.dino_image_size = int(dino_image_size)
        self.siglip_image_size = int(siglip_image_size)
        self.resnet_image_size = int(resnet_image_size)
        if self.resnet_image_size < 32:
            raise ValueError(
                f"resnet_image_size must be at least 32, got {self.resnet_image_size}."
            )
        self.dino = None
        self.siglip = None
        self.resnet = None
        self.n_register = 0
        if vision_backbone == "dino":
            self.dino = _load_dino_model(self.dino_model_path)
            visual_dim = int(self.dino.config.hidden_size)
            self.n_register = int(getattr(self.dino.config, "num_register_tokens", 0))
            self.vision_image_size = self.dino_image_size
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        elif vision_backbone == "siglip":
            self.siglip = _build_siglip_vision_tower(self.siglip_image_size)
            visual_dim = int(self.siglip.config.hidden_size)
            self.vision_image_size = self.siglip_image_size
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        else:
            self.resnet = _build_resnet18_vision_tower()
            visual_dim = 512
            self.vision_image_size = self.resnet_image_size
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        if self.freeze_vision_encoder:
            for parameter in self.vision_encoder.parameters():
                parameter.requires_grad_(False)
        self.register_buffer("_img_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_img_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

        width = int(hidden_dim)
        self.hidden_dim = width
        self.image_proj = nn.Linear(visual_dim, width)
        latent_dim = len(fsq_levels)
        if arch == "fusion":
            self.state_proj = nn.Sequential(
                nn.Linear(state_dim, width),
                nn.GELU(),
                nn.Linear(width, width),
            )
            self.skill_proj = nn.Sequential(
                nn.Linear(latent_dim, width),
                nn.GELU(),
                nn.Linear(width, width),
            )
            self.skill_type_embedding = nn.Parameter(torch.zeros(1, 1, width))
            self.state_type_embedding = nn.Parameter(torch.zeros(1, 1, width))
            self.third_type_embedding = nn.Parameter(torch.zeros(1, 1, width))
            self.wrist_type_embedding = nn.Parameter(torch.zeros(1, 1, width))
            self.layers = nn.ModuleList(
                [MultimodalFusionLayer(width, n_heads, dropout) for _ in range(n_layers)]
            )
            self.fusion_out_norm = DtypeAlignedRMSNorm(width)
            head_dim = width + latent_dim
            self.progress_head = nn.Linear(head_dim, 1)
            self.termination_head = nn.Linear(head_dim, 1)
            for embedding in (
                self.skill_type_embedding,
                self.state_type_embedding,
                self.third_type_embedding,
                self.wrist_type_embedding,
            ):
                nn.init.trunc_normal_(embedding, std=0.02)
        else:
            self.progress_query = nn.Parameter(torch.zeros(1, 1, width))
            self.termination_query = nn.Parameter(torch.zeros(1, 1, width))
            self.state_proj = nn.Linear(state_dim, width)
            self.skill_proj = nn.Linear(latent_dim, width)
        if arch == "small":
            self.layers = nn.ModuleList(
                [QueryTerminatorLayer(width, n_heads, dropout) for _ in range(n_layers)]
            )
            self.query_out_norm = ConditionalRMSNorm(width, width)
        if arch != "fusion":
            self.progress_head = nn.Linear(width, 1)
            self.termination_head = nn.Linear(width, 1)
        self.gradient_checkpointing = False
        self.register_buffer("state_min", torch.as_tensor(state_min, dtype=torch.float32))
        self.register_buffer("state_max", torch.as_tensor(state_max, dtype=torch.float32))
        if arch != "fusion":
            nn.init.trunc_normal_(self.progress_query, std=0.02)
            nn.init.trunc_normal_(self.termination_query, std=0.02)

    @property
    def vision_encoder(self) -> nn.Module:
        if self.dino is not None:
            return self.dino
        if self.siglip is not None:
            return self.siglip
        if self.resnet is not None:
            return self.resnet
        raise RuntimeError("FSQ terminator has no vision encoder.")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.vision_encoder.eval()
        return self

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True
        if not self.freeze_vision_encoder and hasattr(self.vision_encoder, "gradient_checkpointing_enable"):
            self.vision_encoder.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False
        if not self.freeze_vision_encoder and hasattr(self.vision_encoder, "gradient_checkpointing_disable"):
            self.vision_encoder.gradient_checkpointing_disable()

    def _normalize_state(self, state: Tensor) -> Tensor:
        if self.context_mode == "prev_action":
            # The dataset/inference adapter applies the shared clipped action
            # transform and emits an exact all-zero BOS token at t=0.
            return state[..., : self.state_dim]
        lo = self.state_min.to(state.device, state.dtype)
        hi = self.state_max.to(state.device, state.dtype)
        return 2.0 * (state[..., : self.state_dim] - lo) / (hi - lo + 1e-8) - 1.0

    def normalize_previous_action(self, action: Tensor) -> Tensor:
        """Normalize one or more raw emitted actions for terminator inference."""
        if self.context_mode != "prev_action":
            raise RuntimeError(
                "normalize_previous_action is available only for prev_action terminators."
            )
        lo = self.state_min.to(action.device, action.dtype)
        hi = self.state_max.to(action.device, action.dtype)
        normalized = (
            2.0 * (action[..., : self.state_dim] - lo) / (hi - lo + 1e-8) - 1.0
        ).clamp(-1.0, 1.0)
        if self.context_gripper_weight != 1.0:
            normalized = normalized.clone()
            normalized[..., -1] *= math.sqrt(self.context_gripper_weight)
        return normalized

    def _preprocess_image(self, image: Tensor | None) -> Tensor:
        """Convert the FSQ image contract to the shared vision tower input.

        Video decoders used by both training and evaluation return floating-point
        CHW images in [0, 1].  Integer images are also accepted and scaled once.
        Keeping that explicit contract removes the per-camera ``amin().item()`` /
        ``amax().item()`` CUDA synchronizations from every training step.
        """
        if image is None:
            raise ValueError("FSQ terminator always requires both third-person and wrist images.")
        if image.ndim != 4:
            raise ValueError(f"Terminator image must be (B,C,H,W) or (B,H,W,C), got {tuple(image.shape)}")
        if image.shape[-1] in (1, 3):
            image = image.permute(0, 3, 1, 2)
        if image.shape[1] == 1:
            image = image.expand(-1, 3, -1, -1)
        if image.is_floating_point():
            x = image.float()
        else:
            if image.dtype != torch.uint8:
                raise TypeError(
                    "Integer terminator images must be uint8 in [0, 255], "
                    f"got {image.dtype}."
                )
            x = image.float().div_(255.0)
        x = F.interpolate(
            x.clamp(0.0, 1.0),
            size=(self.vision_image_size, self.vision_image_size),
            mode="bilinear",
            align_corners=False,
        )
        return (x - self._img_mean.float()) / self._img_std.float()

    def _encode_image_batch(self, x: Tensor) -> Tensor:
        if self.vision_backbone == "dino":
            x = x.to(dtype=next(self.dino.parameters()).dtype)
            output = self.dino(x).last_hidden_state
            cls = output[:, :1]
            patches = output[:, 1 + self.n_register :]
            return torch.cat([cls, patches], dim=1)
        if self.vision_backbone == "siglip":
            x = x.to(dtype=next(self.siglip.parameters()).dtype)
            return self.siglip(pixel_values=x).last_hidden_state
        x = x.to(dtype=next(self.resnet.parameters()).dtype)
        feature_map = self.resnet(x)
        return feature_map.flatten(2).transpose(1, 2)

    def _image_features(self, image: Tensor | None) -> Tensor:
        """Encode one camera batch; retained for component/evaluation callers."""
        return self._encode_image_batch(self._preprocess_image(image))

    def _prepare_image_tokens(self, third: Tensor | None, wrist: Tensor | None) -> Tensor:
        """Encode top+wrist in one shared-backbone call, then restore camera order."""
        third_input = self._preprocess_image(third)
        wrist_input = self._preprocess_image(wrist)
        if third_input.shape[0] != wrist_input.shape[0]:
            raise ValueError(
                "Third-person and wrist image batches must match, got "
                f"{third_input.shape[0]} and {wrist_input.shape[0]}."
            )
        batch_size = third_input.shape[0]
        features = self._encode_image_batch(torch.cat([third_input, wrist_input], dim=0))
        projected = self.image_proj(features.to(self.image_proj.weight.dtype))
        third_tokens, wrist_tokens = projected.split(batch_size, dim=0)
        # Preserve the historical [all top tokens, all wrist tokens] sequence.
        return torch.cat([third_tokens, wrist_tokens], dim=1)

    @staticmethod
    def _module_dtype(module: nn.Module) -> torch.dtype:
        return next(module.parameters()).dtype

    def _project_skill(self, z_norm: Tensor) -> Tensor:
        return self.skill_proj(z_norm.to(self._module_dtype(self.skill_proj)))

    def _project_state(self, raw_state: Tensor) -> Tensor:
        normalized = self._normalize_state(raw_state)
        return self.state_proj(normalized.to(self._module_dtype(self.state_proj)))

    def _fusion_image_tokens(
        self,
        image_tokens: Tensor,
        *,
        camera_layout: str,
    ) -> Tensor:
        if camera_layout == "both":
            if image_tokens.shape[1] % 2:
                raise ValueError(
                    "Fusion terminator expected equal top/wrist token counts, got "
                    f"{image_tokens.shape[1]} total image tokens."
                )
            per_camera = image_tokens.shape[1] // 2
            third, wrist = image_tokens.split(per_camera, dim=1)
            return torch.cat(
                [
                    third + self.third_type_embedding.to(third.dtype),
                    wrist + self.wrist_type_embedding.to(wrist.dtype),
                ],
                dim=1,
            )
        if camera_layout == "wrist":
            return image_tokens + self.wrist_type_embedding.to(image_tokens.dtype)
        raise ValueError(f"camera_layout must be both|wrist, got {camera_layout!r}.")

    def _forward_fusion(
        self,
        z_norm: Tensor,
        image_tokens: Tensor,
        *,
        raw_state: Tensor | None,
        camera_layout: str = "both",
    ) -> tuple[Tensor, Tensor]:
        """Fuse all modalities symmetrically and read out the updated skill token."""
        if self.arch != "fusion":
            raise RuntimeError("_forward_fusion is available only for arch='fusion'.")
        skill = self._project_skill(z_norm).unsqueeze(1)
        skill = skill + self.skill_type_embedding.to(skill.dtype)
        parts = [skill]
        if raw_state is not None:
            state = self._project_state(raw_state).unsqueeze(1)
            parts.append(state + self.state_type_embedding.to(state.dtype))
        parts.append(
            self._fusion_image_tokens(image_tokens, camera_layout=camera_layout)
        )
        hidden = torch.cat(parts, dim=1)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                hidden = layer(hidden)
        fused_skill = self.fusion_out_norm(hidden[:, 0])
        # A direct z skip mirrors the one-shot reconstructor's short gradient
        # path while the transformed skill token carries multimodal context.
        head_input = torch.cat(
            [fused_skill, z_norm.to(fused_skill.dtype)],
            dim=-1,
        )
        termination = self.termination_head(head_input).squeeze(-1)
        progress = (
            torch.zeros_like(termination)
            if self.termination_only
            else torch.sigmoid(self.progress_head(head_input)).squeeze(-1)
        )
        return progress, termination

    @staticmethod
    def _allow_mask(
        n_image: int,
        device: torch.device,
        *,
        image_allow: Tensor | None = None,
        query_image_allow: Tensor | None = None,
    ) -> Tensor:
        total = n_image + 2
        allow = torch.zeros(total, total, dtype=torch.bool, device=device)
        if image_allow is None:
            allow[:n_image, :n_image] = True
        else:
            if image_allow.shape != (n_image, n_image):
                raise ValueError(
                    "image_allow must have shape "
                    f"({n_image}, {n_image}), got {tuple(image_allow.shape)}."
                )
            allow[:n_image, :n_image] = image_allow.to(
                device=device, dtype=torch.bool
            )
        if query_image_allow is None:
            allow[n_image:, :n_image] = True
        else:
            if query_image_allow.shape != (2, n_image):
                raise ValueError(
                    "query_image_allow must have shape "
                    f"(2, {n_image}), got {tuple(query_image_allow.shape)}."
                )
            allow[n_image:, :n_image] = query_image_allow.to(
                device=device, dtype=torch.bool
            )
        allow[n_image, n_image] = True
        allow[n_image + 1, n_image + 1] = True
        return allow

    def _forward_from_conditions(
        self,
        image_tokens: Tensor,
        norm_cond: Tensor,
        skill_broadcast: Tensor | None,
        *,
        image_allow: Tensor | None = None,
        query_image_allow: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run the shared query decoder from an already-built condition.

        Keeping this path separate lets the image-only terminator reuse the
        exact visual/query architecture while omitting the state projection.
        """
        bsize, n_image = image_tokens.shape[:2]
        queries = torch.cat(
            [
                self.progress_query.expand(bsize, -1, -1),
                self.termination_query.expand(bsize, -1, -1),
            ],
            dim=1,
        ).to(image_tokens.dtype)
        x = torch.cat([image_tokens, queries], dim=1)
        allow = self._allow_mask(
            n_image,
            x.device,
            image_allow=image_allow,
            query_image_allow=query_image_allow,
        )
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    lambda hidden, layer=layer: layer(
                        hidden, n_image, norm_cond, ~allow, skill_broadcast
                    ),
                    x,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                x = layer(x, n_image, norm_cond, ~allow, skill_broadcast)
        query_out = self.query_out_norm(x[:, n_image:], norm_cond)
        termination = self.termination_head(query_out[:, 1]).squeeze(-1)
        if self.termination_only:
            progress = torch.zeros_like(termination)
        else:
            progress = torch.sigmoid(self.progress_head(query_out[:, 0])).squeeze(-1)
        return progress, termination

    def forward(
        self,
        z_norm: Tensor,
        raw_state: Tensor,
        third: Tensor | None,
        wrist: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        image_tokens = self._prepare_image_tokens(third, wrist)
        return self._forward_from_image_tokens(z_norm, raw_state, image_tokens)

    def _forward_from_image_tokens(
        self,
        z_norm: Tensor,
        raw_state: Tensor,
        image_tokens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.arch == "fusion":
            return self._forward_fusion(
                z_norm,
                image_tokens,
                raw_state=raw_state,
                camera_layout="both",
            )
        state_cond = self._project_state(raw_state)
        skill_cond = self._project_skill(z_norm)
        if self.skill_cond_mode == "broadcast":
            norm_cond = state_cond
            skill_broadcast = skill_cond
        else:
            norm_cond = state_cond + skill_cond
            skill_broadcast = None
        return self._forward_from_conditions(image_tokens, norm_cond, skill_broadcast)

    def forward_with_skill_shuffle(
        self,
        z_norm: Tensor,
        shuffled_z_norm: Tensor,
        raw_state: Tensor,
        third: Tensor | None,
        wrist: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Evaluate true and shuffled skills while encoding both cameras once."""
        image_tokens = self._prepare_image_tokens(third, wrist)
        progress, logits = self._forward_from_image_tokens(
            z_norm, raw_state, image_tokens
        )
        shuffled_progress, shuffled_logits = self._forward_from_image_tokens(
            shuffled_z_norm, raw_state, image_tokens
        )
        return progress, logits, shuffled_progress, shuffled_logits

    @torch.no_grad()
    def predict_termination(
        self,
        z_norm: Tensor,
        raw_state: Tensor,
        third: Tensor | None,
        wrist: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        progress, logits = self(z_norm, raw_state, third, wrist)
        return progress, torch.sigmoid(logits)


class FSQImageOnlyQueryTerminator(FSQQueryTerminator):
    """Current top+wrist images and skill code -> progress and termination.

    This deliberately has no state projection, state-normalization buffers, or
    state input.  The visual/query architecture otherwise matches
    :class:`FSQQueryTerminator` exactly.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.state_proj
        del self.state_min
        del self.state_max

    def forward(
        self,
        z_norm: Tensor,
        third: Tensor | None,
        wrist: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        image_tokens = self._prepare_image_tokens(third, wrist)
        return self._forward_without_state(z_norm, image_tokens)

    def _forward_without_state(
        self,
        z_norm: Tensor,
        image_tokens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.arch == "fusion":
            return self._forward_fusion(
                z_norm,
                image_tokens,
                raw_state=None,
                camera_layout="both",
            )
        skill_cond = self._project_skill(z_norm)
        if self.skill_cond_mode == "broadcast":
            norm_cond = torch.zeros_like(skill_cond)
            skill_broadcast = skill_cond
        else:
            norm_cond = skill_cond
            skill_broadcast = None
        return self._forward_from_conditions(image_tokens, norm_cond, skill_broadcast)

    def forward_with_skill_shuffle(
        self,
        z_norm: Tensor,
        shuffled_z_norm: Tensor,
        third: Tensor | None,
        wrist: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        image_tokens = self._prepare_image_tokens(third, wrist)
        progress, logits = self._forward_without_state(z_norm, image_tokens)
        shuffled_progress, shuffled_logits = self._forward_without_state(
            shuffled_z_norm, image_tokens
        )
        return progress, logits, shuffled_progress, shuffled_logits

    @torch.no_grad()
    def predict_termination(
        self,
        z_norm: Tensor,
        third: Tensor | None,
        wrist: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        progress, logits = self(z_norm, third, wrist)
        return progress, torch.sigmoid(logits)


class FSQWristOnlyQueryTerminator(FSQImageOnlyQueryTerminator):
    """Current wrist image and skill code -> progress and termination.

    Like :class:`FSQImageOnlyQueryTerminator`, this model has no state input or
    state projection. It additionally omits the third-person camera entirely,
    so the query decoder attends only to wrist-camera tokens.
    """

    def _prepare_wrist_tokens(self, wrist: Tensor | None) -> Tensor:
        features = self._image_features(wrist)
        return self.image_proj(features.to(self.image_proj.weight.dtype))

    def forward(
        self,
        z_norm: Tensor,
        wrist: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        image_tokens = self._prepare_wrist_tokens(wrist)
        if self.arch == "fusion":
            return self._forward_fusion(
                z_norm,
                image_tokens,
                raw_state=None,
                camera_layout="wrist",
            )
        return self._forward_without_state(z_norm, image_tokens)

    def forward_with_skill_shuffle(
        self,
        z_norm: Tensor,
        shuffled_z_norm: Tensor,
        wrist: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        image_tokens = self._prepare_wrist_tokens(wrist)
        if self.arch == "fusion":
            forward = lambda skill: self._forward_fusion(  # noqa: E731
                skill,
                image_tokens,
                raw_state=None,
                camera_layout="wrist",
            )
        else:
            forward = lambda skill: self._forward_without_state(  # noqa: E731
                skill, image_tokens
            )
        progress, logits = forward(z_norm)
        shuffled_progress, shuffled_logits = forward(shuffled_z_norm)
        return progress, logits, shuffled_progress, shuffled_logits

    @torch.no_grad()
    def predict_termination(
        self,
        z_norm: Tensor,
        wrist: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        progress, logits = self(z_norm, wrist)
        return progress, torch.sigmoid(logits)


class FSQStateRNNTerminator(StateSkillRNNTerminator):
    """Full-skill causal terminator using only raw proprioception and FSQ skill.

    The recurrent architecture deliberately matches the standalone state-RNN
    probe (vanilla tanh RNN, 64-wide input/hidden, one layer).  Raw dataset
    states are quantile-normalized inside the module, so training and online
    one-step inference share the same input contract.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        fsq_levels: list[int],
        state_q01: np.ndarray,
        state_q99: np.ndarray,
        context_mode: str = "proprio",
        termination_only: bool,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            skill_dim=len(fsq_levels),
            input_dim=64,
            hidden_dim=64,
            num_layers=1,
            dropout=0.0,
            termination_only=termination_only,
        )
        self.fsq_levels = [int(level) for level in fsq_levels]
        self.context_mode = str(context_mode)
        self.register_buffer(
            "state_q01",
            torch.as_tensor(state_q01, dtype=torch.float32)[:state_dim],
        )
        self.register_buffer(
            "state_q99",
            torch.as_tensor(state_q99, dtype=torch.float32)[:state_dim],
        )

    def _sequence_inputs(self, z_q: Tensor, states: Tensor) -> Tensor:
        if self.context_mode == "prev_action":
            return StateSkillRNNTerminator._sequence_inputs(
                self, z_q, states[..., : self.state_dim]
            )
        lo = self.state_q01.to(device=states.device, dtype=states.dtype)
        hi = self.state_q99.to(device=states.device, dtype=states.dtype)
        normalized = 2.0 * (states[..., : self.state_dim] - lo) / (hi - lo + 1e-8) - 1.0
        return super()._sequence_inputs(z_q, normalized)

    @torch.no_grad()
    def predict_termination_step(
        self,
        z_norm: Tensor,
        raw_state: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(progress, probability, next_hidden)`` for one online step."""
        progress, logits, next_hidden = self.step_outputs(z_norm, raw_state, hidden)
        return progress, torch.sigmoid(logits), next_hidden


class FSQStateMLPTerminator(StateSkillMLPTerminator):
    """Current-state FSQ terminator used by ``input_space=state, arch=default``."""

    def __init__(
        self,
        *,
        state_dim: int,
        fsq_levels: list[int],
        state_q01: np.ndarray,
        state_q99: np.ndarray,
        context_mode: str = "proprio",
        termination_only: bool,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            skill_dim=len(fsq_levels),
            hidden_dim=64,
            num_layers=2,
            termination_only=termination_only,
        )
        self.fsq_levels = [int(level) for level in fsq_levels]
        self.context_mode = str(context_mode)
        self.register_buffer(
            "state_q01",
            torch.as_tensor(state_q01, dtype=torch.float32)[:state_dim],
        )
        self.register_buffer(
            "state_q99",
            torch.as_tensor(state_q99, dtype=torch.float32)[:state_dim],
        )

    def _normalize_state(self, raw_state: Tensor) -> Tensor:
        if self.context_mode == "prev_action":
            return raw_state[..., : self.state_dim]
        lo = self.state_q01.to(device=raw_state.device, dtype=raw_state.dtype)
        hi = self.state_q99.to(device=raw_state.device, dtype=raw_state.dtype)
        return 2.0 * (raw_state[..., : self.state_dim] - lo) / (hi - lo + 1e-8) - 1.0

    def forward_outputs(self, z_q: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        return super().forward_outputs(z_q, self._normalize_state(state))

    @torch.no_grad()
    def predict_termination(
        self,
        z_norm: Tensor,
        raw_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        progress, logits = self.forward_outputs(z_norm, raw_state)
        return progress, torch.sigmoid(logits)


# -----------------------------------------------------------------------------
# Full model/config and component checkpoint loaders
# -----------------------------------------------------------------------------


def different_code_shuffle_sources(indices: Tensor) -> tuple[Tensor, Tensor]:
    """Choose a deterministic different-code source for every possible row.

    Reusing a same-code row would make a low shuffle delta ambiguous: it could
    mean either that the terminator ignored skill or simply that the shuffled
    discrete code did not change.  This cyclic search picks the first row with
    a different code and returns a validity mask for batches containing only
    one active code.
    """
    flat = indices.reshape(-1)
    size = int(flat.numel())
    sources = torch.arange(size, device=flat.device)
    valid = torch.zeros(size, dtype=torch.bool, device=flat.device)
    for offset in range(1, size):
        candidates = (torch.arange(size, device=flat.device) + offset) % size
        select = (~valid) & (flat[candidates] != flat)
        sources = torch.where(select, candidates, sources)
        valid |= select
    return sources, valid


@dataclass
class SplineFSQAEConfig:
    format_version: int = FORMAT_VERSION
    action_dim: int = 7
    enc_dim: int = 8
    state_dim: int = 8
    n_control: int = 30
    spline_degree: int = 3
    encoder_input_mode: str = "zero_grounded"
    encoder_grounding_convention: str = ENCODER_GROUNDING_CONVENTION
    encoder_length_token: bool = True
    """False: the spline encoder consumes NO length token — duration reaches z
    only through motion shape (probe ported from FSQ-original)."""
    encoder_arch: str = "spline"
    """spline: fixed control-point tokens. action_seq: variable-length ACTION
    sequence transformer (no spline codec / grounding / length-token choices)."""
    autoencoder_mode: str = "legacy"
    """Resolved whole-system preset: raw, zero, action, norm_action, or legacy."""
    action_gripper_weight: float = 1.0
    """Gripper-axis MSE weight for every preset; representation uses its square root."""
    quantizer: str = "fsq"
    """fsq: finite scalar grid. bsq: binary spherical codebook."""
    bsq_code_dim: int = 5
    """BSQ bit count; codebook size is 2**bsq_code_dim. Ignored for FSQ."""
    fsq_entropy: bool = False
    """Apply BSQ-style entropy terms to the FSQ grid: sample entropy
    minimization (confidence — pushes samples off rounding boundaries) and
    batch entropy maximization (code-usage diversity)."""
    entropy_conf_weight: float = 0.1
    entropy_conf_ceiling: float = 0.0
    """Normalized per-sample entropy ceiling in [0, 1]. Confidence pressure is
    a hinge: samples at or below this ceiling receive exactly zero gradient.
    Zero reproduces the historical unconstrained entropy-minimization loss."""
    entropy_div_weight: float = 0.1
    entropy_inv_temperature: float = 10.0
    entropy_joint: bool = True
    """Exact dataset entropy over all prod(fsq_levels) codes (project standard);
    the factorized bound exists only as a fallback for huge codebooks."""
    init_calibration: bool = False
    """One-shot data calibration of the freshly initialized encoder z_head.

    The clean training trajectories are encoded once before optimization.  Each
    output row of z_head is then reparameterized so its initial dataset mean is
    zero and its standard deviation is ``init_calibration_gain``.  No runtime
    normalization layer or persistent distribution constraint is introduced.
    """
    init_calibration_gain: float = 1.0
    init_calibration_samples: int = 0
    """Number of clean training trajectories used for calibration; 0 uses all."""
    pair_loss: str = "none"
    """Pair objective: none, overlap, js, or linear contrastive overlap."""
    pair_weight: float = 0.1
    pair_inv_temperature: float = 5.0
    route_loss: bool = False
    """Add a schedule-free joint decoder-aware code-routing objective.

    Every finite code is evaluated by the enabled reconstructor and termination
    head under the same decoder context. Candidate distortions are detached, so
    this objective updates only the encoder assignment; the regular hard-code
    losses remain the decoders' sole training signals. The soft assignment
    reuses ``pair_inv_temperature`` and has no separate weight.
    """
    pair_warmup: bool = False
    """Enable reconstruction-only warm-up and the following pair-weight ramp."""
    pair_warmup_epochs: int = 0
    """Number of initial reconstruction-only epochs before pair loss starts."""
    pair_ramp_epochs: int = 0
    """Epochs used to linearly ramp pair weight from zero to pair_weight."""
    boundary_aug_pmax: int = 0
    """Legacy fallback used for every directional boundary window."""
    boundary_aug_early_start_pmax: int = -1
    """Maximum frames prepended so the augmented skill starts earlier."""
    boundary_aug_late_start_pmax: int = -1
    """Maximum frames trimmed so the augmented skill starts later."""
    boundary_aug_early_end_pmax: int = -1
    """Maximum frames trimmed so the augmented skill ends earlier."""
    boundary_aug_late_end_pmax: int = -1
    """Maximum frames appended so the augmented skill ends later.

    A negative directional value inherits ``boundary_aug_pmax`` for backward
    compatibility. Zero disables only that direction.
    """
    boundary_aug_distribution: str = "half_normal"
    reconstructor_start_state: bool = True
    """False: the reconstructor drops its skill-start-state token — the action
    chunk becomes a pure (z, progress) motion-program lookup with no spatial
    grounding input."""
    reconstructor_start_state_conditioning: str = "concat"
    """Internal conditioning implementation. ``concat`` preserves historical
    checkpoints; current raw/zero/action presets use ``adaln`` when start-state
    conditioning is enabled."""
    reconstructor_arch: str = "chunk"
    """chunk: per-timestep action-chunk reconstructor (v3 default; runs at the
    M sampled timesteps). oneshot: FSQ-original-style decoder that reconstructs
    the FULL control-point grid ONCE per trajectory from z alone — M then
    applies only to the terminator, and the recon loss ('action' metric slot)
    becomes the ctrl MSE. action_seq: raw full action sequence reconstructed
    once from z by a GRU. action_seq_transformer: the same raw contract with a
    prefix-invariant causal query Transformer whose every block is conditioned
    on z through adaptive LayerNorm and can independently receive start-state
    AdaLN context. Action targets use neither dataset normalization nor teacher
    forcing."""
    reconstructor_output_mode: str = "zero_grounded"
    """Spline/oneshot reconstruction target: raw_state, zero_grounded, or
    start_grounded.
    This is independent of encoder_input_mode; optimal is an encoder-only
    convention because its extra mean-XYZ value is a conditioning token."""
    hidden_dim: int = 256
    fsq_levels: list[int] = field(default_factory=lambda: [5, 5, 5])
    num_layers: int = 2
    dropout: float = 0.1
    length_min: float = 1.0
    length_max: float = 200.0

    max_state_dim: int = 32
    max_action_dim: int = 32
    chunk_size: int = 10
    skill_cond_mode: str = "token"
    pi_base: str = _DEFAULT_PI_BASE

    terminator_arch: str = "small"
    vision_backbone: str = "dino"
    freeze_vision_encoder: bool = True
    dino_model_path: str = _DEFAULT_IMAGE_MODEL
    dino_image_size: int = 224
    siglip_image_size: int = 224
    resnet_image_size: int = 224
    frame_cache_dir: str = ""
    """Completed exact uint8 RGB cache used by visual-terminator datasets."""
    image_encoder_layers: int = 2
    image_encoder_heads: int = 4

    samples_per_skill: int = 2
    end_target_sigma: float = 1.0
    terminator_input_space: str = "both"
    """Terminator observations: context, image (third+wrist), or both."""
    terminator_context: str = "prev_action"
    """Context token contract. New models use the normalized action emitted at
    t-1; ``proprio`` is retained only when loading historical checkpoints."""
    terminator_model: str = "default"
    """default uses the current-step MLP/query model; rnn uses full state history."""
    terminator_progress: bool = True
    terminator_termination: bool = True
    state_rnn_terminator: bool = False
    """Replace the visual query terminator with a full-skill causal RNN that
    consumes only raw proprioception and the FSQ skill latent. Every valid
    timestep is supervised and no camera frame or vision model is loaded."""
    terminator_termination_only: bool = False
    """Train and predict only termination: progress output is fixed to zero and its
    loss term is dropped. The progress query/head stay in the module (they are
    attention-isolated from termination), so checkpoint shapes are unchanged."""
    reconstructor_only: bool = False
    """Train only encoder+FSQ+reconstructor: no terminator module is built, no video
    frames are decoded, and the progress/termination loss terms are dropped."""
    terminator_only: bool = False
    """Train only encoder+FSQ+terminator: no reconstructor module is built and the
    action loss term is dropped. The encoder is driven purely by the terminator's
    progress/termination objectives through the FSQ straight-through estimator."""
    action_loss_weight: float = 1.0
    progress_loss_weight: float = 0.1
    end_loss_weight: float = 0.1
    end_pos_weight: float = 1.0
    end_threshold: float = 0.5

    lr: float = 3e-4
    """Shared learning rate for encoder, reconstructor, and terminator."""
    lr_schedule: str = "cosine"
    """Learning-rate schedule: cosine decays to 1% of lr; constant keeps it fixed."""
    batch_size: int = 64
    num_workers: int = 0
    val_num_workers: int = 0
    epochs: int = 300
    grad_clip: float = 1.0
    gradient_checkpointing: bool = False
    val_split: float = 0.1
    val_every: int = 1
    save_best_model: bool = True
    val_select_action_weight: float | None = None
    val_select_progress_weight: float | None = None
    val_select_end_weight: float | None = None
    log_every: int = 10
    save_path: str | None = None
    checkpoint_every: int = 0
    device: str = "cuda"

    encoder_min: np.ndarray | None = None
    encoder_max: np.ndarray | None = None
    encoder_start_min: np.ndarray | None = None
    encoder_start_max: np.ndarray | None = None
    reconstructor_min: np.ndarray | None = None
    reconstructor_max: np.ndarray | None = None
    state_min: np.ndarray | None = None
    state_max: np.ndarray | None = None
    state_q01: np.ndarray | None = None
    state_q99: np.ndarray | None = None
    action_q01: np.ndarray | None = None
    action_q99: np.ndarray | None = None


class SplineFSQAE(nn.Module):
    """Joint FSQ encoder, image-free transformer reconstructor, and query terminator."""

    def __init__(self, cfg: SplineFSQAEConfig):
        super().__init__()
        # Checkpoints written before joint terminator routing used the narrower
        # reconstruction_route_loss name. Preserve their exact on/off state.
        if "route_loss" not in vars(cfg) and "reconstruction_route_loss" in vars(cfg):
            cfg.route_loss = bool(vars(cfg)["reconstruction_route_loss"])
        if "terminator_context" not in vars(cfg):
            cfg.terminator_context = "proprio"
        # Normalize legacy programmatic configs before validating the cleaned
        # decoder/terminator option surface. New YAML configs already arrive in
        # this form through train_skills_config.py.
        if cfg.reconstructor_only:
            cfg.terminator_progress = False
            cfg.terminator_termination = False
            cfg.state_rnn_terminator = False
        elif cfg.state_rnn_terminator:
            cfg.terminator_input_space = "state"
            cfg.terminator_model = "rnn"
        if cfg.terminator_termination_only and not cfg.reconstructor_only:
            cfg.terminator_progress = False
            cfg.terminator_termination = True
        if int(cfg.format_version) != FORMAT_VERSION:
            raise ValueError(f"Only FSQ format v{FORMAT_VERSION} is supported, got {cfg.format_version}.")
        cfg.quantizer = str(cfg.quantizer).strip().lower()
        if cfg.quantizer not in {"fsq", "bsq"}:
            raise ValueError(f"quantizer must be fsq|bsq, got {cfg.quantizer!r}.")
        if cfg.bsq_code_dim < 2:
            raise ValueError(f"bsq_code_dim must be >= 2, got {cfg.bsq_code_dim}.")
        if cfg.quantizer == "bsq":
            # Decoders and terminators only need the normalized latent width.
            # Binary pseudo-levels retain that established constructor surface;
            # the encoder's actual quantizer is replaced with BSQ below.
            cfg.fsq_levels = [2] * int(cfg.bsq_code_dim)
            if cfg.fsq_entropy:
                raise ValueError(
                    "BSQ in the FSQ training path intentionally supports recon + "
                    "pair loss only; set fsq_entropy=false."
                )
        if cfg.action_dim > cfg.max_action_dim or cfg.state_dim > cfg.max_state_dim:
            raise ValueError(
                f"Real dimensions must fit PI05 padding: action {cfg.action_dim}/{cfg.max_action_dim}, "
                f"state {cfg.state_dim}/{cfg.max_state_dim}."
            )
        if cfg.samples_per_skill < 1 or cfg.chunk_size < 1:
            raise ValueError("samples_per_skill and chunk_size must both be >=1.")
        if cfg.encoder_input_mode not in {
            "zero_grounded", "start_grounded", "raw_state", "optimal"
        }:
            raise ValueError(
                "encoder_input_mode must be zero_grounded|start_grounded|raw_state|optimal, "
                f"got {cfg.encoder_input_mode!r}."
            )
        if (
            cfg.encoder_input_mode in {"zero_grounded", "start_grounded", "optimal"}
            and cfg.encoder_grounding_convention
            != encoder_grounding_convention(cfg.encoder_input_mode)
        ):
            raise ValueError(
                "Encoder grounding convention does not match encoder_input_mode: "
                f"mode={cfg.encoder_input_mode!r}, "
                f"convention={cfg.encoder_grounding_convention!r}, expected="
                f"{encoder_grounding_convention(cfg.encoder_input_mode)!r}."
            )
        if cfg.terminator_arch not in {"small", "fusion"}:
            raise ValueError(
                "terminator_arch must be small|fusion, "
                f"got {cfg.terminator_arch!r}."
            )
        if cfg.terminator_input_space not in {"state", "image", "both"}:
            raise ValueError(
                "terminator_input_space must be state|image|both, "
                f"got {cfg.terminator_input_space!r}."
            )
        if cfg.terminator_context not in {"prev_action", "proprio"}:
            raise ValueError(
                "terminator_context must be prev_action|proprio, "
                f"got {cfg.terminator_context!r}."
            )
        if cfg.terminator_model not in {"default", "rnn"}:
            raise ValueError(
                f"terminator_model must be default|rnn, got {cfg.terminator_model!r}."
            )
        if (
            not cfg.reconstructor_only
            and cfg.terminator_model == "rnn"
            and cfg.terminator_input_space != "state"
        ):
            raise ValueError(
                "The RNN terminator currently supports input_space=state only."
            )
        if cfg.state_rnn_terminator != (
            not cfg.reconstructor_only and cfg.terminator_model == "rnn"
        ):
            raise ValueError(
                "state_rnn_terminator is an internal compatibility flag and must "
                "match terminator_model='rnn'."
            )
        if cfg.reconstructor_only and (cfg.terminator_progress or cfg.terminator_termination):
            raise ValueError(
                "reconstructor_only requires both terminator objectives to be disabled."
            )
        if not cfg.reconstructor_only and not (
            cfg.terminator_progress or cfg.terminator_termination
        ):
            raise ValueError(
                "A built terminator must enable progress, termination, or both."
            )
        if cfg.terminator_termination_only != (
            cfg.terminator_termination and not cfg.terminator_progress
        ):
            raise ValueError(
                "terminator_termination_only is an internal compatibility flag and "
                "must match the selected decoder objectives."
            )
        if cfg.vision_backbone not in {"dino", "siglip", "resnet"}:
            raise ValueError(
                "vision_backbone must be dino|siglip|resnet, "
                f"got {cfg.vision_backbone!r}."
            )
        if cfg.skill_cond_mode not in {"token", "broadcast"}:
            raise ValueError(f"skill_cond_mode must be token|broadcast, got {cfg.skill_cond_mode!r}.")
        if cfg.lr_schedule not in {"cosine", "constant"}:
            raise ValueError(f"lr_schedule must be cosine|constant, got {cfg.lr_schedule!r}.")
        if cfg.lr <= 0:
            raise ValueError(f"lr must be positive, got {cfg.lr}.")
        if cfg.reconstructor_only and cfg.terminator_termination_only:
            raise ValueError(
                "reconstructor_only and terminator_termination_only are mutually "
                "exclusive: reconstructor_only builds no terminator at all."
            )
        if cfg.reconstructor_only and cfg.terminator_only:
            raise ValueError(
                "reconstructor_only and terminator_only are mutually exclusive."
            )
        if cfg.reconstructor_only and cfg.state_rnn_terminator:
            raise ValueError(
                "state_rnn_terminator requires a terminator, but reconstructor_only "
                "removes the terminator branch."
            )
        for name in (
            "encoder_min", "encoder_max", "state_min", "state_max",
            "state_q01", "state_q99", "action_q01", "action_q99",
        ):
            if getattr(cfg, name) is None:
                raise ValueError(f"FSQ config is missing required normalization statistic: {name}")
        if cfg.encoder_input_mode == "optimal":
            for name in ("encoder_start_min", "encoder_start_max"):
                if getattr(cfg, name) is None:
                    raise ValueError(f"Optimal FSQ config is missing required statistic: {name}")
        if cfg.encoder_arch not in {"spline", "action_seq"}:
            raise ValueError(f"encoder_arch must be spline|action_seq, got {cfg.encoder_arch!r}.")
        autoencoder_contracts = {
            "raw": ("spline", "raw_state", "oneshot", "raw_state"),
            "zero": ("spline", "zero_grounded", "oneshot", "zero_grounded"),
            "action": (
                "action_seq",
                "zero_grounded",
                "action_seq_transformer",
                "zero_grounded",
            ),
            "norm_action": (
                "action_seq",
                "zero_grounded",
                "action_seq_transformer",
                "zero_grounded",
            ),
        }
        if cfg.autoencoder_mode not in {"legacy", *autoencoder_contracts}:
            raise ValueError(
                "autoencoder_mode must be raw|zero|action|norm_action|legacy, "
                f"got {cfg.autoencoder_mode!r}."
            )
        if cfg.autoencoder_mode != "legacy":
            actual_contract = (
                cfg.encoder_arch,
                cfg.encoder_input_mode,
                cfg.reconstructor_arch,
                cfg.reconstructor_output_mode,
            )
            expected_contract = autoencoder_contracts[cfg.autoencoder_mode]
            if actual_contract != expected_contract:
                raise ValueError(
                    f"autoencoder_mode={cfg.autoencoder_mode!r} requires "
                    f"{expected_contract}, got {actual_contract}."
                )
            if (
                not math.isfinite(cfg.action_gripper_weight)
                or not 0.0 < cfg.action_gripper_weight <= 1.0
            ):
                raise ValueError(
                    "action_gripper_weight must be in (0, 1], got "
                    f"{cfg.action_gripper_weight}."
                )
            expected_start_state = (
                cfg.reconstructor_start_state_conditioning == "adaln"
            )
            if cfg.reconstructor_start_state != expected_start_state:
                raise ValueError(
                    "Current autoencoder presets require "
                    "reconstructor_start_state_conditioning=concat/adaln to agree "
                    "with reconstructor_start_state; got "
                    f"{cfg.reconstructor_start_state_conditioning!r} and "
                    f"{cfg.reconstructor_start_state}."
                )
        if cfg.reconstructor_start_state_conditioning not in {"concat", "adaln"}:
            raise ValueError(
                "reconstructor_start_state_conditioning must be concat|adaln, "
                f"got {cfg.reconstructor_start_state_conditioning!r}."
            )
        if cfg.pair_loss not in {"none", "overlap", "js", "contrastive"}:
            raise ValueError(
                "pair_loss must be none|overlap|js|contrastive, "
                f"got {cfg.pair_loss!r}."
            )
        if not 0.0 <= cfg.entropy_conf_ceiling <= 1.0:
            raise ValueError(
                "entropy_conf_ceiling must be in [0, 1], "
                f"got {cfg.entropy_conf_ceiling}."
            )
        if cfg.init_calibration_gain <= 0:
            raise ValueError(
                "init_calibration_gain must be positive, "
                f"got {cfg.init_calibration_gain}."
            )
        if cfg.init_calibration_samples < 0:
            raise ValueError(
                "init_calibration_samples must be non-negative, "
                f"got {cfg.init_calibration_samples}."
            )
        if cfg.pair_weight < 0:
            raise ValueError(f"pair_weight must be non-negative, got {cfg.pair_weight}.")
        if cfg.pair_inv_temperature <= 0:
            raise ValueError(
                "pair_inv_temperature must be positive, "
                f"got {cfg.pair_inv_temperature}."
            )
        if cfg.route_loss and cfg.terminator_only:
            raise ValueError(
                "route_loss requires an enabled reconstructor."
            )
        if cfg.route_loss and cfg.reconstructor_arch == "chunk":
            raise ValueError(
                "route_loss requires a whole-skill reconstructor; "
                "the chunk reconstructor is legacy-only."
            )
        if cfg.pair_warmup_epochs < 0 or cfg.pair_ramp_epochs < 0:
            raise ValueError(
                "pair_warmup_epochs and pair_ramp_epochs must be non-negative, "
                f"got {cfg.pair_warmup_epochs} and {cfg.pair_ramp_epochs}."
            )
        directional_pmaxes = resolve_boundary_augmentation_pmaxes(
            cfg.boundary_aug_pmax,
            early_start_pmax=cfg.boundary_aug_early_start_pmax,
            late_start_pmax=cfg.boundary_aug_late_start_pmax,
            early_end_pmax=cfg.boundary_aug_early_end_pmax,
            late_end_pmax=cfg.boundary_aug_late_end_pmax,
        )
        (
            cfg.boundary_aug_early_start_pmax,
            cfg.boundary_aug_late_start_pmax,
            cfg.boundary_aug_early_end_pmax,
            cfg.boundary_aug_late_end_pmax,
        ) = directional_pmaxes
        cfg.boundary_aug_pmax = max(directional_pmaxes)
        cfg.boundary_aug_distribution = normalize_jitter_distribution(
            cfg.boundary_aug_distribution
        )
        if cfg.pair_loss != "none":
            if not any(directional_pmaxes):
                raise ValueError(
                    "FSQ boundary pair loss requires at least one positive "
                    "directional boundary augmentation pmax."
                )
        if cfg.reconstructor_arch not in {
            "chunk", "oneshot", *ACTION_SEQUENCE_RECONSTRUCTOR_ARCHS
        }:
            raise ValueError(
                "reconstructor_arch must be chunk|oneshot|action_seq|"
                "action_seq_transformer, "
                f"got {cfg.reconstructor_arch!r}."
            )
        if cfg.reconstructor_output_mode not in {
            "raw_state", "zero_grounded", "start_grounded"
        }:
            raise ValueError(
                "reconstructor_output_mode must be raw_state|zero_grounded|start_grounded, "
                f"got {cfg.reconstructor_output_mode!r}."
            )
        if cfg.reconstructor_arch == "oneshot":
            if cfg.encoder_arch != "spline":
                raise ValueError("oneshot reconstruction requires encoder_arch='spline'.")
            for name in ("reconstructor_min", "reconstructor_max"):
                value = getattr(cfg, name)
                if value is None:
                    raise ValueError(
                        f"Oneshot FSQ config is missing reconstruction statistic: {name}"
                    )
                if np.asarray(value).shape != (cfg.enc_dim,):
                    raise ValueError(
                        f"{name} must have shape ({cfg.enc_dim},), "
                        f"got {np.asarray(value).shape}."
                    )
        if is_action_sequence_reconstructor_arch(cfg.reconstructor_arch):
            if cfg.encoder_arch != "action_seq":
                raise ValueError(
                    "Action-sequence reconstruction requires encoder_arch='action_seq'."
                )
            if (
                cfg.reconstructor_arch == "action_seq"
                and cfg.reconstructor_start_state
            ):
                # Historical direct configs defaulted this flag to true and
                # the legacy GRU silently disabled it. Keep that checkpoint
                # behavior; the current action preset uses the Transformer.
                cfg.reconstructor_start_state = False
        if (
            cfg.fsq_entropy
            and cfg.entropy_joint
            and math.prod(int(v) for v in cfg.fsq_levels) > 16384
        ):
            raise ValueError(
                "fsq_entropy with entropy_joint enumerates prod(fsq_levels) codes; "
                f"{cfg.fsq_levels} is too large (max 16384)."
            )
        self.cfg = cfg
        if cfg.encoder_arch == "action_seq":
            self.encoder = ActionSeqEncoder(
                action_dim=cfg.action_dim,
                hidden_dim=cfg.hidden_dim,
                fsq_levels=cfg.fsq_levels,
                n_layers=cfg.num_layers,
                n_heads=cfg.image_encoder_heads,
                dropout=cfg.dropout,
                raw_actions=(
                    is_action_sequence_reconstructor_arch(cfg.reconstructor_arch)
                    and cfg.autoencoder_mode != "norm_action"
                ),
                action_q01=cfg.action_q01,
                action_q99=cfg.action_q99,
                clip_normalized_actions=cfg.autoencoder_mode == "norm_action",
                action_gripper_weight=cfg.action_gripper_weight,
            )
        else:
            encoder_cls = (
                SplineFSQEncoder if cfg.encoder_length_token else LengthFreeSplineFSQEncoder
            )
            self.encoder = encoder_cls(
                enc_dim=cfg.enc_dim,
                n_control=cfg.n_control,
                spline_degree=cfg.spline_degree,
                hidden_dim=cfg.hidden_dim,
                fsq_levels=cfg.fsq_levels,
                n_layers=cfg.num_layers,
                n_heads=cfg.image_encoder_heads,
                dropout=cfg.dropout,
                length_min=cfg.length_min,
                length_max=cfg.length_max,
                encoder_min=cfg.encoder_min,
                encoder_max=cfg.encoder_max,
                encoder_input_mode=cfg.encoder_input_mode,
                encoder_start_min=cfg.encoder_start_min,
                encoder_start_max=cfg.encoder_start_max,
                gripper_weight=cfg.action_gripper_weight,
            )
        if cfg.quantizer == "bsq":
            self.encoder.fsq = BSQ(cfg.bsq_code_dim)
        if cfg.terminator_only:
            self.reconstructor = None
        elif cfg.reconstructor_arch == "oneshot":
            # FSQ-original-style whole-trajectory decoder: z [+ start state when
            # reconstructor_start_state] -> the full control-point grid. No
            # length head — termination is the terminator's job.
            self.reconstructor = OneShotTrajectoryDecoder(
                fsq_dim=len(cfg.fsq_levels),
                enc_dim=cfg.enc_dim,
                n_control=cfg.n_control,
                hidden_dim=cfg.hidden_dim,
                n_layers=cfg.num_layers,
                dropout=cfg.dropout,
                predict_length=False,
                state_dim=(
                    cfg.state_dim
                    if cfg.reconstructor_start_state
                    and cfg.reconstructor_start_state_conditioning == "adaln"
                    else cfg.max_state_dim if cfg.reconstructor_start_state else 0
                ),
                start_state_conditioning=cfg.reconstructor_start_state_conditioning,
            )
        elif cfg.reconstructor_arch == "action_seq":
            self.reconstructor = ActionSequenceRNNDecoder(
                fsq_dim=len(cfg.fsq_levels),
                action_dim=cfg.action_dim,
                hidden_dim=cfg.hidden_dim,
                n_layers=cfg.num_layers,
                dropout=cfg.dropout,
                predict_termination=False,
            )
        elif cfg.reconstructor_arch == "action_seq_transformer":
            self.reconstructor = CausalActionSequenceTransformerDecoder(
                fsq_dim=len(cfg.fsq_levels),
                action_dim=cfg.action_dim,
                hidden_dim=cfg.hidden_dim,
                n_layers=cfg.num_layers,
                n_heads=cfg.image_encoder_heads,
                dropout=cfg.dropout,
                state_dim=cfg.state_dim if cfg.reconstructor_start_state else 0,
            )
        else:
            self.reconstructor = MotionChunkReconstructor(
                fsq_levels=cfg.fsq_levels,
                hidden_dim=cfg.hidden_dim,
                n_layers=cfg.num_layers,
                n_heads=cfg.image_encoder_heads,
                dropout=cfg.dropout,
                skill_cond_mode=cfg.skill_cond_mode,
                max_state_dim=cfg.max_state_dim,
                max_action_dim=cfg.max_action_dim,
                chunk_size=cfg.chunk_size,
                use_start_state=cfg.reconstructor_start_state,
            )
        terminator_context_dim = (
            cfg.action_dim if cfg.terminator_context == "prev_action" else cfg.state_dim
        )
        terminator_context_lo = (
            cfg.action_q01 if cfg.terminator_context == "prev_action" else cfg.state_min
        )
        terminator_context_hi = (
            cfg.action_q99 if cfg.terminator_context == "prev_action" else cfg.state_max
        )
        terminator_context_q01 = (
            cfg.action_q01 if cfg.terminator_context == "prev_action" else cfg.state_q01
        )
        terminator_context_q99 = (
            cfg.action_q99 if cfg.terminator_context == "prev_action" else cfg.state_q99
        )
        if cfg.reconstructor_only:
            self.terminator = None
        elif cfg.state_rnn_terminator:
            self.terminator = FSQStateRNNTerminator(
                state_dim=terminator_context_dim,
                fsq_levels=cfg.fsq_levels,
                state_q01=terminator_context_q01,
                state_q99=terminator_context_q99,
                context_mode=cfg.terminator_context,
                termination_only=cfg.terminator_termination_only,
            )
        elif cfg.terminator_input_space == "state":
            self.terminator = FSQStateMLPTerminator(
                state_dim=terminator_context_dim,
                fsq_levels=cfg.fsq_levels,
                state_q01=terminator_context_q01,
                state_q99=terminator_context_q99,
                context_mode=cfg.terminator_context,
                termination_only=cfg.terminator_termination_only,
            )
        else:
            terminator_cls = (
                FSQImageOnlyQueryTerminator
                if cfg.terminator_input_space == "image"
                else FSQQueryTerminator
            )
            self.terminator = terminator_cls(
                state_dim=terminator_context_dim,
                fsq_levels=cfg.fsq_levels,
                hidden_dim=cfg.hidden_dim,
                n_layers=cfg.image_encoder_layers,
                n_heads=cfg.image_encoder_heads,
                dropout=cfg.dropout,
                arch=cfg.terminator_arch,
                vision_backbone=cfg.vision_backbone,
                freeze_vision_encoder=cfg.freeze_vision_encoder,
                dino_model_path=cfg.dino_model_path,
                dino_image_size=cfg.dino_image_size,
                siglip_image_size=cfg.siglip_image_size,
                resnet_image_size=cfg.resnet_image_size,
                skill_cond_mode=cfg.skill_cond_mode,
                state_min=terminator_context_lo,
                state_max=terminator_context_hi,
                context_mode=cfg.terminator_context,
                context_gripper_weight=cfg.action_gripper_weight,
                termination_only=cfg.terminator_termination_only,
            )

    def _decode_reconstruction_route_candidates(
        self,
        *,
        z_norm: Tensor,
        start_state: Tensor,
        lengths: Tensor,
        samples_per_skill: int,
    ) -> dict[str, Tensor | None]:
        """Decode every code under the current decoder context, without gradients.

        Candidate axes are ``(B,K,...)`` for whole-trajectory decoders and
        code-only decoders run the K prototypes once and broadcast them.
        A start-state-conditioned decoder evaluates the exact context/code
        Cartesian product.
        """
        candidates: dict[str, Tensor | None] = {
            "route_candidate_ctrl": None,
            "route_candidate_action_sequence": None,
        }
        if not self.cfg.route_loss:
            return candidates
        if self.reconstructor is None:
            raise RuntimeError(
                "route_loss is enabled but no reconstructor was built."
            )

        code_ids = torch.arange(
            self.fsq.codebook_size, device=z_norm.device, dtype=torch.long
        )
        all_codes = self.fsq.code_to_normalized(code_ids).to(
            device=z_norm.device, dtype=z_norm.dtype
        )
        bsize, code_count = z_norm.shape[0], all_codes.shape[0]
        with torch.no_grad():
            if self.cfg.reconstructor_arch == "oneshot":
                if getattr(self.reconstructor, "state_dim", 0) > 0:
                    dec_state = start_state.view(
                        bsize, samples_per_skill, -1
                    )[:, 0]
                    code_batch = (
                        all_codes.unsqueeze(0)
                        .expand(bsize, code_count, -1)
                        .reshape(bsize * code_count, -1)
                    )
                    state_batch = (
                        dec_state.unsqueeze(1)
                        .expand(bsize, code_count, -1)
                        .reshape(bsize * code_count, -1)
                    )
                    ctrl, _ = self.reconstructor(
                        code_batch, start_state=state_batch
                    )
                    ctrl = ctrl.view(
                        bsize,
                        code_count,
                        self.cfg.n_control,
                        self.cfg.enc_dim,
                    )
                else:
                    ctrl, _ = self.reconstructor(all_codes)
                    ctrl = ctrl.unsqueeze(0).expand(bsize, -1, -1, -1)
                candidates["route_candidate_ctrl"] = ctrl.detach()
            elif is_action_sequence_reconstructor_arch(
                self.cfg.reconstructor_arch
            ):
                if getattr(self.reconstructor, "state_dim", 0) > 0:
                    dec_state = start_state.view(
                        bsize, samples_per_skill, -1
                    )[:, 0]
                    code_batch = (
                        all_codes.unsqueeze(0)
                        .expand(bsize, code_count, -1)
                        .reshape(bsize * code_count, -1)
                    )
                    state_batch = (
                        dec_state.unsqueeze(1)
                        .expand(bsize, code_count, -1)
                        .reshape(bsize * code_count, -1)
                    )
                    sequence, _ = self.reconstructor(
                        code_batch,
                        int(lengths.max()),
                        start_state=state_batch,
                    )
                    sequence = sequence.view(
                        bsize,
                        code_count,
                        int(lengths.max()),
                        self.cfg.action_dim,
                    )
                else:
                    sequence, _ = self.reconstructor(
                        all_codes, int(lengths.max())
                    )
                    sequence = sequence.unsqueeze(0).expand(
                        bsize, -1, -1, -1
                    )
                candidates["route_candidate_action_sequence"] = sequence.detach()
            else:
                raise RuntimeError(
                    "Reconstruction-aware routing supports whole-skill "
                    "reconstructors only; the chunk reconstructor is legacy-only."
                )
        return candidates

    def _decode_termination_route_candidates(
        self,
        *,
        z_norm: Tensor,
        terminator_context: Tensor,
        lengths: Tensor,
        samples_per_skill: int,
        terminator_context_sequence: Tensor | None,
        image_tokens: Tensor | None,
    ) -> dict[str, Tensor | None]:
        """Evaluate the termination head under every code without gradients.

        The returned logits use the same ``(B,K,T)`` contract for both the
        full-sequence state terminator and the sampled-timestep terminators.
        Visual features are supplied by the caller so the expensive backbone
        is run once and shared with the ordinary selected-code forward pass.
        """
        candidates: dict[str, Tensor | None] = {
            "route_candidate_term_logits": None,
        }
        if (
            not self.cfg.route_loss
            or self.terminator is None
            or self.cfg.reconstructor_only
            or not self.cfg.terminator_termination
            or self.cfg.end_loss_weight == 0.0
        ):
            return candidates

        code_ids = torch.arange(
            self.fsq.codebook_size, device=z_norm.device, dtype=torch.long
        )
        all_codes = self.fsq.code_to_normalized(code_ids).to(
            device=z_norm.device, dtype=z_norm.dtype
        )
        bsize = z_norm.shape[0]
        code_count = all_codes.shape[0]

        with torch.no_grad():
            if self.cfg.state_rnn_terminator:
                if terminator_context_sequence is None:
                    raise ValueError(
                        "state_rnn_terminator route scoring requires the cached "
                        "full-skill state sequence."
                    )
                # The state-only path is cheap enough to evaluate all codes in
                # one batch.  Keep B adjacent within each code chunk contract.
                code_batch = (
                    all_codes.unsqueeze(0)
                    .expand(bsize, code_count, -1)
                    .reshape(bsize * code_count, -1)
                )
                state_batch = (
                    terminator_context_sequence.unsqueeze(1)
                    .expand(bsize, code_count, *terminator_context_sequence.shape[1:])
                    .reshape(
                        bsize * code_count,
                        *terminator_context_sequence.shape[1:],
                    )
                )
                length_batch = (
                    lengths.unsqueeze(1)
                    .expand(bsize, code_count)
                    .reshape(bsize * code_count)
                )
                _, logits, _ = self.terminator.forward_all_outputs(
                    code_batch,
                    state_batch,
                    lengths=length_batch,
                )
                candidates["route_candidate_term_logits"] = logits.view(
                    bsize, code_count, logits.shape[-1]
                ).detach()
                return candidates

            sample_count = terminator_context.shape[0]
            expected_samples = bsize * samples_per_skill
            if sample_count != expected_samples:
                raise ValueError(
                    "Terminator route scoring expected B*samples_per_skill states, "
                    f"got {sample_count} vs {expected_samples}."
                )

            if self.cfg.terminator_input_space in {"image", "both"}:
                if image_tokens is None:
                    raise ValueError(
                        "Visual terminator route scoring requires precomputed image tokens."
                    )
                # Long DINO/SigLIP token sequences make code batching memory
                # hungry; ResNet's shorter spatial sequence safely amortizes a
                # few codes per call.  This is an internal execution detail,
                # not a tuning parameter.
                code_chunk_size = 4 if image_tokens.shape[1] <= 128 else 1
            else:
                code_chunk_size = code_count

            logits_by_chunk: list[Tensor] = []
            for code_chunk in all_codes.split(code_chunk_size, dim=0):
                chunk_size = code_chunk.shape[0]
                code_batch = (
                    code_chunk.unsqueeze(0)
                    .expand(sample_count, chunk_size, -1)
                    .reshape(sample_count * chunk_size, -1)
                )
                state_batch = (
                    terminator_context.unsqueeze(1)
                    .expand(sample_count, chunk_size, -1)
                    .reshape(sample_count * chunk_size, -1)
                )
                if self.cfg.terminator_input_space == "state":
                    _, logits = self.terminator.forward_outputs(
                        code_batch, state_batch
                    )
                else:
                    token_batch = (
                        image_tokens.unsqueeze(1)
                        .expand(
                            sample_count,
                            chunk_size,
                            image_tokens.shape[1],
                            image_tokens.shape[2],
                        )
                        .reshape(
                            sample_count * chunk_size,
                            image_tokens.shape[1],
                            image_tokens.shape[2],
                        )
                    )
                    if self.cfg.terminator_input_space == "image":
                        _, logits = self.terminator._forward_without_state(
                            code_batch, token_batch
                        )
                    else:
                        _, logits = self.terminator._forward_from_image_tokens(
                            code_batch, state_batch, token_batch
                        )
                logits_by_chunk.append(logits.view(sample_count, chunk_size))

            sampled_logits = torch.cat(logits_by_chunk, dim=1)
            candidates["route_candidate_term_logits"] = (
                sampled_logits.view(bsize, samples_per_skill, code_count)
                .permute(0, 2, 1)
                .contiguous()
                .detach()
            )
        return candidates

    def gradient_checkpointing_enable(self) -> None:
        if hasattr(self.encoder, "enc_traj_pool"):
            self.encoder.enc_traj_pool.gradient_checkpointing_enable()
        if self.reconstructor is not None:
            if hasattr(self.reconstructor, "pool"):
                self.reconstructor.pool.gradient_checkpointing_enable()
        if self.terminator is not None and hasattr(
            self.terminator, "gradient_checkpointing_enable"
        ):
            self.terminator.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        if hasattr(self.encoder, "enc_traj_pool"):
            self.encoder.enc_traj_pool.gradient_checkpointing_disable()
        if self.reconstructor is not None and hasattr(self.reconstructor, "pool"):
            self.reconstructor.pool.gradient_checkpointing_disable()
        if self.terminator is not None and hasattr(
            self.terminator, "gradient_checkpointing_disable"
        ):
            self.terminator.gradient_checkpointing_disable()

    @property
    def fsq(self) -> FSQ | BSQ:
        return self.encoder.fsq

    # Small read-only surface used by evaluation scripts. Action predictions go
    # through the transformer reconstructor, not a PI05/Gemma action expert.
    @property
    def action_dim(self) -> int:
        return self.cfg.action_dim

    @property
    def n_control(self) -> int:
        return self.cfg.n_control

    @property
    def spline_degree(self) -> int:
        return self.cfg.spline_degree

    @property
    def state_dim(self) -> int:
        return self.cfg.state_dim

    @property
    def chunk_size(self) -> int:
        return self.cfg.chunk_size

    def encode(
        self, ctrl: Tensor, lengths: Tensor, start_pose: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        return self.encoder(ctrl, lengths, start_pose, normalized=True)

    def encode_numpy(self, trajectory: np.ndarray, device: str | torch.device = "cpu") -> np.ndarray:
        if self.cfg.encoder_arch == "action_seq":
            raise ValueError("This model encodes ACTION sequences; use encode_actions_numpy.")
        return self.encoder.encode_numpy(trajectory, device)

    def encode_index(self, trajectory: np.ndarray, device: str | torch.device = "cpu") -> int:
        if self.cfg.encoder_arch == "action_seq":
            raise ValueError("This model encodes ACTION sequences; use encode_actions_index.")
        return self.encoder.encode_index(trajectory, device)

    def _prepare_actions_numpy(self, actions: np.ndarray) -> Tensor:
        actions = np.asarray(actions, dtype=np.float32)
        if is_action_sequence_reconstructor_arch(self.cfg.reconstructor_arch):
            if self.cfg.autoencoder_mode == "norm_action":
                normalized = normalize_action_sequence(
                    actions,
                    self.cfg.action_q01,
                    self.cfg.action_q99,
                    gripper_weight=self.cfg.action_gripper_weight,
                    clip=True,
                )
                return torch.from_numpy(normalized)
            scaled = scale_gripper_features(
                actions,
                gripper_weight=self.cfg.action_gripper_weight,
                gripper_dims=1,
            )
            return torch.from_numpy(scaled)
        lo = np.asarray(self.cfg.action_q01, dtype=np.float32)
        hi = np.asarray(self.cfg.action_q99, dtype=np.float32)
        norm = 2.0 * (actions - lo) / (hi - lo + 1e-8) - 1.0
        return torch.from_numpy(norm)

    def _previous_action_context(
        self,
        emitted_actions: Tensor,
        initial_previous_action: Tensor | None = None,
    ) -> Tensor:
        """Right-shift raw emitted actions into normalized t-1 context tokens."""
        if emitted_actions.ndim != 3:
            raise ValueError(
                "emitted_actions must have shape (B,T,A), got "
                f"{tuple(emitted_actions.shape)}."
            )
        lo = torch.as_tensor(
            self.cfg.action_q01,
            device=emitted_actions.device,
            dtype=emitted_actions.dtype,
        )
        hi = torch.as_tensor(
            self.cfg.action_q99,
            device=emitted_actions.device,
            dtype=emitted_actions.dtype,
        )
        normalized = (
            2.0 * (emitted_actions[..., : self.cfg.action_dim] - lo)
            / (hi - lo + 1e-8)
            - 1.0
        ).clamp(-1.0, 1.0)
        if self.cfg.action_gripper_weight != 1.0:
            normalized = normalized.clone()
            normalized[..., -1] *= math.sqrt(self.cfg.action_gripper_weight)
        previous = torch.zeros_like(normalized)
        previous[:, 1:] = normalized[:, :-1]
        if initial_previous_action is not None:
            initial = initial_previous_action
            if initial.ndim == 1:
                initial = initial.unsqueeze(0)
            expected = (emitted_actions.shape[0], self.cfg.action_dim)
            if tuple(initial.shape) != expected:
                raise ValueError(
                    "initial_previous_action must have shape (B,A), got "
                    f"{tuple(initial.shape)}; expected {expected}."
                )
            initial = initial.to(
                device=emitted_actions.device,
                dtype=emitted_actions.dtype,
            )
            initial_norm = (
                2.0 * (initial - lo) / (hi - lo + 1e-8) - 1.0
            ).clamp(-1.0, 1.0)
            if self.cfg.action_gripper_weight != 1.0:
                initial_norm = initial_norm.clone()
                initial_norm[..., -1] *= math.sqrt(
                    self.cfg.action_gripper_weight
                )
            previous[:, 0] = initial_norm
        return previous

    @torch.no_grad()
    def encode_actions_numpy(
        self, actions: np.ndarray, device: str | torch.device = "cpu"
    ) -> np.ndarray:
        """action_seq encoder: raw (T, A) dataset-unit actions -> z_q."""
        if self.cfg.encoder_arch != "action_seq":
            raise ValueError("encode_actions_numpy requires encoder_arch='action_seq'.")
        acts = self._prepare_actions_numpy(actions).unsqueeze(0).to(device)
        lengths = torch.tensor([acts.shape[1]], dtype=torch.long, device=device)
        z_q, _ = self.encoder(acts, lengths)
        return z_q[0].cpu().numpy()

    @torch.no_grad()
    def encode_actions_index(
        self, actions: np.ndarray, device: str | torch.device = "cpu"
    ) -> int:
        if self.cfg.encoder_arch != "action_seq":
            raise ValueError("encode_actions_index requires encoder_arch='action_seq'.")
        acts = self._prepare_actions_numpy(actions).unsqueeze(0).to(device)
        lengths = torch.tensor([acts.shape[1]], dtype=torch.long, device=device)
        _, index = self.encoder(acts, lengths)
        return int(index.item())

    @torch.no_grad()
    def sample_action_chunks(
        self,
        z_q: Tensor,
        raw_states: Tensor,
        progress: Tensor | None = None,
        *,
        noise: Tensor | None = None,
        num_steps: int = 10,
    ) -> Tensor:
        """Image-free reconstructor action chunks for every ``(B,T)`` progress value, in dataset units."""
        bsize, steps = raw_states.shape[:2]
        _ = noise, num_steps  # kept for API compatibility with older eval callers
        start_raw = raw_states[:, :1, : self.cfg.state_dim].expand(bsize, steps, -1)
        flat_start = start_raw.reshape(bsize * steps, -1)
        lo = torch.as_tensor(self.cfg.state_q01, device=flat_start.device, dtype=flat_start.dtype)
        hi = torch.as_tensor(self.cfg.state_q99, device=flat_start.device, dtype=flat_start.dtype)
        norm = 2.0 * (flat_start - lo) / (hi - lo + 1e-8) - 1.0
        start_state = torch.zeros(
            bsize * steps, self.cfg.max_state_dim, device=flat_start.device, dtype=flat_start.dtype
        )
        start_state[:, : self.cfg.state_dim] = norm
        if progress is None:
            progress = torch.arange(steps, device=raw_states.device, dtype=raw_states.dtype)[None].expand(bsize, -1)
            progress = progress / max(steps - 1, 1)
        progress = progress.to(device=raw_states.device, dtype=raw_states.dtype).reshape(bsize * steps)
        if self.reconstructor is None:
            raise RuntimeError(
                "This FSQ model was trained terminator_only and has no reconstructor."
            )
        if self.cfg.reconstructor_arch != "chunk":
            raise RuntimeError(
                "Per-timestep action chunks require reconstructor_arch='chunk'; "
                f"this model uses {self.cfg.reconstructor_arch!r}."
            )
        z_norm = self.fsq.normalized(z_q).repeat_interleave(steps, dim=0)
        action_norm = self.reconstructor(
            start_state,
            z_norm,
            progress,
        )[..., : self.cfg.action_dim]
        action_lo = torch.as_tensor(
            self.cfg.action_q01, device=action_norm.device, dtype=action_norm.dtype
        )
        action_hi = torch.as_tensor(
            self.cfg.action_q99, device=action_norm.device, dtype=action_norm.dtype
        )
        actions = (action_norm + 1.0) * 0.5 * (action_hi - action_lo) + action_lo
        return actions.view(bsize, steps, self.cfg.chunk_size, self.cfg.action_dim)

    def _normalize_reconstructor_start_state(
        self,
        raw_start_states: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        """Normalize absolute start proprioception for decoder conditioning.

        New AdaLN conditioning uses exact state min/max so its numerical range
        matches the autoencoder's [-1, 1] contract without applying a
        zero-grounded delta range to an absolute pose.  Legacy concat models
        retain their historical q01/q99 convention for checkpoint fidelity.
        """
        if raw_start_states.ndim != 2:
            raise ValueError(
                "raw_start_states must have shape (B, state_dim), got "
                f"{tuple(raw_start_states.shape)}."
            )
        if raw_start_states.shape[-1] < self.cfg.state_dim:
            raise ValueError(
                f"raw_start_states needs at least {self.cfg.state_dim} features, "
                f"got {raw_start_states.shape[-1]}."
            )
        raw = raw_start_states[:, : self.cfg.state_dim].to(dtype=dtype)
        if self.cfg.reconstructor_start_state_conditioning == "adaln":
            lo_values, hi_values = self.cfg.state_min, self.cfg.state_max
        else:
            lo_values, hi_values = self.cfg.state_q01, self.cfg.state_q99
        lo = torch.as_tensor(lo_values, device=raw.device, dtype=raw.dtype)
        hi = torch.as_tensor(hi_values, device=raw.device, dtype=raw.dtype)
        normalized = 2.0 * (raw - lo) / (hi - lo + 1e-8) - 1.0
        if self.cfg.action_gripper_weight != 1.0:
            normalized = normalized.clone()
            normalized[..., -N_GRIPPER_DIMS:] *= math.sqrt(
                self.cfg.action_gripper_weight
            )
        state = torch.zeros(
            raw.shape[0],
            self.cfg.max_state_dim,
            device=raw.device,
            dtype=raw.dtype,
        )
        state[:, : self.cfg.state_dim] = normalized
        return state

    @torch.no_grad()
    def sample_action_sequence(
        self,
        z_q: Tensor,
        steps: int,
        raw_start_states: Tensor | None = None,
    ) -> Tensor:
        """Decode dataset-unit controller actions ``(B, steps, action_dim)``.

        ``raw_start_states`` is required only for an AdaLN-conditioned action
        decoder and is supplied in dataset units.
        """
        if self.reconstructor is None:
            raise RuntimeError(
                "This FSQ model was trained terminator_only and has no reconstructor."
            )
        if not is_action_sequence_reconstructor_arch(self.cfg.reconstructor_arch):
            raise RuntimeError(
                "Full action-sequence decoding requires "
                "an action-sequence reconstructor, "
                f"got {self.cfg.reconstructor_arch!r}."
            )
        z_norm = self.fsq.normalized(z_q)
        start_state = None
        if getattr(self.reconstructor, "state_dim", 0) > 0:
            if raw_start_states is None:
                raise ValueError(
                    "This action-sequence decoder requires raw start states."
                )
            start_state = self._normalize_reconstructor_start_state(
                raw_start_states,
                dtype=z_norm.dtype,
            )
        if getattr(self.reconstructor, "state_dim", 0) > 0:
            action_values, _ = self.reconstructor(
                z_norm,
                int(steps),
                start_state=start_state,
            )
        else:
            action_values, _ = self.reconstructor(z_norm, int(steps))
        # Invert the shared representation weight so public sampling always
        # returns unweighted dataset-unit gripper values.
        values = action_values
        if self.cfg.action_gripper_weight != 1.0:
            values = values.clone()
            values[..., -1] = (
                values[..., -1] / math.sqrt(self.cfg.action_gripper_weight)
            ).clamp(-1.0, 1.0)
        if self.cfg.autoencoder_mode != "norm_action":
            return values

        normalized = values.clone()
        normalized[..., -1].clamp_(-1.0, 1.0)
        lo = torch.as_tensor(
            self.cfg.action_q01,
            device=normalized.device,
            dtype=normalized.dtype,
        )
        hi = torch.as_tensor(
            self.cfg.action_q99,
            device=normalized.device,
            dtype=normalized.dtype,
        )
        return (normalized + 1.0) * 0.5 * (hi - lo) + lo

    @torch.no_grad()
    def sample_control_points(
        self, z_q: Tensor, raw_start_states: Tensor | None = None
    ) -> Tensor:
        """Oneshot reconstructor control points ``(B, n_control, enc_dim)``.

        The oneshot counterpart of ``sample_action_chunks``: one control-point
        grid per skill, returned in the independently configured reconstruction
        output units.
        ``raw_start_states`` are the per-skill start states in dataset units,
        required only when the decoder was built with a start-state input.
        """
        if self.reconstructor is None:
            raise RuntimeError(
                "This FSQ model was trained terminator_only and has no reconstructor."
            )
        if self.cfg.reconstructor_arch != "oneshot":
            raise RuntimeError(
                "Control points exist only for the oneshot reconstructor; this model "
                "uses the per-timestep chunk reconstructor (sample_action_chunks)."
            )
        z_norm = self.fsq.normalized(z_q)
        start_state = None
        if getattr(self.reconstructor, "state_dim", 0) > 0:
            if raw_start_states is None:
                raise ValueError("This oneshot decoder was built with a start-state input.")
            start_state = self._normalize_reconstructor_start_state(
                raw_start_states,
                dtype=z_norm.dtype,
            )
        ctrl_norm, _ = self.reconstructor(z_norm, start_state=start_state)
        if self.cfg.action_gripper_weight != 1.0:
            ctrl_norm = ctrl_norm.clone()
            ctrl_norm[..., -N_GRIPPER_DIMS:] /= math.sqrt(
                self.cfg.action_gripper_weight
            )
        ctrl_lo = torch.as_tensor(
            self.cfg.reconstructor_min, device=ctrl_norm.device, dtype=ctrl_norm.dtype
        )
        ctrl_hi = torch.as_tensor(
            self.cfg.reconstructor_max, device=ctrl_norm.device, dtype=ctrl_norm.dtype
        )
        return (ctrl_norm + 1.0) * 0.5 * (ctrl_hi - ctrl_lo + 1e-8) + ctrl_lo

    @torch.no_grad()
    def decode(
        self,
        z_q: Tensor,
        raw_states: Tensor,
        third: Tensor | None = None,
        wrist: Tensor | None = None,
        _progress_hint: Tensor | None = None,
        *,
        emitted_actions: Tensor | None = None,
        initial_previous_action: Tensor | None = None,
        num_steps: int = 10,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate every trajectory timestep with the transformer reconstructor and terminator.

        This convenience method vectorizes inference over ``(B,T,...)``; training
        still uses the B-trajectory x M-sampled-timestep path. Returned actions are
        de-normalized to dataset units.
        """
        bsize, steps = raw_states.shape[:2]
        flat_state = raw_states.reshape(bsize * steps, -1)[..., : self.cfg.state_dim]
        z_norm = self.fsq.normalized(z_q).repeat_interleave(steps, dim=0)
        actions = self.sample_action_chunks(
            z_q,
            raw_states,
            _progress_hint,
            noise=noise,
            num_steps=num_steps,
        )

        def flatten_camera(value: Tensor | None) -> Tensor | None:
            if value is None:
                return None
            return value.reshape(bsize * steps, *value.shape[2:])

        if self.terminator is None:
            raise RuntimeError(
                "This FSQ model was trained reconstructor_only and has no terminator."
            )
        if (
            self.cfg.terminator_context == "prev_action"
            and self.cfg.terminator_input_space != "image"
        ):
            if emitted_actions is None:
                raise ValueError(
                    "prev_action terminator decode requires the actions actually emitted "
                    "at each timestep so they can be shifted by one step."
                )
            context_sequence = self._previous_action_context(
                emitted_actions,
                initial_previous_action=initial_previous_action,
            )
            flat_context = context_sequence.reshape(
                bsize * steps, self.cfg.action_dim
            )
        else:
            context_sequence = raw_states[..., : self.cfg.state_dim]
            flat_context = flat_state
        if self.cfg.state_rnn_terminator:
            progress, term_logits, _ = self.terminator.forward_all_outputs(
                self.fsq.normalized(z_q),
                context_sequence,
            )
            return actions, progress, term_logits
        if self.cfg.terminator_input_space == "state":
            progress, term_logits = self.terminator.forward_outputs(z_norm, flat_context)
        elif self.cfg.terminator_input_space == "image":
            progress, term_logits = self.terminator(
                z_norm, flatten_camera(third), flatten_camera(wrist)
            )
        else:
            progress, term_logits = self.terminator(
                z_norm, flat_context, flatten_camera(third), flatten_camera(wrist)
            )
        return (
            actions,
            progress.view(bsize, steps),
            term_logits.view(bsize, steps),
        )

    def forward(
        self,
        *,
        ctrl: Tensor,
        lengths: Tensor,
        start_pose: Tensor | None = None,
        start_state: Tensor,
        raw_state: Tensor,
        prev_action: Tensor | None = None,
        progress_target: Tensor,
        third: Tensor | None,
        wrist: Tensor | None,
        samples_per_skill: int,
        terminator_context_sequence: Tensor | None = None,
        terminator_state_sequence: Tensor | None = None,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        action_seq: Tensor | None = None,
        augmented_ctrl: Tensor | None = None,
        augmented_action_seq: Tensor | None = None,
        augmented_lengths: Tensor | None = None,
        augmented_start_pose: Tensor | None = None,
        negative_ctrl: Tensor | None = None,
        negative_action_seq: Tensor | None = None,
        negative_lengths: Tensor | None = None,
        negative_start_pose: Tensor | None = None,
        compute_skill_shuffle: bool = False,
    ) -> dict[str, Tensor]:
        if (
            self.cfg.terminator_context == "prev_action"
            and self.cfg.terminator_input_space != "image"
        ):
            if prev_action is None and self.terminator is not None:
                raise ValueError(
                    "prev_action terminator requires the normalized action emitted at t-1."
                )
            terminator_context = prev_action
        else:
            terminator_context = raw_state
        if terminator_context is None:
            # Reconstructor-only models never consume it; keep the common
            # forward surface tensor-complete without introducing a dummy API.
            terminator_context = raw_state
        if terminator_context_sequence is None:
            # Compatibility for direct callers of historical proprio-RNN models.
            terminator_context_sequence = terminator_state_sequence
        if self.cfg.encoder_arch == "action_seq":
            if action_seq is None:
                raise ValueError("encoder_arch='action_seq' requires the action_seq input.")
            z_e = self.encoder.encode_continuous(
                action_seq[:, : int(lengths.max())], lengths
            )
        else:
            z_e = self.encoder.encode_continuous(ctrl, lengths, start_pose, normalized=True)
        z_q, indices = self.fsq(z_e)
        pair_configured = self.cfg.pair_loss != "none"
        augmented_pair_input = (
            augmented_action_seq
            if self.cfg.encoder_arch == "action_seq"
            else augmented_ctrl
        )
        pair_enabled = pair_configured and augmented_pair_input is not None
        # FSQ uses its bounded grid coordinate; BSQ.bound returns z/||z||.
        u_cont = (
            self.fsq.bound(z_e)
            if self.cfg.fsq_entropy
            or pair_configured
            or self.cfg.route_loss
            else None
        )
        augmented_u_cont = augmented_indices = None
        if pair_enabled:
            if augmented_lengths is None:
                raise ValueError("FSQ pair loss requires augmented_lengths.")
            if self.cfg.encoder_arch == "action_seq":
                augmented_z_e = self.encoder.encode_continuous(
                    augmented_action_seq[:, : int(augmented_lengths.max())],
                    augmented_lengths,
                )
            else:
                augmented_z_e = self.encoder.encode_continuous(
                    augmented_ctrl,
                    augmented_lengths,
                    augmented_start_pose,
                    normalized=True,
                )
            _, augmented_indices = self.fsq(augmented_z_e)
            augmented_u_cont = self.fsq.bound(augmented_z_e)
        negative_u_cont = negative_indices = None
        negative_pair_input = (
            negative_action_seq
            if self.cfg.encoder_arch == "action_seq"
            else negative_ctrl
        )
        if self.cfg.pair_loss == "contrastive" and negative_pair_input is not None:
            if negative_lengths is None:
                raise ValueError("Contrastive pair loss requires negative_lengths.")
            if self.cfg.encoder_arch == "action_seq":
                negative_z_e = self.encoder.encode_continuous(
                    negative_action_seq[:, : int(negative_lengths.max())],
                    negative_lengths,
                )
            else:
                negative_z_e = self.encoder.encode_continuous(
                    negative_ctrl,
                    negative_lengths,
                    negative_start_pose,
                    normalized=True,
                )
            _, negative_indices = self.fsq(negative_z_e)
            negative_u_cont = self.fsq.bound(negative_z_e)
        z_norm = self.fsq.normalized(z_q)
        z_sample = z_norm.repeat_interleave(samples_per_skill, dim=0)
        shuffled_z_norm = shuffled_z_sample = shuffle_valid = None
        if compute_skill_shuffle and self.terminator is not None:
            shuffle_sources, shuffle_valid = different_code_shuffle_sources(indices)
            shuffled_z_norm = z_norm.index_select(0, shuffle_sources)
            shuffled_z_sample = shuffled_z_norm.repeat_interleave(
                samples_per_skill, dim=0
            )
        _ = noise, time  # older validation path passes these; the reconstructor is deterministic.
        ctrl_hat = None
        action_sequence_hat = None
        if self.reconstructor is not None and self.cfg.reconstructor_arch == "oneshot":
            # One decode per TRAJECTORY; samples_per_skill applies only to the
            # terminator below. start_state rows repeat per trajectory, so the
            # first of each group is the per-trajectory skill-start state.
            dec_state = None
            if getattr(self.reconstructor, "state_dim", 0) > 0:
                dec_state = start_state.view(
                    z_norm.shape[0], samples_per_skill, -1
                )[:, 0]
            ctrl_hat, _ = self.reconstructor(z_norm, start_state=dec_state)
        elif (
            self.reconstructor is not None
            and is_action_sequence_reconstructor_arch(self.cfg.reconstructor_arch)
        ):
            dec_state = None
            if getattr(self.reconstructor, "state_dim", 0) > 0:
                dec_state = start_state.view(
                    z_norm.shape[0], samples_per_skill, -1
                )[:, 0]
            if getattr(self.reconstructor, "state_dim", 0) > 0:
                action_sequence_hat, _ = self.reconstructor(
                    z_norm,
                    int(lengths.max()),
                    start_state=dec_state,
                )
            else:
                action_sequence_hat, _ = self.reconstructor(
                    z_norm,
                    int(lengths.max()),
                )
        if (
            self.reconstructor is None
            or ctrl_hat is not None
            or action_sequence_hat is not None
        ):
            actions = torch.zeros(
                z_sample.shape[0],
                self.cfg.chunk_size,
                self.cfg.max_action_dim,
                device=z_sample.device,
                dtype=z_sample.dtype,
            )
        else:
            actions = self.reconstructor(
                start_state,
                z_sample,
                progress_target,
            )
        route_candidates = self._decode_reconstruction_route_candidates(
            z_norm=z_norm,
            start_state=start_state,
            lengths=lengths,
            samples_per_skill=samples_per_skill,
        )
        terminator_image_tokens = None
        if (
            self.cfg.route_loss
            and self.terminator is not None
            and not self.cfg.reconstructor_only
            and self.cfg.terminator_termination
            and self.cfg.end_loss_weight != 0.0
            and not self.cfg.state_rnn_terminator
            and self.cfg.terminator_input_space in {"image", "both"}
        ):
            # Share this graph-bearing visual encoding with the ordinary
            # selected-code terminator pass. Candidate heads run under
            # no_grad, while the selected-code path still trains image_proj
            # and an unfrozen vision backbone normally.
            terminator_image_tokens = self.terminator._prepare_image_tokens(
                third, wrist
            )
        route_candidates.update(
            self._decode_termination_route_candidates(
                z_norm=z_norm,
                terminator_context=terminator_context,
                lengths=lengths,
                samples_per_skill=samples_per_skill,
                terminator_context_sequence=terminator_context_sequence,
                image_tokens=terminator_image_tokens,
            )
        )
        shuffled_progress = shuffled_term_logits = None
        if self.terminator is None:
            zeros = torch.zeros(
                z_sample.shape[0], device=z_sample.device, dtype=actions.dtype
            )
            progress, term_logits = zeros, zeros
        elif self.cfg.state_rnn_terminator:
            if terminator_context_sequence is None:
                raise ValueError(
                    "state_rnn_terminator requires the cached full-skill context sequence."
                )
            progress, term_logits, _ = self.terminator.forward_all_outputs(
                z_norm,
                terminator_context_sequence,
                lengths=lengths,
            )
            if shuffled_z_norm is not None:
                shuffled_progress, shuffled_term_logits, _ = (
                    self.terminator.forward_all_outputs(
                        shuffled_z_norm,
                        terminator_context_sequence,
                        lengths=lengths,
                    )
                )
        elif self.cfg.terminator_input_space == "state":
            progress, term_logits = self.terminator.forward_outputs(
                z_sample, terminator_context
            )
            if shuffled_z_sample is not None:
                shuffled_progress, shuffled_term_logits = self.terminator.forward_outputs(
                    shuffled_z_sample, terminator_context
                )
        elif self.cfg.terminator_input_space == "image":
            if terminator_image_tokens is not None:
                progress, term_logits = self.terminator._forward_without_state(
                    z_sample, terminator_image_tokens
                )
                if shuffled_z_sample is not None:
                    shuffled_progress, shuffled_term_logits = (
                        self.terminator._forward_without_state(
                            shuffled_z_sample, terminator_image_tokens
                        )
                    )
            elif shuffled_z_sample is None:
                progress, term_logits = self.terminator(z_sample, third, wrist)
            else:
                (
                    progress,
                    term_logits,
                    shuffled_progress,
                    shuffled_term_logits,
                ) = self.terminator.forward_with_skill_shuffle(
                    z_sample,
                    shuffled_z_sample,
                    third,
                    wrist,
                )
        else:
            if terminator_image_tokens is not None:
                progress, term_logits = self.terminator._forward_from_image_tokens(
                    z_sample, terminator_context, terminator_image_tokens
                )
                if shuffled_z_sample is not None:
                    shuffled_progress, shuffled_term_logits = (
                        self.terminator._forward_from_image_tokens(
                            shuffled_z_sample,
                            terminator_context,
                            terminator_image_tokens,
                        )
                    )
            elif shuffled_z_sample is None:
                progress, term_logits = self.terminator(
                    z_sample, terminator_context, third, wrist
                )
            else:
                (
                    progress,
                    term_logits,
                    shuffled_progress,
                    shuffled_term_logits,
                ) = self.terminator.forward_with_skill_shuffle(
                    z_sample,
                    shuffled_z_sample,
                    terminator_context,
                    third,
                    wrist,
                )
        result = {
            "z_q": z_q,
            "indices": indices,
            "u_cont": u_cont,
            "augmented_u_cont": augmented_u_cont,
            "augmented_indices": augmented_indices,
            "negative_u_cont": negative_u_cont,
            "negative_indices": negative_indices,
            "ctrl_hat": ctrl_hat,
            "action_sequence_hat": action_sequence_hat,
            "actions": actions,
            "progress": progress,
            "term_logits": term_logits,
            **route_candidates,
        }
        if shuffled_progress is not None and shuffled_term_logits is not None:
            result.update(
                {
                    "skill_shuffle_progress": shuffled_progress,
                    "skill_shuffle_term_logits": shuffled_term_logits,
                    "skill_shuffle_valid": shuffle_valid,
                }
            )
        return result


# Fields added after v3 checkpoints already existed; loaders backfill defaults
# so a pickled pre-probe dataclass cfg still builds the exact same model.
_V3_CFG_BACKFILL = (
    ("encoder_length_token", True),
    ("encoder_arch", "spline"),
    ("autoencoder_mode", "legacy"),
    ("action_gripper_weight", 1.0),
    ("quantizer", "fsq"),
    ("bsq_code_dim", 5),
    ("fsq_entropy", False),
    ("entropy_conf_weight", 0.1),
    ("entropy_conf_ceiling", 0.0),
    ("entropy_div_weight", 0.1),
    ("entropy_inv_temperature", 10.0),
    ("entropy_joint", True),
    ("init_calibration", False),
    ("init_calibration_gain", 1.0),
    ("init_calibration_samples", 0),
    ("pair_loss", "none"),
    ("pair_weight", 0.1),
    ("pair_inv_temperature", 5.0),
    ("route_loss", False),
    ("pair_warmup", False),
    ("pair_warmup_epochs", 0),
    ("pair_ramp_epochs", 0),
    ("boundary_aug_pmax", 0),
    ("boundary_aug_early_start_pmax", -1),
    ("boundary_aug_late_start_pmax", -1),
    ("boundary_aug_early_end_pmax", -1),
    ("boundary_aug_late_end_pmax", -1),
    ("boundary_aug_distribution", "half_normal"),
    ("resnet_image_size", 224),
    ("frame_cache_dir", ""),
    ("reconstructor_start_state", True),
    ("reconstructor_start_state_conditioning", "concat"),
    ("reconstructor_arch", "chunk"),
    ("state_rnn_terminator", False),
    ("terminator_context", "proprio"),
)


def _checkpoint_config(checkpoint: dict[str, Any]) -> SplineFSQAEConfig:
    cfg = checkpoint.get("cfg")
    if cfg is None:
        raise ValueError("FSQ checkpoint has no cfg.")
    if isinstance(cfg, dict):
        # Read checkpoints written before endpoint-weighted reconstruction was removed.
        cfg = dict(cfg)
        legacy_route_loss = cfg.pop("reconstruction_route_loss", None)
        if "route_loss" not in cfg and legacy_route_loss is not None:
            cfg["route_loss"] = bool(legacy_route_loss)
        legacy_lrs = [
            cfg.pop(name)
            for name in ("encoder_lr", "reconstructor_lr", "terminator_lr")
            if name in cfg
        ]
        if "lr" not in cfg and legacy_lrs:
            cfg["lr"] = legacy_lrs[0]
        cfg.pop("weighted_loss", None)
        cfg.pop("weighted_loss_end_weight", None)
        cfg.pop("force_endpoint_sample", None)
        cfg.pop("sample_from_end_window", None)
        cfg.pop("end_window_min_termination", None)
        reconstructor_only = bool(cfg.get("reconstructor_only", False))
        state_rnn = bool(cfg.get("state_rnn_terminator", False))
        cfg.setdefault(
            "terminator_progress",
            not reconstructor_only and not bool(cfg.get("terminator_termination_only", False)),
        )
        cfg.setdefault("terminator_termination", not reconstructor_only)
        cfg.setdefault("terminator_input_space", "state" if state_rnn else "both")
        cfg.setdefault("terminator_model", "rnn" if state_rnn else "default")
        cfg.setdefault("terminator_context", "proprio")
        cfg.setdefault("encoder_grounding_convention", "skill_start_pose_v0")
        legacy_output_mode = (
            "raw_state"
            if cfg.get("encoder_input_mode") == "raw_state"
            else "zero_grounded"
        )
        cfg.setdefault("reconstructor_output_mode", legacy_output_mode)
        cfg.setdefault("reconstructor_min", cfg.get("encoder_min"))
        cfg.setdefault("reconstructor_max", cfg.get("encoder_max"))
        cfg = SplineFSQAEConfig(**cfg)
    if int(getattr(cfg, "format_version", 0)) != FORMAT_VERSION:
        raise ValueError(
            f"Legacy FSQ checkpoint is unsupported: expected format_version={FORMAT_VERSION}, "
            f"got {getattr(cfg, 'format_version', 0)}. Retrain with the transformer-reconstructor FSQ."
        )
    instance_fields = vars(cfg)
    if "terminator_context" not in instance_fields:
        cfg.terminator_context = "proprio"
    if "route_loss" not in instance_fields:
        cfg.route_loss = bool(instance_fields.get("reconstruction_route_loss", False))
    for name, default in _V3_CFG_BACKFILL:
        if not hasattr(cfg, name):
            setattr(cfg, name, default)
    if not hasattr(cfg, "lr"):
        cfg.lr = float(getattr(cfg, "encoder_lr", 3e-4))
    instance_fields = vars(cfg)
    if "encoder_grounding_convention" not in instance_fields:
        cfg.encoder_grounding_convention = "skill_start_pose_v0"
    if "reconstructor_output_mode" not in instance_fields:
        cfg.reconstructor_output_mode = (
            "raw_state"
            if getattr(cfg, "encoder_input_mode", None) == "raw_state"
            else "zero_grounded"
        )
    if "reconstructor_min" not in instance_fields:
        cfg.reconstructor_min = getattr(cfg, "encoder_min", None)
    if "reconstructor_max" not in instance_fields:
        cfg.reconstructor_max = getattr(cfg, "encoder_max", None)
    if (
        getattr(cfg, "encoder_input_mode", None)
        in {"zero_grounded", "start_grounded", "optimal"}
        and cfg.encoder_grounding_convention
        != encoder_grounding_convention(cfg.encoder_input_mode)
    ):
        raise ValueError(
            "FSQ checkpoint grounding convention does not match its encoder input "
            f"mode: mode={cfg.encoder_input_mode!r}, "
            f"convention={cfg.encoder_grounding_convention!r}."
        )
    if "terminator_progress" not in instance_fields:
        cfg.terminator_progress = not bool(getattr(cfg, "reconstructor_only", False)) and not bool(
            getattr(cfg, "terminator_termination_only", False)
        )
    if "terminator_termination" not in instance_fields:
        cfg.terminator_termination = not bool(getattr(cfg, "reconstructor_only", False))
    if "terminator_input_space" not in instance_fields:
        cfg.terminator_input_space = (
            "state" if bool(getattr(cfg, "state_rnn_terminator", False)) else "both"
        )
    if "terminator_model" not in instance_fields:
        cfg.terminator_model = (
            "rnn" if bool(getattr(cfg, "state_rnn_terminator", False)) else "default"
        )
    return cfg


def _terminator_build_config(
    checkpoint: dict[str, Any],
) -> tuple[SplineFSQAEConfig, Any, bool]:
    """Resolve the terminator contract for either joint-v3 or FSQ-original.

    FSQ-original and reconstructor-only v3 checkpoints intentionally contain no
    terminator tensors.  For them, derive only the shared skill/state contract
    and construct a fresh terminator.  Joint-v3 checkpoints retain their
    historical warm start when the component is actually present.
    """
    source_cfg = checkpoint.get("cfg")
    try:
        from FSQ_original import FSQOriginalConfig  # noqa: PLC0415
    except ImportError:
        FSQOriginalConfig = ()  # type: ignore[assignment,misc]

    if isinstance(source_cfg, FSQOriginalConfig):
        if str(getattr(source_cfg, "quantizer", "fsq")) != "fsq":
            raise ValueError(
                "Fresh terminator construction supports FSQ-original checkpoints "
                "only; BSQ uses a different code-coordinate contract."
            )
        if str(getattr(source_cfg, "encoder_arch", "spline")) != "spline":
            raise ValueError(
                "Fresh terminator construction requires an FSQ-original spline "
                "encoder with raw state bounds."
            )
        # The terminator normalizes ABSOLUTE proprioception, so encoder_min/max
        # may stand in for the v3 state_min/max only when the encoder consumed
        # raw states. Under zero_grounded/optimal their XYZ bounds describe
        # mean-centered coordinates, which would silently mis-scale every state.
        encoder_input_mode = str(getattr(source_cfg, "encoder_input_mode", "zero_grounded"))
        if encoder_input_mode != "raw_state":
            raise ValueError(
                "Fresh terminator construction requires an FSQ-original checkpoint "
                f"trained with encoder_input_mode='raw_state', got {encoder_input_mode!r}: "
                "its encoder_min/max contain mean-centered XYZ and cannot normalize the "
                "terminator's absolute state input."
            )
        state_min = getattr(source_cfg, "encoder_min", None)
        state_max = getattr(source_cfg, "encoder_max", None)
        if state_min is None or state_max is None:
            raise ValueError(
                "FSQ-original checkpoint has no encoder_min/encoder_max values "
                "from which to derive terminator state normalization."
            )
        state_min = np.asarray(state_min, dtype=np.float32)
        state_max = np.asarray(state_max, dtype=np.float32)
        if state_min.ndim != 1 or state_max.shape != state_min.shape:
            raise ValueError(
                "FSQ-original state bounds must be matching 1-D arrays, got "
                f"{state_min.shape} and {state_max.shape}."
            )
        cfg = SplineFSQAEConfig(
            action_dim=int(source_cfg.action_dim),
            enc_dim=int(source_cfg.enc_dim),
            state_dim=int(state_min.shape[0]),
            n_control=int(source_cfg.n_control),
            spline_degree=int(source_cfg.spline_degree),
            encoder_input_mode=str(source_cfg.encoder_input_mode),
            hidden_dim=int(source_cfg.hidden_dim),
            fsq_levels=[int(level) for level in source_cfg.fsq_levels],
            num_layers=int(source_cfg.num_layers),
            dropout=float(source_cfg.dropout),
            length_min=float(source_cfg.length_min),
            length_max=float(source_cfg.length_max),
            terminator_arch="small",
            vision_backbone="dino",
            freeze_vision_encoder=True,
            dino_model_path=_DEFAULT_IMAGE_MODEL,
            dino_image_size=224,
            siglip_image_size=224,
            resnet_image_size=224,
            image_encoder_layers=int(source_cfg.num_layers),
            image_encoder_heads=int(source_cfg.num_heads),
            skill_cond_mode="broadcast",
            terminator_context="proprio",
            encoder_min=state_min,
            encoder_max=state_max,
            state_min=state_min,
            state_max=state_max,
        )
        return cfg, source_cfg, False

    cfg = _checkpoint_config(checkpoint)
    model_state = checkpoint.get("model_state", {})
    has_terminator_weights = any(
        key.startswith("terminator.") for key in model_state
    )
    return cfg, cfg, has_terminator_weights


def _new_fsq_terminator(
    terminator_cls: type[FSQQueryTerminator],
    cfg: SplineFSQAEConfig,
    *,
    dino_model_path: str | None,
    termination_only: bool | None = None,
) -> FSQQueryTerminator:
    context_is_action = cfg.terminator_context == "prev_action"
    kwargs = {
        "state_dim": cfg.action_dim if context_is_action else cfg.state_dim,
        "fsq_levels": cfg.fsq_levels,
        "hidden_dim": cfg.hidden_dim,
        "n_layers": cfg.image_encoder_layers,
        "n_heads": cfg.image_encoder_heads,
        "dropout": 0.0,
        "arch": cfg.terminator_arch,
        "vision_backbone": cfg.vision_backbone,
        "freeze_vision_encoder": cfg.freeze_vision_encoder,
        "dino_model_path": dino_model_path or cfg.dino_model_path,
        "dino_image_size": cfg.dino_image_size,
        "siglip_image_size": cfg.siglip_image_size,
        "resnet_image_size": cfg.resnet_image_size,
        "skill_cond_mode": cfg.skill_cond_mode,
        "state_min": cfg.action_q01 if context_is_action else cfg.state_min,
        "state_max": cfg.action_q99 if context_is_action else cfg.state_max,
        "context_mode": cfg.terminator_context,
        "context_gripper_weight": cfg.action_gripper_weight,
    }
    # Every terminator variant subclasses FSQQueryTerminator and forwards **kwargs
    # to it, so the image-only and wrist-only models honor this too. An explicit
    # argument wins over the checkpoint contract; None keeps the cfg default.
    kwargs["termination_only"] = (
        bool(getattr(cfg, "terminator_termination_only", False))
        if termination_only is None
        else bool(termination_only)
    )
    return terminator_cls(**kwargs)


def _load_prefixed(module: nn.Module, state: dict[str, Tensor], prefix: str) -> None:
    selected = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    if not selected:
        raise ValueError(f"FSQ checkpoint contains no '{prefix}*' tensors.")
    missing, unexpected = module.load_state_dict(selected, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Component load failed for {prefix}: missing={missing}, unexpected={unexpected}")


def load_fsq_encoder(path: str | Path, device: str | torch.device = "cpu") -> tuple[SplineFSQEncoder, SplineFSQAEConfig]:
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = _checkpoint_config(checkpoint)
    if cfg.encoder_arch == "action_seq":
        encoder = ActionSeqEncoder(
            action_dim=cfg.action_dim,
            hidden_dim=cfg.hidden_dim,
            fsq_levels=cfg.fsq_levels,
            n_layers=cfg.num_layers,
            n_heads=cfg.image_encoder_heads,
            dropout=0.0,
            raw_actions=(
                is_action_sequence_reconstructor_arch(cfg.reconstructor_arch)
                and cfg.autoencoder_mode != "norm_action"
            ),
            action_q01=cfg.action_q01,
            action_q99=cfg.action_q99,
            clip_normalized_actions=cfg.autoencoder_mode == "norm_action",
            action_gripper_weight=cfg.action_gripper_weight,
        )
        if cfg.quantizer == "bsq":
            encoder.fsq = BSQ(cfg.bsq_code_dim)
        _load_prefixed(encoder, checkpoint["model_state"], "encoder.")
        encoder.cfg = cfg
        encoder.to(device).eval()
        return encoder, cfg
    encoder_cls = SplineFSQEncoder if cfg.encoder_length_token else LengthFreeSplineFSQEncoder
    encoder = encoder_cls(
        enc_dim=cfg.enc_dim,
        n_control=cfg.n_control,
        spline_degree=cfg.spline_degree,
        hidden_dim=cfg.hidden_dim,
        fsq_levels=cfg.fsq_levels,
        n_layers=cfg.num_layers,
        n_heads=cfg.image_encoder_heads,
        dropout=0.0,
        length_min=cfg.length_min,
        length_max=cfg.length_max,
        encoder_min=cfg.encoder_min,
        encoder_max=cfg.encoder_max,
        encoder_input_mode=getattr(cfg, "encoder_input_mode", "zero_grounded"),
        encoder_start_min=getattr(cfg, "encoder_start_min", None),
        encoder_start_max=getattr(cfg, "encoder_start_max", None),
        gripper_weight=cfg.action_gripper_weight,
    )
    if cfg.quantizer == "bsq":
        encoder.fsq = BSQ(cfg.bsq_code_dim)
    _load_prefixed(encoder, checkpoint["model_state"], "encoder.")
    encoder.to(device).eval()
    return encoder, cfg


def load_fsq_terminator(
    path: str | Path,
    device: str | torch.device = "cpu",
    dino_model_path: str | None = None,
) -> tuple[nn.Module, SplineFSQAEConfig]:
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = _checkpoint_config(checkpoint)
    if cfg.state_rnn_terminator:
        context_is_action = cfg.terminator_context == "prev_action"
        terminator = FSQStateRNNTerminator(
            state_dim=cfg.action_dim if context_is_action else cfg.state_dim,
            fsq_levels=cfg.fsq_levels,
            state_q01=cfg.action_q01 if context_is_action else cfg.state_q01,
            state_q99=cfg.action_q99 if context_is_action else cfg.state_q99,
            context_mode=cfg.terminator_context,
            termination_only=cfg.terminator_termination_only,
        )
    elif cfg.terminator_input_space == "state":
        context_is_action = cfg.terminator_context == "prev_action"
        terminator = FSQStateMLPTerminator(
            state_dim=cfg.action_dim if context_is_action else cfg.state_dim,
            fsq_levels=cfg.fsq_levels,
            state_q01=cfg.action_q01 if context_is_action else cfg.state_q01,
            state_q99=cfg.action_q99 if context_is_action else cfg.state_q99,
            context_mode=cfg.terminator_context,
            termination_only=cfg.terminator_termination_only,
        )
    else:
        terminator = _new_fsq_terminator(
            FSQImageOnlyQueryTerminator
            if cfg.terminator_input_space == "image"
            else FSQQueryTerminator,
            cfg,
            dino_model_path=dino_model_path,
        )
    _load_prefixed(terminator, checkpoint["model_state"], "terminator.")
    terminator.to(device).eval()
    return terminator, cfg


def build_trainable_fsq_terminator(
    path: str | Path,
    device: str | torch.device = "cpu",
    dino_model_path: str | None = None,
    termination_only: bool | None = None,
    context: str | None = None,
    default_arch: str | None = None,
    vision_backbone: str | None = None,
    freeze_vision_encoder: bool | None = None,
) -> tuple[FSQQueryTerminator, Any]:
    """Build the state+image terminator used by standalone training.

    Joint-v3 FSQ checkpoints warm-start their saved ``terminator.*`` tensors
    only when the requested contract is identical. A changed context,
    architecture, backbone, or vision-freeze setting intentionally creates a
    fresh terminator while retaining the FSQ/state normalization contract.
    """
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg, source_cfg, has_terminator_weights = _terminator_build_config(checkpoint)
    cfg = copy.copy(cfg)
    source_is_state_image = not (
        getattr(cfg, "state_rnn_terminator", False)
        or getattr(cfg, "terminator_input_space", "both") != "both"
    )

    requested = {
        "terminator_context": context,
        "terminator_arch": default_arch,
        "vision_backbone": vision_backbone,
        "freeze_vision_encoder": freeze_vision_encoder,
    }
    mismatches = []
    if has_terminator_weights and not source_is_state_image:
        mismatches.append(
            "variant: checkpoint is not the requested state+image terminator"
        )
    if has_terminator_weights:
        mismatches.extend(
            f"{field}: checkpoint={getattr(cfg, field)!r}, requested={value!r}"
            for field, value in requested.items()
            if value is not None
            and getattr(cfg, field) != (
                bool(value) if field == "freeze_vision_encoder" else value
            )
        )
    for field, value in requested.items():
        if value is not None:
            setattr(
                cfg,
                field,
                bool(value) if field == "freeze_vision_encoder" else value,
            )
    warm_start = has_terminator_weights and not mismatches
    terminator = _new_fsq_terminator(
        FSQQueryTerminator,
        cfg,
        dino_model_path=dino_model_path,
        termination_only=termination_only,
    )
    if warm_start:
        _load_prefixed(terminator, checkpoint["model_state"], "terminator.")
        log.info(
            "Warm-started state+image terminator from matching FSQ contract: %s.",
            path,
        )
    else:
        reason = "; ".join(mismatches) if mismatches else "no terminator tensors"
        log.info(
            "Initializing a fresh state+image terminator from %s (%s).",
            path,
            reason,
        )
    terminator.to(device).eval()
    return terminator, source_cfg


def build_fsq_image_only_terminator(
    path: str | Path,
    device: str | torch.device = "cpu",
    dino_model_path: str | None = None,
    termination_only: bool | None = None,
) -> tuple[FSQImageOnlyQueryTerminator, SplineFSQAEConfig]:
    """Build a fresh image-only terminator without loading any FSQ model tensor.

    The FSQ checkpoint contributes only the architecture/FSQ contract. The
    selected DINO, SigLIP, or ResNet backbone is initialized normally; every
    image-only terminator-specific tensor keeps its constructor initialization.
    """
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg, source_cfg, _ = _terminator_build_config(checkpoint)
    terminator = _new_fsq_terminator(
        FSQImageOnlyQueryTerminator,
        cfg,
        dino_model_path=dino_model_path,
        termination_only=termination_only,
    )
    terminator.to(device).eval()
    return terminator, source_cfg


def build_fsq_wrist_only_terminator(
    path: str | Path,
    device: str | torch.device = "cpu",
    dino_model_path: str | None = None,
    termination_only: bool | None = None,
) -> tuple[FSQWristOnlyQueryTerminator, SplineFSQAEConfig]:
    """Build a fresh wrist-only terminator without loading any FSQ tensor."""
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg, source_cfg, _ = _terminator_build_config(checkpoint)
    terminator = _new_fsq_terminator(
        FSQWristOnlyQueryTerminator,
        cfg,
        dino_model_path=dino_model_path,
        termination_only=termination_only,
    )
    terminator.to(device).eval()
    return terminator, source_cfg


def load_fsq_reconstructor_state(path: str | Path) -> tuple[dict[str, Tensor], SplineFSQAEConfig]:
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = _checkpoint_config(checkpoint)
    state = {
        k[len("reconstructor.") :]: v
        for k, v in checkpoint["model_state"].items()
        if k.startswith("reconstructor.")
    }
    if not state:
        raise ValueError("FSQ checkpoint has no reconstructor tensors.")
    return state, cfg


def load_fsq_model(
    path: str | Path,
    device: str | torch.device = "cpu",
    dino_model_path: str | None = None,
) -> tuple[SplineFSQAE, SplineFSQAEConfig]:
    """Load all v3 components. Reserved for joint FSQ evaluation/training tools."""
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = _checkpoint_config(checkpoint)
    if dino_model_path and cfg.vision_backbone == "dino":
        cfg.dino_model_path = str(dino_model_path)
    model = SplineFSQAE(cfg)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    dev = torch.device(device)
    model.to(dev).eval()
    return model, cfg


# -----------------------------------------------------------------------------
# B trajectories x M sampled timesteps dataset
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryAugmentationContext:
    """Raw contiguous frames available around one clean skill segment."""

    trajectory: np.ndarray
    start: int
    end: int


def resolve_boundary_augmentation_pmaxes(
    pmax: int,
    *,
    early_start_pmax: int | None = None,
    late_start_pmax: int | None = None,
    early_end_pmax: int | None = None,
    late_end_pmax: int | None = None,
) -> tuple[int, int, int, int]:
    """Resolve four directional windows, retaining the legacy shared fallback.

    The returned order is early-start, late-start, early-end, late-end. A
    directional value of ``None`` or ``-1`` inherits ``pmax``; zero disables
    just that direction.
    """
    legacy = int(pmax)
    if legacy < 0:
        raise ValueError(f"Boundary augmentation pmax must be non-negative, got {legacy}.")

    resolved: list[int] = []
    for name, value in (
        ("early_start_pmax", early_start_pmax),
        ("late_start_pmax", late_start_pmax),
        ("early_end_pmax", early_end_pmax),
        ("late_end_pmax", late_end_pmax),
    ):
        selected = legacy if value is None or int(value) == -1 else int(value)
        if selected < 0:
            raise ValueError(
                f"Boundary augmentation {name} must be non-negative or -1, got {value}."
            )
        resolved.append(selected)
    return resolved[0], resolved[1], resolved[2], resolved[3]


def build_boundary_augmentation_contexts(
    segments: list[np.ndarray],
    metadata: list[dict[str, Any]],
    pmax: int,
) -> list[BoundaryAugmentationContext]:
    """Attach up to ``pmax`` frames from contiguous neighbour skill files.

    The saved FSQ input files contain only the detected segment. Extension is
    therefore allowed only when another segment in the same supplied split is
    exactly adjacent in the original episode. This prevents train examples
    from borrowing trajectory frames from the validation split.
    """
    if len(segments) != len(metadata):
        raise ValueError("Boundary context segments and metadata lengths do not match.")
    if pmax < 0:
        raise ValueError(f"Boundary context pmax must be non-negative, got {pmax}.")

    previous, following = _contiguous_skill_neighbor_maps(metadata)

    contexts: list[BoundaryAugmentationContext] = []
    for i, segment in enumerate(segments):
        segment = np.asarray(segment, dtype=np.float32)
        expected_length = int(metadata[i]["frame_end"]) - int(metadata[i]["frame_start"])
        if expected_length != len(segment):
            raise ValueError(
                f"Skill {i} frame range has length {expected_length}, "
                f"but its trajectory has length {len(segment)}."
            )
        prefix = (
            np.asarray(segments[previous[i]], dtype=np.float32)[-pmax:]
            if pmax > 0 and i in previous
            else segment[:0]
        )
        suffix = (
            np.asarray(segments[following[i]], dtype=np.float32)[:pmax]
            if pmax > 0 and i in following
            else segment[:0]
        )
        if prefix.shape[1:] != segment.shape[1:] or suffix.shape[1:] != segment.shape[1:]:
            raise ValueError(f"Neighbour trajectory dimensions do not match for skill {i}.")
        trajectory = np.concatenate((prefix, segment, suffix), axis=0)
        start = len(prefix)
        contexts.append(
            BoundaryAugmentationContext(
                trajectory=trajectory,
                start=start,
                end=start + len(segment),
            )
        )
    return contexts


def _contiguous_skill_neighbor_maps(
    metadata: list[dict[str, Any]],
) -> tuple[dict[int, int], dict[int, int]]:
    """Return local previous/following indices for exactly adjacent skills."""
    by_episode: dict[tuple[int, int], list[int]] = {}
    for i, item in enumerate(metadata):
        key = (int(item.get("task_id", -1)), int(item["episode_id"]))
        by_episode.setdefault(key, []).append(i)

    previous: dict[int, int] = {}
    following: dict[int, int] = {}
    for ids in by_episode.values():
        ids.sort(
            key=lambda i: (
                int(metadata[i]["frame_start"]),
                int(metadata[i]["skill_index"]),
            )
        )
        for left, right in zip(ids[:-1], ids[1:], strict=True):
            if int(metadata[left]["frame_end"]) == int(metadata[right]["frame_start"]):
                following[left] = right
                previous[right] = left
    return previous, following


def build_adjacent_skill_indices(
    metadata: list[dict[str, Any]],
) -> list[tuple[int, ...]]:
    """List each skill's contiguous in-episode neighbours in local index space.

    Interior skills have ``(previous, following)``; episode-edge skills have
    one entry, and true single-skill episodes have none.  The dataset samples
    uniformly from the returned tuple, so no trajectory-distance target is
    introduced by the negative selection itself.
    """
    previous, following = _contiguous_skill_neighbor_maps(metadata)
    return [
        tuple(
            neighbour
            for neighbour in (previous.get(i), following.get(i))
            if neighbour is not None
        )
        for i in range(len(metadata))
    ]


def build_skill_initial_previous_actions(
    actions: list[np.ndarray],
    metadata: list[dict[str, Any]],
    action_dim: int,
) -> list[np.ndarray | None]:
    """Return the real action immediately preceding each saved skill.

    Saved skill files contain only their own action interval. For every
    non-episode-start skill, action[t-1] is therefore the last action of the
    exactly contiguous preceding skill. Only a true episode start receives
    the all-zero BOS token.
    """
    if len(actions) != len(metadata):
        raise ValueError("Previous-action source and metadata lengths do not match.")
    previous, _ = _contiguous_skill_neighbor_maps(metadata)
    initial: list[np.ndarray | None] = []
    for i, (action, item) in enumerate(zip(actions, metadata, strict=True)):
        frame_start = int(item["frame_start"])
        if frame_start == 0:
            initial.append(None)
            continue
        if i not in previous:
            raise ValueError(
                "Cannot recover action[t-1] for non-episode-start skill "
                f"{i} (task={item.get('task_id', -1)}, "
                f"episode={item.get('episode_id', -1)}, frame_start={frame_start}). "
                "The supplied skill set must include the exactly contiguous "
                "preceding skill."
            )
        source = np.asarray(actions[previous[i]], dtype=np.float32)
        if len(source) == 0 or source.shape[-1] < action_dim:
            raise ValueError(
                f"Invalid preceding action sequence for skill {i}: shape={source.shape}."
            )
        initial.append(source[-1, :action_dim].copy())
    return initial


def sample_boundary_augmented_segment(
    context: BoundaryAugmentationContext,
    *,
    pmax: int,
    early_start_pmax: int | None = None,
    late_start_pmax: int | None = None,
    early_end_pmax: int | None = None,
    late_end_pmax: int | None = None,
    min_length: int,
    distribution: str,
    rng: np.random.Generator | None = None,
    max_attempts: int = 32,
) -> tuple[np.ndarray, int, int]:
    """Move one selected boundary/direction and return trajectory/boundary/offset.

    Boundary is encoded as 0=start and 1=end. The signed offset is applied to
    that boundary in original frame coordinates. Enabled boundaries are chosen
    uniformly, followed by an enabled direction for that boundary. Invalid
    draws retry the direction and magnitude while retaining the selected
    boundary. If no valid nonzero draw is found, an identity augmentation is
    returned.
    """
    if min_length < 1:
        raise ValueError(f"Boundary augmentation min_length must be positive, got {min_length}.")
    early_start, late_start, early_end, late_end = resolve_boundary_augmentation_pmaxes(
        pmax,
        early_start_pmax=early_start_pmax,
        late_start_pmax=late_start_pmax,
        early_end_pmax=early_end_pmax,
        late_end_pmax=late_end_pmax,
    )
    windows = {
        0: ((-1, early_start), (1, late_start)),
        1: ((-1, early_end), (1, late_end)),
    }
    enabled_boundaries = [
        boundary
        for boundary, directions in windows.items()
        if any(limit > 0 for _, limit in directions)
    ]
    if not enabled_boundaries:
        raise ValueError("Boundary augmentation requires at least one positive directional pmax.")
    distribution = normalize_jitter_distribution(distribution)
    r = np.random if rng is None else rng
    if len(enabled_boundaries) == 1:
        boundary = enabled_boundaries[0]
    else:
        boundary = enabled_boundaries[0] if r.random() < 0.5 else enabled_boundaries[1]
    enabled_directions = [(sign, limit) for sign, limit in windows[boundary] if limit > 0]
    for _ in range(max_attempts):
        if len(enabled_directions) == 1:
            sign, directional_pmax = enabled_directions[0]
        else:
            sign, directional_pmax = (
                enabled_directions[0] if r.random() < 0.5 else enabled_directions[1]
            )
        offset = sign * sample_p(directional_pmax, rng=rng, distribution=distribution)
        if offset == 0:
            return context.trajectory[context.start : context.end].copy(), boundary, 0
        start = context.start + offset if boundary == 0 else context.start
        end = context.end + offset if boundary == 1 else context.end
        if start < 0 or end > len(context.trajectory) or end - start < min_length:
            continue
        return context.trajectory[start:end].copy(), boundary, offset
    return context.trajectory[context.start : context.end].copy(), boundary, 0


class FSQTrajectoryDataset(Dataset):
    """One trajectory plus decoder/terminator targets for the selected inputs.

    Image access is lazy and worker-local. A completed random-access RGB cache
    avoids video decoding; the live decoder remains the blank-cache fallback.
    """

    def __init__(
        self,
        segments: list[np.ndarray],
        states: list[np.ndarray],
        actions: list[np.ndarray],
        metadata: list[dict[str, Any]],
        raw_dataset_dir: str | Path,
        cfg: SplineFSQAEConfig,
        *,
        training: bool,
        initial_previous_actions: list[np.ndarray | None] | None = None,
        boundary_contexts: list[BoundaryAugmentationContext] | None = None,
        adjacent_skill_indices: list[tuple[int, ...]] | None = None,
    ):
        if not (len(segments) == len(states) == len(actions) == len(metadata)):
            raise ValueError("FSQ dataset component lengths do not match.")
        self.cfg = cfg
        self.training = bool(training)
        self.pair_augmentation = self.training and cfg.pair_loss != "none"
        if self.pair_augmentation:
            if boundary_contexts is None or len(boundary_contexts) != len(segments):
                raise ValueError(
                    "Training with FSQ pair loss requires one boundary context per segment."
                )
            self.boundary_contexts = boundary_contexts
        else:
            self.boundary_contexts = None
        self.contrastive_pairs = self.training and cfg.pair_loss == "contrastive"
        if self.contrastive_pairs:
            if (
                adjacent_skill_indices is None
                or len(adjacent_skill_indices) != len(segments)
            ):
                raise ValueError(
                    "Training with contrastive pair loss requires adjacent-skill indices."
                )
            self.adjacent_skill_indices = adjacent_skill_indices
        else:
            self.adjacent_skill_indices = None
        self.sampled_timestep_required = (
            (not cfg.terminator_only and cfg.reconstructor_arch == "chunk")
            or (not cfg.reconstructor_only and not cfg.state_rnn_terminator)
        )
        self.samples_per_skill = (
            int(cfg.samples_per_skill) if self.sampled_timestep_required else 1
        )
        if self.samples_per_skill < 1:
            raise ValueError("samples_per_skill must be >=1.")
        self.ctrl: list[np.ndarray] = []
        self.reconstructor_ctrl: list[np.ndarray] | None = (
            [] if cfg.reconstructor_arch == "oneshot" and not cfg.terminator_only else None
        )
        self.start_poses: list[np.ndarray] | None = (
            []
            if cfg.encoder_arch == "spline" and cfg.encoder_input_mode == "optimal"
            else None
        )
        self.lengths: list[int] = []
        self.states = [np.asarray(x, dtype=np.float32) for x in states]
        self.actions = [np.asarray(x, dtype=np.float32) for x in actions]
        self.metadata = metadata
        self.raw_dataset_dir = str(raw_dataset_dir)
        self._raw_dataset = None
        self._uses_visual_samples = (
            not cfg.reconstructor_only
            and not cfg.state_rnn_terminator
            and cfg.terminator_input_space in {"image", "both"}
        )
        self.frame_cache_dir = str(getattr(cfg, "frame_cache_dir", "") or "")
        self._frame_cache = None
        if self.frame_cache_dir and self._uses_visual_samples:
            from fsq_frame_cache import RGBFrameCache

            # Validate the source fingerprint once in the dataset-owning
            # process. DataLoader workers inherit an empty lazy mmap table and
            # open only the video arrays they actually touch.
            self._frame_cache = RGBFrameCache(
                self.frame_cache_dir,
                self.raw_dataset_dir,
                verify_source=True,
            )

        enc_min = np.asarray(cfg.encoder_min, dtype=np.float32)
        enc_max = np.asarray(cfg.encoder_max, dtype=np.float32)
        self.encoder_min = enc_min
        self.encoder_max = enc_max
        recon_min = recon_max = None
        if self.reconstructor_ctrl is not None:
            recon_min = np.asarray(cfg.reconstructor_min, dtype=np.float32)
            recon_max = np.asarray(cfg.reconstructor_max, dtype=np.float32)
        start_min = start_max = None
        if self.start_poses is not None:
            start_min = np.asarray(cfg.encoder_start_min, dtype=np.float32)
            start_max = np.asarray(cfg.encoder_start_max, dtype=np.float32)
        self.encoder_start_min = start_min
        self.encoder_start_max = start_max
        for i, segment in enumerate(segments):
            ctrl, length = spline_encode(
                segment,
                cfg.n_control,
                cfg.spline_degree,
                input_mode=cfg.encoder_input_mode,
            )
            if len(self.states[i]) < length or len(self.actions[i]) < length:
                raise ValueError(
                    f"Skill {i} is shorter than metadata length {length}: "
                    f"states={len(self.states[i])}, actions={len(self.actions[i])}"
                )
            if self._uses_visual_samples and "dataset_from_index" not in metadata[i]:
                raise ValueError(f"Skill {i} metadata has no dataset_from_index.")
            normalized_ctrl = (
                2.0 * (ctrl - enc_min) / (enc_max - enc_min + 1e-8) - 1.0
            ).astype(np.float32)
            self.ctrl.append(
                scale_gripper_features(
                    normalized_ctrl,
                    gripper_weight=cfg.action_gripper_weight,
                    gripper_dims=N_GRIPPER_DIMS,
                )
            )
            if self.reconstructor_ctrl is not None:
                recon_ctrl, recon_length = spline_encode(
                    segment,
                    cfg.n_control,
                    cfg.spline_degree,
                    input_mode=cfg.reconstructor_output_mode,
                )
                if recon_length != length:
                    raise RuntimeError(
                        f"Encoder/reconstructor spline lengths differ: {length} != {recon_length}."
                    )
                normalized_recon_ctrl = (
                    2.0 * (recon_ctrl - recon_min)
                    / (recon_max - recon_min + 1e-8)
                    - 1.0
                ).astype(np.float32)
                self.reconstructor_ctrl.append(
                    scale_gripper_features(
                        normalized_recon_ctrl,
                        gripper_weight=cfg.action_gripper_weight,
                        gripper_dims=N_GRIPPER_DIMS,
                    )
                )
            if self.start_poses is not None:
                start_pose = encoder_grounding_position(segment)
                self.start_poses.append(
                    (2.0 * (start_pose - start_min) / (start_max - start_min + 1e-8) - 1.0).astype(np.float32)
                )
            self.lengths.append(length)

        self.state_q01 = np.asarray(cfg.state_q01, dtype=np.float32)
        self.state_q99 = np.asarray(cfg.state_q99, dtype=np.float32)
        self.action_q01 = np.asarray(cfg.action_q01, dtype=np.float32)
        self.action_q99 = np.asarray(cfg.action_q99, dtype=np.float32)
        if initial_previous_actions is None:
            if not cfg.reconstructor_only and cfg.terminator_context == "prev_action":
                initial_previous_actions = build_skill_initial_previous_actions(
                    self.actions, self.metadata, cfg.action_dim
                )
            else:
                initial_previous_actions = [None] * len(self.actions)
        if len(initial_previous_actions) != len(self.actions):
            raise ValueError(
                "initial_previous_actions must contain one action per skill."
            )
        self.previous_actions = [
            self._normalized_previous_action_values(action, initial)
            for action, initial in zip(
                self.actions, initial_previous_actions, strict=True
            )
        ]
        self.terminator_context_sequences: list[Tensor] | None = None
        self.terminator_progress_targets: list[Tensor] | None = None
        self.terminator_end_targets: list[Tensor] | None = None
        if cfg.state_rnn_terminator and not cfg.reconstructor_only:
            # Cache every compact full-skill context once. Inputs are left-padded
            # because StateSkillRNNTerminator compacts the valid suffix before
            # pack_padded_sequence; outputs/targets are left-aligned.
            max_steps = int(round(cfg.length_max))
            self.terminator_context_sequences = []
            self.terminator_progress_targets = []
            self.terminator_end_targets = []
            for state, previous_action, length in zip(
                self.states, self.previous_actions, self.lengths, strict=True
            ):
                if length > max_steps:
                    raise ValueError(
                        f"Full-skill state sequence length {length} exceeds "
                        f"length_max={max_steps}."
                    )
                context = (
                    previous_action[:length]
                    if cfg.terminator_context == "prev_action"
                    else state[:length, : cfg.state_dim]
                )
                padded_context = torch.zeros(
                    max_steps, context.shape[-1], dtype=torch.float32
                )
                padded_context[-length:] = torch.from_numpy(context.copy())
                positions = np.arange(length, dtype=np.int64)
                progress = torch.zeros(max_steps, dtype=torch.float32)
                progress[:length] = torch.from_numpy(
                    positions.astype(np.float32) / max(length - 1, 1)
                )
                termination = torch.zeros(max_steps, dtype=torch.float32)
                termination[:length] = torch.from_numpy(
                    self._termination_targets(length, positions)
                )
                self.terminator_context_sequences.append(padded_context)
                self.terminator_progress_targets.append(progress)
                self.terminator_end_targets.append(termination)

    def __len__(self) -> int:
        return len(self.ctrl)

    @staticmethod
    def _quantile_norm(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        return (2.0 * (x - lo) / (hi - lo + 1e-8) - 1.0).astype(np.float32)

    def encoder_action_values(self, index: int) -> np.ndarray:
        """Action values seen by the action-sequence encoder.

        The matched action_seq reconstructor defines the new raw-action
        contract. Historical action_seq+chunk models retain their q01/q99
        encoder convention so existing checkpoints remain reproducible.
        """
        return self._encoder_action_values(self.actions[index])

    def _encoder_action_values(self, action: np.ndarray) -> np.ndarray:
        """Apply the checkpoint's action-sequence value convention."""
        if is_action_sequence_reconstructor_arch(self.cfg.reconstructor_arch):
            if self.cfg.autoencoder_mode == "norm_action":
                return normalize_action_sequence(
                    action,
                    self.action_q01,
                    self.action_q99,
                    gripper_weight=self.cfg.action_gripper_weight,
                    clip=True,
                )
            return scale_gripper_features(
                action,
                gripper_weight=self.cfg.action_gripper_weight,
                gripper_dims=1,
            )
        return self._quantile_norm(action, self.action_q01, self.action_q99)

    def _normalized_previous_action_values(
        self,
        action: np.ndarray,
        initial_previous_action: np.ndarray | None,
    ) -> np.ndarray:
        """Right-shift actions, preserving the real cross-skill predecessor."""
        previous = np.zeros(
            (len(action), self.cfg.action_dim), dtype=np.float32
        )
        if len(action) and initial_previous_action is not None:
            raw_initial = np.asarray(initial_previous_action, dtype=np.float32)
            if raw_initial.shape != (self.cfg.action_dim,):
                raise ValueError(
                    "Initial previous action must have shape "
                    f"({self.cfg.action_dim},), got {raw_initial.shape}."
                )
            previous[0] = normalize_action_sequence(
                raw_initial[None],
                self.action_q01,
                self.action_q99,
                gripper_weight=self.cfg.action_gripper_weight,
                clip=True,
            )[0]
        if len(action) > 1:
            previous[1:] = normalize_action_sequence(
                action[:-1, : self.cfg.action_dim],
                self.action_q01,
                self.action_q99,
                gripper_weight=self.cfg.action_gripper_weight,
                clip=True,
            )
        return previous

    def _padded_encoder_action_sequence(self, action: np.ndarray) -> Tensor:
        """Pad one clean/positive/negative encoder sequence for collation.

        Boundary augmentation moves only one boundary, so a positive can grow
        by at most the early-start or late-end window. Keeping that extra room
        in every training item lets raw action pairs remain variable-length;
        the encoder masks all padding and slices each batch to its actual max.
        """
        extension = (
            max(
                self.cfg.boundary_aug_early_start_pmax,
                self.cfg.boundary_aug_late_end_pmax,
            )
            if self.cfg.pair_loss != "none"
            else 0
        )
        pad_steps = int(round(self.cfg.length_max)) + int(extension)
        values = self._encoder_action_values(action)
        if len(values) > pad_steps:
            raise ValueError(
                f"Action sequence length {len(values)} exceeds padded limit={pad_steps}."
            )
        padded = np.zeros((pad_steps, values.shape[-1]), dtype=np.float32)
        padded[: len(values)] = values
        return torch.from_numpy(padded)

    def _termination_targets(self, length: int, sample: np.ndarray) -> np.ndarray:
        distance_to_end = (length - 1 - sample).astype(np.float32)
        if self.cfg.end_target_sigma > 0:
            return np.exp(
                -(distance_to_end ** 2) / (2.0 * self.cfg.end_target_sigma ** 2)
            ).astype(np.float32)
        return (distance_to_end == 0).astype(np.float32)

    def _sample_indices(self, length: int) -> np.ndarray:
        """Uniformly sample M training timesteps; validation uses a deterministic linspace."""
        if not getattr(self, "sampled_timestep_required", True):
            # Oneshot reconstruction and full-sequence RNN termination consume
            # no sampled timestep. Keep one placeholder row for the shared
            # batch contract without invoking the RNG.
            return np.zeros(1, dtype=np.int64)
        m = self.samples_per_skill
        if length < 1:
            raise ValueError(f"Skill length must be positive, got {length}.")
        if self.training:
            return np.sort(
                np.random.choice(length, size=m, replace=length < m).astype(np.int64)
            )
        return np.rint(np.linspace(0, length - 1, m)).astype(np.int64)

    def _action_chunks(self, action: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Stage-1 target: normalize first, then hold beyond the current skill boundary."""
        normalized = self._quantile_norm(action, self.action_q01, self.action_q99)
        m, k, max_dim = len(indices), self.cfg.chunk_size, self.cfg.max_action_dim
        out = np.zeros((m, k, max_dim), dtype=np.float32)
        real_dim = self.cfg.action_dim
        gripper = real_dim - 1
        for row, start in enumerate(indices.tolist()):
            valid = min(k, len(normalized) - start)
            out[row, :valid, :real_dim] = normalized[start : start + valid, :real_dim]
            if valid < k:
                out[row, valid:, gripper] = out[row, valid - 1, gripper]
        return out

    def _get_raw_dataset(self):
        if self._raw_dataset is None:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            root = Path(self.raw_dataset_dir)
            self._raw_dataset = LeRobotDataset(
                repo_id=f"local/{root.name}",
                root=root,
                video_keys_to_load=[
                    "observation.images.image",
                    "observation.images.wrist_image",
                ],
            )
        return self._raw_dataset

    def _sample_images(self, index: int, sample: np.ndarray) -> tuple[Tensor, Tensor]:
        """Load all M frames for each camera in one batched request.

        Calling ``LeRobotDataset.__getitem__`` once per selected timestep forces
        PyAV/TorchCodec to seek/decode separately for every frame.  The M
        timesteps belong to one skill/episode and are sorted, so one request per
        camera can share the seek and sequential decode work.
        """
        dataset = self._get_raw_dataset()
        start = int(self.metadata[index]["dataset_from_index"]) + int(self.metadata[index]["frame_start"])
        indices = [start + int(timestep) for timestep in sample.tolist()]

        # LeRobotDataset exposes single-frame access publicly.  Its worker-local
        # reader owns the same metadata/decoder contract and lets us batch the
        # timestamps without re-opening/seeking the same MP4 M times.
        reader = dataset._ensure_reader()  # noqa: SLF001
        if reader.hf_dataset is None:
            reader.load_and_activate()
        rows = reader.hf_dataset[indices]

        def as_int(value: Any) -> int:
            return int(value.item()) if isinstance(value, Tensor) else int(value)

        def as_float(value: Any) -> float:
            return float(value.item()) if isinstance(value, Tensor) else float(value)

        episode_ids = [as_int(value) for value in rows["episode_index"]]
        if len(set(episode_ids)) != 1:
            raise RuntimeError(f"A skill must stay within one episode, got {episode_ids}.")
        episode_id = episode_ids[0]
        timestamps = [as_float(value) for value in rows["timestamp"]]
        episode = reader._meta.episodes[episode_id]  # noqa: SLF001

        def decode(camera_key: str) -> Tensor:
            video_path = reader.root / reader._meta.get_video_file_path(episode_id, camera_key)  # noqa: SLF001
            from_timestamp = float(episode[f"videos/{camera_key}/from_timestamp"])
            query_timestamps = [from_timestamp + timestamp for timestamp in timestamps]
            if self._frame_cache is not None:
                return self._frame_cache.get_frames(
                    video_path,
                    query_timestamps,
                    reader._tolerance_s,  # noqa: SLF001
                )
            from lerobot.datasets.video_utils import decode_video_frames

            return decode_video_frames(
                video_path,
                query_timestamps,
                reader._tolerance_s,  # noqa: SLF001
                reader._video_backend,  # noqa: SLF001
                # The dataset is AV1.  dav1d's automatic thread pool has been
                # observed deadlocking in a persistent DataLoader worker after
                # many epochs, which blocks the ordered loader indefinitely.
                decoder_num_threads=1,
            )

        return decode("observation.images.image"), decode("observation.images.wrist_image")

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        length = self.lengths[index]
        sample = self._sample_indices(length)
        raw_state = self.states[index][sample, : self.cfg.state_dim]
        prev_action = self.previous_actions[index][sample, : self.cfg.action_dim]
        if (
            self.cfg.reconstructor_start_state
            and self.cfg.reconstructor_start_state_conditioning == "adaln"
        ):
            # Absolute start pose keeps absolute-state statistics even in the
            # zero-grounded autoencoder.  Applying the zero-grounded XYZ range
            # to an absolute XYZ value would mix incompatible coordinates.
            start_lo = np.asarray(self.cfg.state_min, dtype=np.float32)
            start_hi = np.asarray(self.cfg.state_max, dtype=np.float32)
        else:
            # Legacy concat/chunk checkpoints used Stage-1 q01/q99 scaling.
            start_lo, start_hi = self.state_q01, self.state_q99
        start_norm = self._quantile_norm(
            self.states[index][0:1, : self.cfg.state_dim],
            start_lo,
            start_hi,
        )[0]
        start_norm = scale_gripper_features(
            start_norm,
            gripper_weight=self.cfg.action_gripper_weight,
            gripper_dims=N_GRIPPER_DIMS,
        )
        start_state = np.zeros((len(sample), self.cfg.max_state_dim), dtype=np.float32)
        start_state[:, : self.cfg.state_dim] = start_norm
        progress = sample.astype(np.float32) / max(length - 1, 1)
        termination = self._termination_targets(length, sample)
        item = {
            "ctrl": torch.from_numpy(self.ctrl[index]),
            "length": torch.tensor(length, dtype=torch.long),
            "start_state": torch.from_numpy(start_state),
            "raw_state": torch.from_numpy(raw_state.copy()),
            "prev_action": torch.from_numpy(prev_action.copy()),
            "actions": torch.from_numpy(self._action_chunks(self.actions[index], sample)),
            "progress": torch.from_numpy(progress),
            "termination": torch.from_numpy(termination),
            "sample_index": torch.from_numpy(sample),
            "trajectory_index": torch.tensor(index, dtype=torch.long),
        }
        if self.reconstructor_ctrl is not None:
            item["reconstructor_ctrl"] = torch.from_numpy(self.reconstructor_ctrl[index])
        if self.terminator_context_sequences is not None:
            item["terminator_context_sequence"] = self.terminator_context_sequences[index]
            item["terminator_progress"] = self.terminator_progress_targets[index]
            item["terminator_termination"] = self.terminator_end_targets[index]
        elif (
            not self.cfg.reconstructor_only
            and self.cfg.terminator_input_space in {"image", "both"}
        ):
            item["third"], item["wrist"] = self._sample_images(index, sample)
        if self.start_poses is not None:
            item["start_pose"] = torch.from_numpy(self.start_poses[index])
        if self.pair_augmentation:
            augmented, boundary, offset = sample_boundary_augmented_segment(
                self.boundary_contexts[index],
                pmax=self.cfg.boundary_aug_pmax,
                early_start_pmax=self.cfg.boundary_aug_early_start_pmax,
                late_start_pmax=self.cfg.boundary_aug_late_start_pmax,
                early_end_pmax=self.cfg.boundary_aug_early_end_pmax,
                late_end_pmax=self.cfg.boundary_aug_late_end_pmax,
                min_length=max(1, int(round(self.cfg.length_min))),
                distribution=self.cfg.boundary_aug_distribution,
            )
            if self.cfg.encoder_arch == "action_seq":
                augmented_length = len(augmented)
                item["augmented_action_seq"] = self._padded_encoder_action_sequence(
                    augmented
                )
            else:
                augmented_ctrl, augmented_length = spline_encode(
                    augmented,
                    self.cfg.n_control,
                    self.cfg.spline_degree,
                    input_mode=self.cfg.encoder_input_mode,
                )
                augmented_ctrl = (
                    2.0
                    * (augmented_ctrl - self.encoder_min)
                    / (self.encoder_max - self.encoder_min + 1e-8)
                    - 1.0
                ).astype(np.float32)
                augmented_ctrl = scale_gripper_features(
                    augmented_ctrl,
                    gripper_weight=self.cfg.action_gripper_weight,
                    gripper_dims=N_GRIPPER_DIMS,
                )
                item["augmented_ctrl"] = torch.from_numpy(augmented_ctrl)
            item["augmented_length"] = torch.tensor(augmented_length, dtype=torch.long)
            item["augmentation_boundary"] = torch.tensor(boundary, dtype=torch.long)
            item["augmentation_offset"] = torch.tensor(offset, dtype=torch.long)
            if self.start_poses is not None:
                augmented_start = encoder_grounding_position(augmented)
                augmented_start = (
                    2.0
                    * (augmented_start - self.encoder_start_min)
                    / (self.encoder_start_max - self.encoder_start_min + 1e-8)
                    - 1.0
                ).astype(np.float32)
                item["augmented_start_pose"] = torch.from_numpy(augmented_start)
            if self.contrastive_pairs:
                neighbours = self.adjacent_skill_indices[index]
                if neighbours:
                    negative_index = (
                        neighbours[0]
                        if len(neighbours) == 1 or np.random.random() < 0.5
                        else neighbours[1]
                    )
                    negative_valid = True
                else:
                    # Fixed-shape placeholder keeps collation simple.  The loss
                    # masks its negative component for true singleton episodes.
                    negative_index = index
                    negative_valid = False
                if self.cfg.encoder_arch == "action_seq":
                    item["negative_action_seq"] = self._padded_encoder_action_sequence(
                        self.actions[negative_index]
                    )
                else:
                    item["negative_ctrl"] = torch.from_numpy(self.ctrl[negative_index])
                item["negative_length"] = torch.tensor(
                    self.lengths[negative_index], dtype=torch.long
                )
                item["negative_valid"] = torch.tensor(
                    negative_valid, dtype=torch.bool
                )
                item["negative_trajectory_index"] = torch.tensor(
                    negative_index, dtype=torch.long
                )
                if self.start_poses is not None and self.cfg.encoder_arch == "spline":
                    item["negative_start_pose"] = torch.from_numpy(
                        self.start_poses[negative_index]
                    )
        if getattr(self.cfg, "encoder_arch", "spline") == "action_seq":
            # Full sequence padded to length_max; the encoder slices to the
            # batch max and masks padding. The matched action_seq pair is raw
            # for action and normalized/scaled for norm_action; historical
            # action_seq->chunk remains q01/q99 normalized.
            item["encoder_action_seq"] = self._padded_encoder_action_sequence(
                self.actions[index]
            )
        if (
            is_action_sequence_reconstructor_arch(self.cfg.reconstructor_arch)
            and not self.cfg.terminator_only
        ):
            # Raw reconstruction target. Reuse the encoder tensor when present;
            # the matched architecture guarantees it has the same convention.
            if "encoder_action_seq" in item:
                item["reconstructor_action_seq"] = item["encoder_action_seq"]
            else:
                pad_steps = int(round(self.cfg.length_max))
                action = self.actions[index]
                padded = np.zeros((pad_steps, action.shape[-1]), dtype=np.float32)
                padded[: len(action)] = action
                item["reconstructor_action_seq"] = torch.from_numpy(padded)
        return item


def collate_fsq_batch(batch: list[dict[str, Tensor]]) -> dict[str, Tensor | None]:
    out: dict[str, Tensor | None] = {}
    for key in batch[0]:
        out[key] = torch.stack([item[key] for item in batch])
    return out


def _per_trajectory_mean(value: Tensor, bsize: int, samples_per_skill: int) -> Tensor:
    return value.view(bsize, samples_per_skill).mean(dim=1).mean()


def fsq_pair_weight_at_epoch(
    target_weight: float,
    epoch: int,
    warmup_epochs: int,
    ramp_epochs: int,
    *,
    enabled: bool = True,
) -> float:
    """Reconstruction-only warm-up followed by a linear pair-weight ramp."""
    if target_weight < 0:
        raise ValueError(f"Pair target weight must be non-negative, got {target_weight}.")
    if epoch < 1:
        raise ValueError(f"Epoch must be >= 1, got {epoch}.")
    if warmup_epochs < 0 or ramp_epochs < 0:
        raise ValueError(
            "Pair warm-up and ramp epochs must be non-negative, "
            f"got {warmup_epochs} and {ramp_epochs}."
        )
    if not enabled:
        return float(target_weight)
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs == 0:
        return float(target_weight)
    progress = min((epoch - warmup_epochs) / ramp_epochs, 1.0)
    return float(target_weight) * progress


def reconstruction_route_distortions(
    output: dict[str, Tensor],
    batch: dict[str, Tensor | None],
    cfg: SplineFSQAEConfig,
) -> Tensor:
    """Return detached per-trajectory reconstruction costs for every code.

    The result is shaped ``(B,K)`` regardless of the whole-skill decoder
    architecture. Decoder candidates are produced without autograd; keeping
    this helper explicit makes it hard to accidentally train every code toward
    every target and collapse the prototypes to an average.
    """
    if is_action_sequence_reconstructor_arch(cfg.reconstructor_arch):
        candidates = output.get("route_candidate_action_sequence")
        if candidates is None:
            raise ValueError(
                "Route loss requires all-code action-sequence candidates."
            )
        steps = int(candidates.shape[2])
        target = batch["reconstructor_action_seq"][:, :steps].to(candidates)
        target = target[..., : cfg.action_dim]
        positions = torch.arange(steps, device=candidates.device)[None]
        valid = positions < batch["length"].to(candidates.device)[:, None]
        squared = (
            candidates[..., : cfg.action_dim].float()
            - target[:, None].float()
        ).square()
        per_step = squared.mean(dim=-1)
        return (
            (per_step * valid[:, None].to(per_step.dtype)).sum(dim=-1)
            / valid.sum(dim=-1, keepdim=True).clamp_min(1)
        ).detach()

    if cfg.reconstructor_arch == "oneshot":
        candidates = output.get("route_candidate_ctrl")
        if candidates is None:
            raise ValueError(
                "Route loss requires all-code control-point candidates."
            )
        target = batch["reconstructor_ctrl"].to(candidates)
        return (
            candidates.float() - target[:, None].float()
        ).square().mean(dim=(-1, -2)).detach()

    raise ValueError(
        "Route loss requires a whole-skill reconstructor, got "
        f"{cfg.reconstructor_arch!r}."
    )


def termination_route_distortions(
    output: dict[str, Tensor],
    batch: dict[str, Tensor | None],
    cfg: SplineFSQAEConfig,
) -> Tensor:
    """Return detached per-trajectory termination BCE for every candidate code."""
    candidates = output.get("route_candidate_term_logits")
    if candidates is None:
        raise ValueError(
            "Joint route loss requires all-code termination-logit candidates."
        )
    if candidates.ndim != 3:
        raise ValueError(
            "Termination route candidates must have shape (B,K,T), got "
            f"{tuple(candidates.shape)}."
        )
    pos_weight = torch.as_tensor(
        cfg.end_pos_weight, device=candidates.device, dtype=candidates.dtype
    )
    if cfg.state_rnn_terminator:
        steps = int(candidates.shape[-1])
        target = batch["terminator_termination"][:, :steps].to(candidates)
        positions = torch.arange(steps, device=candidates.device)[None]
        valid = positions < batch["length"].to(candidates.device)[:, None]
    else:
        target = batch["termination"].reshape(
            candidates.shape[0], candidates.shape[-1]
        ).to(candidates)
        valid = torch.ones_like(target, dtype=torch.bool)
    per_step = F.binary_cross_entropy_with_logits(
        candidates,
        target[:, None].expand_as(candidates),
        reduction="none",
        pos_weight=pos_weight,
    )
    return (
        (per_step * valid[:, None].to(per_step.dtype)).sum(dim=-1)
        / valid.sum(dim=-1, keepdim=True).clamp_min(1)
    ).detach()


def fsq_reconstruction_loss(
    output: dict[str, Tensor],
    batch: dict[str, Tensor | None],
    cfg: SplineFSQAEConfig,
    *,
    pair_weight: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    bsize = int(batch["ctrl"].shape[0])
    m = cfg.samples_per_skill
    pred = output["actions"][..., : cfg.action_dim]
    ctrl_diag: dict[str, Tensor] = {}
    if cfg.terminator_only:
        action_loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
    elif is_action_sequence_reconstructor_arch(
        getattr(cfg, "reconstructor_arch", "chunk")
    ):
        sequence_pred = output["action_sequence_hat"]
        steps = int(sequence_pred.shape[1])
        sequence_target = batch["reconstructor_action_seq"][:, :steps].to(sequence_pred)
        sequence_target = sequence_target[..., : cfg.action_dim]
        positions = torch.arange(steps, device=sequence_pred.device)[None]
        valid = positions < batch["length"].to(sequence_pred.device)[:, None]
        squared = (sequence_pred - sequence_target).square()
        per_step = squared.mean(dim=-1)
        per_trajectory = (
            (per_step * valid.to(per_step.dtype)).sum(dim=1)
            / valid.sum(dim=1).clamp_min(1)
        )
        action_loss = per_trajectory.mean()

        def masked_group_mean(start: int, stop: int) -> Tensor:
            group = squared[..., start:stop]
            if group.shape[-1] == 0:
                return squared.new_zeros(())
            mask = valid[..., None].to(group.dtype)
            return (group * mask).sum() / (
                valid.sum().clamp_min(1) * group.shape[-1]
            )

        ctrl_diag = {
            "action_xyz": masked_group_mean(0, min(3, cfg.action_dim)).detach(),
            "action_rpy": masked_group_mean(3, min(6, cfg.action_dim)).detach(),
            "action_gripper": masked_group_mean(6, cfg.action_dim).detach(),
            "action_mean_valid_length": valid.sum(dim=1).float().mean().detach(),
        }
    elif getattr(cfg, "reconstructor_arch", "chunk") == "oneshot":
        # One ctrl-grid reconstruction per trajectory; occupies the 'action'
        # metric/weight slot so selection, prints, and wandb panels carry over.
        ctrl_hat = output["ctrl_hat"]
        ctrl_target = batch["reconstructor_ctrl"].to(ctrl_hat)
        ctrl_error = (ctrl_hat - ctrl_target).square()
        action_loss = ctrl_error.mean()
        pose_dims = ctrl_target.shape[-1] - N_GRIPPER_DIMS
        ctrl_diag = {
            "ctrl_pose": ctrl_error[..., :pose_dims].mean().detach(),
            "ctrl_gripper": ctrl_error[..., pose_dims:].mean().detach(),
        }
    else:
        target = batch["actions"].reshape(bsize * m, cfg.chunk_size, cfg.max_action_dim)
        target = target[..., : cfg.action_dim].to(pred)
        per_sample_action = (pred - target).square().mean(dim=(1, 2))
        action_loss = _per_trajectory_mean(per_sample_action, bsize, m)

    sequence_valid = None
    if cfg.state_rnn_terminator and not cfg.reconstructor_only:
        sequence_length = int(output["term_logits"].shape[1])
        positions = torch.arange(
            sequence_length, device=output["term_logits"].device
        )[None]
        sequence_valid = positions < batch["length"].to(positions.device)[:, None]

    if not cfg.terminator_progress or cfg.reconstructor_only:
        progress_loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
    elif sequence_valid is not None:
        progress_per = F.smooth_l1_loss(
            output["progress"],
            batch["terminator_progress"].to(output["progress"]),
            reduction="none",
        )
        progress_loss = (
            progress_per * sequence_valid.to(progress_per.dtype)
        ).sum() / sequence_valid.sum().clamp_min(1)
    else:
        progress_per = F.smooth_l1_loss(
            output["progress"], batch["progress"].reshape(-1).to(output["progress"]), reduction="none"
        )
        progress_loss = _per_trajectory_mean(progress_per, bsize, m)
    if cfg.reconstructor_only or not cfg.terminator_termination:
        end_loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
    else:
        pos_weight = torch.as_tensor(cfg.end_pos_weight, device=output["term_logits"].device)
        end_target = (
            batch["terminator_termination"].to(output["term_logits"])
            if sequence_valid is not None
            else batch["termination"].reshape(-1).to(output["term_logits"])
        )
        end_per = F.binary_cross_entropy_with_logits(
            output["term_logits"],
            end_target,
            reduction="none",
            pos_weight=pos_weight,
        )
        end_loss = (
            (end_per * sequence_valid.to(end_per.dtype)).sum()
            / sequence_valid.sum().clamp_min(1)
            if sequence_valid is not None
            else _per_trajectory_mean(end_per, bsize, m)
        )
    total = (
        cfg.action_loss_weight * action_loss
        + cfg.progress_loss_weight * progress_loss
        + cfg.end_loss_weight * end_loss
    )
    metrics = {
        "action": action_loss.detach(),
        "progress": progress_loss.detach(),
        "termination": end_loss.detach(),
        **ctrl_diag,
    }
    if sequence_valid is not None:
        metrics["terminator_mean_valid_length"] = sequence_valid.sum(dim=1).float().mean().detach()
    if output.get("skill_shuffle_progress") is not None:
        trajectory_valid = output["skill_shuffle_valid"].bool()
        if sequence_valid is not None:
            shuffle_mask = trajectory_valid[:, None] & sequence_valid
        else:
            shuffle_mask = trajectory_valid.repeat_interleave(m)
        shuffle_weight = shuffle_mask.to(output["progress"].dtype)
        denominator = shuffle_weight.sum().clamp_min(1.0)
        deltas: list[Tensor] = []
        if cfg.terminator_progress and not cfg.reconstructor_only:
            progress_delta = (
                (output["progress"] - output["skill_shuffle_progress"])
                .abs()
                .mul(shuffle_weight)
                .sum()
                / denominator
            )
            metrics["skill_shuffle_progress_delta"] = progress_delta.detach()
            deltas.append(progress_delta)
        if cfg.terminator_termination and not cfg.reconstructor_only:
            probability_delta = (
                (
                    output["term_logits"].sigmoid()
                    - output["skill_shuffle_term_logits"].sigmoid()
                )
                .abs()
                .mul(shuffle_weight)
                .sum()
                / denominator
            )
            metrics["skill_shuffle_end_probability_delta"] = probability_delta.detach()
            deltas.append(probability_delta)
        if deltas:
            metrics["skill_shuffle_mean_delta"] = torch.stack(deltas).mean().detach()
        metrics["skill_shuffle_valid_fraction"] = trajectory_valid.float().mean().detach()
    if cfg.route_loss:
        if output.get("u_cont") is None:
            raise ValueError(
                "Route loss requires continuous quantizer coordinates."
            )
        reconstruction_distortions = reconstruction_route_distortions(
            output, batch, cfg
        )
        termination_distortions = None
        joint_distortions = (
            cfg.action_loss_weight * reconstruction_distortions
        )
        if (
            not cfg.reconstructor_only
            and cfg.terminator_termination
            and cfg.end_loss_weight != 0.0
        ):
            termination_distortions = termination_route_distortions(
                output, batch, cfg
            )
            joint_distortions = (
                joint_distortions
                + cfg.end_loss_weight * termination_distortions
            )
        if cfg.quantizer == "bsq":
            route_probabilities = bsq_joint_soft_assignments(
                output["u_cont"], cfg.pair_inv_temperature
            )
        else:
            route_probabilities = fsq_joint_soft_assignments(
                output["u_cont"],
                cfg.fsq_levels,
                cfg.pair_inv_temperature,
            )
        if route_probabilities.shape != joint_distortions.shape:
            raise ValueError(
                "Route probability/distortion shapes do not match: "
                f"{tuple(route_probabilities.shape)} vs "
                f"{tuple(joint_distortions.shape)}."
            )
        reconstruction_route_component = (
            route_probabilities
            * reconstruction_distortions.to(route_probabilities)
        ).sum(dim=-1).mean()
        termination_route_loss = None
        if termination_distortions is not None:
            termination_route_loss = (
                route_probabilities
                * termination_distortions.to(route_probabilities)
            ).sum(dim=-1).mean()
        route_loss = (
            route_probabilities
            * joint_distortions.to(route_probabilities)
        ).sum(dim=-1).mean()
        # Candidate distortions are detached, so this term updates only the
        # soft encoder assignment. The ordinary selected-code losses above
        # remain responsible for training the reconstructor and terminator.
        total = total + route_loss

        oracle_distortion, oracle_code = joint_distortions.min(dim=-1)
        hard_code = output["indices"].long()
        hard_distortion = joint_distortions.gather(
            1, hard_code[:, None]
        ).squeeze(1)
        oracle_probability = route_probabilities.gather(
            1, oracle_code[:, None]
        ).squeeze(1)
        metrics.update(
            {
                "route_loss": route_loss.detach(),
                "route_weighted_loss": route_loss.detach(),
                "route_reconstruction_loss": reconstruction_route_component.detach(),
                "route_hard_distortion": hard_distortion.mean().detach(),
                "route_oracle_distortion": oracle_distortion.mean().detach(),
                "route_regret": (
                    hard_distortion - oracle_distortion
                ).mean().detach(),
                "route_oracle_code_agreement": (
                    hard_code == oracle_code
                ).float().mean().detach(),
                "route_oracle_probability": oracle_probability.mean().detach(),
            }
        )
        if termination_route_loss is not None:
            metrics["route_termination_loss"] = termination_route_loss.detach()
    if getattr(cfg, "fsq_entropy", False) and output.get("u_cont") is not None:
        sample_entropies, dataset_entropy = fsq_entropy_statistics(
            output["u_cont"],
            cfg.fsq_levels,
            cfg.entropy_inv_temperature,
            joint_dataset=cfg.entropy_joint,
        )
        sample_entropy = sample_entropies.mean()
        max_sample_entropy = sum(math.log(int(level)) for level in cfg.fsq_levels)
        entropy_ceiling = cfg.entropy_conf_ceiling * max_sample_entropy
        confidence_violations = (sample_entropies - entropy_ceiling).clamp_min(0.0)
        confidence_loss = confidence_violations.mean()
        total = (
            total
            + cfg.entropy_conf_weight * confidence_loss
            - cfg.entropy_div_weight * dataset_entropy
        )
        metrics["entropy_sample"] = sample_entropy.detach()
        metrics["entropy_sample_normalized"] = (
            sample_entropy / max(max_sample_entropy, 1e-12)
        ).detach()
        metrics["entropy_conf_loss"] = confidence_loss.detach()
        metrics["entropy_conf_active_fraction"] = (
            sample_entropies > entropy_ceiling
        ).float().mean().detach()
        metrics["entropy_conf_ceiling_nats"] = sample_entropy.new_tensor(
            entropy_ceiling
        ).detach()
        metrics["entropy_dataset"] = dataset_entropy.detach()
    if (
        getattr(cfg, "pair_loss", "none") != "none"
        and output.get("augmented_u_cont") is not None
    ):
        if output.get("u_cont") is None:
            raise ValueError("FSQ pair loss requires clean and augmented bounded coordinates.")
        if cfg.quantizer == "bsq":
            positive_overlaps = bsq_pair_joint_overlaps(
                output["u_cont"],
                output["augmented_u_cont"],
                cfg.pair_inv_temperature,
            )
            js_loss = bsq_js_pair_loss(
                output["u_cont"],
                output["augmented_u_cont"],
                cfg.pair_inv_temperature,
            )
        else:
            positive_overlaps = fsq_pair_joint_overlaps(
                output["u_cont"],
                output["augmented_u_cont"],
                cfg.fsq_levels,
                cfg.pair_inv_temperature,
            )
            js_loss = fsq_js_pair_loss(
                output["u_cont"],
                output["augmented_u_cont"],
                cfg.fsq_levels,
                cfg.pair_inv_temperature,
            )
        pair_overlap = positive_overlaps.mean()
        overlap_loss = -positive_overlaps.clamp_min(1e-12).log().mean()
        if cfg.pair_loss == "overlap":
            pair_loss = overlap_loss
        elif cfg.pair_loss == "js":
            pair_loss = js_loss
        elif cfg.pair_loss == "contrastive":
            if (
                output.get("negative_u_cont") is None
                or output.get("negative_indices") is None
                or batch.get("negative_valid") is None
            ):
                raise ValueError(
                    "Contrastive pair loss requires negative coordinates, codes, and validity."
                )
            if cfg.quantizer == "bsq":
                negative_overlaps = bsq_pair_joint_overlaps(
                    output["u_cont"],
                    output["negative_u_cont"],
                    cfg.pair_inv_temperature,
                )
            else:
                negative_overlaps = fsq_pair_joint_overlaps(
                    output["u_cont"],
                    output["negative_u_cont"],
                    cfg.fsq_levels,
                    cfg.pair_inv_temperature,
                )
            negative_valid = batch["negative_valid"].to(
                device=negative_overlaps.device, dtype=torch.bool
            )
            positive_linear = 1.0 - positive_overlaps
            # Eligible anchors receive the symmetric average
            # 0.5 * ((1-positive_overlap) + negative_overlap).  A singleton
            # episode has no valid negative, but can still learn invariance
            # from its positive boundary augmentation.
            pair_loss = torch.where(
                negative_valid,
                0.5 * (positive_linear + negative_overlaps),
                positive_linear,
            ).mean()
            valid_weight = negative_valid.to(negative_overlaps.dtype)
            valid_denominator = valid_weight.sum().clamp_min(1.0)
            negative_linear_loss = (
                negative_overlaps * valid_weight
            ).sum() / valid_denominator
            negative_code_agreement = (
                (output["indices"] == output["negative_indices"])
                .to(negative_overlaps.dtype)
                .mul(valid_weight)
                .sum()
                / valid_denominator
            )
            metrics.update(
                {
                    "pair_positive_linear_loss": positive_linear.mean().detach(),
                    "pair_positive_joint_overlap": pair_overlap.detach(),
                    "pair_positive_code_agreement": (
                        output["indices"] == output["augmented_indices"]
                    ).float().mean().detach(),
                    "pair_negative_linear_loss": negative_linear_loss.detach(),
                    "pair_negative_joint_overlap": negative_linear_loss.detach(),
                    "pair_negative_code_agreement": negative_code_agreement.detach(),
                    "pair_negative_valid_fraction": valid_weight.mean().detach(),
                }
            )
        else:
            raise ValueError(f"Unsupported FSQ pair loss: {cfg.pair_loss!r}.")
        effective_pair_weight = cfg.pair_weight if pair_weight is None else pair_weight
        weighted_pair_loss = effective_pair_weight * pair_loss
        total = total + weighted_pair_loss
        metrics["pair_loss"] = pair_loss.detach()
        metrics["pair_overlap_loss"] = overlap_loss.detach()
        metrics["pair_js_loss"] = js_loss.detach()
        metrics["pair_weighted_loss"] = weighted_pair_loss.detach()
        metrics["pair_weight"] = pair_loss.new_tensor(effective_pair_weight).detach()
        metrics["pair_joint_overlap"] = pair_overlap.detach()
        metrics["pair_code_agreement"] = (
            output["indices"] == output["augmented_indices"]
        ).float().mean().detach()
        metrics["pair_forward_skipped"] = pair_loss.new_zeros(()).detach()
    elif getattr(cfg, "pair_loss", "none") != "none":
        effective_pair_weight = cfg.pair_weight if pair_weight is None else pair_weight
        if effective_pair_weight > 0.0:
            raise ValueError(
                "A positive FSQ pair weight requires augmented pair coordinates."
            )
        zero = total.new_zeros(())
        metrics["pair_weighted_loss"] = zero.detach()
        metrics["pair_weight"] = zero.detach()
        metrics["pair_forward_skipped"] = total.new_ones(()).detach()
    metrics["loss"] = total.detach()
    return total, metrics


@torch.no_grad()
def end_signal_metrics(logits: Tensor, target: Tensor, threshold: float) -> dict[str, float]:
    pred = torch.sigmoid(logits) >= threshold
    truth = target >= 0.5
    tp = (pred & truth).float().sum()
    fp = (pred & ~truth).float().sum()
    fn = (~pred & truth).float().sum()
    eps = torch.finfo(torch.float32).eps
    total = torch.tensor(float(pred.numel())).clamp_min(eps)
    return {
        "acc": float((pred == truth).float().sum() / total),
        "precision": float(tp / (tp + fp).clamp_min(eps)),
        "recall": float(tp / (tp + fn).clamp_min(eps)),
        "positive_rate": float(pred.float().sum() / total),
    }


def absorb_z_head_calibration_(
    z_head: nn.Linear,
    mean: Tensor,
    scale: Tensor,
    *,
    gain: float = 1.0,
) -> None:
    """Fold ``gain * (z - mean) / scale`` into a linear z head in place.

    This is an initialization reparameterization, not a normalization layer:
    after the update the ordinary linear head directly emits the calibrated
    coordinates and remains completely free to move during optimization.
    """
    if not isinstance(z_head, nn.Linear):
        raise TypeError(f"FSQ z_head calibration requires nn.Linear, got {type(z_head)!r}.")
    if z_head.bias is None:
        raise ValueError("FSQ z_head calibration requires a bias term.")
    if gain <= 0:
        raise ValueError(f"FSQ z_head calibration gain must be positive, got {gain}.")
    mean = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
    scale = torch.as_tensor(scale, dtype=torch.float32).reshape(-1)
    if mean.numel() != z_head.out_features or scale.numel() != z_head.out_features:
        raise ValueError(
            "FSQ z_head calibration statistics must match the latent dimension: "
            f"mean={mean.numel()}, scale={scale.numel()}, out={z_head.out_features}."
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
        raise ValueError("FSQ z_head calibration statistics must be finite.")
    if (scale <= 1e-6).any():
        bad = torch.nonzero(scale <= 1e-6, as_tuple=False).reshape(-1).tolist()
        raise ValueError(
            "FSQ z_head calibration found a dead latent axis with std <= 1e-6: "
            f"axes={bad}, std={scale.tolist()}."
        )
    multiplier = (float(gain) / scale).to(
        device=z_head.weight.device, dtype=z_head.weight.dtype
    )
    centered_mean = mean.to(device=z_head.bias.device, dtype=z_head.bias.dtype)
    with torch.no_grad():
        z_head.weight.mul_(multiplier[:, None])
        z_head.bias.sub_(centered_mean).mul_(multiplier)


def _fsq_code_summary_from_latents(
    fsq: FSQ | BSQ, latents: Tensor
) -> dict[str, float]:
    """Small CPU diagnostic used before/after one-shot z-head calibration."""
    # ``calibrate_fsq_z_head_`` deliberately accumulates all encoder outputs on
    # CPU, while the already-built model (and therefore the quantizer buffers)
    # lives on the training device. Run the tiny assignment pass beside those
    # buffers, then return only the indices to CPU for the histogram. The old
    # FSQ-only implementation copied each FSQ buffer to CPU manually; this is
    # the quantizer-agnostic equivalent and also covers BSQ's bit_weights.
    first_buffer = next(fsq.buffers(), None)
    quantizer_device = first_buffer.device if first_buffer is not None else latents.device
    z = latents.detach().to(device=quantizer_device, dtype=torch.float32)
    _, indices = fsq(z)
    indices = indices.to(device="cpu", dtype=torch.long)
    counts = torch.bincount(indices, minlength=fsq.codebook_size)
    return {
        "active_entries": float((counts > 0).sum()),
        "utilization_pct": 100.0 * float((counts > 0).sum()) / fsq.codebook_size,
        "dominant_code_pct": 100.0 * float(counts.max()) / max(1, indices.numel()),
    }


@torch.inference_mode()
def calibrate_fsq_z_head_(
    model: SplineFSQAE,
    dataset: FSQTrajectoryDataset,
    device: torch.device,
    batch_size: int,
    *,
    gain: float,
    max_samples: int = 0,
) -> dict[str, float]:
    """Calibrate a fresh z head once from deterministic clean trajectories."""
    if len(dataset) == 0:
        raise ValueError("FSQ z_head calibration requires a non-empty dataset.")
    if max_samples < 0:
        raise ValueError(f"FSQ z_head calibration max_samples must be >= 0, got {max_samples}.")
    sample_count = len(dataset) if max_samples == 0 else min(max_samples, len(dataset))
    if sample_count == len(dataset):
        sample_ids = np.arange(len(dataset), dtype=np.int64)
    else:
        sample_ids = np.linspace(0, len(dataset) - 1, sample_count, dtype=np.int64)

    was_training = model.training
    model.eval()
    latent_chunks: list[Tensor] = []
    action_seq_arch = getattr(model.cfg, "encoder_arch", "spline") == "action_seq"
    for batch_start in range(0, sample_count, batch_size):
        ids = sample_ids[batch_start : batch_start + batch_size].tolist()
        lengths = torch.as_tensor(
            [dataset.lengths[i] for i in ids], dtype=torch.long, device=device
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            if action_seq_arch:
                actions = [dataset.encoder_action_values(i) for i in ids]
                steps = max(len(action) for action in actions)
                padded = np.zeros(
                    (len(actions), steps, actions[0].shape[-1]), dtype=np.float32
                )
                for row, action in enumerate(actions):
                    padded[row, : len(action)] = action
                action_seq = torch.from_numpy(padded).to(device, non_blocking=True)
                z_e = model.encoder.encode_continuous(action_seq, lengths)
            else:
                ctrl = torch.from_numpy(np.stack([dataset.ctrl[i] for i in ids])).to(
                    device, non_blocking=True
                )
                start_pose = None
                if dataset.start_poses is not None:
                    start_pose = torch.from_numpy(
                        np.stack([dataset.start_poses[i] for i in ids])
                    ).to(device, non_blocking=True)
                z_e = model.encoder.encode_continuous(
                    ctrl, lengths, start_pose, normalized=True
                )
        latent_chunks.append(z_e.float().cpu())

    latents = torch.cat(latent_chunks, dim=0)
    mean = latents.mean(dim=0)
    scale = latents.std(dim=0, correction=0)
    pre = _fsq_code_summary_from_latents(model.fsq, latents)
    absorb_z_head_calibration_(model.encoder.z_head, mean, scale, gain=gain)
    multiplier = float(gain) / scale
    calibrated = (latents - mean) * multiplier
    post = _fsq_code_summary_from_latents(model.fsq, calibrated)
    post_mean = calibrated.mean(dim=0)
    post_std = calibrated.std(dim=0, correction=0)
    model.train(was_training)

    metrics: dict[str, float] = {
        "init_calibration/samples": float(sample_count),
        "init_calibration/gain": float(gain),
    }
    for prefix, summary in (("pre", pre), ("post", post)):
        for name, value in summary.items():
            metrics[f"init_calibration/{prefix}_{name}"] = value
    for axis in range(latents.shape[1]):
        metrics[f"init_calibration/pre_mean_axis_{axis}"] = float(mean[axis])
        metrics[f"init_calibration/pre_std_axis_{axis}"] = float(scale[axis])
        metrics[f"init_calibration/post_mean_axis_{axis}"] = float(post_mean[axis])
        metrics[f"init_calibration/post_std_axis_{axis}"] = float(post_std[axis])
    return metrics


@torch.inference_mode()
def _collect_code_assignments(
    model: SplineFSQAE,
    datasets: tuple[FSQTrajectoryDataset, ...],
    device: torch.device,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    """Encode every skill and measure every quantizer-axis boundary margin.

    Training batches cannot be reused for this metric: they are shuffled and
    are encoded by progressively different model states within an epoch. The
    normalized trajectory control points cached by ``FSQTrajectoryDataset`` are
    sufficient for a deterministic, encoder-only full-dataset pass. The full
    ``(samples, axes)`` margin tensor is retained: the legacy sample-level
    metric still uses its row minimum, while per-axis metrics avoid conflating
    BSQ5's five chances to approach a boundary with FSQ333's three.
    """
    assignments: list[Tensor] = []
    boundary_margins: list[Tensor] = []
    action_seq_arch = getattr(model.cfg, "encoder_arch", "spline") == "action_seq"
    for dataset in datasets:
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            lengths = torch.as_tensor(
                dataset.lengths[start:stop], dtype=torch.long, device=device
            )
            if action_seq_arch:
                acts = [dataset.encoder_action_values(i) for i in range(start, stop)]
                steps = max(len(a) for a in acts)
                batch = np.zeros((len(acts), steps, acts[0].shape[-1]), dtype=np.float32)
                for row, a in enumerate(acts):
                    batch[row, : len(a)] = a
                seq = torch.from_numpy(batch).to(device, non_blocking=True)
            else:
                ctrl = torch.from_numpy(np.stack(dataset.ctrl[start:stop])).to(
                    device, non_blocking=True
                )
                start_pose = None
                if dataset.start_poses is not None:
                    start_pose = torch.from_numpy(
                        np.stack(dataset.start_poses[start:stop])
                    ).to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                if action_seq_arch:
                    z_e = model.encoder.encode_continuous(seq, lengths)
                else:
                    z_e = model.encoder.encode_continuous(
                        ctrl, lengths, start_pose, normalized=True
                    )
                _, indices = model.fsq(z_e)
            # Compute the diagnostic in FP32 even when the encoder forward uses
            # BF16; assignment itself remains identical to the training path.
            margins = model.fsq.boundary_margin(z_e.float())
            assignments.append(indices.reshape(-1).to(device="cpu", dtype=torch.long))
            boundary_margins.append(margins.to(device="cpu", dtype=torch.float32))
    assignment_tensor = (
        torch.cat(assignments) if assignments else torch.empty(0, dtype=torch.long)
    )
    margin_tensor = (
        torch.cat(boundary_margins)
        if boundary_margins
        else torch.empty((0, model.fsq.latent_dim), dtype=torch.float32)
    )
    return assignment_tensor, margin_tensor


def _boundary_margin_metrics(margins: Tensor) -> dict[str, float]:
    """Summarize sample-minimum and per-axis margins on a 0--100 center scale."""
    margins = margins.to(device="cpu", dtype=torch.float32)
    if margins.ndim == 1:
        margins = margins[:, None]
    if margins.ndim != 2:
        raise ValueError(
            f"Boundary margins must have shape (samples, axes), got {tuple(margins.shape)}."
        )
    if margins.numel() == 0:
        return {
            "boundary_margin_mean_pct": math.nan,
            "boundary_margin_p10_pct": math.nan,
            "near_boundary_pct": math.nan,
            "per_axis_boundary_margin_mean_pct": math.nan,
            "per_axis_near_boundary_pct": math.nan,
        }
    margins = margins.clamp(0.0, 0.5)
    # Divide by the maximum center-to-boundary distance (0.5), so 0% means
    # exactly on a decision boundary and 100% means at the center of a bin.
    per_axis_normalized = margins / 0.5
    sample_normalized = per_axis_normalized.amin(dim=-1)
    metrics = {
        "boundary_margin_mean_pct": 100.0 * float(sample_normalized.mean()),
        "boundary_margin_p10_pct": 100.0 * float(torch.quantile(sample_normalized, 0.1)),
        "near_boundary_pct": 100.0 * float((sample_normalized <= 0.1).float().mean()),
        "per_axis_boundary_margin_mean_pct": 100.0 * float(per_axis_normalized.mean()),
        "per_axis_near_boundary_pct": 100.0
        * float((per_axis_normalized <= 0.1).float().mean()),
    }
    for axis in range(per_axis_normalized.shape[1]):
        values = per_axis_normalized[:, axis]
        metrics[f"axis_{axis}_boundary_margin_mean_pct"] = 100.0 * float(values.mean())
        metrics[f"axis_{axis}_boundary_margin_p10_pct"] = 100.0 * float(
            torch.quantile(values, 0.1)
        )
        metrics[f"axis_{axis}_near_boundary_pct"] = 100.0 * float(
            (values <= 0.1).float().mean()
        )
    return metrics


def _code_assignment_stability(
    previous: Tensor,
    current: Tensor,
    codebook_size: int,
    levels: list[int] | None = None,
) -> dict[str, float]:
    """Measure changes in the sample membership of FSQ code entries.

    ``retention`` compares fixed FSQ token IDs directly. ``matched`` first
    finds the best one-to-one relabeling of code IDs, separating genuine sample
    migration from a global permutation of otherwise unchanged groups.
    """
    previous = previous.reshape(-1).to(device="cpu", dtype=torch.long)
    current = current.reshape(-1).to(device="cpu", dtype=torch.long)
    if previous.numel() != current.numel():
        raise ValueError(
            "Code-assignment comparison requires the same samples, got "
            f"{previous.numel()} and {current.numel()}."
        )
    if previous.numel() == 0:
        return {
            "retention_pct": math.nan,
            "change_pct": math.nan,
            "matched_retention_pct": math.nan,
        }
    if (
        int(previous.min()) < 0
        or int(current.min()) < 0
        or int(previous.max()) >= codebook_size
        or int(current.max()) >= codebook_size
    ):
        raise ValueError("Code assignment contains an index outside the FSQ codebook.")

    retention = float((previous == current).float().mean())
    pairs = previous * codebook_size + current
    overlap = torch.bincount(
        pairs, minlength=codebook_size * codebook_size
    ).reshape(codebook_size, codebook_size)
    # Validation-only dependency; no need to load SciPy while importing FSQ.
    from scipy.optimize import linear_sum_assignment

    old_ids, new_ids = linear_sum_assignment(-overlap.numpy())
    matched = float(overlap[old_ids, new_ids].sum()) / previous.numel()
    metrics = {
        "retention_pct": 100.0 * retention,
        "change_pct": 100.0 * (1.0 - retention),
        "matched_retention_pct": 100.0 * matched,
    }
    if levels is not None:
        levels_tensor = torch.as_tensor(levels, dtype=torch.long).reshape(-1)
        if levels_tensor.numel() == 0 or bool((levels_tensor < 2).any()):
            raise ValueError(f"Quantizer levels must all be >= 2, got {levels}.")
        if int(levels_tensor.prod()) != codebook_size:
            raise ValueError(
                "Quantizer levels do not match codebook size: "
                f"levels={levels}, product={int(levels_tensor.prod())}, "
                f"codebook_size={codebook_size}."
            )
        strides = torch.ones_like(levels_tensor)
        if levels_tensor.numel() > 1:
            strides[1:] = torch.cumprod(levels_tensor[:-1], dim=0)
        previous_axes = torch.div(
            previous[:, None], strides[None], rounding_mode="floor"
        ) % levels_tensor[None]
        current_axes = torch.div(
            current[:, None], strides[None], rounding_mode="floor"
        ) % levels_tensor[None]
        axis_changed = previous_axes != current_axes
        metrics["per_axis_change_pct"] = 100.0 * float(axis_changed.float().mean())
        metrics["per_axis_retention_pct"] = 100.0 - metrics["per_axis_change_pct"]
        for axis in range(axis_changed.shape[1]):
            change = 100.0 * float(axis_changed[:, axis].float().mean())
            metrics[f"axis_{axis}_change_pct"] = change
            metrics[f"axis_{axis}_retention_pct"] = 100.0 - change
    return metrics


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def fsq_lr_factor(schedule: str, epoch: int, epochs: int) -> float:
    """Return the LR multiplier for one FSQ epoch."""
    if schedule == "constant":
        return 1.0
    if schedule != "cosine":
        raise ValueError(f"lr_schedule must be cosine|constant, got {schedule!r}.")
    if epochs <= 1:
        return 0.01
    progress = min(max(epoch / (epochs - 1), 0.0), 1.0)
    return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))


def episode_grouped_train_val_ids(
    metadata: list[dict[str, Any]],
    target_val_size: int,
) -> tuple[list[int], list[int]]:
    """Deterministically split whole episodes into validation and training.

    Contrastive negatives must remain beside their anchor, so this split never
    divides one ``(task_id, episode_id)`` across the two datasets.  Validation
    may exceed the requested trajectory count by the final episode's size.
    """
    if target_val_size < 1:
        raise ValueError("Episode-grouped split requires target_val_size >= 1.")
    by_episode: dict[tuple[int, int], list[int]] = {}
    for i, item in enumerate(metadata):
        key = (int(item.get("task_id", -1)), int(item["episode_id"]))
        by_episode.setdefault(key, []).append(i)
    if len(by_episode) < 2:
        raise ValueError(
            "Contrastive pair loss requires at least two episodes so one full "
            "episode can be reserved for validation."
        )

    def episode_hash(key: tuple[int, int]) -> int:
        return int(hashlib.sha1(f"{key[0]}_{key[1]}".encode()).hexdigest(), 16)

    episode_order = sorted(by_episode, key=episode_hash)
    validation_keys: list[tuple[int, int]] = []
    validation_size = 0
    # Always retain the final hashed episode for training.
    for key in episode_order[:-1]:
        if validation_size >= target_val_size:
            break
        validation_keys.append(key)
        validation_size += len(by_episode[key])
    validation_set = set(validation_keys)
    val_ids = [
        i
        for key in episode_order
        if key in validation_set
        for i in by_episode[key]
    ]
    train_ids = [
        i
        for key in episode_order
        if key not in validation_set
        for i in by_episode[key]
    ]
    return train_ids, val_ids


def train_spline_fsqae(
    *,
    segments: list[np.ndarray],
    decoder_states: list[np.ndarray],
    decoder_targets: list[np.ndarray],
    raw_dataset_dir: str | Path,
    cfg: SplineFSQAEConfig,
    wandb_run=None,
    metadata: list[dict[str, Any]],
    resume_from: str | None = None,
) -> SplineFSQAE:
    if not segments:
        raise ValueError("No skill trajectories were provided.")
    if cfg.pair_loss == "contrastive" and len(metadata) != len(segments):
        raise ValueError(
            "Contrastive pair loss requires metadata for every trajectory so "
            "episodes and adjacent skills can be identified."
        )
    directional_pmaxes = resolve_boundary_augmentation_pmaxes(
        cfg.boundary_aug_pmax,
        early_start_pmax=cfg.boundary_aug_early_start_pmax,
        late_start_pmax=cfg.boundary_aug_late_start_pmax,
        early_end_pmax=cfg.boundary_aug_early_end_pmax,
        late_end_pmax=cfg.boundary_aug_late_end_pmax,
    )
    (
        cfg.boundary_aug_early_start_pmax,
        cfg.boundary_aug_late_start_pmax,
        cfg.boundary_aug_early_end_pmax,
        cfg.boundary_aug_late_end_pmax,
    ) = directional_pmaxes
    cfg.boundary_aug_pmax = max(directional_pmaxes)
    sampled_timestep_required = (
        (not cfg.terminator_only and cfg.reconstructor_arch == "chunk")
        or (not cfg.reconstructor_only and not cfg.state_rnn_terminator)
    )
    if not sampled_timestep_required and cfg.samples_per_skill != 1:
        requested_samples = cfg.samples_per_skill
        cfg.samples_per_skill = 1
        print(
            "[FSQ-v3] samples_per_skill "
            f"{requested_samples} -> effective 1: selected decoder objectives "
            "do not use sampled timesteps"
        )
        if wandb_run is not None and hasattr(wandb_run, "config"):
            wandb_run.config.update(
                {"effective_samples_per_skill": 1},
                allow_val_change=True,
            )
    all_initial_previous_actions = None
    if not cfg.reconstructor_only and cfg.terminator_context == "prev_action":
        all_initial_previous_actions = build_skill_initial_previous_actions(
            decoder_targets,
            metadata,
            cfg.action_dim,
        )
    n_val = max(1, int(len(segments) * cfg.val_split))
    if len(metadata) == len(segments) and cfg.pair_loss == "contrastive":
        train_ids, val_ids = episode_grouped_train_val_ids(metadata, n_val)
        order = [*val_ids, *train_ids]
        fingerprint = hashlib.sha1(
            ",".join(
                sorted(
                    f"{metadata[i].get('task_id', -1)}_"
                    f"{metadata[i].get('episode_id', -1)}"
                    for i in val_ids
                )
            ).encode()
        ).hexdigest()[:12]
    elif len(metadata) == len(segments):
        def identity_hash(i: int) -> int:
            item = metadata[i]
            identity = f"{item.get('episode_id', -1)}_{item.get('skill_index', -1)}"
            return int(hashlib.sha1(identity.encode()).hexdigest(), 16)

        order = sorted(range(len(segments)), key=identity_hash)
        fingerprint = hashlib.sha1(
            ",".join(
                sorted(
                    f"{metadata[i].get('episode_id', -1)}_{metadata[i].get('skill_index', -1)}"
                    for i in order[:n_val]
                )
            ).encode()
        ).hexdigest()[:12]
    else:
        order = np.random.default_rng(42).permutation(len(segments)).tolist()
        fingerprint = "seed42"
    if cfg.pair_loss != "contrastive" or len(metadata) != len(segments):
        val_ids, train_ids = order[:n_val], order[n_val:]
    if len(metadata) == len(segments):
        assignment_identity = ",".join(
            f"{metadata[i].get('episode_id', -1)}_{metadata[i].get('skill_index', -1)}"
            for i in (*val_ids, *train_ids)
        )
    else:
        assignment_identity = ",".join(str(i) for i in (*val_ids, *train_ids))
    assignment_fingerprint = hashlib.sha1(assignment_identity.encode()).hexdigest()[:12]
    print(f"[FSQ-v3] trajectories={len(segments)} train={len(train_ids)} val={len(val_ids)} fp={fingerprint}")
    if cfg.pair_loss != "none":
        print(
            f"[FSQ-v3] pair warm-up: {'enabled' if cfg.pair_warmup else 'disabled'}; "
            "reconstruction-only epochs="
            f"{cfg.pair_warmup_epochs}, ramp epochs={cfg.pair_ramp_epochs}, "
            f"target weight={cfg.pair_weight:g}"
        )

    def take(items: list[Any] | None, ids: list[int]):
        return None if items is None else [items[i] for i in ids]

    def dataset(ids: list[int], training: bool) -> FSQTrajectoryDataset:
        selected_segments = take(segments, ids)
        selected_actions = take(decoder_targets, ids)
        selected_metadata = take(metadata, ids)
        boundary_contexts = None
        adjacent_skill_indices = None
        if training and cfg.pair_loss != "none":
            boundary_source = (
                selected_actions if cfg.encoder_arch == "action_seq" else selected_segments
            )
            boundary_contexts = build_boundary_augmentation_contexts(
                boundary_source,
                selected_metadata,
                cfg.boundary_aug_pmax,
            )
        if training and cfg.pair_loss == "contrastive":
            adjacent_skill_indices = build_adjacent_skill_indices(selected_metadata)
            eligible = sum(bool(neighbours) for neighbours in adjacent_skill_indices)
            print(
                "[FSQ-v3] contrastive adjacent negatives: "
                f"{eligible}/{len(adjacent_skill_indices)} anchors eligible "
                f"({100.0 * eligible / max(len(adjacent_skill_indices), 1):.2f}%)"
            )
        return FSQTrajectoryDataset(
            selected_segments,
            take(decoder_states, ids),
            selected_actions,
            selected_metadata,
            raw_dataset_dir,
            cfg,
            training=training,
            initial_previous_actions=take(all_initial_previous_actions, ids),
            boundary_contexts=boundary_contexts,
            adjacent_skill_indices=adjacent_skill_indices,
        )

    train_ds, val_ds = dataset(train_ids, True), dataset(val_ids, False)
    if cfg.num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {cfg.num_workers}.")
    if cfg.val_num_workers < 0:
        raise ValueError(
            f"val_num_workers must be >= 0, got {cfg.val_num_workers}."
        )
    if cfg.val_every < 0:
        raise ValueError(f"val_every must be >= 0, got {cfg.val_every}.")
    if cfg.save_best_model and cfg.val_every == 0:
        raise ValueError("save_best_model requires val_every > 0.")
    print(
        f"[FSQ-v3] data workers: train={cfg.num_workers} "
        f"val={cfg.val_num_workers}; validation every={cfg.val_every or 'off'}"
    )
    device = torch.device(cfg.device)
    # ``step`` moves tensors with non_blocking=True below; pinning makes that transfer genuinely
    # asynchronous instead of synchronizing the GPU after the worker has decoded the next batch.
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fsq_batch,
        pin_memory=pin_memory,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        # Surface a stuck decoder as a failed job instead of silently holding a
        # GPU allocation forever.  Normal batches complete far below 5 minutes.
        timeout=300 if cfg.num_workers > 0 else 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        # Validation defaults to the main process. Re-forking video-decoder
        # workers after every training epoch can intermittently deadlock in
        # AV1/PyAV and used to fail the whole run after the 300 s timeout.
        num_workers=cfg.val_num_workers,
        collate_fn=collate_fsq_batch,
        pin_memory=pin_memory,
        persistent_workers=False,
        prefetch_factor=2 if cfg.val_num_workers > 0 else None,
        timeout=300 if cfg.val_num_workers > 0 else 0,
    )

    model = SplineFSQAE(cfg).to(device)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("[FSQ-v3] gradient checkpointing: enabled")
    else:
        model.gradient_checkpointing_disable()
        print("[FSQ-v3] gradient checkpointing: disabled")

    init_calibration_metrics: dict[str, float] = {}
    if cfg.init_calibration and not resume_from:
        init_calibration_metrics = calibrate_fsq_z_head_(
            model,
            train_ds,
            device,
            cfg.batch_size,
            gain=cfg.init_calibration_gain,
            max_samples=cfg.init_calibration_samples,
        )
        pre_mean = [
            init_calibration_metrics[f"init_calibration/pre_mean_axis_{axis}"]
            for axis in range(model.encoder.z_head.out_features)
        ]
        pre_std = [
            init_calibration_metrics[f"init_calibration/pre_std_axis_{axis}"]
            for axis in range(model.encoder.z_head.out_features)
        ]
        print(
            "[FSQ-v3] z_head init calibration: "
            f"samples={int(init_calibration_metrics['init_calibration/samples'])}, "
            f"gain={cfg.init_calibration_gain:g}, "
            f"pre mean={np.round(pre_mean, 4).tolist()}, "
            f"pre std={np.round(pre_std, 4).tolist()}, "
            "codebook "
            f"{int(init_calibration_metrics['init_calibration/pre_active_entries'])}"
            f"->{int(init_calibration_metrics['init_calibration/post_active_entries'])} active, "
            "dominant "
            f"{init_calibration_metrics['init_calibration/pre_dominant_code_pct']:.1f}%"
            f"->{init_calibration_metrics['init_calibration/post_dominant_code_pct']:.1f}%"
        )
    elif cfg.init_calibration:
        print("[FSQ-v3] z_head init calibration: skipped on resume (already in checkpoint weights)")
    else:
        print("[FSQ-v3] z_head init calibration: disabled")

    param_groups = [
        {"params": model.encoder.parameters(), "lr": cfg.lr, "name": "encoder"},
    ]
    if model.reconstructor is not None:
        param_groups.append(
            {"params": model.reconstructor.parameters(), "lr": cfg.lr, "name": "reconstructor"}
        )
    if model.terminator is not None:
        param_groups.append(
            {"params": model.terminator.parameters(), "lr": cfg.lr, "name": "terminator"}
        )
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: fsq_lr_factor(cfg.lr_schedule, epoch, cfg.epochs),
    )
    save_path = Path(cfg.save_path) if cfg.save_path else Path("FSQ.pt")
    start_epoch, best_val = 1, math.inf
    previous_code_assignments: Tensor | None = None
    previous_code_epoch: int | None = None
    if resume_from:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        resume_cfg = _checkpoint_config(checkpoint)
        resume_directional_pmaxes = resolve_boundary_augmentation_pmaxes(
            getattr(resume_cfg, "boundary_aug_pmax", 0),
            early_start_pmax=getattr(
                resume_cfg, "boundary_aug_early_start_pmax", -1
            ),
            late_start_pmax=getattr(
                resume_cfg, "boundary_aug_late_start_pmax", -1
            ),
            early_end_pmax=getattr(
                resume_cfg, "boundary_aug_early_end_pmax", -1
            ),
            late_end_pmax=getattr(
                resume_cfg, "boundary_aug_late_end_pmax", -1
            ),
        )
        (
            resume_cfg.boundary_aug_early_start_pmax,
            resume_cfg.boundary_aug_late_start_pmax,
            resume_cfg.boundary_aug_early_end_pmax,
            resume_cfg.boundary_aug_late_end_pmax,
        ) = resume_directional_pmaxes
        resume_cfg.boundary_aug_pmax = max(resume_directional_pmaxes)
        resume_input_mode = getattr(resume_cfg, "encoder_input_mode", "zero_grounded")
        if resume_input_mode != cfg.encoder_input_mode:
            raise ValueError(
                "Cannot resume FSQ with a different encoder input convention: "
                f"checkpoint={resume_input_mode!r}, current={cfg.encoder_input_mode!r}."
            )
        if (
            cfg.encoder_input_mode in {"zero_grounded", "start_grounded", "optimal"}
            and resume_cfg.encoder_grounding_convention
            != cfg.encoder_grounding_convention
        ):
            raise ValueError(
                "Cannot resume FSQ across grounding conventions: "
                f"checkpoint={resume_cfg.encoder_grounding_convention!r}, "
                f"current={cfg.encoder_grounding_convention!r}."
            )
        if resume_cfg.reconstructor_output_mode != cfg.reconstructor_output_mode:
            raise ValueError(
                "Cannot resume FSQ with a different reconstruction output convention: "
                f"checkpoint={resume_cfg.reconstructor_output_mode!r}, "
                f"current={cfg.reconstructor_output_mode!r}."
            )
        for probe, default in (
            ("encoder_arch", "spline"),
            ("encoder_length_token", True),
            ("quantizer", "fsq"),
            ("bsq_code_dim", 5),
            ("fsq_entropy", False),
            ("entropy_conf_ceiling", 0.0),
            ("init_calibration", False),
            ("init_calibration_gain", 1.0),
            ("init_calibration_samples", 0),
            ("pair_loss", "none"),
            ("pair_weight", 0.1),
            ("pair_inv_temperature", 5.0),
            ("pair_warmup", False),
            ("pair_warmup_epochs", 0),
            ("pair_ramp_epochs", 0),
            ("boundary_aug_pmax", 0),
            ("boundary_aug_early_start_pmax", -1),
            ("boundary_aug_late_start_pmax", -1),
            ("boundary_aug_early_end_pmax", -1),
            ("boundary_aug_late_end_pmax", -1),
            ("boundary_aug_distribution", "half_normal"),
            ("reconstructor_start_state", True),
            ("reconstructor_start_state_conditioning", "concat"),
            ("reconstructor_arch", "chunk"),
        ):
            resume_value = getattr(resume_cfg, probe, default)
            if resume_value != getattr(cfg, probe):
                raise ValueError(
                    f"Cannot resume FSQ with a different {probe}: "
                    f"checkpoint={resume_value!r}, current={getattr(cfg, probe)!r}. "
                    "Use a different fsq_exp for a new run."
                )
        resume_lr_schedule = getattr(resume_cfg, "lr_schedule", "cosine")
        if resume_lr_schedule != cfg.lr_schedule:
            raise ValueError(
                "Cannot resume FSQ with a different LR schedule: "
                f"checkpoint={resume_lr_schedule!r}, current={cfg.lr_schedule!r}. "
                "Use a different fsq_exp for a new run."
            )
        resume_objectives = (
            bool(resume_cfg.terminator_progress),
            bool(resume_cfg.terminator_termination),
        )
        current_objectives = (
            bool(cfg.terminator_progress),
            bool(cfg.terminator_termination),
        )
        if resume_objectives != current_objectives:
            raise ValueError(
                "Cannot resume FSQ with a different terminator objective: "
                f"checkpoint(progress, termination)={resume_objectives}, "
                f"current={current_objectives}. "
                "Use a different fsq_exp for a new run."
            )
        resume_composition = (
            getattr(resume_cfg, "reconstructor_only", False),
            getattr(resume_cfg, "terminator_only", False),
            getattr(resume_cfg, "terminator_input_space", "both"),
            getattr(resume_cfg, "terminator_model", "default"),
        )
        current_composition = (
            cfg.reconstructor_only,
            cfg.terminator_only,
            cfg.terminator_input_space,
            cfg.terminator_model,
        )
        if resume_composition != current_composition:
            raise ValueError(
                "Cannot resume FSQ with a different model composition: checkpoint "
                "(reconstructor_only, terminator_only, input_space, terminator_model)="
                f"{resume_composition}, "
                f"current={current_composition}. Use a different fsq_exp for a new run."
            )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optim_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = float(checkpoint.get("best_val", checkpoint.get("val_select", math.inf)))
        saved_assignments = checkpoint.get("code_assignments")
        saved_assignment_epoch = checkpoint.get("code_assignments_epoch")
        saved_assignment_fingerprint = checkpoint.get("code_assignments_fingerprint")
        if saved_assignments is not None and saved_assignment_epoch is not None:
            saved_assignments = torch.as_tensor(saved_assignments, dtype=torch.long).reshape(-1)
            fingerprint_matches = (
                saved_assignment_fingerprint is None
                or saved_assignment_fingerprint == assignment_fingerprint
            )
            if saved_assignments.numel() == len(segments) and fingerprint_matches:
                previous_code_assignments = saved_assignments
                previous_code_epoch = int(saved_assignment_epoch)
                print(
                    "[FSQ-v3] restored code-assignment baseline from "
                    f"epoch {previous_code_epoch} ({saved_assignments.numel()} skills)"
                )
            else:
                print(
                    "[FSQ-v3] ignoring incompatible code-assignment baseline: "
                    f"checkpoint={saved_assignments.numel()}/{saved_assignment_fingerprint} "
                    f"current={len(segments)}/{assignment_fingerprint}"
                )
        # Legacy periodic checkpoints only stored their own score. Keep the historical best in
        # FSQ.pt so resume cannot overwrite it with a model that merely beats the periodic snapshot.
        if cfg.save_best_model and save_path.is_file():
            best_checkpoint = torch.load(
                str(save_path), map_location="cpu", weights_only=False, mmap=True)
            best_val = min(best_val, float(best_checkpoint.get("val_select", math.inf)))
            del best_checkpoint
        print(f"[FSQ-v3] resumed {resume_from} at epoch {start_epoch} (best select={best_val:.6f})")
    elif (
        model.terminator is not None
        and cfg.terminator_input_space in {"image", "both"}
    ):
        loaded_vision = initialize_terminator_vision_from_pi05(model.terminator, cfg.pi_base)
        if loaded_vision:
            print(f"[FSQ-v3] initialized {loaded_vision} SigLIP tensors from {cfg.pi_base}")
    if not cfg.save_best_model and save_path.is_file():
        print(
            f"[FSQ-v3] best-model saving is disabled; existing {save_path} "
            "will be left untouched"
        )

    # Older checkpoints predate assignment tracking. Establish their loaded
    # epoch as the baseline now so the first post-resume validation still emits
    # a meaningful retention/change measurement.
    if resume_from and previous_code_assignments is None:
        model.eval()
        previous_code_assignments, _ = _collect_code_assignments(
            model,
            (val_ds, train_ds),
            device,
            cfg.batch_size,
        )
        previous_code_epoch = start_epoch - 1
        print(
            "[FSQ-v3] established code-assignment baseline from loaded epoch "
            f"{previous_code_epoch} ({previous_code_assignments.numel()} skills)"
        )

    if wandb_run is not None:
        # The same epoch aggregate is reported twice: train/val use optimizer_step as x, while
        # train_epoch/val_epoch use epoch as x.  Keeping train and val at the top-level avoids
        # collapsing every curve into one "epoch" workspace section.
        wandb_run.define_metric("epoch")
        wandb_run.define_metric("optimizer_step")
        for name in ("train/*", "val/*", "perf/*", "lr/*"):
            wandb_run.define_metric(name, step_metric="optimizer_step")
        wandb_run.define_metric("codebook/*", step_metric="optimizer_step")
        for name in (
            "train_epoch/*",
            "val_epoch/*",
            "perf_epoch/*",
            "lr_epoch/*",
            "codebook_epoch/*",
        ):
            wandb_run.define_metric(name, step_metric="epoch")
        if init_calibration_metrics:
            wandb_run.log(
                {
                    "epoch": 0,
                    "optimizer_step": 0,
                    **init_calibration_metrics,
                }
            )

    def save(path: str | Path, epoch: int, val: float, select: float, *, resumable: bool) -> None:
        payload = {
            "format_version": FORMAT_VERSION,
            "cfg": cfg,
            "model_state": model.state_dict(),
            "epoch": epoch,
            "val_loss": val,
            "val_select": select,
            "best_val": best_val,
        }
        # FSQ.pt is copied into every SkillVLA data run and only needs component weights.
        # Periodic checkpoints retain optimizer/scheduler state for exact resume.
        if resumable:
            payload["optim_state"] = optimizer.state_dict()
            payload["scheduler_state"] = scheduler.state_dict()
            if previous_code_assignments is not None and previous_code_epoch is not None:
                # A compact baseline lets resumed jobs continue the retention curve instead of
                # spending their first validation point only establishing a new reference.
                payload["code_assignments"] = previous_code_assignments.to(torch.int32)
                payload["code_assignments_epoch"] = previous_code_epoch
                payload["code_assignments_fingerprint"] = assignment_fingerprint
        torch.save(payload, str(path))

    def step(
        batch: dict[str, Tensor | None],
        training: bool,
        batch_index: int,
        epoch: int,
    ):
        moved = {k: (v.to(device, non_blocking=True) if isinstance(v, Tensor) else v) for k, v in batch.items()}
        positive_pair_key = (
            "augmented_action_seq"
            if cfg.encoder_arch == "action_seq"
            else "augmented_ctrl"
        )
        negative_pair_key = (
            "negative_action_seq"
            if cfg.encoder_arch == "action_seq"
            else "negative_ctrl"
        )
        if training and cfg.pair_loss != "none" and positive_pair_key not in moved:
            raise RuntimeError("Training batch is missing the configured FSQ augmentation pair.")
        if training and cfg.pair_loss == "contrastive" and negative_pair_key not in moved:
            raise RuntimeError(
                "Training batch is missing the configured adjacent-skill negative pair."
            )
        bsize = moved["ctrl"].shape[0]
        m = cfg.samples_per_skill
        start_state = moved["start_state"].reshape(bsize * m, cfg.max_state_dim)
        raw_state = moved["raw_state"].reshape(bsize * m, cfg.state_dim)
        prev_action = moved["prev_action"].reshape(bsize * m, cfg.action_dim)
        third = wrist = None
        if "third" in moved:
            third = moved["third"].reshape(bsize * m, *moved["third"].shape[2:])
            wrist = moved["wrist"].reshape(bsize * m, *moved["wrist"].shape[2:])
        noise = time = None
        if not training:
            generator = torch.Generator(device=device).manual_seed(10_000 + batch_index)
            noise = torch.randn(
                (bsize * m, cfg.chunk_size, cfg.max_action_dim),
                generator=generator,
                device=device,
                dtype=moved["actions"].dtype,
            )
            time = torch.full((bsize * m,), 0.5, device=device)
        effective_pair_weight = (
            fsq_pair_weight_at_epoch(
                cfg.pair_weight,
                epoch,
                cfg.pair_warmup_epochs,
                cfg.pair_ramp_epochs,
                enabled=cfg.pair_warmup,
            )
            if training and cfg.pair_loss != "none"
            else 0.0
        )
        compute_pair_forward = effective_pair_weight > 0.0
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                ctrl=moved["ctrl"],
                lengths=moved["length"],
                start_pose=moved.get("start_pose"),
                start_state=start_state,
                raw_state=raw_state,
                prev_action=prev_action,
                progress_target=moved["progress"].reshape(bsize * m),
                third=third,
                wrist=wrist,
                samples_per_skill=m,
                terminator_context_sequence=moved.get("terminator_context_sequence"),
                noise=noise,
                time=time,
                action_seq=moved.get("encoder_action_seq"),
                augmented_ctrl=(
                    moved.get("augmented_ctrl") if compute_pair_forward else None
                ),
                augmented_action_seq=(
                    moved.get("augmented_action_seq") if compute_pair_forward else None
                ),
                augmented_lengths=(
                    moved.get("augmented_length") if compute_pair_forward else None
                ),
                augmented_start_pose=(
                    moved.get("augmented_start_pose") if compute_pair_forward else None
                ),
                negative_ctrl=(
                    moved.get("negative_ctrl") if compute_pair_forward else None
                ),
                negative_action_seq=(
                    moved.get("negative_action_seq") if compute_pair_forward else None
                ),
                negative_lengths=(
                    moved.get("negative_length") if compute_pair_forward else None
                ),
                negative_start_pose=(
                    moved.get("negative_start_pose") if compute_pair_forward else None
                ),
                # Diagnostic only: the visual frontend is shared between the
                # true/shuffled passes, so validation adds one lightweight
                # terminator/fusion pass but never a second ResNet/DINO call.
                compute_skill_shuffle=not training,
            )
            loss, metrics = fsq_reconstruction_loss(
                output,
                moved,
                cfg,
                pair_weight=effective_pair_weight,
            )
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        if cfg.reconstructor_only or not cfg.terminator_termination:
            end_metrics = {}
        elif cfg.state_rnn_terminator:
            positions = torch.arange(
                output["term_logits"].shape[1], device=device
            )[None]
            valid = positions < moved["length"][:, None]
            end_metrics = end_signal_metrics(
                output["term_logits"].detach()[valid],
                moved["terminator_termination"][valid],
                cfg.end_threshold,
            )
        else:
            end_metrics = end_signal_metrics(
                output["term_logits"].detach(),
                moved["termination"].reshape(-1),
                cfg.end_threshold,
            )
        # One FSQ index per input skill is already produced by the encoder. Keep
        # it on-device so epoch-level codebook coverage costs only a boolean
        # scatter, not an additional encode pass or per-batch CPU synchronization.
        return {k: float(v) for k, v in metrics.items()}, end_metrics, bsize, output["indices"].detach()

    # W&B previously received ``step=epoch``, which made one full FSQ epoch look like one
    # optimizer step and therefore incomparable with VLA training dashboards.
    global_step = (start_epoch - 1) * len(train_loader)
    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_sum: dict[str, float] = {}
        train_end: dict[str, float] = {}
        train_count = 0
        train_codes_seen = torch.zeros(model.fsq.codebook_size, dtype=torch.bool, device=device)
        for batch_index, batch in enumerate(train_loader):
            metrics, end_metrics, count, code_indices = step(
                batch, True, batch_index, epoch
            )
            global_step += 1
            train_count += count
            train_codes_seen[code_indices.reshape(-1).long()] = True
            for key, value in metrics.items():
                train_sum[key] = train_sum.get(key, 0.0) + value * count
            for key, value in end_metrics.items():
                train_end[key] = train_end.get(key, 0.0) + value * count
        scheduler.step()

        should_validate = cfg.val_every > 0 and epoch % cfg.val_every == 0
        val_sum: dict[str, float] = {}
        val_end: dict[str, float] = {}
        val_count = 0
        val_codes_seen = torch.zeros(model.fsq.codebook_size, dtype=torch.bool, device=device)
        assignment_metrics: dict[str, float] = {}
        boundary_margin_metrics: dict[str, float] = {}
        assignment_reference_epoch: int | None = None
        full_active_codes = 0
        if should_validate:
            model.eval()
            with torch.no_grad():
                for batch_index, batch in enumerate(val_loader):
                    metrics, end_metrics, count, code_indices = step(
                        batch, False, batch_index, epoch
                    )
                    val_count += count
                    val_codes_seen[code_indices.reshape(-1).long()] = True
                    for key, value in metrics.items():
                        val_sum[key] = val_sum.get(key, 0.0) + value * count
                    for key, value in end_metrics.items():
                        val_end[key] = val_end.get(key, 0.0) + value * count

            # Evaluate sample-to-code membership at one fixed model state. This pass touches only
            # the cached spline inputs; it does not decode images or rerun the reconstructor.
            current_code_assignments, current_boundary_margins = _collect_code_assignments(
                model,
                (val_ds, train_ds),
                device,
                cfg.batch_size,
            )
            boundary_margin_metrics = _boundary_margin_metrics(current_boundary_margins)
            full_active_codes = int(current_code_assignments.unique().numel())
            if previous_code_assignments is not None and previous_code_epoch is not None:
                assignment_reference_epoch = previous_code_epoch
                assignment_metrics = _code_assignment_stability(
                    previous_code_assignments,
                    current_code_assignments,
                    model.fsq.codebook_size,
                    cfg.fsq_levels,
                )
                assignment_metrics["interval_epochs"] = float(epoch - previous_code_epoch)
            previous_code_assignments = current_code_assignments
            previous_code_epoch = epoch

        train_avg = {k: v / max(train_count, 1) for k, v in train_sum.items()}
        val_avg = {k: v / max(val_count, 1) for k, v in val_sum.items()}
        train_end_avg = {k: v / max(train_count, 1) for k, v in train_end.items()}
        val_end_avg = {k: v / max(val_count, 1) for k, v in val_end.items()}
        train_active_codes = int(train_codes_seen.count_nonzero().item())
        val_active_codes = int(val_codes_seen.count_nonzero().item())
        codebook_size = model.fsq.codebook_size
        select = math.nan
        if should_validate:
            select = (
                (cfg.val_select_action_weight if cfg.val_select_action_weight is not None else cfg.action_loss_weight)
                * val_avg["action"]
                + (cfg.val_select_progress_weight if cfg.val_select_progress_weight is not None else cfg.progress_loss_weight)
                * val_avg["progress"]
                + (cfg.val_select_end_weight if cfg.val_select_end_weight is not None else cfg.end_loss_weight)
                * val_avg["termination"]
            )
            if cfg.save_best_model and select < best_val:
                best_val = select
                save(save_path, epoch, val_avg["loss"], select, resumable=False)
        if cfg.checkpoint_every and epoch % cfg.checkpoint_every == 0:
            save(
                save_path.with_name(f"FSQ_epoch{epoch:04d}.pt"),
                epoch,
                val_avg.get("loss", math.nan),
                select,
                resumable=True,
            )

        log = {
            "epoch": epoch,
            "optimizer_step": global_step,
            "perf/seconds": time.perf_counter() - epoch_start,
            "perf/updates_per_sec": len(train_loader) / max(time.perf_counter() - epoch_start, 1e-8),
            **{f"train/{k}": v for k, v in train_avg.items()},
            **{f"train/end_{k}": v for k, v in train_end_avg.items()},
            "train/codebook_utilization_pct": 100.0 * train_active_codes / codebook_size,
            "train/codebook_active_entries": train_active_codes,
            **{f"lr/{group['name']}": group["lr"] for group in optimizer.param_groups},
        }
        log.update({f"train_epoch/{k}": v for k, v in train_avg.items()})
        log.update({f"train_epoch/end_{k}": v for k, v in train_end_avg.items()})
        log.update({
            "train_epoch/codebook_utilization_pct": log["train/codebook_utilization_pct"],
            "train_epoch/codebook_active_entries": train_active_codes,
            "perf_epoch/seconds": log["perf/seconds"],
            "perf_epoch/updates_per_sec": log["perf/updates_per_sec"],
            **{
                f"lr_epoch/{group['name']}": group["lr"]
                for group in optimizer.param_groups
            },
        })
        if should_validate:
            log.update({f"val/{k}": v for k, v in val_avg.items()})
            log.update({f"val/end_{k}": v for k, v in val_end_avg.items()})
            log.update({f"val_epoch/{k}": v for k, v in val_avg.items()})
            log.update({f"val_epoch/end_{k}": v for k, v in val_end_avg.items()})
            log.update({
                "val/codebook_utilization_pct": 100.0 * val_active_codes / codebook_size,
                "val/codebook_active_entries": val_active_codes,
                "val/select": select,
                "val_epoch/codebook_utilization_pct": 100.0 * val_active_codes / codebook_size,
                "val_epoch/codebook_active_entries": val_active_codes,
                "val_epoch/select": select,
            })
            full_codebook_log = {
                "full_active_entries": full_active_codes,
                "full_utilization_pct": 100.0 * full_active_codes / codebook_size,
                **boundary_margin_metrics,
                **assignment_metrics,
            }
            log.update({f"codebook/{key}": value for key, value in full_codebook_log.items()})
            log.update({f"codebook_epoch/{key}": value for key, value in full_codebook_log.items()})
        if wandb_run is not None:
            wandb_run.log(log, step=global_step)
        if epoch == 1 or epoch % cfg.log_every == 0 or should_validate:
            message = (
                f"[FSQ-v3] {epoch:4d}/{cfg.epochs} "
                f"train={train_avg['loss']:.4f}"
            )
            if "pair_weight" in train_avg:
                message += f" pair-w={train_avg['pair_weight']:.6g}"
            if should_validate:
                message += (
                    f" val={val_avg['loss']:.4f} action={val_avg['action']:.4f} "
                    f"prog={val_avg['progress']:.4f} "
                    f"end={val_avg['termination']:.4f} select={select:.4f}"
                )
                if "skill_shuffle_mean_delta" in val_avg:
                    message += (
                        " skill-shuffle="
                        f"{val_avg['skill_shuffle_mean_delta']:.4f}"
                    )
                if assignment_metrics:
                    message += (
                        f" code-retain({assignment_reference_epoch}->{epoch})="
                        f"{assignment_metrics['retention_pct']:.1f}% "
                        f"changed={assignment_metrics['change_pct']:.1f}% "
                        f"axis-changed={assignment_metrics['per_axis_change_pct']:.1f}% "
                        f"matched={assignment_metrics['matched_retention_pct']:.1f}%"
                    )
                else:
                    message += f" code-retain=baseline({full_active_codes}/{codebook_size} active)"
                message += (
                    " boundary-margin="
                    f"{boundary_margin_metrics['boundary_margin_mean_pct']:.1f}% "
                    f"p10={boundary_margin_metrics['boundary_margin_p10_pct']:.1f}% "
                    f"near={boundary_margin_metrics['near_boundary_pct']:.1f}% "
                    "axis-near="
                    f"{boundary_margin_metrics['per_axis_near_boundary_pct']:.1f}%"
                )
            print(message)

    if cfg.save_best_model:
        print(f"[FSQ-v3] done; best val-select={best_val:.6f} -> {save_path}")
    else:
        print(f"[FSQ-v3] done; periodic checkpoints -> {save_path.parent}")
    return model
