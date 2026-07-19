"""Trajectory FSQ with a Stage-1-compatible VSA flow expert and query terminator.

The training unit is a skill trajectory, not a frame:

    B complete skill trajectories -> spline encoder once -> z_q (B, D)
    M sampled timesteps / trajectory -> VSA expert + terminator (B*M)

The action expert is deliberately image-free here.  It is the same Gemma/AdaRMS
core used by Stage-1; Stage-1 later attaches the image condition stream through
attention.  The terminator is the only image consumer.  Its two learned queries
read unpooled DINO or SigLIP tokens from both cameras, while image tokens never
read either query. Raw state and normalized z_q modulate only the query branch.

Checkpoint format v3 has three explicit, independently loadable components:

    encoder.*        trajectory -> FSQ code
    action_expert.*  (state, z_q) -> flow-matching action chunk
    terminator.*     (image, raw state, z_q) -> progress/end

There is intentionally no legacy dense (B,T,...) reconstructor fallback.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import make_interp_spline
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel
from transformers.models.auto import CONFIG_MAPPING

from lerobot.policies.pi05.modeling_pi05 import (
    OPENPI_ATTENTION_MASK_VALUE,
    create_sinusoidal_pos_embedding,
    get_gemma_config,
    make_att_2d_masks,
    sample_beta,
)
from lerobot.policies.pi_gemma import PiGemmaForCausalLM


FORMAT_VERSION = 3
N_GRIPPER_DIMS = 2
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_IMAGE_MODEL = str(_REPO_ROOT / "models" / "dinov3-vits16")
_DEFAULT_PI_BASE = str(_REPO_ROOT / "models" / "pi05_base")


# -----------------------------------------------------------------------------
# Spline trajectory encoder
# -----------------------------------------------------------------------------


def zero_ground_trajectory(trajectory: np.ndarray) -> np.ndarray:
    """Ground pose dimensions at the skill start; retain absolute gripper state."""
    traj = np.asarray(trajectory, dtype=np.float32).copy()
    if len(traj) == 0:
        raise ValueError("Cannot encode an empty skill trajectory.")
    if N_GRIPPER_DIMS >= traj.shape[-1]:
        raise ValueError(
            f"Expected more than {N_GRIPPER_DIMS} state dimensions, got {traj.shape[-1]}."
        )
    offset = traj[0].copy()
    offset[-N_GRIPPER_DIMS:] = 0.0
    return traj - offset


def encoder_start_eef_pose(trajectory: np.ndarray) -> np.ndarray:
    """Absolute skill-start EEF pose(s), excluding trailing gripper-state dimensions."""
    traj = np.asarray(trajectory, dtype=np.float32)
    if len(traj) == 0:
        raise ValueError("Cannot encode an empty skill trajectory.")
    if N_GRIPPER_DIMS >= traj.shape[-1]:
        raise ValueError(
            f"Expected more than {N_GRIPPER_DIMS} state dimensions, got {traj.shape[-1]}."
        )
    return traj[0, :-N_GRIPPER_DIMS].copy()


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
    if input_mode == "raw_state":
        return traj.copy()
    raise ValueError(
        f"encoder_input_mode must be zero_grounded|raw_state|optimal, got {input_mode!r}."
    )


def spline_encode(
    trajectory: np.ndarray,
    n_control: int,
    degree: int,
    *,
    input_mode: str = "zero_grounded",
) -> tuple[np.ndarray, int]:
    """Trajectory -> fixed control points and original skill length."""
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
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.shape[1] != self.n_tokens:
            raise ValueError(f"Expected {self.n_tokens} tokens, got {tokens.shape[1]}.")
        x = self.encoder(tokens + self.pos_embed.to(tokens.dtype))
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

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        half = self.levels_half.to(z.dtype)
        bounded = torch.tanh(z + self.shift.to(z.dtype)) * half - self.offset.to(z.dtype)
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
    ):
        super().__init__()
        if encoder_input_mode not in {"zero_grounded", "raw_state", "optimal"}:
            raise ValueError(
                "encoder_input_mode must be zero_grounded|raw_state|optimal, "
                f"got {encoder_input_mode!r}."
            )
        self.enc_dim = int(enc_dim)
        self.n_control = int(n_control)
        self.spline_degree = int(spline_degree)
        self.encoder_input_mode = encoder_input_mode
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
            start_dim = enc_dim - N_GRIPPER_DIMS
            self.enc_start_proj = nn.Linear(start_dim, hidden_dim)
            self.register_buffer("encoder_start_min", torch.as_tensor(encoder_start_min, dtype=torch.float32))
            self.register_buffer("encoder_start_max", torch.as_tensor(encoder_start_max, dtype=torch.float32))
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
        return 2.0 * (ctrl - lo) / (hi - lo + 1e-8) - 1.0

    def normalize_start_pose(self, start_pose: Tensor) -> Tensor:
        if self.enc_start_proj is None:
            raise ValueError("Only optimal encoder mode accepts a start EEF pose.")
        lo = self.encoder_start_min.to(start_pose.device, start_pose.dtype)
        hi = self.encoder_start_max.to(start_pose.device, start_pose.dtype)
        return 2.0 * (start_pose - lo) / (hi - lo + 1e-8) - 1.0

    def forward(
        self,
        ctrl: Tensor,
        lengths: Tensor,
        start_pose: Tensor | None = None,
        *,
        normalized: bool = True,
    ) -> tuple[Tensor, Tensor]:
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
                raise ValueError("optimal encoder mode requires start_pose on every forward pass.")
            tokens.append(self.enc_start_proj(start_pose).unsqueeze(1))
        tokens.append(length_tok)
        z_e = self.z_head(self.enc_traj_pool(torch.cat(tokens, dim=1)))
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
                encoder_start_eef_pose(trajectory)
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
                encoder_start_eef_pose(trajectory)
            ).float().unsqueeze(0).to(device)
        _, index = self(ctrl_t, length_t, start_t, normalized=False)
        return int(index.item())


# -----------------------------------------------------------------------------
# Stage-1-compatible image-free VSA action expert
# -----------------------------------------------------------------------------


def _build_gemma(variant: str, *, use_adarms: bool = True) -> PiGemmaForCausalLM:
    cfg = get_gemma_config(variant)
    hf = CONFIG_MAPPING["gemma"](
        head_dim=cfg.head_dim,
        hidden_size=cfg.width,
        intermediate_size=cfg.mlp_dim,
        num_attention_heads=cfg.num_heads,
        num_hidden_layers=cfg.depth,
        num_key_value_heads=cfg.num_kv_heads,
        vocab_size=257152,
        hidden_activation="gelu_pytorch_tanh",
        dtype="float32",
        use_adarms=use_adarms,
        adarms_cond_dim=cfg.width if use_adarms else None,
    )
    model = PiGemmaForCausalLM(config=hf)
    model.model.embed_tokens = None
    model.lm_head = None
    model.model.config._attn_implementation = "eager"  # noqa: SLF001
    return model


class VSAFlowExpert(nn.Module):
    """Image-free action stream reused as the Stage-1/2 action expert warm start."""

    def __init__(
        self,
        *,
        variant: str,
        fsq_levels: list[int],
        state_cond_mode: str,
        max_state_dim: int,
        max_action_dim: int,
        chunk_size: int,
        min_period: float,
        max_period: float,
        time_sampling_beta_alpha: float,
        time_sampling_beta_beta: float,
        time_sampling_scale: float,
        time_sampling_offset: float,
    ):
        super().__init__()
        if state_cond_mode not in {"state", "state_skill", "broadcast"}:
            raise ValueError(
                "state_cond_mode must be state|state_skill|broadcast, "
                f"got {state_cond_mode!r}."
            )
        self.variant = variant
        self.state_cond_mode = state_cond_mode
        self.max_state_dim = int(max_state_dim)
        self.max_action_dim = int(max_action_dim)
        self.chunk_size = int(chunk_size)
        self.min_period = float(min_period)
        self.max_period = float(max_period)
        self.time_sampling_beta_alpha = float(time_sampling_beta_alpha)
        self.time_sampling_beta_beta = float(time_sampling_beta_beta)
        self.time_sampling_scale = float(time_sampling_scale)
        self.time_sampling_offset = float(time_sampling_offset)
        self.width = get_gemma_config(variant).width

        self.state_proj = nn.Linear(max_state_dim, self.width)
        self.skill_proj = nn.Linear(len(fsq_levels), self.width)
        self.action_in_proj = nn.Linear(max_action_dim, self.width)
        self.action_out_proj = nn.Linear(self.width, max_action_dim)
        self.time_mlp_in = nn.Linear(self.width, self.width)
        self.time_mlp_out = nn.Linear(self.width, self.width)
        self.gemma_expert = _build_gemma(variant)

    @property
    def working_dtype(self) -> torch.dtype:
        return self.action_in_proj.weight.dtype

    def gradient_checkpointing_enable(self) -> None:
        self.gemma_expert.gradient_checkpointing_enable()

    def sample_noise(self, shape: tuple[int, ...], device: torch.device) -> Tensor:
        return torch.randn(shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize: int, device: torch.device) -> Tensor:
        beta = sample_beta(
            self.time_sampling_beta_alpha, self.time_sampling_beta_beta, bsize, device
        )
        return (beta * self.time_sampling_scale + self.time_sampling_offset).float()

    def _time_cond(self, time: Tensor) -> Tensor:
        t = create_sinusoidal_pos_embedding(
            time, self.width, self.min_period, self.max_period, device=time.device
        ).to(self.working_dtype)
        return F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(t))))

    def _expert_cond(self, time: Tensor, state: Tensor, z_norm: Tensor) -> Tensor:
        cond = self._time_cond(time) + self.state_proj(state.to(self.working_dtype))
        if self.state_cond_mode == "state_skill":
            cond = cond + self.skill_proj(z_norm.to(self.working_dtype))
        return cond

    def _action_prefix(self, z_norm: Tensor) -> Tensor | None:
        if self.state_cond_mode != "state":
            return None
        return self.skill_proj(z_norm.to(self.working_dtype)).unsqueeze(1)

    def _skill_broadcast(self, z_norm: Tensor) -> Tensor | None:
        if self.state_cond_mode != "broadcast":
            return None
        return self.skill_proj(z_norm.to(self.working_dtype))

    def velocity(self, x_t: Tensor, time: Tensor, state: Tensor, z_norm: Tensor) -> Tensor:
        action = self.action_in_proj(x_t.to(self.working_dtype))
        prefix = self._action_prefix(z_norm)
        n_prefix = 0 if prefix is None else prefix.shape[1]
        if prefix is not None:
            action = torch.cat([prefix, action], dim=1)

        bsize, n_tokens = action.shape[:2]
        pad = torch.ones(bsize, n_tokens, dtype=torch.bool, device=action.device)
        ar = ([1] + [0] * (n_prefix - 1)) if n_prefix else []
        ar += [1] + [0] * (self.chunk_size - 1)
        block = torch.tensor(ar, dtype=torch.bool, device=action.device)[None].expand(bsize, -1)
        attn = torch.where(
            make_att_2d_masks(pad, block)[:, None], 0.0, OPENPI_ATTENTION_MASK_VALUE
        )
        positions = torch.arange(n_tokens, device=action.device)[None].expand(bsize, -1)
        hidden = self.gemma_expert.model(
            inputs_embeds=action,
            attention_mask=attn,
            position_ids=positions,
            use_cache=False,
            adarms_cond=self._expert_cond(time, state, z_norm),
            broadcast_cond=self._skill_broadcast(z_norm),
        ).last_hidden_state
        return self.action_out_proj(hidden[:, -self.chunk_size :].to(self.working_dtype)).float()

    def flow_residual(
        self,
        state: Tensor,
        z_norm: Tensor,
        actions: Tensor,
        *,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        bsize = actions.shape[0]
        if noise is None:
            noise = self.sample_noise(tuple(actions.shape), actions.device)
        if time is None:
            time = self.sample_time(bsize, actions.device)
        source = noise.to(actions.dtype)
        x_t = time[:, None, None] * source + (1.0 - time[:, None, None]) * actions
        target_velocity = source - actions
        return target_velocity - self.velocity(x_t, time, state, z_norm)

    @torch.no_grad()
    def sample_actions(
        self,
        state: Tensor,
        z_norm: Tensor,
        *,
        noise: Tensor | None = None,
        num_steps: int = 10,
    ) -> Tensor:
        bsize = state.shape[0]
        if noise is None:
            noise = self.sample_noise(
                (bsize, self.chunk_size, self.max_action_dim), state.device
            )
        x_t = noise
        dt = -1.0 / num_steps
        for step in range(num_steps):
            time = torch.full((bsize,), 1.0 + step * dt, device=state.device)
            x_t = x_t + dt * self.velocity(x_t, time, state, z_norm)
        return x_t


def initialize_action_expert_from_pi05(expert: VSAFlowExpert, pretrained: str | Path) -> int:
    """Selectively load only PI05 action-expert/action-time tensors."""
    from safetensors import safe_open

    path = Path(pretrained)
    if path.is_dir():
        path = path / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"PI05 model.safetensors not found: {path}")

    mapped: dict[str, Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            plain = key[len("model.") :] if key.startswith("model.") else key
            gemma_prefix = "paligemma_with_expert.gemma_expert."
            if plain.startswith(gemma_prefix):
                rest = plain[len(gemma_prefix) :]
                if not rest.startswith("lm_head"):
                    mapped[f"gemma_expert.{rest}"] = handle.get_tensor(key)
            elif plain.startswith(("action_in_proj.", "action_out_proj.", "time_mlp_in.", "time_mlp_out.")):
                mapped[plain] = handle.get_tensor(key)
            elif plain.startswith("action_time_mlp_in."):
                mapped["time_mlp_in." + plain[len("action_time_mlp_in.") :]] = handle.get_tensor(key)
            elif plain.startswith("action_time_mlp_out."):
                mapped["time_mlp_out." + plain[len("action_time_mlp_out.") :]] = handle.get_tensor(key)

    missing, unexpected = expert.load_state_dict(mapped, strict=False)
    allowed_missing = {
        "state_proj.weight", "state_proj.bias", "skill_proj.weight", "skill_proj.bias"
    }
    bad_missing = sorted(k for k in missing if k not in allowed_missing)
    if bad_missing or unexpected:
        raise RuntimeError(
            f"PI05 expert initialization mismatch: missing={bad_missing}, unexpected={sorted(unexpected)}"
        )
    return len(mapped)


def initialize_terminator_vision_from_pi05(
    terminator: "FSQQueryTerminator", pretrained: str | Path
) -> int:
    """Warm-start SigLIP from the same PI05 vision tower used by Stage-1.

    DINO already initializes itself from ``dino_model_path`` and therefore has
    no PI05 mapping. The projection remains task-specific and is learned by FSQ.
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


class QueryTerminatorLayer(nn.Module):
    """Images self-attend; each output query reads images+self, never the other query."""

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.image_norm1 = nn.RMSNorm(hidden_dim)
        self.query_norm1 = ConditionalRMSNorm(hidden_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.image_norm2 = nn.RMSNorm(hidden_dim)
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

    def forward(self, x: Tensor, n_image: int, cond: Tensor, disallow: Tensor) -> Tensor:
        normed = self._norm(x, n_image, cond, second=False)
        attn, _ = self.attention(normed, normed, normed, attn_mask=disallow, need_weights=False)
        x = x + self.dropout(attn)
        x = x + self.dropout(self.ffn(self._norm(x, n_image, cond, second=True)))
        return x


def resolve_image_model_path(name: str) -> str:
    path = Path(name)
    if not path.is_absolute() or path.exists():
        return name
    local = _REPO_ROOT / "models" / path.name
    return str(local) if local.exists() else name


def _build_siglip_vision_tower(image_size: int) -> nn.Module:
    """Build the exact SigLIP tower used by Stage-1's condition stream."""
    from transformers import SiglipVisionModel

    vlm_cfg = CONFIG_MAPPING["paligemma"]()
    vision_cfg = vlm_cfg.vision_config
    vision_cfg.image_size = int(image_size)
    vision_cfg.intermediate_size = 4304
    vision_cfg.projection_dim = 2048
    vision_cfg.projector_hidden_act = "gelu_fast"
    return SiglipVisionModel(vision_cfg)


class FSQQueryTerminator(nn.Module):
    """Live third+wrist images + state/z AdaRMS -> progress and termination.

    The visual frontend and token contract intentionally match Stage-1 cond:
    one shared DINO or SigLIP tower is applied to both cameras, no spatial
    pooling is performed, and both token sequences share one ``image_proj``.

    ``arch='small'`` uses a lightweight query transformer. ``arch='cond'``
    uses an AdaRMS Gemma whose module name is ``cond_encoder`` so its backbone
    remains directly identifiable for a later Stage-1 cond warm start.
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
        cond_encoder_variant: str,
        state_min: np.ndarray,
        state_max: np.ndarray,
    ):
        super().__init__()
        if arch not in {"small", "cond"}:
            raise ValueError(f"terminator_arch must be small|cond, got {arch!r}.")
        if vision_backbone not in {"dino", "siglip"}:
            raise ValueError(f"vision_backbone must be dino|siglip, got {vision_backbone!r}.")
        if arch == "small" and hidden_dim % n_heads:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by n_heads={n_heads}.")
        self.state_dim = int(state_dim)
        self.fsq_levels = [int(x) for x in fsq_levels]
        self.arch = arch
        self.vision_backbone = vision_backbone
        self.freeze_vision_encoder = bool(freeze_vision_encoder)
        self.dino_model_path = resolve_image_model_path(dino_model_path)
        self.dino_image_size = int(dino_image_size)
        self.siglip_image_size = int(siglip_image_size)
        self.cond_encoder_variant = cond_encoder_variant

        self.dino = None
        self.siglip = None
        self.n_register = 0
        if vision_backbone == "dino":
            self.dino = AutoModel.from_pretrained(self.dino_model_path)
            visual_dim = int(self.dino.config.hidden_size)
            self.n_register = int(getattr(self.dino.config, "num_register_tokens", 0))
            self.vision_image_size = self.dino_image_size
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        else:
            self.siglip = _build_siglip_vision_tower(self.siglip_image_size)
            visual_dim = int(self.siglip.config.hidden_size)
            self.vision_image_size = self.siglip_image_size
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        if self.freeze_vision_encoder:
            for parameter in self.vision_encoder.parameters():
                parameter.requires_grad_(False)
        self.register_buffer("_img_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_img_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

        width = int(hidden_dim) if arch == "small" else int(get_gemma_config(cond_encoder_variant).width)
        self.hidden_dim = width
        self.image_proj = nn.Linear(visual_dim, width)
        self.progress_query = nn.Parameter(torch.zeros(1, 1, width))
        self.termination_query = nn.Parameter(torch.zeros(1, 1, width))
        self.state_proj = nn.Linear(state_dim, width)
        self.skill_proj = nn.Linear(len(fsq_levels), width)
        if arch == "small":
            self.layers = nn.ModuleList(
                [QueryTerminatorLayer(width, n_heads, dropout) for _ in range(n_layers)]
            )
            self.cond_encoder = None
            self.query_in_norm = None
            self.query_out_norm = ConditionalRMSNorm(width, width)
        else:
            self.layers = nn.ModuleList()
            # Keep the Gemma itself byte-for-byte compatible with Stage-1 cond
            # (plain RMSNorm). State+skill AdaRMS lives only on the two query
            # tokens, before and after Gemma, so image tokens remain image-only.
            self.cond_encoder = _build_gemma(cond_encoder_variant, use_adarms=False)
            self.query_in_norm = ConditionalRMSNorm(width, width)
            self.query_out_norm = ConditionalRMSNorm(width, width)
        self.progress_head = nn.Linear(width, 1)
        self.termination_head = nn.Linear(width, 1)
        self.register_buffer("state_min", torch.as_tensor(state_min, dtype=torch.float32))
        self.register_buffer("state_max", torch.as_tensor(state_max, dtype=torch.float32))
        nn.init.trunc_normal_(self.progress_query, std=0.02)
        nn.init.trunc_normal_(self.termination_query, std=0.02)

    @property
    def vision_encoder(self) -> nn.Module:
        return self.dino if self.dino is not None else self.siglip

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.vision_encoder.eval()
        return self

    def gradient_checkpointing_enable(self) -> None:
        if self.cond_encoder is not None:
            self.cond_encoder.gradient_checkpointing_enable()
        if not self.freeze_vision_encoder and hasattr(self.vision_encoder, "gradient_checkpointing_enable"):
            self.vision_encoder.gradient_checkpointing_enable()

    def _normalize_state(self, state: Tensor) -> Tensor:
        lo = self.state_min.to(state.device, state.dtype)
        hi = self.state_max.to(state.device, state.dtype)
        return 2.0 * (state[..., : self.state_dim] - lo) / (hi - lo + 1e-8) - 1.0

    def _image_features(self, image: Tensor | None) -> Tensor:
        if image is None:
            raise ValueError("FSQ terminator always requires both third-person and wrist images.")
        if image.ndim != 4:
            raise ValueError(f"Terminator image must be (B,C,H,W) or (B,H,W,C), got {tuple(image.shape)}")
        if image.shape[-1] in (1, 3):
            image = image.permute(0, 3, 1, 2)
        if image.shape[1] == 1:
            image = image.expand(-1, 3, -1, -1)
        x = image.float()
        if x.numel() and float(x.detach().amin()) < -0.05:
            x = (x + 1.0) / 2.0
        if x.numel() and float(x.detach().amax()) > 2.0:
            x = x / 255.0
        x = F.interpolate(
            x.clamp(0.0, 1.0),
            size=(self.vision_image_size, self.vision_image_size),
            mode="bilinear",
            align_corners=False,
        )
        x = (x - self._img_mean.float()) / self._img_std.float()
        if self.vision_backbone == "dino":
            x = x.to(dtype=next(self.dino.parameters()).dtype)
            output = self.dino(x).last_hidden_state
            cls = output[:, :1]
            patches = output[:, 1 + self.n_register :]
            return torch.cat([cls, patches], dim=1)
        x = x.to(dtype=next(self.siglip.parameters()).dtype)
        return self.siglip(pixel_values=x).last_hidden_state

    def _prepare_image_tokens(self, third: Tensor | None, wrist: Tensor | None) -> Tensor:
        # Exactly Stage-1 cond's visual contract: shared backbone, shared projection, concatenation.
        tokens = [self._image_features(third), self._image_features(wrist)]
        return torch.cat(
            [self.image_proj(value.to(self.image_proj.weight.dtype)) for value in tokens], dim=1
        )

    @staticmethod
    def _allow_mask(n_image: int, device: torch.device) -> Tensor:
        total = n_image + 2
        allow = torch.zeros(total, total, dtype=torch.bool, device=device)
        allow[:n_image, :n_image] = True
        allow[n_image, :n_image] = True
        allow[n_image, n_image] = True
        allow[n_image + 1, :n_image] = True
        allow[n_image + 1, n_image + 1] = True
        return allow

    def forward(
        self,
        z_norm: Tensor,
        raw_state: Tensor,
        third: Tensor | None,
        wrist: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        image_tokens = self._prepare_image_tokens(third, wrist)
        bsize, n_image = image_tokens.shape[:2]
        queries = torch.cat(
            [self.progress_query.expand(bsize, -1, -1), self.termination_query.expand(bsize, -1, -1)],
            dim=1,
        ).to(image_tokens.dtype)
        cond = self.state_proj(self._normalize_state(raw_state).to(self.state_proj.weight.dtype))
        cond = cond + self.skill_proj(z_norm.to(self.skill_proj.weight.dtype))
        if self.arch == "cond":
            queries = self.query_in_norm(queries, cond)
        x = torch.cat([image_tokens, queries], dim=1)
        allow = self._allow_mask(n_image, x.device)
        if self.arch == "small":
            for layer in self.layers:
                x = layer(x, n_image, cond, ~allow)
            query_out = self.query_out_norm(x[:, n_image:], cond)
        else:
            attention = torch.where(
                allow[None, None],
                torch.tensor(0.0, device=x.device, dtype=x.dtype),
                torch.tensor(OPENPI_ATTENTION_MASK_VALUE, device=x.device, dtype=x.dtype),
            ).expand(bsize, 1, -1, -1)
            positions = torch.arange(x.shape[1], device=x.device)[None].expand(bsize, -1)
            hidden = self.cond_encoder.model(
                inputs_embeds=x,
                attention_mask=attention,
                position_ids=positions,
                use_cache=False,
                adarms_cond=None,
            ).last_hidden_state
            query_out = self.query_out_norm(hidden[:, n_image:], cond)
        progress = torch.sigmoid(self.progress_head(query_out[:, 0])).squeeze(-1)
        termination = self.termination_head(query_out[:, 1]).squeeze(-1)
        return progress, termination

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


# -----------------------------------------------------------------------------
# Full model/config and component checkpoint loaders
# -----------------------------------------------------------------------------


@dataclass
class SplineFSQAEConfig:
    format_version: int = FORMAT_VERSION
    action_dim: int = 7
    enc_dim: int = 8
    state_dim: int = 8
    n_control: int = 30
    spline_degree: int = 3
    encoder_input_mode: str = "zero_grounded"
    hidden_dim: int = 256
    fsq_levels: list[int] = field(default_factory=lambda: [5, 5, 5])
    num_layers: int = 2
    dropout: float = 0.1
    length_min: float = 1.0
    length_max: float = 200.0

    action_expert_variant: str = "gemma_300m"
    state_cond_mode: str = "state"
    max_state_dim: int = 32
    max_action_dim: int = 32
    chunk_size: int = 10
    pi_base: str = _DEFAULT_PI_BASE
    min_period: float = 4e-3
    max_period: float = 4.0
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001

    terminator_arch: str = "small"
    vision_backbone: str = "dino"
    freeze_vision_encoder: bool = True
    dino_model_path: str = _DEFAULT_IMAGE_MODEL
    dino_image_size: int = 224
    siglip_image_size: int = 224
    cond_encoder_variant: str = "gemma_300m"
    image_encoder_layers: int = 2
    image_encoder_heads: int = 4

    samples_per_skill: int = 2
    end_target_sigma: float = 1.0
    action_loss_weight: float = 1.0
    progress_loss_weight: float = 0.1
    end_loss_weight: float = 0.1
    end_pos_weight: float = 1.0
    weighted_loss: bool = False
    weighted_loss_end_weight: float = 2.0
    end_threshold: float = 0.5

    encoder_lr: float = 3e-4
    terminator_lr: float = 3e-4
    expert_lr: float = 2.5e-5
    batch_size: int = 64
    num_workers: int = 0
    epochs: int = 300
    grad_clip: float = 1.0
    val_split: float = 0.1
    val_select_action_weight: float | None = None
    val_select_progress_weight: float | None = None
    val_select_end_weight: float | None = None
    log_every: int = 10
    save_path: str | None = None
    checkpoint_every: int = 0
    device: str = "cuda"
    expert_dtype: str = "bfloat16"

    encoder_min: np.ndarray | None = None
    encoder_max: np.ndarray | None = None
    encoder_start_min: np.ndarray | None = None
    encoder_start_max: np.ndarray | None = None
    state_min: np.ndarray | None = None
    state_max: np.ndarray | None = None
    state_q01: np.ndarray | None = None
    state_q99: np.ndarray | None = None
    action_q01: np.ndarray | None = None
    action_q99: np.ndarray | None = None


class SplineFSQAE(nn.Module):
    """Joint FSQ encoder, image-free flow expert, and query terminator."""

    def __init__(self, cfg: SplineFSQAEConfig):
        super().__init__()
        if int(cfg.format_version) != FORMAT_VERSION:
            raise ValueError(f"Only FSQ format v{FORMAT_VERSION} is supported, got {cfg.format_version}.")
        if cfg.action_dim > cfg.max_action_dim or cfg.state_dim > cfg.max_state_dim:
            raise ValueError(
                f"Real dimensions must fit PI05 padding: action {cfg.action_dim}/{cfg.max_action_dim}, "
                f"state {cfg.state_dim}/{cfg.max_state_dim}."
            )
        if cfg.expert_dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError(f"expert_dtype must be float32|bfloat16|float16, got {cfg.expert_dtype!r}.")
        if cfg.samples_per_skill < 1 or cfg.chunk_size < 1:
            raise ValueError("samples_per_skill and chunk_size must both be >=1.")
        if cfg.encoder_input_mode not in {"zero_grounded", "raw_state", "optimal"}:
            raise ValueError(
                "encoder_input_mode must be zero_grounded|raw_state|optimal, "
                f"got {cfg.encoder_input_mode!r}."
            )
        if cfg.terminator_arch not in {"small", "cond"}:
            raise ValueError(f"terminator_arch must be small|cond, got {cfg.terminator_arch!r}.")
        if cfg.vision_backbone not in {"dino", "siglip"}:
            raise ValueError(f"vision_backbone must be dino|siglip, got {cfg.vision_backbone!r}.")
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
        self.cfg = cfg
        self.encoder = SplineFSQEncoder(
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
        )
        self.action_expert = VSAFlowExpert(
            variant=cfg.action_expert_variant,
            fsq_levels=cfg.fsq_levels,
            state_cond_mode=cfg.state_cond_mode,
            max_state_dim=cfg.max_state_dim,
            max_action_dim=cfg.max_action_dim,
            chunk_size=cfg.chunk_size,
            min_period=cfg.min_period,
            max_period=cfg.max_period,
            time_sampling_beta_alpha=cfg.time_sampling_beta_alpha,
            time_sampling_beta_beta=cfg.time_sampling_beta_beta,
            time_sampling_scale=cfg.time_sampling_scale,
            time_sampling_offset=cfg.time_sampling_offset,
        )
        self.terminator = FSQQueryTerminator(
            state_dim=cfg.state_dim,
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
            cond_encoder_variant=cfg.cond_encoder_variant,
            state_min=cfg.state_min,
            state_max=cfg.state_max,
        )

    @property
    def fsq(self) -> FSQ:
        return self.encoder.fsq

    # Small read-only surface used by evaluation scripts. These are not legacy
    # decoder modules; all action predictions below go through VSAFlowExpert.
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
        return self.encoder.encode_numpy(trajectory, device)

    def encode_index(self, trajectory: np.ndarray, device: str | torch.device = "cpu") -> int:
        return self.encoder.encode_index(trajectory, device)

    @torch.no_grad()
    def sample_action_chunks(
        self,
        z_q: Tensor,
        raw_states: Tensor,
        *,
        noise: Tensor | None = None,
        num_steps: int = 10,
    ) -> Tensor:
        """Image-free VSA action chunks for every ``(B,T)`` state, in dataset units."""
        bsize, steps = raw_states.shape[:2]
        flat_state = raw_states.reshape(bsize * steps, -1)[..., : self.cfg.state_dim]
        lo = torch.as_tensor(self.cfg.state_q01, device=flat_state.device, dtype=flat_state.dtype)
        hi = torch.as_tensor(self.cfg.state_q99, device=flat_state.device, dtype=flat_state.dtype)
        norm = 2.0 * (flat_state - lo) / (hi - lo + 1e-8) - 1.0
        expert_state = torch.zeros(
            bsize * steps, self.cfg.max_state_dim, device=flat_state.device, dtype=flat_state.dtype
        )
        expert_state[:, : self.cfg.state_dim] = norm
        z_norm = self.fsq.normalized(z_q).repeat_interleave(steps, dim=0)
        action_norm = self.action_expert.sample_actions(
            expert_state, z_norm, noise=noise, num_steps=num_steps
        )[..., : self.cfg.action_dim]
        action_lo = torch.as_tensor(
            self.cfg.action_q01, device=action_norm.device, dtype=action_norm.dtype
        )
        action_hi = torch.as_tensor(
            self.cfg.action_q99, device=action_norm.device, dtype=action_norm.dtype
        )
        actions = (action_norm + 1.0) * 0.5 * (action_hi - action_lo) + action_lo
        return actions.view(bsize, steps, self.cfg.chunk_size, self.cfg.action_dim)

    @torch.no_grad()
    def decode(
        self,
        z_q: Tensor,
        raw_states: Tensor,
        third: Tensor,
        wrist: Tensor | None = None,
        _progress_hint: Tensor | None = None,
        *,
        num_steps: int = 10,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate every trajectory timestep with the v3 expert and terminator.

        This convenience method vectorizes inference over ``(B,T,...)``; training
        still uses the B-trajectory x M-sampled-timestep path. Returned actions are
        de-normalized to dataset units.
        """
        bsize, steps = raw_states.shape[:2]
        flat_state = raw_states.reshape(bsize * steps, -1)[..., : self.cfg.state_dim]
        z_norm = self.fsq.normalized(z_q).repeat_interleave(steps, dim=0)
        actions = self.sample_action_chunks(z_q, raw_states, num_steps=num_steps)

        def flatten_camera(value: Tensor | None) -> Tensor | None:
            if value is None:
                return None
            return value.reshape(bsize * steps, *value.shape[2:])

        progress, term_logits = self.terminator(
            z_norm,
            flat_state,
            flatten_camera(third),
            flatten_camera(wrist),
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
        expert_state: Tensor,
        raw_state: Tensor,
        actions: Tensor,
        third: Tensor | None,
        wrist: Tensor | None,
        samples_per_skill: int,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> dict[str, Tensor]:
        z_q, indices = self.encoder(ctrl, lengths, start_pose, normalized=True)
        z_norm = self.fsq.normalized(z_q)
        z_sample = z_norm.repeat_interleave(samples_per_skill, dim=0)
        residual = self.action_expert.flow_residual(
            expert_state, z_sample, actions, noise=noise, time=time
        )
        progress, term_logits = self.terminator(z_sample, raw_state, third, wrist)
        return {
            "z_q": z_q,
            "indices": indices,
            "flow_residual": residual,
            "progress": progress,
            "term_logits": term_logits,
        }


def _checkpoint_config(checkpoint: dict[str, Any]) -> SplineFSQAEConfig:
    cfg = checkpoint.get("cfg")
    if cfg is None:
        raise ValueError("FSQ checkpoint has no cfg.")
    if isinstance(cfg, dict):
        cfg = SplineFSQAEConfig(**cfg)
    if int(getattr(cfg, "format_version", 0)) != FORMAT_VERSION:
        raise ValueError(
            f"Legacy FSQ checkpoint is unsupported: expected format_version={FORMAT_VERSION}, "
            f"got {getattr(cfg, 'format_version', 0)}. Retrain with the VSA FSQ architecture."
        )
    return cfg


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
    encoder = SplineFSQEncoder(
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
    )
    _load_prefixed(encoder, checkpoint["model_state"], "encoder.")
    encoder.to(device).eval()
    return encoder, cfg


def load_fsq_terminator(
    path: str | Path,
    device: str | torch.device = "cpu",
    dino_model_path: str | None = None,
) -> tuple[FSQQueryTerminator, SplineFSQAEConfig]:
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = _checkpoint_config(checkpoint)
    terminator = FSQQueryTerminator(
        state_dim=cfg.state_dim,
        fsq_levels=cfg.fsq_levels,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.image_encoder_layers,
        n_heads=cfg.image_encoder_heads,
        dropout=0.0,
        arch=cfg.terminator_arch,
        vision_backbone=cfg.vision_backbone,
        freeze_vision_encoder=cfg.freeze_vision_encoder,
        dino_model_path=dino_model_path or cfg.dino_model_path,
        dino_image_size=cfg.dino_image_size,
        siglip_image_size=cfg.siglip_image_size,
        cond_encoder_variant=cfg.cond_encoder_variant,
        state_min=cfg.state_min,
        state_max=cfg.state_max,
    )
    _load_prefixed(terminator, checkpoint["model_state"], "terminator.")
    terminator.to(device).eval()
    return terminator, cfg


def load_fsq_action_expert_state(path: str | Path) -> tuple[dict[str, Tensor], SplineFSQAEConfig]:
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = _checkpoint_config(checkpoint)
    state = {
        k[len("action_expert.") :]: v
        for k, v in checkpoint["model_state"].items()
        if k.startswith("action_expert.")
    }
    if not state:
        raise ValueError("FSQ checkpoint has no action_expert tensors.")
    return state, cfg


def load_fsq_cond_warmstart_state(
    path: str | Path,
) -> tuple[dict[str, Tensor] | None, SplineFSQAEConfig]:
    """Return the Stage-1-cond-compatible part of a ``cond`` terminator.

    Learned progress/termination queries, their heads, and state/skill query
    modulation are deliberately excluded. A ``small`` checkpoint returns None.
    """
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = _checkpoint_config(checkpoint)
    if cfg.terminator_arch != "cond":
        return None, cfg
    component_prefixes = (
        "terminator.cond_encoder.",
        "terminator.image_proj.",
        f"terminator.{cfg.vision_backbone}.",
    )
    state = {}
    for key, value in checkpoint["model_state"].items():
        if key.startswith(component_prefixes):
            state[key[len("terminator.") :]] = value
    if not state:
        raise ValueError("FSQ cond terminator has no reusable cond/vision tensors.")
    return state, cfg


def load_fsq_model(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[SplineFSQAE, SplineFSQAEConfig]:
    """Load all v3 components. Reserved for joint FSQ evaluation/training tools."""
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = _checkpoint_config(checkpoint)
    model = SplineFSQAE(cfg)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    dev = torch.device(device)
    if dev.type == "cuda" and cfg.expert_dtype == "bfloat16":
        model.action_expert.to(dtype=torch.bfloat16)
    elif dev.type == "cuda" and cfg.expert_dtype == "float16":
        model.action_expert.to(dtype=torch.float16)
    model.to(dev).eval()
    return model, cfg


# -----------------------------------------------------------------------------
# B trajectories x M sampled timesteps dataset
# -----------------------------------------------------------------------------


class FSQTrajectoryDataset(Dataset):
    """One trajectory plus M sampled timesteps with live raw camera frames.

    Video decoding is lazy and worker-local. Only the selected M timesteps are
    read; there is no episode-wide DINO warm pass or in-RAM feature cache.
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
    ):
        if not (len(segments) == len(states) == len(actions) == len(metadata)):
            raise ValueError("FSQ dataset component lengths do not match.")
        self.cfg = cfg
        self.training = bool(training)
        self.samples_per_skill = int(cfg.samples_per_skill)
        if self.samples_per_skill < 1:
            raise ValueError("samples_per_skill must be >=1.")
        self.ctrl: list[np.ndarray] = []
        self.start_poses: list[np.ndarray] | None = [] if cfg.encoder_input_mode == "optimal" else None
        self.lengths: list[int] = []
        self.states = [np.asarray(x, dtype=np.float32) for x in states]
        self.actions = [np.asarray(x, dtype=np.float32) for x in actions]
        self.metadata = metadata
        self.raw_dataset_dir = str(raw_dataset_dir)
        self._raw_dataset = None

        enc_min = np.asarray(cfg.encoder_min, dtype=np.float32)
        enc_max = np.asarray(cfg.encoder_max, dtype=np.float32)
        start_min = start_max = None
        if self.start_poses is not None:
            start_min = np.asarray(cfg.encoder_start_min, dtype=np.float32)
            start_max = np.asarray(cfg.encoder_start_max, dtype=np.float32)
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
            if "dataset_from_index" not in metadata[i]:
                raise ValueError(f"Skill {i} metadata has no dataset_from_index.")
            self.ctrl.append((2.0 * (ctrl - enc_min) / (enc_max - enc_min + 1e-8) - 1.0).astype(np.float32))
            if self.start_poses is not None:
                start_pose = encoder_start_eef_pose(segment)
                self.start_poses.append(
                    (2.0 * (start_pose - start_min) / (start_max - start_min + 1e-8) - 1.0).astype(np.float32)
                )
            self.lengths.append(length)

        self.state_q01 = np.asarray(cfg.state_q01, dtype=np.float32)
        self.state_q99 = np.asarray(cfg.state_q99, dtype=np.float32)
        self.action_q01 = np.asarray(cfg.action_q01, dtype=np.float32)
        self.action_q99 = np.asarray(cfg.action_q99, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.ctrl)

    @staticmethod
    def _quantile_norm(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        return (2.0 * (x - lo) / (hi - lo + 1e-8) - 1.0).astype(np.float32)

    def _sample_indices(self, length: int) -> np.ndarray:
        m = self.samples_per_skill
        if self.training:
            if length >= m:
                return np.sort(np.random.choice(length, size=m, replace=False)).astype(np.int64)
            return np.sort(np.random.choice(length, size=m, replace=True)).astype(np.int64)
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
        """Decode all M frames for each camera in one batched video request.

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

        from lerobot.datasets.video_utils import decode_video_frames

        def decode(camera_key: str) -> Tensor:
            video_path = reader.root / reader._meta.get_video_file_path(episode_id, camera_key)  # noqa: SLF001
            from_timestamp = float(episode[f"videos/{camera_key}/from_timestamp"])
            return decode_video_frames(
                video_path,
                [from_timestamp + timestamp for timestamp in timestamps],
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
        norm_state = self._quantile_norm(raw_state, self.state_q01, self.state_q99)
        expert_state = np.zeros((len(sample), self.cfg.max_state_dim), dtype=np.float32)
        expert_state[:, : self.cfg.state_dim] = norm_state
        progress = sample.astype(np.float32) / max(length - 1, 1)
        distance_to_end = (length - 1 - sample).astype(np.float32)
        if self.cfg.end_target_sigma > 0:
            termination = np.exp(
                -(distance_to_end ** 2) / (2.0 * self.cfg.end_target_sigma ** 2)
            ).astype(np.float32)
        else:
            termination = (distance_to_end == 0).astype(np.float32)
        third, wrist = self._sample_images(index, sample)
        item = {
            "ctrl": torch.from_numpy(self.ctrl[index]),
            "length": torch.tensor(length, dtype=torch.long),
            "expert_state": torch.from_numpy(expert_state),
            "raw_state": torch.from_numpy(raw_state.copy()),
            "actions": torch.from_numpy(self._action_chunks(self.actions[index], sample)),
            "third": third,
            "wrist": wrist,
            "progress": torch.from_numpy(progress),
            "termination": torch.from_numpy(termination),
            "sample_index": torch.from_numpy(sample),
            "trajectory_index": torch.tensor(index, dtype=torch.long),
        }
        if self.start_poses is not None:
            item["start_pose"] = torch.from_numpy(self.start_poses[index])
        return item


def collate_fsq_batch(batch: list[dict[str, Tensor]]) -> dict[str, Tensor | None]:
    out: dict[str, Tensor | None] = {}
    for key in batch[0]:
        out[key] = torch.stack([item[key] for item in batch])
    return out


def _per_trajectory_mean(value: Tensor, bsize: int, samples_per_skill: int) -> Tensor:
    return value.view(bsize, samples_per_skill).mean(dim=1).mean()


def fsq_vsa_loss(
    output: dict[str, Tensor],
    batch: dict[str, Tensor | None],
    cfg: SplineFSQAEConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    bsize = int(batch["ctrl"].shape[0])
    m = cfg.samples_per_skill
    residual = output["flow_residual"][..., : cfg.action_dim]
    per_sample_action = residual.square().mean(dim=(1, 2))
    action_plain = _per_trajectory_mean(per_sample_action, bsize, m)
    if cfg.weighted_loss:
        if cfg.weighted_loss_end_weight <= 0:
            raise ValueError(
                "weighted_loss_end_weight must be positive, "
                f"got {cfg.weighted_loss_end_weight}."
            )
        progress = batch["progress"].reshape(-1).to(per_sample_action)
        weight = 1.0 + (cfg.weighted_loss_end_weight - 1.0) * progress
        weighted = (per_sample_action * weight).view(bsize, m).sum(dim=1) / weight.view(bsize, m).sum(dim=1)
        action_objective = weighted.mean()
    else:
        action_objective = action_plain

    progress_per = F.smooth_l1_loss(
        output["progress"], batch["progress"].reshape(-1).to(output["progress"]), reduction="none"
    )
    progress_loss = _per_trajectory_mean(progress_per, bsize, m)
    pos_weight = torch.as_tensor(cfg.end_pos_weight, device=output["term_logits"].device)
    end_per = F.binary_cross_entropy_with_logits(
        output["term_logits"],
        batch["termination"].reshape(-1).to(output["term_logits"]),
        reduction="none",
        pos_weight=pos_weight,
    )
    end_loss = _per_trajectory_mean(end_per, bsize, m)
    total = (
        cfg.action_loss_weight * action_objective
        + cfg.progress_loss_weight * progress_loss
        + cfg.end_loss_weight * end_loss
    )
    metrics = {
        "loss": total.detach(),
        "action": action_plain.detach(),
        "action_objective": action_objective.detach(),
        "progress": progress_loss.detach(),
        "termination": end_loss.detach(),
    }
    return total, metrics


@torch.no_grad()
def end_signal_metrics(logits: Tensor, target: Tensor, threshold: float) -> dict[str, float]:
    pred = torch.sigmoid(logits) >= threshold
    truth = target >= 0.5
    tp = (pred & truth).sum().float()
    fp = (pred & ~truth).sum().float()
    fn = (~pred & truth).sum().float()
    return {
        "acc": float((pred == truth).float().mean()),
        "precision": float(tp / (tp + fp).clamp_min(1.0)),
        "recall": float(tp / (tp + fn).clamp_min(1.0)),
        "positive_rate": float(pred.float().mean()),
    }


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


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
    n_val = max(1, int(len(segments) * cfg.val_split))
    if len(metadata) == len(segments):
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
    val_ids, train_ids = order[:n_val], order[n_val:]
    print(f"[FSQ-v3] trajectories={len(segments)} train={len(train_ids)} val={len(val_ids)} fp={fingerprint}")

    def take(items: list[Any] | None, ids: list[int]):
        return None if items is None else [items[i] for i in ids]

    def dataset(ids: list[int], training: bool) -> FSQTrajectoryDataset:
        return FSQTrajectoryDataset(
            take(segments, ids),
            take(decoder_states, ids),
            take(decoder_targets, ids),
            take(metadata, ids),
            raw_dataset_dir,
            cfg,
            training=training,
        )

    train_ds, val_ds = dataset(train_ids, True), dataset(val_ids, False)
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
        prefetch_factor=1 if cfg.num_workers > 0 else None,
        # Surface a stuck decoder as a failed job instead of silently holding a
        # GPU allocation forever.  Normal batches complete far below 5 minutes.
        timeout=300 if cfg.num_workers > 0 else 0,
    )
    val_workers = min(2, cfg.num_workers)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=val_workers,
        collate_fn=collate_fsq_batch,
        pin_memory=pin_memory,
        persistent_workers=False,
        prefetch_factor=1 if val_workers > 0 else None,
        timeout=300 if val_workers > 0 else 0,
    )

    model = SplineFSQAE(cfg).to(device)
    if cfg.expert_dtype == "bfloat16" and device.type == "cuda":
        model.action_expert.to(dtype=torch.bfloat16)
    elif cfg.expert_dtype == "float16" and device.type == "cuda":
        model.action_expert.to(dtype=torch.float16)
    model.action_expert.gradient_checkpointing_enable()
    model.terminator.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": cfg.encoder_lr, "name": "encoder"},
            {"params": model.action_expert.parameters(), "lr": cfg.expert_lr, "name": "action_expert"},
            {"params": model.terminator.parameters(), "lr": cfg.terminator_lr, "name": "terminator"},
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )

    def lr_factor(epoch: int) -> float:
        if cfg.epochs <= 1:
            return 0.01
        progress = min(max(epoch / (cfg.epochs - 1), 0.0), 1.0)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    start_epoch, best_val = 1, math.inf
    if resume_from:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        resume_cfg = _checkpoint_config(checkpoint)
        resume_input_mode = getattr(resume_cfg, "encoder_input_mode", "zero_grounded")
        if resume_input_mode != cfg.encoder_input_mode:
            raise ValueError(
                "Cannot resume FSQ with a different encoder input convention: "
                f"checkpoint={resume_input_mode!r}, current={cfg.encoder_input_mode!r}."
            )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optim_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = float(checkpoint.get("val_select", math.inf))
        print(f"[FSQ-v3] resumed {resume_from} at epoch {start_epoch}")
    else:
        loaded = initialize_action_expert_from_pi05(model.action_expert, cfg.pi_base)
        print(f"[FSQ-v3] initialized {loaded} action-expert tensors from {cfg.pi_base}")
        loaded_vision = initialize_terminator_vision_from_pi05(model.terminator, cfg.pi_base)
        if loaded_vision:
            print(f"[FSQ-v3] initialized {loaded_vision} SigLIP tensors from {cfg.pi_base}")

    if wandb_run is not None:
        # The same epoch aggregate is reported twice: train/val use optimizer_step as x, while
        # train_epoch/val_epoch use epoch as x.  Keeping train and val at the top-level avoids
        # collapsing every curve into one "epoch" workspace section.
        wandb_run.define_metric("epoch")
        wandb_run.define_metric("optimizer_step")
        for name in ("train/*", "val/*", "perf/*", "lr/*"):
            wandb_run.define_metric(name, step_metric="optimizer_step")
        for name in ("train_epoch/*", "val_epoch/*", "perf_epoch/*", "lr_epoch/*"):
            wandb_run.define_metric(name, step_metric="epoch")

    def save(path: str | Path, epoch: int, val: float, select: float, *, resumable: bool) -> None:
        payload = {
            "format_version": FORMAT_VERSION,
            "cfg": cfg,
            "model_state": model.state_dict(),
            "epoch": epoch,
            "val_loss": val,
            "val_select": select,
        }
        # FSQ.pt is copied into every SkillVLA data run and only needs component weights.
        # Periodic checkpoints retain optimizer/scheduler state for exact resume.
        if resumable:
            payload["optim_state"] = optimizer.state_dict()
            payload["scheduler_state"] = scheduler.state_dict()
        torch.save(payload, str(path))

    def step(batch: dict[str, Tensor | None], training: bool, batch_index: int):
        moved = {k: (v.to(device, non_blocking=True) if isinstance(v, Tensor) else v) for k, v in batch.items()}
        bsize = moved["ctrl"].shape[0]
        m = cfg.samples_per_skill
        expert_state = moved["expert_state"].reshape(bsize * m, cfg.max_state_dim)
        raw_state = moved["raw_state"].reshape(bsize * m, cfg.state_dim)
        actions = moved["actions"].reshape(bsize * m, cfg.chunk_size, cfg.max_action_dim)
        third = moved["third"].reshape(bsize * m, *moved["third"].shape[2:])
        wrist = moved["wrist"].reshape(bsize * m, *moved["wrist"].shape[2:])
        noise = time = None
        if not training:
            generator = torch.Generator(device=device).manual_seed(10_000 + batch_index)
            noise = torch.randn(actions.shape, generator=generator, device=device, dtype=actions.dtype)
            time = torch.full((bsize * m,), 0.5, device=device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                ctrl=moved["ctrl"],
                lengths=moved["length"],
                start_pose=moved.get("start_pose"),
                expert_state=expert_state,
                raw_state=raw_state,
                actions=actions,
                third=third,
                wrist=wrist,
                samples_per_skill=m,
                noise=noise,
                time=time,
            )
            loss, metrics = fsq_vsa_loss(output, moved, cfg)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        end_metrics = end_signal_metrics(
            output["term_logits"].detach(),
            moved["termination"].reshape(-1),
            cfg.end_threshold,
        )
        # One FSQ index per input skill is already produced by the encoder. Keep
        # it on-device so epoch-level codebook coverage costs only a boolean
        # scatter, not an additional encode pass or per-batch CPU synchronization.
        return {k: float(v) for k, v in metrics.items()}, end_metrics, bsize, output["indices"].detach()

    save_path = Path(cfg.save_path) if cfg.save_path else Path("FSQ.pt")
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
            metrics, end_metrics, count, code_indices = step(batch, True, batch_index)
            global_step += 1
            train_count += count
            train_codes_seen[code_indices.reshape(-1).long()] = True
            for key, value in metrics.items():
                train_sum[key] = train_sum.get(key, 0.0) + value * count
            for key, value in end_metrics.items():
                train_end[key] = train_end.get(key, 0.0) + value * count
        scheduler.step()

        model.eval()
        val_sum: dict[str, float] = {}
        val_end: dict[str, float] = {}
        val_count = 0
        val_codes_seen = torch.zeros(model.fsq.codebook_size, dtype=torch.bool, device=device)
        with torch.no_grad():
            for batch_index, batch in enumerate(val_loader):
                metrics, end_metrics, count, code_indices = step(batch, False, batch_index)
                val_count += count
                val_codes_seen[code_indices.reshape(-1).long()] = True
                for key, value in metrics.items():
                    val_sum[key] = val_sum.get(key, 0.0) + value * count
                for key, value in end_metrics.items():
                    val_end[key] = val_end.get(key, 0.0) + value * count

        train_avg = {k: v / max(train_count, 1) for k, v in train_sum.items()}
        val_avg = {k: v / max(val_count, 1) for k, v in val_sum.items()}
        train_end_avg = {k: v / max(train_count, 1) for k, v in train_end.items()}
        val_end_avg = {k: v / max(val_count, 1) for k, v in val_end.items()}
        train_active_codes = int(train_codes_seen.count_nonzero().item())
        val_active_codes = int(val_codes_seen.count_nonzero().item())
        codebook_size = model.fsq.codebook_size
        select = (
            (cfg.val_select_action_weight if cfg.val_select_action_weight is not None else cfg.action_loss_weight)
            * val_avg["action"]
            + (cfg.val_select_progress_weight if cfg.val_select_progress_weight is not None else cfg.progress_loss_weight)
            * val_avg["progress"]
            + (cfg.val_select_end_weight if cfg.val_select_end_weight is not None else cfg.end_loss_weight)
            * val_avg["termination"]
        )
        if select < best_val:
            best_val = select
            save(save_path, epoch, val_avg["loss"], select, resumable=False)
        if cfg.checkpoint_every and epoch % cfg.checkpoint_every == 0:
            save(
                save_path.with_name(f"FSQ_epoch{epoch:04d}.pt"),
                epoch,
                val_avg["loss"],
                select,
                resumable=True,
            )

        log = {
            "epoch": epoch,
            "optimizer_step": global_step,
            "perf/seconds": time.perf_counter() - epoch_start,
            "perf/updates_per_sec": len(train_loader) / max(time.perf_counter() - epoch_start, 1e-8),
            **{f"train/{k}": v for k, v in train_avg.items()},
            **{f"val/{k}": v for k, v in val_avg.items()},
            **{f"train/end_{k}": v for k, v in train_end_avg.items()},
            **{f"val/end_{k}": v for k, v in val_end_avg.items()},
            "train/codebook_utilization_pct": 100.0 * train_active_codes / codebook_size,
            "train/codebook_active_entries": train_active_codes,
            "val/codebook_utilization_pct": 100.0 * val_active_codes / codebook_size,
            "val/codebook_active_entries": val_active_codes,
            "val/select": select,
            "lr/encoder": optimizer.param_groups[0]["lr"],
            "lr/expert": optimizer.param_groups[1]["lr"],
            "lr/terminator": optimizer.param_groups[2]["lr"],
        }
        log.update({f"train_epoch/{k}": v for k, v in train_avg.items()})
        log.update({f"val_epoch/{k}": v for k, v in val_avg.items()})
        log.update({f"train_epoch/end_{k}": v for k, v in train_end_avg.items()})
        log.update({f"val_epoch/end_{k}": v for k, v in val_end_avg.items()})
        log.update({
            "train_epoch/codebook_utilization_pct": log["train/codebook_utilization_pct"],
            "train_epoch/codebook_active_entries": train_active_codes,
            "val_epoch/codebook_utilization_pct": log["val/codebook_utilization_pct"],
            "val_epoch/codebook_active_entries": val_active_codes,
            "val_epoch/select": select,
            "perf_epoch/seconds": log["perf/seconds"],
            "perf_epoch/updates_per_sec": log["perf/updates_per_sec"],
            "lr_epoch/encoder": log["lr/encoder"],
            "lr_epoch/expert": log["lr/expert"],
            "lr_epoch/terminator": log["lr/terminator"],
        })
        if wandb_run is not None:
            wandb_run.log(log, step=global_step)
        if epoch == 1 or epoch % cfg.log_every == 0:
            print(
                f"[FSQ-v3] {epoch:4d}/{cfg.epochs} "
                f"train={train_avg['loss']:.4f} val={val_avg['loss']:.4f} "
                f"action={val_avg['action']:.4f} prog={val_avg['progress']:.4f} "
                f"end={val_avg['termination']:.4f} select={select:.4f}"
            )

    print(f"[FSQ-v3] done; best val-select={best_val:.6f} -> {save_path}")
    return model
