"""
Spline FSQ-AE — Finite Scalar Quantization skill encoder with observation-conditioned decoder.

Architecture
------------
  Encoder (action-only, fully Transformer-based):
    enc_ctrl_proj/enc_len_proj (Linear A→H, 1→H): per-control-point + length tokens
    enc_traj_pool (Transformer + query pool): control-point/length tokens → pooled latent (H)
    z_head (Linear H→D): produce pre-quantization latent
    FSQ: levels=[5,5,5], codebook_size=125, D=3, no learnable params
  The encoder sees ONLY the (zero-grounded) action trajectory — no images. z therefore
  encodes the task-agnostic motion ("what motion", translation-invariant); scene grounding
  ("where/what object") is supplied to the decoder via its start-frame inputs instead.

  Decoder (3 Transformer sub-networks, per timestep):
    dec_image_encoder_term[_wrist] (DINO) → termination image feature (B,T,H)
    (1) skill_decoder: [z token, state token] → TokenTransformerPool → latent (H)
    (2) motion_reconstructor: [z, start_state, progress] tokens (image-free)
            → TokenTransformerPool → delta_head → dataset action chunk (K*A);
            the historical "delta" name is kept for the action target.
    (3) skill_termination_predictor: [latent, img_feat(no flags)] tokens
            → TokenTransformerPool → progress_head (1, sigmoid→[0,1])
                                   + termination_head (1, end logit).
    progress is normalized per skill: 0 at the start, 1 at the skill end.
    In training the GT progress feeds the motion tokens; the termination
    network learns to predict it.

encode_numpy        → latent vector z_q (numpy, shape D)
encode_index        → scalar codebook index (int)
predict_termination → skillVLA path: (z_q, states, dino) → (progress, termination),
                      runs only skill_decoder + skill_termination_predictor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import make_interp_spline
from torch.utils.data import DataLoader, Dataset, Sampler


# ── Spline codec ──────────────────────────────────────────────────────────────

# The encoder trajectory is the full observation.state: EEF pose dims followed by the trailing
# gripper-STATE dims (LIBERO = 2 finger positions). Gripper dims stay ABSOLUTE (not zero-grounded)
# and use a low-degree spline (near-step signal → avoids cubic overshoot at open/close transitions).
N_GRIPPER_DIMS = 2


def zero_ground_trajectory(trajectory: np.ndarray) -> np.ndarray:
    """Make the pose trajectory relative to its skill-start frame; the gripper dims stay absolute.

    Translation-invariant encoder input: the same motion gets the same code regardless of where in
    the workspace it starts (kills the start-position jitter and the scene-layout leak that absolute
    coordinates carry). Spline interpolation passes through the start point, so subtracting here is
    equivalent to subtracting on the control points."""
    traj = np.asarray(trajectory, dtype=np.float32).copy()
    offset = traj[0].copy()
    offset[-N_GRIPPER_DIMS:] = 0.0  # keep gripper-STATE dims absolute (they are not a pose)
    return traj - offset


def spline_encode(
    trajectory: np.ndarray,  # (T, action_dim)
    n_control: int,
    degree: int,
) -> tuple[np.ndarray, int]:
    """trajectory → zero-grounded control points (n_control, action_dim) + original length T."""
    trajectory = zero_ground_trajectory(trajectory)
    T, D = trajectory.shape
    t_orig = np.linspace(0.0, 1.0, T)
    t_ctrl = np.linspace(0.0, 1.0, n_control)
    ctrl_pts = np.zeros((n_control, D), dtype=np.float32)
    for d in range(D):
        # trailing gripper-state dims: low-degree (near-step, no overshoot); pose dims: full degree
        k = 1 if d >= D - N_GRIPPER_DIMS else degree
        k = min(k, T - 1)
        spl = make_interp_spline(t_orig, trajectory[:, d], k=k)
        ctrl_pts[:, d] = spl(t_ctrl)
    return ctrl_pts, T


class ImageTokenEncoder(nn.Module):
    """Turn DINO CLS+patch tokens into one image feature.

    Patches are first projected to an internal token dimension N (token_dim),
    mixed by a Transformer at that dimension, pooled by a learned query, and only
    then lifted to hidden_dim. This decouples the per-patch processing width from
    the model-wide hidden_dim.
    """

    def __init__(
        self,
        feat_dim: int,
        n_tokens: int,
        hidden_dim: int,
        token_dim: int,
        n_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert n_tokens >= 1, "n_tokens must include at least the CLS token"
        assert token_dim % n_heads == 0, "token_dim must be divisible by n_heads"
        self.n_tokens = n_tokens
        self.token_dim = token_dim
        self.token_proj = nn.Linear(feat_dim, token_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, token_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.query = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.pool = nn.MultiheadAttention(token_dim, n_heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(token_dim, hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens:  (B, T, N, F), token 0 is CLS and tokens 1..N-1 are patches.
        returns: (B, T, hidden_dim)
        """
        B, T, N, _ = tokens.shape
        assert N == self.n_tokens, f"expected {self.n_tokens} image tokens, got {N}"
        x = self.token_proj(tokens.reshape(B * T, N, -1))   # (B*T, N, token_dim)
        x = x + self.pos_embed.to(dtype=x.dtype)
        x = self.encoder(x)
        q = self.query.to(dtype=x.dtype).expand(B * T, -1, -1)
        pooled, _ = self.pool(q, x, x, need_weights=False)
        return self.out_norm(self.out_proj(pooled.squeeze(1))).view(B, T, -1)


class TokenTransformerPool(nn.Module):
    """Mix a set of H-dim tokens with a Transformer, then pool to one vector.

    Input  : (B, N, H) tokens already projected to hidden_dim.
    Output : (B, H) — a learned query cross-attends over the encoded tokens.
    A learnable positional embedding distinguishes token slots (e.g. control-point
    order, or the start/traj/end roles in the fusion stage).
    """

    def __init__(
        self,
        hidden_dim: int,
        n_tokens: int,
        n_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.n_tokens = n_tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pool = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, N, H) → (B, H)."""
        B, N, _ = tokens.shape
        assert N == self.n_tokens, f"expected {self.n_tokens} tokens, got {N}"
        x = tokens + self.pos_embed.to(dtype=tokens.dtype)
        x = self.encoder(x)
        q = self.query.to(dtype=x.dtype).expand(B, -1, -1)
        pooled, _ = self.pool(q, x, x, need_weights=False)
        return self.out_norm(pooled.squeeze(1))


# ── Finite Scalar Quantization ────────────────────────────────────────────────

class FSQ(nn.Module):
    """FSQ: each dim quantized via tanh-scaling + rounding. No learnable params.

    levels=[5,5,5] → half=[2,2,2], strides=[1,5,25], codebook_size=125.
    Straight-through estimator preserves encoder gradients through rounding.
    """

    def __init__(self, levels: list[int]) -> None:
        super().__init__()
        D = len(levels)
        levels_half = torch.tensor([(L - 1) / 2.0 for L in levels], dtype=torch.float32)
        self.register_buffer("levels_half", levels_half)
        strides = torch.ones(D, dtype=torch.long)
        for i in range(1, D):
            strides[i] = strides[i - 1] * levels[i - 1]
        self.register_buffer("strides", strides)
        self.codebook_size = int(np.prod(levels))
        self.latent_dim = D

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z (*, D) → z_q (*, D), indices (*,)."""
        lh = self.levels_half.to(z.dtype)
        z_scaled = torch.tanh(z) * lh              # bounded to [-lh, +lh]
        z_int = torch.round(z_scaled)
        z_q = z_scaled + (z_int - z_scaled).detach()  # straight-through
        indices = ((z_int + lh).long() * self.strides).sum(dim=-1)
        return z_q, indices


# ── Config ────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_IMAGE_MODEL = str(_REPO_ROOT / "models" / "dinov3-vits16")


def resolve_image_model_path(name: str) -> str:
    """Map an absolute model path from another machine (e.g. stored in a checkpoint cfg)
    to this repo's ``models/<basename>`` if the original path does not exist here.
    HF hub repo ids and existing paths pass through unchanged."""
    p = Path(name)
    if not p.is_absolute() or p.exists():
        return name
    local = _REPO_ROOT / "models" / p.name
    return str(local) if local.exists() else name


@dataclass
class SplineFSQAEConfig:
    action_dim: int = 7          # decoder output / action-chunk target dim
    enc_dim: int = 8             # encoder trajectory dim = full state (EEF pose + gripper-state)
    state_dim: int = 7
    n_control: int = 30
    spline_degree: int = 3
    hidden_dim: int = 256
    fsq_levels: list[int] = field(default_factory=lambda: [5, 5, 5])
    num_layers: int = 3
    dropout: float = 0.1
    feat_dim: int = 384       # DINO token feature dimension
    n_tokens: int = 65        # 1 CLS + 64 patch tokens (8×8 grid)
    image_model_name: str = _DEFAULT_IMAGE_MODEL
    image_size: int = 224
    patch_grid: int = 8
    n_patch_raw: int = 196
    terminator_use_third: bool = True
    """Terminator 3rd-person camera. With terminator_use_wrist → 3 modes (third,wrist):
    (T,F)=3rd-only, (T,T)=both, (F,T)=wrist-only; (F,F) invalid. Default T = backward-compatible."""
    terminator_use_wrist: bool = False
    """Terminator camera inputs: False = 3rd-person only ([z, img, state], original single-camera
    FSQ — keeps old checkpoints loadable); True = 3rd-person + wrist ([z, img, wrist, state], two
    separate DINO-token encoders)."""
    image_token_dim: int = 128  # internal per-patch width N in the image encoders
    image_encoder_layers: int = 1
    image_encoder_heads: int = 4
    chunk_size: int = 10  # motion always predicts a K-step action chunk
    length_min: float = 0.0     # skill-length min-max stats for [-1,1] length-token normalization
    length_max: float = 200.0
    end_target_sigma: float = 0.0
    """Soft termination target: Gaussian bump (std in FRAMES) peaking at the skill's last frame
    instead of a 1-frame spike. DP skill boundaries are fuzzy, so a spike makes the head memorize
    exact end frames (val overfit). σ≈2-3 encodes that ±tolerance and gives dense gradients; the
    bump's ≥0.5 region also makes the recall/precision metric window-tolerant for free. 0 = hard."""
    delta_loss_weight: float = 1.0
    progress_loss_weight: float = 1.0
    end_loss_weight: float = 1.0
    end_pos_weight: float = 1.0
    end_threshold: float = 0.5
    lr: float = 3e-4
    batch_size: int = 64
    num_workers: int = 8
    epochs: int = 100
    grad_clip: float = 1.0
    device: str = "cuda"
    val_split: float = 0.1
    log_every: int = 10
    save_path: str | None = None
    checkpoint_every: int = 0
    action_min: np.ndarray | None = None
    action_max: np.ndarray | None = None
    delta_min: np.ndarray | None = None
    delta_max: np.ndarray | None = None
    state_min: np.ndarray | None = None   # decoder/terminator proprioception min-max ([-1,1] norm)
    state_max: np.ndarray | None = None


# ── Model ─────────────────────────────────────────────────────────────────────

class SplineFSQAE(nn.Module):
    """Spline-encoded FSQ autoencoder with DINO-token-conditioned decoder."""

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        enc_dim: int = 8,
        n_control: int = 30,
        spline_degree: int = 3,
        hidden_dim: int = 256,
        fsq_levels: list[int] | None = None,
        num_layers: int = 3,
        dropout: float = 0.1,
        feat_dim: int = 384,
        n_tokens: int = 65,
        image_model_name: str = _DEFAULT_IMAGE_MODEL,
        image_size: int = 224,
        patch_grid: int = 8,
        n_patch_raw: int = 196,
        terminator_use_third: bool = True,
        terminator_use_wrist: bool = False,
        image_token_dim: int = 128,
        image_encoder_layers: int = 1,
        image_encoder_heads: int = 4,
        chunk_size: int = 10,
        length_min: float = 0.0,
        length_max: float = 200.0,
        action_min: np.ndarray | None = None,
        action_max: np.ndarray | None = None,
        delta_min: np.ndarray | None = None,
        delta_max: np.ndarray | None = None,
        state_min: np.ndarray | None = None,
        state_max: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        if fsq_levels is None:
            fsq_levels = [5, 5, 5]

        self.action_dim = action_dim
        self.enc_dim = enc_dim
        self.state_dim = state_dim
        self.n_control = n_control
        self.spline_degree = spline_degree
        self.length_min = length_min
        self.length_max = length_max
        self.chunk_size = chunk_size
        self.feat_dim = feat_dim
        self.n_tokens = n_tokens
        self.image_model_name = resolve_image_model_path(image_model_name)
        self.image_size = image_size
        self.patch_grid = patch_grid
        self.n_patch_raw = n_patch_raw
        self.terminator_use_third = bool(terminator_use_third)
        self.terminator_use_wrist = terminator_use_wrist

        self.fsq = FSQ(fsq_levels)
        D = self.fsq.latent_dim
        H = hidden_dim

        # ── Encoder (action trajectory only — no images) ──────────────────────
        # Each (zero-grounded) control point + the length scalar becomes one H-dim token,
        # a Transformer mixes them, and a learned query pools them into the latent.
        self.enc_ctrl_proj = nn.Linear(enc_dim, H)
        self.enc_len_proj = nn.Linear(1, H)
        self.enc_traj_pool = TokenTransformerPool(
            hidden_dim=H,
            n_tokens=n_control + 1,
            n_layers=image_encoder_layers,
            n_heads=image_encoder_heads,
            dropout=dropout,
        )
        self.z_head = nn.Linear(H, D)

        # ── Decoder (2 sub-networks: terminator + reconstructor; no skill_decoder) ──
        self.n_patches = n_tokens - 1  # token 0 = CLS, tokens 1.. = patches

        def _plain_image_encoder() -> "ImageTokenEncoder":
            return ImageTokenEncoder(
                feat_dim=feat_dim, n_tokens=n_tokens, hidden_dim=H, token_dim=image_token_dim,
                n_layers=image_encoder_layers, n_heads=image_encoder_heads, dropout=dropout,
            )

        # z (skill code) → token, shared by both sub-networks.
        self.dec_z_proj = nn.Linear(D, H)

        # (A) terminator → progress + termination.
        #   single (terminator_use_wrist=False): [z, current 3rd img, current state]            (3)
        #   dual   (terminator_use_wrist=True):  [z, current 3rd img, current wrist img, state]  (4)
        # Cameras use separate DINO-token encoders. single keeps the original (1-camera) FSQ so
        # old checkpoints stay loadable.
        if not (terminator_use_third or terminator_use_wrist):
            raise ValueError("terminator needs ≥1 camera: set terminator_use_third and/or terminator_use_wrist.")
        self.dec_image_encoder_term = (
            _plain_image_encoder() if terminator_use_third else None  # 3rd-person, separate params
        )
        self.dec_image_encoder_term_wrist = (
            _plain_image_encoder() if terminator_use_wrist else None  # wrist, separate params
        )
        self.term_state_proj = nn.Linear(state_dim, H)              # separate params
        # terminator tokens: z + (3rd?) + (wrist?) + state
        self.term_pool = TokenTransformerPool(
            hidden_dim=H, n_tokens=(1 + int(terminator_use_third) + int(terminator_use_wrist) + 1),
            n_layers=image_encoder_layers, n_heads=image_encoder_heads, dropout=dropout,
        )
        self.progress_head = nn.Linear(H, 1)
        self.termination_head = nn.Linear(H, 1)

        # (B) reconstructor → action chunk: [z, start_state, progress] (image-free).
        # Everything except progress is fixed at the skill start; z is the sole motion source.
        self.recon_state_proj = nn.Linear(state_dim, H)             # separate params (start state)
        self.motion_prog_proj = nn.Linear(1, H)
        self.motion_pool = TokenTransformerPool(
            hidden_dim=H, n_tokens=3,  # [z, start_state, progress]
            n_layers=image_encoder_layers, n_heads=image_encoder_heads, dropout=dropout,
        )
        self.delta_head = nn.Linear(H, chunk_size * action_dim)

        _amin = action_min if action_min is not None else np.zeros(enc_dim, dtype=np.float32)
        _amax = action_max if action_max is not None else np.ones(enc_dim, dtype=np.float32)
        _dmin = delta_min if delta_min is not None else -np.ones(action_dim, dtype=np.float32)
        _dmax = delta_max if delta_max is not None else np.ones(action_dim, dtype=np.float32)
        _smin = state_min if state_min is not None else np.zeros(state_dim, dtype=np.float32)
        _smax = state_max if state_max is not None else np.ones(state_dim, dtype=np.float32)
        self.register_buffer("action_min", torch.tensor(_amin, dtype=torch.float32))
        self.register_buffer("action_max", torch.tensor(_amax, dtype=torch.float32))
        self.register_buffer("delta_min",  torch.tensor(_dmin, dtype=torch.float32))
        self.register_buffer("delta_max",  torch.tensor(_dmax, dtype=torch.float32))
        self.register_buffer("state_min",  torch.tensor(_smin, dtype=torch.float32))
        self.register_buffer("state_max",  torch.tensor(_smax, dtype=torch.float32))
        object.__setattr__(self, "_raw_image_model", None)
        object.__setattr__(self, "_raw_image_mean", None)
        object.__setattr__(self, "_raw_image_std", None)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _np_norm_actions(self, ctrl_pts: np.ndarray) -> np.ndarray:
        lo = self.action_min.cpu().numpy()
        hi = self.action_max.cpu().numpy()
        return (ctrl_pts - lo) / (hi - lo + 1e-8) * 2.0 - 1.0

    def _norm_state(self, s: torch.Tensor) -> torch.Tensor:
        """Min-max normalize raw proprioception to [-1,1]. Absolute (not grounded) — same basis for
        the terminator's current state and the reconstructor's start state."""
        lo = self.state_min.to(s.device, s.dtype)
        hi = self.state_max.to(s.device, s.dtype)
        return (s - lo) / (hi - lo + 1e-8) * 2.0 - 1.0

    # ── encode ────────────────────────────────────────────────────────────────

    def encode(
        self,
        ctrl_pts: torch.Tensor,      # (B, n_control, enc_dim) — zero-grounded, normalized
        lengths: torch.Tensor,       # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Action-only encode. Returns (z_q, indices)."""
        B = ctrl_pts.size(0)
        # Tokens: per control point + one length token → Transformer pool → latent.
        ctrl_tok = self.enc_ctrl_proj(ctrl_pts)                                    # (B, n_control, H)
        # length token: min-max normalized to [-1,1] (same scheme as the control points)
        l_norm = (((lengths.float() - self.length_min) / (self.length_max - self.length_min + 1e-8))
                  * 2.0 - 1.0).view(B, 1, 1).to(ctrl_pts.dtype)
        len_tok = self.enc_len_proj(l_norm)                                        # (B, 1, H)
        h = self.enc_traj_pool(torch.cat([ctrl_tok, len_tok], dim=1))             # (B, H)
        z_e = self.z_head(h)
        return self.fsq(z_e)

    # ── decode ────────────────────────────────────────────────────────────────

    def _load_raw_image_model(self, device: torch.device):
        model = getattr(self, "_raw_image_model", None)
        if model is not None:
            return model
        from transformers import AutoImageProcessor, AutoModel  # noqa: PLC0415

        model = AutoModel.from_pretrained(self.image_model_name)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval().to(device)
        try:
            proc = AutoImageProcessor.from_pretrained(self.image_model_name)
            mean = getattr(proc, "image_mean", [0.485, 0.456, 0.406])
            std = getattr(proc, "image_std", [0.229, 0.224, 0.225])
        except Exception:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        object.__setattr__(self, "_raw_image_model", model)
        object.__setattr__(self, "_raw_image_mean", torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1))
        object.__setattr__(self, "_raw_image_std", torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1))
        return model

    @torch.no_grad()
    def _raw_images_to_tokens(self, images: torch.Tensor, *, target_t: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Raw RGB images → DINO CLS + pooled patch tokens, shape (B,T,N,F)."""
        x = images.to(device=device)
        if x.ndim == 4:
            x = x.unsqueeze(1)  # (B,1,C,H,W) or (B,1,H,W,C)
        if x.ndim != 5:
            raise ValueError(
                "Raw FSQ decoder images must have shape (B,C,H,W), (B,H,W,C), "
                "(B,T,C,H,W), or (B,T,H,W,C)."
            )
        if x.shape[-1] in (1, 3):
            x = x.permute(0, 1, 4, 2, 3)  # channels-last → channels-first
        if x.shape[2] == 1:
            x = x.expand(-1, -1, 3, -1, -1)
        if x.shape[2] != 3:
            raise ValueError(f"Raw FSQ decoder images must have 3 channels, got shape {tuple(images.shape)}.")
        if x.shape[1] == 1 and target_t > 1:
            x = x.expand(x.shape[0], target_t, x.shape[2], x.shape[3], x.shape[4])
        if x.shape[1] != target_t:
            raise ValueError(f"Raw FSQ decoder image time mismatch: got {x.shape[1]}, expected {target_t}.")

        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W).float()
        if x.numel() and float(x.detach().amin()) < -0.05:
            x = (x + 1.0) / 2.0
        if x.numel() and float(x.detach().amax()) > 2.0:
            x = x / 255.0
        x = x.clamp(0.0, 1.0)
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        model = self._load_raw_image_model(device)
        mean = getattr(self, "_raw_image_mean").to(device=device)
        std = getattr(self, "_raw_image_std").to(device=device)
        x = (x - mean) / std
        device_type = device.type
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == "cuda")):
            out = model(pixel_values=x)
        last_hidden = out.last_hidden_state.float()
        b_frames, _, feat_dim = last_hidden.shape
        cls_tok = last_hidden[:, :1, :]
        patch_tok = last_hidden[:, -self.n_patch_raw:, :]
        sqrt_raw = int(math.sqrt(self.n_patch_raw))
        patch_spatial = patch_tok.reshape(b_frames, sqrt_raw, sqrt_raw, feat_dim).permute(0, 3, 1, 2)
        patch_pooled = F.adaptive_avg_pool2d(patch_spatial, (self.patch_grid, self.patch_grid))
        patch_pooled = patch_pooled.permute(0, 2, 3, 1).reshape(b_frames, self.patch_grid * self.patch_grid, feat_dim)
        tokens = torch.cat([cls_tok, patch_pooled], dim=1).view(B, T, 1 + self.patch_grid * self.patch_grid, feat_dim)
        return tokens.to(dtype=dtype)

    def _prepare_decoder_tokens(self, image_input: torch.Tensor, *, states: torch.Tensor) -> torch.Tensor:
        """Accept precomputed DINO tokens or raw RGB images and return (B,T,N,F) tokens."""
        if image_input.ndim == 4 and image_input.shape[-2] == self.n_tokens and image_input.shape[-1] == self.feat_dim:
            return image_input.to(device=states.device, dtype=states.dtype)
        if image_input.ndim == 3 and image_input.shape[-2] == self.n_tokens and image_input.shape[-1] == self.feat_dim:
            tokens = image_input.unsqueeze(1).to(device=states.device, dtype=states.dtype)
            if states.shape[1] > 1:
                tokens = tokens.expand(tokens.shape[0], states.shape[1], tokens.shape[2], tokens.shape[3])
            return tokens
        return self._raw_images_to_tokens(image_input, target_t=states.shape[1], device=states.device, dtype=states.dtype)

    def decode(
        self,
        z: torch.Tensor,                  # (B, D)
        states: torch.Tensor,             # (B, T, state_dim)
        dec_tokens: torch.Tensor,         # 3rd-person DINO tokens (B,T,N,F)/(B,N,F) or raw images
        dec_tokens_wrist: torch.Tensor | None = None,  # wrist tokens/images (dual terminator only)
        frame_progress: torch.Tensor | None = None,  # (B, T), per-skill progress in [0,1]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          delta          (B, T, K, A) — dataset action chunk
          progress_pred  (B, T) — predicted per-skill progress in [0, 1]
          term_logits    (B, T) — skill-termination logits
        """
        dec_tokens = self._prepare_decoder_tokens(dec_tokens, states=states)
        if self.terminator_use_wrist:
            dec_tokens_wrist = self._prepare_decoder_tokens(dec_tokens_wrist, states=states)
        B, T = dec_tokens.shape[:2]

        if frame_progress is None:
            fi = torch.arange(T, device=states.device, dtype=states.dtype).view(1, T)
            fi = fi.expand(B, T) / max(T - 1, 1)
        else:
            fi = frame_progress.to(device=states.device, dtype=states.dtype)
        prog_tok = self.motion_prog_proj(fi.unsqueeze(-1))           # (B, T, H)

        z_seq = z.unsqueeze(1).expand(B, T, -1).to(states.dtype)     # (B, T, D)
        z_tok = self.dec_z_proj(z_seq)                               # (B, T, H) skill code, shared

        # ── (A) terminator: [z, current 3rd img, current wrist img, current state] (per step) ──
        progress_pred, term_logits = self._terminate(z_tok, states, dec_tokens, dec_tokens_wrist)

        # ── (B) reconstructor: start-frame state fixed, only progress varies per step ──
        delta = self._reconstruct_chunk(z_tok, states, prog_tok)

        return delta, progress_pred, term_logits

    def _terminate(
        self,
        z_tok: torch.Tensor,                     # (B, T, H) projected skill code
        states: torch.Tensor,                    # (B, T, state_dim) current proprioception
        dec_tokens: torch.Tensor,                # (B, T, N, F) current 3rd-person tokens
        dec_tokens_wrist: torch.Tensor | None,   # (B, T, N, F) current wrist tokens (dual mode only)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Terminator head → (progress, term_logits).
          single: [z, current 3rd img, current state]
          dual:   [z, current 3rd img, current wrist img, current state]
        progress (B, T) in [0, 1] (sigmoid); term_logits (B, T) raw logits. Shared by decode()
        (FSQ training) and predict_termination() (skillVLA inference, which sigmoids the logits)."""
        B, T = z_tok.shape[:2]
        s_term = self.term_state_proj(self._norm_state(states))       # (B, T, H)
        toks = [z_tok]
        if self.terminator_use_third:
            toks.append(self.dec_image_encoder_term(dec_tokens))               # (B, T, H) 3rd-person
        if self.terminator_use_wrist:
            if dec_tokens_wrist is None:
                raise ValueError("terminator_use_wrist=True requires wrist tokens.")
            toks.append(self.dec_image_encoder_term_wrist(dec_tokens_wrist))   # (B, T, H) wrist
        toks.append(s_term)
        tm_tokens = torch.stack(toks, dim=2).reshape(B * T, len(toks), -1)
        tm = self.term_pool(tm_tokens).view(B, T, -1)                 # (B, T, H)
        progress = torch.sigmoid(self.progress_head(tm)).squeeze(-1)  # (B, T) in [0, 1]
        term_logits = self.termination_head(tm).squeeze(-1)           # (B, T)
        return progress, term_logits

    def _reconstruct_chunk(
        self,
        z_tok: torch.Tensor,        # (B, T, H) projected skill code
        states: torch.Tensor,       # (B, T, state_dim) — only states[:, :1] (skill start) is used
        prog_tok: torch.Tensor,     # (B, T, H) projected per-step progress
    ) -> torch.Tensor:
        """Reconstructor motion branch → action chunk (B, T, K, A), denormalized.

        Image-free: inputs except progress are fixed at the skill start:
          [z, start_state, progress]            (3)
        z is the sole motion source (the skill-start image channel was removed).
        Shared by decode() (FSQ training, GT progress) and predict_action_chunk()
        (skillVLA inference, terminator progress).
        """
        B, T = z_tok.shape[:2]
        start_state = self.recon_state_proj(self._norm_state(states[:, :1])).expand(B, T, -1)   # (B, T, H)
        recon_tokens = [z_tok, start_state, prog_tok]
        mo = self.motion_pool(
            torch.stack(recon_tokens, dim=2).reshape(B * T, len(recon_tokens), -1)
        ).view(B, T, -1)                                                                # (B, T, H)
        dmin = self.delta_min.to(mo.device, mo.dtype).view(1, 1, 1, -1)
        dmax = self.delta_max.to(mo.device, mo.dtype).view(1, 1, 1, -1)
        d_tanh = torch.tanh(self.delta_head(mo)).view(B, T, self.chunk_size, self.action_dim)
        return (d_tanh + 1.0) / 2.0 * (dmax - dmin) + dmin                              # (B, T, K, A)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        ctrl_pts: torch.Tensor,      # (B, n_control, action_dim)
        lengths: torch.Tensor,       # (B,)
        states: torch.Tensor,             # (B, T, state_dim)
        dec_tokens: torch.Tensor,         # 3rd-person (B, T, n_tokens, feat_dim)
        dec_tokens_wrist: torch.Tensor | None = None,  # wrist (dual terminator only)
        frame_progress: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (delta, progress_pred, term_logits, indices)."""
        z_q, indices = self.encode(ctrl_pts, lengths)
        delta, progress_pred, term_logits = self.decode(
            z_q, states, dec_tokens, dec_tokens_wrist, frame_progress,
        )
        return delta, progress_pred, term_logits, indices

    # ── skillVLA-facing path (terminator + reconstructor) ─────────────────────

    @torch.no_grad()
    def predict_termination(
        self,
        z: torch.Tensor,                  # (B, D) skill code in z_q space (e.g. predicted by skillVLA)
        states: torch.Tensor,             # (B, T, state_dim) current proprioception
        dec_tokens: torch.Tensor,         # current 3rd-person tokens (B,T,N,F)/(B,N,F) or raw images
        dec_tokens_wrist: torch.Tensor | None = None,  # current wrist tokens (dual terminator only)
        quantize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run only the terminator (single: [z, 3rd img, state]; dual: + wrist img) →
        progress, termination.

        skillVLA's per-step path. z is snapped to the FSQ grid when quantize=True.
        Returns: progress (B, T) in [0, 1], termination (B, T) probability (sigmoid).
        """
        dec_tokens = self._prepare_decoder_tokens(dec_tokens, states=states)
        if self.terminator_use_wrist:
            dec_tokens_wrist = self._prepare_decoder_tokens(dec_tokens_wrist, states=states)
        B, T = dec_tokens.shape[:2]
        if quantize:
            lh = self.fsq.levels_half.to(z.device, z.dtype)
            z = torch.maximum(torch.minimum(torch.round(z), lh), -lh)
        z_tok = self.dec_z_proj(z.unsqueeze(1).expand(B, T, -1).to(states.dtype))
        progress, term_logits = self._terminate(z_tok, states, dec_tokens, dec_tokens_wrist)
        return progress, torch.sigmoid(term_logits)

    @torch.no_grad()
    def predict_action_chunk(
        self,
        z: torch.Tensor,            # (B, D) skill code (e.g. predicted by skillVLA)
        states: torch.Tensor,       # (B, T, state_dim) skill-START proprioception (only [:, :1] used)
        progress: torch.Tensor,     # (B,) or (B, T) per-step progress (e.g. terminator output)
        quantize: bool = True,
    ) -> torch.Tensor:
        """Run only the reconstructor: [z, start_state, progress] → action chunk (image-free).

        skillVLA's branch-2 path: states is the skill-START proprioception (fixed) and progress is
        the terminator's output. Returns delta (B, T, K, A) — denormalized actions.
        """
        B, T = states.shape[:2]
        if quantize:
            lh = self.fsq.levels_half.to(z.device, z.dtype)
            z = torch.maximum(torch.minimum(torch.round(z), lh), -lh)
        z_tok = self.dec_z_proj(z.unsqueeze(1).expand(B, T, -1).to(states.dtype))
        prog = progress.to(device=states.device, dtype=states.dtype)
        prog = prog.view(B, 1).expand(B, T) if prog.ndim == 1 else prog
        prog_tok = self.motion_prog_proj(prog.unsqueeze(-1))
        return self._reconstruct_chunk(z_tok, states, prog_tok)

    # ── inference helpers ─────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_numpy(
        self,
        trajectory: np.ndarray,      # (T, enc_dim) — full state trajectory
        device: str = "cpu",
    ) -> np.ndarray:
        """Returns latent vector z_q as numpy array, shape (D,)."""
        ctrl_pts, T = spline_encode(trajectory, self.n_control, self.spline_degree)
        cp_norm = self._np_norm_actions(ctrl_pts)
        cp_t = torch.from_numpy(cp_norm).float().unsqueeze(0).to(device)
        l_t  = torch.tensor([T], dtype=torch.long, device=device)
        z_q, _ = self.encode(cp_t, l_t)
        return z_q.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def encode_index(
        self,
        trajectory: np.ndarray,
        device: str = "cpu",
    ) -> int:
        """Returns scalar FSQ codebook index."""
        ctrl_pts, T = spline_encode(trajectory, self.n_control, self.spline_degree)
        cp_norm = self._np_norm_actions(ctrl_pts)
        cp_t = torch.from_numpy(cp_norm).float().unsqueeze(0).to(device)
        l_t  = torch.tensor([T], dtype=torch.long, device=device)
        _, idx = self.encode(cp_t, l_t)
        return int(idx[0].item())


# ── Dataset ───────────────────────────────────────────────────────────────────

class SplineFSQDataset(Dataset):
    """Dataset for SplineFSQAE.

    dec_tokens       : list of (T, n_tokens, feat_dim) float32 arrays — 3rd-person camera.
                       Token 0 = CLS; tokens 1..n_tokens-1 = patch tokens.
    dec_tokens_wrist : list of (T, n_tokens, feat_dim) wrist-camera arrays, or None for the
                       single-camera terminator (then no wrist tokens are emitted).
    states           : list of (T, 7) arrays (proprioception including gripper).
    deltas           : list of arrays with shape at least (T, action_dim).
                       Values are dataset actions. In chunk mode each array may extend
                       past the current skill to the end of the episode so chunks can
                       be supervised across skill boundaries.
    """

    def __init__(
        self,
        segments: list[np.ndarray],
        dec_tokens: list[np.ndarray],
        dec_tokens_wrist: list[np.ndarray] | None,
        states: list[np.ndarray],
        deltas: list[np.ndarray],
        n_control: int,
        spline_degree: int,
        action_min: np.ndarray,
        action_max: np.ndarray,
        delta_min: np.ndarray,
        delta_max: np.ndarray,
        chunk_size: int = 10,
        end_target_sigma: float = 0.0,
    ) -> None:
        self.chunk_size = chunk_size
        self.end_target_sigma = float(end_target_sigma)
        self.action_min = action_min.astype(np.float32)
        self.action_max = action_max.astype(np.float32)

        self.ctrl_pts: list[np.ndarray] = []
        self.lengths: list[int] = []
        # Keep DINO tokens in their on-disk dtype (float16): upcasting to float32 here doubled RAM
        # (e.g. pg14 = 86GB → 172GB) and OOM'd a 128G node. The collate buffer is float32 and upcasts
        # each batch on assignment, so the model still trains in float32 (identical inputs, half the RAM).
        self.dec_tokens       = [np.ascontiguousarray(t) for t in dec_tokens]
        self.dec_tokens_wrist = (
            None if dec_tokens_wrist is None else [np.ascontiguousarray(t) for t in dec_tokens_wrist]
        )
        self.states  = [s.astype(np.float32) for s in states]
        self.deltas  = [d.astype(np.float32) for d in deltas]

        for i, seg in enumerate(segments):
            cp, T = spline_encode(seg.astype(np.float32), n_control, spline_degree)
            if self.dec_tokens[i].shape[0] < T:
                raise ValueError(f"dec_tokens[{i}] length {self.dec_tokens[i].shape[0]} < skill length {T}")
            if self.dec_tokens_wrist is not None and self.dec_tokens_wrist[i].shape[0] < T:
                raise ValueError(f"dec_tokens_wrist[{i}] length {self.dec_tokens_wrist[i].shape[0]} < skill length {T}")
            if self.deltas[i].shape[0] < T:
                raise ValueError(f"deltas[{i}] length {self.deltas[i].shape[0]} < skill length {T}")
            cp_norm = (cp - self.action_min) / (self.action_max - self.action_min + 1e-8) * 2.0 - 1.0
            self.ctrl_pts.append(cp_norm)
            self.lengths.append(T)

    def __len__(self) -> int:
        return len(self.ctrl_pts)

    def __getitem__(self, idx: int) -> dict:
        T = self.lengths[idx]
        delta = self.deltas[idx]   # (T_future, A), dataset actions
        tokens = self.dec_tokens[idx]              # (T, n_tokens, F) 3rd-person
        T_future = len(delta)

        # Per-skill progress: 0 at the first step, 1 at the last step.
        progress = torch.arange(T, dtype=torch.float32) / max(T - 1, 1)  # (T,)
        # Termination target: hard 1-frame spike, or (end_target_sigma>0) a soft Gaussian bump that
        # rises to 1.0 at the last frame (a half-bump, since no frames exist past the end). The soft
        # form matches the fuzzy DP boundary and curbs the val overfit a sharp spike causes.
        if self.end_target_sigma > 0.0:
            t = torch.arange(T, dtype=torch.float32)
            termination = torch.exp(-((t - (T - 1)) ** 2) / (2.0 * self.end_target_sigma ** 2))
        else:
            termination = torch.zeros(T)
            termination[T - 1] = 1.0

        item: dict = {
            "ctrl":         torch.from_numpy(self.ctrl_pts[idx]),      # (n_control, A)
            "length":       torch.tensor(T, dtype=torch.long),
            "dec_tokens":       torch.from_numpy(tokens[:T]),         # (T, n_tokens, F) 3rd-person
            "state":        torch.from_numpy(self.states[idx][:T]),    # (T, 7)
            "frame_progress": progress,    # (T,) GT progress: motion input + termination-head target
            "termination":   termination,  # (T,) per-step end flag
        }
        if self.dec_tokens_wrist is not None:  # dual terminator only
            item["dec_tokens_wrist"] = torch.from_numpy(self.dec_tokens_wrist[idx][:T])

        # Chunk targets: pred[t,k] supervised by action[t+k]; slots may cross the
        # skill boundary into the next skill (same episode), only episode-end slots invalid.
        K = self.chunk_size
        t_idx = np.arange(T).reshape(-1, 1) + np.arange(K).reshape(1, -1)  # (T, K)
        valid = (t_idx < T_future)                                          # (T, K) bool
        t_clamped = np.minimum(t_idx, T_future - 1)
        chunk_target = delta[t_clamped]          # (T, K, A)
        chunk_target[~valid] = 0.0
        item["delta"]       = torch.from_numpy(chunk_target)              # (T, K, A)
        item["chunk_valid"] = torch.from_numpy(valid.astype(np.float32))  # (T, K)

        return item


# ── Collate ───────────────────────────────────────────────────────────────────

def collate_fsq_batch(batch: list[dict]) -> dict:
    lengths = torch.stack([b["length"] for b in batch])
    max_T = int(lengths.max().item())
    B = len(batch)

    ctrl         = torch.stack([b["ctrl"]         for b in batch])  # (B, n_ctrl, A)
    n_tokens = batch[0]["dec_tokens"].shape[1]
    feat_dim = batch[0]["dec_tokens"].shape[2]
    state_dim = batch[0]["state"].shape[-1]

    has_wrist = "dec_tokens_wrist" in batch[0]
    dec_tokens       = torch.zeros(B, max_T, n_tokens, feat_dim)
    dec_tokens_wrist = torch.zeros(B, max_T, n_tokens, feat_dim) if has_wrist else None
    state      = torch.zeros(B, max_T, state_dim)
    frame_progress = torch.zeros(B, max_T)
    termination = torch.zeros(B, max_T)
    mask       = torch.zeros(B, max_T, dtype=torch.bool)

    K, A = batch[0]["delta"].shape[1], batch[0]["delta"].shape[2]
    delta       = torch.zeros(B, max_T, K, A)
    chunk_valid = torch.zeros(B, max_T, K)

    for i, b in enumerate(batch):
        T = int(b["length"].item())
        dec_tokens[i, :T] = b["dec_tokens"]
        if has_wrist:
            dec_tokens_wrist[i, :T] = b["dec_tokens_wrist"]
        state[i, :T]      = b["state"]
        frame_progress[i, :T] = b["frame_progress"]
        termination[i, :T] = b["termination"]
        delta[i, :T]      = b["delta"]
        chunk_valid[i, :T] = b["chunk_valid"]
        mask[i, :T]       = True

    return {
        "ctrl":         ctrl,
        "lengths":      lengths,
        "dec_tokens":       dec_tokens,
        "dec_tokens_wrist": dec_tokens_wrist,
        "state":        state,
        "frame_progress": frame_progress,
        "delta":        delta,
        "termination":  termination,   # (B, max_T) per-step end flag
        "mask":         mask,
        "chunk_valid":  chunk_valid,   # (B, max_T, K)
    }


# ── Length-bucketed batching ──────────────────────────────────────────────────


class LengthBucketBatchSampler(Sampler):
    """Group similar-length skills into the same batch to cut padding waste, while keeping
    randomness. Per epoch (shuffle=True): permute all indices → cut into mega-buckets of
    batch_size*bucket_mult → sort within each bucket by length → form batches → shuffle batch order.

    Every skill appears exactly once per epoch, so epoch-level gradient coverage is unchanged; only
    mini-batch composition differs (a standard, low-impact technique). The decoder processes
    (B, max_T) tensors, so a batch's cost scales with its longest skill — homogeneous batches mean
    far fewer padding frames decoded. shuffle=False gives a deterministic global length sort (val)."""

    def __init__(self, lengths, batch_size: int, shuffle: bool = True, bucket_mult: int = 40, seed: int = 0):
        self.lengths = np.asarray(lengths)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.bucket_size = max(self.batch_size * bucket_mult, self.batch_size)
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.lengths)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            idx = rng.permutation(n)
        else:
            idx = np.argsort(self.lengths, kind="stable")  # deterministic global sort
        batches = []
        for s in range(0, n, self.bucket_size):
            bucket = idx[s:s + self.bucket_size]
            bucket = bucket[np.argsort(self.lengths[bucket], kind="stable")]
            for b in range(0, len(bucket), self.batch_size):
                batches.append(bucket[b:b + self.batch_size].tolist())
        if self.shuffle:
            rng.shuffle(batches)
        self.epoch += 1
        return iter(batches)


# ── Loss ──────────────────────────────────────────────────────────────────────

def fsqae_loss(
    pred_delta: torch.Tensor,       # (B, T, K, A)
    pred_progress: torch.Tensor,    # (B, T) in [0, 1]
    pred_term_logits: torch.Tensor, # (B, T)
    target_delta: torch.Tensor,
    target_progress: torch.Tensor,  # (B, T) in [0, 1]
    target_term: torch.Tensor,      # (B, T)
    mask: torch.Tensor,             # (B, T) bool
    delta_min: torch.Tensor,        # (A,)
    delta_max: torch.Tensor,        # (A,)
    chunk_valid: torch.Tensor,      # (B, T, K)
    delta_loss_weight: float = 1.0,
    progress_loss_weight: float = 1.0,
    end_loss_weight: float = 1.0,
    end_pos_weight: float | torch.Tensor = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (total, delta_loss, progress_loss, end_loss). No VQ loss term."""
    dev, dt = pred_delta.device, pred_delta.dtype

    # Action chunk loss (B, T, K, A): normalize to [-1,1], SmoothL1, valid-slot mean.
    dmin  = delta_min.to(dev, dt).view(1, 1, 1, -1)
    dmax  = delta_max.to(dev, dt).view(1, 1, 1, -1)
    scale = (dmax - dmin).clamp_min(1e-8)
    per_step_ka = F.smooth_l1_loss(
        (pred_delta   - dmin) / scale * 2.0 - 1.0,
        (target_delta - dmin) / scale * 2.0 - 1.0,
        reduction="none",
    ).mean(dim=-1)  # (B, T, K)
    cv = chunk_valid.to(dev, dt)
    per_step = (per_step_ka * cv).sum(dim=-1) / cv.sum(dim=-1).clamp_min(1.0)  # (B, T)

    m = mask.float()
    delta_loss = ((per_step * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)).mean()

    # Progress regression (per-step, masked).
    prog_per = F.smooth_l1_loss(pred_progress, target_progress.to(dev, dt), reduction="none")
    progress_loss = ((prog_per * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)).mean()

    # Termination classification (per-step, masked).
    pos_w = torch.as_tensor(end_pos_weight, device=dev, dtype=dt)
    end_per = F.binary_cross_entropy_with_logits(
        pred_term_logits, target_term.to(dev, dt), reduction="none", pos_weight=pos_w
    )
    end_loss = (end_per * m).sum() / m.sum().clamp_min(1.0)

    total = (
        delta_loss_weight * delta_loss
        + progress_loss_weight * progress_loss
        + end_loss_weight * end_loss
    )
    return total, delta_loss, progress_loss, end_loss


# ── End-signal metrics ────────────────────────────────────────────────────────

@torch.no_grad()
def end_signal_metrics(
    pred_end_logits: torch.Tensor,   # (B, T)
    target_end: torch.Tensor,        # (B, T)
    mask: torch.Tensor,              # (B, T)
    threshold: float = 0.5,
) -> dict[str, float]:
    valid = mask.bool()
    if valid.sum().item() == 0:
        return {"acc": 0.0, "precision": 0.0, "recall": 0.0, "positive_rate": 0.0}
    pred     = torch.sigmoid(pred_end_logits) >= threshold
    target   = target_end >= 0.5
    pred_v   = pred[valid]
    target_v = target[valid]
    tp = (pred_v & target_v).sum().float()
    fp = (pred_v & ~target_v).sum().float()
    fn = (~pred_v & target_v).sum().float()
    return {
        "acc":           float((pred_v == target_v).float().mean()),
        "precision":     float(tp / (tp + fp).clamp_min(1.0)),
        "recall":        float(tp / (tp + fn).clamp_min(1.0)),
        "positive_rate": float(pred_v.float().mean()),
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train_spline_fsqae(
    segments: list[np.ndarray],
    dec_tokens: list[np.ndarray],
    dec_tokens_wrist: list[np.ndarray] | None,
    decoder_states: list[np.ndarray],
    decoder_targets: list[np.ndarray],
    cfg: SplineFSQAEConfig,
    wandb_run=None,
    metadata: list[dict] | None = None,
    resume_from: str | None = None,
) -> SplineFSQAE:
    if not segments:
        raise ValueError("No skill segments provided.")

    enc_dim = cfg.enc_dim if cfg.enc_dim > 0 else segments[0].shape[-1]
    action_dim = cfg.action_dim if cfg.action_dim > 0 else decoder_targets[0].shape[-1]
    # Encoder normalization stats must match what the encoder actually sees: zero-grounded control
    # points (spline_encode grounds internally). Computing min/max on grounded segments keeps the
    # normalization centered on the relative-pose distribution.
    grounded = np.concatenate([zero_ground_trajectory(s) for s in segments])
    a_min = cfg.action_min if cfg.action_min is not None else grounded.min(axis=0)
    a_max = cfg.action_max if cfg.action_max is not None else grounded.max(axis=0)
    d_min = cfg.delta_min  if cfg.delta_min  is not None else np.concatenate(decoder_targets).min(axis=0)
    d_max = cfg.delta_max  if cfg.delta_max  is not None else np.concatenate(decoder_targets).max(axis=0)
    s_min = cfg.state_min  if cfg.state_min  is not None else np.concatenate(decoder_states).min(axis=0)
    s_max = cfg.state_max  if cfg.state_max  is not None else np.concatenate(decoder_states).max(axis=0)

    print(
        f"[SplineFSQAE] {len(segments)} segments | "
        f"n_control={cfg.n_control} fsq={cfg.fsq_levels} "
        f"hidden={cfg.hidden_dim} K={cfg.chunk_size}"
    )

    n_val = max(1, int(len(segments) * cfg.val_split))
    perm = np.random.permutation(len(segments))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    def take(xs: list, ids) -> list:
        return [xs[i] for i in ids]

    def mk_ds(ids) -> SplineFSQDataset:
        return SplineFSQDataset(
            take(segments, ids),
            take(dec_tokens, ids),
            take(dec_tokens_wrist, ids) if dec_tokens_wrist is not None else None,
            take(decoder_states, ids),
            take(decoder_targets, ids),
            cfg.n_control, cfg.spline_degree,
            a_min, a_max, d_min, d_max,
            cfg.chunk_size,
            cfg.end_target_sigma,
        )

    train_ds, val_ds = mk_ds(train_idx), mk_ds(val_idx)
    train_loader = DataLoader(
        train_ds, collate_fn=collate_fsq_batch,
        batch_sampler=LengthBucketBatchSampler(train_ds.lengths, cfg.batch_size, shuffle=True),
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, collate_fn=collate_fsq_batch,
        batch_sampler=LengthBucketBatchSampler(val_ds.lengths, cfg.batch_size, shuffle=False),
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=cfg.num_workers > 0,
    )

    model = SplineFSQAE(
        action_dim=action_dim,
        enc_dim=enc_dim,
        state_dim=cfg.state_dim,
        n_control=cfg.n_control,
        spline_degree=cfg.spline_degree,
        hidden_dim=cfg.hidden_dim,
        fsq_levels=cfg.fsq_levels,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        feat_dim=cfg.feat_dim,
        n_tokens=cfg.n_tokens,
        patch_grid=cfg.patch_grid,
        terminator_use_third=cfg.terminator_use_third,
        terminator_use_wrist=cfg.terminator_use_wrist,
        image_token_dim=cfg.image_token_dim,
        image_encoder_layers=cfg.image_encoder_layers,
        image_encoder_heads=cfg.image_encoder_heads,
        chunk_size=cfg.chunk_size,
        length_min=cfg.length_min, length_max=cfg.length_max,
        action_min=a_min, action_max=a_max,
        delta_min=d_min,  delta_max=d_max,
        state_min=s_min,  state_max=s_max,
    ).to(cfg.device)

    print(f"[SplineFSQAE] trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)
    start_epoch = 1
    best_val = math.inf

    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "optim_state" in ckpt:
            optim.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("val_loss", math.inf)
        print(f"[SplineFSQAE] resumed from {resume_from}, starting epoch {start_epoch}")

    def _save(path: str, epoch: int, val_loss: float) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "optim_state": optim.state_dict(),
                "cfg": cfg,
                "epoch": epoch,
                "val_loss": val_loss,
            },
            path,
        )

    def _step(batch: dict, train: bool):
        dev = cfg.device
        ctrl      = batch["ctrl"].to(dev)
        lengths   = batch["lengths"].to(dev)
        dec_tok   = batch["dec_tokens"].to(dev)
        dec_tok_wrist = batch["dec_tokens_wrist"]
        dec_tok_wrist = dec_tok_wrist.to(dev) if dec_tok_wrist is not None else None
        states    = batch["state"].to(dev)
        frame_progress = batch["frame_progress"].to(dev)
        tgt_delta = batch["delta"].to(dev)
        tgt_prog  = batch["frame_progress"].to(dev)   # GT progress is the termination-head target
        tgt_term  = batch["termination"].to(dev)
        mask      = batch["mask"].to(dev)
        cv        = batch["chunk_valid"].to(dev) if batch["chunk_valid"] is not None else None

        with torch.autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.bfloat16):
            pred_delta, pred_prog, pred_term, _ = model(
                ctrl, lengths, states, dec_tok, dec_tok_wrist, frame_progress,
            )
            total, d_l, p_l, e_l = fsqae_loss(
                pred_delta, pred_prog, pred_term, tgt_delta, tgt_prog, tgt_term, mask,
                model.delta_min, model.delta_max, cv,
                cfg.delta_loss_weight, cfg.progress_loss_weight, cfg.end_loss_weight, cfg.end_pos_weight,
            )
        if train:
            optim.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
        em = end_signal_metrics(pred_term, tgt_term, mask, cfg.end_threshold)
        return total.item(), d_l.item(), p_l.item(), e_l.item(), em

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        t_tot = t_d = t_p = t_e = t_acc = t_rec = n_tr = 0.0
        for batch in train_loader:
            tot, d, p, e, em = _step(batch, train=True)
            t_tot += tot; t_d += d; t_p += p; t_e += e
            t_acc += em["acc"]; t_rec += em["recall"]; n_tr += 1

        scheduler.step()

        model.eval()
        v_tot = v_d = v_p = v_e = v_acc = v_rec = n_vl = 0.0
        with torch.no_grad():
            for batch in val_loader:
                tot, d, p, e, em = _step(batch, train=False)
                v_tot += tot; v_d += d; v_p += p; v_e += e
                v_acc += em["acc"]; v_rec += em["recall"]; n_vl += 1

        n_tr = max(1, n_tr); n_vl = max(1, n_vl)
        train_loss = t_tot / n_tr
        val_loss = v_tot / n_vl
        log = {
            "train/loss": train_loss,
            "train/delta_loss": t_d / n_tr,
            "train/progress_loss": t_p / n_tr,
            "train/end_loss": t_e / n_tr,
            "train/end_acc": t_acc / n_tr,
            "train/end_recall": t_rec / n_tr,
            "val/loss": val_loss,
            "val/delta_loss": v_d / n_vl,
            "val/progress_loss": v_p / n_vl,
            "val/end_loss": v_e / n_vl,
            "val/end_acc": v_acc / n_vl,
            "val/end_recall": v_rec / n_vl,
            "lr": scheduler.get_last_lr()[0],
            "end_pos_weight": cfg.end_pos_weight,
        }
        if wandb_run is not None:
            wandb_run.log(log, step=epoch)

        if epoch % cfg.log_every == 0 or epoch == 1:
            print(
                f"[SplineFSQAE] {epoch:4d}/{cfg.epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"delta={log['train/delta_loss']:.4f}  prog={log['train/progress_loss']:.4f}  "
                f"end={log['train/end_loss']:.4f}  "
                f"end_acc={log['train/end_acc']:.3f}  end_rec={log['train/end_recall']:.3f}"
            )

        if cfg.save_path is not None and val_loss < best_val:
            best_val = val_loss
            _save(cfg.save_path, epoch, val_loss)

        if cfg.checkpoint_every > 0 and epoch % cfg.checkpoint_every == 0 and cfg.save_path is not None:
            ckpt_path = cfg.save_path.replace(".pt", f"_epoch{epoch:04d}.pt")
            _save(ckpt_path, epoch, val_loss)

    if cfg.save_path is not None and best_val == math.inf:
        _save(cfg.save_path, cfg.epochs, val_loss)

    print(f"[SplineFSQAE] done. best val loss: {best_val:.4f}")
    return model
