"""
Spline FSQ-AE — Finite Scalar Quantization skill encoder with observation-conditioned decoder.

Architecture
------------
  Encoder (fully Transformer-based):
    enc_image_encoder: pure-DINO mini encoder → 1 token each for start/end image
    enc_ctrl_proj/enc_len_proj (Linear A→H, 1→H): per-control-point + length tokens
    enc_traj_pool (Transformer + query pool): control-point/length tokens → 1 traj token
    enc_fusion_pool (Transformer + query pool): [start, traj, end] tokens → fused latent
    z_head (Linear H→D): produce pre-quantization latent
    FSQ: levels=[5,5,5], codebook_size=125, D=3, no learnable params

  Decoder (3 Transformer sub-networks, per timestep):
    dec_image_encoder       (DINO+flags) → motion image feature (B,T,H)
    dec_image_encoder_plain (DINO only)  → termination image feature (B,T,H)
    (1) skill_decoder: [z token, state token] → TokenTransformerPool → latent (H)
    (2) motion_reconstructor: [latent, img_feat(+flags), progress] tokens
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import make_interp_spline
from torch.utils.data import DataLoader, Dataset


# ── Spline codec ──────────────────────────────────────────────────────────────

GRIPPER_DIM = -1


def spline_encode(
    trajectory: np.ndarray,  # (T, action_dim)
    n_control: int,
    degree: int,
) -> tuple[np.ndarray, int]:
    """trajectory → control points (n_control, action_dim) + original length T."""
    T, D = trajectory.shape
    t_orig = np.linspace(0.0, 1.0, T)
    t_ctrl = np.linspace(0.0, 1.0, n_control)
    ctrl_pts = np.zeros((n_control, D), dtype=np.float32)
    gripper_idx = (D + GRIPPER_DIM) % D
    for d in range(D):
        k = 0 if d == gripper_idx else degree
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

    image_mode:
      - "dino_only": project DINO tokens 384 → N.
      - "dino_flags": project DINO tokens 384 → N-2, then append [is_changed,
        is_green] (CLS gets [0, 0]) so the internal width is still N. Flags thus
        occupy 2 of N channels — smaller N ⇒ larger relative flag weight.
    """

    def __init__(
        self,
        feat_dim: int,
        n_tokens: int,
        hidden_dim: int,
        token_dim: int,
        image_mode: str = "dino_flags",
        n_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert image_mode in ("dino_only", "dino_flags"), \
            f"image_mode must be 'dino_only' or 'dino_flags', got {image_mode!r}"
        assert n_tokens >= 1, "n_tokens must include at least the CLS token"
        assert token_dim % n_heads == 0, "token_dim must be divisible by n_heads"
        assert token_dim > 2, "token_dim must exceed the 2 flag channels"
        self.image_mode = image_mode
        self.n_tokens = n_tokens
        self.token_dim = token_dim
        # In flag mode the projection leaves room for the 2 appended flag channels.
        proj_out = token_dim - 2 if image_mode == "dino_flags" else token_dim
        self.token_proj = nn.Linear(feat_dim, proj_out)
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

    def forward(self, tokens: torch.Tensor, patch_flags: torch.Tensor | None = None) -> torch.Tensor:
        """
        tokens:      (B, T, N, F), token 0 is CLS and tokens 1..N-1 are patches.
        patch_flags: (B, T, N-1, 2), per-timestep patch [is_changed, is_green].
                     Ignored in dino_only mode.
        returns:     (B, T, hidden_dim)
        """
        B, T, N, _ = tokens.shape
        assert N == self.n_tokens, f"expected {self.n_tokens} image tokens, got {N}"
        x = self.token_proj(tokens.reshape(B * T, N, -1))   # (B*T, N, token_dim or token_dim-2)
        if self.image_mode == "dino_flags":
            assert patch_flags is not None, "patch_flags are required when image_mode='dino_flags'"
            assert patch_flags.shape == (B, T, N - 1, 2), \
                f"expected patch_flags {(B, T, N - 1, 2)}, got {tuple(patch_flags.shape)}"
            cls_flags = x.new_zeros(B * T, 1, 2)
            pf = patch_flags.to(device=x.device, dtype=x.dtype).reshape(B * T, N - 1, 2)
            x = torch.cat([x, torch.cat([cls_flags, pf], dim=1)], dim=-1)  # (B*T, N, token_dim)
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

@dataclass
class SplineFSQAEConfig:
    action_dim: int = 7
    state_dim: int = 7
    n_control: int = 30
    spline_degree: int = 3
    hidden_dim: int = 256
    fsq_levels: list[int] = field(default_factory=lambda: [5, 5, 5])
    num_layers: int = 3
    dropout: float = 0.1
    feat_dim: int = 384       # DINO token feature dimension
    n_tokens: int = 65        # 1 CLS + 64 patch tokens (8×8 grid)
    image_model_name: str = "/data2/dohyeon/SBD/models/dinov3-vits16"
    image_size: int = 224
    patch_grid: int = 8
    n_patch_raw: int = 196
    decoder_image_mode: str = "dino_flags"  # "dino_only" or "dino_flags"
    image_token_dim: int = 128  # internal per-patch width N in the image encoders
    image_encoder_layers: int = 1
    image_encoder_heads: int = 4
    chunk_size: int = 10  # motion always predicts a K-step action chunk
    max_length: float = 200.0
    delta_loss_weight: float = 1.0
    progress_loss_weight: float = 1.0
    end_loss_weight: float = 1.0
    end_pos_weight: float = 1.0
    end_threshold: float = 0.5
    lr: float = 3e-4
    batch_size: int = 64
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


# ── Model ─────────────────────────────────────────────────────────────────────

class SplineFSQAE(nn.Module):
    """Spline-encoded FSQ autoencoder with DINO-token-conditioned decoder."""

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        n_control: int = 30,
        spline_degree: int = 3,
        hidden_dim: int = 256,
        fsq_levels: list[int] | None = None,
        num_layers: int = 3,
        dropout: float = 0.1,
        feat_dim: int = 384,
        n_tokens: int = 65,
        image_model_name: str = "/data2/dohyeon/SBD/models/dinov3-vits16",
        image_size: int = 224,
        patch_grid: int = 8,
        n_patch_raw: int = 196,
        decoder_image_mode: str = "dino_flags",
        image_token_dim: int = 128,
        image_encoder_layers: int = 1,
        image_encoder_heads: int = 4,
        chunk_size: int = 10,
        max_length: float = 200.0,
        action_min: np.ndarray | None = None,
        action_max: np.ndarray | None = None,
        delta_min: np.ndarray | None = None,
        delta_max: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        if fsq_levels is None:
            fsq_levels = [5, 5, 5]
        assert decoder_image_mode in ("dino_only", "dino_flags"), \
            f"decoder_image_mode must be 'dino_only' or 'dino_flags', got {decoder_image_mode!r}"

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_control = n_control
        self.spline_degree = spline_degree
        self.max_length = max_length
        self.decoder_image_mode = decoder_image_mode
        self.chunk_size = chunk_size
        self.feat_dim = feat_dim
        self.n_tokens = n_tokens
        self.image_model_name = image_model_name
        self.image_size = image_size
        self.patch_grid = patch_grid
        self.n_patch_raw = n_patch_raw

        self.fsq = FSQ(fsq_levels)
        D = self.fsq.latent_dim
        H = hidden_dim

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc_image_encoder = ImageTokenEncoder(
            feat_dim=feat_dim,
            n_tokens=n_tokens,
            hidden_dim=H,
            token_dim=image_token_dim,
            image_mode="dino_only",
            n_layers=image_encoder_layers,
            n_heads=image_encoder_heads,
            dropout=dropout,
        )
        # Trajectory: each control point + the length scalar becomes one H-dim token,
        # a Transformer mixes them, and a learned query pools them into one traj token.
        self.enc_ctrl_proj = nn.Linear(action_dim, H)
        self.enc_len_proj = nn.Linear(1, H)
        self.enc_traj_pool = TokenTransformerPool(
            hidden_dim=H,
            n_tokens=n_control + 1,
            n_layers=image_encoder_layers,
            n_heads=image_encoder_heads,
            dropout=dropout,
        )
        # Fusion: [start_img, traj, end_img] tokens → Transformer → pooled latent.
        self.enc_fusion_pool = TokenTransformerPool(
            hidden_dim=H,
            n_tokens=3,
            n_layers=image_encoder_layers,
            n_heads=image_encoder_heads,
            dropout=dropout,
        )
        self.z_head = nn.Linear(H, D)

        # ── Decoder (3 Transformer sub-networks) ──────────────────────────────
        self.n_patches = n_tokens - 1  # token 0 = CLS, tokens 1.. = patches
        # Motion image encoder follows decoder_image_mode (DINO + optional flags).
        self.dec_image_encoder = ImageTokenEncoder(
            feat_dim=feat_dim,
            n_tokens=n_tokens,
            hidden_dim=H,
            token_dim=image_token_dim,
            image_mode=decoder_image_mode,
            n_layers=image_encoder_layers,
            n_heads=image_encoder_heads,
            dropout=dropout,
        )
        # Termination image encoder always uses pure DINO (no flags).
        self.dec_image_encoder_plain = ImageTokenEncoder(
            feat_dim=feat_dim,
            n_tokens=n_tokens,
            hidden_dim=H,
            token_dim=image_token_dim,
            image_mode="dino_only",
            n_layers=image_encoder_layers,
            n_heads=image_encoder_heads,
            dropout=dropout,
        )

        # (1) skill_decoder: [z token, state token] → latent (H).
        self.dec_z_proj = nn.Linear(D, H)
        self.dec_state_proj = nn.Linear(state_dim, H)
        self.skill_decoder_pool = TokenTransformerPool(
            hidden_dim=H, n_tokens=2,
            n_layers=image_encoder_layers, n_heads=image_encoder_heads, dropout=dropout,
        )

        # (2) motion_reconstructor: [latent, img_feat(+flags), progress] → action chunk.
        self.motion_prog_proj = nn.Linear(1, H)
        self.motion_pool = TokenTransformerPool(
            hidden_dim=H, n_tokens=3,
            n_layers=image_encoder_layers, n_heads=image_encoder_heads, dropout=dropout,
        )
        self.delta_head = nn.Linear(H, chunk_size * action_dim)

        # (3) skill_termination_predictor: [latent, img_feat(no flags)] → progress + termination.
        self.term_pool = TokenTransformerPool(
            hidden_dim=H, n_tokens=2,
            n_layers=image_encoder_layers, n_heads=image_encoder_heads, dropout=dropout,
        )
        self.progress_head = nn.Linear(H, 1)
        self.termination_head = nn.Linear(H, 1)

        _amin = action_min if action_min is not None else np.zeros(action_dim, dtype=np.float32)
        _amax = action_max if action_max is not None else np.ones(action_dim, dtype=np.float32)
        _dmin = delta_min if delta_min is not None else -np.ones(action_dim, dtype=np.float32)
        _dmax = delta_max if delta_max is not None else np.ones(action_dim, dtype=np.float32)
        self.register_buffer("action_min", torch.tensor(_amin, dtype=torch.float32))
        self.register_buffer("action_max", torch.tensor(_amax, dtype=torch.float32))
        self.register_buffer("delta_min",  torch.tensor(_dmin, dtype=torch.float32))
        self.register_buffer("delta_max",  torch.tensor(_dmax, dtype=torch.float32))
        object.__setattr__(self, "_raw_image_model", None)
        object.__setattr__(self, "_raw_image_mean", None)
        object.__setattr__(self, "_raw_image_std", None)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _np_norm_actions(self, ctrl_pts: np.ndarray) -> np.ndarray:
        lo = self.action_min.cpu().numpy()
        hi = self.action_max.cpu().numpy()
        return (ctrl_pts - lo) / (hi - lo + 1e-8) * 2.0 - 1.0

    # ── encode ────────────────────────────────────────────────────────────────

    def encode(
        self,
        ctrl_pts: torch.Tensor,      # (B, n_control, action_dim)
        lengths: torch.Tensor,       # (B,)
        start_tokens: torch.Tensor,  # (B, n_tokens, feat_dim)
        end_tokens: torch.Tensor,    # (B, n_tokens, feat_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (z_q, indices)."""
        B = ctrl_pts.size(0)
        start_feat = self.enc_image_encoder(start_tokens.unsqueeze(1)).squeeze(1)  # (B, H)
        end_feat   = self.enc_image_encoder(end_tokens.unsqueeze(1)).squeeze(1)    # (B, H)

        # Trajectory tokens: per control point + one length token → traj token.
        ctrl_tok = self.enc_ctrl_proj(ctrl_pts)                                    # (B, n_control, H)
        l_norm = (lengths.float() / self.max_length).view(B, 1, 1).to(ctrl_pts.dtype)
        len_tok = self.enc_len_proj(l_norm)                                        # (B, 1, H)
        traj_feat = self.enc_traj_pool(torch.cat([ctrl_tok, len_tok], dim=1))      # (B, H)

        # Fuse in order: start image → trajectory → end image.
        fusion_tokens = torch.stack([start_feat, traj_feat, end_feat], dim=1)      # (B, 3, H)
        h = self.enc_fusion_pool(fusion_tokens)                                    # (B, H)
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
        z: torch.Tensor,             # (B, D)
        states: torch.Tensor,        # (B, T, state_dim)
        dec_tokens: torch.Tensor,    # DINO tokens (B,T,N,F)/(B,N,F) or raw images
        patch_flags: torch.Tensor,   # (B, T, n_patches, 2)
        frame_progress: torch.Tensor | None = None,  # (B, T), per-skill progress in [0,1]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          delta          (B, T, K, A) — dataset action chunk
          progress_pred  (B, T) — predicted per-skill progress in [0, 1]
          term_logits    (B, T) — skill-termination logits
        """
        dec_tokens = self._prepare_decoder_tokens(dec_tokens, states=states)
        B, T = dec_tokens.shape[:2]

        img_feat = self.dec_image_encoder(dec_tokens, patch_flags)   # (B, T, H) motion (+flags)
        img_feat_plain = self.dec_image_encoder_plain(dec_tokens)    # (B, T, H) termination (no flags)

        if frame_progress is None:
            fi = torch.arange(T, device=states.device, dtype=states.dtype).view(1, T)
            fi = fi.expand(B, T) / max(T - 1, 1)
        else:
            fi = frame_progress.to(device=states.device, dtype=states.dtype)
        prog_in = fi.unsqueeze(-1)  # (B, T, 1)

        # (1) skill_decoder: [z token, state token] → latent.
        z_seq = z.unsqueeze(1).expand(B, T, -1).to(states.dtype)     # (B, T, D)
        z_tok = self.dec_z_proj(z_seq)                               # (B, T, H)
        s_tok = self.dec_state_proj(states)                          # (B, T, H)
        sd_tokens = torch.stack([z_tok, s_tok], dim=2).reshape(B * T, 2, -1)
        latent = self.skill_decoder_pool(sd_tokens).view(B, T, -1)   # (B, T, H)

        # (2) motion_reconstructor: [latent, img_feat(+flags), progress] → action chunk.
        prog_tok = self.motion_prog_proj(prog_in)                    # (B, T, H)
        mo_tokens = torch.stack([latent, img_feat, prog_tok], dim=2).reshape(B * T, 3, -1)
        mo = self.motion_pool(mo_tokens).view(B, T, -1)              # (B, T, H)

        dmin = self.delta_min.to(z.device, z.dtype).view(1, 1, 1, -1)
        dmax = self.delta_max.to(z.device, z.dtype).view(1, 1, 1, -1)
        K, A = self.chunk_size, self.action_dim
        d_tanh = torch.tanh(self.delta_head(mo)).view(B, T, K, A)
        delta = (d_tanh + 1.0) / 2.0 * (dmax - dmin) + dmin           # (B, T, K, A)

        # (3) skill_termination_predictor: [latent, img_feat(no flags)] → progress + termination.
        tm_tokens = torch.stack([latent, img_feat_plain], dim=2).reshape(B * T, 2, -1)
        tm = self.term_pool(tm_tokens).view(B, T, -1)                # (B, T, H)
        progress_pred = torch.sigmoid(self.progress_head(tm)).squeeze(-1)  # (B, T) in [0, 1]
        term_logits = self.termination_head(tm).squeeze(-1)          # (B, T)

        return delta, progress_pred, term_logits

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        ctrl_pts: torch.Tensor,      # (B, n_control, action_dim)
        lengths: torch.Tensor,       # (B,)
        start_tokens: torch.Tensor,  # (B, n_tokens, feat_dim)
        end_tokens: torch.Tensor,    # (B, n_tokens, feat_dim)
        states: torch.Tensor,        # (B, T, state_dim)
        dec_tokens: torch.Tensor,    # (B, T, n_tokens, feat_dim)
        patch_flags: torch.Tensor,   # (B, T, n_patches, 2)
        frame_progress: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (delta, progress_pred, term_logits, indices)."""
        z_q, indices = self.encode(ctrl_pts, lengths, start_tokens, end_tokens)
        delta, progress_pred, term_logits = self.decode(z_q, states, dec_tokens, patch_flags, frame_progress)
        return delta, progress_pred, term_logits, indices

    # ── skillVLA inference path ───────────────────────────────────────────────

    @torch.no_grad()
    def predict_termination(
        self,
        z: torch.Tensor,            # (B, D) skill latent in z_q space (e.g. predicted by skillVLA)
        states: torch.Tensor,       # (B, T, state_dim)
        dec_tokens: torch.Tensor,   # DINO tokens (B,T,N,F)/(B,N,F) or raw images
        quantize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run only skill_decoder + skill_termination_predictor (no motion, no flags).

        This is the subgraph skillVLA uses: a predicted skill latent z (D-dim, in
        z_q space) plus proprioceptive states and the current DINO image tokens
        yield per-step skill progress and termination. The motion_reconstructor and
        the flag image encoder are skipped entirely.

        z is treated as a point in the FSQ codeword space ([-lh, lh] per dim).
        With quantize=True it is snapped to the nearest codebook grid point
        (round + clamp), matching the discrete z_q the decoder saw in training.

        Returns:
          progress     (B, T) in [0, 1]
          termination  (B, T) termination probability (sigmoid applied)
        """
        dec_tokens = self._prepare_decoder_tokens(dec_tokens, states=states)
        B, T = dec_tokens.shape[:2]

        if quantize:
            lh = self.fsq.levels_half.to(z.device, z.dtype)
            z = torch.maximum(torch.minimum(torch.round(z), lh), -lh)

        img_feat_plain = self.dec_image_encoder_plain(dec_tokens)        # (B, T, H)

        z_seq = z.unsqueeze(1).expand(B, T, -1).to(states.dtype)         # (B, T, D)
        z_tok = self.dec_z_proj(z_seq)
        s_tok = self.dec_state_proj(states)
        sd_tokens = torch.stack([z_tok, s_tok], dim=2).reshape(B * T, 2, -1)
        latent = self.skill_decoder_pool(sd_tokens).view(B, T, -1)       # (B, T, H)

        tm_tokens = torch.stack([latent, img_feat_plain], dim=2).reshape(B * T, 2, -1)
        tm = self.term_pool(tm_tokens).view(B, T, -1)
        progress = torch.sigmoid(self.progress_head(tm)).squeeze(-1)         # (B, T)
        termination = torch.sigmoid(self.termination_head(tm)).squeeze(-1)   # (B, T)
        return progress, termination

    # ── inference helpers ─────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_numpy(
        self,
        trajectory: np.ndarray,      # (T, action_dim)
        start_tokens: np.ndarray,    # (n_tokens, feat_dim)
        end_tokens: np.ndarray,      # (n_tokens, feat_dim)
        device: str = "cpu",
    ) -> np.ndarray:
        """Returns latent vector z_q as numpy array, shape (D,)."""
        ctrl_pts, T = spline_encode(trajectory, self.n_control, self.spline_degree)
        cp_norm = self._np_norm_actions(ctrl_pts)
        cp_t = torch.from_numpy(cp_norm).float().unsqueeze(0).to(device)
        l_t  = torch.tensor([T], dtype=torch.long, device=device)
        st_t = torch.from_numpy(start_tokens.astype(np.float32)).unsqueeze(0).to(device)
        et_t = torch.from_numpy(end_tokens.astype(np.float32)).unsqueeze(0).to(device)
        z_q, _ = self.encode(cp_t, l_t, st_t, et_t)
        return z_q.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def encode_index(
        self,
        trajectory: np.ndarray,
        start_tokens: np.ndarray,
        end_tokens: np.ndarray,
        device: str = "cpu",
    ) -> int:
        """Returns scalar FSQ codebook index."""
        ctrl_pts, T = spline_encode(trajectory, self.n_control, self.spline_degree)
        cp_norm = self._np_norm_actions(ctrl_pts)
        cp_t = torch.from_numpy(cp_norm).float().unsqueeze(0).to(device)
        l_t  = torch.tensor([T], dtype=torch.long, device=device)
        st_t = torch.from_numpy(start_tokens.astype(np.float32)).unsqueeze(0).to(device)
        et_t = torch.from_numpy(end_tokens.astype(np.float32)).unsqueeze(0).to(device)
        _, idx = self.encode(cp_t, l_t, st_t, et_t)
        return int(idx[0].item())


# ── Dataset ───────────────────────────────────────────────────────────────────

class SplineFSQDataset(Dataset):
    """Dataset for SplineFSQAE.

    dec_tokens : list of (T, n_tokens, feat_dim) float32 arrays.
                 Token 0 = CLS; tokens 1..n_tokens-1 = patch tokens.
    patch_flags: list of (T, n_patches, 2) float32 arrays — timestep-aligned
                 [is_red, is_green] flags per patch.
    states     : list of (T, 7) arrays (proprioception including gripper).
    deltas     : list of arrays with shape at least (T, action_dim).
                 Values are dataset actions. In chunk mode each array may extend
                 past the current skill to the end of the episode so chunks can
                 be supervised across skill boundaries.
    """

    def __init__(
        self,
        segments: list[np.ndarray],
        dec_tokens: list[np.ndarray],
        patch_flags: list[np.ndarray],
        states: list[np.ndarray],
        deltas: list[np.ndarray],
        n_control: int,
        spline_degree: int,
        action_min: np.ndarray,
        action_max: np.ndarray,
        delta_min: np.ndarray,
        delta_max: np.ndarray,
        max_length: float,
        chunk_size: int = 10,
    ) -> None:
        self.chunk_size = chunk_size
        self.max_length = max_length
        self.action_min = action_min.astype(np.float32)
        self.action_max = action_max.astype(np.float32)

        self.ctrl_pts: list[np.ndarray] = []
        self.lengths: list[int] = []
        self.dec_tokens  = [t.astype(np.float32) for t in dec_tokens]
        self.patch_flags = [f.astype(np.float32) for f in patch_flags]
        self.states  = [s.astype(np.float32) for s in states]
        self.deltas  = [d.astype(np.float32) for d in deltas]

        for i, seg in enumerate(segments):
            cp, T = spline_encode(seg.astype(np.float32), n_control, spline_degree)
            if self.dec_tokens[i].shape[0] < T:
                raise ValueError(f"dec_tokens[{i}] length {self.dec_tokens[i].shape[0]} < skill length {T}")
            if self.patch_flags[i].shape[0] < T:
                raise ValueError(f"patch_flags[{i}] length {self.patch_flags[i].shape[0]} < skill length {T}")
            if self.deltas[i].shape[0] < T:
                raise ValueError(f"deltas[{i}] length {self.deltas[i].shape[0]} < skill length {T}")
            if self.patch_flags[i].ndim != 3 or self.patch_flags[i].shape[-1] != 2:
                raise ValueError(f"patch_flags[{i}] must have shape (T, n_patches, 2), got {self.patch_flags[i].shape}")
            cp_norm = (cp - self.action_min) / (self.action_max - self.action_min + 1e-8) * 2.0 - 1.0
            self.ctrl_pts.append(cp_norm)
            self.lengths.append(T)

    def __len__(self) -> int:
        return len(self.ctrl_pts)

    def __getitem__(self, idx: int) -> dict:
        T = self.lengths[idx]
        delta = self.deltas[idx]   # (T_future, A), dataset actions
        tokens = self.dec_tokens[idx]  # (T, n_tokens, F)
        T_future = len(delta)

        # Per-skill progress: 0 at the first step, 1 at the last step.
        progress = torch.arange(T, dtype=torch.float32) / max(T - 1, 1)  # (T,)
        # Termination: 1 at the last step, else 0.
        termination = torch.zeros(T)
        termination[T - 1] = 1.0

        item: dict = {
            "ctrl":         torch.from_numpy(self.ctrl_pts[idx]),      # (n_control, A)
            "length":       torch.tensor(T, dtype=torch.long),
            "start_tokens": torch.from_numpy(tokens[0]),               # (n_tokens, F)
            "end_tokens":   torch.from_numpy(tokens[T - 1]),           # (n_tokens, F)
            "dec_tokens":   torch.from_numpy(tokens[:T]),              # (T, n_tokens, F)
            "patch_flags":  torch.from_numpy(self.patch_flags[idx][:T]),   # (T, n_patches, 2)
            "state":        torch.from_numpy(self.states[idx][:T]),    # (T, 7)
            "frame_progress": progress,    # (T,) GT progress: motion input + termination-head target
            "termination":   termination,  # (T,) per-step end flag
        }

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
    start_tokens = torch.stack([b["start_tokens"] for b in batch])  # (B, n_tokens, F)
    end_tokens   = torch.stack([b["end_tokens"]   for b in batch])  # (B, n_tokens, F)
    n_tokens = batch[0]["dec_tokens"].shape[1]
    feat_dim = batch[0]["dec_tokens"].shape[2]
    n_patches = batch[0]["patch_flags"].shape[1]
    state_dim = batch[0]["state"].shape[-1]

    dec_tokens = torch.zeros(B, max_T, n_tokens, feat_dim)
    patch_flags = torch.zeros(B, max_T, n_patches, 2)
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
        patch_flags[i, :T] = b["patch_flags"]
        state[i, :T]      = b["state"]
        frame_progress[i, :T] = b["frame_progress"]
        termination[i, :T] = b["termination"]
        delta[i, :T]      = b["delta"]
        chunk_valid[i, :T] = b["chunk_valid"]
        mask[i, :T]       = True

    return {
        "ctrl":         ctrl,
        "lengths":      lengths,
        "start_tokens": start_tokens,
        "end_tokens":   end_tokens,
        "dec_tokens":   dec_tokens,
        "patch_flags":  patch_flags,
        "state":        state,
        "frame_progress": frame_progress,
        "delta":        delta,
        "termination":  termination,   # (B, max_T) per-step end flag
        "mask":         mask,
        "chunk_valid":  chunk_valid,   # (B, max_T, K)
    }


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
    patch_flags: list[np.ndarray],
    decoder_states: list[np.ndarray],
    decoder_targets: list[np.ndarray],
    cfg: SplineFSQAEConfig,
    wandb_run=None,
    metadata: list[dict] | None = None,
    resume_from: str | None = None,
) -> SplineFSQAE:
    if not segments:
        raise ValueError("No skill segments provided.")

    action_dim = cfg.action_dim if cfg.action_dim > 0 else segments[0].shape[-1]
    a_min = cfg.action_min if cfg.action_min is not None else np.concatenate(segments).min(axis=0)
    a_max = cfg.action_max if cfg.action_max is not None else np.concatenate(segments).max(axis=0)
    d_min = cfg.delta_min  if cfg.delta_min  is not None else np.concatenate(decoder_targets).min(axis=0)
    d_max = cfg.delta_max  if cfg.delta_max  is not None else np.concatenate(decoder_targets).max(axis=0)

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
            take(patch_flags, ids),
            take(decoder_states, ids),
            take(decoder_targets, ids),
            cfg.n_control, cfg.spline_degree,
            a_min, a_max, d_min, d_max,
            cfg.max_length,
            cfg.chunk_size,
        )

    train_loader = DataLoader(mk_ds(train_idx), batch_size=cfg.batch_size, shuffle=True,  collate_fn=collate_fsq_batch)
    val_loader   = DataLoader(mk_ds(val_idx),   batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fsq_batch)

    model = SplineFSQAE(
        action_dim=action_dim,
        state_dim=cfg.state_dim,
        n_control=cfg.n_control,
        spline_degree=cfg.spline_degree,
        hidden_dim=cfg.hidden_dim,
        fsq_levels=cfg.fsq_levels,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        feat_dim=cfg.feat_dim,
        n_tokens=cfg.n_tokens,
        decoder_image_mode=cfg.decoder_image_mode,
        image_token_dim=cfg.image_token_dim,
        image_encoder_layers=cfg.image_encoder_layers,
        image_encoder_heads=cfg.image_encoder_heads,
        chunk_size=cfg.chunk_size,
        max_length=cfg.max_length,
        action_min=a_min, action_max=a_max,
        delta_min=d_min,  delta_max=d_max,
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
        start_tok = batch["start_tokens"].to(dev)
        end_tok   = batch["end_tokens"].to(dev)
        dec_tok   = batch["dec_tokens"].to(dev)
        pf        = batch["patch_flags"].to(dev)
        states    = batch["state"].to(dev)
        frame_progress = batch["frame_progress"].to(dev)
        tgt_delta = batch["delta"].to(dev)
        tgt_prog  = batch["frame_progress"].to(dev)   # GT progress is the termination-head target
        tgt_term  = batch["termination"].to(dev)
        mask      = batch["mask"].to(dev)
        cv        = batch["chunk_valid"].to(dev) if batch["chunk_valid"] is not None else None

        pred_delta, pred_prog, pred_term, _ = model(
            ctrl, lengths, start_tok, end_tok, states, dec_tok, pf, frame_progress
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
