"""
Spline FSQ-AE — Finite Scalar Quantization skill encoder with observation-conditioned decoder.

Architecture
------------
  Encoder:
    enc_img_proj (shared Linear F→H): mean-pool projected start/end DINO tokens
    enc_traj_proj (Linear n_control*A+1 → H): control points + length_norm
    enc_mlp (MLP 3H→H): fuse start/end/traj features
    z_head (Linear H→D): produce pre-quantization latent
    FSQ: levels=[5,5,5], codebook_size=125, D=3, no learnable params

  Decoder image encoder (per timestep):
    dino_only:  DINO tokens → Linear(F→H) → positional embedding
    dino_flags: DINO tokens + patch flags → Linear(F+2→H) → positional embedding
    Both modes then use a small Transformer encoder and learned-query pooling.
    dec_mlp (MLP D+S+H+1 → H): z + state(7D) + img_feat + frame_idx_norm
    delta_head: Linear(H → A) in single_step, Linear(H → K*A) in chunk
    end_head:   Linear(H → 1) in single_step, Linear(H → K) in chunk
                — chunk mode predicts which slot inside the K-step chunk ends the skill

encode_numpy  → latent vector z_q (numpy, shape D)
encode_index  → scalar codebook index (int)
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


# ── MLP block ─────────────────────────────────────────────────────────────────

class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageTokenEncoder(nn.Module):
    """Turn DINO CLS+patch tokens into one decoder image feature.

    image_mode:
      - "dino_only": use pure DINO CLS + patch tokens.
      - "dino_flags": append [is_red, is_green] to patch tokens; CLS gets [0, 0].
    """

    def __init__(
        self,
        feat_dim: int,
        n_tokens: int,
        hidden_dim: int,
        image_mode: str = "dino_flags",
        n_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert image_mode in ("dino_only", "dino_flags"), \
            f"image_mode must be 'dino_only' or 'dino_flags', got {image_mode!r}"
        assert n_tokens >= 1, "n_tokens must include at least the CLS token"
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.image_mode = image_mode
        self.n_tokens = n_tokens
        in_dim = feat_dim + (2 if image_mode == "dino_flags" else 0)
        self.token_proj = nn.Linear(in_dim, hidden_dim)
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

    def forward(self, tokens: torch.Tensor, patch_flags: torch.Tensor) -> torch.Tensor:
        """
        tokens:      (B, T, N, F), token 0 is CLS and tokens 1..N-1 are patches.
        patch_flags: (B, N-1, 2), per-patch [is_red, is_green]. Ignored in dino_only mode.
        returns:     (B, T, H)
        """
        B, T, N, _ = tokens.shape
        assert N == self.n_tokens, f"expected {self.n_tokens} image tokens, got {N}"
        if self.image_mode == "dino_flags":
            cls_flags = tokens.new_zeros(B, T, 1, 2)
            pf = patch_flags.unsqueeze(1).expand(B, T, -1, -1).to(tokens.dtype)
            tokens = torch.cat([tokens, torch.cat([cls_flags, pf], dim=2)], dim=-1)

        x = self.token_proj(tokens.reshape(B * T, N, -1))
        x = x + self.pos_embed.to(dtype=x.dtype)
        x = self.encoder(x)
        q = self.query.to(dtype=x.dtype).expand(B * T, -1, -1)
        pooled, _ = self.pool(q, x, x, need_weights=False)
        return self.out_norm(pooled.squeeze(1)).view(B, T, -1)


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
    decoder_image_mode: str = "dino_flags"  # "dino_only" or "dino_flags"
    image_encoder_layers: int = 1
    image_encoder_heads: int = 4
    decoder_output_mode: str = "single_step"  # "single_step" or "chunk"
    chunk_size: int = 10
    max_length: float = 200.0
    delta_loss_weight: float = 1.0
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
        decoder_image_mode: str = "dino_flags",
        image_encoder_layers: int = 1,
        image_encoder_heads: int = 4,
        decoder_output_mode: str = "single_step",
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
        assert decoder_output_mode in ("single_step", "chunk"), \
            f"decoder_output_mode must be 'single_step' or 'chunk', got {decoder_output_mode!r}"
        assert decoder_image_mode in ("dino_only", "dino_flags"), \
            f"decoder_image_mode must be 'dino_only' or 'dino_flags', got {decoder_image_mode!r}"

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_control = n_control
        self.spline_degree = spline_degree
        self.max_length = max_length
        self.decoder_output_mode = decoder_output_mode
        self.decoder_image_mode = decoder_image_mode
        self.chunk_size = chunk_size

        self.fsq = FSQ(fsq_levels)
        D = self.fsq.latent_dim
        H = hidden_dim

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc_img_proj = nn.Linear(feat_dim, H)  # shared for start / end tokens
        self.enc_traj_proj = nn.Linear(n_control * action_dim + 1, H)
        enc_layers: list[nn.Module] = [MLPBlock(3 * H, H, dropout)]
        for _ in range(num_layers - 1):
            enc_layers.append(MLPBlock(H, H, dropout))
        self.enc_mlp = nn.Sequential(*enc_layers)
        self.z_head = nn.Linear(H, D)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.n_patches = n_tokens - 1  # token 0 = CLS, tokens 1.. = patches
        self.image_encoder = ImageTokenEncoder(
            feat_dim=feat_dim,
            n_tokens=n_tokens,
            hidden_dim=H,
            image_mode=decoder_image_mode,
            n_layers=image_encoder_layers,
            n_heads=image_encoder_heads,
            dropout=dropout,
        )
        dec_in = D + state_dim + H + 1   # z, state, img_feat, frame_idx_norm
        dec_layers: list[nn.Module] = [MLPBlock(dec_in, H, dropout)]
        for _ in range(num_layers - 1):
            dec_layers.append(MLPBlock(H, H, dropout))
        self.dec_mlp = nn.Sequential(*dec_layers)

        delta_out = chunk_size * action_dim if decoder_output_mode == "chunk" else action_dim
        self.delta_head = nn.Linear(H, delta_out)
        end_out = chunk_size if decoder_output_mode == "chunk" else 1
        self.end_head = nn.Linear(H, end_out)

        _amin = action_min if action_min is not None else np.zeros(action_dim, dtype=np.float32)
        _amax = action_max if action_max is not None else np.ones(action_dim, dtype=np.float32)
        _dmin = delta_min if delta_min is not None else -np.ones(action_dim, dtype=np.float32)
        _dmax = delta_max if delta_max is not None else np.ones(action_dim, dtype=np.float32)
        self.register_buffer("action_min", torch.tensor(_amin, dtype=torch.float32))
        self.register_buffer("action_max", torch.tensor(_amax, dtype=torch.float32))
        self.register_buffer("delta_min",  torch.tensor(_dmin, dtype=torch.float32))
        self.register_buffer("delta_max",  torch.tensor(_dmax, dtype=torch.float32))

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
        start_feat = self.enc_img_proj(start_tokens).mean(dim=1)  # (B, H)
        end_feat   = self.enc_img_proj(end_tokens).mean(dim=1)    # (B, H)
        ctrl_flat = ctrl_pts.reshape(B, -1)
        l_norm = (lengths.float() / self.max_length).unsqueeze(-1).to(ctrl_pts.dtype)
        traj_feat = self.enc_traj_proj(torch.cat([ctrl_flat, l_norm], dim=-1))  # (B, H)
        h = self.enc_mlp(torch.cat([start_feat, end_feat, traj_feat], dim=-1))  # (B, H)
        z_e = self.z_head(h)
        return self.fsq(z_e)

    # ── decode ────────────────────────────────────────────────────────────────

    def decode(
        self,
        z: torch.Tensor,             # (B, D)
        states: torch.Tensor,        # (B, T, state_dim)
        dec_tokens: torch.Tensor,    # (B, T, n_tokens, feat_dim)
        patch_flags: torch.Tensor,   # (B, n_patches, 2)
        frame_indices: torch.Tensor | None = None,  # (B, T) raw frame idx
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          single_step → delta (B, T, A),    end_logit (B, T)
          chunk       → delta (B, T, K, A), end_logit (B, T, K)
        """
        B, T, n_tokens, _ = dec_tokens.shape

        img_feat = self.image_encoder(dec_tokens, patch_flags)  # (B, T, H)

        if frame_indices is None:
            fi = torch.arange(T, device=states.device, dtype=states.dtype).view(1, T)
            fi = fi.expand(B, T)
        else:
            fi = frame_indices.to(device=states.device, dtype=states.dtype)
        fi = (fi / self.max_length).unsqueeze(-1)  # (B, T, 1)

        z_seq = z.unsqueeze(1).expand(B, T, -1).to(states.dtype)
        x = torch.cat([z_seq, states, img_feat, fi], dim=-1)       # (B, T, D+S+H+1)
        h = self.dec_mlp(x.reshape(B * T, -1)).view(B, T, -1)      # (B, T, H)

        dmin = self.delta_min.to(z.device, z.dtype)
        dmax = self.delta_max.to(z.device, z.dtype)

        if self.decoder_output_mode == "single_step":
            d_tanh = torch.tanh(self.delta_head(h))
            delta = (d_tanh + 1.0) / 2.0 * (dmax - dmin) + dmin   # (B, T, A)
        else:
            K, A = self.chunk_size, self.action_dim
            d_tanh = torch.tanh(self.delta_head(h)).view(B, T, K, A)
            dmin_k = dmin.view(1, 1, 1, -1)
            dmax_k = dmax.view(1, 1, 1, -1)
            delta = (d_tanh + 1.0) / 2.0 * (dmax_k - dmin_k) + dmin_k  # (B, T, K, A)

        end_logits = self.end_head(h)
        if self.decoder_output_mode == "single_step":
            end_logits = end_logits.squeeze(-1)  # (B, T)
        return delta, end_logits

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        ctrl_pts: torch.Tensor,      # (B, n_control, action_dim)
        lengths: torch.Tensor,       # (B,)
        start_tokens: torch.Tensor,  # (B, n_tokens, feat_dim)
        end_tokens: torch.Tensor,    # (B, n_tokens, feat_dim)
        states: torch.Tensor,        # (B, T, state_dim)
        dec_tokens: torch.Tensor,    # (B, T, n_tokens, feat_dim)
        patch_flags: torch.Tensor,   # (B, n_patches, 2)
        frame_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (delta, end_logits, indices)."""
        z_q, indices = self.encode(ctrl_pts, lengths, start_tokens, end_tokens)
        delta, end_logits = self.decode(z_q, states, dec_tokens, patch_flags, frame_indices)
        return delta, end_logits, indices

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
    patch_flags: list of (n_patches, 2) float32 arrays — [is_red, is_green] per patch.
                 Constant per skill; broadcast to all T in model forward.
    states     : list of (T, 7) arrays (proprioception including gripper).
    deltas     : list of (T, action_dim) arrays — target action deltas.
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
        decoder_output_mode: str = "single_step",
        chunk_size: int = 10,
    ) -> None:
        self.decoder_output_mode = decoder_output_mode
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

        for seg in segments:
            cp, T = spline_encode(seg.astype(np.float32), n_control, spline_degree)
            cp_norm = (cp - self.action_min) / (self.action_max - self.action_min + 1e-8) * 2.0 - 1.0
            self.ctrl_pts.append(cp_norm)
            self.lengths.append(T)

    def __len__(self) -> int:
        return len(self.ctrl_pts)

    def __getitem__(self, idx: int) -> dict:
        T = self.lengths[idx]
        delta = self.deltas[idx]   # (T, A)
        tokens = self.dec_tokens[idx]  # (T, n_tokens, F)

        item: dict = {
            "ctrl":         torch.from_numpy(self.ctrl_pts[idx]),      # (n_control, A)
            "length":       torch.tensor(T, dtype=torch.long),
            "start_tokens": torch.from_numpy(tokens[0]),               # (n_tokens, F)
            "end_tokens":   torch.from_numpy(tokens[T - 1]),           # (n_tokens, F)
            "dec_tokens":   torch.from_numpy(tokens[:T]),              # (T, n_tokens, F)
            "patch_flags":  torch.from_numpy(self.patch_flags[idx]),   # (n_patches, 2)
            "state":        torch.from_numpy(self.states[idx][:T]),    # (T, 7)
            "frame_idx":    torch.arange(T, dtype=torch.float32),      # (T,)
        }

        if self.decoder_output_mode == "single_step":
            item["delta"] = torch.from_numpy(delta[:T])   # (T, A)
            end = torch.zeros(T)
            end[-1] = 1.0
            item["end"] = end
        else:
            K, A = self.chunk_size, delta.shape[-1]
            # vectorised chunk construction
            t_idx = np.arange(T).reshape(-1, 1) + np.arange(K).reshape(1, -1)  # (T, K)
            valid = (t_idx < T)                                                  # (T, K) bool
            t_clamped = np.minimum(t_idx, T - 1)
            chunk_target = delta[t_clamped]          # (T, K, A)
            chunk_target[~valid] = 0.0
            chunk_valid = valid.astype(np.float32)   # (T, K)
            # end flag: one slot per chunk horizon. If skill ends at t+k, slot k is 1.
            chunk_end = (t_idx == (T - 1)).astype(np.float32)  # (T, K)
            item["delta"]       = torch.from_numpy(chunk_target)  # (T, K, A)
            item["chunk_valid"] = torch.from_numpy(chunk_valid)   # (T, K)
            item["end"]         = torch.from_numpy(chunk_end)     # (T, K)

        return item


# ── Collate ───────────────────────────────────────────────────────────────────

def collate_fsq_batch(batch: list[dict]) -> dict:
    lengths = torch.stack([b["length"] for b in batch])
    max_T = int(lengths.max().item())
    B = len(batch)

    ctrl         = torch.stack([b["ctrl"]         for b in batch])  # (B, n_ctrl, A)
    start_tokens = torch.stack([b["start_tokens"] for b in batch])  # (B, n_tokens, F)
    end_tokens   = torch.stack([b["end_tokens"]   for b in batch])  # (B, n_tokens, F)
    patch_flags  = torch.stack([b["patch_flags"]  for b in batch])  # (B, n_patches, 2)

    n_tokens = batch[0]["dec_tokens"].shape[1]
    feat_dim = batch[0]["dec_tokens"].shape[2]
    state_dim = batch[0]["state"].shape[-1]

    dec_tokens = torch.zeros(B, max_T, n_tokens, feat_dim)
    state      = torch.zeros(B, max_T, state_dim)
    frame_idx  = torch.zeros(B, max_T)
    mask       = torch.zeros(B, max_T, dtype=torch.bool)

    is_chunk = "chunk_valid" in batch[0]
    if is_chunk:
        K, A = batch[0]["delta"].shape[1], batch[0]["delta"].shape[2]
        delta       = torch.zeros(B, max_T, K, A)
        end         = torch.zeros(B, max_T, K)
        chunk_valid = torch.zeros(B, max_T, K)
    else:
        A = batch[0]["delta"].shape[1]
        delta       = torch.zeros(B, max_T, A)
        end         = torch.zeros(B, max_T)
        chunk_valid = None

    for i, b in enumerate(batch):
        T = int(b["length"].item())
        dec_tokens[i, :T] = b["dec_tokens"]
        state[i, :T]      = b["state"]
        frame_idx[i, :T]  = b["frame_idx"]
        delta[i, :T]      = b["delta"]
        end[i, :T]        = b["end"]
        mask[i, :T]       = True
        if is_chunk:
            chunk_valid[i, :T] = b["chunk_valid"]

    return {
        "ctrl":         ctrl,
        "lengths":      lengths,
        "start_tokens": start_tokens,
        "end_tokens":   end_tokens,
        "dec_tokens":   dec_tokens,
        "patch_flags":  patch_flags,
        "state":        state,
        "frame_idx":    frame_idx,
        "delta":        delta,
        "end":          end,
        "mask":         mask,
        "chunk_valid":  chunk_valid,  # None in single_step mode
    }


# ── Loss ──────────────────────────────────────────────────────────────────────

def fsqae_loss(
    pred_delta: torch.Tensor,       # (B, T, A) or (B, T, K, A)
    pred_end_logits: torch.Tensor,  # (B, T) or (B, T, K)
    target_delta: torch.Tensor,
    target_end: torch.Tensor,       # (B, T) or (B, T, K)
    mask: torch.Tensor,             # (B, T) bool
    delta_min: torch.Tensor,        # (A,)
    delta_max: torch.Tensor,        # (A,)
    chunk_valid: torch.Tensor | None = None,  # (B, T, K)
    delta_loss_weight: float = 1.0,
    end_loss_weight: float = 1.0,
    end_pos_weight: float | torch.Tensor = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (total, delta_loss, end_loss). No VQ loss term."""
    dev, dt = pred_delta.device, pred_delta.dtype

    if pred_delta.ndim == 3:
        # single_step
        dmin  = delta_min.to(dev, dt).view(1, 1, -1)
        dmax  = delta_max.to(dev, dt).view(1, 1, -1)
        scale = (dmax - dmin).clamp_min(1e-8)
        per_step = F.smooth_l1_loss(
            (pred_delta   - dmin) / scale * 2.0 - 1.0,
            (target_delta - dmin) / scale * 2.0 - 1.0,
            reduction="none",
        ).mean(dim=-1)  # (B, T)
    else:
        # chunk
        dmin  = delta_min.to(dev, dt).view(1, 1, 1, -1)
        dmax  = delta_max.to(dev, dt).view(1, 1, 1, -1)
        scale = (dmax - dmin).clamp_min(1e-8)
        per_step_ka = F.smooth_l1_loss(
            (pred_delta   - dmin) / scale * 2.0 - 1.0,
            (target_delta - dmin) / scale * 2.0 - 1.0,
            reduction="none",
        ).mean(dim=-1)  # (B, T, K)
        if chunk_valid is not None:
            cv = chunk_valid.to(dev, dt)
            per_step = (per_step_ka * cv).sum(dim=-1) / cv.sum(dim=-1).clamp_min(1.0)
        else:
            per_step = per_step_ka.mean(dim=-1)

    m = mask.float()
    delta_loss = ((per_step * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)).mean()

    pos_w = torch.as_tensor(end_pos_weight, device=dev, dtype=dt)
    end_per = F.binary_cross_entropy_with_logits(
        pred_end_logits, target_end.to(dev, dt), reduction="none", pos_weight=pos_w
    )
    if end_per.ndim == 3:
        end_mask = m.unsqueeze(-1)
    else:
        end_mask = m
    end_loss = (end_per * end_mask).sum() / end_mask.sum().clamp_min(1.0)

    total = delta_loss_weight * delta_loss + end_loss_weight * end_loss
    return total, delta_loss, end_loss


# ── End-signal metrics ────────────────────────────────────────────────────────

@torch.no_grad()
def end_signal_metrics(
    pred_end_logits: torch.Tensor,
    target_end: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    valid = mask.bool()
    if pred_end_logits.ndim == 3:
        valid = valid.unsqueeze(-1).expand_as(pred_end_logits)
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

    if cfg.end_pos_weight <= 0:
        total_steps = sum(len(x) for x in decoder_targets)
        if cfg.decoder_output_mode == "chunk":
            positives = sum(min(cfg.chunk_size, len(x)) for x in decoder_targets)
            total_slots = total_steps * cfg.chunk_size
            negatives = total_slots - positives
        else:
            positives = len(decoder_targets)
            negatives = total_steps - positives
        cfg.end_pos_weight = max(1.0, float(negatives / max(1, positives)))

    print(
        f"[SplineFSQAE] {len(segments)} segments | "
        f"n_control={cfg.n_control} fsq={cfg.fsq_levels} "
        f"hidden={cfg.hidden_dim} mode={cfg.decoder_output_mode}"
        + (f" K={cfg.chunk_size}" if cfg.decoder_output_mode == "chunk" else "")
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
            cfg.decoder_output_mode,
            cfg.chunk_size,
        )

    train_loader = DataLoader(mk_ds(train_idx), batch_size=cfg.batch_size, shuffle=True,  collate_fn=collate_fsq_batch)
    val_loader   = DataLoader(mk_ds(val_idx),   batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fsq_batch)

    model = SplineFSQAE(
        action