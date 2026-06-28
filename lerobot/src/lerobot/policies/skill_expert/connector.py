"""Connector — Stage-1 future-conditioning module (Perceiver pooler → VAE latent).

Encodes the skill's END frame (3rd + wrist images) + END state into a small VAE latent ``z``,
modulated by the GT skill code (z_q) via AdaLN. ``z`` is handed to the action expert as
prefix token(s): it supplies the WITHIN-skill motion detail that (skill, current obs)
underdetermine — a skill is a coarse cluster of 100+ motions, so the action expert needs the
future to disambiguate which one.

Design notes (decided in the Stage-1 redesign):
  - Its DINO encoder is its OWN instance and ALWAYS frozen (independent of the expert-vision
    DINO). Frozen self-supervised features give OOD-robust goal representations; fine-tuning
    them would erode exactly the generalization we want.
  - Perceiver: ``L`` learned latent queries cross-attend the (two-view DINO patches + state)
    KV set → fixed-size output regardless of patch count. Input size ≠ model size.
  - Skill conditions WHAT the queries look for via AdaLN (not a KV token — avoids the
    1-vs-many washout the skill suffered as a lone token among image patches).
  - VAE head (μ, logσ) + free-bits → a region-robust, anchored latent (helps Stage-2 infer z
    from the VLM and few-shot adaptation; see the design discussion).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModel


def _modulate(x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    """DiT/AdaLN: x * (1 + scale) + shift, with (scale, shift) broadcast over the token axis."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _PerceiverBlock(nn.Module):
    """cross-attn (latents ← KV) + self-attn (latents) + FFN, all AdaLN-modulated by the skill cond."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, cond_dim: int | None = None):
        super().__init__()
        cond_dim = cond_dim or dim
        self.norm_q = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_s = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_f = nn.LayerNorm(dim, elementwise_affine=False)
        hidden = int(dim * mlp_ratio)
        self.ff = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        # AdaLN: skill cond → per-block (scale, shift) for the 3 modulated norms. Zero-init → blocks
        # start as identity (DiT AdaLN-Zero), so a fresh connector doesn't disrupt warm-started parts.
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, q: Tensor, kv: Tensor, cond: Tensor) -> Tensor:
        s_c, h_c, s_s, h_s, s_f, h_f = self.ada(cond).chunk(6, dim=-1)
        kvn = self.norm_kv(kv)
        q = q + self.cross(_modulate(self.norm_q(q), s_c, h_c), kvn, kvn)[0]
        qn = _modulate(self.norm_s(q), s_s, h_s)
        q = q + self.self_attn(qn, qn, qn)[0]
        q = q + self.ff(_modulate(self.norm_f(q), s_f, h_f))
        return q


class Connector(nn.Module):
    """End-frame (+ state, skill) → VAE latent z. See module docstring."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        d = int(config.connector_width)

        # ── Own frozen DINO (full patches, NO 8x8 pooling — same pipeline as expert-vision) ──
        self.dino = AutoModel.from_pretrained(config.connector_dino_model_path)
        for p in self.dino.parameters():
            p.requires_grad_(False)
        self.dino.eval()
        vis_dim = int(self.dino.config.hidden_size)
        self.n_register = int(getattr(self.dino.config, "num_register_tokens", 0))
        patch = int(getattr(self.dino.config, "patch_size", 16))
        self.image_size = int(config.connector_dino_image_size)
        grid = self.image_size // patch
        self.n_patch = grid * grid
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet
        self.register_buffer("_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

        # ── KV assembly: patch proj + 2D pos-emb + per-view id + state token (own type-emb) ──
        self.feat_proj = nn.Linear(vis_dim, d)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_patch, d))     # learned 2D positional emb
        self.view_emb = nn.Parameter(torch.zeros(2, d))                  # [3rd, wrist] view-id
        self.state_proj = nn.Linear(config.max_state_dim, d)
        self.state_type = nn.Parameter(torch.zeros(1, 1, d))             # state token type-emb
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.view_emb, std=0.02)
        nn.init.trunc_normal_(self.state_type, std=0.02)

        # ── Skill → AdaLN condition (z_q grid coord, D dims → connector width) ──
        self.skill_cond = nn.Linear(len(config.skill_fsq_levels), d)

        # ── Perceiver latents + blocks ──
        L = int(config.connector_n_latents)
        self.latents = nn.Parameter(torch.zeros(1, L, d))
        nn.init.trunc_normal_(self.latents, std=0.02)
        self.blocks = nn.ModuleList(
            _PerceiverBlock(d, int(config.connector_n_heads), cond_dim=d)
            for _ in range(int(config.connector_depth))
        )
        self.norm_out = nn.LayerNorm(d)

        # ── VAE head ──
        zdim = int(config.connector_z_dim)
        self.head_mu = nn.Linear(d, zdim)
        self.head_logsig = nn.Linear(d, zdim)
        self.free_bits = float(config.connector_free_bits)

    @property
    def out_dim(self) -> int:
        return int(self.config.connector_z_dim)

    @property
    def n_latents(self) -> int:
        return int(self.config.connector_n_latents)

    def train(self, mode: bool = True):
        """Keep the frozen DINO in eval mode even when the policy is put in train() (no dropout drift)."""
        super().train(mode)
        self.dino.eval()
        return self

    @torch.no_grad()
    def _dino_patches(self, image: Tensor) -> Tensor:
        """image (B, C, H, W) in [0, 1] → (B, n_patch, vis_dim) full patches (CLS + registers dropped)."""
        x = image.to(torch.float32)
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = (x - self._mean) / self._std
        x = x.to(dtype=next(self.dino.parameters()).dtype)
        out = self.dino(x).last_hidden_state
        return out[:, 1 + self.n_register :, :]

    def forward(
        self, end_images: list[Tensor], end_state: Tensor, skill_zq: Tensor, sample: bool
    ) -> tuple[Tensor, Tensor]:
        """end_images: [3rd, wrist] each (B, C, H, W) in [0,1]; end_state (B, state_dim);
        skill_zq (B, D) normalized FSQ grid coord in [-1,1]. sample → VAE reparam (train) vs μ (eval).

        Returns z (B, L, z_dim) and the (free-bits) KL scalar.
        """
        dtype = self.feat_proj.weight.dtype
        bsize = end_state.shape[0]

        kv = []
        for vi, img in enumerate(end_images):
            f = self.feat_proj(self._dino_patches(img).to(dtype))                 # (B, n_patch, d)
            f = f + self.pos_emb.to(dtype) + self.view_emb[vi].view(1, 1, -1).to(dtype)
            kv.append(f)
        st = self.state_proj(end_state.to(dtype)).unsqueeze(1) + self.state_type.to(dtype)  # (B, 1, d)
        kv.append(st)
        kv = torch.cat(kv, dim=1)                                                 # (B, 2*n_patch + 1, d)

        cond = self.skill_cond(skill_zq.to(dtype))                               # (B, d)
        q = self.latents.to(dtype).expand(bsize, -1, -1)
        for blk in self.blocks:
            q = blk(q, kv, cond)
        q = self.norm_out(q)

        mu = self.head_mu(q)                                                      # (B, L, z_dim)
        logsig = self.head_logsig(q).clamp(-10.0, 5.0)
        z = mu + torch.exp(logsig) * torch.randn_like(mu) if sample else mu

        # Free-bits KL: per-dim max(λ, KL_i) → no pressure below λ nats (prevents posterior collapse),
        # summed over z_dim, averaged over batch & latents.
        kl_per = 0.5 * (mu.pow(2) + torch.exp(2.0 * logsig) - 2.0 * logsig - 1.0)  # (B, L, z_dim)
        kl = torch.clamp(kl_per, min=self.free_bits).sum(dim=-1).mean()
        return z, kl
