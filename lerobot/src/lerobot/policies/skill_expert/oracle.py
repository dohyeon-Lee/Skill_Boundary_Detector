"""Oracle — Stage-1 residual module (trajectory pooler → VAE latent r).

Encodes a per-skill/per-step trajectory + the GT skill code z into a small VAE latent ``r``
(n_tokens, r_dim), modulated by z via AdaLN. ``r`` is handed to the action expert as a prefix token:
it supplies the motion detail that (skill z, current obs) underdetermine — a skill is a coarse
cluster of 100+ motions, so the action expert needs more to disambiguate which one.

Two input modes (config.oracle_input_source):
  - "state": the skill's full STATE trajectory (start→end) resampled to oracle_resample_n control
    points + a LENGTH token (duration lost by resampling). Per-skill (constant within a skill, so r is
    too) — matches Stage-2 emitting r once at skill start. Appearance-invariant, geometric.
  - "action": the current GT ACTION chunk (chunk_size × action_dim), fixed length → NO resample,
    NO length token. Per-STEP (the chunk slides each frame → r changes). Carries the fine-motion detail
    z's coarse code can't hold; motion patterns recur across tasks so the Oracle still generalizes to
    new tasks (language would not — hence action, not language). Keep r_dim TIGHT (residual, not a copy).

Common: Perceiver — ``n_tokens`` learned latent queries cross-attend the trajectory KV set →
fixed-size output. Skill z conditions WHAT the queries look for via AdaLN (DiT AdaLN-Zero). VAE head
(μ, logσ) + free-bits → a small anchored latent; CFG-style dropout in the action expert forces r to be
a RESIDUAL beyond z (see modeling_skill_expert.py).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


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
        # start as identity (DiT AdaLN-Zero), so a fresh Oracle doesn't disrupt warm-started parts.
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


class Oracle(nn.Module):
    """Skill state-trajectory (+ skill z) → VAE latent r. See module docstring."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        d = int(config.oracle_width)
        self.input_source = str(config.oracle_input_source)

        # ── KV assembly: per-token input proj + learned positional emb over the L tokens. "state" mode
        # also adds a LENGTH token (skill duration — lost by resampling to a fixed N, so fed explicitly,
        # mirroring FSQ's "control points + length"); "action" mode is fixed length so it has neither
        # resample nor length token. Param names for "state" are kept (state_proj/length_*) so existing
        # state-mode checkpoints load unchanged. ──
        if self.input_source == "action":
            n = int(config.chunk_size)                                  # GT action chunk: fixed length
            self.action_proj = nn.Linear(config.max_action_dim, d)
        else:
            n = int(config.oracle_resample_n)                           # resampled state-traj control points
            self.state_proj = nn.Linear(config.max_state_dim, d)
            self.length_proj = nn.Linear(1, d)
            self.length_type = nn.Parameter(torch.zeros(1, 1, d))       # length-token type emb
            nn.init.trunc_normal_(self.length_type, std=0.02)
        self.pos_emb = nn.Parameter(torch.zeros(1, n, d))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # ── Skill z → AdaLN condition (z_q grid coord, D dims → oracle width) ──
        self.skill_cond = nn.Linear(len(config.skill_fsq_levels), d)

        # ── Perceiver latents + blocks ──
        self.latents = nn.Parameter(torch.zeros(1, int(config.oracle_n_tokens), d))
        nn.init.trunc_normal_(self.latents, std=0.02)
        self.blocks = nn.ModuleList(
            _PerceiverBlock(d, int(config.oracle_n_heads), cond_dim=d)
            for _ in range(int(config.oracle_depth))
        )
        self.norm_out = nn.LayerNorm(d)

        # ── VAE head ──
        r_dim = int(config.oracle_r_dim)
        self.head_mu = nn.Linear(d, r_dim)
        self.head_logsig = nn.Linear(d, r_dim)
        self.free_bits = float(config.oracle_free_bits)

    @property
    def out_dim(self) -> int:
        return int(self.config.oracle_r_dim)

    @property
    def n_tokens(self) -> int:
        return int(self.config.oracle_n_tokens)

    def forward(
        self, traj: Tensor, length: Tensor | None, skill_zq: Tensor, sample: bool
    ) -> tuple[Tensor, Tensor]:
        """traj — the Oracle input trajectory: "state" mode = (B, N, max_state_dim) resampled state-traj;
        "action" mode = (B, chunk_size, max_action_dim) GT action chunk.
        length (B,) — skill duration in frames ("state" mode only; lost by resampling); None for "action".
        skill_zq (B, D) normalized FSQ grid coord in [-1, 1]. sample → VAE reparam (train) vs μ (eval).

        Returns r (B, n_tokens, r_dim) and the (free-bits) KL scalar.
        """
        proj = self.action_proj if self.input_source == "action" else self.state_proj
        dtype = proj.weight.dtype
        bsize = traj.shape[0]

        kv = proj(traj.to(dtype)) + self.pos_emb.to(dtype)                               # (B, L, d)
        if self.input_source != "action":                                               # state: + length token
            len_feat = torch.log1p(length.to(dtype).clamp(min=0)).view(bsize, 1, 1)      # scale-robust
            len_tok = self.length_proj(len_feat) + self.length_type.to(dtype)            # (B, 1, d)
            kv = torch.cat([kv, len_tok], dim=1)                                         # (B, N+1, d)
        cond = self.skill_cond(skill_zq.to(dtype))                                       # (B, d)
        q = self.latents.to(dtype).expand(bsize, -1, -1)
        for blk in self.blocks:
            q = blk(q, kv, cond)
        q = self.norm_out(q)

        mu = self.head_mu(q)                                                     # (B, n_tokens, r_dim)
        logsig = self.head_logsig(q).clamp(-10.0, 5.0)
        r = mu + torch.exp(logsig) * torch.randn_like(mu) if sample else mu

        # Free-bits KL: per-dim max(λ, KL_i) → no pressure below λ nats (prevents posterior collapse),
        # summed over r_dim, averaged over batch & tokens.
        kl_per = 0.5 * (mu.pow(2) + torch.exp(2.0 * logsig) - 2.0 * logsig - 1.0)  # (B, n_tokens, r_dim)
        kl = torch.clamp(kl_per, min=self.free_bits).sum(dim=-1).mean()
        return r, kl
