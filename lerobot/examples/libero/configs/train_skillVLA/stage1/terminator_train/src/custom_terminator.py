"""A genuinely different terminator architecture (vs the FSQ terminator's pool-to-one-token).

Raw images (3rd / wrist, toggled) → FROZEN DINO (size s/b/l, selectable) → ALL patch tokens (optionally
8×8-pooled like the FSQ build, or full grid) → a small DiT/AdaRMS transformer over the full token sequence,
conditioned on the skill code (+ optional state) via AdaRMS → learned-query pool → progress + termination.

Toggles (the model is built from them): use_third, use_wrist, use_state (skill is always the conditioning).
Shared by the trainer and (Phase 2) the eval — both feed RAW images, so train/eval encode identically.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_DINO_PATHS = {"s": "models/dinov3-vits16", "b": "models/dinov3-vitb16", "l": "models/dinov3-vitl16"}


class _RMSNorm(nn.Module):
    """RMSNorm WITHOUT a learned scale — the scale/shift come from AdaRMS (modulate)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def _modulate(x, scale, shift):
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FrozenDinoTokenizer(nn.Module):
    """Raw RGB image → frozen DINO → CLS + (optionally 8×8-pooled) patch tokens. Mirrors FSQ's
    _raw_images_to_tokens (same preprocessing/pooling) so train == eval == FSQ image features."""

    def __init__(self, dino_path: str, image_size: int = 224, patch_pool_8x8: bool = True):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel  # noqa: PLC0415
        self.dino = AutoModel.from_pretrained(dino_path)
        for p in self.dino.parameters():
            p.requires_grad_(False)
        self.dino.eval()
        self.feat_dim = int(self.dino.config.hidden_size)
        patch = int(getattr(self.dino.config, "patch_size", 16))
        self.image_size = int(image_size)
        native = self.image_size // patch                          # native patch grid (e.g. 224/16 = 14)
        self.patch_grid = 8 if patch_pool_8x8 else native          # 8×8 pool (FSQ-style) or full grid
        self.n_patch_raw = native ** 2                             # e.g. 14² = 196
        try:
            proc = AutoImageProcessor.from_pretrained(dino_path)
            mean = getattr(proc, "image_mean", [0.485, 0.456, 0.406])
            std = getattr(proc, "image_std", [0.229, 0.224, 0.225])
        except Exception:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.register_buffer("_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    @property
    def n_tokens(self) -> int:
        return 1 + self.patch_grid * self.patch_grid

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W) or (B,H,W,C), any range → (B, 1+grid², feat_dim)."""
        x = images
        if x.shape[-1] in (1, 3) and x.ndim == 4:                  # channels-last → CHW
            x = x.permute(0, 3, 1, 2)
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        x = x.float()
        if x.numel() and float(x.detach().amin()) < -0.05:        # [-1,1] → [0,1]
            x = (x + 1.0) / 2.0
        if x.numel() and float(x.detach().amax()) > 2.0:          # [0,255] → [0,1]
            x = x / 255.0
        x = x.clamp(0.0, 1.0)
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = (x - self._mean.to(x)) / self._std.to(x)
        dev = x.device
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=(dev.type == "cuda")):
            out = self.dino(pixel_values=x.to(next(self.dino.parameters()).dtype))
        lh = out.last_hidden_state.float()
        cls = lh[:, :1, :]
        patch = lh[:, -self.n_patch_raw:, :]                       # drop CLS + register tokens
        s = int(math.sqrt(self.n_patch_raw))
        sp = patch.reshape(patch.shape[0], s, s, self.feat_dim).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(sp, (self.patch_grid, self.patch_grid))   # 8×8 pool (or native grid)
        pooled = pooled.permute(0, 2, 3, 1).reshape(patch.shape[0], self.patch_grid ** 2, self.feat_dim)
        return torch.cat([cls, pooled], dim=1)                     # (B, 1+grid², feat_dim)


class AdaRMSBlock(nn.Module):
    """DiT-style block: AdaRMS-modulated self-attention + MLP, conditioned on `cond` (skill[+state])."""

    def __init__(self, dim: int, heads: int, cond_dim: int):
        super().__init__()
        self.n1 = _RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = _RMSNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.ada = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)                              # adaLN-Zero: blocks start as identity

    def forward(self, x, cond):
        s1, b1, g1, s2, b2, g2 = self.ada(cond).chunk(6, dim=-1)
        h = _modulate(self.n1(x), s1, b1)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + g1.unsqueeze(1) * a
        x = x + g2.unsqueeze(1) * self.mlp(_modulate(self.n2(x), s2, b2))
        return x


class CustomTerminator(nn.Module):
    def __init__(self, *, vocab: int, state_dim: int, dino_size: str = "s", image_size: int = 224,
                 patch_pool_8x8: bool = True, use_third: bool = True, use_wrist: bool = True,
                 use_state: bool = True, dim: int = 256, layers: int = 3, heads: int = 4,
                 project_root: str = ""):
        super().__init__()
        if not (use_third or use_wrist):
            raise ValueError("CustomTerminator needs at least one image (use_third or use_wrist).")
        self.use_third, self.use_wrist, self.use_state = use_third, use_wrist, use_state
        rel = _DINO_PATHS.get(dino_size.lower())
        if rel is None:
            raise ValueError(f"dino_size must be s|b|l, got {dino_size!r}")
        dino_path = f"{project_root.rstrip('/')}/{rel}" if project_root else rel
        self.tok = FrozenDinoTokenizer(dino_path, image_size=image_size, patch_pool_8x8=patch_pool_8x8)
        self.img_proj = nn.Linear(self.tok.feat_dim, dim)
        self.cam_emb = nn.Parameter(torch.zeros(2, dim))          # [3rd, wrist] camera-type embedding
        self.skill_emb = nn.Embedding(vocab, dim)
        self.state_proj = nn.Linear(state_dim, dim) if use_state else None
        self.blocks = nn.ModuleList([AdaRMSBlock(dim, heads, dim) for _ in range(layers)])
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pool = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.out_norm = _RMSNorm(dim)
        self.progress_head = nn.Linear(dim, 1)
        self.termination_head = nn.Linear(dim, 1)

    def forward(self, skill_code, state, img_third=None, img_wrist=None):
        toks = []
        if self.use_third:
            toks.append(self.img_proj(self.tok(img_third)) + self.cam_emb[0])
        if self.use_wrist:
            toks.append(self.img_proj(self.tok(img_wrist)) + self.cam_emb[1])
        x = torch.cat(toks, dim=1)                                # (B, N, dim)
        cond = self.skill_emb(skill_code.long().view(-1))         # (B, dim)
        if self.state_proj is not None:
            cond = cond + self.state_proj(state.float())
        for blk in self.blocks:
            x = blk(x, cond)
        q = self.query.expand(x.shape[0], -1, -1)
        pooled, _ = self.pool(q, x, x, need_weights=False)        # learned-query attention pool
        pooled = self.out_norm(pooled[:, 0])
        progress = torch.sigmoid(self.progress_head(pooled)).squeeze(-1)
        term_logits = self.termination_head(pooled).squeeze(-1)
        return progress, term_logits
