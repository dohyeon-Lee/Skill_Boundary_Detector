import torch
import torch.nn as nn


class _JointReaderLayer(nn.Module):
    """One JOINT concat-KV layer: the probe queries attend [probes' own K/V ; VLM K/V] in ONE softmax
    (own and VLM keys compete in the same normalization — the cond/action "joint self-attention" pattern,
    NOT a two-softmax self-then-cross decoder). VLM is read-only (its tokens are only keys/values, never
    queries), so the VLM stays unperturbed. Pre-norm; residual add."""

    def __init__(self, width: int, heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(width)
        self.norm_kv = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * mlp_ratio), nn.GELU(), nn.Linear(width * mlp_ratio, width),
        )

    def forward(self, probes: torch.Tensor, vlm_out: torch.Tensor, vlm_key_ignore: torch.Tensor) -> torch.Tensor:
        # probes (B, N, W); vlm_out (B, P, W); vlm_key_ignore (B, P) True = MASK OUT that VLM key.
        q = self.norm_q(probes)
        kv = torch.cat([q, self.norm_kv(vlm_out)], dim=1)                 # JOINT: probe K/V ++ VLM K/V
        b, n = q.shape[:2]
        probe_keep = torch.zeros(b, n, dtype=torch.bool, device=probes.device)  # probes always attendable
        key_padding = torch.cat([probe_keep, vlm_key_ignore], dim=1)      # (B, N+P) True = ignore
        attn, _ = self.attn(q, kv, kv, key_padding_mask=key_padding, need_weights=False)
        probes = probes + attn
        probes = probes + self.mlp(self.norm_mlp(probes))
        return probes


class SkillReader(nn.Module):
    """Standalone JOINT concat-KV skill reader (post-VLM, FINAL-layer read).

    N learned probe tokens read the VLM's FINAL-layer output via joint concat-KV attention (probes'
    own K/V compete with the VLM K/V in one softmax — so drift enters gracefully: as the VLM keys become
    unfamiliar the probes can fall back on their own stable self-value). The VLM is READ-ONLY (never
    attends the probes → stays pristine, its pretrained computation intact). ``vlm_key_ignore`` gates
    which VLM tokens are attendable (images always, language iff attend_language) — supplied by the caller.
    The pooled probe hidden feeds the SkillHead. No RoPE (attention-pooling; probe positions are free).

    forward: vlm_out (B, P, W), vlm_key_ignore (B, P) → skill hidden (B, W)."""

    def __init__(self, width: int, depth: int = 2, heads: int = 8, num_probes: int = 4,
                 mlp_ratio: int = 4, dropout: float = 0.0, pool: str = "mean"):
        super().__init__()
        if width % heads != 0:
            raise ValueError(f"SkillReader width ({width}) must be divisible by heads ({heads}).")
        self.pool = pool
        self.probes = nn.Parameter(torch.randn(1, num_probes, width) * 0.02)
        self.layers = nn.ModuleList(
            [_JointReaderLayer(width, heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(width)

    def forward(
        self,
        vlm_out: torch.Tensor,
        vlm_key_ignore: torch.Tensor,
        probe_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        probes = self.probes.expand(vlm_out.shape[0], -1, -1).to(vlm_out.dtype)   # (B, N, W)
        if probe_condition is not None:
            expected = (vlm_out.shape[0], vlm_out.shape[-1])
            if tuple(probe_condition.shape) != expected:
                raise ValueError(
                    "probe_condition must have shape "
                    f"{expected}, got {tuple(probe_condition.shape)}."
                )
            # A conditioning vector shifts every learned probe while leaving
            # the read-only VLM memory pristine.  Existing skill-predictor
            # callers omit this argument and remain bit-for-bit unchanged.
            probes = probes + probe_condition.to(probes.dtype).unsqueeze(1)
        for layer in self.layers:
            probes = layer(probes, vlm_out, vlm_key_ignore)
        probes = self.norm_out(probes)
        return probes[:, 0] if self.pool == "first" else probes.mean(dim=1)       # (B, W)
