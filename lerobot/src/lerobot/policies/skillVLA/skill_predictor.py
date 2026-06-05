import torch
import torch.nn as nn


class SkillPredictor(nn.Module):
    """Reader-query cross-attention head that predicts FSQ latent levels.

    The previous skill token (z_prev) and the normalized skill progress are projected
    into two condition tokens; a set of learned reader-query tokens is appended. All
    tokens self-attend (so the readers absorb the condition), then cross-attend
    (read-only) over the VLM prefix. ONLY the reader outputs are pooled — the condition
    tokens are excluded so their strong residual cannot dilute what the readers extract
    from the prefix. The pooled vector is mapped to dim-wise FSQ logits by an MLP head.

    Input  : z_prev (B, skill_latent_dim), prefix hidden states (B, P, H),
             prefix pad mask (B, P), skill_progress (B,)
    Output : logits (B, fsq_dim, fsq_level) — dim-wise cross-entropy target
    """

    def __init__(
        self,
        skill_latent_dim : int,
        prefix_hidden_dim: int,
        fsq_dim          : int,
        fsq_level        : int,
        hidden_dim       : int = 512,
        num_heads        : int = 8,
        num_layers       : int = 2,
        num_reader_tokens: int = 4,
        dropout          : float = 0.0,
    ):
        super().__init__()
        if prefix_hidden_dim % num_heads != 0:
            raise ValueError(
                f"prefix_hidden_dim ({prefix_hidden_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.fsq_dim           = fsq_dim
        self.fsq_level         = fsq_level
        self.num_reader_tokens = num_reader_tokens

        # Condition tokens carry the situation (z_prev, progress); reader queries are
        # empty learned slots whose only job is to read the prefix.
        self.z_prev_proj   = nn.Linear(skill_latent_dim, prefix_hidden_dim)
        self.progress_proj = nn.Linear(1, prefix_hidden_dim)
        self.input_norm    = nn.LayerNorm(prefix_hidden_dim)
        self.reader_query  = nn.Parameter(torch.zeros(1, num_reader_tokens, prefix_hidden_dim))
        nn.init.normal_(self.reader_query, std=0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=prefix_hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder     = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(prefix_hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(prefix_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, fsq_dim * fsq_level),
        )

    def forward(
        self,
        z_prev           : torch.Tensor,  # (B, skill_latent_dim)
        prefix_hidden    : torch.Tensor,  # (B, prefix_len, prefix_hidden_dim)
        prefix_pad_masks : torch.Tensor,  # (B, prefix_len), True for valid prefix tokens
        skill_progress   : torch.Tensor,  # (B,) or (B, 1)
    ) -> torch.Tensor:                    # (B, fsq_dim, fsq_level) logits
        batch_size = z_prev.shape[0]
        skill_progress = skill_progress.float().view(batch_size, 1)

        cond = torch.stack(
            [self.z_prev_proj(z_prev.float()), self.progress_proj(skill_progress)],
            dim=1,
        )                                                       # (B, 2, H)
        cond = self.input_norm(cond)
        readers = self.reader_query.expand(batch_size, -1, -1)  # (B, N, H)
        tgt = torch.cat([cond, readers], dim=1)                 # (B, 2 + N, H)

        decoded = self.decoder(
            tgt=tgt,                                            # self-attn over [cond, readers]
            memory=prefix_hidden.float(),                       # read-only cross-attn over VLM prefix
            memory_key_padding_mask=~prefix_pad_masks.bool(),
        )
        readers_out = decoded[:, -self.num_reader_tokens:]      # readers only (exclude condition)
        pooled = self.output_norm(readers_out).mean(dim=1)      # (B, H)
        return self.head(pooled).view(batch_size, self.fsq_dim, self.fsq_level)
