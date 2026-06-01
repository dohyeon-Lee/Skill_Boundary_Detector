import torch
import torch.nn as nn


class SkillPredictor(nn.Module):
    """Cross-attention decoder that predicts FSQ latent levels.

    Standard read-only cross-attention, mirroring how the action expert reads the
    VLM prefix: the previous skill token (z_prev) and normalized skill progress are
    projected into two input tokens; a vanilla TransformerDecoder lets them
    cross-attend over the (read-only) VLM prefix; the pooled output is mapped to
    dim-wise FSQ logits. No learned query tokens, no query conditioning.

    Input  : z_prev (B, skill_latent_dim), prefix embeddings (B, P, H),
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
        dropout          : float = 0.0,
    ):
        super().__init__()
        if prefix_hidden_dim % num_heads != 0:
            raise ValueError(
                f"prefix_hidden_dim ({prefix_hidden_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.fsq_dim   = fsq_dim
        self.fsq_level = fsq_level
        self.z_prev_proj   = nn.Linear(skill_latent_dim, prefix_hidden_dim)
        self.progress_proj = nn.Linear(1, prefix_hidden_dim)
        self.input_norm    = nn.LayerNorm(prefix_hidden_dim)
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
        self.head        = nn.Linear(prefix_hidden_dim, fsq_dim * fsq_level)

    def forward(
        self,
        z_prev           : torch.Tensor,  # (B, skill_latent_dim)
        prefix_hidden    : torch.Tensor,  # (B, prefix_len, prefix_hidden_dim)
        prefix_pad_masks : torch.Tensor,  # (B, prefix_len), True for valid prefix tokens
        skill_progress   : torch.Tensor,  # (B,) or (B, 1)
    ) -> torch.Tensor:                    # (B, fsq_dim, fsq_level) logits
        batch_size = z_prev.shape[0]
        skill_progress = skill_progress.float().view(batch_size, 1)

        # Two input tokens: previous skill token + progress.
        tokens = torch.stack(
            [
                self.z_prev_proj(z_prev.float()),
                self.progress_proj(skill_progress),
            ],
            dim=1,
        )                                       # (B, 2, H)
        tokens = self.input_norm(tokens)

        decoded = self.decoder(
            tgt=tokens,                                       # self-attn over [z_prev, progress]
            memory=prefix_hidden.float(),                     # read-only cross-attn over VLM prefix
            memory_key_padding_mask=~prefix_pad_masks.bool(),
        )
        pooled = self.output_norm(decoded).mean(dim=1)        # (B, H)
        return self.head(pooled).view(batch_size, self.fsq_dim, self.fsq_level)
