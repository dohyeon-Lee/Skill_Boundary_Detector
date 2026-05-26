import torch
import torch.nn as nn


class SkillPredictor(nn.Module):
    """Transformer decoder that predicts FSQ latent levels.

    Previous skill token and normalized skill progress are projected into
    condition tokens. Learned skill query tokens attend over those condition
    tokens and raw embed_prefix token embeddings.

    Input  : z_prev (B, skill_latent_dim), prefix embeddings, prefix pad mask,
             skill_progress (B,)
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
        num_query_tokens : int = 4,
        dropout          : float = 0.0,
    ):
        super().__init__()
        if prefix_hidden_dim % num_heads != 0:
            raise ValueError(
                f"prefix_hidden_dim ({prefix_hidden_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.num_query_tokens = num_query_tokens
        self.fsq_dim = fsq_dim
        self.fsq_level = fsq_level
        self.z_prev_token   = nn.Linear(skill_latent_dim, prefix_hidden_dim)
        self.progress_token = nn.Linear(1, prefix_hidden_dim)
        self.condition_norm = nn.LayerNorm(prefix_hidden_dim)
        self.query_tokens   = nn.Parameter(torch.zeros(1, num_query_tokens, prefix_hidden_dim))
        nn.init.normal_(self.query_tokens, std=0.02)
        self.query_conditioner = nn.Linear(skill_latent_dim + 1, num_query_tokens * prefix_hidden_dim)
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
        self.out_mlp = nn.Sequential(
            nn.Linear(prefix_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, fsq_dim * fsq_level),
        )

    def forward(
        self,
        z_prev           : torch.Tensor,  # (B, skill_latent_dim)
        prefix_hidden    : torch.Tensor,  # (B, prefix_len, prefix_hidden_dim)
        prefix_pad_masks : torch.Tensor,  # (B, prefix_len), True for valid prefix tokens
        skill_progress   : torch.Tensor,  # (B,) or (B, 1)
    ) -> torch.Tensor:                    # (B, fsq_dim, fsq_level) logits
        skill_progress = skill_progress.float().view(z_prev.shape[0], 1)
        cond = torch.cat([z_prev, skill_progress], dim=-1).float()
        prefix_hidden = prefix_hidden.float()
        batch_size = z_prev.shape[0]

        condition_tokens = torch.stack(
            [
                self.z_prev_token(z_prev.float()),
                self.progress_token(skill_progress),
            ],
            dim=1,
        )
        condition_tokens = self.condition_norm(condition_tokens)

        cond_query = self.query_conditioner(cond).view(batch_size, self.num_query_tokens, -1)
        query = self.query_tokens.expand(batch_size, -1, -1) + cond_query
        tgt = torch.cat([condition_tokens, query], dim=1)
        key_padding_mask = ~prefix_pad_masks.bool()
        decoded = self.decoder(
            tgt=tgt,
            memory=prefix_hidden,
            memory_key_padding_mask=key_padding_mask,
        )
        query_output = decoded[:, -self.num_query_tokens:]
        pooled = self.output_norm(query_output).mean(dim=1)
        return self.out_mlp(pooled).view(batch_size, self.fsq_dim, self.fsq_level)
