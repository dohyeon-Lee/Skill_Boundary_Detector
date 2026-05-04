import torch
import torch.nn as nn


class SkillPredictor(nn.Module):
    """Transformer decoder that predicts the next skill token (codebook index).

    z_prev, skill_progress, and skill_start_state are projected into separate
    condition tokens. Learned skill query tokens attend over those condition
    tokens and contextual Paligemma prefix hidden states.

    Input  : z_prev (B, skill_latent_dim), prefix hidden states, prefix pad mask,
             skill_progress (B,), skill_start_state (B, state_dim)
    Output : logits (B, num_embeddings) — cross-entropy classification target
    """

    def __init__(
        self,
        skill_latent_dim : int,
        prefix_hidden_dim: int,
        state_dim        : int,
        num_embeddings   : int,
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
        self.z_prev_token   = nn.Linear(skill_latent_dim, prefix_hidden_dim)
        self.progress_token = nn.Linear(1, prefix_hidden_dim)
        self.state_token    = nn.Linear(state_dim, prefix_hidden_dim)
        self.condition_norm = nn.LayerNorm(prefix_hidden_dim)
        self.query_tokens   = nn.Parameter(torch.zeros(1, num_query_tokens, prefix_hidden_dim))
        nn.init.normal_(self.query_tokens, std=0.02)
        self.query_conditioner = nn.Linear(skill_latent_dim + 1 + state_dim, num_query_tokens * prefix_hidden_dim)
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
            nn.Linear(hidden_dim // 2, num_embeddings),
        )

    def forward(
        self,
        z_prev           : torch.Tensor,  # (B, skill_latent_dim)
        prefix_hidden    : torch.Tensor,  # (B, prefix_len, prefix_hidden_dim)
        prefix_pad_masks : torch.Tensor,  # (B, prefix_len), True for valid prefix tokens
        skill_progress   : torch.Tensor,  # (B,) or (B, 1)
        skill_start_state: torch.Tensor,  # (B, state_dim)
    ) -> torch.Tensor:                    # (B, num_embeddings) logits
        skill_progress = skill_progress.float().view(z_prev.shape[0], 1)
        cond = torch.cat([z_prev, skill_progress, skill_start_state], dim=-1).float()
        prefix_hidden = prefix_hidden.float()
        batch_size = z_prev.shape[0]

        condition_tokens = torch.stack(
            [
                self.z_prev_token(z_prev.float()),
                self.progress_token(skill_progress),
                self.state_token(skill_start_state.float()),
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
        return self.out_mlp(pooled)
