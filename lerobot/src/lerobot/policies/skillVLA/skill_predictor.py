import torch
import torch.nn as nn


class SkillPredictor(nn.Module):
    """MLP that predicts the next skill latent z_k from [z_{k-1}, pooled VLM prefix].

    Input  : cat([z_prev, prefix_pooled])  shape (B, skill_latent_dim + prefix_hidden_dim)
    Output : z_k                           shape (B, skill_latent_dim)

    Updated only at skill boundary frames (f_b == 1) during training.
    """

    def __init__(self, skill_latent_dim: int, prefix_hidden_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(skill_latent_dim + prefix_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, skill_latent_dim),
        )

    def forward(self, z_prev: torch.Tensor, prefix_pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_prev:        (B, skill_latent_dim)  previous skill latent
            prefix_pooled: (B, prefix_hidden_dim) mean-pooled VLM prefix embeddings
        Returns:
            z_pred: (B, skill_latent_dim)
        """
        x = torch.cat([z_prev, prefix_pooled], dim=-1)
        return self.mlp(x)
