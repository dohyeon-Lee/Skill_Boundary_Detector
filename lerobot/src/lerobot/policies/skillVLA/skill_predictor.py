import torch
import torch.nn as nn


class SkillPredictor(nn.Module):
    """MLP that predicts the next skill latent z_k at every timestep.

    Input  : cat([z_prev, prefix_pooled, skill_progress, skill_start_state])
    Output : z_k  (B, skill_latent_dim)

    skill_progress is the 0-1 normalized progress within the current skill.
    """

    def __init__(
        self,
        skill_latent_dim : int,
        prefix_hidden_dim: int,
        state_dim        : int,
        hidden_dim       : int = 512,
    ):
        super().__init__()
        input_dim = skill_latent_dim + prefix_hidden_dim + 1 + state_dim  # +1 for skill_progress
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, skill_latent_dim),
        )

    def forward(
        self,
        z_prev           : torch.Tensor,  # (B, skill_latent_dim)
        prefix_pooled    : torch.Tensor,  # (B, prefix_hidden_dim)
        skill_progress   : torch.Tensor,  # (B,) or (B, 1)  — 0-1 progress within skill
        skill_start_state: torch.Tensor,  # (B, state_dim)
    ) -> torch.Tensor:
        skill_progress = skill_progress.float().view(z_prev.shape[0], 1)
        x = torch.cat([z_prev, prefix_pooled, skill_progress, skill_start_state], dim=-1)
        return self.mlp(x)
