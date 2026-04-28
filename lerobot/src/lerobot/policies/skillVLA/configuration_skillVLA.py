from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("skill_vla")
@dataclass
class SkillVLAConfig(PI05Config):
    """SkillVLA stage 3: joint flow + skill predictor training.

    All parameters are trainable. Skill predictor gradient flows into VLM.
    Use skillVLA_decouple for stages 1 & 2 (decoupled pre-training).
    """

    model_type: str = "skill_vla"

    skill_latent_dim: int = 64
    skill_predictor_hidden_dim: int = 512
    skill_predictor_state_dim: int = 8    # observation.state dim (dataset-specific)
    skill_predictor_loss_weight: float = 1.0
    predicted_latent_sampling: bool = False
    predicted_latent_start_step: int = 0
    predicted_latent_ramp_steps: int = 20_000
    predicted_latent_max_prob: float = 1.0
    predicted_latent_detach: bool = True

    # Path to pretrained VAE decoder (.pt file). None = no prior (ablation).
    vae_decoder_path: str | None = None

    # If True, action expert cannot attend to language tokens (only image tokens).
    block_lang_to_action: bool = True
