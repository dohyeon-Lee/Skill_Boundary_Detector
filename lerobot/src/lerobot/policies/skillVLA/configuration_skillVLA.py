from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("skill_vla")
@dataclass
class SkillVLAConfig(PI05Config):
    """SkillVLA extends PI05 with a skill predictor and VAE decoder prior."""

    model_type: str = "skill_vla"

    # Skill latent
    skill_latent_dim: int = 64

    # Skill predictor MLP
    skill_predictor_hidden_dim: int = 512
    skill_predictor_loss_weight: float = 1.0

    # Path to pretrained VAE decoder (.pt file). None = no prior (ablation).
    vae_decoder_path: str | None = None

    # If True, action expert cannot attend to language tokens (only image tokens).
    # If False, behavior is identical to PI05 (lang + obs both flow to action expert).
    block_lang_to_action: bool = True

    # Training stage:
    #   1 — train VLM + action expert only (skill predictor frozen, sp_loss skipped)
    #   2 — train skill predictor only (VLM + action expert frozen, flow matching skipped)
    #   3 — joint: teacher forcing + path3 (sp_loss gradient flows into VLM, all params trainable)
    training_stage: int = 1
