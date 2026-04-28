from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("skill_vla_decouple")
@dataclass
class SkillVLADecoupleConfig(PI05Config):
    """SkillVLA stages 1 & 2: decoupled training before joint fine-tuning.

    Stage 1 — train VLM + action expert only (skill predictor frozen, sp_loss skipped).
    Stage 2 — train skill predictor only (VLM + action expert frozen, flow matching skipped).
    """

    model_type: str = "skill_vla_decouple"

    skill_latent_dim: int = 64
    skill_predictor_hidden_dim: int = 512
    skill_predictor_loss_weight: float = 1.0

    # Path to pretrained VAE decoder (.pt file). None = no prior (ablation).
    vae_decoder_path: str | None = None

    # If True, action expert cannot attend to language tokens (only image tokens).
    block_lang_to_action: bool = True

    # 1: VLM + action expert train freely; skill predictor frozen.
    # 2: Skill predictor trains; VLM + action expert frozen.
    training_stage: int = 1
