from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig


@PreTrainedConfig.register_subclass("skill_vla_pretrain")
@dataclass
class SkillVLAPretrainConfig(PI0FastConfig):
    """Autoregressive PaliGemma pretraining on compositional FSQ skill tokens + FAST actions."""

    model_type: str = "skill_vla_pretrain"
    dtype: str = "bfloat16"
    # Motor actions are pre-tokenized full skill trajectories; these dummy values only satisfy the
    # generic LeRobot policy feature contract and do not define a FAST horizon.
    chunk_size: int = 1
    n_action_steps: int = 1
    max_action_tokens: int = 384

    training_mode: str = "lora"  # full | lora
    skill_fsq_levels: list[int] = field(default_factory=lambda: [3, 3, 3])
    skill_unused_start: int = 0
    fast_vocab_size: int = 1024
    transition_packs: str | None = None
    pretrain_target_packs: str | None = None
    transition_randomization: bool = True

    skill_loss_weight: float = 1.0
    fast_loss_weight: float = 1.0
    structure_loss_weight: float = 0.1

    pretrain_lora_rank: int = 16
    pretrain_lora_alpha: float = 32.0
    pretrain_lora_dropout: float = 0.0
    pretrain_lora_targets: str = "q,k,v,o"

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    def __post_init__(self):
        super().__post_init__()
        self.training_mode = str(self.training_mode).strip().lower()
        if self.training_mode not in {"full", "lora"}:
            raise ValueError(f"training_mode must be full|lora, got {self.training_mode!r}.")
        if not self.skill_fsq_levels or any(int(level) < 2 for level in self.skill_fsq_levels):
            raise ValueError(f"skill_fsq_levels must contain levels >= 2, got {self.skill_fsq_levels}.")
        if self.skill_unused_start < 0:
            raise ValueError("skill_unused_start must be >= 0.")
        if self.fast_vocab_size <= 0:
            raise ValueError("fast_vocab_size must be positive.")
        weights = (self.skill_loss_weight, self.fast_loss_weight, self.structure_loss_weight)
        if any(float(weight) < 0.0 for weight in weights):
            raise ValueError(f"Pretraining loss weights must be non-negative, got {weights}.")
        if self.skill_loss_weight == 0.0 or self.fast_loss_weight == 0.0:
            raise ValueError("skill_loss_weight and fast_loss_weight must both be > 0.")
        if self.training_mode == "lora" and self.pretrain_lora_rank <= 0:
            raise ValueError("LoRA mode requires pretrain_lora_rank > 0.")
