from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.skillVLA.configuration_skillVLA import SkillVLAConfig


@PreTrainedConfig.register_subclass("skill_vla_stage0_pretrain")
@dataclass
class SkillVLAStage0PretrainConfig(SkillVLAConfig):
    """Direct Stage-0 whose VLM retains autoregressive skill/FAST pretraining."""

    model_type: str = "skill_vla_stage0_pretrain"
    pretrain_checkpoint_path: str | None = None
    pretrain_training_mode: str = "full"  # full | lora
    pretrain_lora_targets: str = "q,k,v,o"
    pretrain_lora_rank: int = 16
    pretrain_lora_alpha: float = 32.0
    pretrain_lora_dropout: float = 0.0

    skill_unused_start: int = 0
    fast_vocab_size: int = 1024
    fast_skip_tokens: int = 128
    max_action_tokens: int = 384
    transition_packs: str | None = None
    pretrain_target_packs: str | None = None
    transition_randomization: bool = True

    attend_skill: bool = True
    ar_fast_loss: bool = False
    ar_batch_size: int = 2
    ar_skill_loss_weight: float = 1.0
    ar_fast_loss_weight: float = 1.0
    ar_structure_loss_weight: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        self.pretrain_training_mode = str(self.pretrain_training_mode).strip().lower()
        if self.pt_stage != "stage0":
            raise ValueError("skill_vla_stage0_pretrain requires pt_stage='stage0'.")
        if self.pretrain_training_mode not in {"full", "lora"}:
            raise ValueError(
                "pretrain_training_mode must be full|lora, "
                f"got {self.pretrain_training_mode!r}."
            )
        if not str(self.pretrain_checkpoint_path or "").strip():
            raise ValueError("pretrain_checkpoint_path is required.")
        if self.pretrain_training_mode == "lora" and self.pretrain_lora_rank <= 0:
            raise ValueError("LoRA-pretrained VLM requires pretrain_lora_rank > 0.")
        if self.fast_vocab_size <= 0 or self.max_action_tokens <= 0:
            raise ValueError("fast_vocab_size and max_action_tokens must be positive.")
        if self.ar_batch_size <= 0:
            raise ValueError("ar_batch_size must be positive.")
        weights = (
            self.ar_skill_loss_weight,
            self.ar_fast_loss_weight,
            self.ar_structure_loss_weight,
        )
        if self.ar_skill_loss_weight <= 0.0:
            raise ValueError("ar_skill_loss_weight must be > 0; skill-token CE is mandatory.")
        if any(weight < 0.0 for weight in weights):
            raise ValueError(f"Autoregressive loss weights must be non-negative, got {weights}.")
