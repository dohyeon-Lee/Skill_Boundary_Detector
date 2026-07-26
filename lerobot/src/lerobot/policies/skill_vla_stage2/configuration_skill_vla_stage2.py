"""Configuration for the BayesVLA-style post-contact likelihood stage."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.skill_expert.configuration_skill_expert import SkillExpertConfig


@PreTrainedConfig.register_subclass("skill_vla_stage2")
@dataclass
class SkillVLAStage2Config(SkillExpertConfig):
    """Frozen Stage-1 VSA prior plus four language-conditioned action blocks.

    Predictor reader/head and terminator continuation training are optional and
    disabled by default; neither can send gradients into the frozen VSA prior.
    """

    model_type: str = "skill_vla_stage2"
    stage1_checkpoint_path: str | None = None
    training_skill_source: str = "gt"
    likelihood_num_layers: int = 4
    likelihood_cross_attention_heads: int = 8
    finetune_skill_predictor: bool = False
    finetune_terminator: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if not str(self.stage1_checkpoint_path or "").strip():
            raise ValueError("Stage 2 requires stage1_checkpoint_path.")
        if self.training_skill_source not in {"gt", "predictor"}:
            raise ValueError(
                "training_skill_source must be 'gt' or 'predictor', got "
                f"{self.training_skill_source!r}."
            )
        if self.likelihood_num_layers != 4:
            raise ValueError(
                "BayesVLA-matched Stage 2 fixes likelihood_num_layers=4, got "
                f"{self.likelihood_num_layers}."
            )
        if self.likelihood_cross_attention_heads != 8:
            raise ValueError(
                "The gemma_300m likelihood blocks use 8 cross-attention heads."
            )
        if not self.train_skill_predictor:
            raise ValueError(
                "Stage 2 reuses the frozen Stage-1 VLM/predictor and requires "
                "train_skill_predictor=True in the inherited architecture."
            )
        if not self.train_terminator:
            raise ValueError(
                "Stage 2 checkpoints retain the frozen Stage-1 terminator; "
                "train_terminator=True is required in the inherited architecture."
            )
        if not (
            self.skill_predictor_attend_image
            and self.skill_predictor_attend_language
        ):
            raise ValueError(
                "Stage 2 cross-attention requires both image and language tokens "
                "from the frozen VLM."
            )
