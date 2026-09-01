"""Configuration for the BayesVLA-style post-contact likelihood stage."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.skill_expert.configuration_skill_expert import (
    COND_GEMMA_ARCHITECTURE,
    COND_GEMMA_ARCHITECTURE_REVISION,
    SkillExpertConfig,
)


@PreTrainedConfig.register_subclass("skill_vla_stage2")
@dataclass
class SkillVLAStage2Config(SkillExpertConfig):
    """Frozen Stage-1 prior with likelihood-refinement or DSBC Stage-2 blocks.

    The Stage-1 prior checkpoint carries no predictor or terminator of its own.
    Frozen VLM memory comes directly from the canonical pi0.5 base checkpoint.
    ``skill_predictor_checkpoint_path`` is optional and contributes only the
    learned reader/head/LoRA needed when predicted skill codes are requested.
    Terminators are attached externally at evaluation time.
    """

    model_type: str = "skill_vla_stage2"
    architecture: str = COND_GEMMA_ARCHITECTURE
    architecture_revision: str = COND_GEMMA_ARCHITECTURE_REVISION
    # The frozen VLM module must be instantiated (and language tokenization
    # enabled); Stage 2 never trains it.
    train_skill_predictor: bool = True
    stage1_checkpoint_path: str | None = None
    vlm_base_path: str = "models/pi05_base"
    # likelihood preserves the original Stage-2 flow-refinement objective.
    # dsbc learns an initial-noise selector from online FRS targets while the
    # complete Stage-1 VSA, including its action head, remains frozen.
    stage2_mode: str = "likelihood"
    training_skill_source: str = "gt"
    likelihood_num_layers: int = 4
    likelihood_cross_attention_heads: int = 8
    # last: cross-attend the frozen VLM's final hidden state (original design).
    # layer_mix: each block learns a softmax mixture over all VLM layers.
    likelihood_vlm_memory: str = "last"
    # LR multiplier for the language-pathway bootstrap parameters (gate dense
    # layers, the VLM projection, and the layer mix). Counters the zero-gate
    # cold start that keeps the cross-attention pathway from opening.
    likelihood_gate_lr_scale: float = 1.0
    # DSBC predicts either one real-action noise vector shared over the chunk,
    # or one vector for every action token. Padding dimensions are never learned.
    dsbc_noise_output_mode: str = "shared"
    # FRS bounds the predicted normalized noise with bound * tanh(raw).
    dsbc_noise_output_bound: float = 5.0
    dsbc_frs_num_steps: int = 10
    dsbc_anchor_seed: int = 0
    same_skill_batch_enabled: bool = False
    same_skill_batch_fraction: float = 0.5
    same_skill_progress_temperature: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        if not str(self.stage1_checkpoint_path or "").strip():
            raise ValueError("Stage 2 requires stage1_checkpoint_path.")
        if not str(self.vlm_base_path or "").strip():
            raise ValueError(
                "Stage 2 requires vlm_base_path for its frozen VLM memory."
            )
        if self.training_skill_source == "predictor" and not str(
            self.skill_predictor_checkpoint_path or ""
        ).strip():
            raise ValueError(
                "training_skill_source='predictor' requires "
                "skill_predictor_checkpoint_path."
            )
        if self.architecture != COND_GEMMA_ARCHITECTURE:
            raise ValueError(
                "Stage 2 is implemented on the cond_gemma Stage-1 prior; got "
                f"architecture={self.architecture!r}."
            )
        self.stage2_mode = str(self.stage2_mode).strip().lower()
        if self.stage2_mode not in {"likelihood", "dsbc"}:
            raise ValueError(
                "stage2_mode must be 'likelihood' or 'dsbc', got "
                f"{self.stage2_mode!r}."
            )
        if self.action_loss_mode != "flow":
            raise ValueError(
                "Stage 2 matches Stage 1: action_loss_mode is fixed to 'flow'; "
                "configure only cumulative_xyz_loss_enabled and "
                "cumulative_xyz_loss_weight."
            )
        if self.mask_actions_after_skill_end:
            raise ValueError(
                "Stage 2 fixes mask_actions_after_skill_end=False and "
                "supervises the complete action chunk."
            )
        if self.skill_flow_enabled:
            raise ValueError(
                "Stage 2 inherits the weights shaped by the Stage-1 skill-flow "
                "auxiliary, but fixes skill_flow_enabled=False and trains only "
                "the Stage-2 objective."
            )
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
        if self.likelihood_vlm_memory not in {"last", "layer_mix"}:
            raise ValueError(
                "likelihood_vlm_memory must be 'last' or 'layer_mix', got "
                f"{self.likelihood_vlm_memory!r}."
            )
        if self.likelihood_gate_lr_scale <= 0.0:
            raise ValueError("likelihood_gate_lr_scale must be positive.")
        self.dsbc_noise_output_mode = str(
            self.dsbc_noise_output_mode
        ).strip().lower()
        if self.dsbc_noise_output_mode not in {"shared", "per_step"}:
            raise ValueError(
                "dsbc_noise_output_mode must be 'shared' or 'per_step', got "
                f"{self.dsbc_noise_output_mode!r}."
            )
        if self.dsbc_noise_output_bound <= 0.0:
            raise ValueError("dsbc_noise_output_bound must be positive.")
        if self.dsbc_frs_num_steps <= 0:
            raise ValueError("dsbc_frs_num_steps must be positive.")
        if self.dsbc_anchor_seed < 0:
            raise ValueError("dsbc_anchor_seed must be non-negative.")
        if self.stage2_mode == "dsbc" and self.cumulative_xyz_loss_enabled:
            raise ValueError(
                "cumulative_xyz_loss is defined only for likelihood mode; "
                "DSBC is supervised directly on FRS noise."
            )
        if not 0.0 <= self.same_skill_batch_fraction <= 1.0:
            raise ValueError("same_skill_batch_fraction must be in [0, 1].")
        if self.same_skill_progress_temperature <= 0.0:
            raise ValueError("same_skill_progress_temperature must be > 0.")
        if not self.train_skill_predictor:
            raise ValueError(
                "Stage 2 requires train_skill_predictor=True so the frozen VLM "
                "module is instantiated and language tokens are produced; the "
                "predictor itself is never trained in Stage 2."
            )
        if self.train_terminator:
            raise ValueError(
                "Stage 2 trains without a terminator; keep train_terminator=False "
                "and attach one at evaluation time via load_external_terminator."
            )
        if not (
            self.skill_predictor_attend_image
            and self.skill_predictor_attend_language
        ):
            raise ValueError(
                "Stage 2 cross-attention requires both image and language tokens "
                "from the frozen VLM."
            )
