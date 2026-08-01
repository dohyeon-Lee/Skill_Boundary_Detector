"""Configuration for the Stage-1 vision-state-action prior."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE


@PreTrainedConfig.register_subclass("skill_expert")
@dataclass
class SkillExpertConfig(PreTrainedConfig):
    """Stage-1 VSA policy: DINO images plus routed state/skill -> action flow.

    The condition transformer and action expert both use all 18 ``gemma_300m``
    layers. The action expert is initialized from the pi0.5 base checkpoint and
    fully trained. The condition transformer and all VSA projections are fresh.
    """

    model_type: str = "skill_expert"
    dtype: str = "float32"

    action_expert_variant: str = "gemma_300m"
    cond_encoder_variant: str = "gemma_300m"
    chunk_size: int = 10
    n_action_steps: int = 10
    max_state_dim: int = 32
    max_action_dim: int = 32

    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0
    action_loss_mode: str = "flow"

    vision_backbone: str = "dino"
    dino_model_path: str = "models/dinov3-vitl16"
    dino_image_size: int = 224
    freeze_vision_encoder: bool = False
    dino_lr: float | None = None

    conditioning_route: str = "state_cond"
    """Where state and skill condition the two Stage-1 streams:

    - ``state_cond``: state modulates the condition encoder through AdaRMS; skill
      is broadcast into the action expert.
    - ``state_skill_cond``: state still modulates the condition encoder through
      AdaRMS, and skill is broadcast into the condition encoder as well.
    - ``stateonly_cond``: state modulates the condition encoder through AdaRMS;
      skill is omitted from both the condition and action streams.
    - ``skill_cond``: state is absent from the action path; skill alone is
      broadcast into the condition encoder, which uses ordinary RMSNorm.

    In every route the action expert's AdaRMS input is flow time only.
    """
    skill_vocab_size: int = 27
    skill_fsq_levels: list[int] = field(default_factory=lambda: [3, 3, 3])
    transition_jitter_pmax: int = 0
    transition_jitter_distribution: str = "half_normal"

    # Which skill code conditions the action path during offline training.  The
    # predictor route loads only the learned predictor from a previous Stage-1
    # checkpoint; its frozen pi0.5 VLM base is initialized by pretrained_path.
    training_skill_source: str = "gt"
    skill_predictor_checkpoint_path: str | None = None

    # Skill predictor. Old Stage-1 checkpoints use a fully detached VLM; the
    # Stage3-A-matched path keeps the pi0.5 base frozen and trains only a named
    # Q/K/V/O LoRA together with SkillReader/SkillHead.
    train_skill_predictor: bool = False
    skill_predictor_weight: float = 0.5
    skill_predictor_lr_scale: float = 1.0
    skill_predictor_all_layers: bool = False
    skill_predictor_detach_vlm: bool = True
    skill_predictor_lora: bool = False
    skill_predictor_lora_targets: str = "q,k,v,o"
    skill_predictor_lora_rank: int = 8
    skill_predictor_lora_alpha: float = 16.0
    skill_predictor_lora_dropout: float = 0.0
    skill_predictor_lora_lr_scale: float = 10.0
    skill_predictor_vlm_variant: str = "gemma_2b"
    skill_predictor_image_size: int = 224
    skill_predictor_reader_tokens: int = 4
    skill_predictor_reader_depth: int = 2
    skill_predictor_reader_heads: int = 8
    skill_predictor_deadzone_frac: float = 0.0
    skill_predictor_attend_image: bool = True
    skill_predictor_attend_language: bool = True
    tokenizer_path: str | None = None
    tokenizer_max_length: int = 200

    # Parameter-disjoint FSQ terminator co-training on the same Stage-1 batch.
    train_terminator: bool = False
    fsq_path: str | None = None
    terminator_freeze_vision_encoder: bool | None = None
    terminator_dino_model_path: str | None = None
    terminator_lr_scale: float = 1.0
    terminator_end_target_sigma: float = 2.0
    terminator_end_pos_weight: float = 1.0

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )
    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"

    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError(f"dtype must be float32 or bfloat16, got {self.dtype!r}.")
        if self.action_expert_variant != "gemma_300m":
            raise ValueError(
                "Stage 1 fixes both 18-layer streams to gemma_300m; got "
                f"action_expert_variant={self.action_expert_variant!r}."
            )
        if self.cond_encoder_variant != self.action_expert_variant:
            raise ValueError(
                "Stage 1 requires matching 18-layer cond/expert variants; got "
                f"{self.cond_encoder_variant!r} and {self.action_expert_variant!r}."
            )
        if self.vision_backbone != "dino":
            raise ValueError("Stage 1 uses the Stage-0 DINO vision path; vision_backbone must be 'dino'.")
        if self.dino_image_size <= 0:
            raise ValueError("dino_image_size must be positive.")
        if self.dino_lr is not None and self.dino_lr <= 0.0:
            raise ValueError("dino_lr must be positive when set.")
        if self.freeze_vision_encoder and self.dino_lr is not None:
            raise ValueError("dino_lr cannot be set when freeze_vision_encoder=True.")
        if self.conditioning_route not in {
            "state_cond",
            "state_skill_cond",
            "stateonly_cond",
            "skill_cond",
        }:
            raise ValueError(
                "conditioning_route must be 'state_cond', 'state_skill_cond', "
                "'stateonly_cond', or 'skill_cond', got "
                f"{self.conditioning_route!r}."
            )
        if not self.skill_fsq_levels or any(level <= 1 for level in self.skill_fsq_levels):
            raise ValueError(f"skill_fsq_levels must all be greater than one, got {self.skill_fsq_levels}.")
        expected_vocab = math.prod(self.skill_fsq_levels)
        if self.skill_vocab_size != expected_vocab:
            raise ValueError(
                f"skill_vocab_size={self.skill_vocab_size} does not match "
                f"prod(skill_fsq_levels)={expected_vocab}."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps cannot exceed chunk_size.")
        if min(self.max_state_dim, self.max_action_dim, self.num_inference_steps) <= 0:
            raise ValueError("State/action dimensions and num_inference_steps must be positive.")
        if self.action_loss_mode not in {"flow", "flow_endpoint_xyz"}:
            raise ValueError(
                "action_loss_mode must be 'flow' or 'flow_endpoint_xyz', got "
                f"{self.action_loss_mode!r}."
            )
        if self.transition_jitter_pmax < 0:
            raise ValueError("transition_jitter_pmax must be non-negative.")
        if self.transition_jitter_distribution not in {"half_normal", "uniform"}:
            raise ValueError(
                "transition_jitter_distribution must be 'half_normal' or 'uniform', got "
                f"{self.transition_jitter_distribution!r}."
            )
        if self.training_skill_source not in {"gt", "predictor"}:
            raise ValueError(
                "training_skill_source must be 'gt' or 'predictor', got "
                f"{self.training_skill_source!r}."
            )
        predictor_source = self.skill_predictor_checkpoint_path or getattr(
            self, "stage1_checkpoint_path", None
        )
        if self.training_skill_source == "predictor" and not str(
            predictor_source or ""
        ).strip():
            raise ValueError(
                "training_skill_source='predictor' requires "
                "skill_predictor_checkpoint_path."
            )
        if self.uses_skill_predictor:
            if self.skill_predictor_lora and self.skill_predictor_detach_vlm:
                raise ValueError(
                    "skill_predictor_detach_vlm must be False when skill_predictor_lora=True "
                    "so predictor gradients can reach the skill adapter."
                )
            if not self.skill_predictor_lora and not self.skill_predictor_detach_vlm:
                raise ValueError(
                    "skill_predictor_detach_vlm=False requires skill_predictor_lora=True; "
                    "full VLM fine-tuning is not part of the Stage-1 contract."
                )
            if self.skill_predictor_vlm_variant != "gemma_2b":
                raise ValueError("The pi0.5 base predictor VLM must use gemma_2b.")
            if self.skill_predictor_weight <= 0.0:
                raise ValueError("skill_predictor_weight must be positive.")
            if self.skill_predictor_lr_scale <= 0.0:
                raise ValueError("skill_predictor_lr_scale must be positive.")
            if self.skill_predictor_lora:
                if not str(self.skill_predictor_lora_targets).strip():
                    raise ValueError("skill_predictor_lora_targets cannot be empty.")
                if self.skill_predictor_lora_rank <= 0:
                    raise ValueError("skill_predictor_lora_rank must be positive.")
                if self.skill_predictor_lora_alpha <= 0.0:
                    raise ValueError("skill_predictor_lora_alpha must be positive.")
                if self.skill_predictor_lora_dropout < 0.0:
                    raise ValueError("skill_predictor_lora_dropout must be non-negative.")
                if self.skill_predictor_lora_lr_scale <= 0.0:
                    raise ValueError("skill_predictor_lora_lr_scale must be positive.")
            if min(
                self.skill_predictor_image_size,
                self.skill_predictor_reader_tokens,
                self.skill_predictor_reader_depth,
                self.skill_predictor_reader_heads,
                self.tokenizer_max_length,
            ) <= 0:
                raise ValueError("Skill predictor image, reader, and tokenizer sizes must be positive.")
            if self.skill_predictor_deadzone_frac < 0.0:
                raise ValueError("skill_predictor_deadzone_frac must be non-negative.")
            if not (self.skill_predictor_attend_image or self.skill_predictor_attend_language):
                raise ValueError("Skill predictor must attend image and/or language tokens.")
        if self.train_terminator:
            if not str(self.fsq_path or "").strip():
                raise ValueError("train_terminator=True requires fsq_path.")
            if self.terminator_lr_scale <= 0.0:
                raise ValueError("terminator_lr_scale must be positive.")
            if self.terminator_end_target_sigma < 0.0:
                raise ValueError("terminator_end_target_sigma must be non-negative.")
            if self.terminator_end_pos_weight <= 0.0:
                raise ValueError("terminator_end_pos_weight must be positive.")

    def validate_features(self) -> None:
        if self.input_features is None:
            self.input_features = {}
        if self.output_features is None:
            self.output_features = {}
        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE, shape=(self.max_state_dim,)
            )
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION, shape=(self.max_action_dim,)
            )

    @property
    def uses_skill_predictor(self) -> bool:
        """Whether this policy must instantiate/tokenize the predictor path."""
        return self.train_skill_predictor or self.training_skill_source == "predictor"

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
