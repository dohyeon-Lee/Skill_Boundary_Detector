"""Configuration for auxiliary-only skill predictor / terminator training."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
    LRSchedulerConfig,
    WarmupConstantSchedulerConfig,
)
from lerobot.utils.constants import ACTION, OBS_STATE


@PreTrainedConfig.register_subclass("skill_aux")
@dataclass
class SkillAuxConfig(PreTrainedConfig):
    """Train the Stage-1 skill auxiliaries without constructing an action model."""

    model_type: str = "skill_aux"
    dtype: str = "bfloat16"

    max_state_dim: int = 32
    max_action_dim: int = 32
    skill_vocab_size: int = 27
    skill_fsq_levels: list[int] = field(default_factory=lambda: [3, 3, 3])

    train_terminator: bool = True
    fsq_path: str | None = "FSQ.pt"
    terminator_freeze_vision_encoder: bool = True
    terminator_lr_scale: float = 1.0
    terminator_end_target_sigma: float = 2.0
    terminator_end_pos_weight: float = 1.0
    # Drop the progress head from both the model output and the objective, so the
    # terminator is trained purely as a boundary detector. The progress query and
    # head stay in the module for checkpoint-shape compatibility; the attention
    # mask already isolates them from the termination query, so they are inert.
    terminator_termination_only: bool = False

    train_image_only_terminator: bool = False
    image_only_terminator_freeze_vision_encoder: bool = True
    image_only_terminator_lr_scale: float = 1.0
    image_only_terminator_end_target_sigma: float = 2.0
    image_only_terminator_end_pos_weight: float = 1.0
    image_only_terminator_termination_only: bool = False

    train_wrist_only_terminator: bool = False
    wrist_only_terminator_freeze_vision_encoder: bool = True
    wrist_only_terminator_lr_scale: float = 1.0
    wrist_only_terminator_end_target_sigma: float = 2.0
    wrist_only_terminator_end_pos_weight: float = 1.0
    wrist_only_terminator_termination_only: bool = False

    train_state_only_terminator: bool = False
    state_only_terminator_hidden_dim: int = 64
    state_only_terminator_num_layers: int = 2
    state_only_terminator_lr_scale: float = 1.0
    state_only_terminator_end_target_sigma: float = 2.0
    state_only_terminator_end_pos_weight: float = 1.0
    state_only_terminator_balance_positive_negative: bool = True
    state_only_terminator_termination_only: bool = True

    train_state_rnn_terminator: bool = False
    state_rnn_terminator_sequence_length: int = 16
    state_rnn_terminator_full_skill_sequence: bool = True
    state_rnn_terminator_input_dim: int = 64
    state_rnn_terminator_hidden_dim: int = 64
    state_rnn_terminator_num_layers: int = 1
    state_rnn_terminator_dropout: float = 0.0
    state_rnn_terminator_lr_scale: float = 1.0
    state_rnn_terminator_end_target_sigma: float = 2.0
    state_rnn_terminator_end_pos_weight: float = 1.0
    state_rnn_terminator_balance_positive_negative: bool = True
    state_rnn_terminator_termination_only: bool = True

    train_skill_predictor: bool = False
    skill_predictor_checkpoint_path: str | None = None
    skill_predictor_lr_scale: float = 1.0
    skill_predictor_all_layers: bool = True
    skill_predictor_detach_vlm: bool = False
    skill_predictor_lora: bool = True
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
    skill_predictor_deadzone_frac: float = 0.8
    skill_predictor_attend_image: bool = True
    skill_predictor_attend_language: bool = True
    tokenizer_path: str | None = None
    tokenizer_max_length: int = 200

    gradient_checkpointing: bool = False
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1_000
    scheduler_mode: str = "warmup_constant"
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (
            self.train_terminator
            or self.train_image_only_terminator
            or self.train_wrist_only_terminator
            or self.train_state_only_terminator
            or self.train_state_rnn_terminator
            or self.train_skill_predictor
        ):
            raise ValueError(
                "Auxiliary-only training needs terminator.train, "
                "image_only_terminator.train, wrist_only_terminator.train, "
                "state_only_terminator.train, state_rnn_terminator.train, "
                "and/or skill_predictor.train to be true."
            )
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError(f"dtype must be float32 or bfloat16, got {self.dtype!r}.")
        if not self.skill_fsq_levels or any(level <= 1 for level in self.skill_fsq_levels):
            raise ValueError("skill_fsq_levels must contain integers greater than one.")
        expected_vocab = math.prod(self.skill_fsq_levels)
        if self.skill_vocab_size != expected_vocab:
            raise ValueError(
                f"skill_vocab_size={self.skill_vocab_size} does not match "
                f"prod(skill_fsq_levels)={expected_vocab}."
            )
        if min(self.max_state_dim, self.max_action_dim) <= 0:
            raise ValueError("State and action dimensions must be positive.")
        if (
            self.train_terminator
            or self.train_image_only_terminator
            or self.train_wrist_only_terminator
        ):
            if not str(self.fsq_path or "").strip():
                raise ValueError("Terminator training requires fsq_path.")
        if self.train_terminator:
            if self.terminator_lr_scale <= 0.0:
                raise ValueError("terminator_lr_scale must be positive.")
            if self.terminator_end_target_sigma < 0.0:
                raise ValueError("terminator_end_target_sigma must be non-negative.")
            if self.terminator_end_pos_weight <= 0.0:
                raise ValueError("terminator_end_pos_weight must be positive.")
        if self.train_image_only_terminator:
            if self.image_only_terminator_lr_scale <= 0.0:
                raise ValueError("image_only_terminator_lr_scale must be positive.")
            if self.image_only_terminator_end_target_sigma < 0.0:
                raise ValueError(
                    "image_only_terminator_end_target_sigma must be non-negative."
                )
            if self.image_only_terminator_end_pos_weight <= 0.0:
                raise ValueError(
                    "image_only_terminator_end_pos_weight must be positive."
                )
        if self.train_wrist_only_terminator:
            if self.wrist_only_terminator_lr_scale <= 0.0:
                raise ValueError("wrist_only_terminator_lr_scale must be positive.")
            if self.wrist_only_terminator_end_target_sigma < 0.0:
                raise ValueError(
                    "wrist_only_terminator_end_target_sigma must be non-negative."
                )
            if self.wrist_only_terminator_end_pos_weight <= 0.0:
                raise ValueError(
                    "wrist_only_terminator_end_pos_weight must be positive."
                )
        if self.train_state_only_terminator:
            if min(
                self.state_only_terminator_hidden_dim,
                self.state_only_terminator_num_layers,
            ) <= 0:
                raise ValueError("State-only terminator dimensions must be positive.")
            if self.state_only_terminator_lr_scale <= 0.0:
                raise ValueError("state_only_terminator_lr_scale must be positive.")
            if self.state_only_terminator_end_target_sigma < 0.0:
                raise ValueError(
                    "state_only_terminator_end_target_sigma must be non-negative."
                )
            if self.state_only_terminator_end_pos_weight <= 0.0:
                raise ValueError(
                    "state_only_terminator_end_pos_weight must be positive."
                )
        if self.train_state_rnn_terminator:
            if min(
                self.state_rnn_terminator_sequence_length,
                self.state_rnn_terminator_input_dim,
                self.state_rnn_terminator_hidden_dim,
                self.state_rnn_terminator_num_layers,
            ) <= 0:
                raise ValueError("State-RNN terminator dimensions must be positive.")
            if not 0.0 <= self.state_rnn_terminator_dropout < 1.0:
                raise ValueError("state_rnn_terminator_dropout must be in [0, 1).")
            if self.state_rnn_terminator_lr_scale <= 0.0:
                raise ValueError("state_rnn_terminator_lr_scale must be positive.")
            if self.state_rnn_terminator_end_target_sigma < 0.0:
                raise ValueError(
                    "state_rnn_terminator_end_target_sigma must be non-negative."
                )
            if self.state_rnn_terminator_end_pos_weight <= 0.0:
                raise ValueError(
                    "state_rnn_terminator_end_pos_weight must be positive."
                )
        if self.train_skill_predictor:
            if self.skill_predictor_vlm_variant != "gemma_2b":
                raise ValueError("The auxiliary predictor VLM must use gemma_2b.")
            if self.skill_predictor_lr_scale <= 0.0:
                raise ValueError("skill_predictor_lr_scale must be positive.")
            if self.skill_predictor_lora and self.skill_predictor_detach_vlm:
                raise ValueError(
                    "skill_predictor_detach_vlm must be false when skill_predictor_lora=true."
                )
            if not self.skill_predictor_lora and not self.skill_predictor_detach_vlm:
                raise ValueError(
                    "skill_predictor_detach_vlm=false requires skill_predictor_lora=true; "
                    "full predictor-VLM fine-tuning is intentionally unsupported."
                )
            if self.skill_predictor_lora:
                if not self.skill_predictor_lora_targets.strip():
                    raise ValueError("skill_predictor_lora_targets cannot be empty.")
                if self.skill_predictor_lora_rank <= 0 or self.skill_predictor_lora_alpha <= 0.0:
                    raise ValueError("Skill predictor LoRA rank and alpha must be positive.")
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
        if self.scheduler_mode not in {"cosine_decay", "warmup_constant"}:
            raise ValueError("scheduler_mode must be cosine_decay or warmup_constant.")
        if self.scheduler_warmup_steps < 0 or self.scheduler_decay_steps <= 0:
            raise ValueError("Invalid auxiliary scheduler step counts.")

    @property
    def uses_skill_predictor(self) -> bool:
        return self.train_skill_predictor

    @property
    def state_only_auxiliary(self) -> bool:
        """Whether training can skip decoding every camera stream."""
        has_state_model = (
            self.train_state_only_terminator or self.train_state_rnn_terminator
        )
        has_visual_model = (
            self.train_terminator
            or self.train_image_only_terminator
            or self.train_wrist_only_terminator
            or self.train_skill_predictor
        )
        return has_state_model and not has_visual_model

    @property
    def state_full_skill_supervision(self) -> bool:
        """Whether the dataloader can sample endpoint-anchored full skills."""
        return (
            self.train_state_rnn_terminator
            and self.state_rnn_terminator_full_skill_sequence
            and self.state_only_auxiliary
        )

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

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig:
        if self.scheduler_mode == "warmup_constant":
            return WarmupConstantSchedulerConfig(num_warmup_steps=self.scheduler_warmup_steps)
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int] | None:
        if not self.train_state_rnn_terminator:
            return None
        return list(range(1 - self.state_rnn_terminator_sequence_length, 1))

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None
