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
    # Stable semantic identity of the FSQ model that generated the dataset's
    # integer labels. Equal level counts alone do not imply equal code meaning.
    skill_code_space_id: str = ""
    # Training lineage is stored in the policy checkpoint so FT naming and
    # batch size are inherited from PT rather than duplicated in the FT YAML.
    training_batch_size: int = 0
    dataset_source_lineage: list[str] = field(default_factory=list)
    run_suffix_lineage: list[str] = field(default_factory=list)

    train_terminator: bool = True
    fsq_path: str | None = "FSQ.pt"
    terminator_checkpoint_path: str | None = None
    # Legacy joint warm-start path, kept only so old checkpoints still load.
    auxiliary_checkpoint_path: str | None = None
    terminator_architecture_label: str = ""
    terminator_context: str = "prev_action"
    terminator_cameras: str = "both"
    terminator_arch: str = "fusion"
    terminator_vision_backbone: str = "resnet"
    terminator_freeze_vision_encoder: bool = True
    terminator_lr_scale: float = 1.0
    terminator_end_target_sigma: float = 2.0
    terminator_end_pos_weight: float = 1.0
    # Drop the progress head from both the model output and the objective, so the
    # terminator is trained purely as a boundary detector. The progress query and
    # head stay in the module for checkpoint-shape compatibility; the attention
    # mask already isolates them from the termination query, so they are inert.
    terminator_termination_only: bool = False
    # Train the progress head, but stop its loss at the shared representation.
    terminator_progress_detach_backbone: bool = False
    terminator_goal_xyz: bool = False
    terminator_goal_noise_max_m: float = 0.0
    terminator_skill_skip: bool = True
    terminator_proprio_history: bool = False
    terminator_history_length: int = 20
    terminator_history_dim: int = 128
    terminator_history_layers: int = 2
    terminator_history_heads: int = 4
    # Term14: a second proprio token latched at the active skill's start.
    terminator_start_proprio: bool = False
    # tokens | adarms ([current,start,delta], no proprio tokens) |
    # tokens_delta_adarms (current/start tokens + delta AdaRMS)
    terminator_proprio_conditioning: str = "tokens"
    # Training-only noise in normalized proprio space. LIBERO's final two
    # state axes are left/right finger positions and stay noise-free.
    terminator_proprio_noise_magnitude: float = 0.0
    terminator_proprio_noise_distribution: str = "uniform"
    terminator_proprio_noise_exclude_last_n: int = 2
    terminator_proprio_noise_clamp: bool = True
    # Training-only temporal augmentation. The stored dataset stays unchanged;
    # the loader shifts current vision+proprio and the start anchor coherently.
    terminator_start_randomization: bool = False
    terminator_start_randomization_early_frames: int = 0
    terminator_start_randomization_late_frames: int = 0
    terminator_start_randomization_distribution: str = "half_normal"
    terminator_start_randomization_shift_current_observation: bool = False
    terminator_agent_patch_align_weight: float = 0.0
    terminator_wrist_patch_align_weight: float = 0.0
    terminator_patch_align_target_sigma: float = 0.7
    # LIT-style auxiliary: the final goal token (Term12) or a learned token
    # (Term13) predicts normalized grounded chunk-end state [xyz, aa, gripper].
    terminator_chunk_end_pose_mode: str = "off"  # off | goal_token | learned_token
    terminator_chunk_end_state_loss_weight: float = 0.3
    terminator_chunk_end_state_dim: int = 8
    terminator_chunk_end_state_horizon: int = 10
    terminator_chunk_end_state_q01: list[float] = field(default_factory=list)
    terminator_chunk_end_state_q99: list[float] = field(default_factory=list)

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
    # Component-specific FT source. It may differ from the terminator source.
    skill_predictor_checkpoint_path: str | None = None
    skill_predictor_lr_scale: float = 1.0
    skill_predictor_freeze_vlm: bool = True
    skill_predictor_vlm_lr_scale: float = 1.0
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
    skill_predictor_focus_uv_enabled: bool = False
    skill_predictor_focus_uv_loss_weight: float = 0.25
    skill_predictor_end_state_mode: str = "off"  # off | xyz | full_state
    skill_predictor_end_state_dim: int = 8
    skill_predictor_end_state_loss_weight: float = 1.0
    # Which skill code conditions the XYZ/full-state branch.  Scheduled mode
    # linearly replaces GT codes with the predictor's hard skill decisions.
    skill_predictor_end_state_skill_source: str = "gt"  # gt | predicted | scheduled
    skill_predictor_end_state_schedule_start_step: int = 0
    skill_predictor_end_state_schedule_end_step: int = 100_000
    skill_predictor_end_state_schedule_max_probability: float = 1.0
    # mode1: one jittered transition start per occurrence.
    # mode2: every dataset row's true current frame once per epoch.
    skill_predictor_sampling_mode: str = "mode1"
    # Retained for old checkpoint/config compatibility; mode2 no longer uses
    # boundary/interior resampling.
    skill_predictor_boundary_fraction: float = 0.7
    skill_predictor_boundary_window: int = 10
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
        # draccus parses CLI values as YAML, where a bare `off` becomes the boolean False ("False").
        if str(self.skill_predictor_end_state_mode).strip().lower() in {"off", "false"}:
            self.skill_predictor_end_state_mode = "off"
        self.terminator_chunk_end_pose_mode = str(
            self.terminator_chunk_end_pose_mode
        ).strip().lower()
        if self.terminator_chunk_end_pose_mode == "false":
            self.terminator_chunk_end_pose_mode = "off"
        terminator_enabled = any(
            (
                self.train_terminator,
                self.train_image_only_terminator,
                self.train_wrist_only_terminator,
                self.train_state_only_terminator,
                self.train_state_rnn_terminator,
            )
        )
        if not (terminator_enabled or self.train_skill_predictor):
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
        if self.training_batch_size < 0:
            raise ValueError("training_batch_size must be non-negative.")
        if any(not str(source).strip() for source in self.dataset_source_lineage):
            raise ValueError("dataset_source_lineage cannot contain empty values.")
        if any(not str(suffix).strip() for suffix in self.run_suffix_lineage):
            raise ValueError("run_suffix_lineage cannot contain empty values.")
        if (
            self.train_terminator
            or self.train_image_only_terminator
            or self.train_wrist_only_terminator
        ):
            if not str(self.fsq_path or "").strip():
                raise ValueError("Terminator training requires fsq_path.")
        if self.train_terminator:
            if self.terminator_context not in {"prev_action", "proprio", "none"}:
                raise ValueError(
                    "terminator_context must be prev_action, proprio, or none."
                )
            if self.terminator_cameras not in {"both", "top", "wrist"}:
                raise ValueError(
                    "terminator_cameras must be both, top, or wrist."
                )
            if self.terminator_arch not in {"small", "fusion"}:
                raise ValueError("terminator_arch must be small or fusion.")
            if self.terminator_vision_backbone not in {
                "dino",
                "siglip",
                "resnet",
            }:
                raise ValueError(
                    "terminator_vision_backbone must be dino, siglip, or resnet."
                )
            if self.terminator_lr_scale <= 0.0:
                raise ValueError("terminator_lr_scale must be positive.")
            if self.terminator_end_target_sigma < 0.0:
                raise ValueError("terminator_end_target_sigma must be non-negative.")
            if self.terminator_end_pos_weight <= 0.0:
                raise ValueError("terminator_end_pos_weight must be positive.")
            if (
                self.terminator_progress_detach_backbone
                and self.terminator_termination_only
            ):
                raise ValueError(
                    "terminator_progress_detach_backbone requires progress training."
                )
            if self.terminator_goal_xyz and (
                self.terminator_arch != "fusion"
                or self.terminator_context != "proprio"
            ):
                raise ValueError(
                    "terminator_goal_xyz requires fusion architecture and proprio context."
                )
            if self.terminator_proprio_history:
                if (
                    self.terminator_arch != "fusion"
                    or self.terminator_context != "proprio"
                ):
                    raise ValueError(
                        "terminator_proprio_history requires fusion architecture "
                        "and proprio context."
                    )
                if min(
                    self.terminator_history_length,
                    self.terminator_history_dim,
                    self.terminator_history_layers,
                    self.terminator_history_heads,
                ) <= 0:
                    raise ValueError("Terminator history dimensions must be positive.")
                if self.terminator_history_dim % self.terminator_history_heads:
                    raise ValueError(
                        "terminator_history_dim must be divisible by "
                        "terminator_history_heads."
                    )
            if self.terminator_start_proprio and (
                self.terminator_arch != "fusion"
                or self.terminator_context != "proprio"
            ):
                raise ValueError(
                    "terminator_start_proprio requires fusion architecture and proprio context."
                )
            if self.terminator_start_proprio and self.terminator_proprio_history:
                raise ValueError(
                    "terminator_start_proprio and proprio history are separate ablations."
                )
            if self.terminator_proprio_conditioning not in {
                "tokens",
                "adarms",
                "tokens_delta_adarms",
            }:
                raise ValueError(
                    "terminator_proprio_conditioning must be "
                    "tokens|adarms|tokens_delta_adarms."
                )
            if (
                self.terminator_proprio_conditioning != "tokens"
                and not self.terminator_start_proprio
            ):
                raise ValueError(
                    "Proprio AdaRMS conditioning requires "
                    "terminator_start_proprio=true."
                )
            if self.terminator_proprio_noise_magnitude < 0.0:
                raise ValueError(
                    "terminator_proprio_noise_magnitude must be non-negative."
                )
            if self.terminator_proprio_noise_distribution != "uniform":
                raise ValueError(
                    "terminator_proprio_noise_distribution must be uniform."
                )
            if self.terminator_proprio_noise_exclude_last_n < 0 or (
                self.terminator_proprio_noise_magnitude > 0.0
                and self.terminator_proprio_noise_exclude_last_n
                > self.max_state_dim
            ):
                raise ValueError(
                    "terminator_proprio_noise_exclude_last_n must be between "
                    "0 and max_state_dim."
                )
            if (
                self.terminator_proprio_noise_magnitude > 0.0
                and not self.terminator_start_proprio
            ):
                raise ValueError(
                    "Proprio value noise requires terminator_start_proprio=true."
                )
            if self.terminator_start_randomization and not self.terminator_start_proprio:
                raise ValueError(
                    "terminator_start_randomization requires terminator_start_proprio=true."
                )
            if min(
                self.terminator_start_randomization_early_frames,
                self.terminator_start_randomization_late_frames,
            ) < 0:
                raise ValueError("Terminator start-randomization ranges must be non-negative.")
            if self.terminator_start_randomization_distribution not in {
                "half_normal",
                "uniform",
            }:
                raise ValueError(
                    "terminator_start_randomization_distribution must be half_normal or uniform."
                )
            if self.terminator_start_randomization and not (
                self.terminator_start_randomization_shift_current_observation
            ):
                raise ValueError(
                    "Start randomization must shift current vision/proprio together."
                )
            numeric = {
                "terminator_goal_noise_max_m": self.terminator_goal_noise_max_m,
                "terminator_agent_patch_align_weight": self.terminator_agent_patch_align_weight,
                "terminator_wrist_patch_align_weight": self.terminator_wrist_patch_align_weight,
                "terminator_patch_align_target_sigma": self.terminator_patch_align_target_sigma,
            }
            if any(not math.isfinite(value) or value < 0.0 for value in numeric.values()):
                raise ValueError(
                    "Terminator goal noise, alignment weights, and target sigma "
                    "must be finite and non-negative."
                )
            if self.terminator_goal_noise_max_m > 0.0 and not self.terminator_goal_xyz:
                raise ValueError("Goal noise requires terminator_goal_xyz=true.")
            if (
                self.terminator_agent_patch_align_weight > 0.0
                or self.terminator_wrist_patch_align_weight > 0.0
            ) and not self.terminator_goal_xyz:
                raise ValueError("Patch alignment requires terminator_goal_xyz=true.")
            if (
                self.terminator_agent_patch_align_weight > 0.0
                and self.terminator_cameras not in {"both", "top"}
            ):
                raise ValueError("Agent alignment requires the top camera.")
            if (
                self.terminator_wrist_patch_align_weight > 0.0
                and self.terminator_cameras not in {"both", "wrist"}
            ):
                raise ValueError("Wrist alignment requires the wrist camera.")
            chunk_mode = self.terminator_chunk_end_pose_mode
            if chunk_mode not in {"off", "goal_token", "learned_token"}:
                raise ValueError(
                    "terminator_chunk_end_pose_mode must be off, goal_token, or learned_token."
                )
            if chunk_mode != "off":
                if self.terminator_arch != "fusion" or self.terminator_context != "proprio":
                    raise ValueError(
                        "Chunk-end pose prediction requires fusion architecture and proprio context."
                    )
                if chunk_mode == "goal_token" and not self.terminator_goal_xyz:
                    raise ValueError("goal_token chunk-end prediction requires goal XYZ input.")
                if chunk_mode == "learned_token" and self.terminator_goal_xyz:
                    raise ValueError("learned_token chunk-end prediction requires goal_xyz=false.")
                if (
                    self.terminator_agent_patch_align_weight > 0.0
                    or self.terminator_wrist_patch_align_weight > 0.0
                ):
                    raise ValueError(
                        "Chunk-end pose prediction replaces patch alignment; alignment weights must be zero."
                    )
                if self.terminator_chunk_end_state_dim != 8:
                    raise ValueError(
                        "Terminator chunk-end target is grounded state and must have dimension 8."
                    )
                if self.terminator_chunk_end_state_horizon <= 0:
                    raise ValueError("Terminator chunk-end horizon must be positive.")
                if (
                    not math.isfinite(self.terminator_chunk_end_state_loss_weight)
                    or self.terminator_chunk_end_state_loss_weight <= 0.0
                ):
                    raise ValueError("Terminator chunk-end loss weight must be finite and positive.")
                q01 = self.terminator_chunk_end_state_q01
                q99 = self.terminator_chunk_end_state_q99
                if len(q01) != 8 or len(q99) != 8:
                    raise ValueError("Terminator chunk-end q01/q99 must each contain 8 values.")
                if any(
                    not math.isfinite(float(low))
                    or not math.isfinite(float(high))
                    or float(high) <= float(low)
                    for low, high in zip(q01, q99, strict=True)
                ):
                    raise ValueError("Terminator chunk-end q99 must be finite and exceed q01.")
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
            if self.skill_predictor_sampling_mode not in {"mode1", "mode2"}:
                raise ValueError("skill_predictor_sampling_mode must be mode1 or mode2.")
            if not 0.0 <= self.skill_predictor_boundary_fraction <= 1.0:
                raise ValueError("skill_predictor_boundary_fraction must be in [0, 1].")
            if self.skill_predictor_boundary_window < 1:
                raise ValueError("skill_predictor_boundary_window must be positive.")
            if self.skill_predictor_vlm_variant != "gemma_2b":
                raise ValueError("The auxiliary predictor VLM must use gemma_2b.")
            if self.skill_predictor_lr_scale <= 0.0:
                raise ValueError("skill_predictor_lr_scale must be positive.")
            if not self.skill_predictor_freeze_vlm and self.skill_predictor_lora:
                raise ValueError(
                    "skill_predictor_lora must be false when the complete predictor VLM "
                    "is co-trained."
                )
            if self.skill_predictor_lora and self.skill_predictor_detach_vlm:
                raise ValueError(
                    "skill_predictor_detach_vlm must be false when skill_predictor_lora=true."
                )
            if (
                self.skill_predictor_freeze_vlm
                and not self.skill_predictor_lora
                and not self.skill_predictor_detach_vlm
            ):
                raise ValueError(
                    "skill_predictor_detach_vlm=false requires skill_predictor_lora=true; "
                    "set skill_predictor_freeze_vlm=false for full VLM co-training."
                )
            if not self.skill_predictor_freeze_vlm and self.skill_predictor_detach_vlm:
                raise ValueError(
                    "skill_predictor_detach_vlm must be false when the complete VLM is co-trained."
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
            if self.skill_predictor_vlm_lr_scale <= 0.0:
                raise ValueError("skill_predictor_vlm_lr_scale must be positive.")
            if self.skill_predictor_focus_uv_loss_weight < 0.0:
                raise ValueError(
                    "skill_predictor_focus_uv_loss_weight must be non-negative."
                )
            if self.skill_predictor_end_state_mode not in {"off", "xyz", "full_state"}:
                raise ValueError("skill_predictor_end_state_mode must be off|xyz|full_state.")
            if self.skill_predictor_focus_uv_enabled and self.skill_predictor_end_state_mode != "off":
                raise ValueError("The skill predictor cannot train UV and end-state heads together.")
            if self.skill_predictor_end_state_dim < 3:
                raise ValueError("skill_predictor_end_state_dim must be at least 3.")
            if not math.isfinite(self.skill_predictor_end_state_loss_weight) or self.skill_predictor_end_state_loss_weight < 0:
                raise ValueError("skill_predictor_end_state_loss_weight must be finite and non-negative.")
            if self.skill_predictor_end_state_skill_source not in {
                "gt",
                "predicted",
                "scheduled",
            }:
                raise ValueError(
                    "skill_predictor_end_state_skill_source must be "
                    "gt|predicted|scheduled."
                )
            if min(
                self.skill_predictor_end_state_schedule_start_step,
                self.skill_predictor_end_state_schedule_end_step,
            ) < 0:
                raise ValueError("End-state skill schedule steps must be non-negative.")
            if (
                self.skill_predictor_end_state_skill_source == "scheduled"
                and self.skill_predictor_end_state_schedule_end_step
                <= self.skill_predictor_end_state_schedule_start_step
            ):
                raise ValueError(
                    "Scheduled end-state skill conditioning requires end_step > start_step."
                )
            if not (
                math.isfinite(
                    self.skill_predictor_end_state_schedule_max_probability
                )
                and 0.0
                <= self.skill_predictor_end_state_schedule_max_probability
                <= 1.0
            ):
                raise ValueError(
                    "skill_predictor_end_state_schedule_max_probability must be in [0, 1]."
                )
            if (
                self.skill_predictor_end_state_mode == "off"
                and self.skill_predictor_end_state_skill_source != "gt"
            ):
                raise ValueError(
                    "Predicted-skill conditioning requires an XYZ/full-state head."
                )
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
    def predictor_transition_sampling(self) -> bool:
        """Whether predictor training owns its mode-specific batch sampler."""
        # Keep old joint predictor+terminator checkpoints loadable for eval.
        # The training entry point rejects creating any new joint run.
        terminator_enabled = any(
            (
                self.train_terminator,
                self.train_image_only_terminator,
                self.train_wrist_only_terminator,
                self.train_state_only_terminator,
                self.train_state_rnn_terminator,
            )
        )
        return self.train_skill_predictor and not terminator_enabled

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
        if self.train_terminator and self.terminator_proprio_history:
            return list(range(1 - self.terminator_history_length, 1))
        if self.train_state_rnn_terminator:
            return list(range(1 - self.state_rnn_terminator_sequence_length, 1))
        return None

    @property
    def action_delta_indices(self) -> list[int] | None:
        if (
            self.train_terminator
            and self.terminator_chunk_end_pose_mode != "off"
        ):
            return list(range(self.terminator_chunk_end_state_horizon))
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None
