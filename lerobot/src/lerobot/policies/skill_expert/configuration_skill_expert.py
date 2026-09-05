"""Configuration for the Stage-1 vision-state-action prior."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig, MuonConfig
from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
    LRSchedulerConfig,
    WarmupConstantSchedulerConfig,
)
from lerobot.utils.constants import ACTION, OBS_STATE


VSA_ARCHITECTURE = "vsa_perceiver_crossattn"
COND_GEMMA_ARCHITECTURE = "cond_gemma"
# VSA revisions are explicit because the three visual-fusion ablations have
# different parameter/state_dict contracts.
VSA_ARCHITECTURE_REVISION = "interleaved_direct1024_v3"
UNCOMPRESSED_VISUAL_KV_REVISION = "visual_kv_uncompressed_v1"
COMPRESSED_VISUAL_KV_REVISION = "visual_kv_perceiver_v1"
LEGACY_RESIDUAL_VSA_REVISION = "residual_sa18_v2"
# Condition-Gemma family. The first revision preserves the skillVLA_real module
# and state_dict contract; ``conditioning_route`` records whether its skill
# broadcast targets Cond-Gemma (historical) or the expert (current Arch0).
# Arch0_1--0_3 ablate the state AdaRMS target; Arch0_adaRMS and Arch0_token keep
# the Arch0 state/visual paths and only change how skill reaches the expert
# (AdaRMS or one in-context token); the final two revisions move both state and
# skill to explicit expert tokens.
COND_GEMMA_ARCHITECTURE_REVISION = "skillvla_real_v1"
COND_GEMMA_EXPERT_STATE_REVISION = "expert_state_adarms_v1"
COND_GEMMA_DUAL_STATE_REVISION = "cond_expert_state_adarms_v1"
COND_GEMMA_SEPARATE_DUAL_STATE_REVISION = (
    "cond_expert_separate_state_adarms_v1"
)
COND_GEMMA_WRIST_DUAL_STATE_REVISION = "wrist_cond_expert_state_adarms_v1"
COND_GEMMA_SKILL_ADARMS_REVISION = "expert_skill_adarms_v1"
COND_GEMMA_SKILL_ADARMS_ZERO_REVISION = "expert_skill_adarms_zero_v1"
COND_GEMMA_SKILL_TOKEN_REVISION = "expert_skill_token_v1"
COND_GEMMA_ISOLATED_SKILL_TOKEN_REVISION = "expert_skill_token_isolated_v1"
COND_GEMMA_COND_SKILL_BROADCAST_REVISION = "cond_skill_broadcast_v1"
COND_GEMMA_DUAL_SKILL_BROADCAST_REVISION = "dual_skill_broadcast_v1"
COND_GEMMA_EXPERT_TOKENS_REVISION = "expert_tokens_uncompressed_v1"
COND_GEMMA_PERCEIVER_EXPERT_TOKENS_REVISION = "expert_tokens_perceiver_v1"
COND_GEMMA_ARCHITECTURE_LABELS = {
    COND_GEMMA_ARCHITECTURE_REVISION: "arch0",
    COND_GEMMA_EXPERT_STATE_REVISION: "arch0_1",
    COND_GEMMA_DUAL_STATE_REVISION: "arch0_2",
    COND_GEMMA_SEPARATE_DUAL_STATE_REVISION: "arch0_2_sep",
    COND_GEMMA_WRIST_DUAL_STATE_REVISION: "arch0_3",
    # ``architecture_label`` is lowercased on validation, so the canonical label
    # of the arch0_adaRMS ablation is stored lowercase.
    COND_GEMMA_SKILL_ADARMS_REVISION: "arch0_adarms",
    COND_GEMMA_SKILL_ADARMS_ZERO_REVISION: "arch0_adarms_zero",
    COND_GEMMA_SKILL_TOKEN_REVISION: "arch0_token",
    COND_GEMMA_ISOLATED_SKILL_TOKEN_REVISION: "arch0_token_iso",
    COND_GEMMA_COND_SKILL_BROADCAST_REVISION: "arch0_cond",
    COND_GEMMA_DUAL_SKILL_BROADCAST_REVISION: "arch0_both",
    COND_GEMMA_EXPERT_TOKENS_REVISION: "arch1_1",
    COND_GEMMA_PERCEIVER_EXPERT_TOKENS_REVISION: "arch1_2",
}
COND_STATE_ADARMS_REVISIONS = frozenset(
    {
        COND_GEMMA_ARCHITECTURE_REVISION,
        COND_GEMMA_DUAL_STATE_REVISION,
        COND_GEMMA_SEPARATE_DUAL_STATE_REVISION,
        COND_GEMMA_WRIST_DUAL_STATE_REVISION,
        COND_GEMMA_SKILL_ADARMS_REVISION,
        COND_GEMMA_SKILL_ADARMS_ZERO_REVISION,
        COND_GEMMA_SKILL_TOKEN_REVISION,
        COND_GEMMA_ISOLATED_SKILL_TOKEN_REVISION,
        COND_GEMMA_COND_SKILL_BROADCAST_REVISION,
        COND_GEMMA_DUAL_SKILL_BROADCAST_REVISION,
    }
)
# Arch0_adaRMS drops the expert skill broadcast and sums the skill embedding
# into the expert AdaRMS condition next to the timestep instead. The shared
# AdaRMS dense keeps this free; a dedicated dense per signal would add
# 37 x Linear(1024, 3072) ~ 116M parameters for a log2(27)-bit code.
EXPERT_SKILL_ADARMS_REVISIONS = frozenset(
    {COND_GEMMA_SKILL_ADARMS_REVISION, COND_GEMMA_SKILL_ADARMS_ZERO_REVISION}
)
# Arch0_adaRMS pins the skill term at unit RMS, but the trained timestep
# embedding sits near RMS 0.1, so skill entered the shared AdaRMS channel ~8x
# louder than the timestep it has to coexist with. Arch0_adaRMS_zero adds a
# zero-init scalar gain after that norm: skill starts silent, and training picks
# its level against the timestep instead of inheriting a hardcoded ratio.
ZERO_INIT_SKILL_GAIN_REVISIONS = frozenset(
    {COND_GEMMA_SKILL_ADARMS_ZERO_REVISION}
)
# Arch0_token drops the broadcast as well, but promotes the skill to a single
# in-context expert token. State keeps the Arch0 Cond-Gemma AdaRMS path, which
# is what separates it from Arch1_1's two-token [state, skill] context.
EXPERT_SKILL_TOKEN_REVISIONS = frozenset(
    {COND_GEMMA_SKILL_TOKEN_REVISION, COND_GEMMA_ISOLATED_SKILL_TOKEN_REVISION}
)
# Arch0_token_iso additionally blanks the skill token's view of the visual
# prefix, so it stays a static per-code embedding rather than a
# scene-contextualized one. Both directions still hide actions from skill.
ISOLATED_SKILL_TOKEN_REVISIONS = frozenset(
    {COND_GEMMA_ISOLATED_SKILL_TOKEN_REVISION}
)
# Arch0 broadcasts the skill to the action expert only. These two ablate the
# broadcast target instead of the mechanism: Cond-Gemma alone, or both streams.
COND_SKILL_BROADCAST_REVISIONS = frozenset(
    {COND_GEMMA_COND_SKILL_BROADCAST_REVISION}
)
DUAL_SKILL_BROADCAST_REVISIONS = frozenset(
    {COND_GEMMA_DUAL_SKILL_BROADCAST_REVISION}
)
EXPERT_STATE_ADARMS_REVISIONS = frozenset(
    {
        COND_GEMMA_EXPERT_STATE_REVISION,
        COND_GEMMA_DUAL_STATE_REVISION,
        COND_GEMMA_SEPARATE_DUAL_STATE_REVISION,
        COND_GEMMA_WRIST_DUAL_STATE_REVISION,
    }
)
INTERLEAVED_CROSS_ATTENTION = "interleaved_cross_attention"
UNCOMPRESSED_VISUAL_KV_SELF_ATTENTION = (
    "uncompressed_visual_kv_self_attention"
)
COMPRESSED_VISUAL_KV_SELF_ATTENTION = "compressed_visual_kv_self_attention"
LEGACY_RESIDUAL_CROSS_ATTENTION = "residual_cross_attention"
IN_CONTEXT_TOKENS = "in_context_tokens"
GLOBAL_VISUAL_ADARMS = "global_visual_adarms"
VISION_CONDITIONING_MODES = (
    UNCOMPRESSED_VISUAL_KV_SELF_ATTENTION,
    COMPRESSED_VISUAL_KV_SELF_ATTENTION,
    INTERLEAVED_CROSS_ATTENTION,
    IN_CONTEXT_TOKENS,
    GLOBAL_VISUAL_ADARMS,
)
VSA_ARCHITECTURE_LABELS = {
    UNCOMPRESSED_VISUAL_KV_SELF_ATTENTION: "arch1_3",
    COMPRESSED_VISUAL_KV_SELF_ATTENTION: "arch2_1",
    INTERLEAVED_CROSS_ATTENTION: "arch2_2",
    IN_CONTEXT_TOKENS: "arch3",
    GLOBAL_VISUAL_ADARMS: "arch4",
}
VSA_REVISION_MODE_LABELS = {
    UNCOMPRESSED_VISUAL_KV_REVISION: {
        UNCOMPRESSED_VISUAL_KV_SELF_ATTENTION: "arch1_3"
    },
    COMPRESSED_VISUAL_KV_REVISION: {
        COMPRESSED_VISUAL_KV_SELF_ATTENTION: "arch2_1"
    },
    VSA_ARCHITECTURE_REVISION: {
        INTERLEAVED_CROSS_ATTENTION: "arch2_2",
        IN_CONTEXT_TOKENS: "arch3",
        GLOBAL_VISUAL_ADARMS: "arch4",
    },
}
LEGACY_VSA_ARCHITECTURE_LABELS = {
    LEGACY_RESIDUAL_CROSS_ATTENTION: "arch2_2",
    IN_CONTEXT_TOKENS: "arch3",
    GLOBAL_VISUAL_ADARMS: "arch4",
}
COND_GEMMA_ARCHITECTURE_LABEL = "arch0"
CONDITIONING_ROUTES = frozenset(
    {
        "state_cond",
        "state_skill_cond",
        "state_skill_only_cond",
        "stateonly_cond",
        "skillonly_cond",
        "visiononly_cond",
    }
)
STATELESS_CONDITIONING_ROUTES = frozenset({"skillonly_cond", "visiononly_cond"})
SKILLLESS_CONDITIONING_ROUTES = frozenset({"stateonly_cond", "visiononly_cond"})
VISIONLESS_CONDITIONING_ROUTES = frozenset({"state_skill_only_cond"})


def normalize_conditioning_route(route: str) -> str:
    normalized = str(route).strip().lower()
    return "skillonly_cond" if normalized == "skill_cond" else normalized


@PreTrainedConfig.register_subclass("skill_expert")
@dataclass
class SkillExpertConfig(PreTrainedConfig):
    """Stage-1 config with explicit VSA or skillVLA_real condition-Gemma layout."""

    model_type: str = "skill_expert"
    dtype: str = "float32"

    architecture: str = VSA_ARCHITECTURE
    # User-facing experiment name. Empty preserves checkpoints saved before the
    # User-facing ablation label; revisions keep old same-name checkpoints exact.
    architecture_label: str = ""
    architecture_revision: str = VSA_ARCHITECTURE_REVISION
    vision_conditioning_mode: str = INTERLEAVED_CROSS_ATTENTION
    include_state_in_visual_crossattn: bool = True
    include_skill_in_visual_crossattn: bool = True
    action_expert_variant: str = "gemma_300m"
    # Used only by the explicitly selected skillVLA_real condition architecture.
    cond_encoder_variant: str = "gemma_300m"
    conditioning_route: str = "state_skill_cond"
    chunk_size: int = 10
    n_action_steps: int = 5
    max_state_dim: int = 32
    max_action_dim: int = 32
    # Logical FSQ taxonomy identity. Dataset variants such as ``_relabeled``
    # may have a different physical folder while preserving the same code
    # meanings. Empty keeps historical checkpoints backward compatible.
    skill_code_space_id: str = ""
    # Dataset/runtime proprio coordinate contract. Historical checkpoints use
    # world-frame EEF xyz (none); new grounded datasets subtract episode-start xyz.
    proprio_grounding: str = "none"

    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0
    # Kept in serialized configs for checkpoint compatibility. Stage1 fixes it
    # to flow; the only selectable trajectory auxiliary is cumulative XYZ.
    action_loss_mode: str = "flow"
    # When enabled, supervise only action offsets that still belong to the
    # effective (possibly transition-jittered) skill assignment.
    mask_actions_after_skill_end: bool = False
    # Optional prefix-trajectory auxiliary: flow + weight * normalized
    # cumulative clean-action XYZ error. Flow always retains coefficient 1.
    cumulative_xyz_loss_enabled: bool = False
    cumulative_xyz_loss_weight: float = 0.5
    # Training-only skill flow objective. Arch0_skill predicts the selected
    # code's complete canonical trajectory. The *_skill_chunk probes instead
    # predict an extended current-frame action chunk. All variants reuse the
    # exact Action Expert and are absent at inference.
    skill_flow_enabled: bool = False
    skill_flow_weight: float = 1.0
    skill_flow_max_length: int = 0
    skill_flow_target: str = "canonical"
    skill_flow_state_conditioned: bool = False
    skill_flow_chunk_multiplier: int = 1
    # Optional IMLE-style mode assignment for the skill-flow architectures.
    # N mode latents are sampled from the fixed 2D square U[-1, 1]^2 and scored
    # at M shared timesteps. New runs rank on the deployed main action route;
    # ``skill_only`` preserves the original auxiliary-route assignment. The
    # best K latents condition both routes through the Action Expert AdaRMS
    # input. Disabled checkpoints allocate no modules.
    skill_flow_latent_best_of_n_enabled: bool = False
    skill_flow_latent_candidates: int = 5
    skill_flow_latent_top_k: int = 1
    skill_flow_latent_assignment_timesteps: int = 2
    # Keep the dataclass default backward-compatible with checkpoints written
    # before this field existed. The Stage-1 YAML resolver explicitly exports
    # ``main`` for new experiments.
    skill_flow_latent_ranking_route: str = "skill_only"
    skill_flow_latent_dim: int = 2
    skill_flow_latent_distribution: str = "uniform_square"
    skill_flow_latent_gain_init: float = 0.1
    # Keep the small mode-latent projection and its scalar gain in FP32 even
    # when the rest of Stage 1 uses BF16 parameters. This avoids sub-ULP AdamW
    # updates being rounded away; historical checkpoints default to BF16.
    skill_flow_latent_fp32: bool = False

    vision_backbone: str = "dino"
    dino_model_path: str = "models/dinov3-vitl16"
    dino_image_size: int = 224
    dino_lr_scale: float = 0.1
    freeze_vision_encoder: bool = False
    dino_lr: float | None = None
    # Phase-batch sampling was removed from Stage 1, but 286 of the existing
    # checkpoints saved these fields into config.json. draccus rejects unknown
    # fields, so dropping them outright made every one of those checkpoints
    # impossible to evaluate. They are retained purely so historical configs stay
    # loadable: nothing reads them and they are deliberately left unvalidated.
    phase_batch_sampling_enabled: bool = False
    phase_batch_focused_fraction: float = 0.75
    phase_batch_early_fraction: float = 0.5
    phase_batch_early_threshold: float = 0.25
    phase_batch_late_threshold: float = 0.75
    num_visual_latents_per_camera: int = 32
    visual_perceiver_width: int = 1024
    skill_vocab_size: int = 27
    skill_fsq_levels: list[int] = field(default_factory=lambda: [3, 3, 3])
    # Aggregate maximum retained for ISS sizing and legacy checkpoints.
    transition_jitter_pmax: int = 0
    transition_jitter_early_start_pmax: int = -1
    transition_jitter_late_start_pmax: int = -1
    transition_jitter_early_end_pmax: int = -1
    transition_jitter_late_end_pmax: int = -1
    transition_jitter_distribution: str = "half_normal"

    # Which skill code conditions the action path during offline training.  The
    # predictor route loads only the learned predictor from a previous Stage-1
    # checkpoint; its frozen pi0.5 VLM base is initialized by pretrained_path.
    training_skill_source: str = "gt"
    skill_predictor_checkpoint_path: str | None = None

    # Legacy checkpoint-schema fields. New Stage1 runs never set/train these;
    # they remain readable so historical checkpoints can expose their own
    # predictor during eval. A predictor selected as training_skill_source is
    # always frozen.
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

    # Legacy checkpoint-schema fields for historical own-terminator evaluation.
    # The current Stage1 forward and optimizer never train this module.
    train_terminator: bool = False
    fsq_path: str | None = None
    terminator_freeze_vision_encoder: bool | None = None
    terminator_dino_model_path: str | None = None
    """Deprecated checkpoint-compatibility field; FSQ terminators use their own checkpoint config."""
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
    # ``vsa_debug_steps`` keeps the optional first-N smoke-test behavior.
    # The schedule is based on real optimizer steps and remains correct on resume.
    vsa_debug_steps: int = 0
    vsa_debug_schedule: tuple[int, ...] = ()
    compile_model: bool = False
    compile_mode: str = "max-autotune"

    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    # Muon probe: route 2D hidden matrices to Muon (match_rms_adamw scaling, so
    # the AdamW-tuned lr/weight_decay above are reused); everything else keeps
    # AdamW. False preserves the historical single-AdamW behavior exactly.
    use_muon: bool = False
    scheduler_warmup_steps: int = 1_000
    scheduler_mode: str = "cosine_decay"
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self) -> None:
        self.architecture_label = str(self.architecture_label).strip().lower()
        self.skill_flow_target = str(self.skill_flow_target).strip().lower()
        self.skill_flow_latent_distribution = str(
            self.skill_flow_latent_distribution
        ).strip().lower()
        self.skill_flow_latent_ranking_route = str(
            self.skill_flow_latent_ranking_route
        ).strip().lower().replace("-", "_")
        self.proprio_grounding = (
            str(self.proprio_grounding or "none").strip().lower().replace("-", "_")
        )
        self.conditioning_route = normalize_conditioning_route(self.conditioning_route)
        self.vsa_debug_schedule = tuple(int(step) for step in self.vsa_debug_schedule)
        super().__post_init__()
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError(f"dtype must be float32 or bfloat16, got {self.dtype!r}.")
        if self.proprio_grounding not in {"none", "episode_start_xyz"}:
            raise ValueError(
                "proprio_grounding must be none|episode_start_xyz, "
                f"got {self.proprio_grounding!r}."
            )
        if self.action_expert_variant != "gemma_300m":
            raise ValueError(
                "Stage 1 fixes the 18-layer expert to gemma_300m; got "
                f"action_expert_variant={self.action_expert_variant!r}."
            )
        if self.architecture not in {VSA_ARCHITECTURE, COND_GEMMA_ARCHITECTURE}:
            raise ValueError(
                "Stage 1 architecture must be "
                f"{VSA_ARCHITECTURE!r} or {COND_GEMMA_ARCHITECTURE!r}, "
                f"got {self.architecture!r}."
            )
        if self.architecture == VSA_ARCHITECTURE:
            if self.architecture_revision in VSA_REVISION_MODE_LABELS:
                architecture_labels = VSA_REVISION_MODE_LABELS[
                    self.architecture_revision
                ]
                supported_modes = tuple(architecture_labels)
            elif self.architecture_revision == LEGACY_RESIDUAL_VSA_REVISION:
                supported_modes = tuple(LEGACY_VSA_ARCHITECTURE_LABELS)
                architecture_labels = LEGACY_VSA_ARCHITECTURE_LABELS
            else:
                raise ValueError(
                    "Unsupported VSA architecture_revision="
                    f"{self.architecture_revision!r}; expected "
                    f"one of {tuple(VSA_REVISION_MODE_LABELS)!r} for new training or "
                    f"{LEGACY_RESIDUAL_VSA_REVISION!r} for historical evaluation."
                )
            if self.vision_conditioning_mode not in supported_modes:
                raise ValueError(
                    "vision_conditioning_mode must be one of "
                    f"{supported_modes}, got {self.vision_conditioning_mode!r}."
                )
            expected_architecture_label = architecture_labels[
                self.vision_conditioning_mode
            ]
        else:
            if self.architecture_revision not in COND_GEMMA_ARCHITECTURE_LABELS:
                raise ValueError(
                    "Unsupported cond_gemma architecture_revision="
                    f"got {self.architecture_revision!r}."
                )
            if self.cond_encoder_variant != self.action_expert_variant:
                raise ValueError(
                    "cond_gemma requires matching 18-layer cond/expert variants; got "
                    f"{self.cond_encoder_variant!r} and {self.action_expert_variant!r}."
                )
            if self.conditioning_route not in CONDITIONING_ROUTES:
                raise ValueError(
                    f"conditioning_route must be one of {sorted(CONDITIONING_ROUTES)}, "
                    f"got {self.conditioning_route!r}."
                )
            if (
                self.architecture_revision
                in {
                    COND_GEMMA_EXPERT_TOKENS_REVISION,
                    COND_GEMMA_PERCEIVER_EXPERT_TOKENS_REVISION,
                }
                and self.conditioning_route != "state_skill_cond"
            ):
                raise ValueError(
                    f"{COND_GEMMA_ARCHITECTURE_LABELS[self.architecture_revision]} "
                    "fixes conditioning_route='state_skill_cond'; got "
                    f"{self.conditioning_route!r}."
                )
            if (
                self.architecture_revision
                in {
                    COND_GEMMA_EXPERT_STATE_REVISION,
                    COND_GEMMA_DUAL_STATE_REVISION,
                    COND_GEMMA_SEPARATE_DUAL_STATE_REVISION,
                    COND_GEMMA_WRIST_DUAL_STATE_REVISION,
                    COND_GEMMA_SKILL_ADARMS_REVISION,
                    COND_GEMMA_SKILL_ADARMS_ZERO_REVISION,
                    COND_GEMMA_SKILL_TOKEN_REVISION,
                    COND_GEMMA_ISOLATED_SKILL_TOKEN_REVISION,
                    COND_GEMMA_COND_SKILL_BROADCAST_REVISION,
                    COND_GEMMA_DUAL_SKILL_BROADCAST_REVISION,
                }
                and self.conditioning_route != "state_cond"
            ):
                raise ValueError(
                    f"{COND_GEMMA_ARCHITECTURE_LABELS[self.architecture_revision]} "
                    "fixes conditioning_route='state_cond' so skill bypasses "
                    "Cond-Gemma and reaches the expert only; got "
                    f"{self.conditioning_route!r}."
                )
            expected_architecture_label = COND_GEMMA_ARCHITECTURE_LABELS[
                self.architecture_revision
            ]
        if (
            self.architecture_label
            and self.architecture_label != expected_architecture_label
            # Checkpoints produced before the rename saved the current Arch0
            # revision as "arch1". Accept only that historical metadata alias.
            and not (
                self.architecture == COND_GEMMA_ARCHITECTURE
                and self.architecture_revision == COND_GEMMA_ARCHITECTURE_REVISION
                and self.architecture_label == "arch1"
            )
            # Arch0 skill auxiliaries share Arch0's parameter/state-dict
            # contract and differ only by a training-time objective.
            and not (
                self.architecture == COND_GEMMA_ARCHITECTURE
                and self.architecture_revision == COND_GEMMA_ARCHITECTURE_REVISION
                and self.architecture_label
                in {"arch0_skill", "arch0_skill_chunk"}
            )
            # Arch0_2_skill_chunk similarly shares Arch0_2's exact parameter
            # and rollout contract.
            and not (
                self.architecture == COND_GEMMA_ARCHITECTURE
                and self.architecture_revision == COND_GEMMA_DUAL_STATE_REVISION
                and self.architecture_label == "arch0_2_skill_chunk"
            )
            # Checkpoints from before the rename saved the current alternating
            # cross-attention revision as "arch2"; it is now called Arch2_2.
            and not (
                self.architecture == VSA_ARCHITECTURE
                and self.architecture_revision == VSA_ARCHITECTURE_REVISION
                and self.vision_conditioning_mode == INTERLEAVED_CROSS_ATTENTION
                and self.architecture_label == "arch2"
            )
        ):
            raise ValueError(
                f"architecture_label={self.architecture_label!r} does not match "
                f"{self.architecture}/{getattr(self, 'vision_conditioning_mode', '')}: "
                f"expected {expected_architecture_label!r}."
            )
        if self.vision_backbone != "dino":
            raise ValueError("Stage 1 uses the Stage-0 DINO vision path; vision_backbone must be 'dino'.")
        if self.dino_image_size <= 0:
            raise ValueError("dino_image_size must be positive.")
        if self.dino_lr_scale <= 0.0:
            raise ValueError("dino_lr_scale must be positive.")
        if self.architecture == VSA_ARCHITECTURE or (
            self.architecture == COND_GEMMA_ARCHITECTURE
            and self.architecture_revision
            == COND_GEMMA_PERCEIVER_EXPERT_TOKENS_REVISION
        ):
            if not 1 <= self.num_visual_latents_per_camera <= 197:
                raise ValueError(
                    "num_visual_latents_per_camera must be between 1 and 197."
                )
            if self.visual_perceiver_width <= 0:
                raise ValueError("visual_perceiver_width must be positive.")
        else:
            if self.dino_lr is not None and self.dino_lr <= 0.0:
                raise ValueError("dino_lr must be positive when set.")
            if self.freeze_vision_encoder and self.dino_lr is not None:
                raise ValueError("dino_lr cannot be set when freeze_vision_encoder=True.")
        if self.vsa_debug_steps < 0:
            raise ValueError("vsa_debug_steps must be non-negative.")
        if any(step <= 0 for step in self.vsa_debug_schedule):
            raise ValueError("vsa_debug_schedule entries must be positive optimizer steps.")
        if tuple(sorted(set(self.vsa_debug_schedule))) != self.vsa_debug_schedule:
            raise ValueError("vsa_debug_schedule must be sorted and contain no duplicates.")
        if self.scheduler_mode not in {"cosine_decay", "warmup_constant"}:
            raise ValueError(
                "scheduler_mode must be 'cosine_decay' or 'warmup_constant', got "
                f"{self.scheduler_mode!r}."
            )
        if self.scheduler_warmup_steps < 0:
            raise ValueError("scheduler_warmup_steps must be non-negative.")
        if self.scheduler_decay_steps <= 0:
            raise ValueError("scheduler_decay_steps must be positive.")
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
        if self.model_type == "skill_expert" and self.action_loss_mode != "flow":
            raise ValueError(
                "Stage1 action_loss_mode is fixed to 'flow'; configure only "
                "cumulative_xyz_loss_enabled and cumulative_xyz_loss_weight."
            )
        if not math.isfinite(self.cumulative_xyz_loss_weight) or self.cumulative_xyz_loss_weight <= 0:
            raise ValueError("cumulative_xyz_loss_weight must be finite and positive.")
        if not math.isfinite(self.skill_flow_weight) or self.skill_flow_weight <= 0:
            raise ValueError("skill_flow_weight must be finite and positive.")
        if self.skill_flow_max_length < 0:
            raise ValueError("skill_flow_max_length must be non-negative.")
        if self.skill_flow_target not in {"canonical", "extended_chunk"}:
            raise ValueError(
                "skill_flow_target must be canonical|extended_chunk, got "
                f"{self.skill_flow_target!r}."
            )
        if self.skill_flow_chunk_multiplier <= 0:
            raise ValueError("skill_flow_chunk_multiplier must be positive.")
        if self.skill_flow_latent_candidates <= 0:
            raise ValueError("skill_flow_latent_candidates must be positive.")
        if not 1 <= self.skill_flow_latent_top_k <= self.skill_flow_latent_candidates:
            raise ValueError(
                "skill_flow_latent_top_k must be within [1, candidates], got "
                f"{self.skill_flow_latent_top_k} for "
                f"{self.skill_flow_latent_candidates} candidates."
            )
        if self.skill_flow_latent_assignment_timesteps <= 0:
            raise ValueError(
                "skill_flow_latent_assignment_timesteps must be positive."
            )
        if self.skill_flow_latent_ranking_route not in {"main", "skill_only"}:
            raise ValueError(
                "skill_flow_latent_ranking_route must be main|skill_only, got "
                f"{self.skill_flow_latent_ranking_route!r}."
            )
        if self.skill_flow_latent_dim != 2:
            raise ValueError(
                "The Stage-1 mode latent is fixed to two dimensions; got "
                f"skill_flow_latent_dim={self.skill_flow_latent_dim}."
            )
        if self.skill_flow_latent_distribution != "uniform_square":
            raise ValueError(
                "skill_flow_latent_distribution is fixed to 'uniform_square', got "
                f"{self.skill_flow_latent_distribution!r}."
            )
        if (
            not math.isfinite(self.skill_flow_latent_gain_init)
            or self.skill_flow_latent_gain_init <= 0
        ):
            raise ValueError("skill_flow_latent_gain_init must be finite and positive.")
        if self.skill_flow_latent_best_of_n_enabled:
            # Stage 2 disables the training-only skill-flow auxiliary while
            # retaining the Stage-1 mode-latent projection as part of its
            # frozen action prior.
            if not self.skill_flow_enabled and self.model_type != "skill_vla_stage2":
                raise ValueError(
                    "latent Best-of-N requires skill_flow_enabled."
                )
            if self.architecture_label not in {
                "arch0_skill",
                "arch0_skill_chunk",
                "arch0_2_skill_chunk",
            }:
                raise ValueError(
                    "latent Best-of-N is supported only by arch0_skill, "
                    "arch0_skill_chunk, and arch0_2_skill_chunk; got "
                    f"{self.architecture_label!r}."
                )
        elif self.skill_flow_latent_fp32:
            raise ValueError(
                "skill_flow_latent_fp32 requires latent Best-of-N to be enabled."
            )
        if self.skill_flow_enabled:
            supported_skill_flow = {
                "arch0_skill": (
                    COND_GEMMA_ARCHITECTURE_REVISION,
                    "canonical",
                    False,
                ),
                "arch0_skill_chunk": (
                    COND_GEMMA_ARCHITECTURE_REVISION,
                    "extended_chunk",
                    False,
                ),
                "arch0_2_skill_chunk": (
                    COND_GEMMA_DUAL_STATE_REVISION,
                    "extended_chunk",
                    True,
                ),
            }
            expected = supported_skill_flow.get(self.architecture_label)
            actual = (
                self.architecture_revision,
                self.skill_flow_target,
                self.skill_flow_state_conditioned,
            )
            if not (
                self.architecture == COND_GEMMA_ARCHITECTURE
                and self.conditioning_route == "state_cond"
                and expected == actual
            ):
                raise ValueError(
                    "skill_flow_enabled requires one of "
                    "arch0_skill|arch0_skill_chunk|arch0_2_skill_chunk with its "
                    f"fixed target/state contract; got label={self.architecture_label!r}, "
                    f"revision={self.architecture_revision!r}, "
                    f"target={self.skill_flow_target!r}, "
                    f"state_conditioned={self.skill_flow_state_conditioned}."
                )
            if self.skill_flow_max_length <= 0:
                raise ValueError(
                    "Skill-flow architectures require a positive skill_flow_max_length."
                )
            if (
                self.skill_flow_target == "canonical"
                and self.training_skill_source != "gt"
            ):
                raise ValueError(
                    "arch0_skill currently requires training_skill_source='gt'."
                )
            if (
                self.skill_flow_target == "extended_chunk"
                and self.skill_flow_max_length
                != self.chunk_size * self.skill_flow_chunk_multiplier
            ):
                raise ValueError(
                    "Extended skill-flow length must equal chunk_size * "
                    "skill_flow_chunk_multiplier, got "
                    f"{self.skill_flow_max_length} != {self.chunk_size} * "
                    f"{self.skill_flow_chunk_multiplier}."
                )
        if self.transition_jitter_pmax < 0:
            raise ValueError("transition_jitter_pmax must be non-negative.")
        directional_jitter = (
            self.transition_jitter_early_start_pmax,
            self.transition_jitter_late_start_pmax,
            self.transition_jitter_early_end_pmax,
            self.transition_jitter_late_end_pmax,
        )
        if any(value < -1 for value in directional_jitter):
            raise ValueError(
                "transition jitter directional pmax values must be -1 (legacy "
                f"fallback) or non-negative, got {directional_jitter}."
            )
        resolved_jitter = tuple(
            self.transition_jitter_pmax if value < 0 else value
            for value in directional_jitter
        )
        if max(resolved_jitter) > self.transition_jitter_pmax:
            raise ValueError(
                "transition_jitter_pmax must cover every directional window: "
                f"storage={self.transition_jitter_pmax}, directional={resolved_jitter}."
            )
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
            if self.skill_predictor_lora:
                if not str(self.skill_predictor_lora_targets).strip():
                    raise ValueError("skill_predictor_lora_targets cannot be empty.")
                if self.skill_predictor_lora_rank <= 0:
                    raise ValueError("skill_predictor_lora_rank must be positive.")
                if self.skill_predictor_lora_alpha <= 0.0:
                    raise ValueError("skill_predictor_lora_alpha must be positive.")
                if self.skill_predictor_lora_dropout < 0.0:
                    raise ValueError("skill_predictor_lora_dropout must be non-negative.")
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
                raise ValueError("Historical own terminator requires fsq_path.")

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

    def get_optimizer_preset(self) -> AdamWConfig | MuonConfig:
        if self.use_muon:
            # Muon-specific hyperparameters (momentum, ns_steps, ...) stay at
            # MuonConfig defaults; the shared lr/weight_decay are reusable
            # because of the match_rms_adamw update scaling.
            return MuonConfig(
                lr=self.optimizer_lr,
                weight_decay=self.optimizer_weight_decay,
                grad_clip_norm=self.optimizer_grad_clip_norm,
                adamw_betas=self.optimizer_betas,
                adamw_eps=self.optimizer_eps,
            )
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig:
        if self.scheduler_mode == "warmup_constant":
            return WarmupConstantSchedulerConfig(
                num_warmup_steps=self.scheduler_warmup_steps
            )
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
        horizon = self.chunk_size
        if self.skill_flow_enabled and self.skill_flow_target == "extended_chunk":
            horizon = max(horizon, self.skill_flow_max_length)
        return list(range(horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
