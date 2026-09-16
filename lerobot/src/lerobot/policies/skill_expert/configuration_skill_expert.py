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


COND_GEMMA_ARCHITECTURE = "cond_gemma"
COND_GEMMA_ARCHITECTURE_REVISION = "skillvla_real_v1"
FIXED_VISUAL_BOTTLENECK_ARCHITECTURE = "fixed_visual_bottleneck"
FIXED_VISUAL_BOTTLENECK_REVISION = "fixed_visual_bottleneck_v1"
LATE_VISUAL_BOTTLENECK_REVISION = "late_visual_bottleneck_v1"
LAYERWISE_COND_BOTTLENECK_ARCHITECTURE = "layerwise_cond_bottleneck"
LAYERWISE_COND_BOTTLENECK_REVISION = "layerwise_cond_bottleneck_v1"
LAYERWISE_COND_BOTTLENECK_CORE_EXIT_REVISION = "layerwise_cond_bottleneck_core_exit_v1"
LAYERWISE_COND_BOTTLENECK_UV_REVISION = "layerwise_cond_bottleneck_core_exit_uv_v1"
LAYERWISE_COND_BOTTLENECK_LATENT_UV_REVISION = "layerwise_cond_bottleneck_core_exit_latent_uv_v1"
SUPPORTED_ARCHITECTURE_LABELS = frozenset(
    {
        "arch0",
        "arch0_skill",
        "arch0_skill_chunk",
        "arch1",
        "arch1_skill",
        "arch1_skill_chunk",
        "arch2",
        "arch2_skill",
        "arch2_skill_chunk",
        "arch3",
        "arch3_skill",
        "arch3_skill_chunk",
        "arch4",
        "arch4_skill",
        "arch4_skill_chunk",
        "arch5",
        "arch5_skill",
        "arch5_skill_chunk",
        "arch6",
        "arch6_skill",
        "arch6_skill_chunk",
    }
)
INTERLEAVED_CROSS_ATTENTION = "interleaved_cross_attention"
FIXED_BOTTLENECK_CROSS_ATTENTION = "fixed_bottleneck_cross_attention"
LAYERWISE_COND_BOTTLENECK_CROSS_ATTENTION = (
    "layerwise_cond_bottleneck_cross_attention"
)
# These legacy route groups remain exported because Stage 2 imports them while
# loading historical metadata. New Stage-1 configs never select one.
STATELESS_CONDITIONING_ROUTES = frozenset({"skillonly_cond", "visiononly_cond"})
SKILLLESS_CONDITIONING_ROUTES = frozenset({"stateonly_cond", "visiononly_cond"})
VISIONLESS_CONDITIONING_ROUTES = frozenset({"state_skill_only_cond"})


def normalize_conditioning_route(route: str) -> str:
    return str(route).strip().lower()


@PreTrainedConfig.register_subclass("skill_expert")
@dataclass
class SkillExpertConfig(PreTrainedConfig):
    """Configuration shared by the retained Arch0--Arch6 Stage-1 modes."""

    model_type: str = "skill_expert"
    dtype: str = "float32"

    architecture: str = COND_GEMMA_ARCHITECTURE
    architecture_label: str = "arch0"
    architecture_revision: str = COND_GEMMA_ARCHITECTURE_REVISION
    vision_conditioning_mode: str = INTERLEAVED_CROSS_ATTENTION
    include_state_in_visual_crossattn: bool = True
    include_skill_in_visual_crossattn: bool = True
    action_expert_variant: str = "gemma_300m"
    cond_encoder_variant: str = "gemma_300m"
    conditioning_route: str = "state_cond"
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
    # Optional endpoint-focused top-view preprocessing. The coordinates are
    # materialized per canonical skill in ``skill_focus_uv.npz`` and selected
    # with the same k -> k' transition jitter as the skill code. Color and
    # input-blur draws are shared by top and wrist cameras; crop jitter and the
    # focused transform apply only to the top view. ``partial_fov`` retains the
    # historical full-frame blur mask; ``crop`` makes a resized local view with
    # a smaller box/blur focus cue inside it.
    foveated_vision_enabled: bool = False
    foveation_randomization_enabled: bool = False
    foveation_mode: str = "partial_fov"
    foveation_crop_size: int = 128
    foveation_output_size: int = 224
    foveation_inner_box_enabled: bool = True
    foveation_inner_box_mode: str = "blur"
    foveation_inner_box_size: int = 32
    foveation_inner_box_line_width: int = 3
    foveation_shape: str = "square"
    foveation_sharp_size: int = 96
    foveation_feather: int = 20
    foveation_peripheral_mode: str = "blur"
    foveation_peripheral_blur_radius: float = 8.0
    foveation_color_enabled: bool = False
    foveation_brightness_min: float = 0.8
    foveation_brightness_max: float = 1.2
    foveation_contrast_min: float = 0.8
    foveation_contrast_max: float = 1.2
    foveation_saturation_min: float = 0.8
    foveation_saturation_max: float = 1.2
    foveation_hue_min: float = -0.15
    foveation_hue_max: float = 0.15
    foveation_crop_enabled: bool = False
    foveation_crop_offset_min_px: int = -24
    foveation_crop_offset_max_px: int = 24
    foveation_inner_box_offset_min_px: int = -4
    foveation_inner_box_offset_max_px: int = 4
    foveation_input_blur_enabled: bool = False
    foveation_input_blur_min_radius: float = 0.0
    foveation_input_blur_max_radius: float = 4.0
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
    # Arch1/Arch2's fixed DINO interface and Arch3/4/5's recurrent Cond-Gemma
    # interface. Arch1/2 split tokens evenly across top/wrist; Arch3/4/5 read the
    # combined Cond stream. The legacy default keeps old configs loadable.
    visual_bottleneck_tokens: int = 4
    visual_bottleneck_width: int = 256
    visual_bottleneck_heads: int = 4
    visual_bridge_heads: int = 8
    visual_bridge_gate_init: float = 0.01
    # Arch5/6: auxiliary endpoint UV alignment from final Cond/bottleneck tokens.
    # The readout is never fed back into the action policy.
    cond_focus_uv_loss_weight: float = 1.0
    # Arch2/Arch3/Arch4/Arch5 expose their bottleneck only to this many terminal
    # Action-Expert layers. Arch1 ignores this field and retains visual access
    # at every layer.
    visual_bridge_last_n_layers: int = 1
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
    # Auxiliary predictor checkpoints may either keep the pi0.5 VLM frozen
    # (optionally training a named LoRA) or co-train the complete VLM.  Stage 1
    # itself still treats an attached predictor as frozen.
    skill_predictor_freeze_vlm: bool = True
    skill_predictor_vlm_lr_scale: float = 1.0
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
    skill_predictor_focus_uv_enabled: bool = False
    skill_predictor_focus_uv_loss_weight: float = 0.25
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
        self.foveation_shape = str(self.foveation_shape).strip().lower()
        self.foveation_mode = str(self.foveation_mode).strip().lower()
        self.foveation_peripheral_mode = str(self.foveation_peripheral_mode).strip().lower()
        self.foveation_inner_box_mode = str(
            self.foveation_inner_box_mode
        ).strip().lower()
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
        if self.architecture_label not in SUPPORTED_ARCHITECTURE_LABELS:
            raise ValueError(
                "architecture_label must be arch0|arch0_skill|arch0_skill_chunk|"
                "arch1|arch1_skill|arch1_skill_chunk|"
                "arch2|arch2_skill|arch2_skill_chunk|"
                "arch3|arch3_skill|arch3_skill_chunk|"
                "arch4|arch4_skill|arch4_skill_chunk|"
                "arch5|arch5_skill|arch5_skill_chunk|"
                "arch6|arch6_skill|arch6_skill_chunk, "
                f"got {self.architecture_label!r}."
            )
        is_arch1 = self.architecture_label.startswith("arch1")
        is_arch2 = self.architecture_label.startswith("arch2")
        is_arch3 = self.architecture_label.startswith("arch3")
        is_arch4 = self.architecture_label.startswith("arch4")
        is_arch5 = self.architecture_label.startswith("arch5")
        is_arch6 = self.architecture_label.startswith("arch6")
        is_layerwise = is_arch3 or is_arch4 or is_arch5 or is_arch6
        is_visual_bottleneck = is_arch1 or is_arch2
        expected_architecture = (
            LAYERWISE_COND_BOTTLENECK_ARCHITECTURE
            if is_layerwise
            else (
                FIXED_VISUAL_BOTTLENECK_ARCHITECTURE
                if is_visual_bottleneck
                else COND_GEMMA_ARCHITECTURE
            )
        )
        expected_revision = (
            LAYERWISE_COND_BOTTLENECK_LATENT_UV_REVISION
            if is_arch6
            else (
                LAYERWISE_COND_BOTTLENECK_UV_REVISION
                if is_arch5
                else (
                    LAYERWISE_COND_BOTTLENECK_CORE_EXIT_REVISION
                    if is_arch4
                    else (
                        LAYERWISE_COND_BOTTLENECK_REVISION
                        if is_arch3
                        else (
                            LATE_VISUAL_BOTTLENECK_REVISION
                            if is_arch2
                            else (
                                FIXED_VISUAL_BOTTLENECK_REVISION
                                if is_arch1
                                else COND_GEMMA_ARCHITECTURE_REVISION
                            )
                        )
                    )
                )
            )
        )
        expected_vision_mode = (
            LAYERWISE_COND_BOTTLENECK_CROSS_ATTENTION
            if is_layerwise
            else (
                FIXED_BOTTLENECK_CROSS_ATTENTION
                if is_visual_bottleneck
                else INTERLEAVED_CROSS_ATTENTION
            )
        )
        if self.architecture != expected_architecture:
            family_name = (
                "fixed visual bottleneck"
                if is_visual_bottleneck
                else "Cond-Gemma"
            )
            raise ValueError(
                f"{self.architecture_label} requires architecture="
                f"{expected_architecture!r} ({family_name}); got "
                f"{self.architecture!r}."
            )
        if self.architecture_revision != expected_revision:
            raise ValueError(
                f"{self.architecture_label} requires architecture_revision="
                f"{expected_revision!r}; got {self.architecture_revision!r}."
            )
        if (not is_visual_bottleneck) and (
            self.cond_encoder_variant != self.action_expert_variant
        ):
            raise ValueError(
                "Arch0/Arch3/Arch4/Arch5 require matching 18-layer cond/expert variants; got "
                f"{self.cond_encoder_variant!r} and {self.action_expert_variant!r}."
            )
        if self.conditioning_route != "state_cond":
            raise ValueError(
                "The retained Stage1 contract fixes conditioning_route='state_cond'; "
                f"got {self.conditioning_route!r}."
            )
        if self.vision_conditioning_mode != expected_vision_mode:
            raise ValueError(
                f"{self.architecture_label} requires vision_conditioning_mode="
                f"{expected_vision_mode!r}; got "
                f"{self.vision_conditioning_mode!r}."
            )
        if is_visual_bottleneck or is_layerwise:
            if (
                int(self.visual_bottleneck_tokens) <= 0
                or (
                    is_visual_bottleneck
                    and int(self.visual_bottleneck_tokens) % 2 != 0
                )
            ):
                raise ValueError(
                    "visual_bottleneck_tokens must be positive (and even for "
                    "Arch1/Arch2 so top and wrist receive equal query counts); "
                    f"got {self.visual_bottleneck_tokens}."
                )
            fixed_interface = {
                "visual_bottleneck_width": (self.visual_bottleneck_width, 256),
                "visual_bottleneck_heads": (self.visual_bottleneck_heads, 4),
                "visual_bridge_heads": (self.visual_bridge_heads, 8),
                "visual_bridge_gate_init": (self.visual_bridge_gate_init, 0.01),
            }
            changed = {
                name: actual
                for name, (actual, expected) in fixed_interface.items()
                if actual != expected
            }
            if changed:
                raise ValueError(
                    "Arch1/Arch2/Arch3/Arch4/Arch5 fix the non-token bottleneck geometry; "
                    "got overrides "
                    f"{changed}."
                )
        if not 1 <= int(self.visual_bridge_last_n_layers) <= 18:
            raise ValueError(
                "visual_bridge_last_n_layers must be within [1, 18], got "
                f"{self.visual_bridge_last_n_layers}."
            )
        if (is_arch4 or is_arch5 or is_arch6) and int(self.visual_bridge_last_n_layers) == 18:
            raise ValueError(
                "Arch4/Arch5/Arch6 require visual_bridge_last_n_layers <= 17 so the "
                "skill-only motion core contains at least one Expert layer."
            )
        if not math.isfinite(self.cond_focus_uv_loss_weight) or self.cond_focus_uv_loss_weight <= 0:
            raise ValueError("cond_focus_uv_loss_weight must be finite and positive.")
        if not (is_arch5 or is_arch6) and self.cond_focus_uv_loss_weight != 1.0:
            raise ValueError("cond_focus_uv_loss_weight is configurable only for Arch5/Arch6.")
        if not (is_arch2 or is_layerwise) and int(self.visual_bridge_last_n_layers) != 1:
            raise ValueError(
                "visual_bridge_last_n_layers is an Arch2/Arch3/Arch4/Arch5 setting; "
                "Arch0/Arch1 must leave it at the compatibility default 1."
            )
        if self.vision_backbone != "dino":
            raise ValueError("Stage 1 requires the DINO vision path; vision_backbone must be 'dino'.")
        if self.dino_image_size <= 0:
            raise ValueError("dino_image_size must be positive.")
        if self.dino_lr_scale <= 0.0:
            raise ValueError("dino_lr_scale must be positive.")
        if self.dino_lr is not None and self.dino_lr <= 0.0:
            raise ValueError("dino_lr must be positive when set.")
        if self.freeze_vision_encoder and self.dino_lr is not None:
            raise ValueError("dino_lr cannot be set when freeze_vision_encoder=True.")
        if self.foveation_shape not in {"square", "circle"}:
            raise ValueError(
                "foveation_shape must be square|circle, got "
                f"{self.foveation_shape!r}."
            )
        if self.foveation_mode not in {"partial_fov", "crop"}:
            raise ValueError(
                "foveation_mode must be partial_fov|crop, got "
                f"{self.foveation_mode!r}."
            )
        if self.foveation_crop_size <= 0 or self.foveation_output_size <= 0:
            raise ValueError(
                "foveation_crop_size and foveation_output_size must be positive."
            )
        if self.foveation_inner_box_mode not in {"box", "blur"}:
            raise ValueError(
                "foveation_inner_box_mode must be box|blur, got "
                f"{self.foveation_inner_box_mode!r}."
            )
        if (
            self.foveation_inner_box_size <= 0
            or self.foveation_inner_box_line_width <= 0
        ):
            raise ValueError(
                "foveation inner-box size and line width must be positive."
            )
        if self.foveation_inner_box_size > self.foveation_crop_size:
            raise ValueError(
                "foveation_inner_box_size cannot exceed foveation_crop_size."
            )
        if self.foveation_sharp_size <= 0:
            raise ValueError("foveation_sharp_size must be positive.")
        if self.foveation_feather < 0:
            raise ValueError("foveation_feather must be non-negative.")
        if self.foveation_peripheral_mode not in {"blur", "black"}:
            raise ValueError("foveation_peripheral_mode must be blur|black.")
        if self.foveation_mode == "crop" and self.foveation_peripheral_mode == "black":
            raise ValueError("foveation_peripheral_mode=black requires partial_fov.")
        if self.foveation_peripheral_blur_radius < 0:
            raise ValueError(
                "foveation_peripheral_blur_radius must be non-negative."
            )
        for name, low, high, minimum, maximum in (
            (
                "brightness",
                self.foveation_brightness_min,
                self.foveation_brightness_max,
                0.0,
                None,
            ),
            (
                "contrast",
                self.foveation_contrast_min,
                self.foveation_contrast_max,
                0.0,
                None,
            ),
            (
                "saturation",
                self.foveation_saturation_min,
                self.foveation_saturation_max,
                0.0,
                None,
            ),
            (
                "hue",
                self.foveation_hue_min,
                self.foveation_hue_max,
                -0.5,
                0.5,
            ),
            (
                "input_blur_radius",
                self.foveation_input_blur_min_radius,
                self.foveation_input_blur_max_radius,
                0.0,
                None,
            ),
        ):
            if not math.isfinite(low) or not math.isfinite(high) or low > high:
                raise ValueError(
                    f"foveation_{name} range must be finite and ordered, got "
                    f"[{low}, {high}]."
                )
            if low < minimum or (maximum is not None and high > maximum):
                suffix = (
                    f">= {minimum}"
                    if maximum is None
                    else f"within [{minimum}, {maximum}]"
                )
                raise ValueError(f"foveation_{name} values must be {suffix}.")
        if self.foveation_crop_offset_min_px > self.foveation_crop_offset_max_px:
            raise ValueError(
                "foveation crop-offset range must be ordered, got "
                f"[{self.foveation_crop_offset_min_px}, "
                f"{self.foveation_crop_offset_max_px}]."
            )
        if (
            self.foveation_inner_box_offset_min_px
            > self.foveation_inner_box_offset_max_px
        ):
            raise ValueError(
                "foveation inner-box offset range must be ordered, got "
                f"[{self.foveation_inner_box_offset_min_px}, "
                f"{self.foveation_inner_box_offset_max_px}]."
            )
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
                "arch1_skill",
                "arch1_skill_chunk",
                "arch2_skill",
                "arch2_skill_chunk",
                "arch3_skill",
                "arch3_skill_chunk",
                "arch4_skill",
                "arch4_skill_chunk",
                "arch5_skill",
                "arch5_skill_chunk",
                "arch6_skill",
                "arch6_skill_chunk",
            }:
                raise ValueError(
                    "latent Best-of-N is supported only by *_skill and "
                    "*_skill_chunk modes; got "
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
                "arch1_skill": (
                    FIXED_VISUAL_BOTTLENECK_REVISION,
                    "canonical",
                    False,
                ),
                "arch1_skill_chunk": (
                    FIXED_VISUAL_BOTTLENECK_REVISION,
                    "extended_chunk",
                    False,
                ),
                "arch2_skill": (
                    LATE_VISUAL_BOTTLENECK_REVISION,
                    "canonical",
                    False,
                ),
                "arch2_skill_chunk": (
                    LATE_VISUAL_BOTTLENECK_REVISION,
                    "extended_chunk",
                    False,
                ),
                "arch3_skill": (
                    LAYERWISE_COND_BOTTLENECK_REVISION,
                    "canonical",
                    False,
                ),
                "arch3_skill_chunk": (
                    LAYERWISE_COND_BOTTLENECK_REVISION,
                    "extended_chunk",
                    False,
                ),
                "arch4_skill": (
                    LAYERWISE_COND_BOTTLENECK_CORE_EXIT_REVISION,
                    "canonical",
                    False,
                ),
                "arch4_skill_chunk": (
                    LAYERWISE_COND_BOTTLENECK_CORE_EXIT_REVISION,
                    "extended_chunk",
                    False,
                ),
                "arch5_skill": (
                    LAYERWISE_COND_BOTTLENECK_UV_REVISION,
                    "canonical",
                    False,
                ),
                "arch5_skill_chunk": (
                    LAYERWISE_COND_BOTTLENECK_UV_REVISION,
                    "extended_chunk",
                    False,
                ),
                "arch6_skill": (
                    LAYERWISE_COND_BOTTLENECK_LATENT_UV_REVISION,
                    "canonical",
                    False,
                ),
                "arch6_skill_chunk": (
                    LAYERWISE_COND_BOTTLENECK_LATENT_UV_REVISION,
                    "extended_chunk",
                    False,
                ),
            }
            expected = supported_skill_flow.get(self.architecture_label)
            actual = (
                self.architecture_revision,
                self.skill_flow_target,
                self.skill_flow_state_conditioned,
            )
            if not (
                self.architecture == expected_architecture
                and self.conditioning_route == "state_cond"
                and expected == actual
            ):
                raise ValueError(
                    "skill_flow_enabled requires one of "
                    "arch0_skill|arch0_skill_chunk|arch1_skill|"
                    "arch1_skill_chunk|arch2_skill|arch2_skill_chunk|"
                    "arch3_skill|arch3_skill_chunk|arch4_skill|"
                    "arch4_skill_chunk|arch5_skill|arch5_skill_chunk|"
                    "arch6_skill|arch6_skill_chunk with its "
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
                    "Canonical *_skill modes currently require "
                    "training_skill_source='gt'."
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
        if self.model_type == "skill_expert":
            expected_skill_flow = self.architecture_label not in {
                "arch0",
                "arch1",
                "arch2",
                "arch3",
                "arch4",
                "arch5",
                "arch6",
            }
            if self.skill_flow_enabled != expected_skill_flow:
                raise ValueError(
                    f"{self.architecture_label} requires "
                    f"skill_flow_enabled={expected_skill_flow}."
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
            if not self.skill_predictor_freeze_vlm and self.skill_predictor_lora:
                raise ValueError(
                    "skill_predictor_lora must be False when the complete predictor VLM "
                    "is co-trained."
                )
            if self.skill_predictor_lora and self.skill_predictor_detach_vlm:
                raise ValueError(
                    "skill_predictor_detach_vlm must be False when skill_predictor_lora=True "
                    "so predictor gradients can reach the skill adapter."
                )
            if (
                self.skill_predictor_freeze_vlm
                and not self.skill_predictor_lora
                and not self.skill_predictor_detach_vlm
            ):
                raise ValueError(
                    "skill_predictor_detach_vlm=False requires skill_predictor_lora=True; "
                    "set skill_predictor_freeze_vlm=False for full VLM co-training."
                )
            if not self.skill_predictor_freeze_vlm and self.skill_predictor_detach_vlm:
                raise ValueError(
                    "skill_predictor_detach_vlm must be False when the complete VLM is co-trained."
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
            if self.skill_predictor_vlm_lr_scale <= 0.0:
                raise ValueError("skill_predictor_vlm_lr_scale must be positive.")
            if self.skill_predictor_focus_uv_loss_weight < 0.0:
                raise ValueError(
                    "skill_predictor_focus_uv_loss_weight must be non-negative."
                )
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
