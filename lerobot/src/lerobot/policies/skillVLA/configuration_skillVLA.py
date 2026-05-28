from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("skill_vla")
@dataclass
class SkillVLAConfig(PI05Config):
    """SkillVLA with VLM, action expert, skill predictor, and FSQ end decoder.

    The action expert denoises actions from noise. The FSQ decoder is used
    only for end-signal supervision/control.
    """

    model_type: str = "skill_vla"

    skill_latent_dim: int = 3
    skill_fsq_levels: list[int] = field(default_factory=lambda: [5, 5, 5])
    skill_predictor_hidden_dim: int = 512
    skill_predictor_num_heads: int = 8
    skill_predictor_num_layers: int = 2
    skill_predictor_num_query_tokens: int = 4
    skill_predictor_dropout: float = 0.0
    skill_predictor_num_embeddings: int = 125   # FSQ code count; inferred from skill_fsq_levels/checkpoint when possible.
    skill_predictor_loss_weight: float = 1.0

    skill_boundary_random_p: int = 0
    """Maximum simulated end-signal timing error in steps. 0 disables boundary randomization."""

    # Path to pretrained FSQ checkpoint (.pt file).
    vae_decoder_path: str | None = None
    skill_decoder_image_model_name: str | None = None
    """Override the DINO path/repo stored inside the FSQ checkpoint."""
    skill_decoder_loss_weight: float = 1.0
    skill_decoder_delta_loss_weight: float = 10.0
    """Weight for FSQ decoder action/chunk reconstruction loss inside skill_decoder_loss."""
    skill_decoder_end_loss_weight: float = 1.0
    """Weight for FSQ decoder end-signal BCE inside skill_decoder_loss."""
    skill_decoder_end_pos_weight: float = 1.0
    skill_decoder_end_threshold: float = 0.5
    skill_decoder_state_indices: list[int] | None = None
    """Raw observation.state indices fed to the FSQ decoder. None uses the first FSQ state_dim dims."""
    skill_decoder_dino_tokens_path: str | None = None
    """FSQ decoder DINO token npz. Provides per-frame (CLS + pooled patches) tokens as skill_decoder_image."""
    skill_decoder_dino_cache_path: str | None = None
    """Optional .npy cache path for mmap-friendly FSQ decoder DINO tokens."""
    skill_decoder_dino_output_key: str = "skill_decoder_image"
    """Batch key written by the SkillVLA DINO-token dataset wrapper."""
    skill_decoder_dino_build_cache: bool = True
    """Build the .npy mmap cache from the .npz token file when missing."""
    freeze_vae_decoder: bool = False
    freeze_patch_flag_predictor: bool = False
    """Freeze PatchFlagPredictor weights and detach prefix_embs before flag prediction,
    so the skill-decoder loss does not backprop into VLM via the flag path.
    Orthogonal to detach_action_prefix_grad (action flow path)."""
    inference_skill_max_length: int = 200
    skill_decoder_prior_noise_ratio: float = 0.0
    """Mix ratio r for action-expert start source: (1-r)*normalized FSQ prior + r*Gaussian noise."""

    # If True, block action expert attention to language tokens; False keeps full prefix attention.
    block_lang_to_action: bool = True
    detach_action_prefix_grad: bool = False
    """If True, flow/action loss uses prefix attention but does not update VLM/prefix parameters."""

    # Eval-only oracle mode: bypass the skill predictor and feed dataset label
    # skill tokens to the action expert / FSQ end decoder.
    use_label_skill_tokens_eval: bool = False
    label_skill_dataset_dir: str | None = None
    label_skill_episode_offset: int = 0
    compare_label_skill_tokens_eval: bool = False
