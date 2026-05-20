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
    # "context": Paligemma contextual prefix hidden states. "embs": raw embed_prefix token embeddings.
    skill_predictor_prefix_source: str = "context"
    skill_predictor_loss_weight: float = 1.0

    predicted_latent_sampling: bool = False
    predicted_latent_start_step: int = 0
    predicted_latent_ramp_steps: int = 20_000
    predicted_latent_max_prob: float = 1.0
    skill_boundary_random_p: int = 0
    """Maximum simulated end-signal timing error in steps. 0 disables boundary randomization."""

    # Path to pretrained FSQ checkpoint (.pt file).
    vae_decoder_path: str | None = None
    skill_decoder_loss_weight: float = 1.0
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
    inference_skill_max_order: int = 8
    inference_skill_max_length: int = 200

    # If True, action expert cannot attend to language tokens (only image tokens).
    block_lang_to_action: bool = True

    # Eval-only oracle mode: bypass the skill predictor and feed dataset label
    # skill tokens to the action expert / FSQ end decoder.
    use_label_skill_tokens_eval: bool = False
    label_skill_dataset_dir: str | None = None
    label_skill_episode_offset: int = 0
    compare_label_skill_tokens_eval: bool = False
