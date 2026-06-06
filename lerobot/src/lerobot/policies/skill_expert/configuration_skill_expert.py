from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("skill_expert")
@dataclass
class SkillExpertConfig(PI05Config):
    """Stage-1 standalone action expert (no VLM, no language).

    Predicts an action chunk by flow matching from the current 3rd-person + wrist images
    (each encoded by a trainable DINOv3, shared weights), the robot state, and the GT FSQ
    skill code. All tokens self-attend; only the action-token hidden states are decoded
    into actions. The VLM is added in Stage 2 (the `skill_vla` policy), which can be
    initialized from a Stage-1 `skill_expert` checkpoint.

    Inherits the PI05 action-expert / flow-matching knobs (chunk_size, max_action_dim,
    action_expert_variant, time sampling, num_inference_steps, normalization, ...). The
    PaliGemma/SigLIP and language settings are inherited but unused here.
    """

    model_type: str = "skill_expert"

    # ── Vision encoder: DINOv3, trainable, shared across the two cameras ──
    dino_model_path: str = "/data2/dohyeon/SBD/models/dinov3-vits16"
    dino_image_size: int = 224
    """Square size the camera images are resized to before DINOv3 (patch 16 → 14×14)."""
    freeze_dino: bool = False
    dino_lr: float | None = None
    """Separate LR for the DINO encoder. None → use optimizer_lr (same as the rest)."""

    # ── Skill conditioning ──
    skill_vocab_size: int = 125
    """Number of FSQ skill codes (= prod(fsq_levels)). The skill is a discrete code fed
    through an nn.Embedding lookup, exactly like a language token."""
    skill_fsq_levels: list[int] = field(default_factory=lambda: [5, 5, 5])
    """FSQ levels per dim (for the eval FSQ-cube visualization). prod = skill_vocab_size."""

    # ── Eval-only (oracle closed-loop sim). Ignored during training. ──
    fsq_path: str | None = None
    """Frozen FSQ checkpoint ({run_dir}/FSQ.pt): provides the terminator (skill end signal)
    + code->z_q mapping for eval. Unused in training (skill code comes from the dataset)."""
    skill_label_dataset_dir: str | None = None
    """skillvla dataset dir whose skill_sequence columns give the GT skill sequence per task
    (matched to the env task by language) for the oracle eval."""
    skill_end_mode: str = "termination"
    """Which FSQ terminator signal ends the current skill (oracle eval):
    "termination" → termination probability >= skill_end_threshold (e.g. 0.5);
    "progress"    → predicted progress >= skill_end_threshold (e.g. 0.9)."""
    skill_end_threshold: float = 0.5
    """Threshold on the signal selected by skill_end_mode, above which the skill is finished."""
    inference_skill_max_length: int = 200
    """Force-advance the skill after this many steps even if the terminator never fires (0 = off)."""
