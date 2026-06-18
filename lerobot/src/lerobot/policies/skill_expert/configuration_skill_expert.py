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

    # ── Conditioning architecture ──
    expert_arch: str = "fused"
    """How the action expert consumes the conditioning (image/state/skill):
    "fused" → ONE Gemma over [cond tokens, action tokens] with BLOCK-CAUSAL attention: cond is one
              bidirectional block that does NOT attend the noisy action; action attends cond + action
              (DP/pi0-standard obs→action one-directional structure; expert weights also encode cond);
    "joint" → a SEPARATE cond-encoder (own Gemma, same config) encodes the conditioning, and the
              action expert receives ONLY action tokens, reading the cond stream via PI05-style
              joint block attention (block mask: action sees cond+action, cond⊥action). The action
              expert still warm-starts from pi05; the cond-encoder is fresh. The run/output folder
              reflects this so fused/joint checkpoints don't collide."""
    cond_encoder_variant: str | None = None
    """Gemma variant for the joint cond-encoder. None → same as action_expert_variant."""
    skill_inject: str = "action_prefix"
    """[joint only] Where the skill (z_q) + progress tokens enter (fused ALWAYS uses cond tokens):
    "action_prefix" → prepended to the ACTION stream (~1:K among action tokens; current default);
    "cond_token"    → appended to the COND-encoder tokens (~1:400, diluted among image tokens; the
                      "prev"/baseline recipe). Both feed the same Linear(D→width) on z_q (vector input)."""

    # ── Vision encoder (shared across the two cameras) ──
    vision_backbone: str = "dino"
    """Which image encoder feeds the action expert:
    "dino"   → trainable DINOv3 (own weights, ImageNet norm);
    "siglip" → PaliGemma/SigLIP vision tower, warm-started from the pi05 checkpoint's
               vision_tower (robot-adapted prior). Its params are SEPARATE from the Stage-2 VLM."""
    # DINO backbone (vision_backbone="dino")
    dino_model_path: str = "/data2/dohyeon/SBD/models/dinov3-vits16"
    dino_image_size: int = 224
    """Square size the camera images are resized to before DINOv3 (patch 16 → 14×14)."""
    freeze_dino: bool = False
    dino_lr: float | None = None
    """Separate LR for the DINO encoder. None → use optimizer_lr (same as the rest)."""
    # SigLIP backbone (vision_backbone="siglip"); weights come from the pi05 pretrained_path.
    siglip_image_size: int = 224
    """Square size images are resized to before SigLIP (patch 14 → 16×16 = 256 tokens at 224)."""
    freeze_siglip: bool = True
    """Freeze the SigLIP tower (default True: reuse pi05's robot-adapted features; only image_proj trains)."""
    siglip_lr: float | None = None
    """Separate LR for the SigLIP encoder when unfrozen. None → use optimizer_lr."""

    # ── Skill conditioning ──
    skill_vocab_size: int = 125
    """Number of FSQ skill codes (= prod(skill_fsq_levels)); bounds the codes in the dataset."""
    skill_fsq_levels: list[int] = field(default_factory=lambda: [5, 5, 5])
    """FSQ levels per dim. The flat dataset code is mapped back to its FSQ grid coordinate z_q
    (little-endian strides — the codebook's own convention, the same value the FSQ decoder
    consumes), normalized per dim to [-1, 1], and fed through a Linear(D → width) as ONE cond
    token, constant within a skill — neighboring codes stay neighboring. The skill progress is
    a SEPARATE cond token (raw [0, 1] → Linear(1 → width)), mirroring the FSQ decoder's
    dec_z_proj / motion_prog_proj split."""
    progress_jitter: float = 0.1
    """Train-time uniform noise (±jitter, clamped to [0, 1]) on the GT skill progress fed to the
    progress token. The GT progress is skill_ds/(skill_ds+skill_de) (0 at skill start, 1 at its
    last frame — the FSQ terminator's training target); at inference the terminator's PREDICTED
    progress is injected via batch["skill_progress"], so the jitter teaches robustness to that
    estimator's error. 0 = clean GT."""
    use_progress_token: bool = True
    """Whether the skill PROGRESS enters the model as its own token (alongside the z_q skill token).
    True (default) = current recipe (skill + progress). False = drop the progress token entirely;
    the action expert conditions on the skill code only (progress ablation). Applies wherever the
    skill goes: the action-stream prefix (joint + action_prefix → 2→1 tokens) and the cond tokens
    (fused / joint + cond_token). progress_proj stays allocated either way so progress-on/off
    checkpoints stay mutually loadable; only whether its output is used changes."""

    # ── Eval-only (oracle closed-loop sim). Ignored during training. ──
    fsq_path: str | None = None
    """Frozen FSQ checkpoint ({run_dir}/FSQ.pt): provides the terminator (skill end signal)
    + code->z_q mapping for eval. Unused in training (skill code comes from the dataset)."""
    skill_label_dataset_dir: str | None = None
    """skillvla dataset dir whose skill_sequence columns give the GT skill sequence per task
    (matched to the env task by language) for the oracle eval."""
    terminator_dino_model_path: str | None = None
    """DINO weights for the FSQ terminator's raw-image encoder at eval. None → the FSQ
    checkpoint's own image_model_name (auto-resolved to this repo's models/ if absent).
    Kept SEPARATE from dino_model_path, which is the policy's OWN vision backbone and must
    match the checkpoint being loaded — never override that one at eval."""
    skill_advance_mode: str = "terminator"
    """How the oracle eval advances through the GT skill sequence:
    "terminator" → the FSQ terminator decides each transition (skill_end_mode/threshold);
    "gt"         → advance after each skill's GT demo duration (ideal timing; isolates the
                   action expert from terminator timing errors). The terminator still runs so
                   its curves are recorded for the HTML either way."""
    skill_end_mode: str = "termination"
    """Which FSQ terminator signal ends the current skill (used when skill_advance_mode=terminator):
    "termination" → termination probability >= skill_end_threshold (e.g. 0.5);
    "progress"    → predicted progress >= skill_end_threshold (e.g. 0.9)."""
    skill_end_threshold: float = 0.5
    """Threshold on the signal selected by skill_end_mode, above which the skill is finished."""
    inference_skill_max_length: int = 200
    """Force-advance the skill after this many steps even if the terminator never fires (0 = off)."""
