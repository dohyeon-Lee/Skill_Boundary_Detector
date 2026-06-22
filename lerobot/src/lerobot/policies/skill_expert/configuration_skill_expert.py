from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("skill_expert")
@dataclass
class SkillExpertConfig(PI05Config):
    """Stage-1 standalone action expert (no VLM, no language).

    Predicts an action chunk by flow matching from the current 3rd-person + wrist images
    (each encoded by a trainable DINOv3, shared weights), the robot state, and the GT FSQ
    skill code. A fresh cond-encoder (own Gemma) encodes the scene — images + the per-dim
    DISCRETIZED state — and the action expert reads it via PI05-style block attention (action
    sees cond + action; cond ⊥ action), with skill (z_q) + progress prepended to the action
    stream. The VLM is added in Stage 2 (the `skill_vla` policy), which can be initialized from
    a Stage-1 `skill_expert` checkpoint.

    Inherits the PI05 action-expert / flow-matching knobs (chunk_size, max_action_dim,
    action_expert_variant, time sampling, num_inference_steps, normalization, ...). The
    PaliGemma/SigLIP and language settings are inherited but unused here.
    """

    model_type: str = "skill_expert"

    # ── Conditioning architecture (joint: cond-encoder ⊥ action expert) ──
    # A SEPARATE cond-encoder (own Gemma) encodes the scene (images + per-dim discretized state); the
    # action expert receives [skill, progress, action] and reads the cond stream via PI05-style joint
    # block attention (action sees cond + action; cond ⊥ action). The action expert warm-starts from
    # pi05; the cond-encoder is fresh. Skill (z_q) + progress are prepended to the ACTION stream.
    cond_encoder_variant: str | None = None
    """Gemma variant for the cond-encoder. None → same as action_expert_variant."""

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

    # ── State conditioning ──
    state_cond_mode: str = "state_skill"
    """What rides the ACTION expert's flow-time AdaRMS (DiT-style global conditioning, un-droppable by
    attention). The cond-encoder is image-ONLY and plain-RMSNorm in both modes; state never rides the
    (image-dominated) cond stream — the input_probe diagnostic showed a state token buried among ~400
    image tokens is starved (Δ≈0), so state is always summed into the expert AdaRMS instead. The two
    modes differ only in whether SKILL (and progress) also go global vs stay as attended prefix tokens:
      "state"       → AdaRMS conditioning = time + state_proj(state). Skill (z_q) + progress are PREFIX
                      tokens on the action stream (pi0 prefix⊥action: read by the action tokens, do not
                      attend back). Image stays the dominant motion driver; skill is a lighter, attended
                      signal — leaves room for Stage-2 language to modulate the motion.
      "state_skill" → AdaRMS conditioning = time + state_proj(state) + skill_proj(z_q) + progress_proj(prog)
                      (each its own projection, summed — DiT ⊕ pattern). NO prefix tokens at all; skill is
                      a strong global signal (heaviest skill influence).
    progress is included in the AdaRMS / prefix only when use_progress_token=True. state_proj / skill_proj /
    progress_proj are allocated in both modes (only the destination differs), so "state" and "state_skill"
    checkpoints stay structurally comparable."""

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
    the action expert conditions on the skill code only (progress ablation). The skill (+progress)
    tokens are prepended to the ACTION stream (2→1 tokens when off). progress_proj stays allocated
    either way so progress-on/off checkpoints stay mutually loadable; only whether its output is used
    changes."""

    # ── Loss. Boundary handling: at the skill end (k>skill_de) and episode-end pad, the action TARGET is a
    #    HOLD (arm deltas→0, gripper→last valid value) and supervised — NOT masked out. These add the optional
    #    cum term + the action_loss/action_weight toggles below. ──
    cumulative_pos_loss_weight: float = 0.0
    """λ for the optional cumulative-POSITION loss (added to the per-delta flow loss). Actions are
    RELATIVE (delta), so the endpoint pose = the SUM of the chunk's deltas; matching only the last delta
    doesn't fix the landing position. This term integrates the implied per-step action error over the
    chunk (cumsum, on the ARM dims only — the gripper is absolute, excluded) and penalizes the K running
    positions, so the whole trajectory + endpoint match. 0 = OFF (objective = the action term only)."""
    skill_end_loss_weight: float = 1.0
    """R for the end weighting: weight = 1 + (R-1)·progress, where progress is the within-skill position
    (0 at skill start → 1 at skill end). R=1 → uniform; R>1 → skill END positions (the handoff) count more.
    Applied IDENTICALLY to the action loss (when action_weight=True, per-SAMPLE by the chunk endpoint's
    progress) and the cum term (per-STEP in cum_loss="all", per-SAMPLE in "endpoint")."""

    # ── Loss composition — independent toggles (objective = [action term] + λ·[cum term]) ──────────
    action_loss: bool = True
    """Include the per-step flow (action) MSE in the OPTIMIZED objective. False → the objective is the cum
    term ONLY (requires cumulative_pos_loss_weight>0). The plain unweighted action MSE is STILL logged to
    wandb `loss` either way (comparison-only, never backpropped). Run-name: ac (on) / ac_x (off) / ac_w (below)."""
    action_weight: bool = False
    """Weight the action loss PER-SAMPLE (per-chunk) by the endpoint progress sw=1+(R-1)·prog_end (same sw as
    the endpoint cum term) — chunks ending nearer the skill end count more. Only meaningful when
    action_loss=True. The weighted value is logged to wandb `loss_weighted` (a SEPARATE panel); wandb `loss`
    stays the plain unweighted comparison value. Run-name tag: ac_w."""
    cum_loss: str = "all"
    """Cumulative-position term aggregation (ALWAYS R-weighted; on/off via cumulative_pos_loss_weight=λ):
      "all"      → penalize EVERY within-skill running position (integrated trajectory), end-weighted PER-STEP.
      "endpoint" → penalize ONLY the running position at the LAST valid step (chunk end, capped to skill end /
                   padding), R weighting PER-SAMPLE. Run-name: cum_all / cum_ep (only when λ>0)."""

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

    def __post_init__(self):
        super().__post_init__()
        if self.state_cond_mode not in ("state", "state_skill"):
            raise ValueError(
                f"state_cond_mode must be 'state' or 'state_skill' (got {self.state_cond_mode!r})."
            )
        if self.cum_loss not in ("all", "endpoint"):
            raise ValueError(f"cum_loss must be 'all' or 'endpoint' (got {self.cum_loss!r}).")
        if not self.action_loss and float(self.cumulative_pos_loss_weight) <= 0.0:
            raise ValueError(
                "action_loss=False needs cumulative_pos_loss_weight>0 (otherwise the objective is empty)."
            )
