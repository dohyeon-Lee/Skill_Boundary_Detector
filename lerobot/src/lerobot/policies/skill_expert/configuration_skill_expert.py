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
    sees cond + action; cond ⊥ action), with the skill (z_q) prepended to the action
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
    dino_lr: float | None = None
    """Separate LR for the DINO encoder. None → use optimizer_lr (same as the rest)."""
    # SigLIP backbone (vision_backbone="siglip"); weights come from the pi05 pretrained_path.
    siglip_image_size: int = 224
    """Square size images are resized to before SigLIP (patch 14 → 16×16 = 256 tokens at 224)."""
    siglip_lr: float | None = None
    """Separate LR for the SigLIP encoder when unfrozen. None → use optimizer_lr."""
    # Freeze the SELECTED vision backbone (one flag for whichever vision_backbone is active).
    freeze_vision_encoder: bool = False
    """Statically freeze the active vision backbone. dino → False trains its own DINOv3. siglip → often
    True (warm-started from pi05's robot-adapted vision_tower; only image_proj trains). Replaces the old
    per-backbone freeze_dino / freeze_siglip."""

    # ── State conditioning ──
    state_cond_mode: str = "state_skill"
    """What rides the ACTION expert's flow-time AdaRMS (DiT-style global conditioning, un-droppable by
    attention). The cond-encoder is image-ONLY and plain-RMSNorm in both modes; state never rides the
    (image-dominated) cond stream — the input_probe diagnostic showed a state token buried among ~400
    image tokens is starved (Δ≈0), so state is always summed into the expert AdaRMS instead. The two
    modes differ only in whether SKILL also goes global vs stays as an attended prefix token:
      "state"       → AdaRMS conditioning = time + state_proj(state). Skill (z_q) is a PREFIX token on the
                      action stream (pi0 prefix⊥action: read by the action tokens, does not attend back).
                      Image stays the dominant motion driver; skill is a lighter, attended signal — leaves
                      room for Stage-2 language to modulate the motion.
      "state_skill" → AdaRMS conditioning = time + state_proj(state) + skill_proj(z_q) (each its own
                      projection, summed — DiT ⊕ pattern). NO prefix tokens at all; skill is a strong global
                      signal (heaviest skill influence).
    state_proj / skill_proj are allocated in both modes (only the destination differs), so "state" and
    "state_skill" checkpoints stay structurally comparable."""

    # ── Skill conditioning ──
    skill_vocab_size: int = 125
    """Number of FSQ skill codes (= prod(skill_fsq_levels)); bounds the codes in the dataset."""
    skill_fsq_levels: list[int] = field(default_factory=lambda: [5, 5, 5])
    """FSQ levels per dim. The flat dataset code is mapped back to its FSQ grid coordinate z_q
    (little-endian strides — the codebook's own convention, the same value the FSQ decoder
    consumes), normalized per dim to [-1, 1], and fed through a Linear(D → width) as ONE skill
    token, constant within a skill — neighboring codes stay neighboring. (The skill-progress token
    was removed — the action expert conditions on the skill code only.)"""

    # ── Loss = action MSE (flow matching). Boundary handling: at the skill end (k>skill_de) and episode-end
    #    pad, the action TARGET is a HOLD (arm deltas→0, gripper→last valid value) and supervised — NOT masked.
    #    action_weight selects plain vs per-sample end-weighted MSE. ──
    skill_end_loss_weight: float = 1.0
    """R for the end weighting (only used when action_weight=True): sw = 1 + (R-1)·progress, where progress is
    the within-skill position of the chunk's ENDPOINT (0 at skill start → 1 at skill end). R=1 → uniform; R>1
    → chunks landing near the skill END (the handoff) count up to R×."""
    action_weight: bool = False
    """Weight the action MSE PER-SAMPLE (per-chunk) by sw=1+(R-1)·prog_end — chunks ending nearer the skill
    end count more; UNIFORM within a chunk (the K steps share one weight). False → plain (uniform) action MSE.
    The weighted value is logged to wandb `loss_weighted` (SEPARATE panel); wandb `loss` stays the plain
    unweighted comparison value. Run-name tag: ac_w (weighted) / ac (plain)."""

    # ── Co-trained FSQ terminator (skill-end timing for eval; gradient-disjoint from the main model) ──
    train_terminator: bool = False
    """Co-train the isolated FSQ terminator (skill-end timing) alongside Stage-1 — gradient-disjoint
    from the main model (mirrors stage2/FT), warm-started from fsq_path. Lets a checkpoint be evaled
    (skill transitions). Always-on (its own optimizer group)."""
    terminator_end_target_sigma: float = 1.0
    """Termination target = Gaussian bump exp(-de²/2σ²) peaking at the skill end (σ>0); σ≤0 → hard (de==0)."""
    terminator_end_pos_weight: float = 1.0
    """BCE pos_weight for the termination head (skill-end frames are rare positives)."""
    terminator_lr_scale: float = 1.0
    """LR scale (× optimizer_lr) for the co-trained terminator's own param group."""
    # Current-frame FSQ-grid DINO tokens for the terminator (precomputed at build_data; attached by
    # SkillVLADinoTokenDataset — the SAME generic wrapper Stage-2 uses). Required when train_terminator.
    skill_decoder_dino_tokens_path: str | None = None
    skill_decoder_dino_output_key: str = "skill_decoder_dino"
    skill_decoder_dino_cache_path: str | None = None
    skill_decoder_dino_build_cache: bool = True
    # Wrist-camera FSQ-grid DINO tokens — ONLY needed when the FSQ terminator was trained with
    # terminator_use_wrist=True (build_data dino_wrist:true → dino_wrist.npz). Leave blank for
    # 3rd-only ("wow") FSQs. Attached by a SECOND SkillVLADinoTokenDataset wrapper.
    skill_decoder_dino_wrist_tokens_path: str | None = None
    skill_decoder_dino_wrist_output_key: str = "skill_decoder_dino_wrist"
    skill_decoder_dino_wrist_cache_path: str | None = None

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
    """Which FSQ terminator signal(s) end the current skill (used when skill_advance_mode=terminator):
    "termination" → termination probability >= skill_end_threshold (e.g. 0.5);
    "progress"    → predicted progress >= skill_end_progress_threshold (e.g. 0.9);
    "or"          → EITHER crosses (term >= skill_end_threshold OR progress >= skill_end_progress_threshold)
                    — robust default: the two heads peak at different steps, so requiring only one to fire
                    avoids over-running when one head is weak/misaligned;
    "and"         → BOTH cross at the same step (stricter; rarely fires if the heads disagree in timing)."""
    skill_end_threshold: float = 0.5
    """Threshold on the TERMINATION-probability signal, above which the skill is finished (termination/or/and)."""
    skill_end_progress_threshold: float = 0.9
    """Threshold on the predicted-PROGRESS signal, above which the skill is finished (progress/or/and).
    Kept separate from skill_end_threshold since the two heads live on different scales."""
    inference_skill_max_length: int = 200
    """Force-advance the skill after this many steps even if the terminator never fires (0 = off)."""
    eval_use_trained_terminator: bool = True
    """At eval, use the CO-TRAINED terminator saved IN the checkpoint (model.fsq_term_train.*, fine-tuned on
    this dataset via train_terminator) instead of the raw FSQ.pt terminator — overriding only the terminator
    submodules' weights (the DINO backbone still comes from terminator_dino_model_path). Falls back to FSQ.pt
    if the checkpoint carries no co-trained terminator, or when a terminator_path override is given."""
    eval_init_states_path: str | None = None
    """eval_init_states.npz (built by stage1_eval/oracle_matching/) mapping each dataset episode →
    its MuJoCo init_state + scene_file. Makes the closed-loop eval EPISODE-EXACT: every rollout resets
    the sim to a specific dataset episode's scene and is fed THAT episode's GT skill sequence (vs the old
    task-level eval that ran LIBERO's default init states matched only by language). Required by run_eval."""

    def __post_init__(self):
        super().__post_init__()
        if self.state_cond_mode not in ("state", "state_skill"):
            raise ValueError(
                f"state_cond_mode must be 'state' or 'state_skill' (got {self.state_cond_mode!r})."
            )
        if self.train_terminator and not self.fsq_path:
            raise ValueError("train_terminator=True needs fsq_path (the FSQ checkpoint to warm-start).")
        if self.train_terminator and not self.skill_decoder_dino_tokens_path:
            raise ValueError(
                "train_terminator=True needs skill_decoder_dino_tokens_path (FSQ-grid current-frame DINO "
                "tokens, precomputed at build_data; attached by SkillVLADinoTokenDataset)."
            )
