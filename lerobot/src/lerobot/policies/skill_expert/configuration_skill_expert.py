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

    # ── FSQ expert adaptation ──
    # ``lora_rank`` / ``lora_alpha`` / ``lora_dropout`` / ``lora_targets`` are inherited from PI05Config.
    # ``lora_targets`` accepts q/k/v/o attention aliases, gate/up/down (or ``mlp``), and Stage-1-only
    # ``action_in`` / ``action_out`` flow-head aliases. This uses the same named-adapter implementation as
    # SkillVLA Stage 2, but only on the frozen FSQ action side when LoRA is enabled. The condition side
    # remains a normal trainable scene encoder on A batches.
    lora_expert: bool = False
    """True → inject the named ``expert`` LoRA adapter and freeze every FSQ action-side base parameter.
    False → inject no adapter and fine-tune the complete FSQ action side (Gemma plus
    action/time/state/skill projections) together with the image condition side."""
    lora_lr_scale: float = 1.0
    """LR multiplier for LoRA parameters relative to optimizer_lr."""
    image_free_lora_prob: float = 0.0
    """Probability of a B batch during training. B skips image/cond entirely and anchors the current
    image-free action velocity to frozen FSQ: LoRA uses its adapter-disabled base; full fine-tuning uses a
    non-persistent copy of the original FSQ action tensors. A = 1 - this probability uses normal
    image-conditioned action loss."""
    image_free_lora_anchor_weight: float = 1.0
    """Multiplier on the B adapter-to-base velocity anchor loss."""
    eval_image_free_expert: bool = False
    """Eval-only B route: keep the expert LoRA active while removing every image/condition-stream token.
    This is set by the Stage-1 oracle evaluator and is never used during training."""

    # ── Loss = action MSE (flow matching). Boundary handling: at the skill end (k>skill_de) and episode-end
    #    pad, the action TARGET is a HOLD (arm deltas→0, gripper→last valid value) and supervised — NOT masked.
    #    action_weight selects plain vs per-sample skill-progress-weighted MSE. ──
    skill_start_loss_weight: float = 1.0
    """Action-loss weight at skill progress 0 (only used when action_weight=True)."""
    skill_end_loss_weight: float = 1.0
    """Action-loss weight at skill progress 1 (only used when action_weight=True). The per-chunk weight is
    linearly interpolated between skill_start_loss_weight and this value using the chunk ENDPOINT progress."""
    action_weight: bool = False
    """Weight action MSE PER-SAMPLE by start+(end-start)·prog_end. The K steps within a chunk share one
    weight. False gives plain action MSE. B image-free anchor batches never use this weighting."""

    # ── Co-trained FSQ terminator (skill-end timing for eval; gradient-disjoint from the main model) ──
    train_terminator: bool = False
    """Co-train the isolated FSQ terminator (skill-end timing) alongside Stage-1 — gradient-disjoint
    from the main model (mirrors stage2/FT), warm-started from fsq_path. Lets a checkpoint be evaled
    (skill transitions). Always-on (its own optimizer group)."""
    terminator_freeze_vision_encoder: bool | None = None
    """Freeze the co-trained terminator's DINO/SigLIP encoder. None inherits the FSQ checkpoint setting;
    false fine-tunes it in the terminator optimizer group at terminator_lr_scale × optimizer_lr."""
    terminator_end_target_sigma: float = 1.0
    """Termination target = Gaussian bump exp(-de²/2σ²) peaking at the skill end (σ>0); σ≤0 → hard (de==0)."""
    terminator_end_pos_weight: float = 1.0
    """BCE pos_weight for the termination head (skill-end frames are rare positives)."""
    terminator_lr_scale: float = 1.0
    """LR scale (× optimizer_lr) for the co-trained terminator's own param group."""
    # (skill_decoder_dino_* precompute 토큰 필드 은퇴 — terminator는 배치의 observation.images.*를
    #  ONLINE DINO로 직접 토큰화; terminator_dino_model_path가 DINO 경로 오버라이드를 겸함.)

    # ── Eval-only (oracle closed-loop sim). Ignored during training. ──
    fsq_path: str | None = None
    """v3 FSQ checkpoint ({run_dir}/FSQ.pt). It always warm-starts the exact action expert.
    A terminator_arch=cond checkpoint additionally warm-starts Stage-1's vision tower,
    image projection, and condition Gemma; terminator_arch=small does not. The checkpoint
    also supplies the isolated terminator for co-training/eval. The trajectory encoder is
    never instantiated here (skill codes come from the dataset)."""
    skill_label_dataset_dir: str | None = None
    """skillvla dataset dir whose skill_sequence columns give the GT skill sequence per task
    (matched to the env task by language) for the oracle eval."""
    terminator_dino_model_path: str | None = None
    """Optional local override for the FSQ terminator's DINO model path. None → the FSQ
    checkpoint's own dino_model_path (auto-resolved to this repo's models/ if absent).
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
        if self.lora_expert and self.lora_rank <= 0:
            raise ValueError(f"lora_expert=True needs lora_rank > 0 (got {self.lora_rank}).")
        if self.lora_expert and self.lora_alpha <= 0:
            raise ValueError(f"lora_expert=True needs lora_alpha > 0 (got {self.lora_alpha}).")
        if self.lora_lr_scale <= 0.0:
            raise ValueError(f"lora_lr_scale must be > 0 (got {self.lora_lr_scale}).")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError(f"lora_dropout must be in [0, 1), got {self.lora_dropout}.")
        if not 0.0 <= self.image_free_lora_prob < 1.0:
            raise ValueError(
                "image_free_lora_prob must be in [0, 1); keep some A batches to train the image condition "
                f"side (got {self.image_free_lora_prob})."
            )
        if self.image_free_lora_prob > 0.0 and not self.lora_expert and not self.fsq_path:
            raise ValueError("Full-FT image-free B batches need fsq_path for their frozen FSQ teacher.")
        if self.image_free_lora_anchor_weight <= 0.0:
            raise ValueError(
                "image_free_lora_anchor_weight must be > 0 "
                f"(got {self.image_free_lora_anchor_weight})."
            )
        if self.skill_start_loss_weight <= 0.0 or self.skill_end_loss_weight <= 0.0:
            raise ValueError(
                "skill_start_loss_weight and skill_end_loss_weight must both be > 0 "
                f"(got {self.skill_start_loss_weight} and {self.skill_end_loss_weight})."
            )
