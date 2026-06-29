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

    # ── Connector (Stage-1 future-conditioning module → VAE latent z; see connector.py) ───────────
    # A Perceiver pooler with its OWN frozen DINO encodes the skill's END frame (3rd + wrist) + END
    # state into a small VAE latent z, modulated by the GT skill (z_q) via AdaLN. z is fed to the
    # action expert as prefix token(s) — supplying the within-skill motion detail that (skill,
    # current obs) underdetermine (a skill is a coarse cluster of 100+ motions).
    use_connector: bool = True
    """Build + use the connector. False → plain Stage-1 action expert (no z). In the STAGED schedule
    it is forced OFF during phase 1-1 and ON in 1-2 (the z prefix token is added then — Approach-2)."""
    connector_dino_model_path: str = "/data2/dohyeon/SBD/models/dinov3-vitb16"
    """The connector's OWN DINO (independent instance, ALWAYS frozen → OOD-robust goal features).
    May differ from the expert-vision dino_model_path. (vitl16 recommended once downloaded.)"""
    connector_dino_image_size: int = 224
    connector_width: int = 768            # Perceiver hidden dim
    connector_depth: int = 4              # number of Perceiver (cross+self+FFN) blocks
    connector_n_heads: int = 8
    connector_n_latents: int = 4          # L learned latent queries = number of z prefix tokens
    connector_z_dim: int = 64             # per-latent VAE dim (the transfer bottleneck)
    connector_free_bits: float = 0.1      # free-bits λ (nats/dim): reserves capacity, prevents collapse
    connector_kl_weight: float = 1e-3     # β on the (free-bits) KL term added to the objective
    connector_z_consistency_weight: float = 0.0
    """Method-2 appearance-invariance: weight on ‖z − z(aug(end))‖² (stopgrad teacher). 0 = OFF — hooks
    present, concrete randomization deferred."""
    z_ablation_every: int = 0
    """Diagnostic: every N training steps (when the connector is active) run an extra no-grad forward
    with z disabled and log loss_zablate + z_gain (= zablate − plain, how much z helps). 0 = OFF.
    The key probe for 'is z actually used?' across the 3 experiments."""

    # ── Stage-1 experiment design. Each TRAINING RUN is a single fixed phase. "staged" is TWO runs
    # orchestrated by the sbatch (1-1 then 1-2, SEPARATE folders, 1-2 warm-starts from a chosen 1-1
    # checkpoint) — see stage1_train_config.py. The policy only sees a per-phase config. ──
    loss_mode: str = "plain"
    """"plain"          → fixed weighting (action_weighting below), no per-batch module gating.
    "weighted_gated"  → per-batch 50/50 module freeze (connector↔early-weighted / expert-vision↔
                        late-weighted). Needs use_connector; incompatible with freeze_expert_vision."""
    action_weighting: str = "plain"
    """Action-loss weighting for a PLAIN run: 'plain' (uniform) | 'early' (skill-START emphasized) |
    'late' (skill-END emphasized). Staged sets it per phase (1-1: late, 1-2: plain)."""
    gate_prob: float = 0.5
    """weighted_gated: P(connector trains / expert-vision frozen / early-weighted) per batch."""
    freeze_expert_vision: bool = False
    """Statically freeze the expert-vision (cond-encoder + image_proj + its DINO/SigLIP) for the whole
    run. Used by staged phase 1-2 (vision learned in 1-1, frozen while the connector trains)."""
    boundary_mode: str = "hold"
    """How chunk steps that spill PAST the current skill's end are supervised:
      "hold"      → STOP+HOLD target (arm Δ=0, gripper held) → each skill terminates cleanly (modularity).
      "keep_demo" → keep the demo's real actions across the boundary (stage2-style; goal z guides the tail).
    The STAGED schedule FORCES 'hold' in phase 1-1 (no connector goal yet); 1-2 / joint use this toggle."""
    train_terminator: bool = False
    """Co-train the isolated FSQ terminator (skill-end timing) alongside Stage-1 — gradient-disjoint
    from the main model (mirrors stage2/FT), warm-started from fsq_path. Lets a checkpoint be evaled
    (skill transitions). Always-on (its own optimizer group); not gated by schedule/loss_mode."""
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

    # ── Action loss = flow MSE over the chunk. Two forms (driven by the experiment mode): PLAIN (uniform)
    #    or PROGRESS-WEIGHTED per-step (early- or late-emphasized). Boundary tail handled by boundary_mode
    #    (HOLD / keep_demo). No cumulative-position term, no action_loss/action_weight toggles (removed). ──
    skill_end_loss_weight: float = 1.0
    """R = strength of the per-step PROGRESS weighting on the action loss: weight = 1 + (R-1)·base, where
    base = within-skill progress (LATE-emphasis) or 1-progress (EARLY-emphasis). R=1 → uniform (≡ plain).
    Direction (early/late) and whether weighting is on come from the experiment mode (joint-gated batches /
    staged phase 1-1); R only sets how strong the emphasis is."""

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
        if self.loss_mode not in ("plain", "weighted_gated"):
            raise ValueError(f"loss_mode must be 'plain' or 'weighted_gated' (got {self.loss_mode!r}).")
        if self.action_weighting not in ("plain", "early", "late"):
            raise ValueError(f"action_weighting must be 'plain'|'early'|'late' (got {self.action_weighting!r}).")
        if self.boundary_mode not in ("hold", "keep_demo"):
            raise ValueError(f"boundary_mode must be 'hold' or 'keep_demo' (got {self.boundary_mode!r}).")
        if self.loss_mode == "weighted_gated" and not self.use_connector:
            raise ValueError("loss_mode='weighted_gated' needs use_connector=True (it gates the connector).")
        if self.loss_mode == "weighted_gated" and self.freeze_expert_vision:
            raise ValueError("loss_mode='weighted_gated' is incompatible with freeze_expert_vision.")
        if self.train_terminator and not self.fsq_path:
            raise ValueError("train_terminator=True needs fsq_path (the FSQ checkpoint to warm-start).")
        if self.train_terminator and not self.skill_decoder_dino_tokens_path:
            raise ValueError(
                "train_terminator=True needs skill_decoder_dino_tokens_path (FSQ-grid current-frame DINO "
                "tokens, precomputed at build_data; attached by SkillVLADinoTokenDataset)."
            )
