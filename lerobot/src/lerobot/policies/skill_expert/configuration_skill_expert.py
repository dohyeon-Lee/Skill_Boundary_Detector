from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("skill_expert")
@dataclass
class SkillExpertConfig(PI05Config):
    """Stage-1 standalone action expert (no VLM, no language).

    Predicts an action chunk by flow matching from the current 3rd-person + wrist images
    (each encoded by a trainable DINOv3, shared weights), the robot state, and the GT FSQ
    skill code z. A fresh cond-encoder (own Gemma) encodes the scene and the action expert
    reads it via PI05-style block attention (action sees cond + action; cond ⊥ action). This
    base — vision + state + action expert + skill — is the **VSA** model.

    Stage-1 also (optionally) adds the **Oracle**: it encodes the current skill's full state
    trajectory (start→end, appearance-invariant) + skill z into ONE VAE latent token **r** that
    is fed to the action expert as a prefix token. r carries the within-skill motion detail that
    (skill, current obs) underdetermine. r is forced to be a RESIDUAL beyond z via CFG-style
    conditioning dropout: a fraction of batches drop r (replace it with a learned null token),
    so the VSA base must stand on its own and r only adds the gap. Stage-2 (the `skill_vla`
    policy) freezes the Oracle and has the VLM predict r from language + current image.

    Inherits the PI05 action-expert / flow-matching knobs (chunk_size, max_action_dim,
    action_expert_variant, time sampling, num_inference_steps, normalization, ...).
    """

    model_type: str = "skill_expert"

    # ── Conditioning architecture (cond-encoder ⊥ action expert) ──
    # A SEPARATE cond-encoder (own Gemma) encodes the scene (images); the action expert receives
    # [skill, r, action] and reads the cond stream via PI05-style block attention (action sees
    # cond + action; cond ⊥ action). The action expert warm-starts from pi05; the cond-encoder is fresh.
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
    state_cond_mode: str = "state"
    """What rides the ACTION expert's flow-time AdaRMS (DiT-style global conditioning). The cond-encoder
    is image-ONLY in both modes; state always rides the expert AdaRMS (a state token among ~400 image
    tokens is starved). The two modes differ only in whether SKILL z also goes global vs stays a prefix token:
      "state"       → AdaRMS = time + state_proj(state). Skill z is a PREFIX token on the action stream
                      (pi0 prefix⊥action). The r-slot is the next prefix token → prefix = [z, r].
      "state_skill" → AdaRMS = time + state_proj(state) + skill_proj(z). NO skill prefix token; the only
                      prefix token is the r-slot → prefix = [r]."""

    # ── Skill conditioning (z) ──
    skill_vocab_size: int = 125
    """Number of FSQ skill codes (= prod(skill_fsq_levels)); bounds the codes in the dataset."""
    skill_fsq_levels: list[int] = field(default_factory=lambda: [5, 5, 5])
    """FSQ levels per dim. The flat dataset code is mapped to its FSQ grid coordinate z_q (little-endian
    strides — the codebook's own convention), normalized per dim to [-1, 1], and fed through a Linear(D →
    width) as ONE skill token, constant within a skill."""

    # ── Oracle (Stage-1 residual module → 1-token VAE latent r; see oracle.py) ─────────────────────
    # The Oracle encodes the current skill's full state TRAJECTORY (resampled to a fixed length,
    # appearance-invariant) + skill z into a VAE latent r (n_tokens, r_dim), modulated by z via AdaLN.
    # r is fed to the action expert as a prefix token, supplying the within-skill motion detail that
    # (skill, current obs) underdetermine. Forced to be a RESIDUAL via CFG-style dropout (oracle_dropout_p).
    use_oracle: bool = False
    """Build + use the Oracle. False → plain VSA (the r-slot is always the learned null token = no
    residual). The STAGED schedule sets this False in 1-1 and True in 1-2; single-stage sets it True."""
    oracle_resample_n: int = 30
    """Resample the skill's state trajectory to this many control points — MATCH the FSQ n_control so the
    Oracle sees the trajectory at the same resolution the skill code z was derived from."""
    oracle_spline_degree: int = 3
    """Pose-dim spline degree for the resample (gripper dims are always degree 1 — near-step, no overshoot)
    — MATCH the FSQ spline_degree. The trajectory is zero-grounded (pose relative to start, gripper absolute),
    exactly as the FSQ encoder (mirrors FSQ.py spline_encode)."""
    oracle_width: int = 512               # Oracle hidden dim
    oracle_depth: int = 3                 # number of Perceiver (cross+self+FFN) blocks
    oracle_n_heads: int = 8
    oracle_n_tokens: int = 1              # number of r prefix tokens (1 = tightest stage-2 target)
    oracle_r_dim: int = 16                # per-token VAE dim (the transfer bottleneck; SWEEP knob)
    oracle_free_bits: float = 0.1         # free-bits λ (nats/dim): reserves capacity, prevents collapse
    oracle_kl_weight: float = 1e-3        # β on the (free-bits) KL term (SWEEP knob)
    oracle_dropout_p: float = 0.5
    """CFG-style conditioning dropout: P(drop r → learned null this batch). Forces the VSA base to
    stand alone (graceful degradation for stage-2) and r to carry only the residual. (SWEEP knob.)"""
    r_ablation_every: int = 0
    """Diagnostic: every N steps (Oracle on) run an extra no-grad forward with r dropped → log
    loss_rablate + r_gain (= rablate − with-r). The key probe for 'is r actually used?'. 0 = OFF."""

    # ── Stage-1 staging ──
    freeze_vsa_base: bool = False
    """STATICALLY freeze the 1-1 VSA base for the whole run. Always freezes the CONDITIONING ANCHOR
    (skill_proj + the learned null token) so B-mode reproduces 1-1's z/null exactly; vision is frozen too
    iff freeze_vsa_vision. Used by STAGED phase 1-2 (only the action expert + Oracle + r_proj train).
    False for 1-1 and single-stage."""
    freeze_vsa_vision: bool = True
    """Whether the VISION scene encoder (DINO/SigLIP + image_proj + cond-encoder) is frozen along with the
    conditioning anchor (skill_proj), so the base is learned only by the no-r objective and never co-adapts
    to r. Two regimes:
      • STAGED 1-2 (freeze_vsa_base=True): the anchor (skill_proj) is STATICALLY frozen; vision too iff True.
      • SINGLE (freeze_vsa_base=False, use_oracle): PER-BATCH — A-mode (r used) freezes the base, B-mode
        (r dropped) trains it. The anchor (skill_proj) is A-frozen ALWAYS in single; this flag only adds
        VISION to that A-freeze. (The single-stage analog of staged 1-2's static freeze, over the A/B split.)
    Ignored when use_oracle is off (no A/B split, no r). null_r needs no gate — used only in B."""
    boundary_mode: str = "hold"
    """How chunk steps that spill PAST the current skill's end are supervised:
      "hold"      → STOP+HOLD target (arm Δ=0, gripper held) → each skill terminates cleanly (modularity).
      "keep_demo" → keep the demo's real actions across the boundary (goal r guides the tail)."""

    # ── Co-trained FSQ terminator (skill-end timing for eval; orthogonal to the Oracle) ──
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
    checkpoint's own image_model_name. Kept SEPARATE from dino_model_path (the policy's OWN
    vision backbone, which must match the checkpoint being loaded — never override that at eval)."""
    skill_advance_mode: str = "terminator"
    """How the oracle eval advances through the GT skill sequence:
    "terminator" → the FSQ terminator decides each transition (skill_end_mode/threshold);
    "gt"         → advance after each skill's GT demo duration (ideal timing)."""
    skill_end_mode: str = "termination"
    """Which FSQ terminator signal ends the current skill: "termination" (prob ≥ thr) | "progress" (≥ thr)."""
    skill_end_threshold: float = 0.5
    """Threshold on the signal selected by skill_end_mode, above which the skill is finished."""
    inference_skill_max_length: int = 200
    """Force-advance the skill after this many steps even if the terminator never fires (0 = off)."""

    def __post_init__(self):
        super().__post_init__()
        if self.state_cond_mode not in ("state", "state_skill"):
            raise ValueError(f"state_cond_mode must be 'state' or 'state_skill' (got {self.state_cond_mode!r}).")
        if self.boundary_mode not in ("hold", "keep_demo"):
            raise ValueError(f"boundary_mode must be 'hold' or 'keep_demo' (got {self.boundary_mode!r}).")
        if not 0.0 <= self.oracle_dropout_p < 1.0:
            raise ValueError(f"oracle_dropout_p must be in [0, 1) (got {self.oracle_dropout_p}).")
        if self.train_terminator and not self.fsq_path:
            raise ValueError("train_terminator=True needs fsq_path (the FSQ checkpoint to warm-start).")
        if self.train_terminator and not self.skill_decoder_dino_tokens_path:
            raise ValueError(
                "train_terminator=True needs skill_decoder_dino_tokens_path (FSQ-grid current-frame DINO "
                "tokens, precomputed at build_data; attached by SkillVLADinoTokenDataset)."
            )
