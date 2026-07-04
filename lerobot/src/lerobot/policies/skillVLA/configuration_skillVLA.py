from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("skill_vla")
@dataclass
class SkillVLAConfig(PI05Config):
    """Stage-2 SkillVLA.

    A VLM (PaliGemma, warm-started from pi05_base) reads the SKILL-START observation
    (3rd + wrist image + the start state discretized into the prompt, pi05-style + language) and stays
    PRISTINE; a SEPARATE skill reader (learned probe tokens over the VLM's FINAL-layer output, joint
    concat-KV, VLM read-only) → the FSQ head predicts the current skill (regression over per-dim grid
    coords). An action expert — warm-started from a Stage-1 ``skill_expert`` checkpoint — generates the
    action chunk by flow matching from the CURRENT observation, reading the VLM via PI05-style joint block
    attention (action attends the VLM's image tokens, language iff attend_language; the VLM never attends
    the action expert).

    Branch is taken from the loaded Stage-1 checkpoint's ``expert_arch``:
      A (joint): the action expert ALSO cross-attends the Stage-1 cond-encoder (current obs + skill),
                 so it reads two prefixes — cond-encoder (fresh every step) + VLM (cached per skill).
      B (fused): the fused action expert reads the VLM; its own [image/state/skill, action] stream is
                 block-causal — cond (image/state/skill) ⊥ action (cond does not attend the noisy
                 action), action attends cond + action.

    Skill flow: the discrete GT skill is teacher-forced into the cond-encoder (A) / expert (B) via the
    Stage-1 skill embedding at train time (the skill reader's prediction is used at inference); the skill
    reader reads the VLM one-directionally (read-only) and never perturbs it.

    Inputs (per sample, from SkillVLADataset): current obs/actions/language + the (jittered) skill-start
    image/state and skill code. Inherits PI05's PaliGemma + Gemma-expert + flow-matching settings.
    """

    model_type: str = "skill_vla"

    # ── Stage-1 warm-start (action expert [+ cond-encoder for branch A]) ──
    stage1_checkpoint_path: str | None = None
    """A Stage-1 ``skill_expert`` checkpoint. Its config supplies expert_arch (A/B), vision_backbone,
    action_expert_variant and skill_vocab_size; its weights warm-start the action expert (+ cond-encoder
    for A). The VLM is warm-started separately from ``pretrained_path`` (pi05_base)."""

    # ── Skill (VLM head; FSQ codes shared with Stage-1 / FSQ) ──
    skill_fsq_levels: list[int] = field(default_factory=lambda: [5, 5, 5])
    """FSQ levels per dim. The skill decoder REGRESSES one normalized grid coord per dim (levels per dim)
    and rounds; skill vocab = prod(levels). Must match the Stage-1 / FSQ codebook the dataset was built with."""
    # ── Skill reader (SEPARATE post-VLM module; the VLM stays pristine [imgs, lang]) ──
    # N learned probe tokens read the VLM's FINAL-layer output via JOINT concat-KV attention (the probes'
    # own K/V compete with the VLM K/V in one softmax — same joint pattern as cond/action, but read-only
    # so the VLM is never perturbed). Respects attend_language (images always, language iff True). The
    # pooled probe hidden feeds the SkillHead. Freezing the skill decoder for VLM-only finetuning freezes
    # BOTH this reader and the head (freeze_skill_head).
    num_reader_tokens: int = 4
    """Number of learned probe/reader tokens in the skill reader (they self-attend + read the VLM)."""
    reader_depth: int = 2
    """Number of JOINT concat-KV layers in the skill reader (keep shallow: catastrophic-forgetting-friendly)."""
    reader_heads: int = 8
    """Attention heads in the skill reader (must divide the VLM width)."""
    skill_deadzone_frac: float = 0.0
    """SkillHead regression dead-zone as a FRACTION of the per-dim grid spacing. Per-dim margin =
    frac / (levels-1) in the normalized coord frame; once |pred - target| < margin (i.e. the rounded
    code is already correct with a `frac` safety band) that dim's loss is zeroed → training stops nudging
    a code that already decodes right (faster convergence, no post-correct drift). 0 = plain MSE (off).
    Scales with the codebook automatically (spacing = 2/(levels-1)), so it is codebook-independent."""
    skill_loss_weight: float = 0.5
    """λ_skill in ``total = BC + λ_skill * skill_CE``. < 1 keeps the action BC dominant (including the
    BC gradient that flows into the VLM via cross-attention)."""
    # ── Inter-module attention connections (VLM, cond, action expert). Each is a directed edge in the
    # VLM → cond → expert chain (+ the VLM → expert shortcut); toggling them is a design choice. All apply to
    # BOTH the training joint mask AND the cached inference path (train/infer must match → a Stage-2 retrain).
    attend_language: bool = False
    """Whether the VLM's LANGUAGE tokens are attendable (by cond, the action expert, AND the skill reader).
    False = only the VLM IMAGE tokens are read (language excluded → forces visual grounding); True = language
    ALSO attendable (ablation). This is the language MASTER switch: it gates the language subset for every
    VLM→X edge (vlm_cond, vlm_expert, and the skill reader). Run-name tag: lang."""
    vlm_cond: bool = True
    """VLM → cond edge: cond attends the VLM tokens (images always, language iff attend_language). True =
    the cond-encoder is grounded by the VLM's skill-start view (default). False = cond is a plain current-obs
    scene encoder, VLM-blind. Run-name tag (when off): noVc."""
    cond_expert: bool = True
    """cond → expert edge: the action tokens attend the cond (scene) stream. True = the backbone path the
    action expert reads the scene through (default). False = the expert ignores cond (acts on skill z + its
    prefix + the VLM directly if vlm_expert). Run-name tag (when off): noCe."""
    vlm_expert: bool = False
    """VLM → expert (direct) edge: the action tokens attend the VLM tokens directly (images always, language
    iff attend_language), in ADDITION to the cond path. False = the original one-directional chain
    (VLM → cond → action; action sees the VLM only via cond). True = a direct VLM→action shortcut. The skill
    PREFIX still reads only itself; cond ⊥ action unchanged. Run-name tag: ve. (Was: action_attend_vlm.)"""
    vlm_dropout_p: float = 0.0
    """CFG-style VLM dropout (TRAIN only). With this per-batch probability, SEVER the cond→VLM and
    action→VLM attention edges for the whole batch, so the action expert acts on cond (current obs) +
    the GT skill z (prefix/AdaRMS) ALONE — exactly the Stage-1 form — while the VLM stream still runs
    (self-attn), so the skill head is still supervised every batch (no staleness). This teaches the expert
    to not COLLAPSE onto the VLM grounding. p=0 → never dropped (bit-identical to no-dropout; the coin is
    only flipped when p>0). Mask-only (RoPE/positions unchanged from a non-dropped batch — a pure ablation,
    NOT token removal). Intended for cond_skill_source='gt' (the skill is GT-injected, VLM-independent).
    Run-name tag: vdrop{p}."""

    # ── Per-regime freeze for the CFG dropout (ONLY used when vlm_dropout_p > 0) ──
    # A = VLM_VSA (VLM present) batches | B = VSA (VLM dropped = Stage-1 form) batches. True = FREEZE (no
    # param update) that group on that regime's batches. Groups: expert = the action expert + its action/
    # time/state/skill projections; cond = cond-encoder + image_proj; vlm = the VLM LLM. A group
    # is placed in the optimizer iff it trains in AT LEAST one regime; requires_grad is toggled per batch by
    # the coin flip (Adam skips requires_grad=False params → clean freeze, no momentum leak). When
    # vlm_dropout_p == 0 these are IGNORED and the static freeze_* flags below apply (backward-compatible).
    # Vision backbones (freeze_vlm_vision / freeze_expert_vision) + the skill decoder (freeze_skill_head =
    # skill_reader + skill_head) stay static (the skill reader is post-VLM, supervised every batch).
    # ⚠ MULTI-GPU: with num_gpus>1 (DDP) AND a group trained in only ONE regime, ranks that pick different
    #   drop_vlm coins freeze different params → the DDP reducer can desync/hang. Keep num_gpus=1 for
    #   per-regime freeze, or broadcast one coin to all ranks. (num_gpus=1 — the default — is unaffected.)
    freeze_vlm_vsa_expert: bool = False
    freeze_vlm_vsa_cond: bool = False
    freeze_vlm_vsa_vlm: bool = False
    freeze_vsa_expert: bool = False
    freeze_vsa_cond: bool = False
    freeze_vsa_vlm: bool = False

    # ── Freeze toggles (all parts otherwise trained) ──
    freeze_vlm: bool = False
    """Freeze the PaliGemma LLM (the VLM trunk)."""
    freeze_vlm_vision: bool = False
    """Freeze the PaliGemma SigLIP vision tower. Default False (train it for the new skill-start view)."""
    freeze_cond_encoder: bool = True
    """[A only] Freeze the Stage-1 cond-encoder (reuse its current-obs encoding)."""
    freeze_action_expert: bool = False
    """Freeze the warm-started Gemma action expert (default False = fine-tune)."""
    freeze_expert_vision: bool = False
    """Freeze the action-expert-side vision encoder (DINO/SigLIP) inherited from Stage-1."""
    freeze_skill_head: bool = False
    """Freeze the WHOLE skill decoder — the skill reader (probe tokens) AND the SkillHead readout. Default
    False (Stage-2 trains them). Finetuning sets True: the reader + FSQ-coordinate readout stay pinned to
    the (unchanged) codebook the frozen action expert / terminator expect, so ALL skill-prediction
    adaptation goes into the VLM trunk. The frozen decoder still conducts gradient to the trunk (incl. the
    cond_skill_source=pred STE path), it just isn't updated."""

    # ── Differential LR (relative to optimizer_lr) for the warm-started action/cond side ──
    expert_lr_scale: float = 1.0
    """LR multiplier (× optimizer_lr) for the action expert (gemma_expert + action/time projections
    + the ae skill/progress prefix projections). >1 lets the warm-started expert adapt faster to
    using the obs for intra-skill detail."""
    cond_lr_scale: float = 1.0
    """LR multiplier (× optimizer_lr) for the cond side (cond_encoder + image/state projections).
    The VLM and vision backbones keep the base optimizer_lr."""

    # ── Finetuning (FT): self-conditioned skill + terminator co-training ──
    cond_skill_source: str = "gt"
    """Where the skill code fed to the action-prefix (ae) / cond skill-token (fused) comes from at
    TRAIN time. "gt" (Stage-2 default): teacher-force the dataset's GT code. "pred" (FT): use the
    VLM's OWN predicted code (matches inference / removes exposure bias) via an STE-round so the flow
    loss backprops through SkillHead into the VLM trunk. The skill CE loss vs the GT code is kept
    either way. (progress stays GT-jittered in both.)"""
    train_terminator: bool = False
    """FT: also co-train the FSQ terminator on this dataset's GT signals (z←GT code, current DINO +
    state → progress + termination), so the terminator adapts to the new task before it gates skill
    transitions at inference. Trained on a DISJOINT graph (GT/precomputed inputs only) → zero effect
    on the SkillVLA params; only shares the dataloader. Warm-starts from ``fsq_path`` and is exported
    back to an FSQ checkpoint for eval. Needs the dataset to supply current-frame DINO tokens."""
    terminator_lr_scale: float = 1.0
    """LR multiplier (× optimizer_lr) for the co-trained terminator params (train_terminator only)."""
    terminator_end_target_sigma: float = 2.0
    """Soft termination target std in FRAMES: target = exp(-(frames_to_end)² / (2σ²)) (Gaussian that
    is 1.0 at the skill's last frame and decays earlier). Mirrors the FSQ terminator's training."""
    terminator_end_pos_weight: float = 1.0
    """BCE positive-class weight for the co-trained terminator's termination head."""

    # ── Current-frame DINO tokens for the co-trained terminator (data factory wraps the dataset) ──
    skill_decoder_dino_tokens_path: str | None = None
    """Frame-level DINO token npz (build_data's ``dino.npz``). When set, the data factory wraps the
    SkillVLADataset with SkillVLADinoTokenDataset to attach the current frame's 3rd-person DINO tokens
    under ``skill_decoder_dino_output_key`` — the terminator co-training's image input. Train only."""
    skill_decoder_dino_output_key: str = "skill_decoder_dino"
    """Batch key for the attached current-frame DINO tokens (kept distinct from the inference-time
    ``skill_decoder_image`` so the raw-obs processor step can't clobber it during training)."""
    skill_decoder_dino_cache_path: str | None = None
    """Optional mmap cache (.npy) for the token npz; None → next to the npz."""
    skill_decoder_dino_build_cache: bool = True
    """Build the mmap cache from the npz on first use (else require it to exist)."""
    # Wrist-camera FSQ-grid DINO tokens — ONLY needed when the FSQ terminator is DUAL (terminator_use_wrist,
    # e.g. a "both" FSQ). The factory attaches a 2nd SkillVLADinoTokenDataset under the wrist output key.
    skill_decoder_dino_wrist_tokens_path: str | None = None
    """Wrist-camera DINO token npz (build_data's ``dino_wrist.npz``). Required for a dual (use_wrist) FSQ
    terminator; leave None for a 3rd-only ('wow') FSQ."""
    skill_decoder_dino_wrist_output_key: str = "skill_decoder_dino_wrist"
    skill_decoder_dino_wrist_cache_path: str | None = None

    # ── Inference: skill transitions via the frozen FSQ terminator ──
    fsq_path: str | None = None
    """Frozen FSQ checkpoint whose terminator decides skill transitions during closed-loop rollout.
    Unused at training (skill boundaries come from the dataset)."""
    skill_end_mode: str = "termination"
    """Which FSQ signal ends the current skill:
       "termination" → end_prob   >= skill_end_threshold
       "progress"    → progress   >= skill_end_threshold
       "and"         → end_prob >= skill_end_threshold AND progress >= skill_end_progress_threshold
                       (both must hold — guards against early end-prob spikes before the skill is done)."""
    skill_end_threshold: float = 0.5
    skill_end_progress_threshold: float = 0.9
    """Progress gate for skill_end_mode="and": the skill ends only once predicted progress also
    reaches this (e.g. 0.9). Unused in "termination"/"progress" modes."""
    inference_skill_max_length: int = 200
    """Force-advance the skill after this many steps even if the terminator never fires (0 = off)."""

    # ── Oracle eval: feed GT skill codes (per task) into the cond-encoder instead of the VLM ──
    use_gt_skill: bool = False
    """Oracle eval: teacher-force the dataset's GT skill code sequence into the cond-side skill
    embedding (the VLM still encodes the start obs, but its predicted code is bypassed). Isolates
    the action expert / terminator from the VLM's skill-prediction quality."""
    gt_skill_dataset_dir: str | None = None
    """skillvla dataset whose ``skill_sequence`` (per task) supplies the GT skills for oracle eval."""
    skill_advance_mode: str = "terminator"
    """How a skill ends during rollout: "terminator" (FSQ signal >= threshold) or "gt" (advance by
    the GT skill's demo duration; oracle only). The terminator still runs each step either way so
    its curves are recorded for skill_html."""
