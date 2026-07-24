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
    tokenizer_path: str | None = None
    """Local PaliGemma tokenizer directory. Stage-2 training sets this explicitly so preprocessing never
    depends on Hugging Face authentication or a machine-specific cache. None retains legacy Hub behavior."""
    stage1_checkpoint_path: str | None = None
    """A Stage-1 ``skill_expert`` checkpoint. Its config supplies expert_arch (A/B), vision_backbone,
    action_expert_variant and skill_vocab_size; its weights warm-start the action expert (+ cond-encoder
    for A). The VLM is warm-started separately from ``pretrained_path`` (pi05_base).
    None/"" = SCRATCH mode: the expert/cond side is FRESH-initialized (no Stage-1 training at all) and its
    architecture comes from s1_vision_backbone / s1_state_cond_mode below — the VSA identity is then
    carved purely by the vlm_dropout B batches (use vlm_dropout_p > 0, ideally with a decay schedule)."""
    s1_vision_backbone: str = "siglip"
    """SCRATCH mode only (stage1_checkpoint_path empty): the cond-side vision encoder ("dino"|"siglip")."""
    s1_state_cond_mode: str = "state"
    """SCRATCH mode only: Stage-1 conditioning mode ("state"|"state_skill"|"broadcast")."""

    # ── Skill (VLM head; FSQ codes shared with Stage-1 / FSQ) ──
    skill_fsq_levels: list[int] = field(default_factory=lambda: [5, 5, 5])
    """FSQ levels per dim. The skill decoder REGRESSES one normalized grid coord per dim (levels per dim)
    and rounds; skill vocab = prod(levels). Must match the Stage-1 / FSQ codebook the dataset was built with."""
    # ── Skill reader (SEPARATE post-VLM module; the VLM stays pristine [imgs, lang]) ──
    # N learned probe tokens read the VLM's FINAL-layer output via JOINT concat-KV attention (the probes'
    # own K/V compete with the VLM K/V in one softmax — same joint pattern as cond/action, but read-only
    # so the VLM is never perturbed). Respects attend_language (images always, language iff True). The
    # pooled probe hidden feeds the SkillHead. For finetuning the two are frozen SEPARATELY:
    # freeze_skill_reader (this reader) and freeze_skill_head (the FSQ readout).
    num_reader_tokens: int = 4
    """Number of learned probe/reader tokens in the skill reader (they self-attend + read the VLM)."""
    reader_depth: int = 2
    """Number of JOINT concat-KV layers in the skill reader (keep shallow: catastrophic-forgetting-friendly)."""
    reader_heads: int = 8
    """Attention heads in the skill reader (must divide the VLM width)."""
    skill_reader_all_layers: bool = False
    """Feed the skill reader ALL VLM layers' outputs (each normed by the VLM final norm, stacked as
    K/V) instead of only the final layer. A FROZEN LLM's final rep is language-centric; the skill code
    is a bespoke motion primitive whose features often sit in MID layers — giving the reader all layers
    lets its attention softly pick the informative depth (cond already reads all layers). Read-only
    (VLM unperturbed) → train (joint forward) and inference (predict_skill_code) capture the SAME layer
    stack, so no train/infer skew. Cost: KV grows L× (few probes → attention still cheap) + keeps the
    per-layer VLM hiddens for backward. False → final-layer only (byte-identical to before)."""
    skill_deadzone_frac: float = 0.0
    """SkillHead regression dead-zone as a FRACTION of the per-dim grid spacing. Per-dim margin =
    frac / (levels-1) in the normalized coord frame; once |pred - target| < margin (i.e. the rounded
    code is already correct with a `frac` safety band) that dim's loss is zeroed → training stops nudging
    a code that already decodes right (faster convergence, no post-correct drift). 0 = plain MSE (off).
    Scales with the codebook automatically (spacing = 2/(levels-1)), so it is codebook-independent."""
    skill_loss_weight: float = 0.5
    """λ_skill in ``total = BC + λ_skill * skill_CE``. < 1 keeps the action BC dominant (including the
    BC gradient that flows into the VLM via cross-attention)."""
    action_weight: bool = False
    """Stage-2 connected A only: apply reverse skill-progress weighting to the GT flow objective."""
    skill_start_loss_weight: float = 1.0
    """Maximum Stage-2 A weight at skill start. Weight decays linearly to 1 at skill end. The severed
    teacher anchor remains uniformly weighted and is controlled only by cond_severed_anchor_weight."""
    # ── Multi-adapter LoRA (continual learning; all bases FROZEN) ──
    # Three independently selectable adapters share lora_rank/alpha/dropout/targets (inherited from
    # PI05Config). Selected per-forward via lora.active_adapters():
    #   ① lora_skill @ VLM LLM      — Stage-3 skill-prediction view.
    #   ② vlm_lora   @ VLM LLM      — Stage-2 residual extraction from frozen vision/language features.
    #   ③ cond_lora  @ cond encoder — Stage-2 ingestion of that residual by the frozen VSA cond path.
    # A Stage-1 expert LoRA, when present, is restored separately as the normally frozen
    # ``expert_lora`` component of the VSA. Direct-FSQ Stage-0 creates that same named adapter and
    # trains it according to its explicit A/B component matrices.
    # All newly-created adapters start B=0 (model == base at init); FT trains ① while ②③ remain frozen.
    # Keep inherited pi05 ``lora_enable`` OFF: this policy owns these named adapters.
    lora_skill: bool = False
    vlm_lora: bool = False
    cond_lora: bool = False
    lora_lr_scale: float = 1.0
    """Adapter-only LR multiplier (× optimizer_lr) for all newly trainable LoRA params (①②③). Rank-r B=0 adapters
    conventionally want ~10-40× the full-finetune LR; this leaves the full-trained parts (vlm_vision at
    PT, expert at FT) on the base LR instead of raising optimizer_lr globally."""
    lang_bridge: bool = False
    """Stage-0 edge adapter. Adds a zero-initialized low-rank residual only to the VLM-language K/V
    copy read by cond queries. VLM self-attention and the expert's K/V remain unchanged."""
    lang_bridge_rank: int = 64
    lang_bridge_lr_scale: float = 10.0
    """LR multiplier for the language-edge bridge relative to optimizer_lr."""
    # ── LoRA-continual training stages (REPLACES the legacy A/B/C regime_probs / vlm_dropout_p system;
    # those fields remain below only so old checkpoints' config.json still deserializes). ──
    pt_stage: str | None = None
    """Sequential PT stage — interleaved [SKILL, COND] mixing was retired (once the VLM vision is
    COND-owned the two paths share NO trainable params, so they train back-to-back instead):
      "stage0" = direct FSQ route: branch-local component matrices control gradients. Default A trains
                 vlm_lora+language bridge+cond+expert_lora; VLM-severed B trains cond+cond vision+
                 expert_lora. Both receive GT skill and GT action-flow supervision.
      "cond"  = STAGE 2: every batch is a COND/flow batch — trains ②③ + vlm_vision(+projector) against
                the frozen expert with GT-teacher-forced z. No skill loss; reader/head untouched.
      "skill" = STAGE 3: every batch is a SKILL batch — trains ①+reader+head on the skill loss;
                EVERYTHING else (vision included) frozen. Warm-starts from a Stage-0 or Stage-2 checkpoint;
                structurally identical to FT's SKILL regime (its rehearsal).
      None    = not a PT run (FT via regime_probs_ft, or eval)."""
    stage3_parent_stage: str = "stage2"
    """Stage-3 full-checkpoint source (``stage0`` or ``stage2``). A Stage-0 parent has no Stage-1
    config, so this flag tells model construction to recreate its direct expert and expert-LoRA wrappers
    before loading the frozen parent weights."""
    skill_predictor_vlm_path: str | None = None
    """Optional Stage-3 predictor-only PaliGemma source. The main checkpoint still supplies the complete
    frozen motor (its VLM + cond + expert); only the standalone skill-prediction view uses this separate
    VLM. Blank keeps the parent's VLM shared by both views."""
    stage0_vlm_severed_prob: float = 0.3
    """Stage-0 probability of a B batch. The branch-local drop/train component fields below define
    the exact topology and gradient scope; both branches use the same GT flow target."""
    stage0_a_drop_vlm: bool = False
    stage0_b_drop_vlm: bool = True
    stage0_expert_source: str = "fsq"
    """Legacy Stage-0 action-expert initialization switch. Renewed residual Stage-0 requires
    ``pi05_base``: the action expert warm-starts from pi05, while FSQ supplies the skill grid and
    terminator checkpoint."""
    stage0_cond_state_adarms: bool = False
    """Inject the current normalized robot state into every cond-encoder layer through AdaRMS."""
    stage0_vlm_residual: bool = False
    """Renewed Stage-0 motor objective. The base VSA path runs VLM-severed; a single bounded
    cross-attention residual then lets final-normalized VLM tokens correct the raw action-token hidden
    immediately before the expert final norm/action head."""
    stage0_vlm_residual_heads: str = "auto"
    """Cross-attention heads for the VLM residual. ``auto`` reuses the action expert attention head count."""
    stage0_vlm_residual_dropout: float = 0.0
    stage0_vlm_residual_alpha_min: float = 0.1
    stage0_vlm_residual_alpha_max: float = 0.2
    stage0_vlm_residual_init_alpha: float = 0.15
    stage0_vlm_residual_zero_init_output: bool = True
    """Bounded scalar gate: alpha_min + (alpha_max-alpha_min)*sigmoid(gate_logit). The attention
    output projection starts at zero by default, so the residual is initially an exact no-op while
    still receiving a healthy gradient through the nonzero alpha floor."""
    stage0_conditional_loss_weight: float = 1.0
    stage0_unconditional_loss_weight: float = 0.5
    stage0_gradient_routing: str = "shared"
    stage0_uncond_skill_start_loss_weight: float = 1.0
    stage0_uncond_skill_end_loss_weight: float = 1.0
    """Renewed Stage-0 loss:
       λ_cond * GT flow loss with VLM residual + λ_uncond * GT flow loss without the residual.
       The unconditional term can be skill-progress weighted from start_weight to end_weight.
       ``split`` routing lets conditional loss update only the VLM/residual path while unconditional
       loss owns the shared base VSA; ``shared`` preserves the original both-losses-update-base behavior."""
    stage0_language_ranking_enabled: bool = False
    stage0_language_ranking_weight: float = 0.1
    stage0_language_ranking_relative_margin: float = 0.01
    """Optional renewed Stage-0 Exp4 objective. The correct-language conditional flow must beat a
    different-task-language conditional flow by a relative margin, measured only within the current
    skill. Disabled means no negative sampling or extra VLM forward, preserving Exp0-3 exactly."""
    same_skill_batch_enabled: bool = False
    same_skill_batch_fraction: float = 0.5
    same_skill_progress_temperature: float = 0.1
    """Renewed Stage-0 mixed batch sampler. A fraction of each batch is arranged as adjacent
    same-post-jitter-skill/different-task pairs; partner post-jitter progress is a soft nearest-neighbor
    preference. Pair slots share one draw from the configured jitter law and pass the resolved result
    to the dataset, while ordinary random slots retain independent dataset-side jitter."""
    stage0_freeze_vlm_llm: bool = True
    stage0_freeze_vlm_vision: bool = True
    stage0_freeze_cond: bool = False
    stage0_freeze_cond_vision: bool = False
    stage0_freeze_expert: bool = False
    stage0_freeze_skill_reader: bool = True
    stage0_freeze_skill_head: bool = True
    """Static trainability for renewed Stage-0. By default VLM is frozen, while cond + action expert train."""
    stage0_train_skill_predictor: bool = False
    """Renewed Stage-0 auxiliary skill predictor. Trains only skill_reader + skill_head in an isolated
    optimizer step over no-grad frozen VLM hidden, so it cannot change the action/cond/expert/VLM path."""
    stage0_skill_predictor_weight: float = 0.5
    stage0_skill_predictor_lr_scale: float = 1.0
    stage0_skill_predictor_all_layers: bool = False
    stage0_skill_predictor_detach_vlm: bool = True
    """Predictor auxiliary knobs. detach_vlm must stay True for the isolated no-backflow contract."""
    stage0_a_train_components: str = "vlm_lora,lang_bridge,cond,expert_lora"
    stage0_b_train_components: str = "cond,cond_vision,expert_lora"
    """Branch-local Stage-0 gradient matrix. Component names are validated and stored in checkpoints;
    the optimizer contains their union, while each forward enables only its branch's listed components."""
    stage0_wrong_language_weight: float = 0.0
    """Optional A-only ranking loss weight. The same image/state/GT skill is paired with another
    task's language; correct language must produce a lower GT flow error by the configured margin."""
    stage0_wrong_language_margin: float = 0.02
    stage0_expert_lora: bool = True
    stage0_a_skill_start_loss_weight: float = 1.0
    stage0_a_skill_end_loss_weight: float = 1.0
    stage0_b_skill_start_loss_weight: float = 1.0
    stage0_b_skill_end_loss_weight: float = 1.0
    """Stage-0 flow-loss weights at skill progress 0 and 1, independently for A and B. Each branch
    linearly interpolates between its endpoints; (1, 1) is exactly the original uniform objective."""
    stage0_expert_lora_targets: str = "q,k,v,o,mlp,action_out"
    stage0_expert_lora_rank: int = 8
    stage0_expert_lora_alpha: float = 16.0
    stage0_expert_lora_dropout: float = 0.0
    stage0_expert_lora_lr_scale: float = 10.0
    s1_cond_encoder_variant: str | None = None
    s1_dino_model_path: str = "models/dinov3-vits16"
    s1_dino_image_size: int = 224
    s1_siglip_image_size: int = 224
    cond_severed_prob: float = 0.0
    """STAGE-2 severed anchor probability. The batch constructs x_t from the same boundary-aware hold
    action as Stage-1, then compares two VLM-free forwards: frozen VSA with cond_lora OFF (teacher) versus
    cond_lora ON (student). This is function preservation, not GT action supervision, so cond_lora cannot
    learn an image-only motion shortcut. Only cond_lora receives gradient; vlm_lora/vision are outside the
    graph and get (1-p) of the effective Stage-2 updates. 0 = off."""
    cond_severed_anchor_weight: float = 1.0
    """Stage-2 severed function-anchor multiplier. The sampled objective is
    ``(1-p) * connected_flow + p * weight * severed_anchor``; 1 keeps the natural mixture weighting."""
    regime_probs_pt: list[float] | None = None
    """DEPRECATED (kept so existing checkpoints' config.json still deserializes): the retired
    interleaved [SKILL, COND] probabilities. COND-only values ([0,1]) auto-map to pt_stage="cond";
    anything else errors — use pt_stage."""
    regime_probs_ft: list[float] | None = None
    """FT regimes [P(SKILL), P(MOTOR_connected), P(MOTOR_severed)] (e.g. [0.0, 0.5, 0.5]). SKILL trains
    ①+reader+head only if ft_train_skill (else keep its prob 0 — nothing trainable there). MOTOR trains
    the expert: connected = flow through frozen ②③ (VLM residual riding cond); severed = adapters OFF
    (③ bypass → pure Stage-1 VSA, no VLM forward) + Stage-1 hold target + random-skill distillation to
    the frozen-PT teacher (vsa_distill; teacher runs severed/adapter-free too)."""
    skill_action_grounding: bool = False
    """STAGE 3 mode b: ground the skill prediction through the FROZEN motor. Each batch runs the SKILL
    view (① — grad), STE-rounds its prediction to z, then evaluates z through the frozen connected
    motor (②-VLM → cond(+③) → expert, all no-grad cached constants; deployment-identical attention)
    against the GT-action flow target — so the MAIN loss becomes "pick the skill the frozen motor can
    execute", with the GT-code loss as the anchor (weight = skill_loss_weight). Needs the FULL
    frame-level dataset (per-frame action supervision) — NOT the transition pack. Only meaningful with
    pt_stage="skill" (everything but ①/reader/head frozen keeps the signal pure)."""
    transition_packs: str | None = None
    """Comma-separated transitions.npz paths (build_data/src/build_transition_pack.py). When set, the
    dataset factory serves SEGMENT-level transition samples (skill-start±pmax images + language + code)
    instead of frame-level video samples — the deployment distribution of the skill predictor (the VLM
    only runs at transitions). Use for pt_stage="skill" (stage 3) and FT SKILL batches; the FT replay
    buffer is simply the parent PT dataset's pack appended to this list. Empty/None = the regular
    frame-level SkillVLADataset."""
    ft_train_skill: bool = False
    """FT: unfreeze the skill path (adapter ① + skill_reader + skill_head) so SKILL regime batches adapt
    skill prediction to the new task (new vocabulary/objects) — pair with replay to preserve old-task
    skill prediction. False = skill path fully frozen (set regime_probs_ft[0]=0; FT is then motor-only)."""
    # ── Inter-module attention connections (VLM, cond, action expert). Each is a directed edge in the
    # VLM → cond → expert chain (+ the VLM → expert shortcut); toggling them is a design choice. All apply to
    # BOTH the training joint mask AND the cached inference path (train/infer must match → a Stage-2 retrain).
    attend_language: bool = False
    """Whether the VLM's LANGUAGE tokens are attendable (by cond, the action expert, AND the skill reader).
    This is the language MASTER switch, combined with attend_image below to pick WHICH VLM tokens every
    VLM→X edge (vlm_cond, vlm_expert, and the skill reader) reads. Run-name tag: lang."""
    attend_image: bool = True
    """Whether the VLM's IMAGE tokens are attendable (companion to attend_language). Together they pick the
    VLM read-set for cond/expert/reader:
       attend_image=T, attend_language=F → IMAGE only  (default; forces visual grounding)
       attend_image=T, attend_language=T → image + language
       attend_image=F, attend_language=T → LANGUAGE only (read the VLM's language hiddens, not vision)
       attend_image=F, attend_language=F → nothing (disallowed; use vlm_cond/vlm_expert=False to go VLM-blind).
    Default True keeps existing checkpoints unchanged. Run-name tag (when off): noimg."""
    reader_attend_image: bool | None = None
    """SKILL READER 전용 read-set 오버라이드 (None = attend_image 상속 → 기존 체크포인트 동작 불변).
    cond conduit를 lang-only로 조여도(attend_image=False) skill 예측기는 시작 obs 이미지를 계속 봐야
    하므로 분리 — reader_attend_image=true면 reader만 이미지 위치 히든을 읽음. 학습(stage3 ①)과 추론
    (predict_skill_code)이 같은 값을 써야 하므로 stage3/FT는 체크포인트 config.json에서 상속."""
    reader_attend_language: bool | None = None
    """reader_attend_image의 language 축 (None = attend_language 상속). skill 예측은 보통 image+lang
    둘 다 유용: reader_attend_image=true + reader_attend_language=true."""
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
    vlm_dropout_p_end: float | None = None
    """Dropout SCHEDULE: linearly anneal the per-batch prob from ``vlm_dropout_p`` to this value over
    ``vlm_dropout_decay_steps`` steps, then hold. None = constant ``vlm_dropout_p``. E.g. 0.9→0.2:
    early training is mostly VSA (B) batches — carving the standalone motor module (scratch mode) —
    then the budget shifts to VLM grounding (A)."""
    vlm_dropout_decay_steps: int = 0
    """Steps over which vlm_dropout_p anneals to vlm_dropout_p_end (0 = no schedule). The step counter
    is a persistent buffer → resume continues the schedule where it left off."""

    # ── Per-regime freeze for the CFG dropout (ONLY used when vlm_dropout_p > 0) ──
    # A = VLM_VSA (VLM present) batches | B = VSA (VLM dropped = Stage-1 form) batches. True = FREEZE (no
    # param update) that group on that regime's batches. Groups: expert = the action expert + action/time
    # proj; cond = the WHOLE current-obs pipeline (DINO/SigLIP vision encoder → image_proj → cond-encoder);
    # llm = the VLM's Gemma LLM trunk ONLY (renamed from "vlm" — it never covered the vision tower);
    # vlm_vision = the PaliGemma SigLIP vision tower.
    # A group is placed in the optimizer iff it trains in AT LEAST one regime; requires_grad is toggled per
    # batch by the coin flip (Adam skips requires_grad=False params → clean freeze, no momentum leak).
    # SINGLE SOURCE OF TRUTH: when vlm_dropout_p == 0 (no dropout) every batch is an A (VLM-present) batch,
    # so freeze_vlm_vsa_* applies STATICALLY and freeze_vsa_* is ignored (the old separate freeze_vlm /
    # freeze_cond_encoder / freeze_action_expert / freeze_vlm_vision flags were removed). Only the skill
    # decoder has its own static flags (post-VLM, supervised every batch), split into freeze_skill_reader
    # (probe reader) and freeze_skill_head (FSQ readout).
    # NOTE vlm_vision freeze gates only the VISION TOWER params; on a B batch with llm frozen but
    # vlm_vision trainable, the skill loss still updates the tower THROUGH the frozen LLM.
    # ⚠ MULTI-GPU: with num_gpus>1 (DDP) AND a group trained in only ONE regime, ranks that pick different
    #   drop_vlm coins freeze different params → the DDP reducer can desync/hang. Keep num_gpus=1 for
    #   per-regime freeze, or broadcast one coin to all ranks. (num_gpus=1 — the default — is unaffected.)
    # cond = the cond Gemma encoder + image_proj; cond_vision = the cond's OWN DINO/SigLIP vision tower,
    # split out so it can be frozen/unfrozen independently of the rest of the cond pipeline.
    freeze_vlm_vsa_expert: bool = False
    freeze_vlm_vsa_cond: bool = False
    freeze_vlm_vsa_cond_vision: bool = False
    freeze_vlm_vsa_llm: bool = False
    freeze_vlm_vsa_vlm_vision: bool = False
    freeze_vsa_expert: bool = False
    freeze_vsa_cond: bool = False
    freeze_vsa_cond_vision: bool = False
    freeze_vsa_llm: bool = False
    freeze_vsa_vlm_vision: bool = False
    # 3-way regime C (VLM severed, VSA) freeze dict. A uses freeze_vlm_vsa, B uses freeze_vsa, C uses
    # these — so all three regimes are independently settable. Unused in the 2-way (binary) mode.
    freeze_c_expert: bool = False
    freeze_c_cond: bool = False
    freeze_c_cond_vision: bool = False
    freeze_c_llm: bool = False
    freeze_c_vlm_vision: bool = False

    # 3-WAY regime (overrides the binary vlm_dropout coin when set). Per-batch sample of [A, B, C] with
    # these fixed probabilities (no schedule). A = VLM CONNECTED, freeze_vlm_vsa (perception adapts,
    # motor frozen). B = VLM CONNECTED, freeze_vsa (motor/VSA adapts WITH the frozen-VLM context —
    # matches inference). C = VLM SEVERED, freeze_vsa (pure VSA, Stage-1 form). A+B = alternating-freeze
    # on the connected model (never co-adapt at once); C keeps the VSA runnable VLM-free. Blank → the
    # 2-way vlm_dropout_p coin (+ vsa_distill_main_connected) is used instead.
    regime_probs: list[float] | None = None

    # C (severed VSA) batches decode a skill VLM-free, structurally identical to Stage-1. When True, the
    # BC target in C mirrors Stage-1: steps past the CURRENT skill's end (skill_de) — the chunk tail that
    # would otherwise be supervised with the NEXT skill's real actions the severed VSA cannot know — are
    # REPLACED with a HOLD (arm deltas → 0, gripper → last within-skill value) and still supervised, so
    # each skill learns to STOP + HOLD at its boundary (self-contained/reusable; anti-forgetting). A/B
    # (VLM connected) keep the real cross-skill tail. No-op if skill_de is absent from the batch.
    severed_hold_target: bool = True

    # ── Freeze toggles ──
    # (expert / cond / vlm / vlm_vision freezing is ALL governed by freeze_vlm_vsa_* / freeze_vsa_* above —
    #  at p=0 the A dict applies statically. Only the skill decoder keeps its own flags, split in two:)
    freeze_skill_head: bool = False
    """Freeze the SkillHead readout (pooled reader hidden → tanh FSQ-coordinate regression). Finetuning
    sets True: the readout stays pinned to the (unchanged) codebook the frozen action expert / terminator
    expect. The frozen head still conducts gradient to the reader/trunk (incl. the cond_skill_source=pred
    STE path), it just isn't updated."""
    freeze_skill_reader: bool = False
    """Freeze the SkillReader (probe tokens + joint concat-KV layers reading the VLM's final layer).
    Separate from freeze_skill_head: the head's code mapping is codebook-pinned (freeze in FT), while the
    reader is an obs→skill feature extractor one may WANT to adapt to a new task — freeze_skill_head=true
    + freeze_skill_reader=false re-grounds skill prediction through the reader too, not just the trunk."""
    # Explicit Stage-2 trainability matrix. These flags apply only when pt_stage="cond"; Stage-3 and FT
    # retain their own fixed continual-learning scopes. Defaults preserve the original Stage-2 behavior.
    freeze_vlm: bool = True
    freeze_cond: bool = True
    freeze_expert: bool = True
    freeze_vlm_lora: bool = False
    freeze_cond_lora: bool = False
    freeze_expert_lora: bool = True
    freeze_lang_bridge: bool = False
    freeze_vlm_vision: bool = False
    freeze_cond_vision: bool = True

    # ── Continual-learning VSA distillation (FT anti-forgetting) ──
    # In B (VSA-only, VLM-severed) batches, besides the strong GT-skill action loss, feed the SAME
    # (obs, x_t, t) with SAMPLED non-GT skills and match the student's action to a FROZEN teacher =
    # the PT/warm-start VSA. This pins the skill→action function across the codebook (not just the GT
    # skill the new task visits) → the reusable motor repertoire is preserved (LwF-style, replay-free).
    # Off by default (pure ablation). Requires vlm_dropout to fire (needs B batches) and expert/cond
    # unfrozen in B (a frozen VSA can't forget → distillation is moot).
    vsa_distill: bool = False
    """Enable the VSA teacher distillation on sampled skills in B batches."""
    vsa_distill_weight: float = 0.2
    """λ: distillation loss weight relative to the (strong) GT-skill BC loss. Weak (0.1–0.3)."""
    vsa_distill_n_local: int = 2
    """# sampled skills that are FSQ-grid NEIGHBORS of the GT code (obs-conditional; preserves the
    local skill→action geometry so neighbours stay distinct under the new task's adaptation)."""
    vsa_distill_n_global: int = 2
    """# sampled skills drawn from the global code frequency, EXCLUDING the GT code + its neighbours
    (the cross-context repertoire — the region the new task never visits, where forgetting concentrates)."""
    vsa_distill_neighbor_radius: int = 1
    """FSQ-grid radius (±r per dim) defining the GT neighbourhood for the local samples / exclusion set."""
    vsa_distill_freq_path: str | None = None
    """npz with per-code counts (keys 'counts' [frames] + 'motion_counts' [trajectories]) for the global
    sampler / motion-counter SEED. Blank → uniform. Built from the PARENT PT skillvla dataset."""
    vsa_ft_freq_path: str | None = None
    """(FT training) npz with THIS FT dataset's per-code 'motion_counts' — added to the persistent
    cumulative motion counter (skill_motion_counts) at save time, so the lineage accumulates across FT
    stages. Its codes are EXCLUDED from distill sampling (they're directly trained via the GT BC) unless
    their PRIOR (pre-this-FT) motion count clears vsa_distill_prior_pct. Blank → counter not updated."""
    vsa_distill_prior_pct: float = 20.0
    """Percentile (over the PRIOR motion counter's used codes) below which a THIS-FT code is excluded
    from distill sampling. e.g. 20 → an FT-dataset code is distilled only if its accumulated prior
    motion count is in the top 80% of previously-seen codes (enough old knowledge to be worth
    preserving); nearly-empty prior → skip (nothing to preserve, GT BC handles it)."""
    vsa_gt_severed_bc: bool = False
    """(needs vsa_distill_main_connected) ALSO train the GT skill VLM-SEVERED (pure VSA) against the GT
    ACTION — in addition to the VLM-connected main BC — so the VSA-only path also learns the new task's
    GT execution (not just the connected path). This is a real BC loss (vs the GT action, NOT the
    teacher). No-op unless main_connected is on (else the main BC is already the severed one)."""
    vsa_gt_severed_weight: float = 1.0
    """Weight of the severed GT BC relative to the connected main flow BC."""
    vsa_distill_main_connected: bool = False
    """In a B batch, run the MAIN GT BC forward with the VLM CONNECTED (frozen per freeze_vsa) instead
    of VSA-only — the motor then adapts to the new task WITH the VLM context it has at inference
    (removes the train/inference mismatch). Independent of vsa_distill: on its own it just makes B a
    'train motor VLM-connected' regime (an alternating-freeze with A); with vsa_distill on, the
    distillation still uses the pure VSA-only path to preserve the old repertoire. The VLM already runs
    in B (for the skill head), so this only flips the main forward's attention mask — no extra VLM
    forward."""

    # ── EMA-self distillation (Stage-2 forgetting prep) ──
    # On SEVERED (C) batches only — the standalone/pure-VSA regime where random-skill preservation belongs
    # (connected B trains the VSA to cooperate WITH the VLM; random skills have no matching language) —
    # pin sampled non-GT skills to the model's OWN weight-EMA teacher (a shadow copy updated
    # θ_ema ← α·θ_ema + (1−α)·θ EVERY step, so it still absorbs B's drift). GT gets its sharp BC; other
    # skills are held near their recent-self behaviour → the GT update stays local (anti-forgetting prep)
    # WITHOUT a frozen Stage-1 teacher (the EMA tracks Stage-2's own gains, τ≈1/(1−α) steps behind) and
    # WITHOUT cross-skill contamination (each skill is pinned to ITS own past, not to the GT). Reuses the
    # VSA-distill sampler + velocity machinery; the teacher source is the only change.
    ema_self_distill: bool = False
    """Enable the Stage-2 EMA-self distillation term."""
    ema_self_alpha: float = 0.999
    """Weight-EMA decay α. Effective memory ≈ 1/(1−α) steps (0.999 → ~1k steps behind)."""
    ema_self_weight: float = 0.5
    """Weight of the EMA-self distill loss relative to the main BC (0 → MONITOR-ONLY: logged, no gradient)."""
    ema_self_n_local: int = 2
    """# sampled GT-neighbour skills pinned to the EMA teacher (local geometry preservation)."""
    ema_self_n_global: int = 2
    """# sampled global (frequency/uniform) skills pinned to the EMA teacher (broad repertoire)."""

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
    """FT: also co-train the FSQ terminator on this dataset's GT signals (z←GT code, current-frame
    ONLINE DINO + state → progress + termination), so the terminator adapts to the new task before it
    gates skill transitions at inference. Trained on a DISJOINT graph (GT inputs only) → zero effect
    on the SkillVLA params; only shares the dataloader. Warm-starts from ``fsq_path`` and is exported
    back to an FSQ checkpoint for eval. DINO 토큰은 배치의 현재 프레임 이미지에서 라이브로 계산
    (디스크 precompute 없음; 추론과 동일 경로)."""
    terminator_freeze_vision_encoder: bool | None = None
    """Co-trained terminator vision override. True freezes only its vision encoder; False trains the
    vision encoder with the rest of the terminator. None preserves the freeze mode stored in FSQ.pt."""
    terminator_dino_model_path: str | None = None
    """ONLINE DINO의 로컬 모델 경로 오버라이드 (예: {project_root}/models/dinov3-vits16). FSQ ckpt의
    cfg에는 FSQ를 학습한 머신의 절대경로가 기록되어 있어 서버 이식 시 깨질 수 있음 — 설정하면
    terminator 빌드(fsq_term/fsq_term_train)가 이 경로로 DINO를 lazy 로드한다. None = ckpt 기록 사용."""
    terminator_lr_scale: float = 1.0
    """LR multiplier (× optimizer_lr) for the co-trained terminator params (train_terminator only)."""
    terminator_end_target_sigma: float = 2.0
    """Soft termination target std in FRAMES: target = exp(-(frames_to_end)² / (2σ²)) (Gaussian that
    is 1.0 at the skill's last frame and decays earlier). Mirrors the FSQ terminator's training."""
    terminator_end_pos_weight: float = 1.0
    """BCE positive-class weight for the co-trained terminator's termination head."""

    # (skill_decoder_dino_* precompute 토큰 필드 은퇴 — terminator co-training은 배치의 현재 프레임
    #  이미지를 ONLINE DINO로 직접 토큰화. 디스크 dino.npz/캐시/래퍼 전부 제거.)

    # ── Inference: skill transitions via the frozen FSQ terminator ──
    fsq_path: str | None = None
    """Frozen FSQ checkpoint whose terminator decides skill transitions during closed-loop rollout.
    Stage-0 also loads its frozen action expert directly from this checkpoint."""
    skill_end_mode: str = "termination"
    """Which FSQ signal ends the current skill:
       "termination" → end_prob   >= skill_end_threshold
       "progress"    → progress   >= skill_end_threshold
       "and"         → end_prob >= skill_end_threshold AND progress >= skill_end_progress_threshold
                       (both must hold — guards against early end-prob spikes before the skill is done)
       "or"          → end_prob >= skill_end_threshold OR  progress >= skill_end_progress_threshold
                       (either suffices — ends as soon as the terminator OR the progress gate fires)."""
    skill_end_threshold: float = 0.5
    skill_end_progress_threshold: float = 0.9
    """Progress gate for skill_end_mode="and"/"or": the progress-side threshold (e.g. 0.9). Combined with
    end_prob >= skill_end_threshold via AND ("and") or OR ("or"). Unused in "termination"/"progress"."""
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
    eval_drop_vlm: bool = False
    """EVAL-time VLM dropout: sever the cond→VLM and action→VLM attention edges during rollout —
    exactly a training VSA (B) batch (mask-only, RoPE positions unchanged). The VLM still runs
    standalone for the skill prediction (or GT is injected with use_gt_skill); ONLY the discrete
    skill code crosses to the action side. Used by stage2_eval's eval_dropout probe (VLM-attached
    vs VLM-severed side-by-side of the SAME checkpoint)."""
    eval_drop_vlm_keep_adapters: bool = False
    """EVAL-time, ONLY meaningful with eval_drop_vlm=True: keep ``cond_lora`` active while the real
    VLM is severed. ``expert_lora`` remains active in either case because it is part of the Stage-1 VSA.
    This isolates whether the VLM-independent cond adapter path changes the action output vs the pure
    Stage-1 VSA: keep_adapters=True drops only the real VLM; false also disables cond_lora. No effect
    when eval_drop_vlm=False (the full model runs vlm_lora + cond_lora)."""
    eval_vlm_override_path: str | None = None
    """EVAL-only ablation: a checkpoint whose VLM (the WHOLE PaliGemma tower) OVERWRITES this
    checkpoint's VLM after load, keeping this checkpoint's cond + action-expert (+ skill decoder). FT
    eval sets this to the run's Stage-2 source to run the FT'd motor under the ORIGINAL (un-adapted)
    perception. Blank = keep the loaded VLM."""
    eval_skill_predictor_path: str | None = None
    """EVAL-only external skill source. When set to a Stage-3 checkpoint, its frozen VLM + skill
    adapter/reader/head predict each new discrete skill code. The current policy still owns the action
    route and FSQ terminator; only the code returned at a skill boundary is replaced."""

    def __post_init__(self):
        super().__post_init__()
        if self.pt_stage not in (None, "cond", "skill", "stage0"):
            raise ValueError(f"pt_stage must be None|cond|skill|stage0, got {self.pt_stage!r}.")
        if self.stage3_parent_stage not in {"stage0", "stage2"}:
            raise ValueError(
                f"stage3_parent_stage must be stage0|stage2, got {self.stage3_parent_stage!r}."
            )
        if self.lang_bridge and self.lang_bridge_rank <= 0:
            raise ValueError("lang_bridge=True needs lang_bridge_rank > 0.")
        if self.lang_bridge_lr_scale <= 0.0:
            raise ValueError("lang_bridge_lr_scale must be > 0.")
        if not 0.0 <= self.stage0_vlm_severed_prob < 1.0:
            raise ValueError("stage0_vlm_severed_prob must be in [0, 1).")
        allowed_stage0_components = {
            "vlm", "vlm_vision", "vlm_lora", "lang_bridge",
            "cond", "cond_vision", "cond_lora",
            "expert", "expert_lora", "skill_reader", "skill_head",
        }
        for branch, spec in (
            ("A", self.stage0_a_train_components),
            ("B", self.stage0_b_train_components),
        ):
            components = {part.strip() for part in str(spec).split(",") if part.strip()}
            unknown = components - allowed_stage0_components
            if unknown:
                raise ValueError(f"Unknown Stage-0 {branch} train components: {sorted(unknown)}.")
        if self.pt_stage == "stage0":
            if self.stage0_expert_source not in {"fsq", "pi05_base"}:
                raise ValueError(
                    "stage0_expert_source must be fsq|pi05_base, "
                    f"got {self.stage0_expert_source!r}."
                )
            stage0_residual = bool(getattr(self, "stage0_vlm_residual", False))
            if stage0_residual:
                if self.stage0_expert_source != "pi05_base":
                    raise ValueError(
                        "Renewed Stage-0 VLM residual uses the pi05 action expert. "
                        "Set stage0_expert_source='pi05_base'."
                    )
                alpha_min = float(getattr(self, "stage0_vlm_residual_alpha_min", 0.1))
                alpha_max = float(getattr(self, "stage0_vlm_residual_alpha_max", 0.2))
                init_alpha = float(getattr(self, "stage0_vlm_residual_init_alpha", 0.15))
                if alpha_max <= 0.0:
                    raise ValueError("stage0_vlm_residual_alpha_max must be > 0.")
                if not 0.0 <= alpha_min < init_alpha < alpha_max:
                    raise ValueError(
                        "Stage-0 residual alpha must satisfy 0 <= alpha_min < init_alpha < alpha_max, "
                        f"got {alpha_min}, {init_alpha}, and {alpha_max}."
                    )
                if float(getattr(self, "stage0_vlm_residual_dropout", 0.0)) < 0.0:
                    raise ValueError("stage0_vlm_residual_dropout must be >= 0.")
                if float(getattr(self, "stage0_conditional_loss_weight", 1.0)) < 0.0:
                    raise ValueError("stage0_conditional_loss_weight must be >= 0.")
                if float(getattr(self, "stage0_unconditional_loss_weight", 0.5)) < 0.0:
                    raise ValueError("stage0_unconditional_loss_weight must be >= 0.")
                routing = str(getattr(self, "stage0_gradient_routing", "shared")).strip().lower()
                if routing not in {"shared", "split"}:
                    raise ValueError(
                        "stage0_gradient_routing must be shared|split, "
                        f"got {self.stage0_gradient_routing!r}."
                    )
                if float(getattr(self, "stage0_uncond_skill_start_loss_weight", 1.0)) <= 0.0:
                    raise ValueError("stage0_uncond_skill_start_loss_weight must be > 0.")
                if float(getattr(self, "stage0_uncond_skill_end_loss_weight", 1.0)) <= 0.0:
                    raise ValueError("stage0_uncond_skill_end_loss_weight must be > 0.")
                if bool(getattr(self, "stage0_language_ranking_enabled", False)):
                    if not bool(getattr(self, "attend_language", False)):
                        raise ValueError(
                            "stage0_language_ranking_enabled=True requires attend_language=True."
                        )
                    if float(getattr(self, "stage0_language_ranking_weight", 0.1)) <= 0.0:
                        raise ValueError("stage0_language_ranking_weight must be > 0 when enabled.")
                    if float(getattr(self, "stage0_language_ranking_relative_margin", 0.01)) < 0.0:
                        raise ValueError(
                            "stage0_language_ranking_relative_margin must be >= 0."
                        )
                if not 0.0 <= float(getattr(self, "same_skill_batch_fraction", 0.5)) <= 1.0:
                    raise ValueError("same_skill_batch_fraction must be in [0, 1].")
                if float(getattr(self, "same_skill_progress_temperature", 0.1)) <= 0.0:
                    raise ValueError("same_skill_progress_temperature must be > 0.")
                if bool(getattr(self, "stage0_train_skill_predictor", False)):
                    if not bool(getattr(self, "stage0_skill_predictor_detach_vlm", True)):
                        raise ValueError(
                            "stage0_skill_predictor_detach_vlm must remain True so the predictor "
                            "auxiliary cannot backprop into the action VLM."
                        )
                    if float(getattr(self, "stage0_skill_predictor_weight", 0.5)) <= 0.0:
                        raise ValueError("stage0_skill_predictor_weight must be > 0.")
                    if float(getattr(self, "stage0_skill_predictor_lr_scale", 1.0)) <= 0.0:
                        raise ValueError("stage0_skill_predictor_lr_scale must be > 0.")
            if (not stage0_residual) and (self.stage0_a_drop_vlm or not self.stage0_b_drop_vlm):
                raise ValueError("Stage-0 requires A_drop_vlm=False and B_drop_vlm=True.")
            all_components = {
                part.strip()
                for spec in (self.stage0_a_train_components, self.stage0_b_train_components)
                for part in str(spec).split(",") if part.strip()
            }
            if (not stage0_residual) and "vlm_lora" in all_components and not self.vlm_lora:
                raise ValueError("Stage-0 matrix trains vlm_lora but vlm_lora=False.")
            if (not stage0_residual) and "lang_bridge" in all_components and not self.lang_bridge:
                raise ValueError("Stage-0 matrix trains lang_bridge but lang_bridge=False.")
            if (not stage0_residual) and "expert_lora" in all_components and not self.stage0_expert_lora:
                raise ValueError("Stage-0 matrix trains expert_lora but stage0_expert_lora=False.")
        if self.stage0_wrong_language_weight < 0.0 or self.stage0_wrong_language_margin < 0.0:
            raise ValueError("Stage-0 wrong-language weight and margin must be >= 0.")
        stage0_weights = (
            self.stage0_a_skill_start_loss_weight,
            self.stage0_a_skill_end_loss_weight,
            self.stage0_b_skill_start_loss_weight,
            self.stage0_b_skill_end_loss_weight,
        )
        if any(weight <= 0.0 for weight in stage0_weights):
            raise ValueError(f"All Stage-0 A/B skill loss weights must be > 0, got {stage0_weights}.")
        if self.stage0_expert_lora_lr_scale <= 0.0:
            raise ValueError("stage0_expert_lora_lr_scale must be > 0.")
        extra_read_set = bool(getattr(self, "attend_skill", False))
        if not self.attend_image and not self.attend_language and not extra_read_set:
            raise ValueError(
                "The VLM read-set is empty: image/language and all policy-specific token access "
                "flags are disabled. Use vlm_cond=False / vlm_expert=False for a VLM-blind stream."
            )
        # reader read-set (None = attend_* 상속) 도 비어 있으면 안 됨 — skill 예측기가 읽을 토큰이 없음.
        _rai = self.attend_image if self.reader_attend_image is None else self.reader_attend_image
        _ral = self.attend_language if self.reader_attend_language is None else self.reader_attend_language
        if not _rai and not _ral:
            raise ValueError(
                "reader read-set is empty: reader_attend_image/reader_attend_language (or the inherited "
                "attend_*) are both False — the skill reader would have no VLM tokens to attend.")
