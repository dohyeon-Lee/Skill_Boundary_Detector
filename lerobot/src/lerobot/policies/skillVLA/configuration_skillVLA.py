from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("skill_vla")
@dataclass
class SkillVLAConfig(PI05Config):
    """Stage-2 SkillVLA.

    A VLM (PaliGemma, warm-started from pi05_base) reads the SKILL-START observation
    (3rd + wrist image + the start state discretized into the prompt, pi05-style + language) plus a
    learnable skill-query token, and predicts the current skill (one categorical per FSQ dim). An
    action expert — warm-started from a Stage-1 ``skill_expert`` checkpoint — generates the action
    chunk by flow matching from the CURRENT observation, reading the VLM via PI05-style joint block
    attention (action attends the VLM's image/skill tokens, NOT its language; the VLM never attends
    the action expert).

    Branch is taken from the loaded Stage-1 checkpoint's ``expert_arch``:
      A (joint): the action expert ALSO cross-attends the Stage-1 cond-encoder (current obs + skill),
                 so it reads two prefixes — cond-encoder (fresh every step) + VLM (cached per skill).
      B (fused): the fused action expert reads the VLM; its own image/state/skill stay self-attended.

    Skill flow: the discrete GT skill is teacher-forced into the cond-encoder (A) / expert (B) via the
    Stage-1 skill embedding at train time (the VLM's prediction is used at inference); the VLM's
    skill-query hidden additionally reaches the expert through cross-attention.

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
    """FSQ levels per dim. The VLM skill head predicts ONE categorical per dim (sizes = levels);
    skill vocab = prod(levels). Must match the Stage-1 / FSQ codebook the dataset was built with."""
    skill_loss_weight: float = 0.5
    """λ_skill in ``total = BC + λ_skill * skill_CE``. < 1 keeps the action BC dominant (including the
    BC gradient that flows into the VLM via cross-attention)."""

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

    # ── Inference: skill transitions via the frozen FSQ terminator ──
    fsq_path: str | None = None
    """Frozen FSQ checkpoint whose terminator decides skill transitions during closed-loop rollout.
    Unused at training (skill boundaries come from the dataset)."""
    skill_end_mode: str = "termination"
    """Which FSQ signal ends the current skill: "termination" (prob >= threshold) or "progress"."""
    skill_end_threshold: float = 0.5
    inference_skill_max_length: int = 200
    """Force-advance the skill after this many steps even if the terminator never fires (0 = off)."""
