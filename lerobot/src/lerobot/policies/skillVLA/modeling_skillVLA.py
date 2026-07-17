"""Stage-2 SkillVLA — VLM predicts the skill, an action expert flow-matches the chunk.

A PaliGemma VLM (warm-started from pi05_base) reads the SKILL-START observation (3rd + wrist
image + the start state discretized into the prompt, pi05-style + language) and stays PRISTINE; a
SEPARATE skill reader (learned probe tokens over the VLM's FINAL-layer output, joint concat-KV,
VLM read-only) → ``SkillHead`` predicts the current FSQ skill code (regression over per-dim grid
coords). An action expert — warm-started from a Stage-1 ``skill_expert`` checkpoint — generates the
action chunk by flow matching from the CURRENT observation, reading the VLM via PI05-style joint
block attention.

Architecture (joint): three streams in a one-directional chain VLM → cond → action — a Stage-1
cond-encoder over the CURRENT obs (images + per-dim discretized state), the VLM over the START obs,
and the action expert (action tokens only). cond attends itself + the VLM's image hiddens (NOT its
language tokens unless attend_language — language otherwise reaches cond only indirectly via VLM
image hiddens that attended it); the action attends cond + itself (VLM edge iff vlm_expert). Nothing
attends the action.

Skill flow: the skill (z_q) + progress are prepended to the ACTION stream (ae); the GT code is
teacher-forced at train (cond_skill_source=gt) or the reader's own prediction is fed back via STE
(cond_skill_source=pred); the skill reader's pooled hidden is the prediction signal (skill loss).

loss = BC(flow-matching MSE) + skill_loss_weight · skill_CE.

Warm-start: pi05_base → the VLM (paligemma side only); the Stage-1 checkpoint → the action expert,
the cond-encoder, the expert-side vision encoder, and the image/state/skill projections.

NOTE: closed-loop ``select_action`` (FSQ-terminator-driven skill transitions, skill-start obs
caching) is deferred — see TODO(stage2-inference). ``predict_action_chunk`` works for offline /
dataset eval where the skill-start inputs are supplied in the batch.
"""

from __future__ import annotations

import logging
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn
from transformers import AutoModel
from transformers.models.gemma import modeling_gemma

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.modeling_pi05 import (
    PI05Policy,
    PI05Pytorch,
    create_sinusoidal_pos_embedding,
    get_gemma_config,
    make_att_2d_masks,
    pad_vector,
    resize_with_pad_torch,
)
from lerobot.policies.pi05.lora import (
    NamedLoRALinear,
    inject_named_lora,
    route_plain_to_base,
    set_active_adapters,
    target_names_from_spec,
)
from lerobot.policies.pi_gemma import _gated_residual, layernorm_forward
from lerobot.policies.skill_expert.modeling_skill_expert import (
    _build_gemma,
    _build_siglip_vision_tower,
    _load_raw_state_dict,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    OPENPI_ATTENTION_MASK_VALUE,
)

from .configuration_skillVLA import SkillVLAConfig
from .dataset_skillVLA import (
    SKILL_CODE,
    SKILL_START_IMAGE,
    SKILL_START_STATE,
    SKILL_START_WRIST_IMAGE,
)
from .skill_head import SkillHead
from .skill_reader import SkillReader

log = logging.getLogger(__name__)


# ── Generalized N-stream joint block attention ──────────────────────────────────────────────

def compute_layer_multi(layer_idx, hiddens, layers, attention_mask, position_ids, adarms, rotary_emb):
    """One transformer layer shared across N streams (generalizes pi05's ``compute_layer_complete``).

    Every stream ``i`` has its OWN decoder layer ``layers[i]`` (own q/k/v/o + norms + MLP) and AdaRMS
    cond ``adarms[i]``; all streams must share ``head_dim``/``num_heads``. Per-head q/k/v are
    concatenated along the sequence so a SINGLE attention with ``attention_mask`` (B, 1, T, T) couples
    them, then the result is split back per stream. RoPE uses one shared ``rotary_emb`` over the
    concatenated ``position_ids``.
    """
    query_states, key_states, value_states, gates = [], [], [], []
    for hidden, layer, cond in zip(hiddens, layers, adarms, strict=True):
        h, gate = layernorm_forward(layer.input_layernorm, hidden, cond)
        gates.append(gate)
        shape = (*h.shape[:-1], -1, layer.self_attn.head_dim)
        query_states.append(layer.self_attn.q_proj(h).view(shape).transpose(1, 2))
        key_states.append(layer.self_attn.k_proj(h).view(shape).transpose(1, 2))
        value_states.append(layer.self_attn.v_proj(h).view(shape).transpose(1, 2))
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)

    dummy = torch.zeros(
        query_states.shape[0], query_states.shape[2], query_states.shape[-1],
        device=query_states.device, dtype=query_states.dtype,
    )
    cos, sin = rotary_emb(dummy, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )

    host = layers[0].self_attn
    att_output, _ = modeling_gemma.eager_attention_forward(
        host, query_states, key_states, value_states, attention_mask, host.scaling
    )
    n_heads, head_dim = query_states.shape[1], host.head_dim
    att_output = att_output.reshape(att_output.shape[0], -1, n_heads * head_dim)

    outs = []
    start = 0
    for hidden, layer, cond, gate in zip(hiddens, layers, adarms, gates, strict=True):
        end = start + hidden.shape[1]
        slice_att = att_output[:, start:end]
        if slice_att.dtype != layer.self_attn.o_proj.weight.dtype:
            slice_att = slice_att.to(layer.self_attn.o_proj.weight.dtype)
        out = layer.self_attn.o_proj(slice_att)
        out = _gated_residual(hidden, out, gate)              # first residual
        after = out.clone()
        out, gate2 = layernorm_forward(layer.post_attention_layernorm, out, cond)
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out = out.to(torch.bfloat16)
        out = layer.mlp(out)
        out = _gated_residual(after, out, gate2)              # second residual
        outs.append(out)
        start = end
    return outs


# ── Core model ───────────────────────────────────────────────────────────────────────────────

class SkillVLAPytorch(PI05Pytorch):
    """PI05's PaliGemma (VLM) + Gemma action expert, plus a Stage-1 cond-side and a skill head.

    Inherited from ``PI05Pytorch``: ``paligemma_with_expert`` (``.paligemma`` = VLM, ``.gemma_expert``
    = action expert), ``action_in_proj`` / ``action_out_proj`` / ``time_mlp_*``, and the flow-matching
    noise/time samplers. Added here: the Stage-1 cond-side (vision encoder, image/state/skill
    projections, and a joint-mode cond-encoder), the ``SkillReader`` (probe tokens), and the ``SkillHead``.
    """

    def __init__(self, config: SkillVLAConfig, stage1_config, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)
        self.config = config
        self.stage1_config = stage1_config

        vlm_width = get_gemma_config(config.paligemma_variant).width
        self.expert_width = get_gemma_config(config.action_expert_variant).width

        # ── Stage-1 cond-side (current obs → conditioning tokens) ──
        self._build_cond_side(stage1_config)

        # ── VLM skill prediction: learnable query token (VLM width) + per-dim FSQ head ──
        if math.prod(config.skill_fsq_levels) != stage1_config.skill_vocab_size:
            raise ValueError(
                f"prod(skill_fsq_levels)={math.prod(config.skill_fsq_levels)} must equal the Stage-1 "
                f"skill_vocab_size={stage1_config.skill_vocab_size}."
            )
        # Skill prediction lives OUTSIDE the VLM: a separate JOINT concat-KV reader (N learned probes read
        # the VLM's FINAL-layer output, VLM read-only → pristine) → the FSQ regression head. Freezing the
        # skill decoder for VLM-only finetuning freezes BOTH (see _apply_freezes).
        self.skill_reader = SkillReader(
            vlm_width, depth=config.reader_depth, heads=config.reader_heads,
            num_probes=config.num_reader_tokens)
        self.skill_head = SkillHead(vlm_width, config.skill_fsq_levels,
                                    deadzone_frac=getattr(config, "skill_deadzone_frac", 0.0))

        # ── Stage-1 VSA LoRA: restore its adapter as part of the frozen VSA before adding Stage-2/3 adapters.
        self._has_stage1_expert_lora = False
        self._inject_stage1_expert_lora(stage1_config)

        # ── Multi-adapter LoRA (①②③): named low-rank adapters on the FROZEN VLM LLM / cond encoder. The
        # base stays frozen (from_pretrained routes the pretrained plain weights into `.base`); each adapter
        # is B=0 at init → no-op (model == base at step 0). Selected per-forward via lora.active_adapters().
        # All off by default (the three flags False → this is a no-op, byte-identical to before).
        self._inject_lora_adapters()

        # ── FSQ grid buffers (skill cond token + inference-only terminator) ──
        # Flat FSQ code → z_q uses the FSQ codebook's OWN (little-endian) convention
        # (strides[i]=prod(levels[:i]), z_q = level_id - half), independent of SkillHead's
        # internal mixed-radix — both the terminator and _action_prefix read this geometry.
        levels = config.skill_fsq_levels
        strides = torch.ones(len(levels), dtype=torch.long)
        for i in range(1, len(levels)):
            strides[i] = strides[i - 1] * levels[i - 1]
        self.register_buffer("_fsq_strides", strides, persistent=False)
        self.register_buffer("_fsq_levels", torch.tensor(levels, dtype=torch.long), persistent=False)
        self.register_buffer("_fsq_half", torch.tensor([(lv - 1) / 2.0 for lv in levels], dtype=torch.float32), persistent=False)
        self.fsq_term = None
        # vlm_dropout schedule position (PERSISTENT → a resume continues the anneal where it left off)
        self.register_buffer("_vdrop_step", torch.zeros((), dtype=torch.long), persistent=True)

        # Match pi05's working dtype for the trainable Stage-1-side tokenizers + skill head, so the
        # bf16 expert stream never sees float32 inputs (the vision encoders stay float32 like pi05's
        # vision_tower — their features are cast to the working dtype before the projections).
        # Stage-1-side tokenizers run in the expert's working dtype (Stage-1 trained them in bf16).
        # The pi05-inherited action_in/out_proj + time_mlp stay float32 (pi05 convention): their
        # outputs are cast to the working dtype only at the attention boundary (_action_in/_action_out).
        if str(config.dtype) == "bfloat16":
            for m in (self.image_proj, self.state_proj, self.skill_proj,
                      self.cond_encoder, self.skill_reader, self.skill_head):
                if m is not None:
                    m.to(dtype=torch.bfloat16)

        # FT: a TRAINABLE terminator co-trained on GT signals (built last so the bf16 cast above skips
        # it — it stays float32). Disjoint from the SkillVLA params; warm-starts from config.fsq_path.
        self.fsq_term_train = None
        if getattr(config, "train_terminator", False):
            if not getattr(config, "fsq_path", None):
                raise ValueError("train_terminator=True requires config.fsq_path (the FSQ checkpoint to adapt).")
            self._build_terminator_trainable(config.fsq_path)

        # Continual-learning VSA distillation.
        self._vsa_teacher = None            # lazily built frozen PT teacher (list-wrapped when set)
        vocab = int(stage1_config.skill_vocab_size)

        # PERSISTENT cumulative motion counter (per-code trajectory count, accumulated across the FT
        # lineage). Rides in the checkpoint → each FT stage loads the parent's counter and adds its own
        # dataset's counts at save time. Seeded on the FIRST FT from vsa_distill_freq_path (the PT
        # histogram) when the loaded buffer is empty. finalize_motion_counter() (lazy, first train
        # forward) splits it into the PRIOR (used by the sampler) + the updated buffer (saved).
        self.register_buffer("skill_motion_counts", torch.zeros(vocab, dtype=torch.float32), persistent=True)

        def _load_motion(path):
            if not path or not str(path).strip() or not Path(str(path)).is_file():
                return None                        # tolerant: not built → seed/counter simply skipped
            z = np.load(str(path))
            arr = (z["motion_counts"] if "motion_counts" in z else z["counts"]).astype("float32")
            if arr.shape[0] != vocab:
                raise ValueError(f"freq npz len {arr.shape[0]} != skill_vocab {vocab} ({path})")
            return torch.from_numpy(arr)
        self._motion_seed = _load_motion(getattr(config, "vsa_distill_freq_path", None))   # PT histogram (first-FT seed)
        self._motion_ft = _load_motion(getattr(config, "vsa_ft_freq_path", None))          # this FT's contribution
        self._motion_finalized = False
        self._prior_motion = None          # cumulative BEFORE this FT (sampler weighting)
        self._prior_threshold = 0.0        # abs threshold = percentile(prior[prior>0], vsa_distill_prior_pct)

    # ── construction helpers ──
    def _build_cond_side(self, s1) -> None:
        """Stage-1 vision + projections (+ joint cond-encoder). Mirrors ``SkillExpertPytorch``
        so a Stage-1 checkpoint warm-starts these 1:1 (see ``_remap_stage1_keys``)."""
        self.vision_backbone = s1.vision_backbone
        self.dino = self.siglip = None
        self.n_register = 0
        if s1.vision_backbone == "dino":
            self.dino = AutoModel.from_pretrained(s1.dino_model_path)
            vis_dim = int(self.dino.config.hidden_size)
            self.n_register = int(getattr(self.dino.config, "num_register_tokens", 0))
            self.vision_image_size = s1.dino_image_size
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        elif s1.vision_backbone == "siglip":
            self.siglip = _build_siglip_vision_tower(s1.siglip_image_size)
            vis_dim = int(self.siglip.config.hidden_size)
            self.vision_image_size = s1.siglip_image_size
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        else:
            raise ValueError(f"vision_backbone must be 'dino' or 'siglip', got {s1.vision_backbone!r}")
        self.register_buffer("_img_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_img_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

        self.image_proj = nn.Linear(vis_dim, self.expert_width)
        # State: QUANTILE-normalized CONTINUOUS vector → Linear → rides the action expert's flow-time
        # AdaRMS (mirrors Stage-1's state_cond_mode). State is NOT a cond-stream token anymore — buried
        # among image tokens it was starved (see Stage-1 input_probe). cond stays image-ONLY.
        self.state_proj = nn.Linear(s1.max_state_dim, self.expert_width)
        # Skill cond token mirrors Stage-1: flat code → normalized z_q → Linear (constant within a
        # skill) — see _skill_token_from_z. (The skill-progress token was removed.)
        self.skill_proj = nn.Linear(len(s1.skill_fsq_levels), self.expert_width)

        # ae: the cond-encoder encodes the SCENE ONLY (no skill, no adaLN); the skill is injected as a
        # prefix token in the action-expert stream (see _action_prefix). cond stays skill-blind → a
        # clean obs / VLM-grounded channel. use_adarms=False matches the Stage-1 ae cond-encoder.
        variant = s1.cond_encoder_variant or s1.action_expert_variant
        self.cond_encoder = _build_gemma(variant, use_adarms=False)

    # ── module shortcuts ──
    @property
    def _vlm(self):
        return self.paligemma_with_expert.paligemma.model.language_model

    @property
    def _expert(self):
        return self.paligemma_with_expert.gemma_expert.model

    def _resolved_pt_stage(self) -> str | None:
        """config.pt_stage, with the retired regime_probs_pt auto-mapped for old checkpoints/snapshots:
        COND-only probabilities ([0,1]) → "cond" (that is what those runs trained); anything mixed has
        no sequential equivalent → explicit error pointing at pt_stage."""
        stage = getattr(self.config, "pt_stage", None)
        if stage:
            if stage not in ("cond", "skill"):
                raise ValueError(f"pt_stage must be 'cond' (Stage 2) or 'skill' (Stage 3), got {stage!r}.")
            return stage
        legacy = getattr(self.config, "regime_probs_pt", None)
        if legacy:
            probs = [float(x) for x in legacy]
            if len(probs) == 2 and probs[0] == 0.0 and probs[1] > 0.0:
                return "cond"
            raise ValueError(f"regime_probs_pt={probs} is retired (interleaved mixing removed) — "
                             "set pt_stage='cond' or 'skill' instead.")
        return None

    def _inject_stage1_expert_lora(self, stage1_config) -> None:
        """Recreate Stage-1's expert adapter under ``expert_lora`` so it can be loaded and frozen intact.

        Stage-1 LoRA is part of the VSA checkpoint, not a Stage-2 adaptation knob. Its target set and
        rank therefore come from the Stage-1 config, including optional action_in/action_out projections.
        """
        if not bool(getattr(stage1_config, "lora_expert", False)):
            return
        names = target_names_from_spec(str(getattr(stage1_config, "lora_targets", "q,k,v,o")))
        action_names = names & {"action_in_proj", "action_out_proj"}
        gemma_names = names - action_names
        rank = int(getattr(stage1_config, "lora_rank", 8))
        alpha_raw = getattr(stage1_config, "lora_alpha", 2 * rank)
        alpha = 2.0 * rank if str(alpha_raw).strip().lower() in ("", "auto") else float(alpha_raw)
        dropout = float(getattr(stage1_config, "lora_dropout", 0.0))
        count = 0
        if gemma_names:
            count += inject_named_lora(self._expert, gemma_names, "expert_lora", rank, alpha, dropout)
        if action_names:
            count += inject_named_lora(self, action_names, "expert_lora", rank, alpha, dropout)
        if count == 0:
            raise ValueError(
                "Stage-1 lora_expert=True but no expert Linears matched "
                f"lora_targets={getattr(stage1_config, 'lora_targets', None)!r}."
            )
        self._has_stage1_expert_lora = True
        print(
            "[skillVLA Stage-1 LoRA] "
            f"expert_lora@frozen-VSA={count} r={rank} alpha={alpha} "
            f"dropout={dropout} targets={sorted(names)}"
        )

    def _active_adapters(self, adapters=()) -> set[str]:
        """Always retain the Stage-1 VSA adapter; add the view-specific Stage-2/3 adapters."""
        active = set(adapters)
        if self._has_stage1_expert_lora:
            active.add("expert_lora")
        return active

    def _inject_lora_adapters(self) -> None:
        """Inject the new Stage-2/3 LoRAs: skill on VLM, vlm_lora on VLM, cond_lora on cond encoder.

        The expert is deliberately absent here: a Stage-1 expert_lora is restored above as an immutable
        VSA component, and SkillVLA does not create a new trainable expert adapter.
        """
        c = self.config
        # The two legacy flag names are read only to load pre-rename Stage-2 checkpoints. New configs use
        # vlm_lora / cond_lora exclusively.
        use_vlm_lora = bool(getattr(c, "vlm_lora", False) or getattr(c, "lora_cond_vlm", False))
        use_cond_lora = bool(getattr(c, "cond_lora", False) or getattr(c, "lora_cond_bridge", False))
        if not (getattr(c, "lora_skill", False) or use_vlm_lora or use_cond_lora):
            return
        names = target_names_from_spec(getattr(c, "lora_targets", "q,k,v,o"))
        r = int(getattr(c, "lora_rank", 8))
        alpha = float(getattr(c, "lora_alpha", 16.0))
        drop = float(getattr(c, "lora_dropout", 0.0))
        parts = []
        if getattr(c, "lora_skill", False):
            n = inject_named_lora(self._vlm, names, "skill", r, alpha, drop)
            parts.append(f"skill@VLM-LLM={n}")
        if use_vlm_lora:
            n = inject_named_lora(self._vlm, names, "vlm_lora", r, alpha, drop)
            parts.append(f"vlm_lora@VLM-LLM={n}")
        if use_cond_lora:
            n = inject_named_lora(self.cond_encoder, names, "cond_lora", r, alpha, drop)
            parts.append(f"cond_lora@cond-enc={n}")
        print(f"[skillVLA LoRA] r={r} alpha={alpha} dropout={drop} targets={sorted(names)} → "
              + ", ".join(parts))

    # ── CFG per-regime freeze (only active when config.vlm_dropout_p > 0) ──
    def _regime_groups(self) -> dict:
        """Module groups governed by freeze_vlm_vsa_* / freeze_vsa_* — THE freeze unit for expert/cond/vlm/
        vlm_vision (per-batch toggle when vlm_dropout_p>0; the A dict applies STATICALLY when p=0, since
        every batch is then a VLM-present A batch). cond = the WHOLE current-obs pipeline (vision encoder →
        image_proj → cond-encoder) as one unit. Only the skill decoder (skill_reader + skill_head) is NOT
        here — it follows its own static flag regardless of regime (skill supervised every batch)."""
        cond_vision = self.dino if self.vision_backbone == "dino" else self.siglip
        return {
            "expert": [self._expert, self.action_in_proj, self.action_out_proj, self.time_mlp_in, self.time_mlp_out],
            "cond": [self.image_proj, self.cond_encoder],       # cond Gemma + image_proj (vision split out)
            "cond_vision": [cond_vision],                       # DINO/SigLIP for cond — separately freezable
            "llm": [self._vlm],       # the VLM's Gemma LLM trunk ONLY (vision tower = vlm_vision below)
            "vlm_vision": [self.paligemma_with_expert.paligemma.model.vision_tower],
        }

    def named_component_params(self) -> dict:
        """The 5 trainable-component groups for per-component update tracking (train-time drift graph):
        which part is being intensively updated. Finer than _regime_groups — the cond pipeline is split
        into its Gemma cond-encoder+image_proj ("cond") and its DINO/SigLIP vision tower
        ("cond_vision_encoder"). Returns {group: [(param_name, param), ...]} over ALL params (frozen
        groups then simply show ~0 drift, which confirms the freeze)."""
        cond_vision = self.dino if self.vision_backbone == "dino" else self.siglip
        groups = {
            "llm": [self._vlm],                                              # VLM Gemma trunk
            "vlm_vision": [self.paligemma_with_expert.paligemma.model.vision_tower],
            "cond": [self.cond_encoder, self.image_proj],                    # cond Gemma + image_proj
            "cond_vision_encoder": [cond_vision],                           # DINO/SigLIP for cond
            "action_expert": [self._expert, self.action_in_proj, self.action_out_proj,
                              self.time_mlp_in, self.time_mlp_out, self.state_proj, self.skill_proj],
        }
        out: dict = {}
        for name, mods in groups.items():
            plist = []
            for i, mod in enumerate(mods):
                if mod is None:
                    continue
                for pn, p in mod.named_parameters():
                    plist.append((f"{name}.{i}.{pn}", p))
            out[name] = plist
        return out

    @staticmethod
    def _set_requires_grad(modules, flag: bool) -> None:
        for mod in modules:
            if mod is None:
                continue
            for p in mod.parameters():
                p.requires_grad_(flag)

    @property
    def _wdtype(self) -> torch.dtype:
        # The VSA-distill teacher drops its (never-run) VLM layers → fall back to the cond encoder,
        # which is cast to the same working dtype in __init__.
        layers = self._vlm.layers
        ref = layers[0] if len(layers) else self.cond_encoder.model.layers[0]
        return ref.self_attn.q_proj.weight.dtype

    # ── tokenization ──
    def _image_features(self, image: Tensor) -> Tensor:
        """Stage-1 vision: image (B,3,H,W) in [0,1] → (B, n_tokens, vis_dim)."""
        x = image.to(dtype=torch.float32)
        x = F.interpolate(x, size=(self.vision_image_size, self.vision_image_size),
                          mode="bilinear", align_corners=False)
        x = (x - self._img_mean.float()) / self._img_std.float()
        if self.vision_backbone == "dino":
            x = x.to(dtype=next(self.dino.parameters()).dtype)
            out = self.dino(x).last_hidden_state
            return torch.cat([out[:, :1, :], out[:, 1 + self.n_register :, :]], dim=1)
        x = x.to(dtype=next(self.siglip.parameters()).dtype)
        return self.siglip(pixel_values=x).last_hidden_state

    # ── Stage-1 state_cond_mode (read from the embedded Stage-1 config so Stage-2 auto-matches the
    #    Stage-1 expert it warm-starts from). state: the skill rides the action prefix, AdaRMS=time+state
    #    (image-dominant cond → stage-2 language room). state_skill: state+skill BOTH ride the AdaRMS,
    #    NO prefix (strongest skill influence).
    @property
    def _state_cond_mode(self) -> str:
        return getattr(self.stage1_config, "state_cond_mode", "state_skill")

    def _cond_tokens(self, images: list[Tensor]) -> Tensor:
        """Scene cond tokens → (B, M, expert_width): [img1 patches, img2 patches]. IMAGE-ONLY — state
        moved to the AdaRMS (_state_cond) and the skill rides the action prefix / AdaRMS by mode."""
        tokens = [self.image_proj(self._image_features(img).to(self._wdtype)) for img in images]
        return torch.cat(tokens, dim=1)

    def _state_cond(self, state: Tensor) -> Tensor:
        """(B, state_dim) QUANTILE-normalized state → (B, expert_width) AdaRMS conditioning vector.
        Continuous Linear (padded to max_state_dim), always on — state is un-droppable (mirrors Stage-1)."""
        d = self.stage1_config.max_state_dim
        s = state.to(torch.float32)
        if s.shape[-1] < d:
            s = F.pad(s, (0, d - s.shape[-1]))
        return self.state_proj(s[..., :d].to(self._wdtype))

    def _skill_token_from_z(self, zq: Tensor) -> Tensor:
        """Normalized z_q ∈ [-1, 1]^D → Linear → (B, 1, expert_width) skill token. The z may be the
        GT code's grid coord or the VLM's STE-rounded prediction (cond_skill_source=pred)."""
        return self.skill_proj(zq.to(self._wdtype)).unsqueeze(1)

    def _action_prefix_from_z(self, zq: Tensor) -> Tensor | None:
        """Action-stream prefix from a normalized skill z (GT grid coord or the VLM's STE-rounded
        prediction). state mode → the skill token; state_skill mode → None (the skill rides the AdaRMS
        instead). See ``_expert_cond`` / ``_action_prefix``."""
        if self._state_cond_mode == "state_skill":
            return None
        return self._skill_token_from_z(zq)                              # (B, 1, expert_width)

    def _action_prefix(self, skill_code: Tensor) -> Tensor | None:
        """The skill token prepended to the action-expert stream (state mode) or None (state_skill).
        ae injects the skill on the action stream so cond stays skill-blind (Stage-1)."""
        return self._action_prefix_from_z(self._code_to_z(skill_code) / self._fsq_half[None, :])

    def _expert_cond_from_z(self, time: Tensor, state: Tensor, zq: Tensor) -> Tensor:
        """The action expert's AdaRMS conditioning → (B, expert_width). time + state ALWAYS; for
        state_skill ALSO + skill (so the skill modulates via AdaRMS, not a prefix). Mirrors Stage-1."""
        c = self._time_cond(time) + self._state_cond(state)
        if self._state_cond_mode == "state_skill":
            c = c + self.skill_proj(zq.to(self._wdtype))
        return c

    def _expert_cond(self, time: Tensor, state: Tensor, skill_code: Tensor) -> Tensor:
        return self._expert_cond_from_z(time, state, self._code_to_z(skill_code) / self._fsq_half[None, :])

    def _vlm_tokens(
        self, start_images: list[Tensor], lang_tokens: Tensor, lang_masks: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """VLM prefix [start imgs, language] → (embeds (B,nv,W), pad (B,nv), xattn_block (nv,)).

        The VLM stays PRISTINE — no skill-query token is appended; skill prediction is a separate reader
        over this prefix's FINAL-layer output (see SkillReader). ``xattn_block`` marks VLM tokens the
        cond/expert/reader streams must NOT attend: the language sub-block UNLESS attend_language=True.
        So by default they read VLM IMAGE tokens only (visual grounding)."""
        embs, pad, is_lang = [], [], []
        for img in start_images:
            emb = self.paligemma_with_expert.embed_image(img)
            n = emb.shape[1]
            embs.append(emb)
            pad.append(torch.ones(emb.shape[0], n, dtype=torch.bool, device=emb.device))
            is_lang += [False] * n

        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad.append(lang_masks.to(dtype=torch.bool))
        is_lang += [True] * lang_emb.shape[1]

        embeds = torch.cat([e.to(self._wdtype) for e in embs], dim=1)  # img feats are f32; unify to wdtype
        pad = torch.cat(pad, dim=1)
        is_lang = torch.tensor(is_lang, dtype=torch.bool, device=embeds.device)
        # VLM tokens the cond/expert/reader streams must NOT attend — the complement of the read-set picked
        # by attend_image / attend_language. This gates only DOWNSTREAM readers; the VLM's OWN self-attention
        # is unaffected (it always processes images+language), and positions are independent (_joint_positions).
        #   image-only (default): exclude language → forces visual grounding
        #   image+language:       exclude nothing
        #   language-only:        exclude images → reader/cond/action read the VLM's language hiddens only
        xattn_block = torch.zeros_like(is_lang)
        if not bool(getattr(self.config, "attend_language", False)):
            xattn_block = xattn_block | is_lang        # exclude LANGUAGE tokens
        if not bool(getattr(self.config, "attend_image", True)):
            xattn_block = xattn_block | ~is_lang       # exclude IMAGE tokens (language-only)
        # 같은 forward 안에서 reader 전용 read-set(_reader_xattn_block)을 재계산할 수 있게 스냅샷 —
        # cond conduit는 lang-only로 조여도 skill reader는 이미지를 계속 보게 분리(reader_attend_*).
        self._vlm_is_lang = is_lang
        return embeds, pad, xattn_block

    def _time_cond(self, timestep: Tensor) -> Tensor:
        # float32 throughout: time_mlp stays float32 and its output is the expert's AdaRMS cond,
        # whose norm dense (pi05 keeps it float32) requires a float32 condition.
        t = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features,
            min_period=self.config.min_period, max_period=self.config.max_period, device=timestep.device,
        ).to(torch.float32)
        t = F.silu(self.time_mlp_in(t))
        return F.silu(self.time_mlp_out(t))

    def _action_in(self, x_t: Tensor) -> Tensor:
        """pi05: float32 action_in_proj → cast to the working dtype for the (bf16) attention stream."""
        return self.action_in_proj(x_t.to(torch.float32)).to(self._wdtype)

    def _action_out(self, action_hidden: Tensor) -> Tensor:
        """pi05: float32 action_out_proj on a float32 hidden → float32 flow velocity."""
        return self.action_out_proj(action_hidden.to(torch.float32))

    # ── attention masks ──
    def _mask_branch_A(
        self, nc: int, vlm_pad: Tensor, vlm_xattn_block: Tensor, na: int, drop_vlm: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Branch-A (B,1,T,T) additive mask + the (B,T) validity/pad vector. PHYSICAL stream order is
        [cond, vlm, action] (this mask's row/col layout); the RoPE POSITIONS, however, place the VLM
        first — see _joint_positions. One-directional chain VLM → cond → action: the VLM reads only
        itself; cond reads itself + the VLM tokens it is allowed (~vlm_xattn_block: the image/language
        read-set picked by attend_image/attend_language) (the VLM is pristine — no skill-query token;
        skill reaches the action via the FSQ skill vector / AdaRMS, and is read separately by the
        post-VLM skill reader). The action stream is prefix(n_prefix) + action×K, where n_prefix is
        the [skill(, progress)] prefix in `state` mode or 0 in `state_skill` (skill/progress → AdaRMS):
        the conditioning PREFIX is read by the action tokens and READS COND (scene) + itself — exactly
        Stage-1 _run_joint's block mask (the skill token is scene-contextualized over all layers; the
        frozen expert was TRAINED on that distribution) — but never the VLM / action tokens.
        The ACTION tokens ALWAYS attend cond/scene; they ALSO get a
        DIRECT VLM edge iff vlm_expert — and that edge reuses the SAME ~vlm_xattn_block as cond
        (so per the attend_image/attend_language read-set). Without vlm_expert there is no
        VLM edge and the VLM reaches the action only via cond.
        With n_prefix=0 the prefix→prefix block is empty. Nothing attends the action tokens."""
        bsize, nv = vlm_pad.shape
        device = vlm_pad.device
        total = nc + nv + na
        n_prefix = na - self.config.chunk_size                       # [skill, progress] prefix length
        pa = nc + nv                                                 # action-stream start
        pf1 = pa + n_prefix                                          # prefix end / action-token start
        allow = torch.zeros(bsize, total, total, dtype=torch.bool, device=device)
        allow[:, :nc, :nc] = True                                    # cond block
        if self.config.vlm_cond and not drop_vlm:                    # cond → VLM edge (severed on a CFG B batch)
            allow[:, :nc, nc : nc + nv] = (~vlm_xattn_block)[None, None, :]  # cond → vlm tokens: per the attend_image/attend_language read-set
        allow[:, nc : nc + nv, nc : nc + nv] = True                  # vlm block (self-attn kept even when dropped → skill head still supervised)
        allow[:, pa:pf1, pa:pf1] = True                              # prefix self (⊥ VLM, ⊥ action)
        if self.config.cond_expert:                                  # cond → expert edge (action reads the scene)
            allow[:, pa:pf1, :nc] = True                             # prefix → cond: skill 토큰이 장면을 읽음 (= Stage-1)
            allow[:, pf1:, :nc] = True
        allow[:, pf1:, pa:] = True                                   # action → prefix + action
        if self.config.vlm_expert and not drop_vlm:                  # VLM → expert direct edge (severed on a CFG B batch)
            allow[:, pf1:, nc : nc + nv] = (~vlm_xattn_block)[None, None, :]
        col_valid = torch.cat(
            [torch.ones(bsize, nc, dtype=torch.bool, device=device), vlm_pad,
             torch.ones(bsize, na, dtype=torch.bool, device=device)], dim=1)
        allow = allow & col_valid[:, None, :]
        att_4d = torch.where(allow[:, None], 0.0, OPENPI_ATTENTION_MASK_VALUE)
        return att_4d, col_valid

    # ── joint stream runners ──
    def _run_streams(self, hiddens, layers_per_stream, adarms, att_4d, position_ids, collect_idx=None):
        """Run the shared transformer over the streams (pre-final-norm hiddens out). All streams
        share the VLM's RoPE and have equal depth (gemma_*: 18 layers). When gradient checkpointing
        is on (training), each layer is recomputed in backward instead of stored (memory↓, ~25% slower)
        — the inference cache paths are @torch.no_grad so they are unaffected.
        collect_idx: if set, stash stream[collect_idx]'s per-layer (pre-final-norm) hidden into
        self._collected_layers (list, len = depth) — used for skill_reader_all_layers."""
        hiddens = [h.to(self._wdtype) for h in hiddens]
        rotary = self._vlm.rotary_emb
        use_ckpt = getattr(self, "gradient_checkpointing_enabled", False) and self.training
        collected = [] if collect_idx is not None else None
        for layer_idx in range(len(layers_per_stream[0])):
            layers = [ls[layer_idx] for ls in layers_per_stream]
            if use_ckpt:
                hiddens = torch.utils.checkpoint.checkpoint(
                    compute_layer_multi, layer_idx, hiddens, layers, att_4d, position_ids, adarms, rotary,
                    use_reentrant=False, preserve_rng_state=False,
                )
            else:
                hiddens = compute_layer_multi(layer_idx, hiddens, layers, att_4d, position_ids, adarms, rotary)
            if collected is not None:
                collected.append(hiddens[collect_idx])
        if collected is not None:
            self._collected_layers = collected
        return hiddens

    def _joint_positions(self, vlm_pad: Tensor, nc: int, na: int) -> tuple[Tensor, Tensor, Tensor]:
        """RoPE positions for the joint layout, with the VLM placed FIRST in position space — VLM at
        [0..nv), then cond, then action — even though the PHYSICAL stream order stays [cond, vlm, action]
        (attention is set-based, so column order is immaterial once each key carries its RoPE position).
        Two payoffs:
          (1) the VLM shares the SAME [0..) frame as the standalone skill encode (_vlm_prefix_out /
              predict_skill_code) → the skill reader sees identical VLM hiddens at train & inference
              (no RoPE-frame skew between the gt-mode joint read and the pred/inference standalone read);
          (2) cond and action stay ADJACENT with the SAME cond→action relative positions as the Stage-1
              skill_expert ([cond, action]) — so the warm-started (often frozen) action expert sees the
              relative positions it was trained on (the +nv the VLM would otherwise inject cancels).
        Returns (vlm_pos, cond_pos, action_pos), each (B, ·)."""
        dev = vlm_pad.device
        nv_valid = vlm_pad.sum(dim=1, keepdim=True)                    # (B,1) valid VLM length per sample
        vlm_pos = torch.cumsum(vlm_pad, dim=1) - 1                     # [0..nv_valid-1] (padded repeats, masked out)
        cond_pos = nv_valid + torch.arange(nc, device=dev)[None, :]    # cond right after the VLM
        action_pos = nv_valid + nc + torch.arange(na, device=dev)[None, :]
        return vlm_pos, cond_pos, action_pos

    def _joint_forward_A(self, cond_tokens, vlm_embeds, vlm_pad, vlm_xattn_block, action_tokens, expert_cond,
                         drop_vlm=False, collect_vlm_layers=False):
        nc, na = cond_tokens.shape[1], action_tokens.shape[1]
        att_4d, _ = self._mask_branch_A(nc, vlm_pad, vlm_xattn_block, na, drop_vlm=drop_vlm)
        vlm_pos, cond_pos, action_pos = self._joint_positions(vlm_pad, nc, na)
        position_ids = torch.cat([cond_pos, vlm_pos, action_pos], dim=1)   # PHYSICAL order [cond, vlm, action]
        layers_per_stream = [self.cond_encoder.model.layers, self._vlm.layers, self._expert.layers]
        # cond/vlm: plain RMSNorm; action ← AdaRMS(expert_cond = time + state [+ skill + progress for
        # state_skill]). In `state` mode skill/progress ride the action prefix instead (see _action_prefix).
        adarms = [None, None, expert_cond]
        outs = self._run_streams([cond_tokens, vlm_embeds, action_tokens], layers_per_stream, adarms, att_4d,
                                 position_ids, collect_idx=(1 if collect_vlm_layers else None))   # 1 = VLM stream
        cond_out, vlm_out, action_out = outs
        cond_out, _ = layernorm_forward(self.cond_encoder.model.norm, cond_out, None)
        vlm_out, _ = layernorm_forward(self._vlm.norm, vlm_out, None)
        action_out, _ = layernorm_forward(self._expert.norm, action_out, expert_cond)
        if collect_vlm_layers:   # each captured layer normed by the VLM final norm → (B, L, nv, W)
            self._vlm_all_layers = torch.stack(
                [layernorm_forward(self._vlm.norm, h, None)[0] for h in self._collected_layers], dim=1)
        return vlm_out, action_out

    def _joint_forward(self, cond_tokens, vlm_embeds, vlm_pad, vlm_xattn_block, action_tokens, expert_cond,
                       drop_vlm=False, collect_vlm_layers=False):
        """Returns (vlm_out, action_hidden) where action_hidden is the action-CHUNK only — the
        skill/progress prefix (state mode) is dropped by the final slice (no-op in state_skill mode).
        drop_vlm severs cond→VLM / action→VLM for a CFG dropout batch (Stage-1 form). collect_vlm_layers
        stashes the per-layer VLM stack in self._vlm_all_layers (skill_reader_all_layers)."""
        vlm_out, action_out = self._joint_forward_A(
            cond_tokens, vlm_embeds, vlm_pad, vlm_xattn_block, action_tokens, expert_cond,
            drop_vlm=drop_vlm, collect_vlm_layers=collect_vlm_layers)
        return vlm_out, action_out[:, -self.config.chunk_size :]

    def _vsa_action_hidden(self, cond_tokens: Tensor, action_tokens: Tensor, expert_cond: Tensor) -> Tensor:
        """VSA-only (VLM-severed) action forward: cond + action streams ONLY, no VLM stream. In a
        drop_vlm batch the action is VLM-independent (cond→VLM/action→VLM severed), and RoPE is relative
        so dropping the per-sample VLM position offset is a no-op → this reproduces
        _joint_forward(drop_vlm=True)'s action_hidden exactly, but WITHOUT running the VLM (cheap; the
        teacher needs no VLM). Used for the continual-learning VSA distillation (sampled skills)."""
        bsize, nc = cond_tokens.shape[0], cond_tokens.shape[1]
        na = action_tokens.shape[1]
        device = cond_tokens.device
        n_prefix = na - self.config.chunk_size          # [skill(,progress)] prefix (state mode) or 0
        total = nc + na
        pf1 = nc + n_prefix                              # action-token start (prefix at [nc, pf1))
        allow = torch.zeros(bsize, total, total, dtype=torch.bool, device=device)
        allow[:, :nc, :nc] = True                        # cond self (cond→VLM severed = absent here)
        if n_prefix:
            allow[:, nc:pf1, nc:pf1] = True              # prefix self (⊥ action)
        if self.config.cond_expert:
            if n_prefix:
                allow[:, nc:pf1, :nc] = True             # prefix → cond: skill 토큰이 장면을 읽음 (= Stage-1)
            allow[:, pf1:, :nc] = True                   # action → cond (scene)
        allow[:, pf1:, nc:] = True                       # action → prefix + action
        att_4d = torch.where(allow[:, None], 0.0, OPENPI_ATTENTION_MASK_VALUE)
        # cond [0..nc), action [nc..nc+na): the SAME cond→action relative offset (nc) as _joint_positions
        # (which places them after the VLM); relative-RoPE makes the shared offset immaterial.
        cond_pos = torch.arange(nc, device=device)[None, :].expand(bsize, -1)
        action_pos = (nc + torch.arange(na, device=device))[None, :].expand(bsize, -1)
        position_ids = torch.cat([cond_pos, action_pos], dim=1)
        layers = [self.cond_encoder.model.layers, self._expert.layers]
        cond_out, action_out = self._run_streams(
            [cond_tokens, action_tokens], layers, [None, expert_cond], att_4d, position_ids)
        action_out, _ = layernorm_forward(self._expert.norm, action_out, expert_cond)
        return action_out[:, -self.config.chunk_size:]

    def _vsa_velocity(self, cond_tokens: Tensor, x_t: Tensor, time: Tensor, state: Tensor,
                      skill_zq: Tensor) -> Tensor:
        """VSA (VLM-severed) predicted flow velocity for a given normalized skill z_q, reusing the batch's
        cond_tokens/x_t/time/state. skill_zq is (B, D) in [-1,1] (already ÷ _fsq_half)."""
        action_tokens = self._action_in(x_t)
        prefix = self._action_prefix_from_z(skill_zq)
        if prefix is not None:
            action_tokens = torch.cat([prefix, action_tokens], dim=1)
        expert_cond = self._expert_cond_from_z(time, state, skill_zq)
        return self._action_out(self._vsa_action_hidden(cond_tokens, action_tokens, expert_cond))

    # ── continual-learning VSA distillation (FT anti-forgetting) ──
    def _code_multi_index(self, code: Tensor) -> Tensor:
        """Flat FSQ code (N,) → per-dim grid index (N, D)."""
        idx = code.view(-1, 1).long()
        return torch.div(idx, self._fsq_strides[None, :], rounding_mode="floor") % self._fsq_levels[None, :]

    def _multi_index_code(self, mi: Tensor) -> Tensor:
        """Per-dim grid index (N, D) → flat FSQ code (N,)."""
        return (mi * self._fsq_strides[None, :]).sum(dim=1)

    def finalize_motion_counter(self) -> None:
        """Split the loaded cumulative counter into PRIOR (pre-this-FT, used by the sampler) + update the
        persistent buffer to prior+this-FT (saved → next stage's prior). Lazy (first train forward), so it
        runs AFTER from_pretrained's load; a no-op in eval (never called → buffer unchanged)."""
        if self._motion_finalized:
            return
        dev = self.skill_motion_counts.device
        loaded = self.skill_motion_counts.detach()
        # PRIOR = the loaded cumulative counter; if empty (first FT, Stage-2 ckpt has none), seed it.
        prior = loaded.clone()
        if float(prior.sum()) <= 0.0 and self._motion_seed is not None:
            prior = self._motion_seed.to(dev)
        self._prior_motion = prior
        # absolute threshold from the percentile of the prior's USED codes
        pos = prior[prior > 0]
        pct = float(getattr(self.config, "vsa_distill_prior_pct", 20.0))
        self._prior_threshold = float(torch.quantile(pos, pct / 100.0)) if pos.numel() else 0.0
        # UPDATE the persistent buffer (for saving): prior + this FT's motion counts
        if self._motion_ft is not None:
            self.skill_motion_counts.copy_((prior + self._motion_ft.to(dev)).float())
        else:
            self.skill_motion_counts.copy_(prior.float())
        self._motion_finalized = True

    def _sample_distill_codes(self, gt: Tensor, n_local: int | None = None, n_global: int | None = None) -> Tensor:
        """Per-sample sampled skills for the distillation, (B, n_local + n_global):
          - n_local : FSQ-grid neighbours of GT (±radius per dim), EXCLUDING GT — local geometry.
          - n_global: drawn from the PRIOR motion counter, EXCLUDING GT + its neighbourhood — the
                      cross-context repertoire (where forgetting concentrates).
        Both pools ALSO exclude codes the CURRENT FT dataset uses (they're trained directly via GT BC),
        UNLESS such a code's PRIOR motion count clears vsa_distill_prior_pct (enough accumulated old
        knowledge to be worth preserving). Codes with (near-)empty prior are naturally down-weighted."""
        c = self.config
        n_local = int(c.vsa_distill_n_local) if n_local is None else int(n_local)
        n_global = int(c.vsa_distill_n_global) if n_global is None else int(n_global)
        B, dev = gt.shape[0], gt.device
        vocab = int(self.stage1_config.skill_vocab_size)
        r = int(c.vsa_distill_neighbor_radius)
        D = self._fsq_levels.shape[0]
        # neighbourhood: all (2r+1)^D offsets, clamped into each dim's grid → codes (B, G)
        rng = torch.arange(-r, r + 1, device=dev)
        offs = torch.stack(torch.meshgrid([rng] * D, indexing="ij"), dim=-1).reshape(-1, D)  # (G, D)
        mi = self._code_multi_index(gt)                                    # (B, D)
        neigh_mi = (mi[:, None, :] + offs[None, :, :]).clamp_min(0)
        neigh_mi = torch.minimum(neigh_mi, (self._fsq_levels - 1).to(neigh_mi.dtype)[None, None, :])  # (B, G, D)
        neigh_code = self._multi_index_code(neigh_mi.reshape(-1, D)).reshape(B, -1)  # (B, G)
        nb = torch.zeros(B, vocab, device=dev)
        nb.scatter_(1, neigh_code, 1.0)                                    # neighbourhood indicator (incl GT)
        gtoh = torch.zeros(B, vocab, device=dev).scatter_(1, gt.view(-1, 1).long(), 1.0)
        eps = 1e-8

        # PRIOR motion counter (weighting) + per-code EXCLUSION of this-FT codes with low prior.
        prior = self._prior_motion.to(dev) if self._prior_motion is not None else torch.ones(vocab, device=dev)
        selectable = torch.ones(vocab, device=dev)                        # 1 = allowed as a distill sample
        if self._motion_ft is not None:
            ft_low = (self._motion_ft.to(dev) > 0) & (prior < self._prior_threshold)   # FT code, insufficient prior
            selectable = (~ft_low).float()
        sel = selectable[None, :]                                          # (1, vocab)

        local_p = nb * (1.0 - gtoh) * sel + eps                           # neighbours minus GT, minus excluded
        local = torch.multinomial(local_p, n_local, replacement=True) if n_local > 0 \
            else torch.empty(B, 0, dtype=torch.long, device=dev)
        global_p = prior[None, :] * (1.0 - nb) * sel + eps                # prior-weighted, excl GT+neigh+excluded
        glob = torch.multinomial(global_p, n_global, replacement=True) if n_global > 0 \
            else torch.empty(B, 0, dtype=torch.long, device=dev)
        return torch.cat([local, glob], dim=1)                            # (B, n_local + n_global)

    def _ensure_vsa_teacher(self) -> None:
        """Lazily build the FROZEN teacher = the PT/warm-start VSA. Deep-copies self, RESETS its weights
        to the recorded warm-start checkpoint (config.pretrained_path → resume-correct: teacher stays at
        PT even when the student resumes mid-FT), freezes it, and drops the VLM layers it never runs (the
        lean VSA forward uses no VLM stream; only _vlm.rotary_emb is kept). Held in a 1-list so it is NOT
        registered as a submodule (excluded from state_dict + optimizer)."""
        if getattr(self, "_vsa_teacher", None):
            return
        t = self._deepcopy_self_for_teacher()
        pp = getattr(self.config, "pretrained_path", None)
        if pp:
            raw = _load_raw_state_dict(str(pp), {})
            if raw is not None:
                tstate = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in raw.items()}
                t.load_state_dict(tstate, strict=False)
        for p in t.parameters():
            p.requires_grad_(False)
        t.eval()
        # reclaim memory: the lean VSA forward never runs the VLM LLM layers or the VLM vision tower
        # (cond uses its OWN dino/siglip). Keep _vlm.rotary_emb (used by _run_streams).
        try:
            t.paligemma_with_expert.paligemma.model.language_model.layers = nn.ModuleList()
            t.paligemma_with_expert.paligemma.model.vision_tower = None
        except AttributeError:
            pass
        self._vsa_teacher = [t]

    def _distill_velocities(self, teacher, cond_tokens: Tensor, cond_images, x_t: Tensor, time: Tensor,
                            state: Tensor, skill_code: Tensor, n_local: int, n_global: int):
        """MSE between the student and a given TEACHER's VSA velocities on SAMPLED skills, at the SAME
        (obs, x_t, t). Teacher = frozen PT (FT distill) or the weight-EMA self (Stage-2). Returns
        (loss over the sampled n, parts dict {gt_drift(monitor), local, global}) or (0, None) if n==0."""
        codes = self._sample_distill_codes(skill_code, n_local, n_global)         # (B, n) = [local | global]
        n = codes.shape[1]
        if n == 0:
            return x_t.new_zeros(()), None
        # append the GT code as a MONITOR-ONLY column: its teacher-drift is logged for a fair cross-run
        # comparison (BC pushes drift exactly there) but EXCLUDED from the backpropped mean — distilling
        # on GT would fight the BC loss head-on.
        codes_all = torch.cat([codes, skill_code.view(-1, 1).long()], dim=1)      # (B, n+1)
        n_all = codes_all.shape[1]
        z = self._code_to_z(codes_all.reshape(-1)) / self._fsq_half[None, :]       # (B*(n+1), D)
        ct = cond_tokens.repeat_interleave(n_all, dim=0)                   # student cond (adapting)
        xt = x_t.repeat_interleave(n_all, dim=0)
        tm = time.repeat_interleave(n_all, dim=0)
        st = state.repeat_interleave(n_all, dim=0)
        v_student = self._vsa_velocity(ct, xt, tm, st, z)
        with torch.no_grad():
            ct_t = teacher._cond_tokens(cond_images).repeat_interleave(n_all, dim=0)  # teacher cond
            v_teacher = teacher._vsa_velocity(ct_t, xt, tm, st, z)
        # per-sample MSE (B, n+1): [local | global | gt] → backprop mean over the SAMPLED n only.
        per = (v_student - v_teacher).pow(2).mean(dim=(1, 2)).view(-1, n_all)
        nl = min(int(n_local), n)
        parts = {"gt_drift": per[:, n:].mean().detach()}
        if nl > 0:
            parts["local"] = per[:, :nl].mean().detach()
        if n - nl > 0:
            parts["global"] = per[:, nl:n].mean().detach()
        return per[:, :n].mean(), parts

    def _vsa_distill_loss(self, cond_tokens: Tensor, cond_images, x_t: Tensor, time: Tensor,
                          state: Tensor, skill_code: Tensor) -> Tensor:
        """Continual-learning anti-forgetting term (FT, B batches): distil against the FROZEN PT teacher.
        Stashes the local/global split in self._last_vsa_distill_parts for wandb."""
        self._ensure_vsa_teacher()
        loss, parts = self._distill_velocities(
            self._vsa_teacher[0], cond_tokens, cond_images, x_t, time, state, skill_code,
            int(self.config.vsa_distill_n_local), int(self.config.vsa_distill_n_global))
        self._last_vsa_distill_parts = parts
        return loss

    def _ema_self_distill_loss(self, cond_tokens: Tensor, cond_images, x_t: Tensor, time: Tensor,
                               state: Tensor, skill_code: Tensor) -> Tensor:
        """Stage-2 forgetting-prep term: distil sampled non-GT skills against the model's OWN weight-EMA
        (self._ema_teacher, updated each step). Stashes the split in self._last_ema_distill_parts."""
        self._ensure_ema_teacher()
        loss, parts = self._distill_velocities(
            self._ema_teacher[0], cond_tokens, cond_images, x_t, time, state, skill_code,
            int(self.config.ema_self_n_local), int(self.config.ema_self_n_global))
        self._last_ema_distill_parts = parts
        return loss

    def _deepcopy_self_for_teacher(self):
        """deepcopy(self) for a frozen/EMA teacher WITHOUT the transient forward state. torch cannot
        deepcopy non-leaf (graph-attached) tensors — this forward stashes several on self (_vlm_all_layers,
        _collected_layers, _last_* losses) — and a teacher must not carry a nested teacher. Strip those
        keys, copy, restore. Parameters/buffers live in _parameters/_buffers (deepcopied normally)."""
        import copy  # noqa: PLC0415
        strip = [k for k, v in self.__dict__.items()
                 if (isinstance(v, torch.Tensor) and not v.is_leaf)
                 or (isinstance(v, list) and any(isinstance(x, torch.Tensor) for x in v))]
        strip += [k for k in ("_ema_teacher", "_vsa_teacher") if k in self.__dict__]
        saved = {k: self.__dict__.pop(k) for k in dict.fromkeys(strip)}
        try:
            return copy.deepcopy(self)
        finally:
            self.__dict__.update(saved)

    def _ensure_ema_teacher(self) -> None:
        """Lazily build the EMA-self teacher = a frozen deep-copy of the CURRENT model (so it starts at
        the Stage-2 state, NOT the Stage-1 warm-start). Updated each step by _update_ema_teacher. Held in
        a 1-list (not a submodule → excluded from state_dict/optimizer). NOT persisted: on resume it is
        rebuilt from the resumed weights and the EMA history restarts (τ≈1/(1−α) steps to re-warm)."""
        if getattr(self, "_ema_teacher", None):
            return
        t = self._deepcopy_self_for_teacher()
        for p in t.parameters():
            p.requires_grad_(False)
        t.eval()
        try:                                # lean VSA forward never runs the VLM LLM / vision tower
            t.paligemma_with_expert.paligemma.model.language_model.layers = nn.ModuleList()
            t.paligemma_with_expert.paligemma.model.vision_tower = None
        except AttributeError:
            pass
        self._ema_teacher = [t]

    @torch.no_grad()
    def _update_ema_teacher(self) -> None:
        """θ_ema ← α·θ_ema + (1−α)·θ_now over the teacher's (VSA) parameters. Called once per training
        step. First call builds the teacher (= current weights) → a no-op blend that step."""
        self._ensure_ema_teacher()
        a = float(self.config.ema_self_alpha)
        student = dict(self.named_parameters())
        for name, tp in self._ema_teacher[0].named_parameters():
            sp = student.get(name)
            if sp is not None:
                tp.mul_(a).add_(sp.detach().to(tp.dtype), alpha=1.0 - a)

    def _reader_xattn_block(self, fallback: Tensor) -> Tensor:
        """SKILL READER 전용 VLM read-set — reader_attend_image/language(None=attend_* 상속)로 cond/expert의
        read-set과 분리. cond conduit를 lang-only로 조여도(attend_image=False) skill 예측기는 시작 obs
        이미지를 계속 볼 수 있다. 둘 다 None(상속)이거나 is_lang 스냅샷이 없으면 전달된 블록 그대로
        (기존 체크포인트 동작 불변)."""
        c = self.config
        rai = getattr(c, "reader_attend_image", None)
        ral = getattr(c, "reader_attend_language", None)
        if rai is None and ral is None:
            return fallback                                   # 상속 → cond/expert와 동일 read-set
        il = getattr(self, "_vlm_is_lang", None)
        if il is None or il.shape[0] != fallback.shape[0]:
            return fallback
        ai = bool(getattr(c, "attend_image", True)) if rai is None else bool(rai)
        al = bool(getattr(c, "attend_language", False)) if ral is None else bool(ral)
        blk = torch.zeros_like(il)
        if not al:
            blk = blk | il                                    # exclude LANGUAGE tokens
        if not ai:
            blk = blk | ~il                                   # exclude IMAGE tokens
        return blk

    def _skill_hidden(self, vlm_out: Tensor, vlm_pad: Tensor, vlm_xattn_block: Tensor,
                      all_layers: Tensor | None = None) -> Tensor:
        """Pooled skill hidden from the JOINT concat-KV SkillReader over the VLM's FINAL-layer output
        (``vlm_out`` (B,nv,W)). The reader attends the reader read-set — reader_attend_image/language,
        None이면 cond와 같은 attend_image/attend_language(~vlm_xattn_block) 상속 — minus padding →
        SkillHead. Read-only. all_layers (B,L,nv,W): skill_reader_all_layers → the reader's KV is EVERY
        layer's nv tokens stacked (L·nv keys), so its attention can pick the informative depth; the
        ignore mask is tiled L×."""
        vlm_xattn_block = self._reader_xattn_block(vlm_xattn_block)      # reader 전용 read-set 적용
        vlm_key_ignore = (~vlm_pad) | vlm_xattn_block[None, :]           # (B, nv) True = do not attend
        if all_layers is not None:
            B, L, nv, W = all_layers.shape
            return self.skill_reader(all_layers.reshape(B, L * nv, W), vlm_key_ignore.repeat(1, L))
        return self.skill_reader(vlm_out, vlm_key_ignore)

    def _pred_skill_ste_z(self, skill_hidden: Tensor) -> Tensor:
        """cond_skill_source=pred: the VLM's predicted skill as a normalized z, with a straight-through
        round so the cond/prefix sees the SAME discrete code as inference (skill_head.decode), while the
        flow loss still backprops the continuous prediction into the VLM trunk (through SkillHead, which
        stays frozen in FT but conducts the gradient)."""
        z_pred = self.skill_head._pred_z(skill_hidden)                            # (B, D) ∈ (-1,1), differentiable
        code = self.skill_head.decode(skill_hidden)                              # (B,) discrete (== inference)
        z_hard = (self._code_to_z(code) / self._fsq_half[None, :]).to(z_pred.dtype)
        return z_hard + (z_pred - z_pred.detach())                                # STE: fwd=z_hard, grad=z_pred

    # ── training ──
    def _skill_view(self, start_images, lang_tokens, lang_masks) -> Tensor:
        """SKILL view: standalone VLM prefix forward under adapter ① ("skill") → reader → skill_hidden.
        The only regime path that runs the reader; its graph never contains cond/expert, so training it
        can only move ①/reader/head (whatever _apply_continual_freezes left trainable).
        Adapter selection is STICKY (set_active_adapters, no restore): the gradient-checkpoint recompute
        re-runs these layers inside loss.backward(), after any scoped `with` would have exited — the
        global must still hold THIS view's set then, or the recompute diverges from the saved forward."""
        set_active_adapters(self._active_adapters({"skill"}))
        vlm_embeds, vlm_pad, vlm_xattn_block = self._vlm_tokens(start_images, lang_tokens, lang_masks)
        return self._skill_hidden_standalone(vlm_embeds, vlm_pad, vlm_xattn_block)

    def _flow_view(self, cond_images, start_images, lang_tokens, lang_masks, state, skill_zq,
                   x_t, time, severed: bool, severed_adapters: frozenset = frozenset()):
        """FLOW view → (v_t, cond_tokens). Connected reads the VLM under ``vlm_lora`` and ingests it
        through ``cond_lora``. Severed skips the VLM and keeps the complete frozen Stage-1 VSA, including
        ``expert_lora``. ``severed_adapters`` is normally empty (pure VSA) or {"cond_lora"} for the
        Stage-2 conduit-gating batch. Adapter selection is sticky; see _skill_view."""
        if severed:
            set_active_adapters(self._active_adapters(severed_adapters))
            cond_tokens = self._cond_tokens(cond_images)
            return self._vsa_velocity(cond_tokens, x_t, time, state, skill_zq), cond_tokens
        set_active_adapters(self._active_adapters({"vlm_lora", "cond_lora"}))
        cond_tokens = self._cond_tokens(cond_images)
        vlm_embeds, vlm_pad, vlm_xattn_block = self._vlm_tokens(start_images, lang_tokens, lang_masks)
        action_tokens = self._action_in(x_t)
        prefix = self._action_prefix_from_z(skill_zq)   # None in state_skill mode
        if prefix is not None:
            action_tokens = torch.cat([prefix, action_tokens], dim=1)
        expert_cond = self._expert_cond_from_z(time, state, skill_zq)
        _, action_out = self._joint_forward(
            cond_tokens, vlm_embeds, vlm_pad, vlm_xattn_block, action_tokens, expert_cond,
            drop_vlm=False)
        return self._action_out(action_out), cond_tokens

    def _grounded_flow_view(self, cond_images, start_images, lang_tokens, lang_masks, state,
                            skill_zq: Tensor, x_t: Tensor, time: Tensor) -> Tensor:
        """STAGE-3b grounding: evaluate a (STE-predicted) skill z through the FROZEN connected motor →
        v_t, with gradient flowing ONLY into z (→ ①/reader/head). The frozen prefix (②-view VLM + cond
        with ③) is encoded under no_grad into cached K/V constants — z's gradient never traverses those
        weights (z enters via the action prefix/AdaRMS), and constants keep the checkpoint-recompute
        adapter set unambiguous: the sticky set returns to {"skill"} for the grad graph (skill view +
        this UNcheckpointed action-stream), matching the backward recompute of the skill view.
        Attention topology = the deployment/cached inference path (numerically identical to the joint
        forward; vlm_expert honored) → the grounding measures exactly the deployed configuration."""
        bsize, device = state.shape[0], state.device
        n_chunk = self.config.chunk_size
        n_prefix = 0 if self._state_cond_mode == "state_skill" else 1
        na = n_prefix + n_chunk

        set_active_adapters(self._active_adapters({"vlm_lora", "cond_lora"}))
        with torch.no_grad():
            vlm_embeds, vlm_pad, vlm_xattn_block = self._vlm_tokens(start_images, lang_tokens, lang_masks)
            cond_tokens = self._cond_tokens(cond_images)
            nc, nv = cond_tokens.shape[1], vlm_embeds.shape[1]
            vlm_pos, cond_pos, action_pos = self._joint_positions(vlm_pad, nc, na)
            cond_pad = torch.ones(bsize, nc, dtype=torch.bool, device=device)
            vlm_kv, _ = self._encode_prefix_kv(self._vlm.layers, vlm_embeds, vlm_pad, vlm_pos, adarms=None)
            vlm_to_cond = self.config.vlm_cond
            cond_kv, _ = self._encode_prefix_kv(
                self.cond_encoder.model.layers, cond_tokens, cond_pad, cond_pos, adarms=None,
                extra_kv=(vlm_kv if vlm_to_cond else None),
                extra_valid=((vlm_pad & ~vlm_xattn_block[None, :]) if vlm_to_cond else None))
            attend_vlm = bool(self.config.vlm_expert)
            if attend_vlm:                                 # action keys = [cond, VLM, action-stream]
                prefix_kv = [(torch.cat([ck, vk], dim=2), torch.cat([cv, vv], dim=2))
                             for (ck, cv), (vk, vv) in zip(cond_kv, vlm_kv)]
                npre = nc + nv
                cols = torch.cat([cond_pad, vlm_pad,
                                  torch.ones(bsize, na, dtype=torch.bool, device=device)], dim=1)
            else:                                          # action keys = [cond, action-stream]
                prefix_kv, npre = cond_kv, nc
                cols = torch.cat([cond_pad, torch.ones(bsize, na, dtype=torch.bool, device=device)], dim=1)
            allow = torch.zeros(bsize, 1, na, npre + na, dtype=torch.bool, device=device)
            allow[:, :, :n_prefix, npre: npre + n_prefix] = True        # prefix → prefix
            if self.config.cond_expert:
                allow[:, :, :n_prefix, :nc] = True                      # prefix → cond (= Stage-1)
                allow[:, :, n_prefix:, :nc] = True                      # action → cond (scene)
            if attend_vlm:
                allow[:, :, n_prefix:, nc: nc + nv] = (~vlm_xattn_block)[None, None, None, :]
            allow[:, :, n_prefix:, npre:] = True                        # action → prefix + action
            att_4d = torch.where(allow & cols[:, None, None, :], 0.0, OPENPI_ATTENTION_MASK_VALUE)

        # Keep the Stage-1 VSA adapter live for the action stream while matching the skill-view adapter
        # set used by checkpoint recomputation.
        set_active_adapters(self._active_adapters({"skill"}))
        expert_cond = self._expert_cond_from_z(time, state, skill_zq)
        a_in = self._action_in(x_t)
        prefix = self._action_prefix_from_z(skill_zq)      # None in state_skill mode
        h = a_in if prefix is None else torch.cat([prefix, a_in], dim=1)
        for i in range(len(prefix_kv)):
            h = self._action_layer_cached(i, h, prefix_kv, att_4d, action_pos, expert_cond)
        h, _ = layernorm_forward(self._expert.norm, h, expert_cond)
        return self._action_out(h[:, -n_chunk:])

    def forward(
        self,
        cond_images: list[Tensor],
        start_images: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        skill_code: Tensor,
        actions: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        hold_actions: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None]:
        """LoRA-continual regime router → (flow_losses (B, chunk, max_action_dim) | None,
        skill_hidden (B, vlm_width) | None).

        TRAIN — sequential PT stages (pt_stage: every batch is that stage) or FT per-batch multinomial
        (regime_probs_ft; the whole batch is ONE regime):
          SKILL       standalone VLM(+① "skill") → reader → skill_hidden only (flow=None).
                      (= STAGE 3 [pt_stage="skill"], and FT's SKILL regime.)
          COND/MOTOR  joint flow: VLM(+② "vlm_lora") read by cond(+③ "cond_lora") → frozen VSA; GT-teacher-
                      forced z (the motor path never touches the skill predictor); real cross-skill tail.
                      (STAGE 2 [pt_stage="cond"] trains ②③+vision against the FROZEN expert; FT "motor"
                      trains the expert.)
          COND_SEV    (Stage 2, 확률 cond_severed_prob) VLM 없이 ③만 활성. 같은 hold-derived x_t에서
                      ③ OFF frozen Stage-1 VSA 출력을 teacher로 삼아 ③ ON student를 anchor한다.
                      gradient는 ③에만 흐름 (②/vision은 그래프 밖). distill 없음.
          MOTOR_SEV   new adapters OFF (③ bypass → complete Stage-1 VSA), the VLM never runs; Stage-1 hold
                      target; + random-skill distillation to the frozen-PT teacher (also severed).
        EVAL / probes / no-regime training — BOTH views (skill standalone, then flow; z follows
        cond_skill_source; _probe_force_drop severs the flow view), mirroring the 2-forward inference.
        """
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        noise = noise.to(dtype=actions.dtype)
        if self.training:                # lazily split/update the cumulative motion counter (once)
            self.finalize_motion_counter()

        # ── LoRA-continual regime sampling: the whole batch is ONE regime. Trainability is STATIC
        # (_apply_continual_freezes) — no per-batch requires_grad churn: each regime's graph only
        # contains what it may train, and inactive adapters are simply not in the graph. ──
        # FT takes precedence: a parent checkpoint's config.json carries its pt_stage, so an FT run
        # (loading that checkpoint + setting regime_probs_ft) naturally has BOTH — the ft key wins.
        probs_ft = getattr(self.config, "regime_probs_ft", None)
        pt_stage = self._resolved_pt_stage()
        regime = None
        if self.training and probs_ft:
            names = ("skill", "motor", "motor_sev")
            probs = [float(x) for x in probs_ft]
            if len(probs) != len(names):
                raise ValueError(f"regime_probs_ft needs {len(names)} entries {names}, got {probs}.")
            r = int(torch.multinomial(torch.tensor(probs, device=actions.device), 1).item())
            regime = names[r]
        elif self.training and pt_stage:
            regime = pt_stage                # "cond" (Stage 2) / "skill" (Stage 3) — every batch, no coin
            # Stage-2 severed 정규화 배치 (conduit gating): 확률 cond_severed_prob로 VLM을 떼고 ③만
            # 활성인 severed flow — ③이 이미지-단독 경로로 PT prior를 흡수하는 구멍을 닫는다.
            p_sev = float(getattr(self.config, "cond_severed_prob", 0.0) or 0.0)
            if regime == "cond" and p_sev > 0.0 and float(torch.rand(())) < p_sev:
                regime = "cond_sev"
        self._last_regime = regime
        self._last_severed_hold = False
        self._last_cond_anchor_raw = None
        self._last_vsa_distill = None

        # SKILL regime — mode a (default): skill prediction only, no flow this batch.
        # mode b (skill_action_grounding, STAGE 3 only): the SKILL view's STE prediction is ALSO
        # evaluated through the frozen connected motor against the GT-action flow target — the flow
        # loss backprops ONLY into ①/reader/head (everything downstream frozen; pure grounding signal).
        if regime == "skill":
            skill_hidden = self._skill_view(start_images, lang_tokens, lang_masks)
            if not getattr(self.config, "skill_action_grounding", False):
                return None, skill_hidden
            skill_zq = self._pred_skill_ste_z(skill_hidden)             # forward=decoded z, grad=continuous
            t_exp = time[:, None, None]                                 # connected → real cross-skill tail
            x_t = t_exp * noise + (1 - t_exp) * actions
            u_t = noise - actions
            v_t = self._grounded_flow_view(cond_images, start_images, lang_tokens, lang_masks,
                                           state, skill_zq, x_t, time)
            self._last_regime = regime = "skill_ground"                 # wandb: regime/*_skill_ground
            return F.mse_loss(u_t, v_t, reduction="none"), skill_hidden

        # Severed batches use the Stage-1 hold action to construct x_t (stop+hold past skill_de). MOTOR_SEV
        # and probes retain its GT flow target; COND_SEV instead anchors cond_lora to the frozen VSA at
        # that same x_t, so no image-only shortcut is learned.
        severed = regime in ("motor_sev", "cond_sev") or (
            regime is None and bool(getattr(self, "_probe_force_drop", None)))
        severed_hold = severed and hold_actions is not None and getattr(self.config, "severed_hold_target", True)
        self._last_severed_hold = bool(severed_hold)
        acts = hold_actions if severed_hold else actions
        t_exp = time[:, None, None]
        x_t = t_exp * noise + (1 - t_exp) * acts
        u_t = noise - acts

        skill_hidden = None
        if regime is None:
            # Measurement path (eval/probes/plain no-regime training): BOTH views — the standalone skill
            # view here plus the flow view below — mirroring the 2-forward inference (predict skill →
            # act). z follows cond_skill_source ("pred" = the reader's own STE-rounded code, as deployed).
            # NB: TRAINING through this mixed-view path is impossible with adapters configured — the two
            # views need different STICKY adapter sets in ONE backward graph, so the gradient-checkpoint
            # recompute of the first view would run under the second's set. Regime training never mixes.
            if self.training and torch.is_grad_enabled() and (
                    getattr(self.config, "lora_skill", False) or getattr(self.config, "vlm_lora", False)
                    or getattr(self.config, "cond_lora", False)):
                raise ValueError("LoRA adapters are configured but neither pt_stage nor regime_probs_ft "
                                 "is set — plain (no-stage) training cannot mix the skill/flow adapter "
                                 "views under gradient checkpointing. Set pt_stage or regime_probs_ft.")
            skill_hidden = self._skill_view(start_images, lang_tokens, lang_masks)
            if getattr(self.config, "cond_skill_source", "gt") == "pred":
                skill_zq = self._pred_skill_ste_z(skill_hidden)
            else:
                skill_zq = self._code_to_z(skill_code) / self._fsq_half[None, :]
        else:
            # COND / MOTOR / MOTOR_SEV: GT teacher-forced z — the motor path never touches the skill
            # predictor (motor ⊥ skill; the SKILL regime is the only place skill prediction trains).
            skill_zq = self._code_to_z(skill_code) / self._fsq_half[None, :]

        if regime == "cond_sev":
            # Same hold-derived x_t on both sides. Teacher = exact frozen Stage-1 VSA (expert_lora stays
            # active, all new adapters off); student = VLM severed + cond_lora. This is a function anchor,
            # not GT action supervision, so cond_lora cannot learn an image-only motion shortcut.
            with torch.no_grad():
                teacher_v, _ = self._flow_view(
                    cond_images, start_images, lang_tokens, lang_masks, state, skill_zq, x_t, time,
                    severed=True, severed_adapters=frozenset())
            v_t, cond_tokens = self._flow_view(
                cond_images, start_images, lang_tokens, lang_masks, state, skill_zq, x_t, time,
                severed=True, severed_adapters=frozenset({"cond_lora"}))
            anchor_losses = F.mse_loss(teacher_v, v_t, reduction="none")
            self._last_cond_anchor_raw = anchor_losses.detach()
            flow_losses = float(getattr(self.config, "cond_severed_anchor_weight", 1.0)) * anchor_losses
        else:
            v_t, cond_tokens = self._flow_view(
                cond_images, start_images, lang_tokens, lang_masks, state, skill_zq, x_t, time,
                severed=severed, severed_adapters=frozenset())
            flow_losses = F.mse_loss(u_t, v_t, reduction="none")

        # Random-skill distillation to the FROZEN-PT teacher — MOTOR_SEV training batches only: pure-VSA
        # repertoire preservation lives in the standalone regime. Runs adapter-free (the teacher included,
        # via the same active_adapters scope) → student and teacher are compared in the SAME pure
        # skill→motion space the design preserves. (Reuses this batch's cond_tokens/x_t/time/state —
        # SAME "ruler" — so it only measures parameter drift.)
        if regime == "motor_sev" and getattr(self.config, "vsa_distill", False):
            # adapters already OFF (sticky, set by the severed _flow_view) → teacher + student both run
            # in the pure adapter-free skill→motion space, and the backward recompute matches.
            if float(getattr(self.config, "vsa_distill_weight", 0.0)) > 0.0:
                self._last_vsa_distill = self._vsa_distill_loss(
                    cond_tokens, cond_images, x_t, time, state, skill_code)
            else:
                # weight=0 → MONITOR-ONLY: log the same train_distill/* keys with ZERO training
                # effect (no grad graph) — the distill-off-equivalent baseline for drift comparison.
                with torch.no_grad():
                    self._last_vsa_distill = self._vsa_distill_loss(
                        cond_tokens.detach(), cond_images, x_t, time, state, skill_code)
        return flow_losses, skill_hidden

    # ── inference ──
    def _vlm_prefix_out(self, vlm_embeds: Tensor, vlm_pad: Tensor, all_layers: bool = False):
        """VLM transformer over a PRECOMPUTED prefix (bidirectional within valid tokens) → hidden
        (B, nv, vlm_width). Grad-capable: the gemma language_model gradient-checkpoints internally
        when training, so the cond_skill_source=pred path (which keeps the graph) stays memory-safe.
        all_layers → also return the per-layer stack (B, L, nv, W), each normed by the VLM final norm
        (matching the joint forward's captured stack) for skill_reader_all_layers."""
        att_2d = vlm_pad[:, None, :] & vlm_pad[:, :, None]
        # SDPA (the VLM's default attn) requires the additive bias dtype to match the query's; the
        # python-float `torch.where` yields float32, so cast to the bf16 working dtype.
        att_4d = torch.where(att_2d[:, None], 0.0, OPENPI_ATTENTION_MASK_VALUE).to(vlm_embeds.dtype)
        position_ids = torch.cumsum(vlm_pad, dim=1) - 1
        out = self._vlm.forward(
            inputs_embeds=vlm_embeds, attention_mask=att_4d, position_ids=position_ids,
            past_key_values=None, use_cache=False, adarms_cond=None, output_hidden_states=all_layers,
        )
        if all_layers:
            # all_hidden_states = (embeds, layer0_out .. layer(N-2)_out [PRE-final-norm], FINAL-NORMED
            # layer(N-1)_out). Norm each pre-norm layer output; the last entry is ALREADY final-normed
            # (== last_hidden_state) → append it directly (no double-norm). Matches the joint forward's
            # per-layer capture (each of the N layer outputs normed by the VLM final norm).
            normed = [layernorm_forward(self._vlm.norm, h, None)[0] for h in out.hidden_states[1:-1]]
            normed.append(out.last_hidden_state)
            return out.last_hidden_state, torch.stack(normed, dim=1)   # (B, N, nv, W)
        return out.last_hidden_state

    def _skill_hidden_standalone(self, vlm_embeds: Tensor, vlm_pad: Tensor, vlm_xattn_block: Tensor) -> Tensor:
        """Skill hidden from a STANDALONE VLM prefix forward (pred phase-1 + inference). Honors
        skill_reader_all_layers so the reader sees the SAME layer stack as the joint-forward training path."""
        if getattr(self.config, "skill_reader_all_layers", False):
            vlm_out, all_layers = self._vlm_prefix_out(vlm_embeds, vlm_pad, all_layers=True)
            return self._skill_hidden(vlm_out, vlm_pad, vlm_xattn_block, all_layers=all_layers)
        return self._skill_hidden(self._vlm_prefix_out(vlm_embeds, vlm_pad), vlm_pad, vlm_xattn_block)

    @torch.no_grad()
    def predict_skill_code(self, start_images: list[Tensor], lang_tokens: Tensor, lang_masks: Tensor) -> Tensor:
        """Run ONLY the VLM (bidirectional prefix) + skill reader → argmax FSQ skill code (B,).
        SKILL view: adapter ① ("skill"), sticky — callers that need a different view afterwards
        (sample_actions' flow scope) set their own AFTER resolving the skill (2-forward inference)."""
        set_active_adapters(self._active_adapters({"skill"}))
        vlm_embeds, vlm_pad, vlm_xattn_block = self._vlm_tokens(start_images, lang_tokens, lang_masks)
        hidden = self._skill_hidden_standalone(vlm_embeds, vlm_pad, vlm_xattn_block)
        return self.skill_head.decode(hidden)

    # ── cached inference (branch A): VLM cached per skill, cond per call, action per denoise step ──
    def _encode_prefix_kv(self, layers, embeds, pad, position_ids, adarms=None,
                          extra_kv=None, extra_valid=None):
        """Run a prefix stream (bidirectional within its valid tokens) → per-layer post-RoPE
        (K, V) + the pre-final-norm hidden. ``extra_kv``/``extra_valid`` optionally prepend a
        FIXED already-RoPE'd per-layer K/V context the stream may attend (branch A: cond reads
        the cached VLM K/V at vlm-nonlang columns). With the same mask/positions as training,
        this reproduces the joint forward's per-layer K/V for the stream exactly, so the cached
        denoise below is numerically identical to a full joint forward."""
        embeds = embeds.to(self._wdtype)
        att_2d = make_att_2d_masks(pad, torch.zeros_like(pad))                     # (B, n, n) own block
        if extra_kv is not None:
            ne = extra_valid.shape[1]
            extra_cols = extra_valid[:, None, :].expand(-1, pad.shape[1], -1)      # (B, n, ne)
            att_2d = torch.cat([extra_cols, att_2d], dim=2)                        # keys = [extra, own]
        att_4d = torch.where(att_2d[:, None], 0.0, OPENPI_ATTENTION_MASK_VALUE)
        rotary = self._vlm.rotary_emb
        h, kv = embeds, []
        for li, layer in enumerate(layers):
            hn, gate = layernorm_forward(layer.input_layernorm, h, adarms)
            shape = (*hn.shape[:-1], -1, layer.self_attn.head_dim)
            q = layer.self_attn.q_proj(hn).view(shape).transpose(1, 2)
            k = layer.self_attn.k_proj(hn).view(shape).transpose(1, 2)
            v = layer.self_attn.v_proj(hn).view(shape).transpose(1, 2)
            dummy = torch.zeros(q.shape[0], q.shape[2], q.shape[-1], device=q.device, dtype=q.dtype)
            cos, sin = rotary(dummy, position_ids)
            q, k = modeling_gemma.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
            kv.append((k, v))
            if extra_kv is not None:
                k_att = torch.cat([extra_kv[li][0], k], dim=2)
                v_att = torch.cat([extra_kv[li][1], v], dim=2)
            else:
                k_att, v_att = k, v
            att_out, _ = modeling_gemma.eager_attention_forward(
                layer.self_attn, q, k_att, v_att, att_4d, layer.self_attn.scaling)
            att_out = att_out.reshape(att_out.shape[0], -1, q.shape[1] * layer.self_attn.head_dim)
            if att_out.dtype != layer.self_attn.o_proj.weight.dtype:
                att_out = att_out.to(layer.self_attn.o_proj.weight.dtype)
            o = _gated_residual(h, layer.self_attn.o_proj(att_out), gate)
            after = o.clone()
            o, gate2 = layernorm_forward(layer.post_attention_layernorm, o, adarms)
            if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                o = o.to(torch.bfloat16)
            h = _gated_residual(after, layer.mlp(o), gate2)
        return kv, h

    def _action_layer_cached(self, layer_idx, h, prefix_kv, att_4d, position_ids, adarms):
        """One action-expert layer attending the FIXED combined prefix K/V + its own (RoPE'd) tokens."""
        layer = self._expert.layers[layer_idx]
        hn, gate = layernorm_forward(layer.input_layernorm, h, adarms)
        shape = (*hn.shape[:-1], -1, layer.self_attn.head_dim)
        q = layer.self_attn.q_proj(hn).view(shape).transpose(1, 2)
        k = layer.self_attn.k_proj(hn).view(shape).transpose(1, 2)
        v = layer.self_attn.v_proj(hn).view(shape).transpose(1, 2)
        dummy = torch.zeros(q.shape[0], q.shape[2], q.shape[-1], device=q.device, dtype=q.dtype)
        cos, sin = self._vlm.rotary_emb(dummy, position_ids)
        q, k = modeling_gemma.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        pk, pv = prefix_kv[layer_idx]
        key, val = torch.cat([pk, k], dim=2), torch.cat([pv, v], dim=2)
        att_out, _ = modeling_gemma.eager_attention_forward(
            layer.self_attn, q, key, val, att_4d, layer.self_attn.scaling)
        att_out = att_out.reshape(att_out.shape[0], -1, q.shape[1] * layer.self_attn.head_dim)
        if att_out.dtype != layer.self_attn.o_proj.weight.dtype:
            att_out = att_out.to(layer.self_attn.o_proj.weight.dtype)
        o = _gated_residual(h, layer.self_attn.o_proj(att_out), gate)
        after = o.clone()
        o, gate2 = layernorm_forward(layer.post_attention_layernorm, o, adarms)
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            o = o.to(torch.bfloat16)
        return _gated_residual(after, layer.mlp(o), gate2)

    @torch.no_grad()
    def _sample_actions_A(self, cond_images, start_images, lang_tokens, lang_masks, state, skill_code,
                          noise, num_steps):
        """Branch A (ae) cached sampling: VLM encoded once; cond-encoder (SCENE only, skill-blind)
        encoded once reading the VLM cache; each denoise step runs the action expert over the action
        stream against the cond cache (cond⊥action), decoding the K action tokens. The action prefix is
        the skill token in `state` mode or empty in `state_skill` mode (skill → AdaRMS)."""
        bsize, device = state.shape[0], state.device
        n_chunk = self.config.chunk_size
        cond_tokens = self._cond_tokens(cond_images)                    # image-only (state→AdaRMS)
        nc = cond_tokens.shape[1]
        vlm_embeds, vlm_pad, vlm_xattn_block = self._vlm_tokens(start_images, lang_tokens, lang_masks)
        nv = vlm_embeds.shape[1]
        # action stream = prefix(n_prefix) + action(n_chunk). state mode → [skill] prefix (1 token);
        # state_skill → no prefix (the skill rides the AdaRMS).
        n_prefix = 0 if self._state_cond_mode == "state_skill" else 1
        na = n_prefix + n_chunk

        # RoPE positions: VLM placed FIRST ([0..nv)), then cond, then action — the SAME frame as the
        # training joint forward (_joint_positions), so this cached denoise stays numerically identical.
        # (The cached prefix K/V below is still concatenated as [cond, vlm]; column order is immaterial
        # because each key already carries its RoPE position.)
        vlm_pos, cond_pos, action_pos = self._joint_positions(vlm_pad, nc, na)
        cond_pad = torch.ones(bsize, nc, dtype=torch.bool, device=device)

        # VLM: encode once under the CURRENT (flow) adapter scope → cached K/V for cond/action reads.
        vlm_kv, _ = self._encode_prefix_kv(self._vlm.layers, vlm_embeds, vlm_pad, vlm_pos, adarms=None)
        if skill_code is None:
            # The skill must come from the SKILL view (adapter ①) BEFORE the flow scope is set —
            # sample_actions resolves it (predict_skill_code) and always passes it in.
            raise ValueError("skill_code is None — call via sample_actions (it resolves the skill under "
                             "the SKILL adapter view before setting the flow scope).")

        # EVAL-time VLM dropout (eval_drop_vlm): sever cond→VLM and action→VLM exactly like a training
        # VSA (B) batch — mask-only, positions unchanged. The VLM still runs standalone (skill code above
        # / GT); ONLY the discrete skill crosses to the action side, no K/V.
        drop_vlm = bool(getattr(self.config, "eval_drop_vlm", False))

        # cond: SCENE only (plain RMSNorm, skill-blind); reads the cached VLM K/V IFF vlm_cond (else VLM-blind) → cache
        vlm_to_cond = self.config.vlm_cond and not drop_vlm
        cond_kv, _ = self._encode_prefix_kv(
            self.cond_encoder.model.layers, cond_tokens, cond_pad, cond_pos, adarms=None,
            extra_kv=(vlm_kv if vlm_to_cond else None),
            extra_valid=((vlm_pad & ~vlm_xattn_block[None, :]) if vlm_to_cond else None))

        # denoise: action stream = prefix (constant) + action×K; attends the cond cache iff cond_expert
        # (cond⊥action). With vlm_expert, the VLM K/V is ALSO injected so action reads the VLM directly —
        # gated by the SAME ~vlm_xattn_block as cond (per the attend_image/attend_language read-set).
        action_prefix = self._action_prefix(skill_code)  # None in state_skill mode
        attend_vlm = bool(self.config.vlm_expert) and not drop_vlm
        if attend_vlm:                                                  # action keys = [cond, VLM, action-stream]
            prefix_kv = [(torch.cat([ck, vk], dim=2), torch.cat([cv, vv], dim=2))
                         for (ck, cv), (vk, vv) in zip(cond_kv, vlm_kv)]
            npre = nc + nv
            cols = torch.cat([cond_pad, vlm_pad, torch.ones(bsize, na, dtype=torch.bool, device=device)], dim=1)
        else:                                                          # action keys = [cond, action-stream]
            prefix_kv = cond_kv
            npre = nc
            cols = torch.cat([cond_pad, torch.ones(bsize, na, dtype=torch.bool, device=device)], dim=1)
        # prefix는 cond(장면)+자기 자신을 읽음(= Stage-1; ⊥ VLM, ⊥ action); action은 cond (+ VLM via
        # ~vlm_xattn_block if attend_vlm: per the attend_image/attend_language read-set) + prefix + action.
        # n_prefix=0 (state_skill) → the prefix→prefix block is empty.
        allow = torch.zeros(bsize, 1, na, npre + na, dtype=torch.bool, device=device)
        allow[:, :, :n_prefix, npre : npre + n_prefix] = True           # prefix → prefix
        if self.config.cond_expert:                                    # cond → expert edge (action reads the scene)
            allow[:, :, :n_prefix, :nc] = True                          # prefix → cond: skill 토큰이 장면을 읽음 (= Stage-1)
            allow[:, :, n_prefix:, :nc] = True
        if attend_vlm:
            allow[:, :, n_prefix:, nc : nc + nv] = (~vlm_xattn_block)[None, None, None, :]  # action → VLM: per the attend_image/attend_language read-set
        allow[:, :, n_prefix:, npre:] = True                          # action → prefix + action
        att_4d = torch.where(allow & cols[:, None, None, :], 0.0, OPENPI_ATTENTION_MASK_VALUE)

        if noise is None:
            noise = self.sample_noise((bsize, n_chunk, self.config.max_action_dim), device)
        dt, x_t = -1.0 / num_steps, noise
        for step in range(num_steps):
            t = torch.full((bsize,), 1.0 + step * dt, dtype=torch.float32, device=device)
            expert_cond = self._expert_cond(t, state, skill_code)  # time + state [+ skill for state_skill]
            a_in = self._action_in(x_t)
            h = a_in if action_prefix is None else torch.cat([action_prefix, a_in], dim=1)
            for i in range(len(prefix_kv)):
                h = self._action_layer_cached(i, h, prefix_kv, att_4d, action_pos, expert_cond)
            h, _ = layernorm_forward(self._expert.norm, h, expert_cond)
            v_t = self._action_out(h[:, -n_chunk:])                     # decode action positions only
            x_t = x_t + dt * v_t
        return x_t

    @torch.no_grad()
    def sample_actions(
        self,
        cond_images: list[Tensor],
        start_images: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        skill_code: Tensor | None = None,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        """Flow-matching sampling with a cached VLM prefix (per skill). ``skill_code`` (cond teacher
        value) defaults to the VLM prediction. The cond stream (scene) is also cached; each denoise
        step runs only the action stream."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        # 2-forward inference, ORDERED for the sticky adapter global: resolve the skill FIRST (SKILL view,
        # adapter ① inside predict_skill_code), THEN set the FLOW view — adapters ②③ (VLM residual riding
        # cond), or none under severed inference (eval_drop_vlm → pure Stage-1 VSA, = MOTOR_SEV training).
        if skill_code is None:
            skill_code = self.predict_skill_code(start_images, lang_tokens, lang_masks)
        _sever = bool(getattr(self.config, "eval_drop_vlm", False))
        _keep = bool(getattr(self.config, "eval_drop_vlm_keep_adapters", False))
        # Connected → vlm_lora + cond_lora. Severed + keep_adapters isolates cond_lora; severed without
        # it disables all new adapters while expert_lora remains on as part of the Stage-1 VSA.
        flow_adapters = {"vlm_lora", "cond_lora"} if not _sever else ({"cond_lora"} if _keep else set())
        set_active_adapters(self._active_adapters(flow_adapters))
        return self._sample_actions_A(cond_images, start_images, lang_tokens, lang_masks, state,
                                      skill_code, noise, num_steps)

    # ── FSQ terminator (inference-only skill-transition gating) ──
    @staticmethod
    def _construct_fsq(path: str, dino_model_path: str | None = None):
        """Instantiate and load only ``terminator.*`` from a v2 FSQ checkpoint.

        Neither the trajectory encoder nor the 300M FSQ action expert is allocated here.
        ``dino_model_path`` overrides a checkpoint-local DINO path after moving servers."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "examples" / "libero"))
        from FSQ import load_fsq_terminator  # noqa: PLC0415

        terminator, _ = load_fsq_terminator(path, device="cpu", dino_model_path=dino_model_path)
        return terminator

    def load_terminator(self, path: str) -> None:
        """Load the frozen FSQ checkpoint's terminator (decides skill transitions in closed loop).
        Only ``predict_termination`` is used (the action chunk comes from the flow-matching expert)."""
        terminator = self._construct_fsq(path, getattr(self.config, "terminator_dino_model_path", None))
        for p in terminator.parameters():
            p.requires_grad_(False)
        terminator.eval()
        self.fsq_term = terminator.to(device=next(self.parameters()).device)
        log.info("Loaded FSQ terminator from %s (state_dim=%s, vision=%s, arch=%s).",
                 path, getattr(terminator, "state_dim", None),
                 getattr(terminator, "vision_backbone", None), getattr(terminator, "arch", None))

    def _build_terminator_trainable(self, path: str) -> None:
        """Build the trainable terminator architecture from FSQ.pt.

        On a fresh Stage-2 warm-start, ``from_pretrained`` subsequently replaces this initialization with
        Stage-1's co-trained ``fsq_term_train.*`` tensors when they exist. The raw FSQ weights remain the
        fallback for older Stage-1 checkpoints without a co-trained terminator.
        """
        terminator = self._construct_fsq(path, getattr(self.config, "terminator_dino_model_path", None))
        terminator.requires_grad_(True).train()
        if terminator.freeze_vision_encoder:
            terminator.vision_encoder.requires_grad_(False).eval()
        # float32 throughout (the FSQ was trained in fp32; terminator inputs state/dino are fp32).
        self.fsq_term_train = terminator.to(device=next(self.parameters()).device, dtype=torch.float32)
        n_tr = sum(p.numel() for p in terminator.parameters())
        log.info("Built TRAINABLE v2 FSQ terminator from %s (%d params).", path, n_tr)

    def terminator_predict(self, true_code: Tensor, state: Tensor, image: Tensor,
                           wrist_image: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """FT co-training on current raw third-person and wrist frames."""
        terminator = self.fsq_term_train
        dev = next(terminator.parameters()).device
        dtype = next(terminator.parameters()).dtype
        st = state.to(device=dev, dtype=dtype)[..., : int(terminator.state_dim)]
        z_norm = (
            self._code_to_z(true_code.to(self._fsq_strides.device)) / self._fsq_half[None, :]
        ).to(device=dev, dtype=dtype)
        if wrist_image is None:
            raise ValueError("FSQ terminator requires observation.images.wrist_image.")
        third = image.to(device=dev, dtype=dtype)
        wrist = wrist_image.to(device=dev, dtype=dtype)
        return terminator(z_norm, st, third, wrist)

    def _code_to_z(self, code: Tensor) -> Tensor:
        """Flat FSQ code (B,) → z_q (B, D) in the FSQ codebook's coordinate frame."""
        idx = code.view(-1, 1).long()
        strides, levels = self._fsq_strides[None, :], self._fsq_levels[None, :]
        level_ids = torch.div(idx, strides, rounding_mode="floor") % levels
        return level_ids.float() - self._fsq_half[None, :]

    def _prepare_term_state(self, state: Tensor) -> Tensor:
        s = state.to(device=next(self.parameters()).device, dtype=torch.float32)
        if s.ndim == 3 and s.shape[1] == 1:
            s = s[:, 0]
        if s.ndim != 2:
            raise ValueError(f"Current terminator state must be (B,D), got {tuple(s.shape)}.")
        sd = int(getattr(self.fsq_term, "state_dim", s.shape[-1]))
        if s.shape[-1] < sd:
            raise ValueError(f"FSQ terminator expects state_dim={sd}, got {s.shape[-1]}-dim raw state.")
        return s[..., :sd]

    def _prepare_term_image(self, img: Tensor | None) -> Tensor | None:
        """Prepare one current raw RGB frame."""
        if img is None:
            return None
        x = img.to(device=next(self.parameters()).device, dtype=torch.float32)
        return x

    @torch.no_grad()
    def terminator_step(self, code, state, image, wrist=None):
        """Run the FSQ terminator on the CURRENT obs for the active skill → (progress, end_prob), each (B,)."""
        z_norm = self._code_to_z(code) / self._fsq_half[None, :]
        st = self._prepare_term_state(state)
        img = self._prepare_term_image(image)
        if img is None:
            return None
        w = self._prepare_term_image(wrist)
        if w is None:
            raise ValueError("FSQ terminator requires a current wrist image (skill_decoder_wrist).")
        return self.fsq_term.predict_termination(z_norm, st, img, w)


# ── Warm-start key remapping ────────────────────────────────────────────────────────────────

def _apply_vlm_override(model, config, dtype, kwargs) -> None:
    """EVAL-ONLY ablation (config.eval_vlm_override_path): overwrite the loaded VLM (the WHOLE PaliGemma
    tower — vision_tower + multi_modal_projector + language_model) with another checkpoint's VLM, while
    keeping this checkpoint's cond + action-expert (+ skill decoder). Used by FT eval to run the FT'd
    motor under the ORIGINAL Stage-2 VLM ("did the motor adaptation transfer to the un-adapted
    perception?"). The Stage-2 checkpoint is the FT run's own pretrained_path. No-op when unset."""
    path = getattr(config, "eval_vlm_override_path", None)
    if not path or not str(path).strip():
        return
    raw = _load_raw_state_dict(str(path), kwargs)
    if raw is None:
        log.warning("eval_vlm_override_path set but no weights at %s; keeping the FT VLM.", path)
        return
    pref = "paligemma_with_expert.paligemma."          # the VLM tower (vision + projector + LLM)
    override = {}
    for k, v in raw.items():
        key = k[len("model."):] if k.startswith("model.") else k
        if key.startswith(pref):
            override[f"model.{key}"] = v.to(dtype)
    if not override:
        log.warning("eval_vlm_override: no PaliGemma keys found in %s; keeping the FT VLM.", path)
        return
    missing, unexpected = model.load_state_dict(override, strict=False)
    log.info("eval_vlm_override: replaced %d VLM tensors from %s (cond/expert/skill-decoder kept from "
             "the FT checkpoint).", len(override), path)


def _remap_pi05_to_vlm(raw: dict) -> dict:
    """pi05 checkpoint → Stage-2: keep ONLY the PaliGemma VLM (vision tower + projector + LLM).
    The action expert is taken from the Stage-1 checkpoint, so pi05's gemma_expert is dropped."""
    out = {}
    for k, v in raw.items():
        key = k[len("model.") :] if k.startswith("model.") else k
        if key.startswith("paligemma_with_expert.paligemma."):
            out[f"model.{key}"] = v
        # pi05 stores the LM head separately; pi05 policy mirrors it into embed_tokens.
        elif key == "paligemma_with_expert.paligemma.lm_head.weight":
            out["model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"] = v.clone()
    return out


def _remap_stage1_to_expert(raw: dict) -> dict:
    """Stage-1 ``skill_expert`` checkpoint → Stage-2 expert/cond side. Stage-1 keys are ``model.*``:
    gemma_expert → paligemma_with_expert.gemma_expert; action/time projections + cond-side
    (cond_encoder, dino/siglip, image_proj, state_proj, skill_proj) keep their names under ``model.``.
    ``fsq_term_train.*`` (the CO-TRAINED terminator, if Stage-1 was trained with train_terminator) is passed
    through too, so it OVERRIDES the raw-FSQ.pt terminator that __init__ built — including its registered
    DINO backbone tensors. If the Stage-1 ckpt has no such keys, the FSQ.pt initialization remains.
    """
    out = {}
    for k, v in raw.items():
        key = k[len("model.") :] if k.startswith("model.") else k
        # Stage-1 checkpoint adapters were historically named ``expert``. Stage-2 treats that LoRA as
        # a permanently frozen VSA component, so route it into the explicit ``expert_lora`` slot.
        key = key.replace(".adapters.expert.", ".adapters.expert_lora.")
        if key.startswith("gemma_expert."):
            out[f"model.paligemma_with_expert.{key}"] = v
        elif key.startswith((
            "cond_encoder.", "dino.", "siglip.", "image_proj.", "state_proj.", "skill_proj.",
            "action_in_proj.", "action_out_proj.", "time_mlp_in.", "time_mlp_out.",
            "fsq_term_train.",   # CO-TRAINED terminator → overrides the FSQ.pt init (if present in the ckpt)
        )):
            out[f"model.{key}"] = v
    return out


def _reverse_skill_endpoint_weights(valid: Tensor, batch: dict, max_weight: float) -> Tensor:
    """Per-sample Stage-2 A weight: max_weight at skill start, linearly decaying to 1 at skill end."""
    bsize, chunk = valid.shape
    device = valid.device
    idx = torch.arange(chunk, device=device).view(1, chunk).expand(bsize, chunk)
    last_valid = torch.where(valid, idx, torch.full_like(idx, -1)).max(dim=1).values
    anchor = last_valid.clamp(min=0)
    if "skill_ds" in batch and "skill_de" in batch:
        ds = batch["skill_ds"].to(device).float().view(bsize)
        de = batch["skill_de"].to(device).float().view(bsize)
        progress = ((ds + anchor.float()) / (ds + de).clamp(min=1.0)).clamp(0.0, 1.0)
    else:
        progress = anchor.float() / max(chunk - 1, 1)
    weights = 1.0 + (float(max_weight) - 1.0) * (1.0 - progress)
    return weights * (last_valid >= 0).float()


# ── Policy ───────────────────────────────────────────────────────────────────────────────────

class SkillVLAPolicy(PI05Policy):
    """Stage-2 SkillVLA policy: VLM skill prediction + flow-matching action expert (see module doc)."""

    config_class = SkillVLAConfig
    name = "skill_vla"

    def __init__(self, config: SkillVLAConfig, stage1_config=None, **kwargs):
        # Skip PI05Policy.__init__ (it builds a PI05Pytorch); reuse PreTrainedPolicy.__init__ and
        # wire our own model + rtc processor.
        super(PI05Policy, self).__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        if stage1_config is None:
            stage1_config = self._load_stage1_config(config)
        self.stage1_config = stage1_config
        self.model = SkillVLAPytorch(config, stage1_config, rtc_processor=self.rtc_processor)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self._apply_freezes()
        self.model.to(config.device)
        self.reset()

    def named_component_params(self) -> dict:
        """Delegate to the pytorch model — used by lerobot_train's per-component drift graph."""
        return self.model.named_component_params()

    @staticmethod
    def _load_stage1_config(config: SkillVLAConfig):
        if not config.stage1_checkpoint_path:
            # SCRATCH mode: no Stage-1 run at all — build the Stage-1-side architecture config directly
            # from the SkillVLA config (the expert/cond weights stay fresh; the VSA identity is carved by
            # the vlm_dropout B batches instead of a Stage-1 pretrain).
            from lerobot.policies.skill_expert.configuration_skill_expert import SkillExpertConfig  # noqa: PLC0415

            return SkillExpertConfig(
                vision_backbone=str(getattr(config, "s1_vision_backbone", "siglip")),
                state_cond_mode=str(getattr(config, "s1_state_cond_mode", "state")),
                skill_fsq_levels=list(config.skill_fsq_levels),
                skill_vocab_size=int(math.prod(config.skill_fsq_levels)),
            )
        try:
            return PreTrainedConfig.from_pretrained(config.stage1_checkpoint_path)
        except Exception as exc:  # noqa: BLE001
            # A few early Stage-1 checkpoints carry retired skill_decoder_dino_* fields. They are not
            # used by the VSA architecture, but modern strict config decoding rejects them before the
            # warm-start can begin. Keep the checkpoint usable by filtering only unknown legacy fields.
            import json  # noqa: PLC0415
            from dataclasses import fields  # noqa: PLC0415
            from lerobot.policies.skill_expert.configuration_skill_expert import SkillExpertConfig  # noqa: PLC0415

            path = Path(config.stage1_checkpoint_path) / "config.json"
            if not path.is_file():
                raise
            raw = json.loads(path.read_text())
            allowed = {f.name for f in fields(SkillExpertConfig)}
            dropped = sorted(set(raw) - allowed)
            if not dropped:
                raise
            log.warning("Stage-1 config has retired fields; ignoring %s while loading %s (%s)",
                        dropped, path, exc)
            return SkillExpertConfig(**{k: v for k, v in raw.items() if k in allowed})

    def _torch_dtype(self) -> torch.dtype:
        return torch.bfloat16 if str(self.config.dtype) == "bfloat16" else torch.float32

    def _apply_continual_freezes(self) -> None:
        """One-time trainability for the sequential LoRA-continual stages (pt_stage / regime_probs_ft).
        Per-batch freeze toggling is unnecessary because each stage's graph only contains what it may
        train. Stage 2 uses its explicit freeze matrix; Stage 3 and FT retain their fixed scopes.
          STAGE 2 (pt_stage="cond"): defaults to vlm_lora + cond_lora only; all bases, vision towers,
              Stage-1 expert_lora, and skill reader/head are frozen.
          STAGE 3 (pt_stage="skill"): ① adapter + reader/head train; EVERYTHING else frozen — vision
              included (settled in Stage 2), so ① learns on the FINAL vision (no moving target).
          FT (regime_probs_ft): expert (+ z/state projections) trains; ②③ frozen (stable residual
              conduit); ①+reader+head train iff ft_train_skill (pair with replay); vision frozen."""
        c, m = self.config, self.model
        ft = bool(getattr(c, "regime_probs_ft", None))    # FT wins (parent config.json carries pt_stage)
        stage = None if ft else m._resolved_pt_stage()
        g = m._regime_groups()
        stage2 = stage == "cond"
        train_vlm = stage2 and not bool(getattr(c, "freeze_vlm", True))
        train_cond = stage2 and not bool(getattr(c, "freeze_cond", True))
        train_expert = ft or (stage2 and not bool(getattr(c, "freeze_expert", True)))
        train_cond_vision = stage2 and not bool(getattr(c, "freeze_cond_vision", True))
        train_vlm_vision = stage2 and not bool(getattr(c, "freeze_vlm_vision", False))

        # These module sweeps include injected adapters; the adapter loop below applies their own flags.
        m._set_requires_grad(g["llm"], train_vlm)
        m._set_requires_grad(g["cond"], train_cond)
        m._set_requires_grad(g["cond_vision"], train_cond_vision)
        mm_proj = getattr(m.paligemma_with_expert.paligemma.model, "multi_modal_projector", None)
        m._set_requires_grad(g["vlm_vision"] + [mm_proj], train_vlm_vision)
        m._set_requires_grad(g["expert"] + [m.state_proj, m.skill_proj], train_expert)
        train_skill = stage == "skill" or (ft and bool(getattr(c, "ft_train_skill", False)))
        m._set_requires_grad([m.skill_reader, m.skill_head], train_skill)
        adapter_flags = {
            "skill": train_skill,
            "vlm_lora": stage2 and not bool(getattr(c, "freeze_vlm_lora", False)),
            "cond_lora": stage2 and not bool(getattr(c, "freeze_cond_lora", False)),
            "expert_lora": stage2 and not bool(getattr(c, "freeze_expert_lora", True)),
        }
        n_train = 0
        for mod in m.modules():
            if isinstance(mod, NamedLoRALinear):
                for name, ad in mod.adapters.items():
                    t = bool(adapter_flags.get(name, False))
                    for p in ad.parameters():
                        p.requires_grad_(t)
                    n_train += int(t)
        tag = "FT" if ft else f"STAGE{'2' if stage == 'cond' else '3'}({stage})"
        print(f"[skillVLA continual] {tag} — trainable adapters: "
              f"{[k for k, v in adapter_flags.items() if v]} ({n_train} wrapped Linears), "
              f"bases(vlm/cond/expert)={train_vlm}/{train_cond}/{train_expert}, "
              f"vision(vlm/cond)={train_vlm_vision}/{train_cond_vision}, "
              f"reader/head={'train' if train_skill else 'frozen'}")

    def _apply_freezes(self) -> None:
        c, m = self.config, self.model
        # LoRA-continual stages → their OWN static trainability (replaces the legacy freeze flags).
        if m._resolved_pt_stage() or getattr(c, "regime_probs_ft", None):
            self._apply_continual_freezes()
            return
        def freeze(module):
            if module is not None:
                for p in module.parameters():
                    p.requires_grad_(False)
        p = float(getattr(c, "vlm_dropout_p", 0.0))
        p_end = getattr(c, "vlm_dropout_p_end", None)
        decay = int(getattr(c, "vlm_dropout_decay_steps", 0) or 0)
        # regimes active if dropout can EVER fire this run (constant p>0 / schedule reaching >0 / 3-way).
        rp = getattr(c, "regime_probs", None)
        regimes_on = (rp is not None and len(rp) == 3) or p > 0.0 or (decay > 0 and p_end is not None and float(p_end) > 0.0)
        g = m._regime_groups()
        three_way = rp is not None and len(rp) == 3
        if regimes_on:
            # CFG per-regime: statically freeze a group ONLY if frozen in EVERY active regime (→ excluded
            # from the optimizer). Otherwise leave it trainable for the per-batch toggle. 3-way checks all
            # of A(vlm_vsa)/B(vsa)/C(c); 2-way checks A(vlm_vsa)/B(vsa).
            for key in ("expert", "cond", "cond_vision", "llm", "vlm_vision"):
                frozen_all = getattr(c, f"freeze_vlm_vsa_{key}") and getattr(c, f"freeze_vsa_{key}")
                if three_way:
                    frozen_all = frozen_all and getattr(c, f"freeze_c_{key}")
                if frozen_all:
                    m._set_requires_grad(g[key], False)
        else:
            # p == 0: NO dropout → every batch is an A (VLM_VSA, VLM-present) batch, so the A dict applies
            # STATICALLY (single source of truth — the old separate freeze_vlm / freeze_cond_encoder /
            # freeze_action_expert / freeze_vlm_vision flags are gone). freeze_vsa is meaningless here.
            if c.freeze_vlm_vsa_expert:
                m._set_requires_grad(g["expert"], False)
            if c.freeze_vlm_vsa_cond:
                m._set_requires_grad(g["cond"], False)
            if c.freeze_vlm_vsa_cond_vision:
                m._set_requires_grad(g["cond_vision"], False)
            if c.freeze_vlm_vsa_llm:
                m._set_requires_grad(g["llm"], False)
            if c.freeze_vlm_vsa_vlm_vision:
                m._set_requires_grad(g["vlm_vision"], False)
        # Skill decoder: static ALWAYS (regime-independent; supervised every batch). Split flags:
        # the head's FSQ readout is codebook-pinned (frozen in FT), the reader may stay trainable to
        # re-ground obs→skill through the probes too (not just the VLM trunk).
        if getattr(c, "freeze_skill_reader", False):
            freeze(m.skill_reader)
        if getattr(c, "freeze_skill_head", False):
            freeze(m.skill_head)

    def get_optim_params(self):
        """Param groups with a differential LR (× optimizer_lr) for the warm-started action expert and
        cond side; the VLM + vision backbones keep the base LR. Frozen params (requires_grad=False)
        are excluded, so empty groups never reach the optimizer. LoRA adapters get their OWN scale
        (lora_lr_scale, collected FIRST so ③ inside cond_encoder doesn't fall into the cond group):
        rank-r B=0 adapters conventionally want ~10-40× the full-finetune LR, and this keeps the
        full-trained parts (e.g. vlm_vision at PT) at the base LR."""
        base = float(self.config.optimizer_lr)
        es = float(getattr(self.config, "expert_lr_scale", 1.0))
        cs = float(getattr(self.config, "cond_lr_scale", 1.0))
        ls = float(getattr(self.config, "lora_lr_scale", 1.0))
        m = self.model
        # state/skill projections feed the action expert's AdaRMS (or its prefix) → expert side.
        expert_mods = [m.paligemma_with_expert.gemma_expert, m.action_in_proj, m.action_out_proj,
                       m.time_mlp_in, m.time_mlp_out, m.state_proj, m.skill_proj]
        cond_mods = [m.cond_encoder, m.image_proj]  # scene stream only (state moved to the expert AdaRMS)

        chosen: set[int] = set()

        def collect(mods):
            out = []
            for mod in mods:
                if mod is None:
                    continue
                for p in mod.parameters():
                    if p.requires_grad and id(p) not in chosen:
                        chosen.add(id(p))
                        out.append(p)
            return out

        lora_params = collect([mod.adapters for mod in m.modules() if isinstance(mod, NamedLoRALinear)])
        expert_params = collect(expert_mods)
        cond_params = collect(cond_mods)
        # FT: co-trained terminator (disjoint from the rest) gets its own LR scale.
        term_params = collect([m.fsq_term_train]) if getattr(m, "fsq_term_train", None) is not None else []
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in chosen]

        groups: list[dict] = []
        if rest:
            groups.append({"params": rest})                       # base optimizer_lr
        if lora_params:
            groups.append({"params": lora_params, "lr": base * ls})
        if expert_params:
            groups.append({"params": expert_params, "lr": base * es})
        if cond_params:
            groups.append({"params": cond_params, "lr": base * cs})
        if term_params:
            ts = float(getattr(self.config, "terminator_lr_scale", 1.0))
            groups.append({"params": term_params, "lr": base * ts})
        return groups

    # ── batch → model inputs ──
    def _cond_images(self, batch: dict) -> list[Tensor]:
        """CURRENT-obs images for the cond-side, as [0,1] floats (the vision encoder normalizes)."""
        device = next(self.parameters()).device
        present = [k for k in self.config.image_features if k in batch]
        if not present:
            raise ValueError(f"No image features in batch. Expected one of {list(self.config.image_features)}.")
        images = []
        for key in present:
            img = batch[key].to(device=device)
            img = img.float() if img.dtype != torch.float32 else img
            if img.ndim == 4 and img.shape[1] != 3 and img.shape[-1] == 3:
                img = img.permute(0, 3, 1, 2)
            images.append(img)
        return images

    def _preprocess_vlm_tensor(self, img: Tensor) -> Tensor:
        """pi05-style VLM image preprocessing (resize-with-pad to image_resolution, [-1,1])."""
        img = img.to(device=next(self.parameters()).device).float()
        channels_first = img.shape[1] == 3
        if channels_first:
            img = img.permute(0, 2, 3, 1)
        if tuple(img.shape[1:3]) != tuple(self.config.image_resolution):
            img = resize_with_pad_torch(img, *self.config.image_resolution)
        img = img * 2.0 - 1.0
        if channels_first:
            img = img.permute(0, 3, 1, 2)
        return img

    def _dataset_start_images(self, batch: dict) -> list[Tensor]:
        """SKILL-START images for the VLM (offline/dataset eval: the SkillVLADataset keys).
        Distinct from the ``self._start_images`` closed-loop snapshot attribute (set in reset)."""
        out = []
        for key in (SKILL_START_IMAGE, SKILL_START_WRIST_IMAGE):
            if key not in batch:
                raise ValueError(f"Missing '{key}' in batch (SkillVLADataset must supply skill-start images).")
            out.append(self._preprocess_vlm_tensor(batch[key]))
        return out

    def _snapshot_vlm_images(self, batch: dict) -> list[Tensor]:
        """Closed-loop: snapshot the CURRENT cameras as the VLM's skill-start view (same physical
        cameras as SKILL_START_*; preprocessed identically)."""
        present = [k for k in self.config.image_features if k in batch]
        if not present:
            raise ValueError(f"No image features in batch. Expected one of {list(self.config.image_features)}.")
        return [self._preprocess_vlm_tensor(batch[k]) for k in present]

    def _dataset_skill_code(self, batch: dict) -> Tensor:
        """GT skill code from the dataset batch (training). Distinct from the ``self._skill_code``
        closed-loop attribute (the active VLM-predicted skill, set in reset/_begin_skill)."""
        code = batch[SKILL_CODE].view(-1).long()
        return code.clamp(0, self.stage1_config.skill_vocab_size - 1)

    def _build_severed_hold_target(self, actions: Tensor, batch: dict, action_dim: int) -> Tensor | None:
        """Stage-1 hold target for C (severed) batches: boundary steps (past skill_de OR episode-end pad)
        → arm deltas 0, gripper held at the last within-skill value; within-skill steps keep the GT action.
        Mirrors modeling_skill_expert.forward. Returns None (→ the model keeps the real cross-skill tail)
        when severed_hold_target is off or the batch lacks skill_de."""
        if not getattr(self.config, "severed_hold_target", True) or "skill_de" not in batch:
            return None
        dev = actions.device
        bsize, K = actions.shape[:2]
        valid = torch.ones(bsize, K, dtype=torch.bool, device=dev)
        pad = batch.get("action_is_pad")
        if pad is not None:
            valid &= ~pad.to(dev).bool()
        de = batch["skill_de"].to(dev).long().view(bsize, 1)         # frames from t to the skill's last frame
        valid &= torch.arange(K, device=dev).view(1, K) <= de
        idx = torch.arange(K, device=dev).view(1, K).expand(bsize, K)
        last_valid = torch.where(valid, idx, torch.full_like(idx, -1)).max(dim=1).values
        anchor = last_valid.clamp(min=0)                             # last within-skill step (= skill end)
        rows = torch.arange(bsize, device=dev)
        g = action_dim - 1                                           # gripper = last real dim (absolute)
        hold = torch.zeros_like(actions[:, 0])                       # (B, max_action_dim): arm + pad = 0
        hold[:, g] = actions[rows, anchor, g]                        # gripper holds the last valid value
        return torch.where(valid.unsqueeze(-1), actions, hold.unsqueeze(1))

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean"):
        cond_images = self._cond_images(batch)
        start_images = self._dataset_start_images(batch)
        lang_tokens, lang_masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        skill_code = self._dataset_skill_code(batch)
        action_dim = self.config.output_features[ACTION].shape[0]

        # Stage-1-style hold action for severed batches. MOTOR_SEV/probes use it as the GT flow target;
        # COND_SEV uses it only to construct the shared teacher/student x_t for its function anchor.
        # Boundary = past current skill end or episode padding; None when skill_de is unavailable.
        hold_actions = self._build_severed_hold_target(actions, batch, action_dim)

        flow_losses, skill_hidden = self.model.forward(
            cond_images, start_images, lang_tokens, lang_masks, batch[OBS_STATE], skill_code, actions,
            hold_actions=hold_actions)
        regime = getattr(self.model, "_last_regime", None)   # "skill"|"cond"|"motor"|"motor_sev"|None(eval)

        skill_loss = None
        if skill_hidden is not None:                          # SKILL regime or the eval measurement path
            skill_loss = self.model.skill_head.loss(skill_hidden, skill_code)

        flow_loss = plain_flow_loss = flow_within = flow_tail = None
        reverse_weights = None
        vf = valid = None
        if flow_losses is not None:                           # COND/MOTOR regimes or the eval path
            flow_losses = flow_losses[:, :, :action_dim]                  # (B, K, real_dim)
            # Supervise valid chunk steps only: drop EPISODE-END padding (action_is_pad — clamped-last
            # repeats, wrong for delta actions). The skill-TRANSITION boundary is deliberately NOT masked
            # here — steps that spill past the current skill's end (into the next skill) are KEPT, so the
            # chunk's cross-skill tail is still supervised. (cumulative-position loss is NOT ported.)
            bsize, K = flow_losses.shape[:2]
            valid = torch.ones(bsize, K, dtype=torch.bool, device=flow_losses.device)
            pad = batch.get("action_is_pad")
            if pad is not None:
                valid &= ~pad.to(flow_losses.device).bool()
            vf = valid.float().unsqueeze(-1)                              # (B, K, 1)
            n_steps = valid.float().sum().clamp(min=1.0)
            plain_flow_loss = (flow_losses * vf).sum() / (n_steps * action_dim)
            flow_loss = plain_flow_loss
            # Stage-2 connected A only: emphasize early-skill chunks. B remains a uniform frozen-VSA
            # function anchor controlled solely by cond_severed_anchor_weight.
            if regime == "cond" and bool(getattr(self.config, "action_weight", False)):
                per_sample = ((flow_losses * vf).sum(dim=(1, 2))
                              / (valid.float().sum(dim=1).clamp(min=1.0) * action_dim))
                reverse_weights = _reverse_skill_endpoint_weights(
                    valid, batch, float(getattr(self.config, "skill_start_loss_weight", 1.0)))
                flow_loss = ((per_sample * reverse_weights).sum()
                             / reverse_weights.sum().clamp(min=1.0))
            # Logging-only within/tail split. COND is GT flow error; COND_SEV is teacher-anchor error,
            # so their absolute values are not directly comparable even within the current skill.
            if "skill_de" in batch:
                with torch.no_grad():
                    de_b = batch["skill_de"].to(flow_losses.device).long().view(bsize, 1)
                    within = valid & (torch.arange(K, device=flow_losses.device).view(1, K) <= de_b)
                    tail = valid & ~within
                    per_step = flow_losses.detach().sum(dim=-1) / action_dim      # (B, K)
                    if int(within.sum()) > 0:
                        flow_within = (per_step * within).sum() / within.float().sum()
                    if int(tail.sum()) > 0:
                        flow_tail = (per_step * tail).sum() / tail.float().sum()

        # SkillVLA policy objective (wandb train/loss) = whatever THIS batch supervises: flow-only on
        # COND/MOTOR batches, λ·skill-only on SKILL batches, both on the eval measurement path.
        loss = None
        if flow_loss is not None:
            loss = flow_loss
        if skill_loss is not None:
            weighted = self.config.skill_loss_weight * skill_loss
            loss = weighted if loss is None else loss + weighted
        total = loss                                          # backpropped objective (+ distill/terminator below)

        # Random-skill distillation to the frozen-PT teacher (MOTOR_SEV batches only): weak MSE on sampled
        # skills → added to the backpropped `total` but EXCLUDED from wandb `loss`; logged via "distill/*".
        vsa_distill = getattr(self.model, "_last_vsa_distill", None)
        if vsa_distill is not None and float(self.config.vsa_distill_weight) > 0.0:
            total = total + self.config.vsa_distill_weight * vsa_distill

        # FT: co-train the terminator on GT signals. Disjoint graph (GT/precomputed inputs + its own params)
        # → added to the BACKPROPPED `total` (only the terminator gets gradient; SkillVLA untouched), but it
        # is EXCLUDED from wandb `loss` (train/loss = the policy loss only) — logged to its own wandb
        # section via "terminator/*" keys (→ train_terminator/*, mirrors Stage-1).
        term_loss = None
        # The terminator consumes the batch's current raw third+wrist frames through
        # its own Stage-1-compatible live vision frontend (no feature cache).
        # 게이트: transition-pack SKILL 배치(stage3)는 skill_ds/de가 없고 이미지 슬롯이 '현재'가 아니라
        # 스킬-START 프레임 → terminator 학습을 건너뜀 (안 건너뛰면 잘못된 프레임으로 조용히 학습됨).
        if (self.config.train_terminator and self.model.fsq_term_train is not None
                and "skill_ds" in batch):
            for k in ("skill_code_true", "skill_de"):
                if k not in batch:
                    raise ValueError(f"train_terminator=True needs '{k}' in the batch (SkillVLADataset).")
            img_3rd = batch.get(f"{OBS_IMAGES}.image")
            img_wrist = batch.get(f"{OBS_IMAGES}.wrist_image")
            if img_3rd is None or img_wrist is None:
                raise ValueError("train_terminator=True needs both third-person and wrist images.")
            true_code = batch["skill_code_true"].view(-1).long().clamp(0, self.stage1_config.skill_vocab_size - 1)
            # RAW state (skill_decoder_state, snapshotted pre-normalization by the processor) — the FSQ
            # terminator normalizes internally with its own min/max; feeding the already quantile-
            # normalized OBS_STATE would double-normalize and diverge from closed-loop eval (which uses
            # raw skill_decoder_state). Images are IDENTITY-normalized so `primary` needs no raw snapshot.
            # FAIL-FAST (no OBS_STATE fallback): resuming with an OLD saved processor lacking the preserve
            # step would silently re-introduce the double-normalization bug — surface it loudly instead.
            raw_state = batch.get("skill_decoder_state")
            if raw_state is None:
                raise ValueError(
                    "train_terminator=True needs RAW 'skill_decoder_state' in the batch (snapshotted "
                    "pre-normalization by SkillVLAPreserveRawStateProcessorStep). It is missing — likely "
                    "resuming with an OLD pre-processor from before that step existed. Rebuild the "
                    "pre-processor; feeding normalized observation.state would double-normalize the FSQ input.")
            prog_pred, term_logits = self.model.terminator_predict(
                true_code, raw_state, img_3rd, wrist_image=img_wrist)
            ds = batch["skill_ds"].float().view(-1).to(prog_pred.device)
            de = batch["skill_de"].float().view(-1).to(prog_pred.device)
            prog_tgt = (ds / (ds + de).clamp_min(1.0)).clamp(0.0, 1.0)        # = ds/(length-1)
            sigma = float(self.config.terminator_end_target_sigma)
            term_tgt = (torch.exp(-(de ** 2) / (2.0 * sigma ** 2)) if sigma > 0 else (de == 0).float())
            pos_w = torch.tensor(float(self.config.terminator_end_pos_weight),
                                 device=term_logits.device, dtype=term_logits.dtype)
            term_prog_l = F.smooth_l1_loss(prog_pred, prog_tgt.to(prog_pred.dtype))
            term_end_l = F.binary_cross_entropy_with_logits(
                term_logits, term_tgt.to(term_logits.dtype), pos_weight=pos_w)
            term_loss = term_prog_l + term_end_l
            total = total + term_loss

        loss_dict = {"loss": loss.detach().item()}         # wandb train/loss = this batch's policy loss
        if flow_loss is not None:
            loss_dict["loss_flow"] = flow_loss.detach().item()
            if regime == "cond":
                loss_dict["action_loss"] = plain_flow_loss.detach().item()
                loss_dict["regime/A_flow_loss"] = plain_flow_loss.detach().item()
                loss_dict["regime/A_objective_loss"] = flow_loss.detach().item()
                if reverse_weights is not None:
                    loss_dict["regime/A_weight_mean"] = reverse_weights.detach().mean().item()
        if skill_loss is not None:
            loss_dict["loss_skill"] = skill_loss.detach().item()
            with torch.no_grad():
                skill_acc = (self.model.skill_head.decode(skill_hidden) == skill_code).float().mean()
            loss_dict["skill_acc"] = skill_acc.item()
        # Per-regime metrics: the whole batch is ONE regime, so each step contributes to exactly ONE
        # regime's keys ("regime/*" is routed by lerobot_train to a separate wandb section train_regime/*).
        if regime is not None:
            loss_dict[f"regime/loss_{regime}"] = loss.detach().item()
            if flow_loss is not None:
                loss_dict[f"regime/flow_{regime}"] = flow_loss.detach().item()
                if regime == "cond_sev":
                    anchor_raw = getattr(self.model, "_last_cond_anchor_raw", None)
                    if anchor_raw is not None:
                        anchor_raw = anchor_raw[:, :, :action_dim]
                        anchor_mean = (anchor_raw * vf).sum() / (valid.float().sum().clamp(min=1.0) * action_dim)
                        loss_dict["regime/anchor_cond_sev"] = anchor_mean.item()
                    loss_dict["regime/anchor_objective_cond_sev"] = flow_loss.detach().item()
                    loss_dict["regime/B_objective_loss"] = flow_loss.detach().item()
            # For cond_sev these are anchor-within/anchor-tail; for connected cond they are GT flow metrics.
            if flow_within is not None:
                loss_dict[f"regime/flow_within_{regime}"] = flow_within.item()
            if flow_tail is not None:
                loss_dict[f"regime/flow_tail_{regime}"] = flow_tail.item()
            if skill_loss is not None:
                loss_dict[f"regime/skill_{regime}"] = skill_loss.detach().item()
        if getattr(self.model, "_last_severed_hold", False):   # hold action was used to construct severed x_t
            loss_dict["regime/severed_hold"] = 1.0
        # train_distill 섹션: motor_sev 배치마다 vsa_gt = severed GT BC flow (== regime/flow_motor_sev 복사)
        # — distill on/off 무관하게 찍혀서 on/off run을 같은 패널에서 공정 비교. distill이 켜지면 그 옆에
        # vsa_tot(backprop 총합)/vsa_local/vsa_global/vsa_gt_drift(teacher-대비 GT 드리프트, 모니터)가 추가됨.
        if regime == "motor_sev" and flow_loss is not None:
            loss_dict["distill/vsa_gt"] = flow_loss.detach().item()
        if vsa_distill is not None:                      # VSA anti-forgetting distillation
            loss_dict["distill/vsa_tot"] = vsa_distill.detach().item()   # backpropped mean (sampled only)
            _parts = getattr(self.model, "_last_vsa_distill_parts", None) or {}
            for _pk, _pv in _parts.items():              # gt_drift(모니터) / local(GT-이웃) / global(전역 빈도)
                loss_dict[f"distill/vsa_{_pk}"] = float(_pv)
        if term_loss is not None:
            # Separate wandb section (mirrors Stage-1): "terminator/*" keys are routed by lerobot_train to
            # train_terminator/* — total (prog+end), progress SmoothL1, termination BCE. Kept OUT of train/*.
            loss_dict["terminator/loss"] = term_loss.detach().item()
            loss_dict["terminator/progress"] = term_prog_l.detach().item()
            loss_dict["terminator/termination"] = term_end_l.detach().item()
            loss_dict["loss_total"] = total.detach().item()   # the actual BACKPROPPED objective (policy + terminator; not logged)
        if reduction == "none":
            if flow_losses is None:                       # per-sample = flow only; needs a flow batch
                raise ValueError("reduction='none' requires a flow batch (got a SKILL-regime forward).")
            per_sample = (flow_losses * vf).sum(dim=(1, 2)) / (valid.float().sum(dim=1).clamp(min=1.0) * action_dim)
            return per_sample, loss_dict
        return total, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Offline / dataset eval: requires skill-start inputs in the batch."""
        self.eval()
        actions = self.model.sample_actions(
            self._cond_images(batch), self._dataset_start_images(batch),
            batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], batch[OBS_STATE], **kwargs)
        action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :action_dim]

    def reset(self):
        """Per-episode state for the closed loop."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._skill_code: Tensor | None = None   # active skill (VLM-predicted at skill start)
        self._start_images: list[Tensor] | None = None  # skill-start view fed to the VLM
        self._start_lang: tuple[Tensor, Tensor] | None = None  # skill-start prompt (tokens, masks)
        self._skill_steps = 0
        # skill_html trace: one record per executed skill (VLM-predicted code + terminator series).
        self._skill_trace: list[dict] = []
        self._cur_skill: dict | None = None   # in-progress skill record
        self._episode_step = 0                # global env step within the episode
        self._oracle_cursor = 0               # index into the GT skill sequence (oracle eval)

    # ── oracle eval: GT skill sequence injected via the cond-encoder (see config.use_gt_skill) ──
    def _oracle_active(self) -> bool:
        return bool(getattr(self.config, "use_gt_skill", False)) and getattr(self, "_forced_seqs", None) is not None

    def set_forced_skill_token_sequences(self, sequences) -> None:
        """Oracle eval: per-episode GT skill sequences — each item a bare code or
        ``{"token": code, "gt_length": frames}`` dict (gt_length drives the "gt" advance mode and the
        skill_html GT-vs-terminator timeline). Called once per task; reset() then zeroes the cursor."""
        self._forced_seqs, self._gt_lengths = [], []
        for seq in sequences:
            codes, lens = [], []
            for x in seq:
                codes.append(int(x["token"] if isinstance(x, dict) else x))
                lens.append(int(x.get("gt_length", x.get("skill_length", 0))) if isinstance(x, dict) else 0)
            self._forced_seqs.append(codes)
            self._gt_lengths.append(lens)
        self.reset()

    def set_reference_skill_token_sequences(self, sequences) -> None:
        return None  # no skill-predictor comparison in the VLM-skill model

    def get_gt_timeline(self) -> dict[int, list[dict]]:
        """skill_html hook: per batch index → GT skill timeline [{token, length}] (length = GT demo
        frame count per skill) for the GT-vs-terminator transition-timing comparison plot."""
        if not self._oracle_active():
            return {}
        return {
            b: [{"token": int(c), "length": int(n)} for c, n in zip(self._forced_seqs[b], self._gt_lengths[b])]
            for b in range(len(self._forced_seqs))
        }

    def _oracle_code(self, batch: dict) -> Tensor:
        """GT skill code at the current cursor, shaped (B,)."""
        device, bsize = batch[OBS_STATE].device, batch[OBS_STATE].shape[0]
        codes = []
        for b in range(bsize):
            seq = self._forced_seqs[b if b < len(self._forced_seqs) else 0]
            codes.append(seq[min(self._oracle_cursor, len(seq) - 1)])
        return torch.tensor(codes, dtype=torch.long, device=device)

    def _oracle_gt_length(self) -> int:
        lens = self._gt_lengths[0] if self._gt_lengths else []
        return int(lens[min(self._oracle_cursor, len(lens) - 1)]) if lens else 0

    def _oracle_at_last(self) -> bool:
        return self._oracle_active() and self._oracle_cursor >= len(self._forced_seqs[0]) - 1

    def _begin_skill(self, batch: dict, lang_tokens: Tensor, lang_masks: Tensor) -> None:
        """Snapshot the current obs (image + prompt) as the skill-start view and predict the new skill.
        At a skill boundary the current frame IS the skill start, so the processor's current-state
        prompt is the skill-start prompt; freeze it (with the images) for the rest of the skill."""
        self._start_images = self._snapshot_vlm_images(batch)
        self._start_lang = (lang_tokens, lang_masks)
        if self._oracle_active():  # teacher-force the GT code (VLM still encodes start obs for its hidden)
            self._skill_code = self._oracle_code(batch)
            source = "oracle"
        else:
            self._skill_code = self.model.predict_skill_code(self._start_images, lang_tokens, lang_masks)
            source = "pred"
        self._skill_steps = 0
        # open a skill_html trace record for this skill (env 0; eval runs single-env per task)
        self._cur_skill = {
            "batch_index": 0,
            "skill_index": len(self._skill_trace),
            "codebook_token": int(self._skill_code.reshape(-1)[0].item()),
            "episode_timestep": int(self._episode_step),
            "end_probs": [],
            "skill_source": source,
        }

    def _should_end_skill(self, batch: dict) -> bool:
        """Whether the CURRENT observation closes the active skill, before its next action is chosen.

        ``self._skill_steps`` is the number of actions already emitted under the current skill.  This
        call observes its next boundary frame, records that frame at the current offset, and decides
        whether the counter would reach its limit *at this frame*.  ``select_action`` then increments
        the counter and, if needed, swaps skill/clears the action queue BEFORE sampling an action.  This
        matches the Stage-1 oracle evaluator: a terminator firing on ``obs_t`` makes ``obs_t`` the start
        observation of the next skill, rather than executing one stale old-skill action first.

        The FSQ terminator always runs when available (including at the safety cap) so skill-html traces
        retain its final progress/end prediction. ``skill_advance_mode="gt"`` (oracle only) gates on the
        same prospective counter instead of the terminator signal.
        """
        next_steps = self._skill_steps + 1
        cap = int(self.config.inference_skill_max_length)
        cap_fired = cap > 0 and next_steps >= cap
        term_fired = False
        if self.model.fsq_term is not None:
            state, image = batch.get("skill_decoder_state"), batch.get("skill_decoder_image")
            if state is not None and image is not None:
                out = self.model.terminator_step(
                    self._skill_code, state, image, batch.get("skill_decoder_wrist"))
                if out is not None:
                    progress, end_prob = out   # progress only gates skill_end / feeds skill_html (no token)
                    if self._cur_skill is not None:  # per-step series for skill_html (_plot_skill_progress)
                        self._cur_skill["end_probs"].append({
                            "skill_step": int(self._skill_steps),
                            "prob": float(end_prob.reshape(-1)[0].item()),
                            "progress": float(progress.reshape(-1)[0].item()),
                        })
                    mode = str(self.config.skill_end_mode)
                    thr = float(self.config.skill_end_threshold)
                    pthr = float(getattr(self.config, "skill_end_progress_threshold", 0.9))
                    if mode == "and":
                        # both must hold: end-prob crosses threshold AND progress is far enough along
                        term_fired = bool(((end_prob >= thr) & (progress >= pthr)).any().item())
                    elif mode == "or":
                        # either suffices: end-prob crosses threshold OR progress is far enough along
                        term_fired = bool(((end_prob >= thr) | (progress >= pthr)).any().item())
                    else:
                        signal = end_prob if mode == "termination" else progress
                        term_fired = bool((signal >= thr).any().item())
        if self._oracle_active() and str(self.config.skill_advance_mode) == "gt":
            return next_steps >= max(1, self._oracle_gt_length())
        return term_fired or cap_fired

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Closed loop: VLM predicts the skill at each (FSQ-terminator-decided) skill start; the action
        expert flow-matches chunks (VLM cached per skill); the terminator gates skill transitions."""
        self.eval()
        if self.model.fsq_term is None and self.config.fsq_path:
            self.model.load_terminator(self.config.fsq_path)
        lang_tokens, lang_masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]

        if self._skill_code is None:
            self._begin_skill(batch, lang_tokens, lang_masks)

        # Check the terminator on the CURRENT env frame before selecting an action.  If it says the
        # active skill ended at obs_t, obs_t is also the start image/state for the NEXT skill and no
        # queued old-skill action may leak into the environment.  Stage-1's OracleSkillExpertPolicy
        # follows this exact order; keeping it here makes VSA≡Stage-1 a closed-loop invariant, not only
        # an open-loop-forward property.
        ends_current_skill = self._should_end_skill(batch)
        self._skill_steps += 1
        # Oracle at its last GT skill: keep running it to episode end (same Stage-1 behavior).
        if ends_current_skill and not self._oracle_at_last():
            if self._cur_skill is not None:         # close the old skill at this boundary frame
                self._cur_skill["length"] = int(self._skill_steps)
                self._skill_trace.append(self._cur_skill)
                self._cur_skill = None
            if self._oracle_active():
                self._oracle_cursor += 1
            self._action_queue.clear()
            # Begin immediately from THIS current frame.  In the full model this also snapshots the
            # correct VLM skill-start image/prompt; delaying to obs_{t+1} would condition the next skill
            # on a stale old-skill action's aftermath.
            self._skill_code, self._start_images = None, None
            self._begin_skill(batch, lang_tokens, lang_masks)

        if len(self._action_queue) == 0:
            start_lang, start_masks = self._start_lang  # frozen skill-start prompt
            actions = self.model.sample_actions(
                self._cond_images(batch), self._start_images, start_lang, start_masks,
                batch[OBS_STATE], skill_code=self._skill_code)
            action_dim = self.config.output_features[ACTION].shape[0]
            actions = actions[:, : self.config.n_action_steps, :action_dim]
            self._action_queue.extend(actions.transpose(0, 1))

        action = self._action_queue.popleft()
        self._episode_step += 1
        return action

    def get_skill_trace(self) -> list[dict]:
        """skill_html hook: per-skill records (VLM-predicted FSQ code, start timestep + length, and
        the terminator end-probability series). The still-open skill is finalized on read so the last
        skill of an episode is included even if the terminator never fired before ``done``."""
        trace = list(self._skill_trace)
        if self._cur_skill is not None:
            rec = dict(self._cur_skill)
            rec["length"] = int(self._skill_steps)
            trace.append(rec)
        return trace

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, config=None, strict: bool = False, **kwargs):
        """Build + warm-start. If the path is a Stage-2 checkpoint → resume (full load). Otherwise the
        path is pi05_base (VLM warm-start) and the action/cond side comes from
        ``config.stage1_checkpoint_path`` (Stage-1 skill_expert)."""
        if config is None:
            config = PreTrainedConfig.from_pretrained(pretrained_name_or_path, **kwargs)
        stage1_config = cls._load_stage1_config(config)
        model = cls(config, stage1_config=stage1_config, **kwargs)
        dtype = model._torch_dtype()

        raw = _load_raw_state_dict(pretrained_name_or_path, kwargs)
        if raw is None:
            log.warning("SkillVLA: no weights at %s; fresh init.", pretrained_name_or_path)
            return model

        # LoRA-wrapped Linears (①②③ injected in __init__) expect their pretrained weight under
        # `.base.*`; PLAIN checkpoints (pi05_base, stage-1) would otherwise leave them at RANDOM init
        # (the pi05 `.base` incident, replayed — cond attention cosine≈0 vs stage-1 in the first run).
        model_keys = set(model.state_dict().keys())

        def _routed_load(state: dict, what: str):
            state, n_routed = route_plain_to_base(state, model_keys)
            if n_routed:
                log.info("SkillVLA %s: routed %d plain proj weights into '.base.*' (LoRA-wrapped).", what, n_routed)
            return model.load_state_dict(state, strict=False)

        is_stage2 = any((".skill_reader." in k) or (".skill_head." in k) for k in raw)
        if is_stage2:
            state = {(k if k.startswith("model.") else f"model.{k}"): v.to(dtype) for k, v in raw.items()}
            # Pre-rename Stage-2 checkpoints used ``cond`` and ``cond_bridge`` for these exact adapters.
            # Preserve their evaluation/resume path while all newly written checkpoints use the clear names.
            state = {
                k.replace(".adapters.cond_bridge.", ".adapters.cond_lora.")
                 .replace(".adapters.cond.", ".adapters.vlm_lora."): v
                for k, v in state.items()
            }
            missing, unexpected = _routed_load(state, "resume")
            log.info("SkillVLA resume: %d loaded, %d missing, %d unexpected.", len(state), len(missing), len(unexpected))
            _apply_vlm_override(model, config, dtype, kwargs)   # eval-only: swap the VLM (PaliGemma) tower
            return model

        # Fresh: pi05 → VLM, Stage-1 → expert/cond side. SCRATCH mode (no Stage-1): the action expert
        # ALSO warm-starts from pi05's own gemma_expert (+action/time projections) — the same init
        # Stage-1 itself used — while cond/reader/head stay fresh (mirroring Stage-1's fresh cond);
        # the terminator warm-starts from config.fsq_path as usual.
        vlm_state = {k: v.to(dtype) for k, v in _remap_pi05_to_vlm(raw).items()}
        m1, _ = _routed_load(vlm_state, "VLM<-pi05")           # ①② wrap the VLM q/k/v/o → route to .base
        if not config.stage1_checkpoint_path:
            expert_state = {}
            for k, v in raw.items():
                key = k[len("model."):] if k.startswith("model.") else k
                if key.startswith(("paligemma_with_expert.gemma_expert.", "action_in_proj.",
                                   "action_out_proj.", "time_mlp_in.", "time_mlp_out.")):
                    expert_state[f"model.{key}"] = v.to(dtype)
            _routed_load(expert_state, "expert<-pi05")
            log.info("SkillVLA SCRATCH: VLM<-pi05 (%d keys), action expert<-pi05 (%d keys); "
                     "cond/reader/head FRESH (no Stage-1). The VSA is carved by vlm_dropout B batches "
                     "(p=%s→%s).", len(vlm_state), len(expert_state),
                     getattr(config, "vlm_dropout_p", 0.0), getattr(config, "vlm_dropout_p_end", None))
            return model
        s1_raw = _load_raw_state_dict(config.stage1_checkpoint_path, kwargs)
        if s1_raw is None:
            raise ValueError(f"Could not load Stage-1 weights at {config.stage1_checkpoint_path}.")
        expert_state = {k: v.to(dtype) for k, v in _remap_stage1_to_expert(s1_raw).items()}
        missing, unexpected = _routed_load(expert_state, "expert/cond<-stage1")
        if model.model._has_stage1_expert_lora:
            missing_lora = [k for k in missing if ".adapters.expert_lora." in k]
            unexpected_lora = [k for k in unexpected if ".adapters.expert_lora." in k]
            if missing_lora or unexpected_lora:
                raise RuntimeError(
                    "Stage-1 expert_lora failed to restore exactly: "
                    f"missing={len(missing_lora)}, unexpected={len(unexpected_lora)}. "
                    "Check that the Stage-1 checkpoint config and its lora_targets/rank match its weights."
                )
        n_term = sum(1 for k in expert_state if ".fsq_term_train." in k)
        if config.train_terminator and n_term:
            missing_term = [k for k in missing if ".fsq_term_train." in k]
            unexpected_term = [k for k in unexpected if ".fsq_term_train." in k]
            if missing_term or unexpected_term:
                raise RuntimeError(
                    "Stage-1 co-trained terminator failed to restore exactly: "
                    f"missing={len(missing_term)}, unexpected={len(unexpected_term)}. "
                    "Check terminator_dino_model_path and the Stage-1/FSQ terminator architecture."
                )
        log.info(
            "SkillVLA warm-start: VLM<-pi05 (%d keys), expert/cond<-stage1 (%d keys). Terminator: %s. "
            "Fresh: skill_reader + skill_head.", len(vlm_state), len(expert_state),
            f"co-trained from Stage-1 ({n_term} keys)" if n_term else "raw FSQ.pt (Stage-1 co-trained none)",
        )
        return model
