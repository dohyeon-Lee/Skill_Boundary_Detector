"""Stage-2 SkillVLA — VLM predicts the skill, an action expert flow-matches the chunk.

A PaliGemma VLM (warm-started from pi05_base) reads the SKILL-START observation (3rd + wrist
image + the start state discretized into the prompt, pi05-style + language) plus a learnable
skill-query token, and predicts the current FSQ skill code (one categorical per FSQ dim, via
``SkillHead``). An action expert — warm-started from a Stage-1 ``skill_expert`` checkpoint —
generates the action chunk by flow matching from the CURRENT observation, reading the VLM via
PI05-style joint block attention.

Architecture (joint): three streams in a one-directional chain VLM → cond → action — a Stage-1
cond-encoder over the CURRENT obs (images + per-dim discretized state), the VLM over the START obs,
and the action expert (action tokens only). cond attends itself + the VLM's image hiddens (NOT its
language/skill-query tokens — language reaches cond only indirectly via VLM image hiddens that
attended it); the action attends cond + itself (no direct VLM edge). Nothing attends the action.

Skill flow: the skill (z_q) + progress are prepended to the ACTION stream (ae); the GT code is
teacher-forced at train (cond_skill_source=gt) or the VLM's own prediction is fed back via STE
(cond_skill_source=pred); the VLM's skill-query hidden is the categorical-prediction signal (skill CE).

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
from lerobot.policies.pi_gemma import _gated_residual, layernorm_forward
from lerobot.policies.skill_expert.modeling_skill_expert import (
    _build_gemma,
    _build_siglip_vision_tower,
    _load_raw_state_dict,
)
from lerobot.utils.constants import (
    ACTION,
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
    projections, and a joint-mode cond-encoder), a learnable skill-query token, and the ``SkillHead``.
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
        self.skill_query = nn.Parameter(torch.randn(1, 1, vlm_width) * 0.02)
        self.skill_head = SkillHead(vlm_width, config.skill_fsq_levels)

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

        # Match pi05's working dtype for the trainable Stage-1-side tokenizers + skill head, so the
        # bf16 expert stream never sees float32 inputs (the vision encoders stay float32 like pi05's
        # vision_tower — their features are cast to the working dtype before the projections).
        # Stage-1-side tokenizers run in the expert's working dtype (Stage-1 trained them in bf16).
        # The pi05-inherited action_in/out_proj + time_mlp stay float32 (pi05 convention): their
        # outputs are cast to the working dtype only at the attention boundary (_action_in/_action_out).
        if str(config.dtype) == "bfloat16":
            for m in (self.image_proj, self.state_proj, self.skill_proj,
                      self.cond_encoder, self.skill_head):
                if m is not None:
                    m.to(dtype=torch.bfloat16)
            self.skill_query.data = self.skill_query.data.to(torch.bfloat16)

        # FT: a TRAINABLE terminator co-trained on GT signals (built last so the bf16 cast above skips
        # it — it stays float32). Disjoint from the SkillVLA params; warm-starts from config.fsq_path.
        self.fsq_term_train = None
        if getattr(config, "train_terminator", False):
            if not getattr(config, "fsq_path", None):
                raise ValueError("train_terminator=True requires config.fsq_path (the FSQ checkpoint to adapt).")
            self._build_terminator_trainable(config.fsq_path)

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

    @property
    def _wdtype(self) -> torch.dtype:
        return self._vlm.layers[0].self_attn.q_proj.weight.dtype

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
        """VLM prefix [start imgs, language, skill-query] → (embeds (B,nv,W), pad (B,nv), xattn_block (nv,)).

        ``xattn_block`` marks VLM tokens the cond/expert stream must NOT attend: the skill-query
        read-out token always, plus the language sub-block UNLESS cond_attend_language=True. So by
        default cond reads VLM IMAGE tokens only (visual grounding); the skill reaches the action via
        the action-stream prefix (ae), never by attending the skill-query hidden."""
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

        bsize = lang_emb.shape[0]
        embs.append(self.skill_query.expand(bsize, 1, -1))
        pad.append(torch.ones(bsize, 1, dtype=torch.bool, device=lang_emb.device))
        is_lang += [False]

        embeds = torch.cat([e.to(self._wdtype) for e in embs], dim=1)  # img feats are f32; unify to wdtype
        pad = torch.cat(pad, dim=1)
        is_lang = torch.tensor(is_lang, dtype=torch.bool, device=embeds.device)
        # Tokens the cond/expert stream must NOT attend. Always exclude the skill-query (last) token.
        # Language excluded by DEFAULT (cond reads VLM IMAGE tokens only → forces visual grounding);
        # cond_attend_language=True lets cond ALSO attend the VLM language tokens (ablation toggle).
        xattn_block = torch.zeros_like(is_lang)
        xattn_block[-1] = True
        if not bool(getattr(self.config, "cond_attend_language", False)):
            xattn_block = xattn_block | is_lang
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
        self, nc: int, vlm_pad: Tensor, vlm_xattn_block: Tensor, na: int
    ) -> tuple[Tensor, Tensor]:
        """Branch-A (B,1,T,T) additive mask + the (B,T) validity/pad vector. Streams ordered
        [cond, vlm, action], one-directional chain VLM → cond → action: the VLM reads only
        itself; cond reads itself + the VLM tokens it is allowed (~vlm_xattn_block): IMAGE tokens
        ALWAYS, LANGUAGE tokens ONLY if cond_attend_language; the skill-query read-out token is
        ALWAYS excluded (skill reaches the action via the FSQ skill vector / AdaRMS, not by attending
        the skill-query hidden). The action stream is prefix(n_prefix) + action×K, where n_prefix is
        the [skill(, progress)] prefix in `state` mode or 0 in `state_skill` (skill/progress → AdaRMS):
        the conditioning PREFIX is read by the action tokens but reads NOTHING outside itself
        (prefix ⊥ cond, prefix ⊥ action). The ACTION tokens ALWAYS attend cond/scene; they ALSO get a
        DIRECT VLM edge iff action_attend_vlm — and that edge reuses the SAME ~vlm_xattn_block as cond
        (so images always, language iff cond_attend_language). Without action_attend_vlm there is no
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
        allow[:, :nc, nc : nc + nv] = (~vlm_xattn_block)[None, None, :]  # cond → vlm tokens: images always, language iff cond_attend_language (skill-query always excluded)
        allow[:, nc : nc + nv, nc : nc + nv] = True                  # vlm block
        allow[:, pa:pf1, pa:pf1] = True                              # prefix → prefix (skill self; ⊥ cond, ⊥ action)
        allow[:, pf1:, :nc] = True                                   # action → cond (ONLY action tokens read the scene)
        allow[:, pf1:, pa:] = True                                   # action → prefix + action
        if getattr(self.config, "action_attend_vlm", False):        # optional: action → VLM direct, SAME ~vlm_xattn_block as cond (images always, language iff cond_attend_language)
            allow[:, pf1:, nc : nc + nv] = (~vlm_xattn_block)[None, None, :]
        col_valid = torch.cat(
            [torch.ones(bsize, nc, dtype=torch.bool, device=device), vlm_pad,
             torch.ones(bsize, na, dtype=torch.bool, device=device)], dim=1)
        allow = allow & col_valid[:, None, :]
        att_4d = torch.where(allow[:, None], 0.0, OPENPI_ATTENTION_MASK_VALUE)
        return att_4d, col_valid

    # ── joint stream runners ──
    def _run_streams(self, hiddens, layers_per_stream, adarms, att_4d, position_ids):
        """Run the shared transformer over the streams (pre-final-norm hiddens out). All streams
        share the VLM's RoPE and have equal depth (gemma_*: 18 layers). When gradient checkpointing
        is on (training), each layer is recomputed in backward instead of stored (memory↓, ~25% slower)
        — the inference cache paths are @torch.no_grad so they are unaffected."""
        hiddens = [h.to(self._wdtype) for h in hiddens]
        rotary = self._vlm.rotary_emb
        use_ckpt = getattr(self, "gradient_checkpointing_enabled", False) and self.training
        for layer_idx in range(len(layers_per_stream[0])):
            layers = [ls[layer_idx] for ls in layers_per_stream]
            if use_ckpt:
                hiddens = torch.utils.checkpoint.checkpoint(
                    compute_layer_multi, layer_idx, hiddens, layers, att_4d, position_ids, adarms, rotary,
                    use_reentrant=False, preserve_rng_state=False,
                )
            else:
                hiddens = compute_layer_multi(layer_idx, hiddens, layers, att_4d, position_ids, adarms, rotary)
        return hiddens

    def _joint_forward_A(self, cond_tokens, vlm_embeds, vlm_pad, vlm_xattn_block, action_tokens, expert_cond):
        nc, na = cond_tokens.shape[1], action_tokens.shape[1]
        att_4d, pad = self._mask_branch_A(nc, vlm_pad, vlm_xattn_block, na)
        position_ids = torch.cumsum(pad, dim=1) - 1
        layers_per_stream = [self.cond_encoder.model.layers, self._vlm.layers, self._expert.layers]
        # cond/vlm: plain RMSNorm; action ← AdaRMS(expert_cond = time + state [+ skill + progress for
        # state_skill]). In `state` mode skill/progress ride the action prefix instead (see _action_prefix).
        adarms = [None, None, expert_cond]
        outs = self._run_streams([cond_tokens, vlm_embeds, action_tokens], layers_per_stream, adarms, att_4d, position_ids)
        cond_out, vlm_out, action_out = outs
        cond_out, _ = layernorm_forward(self.cond_encoder.model.norm, cond_out, None)
        vlm_out, _ = layernorm_forward(self._vlm.norm, vlm_out, None)
        action_out, _ = layernorm_forward(self._expert.norm, action_out, expert_cond)
        return vlm_out, action_out

    def _joint_forward(self, cond_tokens, vlm_embeds, vlm_pad, vlm_xattn_block, action_tokens, expert_cond):
        """Returns (vlm_out, action_hidden) where action_hidden is the action-CHUNK only — the
        skill/progress prefix (state mode) is dropped by the final slice (no-op in state_skill mode)."""
        vlm_out, action_out = self._joint_forward_A(
            cond_tokens, vlm_embeds, vlm_pad, vlm_xattn_block, action_tokens, expert_cond)
        return vlm_out, action_out[:, -self.config.chunk_size :]

    def _skill_hidden(self, vlm_out: Tensor) -> Tensor:
        """The skill-query hidden (last VLM token) → SkillHead input."""
        return vlm_out[:, -1]

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
    ) -> tuple[Tensor, Tensor]:
        """Single joint forward → (flow_losses (B, chunk, max_action_dim), skill_hidden (B, vlm_width)).

        The skill z fed to the cond/action-prefix is GT teacher-forced (``cond_skill_source="gt"``,
        Stage-2 default) or the VLM's OWN STE-rounded prediction (``"pred"``, FT: matches inference and
        lets the flow loss backprop through SkillHead into the VLM trunk). The skill CE target is the GT
        code either way. BC gradients flow into the action expert, the VLM (via the action's cross-
        attention; plus the cond/prefix skill token in pred mode) and the cond-encoder.

        pred mode runs the VLM once HERE (phase 1) to get the skill before the prefix exists; the joint
        forward reruns it for cross-attention. The VLM stream is independent (self-attn only), so both
        are numerically identical — phase-1's hidden is reused for the skill CE.
        """
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        noise = noise.to(dtype=actions.dtype)
        t_exp = time[:, None, None]
        x_t = t_exp * noise + (1 - t_exp) * actions
        u_t = noise - actions

        vlm_embeds, vlm_pad, vlm_xattn_block = self._vlm_tokens(start_images, lang_tokens, lang_masks)
        pred_hidden = None
        if getattr(self.config, "cond_skill_source", "gt") == "pred":
            pred_hidden = self._skill_hidden(self._vlm_prefix_out(vlm_embeds, vlm_pad))  # phase 1 (grad)
            skill_zq = self._pred_skill_ste_z(pred_hidden)
        else:
            skill_zq = self._code_to_z(skill_code) / self._fsq_half[None, :]

        cond_tokens = self._cond_tokens(cond_images)   # image-only (state→AdaRMS, skill by mode)
        action_tokens = self._action_in(x_t)
        prefix = self._action_prefix_from_z(skill_zq)   # None in state_skill mode
        if prefix is not None:
            action_tokens = torch.cat([prefix, action_tokens], dim=1)
        expert_cond = self._expert_cond_from_z(time, state, skill_zq)

        vlm_out, action_out = self._joint_forward(
            cond_tokens, vlm_embeds, vlm_pad, vlm_xattn_block, action_tokens, expert_cond)
        v_t = self._action_out(action_out)
        skill_hidden = pred_hidden if pred_hidden is not None else self._skill_hidden(vlm_out)
        return F.mse_loss(u_t, v_t, reduction="none"), skill_hidden

    # ── inference ──
    def _vlm_prefix_out(self, vlm_embeds: Tensor, vlm_pad: Tensor) -> Tensor:
        """VLM transformer over a PRECOMPUTED prefix (bidirectional within valid tokens) → hidden
        (B, nv, vlm_width). Grad-capable: the gemma language_model gradient-checkpoints internally
        when training, so the cond_skill_source=pred path (which keeps the graph) stays memory-safe."""
        att_2d = vlm_pad[:, None, :] & vlm_pad[:, :, None]
        # SDPA (the VLM's default attn) requires the additive bias dtype to match the query's; the
        # python-float `torch.where` yields float32, so cast to the bf16 working dtype.
        att_4d = torch.where(att_2d[:, None], 0.0, OPENPI_ATTENTION_MASK_VALUE).to(vlm_embeds.dtype)
        position_ids = torch.cumsum(vlm_pad, dim=1) - 1
        return self._vlm.forward(
            inputs_embeds=vlm_embeds, attention_mask=att_4d, position_ids=position_ids,
            past_key_values=None, use_cache=False, adarms_cond=None,
        ).last_hidden_state

    @torch.no_grad()
    def predict_skill_code(self, start_images: list[Tensor], lang_tokens: Tensor, lang_masks: Tensor) -> Tensor:
        """Run ONLY the VLM (bidirectional prefix) → argmax FSQ skill code (B,)."""
        vlm_embeds, vlm_pad, _ = self._vlm_tokens(start_images, lang_tokens, lang_masks)
        return self.skill_head.decode(self._skill_hidden(self._vlm_prefix_out(vlm_embeds, vlm_pad)))

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

        # positions = single cumsum over [cond, vlm, action] (identical to the training joint forward)
        full_pad = torch.cat([
            torch.ones(bsize, nc, dtype=torch.bool, device=device), vlm_pad,
            torch.ones(bsize, na, dtype=torch.bool, device=device)], dim=1)
        full_pos = torch.cumsum(full_pad, dim=1) - 1
        cond_pos, vlm_pos, action_pos = full_pos[:, :nc], full_pos[:, nc : nc + nv], full_pos[:, nc + nv :]
        cond_pad = torch.ones(bsize, nc, dtype=torch.bool, device=device)

        # VLM: encode once → skill prediction + cached K/V
        vlm_kv, vlm_h = self._encode_prefix_kv(self._vlm.layers, vlm_embeds, vlm_pad, vlm_pos, adarms=None)
        vlm_h, _ = layernorm_forward(self._vlm.norm, vlm_h, None)
        if skill_code is None:
            skill_code = self.skill_head.decode(vlm_h[:, -1])

        # cond: SCENE only (plain RMSNorm, skill-blind), encoded once READING the cached VLM K/V → cache
        cond_kv, _ = self._encode_prefix_kv(
            self.cond_encoder.model.layers, cond_tokens, cond_pad, cond_pos, adarms=None,
            extra_kv=vlm_kv, extra_valid=vlm_pad & ~vlm_xattn_block[None, :])

        # denoise: action stream = prefix (constant) + action×K; attends the cond cache (cond⊥action). With
        # action_attend_vlm, the VLM K/V is ALSO injected so action reads the VLM directly — gated by the
        # SAME ~vlm_xattn_block as cond (images always, language iff cond_attend_language, skill-query never).
        action_prefix = self._action_prefix(skill_code)  # None in state_skill mode
        attend_vlm = getattr(self.config, "action_attend_vlm", False)
        if attend_vlm:                                                  # action keys = [cond, VLM, action-stream]
            prefix_kv = [(torch.cat([ck, vk], dim=2), torch.cat([cv, vv], dim=2))
                         for (ck, cv), (vk, vv) in zip(cond_kv, vlm_kv)]
            npre = nc + nv
            cols = torch.cat([cond_pad, vlm_pad, torch.ones(bsize, na, dtype=torch.bool, device=device)], dim=1)
        else:                                                          # action keys = [cond, action-stream]
            prefix_kv = cond_kv
            npre = nc
            cols = torch.cat([cond_pad, torch.ones(bsize, na, dtype=torch.bool, device=device)], dim=1)
        # prefix ⊥ cond & ⊥ action (reads only itself); action reads cond (+ VLM via ~vlm_xattn_block if
        # attend_vlm: images always, language iff cond_attend_language) + prefix + action.
        # n_prefix=0 (state_skill) → the prefix→prefix block is empty.
        allow = torch.zeros(bsize, 1, na, npre + na, dtype=torch.bool, device=device)
        allow[:, :, :n_prefix, npre : npre + n_prefix] = True           # prefix → prefix
        allow[:, :, n_prefix:, :nc] = True                             # action → cond
        if attend_vlm:
            allow[:, :, n_prefix:, nc : nc + nv] = (~vlm_xattn_block)[None, None, None, :]  # action → VLM: images always, language iff cond_attend_language
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
        return self._sample_actions_A(cond_images, start_images, lang_tokens, lang_masks, state, skill_code,
                                      noise, num_steps)

    # ── FSQ terminator (inference-only skill-transition gating) ──
    # Terminator-path submodules (the ones co-trained in FT; everything else — the FSQ encoder and the
    # reconstructor — stays frozen). dec_z_proj feeds both, but is on the terminator's input path.
    _TERM_TRAIN_MODULES = ("dec_z_proj", "term_state_proj", "dec_image_encoder_term",
                           "dec_image_encoder_term_wrist", "term_pool", "progress_head", "termination_head")

    @staticmethod
    def _construct_fsq(path: str):
        """Build a SplineFSQAE from an FSQ checkpoint + load its weights (terminator + reconstructor;
        the unused encoder weights are dropped). Returns the model on CPU (caller places/casts it)."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "examples" / "libero"))
        import dataclasses  # noqa: PLC0415

        from FSQ import SplineFSQAE  # noqa: PLC0415

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg_dict = dataclasses.asdict(ckpt["cfg"])
        keys = {"action_dim", "enc_dim", "state_dim", "n_control", "spline_degree", "hidden_dim", "fsq_levels",
                "num_layers", "dropout", "length_min", "length_max", "action_min", "action_max", "delta_min", "delta_max", "state_min", "state_max",
                "feat_dim", "n_tokens", "image_encoder_layers", "terminator_use_third", "terminator_use_wrist", "image_encoder_heads",
                "image_model_name", "image_size", "patch_grid", "n_patch_raw", "image_token_dim", "chunk_size",
                "reconstructor_mode"}
        fsq = SplineFSQAE(**{k: v for k, v in cfg_dict.items() if k in keys})
        # The terminator only runs the decoder path; the FSQ encoder is never used here. Drop encoder
        # weights the current model no longer has, keep strictness on terminator/reconstructor weights.
        model_keys = set(fsq.state_dict().keys())
        state = {k: v for k, v in ckpt["model_state"].items() if k in model_keys}
        dropped = [k for k in ckpt["model_state"] if k not in model_keys]
        missing, _ = fsq.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"FSQ terminator checkpoint missing required weights: {sorted(missing)}")
        if dropped:
            log.info("Ignored %d non-terminator FSQ key(s) when loading FSQ (e.g. %s).", len(dropped), dropped[0])
        return fsq

    def load_terminator(self, path: str) -> None:
        """Load the frozen FSQ checkpoint's terminator (decides skill transitions in closed loop).
        Only ``predict_termination`` is used (the action chunk comes from the flow-matching expert)."""
        fsq = self._construct_fsq(path)
        for p in fsq.parameters():
            p.requires_grad_(False)
        fsq.eval()
        self.fsq_term = fsq.to(device=next(self.parameters()).device)
        log.info("Loaded FSQ terminator from %s (state_dim=%s, use_wrist=%s).",
                 path, getattr(fsq, "state_dim", None), getattr(fsq, "terminator_use_wrist", False))

    def _build_terminator_trainable(self, path: str) -> None:
        """FT: a TRAINABLE terminator co-trained on this dataset's GT signals. Warm-starts from the
        FSQ checkpoint; only the terminator-path submodules (_TERM_TRAIN_MODULES) get gradients — the
        FSQ encoder + reconstructor stay frozen. Registered as a submodule so it joins the optimizer
        and is checkpointed; its loss runs on a disjoint graph (GT inputs only) → no SkillVLA effect."""
        fsq = self._construct_fsq(path)
        trainable = {n for n in self._TERM_TRAIN_MODULES if getattr(fsq, n, None) is not None}
        for name, mod in fsq.named_children():
            req = name in trainable
            for p in mod.parameters():
                p.requires_grad_(req)
        fsq.train()
        # float32 throughout (the FSQ was trained in fp32; terminator inputs state/dino are fp32).
        self.fsq_term_train = fsq.to(device=next(self.parameters()).device, dtype=torch.float32)
        n_tr = sum(p.numel() for p in fsq.parameters() if p.requires_grad)
        log.info("Built TRAINABLE FSQ terminator from %s (%d trainable params, modules=%s).",
                 path, n_tr, sorted(trainable))

    def terminator_predict(self, true_code: Tensor, state: Tensor, dino_tokens: Tensor) -> tuple[Tensor, Tensor]:
        """FT co-training forward (grad ON, logits out): GT skill code + current state + current DINO
        tokens → (progress (B,), term_logits (B,)). Inputs are all GT/precomputed, so the graph never
        touches the SkillVLA params (disjoint co-training)."""
        fsq = self.fsq_term_train
        dev = next(fsq.parameters()).device
        st = state.to(device=dev, dtype=torch.float32)
        if st.ndim == 2:
            st = st.unsqueeze(1)                                       # (B, 1, state_dim)
        st = st[..., : int(fsq.state_dim)]
        z = self._code_to_z(true_code.to(self._fsq_strides.device)).to(device=dev, dtype=st.dtype)  # (B, D) unnormalized z_q
        dec = fsq._prepare_decoder_tokens(dino_tokens, states=st)      # (B, 1, N, F)
        B, T = dec.shape[:2]
        lh = fsq.fsq.levels_half.to(z.device, z.dtype)
        zq = torch.maximum(torch.minimum(torch.round(z), lh), -lh)
        z_tok = fsq.dec_z_proj(zq.unsqueeze(1).expand(B, T, -1).to(st.dtype))
        progress, term_logits = fsq._terminate(z_tok, st, dec, None)   # (B, 1), (B, 1)
        return progress[:, 0], term_logits[:, 0]

    def _code_to_z(self, code: Tensor) -> Tensor:
        """Flat FSQ code (B,) → z_q (B, D) in the FSQ codebook's coordinate frame."""
        idx = code.view(-1, 1).long()
        strides, levels = self._fsq_strides[None, :], self._fsq_levels[None, :]
        level_ids = torch.div(idx, strides, rounding_mode="floor") % levels
        return level_ids.float() - self._fsq_half[None, :]

    def _prepare_term_state(self, state: Tensor) -> Tensor:
        s = state.to(device=next(self.parameters()).device, dtype=torch.float32)
        if s.ndim == 2:
            s = s.unsqueeze(1)  # (B, 1, dim)
        sd = int(getattr(self.fsq_term, "state_dim", s.shape[-1]))
        if s.shape[-1] < sd:
            raise ValueError(f"FSQ terminator expects state_dim={sd}, got {s.shape[-1]}-dim raw state.")
        return s[..., :sd]

    def _prepare_term_image(self, img: Tensor | None, steps: int) -> Tensor | None:
        """Accept precomputed FSQ tokens (B,N,F)/(B,T,N,F) or raw RGB; shape to (B,T,…)."""
        if img is None:
            return None
        x = img.to(device=next(self.parameters()).device, dtype=torch.float32)
        n_tok, feat = int(getattr(self.fsq_term, "n_tokens", 0)), int(getattr(self.fsq_term, "feat_dim", 0))
        is_tok_frame = x.ndim == 3 and n_tok and feat and x.shape[-2] == n_tok and x.shape[-1] == feat
        is_tok_seq = x.ndim == 4 and n_tok and feat and x.shape[-2] == n_tok and x.shape[-1] == feat
        if is_tok_frame:
            x = x.unsqueeze(1)
        elif is_tok_seq and x.shape[1] != steps:
            x = x.expand(-1, steps, -1, -1) if x.shape[1] == 1 else x
        return x

    @torch.no_grad()
    def terminator_step(self, code, state, image, wrist=None):
        """Run the FSQ terminator on the CURRENT obs for the active skill → (progress, end_prob), each (B,)."""
        z = self._code_to_z(code)
        st = self._prepare_term_state(state)
        img = self._prepare_term_image(image, st.shape[1])
        if img is None:
            return None
        use_wrist = bool(getattr(self.fsq_term, "terminator_use_wrist", False))
        w = self._prepare_term_image(wrist, st.shape[1]) if use_wrist else None
        if use_wrist and w is None:
            raise ValueError("FSQ terminator_use_wrist=True but no wrist image supplied (skill_decoder_wrist).")
        progress, end_prob = self.fsq_term.predict_termination(z, st, img, w, quantize=True)
        progress = progress[:, 0] if progress.ndim == 2 else progress
        end_prob = end_prob[:, 0] if end_prob.ndim == 2 else end_prob
        return progress, end_prob


# ── Warm-start key remapping ────────────────────────────────────────────────────────────────

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
    (cond_encoder, dino/siglip, image_proj, state_proj, skill_proj) keep their names under ``model.``."""
    out = {}
    for k, v in raw.items():
        key = k[len("model.") :] if k.startswith("model.") else k
        if key.startswith("gemma_expert."):
            out[f"model.paligemma_with_expert.{key}"] = v
        elif key.startswith((
            "cond_encoder.", "dino.", "siglip.", "image_proj.", "state_proj.", "skill_proj.",
            "action_in_proj.", "action_out_proj.", "time_mlp_in.", "time_mlp_out.",
        )):
            out[f"model.{key}"] = v
    return out


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

    @staticmethod
    def _load_stage1_config(config: SkillVLAConfig):
        if not config.stage1_checkpoint_path:
            raise ValueError("SkillVLAConfig.stage1_checkpoint_path is required (Stage-1 skill_expert ckpt).")
        return PreTrainedConfig.from_pretrained(config.stage1_checkpoint_path)

    def _torch_dtype(self) -> torch.dtype:
        return torch.bfloat16 if str(self.config.dtype) == "bfloat16" else torch.float32

    def _apply_freezes(self) -> None:
        c, m = self.config, self.model
        def freeze(module):
            if module is not None:
                for p in module.parameters():
                    p.requires_grad_(False)
        if c.freeze_vlm:
            freeze(m._vlm)
        if c.freeze_vlm_vision:
            freeze(m.paligemma_with_expert.paligemma.model.vision_tower)
        if c.freeze_cond_encoder:
            freeze(m.cond_encoder)
        if c.freeze_action_expert:
            freeze(m._expert)
            for proj in (m.action_in_proj, m.action_out_proj, m.time_mlp_in, m.time_mlp_out):
                freeze(proj)
        if c.freeze_expert_vision:
            freeze(m.dino if m.vision_backbone == "dino" else m.siglip)
        if getattr(c, "freeze_skill_head", False):
            freeze(m.skill_head)   # pin the FSQ readout (skill_query stays trainable)

    def get_optim_params(self):
        """Param groups with a differential LR (× optimizer_lr) for the warm-started action expert and
        cond side; the VLM + vision backbones keep the base LR. Frozen params (requires_grad=False)
        are excluded, so empty groups never reach the optimizer."""
        base = float(self.config.optimizer_lr)
        es = float(getattr(self.config, "expert_lr_scale", 1.0))
        cs = float(getattr(self.config, "cond_lr_scale", 1.0))
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

        expert_params = collect(expert_mods)
        cond_params = collect(cond_mods)
        # FT: co-trained terminator (disjoint from the rest) gets its own LR scale.
        term_params = collect([m.fsq_term_train]) if getattr(m, "fsq_term_train", None) is not None else []
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in chosen]

        groups: list[dict] = []
        if rest:
            groups.append({"params": rest})                       # base optimizer_lr
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

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean"):
        cond_images = self._cond_images(batch)
        start_images = self._dataset_start_images(batch)
        lang_tokens, lang_masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        skill_code = self._dataset_skill_code(batch)

        flow_losses, skill_hidden = self.model.forward(
            cond_images, start_images, lang_tokens, lang_masks, batch[OBS_STATE], skill_code, actions)
        skill_loss = self.model.skill_head.loss(skill_hidden, skill_code)

        action_dim = self.config.output_features[ACTION].shape[0]
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
        flow_loss = (flow_losses * vf).sum() / (n_steps * action_dim)
        loss = flow_loss + self.config.skill_loss_weight * skill_loss   # SkillVLA policy objective (wandb train/loss)
        total = loss                                                    # backpropped objective (+ terminator below)

        # FT: co-train the terminator on GT signals. Disjoint graph (GT/precomputed inputs + its own params)
        # → added to the BACKPROPPED `total` (only the terminator gets gradient; SkillVLA untouched), but it
        # is EXCLUDED from wandb `loss` (train/loss = the policy loss only) — logged separately as loss_terminator.
        term_loss = None
        if self.config.train_terminator and self.model.fsq_term_train is not None:
            for k in ("skill_code_true", "skill_decoder_dino", "skill_ds", "skill_de"):
                if k not in batch:
                    raise ValueError(f"train_terminator=True needs '{k}' in the batch (SkillVLADataset).")
            true_code = batch["skill_code_true"].view(-1).long().clamp(0, self.stage1_config.skill_vocab_size - 1)
            prog_pred, term_logits = self.model.terminator_predict(
                true_code, batch[OBS_STATE], batch["skill_decoder_dino"])
            ds = batch["skill_ds"].float().view(-1).to(prog_pred.device)
            de = batch["skill_de"].float().view(-1).to(prog_pred.device)
            prog_tgt = (ds / (ds + de).clamp_min(1.0)).clamp(0.0, 1.0)        # = ds/(length-1)
            sigma = float(self.config.terminator_end_target_sigma)
            term_tgt = (torch.exp(-(de ** 2) / (2.0 * sigma ** 2)) if sigma > 0 else (de == 0).float())
            pos_w = torch.tensor(float(self.config.terminator_end_pos_weight),
                                 device=term_logits.device, dtype=term_logits.dtype)
            term_loss = (F.smooth_l1_loss(prog_pred, prog_tgt.to(prog_pred.dtype))
                         + F.binary_cross_entropy_with_logits(term_logits, term_tgt.to(term_logits.dtype), pos_weight=pos_w))
            total = total + term_loss

        with torch.no_grad():
            skill_acc = (self.model.skill_head.decode(skill_hidden) == skill_code).float().mean()
        loss_dict = {
            "loss": loss.detach().item(),                  # wandb train/loss = policy loss (flow + λ·skill); NO terminator
            "loss_flow": flow_loss.detach().item(),
            "loss_skill": skill_loss.detach().item(),
            "skill_acc": skill_acc.item(),
        }
        if term_loss is not None:
            loss_dict["loss_terminator"] = term_loss.detach().item()
            loss_dict["loss_total"] = total.detach().item()   # the actual BACKPROPPED objective (policy + terminator)
        if reduction == "none":
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
        """Whether the current skill ends this step. The FSQ terminator runs whenever available (to
        record the skill_html curves and to gate in "terminator" mode); ``skill_advance_mode="gt"``
        (oracle only) instead ends by the GT demo duration. A safety cap force-advances either way."""
        cap = int(self.config.inference_skill_max_length)
        if cap > 0 and self._skill_steps >= cap:
            return True
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
                    if mode == "and":
                        # both must hold: end-prob crosses threshold AND progress is far enough along
                        pthr = float(getattr(self.config, "skill_end_progress_threshold", 0.9))
                        term_fired = bool(((end_prob >= thr) & (progress >= pthr)).any().item())
                    else:
                        signal = end_prob if mode == "termination" else progress
                        term_fired = bool((signal >= thr).any().item())
        if self._oracle_active() and str(self.config.skill_advance_mode) == "gt":
            return self._skill_steps >= max(1, self._oracle_gt_length())
        return term_fired

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

        if len(self._action_queue) == 0:
            start_lang, start_masks = self._start_lang  # frozen skill-start prompt
            actions = self.model.sample_actions(
                self._cond_images(batch), self._start_images, start_lang, start_masks,
                batch[OBS_STATE], skill_code=self._skill_code)
            action_dim = self.config.output_features[ACTION].shape[0]
            actions = actions[:, : self.config.n_action_steps, :action_dim]
            self._action_queue.extend(actions.transpose(0, 1))

        action = self._action_queue.popleft()
        self._skill_steps += 1
        # Oracle at its last GT skill: keep running it to episode end (don't re-begin) — mirrors stage1.
        if self._should_end_skill(batch) and not self._oracle_at_last():
            if self._cur_skill is not None:         # close the skill_html record
                self._cur_skill["length"] = int(self._skill_steps)
                self._skill_trace.append(self._cur_skill)
                self._cur_skill = None
            if self._oracle_active():               # advance the GT skill cursor
                self._oracle_cursor += 1
            self._skill_code, self._start_images = None, None
            self._action_queue.clear()
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

        is_stage2 = any(("skill_query" in k) or (".skill_head." in k) for k in raw)
        if is_stage2:
            state = {(k if k.startswith("model.") else f"model.{k}"): v.to(dtype) for k, v in raw.items()}
            missing, unexpected = model.load_state_dict(state, strict=False)
            log.info("SkillVLA resume: %d loaded, %d missing, %d unexpected.", len(state), len(missing), len(unexpected))
            return model

        # Fresh: pi05 → VLM, Stage-1 → expert/cond side.
        vlm_state = {k: v.to(dtype) for k, v in _remap_pi05_to_vlm(raw).items()}
        m1, _ = model.load_state_dict(vlm_state, strict=False)
        s1_raw = _load_raw_state_dict(config.stage1_checkpoint_path, kwargs)
        if s1_raw is None:
            raise ValueError(f"Could not load Stage-1 weights at {config.stage1_checkpoint_path}.")
        expert_state = {k: v.to(dtype) for k, v in _remap_stage1_to_expert(s1_raw).items()}
        model.load_state_dict(expert_state, strict=False)
        log.info(
            "SkillVLA warm-start: VLM<-pi05 (%d keys), expert/cond<-stage1 (%d keys). "
            "Fresh: skill_query + skill_head.", len(vlm_state), len(expert_state),
        )
        return model
