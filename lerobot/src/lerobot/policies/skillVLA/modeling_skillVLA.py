"""Stage-2 SkillVLA — VLM predicts the skill, an action expert flow-matches the chunk.

A PaliGemma VLM (warm-started from pi05_base) reads the SKILL-START observation (3rd + wrist
image + the start state discretized into the prompt, pi05-style + language) plus a learnable
skill-query token, and predicts the current FSQ skill code (one categorical per FSQ dim, via
``SkillHead``). An action expert — warm-started from a Stage-1 ``skill_expert`` checkpoint —
generates the action chunk by flow matching from the CURRENT observation, reading the VLM via
PI05-style joint block attention.

Branch (from the Stage-1 checkpoint's ``expert_arch``):
  A (joint, primary): three streams — a Stage-1 cond-encoder over the CURRENT obs (+ teacher-forced
      skill), the VLM over the START obs, and the action expert (action tokens only). The action
      attends cond + the VLM's image/skill tokens (NOT its language) + itself; cond ⊥ VLM; neither
      cond nor VLM attends the action.
  B (fused): two streams — the VLM, and the fused expert whose input is [cond tokens, action tokens]
      (full self-attention) and which additionally reads the VLM's image/skill tokens.

Skill flow: the discrete GT skill is teacher-forced into the cond-encoder (A) / fused expert (B)
through the Stage-1 skill embedding at train time (the VLM's prediction is used at inference). The
VLM's skill-query hidden is the categorical-prediction signal (skill CE loss).

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
        self.expert_arch = stage1_config.expert_arch  # "joint" (branch A) | "fused" (branch B)

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

        # ── Inference-only FSQ terminator (loaded lazily from config.fsq_path) ──
        # Flat FSQ code → z_q uses the FSQ codebook's OWN (little-endian) convention
        # (strides[i]=prod(levels[:i]), z_q = level_id - half), independent of SkillHead's
        # internal mixed-radix — the terminator reads FSQ geometry, the skill_emb does not.
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
            for m in (self.image_proj, self.state_proj, self.skill_emb, self.cond_encoder, self.skill_head):
                if m is not None:
                    m.to(dtype=torch.bfloat16)
            self.skill_query.data = self.skill_query.data.to(torch.bfloat16)

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
        self.state_proj = nn.Linear(s1.max_state_dim, self.expert_width)
        self.skill_emb = nn.Embedding(s1.skill_vocab_size, self.expert_width)

        self.cond_encoder = None
        if self.expert_arch == "joint":
            variant = s1.cond_encoder_variant or s1.action_expert_variant
            self.cond_encoder = _build_gemma(variant, use_adarms=False)
        elif self.expert_arch != "fused":
            raise ValueError(f"expert_arch must be 'fused' or 'joint', got {self.expert_arch!r}")

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

    def _cond_image_state_tokens(self, images: list[Tensor], state: Tensor) -> Tensor:
        """[img1 tokens, img2 tokens, state] → (B, M-1, expert_width). The skill token (which needs the
        predicted code at inference) is appended separately by ``_cond_tokens``."""
        tokens = [self.image_proj(self._image_features(img).to(self._wdtype)) for img in images]
        state = pad_vector(state.to(dtype=torch.float32), self.stage1_config.max_state_dim)
        tokens.append(self.state_proj(state.to(self._wdtype)).unsqueeze(1))
        return torch.cat(tokens, dim=1)

    def _skill_token(self, skill_code: Tensor) -> Tensor:
        return self.skill_emb(skill_code.view(-1).long()).unsqueeze(1).to(self._wdtype)

    def _cond_tokens(self, images: list[Tensor], state: Tensor, skill_code: Tensor) -> Tensor:
        """[img1 tokens, img2 tokens, state, skill] → (B, M, expert_width)."""
        base = self._cond_image_state_tokens(images, state)
        return torch.cat([base, self._skill_token(skill_code)], dim=1)

    def _vlm_tokens(
        self, start_images: list[Tensor], lang_tokens: Tensor, lang_masks: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """VLM prefix [start imgs, language, skill-query] → (embeds (B,nv,W), pad (B,nv), is_lang (nv,)).

        ``is_lang`` marks the language sub-block (excluded from the action's cross-attention)."""
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
        return embeds, pad, is_lang

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
        self, nc: int, vlm_pad: Tensor, vlm_is_lang: Tensor, na: int
    ) -> tuple[Tensor, Tensor]:
        """Branch-A (B,1,T,T) additive mask + the (B,T) validity/pad vector. Streams ordered
        [cond, vlm, action]: cond⊥vlm (each bidirectional within itself); action attends
        cond + vlm-nonlang + action; neither cond nor vlm attends action."""
        bsize, nv = vlm_pad.shape
        device = vlm_pad.device
        total = nc + nv + na
        allow = torch.zeros(bsize, total, total, dtype=torch.bool, device=device)
        allow[:, :nc, :nc] = True                                    # cond block
        allow[:, nc : nc + nv, nc : nc + nv] = True                  # vlm block
        allow[:, nc + nv :, :nc] = True                              # action → cond
        allow[:, nc + nv :, nc : nc + nv] = (~vlm_is_lang)[None, None, :]  # action → vlm-nonlang
        allow[:, nc + nv :, nc + nv :] = True                        # action → action
        col_valid = torch.cat(
            [torch.ones(bsize, nc, dtype=torch.bool, device=device), vlm_pad,
             torch.ones(bsize, na, dtype=torch.bool, device=device)], dim=1)
        allow = allow & col_valid[:, None, :]
        att_4d = torch.where(allow[:, None], 0.0, OPENPI_ATTENTION_MASK_VALUE)
        return att_4d, col_valid

    def _mask_branch_B(
        self, vlm_pad: Tensor, vlm_is_lang: Tensor, ne: int
    ) -> tuple[Tensor, Tensor]:
        """Branch-B (B,1,T,T) additive mask + (B,T) pad. Streams ordered [vlm, expert]:
        vlm bidirectional within itself; expert (cond+action) full self-attn + attends vlm-nonlang."""
        bsize, nv = vlm_pad.shape
        device = vlm_pad.device
        total = nv + ne
        allow = torch.zeros(bsize, total, total, dtype=torch.bool, device=device)
        allow[:, :nv, :nv] = True                                    # vlm block
        allow[:, nv:, nv:] = True                                    # expert self (cond+action)
        allow[:, nv:, :nv] = (~vlm_is_lang)[None, None, :]           # expert → vlm-nonlang
        col_valid = torch.cat(
            [vlm_pad, torch.ones(bsize, ne, dtype=torch.bool, device=device)], dim=1)
        allow = allow & col_valid[:, None, :]
        att_4d = torch.where(allow[:, None], 0.0, OPENPI_ATTENTION_MASK_VALUE)
        return att_4d, col_valid

    # ── joint stream runners ──
    def _run_streams(self, hiddens, layers_per_stream, adarms, att_4d, position_ids):
        """Run the shared transformer over the streams (pre-final-norm hiddens out). All streams
        share the VLM's RoPE and have equal depth (gemma_*: 18 layers)."""
        hiddens = [h.to(self._wdtype) for h in hiddens]
        rotary = self._vlm.rotary_emb
        for layer_idx in range(len(layers_per_stream[0])):
            layers = [ls[layer_idx] for ls in layers_per_stream]
            hiddens = compute_layer_multi(layer_idx, hiddens, layers, att_4d, position_ids, adarms, rotary)
        return hiddens

    def _joint_forward_A(self, cond_tokens, vlm_embeds, vlm_pad, vlm_is_lang, action_tokens, time_cond):
        nc, na = cond_tokens.shape[1], action_tokens.shape[1]
        att_4d, pad = self._mask_branch_A(nc, vlm_pad, vlm_is_lang, na)
        position_ids = torch.cumsum(pad, dim=1) - 1
        layers_per_stream = [self.cond_encoder.model.layers, self._vlm.layers, self._expert.layers]
        adarms = [None, None, time_cond]
        outs = self._run_streams([cond_tokens, vlm_embeds, action_tokens], layers_per_stream, adarms, att_4d, position_ids)
        cond_out, vlm_out, action_out = outs
        cond_out, _ = layernorm_forward(self.cond_encoder.model.norm, cond_out, None)
        vlm_out, _ = layernorm_forward(self._vlm.norm, vlm_out, None)
        action_out, _ = layernorm_forward(self._expert.norm, action_out, time_cond)
        return vlm_out, action_out

    def _joint_forward_B(self, cond_tokens, vlm_embeds, vlm_pad, vlm_is_lang, action_tokens, time_cond):
        expert_in = torch.cat([cond_tokens, action_tokens], dim=1)
        ne = expert_in.shape[1]
        att_4d, pad = self._mask_branch_B(vlm_pad, vlm_is_lang, ne)
        position_ids = torch.cumsum(pad, dim=1) - 1
        layers_per_stream = [self._vlm.layers, self._expert.layers]
        adarms = [None, time_cond]
        outs = self._run_streams([vlm_embeds, expert_in], layers_per_stream, adarms, att_4d, position_ids)
        vlm_out, expert_out = outs
        vlm_out, _ = layernorm_forward(self._vlm.norm, vlm_out, None)
        expert_out, _ = layernorm_forward(self._expert.norm, expert_out, time_cond)
        return vlm_out, expert_out

    def _joint_forward(self, cond_tokens, vlm_embeds, vlm_pad, vlm_is_lang, action_tokens, time_cond):
        """Returns (vlm_out, action_hidden) where action_hidden is the action-token chunk only."""
        if self.expert_arch == "joint":
            vlm_out, action_out = self._joint_forward_A(
                cond_tokens, vlm_embeds, vlm_pad, vlm_is_lang, action_tokens, time_cond)
        else:
            vlm_out, expert_out = self._joint_forward_B(
                cond_tokens, vlm_embeds, vlm_pad, vlm_is_lang, action_tokens, time_cond)
            action_out = expert_out[:, -self.config.chunk_size :]
        return vlm_out, action_out

    def _skill_hidden(self, vlm_out: Tensor) -> Tensor:
        """The skill-query hidden (last VLM token) → SkillHead input."""
        return vlm_out[:, -1]

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

        ``skill_code`` is teacher-forced into the cond-side; the skill CE target is computed by the
        policy from the same code. BC gradients flow into the action expert, the VLM (via the action's
        cross-attention) and the cond-encoder; the skill CE gradient flows into the VLM + skill head.
        """
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        noise = noise.to(dtype=actions.dtype)
        t_exp = time[:, None, None]
        x_t = t_exp * noise + (1 - t_exp) * actions
        u_t = noise - actions

        cond_tokens = self._cond_tokens(cond_images, state, skill_code)
        vlm_embeds, vlm_pad, vlm_is_lang = self._vlm_tokens(start_images, lang_tokens, lang_masks)
        action_tokens = self._action_in(x_t)
        time_cond = self._time_cond(time)

        vlm_out, action_out = self._joint_forward(
            cond_tokens, vlm_embeds, vlm_pad, vlm_is_lang, action_tokens, time_cond)
        v_t = self._action_out(action_out)
        return F.mse_loss(u_t, v_t, reduction="none"), self._skill_hidden(vlm_out)

    # ── inference ──
    @torch.no_grad()
    def predict_skill_code(self, start_images: list[Tensor], lang_tokens: Tensor, lang_masks: Tensor) -> Tensor:
        """Run ONLY the VLM (bidirectional prefix) → argmax FSQ skill code (B,)."""
        vlm_embeds, vlm_pad, _ = self._vlm_tokens(start_images, lang_tokens, lang_masks)
        bsize, nv = vlm_pad.shape
        att_2d = vlm_pad[:, None, :] & vlm_pad[:, :, None]
        att_4d = torch.where(att_2d[:, None], 0.0, OPENPI_ATTENTION_MASK_VALUE)
        position_ids = torch.cumsum(vlm_pad, dim=1) - 1
        out = self._vlm.forward(
            inputs_embeds=vlm_embeds, attention_mask=att_4d, position_ids=position_ids,
            past_key_values=None, use_cache=False, adarms_cond=None,
        ).last_hidden_state
        return self.skill_head.decode(self._skill_hidden(out))

    # ── cached inference (branch A): VLM cached per skill, cond per call, action per denoise step ──
    def _encode_prefix_kv(self, layers, embeds, pad, position_ids, adarms=None):
        """Run a prefix stream standalone (bidirectional within its valid tokens) → per-layer
        post-RoPE (K, V) + the pre-final-norm hidden. Because the streams are block-isolated
        (cond ⊥ vlm, both ⊥ action), this reproduces the joint forward's per-layer K/V for the
        stream exactly, so the cached denoise below is numerically identical to a full joint forward.
        """
        embeds = embeds.to(self._wdtype)
        att_2d = make_att_2d_masks(pad, torch.zeros_like(pad))
        att_4d = torch.where(att_2d[:, None], 0.0, OPENPI_ATTENTION_MASK_VALUE)
        rotary = self._vlm.rotary_emb
        h, kv = embeds, []
        for layer in layers:
            hn, gate = layernorm_forward(layer.input_layernorm, h, adarms)
            shape = (*hn.shape[:-1], -1, layer.self_attn.head_dim)
            q = layer.self_attn.q_proj(hn).view(shape).transpose(1, 2)
            k = layer.self_attn.k_proj(hn).view(shape).transpose(1, 2)
            v = layer.self_attn.v_proj(hn).view(shape).transpose(1, 2)
            dummy = torch.zeros(q.shape[0], q.shape[2], q.shape[-1], device=q.device, dtype=q.dtype)
            cos, sin = rotary(dummy, position_ids)
            q, k = modeling_gemma.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
            kv.append((k, v))
            att_out, _ = modeling_gemma.eager_attention_forward(
                layer.self_attn, q, k, v, att_4d, layer.self_attn.scaling)
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
    def _sample_actions_A(self, cond_images, start_images, lang_tokens, lang_masks, state, skill_code, noise, num_steps):
        """Branch A cached sampling: VLM + cond encoded once, only the action expert runs per step."""
        bsize, device = state.shape[0], state.device
        na = self.config.chunk_size
        cond_base = self._cond_image_state_tokens(cond_images, state)   # (B, nc-1, w)
        nc = cond_base.shape[1] + 1
        vlm_embeds, vlm_pad, vlm_is_lang = self._vlm_tokens(start_images, lang_tokens, lang_masks)
        nv = vlm_embeds.shape[1]

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

        # cond: teacher-force the (predicted) skill, encode once → cached K/V
        cond_tokens = torch.cat([cond_base, self._skill_token(skill_code)], dim=1)
        cond_kv, _ = self._encode_prefix_kv(self.cond_encoder.model.layers, cond_tokens, cond_pad, cond_pos, adarms=None)

        # combined prefix K/V (per layer) + action mask (action sees cond + vlm-nonlang + action)
        prefix_kv = [(torch.cat([cond_kv[i][0], vlm_kv[i][0]], dim=2),
                      torch.cat([cond_kv[i][1], vlm_kv[i][1]], dim=2)) for i in range(len(cond_kv))]
        prefix_valid = torch.cat([cond_pad, vlm_pad & ~vlm_is_lang[None, :]], dim=1)
        cols = torch.cat([prefix_valid, torch.ones(bsize, na, dtype=torch.bool, device=device)], dim=1)
        att_4d = torch.where(cols[:, None, None, :].expand(bsize, 1, na, cols.shape[1]),
                             0.0, OPENPI_ATTENTION_MASK_VALUE)

        if noise is None:
            noise = self.sample_noise((bsize, na, self.config.max_action_dim), device)
        dt, x_t = -1.0 / num_steps, noise
        for step in range(num_steps):
            t = torch.full((bsize,), 1.0 + step * dt, dtype=torch.float32, device=device)
            time_cond = self._time_cond(t)
            h = self._action_in(x_t)
            for i in range(len(prefix_kv)):
                h = self._action_layer_cached(i, h, prefix_kv, att_4d, action_pos, time_cond)
            h, _ = layernorm_forward(self._expert.norm, h, time_cond)
            v_t = self._action_out(h)
            x_t = x_t + dt * v_t
        return x_t

    @torch.no_grad()
    def _sample_actions_B(self, cond_images, start_images, lang_tokens, lang_masks, state, skill_code, noise, num_steps):
        """Branch B (fused) cached sampling: only the VLM is a constant prefix (cached per skill). The
        fused expert's suffix = [cond, action] (full self-attn, cond attends the noisy action so it is
        re-run each step) reads the cached VLM-nonlang K/V — pi05 ``denoise_step`` with cond in the suffix."""
        bsize, device = state.shape[0], state.device
        na = self.config.chunk_size
        vlm_embeds, vlm_pad, vlm_is_lang = self._vlm_tokens(start_images, lang_tokens, lang_masks)
        nv = vlm_embeds.shape[1]
        cond_base = self._cond_image_state_tokens(cond_images, state)
        nc = cond_base.shape[1] + 1
        ne = nc + na

        # positions = single cumsum over [vlm, expert(cond+action)] (identical to the joint forward)
        full_pad = torch.cat([vlm_pad, torch.ones(bsize, ne, dtype=torch.bool, device=device)], dim=1)
        full_pos = torch.cumsum(full_pad, dim=1) - 1
        vlm_pos, suffix_pos = full_pos[:, :nv], full_pos[:, nv:]

        vlm_kv, vlm_h = self._encode_prefix_kv(self._vlm.layers, vlm_embeds, vlm_pad, vlm_pos, adarms=None)
        vlm_h, _ = layernorm_forward(self._vlm.norm, vlm_h, None)
        if skill_code is None:
            skill_code = self.skill_head.decode(vlm_h[:, -1])
        cond_tokens = torch.cat([cond_base, self._skill_token(skill_code)], dim=1)

        # suffix rows attend vlm-nonlang + all suffix (full self-attn within cond+action)
        cols = torch.cat([vlm_pad & ~vlm_is_lang[None, :], torch.ones(bsize, ne, dtype=torch.bool, device=device)], dim=1)
        att_4d = torch.where(cols[:, None, None, :].expand(bsize, 1, ne, cols.shape[1]),
                             0.0, OPENPI_ATTENTION_MASK_VALUE)

        if noise is None:
            noise = self.sample_noise((bsize, na, self.config.max_action_dim), device)
        dt, x_t = -1.0 / num_steps, noise
        for step in range(num_steps):
            t = torch.full((bsize,), 1.0 + step * dt, dtype=torch.float32, device=device)
            time_cond = self._time_cond(t)
            h = torch.cat([cond_tokens, self._action_in(x_t)], dim=1)
            for i in range(len(vlm_kv)):
                h = self._action_layer_cached(i, h, vlm_kv, att_4d, suffix_pos, time_cond)
            h, _ = layernorm_forward(self._expert.norm, h, time_cond)
            v_t = self._action_out(h[:, -na:])
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
        value) defaults to the VLM prediction. A also caches cond; B re-runs the fused suffix each step."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        sampler = self._sample_actions_A if self.expert_arch == "joint" else self._sample_actions_B
        return sampler(cond_images, start_images, lang_tokens, lang_masks, state, skill_code, noise, num_steps)

    # ── FSQ terminator (inference-only skill-transition gating) ──
    def load_terminator(self, path: str) -> None:
        """Load the frozen FSQ checkpoint's terminator (decides skill transitions in closed loop).
        Only ``predict_termination`` is used (the action chunk comes from the flow-matching expert)."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "examples" / "libero"))
        import dataclasses  # noqa: PLC0415

        from FSQ import SplineFSQAE  # noqa: PLC0415

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg_dict = dataclasses.asdict(ckpt["cfg"])
        keys = {"action_dim", "state_dim", "n_control", "spline_degree", "hidden_dim", "fsq_levels",
                "num_layers", "dropout", "max_length", "action_min", "action_max", "delta_min", "delta_max",
                "feat_dim", "n_tokens", "image_encoder_layers", "terminator_use_wrist", "image_encoder_heads",
                "image_model_name", "image_size", "patch_grid", "n_patch_raw", "image_token_dim", "chunk_size",
                "reconstructor_mode"}
        fsq = SplineFSQAE(**{k: v for k, v in cfg_dict.items() if k in keys})
        fsq.load_state_dict(ckpt["model_state"])
        for p in fsq.parameters():
            p.requires_grad_(False)
        fsq.eval()
        self.fsq_term = fsq.to(device=next(self.parameters()).device)
        log.info("Loaded FSQ terminator from %s (state_dim=%s, use_wrist=%s).",
                 path, getattr(fsq, "state_dim", None), getattr(fsq, "terminator_use_wrist", False))

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
    (cond_encoder, dino/siglip, image_proj, state_proj, skill_emb) keep their names under ``model.``."""
    out = {}
    for k, v in raw.items():
        key = k[len("model.") :] if k.startswith("model.") else k
        if key.startswith("gemma_expert."):
            out[f"model.paligemma_with_expert.{key}"] = v
        elif key.startswith((
            "cond_encoder.", "dino.", "siglip.", "image_proj.", "state_proj.", "skill_emb.",
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

    def get_optim_params(self):
        return [p for p in self.parameters() if p.requires_grad]

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
        flow_losses = flow_losses[:, :, :action_dim]
        flow_loss = flow_losses.mean()
        total = flow_loss + self.config.skill_loss_weight * skill_loss

        with torch.no_grad():
            skill_acc = (self.model.skill_head.decode(skill_hidden) == skill_code).float().mean()
        loss_dict = {
            "loss": total.detach().item(),
            "loss_flow": flow_loss.detach().item(),
            "loss_skill": skill_loss.detach().item(),
            "skill_acc": skill_acc.item(),
        }
        if reduction == "none":
            return flow_losses.mean(dim=(1, 2)), loss_dict
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

    def _begin_skill(self, batch: dict, lang_tokens: Tensor, lang_masks: Tensor) -> None:
        """Snapshot the current obs (image + prompt) as the skill-start view and predict the new skill.
        At a skill boundary the current frame IS the skill start, so the processor's current-state
        prompt is the skill-start prompt; freeze it (with the images) for the rest of the skill."""
        self._start_images = self._snapshot_vlm_images(batch)
        self._start_lang = (lang_tokens, lang_masks)
        self._skill_code = self.model.predict_skill_code(self._start_images, lang_tokens, lang_masks)
        self._skill_steps = 0

    def _should_end_skill(self, batch: dict) -> bool:
        """Advance the skill when the FSQ terminator fires (or the safety cap is hit)."""
        cap = int(self.config.inference_skill_max_length)
        if cap > 0 and self._skill_steps >= cap:
            return True
        if self.model.fsq_term is None:
            return False
        state, image = batch.get("skill_decoder_state"), batch.get("skill_decoder_image")
        if state is None or image is None:
            return False
        out = self.model.terminator_step(self._skill_code, state, image, batch.get("skill_decoder_wrist"))
        if out is None:
            return False
        progress, end_prob = out
        signal = end_prob if self.config.skill_end_mode == "termination" else progress
        return bool((signal >= float(self.config.skill_end_threshold)).any().item())

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
        if self._should_end_skill(batch):           # next call begins (and re-predicts) a new skill
            self._skill_code, self._start_images = None, None
            self._action_queue.clear()
        return action

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
