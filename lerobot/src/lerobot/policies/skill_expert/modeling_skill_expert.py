"""SkillExpert — Stage-1 standalone action expert (no VLM, no language).

Flow-matching action chunk predictor conditioned on:
  - current 3rd-person + wrist images (trainable DINOv3, shared weights),
  - robot state,
  - GT FSQ skill code as its grid coordinate z_q ∈ [-1,1]^D (one Linear token, constant
    within a skill — neighboring codes stay neighboring),
  - skill progress ∈ [0,1] as a SEPARATE Linear token (mirrors the FSQ decoder's
    dec_z_proj / motion_prog_proj split). GT = skill_ds/(ds+de) at train time (jittered for
    robustness); the FSQ terminator's estimate is injected at inference.

All tokens self-attend; only the action-token hidden states are decoded into actions.
Stage 2 (`skill_vla`) adds the VLM and can init its action expert from a Stage-1 checkpoint.
"""

from __future__ import annotations

import copy
import logging
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModel
from transformers.models.auto import CONFIG_MAPPING

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.modeling_pi05 import (
    OPENPI_ATTENTION_MASK_VALUE,
    compute_layer_complete,
    create_sinusoidal_pos_embedding,
    get_gemma_config,
    layernorm_forward,
    make_att_2d_masks,
    pad_vector,
    sample_beta,
)
from lerobot.policies.pi_gemma import PiGemmaForCausalLM
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_skill_expert import SkillExpertConfig

log = logging.getLogger(__name__)


def _build_siglip_vision_tower(image_size: int):
    """Standalone SigLIP vision tower matching the pi05 PaliGemma vision_tower (So400m, patch 14)
    so the pi05 checkpoint's `...vision_tower.vision_model.*` weights load 1:1 (verified)."""
    from transformers import SiglipVisionModel  # noqa: PLC0415

    vlm_cfg = CONFIG_MAPPING["paligemma"]()
    vc = vlm_cfg.vision_config
    vc.image_size = image_size
    vc.intermediate_size = 4304
    vc.projection_dim = 2048
    vc.projector_hidden_act = "gelu_fast"
    return SiglipVisionModel(vc)


def _build_gemma(variant: str, *, use_adarms: bool) -> PiGemmaForCausalLM:
    """A bare PiGemma transformer (our projections feed it, so no vocab embed table / lm_head).
    use_adarms=True for the action expert (AdaRMS on the flow timestep); False for the cond-encoder."""
    cfg = get_gemma_config(variant)
    hf = CONFIG_MAPPING["gemma"](
        head_dim=cfg.head_dim,
        hidden_size=cfg.width,
        intermediate_size=cfg.mlp_dim,
        num_attention_heads=cfg.num_heads,
        num_hidden_layers=cfg.depth,
        num_key_value_heads=cfg.num_kv_heads,
        vocab_size=257152,
        hidden_activation="gelu_pytorch_tanh",
        dtype="float32",
        use_adarms=use_adarms,
        adarms_cond_dim=cfg.width if use_adarms else None,
    )
    model = PiGemmaForCausalLM(config=hf)
    model.model.embed_tokens = None  # tokens come from our projections, not a vocab table
    model.lm_head = None             # unused (we decode actions via action_out_proj)
    model.model.config._attn_implementation = "eager"  # noqa: SLF001  (custom 4D mask)
    return model


class SkillExpertPytorch(nn.Module):
    """The Stage-1 action expert network (see module docstring)."""

    def __init__(self, config: SkillExpertConfig):
        super().__init__()
        self.config = config
        expert_cfg = get_gemma_config(config.action_expert_variant)
        self.width = expert_cfg.width

        # ── Vision encoder, shared across the two cameras (DINOv3 or SigLIP) ──
        self.vision_backbone = config.vision_backbone
        self.dino = None
        self.siglip = None
        self.n_register = 0
        if config.vision_backbone == "dino":
            self.dino = AutoModel.from_pretrained(config.dino_model_path)
            vis_dim = int(self.dino.config.hidden_size)
            self.n_register = int(getattr(self.dino.config, "num_register_tokens", 0))
            self.vision_image_size = config.dino_image_size
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet
            if config.freeze_dino:
                for p in self.dino.parameters():
                    p.requires_grad_(False)
        elif config.vision_backbone == "siglip":
            # Warm-started from pi05's vision_tower in from_pretrained (params separate from the VLM).
            self.siglip = _build_siglip_vision_tower(config.siglip_image_size)
            vis_dim = int(self.siglip.config.hidden_size)
            self.vision_image_size = config.siglip_image_size
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]  # SigLIP: [0,1] → [-1,1]
            if config.freeze_siglip:
                for p in self.siglip.parameters():
                    p.requires_grad_(False)
        else:
            raise ValueError(f"vision_backbone must be 'dino' or 'siglip', got {config.vision_backbone!r}")
        self.register_buffer("_img_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_img_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

        # ── Token projections (all → expert width) ──
        self.image_proj = nn.Linear(vis_dim, self.width)                # image token → expert token
        # State: QUANTILE-normalized [-1,1] state vector → width (pi0-style continuous projection). It
        # rides the ACTION expert (NOT the image-dominated cond stream, where it was starved): in
        # state_cond_mode="token" its output is a prefix token [state, skill, progress]; in "adaln" it
        # is added to the flow-time AdaRMS conditioning. Allocated in both modes (only destination differs).
        self.state_proj = nn.Linear(config.max_state_dim, self.width)
        # Skill: flat code → FSQ grid coordinate z_q (codebook's little-endian frame, the same
        # value the FSQ decoder consumes), normalized per dim to [-1, 1] → ONE token, constant
        # within a skill. The skill PROGRESS is a SEPARATE token (mirrors the FSQ decoder's
        # dec_z_proj / motion_prog_proj split): raw [0, 1] through its own Linear.
        levels = torch.tensor(config.skill_fsq_levels, dtype=torch.long)
        strides = torch.ones_like(levels)
        for i in range(1, len(config.skill_fsq_levels)):
            strides[i] = strides[i - 1] * config.skill_fsq_levels[i - 1]
        self.register_buffer("_fsq_levels", levels, persistent=False)
        self.register_buffer("_fsq_strides", strides, persistent=False)
        self.register_buffer("_fsq_half", (levels - 1).float() / 2.0, persistent=False)
        self.skill_proj = nn.Linear(len(config.skill_fsq_levels), self.width)  # z_q → 1 token
        self.progress_proj = nn.Linear(1, self.width)                          # progress → 1 token

        # ── Flow-matching action head (mirrors PI05) ──
        self.action_in_proj = nn.Linear(config.max_action_dim, self.width)
        self.action_out_proj = nn.Linear(self.width, config.max_action_dim)
        self.time_mlp_in = nn.Linear(self.width, self.width)
        self.time_mlp_out = nn.Linear(self.width, self.width)

        # ── Action expert transformer (Gemma, AdaRMS conditioned on the flow timestep) ──
        self.gemma_expert = _build_gemma(config.action_expert_variant, use_adarms=True)

        # ── Conditioning (joint): a SEPARATE cond-encoder encodes the scene; the action expert takes
        # [skill, progress, action] and reads the cond stream via PI05-style block attention (cond⊥action).
        self._grad_ckpt = False
        variant = config.cond_encoder_variant or config.action_expert_variant
        # full_adaln: the cond-encoder ALSO gets AdaRMS, conditioned on state (state_proj shared with the
        # action stream) → state modulates the scene encoding too. Other modes: plain RMSNorm cond stream.
        self.cond_encoder = _build_gemma(variant, use_adarms=(config.state_cond_mode == "full_adaln"))

    @property
    def _wdtype(self) -> torch.dtype:
        """Working dtype of the expert stream (drives the cast at every token boundary)."""
        return self.action_in_proj.weight.dtype

    def gradient_checkpointing_enable(self) -> None:
        self._grad_ckpt = True  # _run_joint checkpoints its layer loop
        if hasattr(self.gemma_expert, "gradient_checkpointing_enable"):
            self.gemma_expert.gradient_checkpointing_enable()
        if hasattr(self.cond_encoder, "gradient_checkpointing_enable"):
            self.cond_encoder.gradient_checkpointing_enable()
        if self.vision_backbone == "dino" and not self.config.freeze_dino \
                and hasattr(self.dino, "gradient_checkpointing_enable"):
            self.dino.gradient_checkpointing_enable()
        elif self.vision_backbone == "siglip" and not self.config.freeze_siglip \
                and hasattr(self.siglip, "gradient_checkpointing_enable"):
            self.siglip.gradient_checkpointing_enable()

    # ── Flow-matching samplers (copied from PI05Pytorch) ──
    def sample_noise(self, shape, device) -> Tensor:
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device) -> Tensor:
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    # ── Tokenization ──
    def _image_features(self, image: Tensor) -> Tensor:
        """image (B, C, H, W) in [0, 1] → (B, n_tokens, vis_dim) image tokens.
        DINO: CLS + patches (registers dropped). SigLIP: 256 patch tokens (no CLS)."""
        x = image.to(dtype=torch.float32)
        size = (self.vision_image_size, self.vision_image_size)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        x = (x - self._img_mean.float()) / self._img_std.float()
        if self.vision_backbone == "dino":
            x = x.to(dtype=next(self.dino.parameters()).dtype)
            out = self.dino(x).last_hidden_state        # (B, 1 + n_register + num_patches, dino_dim)
            cls = out[:, :1, :]
            patches = out[:, 1 + self.n_register :, :]  # drop CLS + register tokens, keep patches
            return torch.cat([cls, patches], dim=1)
        x = x.to(dtype=next(self.siglip.parameters()).dtype)
        return self.siglip(pixel_values=x).last_hidden_state  # (B, 256, siglip_dim)

    def _code_to_zq(self, code: Tensor) -> Tensor:
        """Flat FSQ code (B,) → normalized grid coordinate z_q/half ∈ [-1, 1]^D (little-endian
        strides — the FSQ codebook's own convention, matching the FSQ decoder's z input)."""
        idx = code.view(-1, 1).long()
        level_ids = torch.div(idx, self._fsq_strides[None, :], rounding_mode="floor") % self._fsq_levels[None, :]
        return (level_ids.float() - self._fsq_half[None, :]) / self._fsq_half[None, :]

    def _cond_tokens(self, images: list[Tensor]) -> Tensor:
        """Scene conditioning tokens → (B, M, width): [img1 patches, img2 patches] — image ONLY.
        State, skill (z_q) and progress are NOT here — they ride the action expert (see _action_prefix
        / _expert_cond), so they escape the image-dominated cond stream where state was starved."""
        tokens = [self.image_proj(self._image_features(image).to(self._wdtype)) for image in images]
        return torch.cat(tokens, dim=1)

    def _action_prefix(self, skill_code: Tensor, skill_progress: Tensor, state: Tensor) -> Tensor | None:
        """Tokens prepended to the action stream (read by action tokens; prefix ⊥ action). Order:
        [state?, skill, progress?]. state_cond_mode="token" → a pi0-style continuous state token leads;
        "adaln"/"full_adaln" → no state token (state drives AdaRMS via _expert_cond); "ae_adaln" → NO
        prefix at all (state+skill+progress ALL ride the action AdaRMS). use_progress_token=False drops
        the progress token."""
        if self.config.state_cond_mode == "ae_adaln":
            return None
        toks = []
        if self.config.state_cond_mode == "token":
            toks.append(self.state_proj(state.to(self._wdtype)).unsqueeze(1))   # pi0-style state token
        toks.append(self.skill_proj(self._code_to_zq(skill_code).to(self._wdtype)).unsqueeze(1))
        if self.config.use_progress_token:
            toks.append(self.progress_proj(skill_progress.view(-1, 1).float().to(self._wdtype)).unsqueeze(1))
        return torch.cat(toks, dim=1)

    def _time_cond(self, timestep: Tensor) -> Tensor:
        t = create_sinusoidal_pos_embedding(
            timestep, self.width, self.config.min_period, self.config.max_period, device=timestep.device
        ).to(self._wdtype)
        t = F.silu(self.time_mlp_in(t))
        t = F.silu(self.time_mlp_out(t))
        return t

    def _state_cond(self, state: Tensor) -> Tensor:
        """Shared state→AdaRMS conditioning vector (the "same adaLN weight" injected into both streams
        in full_adaln): state_proj(state) → (B, width)."""
        return self.state_proj(state.to(self._wdtype))

    def _expert_cond(self, timestep: Tensor, state: Tensor,
                     skill_code: Tensor, skill_progress: Tensor) -> Tensor:
        """AdaRMS conditioning for the ACTION stream — each signal through its own projection, SUMMED
        (DiT-style ⊕): flow-time always; + state in adaln/full_adaln/ae_adaln; + skill (z_q) + progress
        in ae_adaln (which carries NO prefix tokens — everything but the image rides this AdaRMS)."""
        c = self._time_cond(timestep)
        if self.config.state_cond_mode in ("adaln", "full_adaln", "ae_adaln"):
            c = c + self._state_cond(state)
        if self.config.state_cond_mode == "ae_adaln":
            c = c + self.skill_proj(self._code_to_zq(skill_code).to(self._wdtype))
            if self.config.use_progress_token:
                c = c + self.progress_proj(skill_progress.view(-1, 1).float().to(self._wdtype))
        return c

    def _cond_cond(self, state: Tensor) -> Tensor | None:
        """AdaRMS conditioning for the COND stream: state_proj(state) in full_adaln (state modulates the
        scene encoding too, sharing state_proj with the action stream), else None (plain RMSNorm)."""
        return self._state_cond(state) if self.config.state_cond_mode == "full_adaln" else None

    def _run_joint(self, cond_tokens: Tensor, x_t: Tensor, expert_cond: Tensor,
                   action_prefix: Tensor | None = None, cond_cond: Tensor | None = None) -> Tensor:
        """Joint block attention over two streams (PI05 VLM↔expert pattern, cond-encoder as prefix).
        Three blocks: cond (scene, image-only) ⊥ everything; the conditioning prefix
        ([state?, skill, progress?]) reads cond + itself but NOT the action tokens (pi0 prefix⊥action);
        the action tokens read cond + prefix + action. Only the action positions (last K) decode the
        velocity. `expert_cond` is the fused flow-time(+state) AdaRMS conditioning for the action stream;
        `cond_cond` is the cond stream's AdaRMS conditioning (state in full_adaln, else None)."""
        action_tokens = self.action_in_proj(x_t.to(self._wdtype))
        n_chunk = action_tokens.shape[1]
        n_prefix = action_prefix.shape[1] if action_prefix is not None else 0
        if action_prefix is not None:
            action_tokens = torch.cat([action_prefix, action_tokens], dim=1)  # [state?, skill, progress?, action×K]
        embeds = [cond_tokens, action_tokens]
        bsize, n_cond = cond_tokens.shape[:2]
        n_act = action_tokens.shape[1]
        device = cond_tokens.device

        pad_masks = torch.ones(bsize, n_cond + n_act, dtype=torch.bool, device=device)
        # cond block (0…0, bidirectional, ⊥action); prefix block (1,0…) reads cond+prefix only;
        # action block (1,0…) reads cond+prefix+action. (pi0: conditioning prefix does not see actions.)
        ar_list = [0] * n_cond
        if n_prefix > 0:
            ar_list += [1] + [0] * (n_prefix - 1)
        ar_list += [1] + [0] * (n_chunk - 1)
        ar = torch.tensor(ar_list, dtype=torch.bool, device=device)
        att_masks = ar[None, :].expand(bsize, n_cond + n_act)
        att_2d_4d = make_att_2d_masks(pad_masks, att_masks)[:, None, :, :]
        att_2d_4d = torch.where(att_2d_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        time_cond = expert_cond
        # cond stream: AdaRMS(state) in full_adaln else plain RMSNorm; action stream: AdaRMS(time[+state])
        adarms_cond = [cond_cond, time_cond]
        # Reuse PI05's two-stream layer; cond_encoder plays the "prefix" model via a tiny shim.
        shim = SimpleNamespace(model=SimpleNamespace(language_model=self.cond_encoder.model))
        use_ckpt = self._grad_ckpt and self.training
        for layer_idx in range(self.cond_encoder.model.config.num_hidden_layers):
            if use_ckpt:
                embeds = torch.utils.checkpoint.checkpoint(
                    compute_layer_complete, layer_idx, embeds, att_2d_4d, position_ids, adarms_cond,
                    use_reentrant=False, preserve_rng_state=False,
                    paligemma=shim, gemma_expert=self.gemma_expert,
                )
            else:
                embeds = compute_layer_complete(
                    layer_idx, embeds, att_2d_4d, position_ids, adarms_cond,
                    paligemma=shim, gemma_expert=self.gemma_expert,
                )
        action_hidden, _ = layernorm_forward(self.gemma_expert.model.norm, embeds[1], time_cond)
        return self.action_out_proj(action_hidden[:, -n_chunk:].to(self._wdtype)).float()  # action positions only

    # ── Training / inference ──
    def forward(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        state: Tensor,
        skill_code: Tensor,
        skill_progress: Tensor,
        actions: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        """Flow-matching loss. Returns per-(b, t, dim) MSE: (B, chunk_size, max_action_dim)."""
        cond = self._cond_tokens(images)
        action_prefix = self._action_prefix(skill_code, skill_progress, state)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        source = self.sample_noise(actions.shape, actions.device) if noise is None else noise
        source = source.to(dtype=actions.dtype)
        time_exp = time[:, None, None]
        x_t = time_exp * source + (1 - time_exp) * actions
        u_t = source - actions
        expert_cond = self._expert_cond(time, state, skill_code, skill_progress)
        v_t = self._run_joint(cond, x_t, expert_cond, action_prefix, self._cond_cond(state))
        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        state: Tensor,
        skill_code: Tensor,
        skill_progress: Tensor,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        bsize, device = state.shape[0], state.device
        if noise is None:
            noise = self.sample_noise((bsize, self.config.chunk_size, self.config.max_action_dim), device)
        # cond⊥action, so the cond stream (image-only) is identical every step → encode it once, cache
        # its per-layer K/V, and run only the action stream ([prefix, action]) against the cache.
        cond = self._cond_tokens(images)
        action_prefix = self._action_prefix(skill_code, skill_progress, state)
        return self._sample_joint_cached(cond, noise, num_steps, action_prefix, state, skill_code, skill_progress)

    def _sample_joint_cached(self, cond_tokens: Tensor, noise: Tensor, num_steps: int,
                             action_prefix: Tensor | None = None, state: Tensor | None = None,
                             skill_code: Tensor | None = None, skill_progress: Tensor | None = None) -> Tensor:
        """Joint-mode inference with a cached cond stream (mirrors PI05 prefix-cache / denoise_step).
        The cond-encoder runs ONCE → per-layer K/V cache; each denoising step runs only the action
        stream ([state?, skill, progress?, action]) against the cache (cond⊥action), with the
        conditioning prefix ⊥ the action tokens (pi0) and state-fused AdaRMS via _expert_cond."""
        bsize, n_cond = cond_tokens.shape[:2]
        n_chunk = noise.shape[1]
        n_prefix = action_prefix.shape[1] if action_prefix is not None else 0
        n_act = n_prefix + n_chunk
        device = cond_tokens.device

        # Prefix (cond): one bidirectional block, encoded once → past_key_values.
        prefix_pad = torch.ones(bsize, n_cond, dtype=torch.bool, device=device)
        prefix_att = torch.zeros(bsize, n_cond, dtype=torch.bool, device=device)  # all-0 → bidirectional
        prefix_4d = torch.where(make_att_2d_masks(prefix_pad, prefix_att)[:, None], 0.0, OPENPI_ATTENTION_MASK_VALUE)
        prefix_pos = torch.cumsum(prefix_pad, dim=1) - 1
        past_key_values = self.cond_encoder.model.forward(
            inputs_embeds=cond_tokens, attention_mask=prefix_4d, position_ids=prefix_pos,
            past_key_values=None, use_cache=True, adarms_cond=self._cond_cond(state),
        ).past_key_values

        # Suffix: conditioning prefix block ([state?, skill, progress?]) reads cond+prefix only; action
        # block reads cond+prefix+action (prefix ⊥ action, pi0). All suffix tokens see all cond.
        suffix_pad = torch.ones(bsize, n_act, dtype=torch.bool, device=device)
        suffix_ar_list = ([1] + [0] * (n_prefix - 1)) if n_prefix > 0 else []
        suffix_ar_list += [1] + [0] * (n_chunk - 1)
        suffix_ar = torch.tensor(suffix_ar_list, dtype=torch.bool, device=device)
        suffix_att = suffix_ar[None, :].expand(bsize, n_act)
        prefix_pad_2d = prefix_pad[:, None, :].expand(bsize, n_act, n_cond)
        full_2d = torch.cat([prefix_pad_2d, make_att_2d_masks(suffix_pad, suffix_att)], dim=2)
        full_4d = torch.where(full_2d[:, None], 0.0, OPENPI_ATTENTION_MASK_VALUE)
        suffix_pos = n_cond + torch.cumsum(suffix_pad, dim=1) - 1

        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            t = torch.full((bsize,), 1.0 + step * dt, dtype=torch.float32, device=device)
            action_tokens = self.action_in_proj(x_t.to(self._wdtype))
            if action_prefix is not None:
                action_tokens = torch.cat([action_prefix, action_tokens], dim=1)  # [state?, skill, progress?, action×K]
            action_hidden = self.gemma_expert.model.forward(
                inputs_embeds=action_tokens, attention_mask=full_4d, position_ids=suffix_pos,
                past_key_values=copy.deepcopy(past_key_values), use_cache=False,
                adarms_cond=self._expert_cond(t, state, skill_code, skill_progress),
            ).last_hidden_state
            v_t = self.action_out_proj(action_hidden[:, -n_chunk:].to(self._wdtype)).float()  # action positions only
            x_t = x_t + dt * v_t
        return x_t


# ── Checkpoint loading helpers (Stage-1 init from a PI05 checkpoint, or resume) ──────────

def _map_pi05_key(key: str, vision_backbone: str = "dino") -> str | None:
    """Map a PI05 checkpoint key to the SkillExpert model key, or None to drop it.

    Always keeps the action expert: the Gemma transformer (incl. AdaRMS norms), the action
    in/out projections, and the time MLP. When vision_backbone="siglip", ALSO warm-start the
    expert's vision tower from PI05's `vision_tower` (robot-adapted prior). Everything else
    (PaliGemma LLM, multi_modal_projector, expert lm_head, discretized-state path) is dropped.
    """
    if key.startswith("paligemma_with_expert.gemma_expert."):
        rest = key[len("paligemma_with_expert.gemma_expert.") :]
        if rest.startswith("lm_head"):
            return None
        return f"model.gemma_expert.{rest}"
    # SigLIP backbone: PI05 vision_tower → expert's own SigLIP (keys are vision_model.*).
    if vision_backbone == "siglip":
        vt = "paligemma_with_expert.paligemma.model.vision_tower."
        if key.startswith(vt):
            return f"model.siglip.{key[len(vt):]}"
    if key.startswith("paligemma_with_expert."):
        return None
    for proj in ("action_in_proj.", "action_out_proj.", "time_mlp_in.", "time_mlp_out."):
        if key.startswith(proj):
            return f"model.{key}"
    # openpi legacy naming for the time MLP
    if key.startswith("action_time_mlp_in."):
        return "model.time_mlp_in." + key[len("action_time_mlp_in.") :]
    if key.startswith("action_time_mlp_out."):
        return "model.time_mlp_out." + key[len("action_time_mlp_out.") :]
    return None


def _build_state_dict(raw: dict, vision_backbone: str = "dino") -> dict:
    """Remap a raw checkpoint to SkillExpert keys. Handles a PI05 checkpoint (partial init)
    and a SkillExpert checkpoint (resume), keyed by whether PI05 prefixes are present."""
    is_pi05 = any("paligemma_with_expert" in k for k in raw)
    out = {}
    for k, v in raw.items():
        nk = _map_pi05_key(k, vision_backbone) if is_pi05 else (k if k.startswith("model.") else f"model.{k}")
        if nk is not None:
            out[nk] = v
    return out


def _load_raw_state_dict(path: str | Path, kwargs: dict) -> dict | None:
    """Load model.safetensors from a local file/dir or a HF repo id."""
    from safetensors.torch import load_file

    p = Path(path)
    if p.is_file():
        return load_file(str(p))
    if (p / "model.safetensors").is_file():
        return load_file(str(p / "model.safetensors"))
    try:
        from transformers.utils import cached_file

        resolved = cached_file(
            str(path),
            "model.safetensors",
            cache_dir=kwargs.get("cache_dir"),
            token=kwargs.get("token"),
            revision=kwargs.get("revision"),
            local_files_only=kwargs.get("local_files_only", False),
        )
        return load_file(resolved)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not resolve weights at %s: %s", path, exc)
        return None


class SkillExpertPolicy(PreTrainedPolicy):
    """Stage-1 action-expert policy (no VLM).

    Trains the SkillExpert by flow matching on (image, skill, state, action). Initializing
    from a PI05 checkpoint (e.g. lerobot/pi05_base) reuses the action expert's motion prior;
    the DINO encoder and the image/state/skill projections start fresh. A Stage-1 checkpoint
    can later seed the Stage-2 `skill_vla` action expert.
    """

    config_class = SkillExpertConfig
    name = "skill_expert"

    def __init__(self, config: SkillExpertConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = SkillExpertPytorch(config)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(device=config.device, dtype=self._torch_dtype())
        self.reset()

    def _torch_dtype(self) -> torch.dtype:
        return torch.bfloat16 if str(self.config.dtype) == "bfloat16" else torch.float32

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def get_optim_params(self):
        # Optional separate LR for the vision encoder (dino_lr / siglip_lr) vs everything else.
        if self.model.vision_backbone == "dino":
            vis_module, vis_lr = self.model.dino, self.config.dino_lr
        else:
            vis_module, vis_lr = self.model.siglip, self.config.siglip_lr
        if vis_lr is None:
            return self.parameters()
        vis_ids = {id(p) for p in vis_module.parameters()}
        vis = [p for p in vis_module.parameters() if p.requires_grad]
        others = [p for p in self.parameters() if id(p) not in vis_ids and p.requires_grad]
        return [{"params": others}, {"params": vis, "lr": vis_lr}]

    # ── batch → model inputs ──
    def _collect_images(self, batch: dict) -> tuple[list[Tensor], list[Tensor]]:
        """Gather present camera images as [0, 1] float tensors. The vision encoder's own
        normalization (ImageNet for DINO, [-1,1] for SigLIP) is applied inside the model."""
        device = next(self.parameters()).device
        images, masks = [], []
        present = [k for k in self.config.image_features if k in batch]
        if not present:
            raise ValueError(f"No image features in batch. Expected one of {list(self.config.image_features)}.")
        for key in present:
            img = batch[key].to(device=device)
            if img.dtype != torch.float32:
                img = img.float()
            if img.ndim == 4 and img.shape[1] != 3 and img.shape[-1] == 3:
                img = img.permute(0, 3, 1, 2)  # channels-last → channels-first
            images.append(img)
            masks.append(torch.ones(img.shape[0], dtype=torch.bool, device=device))
        return images, masks

    def _skill_code(self, batch: dict) -> Tensor:
        """Current GT FSQ skill code = skill_sequence[skill_index]."""
        seq = batch["skill_sequence"].long()
        idx = batch["skill_index"].long().view(-1, 1).clamp(0, seq.shape[1] - 1)
        code = seq.gather(1, idx).squeeze(1)
        return code.clamp(0, self.config.skill_vocab_size - 1)

    def _skill_progress(self, batch: dict) -> Tensor:
        """Per-frame skill progress ∈ [0, 1] (0 at skill start, 1 at its last frame — the FSQ
        terminator's training target). ``skill_progress`` in the batch (injected at inference
        with the terminator's prediction) wins; otherwise GT = skill_ds / (skill_ds+skill_de)."""
        if "skill_progress" in batch:
            return batch["skill_progress"].float().view(-1).clamp(0.0, 1.0)
        ds = batch["skill_ds"].float().view(-1)
        de = batch["skill_de"].float().view(-1)
        return ds / (ds + de).clamp(min=1.0)

    def forward(self, batch: dict, reduction: str = "mean"):
        images, img_masks = self._collect_images(batch)
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        progress = self._skill_progress(batch)
        if self.training and self.config.progress_jitter > 0:
            # robustness to the terminator's progress-estimation error at inference
            jit = (torch.rand_like(progress) * 2.0 - 1.0) * self.config.progress_jitter
            progress = (progress + jit).clamp(0.0, 1.0)
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)   # → state_proj's fixed width (pi0)
        losses = self.model.forward(
            images, img_masks, state, self._skill_code(batch), progress, actions)
        real_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :real_dim]
        loss_dict = {"loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist()}
        if reduction == "none":
            per_sample = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample.mean().detach().item()
            return per_sample, loss_dict
        loss = losses.mean()
        loss_dict["loss"] = loss.detach().item()
        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, **kwargs) -> Tensor:
        self.eval()
        images, img_masks = self._collect_images(batch)
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)   # → state_proj's fixed width (pi0)
        actions = self.model.sample_actions(
            images, img_masks, state, self._skill_code(batch), self._skill_progress(batch), **kwargs)
        real_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :real_dim]

    @torch.no_grad()
    def select_action(self, batch: dict, **kwargs) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, config=None, strict: bool = False, **kwargs):
        """Build the model and partially load weights. A PI05 checkpoint seeds only the action
        expert (motion prior); a SkillExpert checkpoint resumes fully. DINO/skill/state/image
        projections that are absent stay freshly initialized (reported as missing keys)."""
        if config is None:
            config = PreTrainedConfig.from_pretrained(pretrained_name_or_path, **kwargs)
        model = cls(config, **kwargs)
        raw = _load_raw_state_dict(pretrained_name_or_path, kwargs)
        if raw is None:
            log.warning("SkillExpert: no weights loaded, using fresh init.")
            return model
        state_dict = {k: v.to(model._torch_dtype()) for k, v in _build_state_dict(raw, config.vision_backbone).items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        log.info(
            "SkillExpert weights: %d mapped & loaded, %d missing (fresh init), %d unexpected.",
            len(state_dict), len(missing), len(unexpected),
        )
        return model
