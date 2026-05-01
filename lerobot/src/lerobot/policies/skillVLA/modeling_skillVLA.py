"""SkillVLA policy — stage 3 joint training.

Extends PI05 with skill conditioning and a frozen VAE decoder prior.
All parameters are trainable; skill predictor gradient flows into the VLM.

Architecture changes vs PI05:
  1. embed_suffix  : z (skill latent) projected and prepended to action embeddings.
  2. forward       : prior-start flow matching + SP loss jointly.
  3. denoise_step  : passes z into embed_suffix during inference denoising.
  4. sample_actions: accepts z; caches prior and slices per step.
  5. select_action : skill predictor predicts z; manages prior / step-counter.

For stages 1 & 2 (decoupled pre-training) use skillVLA_decouple instead.

Expected batch keys beyond standard PI05:
  - "skill_latent"        : (B, skill_latent_dim)   current skill z
  - "skill_latent_prev"   : (B, skill_latent_dim)   previous skill z'
  - "skill_boundary"      : (B,)                    1 at first frame of new skill
  - "skill_start_state"   : (B, state_dim)          proprioceptive state at skill start
  - "skill_frame_index"   : (B,)                    step index within current skill (0-based)
  - "skill_progress"      : (B,)                    0-1 progress within current skill
"""

from __future__ import annotations

import copy
import logging
import sys
from collections import deque
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import Tensor, nn

from lerobot.policies.pi05.modeling_pi05 import (
    PI05Policy,
    PI05Pytorch,
    get_gemma_config,
    make_att_2d_masks,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from .configuration_skillVLA import SkillVLAConfig
from .processor_skillVLA import OBS_LANG_TO_ACTION_ATTENTION_MASK
from .skill_predictor import SkillPredictor

log = logging.getLogger(__name__)


# ── Core model ────────────────────────────────────────────────────────────────

class SkillVLAPytorch(PI05Pytorch):
    """PI05Pytorch + skill conditioning + frozen VAE decoder prior."""

    def __init__(self, config: SkillVLAConfig, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)

        action_expert_config = get_gemma_config(config.action_expert_variant)
        paligemma_config     = get_gemma_config(config.paligemma_variant)

        self.z_to_suffix_proj = nn.Linear(config.skill_latent_dim, action_expert_config.width)

        self.skill_predictor = SkillPredictor(
            skill_latent_dim  = config.skill_latent_dim,
            prefix_hidden_dim = paligemma_config.width,
            state_dim         = config.skill_predictor_state_dim,
            hidden_dim        = config.skill_predictor_hidden_dim,
            num_heads         = config.skill_predictor_num_heads,
            num_layers        = config.skill_predictor_num_layers,
            num_query_tokens  = config.skill_predictor_num_query_tokens,
            dropout           = config.skill_predictor_dropout,
        )

        self.vae_decoder = None
        self.register_buffer("_action_q01", torch.empty(0), persistent=False)
        self.register_buffer("_action_q99", torch.empty(0), persistent=False)
        self.register_buffer("_predicted_latent_train_step", torch.zeros((), dtype=torch.long), persistent=False)
        self._last_predicted_latent_prob = 0.0
        self._last_predicted_latent_fraction = 0.0
        self._last_prior_raw_actions: Tensor | None = None
        self._last_prior_normalized_actions: Tensor | None = None
        self._last_prior_lengths: list[int] = []
        if config.vae_decoder_path:
            self.load_vae_decoder(config.vae_decoder_path)

    def set_action_quantile_stats(self, q01: Tensor | np.ndarray | None, q99: Tensor | np.ndarray | None) -> None:
        if q01 is None or q99 is None:
            return
        self._action_q01 = torch.as_tensor(q01, dtype=torch.float32)
        self._action_q99 = torch.as_tensor(q99, dtype=torch.float32)

    def _normalize_prior_actions(self, prior: Tensor) -> Tensor:
        """Map raw VAE-decoded actions into the policy's normalized action space."""
        if self._action_q01.numel() == 0 or self._action_q99.numel() == 0:
            raise RuntimeError(
                "Action quantile stats are required to normalize VAE prior actions. "
                "Run with dataset stats or load a checkpoint containing the normalizer safetensors."
            )

        q01 = self._action_q01.to(device=prior.device, dtype=prior.dtype)
        q99 = self._action_q99.to(device=prior.device, dtype=prior.dtype)
        denom = torch.where(q99 == q01, torch.full_like(q99, 1e-8), q99 - q01)
        return 2.0 * (prior - q01) / denom - 1.0

    # ── VAE decoder ───────────────────────────────────────────────────────────

    def load_vae_decoder(self, path: str) -> None:
        sys.path.insert(
            0, str(Path(__file__).resolve().parents[4] / "examples" / "libero")
        )
        import dataclasses  # noqa: PLC0415
        from spline_vae import GRIPPER_DIM, SplineVAE, spline_decode  # noqa: PLC0415

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["cfg"]
        cfg_dict = dataclasses.asdict(cfg)
        vae_keys = {"action_dim", "state_dim", "n_control", "spline_degree",
                    "hidden_dim", "latent_dim", "num_layers", "dropout",
                    "max_length", "action_min", "action_max"}
        vae = SplineVAE(**{k: v for k, v in cfg_dict.items() if k in vae_keys})
        vae.load_state_dict(ckpt["model_state"])
        for p in vae.parameters():
            p.requires_grad_(False)
        self.vae_decoder = vae
        self._spline_decode = spline_decode
        self._gripper_dim   = GRIPPER_DIM
        log.info(f"Loaded frozen VAE decoder from {path}")

    @torch.no_grad()
    def _compute_full_prior(
        self, z: Tensor, skill_start_state: Tensor
    ) -> Tensor:
        """Run frozen VAE decoder (GPU) → spline interpolation (CPU) → action sequence.

        Args:
            z                : (B, skill_latent_dim)
            skill_start_state: (B, state_dim)
        Returns:
            prior: (B, skill_len, action_dim)  padded to the longest decoded length
        """
        if self.vae_decoder is None:
            self._last_prior_raw_actions = None
            self._last_prior_normalized_actions = None
            self._last_prior_lengths = []
            return None

        vae    = self.vae_decoder
        device = z.device

        # ── MLP on the same device as the VAE weights (GPU after model.to(device)) ──
        vae_device = next(vae.parameters()).device
        ctrl_pts_norm, len_norm = vae.decode(
            z.to(vae_device).to(next(vae.parameters()).dtype),
            skill_start_state.to(vae_device).to(next(vae.parameters()).dtype),
        )  # (B, N, action_dim), (B, 1)

        # ── CPU for spline interpolation ────────────────────────────────────────────
        ctrl_np = ctrl_pts_norm.float().cpu().numpy()   # (B, N, action_dim)
        len_np  = len_norm.float().cpu().numpy()         # (B, 1)
        lo      = vae.action_min.cpu().numpy()
        hi      = vae.action_max.cpu().numpy()
        gripper_idx = (vae.action_dim + self._gripper_dim) % vae.action_dim

        B = z.shape[0]
        results = []
        for b in range(B):
            ctrl = ctrl_np[b].copy()
            # gripper dim: raw logit → binary ±1
            ctrl[:, gripper_idx] = np.where(
                1.0 / (1.0 + np.exp(-ctrl[:, gripper_idx])) > 0.5, 1.0, -1.0
            )
            # denormalize [-1,1] → raw action space
            ctrl = (ctrl + 1) / 2 * (hi - lo + 1e-8) + lo
            T = max(2, round(float(len_np[b, 0]) * vae.max_length))
            results.append(self._spline_decode(ctrl, T, vae.spline_degree))

        max_T      = max(r.shape[0] for r in results)
        action_dim = results[0].shape[1]
        padded = np.zeros((B, max_T, action_dim), dtype=np.float32)
        for b, r in enumerate(results):
            padded[b, :r.shape[0]] = r

        if self.config.vae_prior_position_clip > 0:
            # Action convention: [dx, dy, dz, droll, dpitch, dyaw, gripper].
            # Clamp position in raw action units before quantile normalization.
            pos_clip = float(self.config.vae_prior_position_clip)
            padded[:, :, :3] = np.clip(padded[:, :, :3], -pos_clip, pos_clip)

        if self.config.zero_vae_prior_orientation:
            # Zero in raw action space before quantile normalization.
            padded[:, :, 3:6] = 0.0
        elif self.config.vae_prior_orientation_clip > 0:
            # Keep orientation prior in raw action units, but prevent large
            # decoded rotations from dominating the denoising start point.
            ori_clip = float(self.config.vae_prior_orientation_clip)
            padded[:, :, 3:6] = np.clip(padded[:, :, 3:6], -ori_clip, ori_clip)

        prior = torch.from_numpy(padded).to(device)
        normalized_prior = self._normalize_prior_actions(prior)

        self._last_prior_raw_actions = prior.detach().cpu()
        self._last_prior_normalized_actions = normalized_prior.detach().cpu()
        self._last_prior_lengths = [int(r.shape[0]) for r in results]

        return normalized_prior

    def _get_prior_slice(
        self, prior: Tensor, skill_frame_index: Tensor
    ) -> Tensor:
        """Slice prior for each sample at its current skill step, pad with last action.

        Args:
            prior            : (B, skill_len, action_dim)
            skill_frame_index: (B,) int  — 0-based step within skill
        Returns:
            prior_slice: (B, chunk_size, action_dim)
        """
        B, skill_len, action_dim = prior.shape
        chunk_size = self.config.chunk_size
        result = torch.zeros(B, chunk_size, action_dim, device=prior.device, dtype=prior.dtype)
        for b in range(B):
            t   = max(0, int(skill_frame_index[b].item()))
            if t >= skill_len:
                result[b] = prior[b, skill_len - 1]
                continue
            end = min(t + chunk_size, skill_len)
            n   = end - t
            result[b, :n] = prior[b, t:end]
            if n < chunk_size:
                result[b, n:] = prior[b, end - 1]  # repeat last action
        return result

    # ── embed_prefix override: record lang token count ───────────────────────

    def embed_prefix(self, images, img_masks, tokens, masks, lang_to_action_masks: Tensor | None = None):
        embs, pad_masks, att_masks = super().embed_prefix(images, img_masks, tokens, masks)
        self._n_lang_tokens = tokens.shape[1]
        self._lang_to_action_masks = lang_to_action_masks
        return embs, pad_masks, att_masks

    def _block_lang_attention(
        self,
        att_2d_masks: Tensor,
        n_prefix: int,
        lang_to_action_masks: Tensor | None = None,
    ) -> Tensor:
        """Block action→task-language columns while allowing state/action-marker tokens."""
        n_lang = getattr(self, "_n_lang_tokens", 0)
        if n_lang > 0:
            att_2d_masks = att_2d_masks.clone()
            lang_start = n_prefix - n_lang
            att_2d_masks[:, n_prefix:, lang_start:n_prefix] = False
            if lang_to_action_masks is not None:
                visible_lang = lang_to_action_masks.to(device=att_2d_masks.device, dtype=torch.bool)
                att_2d_masks[:, n_prefix:, lang_start:n_prefix] = visible_lang[:, None, :]
        return att_2d_masks

    def contextualize_prefix(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        *,
        use_cache: bool = False,
    ):
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_4d = self._prepare_attention_masks_4d(prefix_att_2d)

        if (
            self.paligemma_with_expert.paligemma.model.language_model
            .layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(torch.bfloat16)

        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        (prefix_context, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=use_cache,
        )
        return prefix_context, past_key_values

    def get_skill_predictor_prefix(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        *,
        use_cache: bool = False,
    ):
        prefix_source = self.config.skill_predictor_prefix_source
        if prefix_source == "context":
            return self.contextualize_prefix(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                use_cache=use_cache,
            )
        if prefix_source == "embs":
            return prefix_embs, None
        raise ValueError(
            "Unsupported skill_predictor_prefix_source="
            f"{prefix_source!r}. Expected 'context' or 'embs'."
        )

    # ── embed_suffix override: inject z ──────────────────────────────────────

    def embed_suffix(self, noisy_actions: Tensor, timestep: Tensor, z: Tensor | None = None):
        embs, pad_masks, att_masks, adarms_cond = super().embed_suffix(noisy_actions, timestep)
        if z is not None:
            B = embs.shape[0]
            z_token = self.z_to_suffix_proj(z.to(embs.dtype)).unsqueeze(1)  # (B, 1, width)
            embs = torch.cat([z_token, embs], dim=1)                         # (B, 1+chunk_size, width)
            # att_mask=0: z_token behaves like prefix — action tokens can attend to it, not vice versa
            z_pad = torch.ones(B, 1, dtype=pad_masks.dtype, device=pad_masks.device)
            z_att = torch.zeros(B, 1, dtype=att_masks.dtype, device=att_masks.device)
            pad_masks = torch.cat([z_pad, pad_masks], dim=1)
            att_masks = torch.cat([z_att, att_masks], dim=1)
        return embs, pad_masks, att_masks, adarms_cond

    def _predicted_latent_prob(self, train_step: int | None) -> float:
        if not self.config.predicted_latent_sampling:
            return 0.0
        step = 0 if train_step is None else train_step
        if step < self.config.predicted_latent_start_step:
            return 0.0
        max_prob = float(self.config.predicted_latent_max_prob)
        if self.config.predicted_latent_ramp_steps <= 0:
            return max(0.0, min(1.0, max_prob))
        progress = (step - self.config.predicted_latent_start_step) / self.config.predicted_latent_ramp_steps
        return max(0.0, min(1.0, max_prob * progress))

    def _sample_z_for_action(
        self,
        z: Tensor | None,
        z_pred: Tensor | None,
        train_step: int | None,
    ) -> Tensor | None:
        """Scheduled-sample the latent used by the action expert and VAE prior."""
        self._last_predicted_latent_prob = 0.0
        self._last_predicted_latent_fraction = 0.0
        if z is None or z_pred is None:
            return z

        prob = self._predicted_latent_prob(train_step)
        self._last_predicted_latent_prob = prob
        if prob <= 0.0:
            return z

        pred = z_pred.detach() if self.config.predicted_latent_detach else z_pred
        mask = torch.rand(z.shape[0], 1, device=z.device) < prob
        self._last_predicted_latent_fraction = float(mask.float().mean().item())
        return torch.where(mask, pred.to(z.dtype), z)

    # ── Training forward ─────────────────────────────────────────────────────

    def forward(
        self,
        images,
        img_masks,
        tokens,
        masks,
        actions          : Tensor,           # (B, chunk_size, max_action_dim)
        z                : Tensor | None = None,
        z_prev           : Tensor | None = None,
        f_b              : Tensor | None = None,   # (B,) 1 at skill start
        skill_start_state: Tensor | None = None,   # (B, state_dim)
        skill_frame_index: Tensor | None = None,   # (B,) int
        skill_progress   : Tensor | None = None,   # (B,) float in [0, 1]
        lang_to_action_masks: Tensor | None = None,
        noise            : Tensor | None = None,
        time             : Tensor | None = None,
        detach_sp_prefix : bool          = True,   # False → sp_loss gradient flows into VLM
        train_step       : int | None    = None,
    ) -> Tuple[Tensor, Tensor]:
        """Returns (flow_loss [B,chunk,max_dim], skill_predictor_loss scalar)."""
        if f_b is not None and f_b.ndim > 1:
            f_b = f_b.squeeze(-1)
        if skill_frame_index is not None and skill_frame_index.ndim > 1:
            skill_frame_index = skill_frame_index.squeeze(-1)
        if skill_progress is not None and skill_progress.ndim > 1:
            skill_progress = skill_progress.squeeze(-1)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks, lang_to_action_masks=lang_to_action_masks
        )

        # Skill predictor is supervised against teacher-forced z, while z_for_action
        # can gradually switch to predicted z for the action expert and VAE prior.
        z_pred = None
        sp_loss = torch.tensor(0.0, device=actions.device)
        if (
            z is not None
            and z_prev is not None
            and (skill_progress is not None or skill_frame_index is not None)
            and skill_start_state is not None
        ):
            skill_predictor_progress = skill_progress if skill_progress is not None else skill_frame_index
            skill_predictor_prefix, _ = self.get_skill_predictor_prefix(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                use_cache=False,
            )
            sp_prefix = skill_predictor_prefix.detach() if detach_sp_prefix else skill_predictor_prefix
            z_pred = self.skill_predictor(
                z_prev.float(),
                sp_prefix,
                prefix_pad_masks,
                skill_predictor_progress,
                skill_start_state.float(),
            )
            sp_loss = F.mse_loss(z_pred, z.to(z_pred.dtype))

        z_for_action = self._sample_z_for_action(z, z_pred, train_step)

        # ── Prior-start flow matching ─────────────────────────────────────────
        # Source (t=1): prior_slice if available, else Gaussian noise.
        # Target (t=0): full actions (no residual subtraction).
        # Path: x_t = t * source + (1-t) * actions,  u_t = source - actions.
        prior_pad = None
        if z_for_action is not None and skill_start_state is not None and skill_frame_index is not None:
            full_prior = self._compute_full_prior(z_for_action, skill_start_state)
            if full_prior is not None:
                prior_slice = self._get_prior_slice(full_prior, skill_frame_index)
                prior_slice = prior_slice * float(self.config.vae_prior_scale)
                prior_pad = torch.zeros_like(actions)
                action_dim = prior_slice.shape[-1]
                prior_pad[:, :, :action_dim] = prior_slice.to(actions.dtype)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        source    = prior_pad if prior_pad is not None else noise
        time_exp  = time[:, None, None]
        x_t = time_exp * source + (1 - time_exp) * actions
        u_t = source - actions

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            x_t, time, z=z_for_action
        )

        if (
            self.paligemma_with_expert.paligemma.model.language_model
            .layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(torch.bfloat16)
            prefix_embs = prefix_embs.to(torch.bfloat16)

        pad_masks     = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks     = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks  = make_att_2d_masks(pad_masks, att_masks)
        if self.config.block_lang_to_action:
            att_2d_masks = self._block_lang_attention(
                att_2d_masks, prefix_pad_masks.shape[1], lang_to_action_masks=lang_to_action_masks
            )
        position_ids  = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def _fwd(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask  = att_2d_masks_4d,
                position_ids    = position_ids,
                past_key_values = None,
                inputs_embeds   = [prefix_embs, suffix_embs],
                use_cache       = False,
                adarms_cond     = [None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            _fwd, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )
        suffix_out = suffix_out[:, -self.config.chunk_size:].to(torch.float32)
        v_t        = self._apply_checkpoint(self.action_out_proj, suffix_out)
        flow_loss  = F.mse_loss(u_t, v_t, reduction="none")

        return flow_loss, sp_loss

    # ── Inference: denoise_step override (pass z) ────────────────────────────

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep, z=None):
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            x_t, timestep, z=z
        )

        suffix_len  = suffix_pad_masks.shape[1]
        batch_size  = prefix_pad_masks.shape[0]
        prefix_len  = prefix_pad_masks.shape[1]

        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        if self.config.block_lang_to_action:
            n_lang = getattr(self, "_n_lang_tokens", 0)
            if n_lang > 0:
                lang_to_action_masks = getattr(self, "_lang_to_action_masks", None)
                lang_start = prefix_len - n_lang
                prefix_pad_2d = prefix_pad_2d.clone()
                prefix_pad_2d[:, :, lang_start:prefix_len] = False
                if lang_to_action_masks is not None:
                    visible_lang = lang_to_action_masks.to(device=prefix_pad_2d.device, dtype=torch.bool)
                    prefix_pad_2d[:, :, lang_start:prefix_len] = visible_lang[:, None, :]
        suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d   = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)

        prefix_offsets  = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids    = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_4d  = self._prepare_attention_masks_4d(full_att_2d)

        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        past_key_values = copy.deepcopy(past_key_values)
        (_, suffix_out_full), _ = self.paligemma_with_expert.forward(
            attention_mask  = full_att_2d_4d,
            position_ids    = position_ids,
            past_key_values = past_key_values,
            inputs_embeds   = [None, suffix_embs],
            use_cache       = True,
            adarms_cond     = [None, adarms_cond],
        )
        suffix_out = suffix_out_full[:, -self.config.chunk_size:].to(torch.float32)
        return self.action_out_proj(suffix_out)

    # ── Inference: sample_actions override (pass z, add prior) ───────────────

    @torch.no_grad()
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        z                   : Tensor | None = None,   # (B, skill_latent_dim)
        skill_start_state   : Tensor | None = None,   # (B, state_dim)  — at skill boundary
        prior_cache         : Tensor | None = None,   # (B, skill_len, action_dim) precomputed
        skill_step          : int = 0,                # current step within skill
        lang_to_action_masks: Tensor | None = None,
        noise               : Tensor | None = None,
        num_steps           : int | None = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor | None]:
        """Denoise from prior (or noise) → action.  Returns (actions, prior_cache)."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize  = tokens.shape[0]
        device = tokens.device

        # Compute / reuse full prior
        if prior_cache is None and z is not None and skill_start_state is not None:
            prior_cache = self._compute_full_prior(z, skill_start_state)

        # Starting point: prior slice at current skill_step (fast path), else Gaussian noise.
        # The decoder may still be loaded when use_vae_prior=false so eval can plot decoded skills.
        shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
        if prior_cache is not None and self.config.use_vae_prior:
            idx         = torch.full((bsize,), skill_step, dtype=torch.long, device=device)
            prior_slice = self._get_prior_slice(prior_cache, idx)
            prior_slice = prior_slice * float(self.config.vae_prior_scale)
            x_t = torch.zeros(shape, device=device, dtype=prior_slice.dtype)
            x_t[:, :, :prior_slice.shape[-1]] = prior_slice
        else:
            x_t = self.sample_noise(shape, device) if noise is None else noise

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks, lang_to_action_masks=lang_to_action_masks
        )
        _, past_key_values = self.contextualize_prefix(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            use_cache=True,
        )

        dt = -1.0 / num_steps
        for step in range(num_steps):
            t_val    = 1.0 + step * dt
            t_tensor = torch.tensor(t_val, dtype=torch.float32, device=device).expand(bsize)
            v_t = self.denoise_step(prefix_pad_masks, past_key_values, x_t, t_tensor, z=z)
            x_t = x_t + dt * v_t

        # x_t converges to the full action (no residual addition needed)
        return x_t, prior_cache


# ── Policy wrapper (stage 3 only) ─────────────────────────────────────────────

class SkillVLAPolicy(PI05Policy):
    """Stage 3: joint flow-matching + skill-predictor training.

    All parameters trainable. Skill predictor MSE gradient flows into the VLM
    (detach_sp_prefix=False). Teacher-forced skill latents from the dataset are
    used during training; at inference the skill predictor auto-regressively
    predicts z from the previous skill and the current VLM prefix.
    """

    config_class = SkillVLAConfig
    name         = "skill_vla"

    def __init__(self, config: SkillVLAConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = SkillVLAPytorch(config, rtc_processor=self.rtc_processor)
        self._set_action_stats_for_prior(kwargs.get("dataset_stats"))
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)
        self.reset()

    def _set_action_stats_for_prior(self, dataset_stats=None) -> None:
        action_stats = (dataset_stats or {}).get(ACTION, {}) if dataset_stats else {}
        q01 = action_stats.get("q01")
        q99 = action_stats.get("q99")

        if (q01 is None or q99 is None) and self.config.pretrained_path is not None:
            stats_paths = sorted(Path(self.config.pretrained_path).glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
            if stats_paths:
                stats = load_file(str(stats_paths[0]))
                q01 = stats.get("action.q01")
                q99 = stats.get("action.q99")

        self.model.set_action_quantile_stats(q01, q99)
        if self.model._action_q01.numel() == 0:
            log.warning("Action quantile stats not found; VAE prior normalization will fail if a prior is used.")

    # ── Training ──────────────────────────────────────────────────────────────

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean"):
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks  = batch[OBS_LANGUAGE_ATTENTION_MASK]

        z                 = batch.get("skill_latent")
        z_prev            = batch.get("skill_latent_prev")
        f_b               = batch.get("skill_boundary")
        actions           = self.prepare_action(batch)
        skill_start_state = batch.get("skill_start_state")
        skill_frame_index = batch.get("skill_frame_index")
        skill_progress    = batch.get("skill_progress")
        lang_to_action_masks = batch.get(OBS_LANG_TO_ACTION_ATTENTION_MASK)
        train_step = int(self.model._predicted_latent_train_step.item())

        # SP gradient flows into VLM (detach_sp_prefix=False)
        flow_losses, sp_loss = self.model.forward(
            images, img_masks, tokens, masks, actions,
            z=z, z_prev=z_prev, f_b=f_b,
            skill_start_state=skill_start_state,
            skill_frame_index=skill_frame_index,
            skill_progress=skill_progress,
            lang_to_action_masks=lang_to_action_masks,
            detach_sp_prefix=False,
            train_step=train_step,
        )
        if self.training:
            self.model._predicted_latent_train_step.add_(1)

        action_dim  = self.config.output_features[ACTION].shape[0]
        flow_losses = flow_losses[:, :, :action_dim]
        flow_loss   = flow_losses.mean()
        total_loss  = flow_loss + self.config.skill_predictor_loss_weight * sp_loss

        n_boundaries = int(f_b.bool().sum().item()) if f_b is not None else 0
        loss_dict = {
            "loss":                        total_loss.item(),
            "loss_flow":                   flow_loss.item(),
            "loss_skill_predictor":        sp_loss.item(),
            "n_skill_boundaries_in_batch": n_boundaries,
            "predicted_latent_prob":        self.model._last_predicted_latent_prob,
            "predicted_latent_fraction":    self.model._last_predicted_latent_fraction,
            "loss_per_dim":                flow_losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }

        if reduction == "none":
            return flow_losses.mean(dim=(1, 2)), loss_dict
        return total_loss, loss_dict

    # ── Inference state ───────────────────────────────────────────────────────

    def reset(self):
        super().reset()
        self._current_z        : Tensor | None = None
        self._prior_cache      : Tensor | None = None
        self._skill_step       : int           = 0
        self._trigger_new_skill: bool          = False
        self._action_queue     : deque         = deque(maxlen=self.config.n_action_steps)
        self._skill_trace      : list[dict]    = []
        self._active_skill_trace_indices: list[int | None] = []
        self._episode_timestep : int           = 0

    def _update_skill(
        self,
        skill_predictor_prefix: Tensor,
        prefix_pad_masks: Tensor,
        current_state: Tensor,
        skill_progress: Tensor,
    ) -> None:
        """Skill predictor → update z; VAE decoder → cache full prior.

        Called at skill boundaries. Passes the current skill progress and
        current observation state.
        """
        b      = skill_predictor_prefix.shape[0]
        device = skill_predictor_prefix.device
        z_prev = self._current_z if self._current_z is not None else \
                 torch.zeros(b, self.config.skill_latent_dim, device=device)
        self._current_z   = self.model.skill_predictor(
            z_prev.float(),
            skill_predictor_prefix,
            prefix_pad_masks,
            skill_progress.float(),
            current_state.float(),
        )
        self._prior_cache = self.model._compute_full_prior(self._current_z, current_state)
        self._record_decoded_skill_actions()
        self._skill_step  = 0
        self._trigger_new_skill = False

    def _record_decoded_skill_actions(self) -> None:
        raw_actions = self.model._last_prior_raw_actions
        normalized_actions = self.model._last_prior_normalized_actions
        lengths = self.model._last_prior_lengths
        self._active_skill_trace_indices = []
        if raw_actions is None or normalized_actions is None:
            return

        batch_size = raw_actions.shape[0]
        for batch_index in range(batch_size):
            length = lengths[batch_index] if batch_index < len(lengths) else raw_actions.shape[1]
            trace_index = len(self._skill_trace)
            self._skill_trace.append(
                {
                    "skill_index": trace_index,
                    "batch_index": batch_index,
                    "length": int(length),
                    "episode_timestep": self._episode_timestep,
                    "raw_actions": raw_actions[batch_index, :length].numpy().copy(),
                    "normalized_actions": normalized_actions[batch_index, :length].numpy().copy(),
                    "expert_raw_actions": [],
                    "expert_normalized_actions": [],
                    "z": self._current_z[batch_index].detach().cpu().numpy().copy(),
                }
            )
            self._active_skill_trace_indices.append(trace_index)

    def _denormalize_action_chunk(self, actions: Tensor) -> Tensor | None:
        if self.model._action_q01.numel() == 0 or self.model._action_q99.numel() == 0:
            return None

        q01 = self.model._action_q01.to(device=actions.device, dtype=actions.dtype)
        q99 = self.model._action_q99.to(device=actions.device, dtype=actions.dtype)
        action_dim = actions.shape[-1]
        q01 = q01[:action_dim]
        q99 = q99[:action_dim]
        return (actions + 1.0) / 2.0 * (q99 - q01) + q01

    def _record_expert_action_chunk(self, actions: Tensor, skill_step: int) -> None:
        """Append action expert outputs to the active skill trace for overlay plots."""
        if not self._active_skill_trace_indices:
            return

        raw_actions = self._denormalize_action_chunk(actions)
        actions_cpu = actions.detach().cpu()
        raw_actions_cpu = raw_actions.detach().cpu() if raw_actions is not None else None

        batch_size = actions.shape[0]
        for batch_index in range(batch_size):
            if batch_index >= len(self._active_skill_trace_indices):
                continue
            trace_index = self._active_skill_trace_indices[batch_index]
            if trace_index is None or trace_index >= len(self._skill_trace):
                continue

            record = self._skill_trace[trace_index]
            length = int(record.get("length", actions.shape[1]))
            start = max(0, int(skill_step))
            if start >= length:
                continue
            end = min(start + actions.shape[1], length)
            n = end - start
            record["expert_normalized_actions"].append(
                {
                    "start": start,
                    "actions": actions_cpu[batch_index, :n].numpy().copy(),
                }
            )
            if raw_actions_cpu is not None:
                record["expert_raw_actions"].append(
                    {
                        "start": start,
                        "actions": raw_actions_cpu[batch_index, :n].numpy().copy(),
                    }
                )

    def get_skill_trace(self) -> list[dict]:
        return list(self._skill_trace)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks  = batch[OBS_LANGUAGE_ATTENTION_MASK]
        lang_to_action_masks = batch.get(OBS_LANG_TO_ACTION_ATTENTION_MASK)

        if self._current_z is None or self._trigger_new_skill:
            state = batch.get("skill_start_state")
            if state is None:
                state = batch.get("observation.state")
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
                images, img_masks, tokens, masks, lang_to_action_masks=lang_to_action_masks
            )
            skill_predictor_prefix, _ = self.model.get_skill_predictor_prefix(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                use_cache=False,
            )
            skill_progress = torch.full(
                (tokens.shape[0],),
                0,
                device=tokens.device,
                dtype=torch.float32,
            )
            self._update_skill(skill_predictor_prefix, prefix_pad_masks, state, skill_progress)

        if len(self._action_queue) == 0:
            actions, _ = self.model.sample_actions(
                images, img_masks, tokens, masks,
                z           = self._current_z,
                prior_cache = self._prior_cache,
                skill_step  = self._skill_step,
                lang_to_action_masks=lang_to_action_masks,
            )
            action_dim = self.config.output_features[ACTION].shape[0]
            actions = actions[:, : self.config.n_action_steps, :action_dim]
            self._record_expert_action_chunk(actions, self._skill_step)
            self._action_queue.extend(actions.transpose(0, 1))

            self._skill_step += self.config.n_action_steps
            if self._prior_cache is not None:
                if self._skill_step >= self._prior_cache.shape[1]:
                    self._trigger_new_skill = True

        action = self._action_queue.popleft()
        self._episode_timestep += 1
        return action
