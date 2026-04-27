"""SkillVLA policy — extends PI05 with a skill predictor and frozen VAE decoder prior.

Architecture changes vs PI05:
  1. embed_suffix : z (skill latent) projected and added to action embeddings.
  2. forward      : residual target = actions - prior_slice; skill predictor loss at f_b==1.
  3. denoise_step : passes z into embed_suffix during inference denoising.
  4. sample_actions: accepts z; caches prior and slices per step.
  5. select_action : manages z / prior / step-counter across timesteps.

Everything else (VLM, action expert, flow matching, processors) is inherited from PI05.

Expected batch keys beyond standard PI05:
  - "skill_latent"        : (B, skill_latent_dim)   current skill z
  - "skill_latent_prev"   : (B, skill_latent_dim)   previous skill z'
  - "skill_boundary"      : (B,)                    1 at first frame of new skill
  - "skill_start_state"   : (B, state_dim)          proprioceptive state at skill start
  - "skill_frame_index"   : (B,)                    step index within current skill (0-based)
"""

from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.pi05.modeling_pi05 import (
    PI05Policy,
    PI05Pytorch,
    get_gemma_config,
    make_att_2d_masks,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from .configuration_skillVLA import SkillVLAConfig
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
            hidden_dim        = config.skill_predictor_hidden_dim,
        )

        self.vae_decoder = None
        if config.vae_decoder_path:
            self.load_vae_decoder(config.vae_decoder_path)

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

        return torch.from_numpy(padded).to(device)

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
            t   = int(skill_frame_index[b].item())
            end = min(t + chunk_size, skill_len)
            n   = end - t
            result[b, :n] = prior[b, t:end]
            if n < chunk_size:
                result[b, n:] = prior[b, end - 1]  # repeat last action
        return result

    # ── embed_prefix override: record lang token count ───────────────────────

    def embed_prefix(self, images, img_masks, tokens, masks):
        embs, pad_masks, att_masks = super().embed_prefix(images, img_masks, tokens, masks)
        self._n_lang_tokens = tokens.shape[1]
        return embs, pad_masks, att_masks

    def _block_lang_attention(self, att_2d_masks: Tensor, n_prefix: int) -> Tensor:
        """Zero out action→language columns in att_2d_masks (in-place)."""
        n_lang = getattr(self, "_n_lang_tokens", 0)
        if n_lang > 0:
            att_2d_masks = att_2d_masks.clone()
            att_2d_masks[:, n_prefix:, n_prefix - n_lang : n_prefix] = False
        return att_2d_masks

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
        noise            : Tensor | None = None,
        time             : Tensor | None = None,
        detach_sp_prefix : bool          = True,   # False → sp_loss gradient flows into VLM
    ) -> Tuple[Tensor, Tensor]:
        """Returns (flow_loss [B,chunk,max_dim], skill_predictor_loss scalar)."""

        # ── Residual target ───────────────────────────────────────────────────
        if z is not None and skill_start_state is not None and skill_frame_index is not None:
            full_prior = self._compute_full_prior(z, skill_start_state)
            if full_prior is not None:
                prior_slice = self._get_prior_slice(full_prior, skill_frame_index)
                # Pad prior to max_action_dim
                pad = torch.zeros_like(actions)
                action_dim = prior_slice.shape[-1]
                pad[:, :, :action_dim] = prior_slice.to(actions.dtype)
                target = actions - pad
            else:
                target = actions
        else:
            target = actions

        # ── Flow matching ─────────────────────────────────────────────────────
        if noise is None:
            noise = self.sample_noise(target.shape, target.device)
        if time is None:
            time = self.sample_time(target.shape[0], target.device)

        time_exp = time[:, None, None]
        x_t = time_exp * noise + (1 - time_exp) * target
        u_t = noise - target

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            x_t, time, z=z
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
            att_2d_masks = self._block_lang_attention(att_2d_masks, prefix_pad_masks.shape[1])
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

        # ── Skill predictor loss (f_b == 1 frames only) ───────────────────────
        sp_loss = torch.tensor(0.0, device=actions.device)
        if z is not None and z_prev is not None and f_b is not None:
            boundary = f_b.bool()
            if boundary.any():
                prefix_pooled = prefix_embs.float().mean(dim=1)
                if detach_sp_prefix:
                    prefix_pooled = prefix_pooled.detach()
                z_pred = self.skill_predictor(
                    z_prev[boundary].to(prefix_pooled.dtype),
                    prefix_pooled[boundary],
                )
                sp_loss = F.mse_loss(z_pred, z[boundary].to(z_pred.dtype))

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
                prefix_pad_2d = prefix_pad_2d.clone()
                prefix_pad_2d[:, :, prefix_len - n_lang : prefix_len] = False
        suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d   = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)

        prefix_offsets  = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids    = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_4d  = self._prepare_attention_masks_4d(full_att_2d)

        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        past_key_values = copy.deepcopy(past_key_values)
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask  = full_att_2d_4d,
            position_ids    = position_ids,
            past_key_values = past_key_values,
            inputs_embeds   = [None, suffix_embs],
            use_cache       = True,
            adarms_cond     = [None, adarms_cond],
        )
        suffix_out = outputs_embeds[:, -self.config.chunk_size:].to(torch.float32)
        return self.action_out_proj(suffix_out)

    # ── Inference: sample_actions override (pass z, add prior) ───────────────

    @torch.no_grad()
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        z                : Tensor | None = None,   # (B, skill_latent_dim)
        skill_start_state: Tensor | None = None,   # (B, state_dim)  — at skill boundary
        prior_cache      : Tensor | None = None,   # (B, skill_len, action_dim) precomputed
        skill_step       : int = 0,                # current step within skill
        noise            : Tensor | None = None,
        num_steps        : int | None = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor | None]:
        """Returns (residual_actions, full_prior) where full_prior is computed if not cached."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize  = tokens.shape[0]
        device = tokens.device

        # Compute / reuse full prior
        if prior_cache is None and z is not None and skill_start_state is not None:
            prior_cache = self._compute_full_prior(z, skill_start_state)

        if noise is None:
            noise = self.sample_noise(
                (bsize, self.config.chunk_size, self.config.max_action_dim), device
            )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks
        )
        prefix_att_2d     = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_4d  = self._prepare_attention_masks_4d(prefix_att_2d)

        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask  = prefix_att_2d_4d,
            position_ids    = prefix_position_ids,
            past_key_values = None,
            inputs_embeds   = [prefix_embs, None],
            use_cache       = True,
        )

        dt  = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            t_val   = 1.0 + step * dt
            t_tensor = torch.tensor(t_val, dtype=torch.float32, device=device).expand(bsize)
            v_t = self.denoise_step(prefix_pad_masks, past_key_values, x_t, t_tensor, z=z)
            x_t = x_t + dt * v_t

        # x_t is the residual action ã; add prior slice → full action
        if prior_cache is not None:
            idx   = torch.full((bsize,), skill_step, dtype=torch.long, device=device)
            prior_slice = self._get_prior_slice(prior_cache, idx)
            pad         = torch.zeros_like(x_t)
            adim        = prior_slice.shape[-1]
            pad[:, :, :adim] = prior_slice.to(x_t.dtype)
            x_t = x_t + pad

        return x_t, prior_cache


# ── Policy wrapper ────────────────────────────────────────────────────────────

class SkillVLAPolicy(PI05Policy):
    config_class = SkillVLAConfig
    name         = "skill_vla"

    def __init__(self, config: SkillVLAConfig, **kwargs):
        super().__init__(config, **kwargs)
        # Replace PI05Pytorch with SkillVLAPytorch
        self.model = SkillVLAPytorch(config, rtc_processor=self.rtc_processor)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self._apply_stage_freezing()
        self.model.to(config.device)
        self.reset()

    def _apply_stage_freezing(self) -> None:
        """Freeze parameters according to training_stage."""
        stage = self.config.training_stage
        if stage == 1:
            # VLM + action expert train freely; skill predictor is idle in stage 1
            for p in self.model.skill_predictor.parameters():
                p.requires_grad_(False)
            log.info("Stage 1: skill predictor frozen")
        elif stage == 2:
            # Freeze everything, then unfreeze only skill predictor
            for p in self.model.parameters():
                p.requires_grad_(False)
            for p in self.model.skill_predictor.parameters():
                p.requires_grad_(True)
            log.info("Stage 2: VLM + action expert frozen, skill predictor trainable")
        elif stage == 3:
            # All params trainable; sp_loss gradient flows into VLM (path 3)
            log.info("Stage 3: joint training — teacher forcing + path3 active")
        else:
            raise ValueError(f"training_stage must be 1, 2, or 3, got {stage}")

    # ── Training ─────────────────────────────────────────────────────────────

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean"):
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks  = batch[OBS_LANGUAGE_ATTENTION_MASK]

        z      = batch.get("skill_latent")
        z_prev = batch.get("skill_latent_prev")
        f_b    = batch.get("skill_boundary")

        # ── Stage 2: skill predictor only (lightweight — skip flow matching) ──
        if self.config.training_stage == 2:
            sp_loss = self._sp_loss_only(images, img_masks, tokens, masks, z, z_prev, f_b)
            total_loss = self.config.skill_predictor_loss_weight * sp_loss
            loss_dict = {
                "loss":                 total_loss.item(),
                "loss_flow":            0.0,
                "loss_skill_predictor": sp_loss.item(),
            }
            return total_loss, loss_dict

        # ── Stage 1 / 3: flow matching ────────────────────────────────────────
        actions           = self.prepare_action(batch)
        skill_start_state = batch.get("skill_start_state")
        skill_frame_index = batch.get("skill_frame_index")

        # Stage 3: pass f_b so sp_loss is computed; gradient flows into VLM (no detach)
        # Stage 1: f_b=None skips sp_loss entirely
        stage = self.config.training_stage
        flow_losses, sp_loss = self.model.forward(
            images, img_masks, tokens, masks, actions,
            z=z, z_prev=z_prev if stage == 3 else None,
            f_b=f_b if stage == 3 else None,
            skill_start_state=skill_start_state,
            skill_frame_index=skill_frame_index,
            detach_sp_prefix=(stage != 3),
        )

        action_dim  = self.config.output_features[ACTION].shape[0]
        flow_losses = flow_losses[:, :, :action_dim]
        flow_loss   = flow_losses.mean()
        total_loss  = flow_loss + self.config.skill_predictor_loss_weight * sp_loss

        n_boundaries = int(f_b.bool().sum().item()) if f_b is not None else 0
        loss_dict = {
            "loss":                       total_loss.item(),
            "loss_flow":                  flow_loss.item(),
            "loss_skill_predictor":       sp_loss.item(),
            "n_skill_boundaries_in_batch": n_boundaries,
            "loss_per_dim":               flow_losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }

        if reduction == "none":
            per_sample = flow_losses.mean(dim=(1, 2))
            return per_sample, loss_dict
        return total_loss, loss_dict

    def _sp_loss_only(
        self,
        images, img_masks, tokens, masks,
        z: Tensor | None,
        z_prev: Tensor | None,
        f_b: Tensor | None,
    ) -> Tensor:
        """Lightweight forward for stage 2: embed prefix → skill predictor → MSE loss."""
        device = tokens.device
        zero = torch.zeros(1, device=device).squeeze()

        if z is None or z_prev is None or f_b is None:
            return zero

        boundary = f_b.bool()
        if not boundary.any():
            return zero

        prefix_embs, _, _ = self.model.embed_prefix(images, img_masks, tokens, masks)
        prefix_pooled = prefix_embs.float().mean(dim=1)  # VLM frozen → no need to detach
        z_pred = self.model.skill_predictor(
            z_prev[boundary].to(prefix_pooled.dtype),
            prefix_pooled[boundary],
        )
        return F.mse_loss(z_pred, z[boundary].to(z_pred.dtype))

    # ── Inference state ───────────────────────────────────────────────────────

    def reset(self):
        super().reset()
        self._current_z      : Tensor | None = None
        self._prior_cache    : Tensor | None = None
        self._skill_step     : int           = 0
        self._trigger_new_skill: bool        = False

    def _update_skill(self, prefix_embs: Tensor, skill_start_state: Tensor) -> None:
        """Run skill predictor → update z; run VAE decoder → cache full prior."""
        b      = prefix_embs.shape[0]
        device = prefix_embs.device
        z_prev = self._current_z if self._current_z is not None else \
                 torch.zeros(b, self.config.skill_latent_dim, device=device)
        prefix_pooled     = prefix_embs.float().mean(dim=1)
        self._current_z   = self.model.skill_predictor(z_prev, prefix_pooled)
        self._prior_cache = self.model._compute_full_prior(self._current_z, skill_start_state)
        self._skill_step  = 0
        self._trigger_new_skill = False

    # ── Inference forward ─────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks  = batch[OBS_LANGUAGE_ATTENTION_MASK]

        # New skill: first call or prior end was reached last step
        if self._current_z is None or self._trigger_new_skill:
            state = batch.get("observation.state")
            prefix_embs, _, _ = self.model.embed_prefix(images, img_masks, tokens, masks)
            self._update_skill(prefix_embs, state)

        actions, _ = self.model.sample_actions(
            images, img_masks, tokens, masks,
            z           = self._current_z,
            prior_cache = self._prior_cache,
            skill_step  = self._skill_step,
        )

        self._skill_step += 1

        # Check if current chunk reaches the end of the prior → trigger next step
        if self._prior_cache is not None:
            prior_length = self._prior_cache.shape[1]
            if self._skill_step + self.config.chunk_size >= prior_length:
                self._trigger_new_skill = True

        action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :action_dim]
