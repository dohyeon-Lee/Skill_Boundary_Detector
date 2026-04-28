"""SkillVLA Decouple policy — stages 1 and 2 of SkillVLA training.

Stage 1: Train VLM + action expert with teacher-forced skill latents from the dataset.
         Skill predictor is frozen and sp_loss is skipped entirely.
         Residual target = actions - VAE-decoder prior.

Stage 2: Freeze VLM + action expert; train skill predictor only (no flow matching).
         Lightweight forward: embed prefix → predict z → MSE loss at boundary frames.

The core model (SkillVLAPytorch) is imported from skillVLA to avoid duplication.
After completing stage 2, switch to skillVLA (stage 3) for joint fine-tuning.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.skillVLA.modeling_skillVLA import SkillVLAPytorch
from lerobot.policies.skillVLA.processor_skillVLA import OBS_LANG_TO_ACTION_ATTENTION_MASK
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from .configuration_skillVLA_decouple import SkillVLADecoupleConfig

log = logging.getLogger(__name__)


class SkillVLADecouplePolicy(PI05Policy):
    config_class = SkillVLADecoupleConfig
    name         = "skill_vla_decouple"

    def __init__(self, config: SkillVLADecoupleConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = SkillVLAPytorch(config, rtc_processor=self.rtc_processor)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self._apply_stage_freezing()
        self.model.to(config.device)
        self.reset()

    # ── Parameter freezing ────────────────────────────────────────────────────

    def _apply_stage_freezing(self) -> None:
        stage = self.config.training_stage
        if stage == 1:
            for p in self.model.skill_predictor.parameters():
                p.requires_grad_(False)
            log.info("Stage 1: skill predictor frozen")
        elif stage == 2:
            for p in self.model.parameters():
                p.requires_grad_(False)
            for p in self.model.skill_predictor.parameters():
                p.requires_grad_(True)
            log.info("Stage 2: VLM + action expert frozen, skill predictor trainable")
        else:
            raise ValueError(f"SkillVLADecouplePolicy: training_stage must be 1 or 2, got {stage}")

    # ── Training ──────────────────────────────────────────────────────────────

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean"):
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks  = batch[OBS_LANGUAGE_ATTENTION_MASK]

        z      = batch.get("skill_latent")
        z_prev = batch.get("skill_latent_prev")
        f_b    = batch.get("skill_boundary")
        lang_to_action_masks = batch.get(OBS_LANG_TO_ACTION_ATTENTION_MASK)

        # ── Stage 2: skill predictor only ─────────────────────────────────────
        if self.config.training_stage == 2:
            sp_loss    = self._sp_loss_only(images, img_masks, tokens, masks, z, z_prev, f_b, lang_to_action_masks)
            total_loss = self.config.skill_predictor_loss_weight * sp_loss
            return total_loss, {
                "loss":                 total_loss.item(),
                "loss_flow":            0.0,
                "loss_skill_predictor": sp_loss.item(),
            }

        # ── Stage 1: flow matching (SP frozen, sp_loss skipped) ───────────────
        actions           = self.prepare_action(batch)
        skill_start_state = batch.get("skill_start_state")
        skill_frame_index = batch.get("skill_frame_index")

        # z_prev=None, f_b=None → sp_loss branch inside model.forward is skipped.
        # prior-start flow matching is handled inside model.forward automatically.
        flow_losses, sp_loss = self.model.forward(
            images, img_masks, tokens, masks, actions,
            z=z, z_prev=None, f_b=None,
            skill_start_state=skill_start_state,
            skill_frame_index=skill_frame_index,
            lang_to_action_masks=lang_to_action_masks,
            detach_sp_prefix=True,
        )

        action_dim  = self.config.output_features[ACTION].shape[0]
        flow_losses = flow_losses[:, :, :action_dim]
        flow_loss   = flow_losses.mean()
        total_loss  = flow_loss  # sp_loss is always 0 in stage 1

        loss_dict = {
            "loss":          total_loss.item(),
            "loss_flow":     flow_loss.item(),
            "loss_per_dim":  flow_losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }

        if reduction == "none":
            return flow_losses.mean(dim=(1, 2)), loss_dict
        return total_loss, loss_dict

    def _sp_loss_only(
        self,
        images, img_masks, tokens, masks,
        z: Tensor | None,
        z_prev: Tensor | None,
        f_b: Tensor | None,
        lang_to_action_masks: Tensor | None = None,
    ) -> Tensor:
        """Stage 2: embed prefix → skill predictor → MSE loss at boundary frames only."""
        device = tokens.device
        zero   = torch.zeros(1, device=device).squeeze()

        if z is None or z_prev is None or f_b is None:
            return zero

        boundary = f_b.bool()
        if not boundary.any():
            return zero

        prefix_embs, _, _ = self.model.embed_prefix(images, img_masks, tokens, masks, lang_to_action_masks=lang_to_action_masks)
        prefix_pooled = prefix_embs.float().mean(dim=1)  # VLM frozen → no need to detach
        z_pred = self.model.skill_predictor(
            z_prev[boundary].to(prefix_pooled.dtype),
            prefix_pooled[boundary],
        )
        return F.mse_loss(z_pred, z[boundary].to(z_pred.dtype))

    # ── Inference state ───────────────────────────────────────────────────────

    def reset(self):
        super().reset()
        self._current_z        : Tensor | None = None
        self._prior_cache      : Tensor | None = None
        self._skill_step       : int           = 0
        self._trigger_new_skill: bool          = False
        self._action_queue     : deque         = deque(maxlen=self.config.n_action_steps)

    def _update_skill(self, prefix_embs: Tensor, skill_start_state: Tensor) -> None:
        b      = prefix_embs.shape[0]
        device = prefix_embs.device
        z_prev = self._current_z if self._current_z is not None else \
                 torch.zeros(b, self.config.skill_latent_dim, device=device)
        prefix_pooled     = prefix_embs.float().mean(dim=1)
        self._current_z   = self.model.skill_predictor(z_prev, prefix_pooled)
        self._prior_cache = self.model._compute_full_prior(self._current_z, skill_start_state)
        self._skill_step  = 0
        self._trigger_new_skill = False

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks  = batch[OBS_LANGUAGE_ATTENTION_MASK]

        lang_to_action_masks = batch.get(OBS_LANG_TO_ACTION_ATTENTION_MASK)

        if self._current_z is None or self._trigger_new_skill:
            state = batch.get("observation.state")
            prefix_embs, _, _ = self.model.embed_prefix(
                images, img_masks, tokens, masks, lang_to_action_masks=lang_to_action_masks
            )
            self._update_skill(prefix_embs, state)

        if len(self._action_queue) == 0:
            actions, _ = self.model.sample_actions(
                images, img_masks, tokens, masks,
                z                    = self._current_z,
                prior_cache          = self._prior_cache,
                skill_step           = self._skill_step,
                lang_to_action_masks = lang_to_action_masks,
            )
            action_dim = self.config.output_features[ACTION].shape[0]
            actions = actions[:, : self.config.n_action_steps, :action_dim]
            self._action_queue.extend(actions.transpose(0, 1))

            self._skill_step += self.config.n_action_steps
            if self._prior_cache is not None:
                if self._skill_step >= self._prior_cache.shape[1]:
                    self._trigger_new_skill = True

        return self._action_queue.popleft()
