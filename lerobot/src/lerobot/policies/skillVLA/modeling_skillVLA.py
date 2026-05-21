"""SkillVLA policy — stage 3 joint training.

Extends PI05 with skill conditioning and an FSQ end-signal decoder.
All parameters are trainable; skill predictor gradient flows into the VLM.

Architecture changes vs PI05:
  1. embed_suffix  : z (skill latent) projected and prepended to action embeddings.
  2. forward       : flow matching + FSQ skill predictor + skill decoder losses.
  3. denoise_step  : passes z into embed_suffix during inference denoising.
  4. sample_actions: accepts z and denoises from noise.
  5. select_action : skill predictor predicts z; manages skill step.

For stages 1 & 2 (decoupled pre-training) use skillVLA_decouple instead.

Expected batch keys beyond standard PI05:
  - "skill_index"          : (B,) current skill index in skill_sequence (BOS=0)
  - "skill_sequence"       : (B, max_order+2) [BOS, skills..., EOS, PAD...]
  - "skill_length_sequence": (B, max_order+2) aligned skill lengths
  - "skill_ds" / "skill_de": (B,) distance from skill start/end
  - "skill_boundary"       : (B,) 1 at current skill end
  - "skill_max_order"      : (B,) maximum real skill count
  - "skill_max_length"     : (B,) FSQ max skill length
"""

from __future__ import annotations

import copy
import math
import logging
import sys
from collections import deque
from pathlib import Path

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
from .processor_skillVLA import OBS_LANG_TO_ACTION_ATTENTION_MASK
from .skill_predictor import SkillPredictor

log = logging.getLogger(__name__)


# ── Core model ────────────────────────────────────────────────────────────────

class SkillVLAPytorch(PI05Pytorch):
    """PI05Pytorch + skill conditioning + FSQ end-signal decoder."""

    def __init__(self, config: SkillVLAConfig, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)

        action_expert_config = get_gemma_config(config.action_expert_variant)
        paligemma_config     = get_gemma_config(config.paligemma_variant)

        fsq_levels = self._resolve_fsq_levels(config.vae_decoder_path, config.skill_fsq_levels)
        if len(set(fsq_levels)) != 1:
            raise ValueError(f"SkillVLA currently expects equal FSQ levels per dim, got {fsq_levels}.")
        config.skill_fsq_levels = fsq_levels
        config.skill_latent_dim = len(fsq_levels)
        config.skill_predictor_num_embeddings = int(math.prod(fsq_levels))
        self.fsq_level = fsq_levels[0]
        self.register_buffer("_fsq_levels", torch.tensor(fsq_levels, dtype=torch.long), persistent=False)
        strides = torch.ones(len(fsq_levels), dtype=torch.long)
        for i in range(1, len(fsq_levels)):
            strides[i] = strides[i - 1] * fsq_levels[i - 1]
        self.register_buffer("_fsq_strides", strides, persistent=False)
        self.register_buffer(
            "_fsq_half",
            torch.tensor([(level - 1) / 2.0 for level in fsq_levels], dtype=torch.float32),
            persistent=False,
        )

        skill_condition_dim = config.skill_latent_dim + config.skill_latent_dim * self.fsq_level
        self.z_to_suffix_proj = nn.Sequential(
            nn.Linear(skill_condition_dim, config.skill_predictor_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.skill_predictor_hidden_dim),
            nn.Linear(config.skill_predictor_hidden_dim, action_expert_config.width),
        )

        self.skill_predictor = SkillPredictor(
            skill_latent_dim  = config.skill_latent_dim,
            prefix_hidden_dim = paligemma_config.width,
            fsq_dim           = config.skill_latent_dim,
            fsq_level         = self.fsq_level,
            hidden_dim        = config.skill_predictor_hidden_dim,
            num_heads         = config.skill_predictor_num_heads,
            num_layers        = config.skill_predictor_num_layers,
            num_query_tokens  = config.skill_predictor_num_query_tokens,
            dropout           = config.skill_predictor_dropout,
        )

        self.vae_decoder = None
        self.special_skill_embeddings = nn.Embedding(3, config.skill_latent_dim)
        self.register_buffer("_predicted_latent_train_step", torch.zeros((), dtype=torch.long), persistent=False)
        self._last_predicted_latent_prob = 0.0
        self._last_predicted_latent_fraction = 0.0
        self._last_sp_loss_components: dict[str, float] = {}
        self._last_skill_decoder_components: dict[str, float] = {}
        if config.vae_decoder_path:
            self.load_vae_decoder(config.vae_decoder_path)

    # ── FSQ decoder ───────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_fsq_levels(path: str | None, fallback: list[int]) -> list[int]:
        if not path:
            return list(fallback)
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            cfg = ckpt.get("cfg")
            levels = getattr(cfg, "fsq_levels", None)
            if levels is not None:
                return [int(x) for x in levels]
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not infer FSQ levels from %s: %s", path, exc)
        return list(fallback)

    def load_vae_decoder(self, path: str) -> None:
        sys.path.insert(
            0, str(Path(__file__).resolve().parents[4] / "examples" / "libero")
        )
        import dataclasses  # noqa: PLC0415
        from FSQ import SplineFSQAE  # noqa: PLC0415

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg  = ckpt["cfg"]
        cfg_dict = dataclasses.asdict(cfg)
        image_model_name = self.config.skill_decoder_image_model_name
        if image_model_name:
            cfg_dict["image_model_name"] = image_model_name
        keys = {"action_dim", "state_dim", "n_control", "spline_degree",
                "hidden_dim", "fsq_levels", "num_layers", "dropout",
                "max_length", "action_min", "action_max", "delta_min", "delta_max",
                "feat_dim", "n_tokens", "decoder_image_mode", "image_encoder_layers",
                "image_encoder_heads", "image_model_name", "image_size", "patch_grid",
                "n_patch_raw", "decoder_output_mode", "chunk_size"}
        vae = SplineFSQAE(**{k: v for k, v in cfg_dict.items() if k in keys})
        vae.load_state_dict(ckpt["model_state"])
        for p in vae.parameters():
            p.requires_grad_(False)
        if not self.config.freeze_vae_decoder:
            for name in ("dec_image_encoder", "dec_mlp", "delta_head", "end_head"):
                module = getattr(vae, name, None)
                if module is not None:
                    for p in module.parameters():
                        p.requires_grad_(True)
        self.vae_decoder    = vae
        log.info(
            "Loaded FSQ decoder from %s (levels=%s, freeze=%s, image_model=%s)",
            path,
            self.config.skill_fsq_levels,
            self.config.freeze_vae_decoder,
            getattr(vae, "image_model_name", None),
        )

    def _token_to_z(self, tokens: Tensor) -> Tensor:
        """FSQ scalar index (B,) or (B,1) → quantized z vector (B, fsq_dim)."""
        idx = tokens.view(-1).long()
        strides = self._fsq_strides.to(device=idx.device)
        half = self._fsq_half.to(device=idx.device, dtype=torch.float32)
        levels = self._fsq_levels.to(device=idx.device)
        level_ids = torch.div(idx[:, None], strides[None, :], rounding_mode="floor") % levels[None, :]
        return level_ids.to(torch.float32) - half[None, :]

    def _token_to_fsq_targets(self, tokens: Tensor) -> Tensor:
        """FSQ scalar index (B,) → per-dim class ids (B, fsq_dim)."""
        idx = tokens.view(-1).long()
        strides = self._fsq_strides.to(device=idx.device)
        levels = self._fsq_levels.to(device=idx.device)
        return (torch.div(idx[:, None], strides[None, :], rounding_mode="floor") % levels[None, :]).long()

    def _fsq_logits_to_z(self, logits: Tensor) -> Tensor:
        """Dim-wise logits (B,D,L) → hard FSQ z_q vector (B,D)."""
        level_ids = logits.argmax(dim=-1)
        return level_ids.to(torch.float32) - self._fsq_half.to(device=logits.device)[None, :]

    def _fsq_logits_to_token(self, logits: Tensor) -> Tensor:
        """Dim-wise logits (B,D,L) → scalar FSQ index (B,)."""
        level_ids = logits.argmax(dim=-1).long()
        strides = self._fsq_strides.to(device=logits.device)
        return (level_ids * strides[None, :]).sum(dim=-1)

    def _z_to_fsq_level_ids(self, z: Tensor) -> Tensor:
        half = self._fsq_half.to(device=z.device, dtype=z.dtype)
        levels = self._fsq_levels.to(device=z.device)
        level_ids = torch.round(z + half[None, :]).long()
        return level_ids.clamp_min(0).minimum((levels - 1)[None, :])

    def _skill_action_condition(self, z: Tensor) -> Tensor:
        """FSQ z_q → concat[z_q, dim-wise one-hot] for action-expert conditioning."""
        z = z.float()
        level_ids = self._z_to_fsq_level_ids(z)
        one_hot = F.one_hot(level_ids, num_classes=self.fsq_level).to(dtype=z.dtype)
        return torch.cat([z, one_hot.flatten(start_dim=1)], dim=-1)

    def _skill_token_embedding(self, tokens: Tensor) -> Tensor:
        """Map FSQ tokens and BOS/EOS/PAD special tokens to predictor inputs."""
        idx = tokens.view(-1).long()
        out = torch.zeros(idx.shape[0], self.config.skill_latent_dim, device=idx.device, dtype=torch.float32)
        real_mask = (idx >= 0) & (idx < self.config.skill_predictor_num_embeddings)
        if real_mask.any():
            out[real_mask] = self._token_to_z(idx[real_mask]).to(out.dtype)
        special_mask = ~real_mask
        if special_mask.any():
            special_idx = (idx[special_mask] - self.config.skill_predictor_num_embeddings).clamp(0, 2)
            out[special_mask] = self.special_skill_embeddings(special_idx).to(out.dtype)
        return out

    def _safe_real_tokens(self, tokens: Tensor, fallback: Tensor) -> Tensor:
        """Use fallback when scheduled sampling predicts a special token."""
        tokens = tokens.view(-1).long()
        fallback = fallback.view(-1).long()
        valid = (tokens >= 0) & (tokens < self.config.skill_predictor_num_embeddings)
        return torch.where(valid, tokens, fallback)

    def _skill_predictor_loss(self, z_pred: Tensor, token_target: Tensor) -> Tensor:
        """Dim-wise CE over FSQ levels. z_pred: (B,D,L), token_target: scalar FSQ index."""
        token_target = token_target.view(-1).long()
        valid = (token_target >= 0) & (token_target < self.config.skill_predictor_num_embeddings)
        if not valid.any():
            loss = z_pred.float().sum() * 0.0
        else:
            target_levels = self._token_to_fsq_targets(token_target[valid]).to(z_pred.device)
            logits = z_pred[valid].float()
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_levels.reshape(-1),
            )
        self._last_sp_loss_components = {"fsq_dim_ce": float(loss.detach().cpu())}
        return loss

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
            z_cond = self._skill_action_condition(z).to(embs.dtype)
            z_token = self.z_to_suffix_proj(z_cond).unsqueeze(1)  # (B, 1, width)
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

    def _sample_tokens_for_action(
        self,
        label_tokens: Tensor,
        z_pred: Tensor | None,
        fallback_tokens: Tensor,
        train_step: int | None,
    ) -> tuple[Tensor, Tensor]:
        """Scheduled-sample real FSQ tokens for action expert and skill decoder."""
        self._last_predicted_latent_prob = 0.0
        self._last_predicted_latent_fraction = 0.0
        label_tokens = label_tokens.view(-1).long()
        fallback_tokens = fallback_tokens.view(-1).long()
        use_pred = torch.zeros(label_tokens.shape[0], dtype=torch.bool, device=label_tokens.device)
        if z_pred is None:
            return label_tokens, use_pred

        prob = self._predicted_latent_prob(train_step)
        self._last_predicted_latent_prob = prob
        if prob <= 0.0:
            return label_tokens, use_pred

        pred_tokens = self._fsq_logits_to_token(z_pred.detach())
        pred_tokens = self._safe_real_tokens(pred_tokens, fallback_tokens)
        use_pred = torch.rand(label_tokens.shape[0], device=label_tokens.device) < prob
        self._last_predicted_latent_fraction = float(use_pred.float().mean().item())
        return torch.where(use_pred, pred_tokens, label_tokens), use_pred

    def _gather_skill_sequence(self, sequence: Tensor, index: Tensor) -> Tensor:
        index = index.view(-1).long().clamp(0, sequence.shape[1] - 1)
        return sequence.long().gather(1, index[:, None]).squeeze(1)

    def _build_skill_training_targets(
        self,
        skill_index: Tensor,
        skill_sequence: Tensor,
        skill_length_sequence: Tensor,
        skill_sequence_len: Tensor,
        skill_ds: Tensor,
        skill_de: Tensor,
        skill_boundary: Tensor,
        skill_max_order: Tensor,
        skill_max_length: Tensor,
    ) -> dict[str, Tensor]:
        device = skill_sequence.device
        k = skill_index.view(-1).long()
        ds = skill_ds.view(-1).long()
        de = skill_de.view(-1).long()
        seq_len = skill_sequence_len.view(-1).long()
        last_real = (seq_len - 2).clamp_min(1)
        max_order = skill_max_order.view(-1).float().clamp_min(1.0)
        max_length = skill_max_length.view(-1).float().clamp_min(1.0)

        current = self._gather_skill_sequence(skill_sequence, k)
        prev_idx = (k - 1).clamp_min(0)
        prev = self._gather_skill_sequence(skill_sequence, prev_idx)

        predictor_input_token = prev.clone()
        predictor_input_index = prev_idx.float() / max_order
        predictor_progress = ds.float() / max_length
        predictor_target = current.clone()

        shifted_action_token = current.clone()
        shifted_decoder_progress = predictor_progress.clone()
        shifted_decoder_target = skill_boundary.view(-1).float()

        random_p = int(self.config.skill_boundary_random_p)
        if self.training and random_p > 0:
            p = torch.randint(1, random_p + 1, k.shape, device=device)
            can_early = (ds != 0) & (de != 0) & (de <= p) & (k < last_real)
            can_late = (ds != 0) & (de != 0) & (ds <= p) & (k > 1)
            both = can_early & can_late
            choose_early = can_early.clone()
            if both.any():
                coin = torch.rand(k.shape, device=device) < 0.5
                choose_early = torch.where(both, coin, choose_early)
            choose_late = can_late & ~choose_early

            if choose_early.any():
                next_idx = (k + 1).clamp(max=skill_sequence.shape[1] - 1)
                predictor_input_token = torch.where(choose_early, current, predictor_input_token)
                predictor_input_index = torch.where(choose_early, k.float() / max_order, predictor_input_index)
                early_progress = (p.float() - de.float()).clamp_min(0.0) / max_length
                predictor_progress = torch.where(choose_early, early_progress, predictor_progress)
                next_token = self._gather_skill_sequence(skill_sequence, next_idx)
                predictor_target = torch.where(choose_early, next_token, predictor_target)
                shifted_action_token = torch.where(choose_early, next_token, shifted_action_token)
                shifted_decoder_progress = torch.where(choose_early, early_progress, shifted_decoder_progress)
                shifted_decoder_target = torch.where(choose_early, torch.zeros_like(shifted_decoder_target), shifted_decoder_target)

            if choose_late.any():
                prev_prev_idx = (k - 2).clamp_min(0)
                prev_skill_idx = (k - 1).clamp_min(0)
                prev_prev_token = self._gather_skill_sequence(skill_sequence, prev_prev_idx)
                prev_skill_token = self._gather_skill_sequence(skill_sequence, prev_skill_idx)
                prev_skill_len = self._gather_skill_sequence(skill_length_sequence, prev_skill_idx).float()
                late_progress = (prev_skill_len + ds.float()) / max_length
                predictor_input_token = torch.where(choose_late, prev_prev_token, predictor_input_token)
                predictor_input_index = torch.where(choose_late, prev_prev_idx.float() / max_order, predictor_input_index)
                predictor_progress = torch.where(choose_late, late_progress, predictor_progress)
                predictor_target = torch.where(choose_late, prev_skill_token, predictor_target)
                shifted_action_token = torch.where(choose_late, prev_skill_token, shifted_action_token)
                shifted_decoder_progress = torch.where(choose_late, late_progress, shifted_decoder_progress)
                random_end = (torch.rand(k.shape, device=device) < 0.5).float()
                shifted_decoder_target = torch.where(choose_late, random_end, shifted_decoder_target)

        return {
            "predictor_input_token": predictor_input_token,
            "predictor_input_index": predictor_input_index,
            "predictor_progress": predictor_progress,
            "predictor_target": predictor_target,
            "action_label_token": current,
            "shifted_action_token": shifted_action_token,
            "decoder_label_progress": ds.float() / max_length,
            "shifted_decoder_progress": shifted_decoder_progress,
            "decoder_label_target": skill_boundary.view(-1).float(),
            "shifted_decoder_target": shifted_decoder_target,
        }

    def _prepare_skill_decoder_state(self, states: Tensor, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        states = states.to(device=device, dtype=dtype)
        if states.ndim == 2:
            states = states.unsqueeze(1)

        expected_dim = int(getattr(self.vae_decoder, "state_dim", states.shape[-1]))
        raw_dim = int(states.shape[-1])
        indices = self.config.skill_decoder_state_indices
        if indices is not None:
            idx = torch.as_tensor(indices, dtype=torch.long, device=states.device)
            states = states.index_select(-1, idx)
        elif raw_dim > expected_dim:
            states = states[..., :expected_dim]

        if states.shape[-1] != expected_dim:
            raise ValueError(
                f"FSQ decoder expects state_dim={expected_dim}, "
                f"but got skill_decoder_state dim={raw_dim}. "
                "Set --policy.skill_decoder_state_indices to select the raw state dims used by FSQ."
            )
        self._last_skill_decoder_components["state_raw_dim"] = float(raw_dim)
        self._last_skill_decoder_components["state_used_dim"] = float(expected_dim)
        return states

    def _prepare_skill_decoder_tokens(
        self,
        images: Tensor | None,
        *,
        batch_size: int,
        steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        """Return FSQ decoder image input. It may be precomputed DINO tokens or raw RGB images."""
        if images is None:
            return None
        image_input = images.to(device=device, dtype=dtype)
        if image_input.shape[0] != batch_size:
            raise ValueError(f"FSQ decoder image batch mismatch: got {image_input.shape[0]}, expected {batch_size}.")
        n_tokens = int(getattr(self.vae_decoder, "n_tokens", 0))
        feat_dim = int(getattr(self.vae_decoder, "feat_dim", 0))
        is_token_frame = (
            image_input.ndim == 3
            and n_tokens > 0
            and feat_dim > 0
            and image_input.shape[-2] == n_tokens
            and image_input.shape[-1] == feat_dim
        )
        is_token_sequence = (
            image_input.ndim == 4
            and n_tokens > 0
            and feat_dim > 0
            and image_input.shape[-2] == n_tokens
            and image_input.shape[-1] == feat_dim
        )
        if is_token_frame:
            image_input = image_input.unsqueeze(1)
        elif is_token_sequence and image_input.shape[1] != steps:
            if image_input.shape[1] == 1:
                image_input = image_input.expand(-1, steps, -1, -1)
            else:
                raise ValueError(
                    f"FSQ decoder image time mismatch: got T={image_input.shape[1]}, expected {steps}."
                )
        return image_input

    def _zero_patch_flags(self, batch_size: int, steps: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        n_patches = int(getattr(self.vae_decoder, "n_patches", 64))
        return torch.zeros(batch_size, steps, n_patches, 2, device=device, dtype=dtype)

    def _skill_decoder_end_loss(
        self,
        tokens: Tensor,
        states: Tensor | None,
        images: Tensor | None,
        progress: Tensor,
        target: Tensor,
    ) -> Tensor:
        self._last_skill_decoder_components = {
            "skipped": 0.0,
            "has_state": float(states is not None),
            "has_image": float(images is not None),
        }
        if self.vae_decoder is None or states is None:
            self._last_skill_decoder_components["skipped"] = 1.0
            return torch.tensor(0.0, device=target.device)
        vae = self.vae_decoder
        vae.train(self.training and not self.config.freeze_vae_decoder)
        z = self._token_to_z(tokens).to(device=target.device)
        states = self._prepare_skill_decoder_state(states, device=target.device, dtype=z.dtype)
        image_tokens = self._prepare_skill_decoder_tokens(
            images,
            batch_size=z.shape[0],
            steps=states.shape[1],
            device=target.device,
            dtype=z.dtype,
        )
        if image_tokens is None:
            self._last_skill_decoder_components["skipped"] = 1.0
            return torch.tensor(0.0, device=target.device)
        patch_flags = self._zero_patch_flags(z.shape[0], states.shape[1], device=target.device, dtype=z.dtype)
        _, end_logits = vae.decode(z, states, image_tokens, patch_flags, progress.view(-1, 1))
        pred = (torch.sigmoid(end_logits.squeeze(1).float()) >= float(self.config.skill_decoder_end_threshold)).float()
        target_f = target.float()
        self._last_skill_decoder_components.update(
            {
                "target_pos_rate": float(target_f.mean().detach().cpu()),
                "pred_pos_rate": float(pred.mean().detach().cpu()),
            }
        )
        pos_w = torch.as_tensor(
            self.config.skill_decoder_end_pos_weight,
            device=target.device,
            dtype=end_logits.dtype,
        )
        return F.binary_cross_entropy_with_logits(
            end_logits.squeeze(1).float(),
            target_f,
            pos_weight=pos_w,
        )

    @torch.no_grad()
    def skill_decoder_delta_end(
        self,
        z: Tensor | None,
        state: Tensor | None,
        image: Tensor | None,
        progress: Tensor,
    ) -> tuple[Tensor, Tensor] | None:
        if self.vae_decoder is None or z is None or state is None:
            return None
        self.vae_decoder.eval()
        state = self._prepare_skill_decoder_state(state, device=z.device, dtype=z.dtype)
        image_tokens = self._prepare_skill_decoder_tokens(
            image,
            batch_size=z.shape[0],
            steps=state.shape[1],
            device=z.device,
            dtype=z.dtype,
        )
        if image_tokens is None:
            return None
        patch_flags = self._zero_patch_flags(z.shape[0], state.shape[1], device=z.device, dtype=z.dtype)
        delta, end_logits = self.vae_decoder.decode(
            z,
            state,
            image_tokens,
            patch_flags,
            progress.to(device=z.device, dtype=z.dtype).view(-1, 1),
        )
        if delta.ndim == 4:
            delta = delta[:, :, 0, :]
        return delta.squeeze(1), torch.sigmoid(end_logits.squeeze(1))

    @torch.no_grad()
    def skill_decoder_end_prob(
        self,
        z: Tensor | None,
        state: Tensor | None,
        image: Tensor | None,
        progress: Tensor,
    ) -> Tensor | None:
        out = self.skill_decoder_delta_end(z, state, image, progress)
        if out is None:
            return None
        return out[1]

    # ── Training forward ─────────────────────────────────────────────────────

    def forward(
        self,
        images,
        img_masks,
        tokens,
        masks,
        actions          : Tensor,           # (B, chunk_size, max_action_dim)
        skill_index      : Tensor | None = None,
        skill_sequence   : Tensor | None = None,
        skill_length_sequence: Tensor | None = None,
        skill_sequence_len: Tensor | None = None,
        skill_ds         : Tensor | None = None,
        skill_de         : Tensor | None = None,
        skill_boundary   : Tensor | None = None,
        skill_max_order  : Tensor | None = None,
        skill_max_length : Tensor | None = None,
        skill_decoder_state: Tensor | None = None,
        skill_decoder_image: Tensor | None = None,
        lang_to_action_masks: Tensor | None = None,
        noise            : Tensor | None = None,
        time             : Tensor | None = None,
        detach_sp_prefix : bool          = True,   # False → sp_loss gradient flows into VLM
        train_step       : int | None    = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (flow_loss [B,chunk,max_dim], skill_predictor_loss, skill_decoder_loss)."""
        for name in (
            "skill_index",
            "skill_sequence",
            "skill_length_sequence",
            "skill_sequence_len",
            "skill_ds",
            "skill_de",
            "skill_boundary",
            "skill_max_order",
            "skill_max_length",
        ):
            if locals()[name] is None:
                raise ValueError(f"{name} is required for SkillVLA training.")
        if skill_index.ndim > 1:
            skill_index = skill_index.squeeze(-1)
        if skill_sequence_len.ndim > 1:
            skill_sequence_len = skill_sequence_len.squeeze(-1)
        if skill_ds.ndim > 1:
            skill_ds = skill_ds.squeeze(-1)
        if skill_de.ndim > 1:
            skill_de = skill_de.squeeze(-1)
        if skill_boundary.ndim > 1:
            skill_boundary = skill_boundary.squeeze(-1)
        if skill_max_order.ndim > 1:
            skill_max_order = skill_max_order.squeeze(-1)
        if skill_max_length.ndim > 1:
            skill_max_length = skill_max_length.squeeze(-1)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks, lang_to_action_masks=lang_to_action_masks
        )

        skill_targets = self._build_skill_training_targets(
            skill_index,
            skill_sequence,
            skill_length_sequence,
            skill_sequence_len,
            skill_ds,
            skill_de,
            skill_boundary,
            skill_max_order,
            skill_max_length,
        )
        predictor_input_z = self._skill_token_embedding(skill_targets["predictor_input_token"]).to(actions.device)
        skill_predictor_prefix, _ = self.get_skill_predictor_prefix(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            use_cache=False,
        )
        sp_prefix = skill_predictor_prefix.detach() if detach_sp_prefix else skill_predictor_prefix
        z_pred = self.skill_predictor(
            predictor_input_z.float(),
            sp_prefix,
            prefix_pad_masks,
            skill_targets["predictor_input_index"].to(actions.device),
            skill_targets["predictor_progress"].to(actions.device),
        )
        sp_loss = self._skill_predictor_loss(z_pred, skill_targets["predictor_target"].to(actions.device))

        zero_fallback = torch.zeros_like(skill_targets["action_label_token"].to(actions.device))
        action_label_tokens = self._safe_real_tokens(
            skill_targets["action_label_token"].to(actions.device),
            zero_fallback,
        )
        shifted_action_tokens = self._safe_real_tokens(
            skill_targets["shifted_action_token"].to(actions.device),
            action_label_tokens,
        )
        sampled_tokens, use_pred_mask = self._sample_tokens_for_action(
            action_label_tokens,
            z_pred,
            shifted_action_tokens,
            train_step,
        )
        z_for_action = self._token_to_z(sampled_tokens).to(actions.device)

        decoder_progress = torch.where(
            use_pred_mask,
            skill_targets["shifted_decoder_progress"].to(actions.device),
            skill_targets["decoder_label_progress"].to(actions.device),
        )
        decoder_target = torch.where(
            use_pred_mask,
            skill_targets["shifted_decoder_target"].to(actions.device),
            skill_targets["decoder_label_target"].to(actions.device),
        )
        if self.config.freeze_vae_decoder:
            skill_decoder_loss = torch.tensor(0.0, device=actions.device)
        else:
            skill_decoder_loss = self._skill_decoder_end_loss(
                sampled_tokens,
                skill_decoder_state,
                skill_decoder_image,
                decoder_progress,
                decoder_target,
            )

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        source    = noise
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

        return flow_loss, sp_loss, skill_decoder_loss

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

    # ── Inference: sample_actions override (pass z) ─────────────────────────

    @torch.no_grad()
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        z                   : Tensor | None = None,   # (B, skill_latent_dim)
        lang_to_action_masks: Tensor | None = None,
        noise               : Tensor | None = None,
        num_steps           : int | None = None,
        **kwargs,
    ) -> Tensor:
        """Denoise from noise to an action chunk conditioned on the active skill."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize  = tokens.shape[0]
        device = tokens.device

        # Starting point: action expert now denoises from noise; FSQ decoder
        # is used only for end-signal supervision/control.
        shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
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
        return x_t


# ── Policy wrapper (stage 3 only) ─────────────────────────────────────────────

class SkillVLAPolicy(PI05Policy):
    """Stage 3: joint flow-matching + skill-predictor training.

    All parameters trainable. Skill predictor CE gradient flows into the VLM
    (detach_sp_prefix=False). Teacher-forced skill latents from the dataset are
    used during training; at inference the skill predictor auto-regressively
    predicts z from the previous skill and the current VLM prefix.
    """

    config_class = SkillVLAConfig
    name         = "skill_vla"
    _VAE_DECODER_CHECKPOINT_PREFIXES = (
        "model.vae_decoder.enc_image_encoder.",
        "model.vae_decoder.enc_traj_proj.",
        "model.vae_decoder.enc_mlp.",
        "model.vae_decoder.z_head.",
        "model.vae_decoder.dec_image_encoder.",
        "model.vae_decoder.dec_mlp.",
        "model.vae_decoder.delta_head.",
        "model.vae_decoder.end_head.",
    )

    def __init__(self, config: SkillVLAConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = SkillVLAPytorch(config, rtc_processor=self.rtc_processor)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)
        self.reset()

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        if self.model.vae_decoder is None:
            return state

        for key in list(state.keys()):
            if key.startswith("model.vae_decoder.") and not key.startswith(self._VAE_DECODER_CHECKPOINT_PREFIXES):
                del state[key]
        return state

    # ── Training ──────────────────────────────────────────────────────────────

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean"):
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks  = batch[OBS_LANGUAGE_ATTENTION_MASK]

        actions           = self.prepare_action(batch)
        lang_to_action_masks = batch.get(OBS_LANG_TO_ACTION_ATTENTION_MASK)
        train_step = int(self.model._predicted_latent_train_step.item())
        skill_decoder_state = batch.get("skill_decoder_state")
        if skill_decoder_state is None:
            raise ValueError(
                "skill_decoder_state is required for SkillVLA training. "
                "It should be copied from raw observation.state before normalization."
            )
        skill_decoder_image = batch.get("skill_decoder_image")
        if skill_decoder_image is None and images:
            skill_decoder_image = images[0]

        # SP gradient flows into VLM (detach_sp_prefix=False)
        flow_losses, sp_loss, sd_loss = self.model.forward(
            images, img_masks, tokens, masks, actions,
            skill_index=batch.get("skill_index"),
            skill_sequence=batch.get("skill_sequence"),
            skill_length_sequence=batch.get("skill_length_sequence"),
            skill_sequence_len=batch.get("skill_sequence_len"),
            skill_ds=batch.get("skill_ds"),
            skill_de=batch.get("skill_de"),
            skill_boundary=batch.get("skill_boundary"),
            skill_max_order=batch.get("skill_max_order"),
            skill_max_length=batch.get("skill_max_length"),
            skill_decoder_state=skill_decoder_state,
            skill_decoder_image=skill_decoder_image,
            lang_to_action_masks=lang_to_action_masks,
            detach_sp_prefix=False,
            train_step=train_step,
        )
        if self.training:
            self.model._predicted_latent_train_step.add_(1)

        action_dim  = self.config.output_features[ACTION].shape[0]
        flow_losses = flow_losses[:, :, :action_dim]
        flow_loss   = flow_losses.mean()
        total_loss  = (
            flow_loss
            + self.config.skill_predictor_loss_weight * sp_loss
            + self.config.skill_decoder_loss_weight * sd_loss
        )

        f_b = batch.get("skill_boundary")
        n_boundaries = int(f_b.bool().sum().item()) if f_b is not None else 0
        loss_dict = {
            "loss":                        total_loss.item(),
            "loss_flow":                   flow_loss.item(),
            "loss_skill_predictor":        sp_loss.item(),
            "loss_skill_decoder":          sd_loss.item(),
            "n_skill_boundaries_in_batch": n_boundaries,
            "predicted_latent_prob":        self.model._last_predicted_latent_prob,
            "predicted_latent_fraction":    self.model._last_predicted_latent_fraction,
        }
        per_dim = flow_losses.mean(dim=[0, 1]).detach().cpu().tolist()
        for dim, value in enumerate(per_dim):
            loss_dict[f"loss_per_dim/{dim}"] = float(value)
        for name, value in self.model._last_sp_loss_components.items():
            loss_dict[f"loss_skill_predictor_{name}"] = value
        for name, value in self.model._last_skill_decoder_components.items():
            loss_dict[f"loss_skill_decoder_{name}"] = value

        if reduction == "none":
            return flow_losses.mean(dim=(1, 2)), loss_dict
        return total_loss, loss_dict

    # ── Inference state ───────────────────────────────────────────────────────

    def reset(self):
        super().reset()
        self._current_z        : Tensor | None = None
        self._current_token    : Tensor | None = None
        self._skill_step       : int           = 0
        self._skill_index      : int           = 0
        self._trigger_new_skill: bool          = False
        self._action_queue     : deque         = deque(maxlen=self.config.n_action_steps)
        self._episode_timestep : int           = 0
        self._skill_trace      : list[dict]    = []
        self._active_skill_trace_indices: list[int | None] = []
        n_forced = len(self._forced_skill_token_sequences) if hasattr(self, "_forced_skill_token_sequences") else 0
        self._forced_skill_token_cursors: list[int] = [0] * n_forced
        n_ref = len(self._reference_skill_token_sequences) if hasattr(self, "_reference_skill_token_sequences") else 0
        self._reference_skill_token_cursors: list[int] = [0] * n_ref

    def set_forced_skill_token_sequences(self, sequences: list[list[int]] | None) -> None:
        """Force eval to use label skill tokens instead of skill predictor outputs."""
        self._forced_skill_token_sequences = [list(seq) for seq in sequences] if sequences is not None else []
        self._forced_skill_token_cursors = [0] * len(self._forced_skill_token_sequences)

    def set_reference_skill_token_sequences(self, sequences: list[list[int]] | None) -> None:
        """Attach label skill records for logging while still using predictor tokens."""
        self._reference_skill_token_sequences = [list(seq) for seq in sequences] if sequences is not None else []
        self._reference_skill_token_cursors = [0] * len(self._reference_skill_token_sequences)

    def _next_skill_records_from(
        self,
        sequences_attr: str,
        cursors_attr: str,
        batch_size: int,
    ) -> list[dict] | None:
        sequences = getattr(self, sequences_attr, [])
        if not sequences:
            return None

        cursors = getattr(self, cursors_attr)
        records = []
        for batch_index in range(batch_size):
            if batch_index >= len(sequences) or len(sequences[batch_index]) == 0:
                return None
            cursor = cursors[batch_index]
            cursor = min(cursor, len(sequences[batch_index]) - 1)
            item = sequences[batch_index][cursor]
            if isinstance(item, dict):
                record = dict(item)
                record["token"] = int(record["token"])
            else:
                record = {"token": int(item)}
            records.append(record)
            cursors[batch_index] = cursor + 1
        return records

    def _next_forced_skill_records(self, batch_size: int) -> list[dict] | None:
        return self._next_skill_records_from(
            "_forced_skill_token_sequences",
            "_forced_skill_token_cursors",
            batch_size,
        )

    def _next_reference_skill_records(self, batch_size: int) -> list[dict] | None:
        return self._next_skill_records_from(
            "_reference_skill_token_sequences",
            "_reference_skill_token_cursors",
            batch_size,
        )

    def _record_skill_start(
        self,
        tokens: Tensor,
        *,
        source: str,
        label_records: list[dict] | None = None,
    ) -> None:
        self._active_skill_trace_indices = []
        tokens_cpu = tokens.detach().cpu().view(-1)
        for batch_index, token in enumerate(tokens_cpu.tolist()):
            label_record = label_records[batch_index] if label_records and batch_index < len(label_records) else None
            label_token = int(label_record["token"]) if label_record is not None and "token" in label_record else None
            skill_length = None
            if label_record is not None:
                skill_length = label_record.get("skill_length")
            trace_index = len(self._skill_trace)
            self._skill_trace.append(
                {
                    "skill_index": trace_index,
                    "batch_index": batch_index,
                    "episode_timestep": int(self._episode_timestep),
                    "codebook_token": int(token),
                    "skill_source": source,
                    "label_codebook_token": label_token,
                    "token_match": (int(token) == label_token) if label_token is not None else None,
                    "has_label_records": label_record is not None,
                    "has_label_prior": False,
                    "dataset_skill_length": int(skill_length) if skill_length is not None else None,
                    "length": 0,
                    "end_signal_timestep": None,
                    "end_signal_skill_step": None,
                    "end_signal_prob": None,
                    "end_probs": [],
                    "expert_actions": [],
                    "decoder_actions": [],
                }
            )
            self._active_skill_trace_indices.append(trace_index)

    def _update_active_skill_trace_length(self) -> None:
        for trace_index in self._active_skill_trace_indices:
            if trace_index is None or trace_index >= len(self._skill_trace):
                continue
            self._skill_trace[trace_index]["length"] = int(self._skill_step)

    def _record_end_signal(self, end_prob: Tensor) -> None:
        probs = end_prob.detach().float().cpu().view(-1).tolist()
        for batch_index, prob in enumerate(probs):
            if batch_index >= len(self._active_skill_trace_indices):
                continue
            trace_index = self._active_skill_trace_indices[batch_index]
            if trace_index is None or trace_index >= len(self._skill_trace):
                continue
            record = self._skill_trace[trace_index]
            record.setdefault("end_probs", []).append(
                {
                    "episode_timestep": int(self._episode_timestep),
                    "skill_step": int(self._skill_step),
                    "prob": float(prob),
                }
            )
            if prob >= float(self.config.skill_decoder_end_threshold) and record.get("end_signal_timestep") is None:
                record["end_signal_timestep"] = int(self._episode_timestep)
                record["end_signal_skill_step"] = int(self._skill_step)
                record["end_signal_prob"] = float(prob)

    def _record_decoder_delta(self, delta: Tensor) -> None:
        deltas = delta.detach().float().cpu().view(delta.shape[0], -1).tolist()
        for batch_index, values in enumerate(deltas):
            if batch_index >= len(self._active_skill_trace_indices):
                continue
            trace_index = self._active_skill_trace_indices[batch_index]
            if trace_index is None or trace_index >= len(self._skill_trace):
                continue
            self._skill_trace[trace_index].setdefault("decoder_actions", []).append(
                {
                    "episode_timestep": int(self._episode_timestep),
                    "skill_step": int(self._skill_step),
                    "delta": [float(v) for v in values],
                }
            )

    def record_executed_action(self, action: Tensor) -> None:
        actions = action.detach().float().cpu().view(action.shape[0], -1).tolist()
        # select_action increments these counters before the env step; the action
        # being recorded belongs to the previous policy timestep.
        episode_timestep = max(0, int(self._episode_timestep) - 1)
        skill_step = max(0, int(self._skill_step) - 1)
        for batch_index, values in enumerate(actions):
            if batch_index >= len(self._active_skill_trace_indices):
                continue
            trace_index = self._active_skill_trace_indices[batch_index]
            if trace_index is None or trace_index >= len(self._skill_trace):
                continue
            self._skill_trace[trace_index].setdefault("expert_actions", []).append(
                {
                    "episode_timestep": episode_timestep,
                    "skill_step": skill_step,
                    "action": [float(v) for v in values],
                }
            )

    def get_skill_trace(self) -> list[dict]:
        self._update_active_skill_trace_length()
        return list(self._skill_trace)

    def _update_skill_from_label_records(self, label_records: list[dict], current_state: Tensor) -> None:
        """Update active skill from oracle dataset tokens, bypassing the skill predictor."""
        label_tokens = torch.tensor(
            [int(record["token"]) for record in label_records],
            dtype=torch.long,
            device=current_state.device,
        )
        self._current_token = label_tokens
        self._current_z = self.model._token_to_z(label_tokens).to(device=current_state.device)
        self._skill_step = 0
        self._skill_index += 1
        self._trigger_new_skill = False
        self._record_skill_start(label_tokens, source="label", label_records=label_records)

    def _update_skill(
        self,
        skill_predictor_prefix: Tensor,
        prefix_pad_masks: Tensor,
        current_state: Tensor,
        skill_progress: Tensor,
    ) -> None:
        """Run the skill predictor at a skill boundary and store the active token."""
        b      = skill_predictor_prefix.shape[0]
        device = skill_predictor_prefix.device
        if self._current_token is None:
            input_token = torch.full(
                (b,),
                self.config.skill_predictor_num_embeddings + 1,  # BOS
                dtype=torch.long,
                device=device,
            )
        else:
            input_token = self._current_token.to(device=device)
        z_prev = self.model._skill_token_embedding(input_token).to(device=device)
        skill_index_norm = torch.full(
            (b,),
            float(self._skill_index) / max(1, int(self.config.inference_skill_max_order)),
            device=device,
            dtype=torch.float32,
        )

        sp_logits = self.model.skill_predictor(
            z_prev.float(),
            skill_predictor_prefix,
            prefix_pad_masks,
            skill_index_norm,
            skill_progress.float(),
        )
        # logits (B,D,L) → per-dim argmax → scalar FSQ index + z_q vector
        pred_tokens    = self.model._fsq_logits_to_token(sp_logits)
        self._current_token = pred_tokens
        self._current_z = self.model._fsq_logits_to_z(sp_logits).to(dtype=z_prev.dtype, device=device)

        self._skill_step  = 0
        self._skill_index += 1
        self._trigger_new_skill = False
        reference_records = self._next_reference_skill_records(b)
        self._record_skill_start(pred_tokens, source="pred", label_records=reference_records)

    def _decoder_image_from_batch(self, batch: dict[str, Tensor], images: list[Tensor]) -> Tensor | None:
        if "skill_decoder_image" in batch:
            return batch["skill_decoder_image"]
        return images[0] if images else None

    def _maybe_trigger_skill_end(self, batch: dict[str, Tensor], images: list[Tensor]) -> None:
        if self._trigger_new_skill or self._current_z is None:
            return
        state = batch.get("skill_decoder_state")
        if state is None:
            raise ValueError(
                "skill_decoder_state is required for FSQ end-signal inference. "
                "It should be copied from raw observation.state before normalization."
            )
        progress = torch.full(
            (self._current_z.shape[0],),
            float(self._skill_step) / max(1, int(self.config.inference_skill_max_length)),
            dtype=torch.float32,
            device=self._current_z.device,
        )
        decoder_out = self.model.skill_decoder_delta_end(
            self._current_z,
            state,
            self._decoder_image_from_batch(batch, images),
            progress,
        )
        end_prob = None if decoder_out is None else decoder_out[1]
        if end_prob is not None:
            self._record_decoder_delta(decoder_out[0])
            self._record_end_signal(end_prob)
        if end_prob is not None and bool((end_prob >= float(self.config.skill_decoder_end_threshold)).any().item()):
            self._trigger_new_skill = True

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks  = batch[OBS_LANGUAGE_ATTENTION_MASK]
        lang_to_action_masks = batch.get(OBS_LANG_TO_ACTION_ATTENTION_MASK)

        if self._current_z is None or (self._trigger_new_skill and len(self._action_queue) == 0):
            state = batch.get("skill_decoder_state")
            if state is None:
                raise ValueError(
                    "skill_decoder_state is required for SkillVLA inference. "
                    "It should be copied from raw observation.state before normalization."
                )
            label_records = self._next_forced_skill_records(tokens.shape[0])
            if label_records is not None:
                self._update_skill_from_label_records(label_records, state)
            else:
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
                self._update_skill(
                    skill_predictor_prefix,
                    prefix_pad_masks,
                    state,
                    skill_progress,
                )

        if len(self._action_queue) == 0:
            actions = self.model.sample_actions(
                images, img_masks, tokens, masks,
                z           = self._current_z,
                lang_to_action_masks=lang_to_action_masks,
            )
            action_dim = self.config.output_features[ACTION].shape[0]
            actions = actions[:, : self.config.n_action_steps, :action_dim]
            self._action_queue.extend(actions.transpose(0, 1))

        self._maybe_trigger_skill_end(batch, images)

        action = self._action_queue.popleft()
        self._skill_step += 1
        self._episode_timestep += 1
        self._update_active_skill_trace_length()
        return action
