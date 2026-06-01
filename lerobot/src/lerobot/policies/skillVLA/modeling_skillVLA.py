"""SkillVLA policy — stage 3 joint training.

Extends PI05 with a skill predictor and an FSQ end-signal decoder.
All parameters are trainable; skill predictor gradient flows into the VLM.

Architecture changes vs PI05:
  1. forward      : flow matching + FSQ skill predictor loss.
  2. select_action: skill predictor predicts z for FSQ end-signal control.

For stages 1 & 2 (decoupled pre-training) use skillVLA_decouple instead.

Expected batch keys beyond standard PI05:
  - "skill_index"          : (B,) current skill index in skill_sequence (BOS=0)
  - "skill_sequence"       : (B, S) [BOS, skills..., EOS, PAD...]
  - "skill_length_sequence": (B, S) aligned skill lengths
  - "skill_ds" / "skill_de": (B,) distance from skill start/end
  - "skill_boundary"       : (B,) 1 at current skill end
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
from .skill_predictor import SkillPredictor

log = logging.getLogger(__name__)


# ── Core model ────────────────────────────────────────────────────────────────

class SkillVLAPytorch(PI05Pytorch):
    """PI05Pytorch + skill predictor + frozen FSQ end-signal decoder."""

    def __init__(self, config: SkillVLAConfig, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)

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

        self.skill_predictor = SkillPredictor(
            skill_latent_dim  = config.skill_latent_dim,
            prefix_hidden_dim = paligemma_config.width,
            fsq_dim           = config.skill_latent_dim,
            fsq_level         = self.fsq_level,
            hidden_dim        = config.skill_predictor_hidden_dim,
            num_heads         = config.skill_predictor_num_heads,
            num_layers        = config.skill_predictor_num_layers,
            dropout           = config.skill_predictor_dropout,
        )
        self.action_skill_memory_proj = nn.Sequential(
            nn.Linear(config.skill_latent_dim, paligemma_config.width),
            nn.SiLU(),
            nn.LayerNorm(paligemma_config.width),
        )
        self.action_progress_memory_proj = nn.Sequential(
            nn.Linear(1, paligemma_config.width),
            nn.SiLU(),
            nn.LayerNorm(paligemma_config.width),
        )
        self.action_fsq_latent_suffix_proj: nn.Module | None = None

        self.vae_decoder = None
        self.special_skill_embeddings = nn.Embedding(3, config.skill_latent_dim)
        self._last_sp_loss_components: dict[str, float] = {}
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
                "n_patch_raw", "image_token_dim", "chunk_size"}
        vae = SplineFSQAE(**{k: v for k, v in cfg_dict.items() if k in keys})
        vae.load_state_dict(ckpt["model_state"])
        for p in vae.parameters():
            p.requires_grad_(False)
        self.vae_decoder    = vae
        action_expert_config = get_gemma_config(self.config.action_expert_variant)
        fsq_hidden_dim = int(getattr(cfg, "hidden_dim", getattr(vae.dec_z_proj, "out_features")))
        self.action_fsq_latent_suffix_proj = nn.Sequential(
            nn.Linear(fsq_hidden_dim, action_expert_config.width),
            nn.SiLU(),
            nn.LayerNorm(action_expert_config.width),
        )
        log.info(
            "Loaded frozen FSQ decoder from %s (levels=%s, image_model=%s)",
            path,
            self.config.skill_fsq_levels,
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

    def _safe_real_tokens(self, tokens: Tensor, fallback: Tensor) -> Tensor:
        """Use fallback when a token is BOS/EOS/PAD but FSQ z is required."""
        tokens = tokens.view(-1).long()
        fallback = fallback.view(-1).long()
        valid = (tokens >= 0) & (tokens < self.config.skill_predictor_num_embeddings)
        return torch.where(valid, tokens, fallback)

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

    def _action_memory_tokens(self, skill_z: Tensor, progress: Tensor, *, dtype: torch.dtype) -> Tensor:
        skill_tok = self.action_skill_memory_proj(skill_z.float()).unsqueeze(1)
        progress_tok = self.action_progress_memory_proj(progress.float().view(-1, 1)).unsqueeze(1)
        return torch.cat([skill_tok, progress_tok], dim=1).to(dtype=dtype)

    def _fsq_latent_suffix_token(self, skill_z: Tensor, states: Tensor | None, *, device: torch.device) -> Tensor | None:
        if not self.config.use_fsq_latent_suffix:
            return None
        if self.vae_decoder is None or self.action_fsq_latent_suffix_proj is None or states is None:
            raise ValueError("FSQ skill latent suffix is enabled, but FSQ decoder/state is missing.")
        states = self._prepare_skill_decoder_state(states, device=device, dtype=skill_z.dtype)
        fsq_latent = self.vae_decoder.predict_skill_latent(skill_z.to(device=device), states, quantize=True)
        fsq_latent = fsq_latent[:, 0].float() if fsq_latent.ndim == 3 else fsq_latent.float()
        return self.action_fsq_latent_suffix_proj(fsq_latent).unsqueeze(1)

    def _prepend_latent_suffix_token(
        self,
        suffix_embs: Tensor,
        suffix_pad_masks: Tensor,
        suffix_att_masks: Tensor,
        latent_token: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if latent_token is None:
            return suffix_embs, suffix_pad_masks, suffix_att_masks
        latent_token = latent_token.to(device=suffix_embs.device, dtype=suffix_embs.dtype)
        latent_pad = torch.ones(
            suffix_embs.shape[0], 1, dtype=suffix_pad_masks.dtype, device=suffix_pad_masks.device
        )
        latent_att = torch.ones(
            suffix_embs.shape[0], 1, dtype=suffix_att_masks.dtype, device=suffix_att_masks.device
        )
        return (
            torch.cat([latent_token, suffix_embs], dim=1),
            torch.cat([latent_pad, suffix_pad_masks], dim=1),
            torch.cat([latent_att, suffix_att_masks], dim=1),
        )

    def _append_action_memory(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        memory_tokens: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        B, M = memory_tokens.shape[:2]
        memory_pad = torch.ones(B, M, dtype=prefix_pad_masks.dtype, device=prefix_pad_masks.device)
        memory_att = torch.zeros(B, M, dtype=prefix_att_masks.dtype, device=prefix_att_masks.device)
        return (
            torch.cat([prefix_embs, memory_tokens.to(device=prefix_embs.device, dtype=prefix_embs.dtype)], dim=1),
            torch.cat([prefix_pad_masks, memory_pad], dim=1),
            torch.cat([prefix_att_masks, memory_att], dim=1),
        )

    def _isolate_action_memory(self, att_2d_masks: Tensor, prefix_len: int, memory_len: int) -> Tensor:
        if memory_len <= 0:
            return att_2d_masks
        mem_start = prefix_len
        mem_end = prefix_len + memory_len
        att_2d_masks = att_2d_masks.clone()
        att_2d_masks[:, :prefix_len, mem_start:mem_end] = False
        att_2d_masks[:, mem_start:mem_end, :mem_end] = False
        idx = torch.arange(memory_len, device=att_2d_masks.device)
        att_2d_masks[:, mem_start + idx, mem_start + idx] = True
        return att_2d_masks

    def _contextualize_prefix_with_action_memory(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        memory_tokens: Tensor,
        *,
        use_cache: bool,
    ) -> tuple[Tensor, object]:
        prefix_len = prefix_pad_masks.shape[1]
        memory_len = memory_tokens.shape[1]
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._append_action_memory(
            prefix_embs, prefix_pad_masks, prefix_att_masks, memory_tokens
        )
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_2d = self._isolate_action_memory(prefix_att_2d, prefix_len, memory_len)
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
        return prefix_context, prefix_pad_masks, past_key_values

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
    ) -> dict[str, Tensor]:
        device = skill_sequence.device
        k = skill_index.view(-1).long()
        ds = skill_ds.view(-1).long()
        de = skill_de.view(-1).long()
        seq_len = skill_sequence_len.view(-1).long()
        last_real = (seq_len - 2).clamp_min(1)

        def _skill_progress(step: Tensor, length: Tensor) -> Tensor:
            denom = length.float().clamp_min(1.0)
            return (step.float() / denom).clamp(0.0, 1.0)

        current = self._gather_skill_sequence(skill_sequence, k)
        current_len = self._gather_skill_sequence(skill_length_sequence, k).float().clamp_min(1.0)
        prev_idx = (k - 1).clamp_min(0)
        prev = self._gather_skill_sequence(skill_sequence, prev_idx)

        predictor_input_token = prev.clone()
        predictor_progress = _skill_progress(ds, current_len)
        predictor_target = current.clone()

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
                next_len = self._gather_skill_sequence(skill_length_sequence, next_idx).float().clamp_min(1.0)
                early_denom = (next_len + p.float()).clamp_min(1.0)
                early_progress = ((p.float() - de.float()).clamp_min(0.0) / early_denom).clamp(0.0, 1.0)
                predictor_progress = torch.where(choose_early, early_progress, predictor_progress)
                next_token = self._gather_skill_sequence(skill_sequence, next_idx)
                predictor_target = torch.where(choose_early, next_token, predictor_target)

            if choose_late.any():
                prev_prev_idx = (k - 2).clamp_min(0)
                prev_skill_idx = (k - 1).clamp_min(0)
                prev_prev_token = self._gather_skill_sequence(skill_sequence, prev_prev_idx)
                prev_skill_token = self._gather_skill_sequence(skill_sequence, prev_skill_idx)
                prev_skill_len = self._gather_skill_sequence(skill_length_sequence, prev_skill_idx).float().clamp_min(1.0)
                late_denom = (prev_skill_len + p.float()).clamp_min(1.0)
                late_progress = ((prev_skill_len + ds.float()) / late_denom).clamp(0.0, 1.0)
                predictor_input_token = torch.where(choose_late, prev_prev_token, predictor_input_token)
                predictor_progress = torch.where(choose_late, late_progress, predictor_progress)
                predictor_target = torch.where(choose_late, prev_skill_token, predictor_target)

        return {
            "predictor_input_token": predictor_input_token,
            "predictor_progress": predictor_progress,
            "predictor_target": predictor_target,
            "action_condition_token": predictor_target,
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

    @torch.no_grad()
    def skill_decoder_progress_end(
        self,
        z: Tensor | None,
        state: Tensor | None,
        image: Tensor | None,
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
        progress, end_prob = self.vae_decoder.predict_termination(z, state, image_tokens, quantize=True)
        progress = progress[:, 0] if progress.ndim == 2 else progress
        end_prob = end_prob[:, 0] if end_prob.ndim == 2 else end_prob
        return progress, end_prob

    @torch.no_grad()
    def skill_decoder_end_prob(
        self,
        z: Tensor | None,
        state: Tensor | None,
        image: Tensor | None,
    ) -> Tensor | None:
        out = self.skill_decoder_progress_end(z, state, image)
        if out is None:
            return None
        return out[1]

    def _action_progress_from_fsq(
        self,
        z: Tensor,
        states: Tensor | None,
        images: Tensor | None,
        *,
        device: torch.device,
    ) -> Tensor:
        out = self.skill_decoder_progress_end(z.to(device=device), states, images)
        if out is None:
            raise ValueError("FSQ progress output is required for SkillVLA action memory conditioning.")
        return out[0].to(device=device, dtype=torch.float32)

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        fsq_latent_token: Tensor | None = None,
    ):
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self._prepend_latent_suffix_token(
            suffix_embs, suffix_pad_masks, suffix_att_masks, fsq_latent_token
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        past_key_values = copy.deepcopy(past_key_values)
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    @torch.no_grad()
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        *,
        skill_z: Tensor,
        skill_progress: Tensor,
        skill_decoder_state: Tensor | None = None,
        noise=None,
        num_steps=None,
        **kwargs,
    ) -> Tensor:
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device
        if noise is None:
            noise = self.sample_noise(
                (bsize, self.config.chunk_size, self.config.max_action_dim),
                device,
            )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        action_memory_tokens = self._action_memory_tokens(skill_z, skill_progress, dtype=prefix_embs.dtype)
        fsq_latent_token = self._fsq_latent_suffix_token(skill_z, skill_decoder_state, device=device)
        _, prefix_memory_pad_masks, past_key_values = self._contextualize_prefix_with_action_memory(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            action_memory_tokens,
            use_cache=True,
        )

        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    prefix_pad_masks=prefix_memory_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                    fsq_latent_token=fsq_latent_token,
                )

            if self._rtc_enabled():
                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
                    inference_delay=kwargs.get("inference_delay"),
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=kwargs.get("execution_horizon"),
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

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
        skill_decoder_state: Tensor | None = None,
        skill_decoder_image: Tensor | None = None,
        noise            : Tensor | None = None,
        time             : Tensor | None = None,
        detach_sp_prefix : bool          = True,   # False → sp_loss gradient flows into VLM
    ) -> tuple[Tensor, Tensor]:
        """Returns (flow_loss [B,chunk,max_dim], skill_predictor_loss)."""
        for name in (
            "skill_index",
            "skill_sequence",
            "skill_length_sequence",
            "skill_sequence_len",
            "skill_ds",
            "skill_de",
            "skill_boundary",
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

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)

        skill_targets = self._build_skill_training_targets(
            skill_index,
            skill_sequence,
            skill_length_sequence,
            skill_sequence_len,
            skill_ds,
            skill_de,
            skill_boundary,
        )
        predictor_input_z = self._skill_token_embedding(skill_targets["predictor_input_token"]).to(actions.device)

        fallback_token = torch.zeros_like(skill_targets["action_condition_token"].to(actions.device))
        action_condition_token = self._safe_real_tokens(
            skill_targets["action_condition_token"].to(actions.device),
            fallback_token,
        )
        action_skill_z = self._token_to_z(action_condition_token).to(actions.device)
        action_progress = self._action_progress_from_fsq(
            action_skill_z,
            skill_decoder_state,
            skill_decoder_image,
            device=actions.device,
        )
        action_memory_tokens = self._action_memory_tokens(action_skill_z, action_progress, dtype=prefix_embs.dtype)
        fsq_latent_token = self._fsq_latent_suffix_token(action_skill_z, skill_decoder_state, device=actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        source = self.sample_noise(actions.shape, actions.device).to(dtype=actions.dtype) if noise is None else noise
        time_exp  = time[:, None, None]
        x_t = time_exp * source + (1 - time_exp) * actions
        u_t = source - actions

        prefix_len = prefix_pad_masks.shape[1]

        # The action expert runs the prefix through the PaliGemma LLM; the skill
        # predictor reuses that SAME gemma-contextualized prefix (prefix portion —
        # action-memory tokens are attention-isolated so they don't affect it).
        if self.config.detach_action_prefix_grad:
            with torch.no_grad():
                # gradient_checkpointing forces use_cache=False inside the model, which
                # prevents KV-cache construction. Disable it for this call since we are
                # already under no_grad — activations are not saved either way.
                _pg_lm = self.paligemma_with_expert.paligemma.model.language_model
                _ge_m  = self.paligemma_with_expert.gemma_expert.model
                _pg_gc, _ge_gc = _pg_lm.gradient_checkpointing, _ge_m.gradient_checkpointing
                _pg_lm.gradient_checkpointing = False
                _ge_m.gradient_checkpointing  = False
                prefix_context, prefix_memory_pad_masks, past_key_values = self._contextualize_prefix_with_action_memory(
                    prefix_embs.detach(),
                    prefix_pad_masks,
                    prefix_att_masks,
                    action_memory_tokens,
                    use_cache=True,
                )
                _pg_lm.gradient_checkpointing = _pg_gc
                _ge_m.gradient_checkpointing  = _ge_gc
            v_t = self.denoise_step(
                prefix_memory_pad_masks,
                past_key_values,
                x_t,
                time,
                fsq_latent_token=fsq_latent_token,
            )
            # Action grad is detached above (no_grad). If the skill-predictor loss
            # should still train the VLM, re-contextualize the prefix WITH grad
            # (image+lang only — one extra prefix pass, only in this config);
            # otherwise reuse the detached context.
            if detach_sp_prefix:
                sp_prefix_ctx = prefix_context[:, :prefix_len]
            else:
                sp_prefix_ctx, _ = self.contextualize_prefix(prefix_embs, prefix_pad_masks, prefix_att_masks)
                sp_prefix_ctx = sp_prefix_ctx[:, :prefix_len]
        else:
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)
            suffix_embs, suffix_pad_masks, suffix_att_masks = self._prepend_latent_suffix_token(
                suffix_embs, suffix_pad_masks, suffix_att_masks, fsq_latent_token
            )

            if (
                self.paligemma_with_expert.paligemma.model.language_model
                .layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
            ):
                suffix_embs = suffix_embs.to(torch.bfloat16)
                prefix_embs = prefix_embs.to(torch.bfloat16)
                action_memory_tokens = action_memory_tokens.to(torch.bfloat16)

            memory_len = action_memory_tokens.shape[1]
            prefix_memory_embs, prefix_memory_pad_masks, prefix_memory_att_masks = self._append_action_memory(
                prefix_embs, prefix_pad_masks, prefix_att_masks, action_memory_tokens
            )
            pad_masks     = torch.cat([prefix_memory_pad_masks, suffix_pad_masks], dim=1)
            att_masks     = torch.cat([prefix_memory_att_masks, suffix_att_masks], dim=1)
            att_2d_masks  = make_att_2d_masks(pad_masks, att_masks)
            att_2d_masks = self._isolate_action_memory(att_2d_masks, prefix_len, memory_len)
            position_ids  = torch.cumsum(pad_masks, dim=1) - 1
            att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

            def _fwd(prefix_memory_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
                (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                    attention_mask  = att_2d_masks_4d,
                    position_ids    = position_ids,
                    past_key_values = None,
                    inputs_embeds   = [prefix_memory_embs, suffix_embs],
                    use_cache       = False,
                    adarms_cond     = [None, adarms_cond],
                )
                return prefix_out, suffix_out

            prefix_out, suffix_out = self._apply_checkpoint(
                _fwd, prefix_memory_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
            )
            sp_prefix_ctx = prefix_out[:, :prefix_len]
            if detach_sp_prefix:
                sp_prefix_ctx = sp_prefix_ctx.detach()
            suffix_out = suffix_out[:, -self.config.chunk_size:].to(torch.float32)
            v_t        = self._apply_checkpoint(self.action_out_proj, suffix_out)
        flow_loss  = F.mse_loss(u_t, v_t, reduction="none")

        # Skill predictor reads the gemma-contextualized prefix (read-only cross-attn).
        z_pred = self.skill_predictor(
            predictor_input_z.float(),
            sp_prefix_ctx,
            prefix_pad_masks,
            skill_targets["predictor_progress"].to(actions.device),
        )
        sp_loss = self._skill_predictor_loss(z_pred, skill_targets["predictor_target"].to(actions.device))

        return flow_loss, sp_loss

# ── Policy wrapper (stage 3 only) ─────────────────────────────────────────────

class SkillVLAPolicy(PI05Policy):
    """Stage 3: joint flow-matching + skill-predictor training.

    The skill predictor reads the same gemma-contextualized prefix the action expert
    uses (read-only cross-attention) and predicts the FSQ skill token. Whether each
    loss updates the VLM is controlled independently by detach_sp_prefix /
    detach_action_prefix_grad (both default False → both train it). The action expert
    follows the PI05 denoising path; predicted skills drive FSQ end-signal control.
    """

    config_class = SkillVLAConfig
    name         = "skill_vla"
    _VAE_DECODER_CHECKPOINT_PREFIXES = (
        "model.vae_decoder.enc_image_encoder.",
        "model.vae_decoder.enc_ctrl_proj.",
        "model.vae_decoder.enc_len_proj.",
        "model.vae_decoder.enc_traj_pool.",
        "model.vae_decoder.enc_fusion_pool.",
        "model.vae_decoder.z_head.",
        "model.vae_decoder.dec_image_encoder_plain.",
        "model.vae_decoder.dec_z_proj.",
        "model.vae_decoder.dec_state_proj.",
        "model.vae_decoder.skill_decoder_pool.",
        "model.vae_decoder.term_pool.",
        "model.vae_decoder.progress_head.",
        "model.vae_decoder.termination_head.",
    )

    def __init__(self, config: SkillVLAConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = SkillVLAPytorch(config, rtc_processor=self.rtc_processor)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)
        self.reset()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return super().from_pretrained(*args, **kwargs)

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
        skill_decoder_state = batch.get("skill_decoder_state")
        if skill_decoder_state is None:
            raise ValueError(
                "skill_decoder_state is required for SkillVLA action memory conditioning. "
                "It should be copied from raw observation.state before normalization."
            )
        skill_decoder_image = batch.get("skill_decoder_image")
        if skill_decoder_image is None and images:
            skill_decoder_image = images[0]

        # detach_sp_prefix / detach_action_prefix_grad independently control whether
        # each loss updates the VLM (both default False → both train it).
        flow_losses, sp_loss = self.model.forward(
            images, img_masks, tokens, masks, actions,
            skill_index=batch.get("skill_index"),
            skill_sequence=batch.get("skill_sequence"),
            skill_length_sequence=batch.get("skill_length_sequence"),
            skill_sequence_len=batch.get("skill_sequence_len"),
            skill_ds=batch.get("skill_ds"),
            skill_de=batch.get("skill_de"),
            skill_boundary=batch.get("skill_boundary"),
            skill_decoder_state=skill_decoder_state,
            skill_decoder_image=skill_decoder_image,
            detach_sp_prefix=self.config.detach_sp_prefix,
        )

        action_dim  = self.config.output_features[ACTION].shape[0]
        flow_losses = flow_losses[:, :, :action_dim]
        flow_loss   = flow_losses.mean()
        total_loss  = (
            flow_loss
            + self.config.skill_predictor_loss_weight * sp_loss
        )

        f_b = batch.get("skill_boundary")
        n_boundaries = int(f_b.bool().sum().item()) if f_b is not None else 0
        loss_dict = {
            "loss":                        total_loss.item(),
            "loss_flow":                   flow_loss.item(),
            "loss_skill_predictor":        sp_loss.item(),
            "n_skill_boundaries_in_batch": n_boundaries,
            "detach_action_prefix_grad":   float(self.config.detach_action_prefix_grad),
        }
        per_dim = flow_losses.mean(dim=[0, 1]).detach().cpu().tolist()
        for dim, value in enumerate(per_dim):
            loss_dict[f"loss_per_dim/{dim}"] = float(value)
        for name, value in self.model._last_sp_loss_components.items():
            loss_dict[f"loss_skill_predictor_{name}"] = value

        if reduction == "none":
            return flow_losses.mean(dim=(1, 2)), loss_dict
        return total_loss, loss_dict

    # ── Inference state ───────────────────────────────────────────────────────

    def reset(self):
        super().reset()
        self._current_z        : Tensor | None = None
        self._current_token    : Tensor | None = None
        self._current_progress : Tensor | None = None
        self._skill_step       : int           = 0
        self._trigger_new_skill: bool          = False
        self._action_queue     : deque         = deque(maxlen=self.config.n_action_steps)
        self._episode_timestep : int           = 0
        self._skill_trace      : list[dict]    = []
        self._active_skill_trace_indices: list[int | None] = []
        self._prev_skill_decoder_state: Tensor | None = None
        self._last_executed_gripper: dict[int, float] = {}
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

    def _record_expert_chunk(self, actions: Tensor) -> None:
        actions_cpu = actions.detach().float().cpu()
        if actions_cpu.ndim != 3:
            return
        chunks = actions_cpu.tolist()
        for batch_index, chunk in enumerate(chunks):
            if batch_index >= len(self._active_skill_trace_indices):
                continue
            trace_index = self._active_skill_trace_indices[batch_index]
            if trace_index is None or trace_index >= len(self._skill_trace):
                continue
            self._skill_trace[trace_index].setdefault("expert_action_chunks", []).append(
                {
                    "episode_timestep": int(self._episode_timestep),
                    "skill_step": int(self._skill_step),
                    "chunk": [[float(v) for v in row] for row in chunk],
                }
            )

    def _record_state_delta(self, state: Tensor) -> None:
        """Record actual state delta (state_t - state_{t-1}) using same dims as FSQ decoder."""
        indices = self.config.skill_decoder_state_indices
        if indices is not None:
            idx = torch.as_tensor(indices, dtype=torch.long, device=state.device)
            sel = state.index_select(-1, idx).detach().float().cpu()
        else:
            vae = getattr(self.model, "vae_decoder", None)
            expected = int(getattr(vae, "state_dim", state.shape[-1])) if vae is not None else state.shape[-1]
            sel = state[..., :expected].detach().float().cpu()

        if self._prev_skill_decoder_state is None:
            self._prev_skill_decoder_state = sel
            return

        delta = sel - self._prev_skill_decoder_state
        self._prev_skill_decoder_state = sel

        if not self._active_skill_trace_indices:
            return
        deltas_list = delta.view(delta.shape[0], -1).tolist()
        for batch_index, values in enumerate(deltas_list):
            if batch_index >= len(self._active_skill_trace_indices):
                continue
            trace_index = self._active_skill_trace_indices[batch_index]
            if trace_index is None or trace_index >= len(self._skill_trace):
                continue
            # Replace last dim with actual gripper action (matches FSQ training convention)
            gripper = self._last_executed_gripper.get(batch_index)
            if gripper is not None and len(values) > 0:
                values = list(values[:-1]) + [gripper]
            self._skill_trace[trace_index].setdefault("state_deltas", []).append(
                {
                    "episode_timestep": max(0, int(self._episode_timestep) - 1),
                    "skill_step": max(0, int(self._skill_step) - 1),
                    "delta": [float(v) for v in values],
                }
            )

    def record_executed_action(self, action: Tensor) -> None:
        actions = action.detach().float().cpu().view(action.shape[0], -1)
        # Store last gripper action per batch element for state-delta gripper correction.
        for b in range(actions.shape[0]):
            self._last_executed_gripper[b] = float(actions[b, -1])
        actions_list = actions.tolist()
        # select_action increments these counters before the env step; the action
        # being recorded belongs to the previous policy timestep.
        episode_timestep = max(0, int(self._episode_timestep) - 1)
        skill_step = max(0, int(self._skill_step) - 1)
        for batch_index, values in enumerate(actions_list):
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
        self._current_progress = torch.zeros(label_tokens.shape[0], device=current_state.device, dtype=torch.float32)
        self._skill_step = 0
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

        sp_logits = self.model.skill_predictor(
            z_prev.float(),
            skill_predictor_prefix,
            prefix_pad_masks,
            skill_progress.float(),
        )
        # logits (B,D,L) → per-dim argmax → scalar FSQ index + z_q vector
        pred_tokens    = self.model._fsq_logits_to_token(sp_logits)
        self._current_token = pred_tokens
        self._current_z = self.model._fsq_logits_to_z(sp_logits).to(dtype=z_prev.dtype, device=device)
        self._current_progress = torch.zeros(b, device=device, dtype=torch.float32)

        self._skill_step  = 0
        self._trigger_new_skill = False
        reference_records = self._next_reference_skill_records(b)
        self._record_skill_start(pred_tokens, source="pred", label_records=reference_records)

    def _decoder_image_from_batch(self, batch: dict[str, Tensor], images: list[Tensor]) -> Tensor | None:
        if "skill_decoder_image" in batch:
            return batch["skill_decoder_image"]
        return images[0] if images else None

    def _start_skill_from_batch(
        self,
        batch: dict[str, Tensor],
        images: list[Tensor],
        img_masks: list[Tensor],
        tokens: Tensor,
        masks: Tensor,
    ) -> None:
        state = batch.get("skill_decoder_state")
        if state is None:
            raise ValueError(
                "skill_decoder_state is required for SkillVLA inference. "
                "It should be copied from raw observation.state before normalization."
            )
        label_records = self._next_forced_skill_records(tokens.shape[0])
        if label_records is not None:
            self._update_skill_from_label_records(label_records, state)
            return

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(images, img_masks, tokens, masks)
        # Match training: the skill predictor reads the gemma-contextualized prefix.
        prefix_ctx, _ = self.model.contextualize_prefix(prefix_embs, prefix_pad_masks, prefix_att_masks)
        skill_progress = torch.zeros(tokens.shape[0], device=tokens.device, dtype=torch.float32)
        self._update_skill(prefix_ctx, prefix_pad_masks, state, skill_progress)

    def _refresh_current_progress(self, batch: dict[str, Tensor], images: list[Tensor]) -> Tensor | None:
        if self._current_z is None:
            return None
        state = batch.get("skill_decoder_state")
        if state is None:
            raise ValueError(
                "skill_decoder_state is required for FSQ progress inference. "
                "It should be copied from raw observation.state before normalization."
            )
        decoder_out = self.model.skill_decoder_progress_end(
            self._current_z,
            state,
            self._decoder_image_from_batch(batch, images),
        )
        if decoder_out is None:
            return None
        self._current_progress = decoder_out[0].to(device=self._current_z.device, dtype=torch.float32)
        return decoder_out[1]

    def _maybe_trigger_skill_end(self, batch: dict[str, Tensor], images: list[Tensor]) -> None:
        if self._trigger_new_skill or self._current_z is None:
            return
        # Safety cap (inference only): force a skill switch once the current skill has
        # run for inference_skill_max_length steps, even if FSQ termination never fired.
        max_len = int(self.config.inference_skill_max_length)
        if max_len > 0 and self._skill_step >= max_len:
            self._trigger_new_skill = True
            return
        end_prob = self._refresh_current_progress(batch, images)
        if end_prob is not None:
            self._record_end_signal(end_prob)
        if end_prob is not None and bool((end_prob >= float(self.config.skill_decoder_end_threshold)).any().item()):
            self._trigger_new_skill = True

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks  = batch[OBS_LANGUAGE_ATTENTION_MASK]

        if self._current_z is None or (self._trigger_new_skill and len(self._action_queue) == 0):
            self._start_skill_from_batch(batch, images, img_masks, tokens, masks)

        self._maybe_trigger_skill_end(batch, images)
        if self._trigger_new_skill and len(self._action_queue) == 0:
            self._start_skill_from_batch(batch, images, img_masks, tokens, masks)

        if len(self._action_queue) == 0:
            state = batch.get("skill_decoder_state")
            if state is None:
                raise ValueError(
                    "skill_decoder_state is required for FSQ termination inference. "
                    "It should be copied from raw observation.state before normalization."
            )
            self._refresh_current_progress(batch, images)
            if self._current_progress is None:
                raise ValueError("FSQ progress output is required before action chunk generation.")
            actions = self.model.sample_actions(
                images, img_masks, tokens, masks,
                skill_z=self._current_z,
                skill_progress=self._current_progress,
                skill_decoder_state=state,
            )
            action_dim = self.config.output_features[ACTION].shape[0]
            self._record_expert_chunk(actions[:, :, :action_dim])
            actions = actions[:, : self.config.n_action_steps, :action_dim]
            self._action_queue.extend(actions.transpose(0, 1))

        action = self._action_queue.popleft()
        raw_state = batch.get("skill_decoder_state")
        if raw_state is not None:
            self._record_state_delta(raw_state)
        self._skill_step += 1
        self._episode_timestep += 1
        self._update_active_skill_trace_length()
        return action
