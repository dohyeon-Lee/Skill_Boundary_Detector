"""Pretrained pi0.5 image/text prefix with the existing aux3 Expert interface."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.utils.checkpoint
from transformers import AutoTokenizer

from lerobot.policies.pi05.modeling_pi05 import (
    OPENPI_ATTENTION_MASK_VALUE, _gated_residual, layernorm_forward, resize_with_pad_torch,
)
from .modeling_utils import build_paligemma_model
from .independent_skill_planner import IndependentSkillPlannerExpert
from .vanila import VanilaSkillExpert
from .layerwise_cond_bottleneck import _LayerwiseConditionReader, LayerwiseCondBottleneckSkillExpert
from .wrist_patch_alignment import WristPatchAlignmentHead


def conditioning_prompts(state, coordinates, uv, valid):
    """Keep pi0.5's exact state discretization; UV uses 256 clipped bins.

    FSQ coordinates retain their normalized lattice values (-1, 0, 1 for
    FSQ333). No dataset task/instruction is consumed. Tokenization deliberately
    has no input gradient, just like pi0.5's original discrete state prefix.
    """
    for tensor in (state, coordinates, uv, valid):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("VLM prefix conditions must be finite.")
    states = np.digitize(state.detach().float().cpu().numpy(),
                         bins=np.linspace(-1, 1, 257)[:-1]) - 1
    skills = coordinates.detach().float().cpu().numpy()
    goals = ((uv.detach().float().clamp(-1, 1) + 1) * (255 / 2)).round().long().cpu().tolist()
    validity = valid.detach().reshape(-1).cpu().tolist()
    return [
        f"Task: Skill: {' '.join(format(float(x), '.6g') for x in skill)}; "
        f"Gripper goal in top-view image: {' '.join(map(str, goal)) if ok > 0 else 'none'}, "
        f"State: {' '.join(map(str, proprio))};\nAction: "
        for proprio, skill, goal, ok in zip(states, skills, goals, validity, strict=True)
    ]


class Pi05ConditionEncoder(nn.Module):
    """One registered pretrained trunk; .model exposes its 18-layer Gemma."""
    def __init__(self, config):
        super().__init__()
        self.vlm = build_paligemma_model("gemma_2b", image_size=config.dino_image_size)
        self.model.config._attn_implementation = config.cond_attention_backend
        self.freeze_images = config.freeze_vision_encoder
        if self.freeze_images:
            self.vlm.vision_tower.requires_grad_(False)
            self.vlm.vision_tower.eval()

    @property
    def model(self):
        return self.vlm.language_model

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_images:
            self.vlm.vision_tower.eval()
        return self

    def gradient_checkpointing_enable(self):
        if not self.freeze_images and hasattr(self.vlm.vision_tower, "gradient_checkpointing_enable"):
            self.vlm.vision_tower.gradient_checkpointing_enable()


class Pi05VLMVisionMixin:
    def __init__(self, config):
        super().__init__(config)
        width = int(self.cond_encoder.model.config.hidden_size)
        if len(self.cond_encoder.model.layers) != len(self.gemma_expert.model.layers):
            raise ValueError("VLM and Expert must have matching layer counts.")
        if not config.tokenizer_path:
            raise ValueError("pi05vlm requires a local PaliGemma tokenizer_path.")
        self.prefix_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, local_files_only=True)
        self.prefix_tokenizer.padding_side = "right"
        self.uses_cond_state_adarms = False
        # These old Cond-only adapters must not remain as unused parameters.
        self.state_proj = nn.Identity()
        self.cond_skill_condition = nn.Identity()
        self.cond_end_pose_condition = nn.Identity()
        self.layerwise_condition_memory_norm = nn.LayerNorm(width)
        self.layerwise_condition_readers = nn.ModuleList([
            _LayerwiseConditionReader(config.visual_bottleneck_width, width, config.visual_bottleneck_heads)
            for _ in self.cond_encoder.model.layers
        ])
        self.agent_patch_align_head = WristPatchAlignmentHead(config.visual_bottleneck_width, width)
        self.wrist_patch_align_head = WristPatchAlignmentHead(config.visual_bottleneck_width, width)

    def _condition_tokens(self, images, *, batch_size=None, skill_code=None):
        del batch_size, skill_code
        if len(images) != 2:
            raise ValueError("pi05vlm requires Top and Wrist images.")
        tokens = []
        for image in images:
            # Images already carry the existing crop/randomization. Apply the
            # pretrained SigLIP [-1,1] preprocessing instead of DINO/ImageNet.
            image = image.float()
            size = int(self.config.dino_image_size)
            if image.shape[-2:] != (size, size):
                image = resize_with_pad_torch(image, size, size)
            features = self.cond_encoder.vlm.get_image_features(image * 2 - 1).pooler_output
            tokens.append((features * math.sqrt(features.shape[-1])).to(self.working_dtype))
        if tokens[0].shape != tokens[1].shape:
            raise ValueError("Top and Wrist must have matching image token shapes.")
        return torch.cat(tokens, dim=1)

    def _project_condition_state(self, state, focus_uv=None, skill_code=None, end_pose=None):
        del focus_uv
        if state is None or skill_code is None:
            raise ValueError("pi05vlm requires proprio and skill.")
        # Preserve current proprio in ALL late Expert AdaRMS layers; only the
        # VLM's conditioning interface changes to discrete prefix tokens.
        self._bridge_condition_shift = self.bridge_proprio_condition(state.float()).to(self.working_dtype)
        uv, valid, _ = self._split_uv_goal(end_pose)
        return (state, self._code_to_zq(skill_code), uv, valid)

    def _condition_activation_tensors(self, representation):
        # The quantized text prefix carries four inputs, not one AdaRMS tensor.
        return dict(zip(("state_input", "skill_input", "goal_uv_input", "goal_uv_valid"), representation, strict=True))

    def _input_sensitivity_stats(self, **kwargs):
        # Diagnostic forwards must not replace the real forward's auxiliary
        # readouts with no-grad probe tensors before the policy computes loss.
        names = ("_final_agent_patch_tokens", "_final_wrist_patch_tokens", "_final_dual_patch_query")
        saved = {name: getattr(self, name) for name in names}
        try:
            return super()._input_sensitivity_stats(**kwargs)
        finally:
            for name, value in saved.items():
                setattr(self, name, value)

    def _embed_condition_text(self, condition, device):
        prompts = conditioning_prompts(*condition)
        encoded = self.prefix_tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
        if encoded["input_ids"].shape[1] > self.config.tokenizer_max_length:
            raise ValueError("Skill/UV/state prefix exceeds tokenizer_max_length; refusing to truncate conditions.")
        ids = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device=device, dtype=torch.bool)
        embedding = self.cond_encoder.model.embed_tokens(ids)
        return (embedding * math.sqrt(embedding.shape[-1])).to(self.working_dtype), mask

    def _vlm_layer_step(self, index, hidden, latent, mask, positions, rotary, image_tokens):
        layer = self.cond_encoder.model.layers[index]
        normalized, gate = layernorm_forward(layer.input_layernorm, hidden, None)
        options = {"is_causal": False} if self._cond_attention_is_sdpa else {}
        attended, _ = layer.self_attn(normalized, attention_mask=mask, position_ids=positions,
                                     past_key_values=None, use_cache=False, position_embeddings=rotary,
                                     **options)
        hidden = _gated_residual(hidden, attended, gate)
        normalized, gate = layernorm_forward(layer.post_attention_layernorm, hidden, None)
        hidden = _gated_residual(hidden, layer.mlp(normalized.to(layer.mlp.up_proj.weight.dtype)), gate)
        if index == len(self.cond_encoder.model.layers) - 1:
            hidden, _ = layernorm_forward(self.cond_encoder.model.norm, hidden, None)
        # Conditions can influence every image through bidirectional prefix
        # attention, but bottleneck/patch heads only read real image tokens.
        memory = self.layerwise_condition_memory_norm(hidden[:, :image_tokens].float())
        return hidden, self._update_layerwise_latent(index, latent, memory)

    def _encode_layerwise_latents(self, condition_tokens, condition_state):
        text, text_mask = self._embed_condition_text(condition_state, condition_tokens.device)
        count = condition_tokens.shape[1]
        hidden = torch.cat((condition_tokens, text), dim=1)
        valid = torch.cat((torch.ones(hidden.shape[0], count, device=hidden.device, dtype=torch.bool), text_mask), dim=1)
        # Match pi0.5's all-visible prefix block, NOT a causal text mask.
        allowed = valid[:, None, None, :].expand(-1, 1, hidden.shape[1], -1)
        mask = torch.where(allowed, 0.0, OPENPI_ATTENTION_MASK_VALUE).to(hidden.dtype)
        positions = (valid.long().cumsum(1) - 1).clamp_min(0)
        rotary = self.cond_encoder.model.rotary_emb(hidden, positions)
        latent = self._initial_condition_latent(hidden.shape[0])
        latents = []
        for index in range(len(self.cond_encoder.model.layers)):
            args = (index, hidden, latent, mask, positions, rotary, count)
            if self._checkpoint_layer("vlm", index, len(self.cond_encoder.model.layers)):
                hidden, latent = torch.utils.checkpoint.checkpoint(
                    self._vlm_layer_step, *args, use_reentrant=False, preserve_rng_state=True)
            else:
                hidden, latent = self._vlm_layer_step(*args)
            latents.append(latent)
        self._on_final_condition_hidden(hidden[:, :count])
        self._on_final_bottleneck_latent(latent)
        return latents

    def _initial_condition_latent(self, batch_size):
        return self.layerwise_latent_queries[None].expand(batch_size, -1, -1)

    def _on_final_condition_hidden(self, hidden):
        # SigLIP has no DINO CLS token to discard.
        from .bottleneck_video import capture_condition
        capture_condition(self, hidden, cls_tokens=False)
        half = hidden.shape[1] // 2
        self._final_agent_patch_tokens = hidden[:, :half] if self.training else None
        self._final_wrist_patch_tokens = hidden[:, half:] if self.training else None


class Pi05VLMIndependentSkillPlannerExpert(Pi05VLMVisionMixin, IndependentSkillPlannerExpert):
    """Pretrained VLM with the independent Fullskill/Action routes."""


class Pi05VLMVanilaSkillExpert(Pi05VLMVisionMixin, VanilaSkillExpert):
    """No-plan Expert with split inputs or uniform all-layer Z18 conditioning."""

    def __init__(self, config):
        super().__init__(config)
        self.split_layer = len(self.gemma_expert.model.layers) - config.visual_bridge_last_n_layers
        if config.uses_uniform_pi05_vanila_expert:
            self.split_layer = len(self.gemma_expert.model.layers)
            self.start_pose_condition = nn.Identity()
            self.bridge_proprio_condition = nn.Identity()
        self._split_time_condition = None

    @property
    def visual_bridge_start_layer(self):
        depth = len(self.gemma_expert.model.layers)
        if self.config.uses_uniform_pi05_vanila_expert:
            return 0
        if self.config.architecture_label == "vanila_last_pi05vlm_9_9":
            # Every late Expert layer reads the same final VLM bottleneck Z18.
            return depth - self.config.visual_bridge_last_n_layers
        return depth - 1

    def _project_condition_state(self, state, focus_uv=None, skill_code=None, end_pose=None):
        if self.config.uses_uniform_pi05_vanila_expert:
            if state is None or skill_code is None:
                raise ValueError("pi05vlm requires current proprio and skill for the VLM prefix.")
            self._bridge_condition_shift = self._current_proprio_condition = None
            uv, valid, _ = self._split_uv_goal(end_pose)
            return state, self._code_to_zq(skill_code), uv, valid
        condition = super()._project_condition_state(state, focus_uv, skill_code, end_pose)
        self._current_proprio_condition = self._bridge_condition_shift
        return condition

    def _expert_condition(self, timestep, projected_state=None, skill_code=None,
                          mode_latent=None, end_pose=None):
        if self.config.uses_uniform_pi05_vanila_expert:
            condition = self._time_condition(timestep)
            self._split_time_condition = condition
            if not self.config.uses_vision_only_expert:
                uv, valid, _ = self._split_uv_goal(end_pose)
                condition = condition + (self.end_pose_condition(uv) * valid).to(condition.dtype)
            return condition
        uv, valid, start = self._split_uv_goal(end_pose)
        self._split_time_condition = self._time_condition(timestep)
        condition = self._split_time_condition + self.start_pose_condition(start).to(self.working_dtype)
        if not self.config.omits_skill_expert_uv_goal:
            condition = condition + (self.end_pose_condition(uv) * valid).to(condition.dtype)
        return condition

    def _action_layer_condition(self, layer_index, expert_condition):
        if self.config.uses_uniform_pi05_vanila_expert:
            return expert_condition
        if layer_index < self.split_layer:
            return expert_condition
        if self._split_time_condition is None or self._current_proprio_condition is None:
            raise RuntimeError("Split Vanilla requires the current flow time and proprio condition")
        return self._split_time_condition + self._current_proprio_condition.to(self._split_time_condition.dtype)

    def _expert_layer_with_latent_bridge(self, *args):
        # Conditions are already materialized below. Bypass Arch17's extra
        # bridge-only proprio addition so L18 receives proprio exactly once.
        return LayerwiseCondBottleneckSkillExpert._expert_layer_with_latent_bridge(self, *args)

    def _run_expert_with_layerwise_latents(self, noisy_actions, expert_condition,
                                         expert_skill, layer_latents, *, return_all_layers=False):
        hidden = self._action_tokens(noisy_actions)
        mask, positions, rotary = self._sequence_geometry(hidden, self.gemma_expert.model)
        depth = len(self.gemma_expert.model.layers)
        late_condition = self._action_layer_condition(self.split_layer, expert_condition)
        all_hidden = []
        for index in range(depth):
            condition = expert_condition if index < self.split_layer else late_condition
            skill = expert_skill if index < self.split_layer and not self.config.uses_vision_only_expert else None
            latent = layer_latents[-1] if self._layer_uses_visual_bridge(index) else None
            args = (index, hidden, mask, positions, condition, skill, latent, rotary)
            if self._checkpoint_layer("action_expert", index, depth):
                hidden = torch.utils.checkpoint.checkpoint(
                    self._expert_layer_with_latent_bridge, *args,
                    use_reentrant=False, preserve_rng_state=True,
                )
            else:
                hidden = self._expert_layer_with_latent_bridge(*args)
            if return_all_layers:
                all_hidden.append(layernorm_forward(self.gemma_expert.model.norm, hidden, condition)[0])
        hidden = layernorm_forward(self.gemma_expert.model.norm, hidden, late_condition)[0]
        return (hidden, torch.stack(all_hidden, dim=1)) if return_all_layers else hidden

    def _input_sensitivity_stats(self, **kwargs):
        time_condition = self._split_time_condition
        try:
            return super()._input_sensitivity_stats(**kwargs)
        finally:
            self._split_time_condition = time_condition
