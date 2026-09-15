"""pi0.5-VLM skill predictor used by Stage 1 and auxiliary training."""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.pi05.lora import (
    NamedLoRALinear,
    inject_named_lora,
    set_active_adapters,
    target_names_from_spec,
)
from lerobot.policies.pi05.modeling_pi05 import (
    OPENPI_ATTENTION_MASK_VALUE,
    layernorm_forward,
    resize_with_pad_torch,
)
from lerobot.policies.skillVLA.skill_head import SkillHead
from lerobot.policies.skillVLA.skill_reader import SkillReader

from .configuration_skill_expert import SkillExpertConfig
from .modeling_utils import build_paligemma_model


class FrozenVLMSkillPredictor(nn.Module):
    """Read skill-start observations and predict skill plus optional focus UV.

    By default the pi0.5 VLM stays frozen and an optional named LoRA is trained.
    Auxiliary training may instead explicitly co-train the complete VLM.  The
    skill and UV branches share one VLM forward but use independent readers;
    the UV reader is conditioned on the selected FSQ coordinates.
    """

    def __init__(self, config: SkillExpertConfig):
        super().__init__()
        self.config = config
        self.vlm = build_paligemma_model(
            config.skill_predictor_vlm_variant,
            image_size=config.skill_predictor_image_size,
        )
        width = int(self.vlm.language_model.config.hidden_size)
        self.reader = SkillReader(
            width,
            depth=config.skill_predictor_reader_depth,
            heads=config.skill_predictor_reader_heads,
            num_probes=config.skill_predictor_reader_tokens,
        )
        self.head = SkillHead(
            width,
            config.skill_fsq_levels,
            deadzone_frac=config.skill_predictor_deadzone_frac,
        )
        self.focus_uv_reader: SkillReader | None = None
        self.focus_uv_skill_projection: nn.Module | None = None
        self.focus_uv_head: nn.Module | None = None
        if config.skill_predictor_focus_uv_enabled:
            self.focus_uv_reader = SkillReader(
                width,
                depth=config.skill_predictor_reader_depth,
                heads=config.skill_predictor_reader_heads,
                num_probes=config.skill_predictor_reader_tokens,
            )
            self.focus_uv_skill_projection = nn.Sequential(
                nn.Linear(len(config.skill_fsq_levels), width),
                nn.SiLU(),
                nn.Linear(width, width),
            )
            self.focus_uv_head = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, width),
                nn.SiLU(),
                nn.Linear(width, 2),
                nn.Tanh(),
            )

        levels = torch.tensor(config.skill_fsq_levels, dtype=torch.long)
        strides = torch.ones_like(levels)
        for index in range(1, len(config.skill_fsq_levels)):
            strides[index] = strides[index - 1] * levels[index - 1]
        self.register_buffer("_focus_fsq_levels", levels, persistent=False)
        self.register_buffer("_focus_fsq_strides", strides, persistent=False)
        self.register_buffer(
            "_focus_fsq_half", (levels - 1).float() / 2.0, persistent=False
        )

        self.lora_layer_count = 0
        if config.skill_predictor_lora:
            self.lora_layer_count = self.add_lora_adapter(
                "skill",
                config.skill_predictor_lora_targets,
                config.skill_predictor_lora_rank,
                config.skill_predictor_lora_alpha,
                config.skill_predictor_lora_dropout,
            )

        # The main Stage-1 optimizer must never register predictor parameters.
        self.vlm.requires_grad_(False)
        self.reader.requires_grad_(False)
        self.head.requires_grad_(False)
        if self.focus_uv_reader is not None:
            self.focus_uv_reader.requires_grad_(False)
            self.focus_uv_skill_projection.requires_grad_(False)
            self.focus_uv_head.requires_grad_(False)
        self._train_vlm_base = False
        self.vlm.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._train_vlm_base:
            self.vlm.train(mode)
        else:
            self.vlm.eval()
        # A LoRA-only run keeps the vision tower deterministic while enabling
        # language-model checkpointing and adapter dropout.
        if self._lora_attached_to_loss and not self._train_vlm_base:
            self.vlm.language_model.train(mode)
        return self

    @property
    def _lora_attached_to_loss(self) -> bool:
        return bool(
            self.config.skill_predictor_lora
            and not self.config.skill_predictor_detach_vlm
        )

    @property
    def _vlm_attached_to_loss(self) -> bool:
        return self._train_vlm_base or self._lora_attached_to_loss

    def set_train_vlm_base(self, enabled: bool) -> None:
        """Enable complete VLM co-training for the auxiliary predictor only."""
        enabled = bool(enabled)
        if enabled and self.config.skill_predictor_lora:
            raise ValueError("Complete predictor-VLM co-training cannot use LoRA.")
        self._train_vlm_base = enabled
        self.vlm.requires_grad_(enabled)
        if enabled:
            self.vlm.train(self.training)
        else:
            self.vlm.eval()

    def vlm_parameters(self) -> list[nn.Parameter]:
        return list(self.vlm.parameters()) if self._train_vlm_base else []

    def reader_head_parameters(self) -> list[nn.Parameter]:
        parameters = [*self.reader.parameters(), *self.head.parameters()]
        if self.focus_uv_reader is not None:
            parameters.extend(self.focus_uv_reader.parameters())
            parameters.extend(self.focus_uv_skill_projection.parameters())
            parameters.extend(self.focus_uv_head.parameters())
        return parameters

    def add_lora_adapter(
        self,
        name: str,
        targets: str,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> int:
        """Attach one independently selectable LoRA branch to the frozen VLM."""
        target_names = target_names_from_spec(targets)
        wrapped = inject_named_lora(
            self.vlm.language_model,
            target_names,
            name,
            rank,
            alpha,
            dropout,
        )
        if wrapped == 0:
            raise RuntimeError(
                f"{name} LoRA did not match any VLM projection; targets={targets!r}."
            )
        qkvo = {"q_proj", "k_proj", "v_proj", "o_proj"}
        if target_names == qkvo:
            expected = 4 * int(
                self.vlm.language_model.config.num_hidden_layers
            )
            if wrapped != expected:
                raise RuntimeError(
                    f"{name} Q/K/V/O LoRA must cover every VLM layer; "
                    f"wrapped={wrapped}, expected={expected}."
                )
        return wrapped

    def adapter_parameters(self, name: str) -> list[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        for module in self.vlm.language_model.modules():
            if isinstance(module, NamedLoRALinear) and name in module.adapters:
                parameters.extend(module.adapters[name].parameters())
        return parameters

    def lora_parameters(self) -> list[nn.Parameter]:
        return self.adapter_parameters("skill")

    def set_adapter_training(self, name: str, mode: bool) -> None:
        """Toggle only one adapter's dropout/training state, not the base VLM."""
        for module in self.vlm.language_model.modules():
            if isinstance(module, NamedLoRALinear) and name in module.adapters:
                module.adapters[name].train(mode)

    def auxiliary_parameters(self) -> list[nn.Parameter]:
        return [
            *self.reader_head_parameters(),
            *self.lora_parameters(),
            *self.vlm_parameters(),
        ]

    def gradient_checkpointing_enable(self) -> None:
        # PiGemma decoder layers inherit Transformers' GradientCheckpointingLayer.
        # Calling the public enable method propagates both the flag and the
        # checkpoint function to every layer; setting only the top-level boolean
        # leaves the layer loop completely uncheckpointed.
        self.vlm.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    @staticmethod
    def _activate_skill_adapter() -> None:
        # Sticky selection is required when a checkpointed forward is later
        # recomputed during backward.
        set_active_adapters({"skill"})

    def _preprocess_image(self, image: Tensor) -> Tensor:
        image = image.to(torch.float32)
        channels_first = image.shape[1] == 3
        if channels_first:
            image = image.permute(0, 2, 3, 1)
        target = (self.config.skill_predictor_image_size,) * 2
        if tuple(image.shape[1:3]) != target:
            image = resize_with_pad_torch(image, *target)
        image = image * 2.0 - 1.0
        return image.permute(0, 3, 1, 2) if channels_first else image

    def _embed_prefix(
        self,
        images: list[Tensor],
        language_tokens: Tensor,
        language_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        embeddings: list[Tensor] = []
        padding: list[Tensor] = []
        is_language: list[bool] = []
        working_dtype = self.vlm.language_model.embed_tokens.weight.dtype

        for image in images:
            image = self._preprocess_image(image)
            output_dtype = image.dtype
            image_output = self.vlm.get_image_features(image)
            image_embedding = image_output.pooler_output * math.sqrt(
                image_output.pooler_output.shape[-1]
            )
            image_embedding = image_embedding.to(output_dtype)
            token_count = image_embedding.shape[1]
            embeddings.append(image_embedding)
            padding.append(
                torch.ones(
                    image_embedding.shape[0],
                    token_count,
                    dtype=torch.bool,
                    device=image_embedding.device,
                )
            )
            is_language.extend([False] * token_count)

        language_embedding = self.vlm.language_model.embed_tokens(language_tokens)
        language_embedding = language_embedding * math.sqrt(language_embedding.shape[-1])
        embeddings.append(language_embedding)
        padding.append(language_mask.bool())
        is_language.extend([True] * language_embedding.shape[1])

        prefix = torch.cat([embedding.to(working_dtype) for embedding in embeddings], dim=1)
        valid = torch.cat(padding, dim=1)
        language_positions = torch.tensor(
            is_language, dtype=torch.bool, device=prefix.device
        )
        readable = torch.zeros_like(language_positions)
        if self.config.skill_predictor_attend_image:
            readable |= ~language_positions
        if self.config.skill_predictor_attend_language:
            readable |= language_positions
        if not bool(readable.any()):
            raise ValueError("Skill predictor must attend image and/or language tokens.")
        key_ignore = (~valid) | ~readable[None]
        return prefix, valid, key_ignore

    def _encode_prefix(
        self,
        prefix: Tensor,
        valid: Tensor,
        *,
        all_layers: bool | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        attention = valid[:, None, :] & valid[:, :, None]
        attention = torch.where(
            attention[:, None], 0.0, OPENPI_ATTENTION_MASK_VALUE
        ).to(prefix.dtype)
        positions = torch.cumsum(valid, dim=1) - 1
        if all_layers is None:
            all_layers = self.config.skill_predictor_all_layers
        output = self.vlm.language_model.forward(
            inputs_embeds=prefix,
            attention_mask=attention,
            position_ids=positions,
            past_key_values=None,
            use_cache=False,
            adarms_cond=None,
            output_hidden_states=all_layers,
        )
        if not all_layers:
            return output.last_hidden_state, None
        normalized = [
            layernorm_forward(self.vlm.language_model.norm, hidden, None)[0]
            for hidden in output.hidden_states[1:-1]
        ]
        normalized.append(output.last_hidden_state)
        return output.last_hidden_state, torch.stack(normalized, dim=1)

    @torch.no_grad()
    def encode_last_hidden(
        self,
        images: list[Tensor],
        language_tokens: Tensor,
        language_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return the skill-LoRA-adapted VLM's final joint token sequence."""
        self._activate_skill_adapter()
        return self._encode_last_hidden(images, language_tokens, language_mask)

    @torch.no_grad()
    def encode_base_last_hidden(
        self,
        images: list[Tensor],
        language_tokens: Tensor,
        language_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return pure frozen-base VLM memory with every named adapter disabled.

        Stage 2 uses this path for every likelihood block, while predictor
        ``loss``/``predict`` keep using the independently trained ``skill`` LoRA.
        """
        set_active_adapters(set())
        return self._encode_last_hidden(images, language_tokens, language_mask)

    @torch.no_grad()
    def encode_base_hidden_stack(
        self,
        images: list[Tensor],
        language_tokens: Tensor,
        language_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return the frozen-base VLM's per-layer memory stack, adapters disabled.

        Shapes: ``(B, num_layers, N, width)`` plus the ``(B, N)`` key-padding
        mask. Every intermediate layer is normalized with the final norm (the
        reader's ``all_layers`` contract) so Stage-2 layer mixing combines
        comparably scaled representations.
        """
        set_active_adapters(set())
        prefix, valid, _ = self._embed_prefix(images, language_tokens, language_mask)
        _, layer_stack = self._encode_prefix(prefix, valid, all_layers=True)
        return layer_stack.detach(), (~valid).detach()

    def encode_named_last_hidden(
        self,
        name: str,
        images: list[Tensor],
        language_tokens: Tensor,
        language_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Encode VLM memory with exactly one trainable named adapter active."""
        set_active_adapters({name})
        prefix, valid, _ = self._embed_prefix(
            images, language_tokens, language_mask
        )
        hidden, _ = self._encode_prefix(prefix, valid, all_layers=False)
        return hidden, (~valid).detach()

    def encode_named_hidden_stack(
        self,
        name: str,
        images: list[Tensor],
        language_tokens: Tensor,
        language_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Encode every VLM layer with exactly one named adapter active."""
        set_active_adapters({name})
        prefix, valid, _ = self._embed_prefix(
            images, language_tokens, language_mask
        )
        _, layer_stack = self._encode_prefix(prefix, valid, all_layers=True)
        return layer_stack, (~valid).detach()

    def _encode_last_hidden(
        self,
        images: list[Tensor],
        language_tokens: Tensor,
        language_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Encode joint image/language tokens under the caller-selected adapter route."""
        prefix, valid, _ = self._embed_prefix(images, language_tokens, language_mask)
        hidden, _ = self._encode_prefix(prefix, valid, all_layers=False)
        return hidden.detach(), (~valid).detach()

    def loss(
        self,
        images: list[Tensor],
        language_tokens: Tensor,
        language_mask: Tensor,
        skill_code: Tensor,
    ) -> tuple[Tensor, float]:
        memory, key_ignore = self.reader_memory(images, language_tokens, language_mask)
        reader_hidden = self.reader(memory, key_ignore)
        loss = self.head.loss(reader_hidden, skill_code)
        with torch.no_grad():
            accuracy = (self.head.decode(reader_hidden) == skill_code).float().mean().item()
        return loss, accuracy

    def loss_with_focus_uv(
        self,
        images: list[Tensor],
        language_tokens: Tensor,
        language_mask: Tensor,
        skill_code: Tensor,
        focus_uv: Tensor,
        focus_valid: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """Joint skill/UV objective using GT skill to condition the UV branch."""
        if self.focus_uv_reader is None:
            raise RuntimeError("Focus-UV prediction is not enabled.")
        memory, key_ignore = self.reader_memory(images, language_tokens, language_mask)
        skill_hidden = self.reader(memory, key_ignore)
        skill_loss = self.head.loss(skill_hidden, skill_code)
        with torch.no_grad():
            accuracy = (
                self.head.decode(skill_hidden) == skill_code
            ).float().mean().item()

        predicted_uv = self.predict_focus_uv_from_memory(
            memory, key_ignore, skill_code
        )
        target_uv = focus_uv.to(
            device=predicted_uv.device, dtype=predicted_uv.dtype
        ).reshape(-1, 2)
        valid = focus_valid.to(device=predicted_uv.device).reshape(-1).bool()
        if target_uv.shape[0] != predicted_uv.shape[0] or valid.shape[0] != predicted_uv.shape[0]:
            raise ValueError(
                "Focus-UV batch shape mismatch: "
                f"predicted={tuple(predicted_uv.shape)}, target={tuple(target_uv.shape)}, "
                f"valid={tuple(valid.shape)}."
            )
        if bool(valid.any()):
            uv_loss = F.smooth_l1_loss(predicted_uv[valid], target_uv[valid])
            uv_mae = (predicted_uv[valid] - target_uv[valid]).abs().mean()
        else:
            # Keep a differentiable zero so a batch containing only invalid
            # projections does not contaminate the skill objective.
            uv_loss = predicted_uv.sum() * 0.0
            uv_mae = predicted_uv.detach().new_zeros(())
        total = skill_loss + self.config.skill_predictor_focus_uv_loss_weight * uv_loss
        return total, {
            "skill_loss": float(skill_loss.detach()),
            "skill_accuracy": float(accuracy),
            "focus_uv_loss": float(uv_loss.detach()),
            "focus_uv_mae": float(uv_mae.detach()),
            "focus_uv_valid_fraction": float(valid.float().mean().detach()),
            "total_loss": float(total.detach()),
        }

    def reader_memory(
        self,
        images: list[Tensor],
        language_tokens: Tensor,
        language_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute the shared VLM memory once for all predictor heads."""
        self._activate_skill_adapter()
        context = nullcontext() if self._vlm_attached_to_loss else torch.no_grad()
        with context:
            prefix, valid, key_ignore = self._embed_prefix(
                images, language_tokens, language_mask
            )
            hidden, layer_stack = self._encode_prefix(prefix, valid)
        if not self._vlm_attached_to_loss:
            hidden = hidden.detach()
            layer_stack = None if layer_stack is None else layer_stack.detach()

        if layer_stack is not None:
            batch, layers, tokens, width = layer_stack.shape
            return (
                layer_stack.reshape(batch, layers * tokens, width),
                key_ignore.repeat(1, layers),
            )
        return hidden, key_ignore

    def reader_hidden(
        self,
        images: list[Tensor],
        language_tokens: Tensor,
        language_mask: Tensor,
    ) -> Tensor:
        """Return the trainable reader output over frozen (or LoRA) VLM memory."""
        memory, key_ignore = self.reader_memory(images, language_tokens, language_mask)
        return self.reader(memory, key_ignore)

    def _skill_code_coordinates(self, skill_code: Tensor, dtype: torch.dtype) -> Tensor:
        index = skill_code.reshape(-1, 1).long()
        level_ids = (
            torch.div(index, self._focus_fsq_strides[None], rounding_mode="floor")
            % self._focus_fsq_levels[None]
        )
        return (
            (level_ids.float() - self._focus_fsq_half[None])
            / self._focus_fsq_half[None]
        ).to(dtype=dtype)

    def predict_focus_uv_from_memory(
        self,
        memory: Tensor,
        key_ignore: Tensor,
        skill_code: Tensor,
    ) -> Tensor:
        """Predict normalized endpoint UV from shared VLM memory and one skill."""
        if (
            self.focus_uv_reader is None
            or self.focus_uv_skill_projection is None
            or self.focus_uv_head is None
        ):
            raise RuntimeError("Focus-UV prediction is not enabled.")
        coordinates = self._skill_code_coordinates(skill_code, memory.dtype)
        skill_condition = self.focus_uv_skill_projection(coordinates)
        uv_hidden = self.focus_uv_reader(
            memory, key_ignore, probe_condition=skill_condition
        )
        return self.focus_uv_head(uv_hidden)

    def predict_focus_uv(
        self,
        images: list[Tensor],
        language_tokens: Tensor,
        language_mask: Tensor,
        skill_code: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Predict skill and UV; inference conditions UV on the hard predicted skill."""
        memory, key_ignore = self.reader_memory(images, language_tokens, language_mask)
        skill_hidden = self.reader(memory, key_ignore)
        if skill_code is None:
            skill_code = self.head.decode(skill_hidden)
        skill_code = skill_code.reshape(-1).long()
        uv = self.predict_focus_uv_from_memory(memory, key_ignore, skill_code)
        return skill_code, uv

    def predict_continuous(
        self,
        images: list[Tensor],
        language_tokens: Tensor,
        language_mask: Tensor,
    ) -> Tensor:
        """Predict differentiable normalized FSQ coordinates in ``[-1, 1]``."""
        return self.head.predict_continuous(
            self.reader_hidden(images, language_tokens, language_mask)
        )

    def predict_continuous_from_hidden_stack(
        self,
        layer_stack: Tensor,
        key_ignore: Tensor,
        *,
        language_token_count: int,
    ) -> Tensor:
        """Read continuous skills from one already-computed base-VLM stack.

        Stage 2 uses this to share the expensive frozen VLM forward between
        its self-routed skill predictor and the noise reader.
        """
        if layer_stack.ndim != 4:
            raise ValueError(
                "VLM hidden stack must have shape [B,L,N,D], got "
                f"{tuple(layer_stack.shape)}."
            )
        if key_ignore.shape != layer_stack.shape[:1] + layer_stack.shape[2:3]:
            raise ValueError(
                "VLM key mask must have shape [B,N], got "
                f"{tuple(key_ignore.shape)} for stack {tuple(layer_stack.shape)}."
            )
        language_token_count = int(language_token_count)
        if not 0 <= language_token_count <= layer_stack.shape[2]:
            raise ValueError(
                "language_token_count must fit the VLM token sequence, got "
                f"{language_token_count} for {layer_stack.shape[2]} tokens."
            )
        key_ignore = key_ignore.clone()
        image_token_count = layer_stack.shape[2] - language_token_count
        if not self.config.skill_predictor_attend_image:
            key_ignore[:, :image_token_count] = True
        if not self.config.skill_predictor_attend_language:
            key_ignore[:, image_token_count:] = True
        if self.config.skill_predictor_all_layers:
            batch, layers, tokens, width = layer_stack.shape
            hidden = layer_stack.reshape(batch, layers * tokens, width)
            mask = key_ignore.repeat(1, layers)
        else:
            hidden = layer_stack[:, -1]
            mask = key_ignore
        reader_hidden = self.reader(hidden, mask)
        return self.head.predict_continuous(reader_hidden)

    @torch.no_grad()
    def predict(
        self,
        images: list[Tensor],
        language_tokens: Tensor,
        language_mask: Tensor,
    ) -> Tensor:
        """Predict one FSQ skill code from a runtime skill-start observation."""
        coordinates = self.predict_continuous(
            images, language_tokens, language_mask
        )
        code, _, _ = self.head.quantize_coordinates(coordinates)
        return code
