"""Stage3-A-matched pi0.5-VLM skill predictor for Stage 1."""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
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
    """Read skill-start image/language tokens with an optional skill-only LoRA.

    The pi0.5 VLM base and vision tower always stay frozen. In the Stage3-A
    configuration, Q/K/V/O skill adapters across all 18 language layers train
    together with the standalone joint-KV reader and FSQ regression head.
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

        self.lora_layer_count = 0
        if config.skill_predictor_lora:
            target_names = target_names_from_spec(
                config.skill_predictor_lora_targets
            )
            self.lora_layer_count = inject_named_lora(
                self.vlm.language_model,
                target_names,
                "skill",
                config.skill_predictor_lora_rank,
                config.skill_predictor_lora_alpha,
                config.skill_predictor_lora_dropout,
            )
            if self.lora_layer_count == 0:
                raise RuntimeError(
                    "Skill predictor LoRA did not match any VLM projection; "
                    f"targets={config.skill_predictor_lora_targets!r}."
                )
            qkvo = {"q_proj", "k_proj", "v_proj", "o_proj"}
            if target_names == qkvo:
                expected = 4 * int(
                    self.vlm.language_model.config.num_hidden_layers
                )
                if self.lora_layer_count != expected:
                    raise RuntimeError(
                        "Skill predictor Q/K/V/O LoRA must cover every VLM layer; "
                        f"wrapped={self.lora_layer_count}, expected={expected}."
                    )

        # The main Stage-1 optimizer must never register predictor parameters.
        self.vlm.requires_grad_(False)
        self.reader.requires_grad_(False)
        self.head.requires_grad_(False)
        self.vlm.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.vlm.eval()
        # Only the language-model adapter needs training behavior. The frozen
        # vision tower remains deterministic, while this also enables the VLM's
        # gradient-checkpointing path and LoRA dropout when configured.
        if self._lora_attached_to_loss:
            self.vlm.language_model.train(mode)
        return self

    @property
    def _lora_attached_to_loss(self) -> bool:
        return bool(
            self.config.skill_predictor_lora
            and not self.config.skill_predictor_detach_vlm
        )

    def reader_head_parameters(self) -> list[nn.Parameter]:
        return [*self.reader.parameters(), *self.head.parameters()]

    def lora_parameters(self) -> list[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        for module in self.vlm.language_model.modules():
            if isinstance(module, NamedLoRALinear) and "skill" in module.adapters:
                parameters.extend(module.adapters["skill"].parameters())
        return parameters

    def auxiliary_parameters(self) -> list[nn.Parameter]:
        return [*self.reader_head_parameters(), *self.lora_parameters()]

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
        self._activate_skill_adapter()
        # Legacy checkpoints detach the complete VLM graph. Stage3-A keeps the
        # graph only as far as the skill LoRA; every base tensor is still frozen.
        context = nullcontext() if self._lora_attached_to_loss else torch.no_grad()
        with context:
            prefix, valid, key_ignore = self._embed_prefix(
                images, language_tokens, language_mask
            )
            hidden, layer_stack = self._encode_prefix(prefix, valid)
        if not self._lora_attached_to_loss:
            hidden = hidden.detach()
            layer_stack = None if layer_stack is None else layer_stack.detach()

        if layer_stack is not None:
            batch, layers, tokens, width = layer_stack.shape
            reader_hidden = self.reader(
                layer_stack.reshape(batch, layers * tokens, width),
                key_ignore.repeat(1, layers),
            )
        else:
            reader_hidden = self.reader(hidden, key_ignore)
        loss = self.head.loss(reader_hidden, skill_code)
        with torch.no_grad():
            accuracy = (self.head.decode(reader_hidden) == skill_code).float().mean().item()
        return loss, accuracy

    @torch.no_grad()
    def predict(
        self,
        images: list[Tensor],
        language_tokens: Tensor,
        language_mask: Tensor,
    ) -> Tensor:
        """Predict one FSQ skill code from a runtime skill-start observation."""
        self._activate_skill_adapter()
        prefix, valid, key_ignore = self._embed_prefix(
            images, language_tokens, language_mask
        )
        hidden, layer_stack = self._encode_prefix(prefix, valid)
        if layer_stack is not None:
            batch, layers, tokens, width = layer_stack.shape
            reader_hidden = self.reader(
                layer_stack.reshape(batch, layers * tokens, width),
                key_ignore.repeat(1, layers),
            )
        else:
            reader_hidden = self.reader(hidden, key_ignore)
        return self.head.decode(reader_hidden)
