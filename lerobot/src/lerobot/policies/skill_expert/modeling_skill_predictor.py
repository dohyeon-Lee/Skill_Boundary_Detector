"""Frozen-pi0.5-VLM skill predictor used as a Stage-1 auxiliary task."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

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
    """Read skill-start image/language tokens without updating the pi0.5 VLM.

    This is the renewed Stage-0 predictor contract: PaliGemma is a frozen feature
    producer and only the standalone joint-KV reader and FSQ regression head train.
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

        # The main Stage-1 optimizer must never register predictor parameters.
        self.vlm.requires_grad_(False)
        self.reader.requires_grad_(False)
        self.head.requires_grad_(False)
        self.vlm.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.vlm.eval()
        return self

    def auxiliary_parameters(self) -> list[nn.Parameter]:
        return [*self.reader.parameters(), *self.head.parameters()]

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
        """Return the frozen VLM's final joint image/language token sequence.

        Stage 2 uses this sequence directly as the shared cross-attention memory
        for every likelihood block.  The second return value follows PyTorch's
        key-padding convention (``True`` means ignore).
        """
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
        # Keep the complete VLM graph out of the auxiliary backward pass.
        with torch.no_grad():
            prefix, valid, key_ignore = self._embed_prefix(
                images, language_tokens, language_mask
            )
            hidden, layer_stack = self._encode_prefix(prefix, valid)
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
