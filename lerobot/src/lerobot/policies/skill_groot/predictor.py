"""The existing mode2 skill/UV readers over the pretrained GR00T VLM."""
from __future__ import annotations

from contextlib import nullcontext
import torch
from transformers import AutoTokenizer

from lerobot.policies.skill_expert.modeling_skill_predictor import FrozenVLMSkillPredictor
from .backbone import GrootVLM
from .native import load_native


def predictor_vlm(config):
    model = load_native(config.groot_base_path)
    return GrootVLM(model.backbone, model.config.model_name, config.groot_base_path)  # DiT is not part of the Predictor.


class GrootSkillPredictor(FrozenVLMSkillPredictor):
    def __init__(self, config):
        if config.skill_predictor_lora:
            raise ValueError("GR00T Predictor currently supports full/frozen/vision-only tuning; disable pi0.5-specific LoRA")
        super().__init__(config)
        self.text_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, local_files_only=True)

    def gradient_checkpointing_enable(self, *, llm=True, vision=True):
        self.vlm.configure_checkpointing(llm=llm, vision=vision)
        self._gradient_checkpointing_enabled = True

    def vlm_parameters(self):
        # Qwen's merger belongs to the vision tower, unlike PaliGemma's
        # separate projector; never register its parameters twice.
        if self._train_vlm_base:
            return list(self.vlm.parameters())
        return list(self.vlm.vision_tower.parameters()) if self._train_vision_encoder else []

    def reader_memory(self, images, language_tokens, language_mask, *, top_crop=None):
        images = list(images)
        images[0] = self.prepare_top_image(images[0], top_crop=top_crop)
        # Tokenization stays a saved processor contract. Re-encode its task text
        # using native Qwen's processor; no PaliGemma model weights are involved.
        texts = [self.text_tokenizer.decode(tokens[mask.bool()].detach().cpu().tolist(), skip_special_tokens=True)
                 for tokens, mask in zip(language_tokens, language_mask, strict=True)]
        context = nullcontext() if self._vlm_attached_to_loss else torch.no_grad()
        with context:
            layers, valid, image, _ = self.vlm.encode(images, texts)
            hidden = [self.vlm.language_model.norm(value) for value in layers]
            if self.config.skill_predictor_all_layers:
                memory = torch.cat(hidden, dim=1)
            else:
                memory = hidden[-1]
        readable = (image if self.config.skill_predictor_attend_image else torch.zeros_like(image))
        if self.config.skill_predictor_attend_language:
            readable = readable | ~image
        keep = valid & readable
        if not bool(keep.any(-1).all()):
            raise ValueError("GR00T Predictor must read image and/or language tokens")
        key_ignore = ~keep
        if self.config.skill_predictor_all_layers:
            key_ignore = key_ignore.repeat(1, len(layers))
        return memory, key_ignore

    def _encode_last_hidden(self, images, language_tokens, language_mask):
        return self.reader_memory(images, language_tokens, language_mask)
