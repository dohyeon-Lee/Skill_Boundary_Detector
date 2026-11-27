"""Native causal Qwen3-VL forward, with read-only per-layer observation taps."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GrootVLM(nn.Module):
    def __init__(self, backbone, model_name="nvidia/Cosmos-Reason2-2B", base_path=None):
        super().__init__()
        self.backbone = backbone
        from pathlib import Path
        local = Path(base_path) / "processor" if base_path else None
        if local is not None and local.is_dir():
            model_name = str(local)
        # Match the N1.7 processor explicitly. AutoProcessor in Transformers 5
        # misidentifies Cosmos' legacy image-processor sidecar as a video config.
        from transformers import AutoTokenizer, Qwen2VLImageProcessor, Qwen3VLVideoProcessor, Qwen3VLProcessor
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if len(tokenizer) < 150000:
            raise FileNotFoundError(
                f"Incomplete GR00T tokenizer at {model_name}: download the Cosmos-Reason2-2B "
                "tokenizer/processor assets (weights are already in the GR00T base checkpoint)."
            )
        self.processor = Qwen3VLProcessor(
            image_processor=Qwen2VLImageProcessor.from_pretrained(model_name),
            video_processor=Qwen3VLVideoProcessor.from_pretrained(model_name),
            tokenizer=tokenizer, chat_template=tokenizer.chat_template,
        )
        self.processor.tokenizer.padding_side = "left"
        # Each observation is encoded in full; no generation/KV-cache state
        # is reused across training frames or different skill conditions.
        self.backbone.model.config.use_cache = False
        self.language_model.config.use_cache = False

    def configure_checkpointing(self, *, llm=False, vision=False):
        from lerobot.utils.gradient_checkpointing import checkpoint_indices
        for module, selection, name in ((self.language_model, llm, "layers"),
                                        (self.vision_tower, vision, "blocks")):
            blocks = getattr(module, name)
            selected = checkpoint_indices(selection, len(blocks))
            module.gradient_checkpointing_disable()
            if selected:
                module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                for index, block in enumerate(blocks):
                    block.gradient_checkpointing = index in selected
        self.backbone.model.config.use_cache = False
        self.language_model.config.use_cache = False

    @property
    def language_model(self):
        return self.backbone.language_model

    @property
    def vision_tower(self):
        return self.backbone.visual

    @property
    def multi_modal_projector(self):
        # Qwen's merger is part of its native vision tower.
        return self.backbone.visual.merger

    def encode(self, images, texts):
        from transformers.feature_extraction_utils import BatchFeature
        device = next(self.backbone.parameters()).device
        batch, cameras = images[0].shape[0], len(images)
        frames = [F.interpolate(image.float(), size=(224, 224), mode="bilinear", align_corners=False)
                  for image in images]
        prompts, pixels = [], []
        for row in range(batch):
            content = [{"type": "image"} for _ in frames] + [{"type": "text", "text": texts[row]}]
            prompts.append(self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=False))
            pixels.extend(image[row].detach().cpu() for image in frames)
        packed = self.processor(text=prompts, images=pixels, padding=True, return_tensors="pt",
                                do_resize=False, do_rescale=False)
        packed = packed.to(device)
        values = []
        def tap(_module, _args, output):
            values.append(output[0] if isinstance(output, (tuple, list)) else output)
        hooks = [layer.register_forward_hook(tap) for layer in self.language_model.layers]
        try:
            result = self.backbone(BatchFeature(dict(packed)))
        finally:
            for hook in hooks:
                hook.remove()
        valid, image_mask = result.backbone_attention_mask, result.image_mask
        camera_masks = []
        merged = int(getattr(self.processor.image_processor, "merge_size", 2)) ** 2
        grids = packed["image_grid_thw"].reshape(batch, cameras, 3)
        counts = grids.prod(-1) // merged
        if not bool((counts == counts[0, 0]).all()):
            raise ValueError("SkillVLA GR00T requires equal square camera grids within a batch")
        for camera in range(cameras):
            mask = torch.zeros_like(image_mask)
            for row in range(batch):
                slots = image_mask[row].nonzero().flatten()
                start = int(counts[row, :camera].sum())
                end = start + int(counts[row, camera])
                mask[row, slots[start:end]] = True
            camera_masks.append(mask)
        return values, valid, image_mask, camera_masks
