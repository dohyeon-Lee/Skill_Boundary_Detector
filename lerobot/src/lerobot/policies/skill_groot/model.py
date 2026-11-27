"""Two read-only recurrent bottlenecks over pretrained GR00T N1.7.

The native causal VLM, state/action encoders, DiT blocks and action decoder
retain their weights. Only the DiT conditioning memory and early skill/UV
conditioning change. There is no raw VLM -> DiT bypass.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .backbone import GrootVLM
from .native import load_native, native_modules
from lerobot.utils.gradient_checkpointing import checkpoint_indices


class Reader(nn.Module):
    def __init__(self, width, memory_width, heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.cross = nn.MultiheadAttention(width, heads, kdim=memory_width, vdim=memory_width, batch_first=True)
        self.self_norm = nn.LayerNorm(width)
        self.self_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 4), nn.SiLU(), nn.Linear(width * 4, width))

    def read(self, query, memory, keep):
        if not bool(keep.any(-1).all()):
            raise ValueError("Bottleneck memory mask must keep at least one token per row")
        value, _ = self.cross(self.norm(query), memory, memory, key_padding_mask=~keep, need_weights=False)
        return query + value

    def mix(self, latent):
        norm = self.self_norm(latent)
        value, _ = self.self_attn(norm, norm, norm, need_weights=False)
        latent = latent + value
        return latent + self.mlp(latent)


class Bottleneck(nn.Module):
    def __init__(self, depth, memory_width, width=512, separate=False):
        super().__init__()
        self.separate = separate
        self.queries = nn.Parameter(torch.randn(100, width) * .02)
        self.readers = nn.ModuleList([Reader(width, memory_width) for _ in range(depth)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, memory_width))

    def forward(self, layers, valid, image, cameras):
        z = self.queries[None].expand(layers[0].shape[0], -1, -1)
        for reader, memory in zip(self.readers, layers, strict=True):
            memory = memory.float()
            # Dedicated aux queries read Top / Wrist / all images directly.
            pieces = [reader.read(z[:, i:i + 1], memory, mask) for i, mask in
                      enumerate((cameras[0], cameras[1], image))]
            if self.separate:
                pieces += [reader.read(z[:, 3:50], memory, image),
                           reader.read(z[:, 50:], memory, valid & ~image)]
            else:
                pieces.append(reader.read(z[:, 3:], memory, valid))
            # Both representations exchange information, including auxiliary queries.
            z = reader.mix(torch.cat(pieces, dim=1))
        return z, self.output(z)


class DetachedTermination(nn.Module):
    def __init__(self, depth, width):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(8, 256) * .02)
        self.readers = nn.ModuleList([Reader(256, width) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 1))

    def forward(self, layers, valid):
        latent = self.queries[None].expand(layers[0].shape[0], -1, -1)
        for reader, memory in zip(self.readers, layers, strict=True):
            latent = reader.mix(reader.read(latent, memory.detach().float(), valid))
        return self.head(latent.mean(1)).squeeze(-1)


@dataclass
class CameraMemory:
    images: list
    offsets: list

    def __getitem__(self, rows):
        return CameraMemory([x[rows] for x in self.images], [x[rows] for x in self.offsets])

    def uv(self, uv, camera=0):
        # Native GR00T camera geometry: resize 256, crop 230, then VLM patchification.
        return 2 * (((uv.float() + 1) * .5 * 256 - self.offsets[camera]) / 230) - 1


def crop_cameras(images, training):
    cropped, offsets = [], []
    for image in images:
        image = F.interpolate(image.float(), size=(256, 256), mode="bilinear", align_corners=False)
        batch = image.shape[0]
        xy = (torch.randint(27, (batch, 2), device=image.device) if training
              else torch.full((batch, 2), 13, device=image.device))
        cropped.append(torch.stack([image[row, :, y:y + 230, x:x + 230] for row, (x, y) in enumerate(xy.tolist())]))
        offsets.append(xy.float())
    return CameraMemory(cropped, offsets)


def zero_last(module):
    nn.init.zeros_(module[-1].weight)
    if module[-1].bias is not None:
        nn.init.zeros_(module[-1].bias)
    return module


class GrootSkillModel(nn.Module):
    def __init__(self, config, *, native=None):
        super().__init__()
        self.config = config
        loaded = load_native(config.groot_base_path) if native is None else native
        self.vlm = GrootVLM(loaded.backbone, loaded.config.model_name, getattr(config, "groot_base_path", None))
        self.expert = loaded.action_head
        self.native_config = loaded.config
        self.depth = len(self.expert.model.transformer_blocks)
        if self.depth % 4:
            raise ValueError("GR00T Expert must split into two equal cross+self groups")
        self.split = self.depth // 2
        depth = len(self.vlm.language_model.layers)
        memory_width = int(loaded.config.backbone_embedding_dim)
        latent_width = int(config.visual_bottleneck_width)
        inner = self.expert.model.transformer_blocks[0].dim
        separate = config.skillvla_backbone == "groot_seperate"
        self.bottlenecks = nn.ModuleList([Bottleneck(depth, memory_width, latent_width, separate) for _ in range(2)])
        self.skill_projection = zero_last(nn.Sequential(nn.Linear(len(config.skill_fsq_levels), inner), nn.SiLU(), nn.Linear(inner, inner)))
        self.uv_projection = zero_last(nn.Sequential(nn.Linear(2, inner), nn.SiLU(), nn.Linear(inner, inner)))
        plan_dim = config.skill_plan_state_dim
        self.plan_input = nn.Linear(plan_dim, inner)
        self.plan_head = nn.Sequential(nn.LayerNorm(inner), nn.Linear(inner, latent_width), nn.SiLU(), nn.Linear(latent_width, plan_dim))
        from lerobot.policies.skill_expert.wrist_patch_alignment import WristPatchAlignmentHead
        self.patch_heads = nn.ModuleList([nn.ModuleList([WristPatchAlignmentHead(latent_width, memory_width) for _ in range(2)]) for _ in range(2)])
        self.xyz_heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(latent_width), nn.Linear(latent_width, 3)) for _ in range(2)])
        self.vlm_termination_bottleneck = DetachedTermination(depth, memory_width) if config.vlm_termination_enabled else None
        levels = torch.tensor(config.skill_fsq_levels)
        self.register_buffer("levels", levels, persistent=False)
        self.register_buffer("strides", torch.cat((torch.ones(1, dtype=torch.long), levels.cumprod(0)[:-1])), persistent=False)
        self._term_logits = self._camera_memory = None
        self._aux = None
        self._step = 0
        if config.newtask_ft_enabled and config.newtask_ft_freeze_expert:
            # The entire native action head is the pretrained motion prior,
            # including its VL norm/self-attention frontend. Frozen parameters
            # still propagate loss to the new bottlenecks and trainable VLM.
            for module in (self.expert, self.skill_projection, self.uv_projection,
                           self.plan_input, self.plan_head):
                if module is not None:
                    module.requires_grad_(False)
            self.expert.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.config.newtask_ft_enabled and self.config.newtask_ft_freeze_expert:
            # Match native GR00T's frozen-module mode: fixed DiT/VL frontend
            # weights should not acquire training-time attention/FFN dropout.
            self.expert.eval()
        return self

    def coordinates(self, code):
        digits = (code.reshape(-1, 1).long() // self.strides) % self.levels
        return (digits.float() - (self.levels - 1) / 2) / ((self.levels - 1) / 2).clamp_min(1)

    def prompts(self, code, uv):
        # Deliberately NO proprio in the GR00T VLM prefix.
        coordinates = self.coordinates(code).detach().cpu().tolist()
        return [f"Skill: {' '.join(f'{x:.6f}' for x in z)}; Gripper goal in top-view image: {u:.6f} {v:.6f};"
                for z, (u, v) in zip(coordinates, uv.detach().cpu().tolist(), strict=True)]

    def observe(self, images, code, uv):
        cache = crop_cameras(images, self.training)
        uv = cache.uv(uv)
        layers, valid, image, cameras = self.vlm.encode(cache.images, self.prompts(code, uv))
        banks, queries = [], []
        for bottle in self.bottlenecks:
            query, memory = bottle(layers, valid, image, cameras)
            queries.append(query)
            # Retain GR00T's learned VL norm and VL self-attention frontend.
            from transformers.feature_extraction_utils import BatchFeature
            banks.append(self.expert.process_backbone_output(BatchFeature({"backbone_features": memory.to(layers[-1].dtype)})).backbone_features)
        patch_features = [layers[-1][mask].reshape(layers[-1].shape[0], -1, layers[-1].shape[-1]).float() for mask in cameras]
        self._aux = [(self.patch_heads[i][0](query[:, :1], patch_features[0]),
                      self.patch_heads[i][1](query[:, 1:2], patch_features[1]), self.xyz_heads[i](query[:, 2]))
                     for i, query in enumerate(queries)]
        self._camera_memory = cache
        if self.vlm_termination_bottleneck is not None:
            self._term_logits = self.vlm_termination_bottleneck(layers, valid)
        return banks, uv

    def _run(self, hidden, time, code, uv, banks, *, planner=False):
        dit = self.expert.model
        temb = dit.timestep_encoder(time)
        early = temb + self.uv_projection(uv.float()).to(temb.dtype)
        skill = self.skill_projection(self.coordinates(code)).to(hidden.dtype)
        limit = self.split if planner else self.depth
        selection = getattr(self.config,
                            "skill_planner_checkpointing" if planner else "action_expert_checkpointing", None)
        selected = checkpoint_indices(self.config.gradient_checkpointing if selection is None else selection, limit)
        for index, block in enumerate(dit.transformer_blocks[:limit]):
            first, cross = index < self.split, index % 2 == 0
            memory = banks[0 if first else 1]
            mask = torch.ones(memory.shape[:2], device=memory.device, dtype=torch.bool)
            if cross and self.config.skillvla_backbone == "groot_seperate":
                # Keep native text / image alternation: non-image first.
                mask[:, :50 if index % 4 == 0 else 0] = False
                if index % 4 != 0:
                    mask[:, 50:] = False
            condition = early if first else temb
            def run(x, c, s, m, keep, *, layer=block, use_cross=cross, inject=first and not cross):
                norm = layer.norm1(x, c)
                if layer.pos_embed is not None:
                    norm = layer.pos_embed(norm)
                if inject:
                    # State token stays native; only action/fullskill tokens receive skill broadcast.
                    norm = torch.cat((norm[:, :1], norm[:, 1:] + s[:, None]), dim=1)
                update = layer.attn1(norm, encoder_hidden_states=m if use_cross else None,
                                     attention_mask=keep if use_cross else None)
                if layer.final_dropout is not None:
                    update = layer.final_dropout(update)
                x = x + update
                return x + layer.ff(layer.norm3(x))
            hidden = checkpoint(run, hidden, condition, skill, memory, mask, use_reentrant=False, preserve_rng_state=True) if index in selected and self.training else run(hidden, condition, skill, memory, mask)
        if planner:
            return self.plan_head(hidden[:, 1:].float())
        shift, scale = dit.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        return dit.proj_out_2(dit.norm_out(hidden) * (1 + scale[:, None]) + shift[:, None])

    def state_token(self, state):
        batch = state.shape[0]
        state = F.pad(state, (0, self.native_config.max_state_dim - state.shape[-1]))
        ids = torch.full((batch,), 2, dtype=torch.long, device=state.device)  # native libero_sim
        encoded = self.expert.state_encoder(state[:, None].to(next(self.expert.parameters()).dtype), ids)
        if self.training and self.expert.state_dropout_prob > 0:
            encoded = encoded * (torch.rand(batch, 1, 1, device=encoded.device) >= self.expert.state_dropout_prob)
        return encoded, ids

    def velocity(self, actions, state_token, ids, time, code, uv, banks):
        padded = F.pad(actions, (0, self.native_config.max_action_dim - actions.shape[-1]))
        tokens = self.expert.action_encoder(padded.to(state_token.dtype), time, ids)
        if self.native_config.add_pos_embed:
            tokens = tokens + self.expert.position_embedding(torch.arange(tokens.shape[1], device=tokens.device))[None]
        output = self._run(torch.cat((state_token, tokens), 1), time, code, uv, banks)
        return self.expert.action_decoder(output, ids)[:, 1:, :actions.shape[-1]].float()

    def flow_loss(self, target, state_token, ids, code, uv, banks, *, planner=False):
        _, flow = native_modules()
        noise = torch.randn_like(target)
        time = self.expert.sample_time(target.shape[0], device=target.device, dtype=target.dtype)
        noisy, velocity, model_time = flow.make_flow_matching_inputs(target, noise, time, convention=flow.FlowConvention.NOISE_AT_ZERO)
        buckets = (model_time * self.native_config.num_timestep_buckets).long()
        if planner:
            tokens = self.plan_input(noisy.float()).to(state_token.dtype)
            pred = self._run(torch.cat((state_token, tokens), 1), buckets, code, uv, banks, planner=True)
        else:
            pred = self.velocity(noisy, state_token, ids, buckets, code, uv, banks)
        return (pred.float() - velocity.float()).square()

    def take_vlm_termination_outputs(self):
        result = self._term_logits, self._camera_memory
        self._term_logits = None
        return result

    def vlm_termination_from_image_tokens(self, memory, state, code, goal):
        del state
        uv = memory.uv(goal[:, :2])
        with torch.no_grad():
            layers, valid, _, _ = self.vlm.encode(memory.images, self.prompts(code, uv))
        return self.vlm_termination_bottleneck(layers, valid)

    def set_training_step(self, step):
        self._step = step
