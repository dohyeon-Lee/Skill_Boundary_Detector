"""SkillExpert — Stage-1 standalone action expert (no VLM, no language).

Flow-matching action chunk predictor conditioned on:
  - current 3rd-person + wrist images (trainable DINOv3, shared weights),
  - robot state,
  - GT FSQ skill code (nn.Embedding lookup, like a language token).

All tokens self-attend; only the action-token hidden states are decoded into actions.
Stage 2 (`skill_vla`) adds the VLM and can init its action expert from a Stage-1 checkpoint.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModel
from transformers.models.auto import CONFIG_MAPPING

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.modeling_pi05 import (
    OPENPI_ATTENTION_MASK_VALUE,
    create_sinusoidal_pos_embedding,
    get_gemma_config,
    make_att_2d_masks,
    pad_vector,
    sample_beta,
)
from lerobot.policies.pi_gemma import PiGemmaForCausalLM
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_skill_expert import SkillExpertConfig

log = logging.getLogger(__name__)


def _build_siglip_vision_tower(image_size: int):
    """Standalone SigLIP vision tower matching the pi05 PaliGemma vision_tower (So400m, patch 14)
    so the pi05 checkpoint's `...vision_tower.vision_model.*` weights load 1:1 (verified)."""
    from transformers import SiglipVisionModel  # noqa: PLC0415

    vlm_cfg = CONFIG_MAPPING["paligemma"]()
    vc = vlm_cfg.vision_config
    vc.image_size = image_size
    vc.intermediate_size = 4304
    vc.projection_dim = 2048
    vc.projector_hidden_act = "gelu_fast"
    return SiglipVisionModel(vc)


class SkillExpertPytorch(nn.Module):
    """The Stage-1 action expert network (see module docstring)."""

    def __init__(self, config: SkillExpertConfig):
        super().__init__()
        self.config = config
        expert_cfg = get_gemma_config(config.action_expert_variant)
        self.width = expert_cfg.width

        # ── Vision encoder, shared across the two cameras (DINOv3 or SigLIP) ──
        self.vision_backbone = config.vision_backbone
        self.dino = None
        self.siglip = None
        self.n_register = 0
        if config.vision_backbone == "dino":
            self.dino = AutoModel.from_pretrained(config.dino_model_path)
            vis_dim = int(self.dino.config.hidden_size)
            self.n_register = int(getattr(self.dino.config, "num_register_tokens", 0))
            self.vision_image_size = config.dino_image_size
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet
            if config.freeze_dino:
                for p in self.dino.parameters():
                    p.requires_grad_(False)
        elif config.vision_backbone == "siglip":
            # Warm-started from pi05's vision_tower in from_pretrained (params separate from the VLM).
            self.siglip = _build_siglip_vision_tower(config.siglip_image_size)
            vis_dim = int(self.siglip.config.hidden_size)
            self.vision_image_size = config.siglip_image_size
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]  # SigLIP: [0,1] → [-1,1]
            if config.freeze_siglip:
                for p in self.siglip.parameters():
                    p.requires_grad_(False)
        else:
            raise ValueError(f"vision_backbone must be 'dino' or 'siglip', got {config.vision_backbone!r}")
        self.register_buffer("_img_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_img_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

        # ── Token projections (all → expert width) ──
        self.image_proj = nn.Linear(vis_dim, self.width)                # image token → expert token
        self.state_proj = nn.Linear(config.max_state_dim, self.width)   # continuous state → 1 token
        self.skill_emb = nn.Embedding(config.skill_vocab_size, self.width)  # discrete FSQ code → 1 token

        # ── Flow-matching action head (mirrors PI05) ──
        self.action_in_proj = nn.Linear(config.max_action_dim, self.width)
        self.action_out_proj = nn.Linear(self.width, config.max_action_dim)
        self.time_mlp_in = nn.Linear(self.width, self.width)
        self.time_mlp_out = nn.Linear(self.width, self.width)

        # ── Gemma expert transformer (AdaRMS conditioned on the flow timestep) ──
        expert_hf = CONFIG_MAPPING["gemma"](
            head_dim=expert_cfg.head_dim,
            hidden_size=expert_cfg.width,
            intermediate_size=expert_cfg.mlp_dim,
            num_attention_heads=expert_cfg.num_heads,
            num_hidden_layers=expert_cfg.depth,
            num_key_value_heads=expert_cfg.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=True,
            adarms_cond_dim=expert_cfg.width,
        )
        self.gemma_expert = PiGemmaForCausalLM(config=expert_hf)
        self.gemma_expert.model.embed_tokens = None  # tokens come from our projections, not a vocab table
        self.gemma_expert.lm_head = None             # unused (we decode actions via action_out_proj)
        self.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001  (custom 4D mask)

    @property
    def _wdtype(self) -> torch.dtype:
        """Working dtype of the expert stream (drives the cast at every token boundary)."""
        return self.action_in_proj.weight.dtype

    def gradient_checkpointing_enable(self) -> None:
        if hasattr(self.gemma_expert, "gradient_checkpointing_enable"):
            self.gemma_expert.gradient_checkpointing_enable()
        if self.vision_backbone == "dino" and not self.config.freeze_dino \
                and hasattr(self.dino, "gradient_checkpointing_enable"):
            self.dino.gradient_checkpointing_enable()
        elif self.vision_backbone == "siglip" and not self.config.freeze_siglip \
                and hasattr(self.siglip, "gradient_checkpointing_enable"):
            self.siglip.gradient_checkpointing_enable()

    # ── Flow-matching samplers (copied from PI05Pytorch) ──
    def sample_noise(self, shape, device) -> Tensor:
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device) -> Tensor:
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    # ── Tokenization ──
    def _image_features(self, image: Tensor) -> Tensor:
        """image (B, C, H, W) in [0, 1] → (B, n_tokens, vis_dim) image tokens.
        DINO: CLS + patches (registers dropped). SigLIP: 256 patch tokens (no CLS)."""
        x = image.to(dtype=torch.float32)
        size = (self.vision_image_size, self.vision_image_size)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        x = (x - self._img_mean.float()) / self._img_std.float()
        if self.vision_backbone == "dino":
            x = x.to(dtype=next(self.dino.parameters()).dtype)
            out = self.dino(x).last_hidden_state        # (B, 1 + n_register + num_patches, dino_dim)
            cls = out[:, :1, :]
            patches = out[:, 1 + self.n_register :, :]  # drop CLS + register tokens, keep patches
            return torch.cat([cls, patches], dim=1)
        x = x.to(dtype=next(self.siglip.parameters()).dtype)
        return self.siglip(pixel_values=x).last_hidden_state  # (B, 256, siglip_dim)

    def _cond_tokens(self, images: list[Tensor], state: Tensor, skill_code: Tensor) -> Tensor:
        """Conditioning tokens [img1 tokens, img2 tokens, state, skill] → (B, M, width)."""
        tokens = [self.image_proj(self._image_features(image).to(self._wdtype)) for image in images]
        state = pad_vector(state.to(dtype=torch.float32), self.config.max_state_dim)
        tokens.append(self.state_proj(state.to(self._wdtype)).unsqueeze(1))
        tokens.append(self.skill_emb(skill_code.view(-1).long()).unsqueeze(1))
        return torch.cat(tokens, dim=1)

    def _time_cond(self, timestep: Tensor) -> Tensor:
        t = create_sinusoidal_pos_embedding(
            timestep, self.width, self.config.min_period, self.config.max_period, device=timestep.device
        ).to(self._wdtype)
        t = F.silu(self.time_mlp_in(t))
        t = F.silu(self.time_mlp_out(t))
        return t

    def _run_expert(self, cond_tokens: Tensor, x_t: Tensor, timestep: Tensor) -> Tensor:
        """Full self-attention over [cond tokens, noisy action tokens]; decode action velocity."""
        action_tokens = self.action_in_proj(x_t.to(self._wdtype))     # (B, chunk, width)
        tokens = torch.cat([cond_tokens, action_tokens], dim=1)       # (B, M + chunk, width)
        bsize, seq_len = tokens.shape[:2]
        device = tokens.device

        pad_masks = torch.ones(bsize, seq_len, dtype=torch.bool, device=device)
        att_masks = torch.zeros(bsize, seq_len, dtype=torch.bool, device=device)  # all 0 → full self-attn
        att_2d_4d = make_att_2d_masks(pad_masks, att_masks)[:, None, :, :]
        att_2d_4d = torch.where(att_2d_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        out = self.gemma_expert.model.forward(
            inputs_embeds=tokens,
            attention_mask=att_2d_4d,
            position_ids=position_ids,
            adarms_cond=self._time_cond(timestep),
            use_cache=False,  # full self-attn, no KV cache (silences the gradient-checkpointing warning)
        ).last_hidden_state
        action_hidden = out[:, -self.config.chunk_size :]
        return self.action_out_proj(action_hidden.to(self._wdtype)).float()

    # ── Training / inference ──
    def forward(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        state: Tensor,
        skill_code: Tensor,
        actions: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        """Flow-matching loss. Returns per-(b, t, dim) MSE: (B, chunk_size, max_action_dim)."""
        cond = self._cond_tokens(images, state, skill_code)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        source = self.sample_noise(actions.shape, actions.device) if noise is None else noise
        source = source.to(dtype=actions.dtype)
        time_exp = time[:, None, None]
        x_t = time_exp * source + (1 - time_exp) * actions
        u_t = source - actions
        v_t = self._run_expert(cond, x_t, time)
        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        state: Tensor,
        skill_code: Tensor,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        bsize, device = state.shape[0], state.device
        if noise is None:
            noise = self.sample_noise((bsize, self.config.chunk_size, self.config.max_action_dim), device)
        cond = self._cond_tokens(images, state, skill_code)  # constant across denoising steps
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            t = torch.full((bsize,), 1.0 + step * dt, dtype=torch.float32, device=device)
            x_t = x_t + dt * self._run_expert(cond, x_t, t)
        return x_t


# ── Checkpoint loading helpers (Stage-1 init from a PI05 checkpoint, or resume) ──────────

def _map_pi05_key(key: str, vision_backbone: str = "dino") -> str | None:
    """Map a PI05 checkpoint key to the SkillExpert model key, or None to drop it.

    Always keeps the action expert: the Gemma transformer (incl. AdaRMS norms), the action
    in/out projections, and the time MLP. When vision_backbone="siglip", ALSO warm-start the
    expert's vision tower from PI05's `vision_tower` (robot-adapted prior). Everything else
    (PaliGemma LLM, multi_modal_projector, expert lm_head, discretized-state path) is dropped.
    """
    if key.startswith("paligemma_with_expert.gemma_expert."):
        rest = key[len("paligemma_with_expert.gemma_expert.") :]
        if rest.startswith("lm_head"):
            return None
        return f"model.gemma_expert.{rest}"
    # SigLIP backbone: PI05 vision_tower → expert's own SigLIP (keys are vision_model.*).
    if vision_backbone == "siglip":
        vt = "paligemma_with_expert.paligemma.model.vision_tower."
        if key.startswith(vt):
            return f"model.siglip.{key[len(vt):]}"
    if key.startswith("paligemma_with_expert."):
        return None
    for proj in ("action_in_proj.", "action_out_proj.", "time_mlp_in.", "time_mlp_out."):
        if key.startswith(proj):
            return f"model.{key}"
    # openpi legacy naming for the time MLP
    if key.startswith("action_time_mlp_in."):
        return "model.time_mlp_in." + key[len("action_time_mlp_in.") :]
    if key.startswith("action_time_mlp_out."):
        return "model.time_mlp_out." + key[len("action_time_mlp_out.") :]
    return None


def _build_state_dict(raw: dict, vision_backbone: str = "dino") -> dict:
    """Remap a raw checkpoint to SkillExpert keys. Handles a PI05 checkpoint (partial init)
    and a SkillExpert checkpoint (resume), keyed by whether PI05 prefixes are present."""
    is_pi05 = any("paligemma_with_expert" in k for k in raw)
    out = {}
    for k, v in raw.items():
        nk = _map_pi05_key(k, vision_backbone) if is_pi05 else (k if k.startswith("model.") else f"model.{k}")
        if nk is not None:
            out[nk] = v
    return out


def _load_raw_state_dict(path: str | Path, kwargs: dict) -> dict | None:
    """Load model.safetensors from a local file/dir or a HF repo id."""
    from safetensors.torch import load_file

    p = Path(path)
    if p.is_file():
        return load_file(str(p))
    if (p / "model.safetensors").is_file():
        return load_file(str(p / "model.safetensors"))
    try:
        from transformers.utils import cached_file

        resolved = cached_file(
            str(path),
            "model.safetensors",
            cache_dir=kwargs.get("cache_dir"),
            token=kwargs.get("token"),
            revision=kwargs.get("revision"),
            local_files_only=kwargs.get("local_files_only", False),
        )
        return load_file(resolved)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not resolve weights at %s: %s", path, exc)
        return None


class SkillExpertPolicy(PreTrainedPolicy):
    """Stage-1 action-expert policy (no VLM).

    Trains the SkillExpert by flow matching on (image, skill, state, action). Initializing
    from a PI05 checkpoint (e.g. lerobot/pi05_base) reuses the action expert's motion prior;
    the DINO encoder and the image/state/skill projections start fresh. A Stage-1 checkpoint
    can later seed the Stage-2 `skill_vla` action expert.
    """

    config_class = SkillExpertConfig
    name = "skill_expert"

    def __init__(self, config: SkillExpertConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = SkillExpertPytorch(config)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(device=config.device, dtype=self._torch_dtype())
        self.reset()

    def _torch_dtype(self) -> torch.dtype:
        return torch.bfloat16 if str(self.config.dtype) == "bfloat16" else torch.float32

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def get_optim_params(self):
        # Optional separate LR for the vision encoder (dino_lr / siglip_lr) vs everything else.
        if self.model.vision_backbone == "dino":
            vis_module, vis_lr = self.model.dino, self.config.dino_lr
        else:
            vis_module, vis_lr = self.model.siglip, self.config.siglip_lr
        if vis_lr is None:
            return self.parameters()
        vis_ids = {id(p) for p in vis_module.parameters()}
        vis = [p for p in vis_module.parameters() if p.requires_grad]
        others = [p for p in self.parameters() if id(p) not in vis_ids and p.requires_grad]
        return [{"params": others}, {"params": vis, "lr": vis_lr}]

    # ── batch → model inputs ──
    def _collect_images(self, batch: dict) -> tuple[list[Tensor], list[Tensor]]:
        """Gather present camera images as [0, 1] float tensors. The vision encoder's own
        normalization (ImageNet for DINO, [-1,1] for SigLIP) is applied inside the model."""
        device = next(self.parameters()).device
        images, masks = [], []
        present = [k for k in self.config.image_features if k in batch]
        if not present:
            raise ValueError(f"No image features in batch. Expected one of {list(self.config.image_features)}.")
        for key in present:
            img = batch[key].to(device=device)
            if img.dtype != torch.float32:
                img = img.float()
            if img.ndim == 4 and img.shape[1] != 3 and img.shape[-1] == 3:
                img = img.permute(0, 3, 1, 2)  # channels-last → channels-first
            images.append(img)
            masks.append(torch.ones(img.shape[0], dtype=torch.bool, device=device))
        return images, masks

    def _skill_code(self, batch: dict) -> Tensor:
        """Current GT FSQ skill code = skill_sequence[skill_index]."""
        seq = batch["skill_sequence"].long()
        idx = batch["skill_index"].long().view(-1, 1).clamp(0, seq.shape[1] - 1)
        code = seq.gather(1, idx).squeeze(1)
        return code.clamp(0, self.config.skill_vocab_size - 1)

    def forward(self, batch: dict, reduction: str = "mean"):
        images, img_masks = self._collect_images(batch)
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        losses = self.model.forward(images, img_masks, batch[OBS_STATE], self._skill_code(batch), actions)
        real_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :real_dim]
        loss_dict = {"loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist()}
        if reduction == "none":
            per_sample = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample.mean().detach().item()
            return per_sample, loss_dict
        loss = losses.mean()
        loss_dict["loss"] = loss.detach().item()
        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, **kwargs) -> Tensor:
        self.eval()
        images, img_masks = self._collect_images(batch)
        actions = self.model.sample_actions(images, img_masks, batch[OBS_STATE], self._skill_code(batch), **kwargs)
        real_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :real_dim]

    @torch.no_grad()
    def select_action(self, batch: dict, **kwargs) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, config=None, strict: bool = False, **kwargs):
        """Build the model and partially load weights. A PI05 checkpoint seeds only the action
        expert (motion prior); a SkillExpert checkpoint resumes fully. DINO/skill/state/image
        projections that are absent stay freshly initialized (reported as missing keys)."""
        if config is None:
            config = PreTrainedConfig.from_pretrained(pretrained_name_or_path, **kwargs)
        model = cls(config, **kwargs)
        raw = _load_raw_state_dict(pretrained_name_or_path, kwargs)
        if raw is None:
            log.warning("SkillExpert: no weights loaded, using fresh init.")
            return model
        state_dict = {k: v.to(model._torch_dtype()) for k, v in _build_state_dict(raw, config.vision_backbone).items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        log.info(
            "SkillExpert weights: %d mapped & loaded, %d missing (fresh init), %d unexpected.",
            len(state_dict), len(missing), len(unexpected),
        )
        return model
