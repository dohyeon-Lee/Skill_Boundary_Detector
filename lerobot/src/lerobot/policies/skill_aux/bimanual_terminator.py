"""ABC extension of the existing fusion Terminator: three cameras and two UV goals."""

import sys
from pathlib import Path

import torch
from torch import nn

_FSQ_SRC = Path(__file__).resolve().parents[4] / "skillVLA/libero/configs/train_skills/FSQ/src"
if str(_FSQ_SRC) not in sys.path:
    sys.path.insert(0, str(_FSQ_SRC))
from FSQ import FSQQueryTerminator, _checkpoint_config, _new_fsq_terminator


class BimanualTerminator(FSQQueryTerminator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.arch != "fusion" or self.camera_mode != "both":
            raise ValueError("ABC Terminator requires fusion and all three cameras.")
        width = self.third_type_embedding.shape[-1]
        self.right_wrist_type_embedding = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.normal_(self.right_wrist_type_embedding, std=0.02)
        if self.goal_uv:
            self.goal_proj[0] = nn.Linear(4, width)

    def _prepare_image_tokens(self, third, wrist):
        if wrist is None or wrist.ndim != 5 or wrist.shape[1] != 2:
            raise ValueError("ABC wrist images must be [B,2,C,H,W], left then right.")
        inputs = [self._preprocess_image(image) for image in (third, wrist[:, 0], wrist[:, 1])]
        if len({image.shape[0] for image in inputs}) != 1:
            raise ValueError("ABC camera batch sizes differ.")
        features = self._encode_image_batch(torch.cat(inputs, dim=0))
        projected = self.image_proj(features)
        return torch.cat(projected.chunk(3, dim=0), dim=1)

    def _fusion_image_tokens(self, image_tokens, *, camera_layout):
        if image_tokens.shape[1] % 3:
            raise ValueError("ABC requires equal top/left/right image token counts.")
        parts = image_tokens.chunk(3, dim=1)
        embeddings = (self.third_type_embedding, self.wrist_type_embedding, self.right_wrist_type_embedding)
        return torch.cat([part + embedding.to(part.dtype) for part, embedding in zip(parts, embeddings)], dim=1)

    def _project_uv_goal(self, goal_uv):
        if goal_uv.shape[-1] != 4 or goal_uv.ndim != 2 or not torch.isfinite(goal_uv).all():
            raise ValueError("ABC goal_uv must be finite [B,4], ordered uL,vL,uR,vR.")
        return self.goal_proj(goal_uv.to(self._module_dtype(self.goal_proj)))


def build_bimanual_terminator(config):
    if config.terminator_fsq_warm_start:
        raise ValueError("ABC standalone Terminators initialize fresh, without LIBERO FSQ Terminator weights.")
    cfg = _checkpoint_config(torch.load(config.fsq_path, map_location="cpu", weights_only=False))
    if cfg.state_dim != 14:
        raise ValueError("ABC Terminator requires a 14D FSQ state contract.")
    cfg.terminator_context = config.terminator_context
    cfg.terminator_cameras = config.terminator_cameras
    cfg.terminator_arch = config.terminator_arch
    cfg.vision_backbone = config.terminator_vision_backbone
    cfg.freeze_vision_encoder = config.terminator_freeze_vision_encoder
    for field in ("goal_uv", "goal_xyz", "skill_skip", "start_proprio", "current_proprio", "proprio_conditioning"):
        setattr(cfg, "terminator_" + field, getattr(config, "terminator_" + field))
    return _new_fsq_terminator(
        BimanualTerminator, cfg,
        dino_model_path=config.terminator_dino_model_path,
        termination_only=config.terminator_termination_only,
    )


def project_bimanual_goal(xyz, camera, valid, *, image_size, noise_max=0.0):
    """Perturb in world metres, then project both arms through the same Top camera."""
    xyz = xyz.float()
    if noise_max:
        xyz = xyz + (torch.rand_like(xyz) * 2 - 1) * noise_max
    projected = torch.einsum("bij,bnj->bni", camera.float(), torch.cat((xyz, torch.ones_like(xyz[..., :1])), -1))
    depth = projected[..., 2]
    finite = valid.bool() & torch.isfinite(projected).all(-1) & (depth > 1e-8)
    pixels = projected[..., :2] / depth.clamp_min(1e-8)[..., None]
    width, height = image_size
    uv = 2 * pixels / xyz.new_tensor([width - 1, height - 1]) - 1
    return torch.where(finite[..., None], uv.clamp(-1, 1), torch.zeros_like(uv)).flatten(1)
