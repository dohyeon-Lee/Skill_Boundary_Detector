"""Encode skillset trajectories with a trained FSQ checkpoint.

This produces the skill_latents*.npz file consumed by codebook_visualizer.py
and decoder_eval.py. The FSQ encoder uses spline control points, skill length,
and start/end DINO tokens (encoding uses only the pure-DINO encoder).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from FSQ import SplineFSQAE, SplineFSQAEConfig
from train_FSQ import _compute_skill_orders, load_skill_files


@dataclass
class Args:
    model_path: str
    """FSQ checkpoint path: FSQ.pt or FSQ_epochXXXX.pt."""

    skills_dir: str
    """Directory containing per-skill npz files."""

    output_path: str
    """Output skill_latents npz path."""

    eef_dims: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    gripper_action_dim: int = -1
    device: str = "cuda"


def load_model(model_path: Path, device: str) -> SplineFSQAE:
    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    cfg: SplineFSQAEConfig = ckpt["cfg"]
    model = SplineFSQAE(
        action_dim=cfg.action_dim,
        state_dim=cfg.state_dim,
        n_control=cfg.n_control,
        spline_degree=cfg.spline_degree,
        hidden_dim=cfg.hidden_dim,
        fsq_levels=cfg.fsq_levels,
        num_layers=cfg.num_layers,
        dropout=0.0,
        feat_dim=cfg.feat_dim,
        n_tokens=cfg.n_tokens,
        image_model_name=getattr(cfg, "image_model_name", "/data2/dohyeon/SBD/models/dinov3-vits16"),
        image_size=getattr(cfg, "image_size", 224),
        patch_grid=getattr(cfg, "patch_grid", 8),
        n_patch_raw=getattr(cfg, "n_patch_raw", 196),
        image_token_dim=getattr(cfg, "image_token_dim", 128),
        image_encoder_layers=getattr(cfg, "image_encoder_layers", 1),
        image_encoder_heads=getattr(cfg, "image_encoder_heads", 4),
        chunk_size=cfg.chunk_size,
        max_length=cfg.max_length,
        action_min=cfg.action_min,
        action_max=cfg.action_max,
        delta_min=cfg.delta_min,
        delta_max=cfg.delta_max,
        terminator_use_wrist=getattr(cfg, "terminator_use_wrist", False),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


def main(args: Args) -> None:
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    skills_dir = Path(args.skills_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    segments, _, _, metadata = load_skill_files(
        skills_dir,
        eef_dims=args.eef_dims,
        gripper_action_dim=args.gripper_action_dim,
    )

    model = load_model(Path(args.model_path), device)
    print(f"[FSQ encode] model={args.model_path}")
    print(f"[FSQ encode] device={device} skills={len(segments)}")

    latents = []
    tokens = []
    for seg in tqdm(segments, desc="Encoding FSQ skills"):  # action-only encoder: no images needed
        latents.append(model.encode_numpy(seg, device=device))
        tokens.append(model.encode_index(seg, device=device))

    save_dict: dict[str, np.ndarray] = {
        "latents": np.stack(latents).astype(np.float32),
        "tokens": np.array(tokens, dtype=np.int32),
        "skill_order": np.array(_compute_skill_orders(metadata), dtype=np.float32),
    }
    for key in ("episode_id", "task_id", "skill_index", "frame_start", "frame_end", "length"):
        save_dict[key] = np.array([m[key] for m in metadata])
    np.savez(str(output_path), **save_dict)
    print(f"[FSQ encode] saved -> {output_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
