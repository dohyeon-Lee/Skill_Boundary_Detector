"""
학습된 VQ-VAE 체크포인트로 새 skill 파일들을 인코딩해서 latents .npz를 저장한다.

VQ-VAE 학습 없이 encoder + codebook만 사용하므로,
학습 때 쓰지 않은 데이터(e.g. libero_10)도 처리 가능.

출력 포맷은 train_vae.py 체크포인트 저장과 동일:
  keys: latents, tokens, skill_order, episode_id, task_id, skill_index, frame_start, frame_end, length

Usage:
    python examples/libero/encode_skills.py \
        --model_path   .../spline_vqae_epoch1500.pt \
        --skills_dir   .../libero_10_skillset \
        --output_path  .../spline_vqae_latents_epoch1500_libero10.npz
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path(__file__).parent))
from train_vae import _compute_skill_orders, load_skill_files


@dataclass
class Args:
    model_path: str
    """학습된 VQ-VAE 체크포인트 (.pt)"""
    skills_dir: str
    """build_skill_dataset.py 출력 디렉토리 (*_skillset)"""
    output_path: str
    """저장할 .npz 경로"""
    eef_dims: list[int] | None = None
    """EEF 차원 인덱스. None이면 [0,1,2,3,4,5]"""
    gripper_action_dim: int = -1
    """그리퍼 액션 차원 인덱스 (음수면 마지막 차원)"""
    zero_start_eef: bool = True
    """학습 때와 동일하게 맞춰야 함"""
    device: str = "cpu"


def load_model(model_path: str, device: str):
    from spline_vqae import SplineVQAE

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    epoch = ckpt.get("epoch", 0)

    model = SplineVQAE(
        action_dim=cfg.action_dim if hasattr(cfg, "action_dim") else None,
        state_dim=cfg.state_dim,
        n_control=cfg.n_control,
        spline_degree=cfg.spline_degree,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        num_embeddings=cfg.num_embeddings,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        encoder_type=cfg.encoder_type,
        decoder_rnn_type=cfg.decoder_rnn_type,
        commitment_cost=cfg.commitment_cost,
        max_length=cfg.max_length,
        action_min=cfg.action_min,
        action_max=cfg.action_max,
        delta_min=cfg.delta_min,
        delta_max=cfg.delta_max,
        image_model_name=getattr(cfg, "image_model_name", ""),
        image_feature_dim=getattr(cfg, "image_feature_dim", 384),
        image_size=getattr(cfg, "image_size", 224),
        use_images=getattr(cfg, "use_images", False),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[encode_skills] Loaded model from {model_path} (epoch {epoch})")
    return model, epoch


def main(args: Args) -> None:
    from spline_vqae import encode_skill_vectors, encode_skills

    eef_dims = args.eef_dims if args.eef_dims is not None else [0, 1, 2, 3, 4, 5]

    print(f"[encode_skills] Loading skill files from {args.skills_dir} ...")
    segments, _, _, metadata, skill_orders = load_skill_files(
        Path(args.skills_dir),
        eef_dims=eef_dims,
        gripper_action_dim=args.gripper_action_dim,
        zero_start_eef=args.zero_start_eef,
    )

    model, _ = load_model(args.model_path, args.device)

    print(f"[encode_skills] Encoding {len(segments)} skills ...")
    with torch.no_grad():
        tokens = encode_skills(model, segments, device=args.device, skill_orders=skill_orders)
        vectors = encode_skill_vectors(model, segments, device=args.device, skill_orders=skill_orders)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict: dict = {
        "latents": vectors,
        "tokens": tokens,
        "skill_order": np.array(skill_orders, dtype=np.float32),
    }
    for key in ("episode_id", "task_id", "skill_index", "frame_start", "frame_end", "length"):
        save_dict[key] = np.array([m[key] for m in metadata])

    np.savez(str(output_path), **save_dict)
    print(f"[encode_skills] Saved {len(tokens)} skill tokens → {output_path}")
    print(f"[encode_skills] Token stats — min: {tokens.min()}, max: {tokens.max()}, unique: {len(np.unique(tokens))}")


if __name__ == "__main__":
    main(tyro.cli(Args))
