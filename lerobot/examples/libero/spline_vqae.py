"""
Spline VQ-VAE — Vector-Quantized skill encoder with recurrent observation decoder.

Architecture
------------
  Encoder input : variable-length EEF trajectory (+ gripper), original length,
                  and BOS-aware skill order in the episode.
  Spline codec  : variable-length encoder trajectory → fixed-size control points.
  Encoder (MLP) : flatten(ctrl_pts) + length_norm + skill_order → z_e.
                  Optionally, GRU/BiLSTM encoders can consume the control-point
                  sequence directly.
  VQ layer      : z_e → nearest codebook entry z_q + integer token index.
  Image encoder : frozen DINO/DINOv3 feature per timestep.
  Decoder (MLP) : consumes fixed z_q, current image feature, current absolute
                  EEF state, and max-length-normalized skill-frame index.
  Decoder (RNN) : optional vanilla RNN decoder that consumes fixed z_q, current
                  image feature, and current absolute EEF state recurrently.
  Output        : next-state delta pose + gripper, plus an end-signal logit.
  Loss          : skill-balanced normalized-delta reconstruction loss +
                  end-signal BCE + VQ codebook/commitment loss.

encode_numpy        → integer token index
encode_vector_numpy → quantized codebook vector

Gripper dimension (last) uses degree-0 (nearest-neighbor) spline.
All other dims use the configured spline degree (default: cubic).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import make_interp_spline
from torch.utils.data import DataLoader, Dataset


# ── Spline codec (numpy) ──────────────────────────────────────────────────────

GRIPPER_DIM = -1  # last dim always uses degree-0 spline


def spline_encode(
    trajectory: np.ndarray,   # (T, action_dim)
    n_control: int,
    degree: int,
) -> tuple[np.ndarray, int]:
    """trajectory → control points (n_control, action_dim) + length T."""
    T, D = trajectory.shape
    t_orig = np.linspace(0.0, 1.0, T)
    t_ctrl = np.linspace(0.0, 1.0, n_control)
    ctrl_pts = np.zeros((n_control, D), dtype=np.float32)
    gripper_idx = (D + GRIPPER_DIM) % D
    for d in range(D):
        k = 0 if d == gripper_idx else degree
        k = min(k, T - 1)
        spl = make_interp_spline(t_orig, trajectory[:, d], k=k)
        ctrl_pts[:, d] = spl(t_ctrl)
    return ctrl_pts, T


# ── MLP block ─────────────────────────────────────────────────────────────────

class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Vector Quantizer ──────────────────────────────────────────────────────────

class VectorQuantizer(nn.Module):
    """Nearest-neighbour codebook with straight-through gradient estimator."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_embeddings  = num_embeddings
        self.embedding_dim   = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(
        self, z_e: torch.Tensor  # (B, D)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (z_q_st, z_q, indices, vq_loss).
            z_q_st  : (B, D)  straight-through (gradient flows to encoder)
            z_q     : (B, D)  quantized vector (detached from encoder graph)
            indices : (B,)    long — codebook token indices
            vq_loss : scalar
        """
        # squared distances to all K codebook entries: (B, K)
        d = (
            z_e.pow(2).sum(1, keepdim=True)
            - 2.0 * z_e @ self.embedding.weight.T
            + self.embedding.weight.pow(2).sum(1)
        )
        indices = d.argmin(1)           # (B,)
        z_q     = self.embedding(indices)  # (B, D)

        codebook_loss   = F.mse_loss(z_q,  z_e.detach())
        commitment_loss = F.mse_loss(z_e,  z_q.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        z_q_st = z_e + (z_q - z_e).detach()  # straight-through estimator
        return z_q_st, z_q, indices, vq_loss


# ── Observation-conditioned VQ-VAE ────────────────────────────────────────────

@dataclass
class SplineVQAEConfig:
    action_dim: int = 7
    state_dim: int = 6
    n_control: int = 30
    spline_degree: int = 3
    hidden_dim: int = 256
    latent_dim: int = 64
    num_embeddings: int = 512
    num_layers: int = 3
    dropout: float = 0.1
    encoder_type: str = "mlp"
    decoder_rnn_type: str = "mlp"
    commitment_cost: float = 0.25
    delta_loss_weight: float = 1.0
    end_loss_weight: float = 1.0
    end_pos_weight: float = 1.0
    end_threshold: float = 0.5
    max_length: float = 200.0
    lr: float = 3e-4
    batch_size: int = 64
    epochs: int = 100
    grad_clip: float = 1.0
    device: str = "cuda"
    val_split: float = 0.1
    log_every: int = 10
    save_path: str | None = None
    checkpoint_every: int = 0
    action_min: np.ndarray | None = None
    action_max: np.ndarray | None = None
    delta_min: np.ndarray | None = None
    delta_max: np.ndarray | None = None
    image_model_name: str = "facebook/dinov2-small"
    image_feature_dim: int = 384
    image_size: int = 224
    use_images: bool = True


class FrozenDINOv2Encoder(nn.Module):
    """Frozen visual encoder returning one feature vector per image.

    ViT-style DINO/DINOv3 backbones use the class token. ConvNeXt-style
    DINOv3 backbones are spatially pooled to one vector.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        image_size: int = 224,
        fallback_feature_dim: int = 384,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.fallback_feature_dim = fallback_feature_dim
        self.model = None
        try:
            from transformers import AutoImageProcessor, AutoModel

            self.model = AutoModel.from_pretrained(model_name)
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()

            hidden_size = getattr(self.model.config, "hidden_size", None)
            hidden_sizes = getattr(self.model.config, "hidden_sizes", None)
            if hidden_size is None and hidden_sizes:
                hidden_size = hidden_sizes[-1]
            self.out_dim = int(hidden_size or fallback_feature_dim)

            try:
                processor = AutoImageProcessor.from_pretrained(model_name)
                mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
                std = getattr(processor, "image_std", [0.229, 0.224, 0.225])
            except Exception:
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load frozen visual backbone '{model_name}'. "
                "Install transformers and make sure the pretrained weights are available."
            ) from exc

        mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def train(self, mode: bool = True):
        super().train(False)
        if self.model is not None:
            self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, T, C, H, W), float in [0, 1]."""
        B, T = images.shape[:2]
        x = images.reshape(B * T, *images.shape[2:])
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        out = self.model(pixel_values=x)

        last_hidden = getattr(out, "last_hidden_state", None)
        if last_hidden is not None:
            if last_hidden.ndim == 3:
                feat = last_hidden[:, 0]
            elif last_hidden.ndim == 4:
                feat = last_hidden.mean(dim=(-2, -1))
            else:
                raise ValueError(f"Unsupported last_hidden_state shape: {tuple(last_hidden.shape)}")
        else:
            pooler = getattr(out, "pooler_output", None)
            if pooler is None:
                raise ValueError("Visual backbone output has neither last_hidden_state nor pooler_output.")
            feat = pooler
        return feat.reshape(B, T, -1)


class SplineSkillDataset(Dataset):
    """Skill dataset for observation-conditioned VQ-VAE training.

    `segments` are encoder trajectories (absolute or zero-start EEF + gripper).
    `states` are decoder proprioceptive inputs. `deltas` are the decoder target.
    `images` may contain raw image clips (T,H,W,C) or precomputed DINO features
    (T,F). It may be omitted only when cfg.use_images is False.
    """

    def __init__(
        self,
        segments: list[np.ndarray],
        states: list[np.ndarray],
        deltas: list[np.ndarray],
        images: list[np.ndarray | None],
        skill_orders: list[float],
        n_control: int,
        spline_degree: int,
        action_min: np.ndarray,
        action_max: np.ndarray,
        delta_min: np.ndarray,
        delta_max: np.ndarray,
        max_length: float,
    ) -> None:
        self.max_length = max_length
        self.action_min = action_min.astype(np.float32)
        self.action_max = action_max.astype(np.float32)
        self.delta_min = delta_min.astype(np.float32)
        self.delta_max = delta_max.astype(np.float32)
        self.states = [s.astype(np.float32) for s in states]
        self.images = images
        self.skill_orders = [float(x) for x in skill_orders]
        self.ctrl_pts: list[np.ndarray] = []
        self.lengths: list[int] = []
        self.deltas: list[np.ndarray] = []
        self.end_signals: list[np.ndarray] = []

        for seg, delta in zip(segments, deltas):
            cp, T = spline_encode(seg.astype(np.float32), n_control, spline_degree)
            cp_norm = (cp - self.action_min) / (self.action_max - self.action_min + 1e-8) * 2.0 - 1.0
            end = np.zeros((T,), dtype=np.float32)
            end[-1] = 1.0
            self.ctrl_pts.append(cp_norm.astype(np.float32))
            self.lengths.append(T)
            self.deltas.append(delta.astype(np.float32))  # raw units; denorm in decode()
            self.end_signals.append(end)

    def __len__(self) -> int:
        return len(self.ctrl_pts)

    def __getitem__(self, idx: int):
        visual = self.images[idx]
        if visual is not None:
            visual_t = torch.from_numpy(visual.astype(np.float32))
            if visual_t.ndim == 4 and visual_t.shape[-1] in (1, 3):
                visual_t = (visual_t / 255.0).permute(0, 3, 1, 2)
            elif visual_t.ndim != 2:
                raise ValueError(f"Expected raw image clip (T,H,W,C) or feature clip (T,F), got {tuple(visual_t.shape)}")
        else:
            visual_t = None
        return {
            "ctrl": torch.from_numpy(self.ctrl_pts[idx]),
            "length": torch.tensor(self.lengths[idx], dtype=torch.long),
            "state": torch.from_numpy(self.states[idx]),
            "delta": torch.from_numpy(self.deltas[idx]),
            "end": torch.from_numpy(self.end_signals[idx]),
            "image": visual_t,
            "skill_order": torch.tensor([self.skill_orders[idx]], dtype=torch.float32),
        }


def collate_skill_batch(batch):
    lengths = torch.stack([b["length"] for b in batch])
    max_len = int(lengths.max().item())
    ctrl = torch.stack([b["ctrl"] for b in batch])
    skill_order = torch.stack([b["skill_order"] for b in batch])

    B = len(batch)
    state_dim = batch[0]["state"].shape[-1]
    delta_dim = batch[0]["delta"].shape[-1]
    states = torch.zeros(B, max_len, state_dim, dtype=torch.float32)
    deltas = torch.zeros(B, max_len, delta_dim, dtype=torch.float32)
    ends = torch.zeros(B, max_len, dtype=torch.float32)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    has_images = batch[0]["image"] is not None
    images = None
    if has_images:
        tail_shape = batch[0]["image"].shape[1:]
        images = torch.zeros(B, max_len, *tail_shape, dtype=torch.float32)

    for i, b in enumerate(batch):
        T = int(b["length"].item())
        states[i, :T] = b["state"]
        deltas[i, :T] = b["delta"]
        ends[i, :T] = b["end"]
        mask[i, :T] = True
        if has_images:
            images[i, :T] = b["image"]

    return ctrl, lengths, skill_order, states, deltas, ends, mask, images


class SplineVQAE(nn.Module):
    """Spline-tokenized VQ-VAE with frozen visual per-step decoder."""

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        n_control: int = 30,
        spline_degree: int = 3,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        num_embeddings: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        encoder_type: str = "mlp",
        decoder_rnn_type: str = "mlp",
        commitment_cost: float = 0.25,
        max_length: float = 200.0,
        action_min: np.ndarray | None = None,
        action_max: np.ndarray | None = None,
        delta_min: np.ndarray | None = None,
        delta_max: np.ndarray | None = None,
        image_model_name: str = "facebook/dinov2-small",
        image_feature_dim: int = 384,
        image_size: int = 224,
        use_images: bool = True,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.delta_dim = state_dim + 1
        self.n_control = n_control
        self.spline_degree = spline_degree
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.max_length = max_length
        self.use_images = use_images
        self.encoder_type = encoder_type.lower()
        self.decoder_rnn_type = decoder_rnn_type.lower()

        if self.encoder_type not in {"mlp", "gru", "bilstm"}:
            raise ValueError(f"Unsupported encoder_type={encoder_type!r}. Use 'mlp', 'gru', or 'bilstm'.")
        if self.decoder_rnn_type not in {"mlp", "rnn"}:
            raise ValueError(f"Unsupported decoder_rnn_type={decoder_rnn_type!r}. Use 'mlp' or 'rnn'.")

        if self.encoder_type == "mlp":
            enc_in = n_control * action_dim + 2  # control points + length + skill_order
            layers = [MLPBlock(enc_in, hidden_dim, dropout)]
            for _ in range(num_layers - 1):
                layers.append(MLPBlock(hidden_dim, hidden_dim, dropout))
            self.encoder = nn.Sequential(*layers)
        elif self.encoder_type == "gru":
            self.encoder = nn.GRU(
                input_size=action_dim + 2,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.encoder_out_dim = hidden_dim
        else:
            if hidden_dim % 2 != 0:
                raise ValueError("hidden_dim must be even when encoder_type='bilstm'.")
            self.encoder = nn.LSTM(
                input_size=action_dim + 2,
                hidden_size=hidden_dim // 2,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.encoder_out_dim = hidden_dim
        self.z_head = nn.Linear(hidden_dim, latent_dim)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)

        if use_images:
            self.image_encoder = FrozenDINOv2Encoder(image_model_name, image_size, image_feature_dim)
            self.image_proj = nn.Linear(self.image_encoder.out_dim, hidden_dim)
            self.image_feature_dim = self.image_encoder.out_dim
        else:
            self.image_encoder = None
            self.image_proj = nn.Linear(image_feature_dim, hidden_dim)
            self.image_feature_dim = image_feature_dim

        if self.decoder_rnn_type == "mlp":
            decoder_in = latent_dim + state_dim + hidden_dim + 1  # + normalized frame index within skill
            layers = [MLPBlock(decoder_in, hidden_dim, dropout)]
            for _ in range(num_layers - 1):
                layers.append(MLPBlock(hidden_dim, hidden_dim, dropout))
            self.decoder = nn.Sequential(*layers)
            self.rnn = None
            self.init_hidden = None
        else:
            decoder_in = latent_dim + state_dim + hidden_dim
            self.rnn = nn.RNNCell(decoder_in, hidden_dim)
            self.init_hidden = nn.Linear(latent_dim, hidden_dim)
            self.decoder = None
        self.delta_head = nn.Linear(hidden_dim, self.delta_dim)
        self.end_head = nn.Linear(hidden_dim, 1)

        _min = action_min if action_min is not None else np.zeros(action_dim)
        _max = action_max if action_max is not None else np.ones(action_dim)
        _dmin = delta_min if delta_min is not None else -np.ones(self.delta_dim)
        _dmax = delta_max if delta_max is not None else np.ones(self.delta_dim)
        self.register_buffer("action_min", torch.tensor(_min, dtype=torch.float32))
        self.register_buffer("action_max", torch.tensor(_max, dtype=torch.float32))
        self.register_buffer("delta_min", torch.tensor(_dmin, dtype=torch.float32))
        self.register_buffer("delta_max", torch.tensor(_dmax, dtype=torch.float32))

    def _norm_length(self, length: torch.Tensor) -> torch.Tensor:
        return length.float() / self.max_length

    def _np_norm_actions(self, ctrl_pts: np.ndarray) -> np.ndarray:
        lo = self.action_min.cpu().numpy()
        hi = self.action_max.cpu().numpy()
        return (ctrl_pts - lo) / (hi - lo + 1e-8) * 2.0 - 1.0

    def encode(
        self,
        ctrl_pts: torch.Tensor,
        lengths: torch.Tensor,
        skill_order: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = ctrl_pts.size(0)
        if skill_order is None:
            skill_order = torch.zeros(B, 1, device=ctrl_pts.device, dtype=ctrl_pts.dtype)
        skill_order = skill_order.to(device=ctrl_pts.device, dtype=ctrl_pts.dtype)

        if self.encoder_type == "mlp":
            flat = ctrl_pts.reshape(B, -1)
            l = self._norm_length(lengths.to(ctrl_pts.device)).unsqueeze(-1).to(ctrl_pts.dtype)
            x = torch.cat([flat, l, skill_order], dim=-1)
            h = self.encoder(x)
        else:
            l = self._norm_length(lengths.to(ctrl_pts.device)).view(B, 1, 1).to(ctrl_pts.dtype)
            l = l.expand(B, self.n_control, 1)
            order = skill_order.view(B, 1, 1).expand(B, self.n_control, 1)
            x = torch.cat([ctrl_pts, l, order], dim=-1)
            if self.encoder_type == "gru":
                _, h_n = self.encoder(x)
                h = h_n[-1]
            else:
                _, (h_n, _) = self.encoder(x)
                h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        return self.z_head(h)

    def decode(
        self,
        z: torch.Tensor,
        states: torch.Tensor,
        images: torch.Tensor | None = None,
        frame_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = states.shape
        if self.use_images:
            if images is None:
                raise ValueError("images must be provided when use_images=True")
            if images.ndim == 3:
                image_feat = images
            elif images.ndim == 5:
                with torch.no_grad():
                    image_feat = self.image_encoder(images)
            else:
                raise ValueError(f"Expected precomputed image features (B,T,F) or raw images (B,T,C,H,W), got {tuple(images.shape)}")
        else:
            image_feat = torch.zeros(B, T, self.image_feature_dim, device=states.device, dtype=states.dtype)
        image_feat = self.image_proj(image_feat)

        dmin = self.delta_min.to(z.device, z.dtype)
        dmax = self.delta_max.to(z.device, z.dtype)

        z_seq = z.unsqueeze(1).expand(B, T, -1)
        if frame_index is None:
            frame_index = torch.arange(T, device=states.device, dtype=states.dtype).view(1, T, 1)
            frame_index = frame_index.expand(B, T, 1) / float(self.max_length)
        else:
            frame_index = frame_index.to(device=states.device, dtype=states.dtype)
            if frame_index.ndim == 1:
                frame_index = frame_index.view(B, 1, 1)
            elif frame_index.ndim == 2:
                frame_index = frame_index.unsqueeze(-1)
            if frame_index.shape[1] == 1 and T > 1:
                frame_index = frame_index.expand(B, T, 1)

        if self.decoder_rnn_type == "mlp":
            x = torch.cat([z_seq, states, image_feat, frame_index], dim=-1)
            h = self.decoder(x.reshape(B * T, -1)).view(B, T, -1)
            delta_tanh = torch.tanh(self.delta_head(h))
            delta_raw = (delta_tanh + 1.0) / 2.0 * (dmax - dmin) + dmin
            end_logits = self.end_head(h).squeeze(-1)
            return delta_raw, end_logits

        h = torch.tanh(self.init_hidden(z))
        deltas, end_logits = [], []
        for t in range(T):
            x_t = torch.cat([z_seq[:, t], states[:, t], image_feat[:, t]], dim=-1)
            h = self.rnn(x_t, h)
            delta_tanh = torch.tanh(self.delta_head(h))
            delta_raw = (delta_tanh + 1.0) / 2.0 * (dmax - dmin) + dmin
            deltas.append(delta_raw)
            end_logits.append(self.end_head(h).squeeze(-1))
        return torch.stack(deltas, dim=1), torch.stack(end_logits, dim=1)

    def forward(
        self,
        ctrl_pts: torch.Tensor,
        lengths: torch.Tensor,
        skill_order: torch.Tensor,
        states: torch.Tensor,
        images: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.encode(ctrl_pts, lengths, skill_order)
        z_q_st, _, indices, vq_loss = self.quantizer(z_e)
        delta, end_logits = self.decode(z_q_st, states, images)
        return delta, end_logits, vq_loss, indices

    @torch.no_grad()
    def encode_numpy(
        self,
        trajectory: np.ndarray,
        skill_order: float = 0.0,
        device: str = "cpu",
    ) -> int:
        ctrl_pts, T = spline_encode(trajectory, self.n_control, self.spline_degree)
        ctrl_norm = self._np_norm_actions(ctrl_pts)
        cp_t = torch.from_numpy(ctrl_norm).float().unsqueeze(0).to(device)
        l_t = torch.tensor([T], dtype=torch.long, device=device)
        order_t = torch.tensor([[skill_order]], dtype=torch.float32, device=device)
        z_e = self.encode(cp_t, l_t, order_t)
        _, _, indices, _ = self.quantizer(z_e)
        return int(indices[0].item())

    @torch.no_grad()
    def encode_vector_numpy(
        self,
        trajectory: np.ndarray,
        skill_order: float = 0.0,
        device: str = "cpu",
    ) -> np.ndarray:
        ctrl_pts, T = spline_encode(trajectory, self.n_control, self.spline_degree)
        ctrl_norm = self._np_norm_actions(ctrl_pts)
        cp_t = torch.from_numpy(ctrl_norm).float().unsqueeze(0).to(device)
        l_t = torch.tensor([T], dtype=torch.long, device=device)
        order_t = torch.tensor([[skill_order]], dtype=torch.float32, device=device)
        z_e = self.encode(cp_t, l_t, order_t)
        _, z_q, _, _ = self.quantizer(z_e)
        return z_q.squeeze(0).cpu().numpy()


def spline_vqae_loss(
    pred_delta: torch.Tensor,
    pred_end_logits: torch.Tensor,
    target_delta: torch.Tensor,
    target_end: torch.Tensor,
    mask: torch.Tensor,
    vq_loss: torch.Tensor,
    delta_min: torch.Tensor,
    delta_max: torch.Tensor,
    delta_loss_weight: float = 1.0,
    end_loss_weight: float = 1.0,
    end_pos_weight: float | torch.Tensor = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    delta_min = delta_min.to(device=pred_delta.device, dtype=pred_delta.dtype).view(1, 1, -1)
    delta_max = delta_max.to(device=pred_delta.device, dtype=pred_delta.dtype).view(1, 1, -1)
    scale = (delta_max - delta_min).clamp_min(1e-8)
    pred_delta_norm = (pred_delta - delta_min) / scale * 2.0 - 1.0
    target_delta_norm = (target_delta - delta_min) / scale * 2.0 - 1.0

    per_step_delta = F.smooth_l1_loss(pred_delta_norm, target_delta_norm, reduction="none").mean(dim=-1)
    per_skill_delta = (per_step_delta * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp_min(1.0)
    delta_loss = per_skill_delta.mean()

    pos_w = torch.as_tensor(end_pos_weight, device=pred_end_logits.device, dtype=pred_end_logits.dtype)
    end_loss_raw = F.binary_cross_entropy_with_logits(
        pred_end_logits,
        target_end,
        reduction="none",
        pos_weight=pos_w,
    )
    end_loss = (end_loss_raw * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
    total = delta_loss_weight * delta_loss + end_loss_weight * end_loss + vq_loss
    return total, delta_loss, end_loss, vq_loss


@torch.no_grad()
def end_signal_metrics(
    pred_end_logits: torch.Tensor,
    target_end: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    valid = mask.bool()
    if valid.sum().item() == 0:
        return {"acc": 0.0, "precision": 0.0, "recall": 0.0, "positive_rate": 0.0}

    pred = torch.sigmoid(pred_end_logits) >= threshold
    target = target_end >= 0.5
    pred_v = pred[valid]
    target_v = target[valid]
    tp = (pred_v & target_v).sum().float()
    fp = (pred_v & ~target_v).sum().float()
    fn = (~pred_v & target_v).sum().float()
    acc = (pred_v == target_v).float().mean()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    positive_rate = pred_v.float().mean()
    return {
        "acc": float(acc.cpu()),
        "precision": float(precision.cpu()),
        "recall": float(recall.cpu()),
        "positive_rate": float(positive_rate.cpu()),
    }


def train_spline_vqae(
    segments: list[np.ndarray],
    cfg: SplineVQAEConfig,
    wandb_run=None,
    metadata: list[dict] | None = None,
    resume_from: str | None = None,
    decoder_states: list[np.ndarray] | None = None,
    decoder_targets: list[np.ndarray] | None = None,
    images: list[np.ndarray | None] | None = None,
    skill_orders: list[float] | None = None,
) -> SplineVQAE:
    if len(segments) == 0:
        raise ValueError("No skill segments provided.")
    if decoder_states is None or decoder_targets is None:
        raise ValueError("decoder_states and decoder_targets are required for recurrent VQ-VAE training.")

    action_dim = cfg.action_dim if cfg.action_dim > 0 else segments[0].shape[-1]
    a_min = cfg.action_min if cfg.action_min is not None else np.concatenate(segments, axis=0).min(axis=0)
    a_max = cfg.action_max if cfg.action_max is not None else np.concatenate(segments, axis=0).max(axis=0)
    d_min = cfg.delta_min if cfg.delta_min is not None else np.concatenate(decoder_targets, axis=0).min(axis=0)
    d_max = cfg.delta_max if cfg.delta_max is not None else np.concatenate(decoder_targets, axis=0).max(axis=0)
    images = images if images is not None else [None] * len(segments)
    skill_orders = skill_orders if skill_orders is not None else [0.0] * len(segments)
    if cfg.end_pos_weight <= 0:
        total_steps = sum(len(x) for x in decoder_targets)
        n_positive = max(1, len(decoder_targets))
        cfg.end_pos_weight = max(1.0, float((total_steps - n_positive) / n_positive))

    print(
        f"[SplineVQVAE-RNN] {len(segments)} segments | "
        f"n_control={cfg.n_control} degree={cfg.spline_degree} latent={cfg.latent_dim} "
        f"codebook={cfg.num_embeddings} hidden={cfg.hidden_dim} image_dino={cfg.use_images}"
    )
    print(
        f"[SplineVQVAE-RNN] delta_loss=skill-balanced | "
        f"end_loss=timestep BCE pos_weight={cfg.end_pos_weight:.3f}"
    )

    n_val = max(1, int(len(segments) * cfg.val_split))
    idx = np.random.permutation(len(segments))
    train_idx, val_idx = idx[n_val:], idx[:n_val]

    def take(xs, ids):
        return [xs[i] for i in ids]

    def mk_ds(ids):
        return SplineSkillDataset(
            take(segments, ids),
            take(decoder_states, ids),
            take(decoder_targets, ids),
            take(images, ids),
            take(skill_orders, ids),
            cfg.n_control,
            cfg.spline_degree,
            a_min,
            a_max,
            d_min,
            d_max,
            cfg.max_length,
        )

    train_loader = DataLoader(mk_ds(train_idx), batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_skill_batch)
    val_loader = DataLoader(mk_ds(val_idx), batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_skill_batch)

    model = SplineVQAE(
        action_dim=action_dim,
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
        action_min=a_min,
        action_max=a_max,
        delta_min=d_min,
        delta_max=d_max,
        image_model_name=cfg.image_model_name,
        image_feature_dim=cfg.image_feature_dim,
        image_size=cfg.image_size,
        use_images=cfg.use_images,
    ).to(cfg.device)

    print(f"[SplineVQVAE-RNN] trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)
    start_epoch = 1
    best_val = math.inf

    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "optim_state" in ckpt:
            optim.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[SplineVQVAE-RNN] resumed from {resume_from}, starting epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        t_total = t_delta = t_end = t_vq = 0.0
        t_end_acc = t_end_precision = t_end_recall = t_end_positive_rate = 0.0
        for cp, lengths, order, states, targets, ends, mask, img in train_loader:
            cp = cp.to(cfg.device)
            lengths = lengths.to(cfg.device)
            order = order.to(cfg.device)
            states = states.to(cfg.device)
            targets = targets.to(cfg.device)
            ends = ends.to(cfg.device)
            mask = mask.to(cfg.device)
            img = img.to(cfg.device) if img is not None else None

            pred_delta, pred_end, vq_loss, _ = model(cp, lengths, order, states, img)
            loss, delta_l, end_l, vq_l = spline_vqae_loss(
                pred_delta,
                pred_end,
                targets,
                ends,
                mask,
                vq_loss,
                model.delta_min,
                model.delta_max,
                cfg.delta_loss_weight,
                cfg.end_loss_weight,
                cfg.end_pos_weight,
            )
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            t_total += loss.item()
            t_delta += delta_l.item()
            t_end += end_l.item()
            t_vq += vq_l.item()
            metrics = end_signal_metrics(pred_end, ends, mask, cfg.end_threshold)
            t_end_acc += metrics["acc"]
            t_end_precision += metrics["precision"]
            t_end_recall += metrics["recall"]
            t_end_positive_rate += metrics["positive_rate"]

        scheduler.step()

        model.eval()
        v_total = 0.0
        v_end_acc = v_end_precision = v_end_recall = v_end_positive_rate = 0.0
        with torch.no_grad():
            for cp, lengths, order, states, targets, ends, mask, img in val_loader:
                cp = cp.to(cfg.device)
                lengths = lengths.to(cfg.device)
                order = order.to(cfg.device)
                states = states.to(cfg.device)
                targets = targets.to(cfg.device)
                ends = ends.to(cfg.device)
                mask = mask.to(cfg.device)
                img = img.to(cfg.device) if img is not None else None
                pred_delta, pred_end, vq_loss, _ = model(cp, lengths, order, states, img)
                loss, *_ = spline_vqae_loss(
                    pred_delta,
                    pred_end,
                    targets,
                    ends,
                    mask,
                    vq_loss,
                    model.delta_min,
                    model.delta_max,
                    cfg.delta_loss_weight,
                    cfg.end_loss_weight,
                    cfg.end_pos_weight,
                )
                v_total += loss.item()
                metrics = end_signal_metrics(pred_end, ends, mask, cfg.end_threshold)
                v_end_acc += metrics["acc"]
                v_end_precision += metrics["precision"]
                v_end_recall += metrics["recall"]
                v_end_positive_rate += metrics["positive_rate"]

        n_tr, n_vl = max(1, len(train_loader)), max(1, len(val_loader))
        log = {
            "train/loss": t_total / n_tr,
            "train/delta": t_delta / n_tr,
            "train/end": t_end / n_tr,
            "train/vq": t_vq / n_tr,
            "train/end_acc@threshold": t_end_acc / n_tr,
            "train/end_precision@threshold": t_end_precision / n_tr,
            "train/end_recall@threshold": t_end_recall / n_tr,
            "train/end_positive_rate@threshold": t_end_positive_rate / n_tr,
            "val/loss": v_total / n_vl,
            "val/end_acc@threshold": v_end_acc / n_vl,
            "val/end_precision@threshold": v_end_precision / n_vl,
            "val/end_recall@threshold": v_end_recall / n_vl,
            "val/end_positive_rate@threshold": v_end_positive_rate / n_vl,
            "epoch": epoch,
        }
        if wandb_run is not None:
            wandb_run.log(log)
        if epoch % cfg.log_every == 0 or epoch == 1:
            print(
                f"[SplineVQVAE-RNN] epoch {epoch:4d}/{cfg.epochs} "
                f"train={log['train/loss']:.4f} "
                f"(delta={log['train/delta']:.4f} end={log['train/end']:.4f} vq={log['train/vq']:.4f}) "
                f"val={log['val/loss']:.4f} "
                f"end_acc@{cfg.end_threshold:.2f}={log['val/end_acc@threshold']:.3f} "
                f"end_rec={log['val/end_recall@threshold']:.3f}"
            )

        if v_total < best_val:
            best_val = v_total
            if cfg.save_path:
                torch.save({"model_state": model.state_dict(), "cfg": cfg}, cfg.save_path)

        if cfg.checkpoint_every > 0 and epoch % cfg.checkpoint_every == 0 and cfg.save_path:
            ckpt_path = cfg.save_path.replace(".pt", f"_epoch{epoch:04d}.pt")
            torch.save({"model_state": model.state_dict(), "optim_state": optim.state_dict(), "cfg": cfg, "epoch": epoch}, ckpt_path)
            tokens = [model.encode_numpy(seg, skill_orders[i], device=cfg.device) for i, seg in enumerate(segments)]
            vectors = [model.encode_vector_numpy(seg, skill_orders[i], device=cfg.device) for i, seg in enumerate(segments)]
            latents_path = cfg.save_path.replace(".pt", f"_latents_epoch{epoch:04d}.npz")
            save_dict: dict = {
                "latents": np.stack(vectors).astype(np.float32),
                "tokens": np.array(tokens, dtype=np.int32),
                "skill_order": np.array(skill_orders, dtype=np.float32),
            }
            if metadata is not None:
                for key in ("episode_id", "task_id", "skill_index", "frame_start", "frame_end", "length"):
                    save_dict[key] = np.array([m[key] for m in metadata])
            np.savez(latents_path, **save_dict)
            print(f"[SplineVQVAE-RNN] checkpoint tokens -> {latents_path}")

    model = model.cpu()
    print(f"[SplineVQVAE-RNN] done. best val loss: {best_val / max(1, len(val_loader)):.4f}")
    return model


def encode_skills(
    model: SplineVQAE,
    segments: list[np.ndarray],
    device: str = "cpu",
    skill_orders: list[float] | None = None,
) -> np.ndarray:
    model = model.to(device).eval()
    skill_orders = skill_orders if skill_orders is not None else [0.0] * len(segments)
    tokens = [model.encode_numpy(seg, skill_orders[i], device=device) for i, seg in enumerate(segments)]
    return np.array(tokens, dtype=np.int32)


def encode_skill_vectors(
    model: SplineVQAE,
    segments: list[np.ndarray],
    device: str = "cpu",
    skill_orders: list[float] | None = None,
) -> np.ndarray:
    model = model.to(device).eval()
    skill_orders = skill_orders if skill_orders is not None else [0.0] * len(segments)
    vectors = [model.encode_vector_numpy(seg, skill_orders[i], device=device) for i, seg in enumerate(segments)]
    return np.stack(vectors).astype(np.float32)
