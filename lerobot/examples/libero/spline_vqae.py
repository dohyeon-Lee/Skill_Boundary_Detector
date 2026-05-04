"""
Spline VQAE — Deterministic Vector-Quantized Autoencoder for skill trajectories.

Architecture
------------
  Spline codec  : variable-length action trajectory ↔ fixed-size control points + length
  Encoder (MLP) : flatten(ctrl_pts) + length_norm → z_e  (no mu/logvar, deterministic)
  VQ layer      : z_e → nearest codebook entry z_q + integer token index
  Decoder (MLP) : z_q + initial_state → flatten(ctrl_pts) + length_norm
  Loss          : MSE on ctrl_pts + MSE on length + codebook loss + commitment loss

encode_numpy returns the integer token index.
decode_numpy takes the integer token index and reconstructs the trajectory.

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
from torch.utils.data import DataLoader

from spline_vae import (
    GRIPPER_DIM,
    MLPBlock,
    SplineSkillDataset,
    collate_fn,
    spline_decode,
    spline_encode,
)


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


# ── Model ─────────────────────────────────────────────────────────────────────

class SplineVQAE(nn.Module):
    """Deterministic MLP AE + VQ codebook over spline-encoded action trajectories."""

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
        commitment_cost: float = 0.25,
        max_length: float = 500.0,
        action_min: np.ndarray | None = None,
        action_max: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.action_dim     = action_dim
        self.state_dim      = state_dim
        self.n_control      = n_control
        self.spline_degree  = spline_degree
        self.latent_dim     = latent_dim
        self.num_embeddings = num_embeddings
        self.max_length     = max_length

        # ── encoder ───────────────────────────────────────────────────────────
        enc_in = n_control * action_dim + 1
        layers = [MLPBlock(enc_in, hidden_dim, dropout)]
        for _ in range(num_layers - 1):
            layers.append(MLPBlock(hidden_dim, hidden_dim, dropout))
        self.encoder_mlp = nn.Sequential(*layers)
        self.z_head      = nn.Linear(hidden_dim, latent_dim)  # single deterministic head

        # ── VQ layer ──────────────────────────────────────────────────────────
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)

        # ── decoder ───────────────────────────────────────────────────────────
        dec_in = latent_dim + state_dim
        dec_layers = [MLPBlock(dec_in, hidden_dim, dropout)]
        for _ in range(num_layers - 1):
            dec_layers.append(MLPBlock(hidden_dim, hidden_dim, dropout))
        self.decoder_mlp = nn.Sequential(*dec_layers)
        self.ctrl_head   = nn.Linear(hidden_dim, n_control * action_dim)
        self.length_head = nn.Linear(hidden_dim, 1)

        _min = action_min if action_min is not None else np.zeros(action_dim)
        _max = action_max if action_max is not None else np.ones(action_dim)
        self.register_buffer("action_min", torch.tensor(_min, dtype=torch.float32))
        self.register_buffer("action_max", torch.tensor(_max, dtype=torch.float32))

    # ── helpers ───────────────────────────────────────────────────────────────

    def _norm_length(self, length: torch.Tensor) -> torch.Tensor:
        return length.float() / self.max_length

    def _np_norm_actions(self, ctrl_pts: np.ndarray) -> np.ndarray:
        lo = self.action_min.cpu().numpy()
        hi = self.action_max.cpu().numpy()
        return (ctrl_pts - lo) / (hi - lo + 1e-8) * 2 - 1

    def _np_denorm_actions(self, ctrl_pts: np.ndarray) -> np.ndarray:
        lo = self.action_min.cpu().numpy()
        hi = self.action_max.cpu().numpy()
        return (ctrl_pts + 1) / 2 * (hi - lo + 1e-8) + lo

    # ── encoder ───────────────────────────────────────────────────────────────

    def encode(
        self,
        ctrl_pts: torch.Tensor,  # (B, N, action_dim) normalised
        lengths: torch.Tensor,   # (B,) raw int lengths
    ) -> torch.Tensor:           # z_e : (B, latent_dim)
        B    = ctrl_pts.size(0)
        flat = ctrl_pts.view(B, -1)
        l    = self._norm_length(lengths.to(flat.device)).unsqueeze(-1)
        x    = torch.cat([flat, l], dim=-1)
        h    = self.encoder_mlp(x)
        return self.z_head(h)

    # ── decoder ───────────────────────────────────────────────────────────────

    def decode(
        self,
        z: torch.Tensor,      # (B, latent_dim)
        state: torch.Tensor,  # (B, state_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ctrl_pts_norm (B, N, D) and length_norm (B, 1)."""
        B   = z.size(0)
        x   = torch.cat([z, state], dim=-1) if self.state_dim > 0 else z
        h   = self.decoder_mlp(x)
        raw = self.ctrl_head(h).view(B, self.n_control, self.action_dim)
        gripper_idx = (self.action_dim + GRIPPER_DIM) % self.action_dim
        ctrl_pts = torch.tanh(raw)
        ctrl_pts = ctrl_pts.clone()
        ctrl_pts[:, :, gripper_idx] = raw[:, :, gripper_idx]  # raw logit for BCE
        length_norm = torch.sigmoid(self.length_head(h))
        return ctrl_pts, length_norm

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        ctrl_pts: torch.Tensor,  # (B, N, action_dim) normalised
        lengths: torch.Tensor,   # (B,)
        state: torch.Tensor,     # (B, state_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (recon_ctrl, recon_len, vq_loss, indices)."""
        z_e = self.encode(ctrl_pts, lengths)
        z_q_st, _, indices, vq_loss = self.quantizer(z_e)
        recon_ctrl, recon_len = self.decode(z_q_st, state)
        return recon_ctrl, recon_len, vq_loss, indices

    # ── numpy convenience ─────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_numpy(self, actions: np.ndarray, device: str = "cpu") -> int:
        """(T, action_dim) raw actions → token index (int)."""
        ctrl_pts, T = spline_encode(actions, self.n_control, self.spline_degree)
        ctrl_norm   = self._np_norm_actions(ctrl_pts)
        cp_t = torch.from_numpy(ctrl_norm).float().unsqueeze(0).to(device)
        l_t  = torch.tensor([T], dtype=torch.long)
        z_e  = self.encode(cp_t, l_t)
        _, _, indices, _ = self.quantizer(z_e)
        return int(indices[0].item())

    @torch.no_grad()
    def encode_vector_numpy(self, actions: np.ndarray, device: str = "cpu") -> np.ndarray:
        """(T, action_dim) raw actions → quantized codebook vector z_q (latent_dim,)."""
        ctrl_pts, T = spline_encode(actions, self.n_control, self.spline_degree)
        ctrl_norm   = self._np_norm_actions(ctrl_pts)
        cp_t = torch.from_numpy(ctrl_norm).float().unsqueeze(0).to(device)
        l_t  = torch.tensor([T], dtype=torch.long)
        z_e  = self.encode(cp_t, l_t)
        _, z_q, _, _ = self.quantizer(z_e)
        return z_q.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def decode_numpy(
        self,
        index: int,
        state: np.ndarray,  # (state_dim,)
        device: str = "cpu",
    ) -> np.ndarray:
        """token index + state → reconstructed raw actions (T, action_dim)."""
        idx_t = torch.tensor([index], dtype=torch.long, device=device)
        z_q   = self.quantizer.embedding(idx_t)  # (1, D)
        s_t   = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
        ctrl_norm, len_norm = self.decode(z_q, s_t)
        ctrl_norm_np = ctrl_norm.squeeze(0).cpu().numpy()
        gripper_idx  = (self.action_dim + GRIPPER_DIM) % self.action_dim
        ctrl_norm_np[:, gripper_idx] = np.where(
            1.0 / (1.0 + np.exp(-ctrl_norm_np[:, gripper_idx])) > 0.5, 1.0, -1.0
        )
        ctrl_pts = self._np_denorm_actions(ctrl_norm_np)
        T = max(2, round(float(len_norm.item()) * self.max_length))
        return spline_decode(ctrl_pts, T, self.spline_degree)


# ── Loss ──────────────────────────────────────────────────────────────────────

def spline_vqae_loss(
    recon_ctrl: torch.Tensor,    # (B, N, D)  — gripper dim is raw logit
    recon_len:  torch.Tensor,    # (B, 1)
    target_ctrl: torch.Tensor,   # (B, N, D)  — gripper dim is ±1 normalised
    target_len:  torch.Tensor,   # (B, 1)  normalised
    vq_loss: torch.Tensor,       # scalar from VQ layer
    length_weight: float = 1.0,
    gripper_loss_weight: float = 1.0,
    action_dim: int = 7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gripper_idx = (action_dim + GRIPPER_DIM) % action_dim
    non_gripper = [i for i in range(action_dim) if i != gripper_idx]

    ctrl_loss_cont = F.mse_loss(recon_ctrl[:, :, non_gripper], target_ctrl[:, :, non_gripper])
    gripper_target_01 = (target_ctrl[:, :, gripper_idx] + 1.0) / 2.0
    ctrl_loss_gripper = F.binary_cross_entropy_with_logits(
        recon_ctrl[:, :, gripper_idx], gripper_target_01
    )
    ctrl_loss = ctrl_loss_cont + gripper_loss_weight * ctrl_loss_gripper
    len_loss  = F.mse_loss(recon_len, target_len)
    total     = ctrl_loss + length_weight * len_loss + vq_loss
    return total, ctrl_loss, len_loss, vq_loss


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class SplineVQAEConfig:
    action_dim:    int   = 7
    state_dim:     int   = 7
    n_control:     int   = 30
    spline_degree: int   = 3
    hidden_dim:    int   = 256
    latent_dim:    int   = 64
    num_embeddings: int  = 512
    num_layers:    int   = 3
    dropout:       float = 0.1
    commitment_cost:     float = 0.25
    length_weight:       float = 10.0
    gripper_loss_weight: float = 1.0
    max_length:          float = 500.0
    lr:            float = 3e-4
    batch_size:    int   = 64
    epochs:        int   = 100
    grad_clip:     float = 1.0
    device:        str   = "cuda"
    val_split:     float = 0.1
    log_every:     int   = 10
    save_path:     str | None = None
    checkpoint_every: int = 0
    action_min: np.ndarray | None = None
    action_max: np.ndarray | None = None


# ── Training loop ─────────────────────────────────────────────────────────────

def train_spline_vqae(
    segments: list[np.ndarray],
    init_states: list[np.ndarray],
    cfg: SplineVQAEConfig,
    wandb_run=None,
    metadata: list[dict] | None = None,
    resume_from: str | None = None,
) -> SplineVQAE:
    if len(segments) == 0:
        raise ValueError("No skill segments provided.")

    action_dim = cfg.action_dim if cfg.action_dim > 0 else segments[0].shape[-1]
    a_min = cfg.action_min if cfg.action_min is not None else np.full(action_dim, -1.0)
    a_max = cfg.action_max if cfg.action_max is not None else np.full(action_dim,  1.0)

    print(
        f"[SplineVQAE] {len(segments)} segments | "
        f"n_control={cfg.n_control} degree={cfg.spline_degree} "
        f"latent={cfg.latent_dim} codebook={cfg.num_embeddings} "
        f"hidden={cfg.hidden_dim} layers={cfg.num_layers} "
        f"commitment_cost={cfg.commitment_cost} epochs={cfg.epochs}"
    )

    n_val  = max(1, int(len(segments) * cfg.val_split))
    idx    = np.random.permutation(len(segments))
    train_segs,   val_segs   = [segments[i]   for i in idx[n_val:]], [segments[i]   for i in idx[:n_val]]
    train_states, val_states = [init_states[i] for i in idx[n_val:]], [init_states[i] for i in idx[:n_val]]

    mk_ds = lambda segs, states: SplineSkillDataset(
        segs, states, cfg.n_control, cfg.spline_degree, a_min, a_max, cfg.max_length
    )
    train_loader = DataLoader(mk_ds(train_segs, train_states), batch_size=cfg.batch_size, shuffle=True,  collate_fn=collate_fn, drop_last=False)
    val_loader   = DataLoader(mk_ds(val_segs,   val_states),   batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    model = SplineVQAE(
        action_dim=action_dim, state_dim=cfg.state_dim,
        n_control=cfg.n_control, spline_degree=cfg.spline_degree,
        hidden_dim=cfg.hidden_dim, latent_dim=cfg.latent_dim,
        num_embeddings=cfg.num_embeddings,
        num_layers=cfg.num_layers, dropout=cfg.dropout,
        commitment_cost=cfg.commitment_cost,
        max_length=cfg.max_length, action_min=a_min, action_max=a_max,
    ).to(cfg.device)

    print(f"[SplineVQAE] params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    start_epoch = 1
    optim     = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)

    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "optim_state" in ckpt:
            optim.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=cfg.epochs, eta_min=cfg.lr * 0.01, last_epoch=start_epoch - 2
        )
        print(f"[SplineVQAE] resumed from {resume_from}, starting epoch {start_epoch}")

    best_val = math.inf

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        t_total = t_ctrl = t_len = t_vq = 0.0

        for cp, l_raw, l_norm, state in train_loader:
            cp, l_norm, state = cp.to(cfg.device), l_norm.to(cfg.device), state.to(cfg.device)
            recon_ctrl, recon_len, vq_loss, _ = model(cp, l_raw, state)
            loss, ctrl_l, len_l, vq_l = spline_vqae_loss(
                recon_ctrl, recon_len, cp, l_norm, vq_loss,
                cfg.length_weight, cfg.gripper_loss_weight, cfg.action_dim,
            )
            optim.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            t_total += loss.item(); t_ctrl += ctrl_l.item(); t_len += len_l.item(); t_vq += vq_l.item()

        scheduler.step()

        model.eval(); v_total = 0.0
        with torch.no_grad():
            for cp, l_raw, l_norm, state in val_loader:
                cp, l_norm, state = cp.to(cfg.device), l_norm.to(cfg.device), state.to(cfg.device)
                recon_ctrl, recon_len, vq_loss, _ = model(cp, l_raw, state)
                loss, *_ = spline_vqae_loss(
                    recon_ctrl, recon_len, cp, l_norm, vq_loss,
                    cfg.length_weight, cfg.gripper_loss_weight, cfg.action_dim,
                )
                v_total += loss.item()

        n_tr, n_vl = len(train_loader), len(val_loader)
        log = {
            "train/loss": t_total / n_tr,
            "train/ctrl": t_ctrl  / n_tr,
            "train/len":  t_len   / n_tr,
            "train/vq":   t_vq    / n_tr,
            "val/loss":   v_total / n_vl,
            "epoch": epoch,
        }
        if wandb_run is not None:
            wandb_run.log(log)
        if epoch % cfg.log_every == 0 or epoch == 1:
            print(
                f"[SplineVQAE] epoch {epoch:4d}/{cfg.epochs}  "
                f"train: {log['train/loss']:.4f} "
                f"(ctrl={log['train/ctrl']:.4f} len={log['train/len']:.4f} vq={log['train/vq']:.4f})  "
                f"val: {log['val/loss']:.4f}"
            )

        if v_total < best_val:
            best_val = v_total
            if cfg.save_path:
                torch.save({"model_state": model.state_dict(), "cfg": cfg}, cfg.save_path)

        if cfg.checkpoint_every > 0 and epoch % cfg.checkpoint_every == 0 and cfg.save_path:
            ckpt_path = cfg.save_path.replace(".pt", f"_epoch{epoch:04d}.pt")
            torch.save({
                "model_state": model.state_dict(),
                "optim_state": optim.state_dict(),
                "cfg": cfg, "epoch": epoch,
            }, ckpt_path)
            model.eval()
            tokens  = [model.encode_numpy(seg, device=cfg.device) for seg in segments]
            vectors = [model.encode_vector_numpy(seg, device=cfg.device) for seg in segments]
            latents_path = cfg.save_path.replace(".pt", f"_latents_epoch{epoch:04d}.npz")
            save_dict: dict = {
                "latents": np.stack(vectors).astype(np.float32),   # (N, latent_dim) for t-SNE
                "tokens":  np.array(tokens, dtype=np.int32),       # (N,) for skill predictor
            }
            if metadata is not None:
                for key in ("episode_id", "task_id", "skill_index", "frame_start", "frame_end", "length"):
                    save_dict[key] = np.array([m[key] for m in metadata])
            np.savez(latents_path, **save_dict)
            model.train()
            print(f"[SplineVQAE] checkpoint tokens → {latents_path}")

    model = model.cpu()
    print(f"[SplineVQAE] done. best val loss: {best_val / len(val_loader):.4f}")
    return model


# ── Encode all segments ───────────────────────────────────────────────────────

def encode_skills(
    model: SplineVQAE,
    segments: list[np.ndarray],
    device: str = "cpu",
) -> np.ndarray:
    """Returns (N,) int32 token indices, one per skill segment."""
    model = model.to(device).eval()
    tokens = [model.encode_numpy(seg, device=device) for seg in segments]
    return np.array(tokens, dtype=np.int32)


def encode_skill_vectors(
    model: SplineVQAE,
    segments: list[np.ndarray],
    device: str = "cpu",
) -> np.ndarray:
    """Returns (N, latent_dim) float32 quantized codebook vectors, one per skill segment."""
    model = model.to(device).eval()
    vectors = [model.encode_vector_numpy(seg, device=device) for seg in segments]
    return np.stack(vectors).astype(np.float32)
