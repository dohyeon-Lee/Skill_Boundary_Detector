"""
Spline VAE — MLP Variational Autoencoder for skill trajectories.

Architecture
------------
  Spline codec  : variable-length action trajectory ↔ fixed-size control points + length
  Encoder (MLP) : flatten(ctrl_pts) + length_norm → (mu, logvar)
  Decoder (MLP) : z + initial_state → flatten(ctrl_pts) + length_norm
  Loss          : MSE on ctrl_pts + MSE on length + beta * KL

Gripper dimension (last) uses degree-0 (nearest-neighbor) spline.
All other dims use the configured spline degree (default: cubic).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import make_interp_spline
from torch.utils.data import DataLoader, Dataset


# ── Spline codec (numpy) ───────────────────────────────────────────────────────

GRIPPER_DIM = -1  # last dim always uses degree-0 spline


def spline_encode(
    trajectory: np.ndarray,   # (T, action_dim)
    n_control: int,
    degree: int,
) -> tuple[np.ndarray, int]:
    """trajectory → control points (n_control, action_dim)  +  length T."""
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


def spline_decode(
    ctrl_pts: np.ndarray,  # (n_control, action_dim)
    length: int,
    degree: int,
) -> np.ndarray:
    """Control points + length → reconstructed trajectory (length, action_dim)."""
    n_control, D = ctrl_pts.shape
    t_ctrl = np.linspace(0.0, 1.0, n_control)
    t_out  = np.linspace(0.0, 1.0, length)
    recon = np.zeros((length, D), dtype=np.float32)
    gripper_idx = (D + GRIPPER_DIM) % D
    for d in range(D):
        k = 0 if d == gripper_idx else degree
        k = min(k, n_control - 1)
        spl = make_interp_spline(t_ctrl, ctrl_pts[:, d], k=k)
        recon[:, d] = spl(t_out)
    return recon


# ── MLP block ──────────────────────────────────────────────────────────────────

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


# ── Model ──────────────────────────────────────────────────────────────────────

class SplineVAE(nn.Module):
    """MLP VAE over spline-encoded action trajectories."""

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        n_control: int = 30,
        spline_degree: int = 3,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_length: float = 500.0,
        action_min: np.ndarray | None = None,
        action_max: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.action_dim   = action_dim
        self.state_dim    = state_dim
        self.n_control    = n_control
        self.spline_degree = spline_degree
        self.latent_dim   = latent_dim
        self.max_length   = max_length

        # encoder input: flattened ctrl_pts + normalised length
        enc_in = n_control * action_dim + 1

        layers = [MLPBlock(enc_in, hidden_dim, dropout)]
        for _ in range(num_layers - 1):
            layers.append(MLPBlock(hidden_dim, hidden_dim, dropout))
        self.encoder_mlp = nn.Sequential(*layers)
        self.mu_head     = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # decoder input: z + initial state
        dec_in = latent_dim + state_dim
        dec_layers = [MLPBlock(dec_in, hidden_dim, dropout)]
        for _ in range(num_layers - 1):
            dec_layers.append(MLPBlock(hidden_dim, hidden_dim, dropout))
        self.decoder_mlp  = nn.Sequential(*dec_layers)
        self.ctrl_head    = nn.Linear(hidden_dim, n_control * action_dim)
        self.length_head  = nn.Linear(hidden_dim, 1)

        _min = action_min if action_min is not None else np.zeros(action_dim)
        _max = action_max if action_max is not None else np.ones(action_dim)
        self.register_buffer("action_min", torch.tensor(_min, dtype=torch.float32))
        self.register_buffer("action_max", torch.tensor(_max, dtype=torch.float32))

    # ── helpers ────────────────────────────────────────────────────────────────

    def _norm_actions(self, ctrl_pts: torch.Tensor) -> torch.Tensor:
        """ctrl_pts (B, N, D) → [-1, 1]"""
        lo = self.action_min.view(1, 1, -1)
        hi = self.action_max.view(1, 1, -1)
        return (ctrl_pts - lo) / (hi - lo + 1e-8) * 2 - 1

    def _denorm_actions(self, ctrl_pts: torch.Tensor) -> torch.Tensor:
        lo = self.action_min.view(1, 1, -1)
        hi = self.action_max.view(1, 1, -1)
        return (ctrl_pts + 1) / 2 * (hi - lo + 1e-8) + lo

    def _norm_length(self, length: torch.Tensor) -> torch.Tensor:
        return length.float() / self.max_length

    def _denorm_length(self, length_norm: torch.Tensor) -> torch.Tensor:
        return length_norm * self.max_length

    # ── encoder ────────────────────────────────────────────────────────────────

    def encode(
        self,
        ctrl_pts: torch.Tensor,   # (B, N, action_dim)  already normalised
        lengths: torch.Tensor,    # (B,)  raw int lengths
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = ctrl_pts.size(0)
        flat = ctrl_pts.view(B, -1)                       # (B, N*D)
        l    = self._norm_length(lengths.to(flat.device)).unsqueeze(-1)   # (B, 1)
        x    = torch.cat([flat, l], dim=-1)               # (B, N*D+1)
        h    = self.encoder_mlp(x)
        return self.mu_head(h), self.logvar_head(h)

    # ── reparameterisation ─────────────────────────────────────────────────────

    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return mu
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    # ── decoder ────────────────────────────────────────────────────────────────

    def decode(
        self,
        z: torch.Tensor,      # (B, latent_dim)
        state: torch.Tensor,  # (B, state_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ctrl_pts_norm (B, N, D) and length_norm (B, 1)."""
        B = z.size(0)
        x = torch.cat([z, state], dim=-1) if self.state_dim > 0 else z
        h = self.decoder_mlp(x)
        raw = self.ctrl_head(h).view(B, self.n_control, self.action_dim)
        gripper_idx = (self.action_dim + GRIPPER_DIM) % self.action_dim
        ctrl_pts = torch.tanh(raw)
        # Gripper dim: keep as raw logit (used with BCE loss; tanh is replaced by sigmoid+threshold)
        ctrl_pts = ctrl_pts.clone()
        ctrl_pts[:, :, gripper_idx] = raw[:, :, gripper_idx]
        length_norm = torch.sigmoid(self.length_head(h))   # (B, 1)  ∈ (0, 1)
        return ctrl_pts, length_norm

    # ── forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        ctrl_pts: torch.Tensor,  # (B, N, action_dim)  normalised
        lengths: torch.Tensor,   # (B,)
        state: torch.Tensor,     # (B, state_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (recon_ctrl_pts, recon_length_norm, mu, logvar)."""
        mu, logvar = self.encode(ctrl_pts, lengths)
        z = self.reparameterise(mu, logvar)
        recon_ctrl, recon_len = self.decode(z, state)
        return recon_ctrl, recon_len, mu, logvar

    # ── numpy convenience ──────────────────────────────────────────────────────

    def _np_norm_actions(self, ctrl_pts: np.ndarray) -> np.ndarray:
        lo = self.action_min.cpu().numpy()
        hi = self.action_max.cpu().numpy()
        return (ctrl_pts - lo) / (hi - lo + 1e-8) * 2 - 1

    def _np_denorm_actions(self, ctrl_pts: np.ndarray) -> np.ndarray:
        lo = self.action_min.cpu().numpy()
        hi = self.action_max.cpu().numpy()
        return (ctrl_pts + 1) / 2 * (hi - lo + 1e-8) + lo

    @torch.no_grad()
    def encode_numpy(self, actions: np.ndarray, device: str = "cpu") -> np.ndarray:
        """(T, action_dim) raw actions → latent z (latent_dim,)."""
        ctrl_pts, T = spline_encode(actions, self.n_control, self.spline_degree)
        ctrl_norm   = self._np_norm_actions(ctrl_pts)
        cp_t = torch.from_numpy(ctrl_norm).float().unsqueeze(0).to(device)
        l_t  = torch.tensor([T], dtype=torch.long)
        mu, _ = self.encode(cp_t, l_t)
        return mu.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def decode_numpy(
        self,
        z: np.ndarray,      # (latent_dim,)
        state: np.ndarray,  # (state_dim,)
        device: str = "cpu",
    ) -> np.ndarray:
        """latent z + state → reconstructed raw actions (T, action_dim)."""
        z_t = torch.from_numpy(z).float().unsqueeze(0).to(device)
        s_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        ctrl_norm, len_norm = self.decode(z_t, s_t)
        ctrl_norm_np = ctrl_norm.squeeze(0).cpu().numpy()
        # Gripper: sigmoid(logit) > 0.5 → +1, else → -1  (in normalised ±1 space)
        gripper_idx = (self.action_dim + GRIPPER_DIM) % self.action_dim
        ctrl_norm_np[:, gripper_idx] = np.where(
            1.0 / (1.0 + np.exp(-ctrl_norm_np[:, gripper_idx])) > 0.5, 1.0, -1.0
        )
        ctrl_pts = self._np_denorm_actions(ctrl_norm_np)
        T = max(2, round(float(len_norm.item()) * self.max_length))
        return spline_decode(ctrl_pts, T, self.spline_degree)


# ── Loss ───────────────────────────────────────────────────────────────────────

def spline_vae_loss(
    recon_ctrl: torch.Tensor,    # (B, N, D)  — gripper dim is raw logit
    recon_len:  torch.Tensor,    # (B, 1)
    target_ctrl: torch.Tensor,   # (B, N, D)  — gripper dim is ±1 normalised
    target_len:  torch.Tensor,   # (B, 1)  normalised
    mu: torch.Tensor,            # (B, latent_dim)
    logvar: torch.Tensor,        # (B, latent_dim)
    beta: float = 1.0,
    length_weight: float = 1.0,
    gripper_loss_weight: float = 1.0,
    action_dim: int = 7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gripper_idx = (action_dim + GRIPPER_DIM) % action_dim
    non_gripper = [i for i in range(action_dim) if i != gripper_idx]

    # Continuous dims: MSE
    ctrl_loss_cont = F.mse_loss(recon_ctrl[:, :, non_gripper], target_ctrl[:, :, non_gripper])

    # Gripper dim: BCE (target ±1 → 0/1, prediction is raw logit)
    gripper_target_01 = (target_ctrl[:, :, gripper_idx] + 1.0) / 2.0
    ctrl_loss_gripper = F.binary_cross_entropy_with_logits(
        recon_ctrl[:, :, gripper_idx], gripper_target_01
    )

    ctrl_loss = ctrl_loss_cont + gripper_loss_weight * ctrl_loss_gripper
    len_loss  = F.mse_loss(recon_len, target_len)
    kl_loss   = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total     = ctrl_loss + length_weight * len_loss + beta * kl_loss
    return total, ctrl_loss, len_loss, kl_loss


# ── Dataset ────────────────────────────────────────────────────────────────────

class SplineSkillDataset(Dataset):
    """Pre-encodes trajectories to spline control points at construction time."""

    def __init__(
        self,
        segments: list[np.ndarray],    # (T, action_dim) each
        init_states: list[np.ndarray], # (state_dim,) each
        n_control: int,
        spline_degree: int,
        action_min: np.ndarray,
        action_max: np.ndarray,
        max_length: float,
    ) -> None:
        self.init_states = init_states
        self.max_length  = max_length

        self.ctrl_pts: list[np.ndarray] = []
        self.lengths:  list[int]        = []
        for seg in segments:
            cp, T = spline_encode(seg, n_control, spline_degree)
            cp_norm = (cp - action_min) / (action_max - action_min + 1e-8) * 2 - 1
            self.ctrl_pts.append(cp_norm.astype(np.float32))
            self.lengths.append(T)

    def __len__(self) -> int:
        return len(self.ctrl_pts)

    def __getitem__(self, idx: int):
        cp    = torch.from_numpy(self.ctrl_pts[idx])             # (N, D)
        state = torch.from_numpy(self.init_states[idx].astype(np.float32))
        l_raw = torch.tensor(self.lengths[idx], dtype=torch.long)
        l_norm = torch.tensor(self.lengths[idx] / self.max_length, dtype=torch.float32)
        return cp, l_raw, l_norm, state


def collate_fn(batch):
    cp, l_raw, l_norm, state = zip(*batch)
    return (
        torch.stack(cp),
        torch.stack(l_raw),
        torch.stack(l_norm).unsqueeze(-1),  # (B, 1)
        torch.stack(state),
    )


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class SplineVAEConfig:
    action_dim:    int   = 7
    state_dim:     int   = 7
    n_control:     int   = 30
    spline_degree: int   = 3
    hidden_dim:    int   = 256
    latent_dim:    int   = 64
    num_layers:    int   = 3
    dropout:       float = 0.1
    beta:                float = 1.0
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


# ── Training loop ──────────────────────────────────────────────────────────────

def train_spline_vae(
    segments: list[np.ndarray],
    init_states: list[np.ndarray],
    cfg: SplineVAEConfig,
    wandb_run=None,
    metadata: list[dict] | None = None,
    resume_from: str | None = None,
) -> SplineVAE:
    if len(segments) == 0:
        raise ValueError("No skill segments provided.")

    action_dim = cfg.action_dim if cfg.action_dim > 0 else segments[0].shape[-1]
    a_min = cfg.action_min if cfg.action_min is not None else np.full(action_dim, -1.0)
    a_max = cfg.action_max if cfg.action_max is not None else np.full(action_dim,  1.0)

    print(
        f"[SplineVAE] {len(segments)} segments | "
        f"n_control={cfg.n_control} degree={cfg.spline_degree} "
        f"latent={cfg.latent_dim} hidden={cfg.hidden_dim} layers={cfg.num_layers} "
        f"beta={cfg.beta} epochs={cfg.epochs}"
    )

    n_val    = max(1, int(len(segments) * cfg.val_split))
    idx      = np.random.permutation(len(segments))
    train_segs,  val_segs   = [segments[i]   for i in idx[n_val:]], [segments[i]   for i in idx[:n_val]]
    train_states, val_states = [init_states[i] for i in idx[n_val:]], [init_states[i] for i in idx[:n_val]]

    mk_ds = lambda segs, states: SplineSkillDataset(
        segs, states, cfg.n_control, cfg.spline_degree, a_min, a_max, cfg.max_length
    )
    train_loader = DataLoader(mk_ds(train_segs, train_states),  batch_size=cfg.batch_size, shuffle=True,  collate_fn=collate_fn, drop_last=False)
    val_loader   = DataLoader(mk_ds(val_segs,   val_states),    batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    model = SplineVAE(
        action_dim=action_dim, state_dim=cfg.state_dim,
        n_control=cfg.n_control, spline_degree=cfg.spline_degree,
        hidden_dim=cfg.hidden_dim, latent_dim=cfg.latent_dim,
        num_layers=cfg.num_layers, dropout=cfg.dropout,
        max_length=cfg.max_length, action_min=a_min, action_max=a_max,
    ).to(cfg.device)

    print(f"[SplineVAE] params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    start_epoch = 1
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
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
        print(f"[SplineVAE] resumed from {resume_from}, starting epoch {start_epoch}")

    best_val = math.inf

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        t_total = t_ctrl = t_len = t_kl = 0.0

        for cp, l_raw, l_norm, state in train_loader:
            cp, l_norm, state = cp.to(cfg.device), l_norm.to(cfg.device), state.to(cfg.device)
            recon_ctrl, recon_len, mu, logvar = model(cp, l_raw, state)
            loss, ctrl_l, len_l, kl_l = spline_vae_loss(
                recon_ctrl, recon_len, cp, l_norm, mu, logvar,
                cfg.beta, cfg.length_weight, cfg.gripper_loss_weight, cfg.action_dim,
            )
            optim.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            t_total += loss.item(); t_ctrl += ctrl_l.item(); t_len += len_l.item(); t_kl += kl_l.item()

        scheduler.step()

        model.eval(); v_total = 0.0
        with torch.no_grad():
            for cp, l_raw, l_norm, state in val_loader:
                cp, l_norm, state = cp.to(cfg.device), l_norm.to(cfg.device), state.to(cfg.device)
                recon_ctrl, recon_len, mu, logvar = model(cp, l_raw, state)
                loss, *_ = spline_vae_loss(
                    recon_ctrl, recon_len, cp, l_norm, mu, logvar,
                    cfg.beta, cfg.length_weight, cfg.gripper_loss_weight, cfg.action_dim,
                )
                v_total += loss.item()

        n_tr, n_vl = len(train_loader), len(val_loader)
        log = {
            "train/loss":  t_total / n_tr,
            "train/ctrl":  t_ctrl  / n_tr,
            "train/len":   t_len   / n_tr,
            "train/kl":    t_kl    / n_tr,
            "val/loss":    v_total / n_vl,
            "epoch": epoch,
        }
        if wandb_run is not None:
            wandb_run.log(log)
        if epoch % cfg.log_every == 0 or epoch == 1:
            print(
                f"[SplineVAE] epoch {epoch:4d}/{cfg.epochs}  "
                f"train: {log['train/loss']:.4f} "
                f"(ctrl={log['train/ctrl']:.4f} len={log['train/len']:.4f} kl={log['train/kl']:.4f})  "
                f"val: {log['val/loss']:.4f}"
            )

        if v_total < best_val:
            best_val = v_total
            if cfg.save_path:
                torch.save({"model_state": model.state_dict(), "cfg": cfg}, cfg.save_path)

        if cfg.checkpoint_every > 0 and epoch % cfg.checkpoint_every == 0 and cfg.save_path:
            ckpt_path = cfg.save_path.replace(".pt", f"_epoch{epoch:04d}.pt")
            torch.save({"model_state": model.state_dict(), "optim_state": optim.state_dict(), "cfg": cfg, "epoch": epoch}, ckpt_path)
            model.eval()
            codes = [model.encode_numpy(seg, device=cfg.device) for seg in segments]
            latents_path = cfg.save_path.replace(".pt", f"_latents_epoch{epoch:04d}.npz")
            save_dict: dict = {"latents": np.stack(codes)}
            if metadata is not None:
                for key in ("episode_id", "task_id", "skill_index", "frame_start", "frame_end", "length"):
                    save_dict[key] = np.array([m[key] for m in metadata])
            np.savez(latents_path, **save_dict)
            model.train()
            print(f"[SplineVAE] checkpoint latents → {latents_path}")

    model = model.cpu()
    print(f"[SplineVAE] done. best val loss: {best_val / len(val_loader):.4f}")
    return model


# ── Encode all segments ────────────────────────────────────────────────────────

def encode_skills(
    model: SplineVAE,
    segments: list[np.ndarray],
    device: str = "cpu",
) -> np.ndarray:
    model = model.to(device).eval()
    return np.stack([model.encode_numpy(seg, device=device) for seg in segments])
