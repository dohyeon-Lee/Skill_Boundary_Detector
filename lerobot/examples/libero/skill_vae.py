"""
Skill VAE — BiLSTM Variational Autoencoder for variable-length skill trajectories.

Architecture:
  Encoder : BiLSTM over action sequence
            → last hidden (fwd + bwd concat) → Linear → (mu, logvar)
  Decoder : LSTM with h0/c0 initialised from z
            → teacher-forced on GT actions during training
            → autoregressive at inference time
  Loss    : masked MSE reconstruction + beta-weighted KL divergence

Input  : action trajectory  (T, action_dim)
Output : reconstructed action trajectory (T, action_dim)
         latent code z  (latent_dim,)

Variable-length handling:
  Sequences within a batch are right-padded to the longest sequence.
  pack_padded_sequence / pad_packed_sequence are used so padding frames
  do not affect LSTM hidden states or loss computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset


# ── Model ──────────────────────────────────────────────────────────────────────

class SkillVAE(nn.Module):
    """BiLSTM Variational Autoencoder for skill action trajectories."""

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        enc_input_dim = action_dim
        enc_drop = dropout if num_layers > 1 else 0.0
        self.encoder_lstm = nn.LSTM(
            enc_input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=enc_drop,
        )
        # BiLSTM last layer: fwd (hidden_dim) + bwd (hidden_dim) → 2*hidden_dim
        self.mu_head = nn.Linear(hidden_dim * 2, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim * 2, latent_dim)

        # Decoder LSTM: h0 and c0 each have shape (num_layers, B, hidden_dim)
        dec_drop = dropout if num_layers > 1 else 0.0
        self.z_to_h = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.z_to_c = nn.Linear(latent_dim, hidden_dim * num_layers)
        # Decoder input: shifted GT actions (teacher forcing)
        self.decoder_lstm = nn.LSTM(
            action_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dec_drop,
        )
        self.output_head = nn.Linear(hidden_dim, action_dim)

        # Learned start token fed as the first decoder input
        self.start_token = nn.Parameter(torch.zeros(1, 1, action_dim))

        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    # ── Encoder ────────────────────────────────────────────────────────────────

    def encode(
        self,
        actions: torch.Tensor,     # (B, T_max, action_dim)  padded
        lengths: torch.Tensor,     # (B,) int64 CPU tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, logvar), each (B, latent_dim)."""
        packed = pack_padded_sequence(actions, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.encoder_lstm(packed)
        # h: (num_layers * 2, B, hidden_dim)  [fwd/bwd interleaved per layer]
        # Last layer: index -2 (fwd) and -1 (bwd)
        h_fwd = h[-2]   # (B, hidden_dim)
        h_bwd = h[-1]   # (B, hidden_dim)
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)  # (B, hidden_dim*2)
        return self.mu_head(h_cat), self.logvar_head(h_cat)

    # ── Reparameterisation ─────────────────────────────────────────────────────

    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ── Decoder ────────────────────────────────────────────────────────────────

    def _z_to_hidden(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map latent z → (h0, c0) for decoder LSTM."""
        B = z.size(0)
        h0 = self.z_to_h(z).view(B, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        c0 = self.z_to_c(z).view(B, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        return h0, c0

    def decode(
        self,
        z: torch.Tensor,           # (B, latent_dim)
        actions_in: torch.Tensor,  # (B, T_max, action_dim) teacher-forced (GT actions)
        lengths: torch.Tensor,     # (B,) int64 CPU tensor — lengths of actions_in
    ) -> torch.Tensor:
        """Teacher-forced decode. Return reconstructed actions (B, T_max, action_dim)."""
        B, T, _ = actions_in.shape
        h0, c0 = self._z_to_hidden(z)

        # Prepend start token; drop last GT action → decoder sees [start, a0, ..., a_{T-2}]
        start = self.start_token.expand(B, 1, -1)
        dec_in = torch.cat([start, actions_in[:, :-1, :]], dim=1)  # (B, T, action_dim)

        packed = pack_padded_sequence(dec_in, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.decoder_lstm(packed, (h0, c0))
        out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=T)
        return self.output_head(out)  # (B, T, action_dim)

    @torch.no_grad()
    def decode_autoregressive(
        self,
        z: torch.Tensor,   # (B, latent_dim)
        length: int,
    ) -> torch.Tensor:
        """Autoregressive decode (inference). Return (B, length, action_dim)."""
        B = z.size(0)
        h, c = self._z_to_hidden(z)
        x = self.start_token.expand(B, 1, -1)
        outputs = []
        for _ in range(length):
            out, (h, c) = self.decoder_lstm(x, (h, c))
            a = self.output_head(out)   # (B, 1, action_dim)
            outputs.append(a)
            x = a
        return torch.cat(outputs, dim=1)  # (B, length, action_dim)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        actions: torch.Tensor,   # (B, T_max, action_dim)
        lengths: torch.Tensor,   # (B,) int64
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (recon_actions, mu, logvar)."""
        mu, logvar = self.encode(actions, lengths)
        z = self.reparameterise(mu, logvar)
        recon = self.decode(z, actions, lengths)
        return recon, mu, logvar

    # ── Convenience encode/decode for numpy arrays ─────────────────────────────

    @torch.no_grad()
    def encode_numpy(
        self,
        actions: np.ndarray,   # (T, action_dim)
        device: str = "cpu",
    ) -> np.ndarray:
        """Encode a single skill. Returns z (latent_dim,) — the mean."""
        a = torch.from_numpy(actions).float().unsqueeze(0).to(device)
        l = torch.tensor([len(actions)], dtype=torch.long)
        mu, _ = self.encode(a, l)
        return mu.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def decode_numpy(
        self,
        z: np.ndarray,   # (latent_dim,)
        length: int,
        device: str = "cpu",
    ) -> np.ndarray:
        """Decode a latent code. Returns reconstructed actions (length, action_dim)."""
        z_t = torch.from_numpy(z).float().unsqueeze(0).to(device)
        recon = self.decode_autoregressive(z_t, length)
        return recon.squeeze(0).cpu().numpy()


# ── Loss ───────────────────────────────────────────────────────────────────────

def vae_loss(
    recon: torch.Tensor,    # (B, T_max, action_dim)
    target: torch.Tensor,   # (B, T_max, action_dim)
    mu: torch.Tensor,       # (B, latent_dim)
    logvar: torch.Tensor,   # (B, latent_dim)
    lengths: torch.Tensor,  # (B,) int64
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Masked reconstruction MSE + beta * KL divergence.

    Returns (total_loss, recon_loss, kl_loss).
    """
    B, T, D = target.shape
    # Build validity mask (B, T, 1)
    mask = torch.zeros(B, T, device=target.device)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1.0
    mask = mask.unsqueeze(-1)  # (B, T, 1)

    # Per-element MSE, averaged over valid positions
    sq_err = ((recon - target) ** 2) * mask            # (B, T, D)
    recon_loss = sq_err.sum() / (mask.sum() * D + 1e-8)

    # KL: mean over batch and latent dims
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


# ── Dataset ────────────────────────────────────────────────────────────────────

class SkillDataset(Dataset):
    """Dataset of variable-length skill segments.

    Each item is an action array:
        action_array : (T_i, action_dim)  float32 numpy
    """

    def __init__(
        self,
        segments: list[np.ndarray],
    ) -> None:
        self.segments = segments

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        actions = self.segments[idx]
        return (
            torch.from_numpy(actions.astype(np.float32)),
            len(actions),
        )

    @staticmethod
    def collate_fn(
        batch: list[tuple[torch.Tensor, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad to longest sequence in batch and return (actions, lengths)."""
        actions_list, lengths = zip(*batch)
        actions_pad = pad_sequence(actions_list, batch_first=True, padding_value=0.0)
        lengths_t   = torch.tensor(lengths, dtype=torch.long)
        return actions_pad, lengths_t


# ── Training config ────────────────────────────────────────────────────────────

@dataclass
class VAEConfig:
    action_dim: int = 7
    hidden_dim: int = 256
    latent_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    beta: float = 1.0          # KL weight in ELBO
    lr: float = 3e-4
    batch_size: int = 32
    epochs: int = 100
    grad_clip: float = 1.0
    device: str = "cuda"
    val_split: float = 0.1
    log_every: int = 10        # print every N epochs
    save_path: str | None = None
    checkpoint_every: int = 0  # 0 = disabled, N = save checkpoint every N epochs


# ── Training loop ──────────────────────────────────────────────────────────────

def train_skill_vae(
    segments: list[np.ndarray],
    cfg: VAEConfig,
    wandb_run=None,
    metadata: list[dict] | None = None,
) -> SkillVAE:
    """Train a SkillVAE on the given list of action_array segments.

    Returns the trained model (on CPU).
    """
    if len(segments) == 0:
        raise ValueError("No skill segments provided for VAE training.")

    print(f"[VAE] Training on {len(segments)} skill segments "
          f"(latent_dim={cfg.latent_dim}, hidden_dim={cfg.hidden_dim}, "
          f"epochs={cfg.epochs}, beta={cfg.beta})")

    # Infer dims from data if not set
    example_a = segments[0]
    action_dim = cfg.action_dim if cfg.action_dim > 0 else example_a.shape[-1]

    # Train / val split
    n_val = max(1, int(len(segments) * cfg.val_split))
    indices = np.random.permutation(len(segments))
    train_segs = [segments[i] for i in indices[n_val:]]
    val_segs   = [segments[i] for i in indices[:n_val]]

    train_ds = SkillDataset(train_segs)
    val_ds   = SkillDataset(val_segs)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=SkillDataset.collate_fn, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=SkillDataset.collate_fn,
    )

    model = SkillVAE(
        action_dim=action_dim,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(cfg.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[VAE] Parameters: {n_params:,}  |  action_dim={action_dim}")

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)

    best_val = math.inf

    for epoch in range(1, cfg.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        t_total = t_recon = t_kl = 0.0
        for actions, lengths in train_loader:
            actions = actions.to(cfg.device)
            lengths = lengths.to(cfg.device)

            recon, mu, logvar = model(actions, lengths)
            loss, recon_l, kl_l = vae_loss(recon, actions, mu, logvar, lengths, cfg.beta)

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

            t_total += loss.item()
            t_recon += recon_l.item()
            t_kl    += kl_l.item()

        scheduler.step()

        # ── Validation ─────────────────────────────────────────────────────────
        model.eval()
        v_total = 0.0
        with torch.no_grad():
            for actions, lengths in val_loader:
                actions = actions.to(cfg.device)
                recon, mu, logvar = model(actions, lengths)
                loss, _, _ = vae_loss(recon, actions, mu, logvar, lengths, cfg.beta)
                v_total += loss.item()

        n_train = len(train_loader)
        n_val_b = len(val_loader)
        log_dict = {
            "train/loss": t_total / n_train,
            "train/recon": t_recon / n_train,
            "train/kl": t_kl / n_train,
            "val/loss": v_total / n_val_b,
            "epoch": epoch,
        }
        if wandb_run is not None:
            wandb_run.log(log_dict)
        if epoch % cfg.log_every == 0 or epoch == 1:
            print(
                f"[VAE] epoch {epoch:4d}/{cfg.epochs}  "
                f"train: {log_dict['train/loss']:.4f} "
                f"(recon={log_dict['train/recon']:.4f}, kl={log_dict['train/kl']:.4f})  "
                f"val: {log_dict['val/loss']:.4f}"
            )

        if v_total < best_val:
            best_val = v_total
            if cfg.save_path:
                torch.save(
                    {"model_state": model.state_dict(), "cfg": cfg},
                    cfg.save_path,
                )

        if cfg.checkpoint_every > 0 and epoch % cfg.checkpoint_every == 0 and cfg.save_path:
            ckpt_path = cfg.save_path.replace(".pt", f"_epoch{epoch:04d}.pt")
            torch.save({"model_state": model.state_dict(), "cfg": cfg, "epoch": epoch}, ckpt_path)
            # Also save latent codes at this checkpoint
            model.eval()
            codes = []
            with torch.no_grad():
                for acts in segments:
                    z = model.encode_numpy(acts, device=cfg.device)
                    codes.append(z)
            latents_ckpt_path = cfg.save_path.replace(".pt", f"_latents_epoch{epoch:04d}.npz")
            save_dict: dict = {"latents": np.stack(codes)}
            if metadata is not None:
                for key in ("episode_id", "skill_index", "frame_start", "frame_end", "length"):
                    save_dict[key] = np.array([m[key] for m in metadata])
            np.savez(latents_ckpt_path, **save_dict)
            model.train()
            print(f"[VAE] Checkpoint latents saved → {latents_ckpt_path}")

    model = model.cpu()
    print(f"[VAE] Training complete. Best val loss: {best_val/len(val_loader):.4f}")
    return model


# ── Encode skill segments ──────────────────────────────────────────────────────

def encode_skills(
    model: SkillVAE,
    segments: list[np.ndarray],
    device: str = "cpu",
) -> np.ndarray:
    """Encode all segments and return latent codes (N, latent_dim)."""
    model = model.to(device).eval()
    codes = []
    for actions in segments:
        z = model.encode_numpy(actions, device=device)
        codes.append(z)
    return np.stack(codes)


# ── Utility: extract skill segments from episode data ─────────────────────────

def extract_skill_segments(
    gt_actions: np.ndarray,     # (T, action_dim)
    states: np.ndarray,         # (T, state_dim)
    boundary_ts: list[int],     # replan-frame indices of skill boundaries (from SBD)
    ep_end: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cut (gt_actions, states) into skill segments at boundary_ts.

    boundary_ts are indices relative to the start of gt_actions (i.e. replan-chunk
    indices × replan_interval).  Returns a list of (action_seg, state_seg) tuples,
    one per skill.
    """
    T = len(gt_actions)
    if ep_end is None:
        ep_end = T

    cuts = sorted(set(boundary_ts))
    # Build segment start/end pairs: [0, cut0, cut1, ..., T]
    breakpoints = [0] + [min(c, T) for c in cuts] + [T]
    breakpoints = sorted(set(breakpoints))

    segments = []
    for i in range(len(breakpoints) - 1):
        s, e = breakpoints[i], breakpoints[i + 1]
        if e - s < 2:
            continue  # skip degenerate single-frame "skills"
        segments.append((
            gt_actions[s:e].astype(np.float32),
            states[s:e].astype(np.float32),
        ))
    return segments
