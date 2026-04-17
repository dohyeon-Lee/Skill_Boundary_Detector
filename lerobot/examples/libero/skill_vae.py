"""
Skill VAE — BiLSTM Variational Autoencoder for variable-length skill trajectories.

Architecture:
  Encoder : BiLSTM over action sequence
            → last hidden (fwd + bwd concat) → Linear → (mu, logvar)
  Decoder : LSTM with h0/c0 initialised from z
            → teacher-forced on GT actions during training
            → autoregressive at inference time
            → two heads: output_head (action) + stop_head (stop probability)
  Loss    : masked MSE reconstruction + stop BCE + beta-weighted KL divergence

Variable-length handling:
  Sequences within a batch are right-padded to the longest sequence.
  pack_padded_sequence / pad_packed_sequence are used so padding frames
  do not affect LSTM hidden states or loss computation.
  At inference, generation stops when stop_prob > stop_threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset


# ── Model ──────────────────────────────────────────────────────────────────────

class SkillVAE(nn.Module):
    """BiLSTM Variational Autoencoder for skill action trajectories."""

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        stop_threshold: float = 0.5,
        max_decode_steps: int = 500,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.state_dim  = state_dim
        self.num_layers = num_layers
        self.stop_threshold = stop_threshold
        self.max_decode_steps = max_decode_steps

        enc_drop = dropout if num_layers > 1 else 0.0
        self.encoder_lstm = nn.LSTM(
            action_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=enc_drop,
        )
        self.mu_head     = nn.Linear(hidden_dim * 2, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim * 2, latent_dim)

        dec_drop = dropout if num_layers > 1 else 0.0
        dec_cond_dim = latent_dim + state_dim
        self.z_to_h = nn.Linear(dec_cond_dim, hidden_dim * num_layers)
        self.z_to_c = nn.Linear(dec_cond_dim, hidden_dim * num_layers)
        self.decoder_lstm = nn.LSTM(
            action_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dec_drop,
        )
        self.output_head = nn.Linear(hidden_dim, action_dim)
        self.stop_head   = nn.Linear(hidden_dim, 1)

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
        actions: torch.Tensor,   # (B, T_max, action_dim)
        lengths: torch.Tensor,   # (B,) int64 CPU tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, logvar), each (B, latent_dim)."""
        packed = pack_padded_sequence(actions, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.encoder_lstm(packed)
        h_fwd = h[-2]
        h_bwd = h[-1]
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)
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

    def _z_to_hidden(self, z: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = z.size(0)
        cond = torch.cat([z, state], dim=-1)
        h0 = self.z_to_h(cond).view(B, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        c0 = self.z_to_c(cond).view(B, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        return h0, c0

    def decode(
        self,
        z: torch.Tensor,                    # (B, latent_dim)
        actions_in: torch.Tensor,  # (B, T_max, action_dim) teacher-forced
        lengths: torch.Tensor,    # (B,) int64 CPU tensor
        state: torch.Tensor,      # (B, state_dim) initial eef state
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced decode.

        Returns:
            recon  : (B, T_max, action_dim)
            stop_logits : (B, T_max, 1)
        """
        B, T, _ = actions_in.shape
        h0, c0 = self._z_to_hidden(z, state)

        start = self.start_token.expand(B, 1, -1)
        dec_in = torch.cat([start, actions_in[:, :-1, :]], dim=1)

        packed = pack_padded_sequence(dec_in, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.decoder_lstm(packed, (h0, c0))
        out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=T)

        recon       = self.output_head(out)   # (B, T, action_dim)
        stop_logits = self.stop_head(out)     # (B, T, 1)
        return recon, stop_logits

    @torch.no_grad()
    def decode_autoregressive(
        self,
        z: torch.Tensor,      # (B, latent_dim)
        state: torch.Tensor,  # (B, state_dim) initial eef state
    ) -> torch.Tensor:
        """Autoregressive decode. Stops when stop_prob > threshold or max_decode_steps.

        Returns:
            actions : (B, T, action_dim)  where T varies per sample in batch
                      (all samples stop at the same step — the earliest stop in batch)
        """
        B = z.size(0)
        h, c = self._z_to_hidden(z, state)
        x = self.start_token.expand(B, 1, -1)
        outputs = []
        for _ in range(self.max_decode_steps):
            out, (h, c) = self.decoder_lstm(x, (h, c))
            a    = self.output_head(out)              # (B, 1, action_dim)
            stop = torch.sigmoid(self.stop_head(out)) # (B, 1, 1)
            outputs.append(a)
            x = a
            if stop.max().item() > self.stop_threshold:
                break
        return torch.cat(outputs, dim=1)  # (B, T, action_dim)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        actions: torch.Tensor,  # (B, T_max, action_dim)
        lengths: torch.Tensor,  # (B,) int64
        state: torch.Tensor,    # (B, state_dim) initial eef state
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (recon_actions, stop_logits, mu, logvar)."""
        mu, logvar = self.encode(actions, lengths)
        z = self.reparameterise(mu, logvar)
        recon, stop_logits = self.decode(z, actions, lengths, state)
        return recon, stop_logits, mu, logvar

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
        z: np.ndarray,      # (latent_dim,)
        state: np.ndarray,  # (state_dim,) initial eef state
        device: str = "cpu",
    ) -> np.ndarray:
        """Decode a latent code. Returns reconstructed actions (T, action_dim)."""
        z_t = torch.from_numpy(z).float().unsqueeze(0).to(device)
        s_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        recon = self.decode_autoregressive(z_t, s_t)
        return recon.squeeze(0).cpu().numpy()


# ── Loss ───────────────────────────────────────────────────────────────────────

def vae_loss(
    recon: torch.Tensor,        # (B, T_max, action_dim)
    stop_logits: torch.Tensor,  # (B, T_max, 1)
    target: torch.Tensor,       # (B, T_max, action_dim)
    mu: torch.Tensor,           # (B, latent_dim)
    logvar: torch.Tensor,       # (B, latent_dim)
    lengths: torch.Tensor,      # (B,) int64
    beta: float = 1.0,
    stop_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Masked reconstruction MSE + stop BCE + beta * KL divergence.

    Returns (total_loss, recon_loss, stop_loss, kl_loss).
    """
    B, T, D = target.shape

    # Validity mask (B, T, 1)
    arange = torch.arange(T, device=target.device).unsqueeze(0)  # (1, T)
    lengths_dev = lengths.to(target.device)
    mask = (arange < lengths_dev.unsqueeze(1)).unsqueeze(-1).float()  # (B, T, 1)

    # Reconstruction loss (masked MSE)
    recon_loss = (((recon - target) ** 2) * mask).sum() / (mask.sum() * D + 1e-8)

    # Stop loss: target is 1 at the last valid step, 0 elsewhere
    stop_target = (arange == (lengths_dev.unsqueeze(1) - 1)).unsqueeze(-1).float()  # (B, T, 1)
    stop_loss = (F.binary_cross_entropy_with_logits(stop_logits, stop_target, reduction="none") * mask).sum() / (mask.sum() + 1e-8)

    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total = recon_loss + stop_weight * stop_loss + beta * kl_loss
    return total, recon_loss, stop_loss, kl_loss


# ── Dataset ────────────────────────────────────────────────────────────────────

class SkillDataset(Dataset):
    """Dataset of variable-length skill segments.

    segments: list of action arrays (T, action_dim)
    init_states: optional list of initial eef states (state_dim,), one per segment
    """

    def __init__(
        self,
        segments: list[np.ndarray],
        init_states: list[np.ndarray],
    ) -> None:
        self.segments    = segments
        self.init_states = init_states

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        actions = self.segments[idx]
        state   = torch.from_numpy(self.init_states[idx].astype(np.float32))
        return torch.from_numpy(actions.astype(np.float32)), len(actions), state

    @staticmethod
    def collate_fn(
        batch: list[tuple[torch.Tensor, int, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actions_list, lengths, states = zip(*batch)
        actions_pad = pad_sequence(actions_list, batch_first=True, padding_value=0.0)
        lengths_t   = torch.tensor(lengths, dtype=torch.long)
        states_t    = torch.stack(states)
        return actions_pad, lengths_t, states_t


# ── Training config ────────────────────────────────────────────────────────────

@dataclass
class VAEConfig:
    action_dim: int = 7
    state_dim: int = 0
    """Dimension of initial eef state fed to decoder. 0 = not used."""
    hidden_dim: int = 256
    latent_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    beta: float = 1.0
    stop_weight: float = 1.0
    stop_threshold: float = 0.5
    max_decode_steps: int = 500
    lr: float = 3e-4
    batch_size: int = 32
    epochs: int = 100
    grad_clip: float = 1.0
    device: str = "cuda"
    val_split: float = 0.1
    log_every: int = 10
    save_path: str | None = None
    checkpoint_every: int = 0


# ── Training loop ──────────────────────────────────────────────────────────────

def train_skill_vae(
    segments: list[np.ndarray],
    cfg: VAEConfig,
    wandb_run=None,
    metadata: list[dict] | None = None,
    init_states: list[np.ndarray],
) -> SkillVAE:
    if len(segments) == 0:
        raise ValueError("No skill segments provided for VAE training.")

    action_dim = cfg.action_dim if cfg.action_dim > 0 else segments[0].shape[-1]

    print(f"[VAE] Training on {len(segments)} skill segments "
          f"(latent_dim={cfg.latent_dim}, hidden_dim={cfg.hidden_dim}, "
          f"epochs={cfg.epochs}, beta={cfg.beta}, stop_weight={cfg.stop_weight}, "
          f"state_dim={cfg.state_dim})")

    n_val = max(1, int(len(segments) * cfg.val_split))
    indices = np.random.permutation(len(segments))
    train_segs   = [segments[i] for i in indices[n_val:]]
    val_segs     = [segments[i] for i in indices[:n_val]]
    train_states = [init_states[i] for i in indices[n_val:]]
    val_states   = [init_states[i] for i in indices[:n_val]]

    train_loader = DataLoader(
        SkillDataset(train_segs, train_states), batch_size=cfg.batch_size, shuffle=True,
        collate_fn=SkillDataset.collate_fn, drop_last=False,
    )
    val_loader = DataLoader(
        SkillDataset(val_segs, val_states), batch_size=cfg.batch_size, shuffle=False,
        collate_fn=SkillDataset.collate_fn,
    )

    model = SkillVAE(
        action_dim=action_dim,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        state_dim=cfg.state_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        stop_threshold=cfg.stop_threshold,
        max_decode_steps=cfg.max_decode_steps,
    ).to(cfg.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[VAE] Parameters: {n_params:,}  |  action_dim={action_dim}")

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)

    best_val = math.inf

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t_total = t_recon = t_stop = t_kl = 0.0
        for actions, lengths, states in train_loader:
            actions = actions.to(cfg.device)
            states  = states.to(cfg.device)

            recon, stop_logits, mu, logvar = model(actions, lengths, states)
            loss, recon_l, stop_l, kl_l = vae_loss(
                recon, stop_logits, actions, mu, logvar, lengths, cfg.beta, cfg.stop_weight
            )

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

            t_total += loss.item()
            t_recon += recon_l.item()
            t_stop  += stop_l.item()
            t_kl    += kl_l.item()

        scheduler.step()

        model.eval()
        v_total = 0.0
        with torch.no_grad():
            for actions, lengths, states in val_loader:
                actions = actions.to(cfg.device)
                states  = states.to(cfg.device)
                recon, stop_logits, mu, logvar = model(actions, lengths, states)
                loss, _, _, _ = vae_loss(
                    recon, stop_logits, actions, mu, logvar, lengths, cfg.beta, cfg.stop_weight
                )
                v_total += loss.item()

        n_train = len(train_loader)
        n_val_b = len(val_loader)
        log_dict = {
            "train/loss":  t_total / n_train,
            "train/recon": t_recon / n_train,
            "train/stop":  t_stop  / n_train,
            "train/kl":    t_kl    / n_train,
            "val/loss":    v_total / n_val_b,
            "epoch": epoch,
        }
        if wandb_run is not None:
            wandb_run.log(log_dict)
        if epoch % cfg.log_every == 0 or epoch == 1:
            print(
                f"[VAE] epoch {epoch:4d}/{cfg.epochs}  "
                f"train: {log_dict['train/loss']:.4f} "
                f"(recon={log_dict['train/recon']:.4f}, "
                f"stop={log_dict['train/stop']:.4f}, "
                f"kl={log_dict['train/kl']:.4f})  "
                f"val: {log_dict['val/loss']:.4f}"
            )

        if v_total < best_val:
            best_val = v_total
            if cfg.save_path:
                torch.save({"model_state": model.state_dict(), "cfg": cfg}, cfg.save_path)

        if cfg.checkpoint_every > 0 and epoch % cfg.checkpoint_every == 0 and cfg.save_path:
            ckpt_path = cfg.save_path.replace(".pt", f"_epoch{epoch:04d}.pt")
            torch.save({"model_state": model.state_dict(), "cfg": cfg, "epoch": epoch}, ckpt_path)
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

