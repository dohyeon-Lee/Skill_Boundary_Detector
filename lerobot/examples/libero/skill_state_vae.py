"""
Skill State VAE — BiLSTM Variational Autoencoder for variable-length skill trajectories.

Architecture:
  Encoder : BiLSTM over (action + state) concatenated sequence
            → last hidden (fwd + bwd concat) → Linear → (mu, logvar)
  Decoder : LSTM with h0/c0 initialised from z
            → teacher-forced on GT (action + state) during training
            → autoregressive at inference time
            → output head splits into action and state reconstructions
  Loss    : masked MSE reconstruction (action + state) + beta-weighted KL divergence

Input  : action trajectory  (T, action_dim)
         state trajectory   (T, state_dim)
Output : reconstructed action trajectory (T, action_dim)
         reconstructed state trajectory  (T, state_dim)
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

class SkillStateVAE(nn.Module):
    """BiLSTM VAE that encodes and reconstructs both action and state trajectories."""

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.state_dim  = state_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        enc_input_dim = action_dim + state_dim
        enc_drop = dropout if num_layers > 1 else 0.0
        self.encoder_lstm = nn.LSTM(
            enc_input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=enc_drop,
        )
        self.mu_head     = nn.Linear(hidden_dim * 2, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim * 2, latent_dim)

        dec_drop = dropout if num_layers > 1 else 0.0
        self.z_to_h = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.z_to_c = nn.Linear(latent_dim, hidden_dim * num_layers)
        # Decoder input: (action + state) concatenated
        self.decoder_lstm = nn.LSTM(
            action_dim + state_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dec_drop,
        )
        # Split output into action and state heads
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.state_head  = nn.Linear(hidden_dim, state_dim)

        # Learned start token: zeros for (action + state)
        self.start_token = nn.Parameter(torch.zeros(1, 1, action_dim + state_dim))

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
        states: torch.Tensor,    # (B, T_max, state_dim)
        lengths: torch.Tensor,   # (B,) int64 CPU tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, logvar), each (B, latent_dim)."""
        x = torch.cat([actions, states], dim=-1)  # (B, T, action_dim+state_dim)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.encoder_lstm(packed)
        h_fwd = h[-2]
        h_bwd = h[-1]
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)  # (B, hidden_dim*2)
        return self.mu_head(h_cat), self.logvar_head(h_cat)

    # ── Reparameterisation ─────────────────────────────────────────────────────

    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    # ── Decoder ────────────────────────────────────────────────────────────────

    def _z_to_hidden(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = z.size(0)
        h0 = self.z_to_h(z).view(B, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        c0 = self.z_to_c(z).view(B, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        return h0, c0

    def decode(
        self,
        z: torch.Tensor,          # (B, latent_dim)
        actions_in: torch.Tensor, # (B, T_max, action_dim)
        states_in: torch.Tensor,  # (B, T_max, state_dim)
        lengths: torch.Tensor,    # (B,) int64 CPU tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced decode. Return (recon_actions, recon_states), each (B, T_max, dim)."""
        B, T, _ = actions_in.shape
        h0, c0 = self._z_to_hidden(z)

        x = torch.cat([actions_in, states_in], dim=-1)  # (B, T, action_dim+state_dim)
        start = self.start_token.expand(B, 1, -1)
        dec_in = torch.cat([start, x[:, :-1, :]], dim=1)  # (B, T, action_dim+state_dim)

        packed = pack_padded_sequence(dec_in, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.decoder_lstm(packed, (h0, c0))
        out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=T)

        return self.action_head(out), self.state_head(out)

    @torch.no_grad()
    def decode_autoregressive(
        self,
        z: torch.Tensor,   # (B, latent_dim)
        length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive decode (inference). Return (actions, states), each (B, length, dim)."""
        B = z.size(0)
        h, c = self._z_to_hidden(z)
        x = self.start_token.expand(B, 1, -1)
        action_outputs, state_outputs = [], []
        for _ in range(length):
            out, (h, c) = self.decoder_lstm(x, (h, c))
            a = self.action_head(out)  # (B, 1, action_dim)
            s = self.state_head(out)   # (B, 1, state_dim)
            action_outputs.append(a)
            state_outputs.append(s)
            x = torch.cat([a, s], dim=-1)
        return torch.cat(action_outputs, dim=1), torch.cat(state_outputs, dim=1)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        actions: torch.Tensor,   # (B, T_max, action_dim)
        states: torch.Tensor,    # (B, T_max, state_dim)
        lengths: torch.Tensor,   # (B,) int64
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (recon_actions, recon_states, mu, logvar)."""
        mu, logvar = self.encode(actions, states, lengths)
        z = self.reparameterise(mu, logvar)
        recon_actions, recon_states = self.decode(z, actions, states, lengths)
        return recon_actions, recon_states, mu, logvar

    # ── Convenience encode/decode for numpy arrays ─────────────────────────────

    @torch.no_grad()
    def encode_numpy(
        self,
        actions: np.ndarray,   # (T, action_dim)
        states: np.ndarray,    # (T, state_dim)
        device: str = "cpu",
    ) -> np.ndarray:
        """Encode a single skill. Returns z (latent_dim,) — the mean."""
        a = torch.from_numpy(actions).float().unsqueeze(0).to(device)
        s = torch.from_numpy(states).float().unsqueeze(0).to(device)
        l = torch.tensor([len(actions)], dtype=torch.long)
        mu, _ = self.encode(a, s, l)
        return mu.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def decode_numpy(
        self,
        z: np.ndarray,   # (latent_dim,)
        length: int,
        device: str = "cpu",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decode a latent code. Returns (actions, states), each (length, dim)."""
        z_t = torch.from_numpy(z).float().unsqueeze(0).to(device)
        recon_a, recon_s = self.decode_autoregressive(z_t, length)
        return recon_a.squeeze(0).cpu().numpy(), recon_s.squeeze(0).cpu().numpy()


# ── Loss ───────────────────────────────────────────────────────────────────────

def vae_loss(
    recon_actions: torch.Tensor,  # (B, T_max, action_dim)
    recon_states: torch.Tensor,   # (B, T_max, state_dim)
    target_actions: torch.Tensor, # (B, T_max, action_dim)
    target_states: torch.Tensor,  # (B, T_max, state_dim)
    mu: torch.Tensor,             # (B, latent_dim)
    logvar: torch.Tensor,         # (B, latent_dim)
    lengths: torch.Tensor,        # (B,) int64
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Masked reconstruction MSE (action + state) + beta * KL divergence.

    Returns (total_loss, recon_loss, kl_loss).
    """
    B, T, _ = target_actions.shape
    mask = torch.zeros(B, T, device=target_actions.device)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1.0
    mask = mask.unsqueeze(-1)  # (B, T, 1)

    def masked_mse(recon, target):
        D = target.shape[-1]
        return ((recon - target) ** 2 * mask).sum() / (mask.sum() * D + 1e-8)

    recon_loss = masked_mse(recon_actions, target_actions) + masked_mse(recon_states, target_states)
    kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total      = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


# ── Dataset ────────────────────────────────────────────────────────────────────

class SkillStateDataset(Dataset):
    """Dataset of variable-length skill segments.

    Each item is a (action_array, state_array) pair:
        action_array : (T_i, action_dim)  float32 numpy
        state_array  : (T_i, state_dim)   float32 numpy
    """

    def __init__(self, segments: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self.segments = segments

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        actions, states = self.segments[idx]
        return (
            torch.from_numpy(actions.astype(np.float32)),
            torch.from_numpy(states.astype(np.float32)),
            len(actions),
        )

    @staticmethod
    def collate_fn(
        batch: list[tuple[torch.Tensor, torch.Tensor, int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actions_list, states_list, lengths = zip(*batch)
        actions_pad = pad_sequence(actions_list, batch_first=True, padding_value=0.0)
        states_pad  = pad_sequence(states_list,  batch_first=True, padding_value=0.0)
        lengths_t   = torch.tensor(lengths, dtype=torch.long)
        return actions_pad, states_pad, lengths_t


# ── Training config ────────────────────────────────────────────────────────────

@dataclass
class StateVAEConfig:
    action_dim: int = 7
    state_dim: int = 8
    hidden_dim: int = 256
    latent_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    beta: float = 1.0
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

def train_skill_state_vae(
    segments: list[tuple[np.ndarray, np.ndarray]],
    cfg: StateVAEConfig,
    wandb_run=None,
    metadata: list[dict] | None = None,
) -> SkillStateVAE:
    """Train a SkillStateVAE on the given list of (action_array, state_array) segments."""
    if len(segments) == 0:
        raise ValueError("No skill segments provided for VAE training.")

    print(f"[StateVAE] Training on {len(segments)} skill segments "
          f"(latent_dim={cfg.latent_dim}, hidden_dim={cfg.hidden_dim}, "
          f"epochs={cfg.epochs}, beta={cfg.beta})")

    example_a, example_s = segments[0]
    action_dim = cfg.action_dim if cfg.action_dim > 0 else example_a.shape[-1]
    state_dim  = cfg.state_dim  if cfg.state_dim  > 0 else example_s.shape[-1]

    n_val = max(1, int(len(segments) * cfg.val_split))
    indices = np.random.permutation(len(segments))
    train_segs = [segments[i] for i in indices[n_val:]]
    val_segs   = [segments[i] for i in indices[:n_val]]

    train_ds = SkillStateDataset(train_segs)
    val_ds   = SkillStateDataset(val_segs)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=SkillStateDataset.collate_fn, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=SkillStateDataset.collate_fn,
    )

    model = SkillStateVAE(
        action_dim=action_dim,
        state_dim=state_dim,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(cfg.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[StateVAE] Parameters: {n_params:,}  |  action_dim={action_dim}  state_dim={state_dim}")

    optim     = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)

    best_val = math.inf

    for epoch in range(1, cfg.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        t_total = t_recon = t_kl = 0.0
        for actions, states, lengths in train_loader:
            actions = actions.to(cfg.device)
            states  = states.to(cfg.device)
            lengths = lengths.to(cfg.device)

            recon_a, recon_s, mu, logvar = model(actions, states, lengths)
            loss, recon_l, kl_l = vae_loss(recon_a, recon_s, actions, states, mu, logvar, lengths, cfg.beta)

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
            for actions, states, lengths in val_loader:
                actions = actions.to(cfg.device)
                states  = states.to(cfg.device)
                recon_a, recon_s, mu, logvar = model(actions, states, lengths)
                loss, _, _ = vae_loss(recon_a, recon_s, actions, states, mu, logvar, lengths, cfg.beta)
                v_total += loss.item()

        n_train = len(train_loader)
        n_val_b = len(val_loader)
        log_dict = {
            "train/loss":  t_total / n_train,
            "train/recon": t_recon / n_train,
            "train/kl":    t_kl    / n_train,
            "val/loss":    v_total / n_val_b,
            "epoch":       epoch,
        }
        if wandb_run is not None:
            wandb_run.log(log_dict)
        if epoch % cfg.log_every == 0 or epoch == 1:
            print(
                f"[StateVAE] epoch {epoch:4d}/{cfg.epochs}  "
                f"train: {log_dict['train/loss']:.4f} "
                f"(recon={log_dict['train/recon']:.4f}, kl={log_dict['train/kl']:.4f})  "
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
                for acts, sts in segments:
                    z = model.encode_numpy(acts, sts, device=cfg.device)
                    codes.append(z)
            latents_ckpt_path = cfg.save_path.replace(".pt", f"_latents_epoch{epoch:04d}.npz")
            save_dict: dict = {"latents": np.stack(codes)}
            if metadata is not None:
                for key in ("episode_id", "skill_index", "frame_start", "frame_end", "length"):
                    save_dict[key] = np.array([m[key] for m in metadata])
            np.savez(latents_ckpt_path, **save_dict)
            model.train()
            print(f"[StateVAE] Checkpoint latents saved → {latents_ckpt_path}")

    model = model.cpu()
    print(f"[StateVAE] Training complete. Best val loss: {best_val/len(val_loader):.4f}")
    return model


# ── Encode skill segments ──────────────────────────────────────────────────────

def encode_skills(
    model: SkillStateVAE,
    segments: list[tuple[np.ndarray, np.ndarray]],
    device: str = "cpu",
) -> np.ndarray:
    """Encode all segments and return latent codes (N, latent_dim)."""
    model = model.to(device).eval()
    codes = []
    for actions, states in segments:
        z = model.encode_numpy(actions, states, device=device)
        codes.append(z)
    return np.stack(codes)
