"""One-shot trajectory FSQ autoencoder (original spline VQ-AE style).

Unlike FSQ.py v3 (per-timestep action-chunk reconstructor + image query
terminator), this variant revives the original spline autoencoder contract:

    encoder input  : spline control points of the mean-XYZ-grounded state
                     trajectory [+ absolute mean-XYZ token in optimal mode]
                     + one length token
    decoder output : the SAME normalized control points + normalized length,
                     reconstructed in ONE shot from z alone

There are no images, no terminator, and no per-timestep sampling — a pure
autoencoder over the encoder-input representation. ``spline_decode`` turns
reconstructed control points + length back into a full mean-XYZ-grounded
trajectory. The grounding position is intentionally NOT reconstructed: in
optimal mode it conditions the encoder only, and absolute XYZ is recovered by
adding the caller's known trajectory mean after decoding.

The encoder is the exact ``SplineFSQEncoder`` used by FSQ v3, so its weights
stay transplant-compatible (state-dict prefix ``encoder.*``).
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import make_interp_spline
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from FSQ import (
    ENCODER_GROUNDING_CONVENTION,
    FSQ,
    N_GRIPPER_DIMS,
    ActionSeqEncoder,
    LengthFreeSplineFSQEncoder,
    OneShotTrajectoryDecoder,
    SplineFSQEncoder,
    TokenTransformerPool,
    fsq_entropy_terms,
    _boundary_margin_metrics,
    _code_assignment_stability,
    _collect_code_assignments,
    encoder_grounding_position,
    fsq_lr_factor,
    prepare_encoder_trajectory,
    spline_encode,
)

ORIGINAL_FORMAT_VERSION = 1


class BSQ(nn.Module):
    """Binary Spherical Quantization (Zhao et al., 2024, BSQ-ViT).

    z -> u = z/|z| on the unit sphere; each dimension binarizes to
    sign(u_i)/sqrt(L), so the implicit codebook is the 2^L hypercube corners
    projected onto the sphere. Interface-compatible with FSQ.FSQ (forward /
    normalized / code_to_normalized / boundary_margin / codebook_size) so the
    encoder, codebook diagnostics, and eval reuse everything unchanged.
    """

    def __init__(self, code_dim: int, inv_temperature: float = 10.0):
        super().__init__()
        if code_dim < 2:
            raise ValueError(f"BSQ code_dim must be >= 2, got {code_dim}.")
        self.code_dim = int(code_dim)
        self.codebook_size = 2 ** self.code_dim
        self.latent_dim = self.code_dim
        self.inv_temperature = float(inv_temperature)
        self.register_buffer(
            "bit_weights", (2 ** torch.arange(self.code_dim)).long(), persistent=False
        )

    def unit(self, z: Tensor) -> Tensor:
        return F.normalize(z.float(), dim=-1, eps=1e-8).to(z.dtype)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        u = self.unit(z)
        signs = torch.where(u >= 0, torch.ones_like(u), -torch.ones_like(u))
        u_hat = signs / math.sqrt(self.code_dim)
        z_q = u + (u_hat - u).detach()
        index = ((u >= 0).long() * self.bit_weights).sum(dim=-1)
        return z_q, index

    def normalized(self, z_q: Tensor) -> Tensor:
        """Corner coordinate ±1/sqrt(L) -> ±1 per bit (FSQ's centered-grid scale)."""
        return z_q * math.sqrt(self.code_dim)

    def code_to_normalized(self, code: Tensor) -> Tensor:
        idx = code.view(-1, 1).long()
        bits = torch.div(idx, self.bit_weights[None].to(idx.device), rounding_mode="floor") % 2
        return bits.float() * 2.0 - 1.0

    def boundary_margin(self, z: Tensor) -> Tensor:
        """Distance to the nearest sign flip, on FSQ's 0..0.5 scale.

        |u_i| = 0 sits exactly on a bit boundary; the corner magnitude
        1/sqrt(L) maps to 0.5 (bin center), so FSQ's margin diagnostics and
        near-boundary thresholds read identically.
        """
        u = self.unit(z.float())
        return (u.abs() * math.sqrt(self.code_dim) * 0.5).clamp(0.0, 0.5)


def bsq_entropy_terms(
    u_cont: Tensor, inv_temperature: float, *, joint_dataset: bool = False
) -> tuple[Tensor, Tensor]:
    """LFQ/BSQ entropy objective terms from the per-bit Bernoulli q(c|u).

    Returns (sample_entropy, dataset_entropy) in nats. Minimize the first
    (per-sample bit confidence — pushes |u_i| off the boundary), maximize the
    second (code-usage diversity). Weighted separately by the caller so a
    confidence-only setup (dataset weight 0) stays expressible.

    The sample entropy is exact either way (the per-sample distribution IS the
    bit product). ``joint_dataset=False`` uses the paper's O(L) per-bit
    marginal approximation of H(E[q]) — an UPPER bound (subadditivity) that is
    blind to bit correlations: an antipodal code pair at 50/50 maximizes every
    marginal while the true joint entropy is one bit. ``joint_dataset=True``
    enumerates all 2^L codes and computes the paper's stated objective exactly
    (feasible for small code_dim; the approximation exists for L=18~36).
    """
    p = torch.sigmoid(2.0 * inv_temperature * u_cont.float()).clamp(1e-6, 1.0 - 1e-6)
    bit_entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
    sample_entropy = bit_entropy.sum(dim=-1).mean()
    if not joint_dataset:
        p_bar = p.mean(dim=0)
        dataset_entropy = -(p_bar * p_bar.log() + (1.0 - p_bar) * (1.0 - p_bar).log()).sum()
        return sample_entropy, dataset_entropy
    code_dim = u_cont.shape[-1]
    codes = torch.arange(2 ** code_dim, device=u_cont.device)
    bits = ((codes[:, None] >> torch.arange(code_dim, device=u_cont.device)) & 1).float()
    # log q(c|u) = sum_i [ c_i log p_i + (1-c_i) log(1-p_i) ]  -> (B, 2^L) via matmul
    log_q = p.log() @ bits.T + (1.0 - p).log() @ (1.0 - bits).T
    q_bar = log_q.exp().mean(dim=0).clamp_min(1e-12)
    dataset_entropy = -(q_bar * q_bar.log()).sum()
    return sample_entropy, dataset_entropy


# -----------------------------------------------------------------------------
# Spline decoding (control points + length -> trajectory)
# -----------------------------------------------------------------------------


def spline_decode(ctrl_pts: np.ndarray, length: int, degree: int) -> np.ndarray:
    """Control points + length -> reconstructed trajectory ``(length, dim)``.

    Mirrors ``FSQ.spline_encode``: trailing gripper-state dims use a linear
    spline, pose dims use the configured degree.
    """
    ctrl_pts = np.asarray(ctrl_pts, dtype=np.float32)
    n_control, dim = ctrl_pts.shape
    length = int(length)
    if length < 1:
        raise ValueError(f"Decoded length must be >= 1, got {length}.")
    if length == 1:
        return ctrl_pts[:1].copy()
    t_ctrl = np.linspace(0.0, 1.0, n_control)
    t_out = np.linspace(0.0, 1.0, length)
    recon = np.zeros((length, dim), dtype=np.float32)
    for d in range(dim):
        k = 1 if d >= dim - N_GRIPPER_DIMS else degree
        k = min(k, n_control - 1)
        recon[:, d] = make_interp_spline(t_ctrl, ctrl_pts[:, d], k=k)(t_out)
    return recon


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class FSQOriginalConfig:
    format_version: int = ORIGINAL_FORMAT_VERSION
    enc_dim: int = 8
    n_control: int = 30
    spline_degree: int = 3
    encoder_input_mode: str = "zero_grounded"
    encoder_grounding_convention: str = ENCODER_GROUNDING_CONVENTION
    hidden_dim: int = 256
    fsq_levels: list[int] = field(default_factory=lambda: [3, 3, 3])
    quantizer: str = "fsq"
    """fsq: FSQ grid over fsq_levels. bsq: Binary Spherical Quantization —
    fsq_levels is ignored and the codebook is 2^bsq_code_dim sphere corners."""
    bsq_code_dim: int = 5
    bsq_inv_temperature: float = 10.0
    """Soft-bit sharpness for the BSQ entropy terms (paper often uses 100;
    softer values spread the entropy gradient over more samples)."""
    bsq_entropy_conf_weight: float = 0.1
    """Weight of per-sample entropy MINIMIZATION (bit confidence; pushes u off
    bit boundaries). Set 0 to disable."""
    bsq_entropy_div_weight: float = 0.1
    """Weight of batch entropy MAXIMIZATION (balanced bit usage). The paper's
    gamma=1 corresponds to conf and div weights being equal; 0 = confidence-only."""
    bsq_entropy_joint: bool = True
    """Default (and the project standard): compute the dataset-entropy term
    EXACTLY over all codes (the paper's stated objective). The per-dim
    marginal approximation is an upper bound blind to inter-dim correlations
    (measured antipodal-pair collapse), so False exists only as an internal
    fallback for code_dim > 14 — it is deliberately not exposed in the yamls."""
    fsq_entropy: bool = False
    """FSQ quantizer only: apply the BSQ-style entropy terms (per-dim soft
    level assignment; conf pushes samples off rounding boundaries, div spreads
    code usage), reusing bsq_inv_temperature / bsq_entropy_*_weight /
    bsq_entropy_joint. The attribution ablation for the FSQ-vs-BSQ gap:
    if FSQ+entropy matches BSQ stability, the loss — not the sphere geometry —
    is what matters."""
    bsq_entropy_cov_weight: float = 0.0
    """COVERAGE loss weight (0 = off). Unlike the div term (uniformity prior),
    this only revives dead codes: each code whose soft batch mass falls below
    cov_floor is penalized in proportion to the shortfall, and the pressure is
    EXACTLY zero once every code clears the floor — living codes' shares stay
    untouched. Differentiable analog of VQ dead-code revival. Requires the
    entropy machinery (quantizer='bsq', or fsq with fsq_entropy=True)."""
    bsq_entropy_cov_floor: float = 0.0
    """Batch-mass threshold below which a code counts as dying. 0 = auto
    (1/batch, i.e. "average one sample per batch")."""
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    encoder_length_token: bool = True
    """False: the encoder consumes NO length token — z sees only the (normalized-
    time) control points [+ start token in optimal mode], so absolute duration
    can enter z only through motion shape. Length then becomes a learned
    per-code property instead of an independent input axis."""
    encoder_arch: str = "spline"
    """spline: fixed control-point tokens (encoder_input_mode / length-token
    probes apply). action_seq: variable-length ACTION-sequence transformer —
    no spline codec, no grounding choice (delta actions carry no absolute
    pose), no length token (duration implicit); requires decoder_arch='rnn'
    so input and output are both action sequences."""
    decoder_layers: int = 3
    decoder_arch: str = "oneshot"
    """oneshot: MLP that emits the full control-point grid (+length) at once.
    rnn: z-only GRU unroll (no teacher forcing) that emits one normalized
    ACTION + one termination logit per step; skill length is represented
    implicitly by the termination signal instead of an explicit length head."""
    reconstruct_length: bool = True
    """oneshot only. False: the decoder reconstructs ONLY the control points —
    no length head is built and the length loss term is dropped. Decoding then
    requires an explicit target length (the encoder still consumes the length
    token). Ignored by the rnn arch (termination always handles length)."""
    length_min: float = 1.0
    length_max: float = 200.0
    action_dim: int = 7
    """rnn arch only: dimensionality of the per-step action output."""

    ctrl_loss_weight: float = 1.0
    length_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    """rnn arch: weight of the masked per-step action MSE."""
    term_loss_weight: float = 1.0
    """rnn arch: weight of the masked per-step termination BCE."""
    term_pos_weight: float = 1.0
    """rnn arch: BCE positive-class weight for the termination head."""
    term_sigma: float = 1.0
    """rnn arch: soft termination target std in frames (Gaussian bump at the
    skill end). 0 = hard 1-frame spike."""

    encoder_lr: float = 3e-4
    decoder_lr: float = 3e-4
    lr_schedule: str = "cosine"
    batch_size: int = 64
    num_workers: int = 0
    epochs: int = 300
    grad_clip: float = 1.0
    val_split: float = 0.1
    val_every: int = 1
    save_best_model: bool = True
    # Best-val SELECTION weights. None -> follow the actual loss weights.
    val_select_ctrl_weight: float | None = None
    val_select_length_weight: float | None = None
    log_every: int = 10
    save_path: str | None = None
    checkpoint_every: int = 0
    device: str = "cuda"

    encoder_min: np.ndarray | None = None
    encoder_max: np.ndarray | None = None
    encoder_start_min: np.ndarray | None = None
    encoder_start_max: np.ndarray | None = None
    action_q01: np.ndarray | None = None
    action_q99: np.ndarray | None = None


# -----------------------------------------------------------------------------
# Decoder + model
# -----------------------------------------------------------------------------


class RNNTrajectoryDecoder(nn.Module):
    """z-only GRU unroll: no teacher forcing, so z stays the sole information source.

    Every step consumes the projected skill vector (a constant input sequence)
    from an initial hidden state also derived from z. Two heads emit a
    normalized action (tanh, v3 convention) and a termination logit per step.
    Feeding previous GT states back in would let the recurrence explain the
    trajectory without z (posterior-collapse analog), so it is deliberately
    not done.
    """

    def __init__(
        self,
        *,
        fsq_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"decoder_layers must be >= 1, got {n_layers}.")
        self.action_dim = int(action_dim)
        self.n_layers = int(n_layers)
        self.hidden_dim = int(hidden_dim)
        self.z_proj = nn.Linear(fsq_dim, hidden_dim)
        self.h0_proj = nn.Linear(fsq_dim, n_layers * hidden_dim)
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.term_head = nn.Linear(hidden_dim, 1)

    def forward(self, z_norm: Tensor, steps: int) -> tuple[Tensor, Tensor]:
        bsize = z_norm.shape[0]
        z_tok = self.z_proj(z_norm)
        inputs = z_tok.unsqueeze(1).expand(bsize, int(steps), -1)
        h0 = (
            torch.tanh(self.h0_proj(z_norm))
            .view(bsize, self.n_layers, self.hidden_dim)
            .transpose(0, 1)
            .contiguous()
        )
        hidden, _ = self.gru(inputs, h0)
        actions = torch.tanh(self.action_head(hidden))
        term_logits = self.term_head(hidden).squeeze(-1)
        return actions, term_logits


class SplineFSQOriginalAE(nn.Module):
    """FSQ v3 spline encoder + one-shot control-point/length decoder."""

    def __init__(self, cfg: FSQOriginalConfig):
        super().__init__()
        if int(cfg.format_version) != ORIGINAL_FORMAT_VERSION:
            raise ValueError(
                f"Only FSQ-original format v{ORIGINAL_FORMAT_VERSION} is supported, "
                f"got {cfg.format_version}."
            )
        if cfg.encoder_input_mode not in {"zero_grounded", "raw_state", "optimal"}:
            raise ValueError(
                "encoder_input_mode must be zero_grounded|raw_state|optimal, "
                f"got {cfg.encoder_input_mode!r}."
            )
        if cfg.lr_schedule not in {"cosine", "constant"}:
            raise ValueError(f"lr_schedule must be cosine|constant, got {cfg.lr_schedule!r}.")
        if cfg.decoder_arch not in {"oneshot", "rnn"}:
            raise ValueError(f"decoder_arch must be oneshot|rnn, got {cfg.decoder_arch!r}.")
        if cfg.encoder_arch not in {"spline", "action_seq"}:
            raise ValueError(f"encoder_arch must be spline|action_seq, got {cfg.encoder_arch!r}.")
        if cfg.encoder_arch == "action_seq" and cfg.decoder_arch != "rnn":
            raise ValueError(
                "encoder_arch='action_seq' requires decoder_arch='rnn' "
                "(action-in/action-out autoencoder)."
            )
        if cfg.quantizer not in {"fsq", "bsq"}:
            raise ValueError(f"quantizer must be fsq|bsq, got {cfg.quantizer!r}.")
        if cfg.quantizer == "bsq" and cfg.bsq_entropy_joint and cfg.bsq_code_dim > 14:
            raise ValueError(
                "bsq_entropy_joint enumerates 2^code_dim codes; "
                f"code_dim={cfg.bsq_code_dim} is too large (max 14). "
                "Use the factorized approximation instead."
            )
        if (
            cfg.quantizer == "fsq"
            and cfg.fsq_entropy
            and cfg.bsq_entropy_joint
            and math.prod(int(v) for v in cfg.fsq_levels) > 16384
        ):
            raise ValueError(
                "fsq_entropy with bsq_entropy_joint enumerates prod(fsq_levels) codes; "
                f"{cfg.fsq_levels} is too large (max 16384)."
            )
        if getattr(cfg, "bsq_entropy_cov_weight", 0.0) > 0:
            codebook = (
                2 ** cfg.bsq_code_dim
                if cfg.quantizer == "bsq"
                else math.prod(int(v) for v in cfg.fsq_levels)
            )
            if codebook > 16384:
                raise ValueError(
                    f"bsq_entropy_cov_weight enumerates all {codebook} codes; too large (max 16384)."
                )
            if cfg.quantizer == "fsq" and not cfg.fsq_entropy:
                raise ValueError(
                    "bsq_entropy_cov_weight on the fsq quantizer requires fsq_entropy=True "
                    "(the coverage loss rides on the soft-assignment machinery)."
                )
        for name in ("encoder_min", "encoder_max"):
            if getattr(cfg, name) is None:
                raise ValueError(f"FSQ-original config is missing required statistic: {name}")
        if cfg.encoder_input_mode == "optimal":
            for name in ("encoder_start_min", "encoder_start_max"):
                if getattr(cfg, name) is None:
                    raise ValueError(f"Optimal FSQ-original config is missing statistic: {name}")
        if (
            cfg.encoder_input_mode in {"zero_grounded", "optimal"}
            and cfg.encoder_grounding_convention != ENCODER_GROUNDING_CONVENTION
        ):
            raise ValueError(
                "This FSQ-original checkpoint uses legacy start-pose grounding. "
                "Mean-XYZ grounding requires a new run."
            )
        if cfg.decoder_arch == "rnn":
            if cfg.action_dim < 1:
                raise ValueError(f"rnn decoder requires action_dim >= 1, got {cfg.action_dim}.")
            for name in ("action_q01", "action_q99"):
                if getattr(cfg, name) is None:
                    raise ValueError(f"rnn FSQ-original config is missing statistic: {name}")
        self.cfg = cfg
        if cfg.encoder_arch == "action_seq":
            self.encoder = ActionSeqEncoder(
                action_dim=cfg.action_dim,
                hidden_dim=cfg.hidden_dim,
                fsq_levels=cfg.fsq_levels,
                n_layers=cfg.num_layers,
                n_heads=cfg.num_heads,
                dropout=cfg.dropout,
            )
        else:
            encoder_cls = SplineFSQEncoder if cfg.encoder_length_token else LengthFreeSplineFSQEncoder
            self.encoder = encoder_cls(
                enc_dim=cfg.enc_dim,
                n_control=cfg.n_control,
                spline_degree=cfg.spline_degree,
                hidden_dim=cfg.hidden_dim,
                fsq_levels=cfg.fsq_levels,
                n_layers=cfg.num_layers,
                n_heads=cfg.num_heads,
                dropout=cfg.dropout,
                length_min=cfg.length_min,
                length_max=cfg.length_max,
                encoder_min=cfg.encoder_min,
                encoder_max=cfg.encoder_max,
                encoder_input_mode=cfg.encoder_input_mode,
                encoder_start_min=cfg.encoder_start_min,
                encoder_start_max=cfg.encoder_start_max,
            )
        # BSQ swap: only the projection head and the quantizer change; the
        # spline/transformer encoder and every downstream contract stay intact.
        latent_dim = len(cfg.fsq_levels)
        if cfg.quantizer == "bsq":
            latent_dim = int(cfg.bsq_code_dim)
            self.encoder.z_head = nn.Linear(cfg.hidden_dim, latent_dim)
            self.encoder.fsq = BSQ(latent_dim, cfg.bsq_inv_temperature)
        if cfg.decoder_arch == "rnn":
            self.decoder = RNNTrajectoryDecoder(
                fsq_dim=latent_dim,
                action_dim=cfg.action_dim,
                hidden_dim=cfg.hidden_dim,
                n_layers=cfg.decoder_layers,
                dropout=cfg.dropout,
            )
        else:
            self.decoder = OneShotTrajectoryDecoder(
                fsq_dim=latent_dim,
                enc_dim=cfg.enc_dim,
                n_control=cfg.n_control,
                hidden_dim=cfg.hidden_dim,
                n_layers=cfg.decoder_layers,
                dropout=cfg.dropout,
                predict_length=cfg.reconstruct_length,
            )

    @property
    def fsq(self) -> FSQ:
        return self.encoder.fsq

    def normalize_length(self, lengths: Tensor) -> Tensor:
        return (
            2.0 * (lengths.float() - self.cfg.length_min)
            / (self.cfg.length_max - self.cfg.length_min + 1e-8)
            - 1.0
        )

    def denormalize_length(self, length_norm: Tensor) -> Tensor:
        raw = (length_norm.float() + 1.0) * 0.5 * (
            self.cfg.length_max - self.cfg.length_min + 1e-8
        ) + self.cfg.length_min
        return raw.round().clamp_min(1.0)

    def forward(
        self,
        ctrl: Tensor,
        lengths: Tensor,
        start_pose: Tensor | None = None,
        *,
        unroll_steps: int | None = None,
        action_seq: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        if self.cfg.encoder_arch == "action_seq":
            if action_seq is None:
                raise ValueError("encoder_arch='action_seq' requires the action_seq input.")
            steps_in = int(lengths.max())
            z_e = self.encoder.encode_continuous(action_seq[:, :steps_in], lengths)
        else:
            z_e = self.encoder.encode_continuous(ctrl, lengths, start_pose, normalized=True)
        z_q, indices = self.fsq(z_e)
        z_norm = self.fsq.normalized(z_q)
        # Continuous pre-quantization coordinate for the entropy terms:
        # BSQ -> sphere point u; FSQ (opt-in) -> FSQ.bound grid coordinate.
        u_cont = None
        if isinstance(self.fsq, BSQ):
            u_cont = self.fsq.unit(z_e)
        elif self.cfg.fsq_entropy:
            u_cont = self.fsq.bound(z_e)
        if self.cfg.decoder_arch == "rnn":
            steps = int(unroll_steps) if unroll_steps is not None else int(lengths.max())
            actions_hat, term_logits = self.decoder(z_norm, steps)
            return {
                "z_q": z_q,
                "indices": indices,
                "u_cont": u_cont,
                "actions_hat": actions_hat,
                "term_logits": term_logits,
            }
        ctrl_hat, length_hat = self.decoder(z_norm)
        return {
            "z_q": z_q,
            "indices": indices,
            "u_cont": u_cont,
            "ctrl_hat": ctrl_hat,
            "length_hat": length_hat,
        }

    # ── numpy convenience ─────────────────────────────────────────────────────

    def encode_numpy(self, trajectory: np.ndarray, device: str | torch.device = "cpu") -> np.ndarray:
        if self.cfg.encoder_arch == "action_seq":
            raise ValueError(
                "This model encodes ACTION sequences; use encode_actions_numpy."
            )
        return self.encoder.encode_numpy(trajectory, device)

    def encode_index(self, trajectory: np.ndarray, device: str | torch.device = "cpu") -> int:
        if self.cfg.encoder_arch == "action_seq":
            raise ValueError(
                "This model encodes ACTION sequences; use encode_actions_index."
            )
        return self.encoder.encode_index(trajectory, device)

    def _normalize_actions_numpy(self, actions: np.ndarray) -> Tensor:
        lo = np.asarray(self.cfg.action_q01, dtype=np.float32)
        hi = np.asarray(self.cfg.action_q99, dtype=np.float32)
        norm = 2.0 * (np.asarray(actions, dtype=np.float32) - lo) / (hi - lo + 1e-8) - 1.0
        return torch.from_numpy(norm)

    @torch.no_grad()
    def encode_actions_numpy(
        self, actions: np.ndarray, device: str | torch.device = "cpu"
    ) -> np.ndarray:
        """action_seq encoder: raw (T, A) dataset-unit actions -> z_q."""
        if self.cfg.encoder_arch != "action_seq":
            raise ValueError("encode_actions_numpy requires encoder_arch='action_seq'.")
        acts = self._normalize_actions_numpy(actions).unsqueeze(0).to(device)
        lengths = torch.tensor([acts.shape[1]], dtype=torch.long, device=device)
        z_q, _ = self.encoder(acts, lengths)
        return z_q[0].cpu().numpy()

    @torch.no_grad()
    def encode_actions_index(
        self, actions: np.ndarray, device: str | torch.device = "cpu"
    ) -> int:
        if self.cfg.encoder_arch != "action_seq":
            raise ValueError("encode_actions_index requires encoder_arch='action_seq'.")
        acts = self._normalize_actions_numpy(actions).unsqueeze(0).to(device)
        lengths = torch.tensor([acts.shape[1]], dtype=torch.long, device=device)
        _, index = self.encoder(acts, lengths)
        return int(index.item())

    def _denormalize_ctrl(self, ctrl_norm: Tensor) -> np.ndarray:
        lo = self.encoder.encoder_min.to(ctrl_norm.device, ctrl_norm.dtype)
        hi = self.encoder.encoder_max.to(ctrl_norm.device, ctrl_norm.dtype)
        return ((ctrl_norm + 1.0) * 0.5 * (hi - lo + 1e-8) + lo).cpu().numpy()

    @torch.no_grad()
    def decode_index_numpy(
        self,
        index: int,
        device: str | torch.device = "cpu",
        length: int | None = None,
    ) -> np.ndarray:
        """Codebook index -> encoder-convention trajectory via the one-shot decoder."""
        if self.cfg.decoder_arch != "oneshot":
            raise ValueError("decode_index_numpy is oneshot-only; use rollout_index_numpy for rnn.")
        code = torch.tensor([index], dtype=torch.long, device=device)
        z_norm = self.fsq.code_to_normalized(code).to(device)
        ctrl_hat, length_hat = self.decoder(z_norm)
        ctrl = self._denormalize_ctrl(ctrl_hat[0])
        if length is not None:
            steps = int(length)
        elif length_hat is None:
            raise ValueError(
                "This model was trained with reconstruct_length=False; "
                "pass an explicit length to decode."
            )
        else:
            steps = int(self.denormalize_length(length_hat)[0].item())
        return spline_decode(ctrl, steps, self.cfg.spline_degree)

    @torch.no_grad()
    def reconstruct_numpy(
        self,
        trajectory: np.ndarray,
        device: str | torch.device = "cpu",
        *,
        use_true_length: bool = True,
    ) -> np.ndarray:
        """Full round trip: raw state trajectory -> z -> mean-XYZ-grounded reconstruction.

        The output keeps rotation and gripper absolute, while XYZ is centered
        around the trajectory mean. Add ``trajectory[:, :3].mean(0)`` back to
        reconstructed XYZ to compare in absolute coordinates.
        """
        if self.cfg.decoder_arch != "oneshot":
            raise ValueError("reconstruct_numpy is oneshot-only; use rollout_actions_numpy for rnn.")
        ctrl, length = spline_encode(
            trajectory,
            self.cfg.n_control,
            self.cfg.spline_degree,
            input_mode=self.cfg.encoder_input_mode,
        )
        ctrl_t = self.encoder.normalize_control_points(
            torch.from_numpy(ctrl).float().unsqueeze(0).to(device)
        )
        length_t = torch.tensor([length], dtype=torch.long, device=device)
        start_t = None
        if self.encoder.enc_start_proj is not None:
            start_t = self.encoder.normalize_start_pose(
                torch.from_numpy(encoder_grounding_position(trajectory)).float().unsqueeze(0).to(device)
            )
        output = self(ctrl_t, length_t, start_t)
        ctrl_hat = self._denormalize_ctrl(output["ctrl_hat"][0])
        if use_true_length:
            steps = length
        elif output["length_hat"] is None:
            raise ValueError(
                "This model was trained with reconstruct_length=False; "
                "only use_true_length=True is available."
            )
        else:
            steps = int(self.denormalize_length(output["length_hat"])[0].item())
        return spline_decode(ctrl_hat, steps, self.cfg.spline_degree)

    # ── rnn-arch inference ────────────────────────────────────────────────────

    def _denormalize_actions(self, actions_norm: Tensor) -> np.ndarray:
        lo = torch.as_tensor(self.cfg.action_q01, device=actions_norm.device, dtype=actions_norm.dtype)
        hi = torch.as_tensor(self.cfg.action_q99, device=actions_norm.device, dtype=actions_norm.dtype)
        return ((actions_norm + 1.0) * 0.5 * (hi - lo + 1e-8) + lo).cpu().numpy()

    def _rollout_z(
        self, z_norm: Tensor, max_steps: int | None, threshold: float
    ) -> tuple[np.ndarray, bool]:
        """Unroll until the termination head fires (or the cap). The unroll is
        deterministic in z, so computing the capped sequence once and cutting it
        at the first firing equals a step-by-step rollout."""
        cap = int(max_steps) if max_steps is not None else int(round(self.cfg.length_max))
        actions_norm, term_logits = self.decoder(z_norm, cap)
        fired = (torch.sigmoid(term_logits[0]) >= threshold).nonzero()
        terminated = fired.numel() > 0
        steps = int(fired[0].item()) + 1 if terminated else cap
        return self._denormalize_actions(actions_norm[0, :steps]), terminated

    @torch.no_grad()
    def rollout_actions_numpy(
        self,
        trajectory: np.ndarray,
        device: str | torch.device = "cpu",
        *,
        max_steps: int | None = None,
        threshold: float = 0.5,
    ) -> tuple[np.ndarray, bool]:
        """Encode a state trajectory, then unroll actions until termination.

        Returns (actions in dataset units, terminated_naturally)."""
        if self.cfg.decoder_arch != "rnn":
            raise ValueError("rollout_actions_numpy requires decoder_arch='rnn'.")
        if self.cfg.encoder_arch == "action_seq":
            raise ValueError(
                "This model encodes ACTION sequences; use rollout_from_actions_numpy."
            )
        ctrl, length = spline_encode(
            trajectory,
            self.cfg.n_control,
            self.cfg.spline_degree,
            input_mode=self.cfg.encoder_input_mode,
        )
        ctrl_t = self.encoder.normalize_control_points(
            torch.from_numpy(ctrl).float().unsqueeze(0).to(device)
        )
        length_t = torch.tensor([length], dtype=torch.long, device=device)
        start_t = None
        if self.encoder.enc_start_proj is not None:
            start_t = self.encoder.normalize_start_pose(
                torch.from_numpy(encoder_grounding_position(trajectory)).float().unsqueeze(0).to(device)
            )
        z_q, _ = self.encoder(ctrl_t, length_t, start_t, normalized=True)
        return self._rollout_z(self.fsq.normalized(z_q), max_steps, threshold)

    @torch.no_grad()
    def rollout_from_actions_numpy(
        self,
        actions: np.ndarray,
        device: str | torch.device = "cpu",
        *,
        max_steps: int | None = None,
        threshold: float = 0.5,
    ) -> tuple[np.ndarray, bool]:
        """action_seq encoder round trip: raw actions -> z -> action rollout."""
        if self.cfg.encoder_arch != "action_seq" or self.cfg.decoder_arch != "rnn":
            raise ValueError(
                "rollout_from_actions_numpy requires encoder_arch='action_seq' "
                "and decoder_arch='rnn'."
            )
        acts = self._normalize_actions_numpy(actions).unsqueeze(0).to(device)
        lengths = torch.tensor([acts.shape[1]], dtype=torch.long, device=device)
        z_q, _ = self.encoder(acts, lengths)
        return self._rollout_z(self.fsq.normalized(z_q), max_steps, threshold)

    @torch.no_grad()
    def rollout_index_numpy(
        self,
        index: int,
        device: str | torch.device = "cpu",
        *,
        max_steps: int | None = None,
        threshold: float = 0.5,
    ) -> tuple[np.ndarray, bool]:
        """Codebook index -> action rollout until termination (rnn arch)."""
        if self.cfg.decoder_arch != "rnn":
            raise ValueError("rollout_index_numpy requires decoder_arch='rnn'.")
        code = torch.tensor([index], dtype=torch.long, device=device)
        z_norm = self.fsq.code_to_normalized(code).to(device)
        return self._rollout_z(z_norm, max_steps, threshold)


# -----------------------------------------------------------------------------
# Checkpoint IO
# -----------------------------------------------------------------------------


def load_fsq_original_model(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[SplineFSQOriginalAE, FSQOriginalConfig]:
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = checkpoint.get("cfg")
    if not isinstance(cfg, FSQOriginalConfig):
        raise ValueError(f"Not an FSQ-original checkpoint (cfg missing/wrong type): {path}")
    if (
        cfg.encoder_input_mode in {"zero_grounded", "optimal"}
        and "encoder_grounding_convention" not in vars(cfg)
    ):
        raise ValueError(
            "Legacy FSQ-original grounding checkpoint is incompatible with mean-XYZ grounding."
        )
    model = SplineFSQOriginalAE(cfg)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), cfg


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class FSQOriginalDataset(Dataset):
    """Whole-trajectory items: normalized control points, length, optional grounding position.

    ``start_poses`` is retained as a compatibility name for normalized mean XYZ.
    It and ``ctrl`` / ``lengths`` use the same
    layout as ``FSQTrajectoryDataset`` so FSQ.py's codebook diagnostics
    (``_collect_code_assignments``) can be reused unchanged.
    """

    def __init__(
        self,
        segments: list[np.ndarray],
        metadata: list[dict[str, Any]],
        cfg: FSQOriginalConfig,
        actions: list[np.ndarray] | None = None,
    ):
        if len(segments) != len(metadata):
            raise ValueError("FSQ-original dataset component lengths do not match.")
        if cfg.decoder_arch == "rnn":
            if actions is None or len(actions) != len(segments):
                raise ValueError("rnn decoder_arch requires per-skill actions in the dataset.")
        self.cfg = cfg
        self.metadata = metadata
        self.ctrl: list[np.ndarray] = []
        self.start_poses: list[np.ndarray] | None = (
            [] if cfg.encoder_input_mode == "optimal" else None
        )
        self.lengths: list[int] = []
        # rnn arch: q01/q99-normalized action targets padded to length_max so the
        # default collate stacks them; the loss masks steps >= length.
        self.actions_norm: list[np.ndarray] | None = None
        if cfg.decoder_arch == "rnn":
            pad_steps = int(round(cfg.length_max))
            a_lo = np.asarray(cfg.action_q01, dtype=np.float32)
            a_hi = np.asarray(cfg.action_q99, dtype=np.float32)
            self.actions_norm = []
            for action in actions:
                action = np.asarray(action, dtype=np.float32)
                if len(action) > pad_steps:
                    raise ValueError(
                        f"Skill action length {len(action)} exceeds length_max {pad_steps}."
                    )
                norm = 2.0 * (action - a_lo) / (a_hi - a_lo + 1e-8) - 1.0
                padded = np.zeros((pad_steps, action.shape[-1]), dtype=np.float32)
                padded[: len(action)] = norm
                self.actions_norm.append(padded)

        enc_min = np.asarray(cfg.encoder_min, dtype=np.float32)
        enc_max = np.asarray(cfg.encoder_max, dtype=np.float32)
        start_min = start_max = None
        if self.start_poses is not None:
            start_min = np.asarray(cfg.encoder_start_min, dtype=np.float32)
            start_max = np.asarray(cfg.encoder_start_max, dtype=np.float32)
        for segment in segments:
            ctrl, length = spline_encode(
                segment,
                cfg.n_control,
                cfg.spline_degree,
                input_mode=cfg.encoder_input_mode,
            )
            self.ctrl.append(
                (2.0 * (ctrl - enc_min) / (enc_max - enc_min + 1e-8) - 1.0).astype(np.float32)
            )
            if self.start_poses is not None:
                start_pose = encoder_grounding_position(segment)
                self.start_poses.append(
                    (2.0 * (start_pose - start_min) / (start_max - start_min + 1e-8) - 1.0).astype(
                        np.float32
                    )
                )
            self.lengths.append(length)

    def __len__(self) -> int:
        return len(self.ctrl)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        length = self.lengths[index]
        length_target = (
            2.0 * (float(length) - self.cfg.length_min)
            / (self.cfg.length_max - self.cfg.length_min + 1e-8)
            - 1.0
        )
        item = {
            "ctrl": torch.from_numpy(self.ctrl[index]),
            "length": torch.tensor(length, dtype=torch.long),
            "length_target": torch.tensor(length_target, dtype=torch.float32),
        }
        if self.start_poses is not None:
            item["start_pose"] = torch.from_numpy(self.start_poses[index])
        if self.actions_norm is not None:
            item["actions_norm"] = torch.from_numpy(self.actions_norm[index])
        return item


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------


def _rnn_loss(
    output: dict[str, Tensor],
    batch: dict[str, Tensor],
    cfg: FSQOriginalConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    pred = output["actions_hat"]
    # Steps beyond every skill's length are masked out, so trimming to the
    # shorter of prediction/target width never changes the loss value.
    steps = min(pred.shape[1], batch["actions_norm"].shape[1])
    pred = pred[:, :steps]
    lengths = batch["length"].to(pred.device)
    target = batch["actions_norm"][:, :steps].to(pred)
    step_ids = torch.arange(steps, device=pred.device)
    mask = (step_ids[None] < lengths[:, None]).to(pred.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)

    per_step_action = (pred - target).square().mean(dim=-1)
    action_loss = ((per_step_action * mask).sum(dim=1) / denom).mean()

    distance_to_end = lengths[:, None].to(pred.dtype) - 1.0 - step_ids[None].to(pred.dtype)
    if cfg.term_sigma > 0:
        term_target = torch.exp(-distance_to_end.square() / (2.0 * cfg.term_sigma**2))
    else:
        term_target = (distance_to_end == 0).to(pred.dtype)
    term_logits = output["term_logits"][:, :steps]
    per_step_term = F.binary_cross_entropy_with_logits(
        term_logits,
        term_target.to(term_logits.dtype),
        reduction="none",
        pos_weight=torch.as_tensor(cfg.term_pos_weight, device=term_logits.device),
    )
    term_loss = ((per_step_term * mask.to(per_step_term.dtype)).sum(dim=1) / denom).mean()

    total = cfg.action_loss_weight * action_loss + cfg.term_loss_weight * term_loss
    metrics = {
        "loss": total.detach(),
        "action": action_loss.detach(),
        "termination": term_loss.detach(),
    }
    return total, metrics


def _soft_joint_distribution(u_cont: Tensor, cfg: FSQOriginalConfig) -> Tensor:
    """Soft joint code distribution q(c|x) of shape (B, codebook_size).

    Same factorized-Bernoulli / per-dim-softmax construction the entropy terms
    use, materialized over every code so batch masses can be thresholded."""
    tau = cfg.bsq_inv_temperature
    if cfg.quantizer == "bsq":
        p = torch.sigmoid(2.0 * tau * u_cont.float()).clamp(1e-6, 1.0 - 1e-6)
        code_dim = u_cont.shape[-1]
        codes = torch.arange(2 ** code_dim, device=u_cont.device)
        bits = ((codes[:, None] >> torch.arange(code_dim, device=u_cont.device)) & 1).float()
        log_q = p.log() @ bits.T + (1.0 - p).log() @ (1.0 - bits).T
        return log_q.exp()
    q = None
    for d, level in enumerate(int(v) for v in cfg.fsq_levels):
        centers = torch.arange(level, device=u_cont.device, dtype=torch.float32)
        centers = centers - level // 2
        logits = -tau * (u_cont[:, d : d + 1].float() - centers[None]) ** 2
        p = torch.softmax(logits, dim=-1).clamp(1e-6, 1.0)
        q = p if q is None else (p[:, :, None] * q[:, None, :]).reshape(q.shape[0], -1)
    return q


def coverage_loss(q: Tensor, floor: float) -> Tensor:
    """Dead-code revival hinge: penalize only codes below the mass floor.

    Zero pressure once every code clears the floor — living codes' shares are
    never touched, unlike the entropy-max (uniformity) diversity term."""
    mass = q.mean(dim=0)
    return (torch.relu(floor - mass) / max(floor, 1e-12)).sum()


def _apply_bsq_entropy(
    total: Tensor,
    metrics: dict[str, Tensor],
    output: dict[str, Tensor],
    cfg: FSQOriginalConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    if output.get("u_cont") is None:
        return total, metrics
    if cfg.quantizer == "bsq":
        sample_entropy, dataset_entropy = bsq_entropy_terms(
            output["u_cont"],
            cfg.bsq_inv_temperature,
            joint_dataset=cfg.bsq_entropy_joint,
        )
    elif cfg.fsq_entropy:
        sample_entropy, dataset_entropy = fsq_entropy_terms(
            output["u_cont"],
            cfg.fsq_levels,
            cfg.bsq_inv_temperature,
            joint_dataset=cfg.bsq_entropy_joint,
        )
    else:
        return total, metrics
    total = (
        total
        + cfg.bsq_entropy_conf_weight * sample_entropy
        - cfg.bsq_entropy_div_weight * dataset_entropy
    )
    metrics = {
        **metrics,
        "entropy_sample": sample_entropy.detach(),
        "entropy_dataset": dataset_entropy.detach(),
    }
    cov_weight = getattr(cfg, "bsq_entropy_cov_weight", 0.0)
    if cov_weight > 0:
        q = _soft_joint_distribution(output["u_cont"], cfg)
        floor = getattr(cfg, "bsq_entropy_cov_floor", 0.0) or 1.0 / q.shape[0]
        cov = coverage_loss(q, floor)
        total = total + cov_weight * cov
        metrics["coverage"] = cov.detach()
    metrics["loss"] = total.detach()
    return total, metrics


def fsq_original_loss(
    output: dict[str, Tensor],
    batch: dict[str, Tensor],
    cfg: FSQOriginalConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    if cfg.decoder_arch == "rnn":
        total, metrics = _rnn_loss(output, batch, cfg)
        return _apply_bsq_entropy(total, metrics, output, cfg)
    ctrl_hat = output["ctrl_hat"]
    ctrl_target = batch["ctrl"].to(ctrl_hat)
    ctrl_error = (ctrl_hat - ctrl_target).square()
    ctrl_loss = ctrl_error.mean()
    length_hat = output["length_hat"]
    if length_hat is None:
        length_loss = torch.zeros((), device=ctrl_hat.device, dtype=ctrl_hat.dtype)
    else:
        length_loss = (length_hat - batch["length_target"].to(length_hat)).square().mean()
    total = cfg.ctrl_loss_weight * ctrl_loss + cfg.length_loss_weight * length_loss
    pose_dims = ctrl_target.shape[-1] - N_GRIPPER_DIMS
    metrics = {
        "loss": total.detach(),
        "ctrl": ctrl_loss.detach(),
        "length": length_loss.detach(),
        "ctrl_pose": ctrl_error[..., :pose_dims].mean().detach(),
        "ctrl_gripper": ctrl_error[..., pose_dims:].mean().detach(),
    }
    return _apply_bsq_entropy(total, metrics, output, cfg)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


@torch.inference_mode()
def _collect_assignments(
    model: SplineFSQOriginalAE,
    datasets: tuple[FSQOriginalDataset, ...],
    device: torch.device,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    """Arch-aware code-assignment sweep for the codebook diagnostics.

    The spline arch delegates to FSQ.py's collector (identical inputs); the
    action_seq arch feeds the padded normalized action sequences instead."""
    if model.cfg.encoder_arch != "action_seq":
        return _collect_code_assignments(model, datasets, device, batch_size)
    assignments, margins = [], []
    for dataset in datasets:
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            acts = torch.from_numpy(
                np.stack(dataset.actions_norm[start:stop])
            ).to(device, non_blocking=True)
            lengths = torch.as_tensor(
                dataset.lengths[start:stop], dtype=torch.long, device=device
            )
            steps = int(lengths.max())
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                z_e = model.encoder.encode_continuous(acts[:, :steps], lengths)
                _, indices = model.fsq(z_e)
            margins.append(model.fsq.boundary_margin(z_e.float()).amin(dim=-1).cpu())
            assignments.append(indices.reshape(-1).to(device="cpu", dtype=torch.long))
    return (
        torch.cat(assignments) if assignments else torch.empty(0, dtype=torch.long),
        torch.cat(margins) if margins else torch.empty(0, dtype=torch.float32),
    )


def deterministic_split(
    count: int, metadata: list[dict[str, Any]], val_split: float
) -> tuple[list[int], list[int], str]:
    """Deterministic train/val membership by skill identity (sha1).

    Identical data always yields the same membership regardless of load order.
    Returns (val_ids, train_ids, fingerprint); shared by training and eval so
    an eval never scores training members as held-out.
    """
    n_val = max(1, int(count * val_split))
    if len(metadata) == count:
        def identity_hash(i: int) -> int:
            item = metadata[i]
            identity = f"{item.get('episode_id', -1)}_{item.get('skill_index', -1)}"
            return int(hashlib.sha1(identity.encode()).hexdigest(), 16)

        order = sorted(range(count), key=identity_hash)
        fingerprint = hashlib.sha1(
            ",".join(
                sorted(
                    f"{metadata[i].get('episode_id', -1)}_{metadata[i].get('skill_index', -1)}"
                    for i in order[:n_val]
                )
            ).encode()
        ).hexdigest()[:12]
    else:
        order = np.random.default_rng(42).permutation(count).tolist()
        fingerprint = "seed42"
    return order[:n_val], order[n_val:], fingerprint


def train_fsq_original(
    *,
    segments: list[np.ndarray],
    metadata: list[dict[str, Any]],
    cfg: FSQOriginalConfig,
    actions: list[np.ndarray] | None = None,
    wandb_run=None,
    resume_from: str | None = None,
) -> SplineFSQOriginalAE:
    if not segments:
        raise ValueError("No skill trajectories were provided.")
    if cfg.num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {cfg.num_workers}.")
    if cfg.val_every < 0:
        raise ValueError(f"val_every must be >= 0, got {cfg.val_every}.")
    if cfg.save_best_model and cfg.val_every == 0:
        raise ValueError("save_best_model requires val_every > 0.")

    # Deterministic split by skill identity (same rule as FSQ v3); shared with
    # fsq_original_eval so held-out membership matches training exactly.
    val_ids, train_ids, fingerprint = deterministic_split(
        len(segments), metadata, cfg.val_split
    )
    print(
        f"[FSQ-orig] trajectories={len(segments)} train={len(train_ids)} "
        f"val={len(val_ids)} fp={fingerprint}"
    )

    def dataset(ids: list[int]) -> FSQOriginalDataset:
        return FSQOriginalDataset(
            [segments[i] for i in ids],
            [metadata[i] for i in ids],
            cfg,
            actions=None if actions is None else [actions[i] for i in ids],
        )

    train_ds, val_ds = dataset(train_ids), dataset(val_ids)
    device = torch.device(cfg.device)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model = SplineFSQOriginalAE(cfg).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": cfg.encoder_lr, "name": "encoder"},
            {"params": model.decoder.parameters(), "lr": cfg.decoder_lr, "name": "decoder"},
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: fsq_lr_factor(cfg.lr_schedule, epoch, cfg.epochs),
    )
    save_path = Path(cfg.save_path) if cfg.save_path else Path("FSQ.pt")
    start_epoch, best_val = 1, math.inf
    previous_code_assignments: Tensor | None = None
    previous_code_epoch: int | None = None
    if resume_from:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        resume_cfg = checkpoint.get("cfg")
        if not isinstance(resume_cfg, FSQOriginalConfig):
            raise ValueError(f"Not an FSQ-original checkpoint: {resume_from}")
        if resume_cfg.encoder_input_mode != cfg.encoder_input_mode:
            raise ValueError(
                "Cannot resume FSQ-original with a different encoder input convention: "
                f"checkpoint={resume_cfg.encoder_input_mode!r}, current={cfg.encoder_input_mode!r}."
            )
        if (
            cfg.encoder_input_mode in {"zero_grounded", "optimal"}
            and (
                "encoder_grounding_convention" not in vars(resume_cfg)
                or resume_cfg.encoder_grounding_convention
                != cfg.encoder_grounding_convention
            )
        ):
            raise ValueError(
                "Cannot resume FSQ-original across grounding conventions; start a new run."
            )
        if resume_cfg.lr_schedule != cfg.lr_schedule:
            raise ValueError(
                "Cannot resume FSQ-original with a different LR schedule: "
                f"checkpoint={resume_cfg.lr_schedule!r}, current={cfg.lr_schedule!r}. "
                "Use a different fsq_exp for a new run."
            )
        if list(resume_cfg.fsq_levels) != list(cfg.fsq_levels):
            raise ValueError(
                "Cannot resume FSQ-original with different FSQ levels: "
                f"checkpoint={resume_cfg.fsq_levels}, current={cfg.fsq_levels}."
            )
        resume_quantizer = getattr(resume_cfg, "quantizer", "fsq")
        if resume_quantizer != cfg.quantizer:
            raise ValueError(
                "Cannot resume FSQ-original with a different quantizer: "
                f"checkpoint={resume_quantizer!r}, current={cfg.quantizer!r}."
            )
        if cfg.quantizer == "bsq" and int(getattr(resume_cfg, "bsq_code_dim", 0)) != int(cfg.bsq_code_dim):
            raise ValueError(
                "Cannot resume BSQ with a different code_dim: "
                f"checkpoint={getattr(resume_cfg, 'bsq_code_dim', None)}, current={cfg.bsq_code_dim}."
            )
        resume_entropy_joint = bool(getattr(resume_cfg, "bsq_entropy_joint", False))
        if cfg.quantizer == "bsq" and resume_entropy_joint != cfg.bsq_entropy_joint:
            raise ValueError(
                "Cannot resume BSQ with a different dataset-entropy objective: "
                f"checkpoint joint={resume_entropy_joint}, current={cfg.bsq_entropy_joint}. "
                "Use a different fsq_exp for a new run."
            )
        resume_fsq_entropy = bool(getattr(resume_cfg, "fsq_entropy", False))
        if cfg.quantizer == "fsq" and resume_fsq_entropy != cfg.fsq_entropy:
            raise ValueError(
                "Cannot resume FSQ with a different entropy objective: "
                f"checkpoint fsq_entropy={resume_fsq_entropy}, current={cfg.fsq_entropy}. "
                "Use a different fsq_exp for a new run."
            )
        resume_length_token = getattr(resume_cfg, "encoder_length_token", True)
        if resume_length_token != cfg.encoder_length_token:
            raise ValueError(
                "Cannot resume FSQ-original with a different encoder input contract: "
                f"checkpoint encoder_length_token={resume_length_token}, "
                f"current={cfg.encoder_length_token}. Use a different fsq_exp for a new run."
            )
        resume_encoder_arch = getattr(resume_cfg, "encoder_arch", "spline")
        if resume_encoder_arch != cfg.encoder_arch:
            raise ValueError(
                "Cannot resume FSQ-original with a different encoder architecture: "
                f"checkpoint={resume_encoder_arch!r}, current={cfg.encoder_arch!r}. "
                "Use a different fsq_exp for a new run."
            )
        resume_decoder_arch = getattr(resume_cfg, "decoder_arch", "oneshot")
        if resume_decoder_arch != cfg.decoder_arch:
            raise ValueError(
                "Cannot resume FSQ-original with a different decoder architecture: "
                f"checkpoint={resume_decoder_arch!r}, current={cfg.decoder_arch!r}. "
                "Use a different fsq_exp for a new run."
            )
        resume_reconstruct_length = getattr(resume_cfg, "reconstruct_length", True)
        if resume_reconstruct_length != cfg.reconstruct_length:
            raise ValueError(
                "Cannot resume FSQ-original with a different reconstruction target: "
                f"checkpoint reconstruct_length={resume_reconstruct_length}, "
                f"current={cfg.reconstruct_length}. Use a different fsq_exp for a new run."
            )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optim_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = float(checkpoint.get("best_val", checkpoint.get("val_select", math.inf)))
        if cfg.save_best_model and save_path.is_file():
            best_checkpoint = torch.load(
                str(save_path), map_location="cpu", weights_only=False, mmap=True
            )
            best_val = min(best_val, float(best_checkpoint.get("val_select", math.inf)))
            del best_checkpoint
        print(f"[FSQ-orig] resumed {resume_from} at epoch {start_epoch} (best select={best_val:.6f})")
        model.eval()
        previous_code_assignments, _ = _collect_assignments(
            model, (val_ds, train_ds), device, cfg.batch_size
        )
        previous_code_epoch = start_epoch - 1

    if wandb_run is not None:
        wandb_run.define_metric("epoch")
        wandb_run.define_metric("optimizer_step")
        for name in ("train/*", "val/*", "perf/*", "lr/*", "codebook/*"):
            wandb_run.define_metric(name, step_metric="optimizer_step")
        for name in ("train_epoch/*", "val_epoch/*", "perf_epoch/*", "lr_epoch/*", "codebook_epoch/*"):
            wandb_run.define_metric(name, step_metric="epoch")

    def save(path: str | Path, epoch: int, val: float, select: float, *, resumable: bool) -> None:
        payload = {
            "format_version": ORIGINAL_FORMAT_VERSION,
            "cfg": cfg,
            "model_state": model.state_dict(),
            "epoch": epoch,
            "val_loss": val,
            "val_select": select,
            "best_val": best_val,
        }
        if resumable:
            payload["optim_state"] = optimizer.state_dict()
            payload["scheduler_state"] = scheduler.state_dict()
        torch.save(payload, str(path))

    def step(batch: dict[str, Tensor], training: bool):
        moved = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        unroll_steps = None
        if cfg.decoder_arch == "rnn":
            unroll_steps = int(moved["length"].max().item())
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                moved["ctrl"],
                moved["length"],
                moved.get("start_pose"),
                unroll_steps=unroll_steps,
                action_seq=moved.get("actions_norm") if cfg.encoder_arch == "action_seq" else None,
            )
            loss, metrics = fsq_original_loss(output, moved, cfg)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        return (
            {k: float(v) for k, v in metrics.items()},
            int(moved["ctrl"].shape[0]),
            output["indices"].detach(),
        )

    global_step = (start_epoch - 1) * len(train_loader)
    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_sum: dict[str, float] = {}
        train_count = 0
        train_codes_seen = torch.zeros(model.fsq.codebook_size, dtype=torch.bool, device=device)
        for batch in train_loader:
            metrics, count, code_indices = step(batch, True)
            global_step += 1
            train_count += count
            train_codes_seen[code_indices.reshape(-1).long()] = True
            for key, value in metrics.items():
                train_sum[key] = train_sum.get(key, 0.0) + value * count
        scheduler.step()

        should_validate = cfg.val_every > 0 and epoch % cfg.val_every == 0
        val_sum: dict[str, float] = {}
        val_count = 0
        val_codes_seen = torch.zeros(model.fsq.codebook_size, dtype=torch.bool, device=device)
        assignment_metrics: dict[str, float] = {}
        boundary_metrics: dict[str, float] = {}
        assignment_reference_epoch: int | None = None
        full_active_codes = 0
        if should_validate:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    metrics, count, code_indices = step(batch, False)
                    val_count += count
                    val_codes_seen[code_indices.reshape(-1).long()] = True
                    for key, value in metrics.items():
                        val_sum[key] = val_sum.get(key, 0.0) + value * count
            current_code_assignments, current_boundary_margins = _collect_assignments(
                model, (val_ds, train_ds), device, cfg.batch_size
            )
            boundary_metrics = _boundary_margin_metrics(current_boundary_margins)
            full_active_codes = int(current_code_assignments.unique().numel())
            if previous_code_assignments is not None and previous_code_epoch is not None:
                assignment_reference_epoch = previous_code_epoch
                assignment_metrics = _code_assignment_stability(
                    previous_code_assignments,
                    current_code_assignments,
                    model.fsq.codebook_size,
                )
                assignment_metrics["interval_epochs"] = float(epoch - previous_code_epoch)
            previous_code_assignments = current_code_assignments
            previous_code_epoch = epoch

        train_avg = {k: v / max(train_count, 1) for k, v in train_sum.items()}
        val_avg = {k: v / max(val_count, 1) for k, v in val_sum.items()}
        train_active_codes = int(train_codes_seen.count_nonzero().item())
        val_active_codes = int(val_codes_seen.count_nonzero().item())
        codebook_size = model.fsq.codebook_size
        select = math.nan
        if should_validate:
            if cfg.decoder_arch == "rnn":
                select = (
                    cfg.action_loss_weight * val_avg["action"]
                    + cfg.term_loss_weight * val_avg["termination"]
                )
            else:
                select = (
                    (cfg.val_select_ctrl_weight if cfg.val_select_ctrl_weight is not None else cfg.ctrl_loss_weight)
                    * val_avg["ctrl"]
                    + (cfg.val_select_length_weight if cfg.val_select_length_weight is not None else cfg.length_loss_weight)
                    * val_avg["length"]
                )
            if cfg.save_best_model and select < best_val:
                best_val = select
                save(save_path, epoch, val_avg["loss"], select, resumable=False)
        if cfg.checkpoint_every and epoch % cfg.checkpoint_every == 0:
            save(
                save_path.with_name(f"FSQ_epoch{epoch:04d}.pt"),
                epoch,
                val_avg.get("loss", math.nan),
                select,
                resumable=True,
            )

        log = {
            "epoch": epoch,
            "optimizer_step": global_step,
            "perf/seconds": time.perf_counter() - epoch_start,
            **{f"train/{k}": v for k, v in train_avg.items()},
            "train/codebook_utilization_pct": 100.0 * train_active_codes / codebook_size,
            "train/codebook_active_entries": train_active_codes,
            **{f"lr/{group['name']}": group["lr"] for group in optimizer.param_groups},
        }
        log.update({f"train_epoch/{k}": v for k, v in train_avg.items()})
        log.update({
            "train_epoch/codebook_utilization_pct": log["train/codebook_utilization_pct"],
            "train_epoch/codebook_active_entries": train_active_codes,
            "perf_epoch/seconds": log["perf/seconds"],
            **{f"lr_epoch/{group['name']}": group["lr"] for group in optimizer.param_groups},
        })
        if should_validate:
            log.update({f"val/{k}": v for k, v in val_avg.items()})
            log.update({f"val_epoch/{k}": v for k, v in val_avg.items()})
            log.update({
                "val/codebook_utilization_pct": 100.0 * val_active_codes / codebook_size,
                "val/codebook_active_entries": val_active_codes,
                "val/select": select,
                "val_epoch/codebook_utilization_pct": 100.0 * val_active_codes / codebook_size,
                "val_epoch/codebook_active_entries": val_active_codes,
                "val_epoch/select": select,
            })
            full_codebook_log = {
                "full_active_entries": full_active_codes,
                "full_utilization_pct": 100.0 * full_active_codes / codebook_size,
                **boundary_metrics,
                **assignment_metrics,
            }
            log.update({f"codebook/{key}": value for key, value in full_codebook_log.items()})
            log.update({f"codebook_epoch/{key}": value for key, value in full_codebook_log.items()})
        if wandb_run is not None:
            wandb_run.log(log, step=global_step)
        if epoch == 1 or epoch % cfg.log_every == 0 or should_validate:
            message = f"[FSQ-orig] {epoch:4d}/{cfg.epochs} train={train_avg['loss']:.4f}"
            if should_validate:
                if cfg.decoder_arch == "rnn":
                    message += (
                        f" val={val_avg['loss']:.4f} action={val_avg['action']:.4f} "
                        f"term={val_avg['termination']:.4f} select={select:.4f}"
                    )
                else:
                    message += (
                        f" val={val_avg['loss']:.4f} ctrl={val_avg['ctrl']:.4f} "
                        f"len={val_avg['length']:.4f} select={select:.4f}"
                    )
                if assignment_metrics:
                    message += (
                        f" code-retain({assignment_reference_epoch}->{epoch})="
                        f"{assignment_metrics['retention_pct']:.1f}% "
                        f"changed={assignment_metrics['change_pct']:.1f}% "
                        f"matched={assignment_metrics['matched_retention_pct']:.1f}%"
                    )
                else:
                    message += f" code-retain=baseline({full_active_codes}/{codebook_size} active)"
                message += (
                    " boundary-margin="
                    f"{boundary_metrics['boundary_margin_mean_pct']:.1f}% "
                    f"p10={boundary_metrics['boundary_margin_p10_pct']:.1f}% "
                    f"near={boundary_metrics['near_boundary_pct']:.1f}%"
                )
            print(message)

    if cfg.save_best_model:
        print(f"[FSQ-orig] done; best val-select={best_val:.6f} -> {save_path}")
    else:
        print(f"[FSQ-orig] done; periodic checkpoints -> {save_path.parent}")
    return model
