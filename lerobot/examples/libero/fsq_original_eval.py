"""Standalone evaluation for FSQ-original checkpoints (oneshot | rnn).

Deliberately separate from fsq_eval.py: v3's evaluation contract requires a
sample_action_chunks reconstructor and an image-based terminator, neither of
which exists in the FSQ-original variants. This script scores what each
variant actually models:

  common  : codebook usage (active/effective/top1/entropy) and boundary
            margins, reported for train/val splits (same deterministic split
            as training) plus overall.
  oneshot : zero-grounded STATE trajectory reconstruction MSE decomposed into
            xyz / rpy / gripper, decoded at the true length; plus length
            error when the checkpoint has a length head.
  rnn     : per-step ACTION reconstruction MSE (dataset units, masked to the
            GT length) decomposed into xyz / rpy / gripper; termination
            timing |predicted stop - T|, early rate, and no-fire rate.

Outputs metrics.json and a small fsq_original_eval.html into --output_dir.

Usage:
    python examples/libero/fsq_original_eval.py \
      --model_path  /path/to/FSQ_epoch0200.pt \
      --skills_dir  /path/to/skillset/skills \
      --output_dir  /path/to/output
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


@dataclass
class Args:
    model_path: str = ""
    skills_dir: str = ""
    output_dir: str = ""
    device: str = "cuda"
    batch_size: int = 256
    term_threshold: float = 0.5
    """rnn arch: sigmoid threshold above which the rollout is considered stopped."""
    val_split: float = 0.1
    """Must match training so the val rows really are held-out skills."""
    max_skills: int = 0
    """Evaluate only every k-th skill so ~max_skills remain (0 = all). Smoke use only."""


# ── batched model passes ───────────────────────────────────────────────────────


@torch.no_grad()
def _encode_all(model, dataset, device, batch_size):
    """Encode every skill once: codes, per-sample boundary margins, z_norm,
    and (BSQ only) the continuous sphere point u for bit-level plots."""
    from FSQ_original import BSQ

    is_bsq = isinstance(model.fsq, BSQ)
    is_action_seq = getattr(model.cfg, "encoder_arch", "spline") == "action_seq"
    codes, margins, z_norms, units = [], [], [], []
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        lengths = torch.as_tensor(dataset.lengths[start:stop], dtype=torch.long, device=device)
        if is_action_seq:
            acts = torch.from_numpy(np.stack(dataset.actions_norm[start:stop])).to(device)
            z_e = model.encoder.encode_continuous(acts[:, : int(lengths.max())], lengths)
        else:
            ctrl = torch.from_numpy(np.stack(dataset.ctrl[start:stop])).to(device)
            start_pose = None
            if dataset.start_poses is not None:
                start_pose = torch.from_numpy(np.stack(dataset.start_poses[start:stop])).to(device)
            z_e = model.encoder.encode_continuous(ctrl, lengths, start_pose, normalized=True)
        z_q, index = model.fsq(z_e)
        codes.append(index.cpu())
        margins.append(model.fsq.boundary_margin(z_e.float()).amin(dim=-1).cpu())
        z_norms.append(model.fsq.normalized(z_q).float().cpu())
        if is_bsq:
            units.append(model.fsq.unit(z_e.float()).cpu())
    return (
        torch.cat(codes),
        torch.cat(margins),
        torch.cat(z_norms),
        torch.cat(units) if units else None,
    )


def _codebook_metrics(codes: torch.Tensor, margins: torch.Tensor, codebook_size: int) -> dict:
    counts = torch.bincount(codes.long(), minlength=codebook_size).float()
    total = counts.sum().clamp_min(1.0)
    p = counts / total
    nonzero = p[p > 0]
    entropy_bits = float(-(nonzero * nonzero.log2()).sum())
    normalized_margin = (margins.float().clamp(0.0, 0.5) / 0.5)
    return {
        "n_skills": int(total),
        "active_codes": int((counts > 0).sum()),
        "codebook_size": codebook_size,
        "effective_codes": float(math.exp(-(nonzero * nonzero.log()).sum())),
        "top1_share_pct": float(p.max() * 100.0),
        "top3_share_pct": float(p.sort(descending=True).values[:3].sum() * 100.0),
        "usage_entropy_bits": entropy_bits,
        "boundary_margin_mean_pct": float(normalized_margin.mean() * 100.0),
        "boundary_margin_p10_pct": float(torch.quantile(normalized_margin, 0.1) * 100.0),
        "near_boundary_pct": float((normalized_margin <= 0.1).float().mean() * 100.0),
    }


@torch.no_grad()
def _oneshot_recon_metrics(model, dataset, segments, z_norms, device, batch_size) -> dict:
    """Decode each skill's control points at the TRUE length; MSE in the
    zero-grounded state convention (xyz 0:3, rpy 3:6, gripper trailing dims)."""
    from FSQ import prepare_encoder_trajectory
    from FSQ_original import spline_decode

    ctrl_hats = []
    length_hats = []
    for start in range(0, len(dataset), batch_size):
        z = z_norms[start : start + batch_size].to(device)
        ctrl_hat, length_hat = model.decoder(z)
        ctrl_hats.append(ctrl_hat.float().cpu())
        if length_hat is not None:
            length_hats.append(model.denormalize_length(length_hat).cpu())
    ctrl_hat = torch.cat(ctrl_hats).numpy()

    mse = {"xyz": [], "rpy": [], "gripper": [], "total": []}
    for i, segment in enumerate(segments):
        target = prepare_encoder_trajectory(segment, model.cfg.encoder_input_mode)
        pred_ctrl = model._denormalize_ctrl(torch.from_numpy(ctrl_hat[i]))  # noqa: SLF001
        recon = spline_decode(pred_ctrl, len(target), model.cfg.spline_degree)
        err = (recon - target) ** 2
        mse["xyz"].append(err[:, 0:3].mean())
        mse["rpy"].append(err[:, 3:6].mean())
        mse["gripper"].append(err[:, 6:].mean())
        mse["total"].append(err.mean())
    metrics = {f"recon_mse_{k}": float(np.mean(v)) for k, v in mse.items()}
    if length_hats:
        pred_lengths = torch.cat(length_hats).numpy()
        true_lengths = np.asarray(dataset.lengths, dtype=np.float32)
        metrics["length_abs_err_mean"] = float(np.abs(pred_lengths - true_lengths).mean())
    return metrics


@torch.no_grad()
def _rnn_recon_metrics(model, dataset, actions, z_norms, device, batch_size, threshold) -> dict:
    """Masked per-step action MSE in dataset units + termination timing.

    The unroll is deterministic in z, so one capped-length pass provides both
    the reconstruction (cut at GT length) and the stop step (first firing)."""
    cap = int(round(model.cfg.length_max))
    a_lo = np.asarray(model.cfg.action_q01, dtype=np.float32)
    a_hi = np.asarray(model.cfg.action_q99, dtype=np.float32)

    mse = {"xyz": [], "rpy": [], "gripper": [], "total": []}
    timing_abs, early, no_fire = [], [], []
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        z = z_norms[start:stop].to(device)
        actions_norm, term_logits = model.decoder(z, cap)
        pred = ((actions_norm.float().cpu().numpy() + 1.0) * 0.5 * (a_hi - a_lo + 1e-8)) + a_lo
        fired = torch.sigmoid(term_logits.float().cpu()) >= threshold
        for row, i in enumerate(range(start, stop)):
            true_len = dataset.lengths[i]
            err = (pred[row, :true_len] - actions[i]) ** 2
            mse["xyz"].append(err[:, 0:3].mean())
            mse["rpy"].append(err[:, 3:6].mean())
            mse["gripper"].append(err[:, 6:].mean())
            mse["total"].append(err.mean())
            hit = fired[row].nonzero()
            if hit.numel():
                stop_step = int(hit[0].item()) + 1
                timing_abs.append(abs(stop_step - true_len))
                early.append(stop_step < true_len)
                no_fire.append(False)
            else:
                timing_abs.append(abs(cap - true_len))
                early.append(False)
                no_fire.append(True)
    metrics = {f"recon_mse_{k}": float(np.mean(v)) for k, v in mse.items()}
    metrics.update({
        "termination_abs_err_mean": float(np.mean(timing_abs)),
        "early_rate": float(np.mean(early)),
        "no_fire_rate": float(np.mean(no_fire)),
    })
    return metrics


# ── BSQ figures ────────────────────────────────────────────────────────────────


def _previous_codes(output_dir: Path) -> np.ndarray | None:
    """Latest earlier epoch dir of the same run that saved codes.npy."""
    parent = output_dir.parent
    if not parent.is_dir():
        return None
    candidates = sorted(
        d for d in parent.iterdir()
        if d.is_dir() and d.name < output_dir.name and (d / "codes.npy").is_file()
    )
    return np.load(str(candidates[-1] / "codes.npy")) if candidates else None


def _render_bsq_figures(
    units: torch.Tensor,
    codes: torch.Tensor,
    code_dim: int,
    prev_codes: np.ndarray | None,
) -> list[tuple[str, str]]:
    """Bit histograms + Hamming usage/flow graph, as (title, base64-png)."""
    import base64
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def to_b64(fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    figures: list[tuple[str, str]] = []
    u = units.numpy()
    corner = 1.0 / math.sqrt(code_dim)

    # 1) Per-bit u_i histograms: the bit boundary is u_i = 0, so boundary
    # pile-up (and any confidence-term valley) is directly visible per bit.
    fig, axes = plt.subplots(1, code_dim, figsize=(2.8 * code_dim, 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        ax.hist(u[:, i], bins=60, color="#4878d0")
        ax.axvline(0.0, color="red", lw=1.2)
        ax.axvline(corner, color="gray", ls="--", lw=0.8)
        ax.axvline(-corner, color="gray", ls="--", lw=0.8)
        ax.set_title(f"bit {i}  (+1: {100.0 * float((u[:, i] >= 0).mean()):.0f}%)", fontsize=9)
        ax.set_xlim(-3 * corner, 3 * corner)
    fig.suptitle("continuous u_i per bit — red = bit boundary, dashed = corner ±1/√L")
    figures.append(("bit_histograms", to_b64(fig)))

    # 2) Hamming graph: nodes layered by popcount, gray edges = 1-bit
    # neighborhoods, node size = usage, red edges = skill migration since the
    # previous evaluated epoch (1-bit flips only; larger jumps counted apart).
    n_codes = 2 ** code_dim
    counts = np.bincount(codes.numpy(), minlength=n_codes)
    popcount = np.array([bin(c).count("1") for c in range(n_codes)])
    pos = {}
    for weight in range(code_dim + 1):
        members = [c for c in range(n_codes) if popcount[c] == weight]
        for k, c in enumerate(members):
            pos[c] = (k - (len(members) - 1) / 2.0, weight)
    fig, ax = plt.subplots(figsize=(1.6 * code_dim + 4, 0.9 * code_dim + 3))
    for c in range(n_codes):
        for b in range(code_dim):
            d = c ^ (1 << b)
            if d > c:
                ax.plot(*zip(pos[c], pos[d]), color="#cccccc", lw=0.5, zorder=1)
    moved_title = ""
    if prev_codes is not None and len(prev_codes) == len(codes):
        now = codes.numpy()
        flips = now != prev_codes
        one_bit: dict[tuple[int, int], int] = {}
        multi = 0
        for a, b in zip(prev_codes[flips], now[flips]):
            if bin(int(a) ^ int(b)).count("1") == 1:
                key = (min(int(a), int(b)), max(int(a), int(b)))
                one_bit[key] = one_bit.get(key, 0) + 1
            else:
                multi += 1
        if one_bit:
            peak = max(one_bit.values())
            for (a, b), traffic in one_bit.items():
                ax.plot(*zip(pos[a], pos[b]), color="#d65f5f",
                        lw=0.5 + 4.0 * traffic / peak, alpha=0.8, zorder=2)
        moved_title = (
            f" | moved {int(flips.sum())}/{len(now)} "
            f"(1-bit {sum(one_bit.values())}, multi-bit {multi})"
        )
    size = 40.0 + 1500.0 * counts / max(counts.max(), 1)
    xs = [pos[c][0] for c in range(n_codes)]
    ys = [pos[c][1] for c in range(n_codes)]
    ax.scatter(xs, ys, s=size, c="#4878d0", zorder=3, alpha=0.9)
    for c in range(n_codes):
        ax.annotate(format(c, f"0{code_dim}b"), pos[c], fontsize=6,
                    ha="center", va="center", zorder=4)
    ax.set_yticks(range(code_dim + 1))
    ax.set_ylabel("popcount")
    ax.set_xticks([])
    ax.set_title(f"Hamming graph — node size = usage{moved_title}")
    figures.append(("hamming_usage_flow", to_b64(fig)))
    return figures


# ── report ─────────────────────────────────────────────────────────────────────


def _render_html(payload: dict, figures: list[tuple[str, str]] | None = None) -> str:
    rows = []
    split_names = list(payload["splits"].keys())
    keys = list(payload["splits"][split_names[0]].keys())
    header = "".join(f"<th>{escape(name)}</th>" for name in split_names)
    for key in keys:
        cells = "".join(
            f"<td>{payload['splits'][name][key]:.4f}</td>"
            if isinstance(payload["splits"][name][key], float)
            else f"<td>{escape(str(payload['splits'][name][key]))}</td>"
            for name in split_names
        )
        rows.append(f"<tr><th>{escape(key)}</th>{cells}</tr>")
    title = f"{payload['run_name']} {payload['checkpoint']}"
    figure_html = "".join(
        f"<h3>{escape(name)}</h3><img src='data:image/png;base64,{b64}' style='max-width:100%'>"
        for name, b64 in (figures or [])
    )
    return (
        "<!doctype html><meta charset='utf-8'>"
        f"<title>{escape(title)}</title>"
        "<style>body{font-family:sans-serif;margin:2em}table{border-collapse:collapse}"
        "td,th{border:1px solid #999;padding:4px 10px;text-align:right}"
        "th{background:#f0f0f0}</style>"
        f"<h2>FSQ-original eval — {escape(title)}</h2>"
        f"<p>arch={escape(payload['decoder_arch'])} quantizer={escape(payload['quantizer'])} "
        f"split_fingerprint={escape(payload['split_fingerprint'])}</p>"
        f"<table><tr><th>metric</th>{header}</tr>{''.join(rows)}</table>"
        f"{figure_html}"
    )


def main(args: Args) -> None:
    from FSQ_original import FSQOriginalDataset, deterministic_split, load_fsq_original_model
    from skills_bundle import load_skills

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_fsq_original_model(args.model_path, device)
    segments, actions, metadata = load_skills(Path(args.skills_dir))
    val_ids, train_ids, fingerprint = deterministic_split(
        len(segments), metadata, args.val_split
    )
    if args.max_skills and args.max_skills < len(segments):
        keep_every = max(1, len(segments) // args.max_skills)
        val_ids = val_ids[::keep_every]
        train_ids = train_ids[::keep_every]
        print(f"[FSQ-orig-eval] subsampled to train={len(train_ids)} val={len(val_ids)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {}
    all_codes = all_units = None
    for name, ids in (("train", train_ids), ("val", val_ids), ("all", train_ids + val_ids)):
        subset_segments = [segments[i] for i in ids]
        subset_actions = [actions[i] for i in ids]
        dataset = FSQOriginalDataset(
            subset_segments,
            [metadata[i] for i in ids],
            cfg,
            actions=subset_actions if cfg.decoder_arch == "rnn" else None,
        )
        codes, margins, z_norms, units = _encode_all(model, dataset, device, args.batch_size)
        if name == "all":
            all_codes, all_units = codes, units
        metrics = _codebook_metrics(codes, margins, model.fsq.codebook_size)
        if getattr(cfg, "quantizer", "fsq") == "bsq":
            # Per-bit +1 usage: healthy BSQ keeps every bit away from 0%/100%.
            code_dim = int(cfg.bsq_code_dim)
            bits = ((codes.long()[:, None] >> torch.arange(code_dim)) & 1).float()
            ratio = bits.mean(dim=0) * 100.0
            metrics.update({
                "bit_plus_ratio_min_pct": float(ratio.min()),
                "bit_plus_ratio_max_pct": float(ratio.max()),
                "bit_plus_ratio_mean_dev_pct": float((ratio - 50.0).abs().mean()),
            })
        if cfg.decoder_arch == "rnn":
            metrics.update(
                _rnn_recon_metrics(
                    model, dataset, subset_actions, z_norms, device,
                    args.batch_size, args.term_threshold,
                )
            )
        else:
            metrics.update(
                _oneshot_recon_metrics(
                    model, dataset, subset_segments, z_norms, device, args.batch_size
                )
            )
        splits[name] = metrics
        print(f"[FSQ-orig-eval] {name}: " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))

    model_path = Path(args.model_path)
    payload = {
        "format": "fsq_original_eval_v1",
        "run_name": model_path.parent.name,
        "checkpoint": model_path.stem,
        "decoder_arch": cfg.decoder_arch,
        "encoder_input_mode": cfg.encoder_input_mode,
        "quantizer": getattr(cfg, "quantizer", "fsq"),
        "codebook_size": int(model.fsq.codebook_size),
        "fsq_levels": [int(v) for v in cfg.fsq_levels],
        "term_threshold": args.term_threshold,
        "val_split": args.val_split,
        "split_fingerprint": fingerprint,
        "splits": splits,
    }
    figures = None
    if all_units is not None:
        # Codes are saved in the deterministic (train_ids + val_ids) order, so
        # sibling epoch evaluations of the same run align skill-for-skill and
        # the Hamming graph can show migration since the previous epoch.
        np.save(str(output_dir / "codes.npy"), all_codes.numpy())
        try:
            figures = _render_bsq_figures(
                all_units, all_codes, int(cfg.bsq_code_dim), _previous_codes(output_dir)
            )
        except Exception as error:  # matplotlib absent/headless quirks must not fail the eval
            print(f"[FSQ-orig-eval] figure rendering skipped: {error}")
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    (output_dir / "fsq_original_eval.html").write_text(_render_html(payload, figures))
    print(f"[FSQ-orig-eval] wrote {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
