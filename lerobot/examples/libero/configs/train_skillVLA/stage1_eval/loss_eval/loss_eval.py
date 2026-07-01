#!/usr/bin/env python3
"""Offline front-vs-back SKILL loss eval (no sim).

The action_weight objective up-weights chunks landing near a skill's END, so its benefit shows up as a
LOWER action MSE in the high within-skill-progress region — which the aggregate `action_loss` hides. That
breakdown was added as a train-time wandb panel (`action_loss_prog/*`), but runs finished BEFORE it existed
can't show it. This tool recovers it OFFLINE from a trained checkpoint: it runs the SAME flow-matching
forward the trainer runs (dataset + processor from the checkpoint), takes the per-chunk MSE
(`forward(..., reduction="none")`), and buckets it by the chunk's ENDPOINT within-skill progress
(skill start 0 → skill end 1) — exactly mirroring modeling_skill_expert's diagnostic.

Reads the SAME `models` list as stage1_eval_config.yaml, so it compares whatever models are listed
(1 → its own buckets; 2+ → a side-by-side table). No LIBERO / sim needed — just the dataset + the model.

  ./run.sh                      # uses ../stage1_eval_config.yaml
  ./run.sh --n_batches 100      # more chunks → tighter estimate (default 50 × batch_size)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))          # stage1_eval/src → _resolve_model, _auto_labels
from stage1_eval_config import _auto_labels, _resolve_model  # noqa: E402

# train_skills_config (yaml helpers) is on sys.path via stage1_eval_config's own insert.
from train_skills_config import get_value, load_config  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.policies.skill_expert.configuration_skill_expert import SkillExpertConfig  # noqa: E402  (registers)
from lerobot.utils.constants import ACTION  # noqa: E402

# Progress buckets (chunk endpoint within-skill position). 90-100 = the skill-END handoff action_weight targets.
_EDGES = [(0.0, 0.5), (0.5, 0.9), (0.9, 1.01)]


def _resolve_models(cfg: dict) -> list[dict]:
    """Resolve the yaml's `models` (or single model_dir) to [{policy_path, skill_dir, label}], reusing the
    stage1_eval emitter's path logic."""
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    skillvla_root = (project_root / str(get_value(cfg, "dataset_root", "dataset"))
                     / str(get_value(cfg, "skillvla_dataset_root", "skillvla_dataset")))
    vla_root = project_root / str(get_value(cfg, "outputs_root", "outputs")) / "skillVLA_stage1"
    source_yaml = str(get_value(cfg, "source_dataset", "")).strip()
    default_ckpt = str(get_value(cfg, "checkpoint", "last"))

    models_yaml = get_value(cfg, "models", None)
    if isinstance(models_yaml, list) and models_yaml:
        entries = [{"model_dir": str(get_value(e, "model_dir")),
                    "checkpoint": str(get_value(e, "checkpoint", default_ckpt)),
                    "label": str(get_value(e, "label", "")).strip()} for e in models_yaml]
    else:
        entries = [{"model_dir": str(get_value(cfg, "model_dir")), "checkpoint": default_ckpt, "label": ""}]

    labels = _auto_labels([e["model_dir"] for e in entries])
    out = []
    for e, auto in zip(entries, labels):
        r = _resolve_model(e["model_dir"], e["checkpoint"], skillvla_root=skillvla_root,
                           vla_root=vla_root, source_yaml=source_yaml)
        out.append({"policy_path": r["policy_path"], "skill_dir": r["skill_label_dataset_dir"],
                    "label": e["label"] or auto, "checkpoint": e["checkpoint"]})
    return out


def _endpoint_progress(batch: dict, K: int, dev) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-chunk ENDPOINT within-skill progress (0 skill-start → 1 skill-end) + a validity mask — mirrors
    modeling_skill_expert.forward (anchor = last within-skill, non-pad step of the chunk)."""
    b = batch["skill_ds"].shape[0]
    ds = batch["skill_ds"].to(dev).float().view(b)
    de = batch["skill_de"].to(dev).float().view(b)
    valid = torch.ones(b, K, dtype=torch.bool, device=dev)
    pad = batch.get("action_is_pad")
    if pad is not None:
        valid &= ~pad.to(dev).bool()
    valid &= torch.arange(K, device=dev).view(1, K) <= de.view(b, 1)      # within the current skill
    idx = torch.arange(K, device=dev).view(1, K).expand(b, K)
    last_valid = torch.where(valid, idx, torch.full_like(idx, -1)).max(dim=1).values   # anchor (-1 if none)
    anchor = last_valid.clamp(min=0).float()
    prog_end = ((ds + anchor) / (ds + de).clamp(min=1.0)).clamp(0.0, 1.0)
    return prog_end, last_valid >= 0


def eval_model(spec: dict, cfg: dict, *, n_batches: int, batch_size: int, device, seed: int) -> dict:
    """Load the checkpoint + its dataset, run the flow-matching forward on n_batches, and accumulate the
    per-chunk plain MSE bucketed by endpoint within-skill progress. Returns {overall, buckets:{k:mean}, counts}."""
    policy_path, skill_dir = str(spec["policy_path"]), str(spec["skill_dir"])
    pcfg = PreTrainedConfig.from_pretrained(policy_path)
    pcfg.pretrained_path = policy_path
    pcfg.device = str(device)
    pcfg.train_terminator = False   # skip the disjoint terminator (unused here); action-expert loss unaffected

    ds_meta_kwargs = dict(root=skill_dir)
    delta = resolve_delta_timestamps(pcfg, LeRobotDataset("x/loss_eval", **ds_meta_kwargs).meta)
    dataset = LeRobotDataset("x/loss_eval", delta_timestamps=delta, video_backend="pyav", **ds_meta_kwargs)
    model = make_policy(cfg=pcfg, ds_meta=dataset.meta)
    model.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=pcfg, pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}})

    # SHUFFLE with a fixed generator → a REPRESENTATIVE sample spread across all tasks/skills (NOT just the
    # first episodes, as shuffle=False would give), and the SAME order across models on the same dataset
    # (paired, lower comparison variance). n_batches <= 0 → a full pass over the dataset.
    g = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g,
                                         num_workers=4, drop_last=True)
    n_iter = n_batches if n_batches > 0 else len(loader)
    frac = 100.0 * min(n_iter * batch_size, len(dataset)) / max(len(dataset), 1)
    print(f"  dataset {len(dataset):,} chunks → sampling {n_iter} batches = {n_iter * batch_size:,} "
          f"({frac:.1f}%){' [FULL pass]' if n_batches <= 0 else ''}")
    sums = {f"{int(lo*100):02d}-{int(min(hi,1.0)*100):02d}": 0.0 for lo, hi in _EDGES}
    cnts = {k: 0 for k in sums}
    tot_sum, tot_cnt = 0.0, 0
    torch.manual_seed(seed)   # fixed → paired flow-times across models (lower comparison variance)
    it = iter(loader)
    for _ in range(n_iter):
        try:
            batch = next(it)
        except StopIteration:
            break
        batch = preprocessor(batch)
        with torch.no_grad():
            per_sample, _ = model.forward(batch, reduction="none")       # (B,) per-chunk plain MSE
        per_sample = per_sample.detach().float()
        K = batch[ACTION].shape[1]
        prog_end, valid_s = _endpoint_progress(batch, K, per_sample.device)
        ps = per_sample[valid_s]
        pe = prog_end[valid_s]
        tot_sum += float(ps.sum()); tot_cnt += int(ps.numel())
        for lo, hi in _EDGES:
            m = (pe >= lo) & (pe < hi)
            key = f"{int(lo*100):02d}-{int(min(hi,1.0)*100):02d}"
            sums[key] += float(ps[m].sum()); cnts[key] += int(m.sum())
    buckets = {k: (sums[k] / cnts[k] if cnts[k] else float("nan")) for k in sums}
    return {"label": spec["label"], "overall": (tot_sum / tot_cnt if tot_cnt else float("nan")),
            "n_chunks": tot_cnt, "buckets": buckets, "counts": cnts}


def _keys() -> list[str]:
    return [f"{int(lo*100):02d}-{int(min(hi, 1.0)*100):02d}" for lo, hi in _EDGES]


def _bar(v: float, vmax: float, width: int = 24) -> str:
    """A little ASCII bar (│████░░) so the table is readable without opening the PNG."""
    if not math.isfinite(v) or vmax <= 0:
        return " " * width
    n = int(round(width * v / vmax))
    return "█" * n + "░" * (width - n)


def _format_report(results: list[dict]) -> str:
    """Human-readable text report: a numeric table + an ASCII bar per (model, bucket), + the winner."""
    keys = _keys()
    lines = ["Front→back SKILL loss  (per-chunk action MSE, bucketed by within-skill ENDPOINT progress)",
             "action_weight up-weights the skill END → a LOWER 90-100 bar than plain means it worked.", ""]
    head = f"{'model':<18} {'overall':>9} " + " ".join(f"{'prog '+k:>11}" for k in keys)
    lines.append(head)
    lines.append("-" * len(head))
    for r in results:
        cells = " ".join(f"{r['buckets'][k]:>11.5f}" if math.isfinite(r['buckets'][k]) else f"{'nan':>11}"
                         for k in keys)
        lines.append(f"{r['label']:<18} {r['overall']:>9.5f} {cells}")
    # per-bucket ASCII bars (shared scale per bucket → bars comparable across models within a column)
    for k in keys:
        vals = [r["buckets"][k] for r in results]
        vmax = max((v for v in vals if math.isfinite(v)), default=0.0)
        lines.append("")
        lines.append(f"prog {k}  (max={vmax:.5f})")
        for r in results:
            v = r["buckets"][k]
            vs = f"{v:.5f}" if math.isfinite(v) else "nan"
            lines.append(f"  {r['label']:<16} {_bar(v, vmax)} {vs}")
    lines.append("")
    lines.append("chunks/bucket: " + "; ".join(f"{r['label']}={r['counts']}" for r in results))
    if len(results) >= 2:
        hi = keys[-1]
        finite = [(r["label"], r["buckets"][hi]) for r in results if math.isfinite(r["buckets"][hi])]
        if finite:
            best = min(finite, key=lambda t: t[1])
            lines.append(f"\n→ lowest skill-END (prog {hi}) loss: {best[0]} ({best[1]:.5f}) — "
                         "where action_weight should help most.")
    return "\n".join(lines)


def _plot(results: list[dict], out_png: Path) -> Path | None:
    """Grouped bar chart (buckets on x, one bar per model) → PNG. Skips gracefully if matplotlib is absent."""
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] PNG skipped (matplotlib unavailable): {exc}")
        return None
    keys = _keys()
    x = np.arange(len(keys))
    n = max(len(results), 1)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(2.4 * len(keys) + 2, 5))
    for i, r in enumerate(results):
        vals = [r["buckets"][k] if math.isfinite(r["buckets"][k]) else 0.0 for k in keys]
        bars = ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=r["label"])
        ax.bar_label(bars, fmt="%.4f", fontsize=7, rotation=90, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"prog\n{k}" for k in keys])
    ax.set_ylabel("per-chunk action MSE")
    ax.set_title("Front→back skill loss  (lower at 90-100 = action_weight bought skill-end accuracy)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=_HERE / "loss_eval_config.yaml")
    ap.add_argument("--n_batches", type=int, default=None, help="override yaml n_batches")
    ap.add_argument("--batch_size", type=int, default=None, help="override yaml batch_size")
    ap.add_argument("--seed", type=int, default=None, help="override yaml seed")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    n_batches = args.n_batches if args.n_batches is not None else int(get_value(cfg, "n_batches", 50))
    batch_size = args.batch_size if args.batch_size is not None else int(get_value(cfg, "batch_size", 64))
    seed = args.seed if args.seed is not None else int(get_value(cfg, "seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = _resolve_models(cfg)
    print(f"loss-eval: {len(specs)} model(s), {n_batches}×{batch_size} chunks each, device={device}")
    results = []
    for spec in specs:
        print(f"\n[{spec['label']}] {spec['policy_path']}")
        results.append(eval_model(spec, cfg, n_batches=n_batches, batch_size=batch_size,
                                  device=device, seed=seed))
    report = _format_report(results)
    print("\n" + report)

    base = Path(args.out or (_HERE / "outputs" / ("loss_eval_" + "_vs_".join(r["label"] for r in results))))
    base = base.with_suffix("")                  # strip any extension the user passed
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".txt").write_text(report + "\n")
    base.with_suffix(".json").write_text(json.dumps(results, indent=2))
    png = _plot(results, base.with_suffix(".png"))
    print("\nwrote:")
    for p in (base.with_suffix(".txt"), base.with_suffix(".json"), png):
        if p:
            print(f"  {p}")


if __name__ == "__main__":
    main()
