#!/usr/bin/env python
"""Per-block parameter movement during FT: how much did each block get pushed to learn the new task?

Compares an FT checkpoint against its PT source, per block:
  abs   = ‖θ_FT − θ_PT‖                      (raw movement)
  rel   = ‖θ_FT − θ_PT‖ / ‖θ_PT‖             (movement relative to block scale — fair across blocks)
  cos   = ⟨θ_FT−θ_PT, θ_PT⟩ / (‖·‖‖·‖)       (direction: did it grow along or away from PT weights)

Pairs with the flatness map: forgetting ≈ (block fragility) × (block movement). If cyclic forgets
less because it MOVES the VLM less → rel(VLM) smaller for cyclic; if because the VLM is flatter
(moves similarly but survives) → rel(VLM) similar. This script decides which.

Forward-free — just reads the two safetensors. Runs on CPU in seconds.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file

from measure_term2_blocks import block_of

WEIGHTS = "model.safetensors"


def load_model_sd(ckpt_dir: Path) -> dict:
    return load_file(str(ckpt_dir / WEIGHTS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft_ckpt", type=Path, required=True)   # .../pretrained_model
    ap.add_argument("--pt_ckpt", type=Path, required=True)   # .../pretrained_model (FT source)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    ft = load_model_sd(args.ft_ckpt)
    pt = load_model_sd(args.pt_ckpt)
    common = [k for k in ft if k in pt and ft[k].shape == pt[k].shape and torch.is_floating_point(ft[k])]
    missing = [k for k in ft if k not in pt or ft[k].shape != pt[k].shape]
    if missing:
        print(f"  (skipped {len(missing)} non-matching keys, e.g. {missing[:2]})")

    # accumulate per block: Σ‖Δ‖², Σ‖θ_pt‖², Σ⟨Δ,θ_pt⟩
    blk = defaultdict(lambda: {"d2": 0.0, "p2": 0.0, "dp": 0.0, "n": 0})
    for k in common:
        d = (ft[k].float() - pt[k].float())
        p = pt[k].float()
        b = block_of(k)
        blk[b]["d2"] += float((d * d).sum())
        blk[b]["p2"] += float((p * p).sum())
        blk[b]["dp"] += float((d * p).sum())
        blk[b]["n"] += d.numel()

    summary = {}
    for b, s in blk.items():
        absmov = s["d2"] ** 0.5
        pnorm = s["p2"] ** 0.5
        summary[b] = {
            "abs_movement": round(absmov, 4),
            "rel_movement": round(absmov / (pnorm + 1e-12), 5),
            "cos_with_pt": round(s["dp"] / (absmov * pnorm + 1e-12), 4),
            "n_params": s["n"],
        }
    # overall
    tot = {k: sum(blk[b][k] for b in blk) for k in ["d2", "p2", "dp"]}
    summary["_overall"] = {
        "rel_movement": round(tot["d2"] ** 0.5 / (tot["p2"] ** 0.5 + 1e-12), 5),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ft_delta_blocks.json").write_text(json.dumps(summary, indent=2))

    order = ["vision_tower", "language_model", "action_expert", "projectors", "flow_head", "other"]
    print(f"\n{'block':<16}{'rel_move':>10}{'cos_pt':>9}{'abs_move':>11}")
    for b in order:
        if b in summary:
            s = summary[b]
            print(f"{b:<16}{s['rel_movement']:>10.5f}{s['cos_with_pt']:>9.3f}{s['abs_movement']:>11.3f}")
    print(f"{'OVERALL':<16}{summary['_overall']['rel_movement']:>10.5f}")
    print(f"DONE -> {args.out_dir}")


if __name__ == "__main__":
    main()
