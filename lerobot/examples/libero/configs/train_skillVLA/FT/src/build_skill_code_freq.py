#!/usr/bin/env python3
"""Build the per-code skill histogram for the VSA-distillation global sampler.

Counts the GT skill codes over the PARENT PT skillvla dataset (the repertoire FT should not forget) →
skill_code_freq.npz (key 'counts', len == skill_vocab_size) next to that dataset's FSQ run dir. The FT
emitter auto-picks it up as vsa_distill_freq_path (blank → the model falls back to a uniform sampler).

The codes come from the dataset parquet: each frame's GT code = skill_sequence[skill_index] (so the
histogram is FRAME-WEIGHTED = actual execution frequency). No model needed.

Usage:
  build_skill_code_freq.py --pt_dataset_root .../{source}/{run_tag}/skillvla [--fsq_levels 5,5,5]
  build_skill_code_freq.py --stage2_run <Stage-2 folder name>   # derive the PT dataset from its config
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_CONFIGS = _HERE.parents[3]
sys.path.insert(0, str(_CONFIGS / "train_skills" / "src"))
from train_skills_config import get_value, load_config  # noqa: E402

GLOBAL_CFG = _CONFIGS / "global_config.yaml"


def build(pt_root: Path, vocab: int) -> np.ndarray:
    """Frame-weighted GT code histogram: per frame, code = skill_sequence[skill_index]."""
    import pyarrow.parquet as pq  # noqa: PLC0415
    files = sorted(glob.glob(str(pt_root / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet under {pt_root}/data/")
    counts = np.zeros(vocab, dtype="float64")
    for f in files:
        t = pq.read_table(f, columns=["skill_index", "skill_sequence"])
        idx = np.asarray(t.column("skill_index").to_pylist(), dtype="int64").reshape(-1)
        seq = t.column("skill_sequence").to_pylist()           # list of per-frame sequences
        for i, s in zip(idx, seq):
            s = np.asarray(s).reshape(-1)
            if 0 <= i < s.shape[0]:
                c = int(s[i])
                if 0 <= c < vocab:
                    counts[c] += 1.0
    return counts.astype("float32")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt_dataset_root", type=Path, help=".../{source}/{run_tag}/skillvla")
    ap.add_argument("--stage2_run", type=str, help="Stage-2 folder under {outputs}/skillVLA_stage2/")
    ap.add_argument("--fsq_levels", type=str, default="", help="e.g. 5,5,5 (else parsed from run_tag)")
    ap.add_argument("--config", type=Path, default=GLOBAL_CFG)
    args = ap.parse_args()

    cfg = load_config(args.config)
    project_root = Path(str(get_value(cfg, "project_root"))).expanduser()
    outputs_root = project_root / str(get_value(cfg, "outputs_root", "outputs"))

    pt_root = args.pt_dataset_root
    if pt_root is None:
        if not args.stage2_run:
            ap.error("pass --pt_dataset_root or --stage2_run")
        tc = outputs_root / "skillVLA_stage2" / args.stage2_run / "checkpoints"
        last = sorted((p.name for p in tc.glob("*") if p.name.isdigit()), key=int)[-1]
        ds = json.loads((tc / last / "pretrained_model" / "train_config.json").read_text())["dataset"]
        pt_root = Path(str(ds["root"]))
    pt_root = pt_root.resolve()

    run_tag = pt_root.parent.name
    if args.fsq_levels:
        levels = [int(x) for x in args.fsq_levels.split(",")]
    else:
        import re
        levels = [int(d) for d in re.search(r"FSQ(\d+)", run_tag).group(1)]
    vocab = int(np.prod(levels))

    counts = build(pt_root, vocab)
    out = pt_root.parent / "skill_code_freq.npz"
    np.savez(out, counts=counts, levels=np.array(levels))
    nz = int((counts > 0).sum())
    print(f"wrote {out}\n  vocab={vocab}  used_codes={nz}/{vocab}  total_samples={int(counts.sum())}")
    top = np.argsort(counts)[::-1][:8]
    print("  top codes:", [(int(c), int(counts[c])) for c in top])


if __name__ == "__main__":
    main()
