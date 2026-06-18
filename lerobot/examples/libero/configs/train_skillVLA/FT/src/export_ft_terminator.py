#!/usr/bin/env python3
"""Export the FT-co-trained terminator into a standalone FSQ checkpoint for FT_eval.

During FT the terminator lives in the policy as ``model.fsq_term_train.*`` (a full SplineFSQAE whose
terminator-path submodules were adapted; the rest stayed frozen = identical to the base FSQ). This
takes those weights out of the FT checkpoint and merges them onto the base FSQ.pt's ``model_state``,
writing an ``FSQ_ft.pt`` in the SAME format ({"cfg", "model_state"}) that the policy's
``load_terminator`` consumes. FT_eval then points ``fsq_path`` at this file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# The base FSQ.pt pickles a SplineFSQAEConfig from FSQ.py (examples/libero) → put it on sys.path so
# torch.load can unpickle the cfg (mirrors SkillVLAPytorch._construct_fsq).
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

_PREFIX = "fsq_term_train."


def _load_ft_state(pretrained_model: Path, *, required: bool = True) -> dict | None:
    """Read the FT checkpoint's policy weights and return the SplineFSQAE-keyed terminator state
    (``model.fsq_term_train.<k>`` → ``<k>``). Supports safetensors and torch .bin/.pt. Returns None if
    the checkpoint has no co-trained terminator and ``required=False``."""
    st = pretrained_model / "model.safetensors"
    if st.is_file():
        from safetensors.torch import load_file  # noqa: PLC0415

        raw = load_file(str(st))
    else:
        cand = next((p for p in (pretrained_model / "pytorch_model.bin",
                                 pretrained_model / "model.bin") if p.is_file()), None)
        if cand is None:
            raise FileNotFoundError(f"No model weights under {pretrained_model} (model.safetensors/.bin).")
        raw = torch.load(str(cand), map_location="cpu", weights_only=False)
    out = {}
    for k, v in raw.items():
        i = k.find(_PREFIX)
        if i != -1:
            out[k[i + len(_PREFIX):]] = v
    if not out:
        if not required:
            return None
        raise ValueError(
            f"No '{_PREFIX}' weights in {pretrained_model} — was the run trained with train_terminator=true?")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft_run_dir", type=Path, required=True, help="FT output dir (…/skillVLA_FT/<run>)")
    ap.add_argument("--checkpoint", default="last", help="checkpoint step or 'last'")
    ap.add_argument("--base_fsq", type=Path, required=True, help="base FSQ.pt (same codebook) to merge onto")
    ap.add_argument("--out", type=Path, required=True, help="output FSQ_ft.pt path")
    ap.add_argument("--skip_if_no_terminator", action="store_true",
                    help="exit 0 without writing if the checkpoint has no co-trained terminator")
    args = ap.parse_args()

    pretrained_model = args.ft_run_dir / "checkpoints" / str(args.checkpoint) / "pretrained_model"
    ft_state = _load_ft_state(pretrained_model, required=not args.skip_if_no_terminator)
    if ft_state is None:
        print(f"[export] no co-trained terminator in {pretrained_model}; skipping (eval falls back to base FSQ).")
        return

    base = torch.load(str(args.base_fsq), map_location="cpu", weights_only=False)
    if "model_state" not in base or "cfg" not in base:
        raise ValueError(f"{args.base_fsq} is not an FSQ checkpoint ({{cfg, model_state}}).")
    merged = dict(base["model_state"])
    applied = [k for k in ft_state if k in merged]
    extra = [k for k in ft_state if k not in merged]
    merged.update({k: v for k, v in ft_state.items() if k in merged})  # only known FSQ keys

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": base["cfg"], "model_state": merged}, str(args.out))
    print(f"[export] FSQ_ft.pt written: {args.out}")
    print(f"[export]   merged {len(applied)} terminator/decoder tensors from {pretrained_model}")
    if extra:
        print(f"[export]   ignored {len(extra)} non-FSQ keys (e.g. {extra[0]})")


if __name__ == "__main__":
    main()
