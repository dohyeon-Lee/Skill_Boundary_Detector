#!/usr/bin/env python3
"""Resolve BSQ training settings.

Delegates everything to the FSQ_original resolver (same structure, same
fsq_orig_* knobs), then swaps the run-name quantizer tag fsq{levels} ->
bsq{code_dim} and adds the BSQ-only keys. Runs live under the same
outputs/FSQ root; the bsq tag keeps them disjoint from every FSQ run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "FSQ_original" / "src"))
sys.path.insert(0, str(_HERE.parent.parent / "src"))

from fsq_original_config import resolve as _resolve_fsq_original  # noqa: E402
from train_skills_config import as_bool, get_value, load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent / "bsq_config.yaml"


def resolve(cfg: dict, dataset: str | None = None) -> dict:
    settings = _resolve_fsq_original(cfg, dataset=dataset)
    code_dim = int(get_value(cfg, "bsq_code_dim", 5))
    if code_dim < 2:
        raise ValueError(f"bsq_code_dim must be >= 2, got {code_dim}.")
    fsq_tag = settings["fsq_tag"]
    bsq_tag = f"bsq{code_dim}"
    run_name = settings["fsq_run_name"]
    # Idempotent: under Slurm the sbatch re-resolves with FSQ_RUN_NAME already
    # exported by the submit shell (get_value prefers env), so the name may
    # already carry the bsq tag.
    if f"_{bsq_tag}" not in run_name:
        if f"_{fsq_tag}" not in run_name:
            raise ValueError(
                f"Could not place the bsq tag: neither {bsq_tag!r} nor {fsq_tag!r} "
                f"in {run_name!r}."
            )
        run_name = run_name.replace(f"_{fsq_tag}", f"_{bsq_tag}", 1)
    settings.update({
        "fsq_run_name": run_name,
        "fsq_output_dir": settings["fsq_outputs_root"] / run_name,
        "fsq_tag": bsq_tag,
        "bsq_code_dim": code_dim,
        "bsq_inv_temperature": str(get_value(cfg, "bsq_inv_temperature", 10.0)),
        "bsq_entropy_conf_weight": str(get_value(cfg, "bsq_entropy_conf_weight", 0.1)),
        "bsq_entropy_div_weight": str(get_value(cfg, "bsq_entropy_div_weight", 0.1)),
        "bsq_entropy_joint": as_bool(get_value(cfg, "bsq_entropy_joint", True)),
    })
    return settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    settings = resolve(cfg, dataset=args.dataset)
    if args.shell:
        print_shell(settings)
        return
    for key, value in settings.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
