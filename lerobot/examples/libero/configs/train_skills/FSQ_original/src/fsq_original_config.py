#!/usr/bin/env python3
"""Resolve FSQ-original (one-shot reconstruction) training settings.

Reuses the shared train_skills resolver for dataset/skillset selection, run
naming, and Slurm resources, then adds the FSQ-original-only keys. Runs live
under the SAME outputs/FSQ root as v3 so downstream tooling (skill encoding,
eval, data building) resolves them identically. Run identity comes purely
from fsq_exp — pick distinct tags per variant; cross-format mixups are still
caught by the checkpoint-type resume guards.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "src"))

from train_skills_config import (  # noqa: E402
    as_bool,
    get_value,
    load_config,
    print_shell,
    train_settings,
)

DEFAULT_CONFIG_PATH = _HERE.parent / "fsq_original_config.yaml"


def resolve(cfg: dict, dataset: str | None = None) -> dict:
    settings = train_settings(cfg, dataset=dataset)
    decoder_layers = int(get_value(cfg, "fsq_orig_decoder_layers", 3))
    if decoder_layers < 1:
        raise ValueError(f"fsq_orig_decoder_layers must be >= 1, got {decoder_layers}.")
    decoder_arch = str(get_value(cfg, "fsq_orig_decoder_arch", "oneshot")).strip().lower()
    if decoder_arch not in {"oneshot", "rnn"}:
        raise ValueError(f"fsq_orig_decoder_arch must be oneshot|rnn, got {decoder_arch!r}.")
    encoder_arch = str(get_value(cfg, "fsq_orig_encoder_arch", "spline")).strip().lower()
    if encoder_arch not in {"spline", "action_seq"}:
        raise ValueError(f"fsq_orig_encoder_arch must be spline|action_seq, got {encoder_arch!r}.")
    if encoder_arch == "action_seq" and decoder_arch != "rnn":
        raise ValueError("fsq_orig_encoder_arch=action_seq requires fsq_orig_decoder_arch=rnn.")
    settings.update({
        "fsq_orig_decoder_layers": decoder_layers,
        "fsq_orig_decoder_arch": decoder_arch,
        "fsq_orig_encoder_arch": encoder_arch,
        "fsq_orig_reconstruct_length": as_bool(
            get_value(cfg, "fsq_orig_reconstruct_length", True)
        ),
        "fsq_orig_encoder_length_token": as_bool(
            get_value(cfg, "fsq_orig_encoder_length_token", True)
        ),
        "fsq_orig_action_loss_weight": str(get_value(cfg, "fsq_orig_action_loss_weight", 1.0)),
        "fsq_orig_term_loss_weight": str(get_value(cfg, "fsq_orig_term_loss_weight", 1.0)),
        "fsq_orig_term_pos_weight": str(get_value(cfg, "fsq_orig_term_pos_weight", 1.0)),
        "fsq_orig_term_sigma": str(get_value(cfg, "fsq_orig_term_sigma", 1.0)),
        "fsq_orig_decoder_lr": str(
            get_value(cfg, "fsq_orig_decoder_lr", get_value(cfg, "fsq_encoder_lr", "3e-4"))
        ),
        "fsq_orig_ctrl_loss_weight": str(get_value(cfg, "fsq_orig_ctrl_loss_weight", 1.0)),
        "fsq_orig_length_loss_weight": str(get_value(cfg, "fsq_orig_length_loss_weight", 1.0)),
        # Empty -> "" (sbatch omits the flag -> selection follows the loss weights).
        "fsq_orig_val_select_ctrl_weight": str(
            get_value(cfg, "fsq_orig_val_select_ctrl_weight", "") or ""
        ),
        "fsq_orig_val_select_length_weight": str(
            get_value(cfg, "fsq_orig_val_select_length_weight", "") or ""
        ),
        # BSQ-style entropy knobs; on FSQ runs they only act when
        # fsq_orig_entropy=true (the FSQ-vs-BSQ attribution ablation).
        "fsq_orig_entropy": as_bool(get_value(cfg, "fsq_orig_entropy", False)),
        "bsq_inv_temperature": str(get_value(cfg, "bsq_inv_temperature", 10.0)),
        "bsq_entropy_conf_weight": str(get_value(cfg, "bsq_entropy_conf_weight", 0.1)),
        "bsq_entropy_div_weight": str(get_value(cfg, "bsq_entropy_div_weight", 0.1)),
        # Joint (exact) dataset entropy is the project standard; not a yaml choice.
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
