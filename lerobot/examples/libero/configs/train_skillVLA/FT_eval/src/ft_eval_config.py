#!/usr/bin/env python3
"""Resolve FT evaluation through the maintained Stage-2 evaluator."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_STAGE2_EVAL_SRC = _HERE.parent.parent.parent / "stage2_eval" / "src"
sys.path.insert(0, str(_STAGE2_EVAL_SRC))

from stage2_eval_config import build_settings as _build_stage2_settings  # noqa: E402

sys.path.insert(0, str(_HERE.parent.parent.parent.parent / "train_skills" / "src"))
from train_skills_config import load_config, print_shell  # noqa: E402

DEFAULT_CONFIG_PATH = _HERE.parent.parent / "ft_eval_config.yaml"


def build_settings(config: dict) -> dict:
    """Use Stage-2 contracts, but default model and output roots to FT."""
    resolved_config = copy.deepcopy(config)
    resolved_config.setdefault("outputs_subdir", "skillVLA_FT")
    settings = _build_stage2_settings(resolved_config)

    # Keep evaluation products physically separate from stage2_eval outputs.
    output_name = Path(settings["eval_out_dir"]).name
    settings["eval_out_dir"] = _HERE.parent.parent / "outputs" / output_name
    settings["wandb_run_name"] = settings["wandb_run_name"].replace(
        "S2eval_", "FTeval_", 1
    )
    return settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    settings = build_settings(load_config(args.config))
    if args.shell:
        print_shell(settings)
    else:
        for key, value in settings.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
