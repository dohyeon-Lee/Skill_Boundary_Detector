#!/usr/bin/env python3
"""Reuse ABC's synchronized comparison for saved DROID SBD reports."""
from __future__ import annotations

import argparse
from pathlib import Path

from config import load_workflow


EVAL_DIR = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=EVAL_DIR / 'outputs/together/together_config.yaml')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    report = load_workflow('together_report').build(
        args.config, dry_run=args.dry_run, outputs_root=EVAL_DIR / 'outputs')
    if not args.dry_run:
        # Retain the shared template/compatibility marker and all sync logic.
        temporary = report.with_suffix('.html.tmp')
        temporary.write_text(report.read_text().replace('ABC SBD Together', 'DROID SBD Together'))
        temporary.replace(report)


if __name__ == '__main__':
    main()
