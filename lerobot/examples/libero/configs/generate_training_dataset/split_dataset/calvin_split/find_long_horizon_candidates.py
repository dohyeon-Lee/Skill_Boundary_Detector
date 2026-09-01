#!/usr/bin/env python3
"""Find and visualize repeated long-horizon task combinations in CALVIN play data."""

from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR / "src"))

from calvin_long_horizon import run  # noqa: E402
from calvin_long_horizon_config import (  # noqa: E402
    load_config,
    reject_cli_arguments,
    settings,
)


def main() -> None:
    reject_cli_arguments()
    try:
        run(settings(load_config()))
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

