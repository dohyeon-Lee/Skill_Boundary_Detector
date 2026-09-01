#!/usr/bin/env python3
"""Build CALVIN play-pretrain, language-pretrain, and held-out datasets."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR / "src"))

from calvin_task_split import build  # noqa: E402
from calvin_task_split_config import DEFAULT_CONFIG_PATH, load_settings  # noqa: E402


def main() -> None:
    config_path = Path(
        os.environ.get("CALVIN_LONG_HORIZON_SPLIT_CONFIG", DEFAULT_CONFIG_PATH)
    )
    try:
        build(load_settings(config_path))
    except (OSError, TypeError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
