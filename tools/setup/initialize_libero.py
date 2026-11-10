"""Create LIBERO's default paths without its first-import interactive prompt."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import yaml


def initialize() -> Path:
    folder = Path(os.environ.get("LIBERO_CONFIG_PATH", "~/.libero")).expanduser()
    config = folder / "config.yaml"
    if config.exists():
        return config
    # Inspect only the top-level package; importing libero.libero would prompt.
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("LIBERO package is not installed")
    benchmark = Path(next(iter(spec.submodule_search_locations))) / "libero"
    if not (benchmark / "__init__.py").is_file():
        raise RuntimeError("LIBERO benchmark package was not found")
    values = {"benchmark_root": str(benchmark), "bddl_files": str(benchmark / "bddl_files"),
              "init_states": str(benchmark / "init_files"), "datasets": str(benchmark.parent / "datasets"),
              "assets": str(benchmark / "assets")}
    folder.mkdir(parents=True, exist_ok=True)
    try:
        with config.open("x") as stream:
            yaml.safe_dump(values, stream)
    except FileExistsError:
        pass
    return config


if __name__ == "__main__":
    print("LIBERO paths ready:", initialize())
