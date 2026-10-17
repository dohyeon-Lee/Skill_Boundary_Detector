#!/usr/bin/env python3
"""Validate the isolated MolmoAct2 environment without models, data, or CUDA startup."""
from __future__ import annotations

import importlib.metadata as metadata
import importlib.util
import os
from pathlib import Path
import platform
import re
import runpy
import sys


def read_pins(lock: Path) -> dict[str, str]:
    pins = {}
    for line in lock.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if match is None:
            raise ValueError(f"Expected an exact package pin in {lock}: {line!r}")
        name = re.sub(r"[-_.]+", "-", match[1]).lower()
        if name in pins:
            raise ValueError(f"Duplicate package pin: {name}")
        pins[name] = match[2]
    if not pins:
        raise ValueError(f"Empty dependency lock: {lock}")
    return pins


def package_problems(pins: dict[str, str]) -> list[str]:
    try:
        from packaging.version import Version
    except ImportError:
        Version = str  # Report a missing packaging pin alongside the other missing packages.
    problems = []
    for name, expected in pins.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            problems.append(f"missing: {name}=={expected}")
            continue
        # PEP 440 normalizes legacy versions (thop's '-' becomes '.post'), while
        # still comparing the CUDA wheel suffix exactly.
        if Version(actual) != Version(expected):
            problems.append(f"{name}: {actual} (expected {expected})")
    return problems


def editable_problems(source: Path) -> list[str]:
    problems = []
    for module, directory in (("olmo", source / "olmo"), ("lerobot", source / "lerobot/src/lerobot")):
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            problems.append(f"missing editable package: {module}")
        elif not Path(spec.origin).resolve().is_relative_to(directory.resolve()):
            problems.append(f"wrong {module}: {spec.origin} (expected {directory})")
    return problems


def main() -> None:
    module = Path(__file__).resolve().parents[2]
    root = module.parents[4]
    source = root / "third_party/molmoact2/experiments"
    expected_venv = root / ".venv_molmoact2"
    problems = []
    if sys.version_info[:2] != (3, 12):
        problems.append(f"Python {platform.python_version()} (expected 3.12.x)")
    if Path(sys.prefix).resolve() != expected_venv.resolve():
        problems.append(f"wrong venv: {sys.prefix} (expected {expected_venv})")
    pins = read_pins(module / "requirements.lock.txt")
    problems.extend(package_problems(pins))
    # Check installed editables BEFORE inserting any source path, to catch cross-env imports.
    problems.extend(editable_problems(source))
    if problems:
        raise RuntimeError("MolmoAct2 environment check failed:\n  - " + "\n  - ".join(problems))

    os.environ["MOLMO_DATA_DIR"] = str(root / ".cache/molmoact2")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    sys.path.insert(0, str(source))  # launch_scripts is not an installed package.
    import torch
    import torchvision
    import transformers
    import av

    if torch.version.cuda != "12.8":
        raise RuntimeError(f"Expected CUDA 12.8 torch wheels, got {torch.version.cuda}")
    # __main__ is deliberately NOT used: imports only, no model/data loading or training.
    runpy.run_path(str(source / "launch_scripts/train_lerobot.py"), run_name="_molmoact2_setup_check")
    print(f"MolmoAct2 환경 검증 통과: Python {platform.python_version()}, 고정 패키지 {len(pins)}개 일치")
    print(f"  torch {torch.__version__}, torchvision {torchvision.__version__}, transformers {transformers.__version__}, av {av.__version__}")
    print(f"  Original trainer imports OK; LeRobot -> {source / 'lerobot/src/lerobot'}")


if __name__ == "__main__":
    main()
