#!/usr/bin/env python3
"""Validate the isolated official ABC Simulator environment and evaluation assets."""
from __future__ import annotations

import importlib.metadata as metadata
import importlib.util
import os
from pathlib import Path
import platform
import re
import sys


ASSET_MANIFEST_FORMAT = "abc_sim_assets_manifest/v1"
ROBOCASA_PACKS = ("objects_lightwheel", "objaverse")


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
    from packaging.version import Version

    problems = []
    for name, expected in pins.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            problems.append(f"missing: {name}=={expected}")
            continue
        if Version(actual) != Version(expected):
            problems.append(f"{name}: {actual} (expected {expected})")
    return problems


def editable_problems(source: Path) -> list[str]:
    problems = []
    for module in ("abc_sim", "abc_minimal"):
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            problems.append(f"missing editable package: {module}")
        elif not Path(spec.origin).resolve().is_relative_to(source.resolve()):
            problems.append(f"wrong {module}: {spec.origin} (expected under {source})")
    return problems


def asset_problems(source: Path) -> list[str]:
    import json

    manifest_path = source / "abc_sim/models/assets_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"unreadable asset manifest {manifest_path}: {exc}"]
    if manifest.get("format") != ASSET_MANIFEST_FORMAT:
        return [f"unexpected asset manifest format: {manifest.get('format')!r}"]

    marker_dir = source / "abc_sim/models/assets/.abc_sim_asset_packages"
    problems = []
    for package in manifest.get("packages", []):
        name = package["name"]
        marker = marker_dir / f"{name}.sha256"
        try:
            marker_hash = marker.read_text().strip()
        except OSError:
            marker_hash = None
        if marker_hash != package["sha256"]:
            problems.append(f"asset package not installed or stale: {name}")
            continue
        destination = source / package["extract_dir"]
        missing = [entry for entry in package["contains"] if not (destination / entry).exists()]
        if missing:
            problems.append(f"asset package {name} is missing: {', '.join(missing)}")

    robocasa = source / "abc_sim/models/assets_robocasa"
    for pack in ROBOCASA_PACKS:
        directory = robocasa / pack
        if not directory.is_dir() or not any(directory.iterdir()):
            problems.append(f"RoboCasa object pack not installed: {pack}")
    return problems


def main() -> None:
    module = Path(__file__).resolve().parent
    root = module.parents[5]
    lerobot_src = root / "lerobot" / "src"
    sys.path.insert(0, str(lerobot_src))
    source = root / "third_party/abc"
    expected_venv = root / ".venv_abc_sim"
    problems = []
    if sys.version_info[:2] != (3, 12):
        problems.append(f"Python {platform.python_version()} (expected 3.12.x)")
    if Path(sys.prefix).resolve() != expected_venv.resolve():
        problems.append(f"wrong venv: {sys.prefix} (expected {expected_venv})")
    pins = read_pins(module / "requirements.lock.txt")
    problems.extend(package_problems(pins))
    problems.extend(editable_problems(source))
    problems.extend(asset_problems(source))
    try:
        conflicting_mujoco = metadata.version("mujoco-core")
    except metadata.PackageNotFoundError:
        pass
    else:
        problems.append(
            f"conflicting mujoco-core=={conflicting_mujoco}; ABC requires the MuJoCo 3.8 distribution only"
        )
    if problems:
        raise RuntimeError("ABC Sim environment check failed:\n  - " + "\n  - ".join(problems))

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    import abc_minimal
    import abc_sim
    import mujoco
    import mujoco_warp
    import torch
    import warp
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    if torch.version.cuda != "12.8":
        raise RuntimeError(f"Expected CUDA 12.8 torch wheels, got {torch.version.cuda}")
    if not callable(abc_sim.make_env):
        raise RuntimeError("abc_sim.make_env is unavailable")
    if PI05Config.__name__ != "PI05Config" or PI05Policy.__name__ != "PI05Policy":
        raise RuntimeError("LeRobot PI0.5 policy runtime is unavailable")
    smoke_model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    mujoco.mj_forward(smoke_model, mujoco.MjData(smoke_model))
    print(
        f"ABC Sim 환경 검증 통과: Python {platform.python_version()}, "
        f"고정 패키지 {len(pins)}개, PI0.5 runtime 및 전체 평가 에셋 일치"
    )
    print(
        f"  torch {torch.__version__}, mujoco {mujoco.__version__}, "
        f"mujoco-warp {metadata.version('mujoco-warp')}, warp {warp.__version__}, source -> {source}"
    )


if __name__ == "__main__":
    main()
