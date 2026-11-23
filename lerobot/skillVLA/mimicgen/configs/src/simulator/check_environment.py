#!/usr/bin/env python3
"""Validate pinned Coffee assets and CPU physics, with optional RGB rendering."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import importlib.util
from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
SOURCE = ROOT / "third_party/mimicgen"
VENV = ROOT / ".venv_mimicgen"


def read_pins(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?==([^\s;]+)", line)
        if match is None:
            raise ValueError(f"Expected an exact package pin: {line!r}")
        name = re.sub(r"[-_.]+", "-", match[1]).lower()
        if name in pins:
            raise ValueError(f"Duplicate package pin: {name}")
        pins[name] = match[2]
    if not pins:
        raise ValueError("Empty dependency lock.")
    return pins


def installation_problems() -> list[str]:
    problems = []
    if sys.version_info[:2] != (3, 12) or Path(sys.prefix).resolve() != VENV.resolve():
        problems.append(f"Use {VENV}/bin/python (Python 3.12), got {sys.executable}")
    expected = (HERE / "source_revision.txt").read_text().strip()
    try:
        actual = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
        if actual != expected:
            problems.append(f"MimicGen source {actual}, expected {expected}")
    except (OSError, subprocess.CalledProcessError):
        problems.append(f"Missing MimicGen Git source: {SOURCE}")
    from packaging.version import Version

    for name, pin in read_pins(HERE / "requirements.lock.txt").items():
        try:
            actual = metadata.version(name)
            if Version(actual) != Version(pin):
                problems.append(f"{name}: {actual}, expected {pin}")
        except metadata.PackageNotFoundError:
            problems.append(f"Missing {name}=={pin}")
    spec = importlib.util.find_spec("mimicgen")
    if spec is None or spec.origin is None or not Path(spec.origin).resolve().is_relative_to(SOURCE.resolve()):
        problems.append("mimicgen must import from this project's pinned third_party/mimicgen")
    assets = SOURCE / "mimicgen/models/robosuite/assets"
    for relative in (
        "objects/coffee_pod.xml", "objects/coffee_body.xml", "objects/coffee_lid.xml",
        "objects/coffee_base.xml", "objects/meshes/coffee_pod.stl",
        "objects/meshes/coffee_body.stl", "objects/meshes/coffee_lid.stl",
        "shapenet_core/mugs/3143a4ac/model.xml",
    ):
        path = assets / relative
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"Missing Coffee asset: {path}")
        else:
            with path.open("rb") as stream:
                if stream.read(100).startswith(b"version https://git-lfs.github.com/spec/"):
                    problems.append(f"Coffee asset is an unresolved Git LFS pointer: {path}")
    return problems


def coffee_check(*, render: bool) -> None:
    import numpy as np
    import robosuite
    import mimicgen  # noqa: F401 -- register official environments
    from mimicgen.envs.robosuite.coffee import Coffee_D0, CoffeePreparation_D0  # noqa: F401
    from robosuite.utils.camera_utils import get_camera_transform_matrix

    cameras = ("agentview", "robot0_eye_in_hand")
    controller = robosuite.load_controller_config(default_controller="OSC_POSE")
    env = robosuite.make(
        env_name="Coffee_D0", robots="Panda", gripper_types="PandaGripper",
        controller_configs=controller, control_freq=20, horizon=400, ignore_done=True,
        use_object_obs=True, use_camera_obs=render,
        has_renderer=False, has_offscreen_renderer=render,
        camera_names=list(cameras), camera_heights=256, camera_widths=256,
        reward_shaping=False, hard_reset=False,
    )
    try:
        # robosuite 1.4.1 computes raw joints for its sin/cos sensors but excludes
        # them from returned observations unless explicitly activated.
        env.modify_observable("robot0_joint_pos", "active", True)
        np.random.seed(0)
        observation = env.reset()
        assert env.action_dim == 7, f"Expected a 7D OSC action, got {env.action_dim}"
        for name, shape in {"robot0_eef_pos": (3,), "robot0_eef_quat": (4,),
                            "robot0_gripper_qpos": (2,), "robot0_joint_pos": (7,)}.items():
            value = np.asarray(observation[name])
            assert value.shape == shape and np.isfinite(value).all(), (name, value)
        action = np.zeros(7)
        action[-1] = -1
        for _ in range(2):
            observation, reward, done, info = env.step(action)
        assert np.isfinite(env.sim.get_state().flatten()).all(), "Non-finite simulator state"
        assert isinstance(env._check_success(), (bool, np.bool_)), "Invalid task success result"
        for camera in cameras:
            matrix = get_camera_transform_matrix(env.sim, camera, 256, 256)
            assert matrix.shape == (4, 4) and np.isfinite(matrix).all(), camera
            if render:
                image = np.asarray(observation[camera + "_image"])
                assert image.shape == (256, 256, 3) and image.dtype == np.uint8, (camera, image.shape)
                assert image.max() > image.min(), f"Blank camera image: {camera}"
            print(f"  {camera}: calibration OK" + (", RGB 256x256 OK" if render else " (render skipped)"))
        print("Coffee_D0: Panda/PandaGripper, OSC delta 7D, 20 Hz, reset + 2 physics steps OK")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="Render both cameras; needs EGL or OSMesa.")
    args = parser.parse_args()
    problems = installation_problems()
    if problems:
        raise SystemExit("MimicGen installation check failed:\n  - " + "\n  - ".join(problems))
    print("Pinned dependencies, editable source, and Coffee assets: OK", flush=True)
    try:
        coffee_check(render=args.render)
    except Exception as error:
        if args.render:
            raise RuntimeError(
                "Coffee render check failed. Run on an allocated GPU with MUJOCO_GL=egl, "
                "or use MUJOCO_GL=osmesa with OSMesa installed. No Slurm allocation was requested."
            ) from error
        raise


if __name__ == "__main__":
    main()
