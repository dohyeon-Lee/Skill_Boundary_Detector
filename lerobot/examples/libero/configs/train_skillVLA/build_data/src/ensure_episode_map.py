"""Create the source episode map needed by UV generation, once per dataset."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import tempfile


@contextmanager
def _matching_environment(project_root: Path, *, render: bool):
    env = os.environ.copy()
    if not render:
        yield env
        return

    # LangGap matches rendered candidates. Keep its simulator configuration
    # private to this build instead of writing a shared ~/.libero/config.yaml.
    libero_root = project_root / "tools/lerobot-libero/libero"
    benchmark_root = libero_root / "libero"
    if not (benchmark_root / "init_files").is_dir():
        raise FileNotFoundError(f"LIBERO init_files not found: {benchmark_root / 'init_files'}")
    with tempfile.TemporaryDirectory(prefix="skillvla_libero_") as folder:
        Path(folder, "config.yaml").write_text(
            f"benchmark_root: {benchmark_root}\n"
            f"bddl_files: {benchmark_root / 'bddl_files'}\n"
            f"init_states: {benchmark_root / 'init_files'}\n"
            f"datasets: {libero_root / 'datasets'}\n"
            f"assets: {benchmark_root / 'assets'}\n"
        )
        env["LIBERO_CONFIG_PATH"] = folder
        env.setdefault("MUJOCO_GL", "egl")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(project_root / "lerobot/src"), str(project_root / "tools/lerobot-libero")]
            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        yield env


def ensure_episode_map(
    source_dataset_dir: Path,
    original_dataset_dir: Path,
    output: Path,
    project_root: Path,
) -> bool:
    """Reuse an existing map, or build it without overwriting other writers."""
    if output.is_file():
        print(f"Episode map already exists → skip: {output}", flush=True)
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    # PT/FT or different FSQ builds may share this source-level map.
    with output.with_name(f".{output.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.is_file():
            print(f"Episode map already exists → skip: {output}", flush=True)
            return False

        langgap = source_dataset_dir.name.startswith("langgap_")
        builder = Path(__file__).with_name(
            "build_langgap_init_states.py" if langgap else "build_init_states.py"
        )
        print(f"Build source episode map → {output}", flush=True)
        # Publish only a successful complete map; a failed writer must not
        # leave a partial NPZ that the next build would mistake for completion.
        with tempfile.TemporaryDirectory(prefix=f".{output.stem}.", dir=output.parent) as folder:
            temporary = Path(folder) / output.name
            command = [sys.executable, "-u", str(builder),
                       "--lerobot_dataset", str(source_dataset_dir), "--out", str(temporary)]
            if langgap:
                command.extend(["--cache-dir", str(output.parent / ".langgap_episode_exact_cache")])
            else:
                command.extend(["--orig_dataset", str(original_dataset_dir)])
            try:
                with _matching_environment(project_root, render=langgap) as env:
                    subprocess.run(command, check=True, env=env)
            finally:
                diagnostics = temporary.with_suffix(".diagnostics.json")
                if diagnostics.is_file():
                    diagnostics.replace(output.with_suffix(".diagnostics.json"))
            if not temporary.is_file():
                raise RuntimeError(f"Episode matching finished without creating {output}")
            temporary.replace(output)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset-dir", type=Path, required=True)
    parser.add_argument("--original-dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    ensure_episode_map(args.source_dataset_dir, args.original_dataset_dir, args.output, args.project_root)


if __name__ == "__main__":
    main()
