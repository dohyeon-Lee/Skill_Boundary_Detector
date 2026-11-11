"""Create the source episode map needed by UV generation, once per dataset."""

from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import tempfile


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

        builder = Path(__file__).with_name("build_init_states.py")
        print(f"Build source episode map → {output}", flush=True)
        # Publish only a successful complete map; a failed writer must not
        # leave a partial NPZ that the next build would mistake for completion.
        with tempfile.TemporaryDirectory(prefix=f".{output.stem}.", dir=output.parent) as folder:
            temporary = Path(folder) / output.name
            command = [sys.executable, "-u", str(builder),
                       "--lerobot_dataset", str(source_dataset_dir), "--out", str(temporary),
                       "--orig_dataset", str(original_dataset_dir)]
            try:
                subprocess.run(command, check=True)
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
