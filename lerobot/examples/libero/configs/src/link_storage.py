#!/usr/bin/env python3
"""Link a server's storage volumes into the checkout (``storage_*`` keys of servers/<name>.yaml).

Every module expects ``<project_root>/{models, <dataset_root>, <outputs_root>}``. On a machine whose
data lives elsewhere (RunPod: /workspace-global) this script makes that tree out of symlinks, so no
code or config path changes. The volume keeps the folder names used on every other server:

  storage_volume/models/<model>                 -> <repo>/models/<model>   (per file where git
                                                                            already has the folder)
  storage_volume/dataset*/                      -> <repo>/dataset*         (dataset_filtered, ...)
  storage_volume/outputs*/<group>/<run>         -> storage_outputs/outputs*/<group>/<run>
                                                   (start checkpoints, read by warm starts)
  storage_outputs/outputs*/                     -> <repo>/outputs*         (new runs are written here;
                                                                            <outputs_root> is created)

Other folders on the volume are left alone. Runs from setup_env.sh and after ``hf_sync.sh pull``;
safe to re-run after adding data. Existing real files and directories are never replaced (reported
as conflicts). A server without storage_* keys needs nothing.
"""

from __future__ import annotations

import argparse
import filecmp
import os
import runpy
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _is_named(path: Path, prefix: str) -> bool:
    return path.is_dir() and (path.name == prefix or path.name.startswith(prefix + "_"))


class Linker:
    def __init__(self, dry_run: bool):
        self.dry_run = dry_run
        self.counts = {"linked": 0, "ok": 0, "conflict": 0}

    def link(self, destination: Path, target: Path) -> None:
        if destination.is_symlink():
            if Path(os.readlink(destination)) == target:
                self.counts["ok"] += 1
                return
            print(f"  conflict: {destination} -> {os.readlink(destination)} (wanted {target})")
            self.counts["conflict"] += 1
            return
        if destination.exists():
            # A git-tracked small file (e.g. a model's config.json) identical to the volume copy is fine.
            same_file = destination.is_file() and target.is_file() and filecmp.cmp(destination, target, shallow=False)
            if destination.resolve() == target.resolve() or same_file:
                self.counts["ok"] += 1
            else:
                print(f"  conflict: {destination} already exists (not linked to {target})")
                self.counts["conflict"] += 1
            return
        print(f"  link: {destination} -> {target}")
        if not self.dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(target)
        self.counts["linked"] += 1

    def merge_tree(self, destination: Path, source: Path) -> None:
        """Link ``source`` as a whole, or file by file where ``destination`` is a real directory."""
        if not destination.exists() or destination.is_symlink():
            self.link(destination, source)
            return
        for child in sorted(source.iterdir()):
            if child.is_dir() and not child.is_symlink():
                self.merge_tree(destination / child.name, child)
            else:
                self.link(destination / child.name, child)


def _directory(config: dict, key: str) -> Path | None:
    value = str(config.get(key, "") or "").strip()
    return Path(value).expanduser() if value else None


def link_storage(config: dict, *, dry_run: bool = False) -> dict[str, int]:
    repo = Path(str(config["project_root"])).expanduser()
    linker = Linker(dry_run)
    volume = _directory(config, "storage_volume")
    outputs = _directory(config, "storage_outputs")
    if volume is None and outputs is None:
        print(f"server {config.get('server', '?')}: no storage_* keys; nothing to link.")
        return linker.counts
    if volume is not None and not volume.is_dir():
        raise FileNotFoundError(f"storage_volume={volume} does not exist (mount the volume or fix servers/*.yaml).")

    if outputs is not None:
        outputs_root = str(config.get("outputs_root", "outputs") or "outputs")
        if not dry_run:
            (outputs / outputs_root).mkdir(parents=True, exist_ok=True)
        # Only outputs*/ folders: storage_outputs may be a shared parent such as /workspace.
        names = {outputs_root} | ({path.name for path in outputs.iterdir() if _is_named(path, "outputs")} if outputs.is_dir() else set())
        for name in sorted(names):
            linker.link(repo / name, outputs / name)
    if volume is not None:
        for child in sorted(volume.iterdir()):
            if child.name == "models" and child.is_dir():
                for model in sorted(child.iterdir()):
                    linker.merge_tree(repo / "models" / model.name, model)
            elif _is_named(child, "dataset"):
                linker.link(repo / child.name, child)
            elif _is_named(child, "outputs"):
                if outputs is None:
                    raise ValueError("Start checkpoints on the volume need storage_outputs (runs are linked into it).")
                for run in sorted(child.glob("*/*")):
                    if run.is_dir():
                        linker.link(outputs / child.name / run.relative_to(child), run)
        dataset_root = str(config.get("dataset_root", "") or "")
        if dataset_root and not (repo / dataset_root).exists():
            print(f"  note: dataset_root={dataset_root} is not in {volume} yet (bash hf_sync.sh pull ...)")
    counts = linker.counts
    print(f"server {config.get('server', '?')}: {counts['linked']} linked, {counts['ok']} already ok, "
          f"{counts['conflict']} conflict(s){' (dry run)' if dry_run else ''}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=_HERE.parent / "global_config.yaml")
    parser.add_argument("--server", default=None, help="override server detection (like SBD_SERVER)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load = runpy.run_path(str(_HERE / "global_config_loader.py"))["load_global_config"]
    counts = link_storage(load(args.config, args.server), dry_run=args.dry_run)
    raise SystemExit(1 if counts["conflict"] else 0)


if __name__ == "__main__":
    main()
