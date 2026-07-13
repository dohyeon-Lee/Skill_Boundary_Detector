#!/usr/bin/env python3
"""Download an ABC-130k mcap SUBSET from the HF hub into the staging area.

Layout on the hub (XDOF/ABC-130k): ``data/{split}/<task>/<episode>/episode.mcap``
(+ optional ``annotation.mcap``). The full repo is >1 TB / 130k episodes, so we never
enumerate the whole tree — we list ONE level at a time with HfFileSystem (fast) and
descend only into the selected tasks/episodes.

Staging destination (repo-relative tree preserved for resume-friendliness):
    {abc_root}/_mcap/{subset_name}/data/{split}/<task>/<episode>/episode.mcap
plus a ``manifest.json`` (provenance: which tasks/episodes were selected).

Idempotent: files already present with size>0 are skipped without network calls.
DRY_RUN=1 (or --dry-run) lists the discovered structure + planned downloads only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from ABC_dataset_config import DEFAULT_CONFIG_PATH, abc_root, load_config, subsets  # noqa: E402


def _ls_dirs(fs, path: str) -> list[str]:
    """Names of directory entries directly under `path` (single-level, sorted)."""
    out = []
    for e in fs.ls(path, detail=True):
        if e.get("type") == "directory":
            out.append(e["name"].rstrip("/").rsplit("/", 1)[-1])
    return sorted(out)


def _ls_files(fs, path: str) -> list[str]:
    """Names of file entries directly under `path` (single-level, sorted)."""
    out = []
    for e in fs.ls(path, detail=True):
        if e.get("type") == "file":
            out.append(e["name"].rsplit("/", 1)[-1])
    return sorted(out)


def plan_subset(fs, repo: str, name: str, spec: dict) -> list[str]:
    """Resolve one subset spec to a list of repo-relative file paths to download."""
    split = str(spec.get("split", "train"))
    base = f"datasets/{repo}/data/{split}"
    want_tasks = [str(t) for t in (spec.get("tasks") or [])]
    max_tasks = int(spec.get("max_tasks") or 0)
    max_eps = int(spec.get("max_episodes_per_task") or 0)
    with_ann = bool(spec.get("include_annotations", False))

    tasks = _ls_dirs(fs, base)
    if not tasks:
        # Flat layout fallback: mcap files directly under data/{split}/
        files = [f for f in _ls_files(fs, base) if f.endswith(".mcap")]
        if files:
            print(f"  [{name}] flat layout ({len(files)} mcap files directly under data/{split}/)")
            picked = files[:max_eps] if max_eps else files
            return [f"data/{split}/{f}" for f in picked]
        raise RuntimeError(f"data/{split}/ under {repo} has neither task dirs nor mcap files")

    if want_tasks:
        missing = [t for t in want_tasks if t not in tasks]
        if missing:
            raise RuntimeError(
                f"[{name}] tasks not found on hub: {missing}\n"
                f"  available (first 30): {tasks[:30]}")
        chosen = want_tasks
    else:
        chosen = tasks[:max_tasks] if max_tasks else tasks
    print(f"  [{name}] split={split}  tasks {len(chosen)}/{len(tasks)}: {chosen}")

    plan: list[str] = []
    for task in chosen:
        eps = _ls_dirs(fs, f"{base}/{task}")
        picked = eps[:max_eps] if max_eps else eps
        print(f"    {task}: episodes {len(picked)}/{len(eps)}")
        if not picked and not eps:
            # Episodes may be flat files under the task dir
            files = [f for f in _ls_files(fs, f"{base}/{task}") if f.endswith(".mcap")]
            picked_files = files[:max_eps] if max_eps else files
            plan.extend(f"data/{split}/{task}/{f}" for f in picked_files)
            print(f"    {task}: flat mcap files {len(picked_files)}/{len(files)}")
            continue
        for ep in picked:
            files = _ls_files(fs, f"{base}/{task}/{ep}")
            for f in files:
                if f == "episode.mcap" or (with_ann and f == "annotation.mcap"):
                    plan.append(f"data/{split}/{task}/{ep}/{f}")
    return plan


def download(repo: str, rel_files: list[str], dest_root: Path) -> None:
    from huggingface_hub import hf_hub_download

    todo = [f for f in rel_files
            if not ((dest_root / f).is_file() and (dest_root / f).stat().st_size > 0)]
    print(f"  files: {len(rel_files)} planned, {len(rel_files) - len(todo)} already present, "
          f"{len(todo)} to download")
    for i, rel in enumerate(todo, 1):
        hf_hub_download(repo_id=repo, repo_type="dataset", filename=rel,
                        local_dir=str(dest_root))
        print(f"    [{i}/{len(todo)}] {rel}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--only", default=os.environ.get("ABC_ONLY", ""),
                        help="Space/comma-separated subset names (default: all in yaml)")
    parser.add_argument("--dry-run", action="store_true",
                        default=os.environ.get("DRY_RUN", "") not in ("", "0"),
                        help="List discovered structure + plan; download nothing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo = str(cfg.get("abc_hf_repo", "XDOF/ABC-130k"))
    root = abc_root(cfg)
    specs = subsets(cfg)
    only = {s for chunk in args.only.replace(",", " ").split() for s in [chunk] if s}
    if only:
        unknown = only - set(specs)
        if unknown:
            raise SystemExit(f"unknown subset(s) {sorted(unknown)}; yaml has {sorted(specs)}")
        specs = {k: v for k, v in specs.items() if k in only}
    if not specs:
        raise SystemExit("no abc_subsets configured")

    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()

    print(f"repo: {repo}  →  staging: {root}/_mcap/<name>/")
    for name, spec in specs.items():
        plan = plan_subset(fs, repo, name, spec)
        dest = root / "_mcap" / name
        if args.dry_run:
            print(f"  [{name}] DRY RUN — {len(plan)} files would go to {dest}")
            for rel in plan[:8]:
                print(f"    {rel}")
            if len(plan) > 8:
                print(f"    … +{len(plan) - 8} more")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        download(repo, plan, dest)
        manifest = {
            "hf_repo": repo,
            "subset": name,
            "spec": spec,
            "files": plan,
        }
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"  [{name}] done → {dest}  (manifest.json: {len(plan)} files)")


if __name__ == "__main__":
    main()
