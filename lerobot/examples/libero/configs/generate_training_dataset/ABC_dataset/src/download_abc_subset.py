#!/usr/bin/env python3
"""Download ABC-130k subsets by driving the abcdl_RLLAB selective downloader.

The actual download engine is ``{abcdl_repo}/download/src/download_abc.py`` — the
group/task/episode-selective tool (paper's 7 primitive categories → 197-task taxonomy
lives in ``{abcdl_repo}/download/config.yaml`` and is read from there as the single
source of truth; ``extra_groups:`` in ABC_dataset_config.yaml merges on top).

This wrapper, per subset in ``abc_subsets``:
  1. composes a downloader config = {repo_id, split, out_dir=_mcap/{name},
     groups (taxonomy + extras), downloads (the subset's entries), ...}
  2. writes it to ``{abc_root}/_mcap/{name}/.download_config.yaml`` (provenance)
  3. runs the engine with it (``--dry-run`` passes through)

Staging layout stays converter-compatible: ``_mcap/{name}/data/{split}/<task>/<ep>/…``
(group_subdirs off — the subset name itself is the grouping downstream cares about).

NOTE: XDOF/ABC-130k is a GATED dataset — accept the license on the hub page and
``huggingface-cli login`` (or set HF_TOKEN) before a real download. The engine prints
a friendly hint on auth failures.

Usage:
    python download_abc_subset.py [--only abc_toy] [--dry-run]
    python download_abc_subset.py --list-tasks [--counts] [--split val]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from ABC_dataset_config import DEFAULT_CONFIG_PATH, abc_root, abcdl_repo, load_config, subsets  # noqa: E402


def _engine_paths(cfg: dict) -> tuple[Path, Path]:
    """(downloader script, its taxonomy config) inside the abcdl repo — fail with cure."""
    repo = abcdl_repo(cfg)
    tool = repo / "download" / "src" / "download_abc.py"
    taxonomy = repo / "download" / "config.yaml"
    if not tool.exists():
        raise SystemExit(
            f"selective downloader not found: {tool}\n"
            f"  abcdl_repo({repo})가 구버전(다운로드 툴 없는 GitHub main)일 수 있음 — "
            "yonsei에서 최신 abcdl_RLLAB(download/ 포함)를 동기화하세요 (sync_server.sh)")
    return tool, taxonomy


def _load_groups(taxonomy: Path, cfg: dict) -> dict:
    """Official 7-category taxonomy from the abcdl downloader config + local extras."""
    groups: dict = {}
    if taxonomy.exists():
        groups.update((yaml.safe_load(taxonomy.read_text()) or {}).get("groups") or {})
    groups.update(cfg.get("extra_groups") or {})
    return groups


def _run_engine(tool: Path, config_path: Path, repo: Path, extra: list[str]) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo}:{env.get('PYTHONPATH', '')}"  # engine's optional abcdl import
    subprocess.run([sys.executable, str(tool), "--config", str(config_path), *extra],
                   check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--only", default=os.environ.get("ABC_ONLY", ""),
                        help="Space/comma-separated subset names (default: all in yaml)")
    parser.add_argument("--dry-run", action="store_true",
                        default=os.environ.get("DRY_RUN", "") not in ("", "0"),
                        help="Resolve + print the plan via the engine; download nothing")
    parser.add_argument("--list-tasks", action="store_true",
                        help="Print available task-folder names and exit")
    parser.add_argument("--counts", action="store_true",
                        help="With --list-tasks: also print per-task episode counts (slow)")
    parser.add_argument("--split", default=None, help="Split for --list-tasks (default train)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_id = str(cfg.get("abc_hf_repo", "XDOF/ABC-130k"))
    root = abc_root(cfg)
    repo = abcdl_repo(cfg)
    tool, taxonomy = _engine_paths(cfg)
    dl_workers = int(cfg.get("download_workers", 8))

    if args.list_tasks:
        gen = {"repo_id": repo_id, "split": args.split or "train", "downloads": []}
        tmp = root / "_mcap" / ".list_tasks_config.yaml"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(yaml.safe_dump(gen, sort_keys=False, allow_unicode=True))
        _run_engine(tool, tmp, repo, ["--list-tasks"] + (["--counts"] if args.counts else []))
        return

    specs = subsets(cfg)
    only = {s for s in args.only.replace(",", " ").split() if s}
    if only:
        unknown = only - set(specs)
        if unknown:
            raise SystemExit(f"unknown subset(s) {sorted(unknown)}; yaml has {sorted(specs)}")
        specs = {k: v for k, v in specs.items() if k in only}
    if not specs:
        raise SystemExit("no abc_subsets configured")

    groups = _load_groups(taxonomy, cfg)
    print(f"engine: {tool}\ntaxonomy: {len(groups)} groups "
          f"({', '.join(sorted(groups)) or '—'})")

    for name, spec in specs.items():
        entries = list(spec.get("downloads") or [])
        if not entries:
            raise SystemExit(f"[{name}] subset needs a non-empty `downloads:` list "
                             "(entries of group/task/tasks + episodes)")
        dest = root / "_mcap" / name
        gen = {
            "repo_id": repo_id,
            "split": str(spec.get("split", "train")),
            "out_dir": str(dest),
            "group_subdirs": False,   # converter-friendly repo mirror: data/{split}/{task}/{ep}
            "include_meta": False,
            "max_workers": dl_workers,
            "hf_transfer": True,
            "convert_to_abcdl": False,  # ②③④는 build_ABC_dataset.sh가 idempotent하게 수행
            "downloads": entries,
            "groups": groups,
        }
        dest.mkdir(parents=True, exist_ok=True)
        gen_path = dest / ".download_config.yaml"  # provenance: 정확히 뭘 요청했는지 기록
        gen_path.write_text(yaml.safe_dump(gen, sort_keys=False, allow_unicode=True))
        print(f"\n════ subset {name} → {dest} ════")
        _run_engine(tool, gen_path, repo, ["--dry-run"] if args.dry_run else [])


if __name__ == "__main__":
    main()
