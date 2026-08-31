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
  3. skips only when that subset's request hash + downloaded MCAP manifest still match
     ``.download_complete.json``; otherwise runs/resumes the engine
  4. writes the completion marker only after the engine exits successfully

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
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from ABC_dataset_config import DEFAULT_CONFIG_PATH, abc_root, abcdl_repo, load_config, subsets  # noqa: E402


COMPLETION_MARKER = ".download_complete.json"
COMPLETION_SCHEMA_VERSION = 1


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


def _request_sha256(generated_config: dict) -> str:
    """Hash only fields that change which remote files belong to the subset."""
    ignored = {"out_dir", "max_workers", "hf_transfer", "convert_to_abcdl"}
    request = {
        key: value for key, value in generated_config.items() if key not in ignored
    }
    canonical = json.dumps(
        request,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _pending_download_files(dest: Path) -> list[Path]:
    return sorted(
        path
        for path in dest.rglob("*")
        if path.is_file() and path.suffix in {".incomplete", ".lock"}
    )


def _downloaded_episode_mcaps(dest: Path) -> list[dict[str, int | str]]:
    files: list[dict[str, int | str]] = []
    for path in sorted(dest.rglob("episode.mcap")):
        if path.is_file() and (size := path.stat().st_size) > 0:
            files.append({"path": path.relative_to(dest).as_posix(), "size": size})
    return files


def _completion_status(dest: Path, request_sha256: str) -> tuple[bool, str]:
    marker_path = dest / COMPLETION_MARKER
    if not marker_path.is_file():
        return False, "completion marker missing"
    try:
        marker = json.loads(marker_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return False, f"invalid completion marker: {error}"
    if marker.get("schema_version") != COMPLETION_SCHEMA_VERSION:
        return False, "completion marker schema changed"
    if marker.get("request_sha256") != request_sha256:
        return False, "subset request changed"
    if pending := _pending_download_files(dest):
        return False, f"{len(pending)} incomplete/lock file(s) remain"

    files = marker.get("episode_mcaps")
    if not isinstance(files, list) or not files:
        return False, "completion marker has no episode MCAP manifest"
    for item in files:
        if not isinstance(item, dict):
            return False, "invalid episode MCAP manifest entry"
        relative = Path(str(item.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            return False, f"unsafe episode MCAP path in marker: {relative}"
        path = dest / relative
        try:
            expected_size = int(item["size"])
        except (KeyError, TypeError, ValueError):
            return False, f"invalid size for {relative}"
        if not path.is_file():
            return False, f"missing {relative}"
        if expected_size <= 0 or path.stat().st_size != expected_size:
            return False, f"size mismatch for {relative}"
    return True, f"{len(files)} episode MCAP(s) verified"


def _write_completion_marker(dest: Path, subset_name: str, request_sha256: str) -> None:
    pending = _pending_download_files(dest)
    if pending:
        raise RuntimeError(
            f"[{subset_name}] download engine exited but {len(pending)} incomplete/lock file(s) remain"
        )
    episode_mcaps = _downloaded_episode_mcaps(dest)
    if not episode_mcaps:
        raise RuntimeError(
            f"[{subset_name}] download engine exited without any non-empty episode.mcap files in {dest}"
        )
    marker = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "subset": subset_name,
        "request_sha256": request_sha256,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "episode_mcaps": episode_mcaps,
    }
    marker_path = dest / COMPLETION_MARKER
    temporary = marker_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(marker_path)


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
    dl_workers = int(cfg.get("download_workers", 4))
    # 검증된 filtered_dataset 패턴(download_filtered_libero.sh)의 결론을 따른다: 순정 파이썬
    # 백엔드는 read timeout이 없어 죽은 소켓에 "영원히" 매달리고, hf_transfer(Rust)+read timeout이
    # 죽은 소켓을 끊고 자동 재시도한다 → hf_transfer가 hang을 '고치는' 쪽. 기본 True.
    # (프로세스가 timeout도 못 넘기고 wedge되는 병적 케이스는 download_ABC.sh의 워치독이 재시작으로 커버.)
    hf_transfer = bool(cfg.get("download_hf_transfer", True))

    if args.list_tasks:
        gen = {"repo_id": repo_id, "split": args.split or "train", "downloads": []}
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(yaml.safe_dump(gen, sort_keys=False, allow_unicode=True))
        try:
            _run_engine(tool, Path(f.name), repo,
                        ["--list-tasks"] + (["--counts"] if args.counts else []))
        finally:
            os.unlink(f.name)
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
            "hf_transfer": hf_transfer,
            "convert_to_abcdl": False,  # ②③④는 build_ABC_dataset.sh가 idempotent하게 수행
            "downloads": entries,
            "groups": groups,
        }
        request_sha256 = _request_sha256(gen)
        print(f"\n════ subset {name} → {dest} ════")
        if args.dry_run:
            # dry-run은 디스크에 흔적을 남기지 않는다 — 생성 config도 임시파일로.
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
                f.write(yaml.safe_dump(gen, sort_keys=False, allow_unicode=True))
            try:
                _run_engine(tool, Path(f.name), repo, ["--dry-run"])
            finally:
                os.unlink(f.name)
            continue
        dest.mkdir(parents=True, exist_ok=True)
        complete, reason = _completion_status(dest, request_sha256)
        if complete:
            print(f"[skip] {name}: {reason}")
            continue
        print(f"[resume] {name}: {reason}")
        gen_path = dest / ".download_config.yaml"  # provenance: 정확히 뭘 요청했는지 기록
        gen_path.write_text(yaml.safe_dump(gen, sort_keys=False, allow_unicode=True))
        _run_engine(tool, gen_path, repo, [])
        _write_completion_marker(dest, name, request_sha256)
        print(f"[complete] {name}: wrote {dest / COMPLETION_MARKER}")


if __name__ == "__main__":
    main()
