#!/usr/bin/env python3
"""Selective downloader for the ABC-130k dataset (``XDOF/ABC-130k``).

The upstream release is a *gated* HuggingFace dataset laid out flat by task::

    data/{train,val}/<task_name>/episode_<uuid>/episode.mcap [+ annotation.mcap]

There is **no** category folder on the Hub, so the 7 paper "primitive
categories" are expressed here as **user-defined groups** in the YAML config
(a ``groups:`` map of ``group_name -> [task, ...]``). Each download entry then
asks for a group / task / task-list plus how many episodes to pull, and this
script lists the matching episode dirs over ``HfFileSystem`` and pulls only
those via ``snapshot_download(allow_patterns=...)`` — never the whole 3,553 h.

Optionally (``convert_to_abcdl: true``) each downloaded ``episode.mcap`` is run
through :func:`abcdl.convert.mcap_abcdl.mcap_to_abcdl` into the training format.

Usage
-----
    huggingface-cli login                      # gated: accept the license first
    python download_abc.py --config config.yaml
    python download_abc.py --config config.yaml --dry-run      # plan only
    python download_abc.py --config config.yaml --list-tasks   # discover task names
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

try:
    import yaml
except ImportError as e:  # pragma: no cover
    sys.exit("PyYAML is required: pip install pyyaml  (%s)" % e)


REPO_DEFAULT = "XDOF/ABC-130k"


# ---------------------------------------------------------------------------
# HuggingFace helpers
# ---------------------------------------------------------------------------

def _fs(token: Optional[str] = None):
    try:
        from huggingface_hub import HfFileSystem
    except ImportError as e:  # pragma: no cover
        sys.exit("huggingface_hub is required: pip install huggingface_hub  (%s)" % e)
    return HfFileSystem(token=token)


def _auth_hint(err: Exception) -> None:
    """Print a friendly hint when the failure looks like gated/auth."""
    msg = str(err).lower()
    if any(s in msg for s in ("gated", "401", "403", "unauthorized", "awaiting", "access", "authenticated")):
        print(
            "\n[auth] XDOF/ABC-130k is a *gated* dataset. Before downloading:\n"
            "  1) open https://huggingface.co/datasets/XDOF/ABC-130k and accept the conditions\n"
            "  2) run `huggingface-cli login` (or set HF_TOKEN / `token:` in the config)\n",
            file=sys.stderr,
        )


def list_episode_dirs(fs, repo_id: str, split: str, task: str) -> Optional[list]:
    """Full ``episode_<uuid>`` dir paths for one task, sorted; None if task missing."""
    base = f"datasets/{repo_id}/data/{split}/{task}"
    try:
        entries = fs.ls(base, detail=False)
    except FileNotFoundError:
        return None
    eps = sorted(
        e for e in entries
        if os.path.basename(e.rstrip("/")).startswith("episode")
    )
    return eps


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def resolve_tasks(entry: dict, groups: dict) -> list:
    """A download entry names its tasks via `group`, `task`, or `tasks`."""
    if "group" in entry:
        g = entry["group"]
        if g not in groups:
            raise ValueError(
                f"download entry references unknown group {g!r}; "
                f"define it under `groups:` (known: {sorted(groups)})"
            )
        tasks = list(groups[g] or [])
        if not tasks:
            print(f"[warn] group {g!r} is empty — nothing to select", file=sys.stderr)
        return tasks
    if "task" in entry:
        return [entry["task"]]
    if "tasks" in entry:
        return list(entry["tasks"])
    raise ValueError(f"download entry needs one of group/task/tasks: {entry!r}")


def select_episodes(ep_dirs: list, spec) -> list:
    """Pick episodes from the sorted list per `spec`.

    spec = "all" / None  -> every episode
         = int N         -> first N (deterministic, sorted by uuid)
         = list of int   -> those indices
         = list of str   -> episodes whose dir name contains that substring (uuid)
    """
    if spec in (None, "all"):
        return ep_dirs
    if isinstance(spec, bool):  # guard: YAML `true` is not a count
        raise ValueError(f"episodes must be int/'all'/list, got bool {spec!r}")
    if isinstance(spec, int):
        if spec < 0:
            raise ValueError(f"episodes count must be >= 0, got {spec}")
        return ep_dirs[:spec]
    if isinstance(spec, (list, tuple)):
        out: list = []
        for s in spec:
            if isinstance(s, bool):
                raise ValueError(f"bad episode selector {s!r}")
            if isinstance(s, int):
                if 0 <= s < len(ep_dirs):
                    out.append(ep_dirs[s])
                else:
                    print(f"[warn] episode index {s} out of range (0..{len(ep_dirs)-1})", file=sys.stderr)
            else:  # substring / uuid match
                hit = next((e for e in ep_dirs if str(s) in os.path.basename(e)), None)
                if hit:
                    out.append(hit)
                else:
                    print(f"[warn] no episode matching {s!r}", file=sys.stderr)
        # de-dup, keep order
        seen: set = set()
        return [e for e in out if not (e in seen or seen.add(e))]
    raise ValueError(f"bad episodes spec: {spec!r}")


# ---------------------------------------------------------------------------
# Convert step (optional)
# ---------------------------------------------------------------------------

def _find_episode_mcap(ep_dir: str) -> Optional[str]:
    cand = os.path.join(ep_dir, "episode.mcap")
    if os.path.exists(cand):
        return cand
    mcaps = [
        p for p in glob.glob(os.path.join(ep_dir, "*.mcap"))
        if "annotation" not in os.path.basename(p)
    ]
    return mcaps[0] if mcaps else None


def convert_to_abcdl(cfg: dict, selected: list) -> None:
    try:
        from abcdl.convert.mcap_abcdl import mcap_to_abcdl
    except Exception as e:  # noqa: BLE001 - abcdl/torchcodec/ffmpeg may be missing
        print(f"[convert] skipped — cannot import abcdl.convert.mcap_abcdl: {e}", file=sys.stderr)
        return
    size = int(cfg.get("size", 224))
    abcdl_out = cfg.get("abcdl_out_dir", "./abc_abcdl")
    n_ok = n_fail = 0
    for root, group, split, task, names in selected:
        for nm in names:
            ep_dir = os.path.join(root, "data", split, task, nm)
            src = _find_episode_mcap(ep_dir)
            if src is None:
                print(f"[convert] no episode.mcap under {ep_dir}", file=sys.stderr)
                n_fail += 1
                continue
            dst = (os.path.join(abcdl_out, group, split, task, nm) if group
                   else os.path.join(abcdl_out, split, task, nm))
            try:
                mcap_to_abcdl(src, dst, size=size)
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                print(f"[convert] FAILED {src}: {e}", file=sys.stderr)
                n_fail += 1
    print(f"[convert] ok={n_ok} fail={n_fail}  size={size}px  -> {abcdl_out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def cmd_list_tasks(fs, repo_id: str, split: str, with_counts: bool) -> None:
    base = f"datasets/{repo_id}/data/{split}"
    try:
        tasks = sorted(os.path.basename(e.rstrip("/")) for e in fs.ls(base, detail=False))
    except Exception as e:  # noqa: BLE001
        _auth_hint(e)
        raise
    print(f"# {len(tasks)} tasks under data/{split}/ in {repo_id}")
    for t in tasks:
        if with_counts:
            eps = list_episode_dirs(fs, repo_id, split, t) or []
            print(f"{t}\t{len(eps)}")
        else:
            print(t)


def _enable_hf_transfer(want: bool) -> None:
    """Turn on the accelerated ``hf_transfer`` backend when available.

    hf_transfer splits a single large file (the ~70 MB ``episode.mcap`` blobs) into
    parallel range requests for much higher throughput. huggingface_hub reads this
    toggle from the environment at import time, so it must be set *before* the first
    hf import (which happens lazily in :func:`_fs`).
    """
    if want and importlib.util.find_spec("hf_transfer") is not None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    else:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        if want:
            print("[hint] `uv pip install hf_transfer` for faster large-file downloads",
                  file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Selective ABC-130k downloader")
    ap.add_argument("--config", required=True, help="path to the YAML config")
    ap.add_argument("--dry-run", action="store_true", help="resolve + print plan, do not download")
    ap.add_argument("--list-tasks", action="store_true", help="print available task names and exit")
    ap.add_argument("--counts", action="store_true", help="with --list-tasks, also print episode counts (slow)")
    ap.add_argument("--split", default=None, help="override split for --list-tasks")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    repo_id = cfg.get("repo_id", REPO_DEFAULT)
    default_split = cfg.get("split", "train")
    out_dir = cfg.get("out_dir", "./abc_subset")
    token = cfg.get("token") or os.environ.get("HF_TOKEN")
    groups = cfg.get("groups") or {}
    downloads = cfg.get("downloads") or []
    include_meta = bool(cfg.get("include_meta", False))
    max_workers = max(1, int(cfg.get("max_workers", 8)))   # concurrent downloads + listings
    group_subdirs = bool(cfg.get("group_subdirs", False))  # nest output under out_dir/<group>/

    # Accelerated large-file transfer (must be set before the first hf import in _fs).
    _enable_hf_transfer(bool(cfg.get("hf_transfer", True)))

    fs = _fs(token)

    if args.list_tasks:
        cmd_list_tasks(fs, repo_id, args.split or default_split, args.counts)
        return

    # Expand download entries into (group, split, task, episode-spec) work items.
    work: list = []
    for entry in downloads:
        split = entry.get("split", default_split)
        ep_spec = entry.get("episodes", "all")
        group = entry.get("group") or entry.get("label")  # None for bare task/tasks entries
        for task in resolve_tasks(entry, groups):
            work.append((group, split, task, ep_spec))

    # List episodes for every task in parallel — metadata is the slow part when a
    # group has many tasks. Each item resolves to its chosen episode dir names.
    def _resolve(item):
        group, split, task, ep_spec = item
        try:
            ep_dirs = list_episode_dirs(fs, repo_id, split, task)
        except Exception as e:  # noqa: BLE001
            return ("error", group, split, task, e)
        if ep_dirs is None:
            return ("missing", group, split, task, None)
        names = [os.path.basename(e.rstrip("/")) for e in select_episodes(ep_dirs, ep_spec)]
        return ("ok", group, split, task, names)

    results: list = []
    if work:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(work))) as ex:
            results = list(ex.map(_resolve, work))

    # Bucket by download root. With group_subdirs, each group's files go to a
    # separate local_dir out_dir/<group>/ (HF incremental caching still works per
    # root); otherwise everything mirrors the repo under out_dir.
    selected: list = []          # (root, group_label, split, task, names)
    root_patterns: dict = {}     # root -> [allow_pattern, ...]
    for status, group, split, task, payload in results:
        if status == "error":
            _auth_hint(payload)
            raise payload
        if status == "missing":
            print(f"[warn] task not found: data/{split}/{task}", file=sys.stderr)
            continue
        if not payload:
            print(f"[warn] 0 episodes selected for data/{split}/{task}", file=sys.stderr)
            continue
        label = (group or "misc") if group_subdirs else None
        root = os.path.join(out_dir, label) if label else out_dir
        selected.append((root, label, split, task, payload))
        root_patterns.setdefault(root, []).extend(
            f"data/{split}/{task}/{nm}/*" for nm in payload)

    if include_meta:
        root_patterns.setdefault(out_dir, []).extend(["meta/*", "README.md"])

    total_eps = sum(len(n) for _, _, _, _, n in selected)
    ht = "on" if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1" else "off"
    print(f"repo={repo_id}  tasks={len(selected)}  episodes={total_eps}  -> {out_dir}"
          f"  (max_workers={max_workers}, hf_transfer={ht}, group_subdirs={group_subdirs})")
    for root, label, split, task, names in selected:
        print(f"  {os.path.join(root, 'data', split, task)}: {len(names)} ep")

    if args.dry_run:
        print("[dry-run] downloads by root:")
        for root, pats in root_patterns.items():
            print(f"   {root}/")
            for p in pats:
                print(f"      {p}")
        return

    if not root_patterns:
        print("nothing to download (empty selection)")
        return

    from huggingface_hub import snapshot_download
    for root, pats in root_patterns.items():
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns=pats,
                local_dir=root,
                token=token,
                max_workers=max_workers,
            )
        except Exception as e:  # noqa: BLE001
            _auth_hint(e)
            raise
        print(f"[download] done -> {root}")

    if cfg.get("convert_to_abcdl"):
        convert_to_abcdl(cfg, selected)


if __name__ == "__main__":
    main()
