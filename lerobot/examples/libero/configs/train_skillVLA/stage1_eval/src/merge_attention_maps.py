#!/usr/bin/env python3
"""Merge task-owned Arch3 attention reports after each Slurm array job.

The caller holds the same flock as the ordinary Stage-1 metrics merge. Task
pages are completion markers, so a still-running worker is never indexed.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font:15px system-ui;background:#111827;color:#eee;"
        "max-width:900px;margin:32px auto}a{color:#93c5fd}li{margin:8px 0}</style>"
        "</head><body>"
        f"<h1>{html.escape(title)}</h1>{body}</body></html>"
    )


def merge(out_dir: Path, *, expected_tasks: int = 0) -> dict[str, int]:
    """Build panel and root indexes from completed, disjoint task reports."""
    panel_counts: dict[str, int] = {}
    panel_links: list[str] = []
    for panel_dir in sorted((out_dir / "panels").glob("*")):
        attention_dir = panel_dir / "attention_maps"
        if not attention_dir.is_dir():
            continue
        task_pages = sorted(attention_dir.glob("*/index.html"))
        if not task_pages:
            continue
        task_links: list[str] = []
        all_records: list[dict] = []
        for task_page in task_pages:
            task_dir = task_page.parent
            manifest = task_dir / "attention_maps.json"
            if not manifest.is_file():
                continue
            records = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                raise ValueError(f"Invalid attention manifest: {manifest}")
            all_records.extend(records)
            task_links.append(
                f"<li><a href='{html.escape(task_dir.name)}/index.html'>"
                f"{html.escape(task_dir.name)}</a> · {len(records)} maps</li>"
            )
        count = len(task_links)
        if not count:
            continue
        panel_counts[panel_dir.name] = count
        progress = f"{count}/{expected_tasks} tasks" if expected_tasks else f"{count} tasks"
        _atomic_text(
            attention_dir / "index.html",
            _page(f"{panel_dir.name} attention maps", f"<p>{progress}</p><ul>{''.join(task_links)}</ul>"),
        )
        _atomic_text(
            attention_dir / "attention_maps.json",
            json.dumps(all_records, indent=2),
        )
        panel_links.append(
            f"<li><a href='panels/{html.escape(panel_dir.name)}/attention_maps/index.html'>"
            f"{html.escape(panel_dir.name)}</a> · {progress}</li>"
        )
    if panel_links:
        _atomic_text(
            out_dir / "attention_maps.html",
            _page("Stage-1 attention maps", f"<ul>{''.join(panel_links)}</ul>"),
        )
    return panel_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--expected_tasks", type=int, default=0)
    args = parser.parse_args()
    print("attention maps:", merge(args.out_dir, expected_tasks=args.expected_tasks))


if __name__ == "__main__":
    main()
