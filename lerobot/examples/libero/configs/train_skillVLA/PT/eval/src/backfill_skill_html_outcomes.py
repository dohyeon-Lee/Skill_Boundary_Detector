#!/usr/bin/env python3
"""Add episode outcome badges to saved skill HTML without rerunning evaluation.

Dry-run by default. --apply updates only HTML files inside the selected output
directory and saves an original .before-outcome-badges.bak beside each file.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


STYLES = """<style id="episode-outcome-backfill-styles">
.episode-title { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
.episode-result { font-size:12px; font-weight:700; padding:4px 10px; border-radius:999px; }
.result-success { background:#dcfce7; color:#166534; }
.result-failure { background:#fee2e2; color:#991b1b; }
.result-unknown { background:#e5e7eb; color:#4b5563; }
.episode-success { border-left:5px solid #16a34a; }
.episode-failure { border-left:5px solid #dc2626; }
</style>
"""
SCRIPT = """<script id="episode-outcome-backfill-script">
document.querySelectorAll("#app .episode").forEach((section, index) => {
  const episode = DATA.episodes[index];
  if (!episode) return;
  const outcome = episode.success === true ? "success" : (episode.success === false ? "failure" : "unknown");
  section.classList.remove("episode-success", "episode-failure", "episode-unknown");
  section.classList.add(`episode-${outcome}`);
  const title = section.querySelector(".episode-title");
  if (!title) return;
  let badge = title.querySelector(".episode-result");
  if (!badge) { badge = document.createElement("span"); title.appendChild(badge); }
  badge.className = `episode-result result-${outcome}`;
  badge.textContent = {success:"SUCCESS", failure:"FAILURE", unknown:"UNKNOWN"}[outcome];
  badge.title = "Full-task evaluation result; not an individual skill's termination";
});
</script>
"""


def annotate_html(source: str, successes: list[bool], *, task_group: str, task_id: int) -> tuple[str, int]:
    if not successes or any(type(value) is not bool for value in successes):
        raise ValueError("Expected a nonempty list of boolean episode successes.")
    markers = list(re.finditer(r"\bconst\s+DATA\s*=\s*", source))
    if len(markers) != 1:
        raise ValueError("Expected exactly one embedded skill-trace DATA object.")
    start = markers[0].end()
    data, length = json.JSONDecoder().raw_decode(source[start:])
    if data.get("task_group") != task_group or data.get("task_id") != task_id:
        raise ValueError("HTML task identity differs from the evaluation metrics.")
    seen = set()
    for episode in data["episodes"]:
        index = episode["episode_index"]
        if type(index) is not int or not 0 <= index < len(successes) or index in seen:
            raise ValueError(f"Invalid/duplicate HTML episode index: {index!r}")
        seen.add(index)
        episode["success"] = successes[index]
    if not seen:
        raise ValueError("HTML contains no episodes.")
    updated = source[:start] + json.dumps(data) + source[start + length:]
    if 'id="episode-outcome-backfill-script"' not in updated:
        if updated.count("</head>") != 1 or updated.count("</body>") != 1:
            raise ValueError("Expected one HTML head and body.")
        updated = updated.replace("</head>", STYLES + "</head>", 1)
        updated = updated.replace("</body>", SCRIPT + "</body>", 1)
    return updated, len(seen)


def backfill(output_dir: Path, *, apply: bool = False) -> dict[str, int]:
    root = output_dir.resolve(strict=True)
    metrics = json.loads((root / "metrics/eval_info_merged.json").read_text())
    plans = {}
    # Validate every file before making the first change.
    for info in metrics.values():
        for task in info["per_task"]:
            values = task["metrics"]
            for filename in values.get("skill_html_paths", []):
                path = Path(filename)
                path = (root / path if not path.is_absolute() else path).resolve(strict=True)
                if not path.is_relative_to(root):
                    raise ValueError(f"Refusing to modify HTML outside the selected output: {path}")
                source = path.read_text()
                updated, count = annotate_html(
                    source, values["successes"], task_group=task["task_group"], task_id=task["task_id"]
                )
                plan = (updated, count)
                if path in plans and plans[path] != plan:
                    raise ValueError(f"Conflicting metrics for HTML: {path}")
                plans[path] = plan
    if not plans:
        raise ValueError("No saved skill HTML paths found in merged metrics.")
    changed = 0
    for path, (updated, _) in plans.items():
        if path.read_text() == updated:
            continue
        changed += 1
        if not apply:
            continue
        backup = path.with_name(path.name + ".before-outcome-badges.bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(updated)
            shutil.copymode(path, temporary)
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return {"html_files": len(plans), "episodes": sum(plan[1] for plan in plans.values()), "changed": changed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--apply", action="store_true", help="Apply changes with original HTML backups.")
    args = parser.parse_args()
    print(json.dumps({"applied": args.apply, **backfill(args.output_dir, apply=args.apply)}))


if __name__ == "__main__":
    main()
