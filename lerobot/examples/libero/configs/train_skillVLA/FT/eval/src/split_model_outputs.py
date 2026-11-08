"""Separate rollout panels by actual model/checkpoint; keep resumable source runs hidden.

Dry-run by default. Individual panel directories are renamed, not copied. Hidden
source runs retain combined metrics/videos and panel symlinks for evaluator resume.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
from pathlib import Path
from urllib.parse import quote

from organize_outputs import atomic_text, page, run_index, summary


def model_destination(original: str, previous: str, label: str, checkpoint: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError(f"Unsafe model label: {label}")
    previous_path = Path(previous)
    training = next((p for p in previous_path.parts if p.startswith(("expert_", "bridge_", "skillcore_"))), None)
    condition = next((p for p in previous_path.parts if p.startswith(("pred_", "gt_"))), "pred_preskill")
    if "pi05vlm" in label:
        training = training or "expert_frozen_vision_trainable"
        if label.startswith("dual_bottleneck"):
            family = Path("pi05vlm/dual_bottleneck_9_9")
        elif label.startswith("vanila"):
            family = Path("pi05vlm/vanilla_9_9")
        elif label.startswith("denoising"):
            family = Path("pi05vlm/denoising_wovis") / ("no_uv" if "nouv" in label else "with_uv")
        else:
            raise ValueError(f"Unknown VLM model: {label}")
    else:
        if not training:
            raise ValueError(f"Missing training condition: {previous}")
        family = Path(previous_path.parts[0]) / label
    result = family / training / condition
    if "random_range2" in previous_path.parts:
        result /= "random_range2"
    return result / checkpoint


def _checkpoint(folder: Path, panel: Path, label: str, original: str) -> str:
    for path in (folder / "metrics/panel_cache" / panel.name).glob("*.json"):
        policy = json.loads(path.read_text()).get("signature", {}).get("policy_path", "")
        match = re.search(r"/checkpoints/(\d+)/", policy)
        if match:
            step = int(match[1])
            return f"{step // 1000}k" if step % 1000 == 0 else str(step)
    match = re.search(r"_(\d+k)$", label) or re.search(r"_(\d+k)_", original)
    if not match:
        raise ValueError(f"Cannot identify checkpoint for {original}: {label}")
    return match[1]


def plan(root: Path) -> tuple[dict, dict, list]:
    registry = json.loads((root / "output_locations.json").read_text())
    path = root / "model_results.json"
    saved = json.loads(path.read_text()) if path.exists() else {"version": 1, "results": []}
    known = {(r["original"], r["label"]) for r in saved["results"]}
    additions = []
    targets = {r["target"] for r in saved["results"]}
    for original, relative in registry["locations"].items():
        source = root / relative
        if not source.resolve().is_relative_to(root) or not source.is_dir():
            raise ValueError(f"Invalid source: {source}")
        metric = source / "metrics/eval_info_merged.json"
        if not metric.exists():
            continue
        for label in json.loads(metric.read_text()):
            if (original, label) in known:
                continue
            panels = [p for p in (source / "panels").iterdir() if p.name.split("_", 1)[-1] == label]
            if len(panels) != 1:
                raise ValueError(f"Expected exactly one panel for {label}: {panels}")
            panel = panels[0]
            target = model_destination(original, relative, label, _checkpoint(source, panel, label, original)).as_posix()
            if target in targets or (root / target).exists():
                raise FileExistsError(f"Model destination collision: {target}")
            targets.add(target)
            additions.append({"original": original, "label": label, "target": target,
                              "panel": panel.name, "source_relative": relative})
    return registry, saved, additions


def _rewrite(path: Path, replacements: list[tuple[str, str]]) -> None:
    before = path.read_text()
    after = before
    for old, new in replacements:
        after = after.replace(old + "/", new + "/").replace('"' + old + '"', '"' + new + '"')
    if after != before:
        if path.suffix == ".json":
            json.loads(after)
        atomic_text(path, after)


def _root_index(root: Path, records: list[dict]) -> None:
    body = "<h1>FT 평가 결과 — 모델별</h1><p>모델 → 학습 조건 → 평가 입력 → 체크포인트 순서입니다. "
    body += "각 폴더의 성공률·영상·trace는 해당 모델과 체크포인트만 포함합니다.</p>"
    if (root / "analysis/index.html").is_file():
        body += '<p><a href="analysis/index.html">통합 분석 — 전체·태스크별 체크포인트 성공률 변화</a></p>'
    groups: dict[str, list] = {}
    for record in sorted(records, key=lambda r: r["target"]):
        group = record["target"].split("/")[0]
        groups.setdefault(group, []).append(record)
    for group, entries in groups.items():
        body += f"<h2>{html.escape(group)}</h2><table><tr><th>모델 / 조건 / 체크포인트</th><th>성공률</th></tr>"
        for record in entries:
            relative = record["target"]
            body += f'<tr><td><a href="{quote(relative, safe="/")}/index.html">{html.escape(relative)}</a></td><td>{summary(root / relative)}</td></tr>'
        body += "</table>"
    body += '<p><a href="Prev/">Prev</a> · <a href="ablation/">기존 ablation</a> · <a href=".evaluation_runs/">원본 비교/재개 자료</a></p>'
    atomic_text(root / "index.html", page("Model-separated FT results", body))


def split_outputs(root: Path, *, apply: bool = False, charts: bool = True) -> dict:
    root = root.resolve(strict=True)
    registry, saved, additions = plan(root)
    for record in additions:
        print(f'{record["label"]} -> {record["target"]}', flush=True)
    if not apply:
        return {"new_models": len(additions), "existing_models": len(saved["results"])}

    # Persist the complete plan before mutations so interruption at any rename
    # can resume without deriving model identity from the hidden archive name.
    saved["results"].extend(additions)
    atomic_text(root / "model_results.json", json.dumps(saved, indent=2) + "\n")
    # Archive source runs first; registry persists each rename for resumability.
    old_locations = dict(registry["locations"])
    for original, relative in old_locations.items():
        archive = Path(".evaluation_runs") / original
        if relative == archive.as_posix():
            continue
        source, target = root / relative, root / archive
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        registry["locations"][original] = archive.as_posix()
        atomic_text(root / "output_locations.json", json.dumps(registry, indent=2) + "\n")

    for record in saved["results"]:
        archive = root / registry["locations"][record["original"]]
        source = archive / "panels" / record["panel"]
        target = root / record["target"] / "panels" / record["panel"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if source.is_symlink() and source.resolve() == target.resolve():
                continue
            if source.exists():
                raise FileExistsError(source)
        else:
            source.rename(target)
        # Only the hidden evaluator backend gets an alias; visible model data
        # is a real directory with the original media inode, not a linked view.
        source.symlink_to(os.path.relpath(target, source.parent), target_is_directory=True)

    records = saved["results"]
    for original, relative in registry["locations"].items():
        archive = root / relative
        entries = [r for r in records if r["original"] == original]
        replacements = []
        for record in entries:
            for old in {record["source_relative"], str(Path(".evaluation_runs") / original), original}:
                replacements.append((str(root / old / "panels" / record["panel"]),
                                     str(root / record["target"] / "panels" / record["panel"])))
        for old in {old_locations[original], *(r["source_relative"] for r in entries), original}:
            replacements.append((str(root / old), str(archive)))
        for path in (archive / "metrics").rglob("*.json"):
            _rewrite(path, replacements)
        for record in entries:
            folder = root / record["target"]
            for path in folder.glob("panels/*/skill_html/*/*.html"):
                _rewrite(path, replacements)
            metric_dir = folder / "metrics"
            metric_dir.mkdir(exist_ok=True)
            # Keep original numeric values; select the model, never re-evaluate.
            for path in (archive / "metrics").glob("eval_info*.json"):
                data = json.loads(path.read_text())
                if record["label"] in data:
                    atomic_text(metric_dir / path.name, json.dumps({record["label"]: data[record["label"]]}, indent=2) + "\n")
            atomic_text(folder / "source_run.json", json.dumps({
                "label": record["label"], "original_output_name": original,
                "source_run": relative, "source_panel": record["panel"],
            }, indent=2) + "\n")
            if charts:
                import sys
                sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "PT/eval/src"))
                from merge_eval_chunks import draw_chart
                data = json.loads((metric_dir / "eval_info_merged.json").read_text())
                draw_chart(data, [record["label"]], folder / "task_success_rates.png", expected_tasks=10)
            run_index(folder, record["label"], root)
        run_index(archive, original, root)

    _root_index(root, records)
    # Remove only now-empty old classification directories, never their data.
    for relative in sorted(old_locations.values(), key=lambda v: len(Path(v).parts), reverse=True):
        path = root / relative
        while path != root and path.name != ".evaluation_runs":
            if not path.exists():
                path = path.parent
                continue
            try:
                path.rmdir()
            except OSError:
                break
            path = path.parent
    if charts:
        from build_checkpoint_analysis import build_analysis
        build_analysis(root)
    return {"new_models": len(additions), "models": len(records), "source_runs": len(registry["locations"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(split_outputs(args.root, apply=args.apply), indent=2))
