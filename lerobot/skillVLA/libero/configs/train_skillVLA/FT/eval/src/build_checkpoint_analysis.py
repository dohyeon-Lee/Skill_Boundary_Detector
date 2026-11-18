"""Add checkpoint trend reports without moving or changing individual eval results."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

from organize_outputs import atomic_text, page, run_index

# Model identity and artifact presentation may differ in a comparison; actual
# predictor/terminator, RNG, grounding, boundary and oracle contracts may not.
NON_PROTOCOL_KEYS = {
    "policy_path", "architecture", "architecture_label", "architecture_revision",
    "conditioning_route", "tasks", "presentation_videos", "presentation_video_size",
    "presentation_tint_alpha", "video_fullskill_overlay",
}


def protocol_signature(signature: dict) -> dict:
    return {key: value for key, value in signature.items() if key not in NON_PROTOCOL_KEYS}


def _link(target: Path, folder: Path) -> str:
    return quote(os.path.relpath(target, folder), safe="/")


def collect(root: Path) -> dict:
    records = json.loads((root / "model_results.json").read_text())["results"]
    locations = json.loads((root / "output_locations.json").read_text())["locations"]
    groups = defaultdict(list)
    for record in records:
        folder = root / record["target"]
        metric = folder / "metrics/eval_info_merged.json"
        if not metric.is_file():
            continue
        data = json.loads(metric.read_text())[record["label"]]
        caches = sorted((root / locations[record["original"]] / "metrics/panel_cache" / record["panel"]).glob("*.json"))
        if caches:
            protocol = protocol_signature(json.loads(caches[0].read_text())["signature"])
        else:
            # Unknown protocols must never silently merge across source jobs.
            protocol = {"unknown_source_run": record["original"]}
        digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
        target = Path(record["target"])
        training = next(part for part in target.parts if part.startswith(("expert_", "bridge_", "skillcore_")))
        condition = next(part for part in target.parts if part.startswith(("pred_", "gt_")))
        if "random_range2" in target.parts:
            condition += "_range2"
        step = int(target.name[:-1]) * 1000 if target.name.endswith("k") else int(target.name)
        series = re.sub(r"_(?:\d+k|\d{6})$", "", record["label"])
        groups[(target.parts[0], training, condition, digest)].append({
            "record": record, "data": data, "step": step, "series": series,
            "protocol": protocol, "parent": target.parent.as_posix(),
        })
    return dict(groups)


def _table(points: list, folder: Path, root: Path) -> str:
    body = "<table><tr><th>모델</th><th>FT step</th><th>전체 성공률</th><th>완료 task</th></tr>"
    for point in sorted(points, key=lambda p: (p["series"], p["step"])):
        data = point["data"]
        overall = data.get("overall", {})
        path = root / point["record"]["target"] / "index.html"
        rate = overall.get("pc_success")
        text = f"{rate:g}%" if rate is not None else "—"
        body += f'<tr><td><a href="{_link(path, folder)}">{html.escape(point["series"])}</a></td><td>{point["step"] / 1000:g}k</td><td>{text}</td><td>{len(data.get("per_task", []))}</td></tr>'
    return body + "</table>"


def build_analysis(root: Path, *, charts: bool = True) -> dict:
    root = root.resolve(strict=True)
    groups = collect(root)
    analysis_root = root / "analysis"
    analysis_root.mkdir(exist_ok=True)
    if charts:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "PT/eval/src"))
        from merge_eval_chunks import draw_checkpoint_chart, draw_checkpoint_task_charts
    overview = '<a href="../index.html">개별 결과 모아보기</a><h1>FT 체크포인트 통합 분석</h1>'
    overview += "<p>같은 학습 설정·평가 protocol 안에서 비교합니다. 다른 predictor/terminator, oracle, seed, 초기 상태 범위는 섞지 않습니다. "
    overview += "실측 지점만 사용하며 누락된 task는 0%로 채우지 않습니다. 체크포인트가 하나뿐이면 추세선은 만들지 않습니다.</p>"
    model_reports = defaultdict(list)
    group_records = []
    chart_count = 0
    for key, points in sorted(groups.items()):
        family, training, condition, digest = key
        folder = analysis_root / family / training / condition / digest
        folder.mkdir(parents=True, exist_ok=True)
        merged = {}; by_model = defaultdict(list)
        for point in sorted(points, key=lambda p: (p["series"], p["step"])):
            label = f'{point["series"]}_{point["step"] // 1000}k' if point["step"] % 1000 == 0 else f'{point["series"]}_ckpt_{point["step"]:06d}'
            if label in merged:
                raise ValueError(f"Duplicate checkpoint in protocol group: {label}")
            merged[label] = point["data"]
            by_model[point["parent"]].append((label, point))
        atomic_text(folder / "metrics.json", json.dumps({
            "protocol": points[0]["protocol"], "results": merged,
        }, indent=2) + "\n")
        body = f'<a href="{_link(analysis_root / "index.html", folder)}">모든 통합 분석</a>'
        body += f"<h1>{html.escape(family)} / {html.escape(training)} / {html.escape(condition)}</h1>"
        body += '<p><a href="metrics.json">평가 protocol 및 집계 원본</a></p>'
        body += "<h2>전체 성공률 변화 — 모델 간 비교</h2>"
        group_chart = folder / "checkpoint_success_rates.png"
        wrote = charts and draw_checkpoint_chart(merged, list(merged), group_chart, expected_tasks=10)
        if wrote:
            chart_count += 1
            body += '<img style="max-width:100%" src="checkpoint_success_rates.png">'
        else:
            body += "<p>체크포인트가 하나뿐인 설정: 아래 실측 수치를 확인하세요.</p>"
        body += _table(points, folder, root)
        for parent, pairs in sorted(by_model.items()):
            single = {label: p["data"] for label, p in pairs}
            model_folder = root / parent / "analysis" / digest
            model_folder.mkdir(parents=True, exist_ok=True)
            title = pairs[0][1]["series"]
            model_body = f'<a href="{_link(folder / "index.html", model_folder)}">같은 조건의 모델 간 비교</a><h1>{html.escape(title)}</h1>'
            model_body += "<h2>전체 성공률 변화</h2>"
            solo_chart = model_folder / "checkpoint_success_rates.png"
            has_overall = charts and draw_checkpoint_chart(single, list(single), solo_chart, expected_tasks=10)
            task_charts = draw_checkpoint_task_charts(single, list(single), model_folder, expected_tasks=10) if charts else []
            if has_overall:
                chart_count += 1
                model_body += '<img style="max-width:100%" src="checkpoint_success_rates.png">'
            else:
                model_body += "<p>현재 체크포인트 1개 — 추세선 없음.</p>"
            model_body += "<h2>태스크별 성공률 변화</h2>"
            for chart in task_charts:
                chart_count += 1
                model_body += f'<img style="max-width:100%" src="{_link(chart, model_folder)}">'
            model_body += _table([p for _, p in pairs], model_folder, root)
            atomic_text(model_folder / "metrics.json", json.dumps({
                "protocol": points[0]["protocol"], "results": single,
            }, indent=2) + "\n")
            atomic_text(model_folder / "index.html", page(title + " checkpoint analysis", model_body))
            model_reports[parent].append(model_folder / "index.html")
            body += f'<h2>{html.escape(title)} — 태스크별 변화</h2><p><a href="{_link(model_folder / "index.html", folder)}">모델별 전체·태스크 추세 보기</a></p>'
            for chart in task_charts:
                body += f'<img style="max-width:100%" src="{_link(chart, folder)}">'
        atomic_text(folder / "index.html", page("FT checkpoint trends", body))
        relative = (folder / "index.html").relative_to(root).as_posix()
        group_records.append({"family": family, "training": training, "condition": condition,
                              "protocol_id": digest, "report": relative, "checkpoints": len(points)})
        title = f"{family} / {training} / {condition} · {len(points)} results"
        overview += f'<h2><a href="{_link(folder / "index.html", analysis_root)}">{html.escape(title)}</a></h2>'
        if wrote:
            overview += f'<a href="{_link(folder / "index.html", analysis_root)}"><img style="max-width:100%" src="{_link(group_chart, analysis_root)}"></a>'

    for parent, reports in model_reports.items():
        folder = root / parent / "analysis"
        body = '<h1>체크포인트 통합 분석</h1><ul>'
        for i, report in enumerate(reports):
            body += f'<li><a href="{_link(report, folder)}">평가 조건 {i + 1} — 전체·태스크별 성공률 변화</a></li>'
        body += "</ul>"
        atomic_text(folder / "index.html", page("Checkpoint analysis", body))
    atomic_text(analysis_root / "index.html", page("FT integrated checkpoint analysis", overview))
    atomic_text(analysis_root / "analysis_groups.json", json.dumps(group_records, indent=2) + "\n")
    # Refresh navigation only, without touching media or source metric files.
    from split_model_outputs import _root_index
    records = json.loads((root / "model_results.json").read_text())["results"]
    _root_index(root, records)
    for record in records:
        run_index(root / record["target"], record["label"], root)
    return {"protocol_groups": len(groups), "model_reports": len(model_reports), "charts": chart_count}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    args = parser.parse_args()
    print(json.dumps(build_analysis(args.root), indent=2))
