#!/usr/bin/env python3
"""Analyze gripper weighting across zero and normalized-action FSQ models."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import html
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import analyze_action_codebook as base
import analyze_normalized_action as normalized


DISPLAY_NAMES = {
    "js_norm_action": "normalized action · JS · grip 1.0",
    "js_norm_action01": "normalized action · JS · grip 0.1",
    "js_zero": "zero · JS · grip 1.0",
    "js_zero01": "zero · JS · grip 0.1",
    "none_zero": "zero · pair OFF · grip 1.0",
    "none_zero01": "zero · pair OFF · grip 0.1",
}
COLORS = {
    "js_norm_action": "#2a9d8f",
    "js_norm_action01": "#4cc9f0",
    "js_zero": "#e9c46a",
    "js_zero01": "#ef476f",
    "none_zero": "#7b8cde",
    "none_zero01": "#9b5de5",
}
ORDER = {name: index for index, name in enumerate(DISPLAY_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--skill-bundle", type=Path, required=True)
    parser.add_argument("--output-name", default="gripper_effect_analysis.html")
    return parser.parse_args()


def enrich(result: dict[str, Any]) -> None:
    result["display_name"] = DISPLAY_NAMES[result["name"]]
    meta = json.loads((Path(result["run_dir"]) / "fsq_meta.json").read_text())
    result["gripper_weight"] = float(meta.get("action_gripper_weight", 1.0) or 1.0)
    result["normalization"] = result["mode"] == "norm_action"


def add_replay_cohesion(result: dict[str, Any], collection_path: Path) -> None:
    """Add the replay report's task x skill-order cell cohesion metric."""
    collection = json.loads(collection_path.read_text())
    checkpoint = collection["checkpoints"][-1]
    cells: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    for skill in checkpoint["skills"]:
        token = int(skill["token"])
        for occurrence in skill["occurrences"]:
            key = (int(occurrence["task_id"]), int(occurrence["skill_index"]))
            cells[key][token] += 1
    effective: list[float] = []
    top_share: list[float] = []
    for counts in cells.values():
        total = sum(counts.values())
        probability = [count / total for count in counts.values()]
        effective.append(math.exp(-sum(p * math.log(p) for p in probability)))
        top_share.append(max(counts.values()) / total)
    result["replay_cell_count"] = len(cells)
    result["replay_cell_effective"] = float(np.mean(effective))
    result["replay_cell_top_share"] = float(np.mean(top_share))


def save_detail(result: dict[str, Any], output_dir: Path) -> None:
    old_colors = base.MODEL_COLORS.copy()
    base.MODEL_COLORS.update(COLORS)
    try:
        base.save_detail_plots(result, output_dir)
    finally:
        base.MODEL_COLORS.clear()
        base.MODEL_COLORS.update(old_colors)
    (output_dir / "metrics.json").write_text(
        json.dumps(base.json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fieldnames = sorted({key for row in result["codes"] for key in row})
    with (output_dir / "code_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(base.json_ready(result["codes"]))
    detail = normalized.detail_html(result)
    detail = detail.replace(
        "../../normalized_action_analysis.html", "../../gripper_effect_analysis.html"
    ).replace("normalized-action analysis", "gripper-effect analysis")
    (output_dir / "index.html").write_text(detail, encoding="utf-8")


def group_metric(result: dict[str, Any], group: str) -> float:
    return float(
        next(
            row["balanced_accuracy"]
            for row in result["classification_ablation"]
            if row["group"] == group
        )
    )


def save_plots(
    results: list[dict[str, Any]], assignment_nmi: np.ndarray, output_dir: Path
) -> None:
    labels = [result["display_name"] for result in results]
    colors = [COLORS[result["name"]] for result in results]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), constrained_layout=True)
    metrics = (
        ("effective_codes", "Effective codes"),
        ("largest_code_share", "Largest-code share"),
        ("state_direction_nmi", "Actual XYZ direction NMI"),
        ("state_direction_coherence", "Direction coherence"),
        ("gripper_nmi", "Gripper NMI"),
        ("opposite_adjacent_same_code_rate", "Opposite adjacent collision"),
    )
    for axis, (key, title) in zip(axes.flat, metrics, strict=True):
        axis.bar(np.arange(len(results)), [r[key] for r in results], color=colors)
        axis.set_xticks(np.arange(len(results)), labels, rotation=23, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        if "share" in key or "rate" in key:
            axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.savefig(output_dir / "gripper_effect_comparison.png", dpi=170)
    plt.close(fig)

    groups = ("action XYZ", "action rotation", "action gripper", "실제 state motion", "길이·순서")
    x = np.arange(len(groups))
    width = 0.82 / len(results)
    fig, axis = plt.subplots(figsize=(14.5, 6.2), constrained_layout=True)
    for index, result in enumerate(results):
        axis.bar(
            x + (index - (len(results) - 1) / 2) * width,
            [group_metric(result, group) for group in groups],
            width,
            label=result["display_name"],
            color=colors[index],
        )
    axis.set_xticks(x, ("action XYZ", "action rotation", "action gripper", "actual motion", "length/order"))
    axis.set_ylabel("episode-held-out balanced accuracy")
    axis.set_title("Which feature group predicts the assigned code?")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(ncol=2, fontsize=9)
    fig.savefig(output_dir / "gripper_effect_feature_groups.png", dpi=170)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 8.4), constrained_layout=True)
    image = axis.imshow(assignment_nmi, cmap="viridis", vmin=0, vmax=1)
    axis.set_xticks(range(len(results)), labels, rotation=24, ha="right")
    axis.set_yticks(range(len(results)), labels)
    for row in range(len(results)):
        for column in range(len(results)):
            value = assignment_nmi[row, column]
            axis.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value < 0.65 else "black",
            )
    axis.set_title("Code assignment similarity (NMI)")
    fig.colorbar(image, ax=axis, shrink=0.82)
    fig.savefig(output_dir / "gripper_effect_assignment_nmi.png", dpi=170)
    plt.close(fig)


def delta(before: float, after: float, *, percent: bool = False) -> str:
    if percent:
        return f"{before:.1%} → {after:.1%} ({after - before:+.1%}p)"
    return f"{before:.3f} → {after:.3f} ({after - before:+.3f})"


def summary_html(
    results: list[dict[str, Any]],
    assignment_nmi: np.ndarray,
    assignment_ari: np.ndarray,
) -> str:
    lookup = {result["name"]: result for result in results}
    z = lookup["js_zero"]
    z01 = lookup["js_zero01"]
    n = lookup["js_norm_action"]
    n01 = lookup["js_norm_action01"]
    z_none = lookup["none_zero"]
    z01_none = lookup["none_zero01"]
    indices = {result["name"]: i for i, result in enumerate(results)}

    cards = "".join(
        f"<a class='card' href='{html.escape(r['run_name'])}/gripper_effect_analysis/index.html'>"
        f"<span>{html.escape(r['epoch'])} · {html.escape(r['mode'])}</span>"
        f"<h3>{html.escape(r['display_name'])}</h3><dl>"
        f"<dt>effective code</dt><dd>{r['effective_codes']:.2f}</dd>"
        f"<dt>direction NMI</dt><dd>{r['state_direction_nmi']:.3f}</dd>"
        f"<dt>direction purity</dt><dd>{r['state_direction_purity']:.1%}</dd>"
        f"<dt>gripper NMI</dt><dd>{r['gripper_nmi']:.3f}</dd>"
        f"<dt>cell cohesion</dt><dd>{r['replay_cell_effective']:.2f}</dd>"
        f"<dt>opposite collision</dt><dd>{r['opposite_adjacent_same_code_rate']:.1%}</dd>"
        f"</dl><b>상세 분석 열기 →</b></a>"
        for r in results
    )
    rows = "".join(
        "<tr>"
        f"<td><a href='{html.escape(r['run_name'])}/gripper_effect_analysis/index.html'>{html.escape(r['display_name'])}</a></td>"
        f"<td>{r['epoch']}</td><td>{r['effective_codes']:.2f}</td><td>{r['largest_code_share']:.1%}</td>"
        f"<td>{r['state_direction_nmi']:.3f}</td><td>{r['state_direction_purity']:.1%}</td>"
        f"<td>{r['state_direction_coherence']:.3f}</td><td>{r['gripper_nmi']:.3f}</td>"
        f"<td>{group_metric(r, 'action gripper'):.1%}</td><td>{r['replay_cell_effective']:.2f}</td><td>{r['task_nmi']:.3f}</td>"
        f"<td>{r['adjacent_same_code_rate']:.1%}</td><td>{r['opposite_adjacent_same_code_rate']:.1%}</td>"
        f"<td>{r['task6']['first_pair_separation_rate']:.0%}</td></tr>"
        for r in results
    )

    z01_better = (
        z01["state_direction_nmi"] >= n01["state_direction_nmi"]
        and z01["opposite_adjacent_same_code_rate"] <= n01["opposite_adjacent_same_code_rate"]
    )
    verdict = (
        "zero 0.1이 normalized action보다 방향 category와 반대방향 충돌을 함께 더 잘 정리했다."
        if z01_better
        else "zero 0.1과 normalized action은 한쪽의 완승이 아니라 서로 다른 장점이 있다."
    )
    zi, ni = indices["js_zero01"], indices["js_norm_action01"]
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>gripper effect · FSQ codebook comparison</title>
<style>
:root{{--bg:#08111d;--panel:#111f31;--line:#29415d;--text:#edf5ff;--muted:#9eb1c8;--cyan:#65d6ff;--green:#6ee7b7;--amber:#ffc76b}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 12% 0,#183858,var(--bg) 38%);color:var(--text);font:15px/1.65 Inter,system-ui,"Noto Sans KR",sans-serif}}main{{max-width:1480px;margin:auto;padding:52px 28px 90px}}h1{{font-size:clamp(34px,5vw,58px);line-height:1.05;margin:6px 0 15px}}h2{{margin-top:48px}}p{{color:#c8d7e9}}a{{color:var(--cyan)}}.lead{{font-size:17px;max-width:1100px}}.callout{{background:#10283a;border:1px solid #34718b;border-radius:14px;padding:17px 20px;margin:18px 0}}.warn{{background:#2a2216;border-color:#a77c37}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:13px}}.card{{display:block;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:17px;text-decoration:none;color:var(--text)}}.card:hover{{border-color:var(--cyan)}}.card span,.muted{{color:var(--muted)}}.card h3{{margin:6px 0 12px}}dl{{display:grid;grid-template-columns:1fr auto;gap:4px 10px}}dt{{color:var(--muted)}}dd{{margin:0;font-weight:700}}figure{{background:white;padding:10px;border-radius:13px;margin:20px 0}}figure img{{display:block;width:100%}}figcaption{{color:#42536a;padding:8px}}.plots{{display:grid;grid-template-columns:1.25fr 1fr;gap:16px}}.table{{overflow:auto;border:1px solid var(--line);border-radius:11px}}table{{border-collapse:collapse;width:100%;background:#0d1928}}th,td{{padding:9px 11px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}thead th{{background:#182b42;position:sticky;top:0}}.good{{color:var(--green)}}@media(max-width:950px){{.cards,.plots{{grid-template-columns:1fr}}main{{padding:30px 14px}}}}
</style></head><body><main>
<div class="muted">complete training skillset · 11,221 skills · latest evaluated checkpoint per model</div>
<h1>gripper weight와 입력 표현<br>codebook categorization</h1>
<p class="lead">zero-grounded proprio와 normalized action에서 gripper loss weight 1.0/0.1이 code assignment를 어떻게 바꾸는지 비교한다. 0.1은 feature 값을 0.1배 하는 것이 아니라 sqrt(0.1)배 하여 제곱오차 기여를 0.1배로 만드는 설정이다.</p>
<div class="callout"><b class="good">zero 0.1 vs normalized action 0.1:</b> {verdict} 방향 NMI는 {z01['state_direction_nmi']:.3f} vs {n01['state_direction_nmi']:.3f}, purity는 {z01['state_direction_purity']:.1%} vs {n01['state_direction_purity']:.1%}, coherence는 {z01['state_direction_coherence']:.3f} vs {n01['state_direction_coherence']:.3f}, 반대방향 인접 collision은 {z01['opposite_adjacent_same_code_rate']:.1%} vs {n01['opposite_adjacent_same_code_rate']:.1%}다.</div>
<div class="callout"><b>왜 zero 0.1이 눈으로 더 좋아 보일 수 있나:</b> task×skill-order cell 내부 effective code는 zero 0.1이 {z01['replay_cell_effective']:.2f}, normalized action 0.1이 {n01['replay_cell_effective']:.2f}로 zero가 더 응집돼 있다(낮을수록 같은 상황이 한 code로 모임). 전체 effective code도 {z01['effective_codes']:.2f} vs {n01['effective_codes']:.2f}라 zero가 훨씬 compact하다. 반면 normalized action은 episode마다 실제 motion 차이를 더 세밀하게 나눠 방향 지표와 collision이 더 좋다.</div>
<div class="callout"><b>분할량 차이:</b> assignment NMI/ARI는 {assignment_nmi[zi, ni]:.3f}/{assignment_ari[zi, ni]:.3f}로, 같은 gripper weight여도 두 입력은 상당히 다른 partition을 학습했다.</div>
<div class="callout"><b>zero에서 gripper 약화의 순효과(JS):</b> 방향 NMI {delta(z['state_direction_nmi'], z01['state_direction_nmi'])}, gripper NMI {delta(z['gripper_nmi'], z01['gripper_nmi'])}, effective code {z['effective_codes']:.2f} → {z01['effective_codes']:.2f}, 반대방향 collision {delta(z['opposite_adjacent_same_code_rate'], z01['opposite_adjacent_same_code_rate'], percent=True)}다.</div>
<div class="callout"><b>pair OFF에서도 재현되는가:</b> zero grip1 → grip0.1에서 방향 NMI {delta(z_none['state_direction_nmi'], z01_none['state_direction_nmi'])}, gripper NMI {delta(z_none['gripper_nmi'], z01_none['gripper_nmi'])}, effective code {z_none['effective_codes']:.2f} → {z01_none['effective_codes']:.2f}다.</div>
<div class="callout"><b>normalized action의 gripper 약화:</b> grip1(epoch700) → grip0.1(epoch2000)에서 방향 NMI {delta(n['state_direction_nmi'], n01['state_direction_nmi'])}, gripper NMI {delta(n['gripper_nmi'], n01['gripper_nmi'])}, effective code {n['effective_codes']:.2f} → {n01['effective_codes']:.2f}다.</div>
<div class="callout warn"><b>해석 제한:</b> normalized action grip1은 epoch700, grip0.1은 epoch2000이라 gripper weight의 순수 ablation이 아니다. 반면 zero JS와 zero pair-OFF의 1.0↔0.1 비교, 그리고 zero0.1↔normalized-action0.1 비교는 모두 epoch2000이다.</div>
<div class="cards">{cards}</div>
<h2>정량 비교</h2><figure><img src="gripper_effect_comparison.png"><figcaption>사용량, 실제 이동방향, gripper 연관, 반대방향 collision 비교.</figcaption></figure>
<div class="table"><table><thead><tr><th>model</th><th>epoch</th><th>effective</th><th>max share</th><th>direction NMI</th><th>purity</th><th>coherence</th><th>gripper NMI</th><th>gripper-only BA</th><th>cell effective</th><th>task NMI</th><th>adjacent same</th><th>opposite collision</th><th>task6 분리</th></tr></thead><tbody>{rows}</tbody></table></div>
<h2>무엇이 code를 설명하는가</h2><div class="plots"><figure><img src="gripper_effect_feature_groups.png"><figcaption>각 특징 그룹만으로 code를 예측한 episode-held-out balanced accuracy.</figcaption></figure><figure><img src="gripper_effect_assignment_nmi.png"><figcaption>code 번호 permutation에 무관한 assignment partition NMI.</figcaption></figure></div>
<p class="muted">raw metrics: <a href="gripper_effect_analysis.json">gripper_effect_analysis.json</a></p>
</main></body></html>"""


def main() -> None:
    args = parse_args()
    report_root = args.report_root.resolve()
    dataset = base.load_bundle(args.skill_bundle.resolve())
    paths = sorted(report_root.glob("*/metrics/collection.json"))
    found = {json.loads(path.read_text()).get("model_name") for path in paths}
    if found != set(DISPLAY_NAMES):
        raise RuntimeError(f"Expected {sorted(DISPLAY_NAMES)}, found {sorted(found)}")

    paired: list[tuple[dict[str, Any], np.ndarray]] = []
    for path in paths:
        print(f"Analyzing {path.parent.parent.name}", flush=True)
        result = base.model_metrics(path, dataset)
        enrich(result)
        add_replay_cohesion(result, path)
        save_detail(result, path.parent.parent / "gripper_effect_analysis")
        paired.append((result, normalized.load_tokens(path)))
    paired.sort(key=lambda pair: ORDER[pair[0]["name"]])
    results = [pair[0] for pair in paired]
    tokens = [pair[1] for pair in paired]

    count = len(results)
    assignment_nmi = np.eye(count)
    assignment_ari = np.eye(count)
    for first in range(count):
        for second in range(first + 1, count):
            assignment_nmi[first, second] = assignment_nmi[second, first] = normalized_mutual_info_score(tokens[first], tokens[second])
            assignment_ari[first, second] = assignment_ari[second, first] = adjusted_rand_score(tokens[first], tokens[second])

    save_plots(results, assignment_nmi, report_root)
    payload = {"models": results, "assignment_nmi": assignment_nmi, "assignment_ari": assignment_ari}
    (report_root / "gripper_effect_analysis.json").write_text(
        json.dumps(base.json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output_path = report_root / args.output_name
    output_path.write_text(summary_html(results, assignment_nmi, assignment_ari), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
