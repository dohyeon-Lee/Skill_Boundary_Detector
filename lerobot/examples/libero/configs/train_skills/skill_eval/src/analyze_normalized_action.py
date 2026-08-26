#!/usr/bin/env python3
"""Compare raw-action and normalized-action FSQ codebooks.

This builds on :mod:`analyze_action_codebook` so every metric is computed on
the complete training skill bundle.  The replay subset is used only for the
already-generated visual replay pages and the task-6 exact episode selection.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import analyze_action_codebook as base


DISPLAY_NAMES = {
    "action_js": "raw action · JS · grip 1.0",
    "norm_action_js": "normalized action · JS · grip 1.0",
    "norm01_action_js": "normalized action · JS · grip 0.1",
    "norm01_action_none": "normalized action · pair OFF · grip 0.1",
}
MODEL_COLORS = {
    "action_js": "#f4a261",
    "norm_action_js": "#2a9d8f",
    "norm01_action_js": "#4cc9f0",
    "norm01_action_none": "#9b5de5",
}
ORDER = {name: index for index, name in enumerate(DISPLAY_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--skill-bundle", type=Path, required=True)
    parser.add_argument("--output-name", default="normalized_action_analysis.html")
    return parser.parse_args()


def load_tokens(collection_path: Path) -> np.ndarray:
    collection = json.loads(collection_path.read_text())
    checkpoint = collection["checkpoints"][-1]
    model_root = collection_path.parent.parent
    manifest_path = model_root / "checkpoints" / checkpoint["epoch_tag"] / "metrics" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    with np.load(manifest["signature"]["latents_path"], allow_pickle=False) as data:
        return data["tokens"].astype(np.int64)


def enrich(result: dict[str, Any]) -> None:
    result["display_name"] = DISPLAY_NAMES.get(result["name"], result["name"])
    meta = json.loads((Path(result["run_dir"]) / "fsq_meta.json").read_text())
    result["gripper_weight"] = float(meta.get("action_gripper_weight", 1.0))
    result["normalization"] = result["mode"] == "norm_action"


def top_features(result: dict[str, Any], count: int = 7) -> str:
    return ", ".join(
        f"{base.FEATURE_LABELS.get(row['feature'], row['feature'])} ({row['importance']:.3f})"
        for row in result["feature_importance"][:count]
    )


def axis_notes(result: dict[str, Any]) -> str:
    notes = []
    for axis in range(3):
        rows = sorted(
            (row for row in result["axis_correlations"] if row["axis"] == axis),
            key=lambda row: abs(row["correlation"]),
            reverse=True,
        )[:3]
        notes.append(
            f"axis {axis}: "
            + ", ".join(
                f"{base.FEATURE_LABELS.get(row['feature'], row['feature'])} "
                f"ρ={row['correlation']:+.3f}"
                for row in rows
            )
        )
    return " · ".join(notes)


def detail_html(result: dict[str, Any]) -> str:
    replay_metric = (
        f"<div class='metric'><span>task×skill-order cohesion</span>"
        f"<b>{result['replay_cell_effective']:.2f}</b>"
        f"<small>낮을수록 같은 상황이 한 code로 모임</small></div>"
        if "replay_cell_effective" in result
        else ""
    )
    ablation_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['group'])}</td><td>{row['feature_count']}</td>"
        f"<td>{row['accuracy']:.1%}</td><td>{row['balanced_accuracy']:.1%}</td>"
        "</tr>"
        for row in result["classification_ablation"]
    )
    code_rows = "".join(
        "<tr>"
        f"<td>{row['code']}</td><td>{row['count']}</td>"
        + (
            f"<td>{row['share']:.1%}</td>"
            f"<td>{row['state_direction']} ({row['state_direction_purity']:.0%})</td>"
            f"<td>{row['action_direction']} ({row['action_direction_purity']:.0%})</td>"
            f"<td>{row['grip_regime']} ({row['grip_purity']:.0%})</td>"
            f"<td>{row['mean_frames']:.1f}</td>"
            if row["count"]
            else "<td colspan='5'>unused</td>"
        )
        + "</tr>"
        for row in result["codes"]
    )
    task_rows = []
    for episode in result["task6"]["episodes"]:
        text = " · ".join(
            f"s{skill['skill_index']} → c{skill['code']} / {skill['state_direction']} / "
            f"Δp=[{', '.join(f'{value:+.3f}' for value in skill['displacement'])}]"
            for skill in episode["skills"]
        )
        task_rows.append(f"<tr><td>{episode['episode']}</td><td>{html.escape(text)}</td></tr>")
    if result["mode"] == "norm_action":
        preprocessing = (
            "action 각 축을 training q01/q99 기준 [-1,1]로 선형 정규화하고 clipping한 뒤, "
            f"gripper 채널을 sqrt({result['gripper_weight']:g})배 한다(loss에서는 "
            f"weight={result['gripper_weight']:g}). 입력과 reconstruction target에 동일하게 적용된다."
        )
    elif result["mode"] == "action":
        preprocessing = "7D controller action을 별도 정규화 없이 그대로 입력·복원하며 gripper 배율은 1.0이다."
    else:
        preprocessing = (
            "zero-grounded proprio spline control point를 입력·복원한다. "
            f"gripper state 채널은 sqrt({result['gripper_weight']:g})배 "
            f"스케일하여 loss weight={result['gripper_weight']:g}를 적용한다."
        )
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(result['display_name'])} · normalized-action analysis</title>
<style>
:root{{--bg:#08111d;--panel:#111f31;--line:#29415d;--text:#edf5ff;--muted:#9eb1c8;--cyan:#65d6ff;--amber:#ffc76b}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 12% 0,#183858,var(--bg) 38%);color:var(--text);font:15px/1.65 Inter,system-ui,"Noto Sans KR",sans-serif}}main{{max-width:1320px;margin:auto;padding:44px 28px 90px}}a{{color:var(--cyan)}}h1{{font-size:clamp(30px,4vw,50px);line-height:1.1;margin:8px 0}}h2{{margin-top:44px}}p{{color:#c9d6e7}}.back{{display:inline-block;padding:7px 12px;border:1px solid var(--line);border-radius:999px;text-decoration:none}}.chips{{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}}.chips span{{background:#19304a;border-radius:999px;padding:6px 11px}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}}.metric,.callout{{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:15px}}.metric b{{display:block;font-size:24px;color:var(--amber)}}.callout{{margin:18px 0}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}figure{{margin:0;background:white;border-radius:12px;padding:9px}}figure img{{display:block;width:100%}}figcaption{{color:#44546b;padding:7px}}.table{{overflow:auto;border:1px solid var(--line);border-radius:11px}}table{{border-collapse:collapse;width:100%;background:#0e1a2a}}th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child,td:nth-child(2){{text-align:left}}thead th{{background:#182b42;position:sticky;top:0}}.muted{{color:var(--muted)}}@media(max-width:850px){{.metrics,.grid{{grid-template-columns:1fr}}main{{padding:28px 14px}}}}
</style></head><body><main>
<a class="back" href="../../normalized_action_analysis.html">← 전체 모델 비교</a>
<div class="chips"><span>{html.escape(result['epoch'])}</span><span>mode={html.escape(result['mode'])}</span><span>pair={html.escape(result['pair_loss'])}</span><span>grip={result['gripper_weight']:g}</span><span>route=ON</span></div>
<h1>{html.escape(result['display_name'])}</h1>
<p>{html.escape(preprocessing)}</p>
<div class="metrics">
 <div class="metric"><span>effective codes</span><b>{result['effective_codes']:.2f}</b><small>{result['used_codes']}/27 active</small></div>
 <div class="metric"><span>실제 XYZ 방향 NMI</span><b>{result['state_direction_nmi']:.3f}</b><small>purity {result['state_direction_purity']:.1%}</small></div>
 <div class="metric"><span>gripper NMI</span><b>{result['gripper_nmi']:.3f}</b><small>낮을수록 gripper 의존 감소</small></div>
 <div class="metric"><span>인접 same-code</span><b>{result['adjacent_same_code_rate']:.1%}</b><small>반대방향 조건 {result['opposite_adjacent_same_code_rate']:.1%}</small></div>
 {replay_metric}
</div>
<div class="callout"><b>가장 설명력 높은 특징:</b> {html.escape(top_features(result))}</div>
<div class="callout"><b>FSQ 축 판독:</b> {html.escape(axis_notes(result))}</div>
<p class="muted">NMI와 feature importance는 인과효과가 아니라 code assignment와 관측 특징 사이의 연관성이다. 모든 지표는 replay의 1,290 occurrence가 아니라 전체 training skill 11,221개에서 계산했다.</p>
<div class="grid"><figure><img src="code_usage.png"><figcaption>전체 skill의 code 사용량.</figcaption></figure><figure><img src="direction_by_code.png"><figcaption>code 내부 실제 EE 순변위 방향 구성.</figcaption></figure></div>
<h2>Code assignment 설명력</h2>
<div class="grid"><figure><img src="feature_importance.png"><figcaption>모든 특징을 함께 사용한 ExtraTrees importance.</figcaption></figure><figure><img src="fsq_axis_correlation.png"><figcaption>세 FSQ scalar axis와 주요 특징의 Spearman 상관.</figcaption></figure></div>
<div class="table"><table><thead><tr><th>특징 그룹</th><th>차원</th><th>episode-held-out 정확도</th><th>balanced 정확도</th></tr></thead><tbody>{ablation_rows}</tbody></table></div>
<h2>task 6 exact 재검증</h2>
<div class="callout">첫 두 skill의 분리율은 <b>{result['task6']['first_pair_separation_rate']:.0%}</b> ({result['task6']['first_pair_count'] - result['task6']['first_pair_same_code']}/{result['task6']['first_pair_count']})이다.</div>
<div class="table"><table><thead><tr><th>episode</th><th>skill → code / 실제 방향 / 변위</th></tr></thead><tbody>{''.join(task_rows)}</tbody></table></div>
<h2>Code별 요약</h2>
<div class="table"><table><thead><tr><th>code</th><th>count</th><th>share</th><th>실제 방향</th><th>action 방향</th><th>gripper</th><th>평균 길이</th></tr></thead><tbody>{code_rows}</tbody></table></div>
<p class="muted">원자료: <a href="metrics.json">metrics.json</a> · <a href="code_summary.csv">code_summary.csv</a></p>
</main></body></html>"""


def save_detail(result: dict[str, Any], output_dir: Path) -> None:
    old_colors = base.MODEL_COLORS.copy()
    base.MODEL_COLORS.update(MODEL_COLORS)
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
    (output_dir / "index.html").write_text(detail_html(result), encoding="utf-8")


def group_metric(result: dict[str, Any], group: str) -> float:
    row = next(row for row in result["classification_ablation"] if row["group"] == group)
    return float(row["balanced_accuracy"])


def save_summary_plots(
    results: list[dict[str, Any]],
    assignment_nmi: np.ndarray,
    output_dir: Path,
) -> None:
    labels = [result["display_name"] for result in results]
    colors = [MODEL_COLORS[result["name"]] for result in results]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    metrics = (
        ("effective_codes", "Effective codes"),
        ("largest_code_share", "Largest-code share"),
        ("state_direction_nmi", "Actual XYZ direction NMI"),
        ("state_direction_coherence", "Direction coherence"),
        ("gripper_nmi", "Gripper NMI"),
        ("adjacent_same_code_rate", "Adjacent same-code rate"),
    )
    for axis, (key, title) in zip(axes.flat, metrics, strict=True):
        axis.bar(np.arange(len(results)), [result[key] for result in results], color=colors)
        axis.set_xticks(np.arange(len(results)), labels, rotation=20, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        if "share" in key or "rate" in key:
            axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.savefig(output_dir / "normalized_action_comparison.png", dpi=170)
    plt.close(fig)

    groups = ("action XYZ", "action rotation", "action gripper", "실제 state motion", "길이·순서")
    plot_group_labels = ("action XYZ", "action rotation", "action gripper", "actual state motion", "length / order")
    x = np.arange(len(groups))
    width = 0.78 / len(results)
    fig, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    for index, result in enumerate(results):
        axis.bar(
            x + (index - (len(results) - 1) / 2) * width,
            [group_metric(result, group) for group in groups],
            width,
            label=result["display_name"],
            color=colors[index],
        )
    axis.set_xticks(x, plot_group_labels)
    axis.set_ylabel("episode-held-out balanced accuracy")
    axis.set_title("Which feature group predicts the assigned code?")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    fig.savefig(output_dir / "normalized_action_feature_groups.png", dpi=170)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9.2, 7.2), constrained_layout=True)
    image = axis.imshow(assignment_nmi, cmap="viridis", vmin=0, vmax=1)
    axis.set_xticks(range(len(results)), labels, rotation=22, ha="right")
    axis.set_yticks(range(len(results)), labels)
    for row in range(len(results)):
        for column in range(len(results)):
            axis.text(column, row, f"{assignment_nmi[row, column]:.3f}", ha="center", va="center", color="white" if assignment_nmi[row, column] < 0.65 else "black")
    axis.set_title("Code assignment similarity\n(permutation-invariant NMI)")
    fig.colorbar(image, ax=axis, shrink=0.82)
    fig.savefig(output_dir / "normalized_action_assignment_nmi.png", dpi=170)
    plt.close(fig)


def delta_text(before: float, after: float, *, percent: bool = False) -> str:
    delta = after - before
    if percent:
        return f"{before:.1%} → {after:.1%} ({delta:+.1%}p)"
    return f"{before:.3f} → {after:.3f} ({delta:+.3f})"


def summary_html(
    results: list[dict[str, Any]],
    assignment_nmi: np.ndarray,
    assignment_ari: np.ndarray,
) -> str:
    lookup = {result["name"]: result for result in results}
    raw = lookup["action_js"]
    norm_full = lookup["norm_action_js"]
    norm = lookup["norm01_action_js"]
    no_pair = lookup["norm01_action_none"]
    def correlation(result: dict[str, Any], axis: int, feature: str) -> float:
        return next(
            row["correlation"]
            for row in result["axis_correlations"]
            if row["axis"] == axis and row["feature"] == feature
        )
    cards = "".join(
        f"<a class='card' href='{html.escape(result['run_name'])}/normalized_action_analysis/index.html'>"
        f"<span>{html.escape(result['epoch'])} · {html.escape(result['mode'])}</span>"
        f"<h3>{html.escape(result['display_name'])}</h3><dl>"
        f"<dt>effective code</dt><dd>{result['effective_codes']:.2f}</dd>"
        f"<dt>direction NMI</dt><dd>{result['state_direction_nmi']:.3f}</dd>"
        f"<dt>direction purity</dt><dd>{result['state_direction_purity']:.1%}</dd>"
        f"<dt>gripper NMI</dt><dd>{result['gripper_nmi']:.3f}</dd>"
        f"<dt>adjacent same</dt><dd>{result['adjacent_same_code_rate']:.1%}</dd>"
        f"</dl><b>상세 분석 열기 →</b></a>"
        for result in results
    )
    rows = "".join(
        "<tr>"
        f"<td><a href='{html.escape(result['run_name'])}/normalized_action_analysis/index.html'>{html.escape(result['display_name'])}</a></td>"
        f"<td>{result['effective_codes']:.2f}</td><td>{result['largest_code_share']:.1%}</td>"
        f"<td>{result['state_direction_nmi']:.3f}</td><td>{result['state_direction_purity']:.1%}</td>"
        f"<td>{result['state_direction_coherence']:.3f}</td><td>{result['gripper_nmi']:.3f}</td>"
        f"<td>{result['task_nmi']:.3f}</td><td>{result['skill_index_nmi']:.3f}</td>"
        f"<td>{result['adjacent_same_code_rate']:.1%}</td><td>{result['opposite_adjacent_same_code_rate']:.1%}</td>"
        "</tr>"
        for result in results
    )
    raw_index = next(index for index, result in enumerate(results) if result["name"] == "action_js")
    norm_full_index = next(index for index, result in enumerate(results) if result["name"] == "norm_action_js")
    norm_index = next(index for index, result in enumerate(results) if result["name"] == "norm01_action_js")
    pair_index = next(index for index, result in enumerate(results) if result["name"] == "norm01_action_none")
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>normalized action · codebook comparison</title>
<style>
:root{{--bg:#08111d;--panel:#111f31;--line:#29415d;--text:#edf5ff;--muted:#9eb1c8;--cyan:#65d6ff;--green:#6ee7b7;--amber:#ffc76b}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 12% 0,#183858,var(--bg) 38%);color:var(--text);font:15px/1.65 Inter,system-ui,"Noto Sans KR",sans-serif}}main{{max-width:1420px;margin:auto;padding:52px 28px 90px}}h1{{font-size:clamp(34px,5vw,58px);line-height:1.05;letter-spacing:-.04em;margin:6px 0 15px}}h2{{margin-top:48px}}p{{color:#c8d7e9}}a{{color:var(--cyan)}}.lead{{font-size:17px;max-width:1080px}}.callout{{background:#10283a;border:1px solid #34718b;border-radius:14px;padding:17px 20px;margin:18px 0}}.warning{{border-color:#a77c37;background:#2a2216}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:13px}}.card{{display:block;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:17px;text-decoration:none;color:var(--text);transition:.15s}}.card:hover{{transform:translateY(-2px);border-color:var(--cyan)}}.card span,.muted{{color:var(--muted)}}.card h3{{margin:6px 0 12px}}dl{{display:grid;grid-template-columns:1fr auto;gap:4px 10px}}dt{{color:var(--muted)}}dd{{margin:0;font-weight:700}}figure{{background:white;padding:10px;border-radius:13px;margin:20px 0}}figure img{{display:block;width:100%}}figcaption{{color:#42536a;padding:8px}}.plots{{display:grid;grid-template-columns:1.3fr 1fr;gap:16px}}.table{{overflow:auto;border:1px solid var(--line);border-radius:11px}}table{{border-collapse:collapse;width:100%;background:#0d1928}}th,td{{padding:9px 11px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}thead th{{background:#182b42;position:sticky;top:0}}.good{{color:var(--green)}}.amber{{color:var(--amber)}}@media(max-width:950px){{.cards,.plots{{grid-template-columns:1fr}}main{{padding:30px 14px}}}}
</style></head><body><main>
<div class="muted">complete training skillset · 11,221 skills · epoch 2000</div>
<h1>normalized action<br>codebook comparison</h1>
<p class="lead">raw action, 축별 normalized action, gripper loss weight 0.1, JS pair loss 제거를 단계별로 비교한다. 각 카드를 누르면 27개 code의 방향·gripper·FSQ 축 연관을 볼 수 있다.</p>
<div class="callout"><b class="good">normalization 비교(grip 1.0):</b> raw → normalized에서 실제 이동방향 NMI는 {delta_text(raw['state_direction_nmi'], norm_full['state_direction_nmi'])}, purity는 {delta_text(raw['state_direction_purity'], norm_full['state_direction_purity'], percent=True)}, gripper NMI는 {delta_text(raw['gripper_nmi'], norm_full['gripper_nmi'])}다.</div>
<div class="callout"><b>gripper 약화 비교(normalized):</b> weight 1.0 → 0.1에서 방향 NMI는 {delta_text(norm_full['state_direction_nmi'], norm['state_direction_nmi'])}, gripper NMI는 {delta_text(norm_full['gripper_nmi'], norm['gripper_nmi'])}, effective code는 {norm_full['effective_codes']:.2f} → {norm['effective_codes']:.2f}다.</div>
<div class="callout"><b>미세분할은 줄지 않았다:</b> effective code는 {raw['effective_codes']:.2f} → {norm['effective_codes']:.2f}로 오히려 늘었고, 27개 code도 모두 사용했다. 최대 code 점유율은 {raw['largest_code_share']:.1%} → {norm['largest_code_share']:.1%}다. 다만 늘어난 분할이 무작위인 것은 아니다. 방향 NMI·purity·coherence가 모두 상승하고 gripper NMI는 하락했으므로, 더 많은 code를 쓰면서 분할 기준이 gripper에서 XYZ motion 쪽으로 이동했다.</div>
<div class="callout"><b>Gripper가 사라진 것은 아니고 한 축으로 정리됐다:</b> raw에서는 axis 2가 gripper mean과 ρ={correlation(raw, 2, 'grip_mean'):+.3f}이고 axis 0도 gripper delta와 강하게 연결된다. normalized에서는 axis 0이 gripper mean(ρ={correlation(norm, 0, 'grip_mean'):+.3f}), axis 1이 y 방향 action(ρ={correlation(norm, 1, 'act_mean_y'):+.3f}), axis 2가 path length(ρ={correlation(norm, 2, 'state_path_xyz'):+.3f})로 분업한다. 즉 gripper는 여전히 code의 한 자유도를 쓰지만, 나머지 두 축이 motion 방향·크기를 더 선명하게 담당한다.</div>
<div class="callout"><b>JS의 순효과(normalized 내부):</b> pair OFF 대비 JS에서 방향 NMI는 {delta_text(no_pair['state_direction_nmi'], norm['state_direction_nmi'])}, 인접 same-code는 {delta_text(no_pair['adjacent_same_code_rate'], norm['adjacent_same_code_rate'], percent=True)}, 반대방향 인접 collision은 {delta_text(no_pair['opposite_adjacent_same_code_rate'], norm['opposite_adjacent_same_code_rate'], percent=True)}다.</div>
<div class="callout"><b>task 6 exact:</b> 첫 하강 skill과 다음 좌우 이동 skill의 분리율은 raw JS {raw['task6']['first_pair_separation_rate']:.0%}, normalized JS {norm['task6']['first_pair_separation_rate']:.0%}, normalized pair OFF {no_pair['task6']['first_pair_separation_rate']:.0%}다.</div>
<div class="callout"><b>Assignment 자체는 얼마나 바뀌었나:</b> raw↔normalized grip1 NMI/ARI는 {assignment_nmi[raw_index, norm_full_index]:.3f}/{assignment_ari[raw_index, norm_full_index]:.3f}, normalized grip1↔grip0.1은 {assignment_nmi[norm_full_index, norm_index]:.3f}/{assignment_ari[norm_full_index, norm_index]:.3f}, grip0.1 JS↔pair OFF는 {assignment_nmi[norm_index, pair_index]:.3f}/{assignment_ari[norm_index, pair_index]:.3f}다.</div>
<div class="callout warning"><b class="amber">해석 제한:</b> <code>norm_action1</code>은 최종 평가가 epoch700이고 나머지는 epoch2000이다. 따라서 grip1→grip0.1 차이에는 학습 시간 차이가 혼재한다. 반면 <code>zero01 JS</code>↔<code>norm_action01 JS</code>는 둘 다 epoch2000이므로 입력 표현 비교에 더 적합하다.</div>
<div class="cards">{cards}</div>
<h2>정량 비교</h2>
<figure><img src="normalized_action_comparison.png"><figcaption>사용량, 방향성, gripper 의존, 인접 skill collision을 동일 축으로 비교.</figcaption></figure>
<div class="table"><table><thead><tr><th>model</th><th>effective</th><th>max share</th><th>direction NMI</th><th>direction purity</th><th>coherence</th><th>gripper NMI</th><th>task NMI</th><th>index NMI</th><th>adjacent same</th><th>opposite collision</th></tr></thead><tbody>{rows}</tbody></table></div>
<h2>Code를 설명하는 정보</h2>
<div class="plots"><figure><img src="normalized_action_feature_groups.png"><figcaption>한 특징 그룹만으로 code를 예측한 episode-held-out balanced accuracy. 높을수록 해당 정보가 assignment에 강하게 남아 있다.</figcaption></figure><figure><img src="normalized_action_assignment_nmi.png"><figcaption>모델 사이 전체 11,221개 assignment partition 유사도.</figcaption></figure></div>
<h2>읽는 법</h2>
<p>방향 NMI/purity와 coherence가 오르고 gripper NMI 및 gripper-only 예측력이 내려가면, 의도대로 controller의 큰 수치 scale보다 이동방향이 code를 더 많이 결정하게 된 것이다. 반대로 effective code만 줄고 방향 지표도 함께 내려가면 semantic 정돈이 아니라 단순 병합에 가깝다.</p>
<p class="muted">raw metrics: <a href="normalized_action_analysis.json">normalized_action_analysis.json</a></p>
</main></body></html>"""


def main() -> None:
    args = parse_args()
    report_root = args.report_root.resolve()
    dataset = base.load_bundle(args.skill_bundle.resolve())
    paths = sorted(report_root.glob("*/metrics/collection.json"))
    paths = [path for path in paths if json.loads(path.read_text()).get("model_name") in DISPLAY_NAMES]
    if set(json.loads(path.read_text())["model_name"] for path in paths) != set(DISPLAY_NAMES):
        found = [json.loads(path.read_text()).get("model_name") for path in paths]
        raise RuntimeError(f"Expected {sorted(DISPLAY_NAMES)}, found {sorted(found)}")

    results: list[dict[str, Any]] = []
    tokens: list[np.ndarray] = []
    for path in paths:
        print(f"Analyzing {path.parent.parent.name}", flush=True)
        result = base.model_metrics(path, dataset)
        enrich(result)
        save_detail(result, path.parent.parent / "normalized_action_analysis")
        results.append(result)
        tokens.append(load_tokens(path))
    paired = sorted(zip(results, tokens, strict=True), key=lambda pair: ORDER[pair[0]["name"]])
    results = [pair[0] for pair in paired]
    tokens = [pair[1] for pair in paired]

    count = len(results)
    assignment_nmi = np.eye(count)
    assignment_ari = np.eye(count)
    for first in range(count):
        for second in range(first + 1, count):
            assignment_nmi[first, second] = assignment_nmi[second, first] = normalized_mutual_info_score(
                tokens[first], tokens[second]
            )
            assignment_ari[first, second] = assignment_ari[second, first] = adjusted_rand_score(
                tokens[first], tokens[second]
            )

    save_summary_plots(results, assignment_nmi, report_root)
    payload = {
        "models": results,
        "assignment_nmi": assignment_nmi,
        "assignment_ari": assignment_ari,
    }
    (report_root / "normalized_action_analysis.json").write_text(
        json.dumps(base.json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output_path = report_root / args.output_name
    output_path.write_text(summary_html(results, assignment_nmi, assignment_ari), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
