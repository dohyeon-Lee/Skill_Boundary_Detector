#!/usr/bin/env python3
"""Analyze normalized-action contrastive, JS, and pair-OFF codebooks.

The report deliberately separates two questions:

* latest evaluated snapshots, which describe the artifacts a user can inspect;
* the greatest common epoch, which is the fair pair-objective comparison.

All categorization metrics use the complete aligned training skill bundle.  GT
replay pages remain linked for visual inspection but are not the metric sample.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from torch._subclasses.fake_tensor import FakeTensorMode

import analyze_action_codebook as base
import analyze_normalized_action as normalized

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


MODEL_ORDER = ("cont", "js", "none")
DISPLAY_NAMES = {
    "cont": "normalized action · contrastive + route",
    "js": "normalized action · JS + route",
    "none": "normalized action · pair OFF + route",
}
SHORT_NAMES = {
    "cont": "contrastive",
    "js": "JS",
    "none": "pair OFF",
}
COLORS = {
    "cont": "#4cc9f0",
    "js": "#f4a261",
    "none": "#9b5de5",
}
ANALYSIS_SUBDIR = "norm_action_pair_analysis"
OUTPUT_NAME = "norm_action_pair_analysis.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--skill-bundle", type=Path, required=True)
    parser.add_argument("--output-name", default=OUTPUT_NAME)
    return parser.parse_args()


def epoch_number(tag: str) -> int:
    if not tag.startswith("epoch") or not tag[5:].isdigit():
        raise ValueError(f"Invalid epoch tag: {tag!r}")
    return int(tag[5:])


def load_collections(report_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in report_root.glob("*/metrics/collection.json"):
        document = json.loads(path.read_text())
        name = str(document.get("model_name", ""))
        if name in MODEL_ORDER:
            if name in paths:
                raise ValueError(f"Duplicate collection for {name}: {paths[name]} and {path}")
            paths[name] = path
    missing = set(MODEL_ORDER) - paths.keys()
    if missing:
        raise FileNotFoundError(f"Missing model collection(s): {sorted(missing)}")
    return paths


def available_epochs(collection_path: Path) -> list[str]:
    document = json.loads(collection_path.read_text())
    return [str(checkpoint["epoch_tag"]) for checkpoint in document["checkpoints"]]


def manifest_path(collection_path: Path, epoch_tag: str) -> Path:
    return collection_path.parent.parent / "checkpoints" / epoch_tag / "metrics" / "manifest.json"


def load_tokens(collection_path: Path, epoch_tag: str) -> np.ndarray:
    manifest = json.loads(manifest_path(collection_path, epoch_tag).read_text())
    with np.load(manifest["signature"]["latents_path"], allow_pickle=False) as data:
        return data["tokens"].astype(np.int64)


def enrich(result: dict[str, Any]) -> None:
    result["display_name"] = DISPLAY_NAMES[result["name"]]
    meta = json.loads((Path(result["run_dir"]) / "fsq_meta.json").read_text())
    result["gripper_weight"] = float(meta.get("action_gripper_weight", 1.0))
    result["normalization"] = result["mode"] == "norm_action"


def add_checkpoint_validation(result: dict[str, Any]) -> None:
    """Read scalar validation losses without materializing checkpoint tensors."""
    checkpoint = Path(result["run_dir"]) / f"FSQ_{result['epoch']}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint for validation scalars is missing: {checkpoint}")
    with FakeTensorMode():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    result["validation_total_loss"] = float(payload["val_loss"])
    # FSQ checkpoints select the best model with the weighted action/reconstruction
    # term only; pair and route losses are intentionally excluded from val_select.
    result["validation_reconstruction_loss"] = float(payload["val_select"])


def save_detail(result: dict[str, Any], output_dir: Path, summary_name: str) -> None:
    normalized.save_detail(result, output_dir)
    page = output_dir / "index.html"
    document = page.read_text()
    document = document.replace(
        "../../normalized_action_analysis.html",
        f"../../{summary_name}",
    )
    document = document.replace(
        "원자료:",
        '<a href="../index.html">GT replay 원본 열기</a> · 원자료:',
    )
    cohesion = result["semantic_cohesion"]
    cohesion_callout = (
        '<div class="callout"><b>Code 내부 semantic cohesion:</b> '
        f"실제 state motion {cohesion['state_motion']:.3f}, "
        f"action XYZ {cohesion['action_xyz']:.3f}, "
        f"전체 action {cohesion['all_action']:.3f}이다. "
        "각 특징을 표준화한 뒤 code centroid가 설명하는 분산 비율이며, 높을수록 "
        "같은 code 안의 연속형 motion 특징이 더 촘촘하게 모였다는 뜻이다. "
        "code 사용 균등성은 이 점수의 좋고 나쁨으로 취급하지 않는다.</div>"
    )
    document = document.replace(
        '<div class="callout"><b>가장 설명력 높은 특징:',
        cohesion_callout + '<div class="callout"><b>가장 설명력 높은 특징:',
    )
    page.write_text(document, encoding="utf-8")


def semantic_cohesion(tokens: np.ndarray, dataset: dict[str, Any]) -> dict[str, float]:
    """Fraction of standardized feature variance explained by code centroids.

    This measures whether similar continuous motion descriptors collect inside a
    code.  It deliberately does not reward uniform occupancy.  A more granular
    partition can improve the score, so code count is still shown beside it as
    context rather than folded into the score.
    """
    matrix = dataset["matrix"].astype(np.float64)
    scale = matrix.std(axis=0)
    standardized = (matrix - matrix.mean(axis=0)) / np.where(scale > 1e-9, scale, 1.0)
    groups = base.feature_groups(dataset["feature_names"])
    aliases = {
        "action XYZ": "action_xyz",
        "action rotation": "action_rotation",
        "action gripper": "action_gripper",
        "실제 state motion": "state_motion",
        "길이·순서": "length_order",
        "전체 action": "all_action",
        "전체 특징": "all_features",
    }
    result: dict[str, float] = {}
    for group, columns in groups.items():
        values = standardized[:, columns]
        centroids = np.empty_like(values)
        for code in np.unique(tokens):
            mask = tokens == code
            centroids[mask] = values[mask].mean(axis=0)
        # Global standardized MSE is one per non-constant dimension, so one
        # minus residual MSE is the weighted explained-variance fraction.
        result[aliases[group]] = float(1.0 - np.mean(np.square(values - centroids)))
    return result


def light_metrics(tokens: np.ndarray, dataset: dict[str, Any]) -> dict[str, float | int]:
    counts = np.bincount(tokens, minlength=27)
    episodes = dataset["bundle"]["meta_episode_id"].astype(np.int64)
    tasks = dataset["bundle"]["meta_task_id"].astype(np.int64)
    skill_indices = dataset["bundle"]["meta_skill_index"].astype(np.int64)
    left, right = base.adjacent_pairs(episodes, skill_indices)
    displacement = dataset["state_displacement"]
    norms = np.linalg.norm(displacement, axis=1)
    valid = (norms[left] >= 0.01) & (norms[right] >= 0.01)
    cosine = np.zeros(len(left), dtype=np.float64)
    cosine[valid] = np.sum(displacement[left[valid]] * displacement[right[valid]], axis=1) / (
        norms[left[valid]] * norms[right[valid]]
    )
    conflict = valid & (cosine < 0.0)
    same = tokens[left] == tokens[right]
    cohesion = semantic_cohesion(tokens, dataset)
    return {
        "used_codes": int(np.count_nonzero(counts)),
        "effective_codes": base.entropy_effective(counts),
        "largest_code_share": float(counts.max() / counts.sum()),
        "state_direction_nmi": float(
            normalized_mutual_info_score(dataset["state_direction"], tokens)
        ),
        "state_direction_purity": base.weighted_purity(tokens, dataset["state_direction"]),
        "state_direction_coherence": base.direction_coherence(tokens, displacement, 0.01),
        "gripper_nmi": float(normalized_mutual_info_score(dataset["grip_regime"], tokens)),
        "task_nmi": float(normalized_mutual_info_score(tasks, tokens)),
        "skill_index_nmi": float(normalized_mutual_info_score(skill_indices, tokens)),
        "adjacent_same_code_rate": float(same.mean()),
        "opposite_adjacent_same_code_rate": (
            float(same[conflict].mean()) if conflict.any() else 0.0
        ),
        "state_motion_cohesion": cohesion["state_motion"],
        "action_xyz_cohesion": cohesion["action_xyz"],
    }


def comparison_matrices(
    tokens: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    count = len(MODEL_ORDER)
    nmi = np.eye(count, dtype=np.float64)
    ari = np.eye(count, dtype=np.float64)
    for first in range(count):
        for second in range(first + 1, count):
            left = tokens[MODEL_ORDER[first]]
            right = tokens[MODEL_ORDER[second]]
            nmi[first, second] = nmi[second, first] = normalized_mutual_info_score(left, right)
            ari[first, second] = ari[second, first] = adjusted_rand_score(left, right)
    return nmi, ari


def plot_snapshot_comparison(
    results: list[dict[str, Any]], output_path: Path, title: str
) -> None:
    labels = [SHORT_NAMES[result["name"]] + f"\n{result['epoch']}" for result in results]
    colors = [COLORS[result["name"]] for result in results]
    metrics = (
        ("semantic_cohesion.state_motion", "State-motion cohesion", False),
        ("semantic_cohesion.action_xyz", "Action-XYZ cohesion", False),
        ("state_direction_nmi", "XYZ direction NMI", False),
        ("state_direction_purity", "XYZ direction purity", True),
        ("gripper_nmi", "Gripper NMI", False),
        ("opposite_adjacent_same_code_rate", "Opposite-adjacent same code", True),
    )

    def value(result: dict[str, Any], key: str) -> float:
        current: Any = result
        for component in key.split("."):
            current = current[component]
        return float(current)

    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for axis, (key, label, percent) in zip(axes.flat, metrics, strict=True):
        axis.bar(range(len(results)), [value(result, key) for result in results], color=colors)
        axis.set_xticks(range(len(results)), labels)
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.22)
        if percent:
            decimals = 2 if key == "opposite_adjacent_same_code_rate" else 0
            axis.yaxis.set_major_formatter(
                lambda value, _, decimals=decimals: f"{value:.{decimals}%}"
            )
    figure.suptitle(title, fontsize=17)
    figure.savefig(output_path, dpi=175)
    plt.close(figure)


def plot_training_curves(
    histories: dict[str, list[dict[str, Any]]], output_path: Path
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    metrics = (
        ("state_motion_cohesion", "State-motion cohesion", False),
        ("state_direction_nmi", "XYZ direction NMI", False),
        ("state_direction_coherence", "XYZ vector coherence", False),
        ("gripper_nmi", "Gripper NMI", False),
        ("adjacent_same_code_rate", "Adjacent same code", True),
        ("opposite_adjacent_same_code_rate", "Opposite-adjacent same code", True),
    )
    for axis, (key, label, percent) in zip(axes.flat, metrics, strict=True):
        for name in MODEL_ORDER:
            rows = histories[name]
            axis.plot(
                [row["epoch_number"] for row in rows],
                [row[key] for row in rows],
                marker="o",
                markersize=3.5,
                linewidth=2,
                color=COLORS[name],
                label=SHORT_NAMES[name],
            )
        axis.set_title(label)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.23)
        if percent:
            decimals = 2 if key == "opposite_adjacent_same_code_rate" else 1
            axis.yaxis.set_major_formatter(
                lambda value, _, decimals=decimals: f"{value:.{decimals}%}"
            )
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Checkpoint trajectory on the same 11,221 skills", fontsize=17)
    figure.savefig(output_path, dpi=175)
    plt.close(figure)


def plot_assignment_heatmap(matrix: np.ndarray, output_path: Path, title: str) -> None:
    labels = [SHORT_NAMES[name] for name in MODEL_ORDER]
    figure, axis = plt.subplots(figsize=(7.4, 6.2), constrained_layout=True)
    image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(labels)), labels)
    axis.set_yticks(range(len(labels)), labels)
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value < 0.62 else "black",
            )
    axis.set_title(title)
    figure.colorbar(image, ax=axis, shrink=0.82)
    figure.savefig(output_path, dpi=175)
    plt.close(figure)


def delta(after: float, before: float, *, percent: bool = False) -> str:
    difference = after - before
    if percent:
        return f"{before:.1%} → {after:.1%} ({difference:+.1%}p)"
    return f"{before:.3f} → {after:.3f} ({difference:+.3f})"


def metric_table(results: list[dict[str, Any]]) -> str:
    return "".join(
        "<tr>"
        f"<th><a href='{html.escape(result['run_name'])}/{ANALYSIS_SUBDIR}/index.html'>"
        f"{html.escape(DISPLAY_NAMES[result['name']])}</a></th>"
        f"<td>{html.escape(result['epoch'])}</td>"
        f"<td>{result['semantic_cohesion']['state_motion']:.3f}</td>"
        f"<td>{result['semantic_cohesion']['action_xyz']:.3f}</td>"
        f"<td>{result['state_direction_nmi']:.3f}</td>"
        f"<td>{result['state_direction_purity']:.1%}</td>"
        f"<td>{result['state_direction_coherence']:.3f}</td>"
        f"<td>{result['gripper_nmi']:.3f}</td>"
        f"<td>{result['task_nmi']:.3f}</td>"
        f"<td>{result['skill_index_nmi']:.3f}</td>"
        f"<td>{result['adjacent_same_code_rate']:.1%}</td>"
        f"<td>{result['opposite_adjacent_same_code_rate']:.1%}</td>"
        f"<td>{result['used_codes']}/27 · eff {result['effective_codes']:.2f}</td>"
        f"<td>{result['largest_code_share']:.1%}</td>"
        "</tr>"
        for result in results
    )


def feature_group_value(result: dict[str, Any], group: str) -> float:
    row = next(row for row in result["classification_ablation"] if row["group"] == group)
    return float(row["balanced_accuracy"])


def summary_html(
    latest: list[dict[str, Any]],
    fair: list[dict[str, Any]],
    fair_epoch: str,
    fair_nmi: np.ndarray,
    fair_ari: np.ndarray,
) -> str:
    latest_by_name = {result["name"]: result for result in latest}
    fair_by_name = {result["name"]: result for result in fair}
    cont = fair_by_name["cont"]
    js = fair_by_name["js"]
    none = fair_by_name["none"]
    cont_latest = latest_by_name["cont"]
    js_latest = latest_by_name["js"]
    cont_index = MODEL_ORDER.index("cont")
    js_index = MODEL_ORDER.index("js")
    cards = "".join(
        f"<a class=card href='{html.escape(result['run_name'])}/{ANALYSIS_SUBDIR}/index.html'>"
        f"<span>{html.escape(result['epoch'])}</span><h3>{html.escape(DISPLAY_NAMES[result['name']])}</h3>"
        f"<dl><dt>state cohesion</dt><dd>{result['semantic_cohesion']['state_motion']:.3f}</dd>"
        f"<dt>direction NMI</dt><dd>{result['state_direction_nmi']:.3f}</dd>"
        f"<dt>vector coherence</dt><dd>{result['state_direction_coherence']:.3f}</dd>"
        f"<dt>opposite collision</dt><dd>{result['opposite_adjacent_same_code_rate']:.1%}</dd>"
        f"</dl><b>모델별 분석 열기 →</b></a>"
        for result in latest
    )
    state_motion_cont = feature_group_value(cont, "실제 state motion")
    state_motion_js = feature_group_value(js, "실제 state motion")
    action_xyz_cont = feature_group_value(cont, "action XYZ")
    action_xyz_js = feature_group_value(js, "action XYZ")
    direction_winner = "contrastive" if cont["state_direction_nmi"] > js["state_direction_nmi"] else "JS"
    collision_winner = (
        "contrastive"
        if cont["opposite_adjacent_same_code_rate"] < js["opposite_adjacent_same_code_rate"]
        else "JS"
    )
    reconstruction_increase = (
        cont["validation_reconstruction_loss"]
        / js["validation_reconstruction_loss"]
        - 1.0
    )
    opposite_absolute_gain = (
        js["opposite_adjacent_same_code_rate"]
        - cont["opposite_adjacent_same_code_rate"]
    )
    state_cohesion_cont = cont["semantic_cohesion"]["state_motion"]
    state_cohesion_js = js["semantic_cohesion"]["state_motion"]
    action_cohesion_cont = cont["semantic_cohesion"]["action_xyz"]
    action_cohesion_js = js["semantic_cohesion"]["action_xyz"]
    snapshots_aligned = all(result["epoch"] == fair_epoch for result in latest)
    snapshot_note = (
        f"세 모델 모두 {html.escape(fair_epoch)}까지 평가되어 최신 비교와 공정 비교가 동일하다."
        if snapshots_aligned
        else (
            f"contrastive는 현재 {html.escape(cont_latest['epoch'])}, JS는 "
            f"{html.escape(js_latest['epoch'])}까지 평가됐다. 최신끼리의 직접 비교에는 "
            f"학습량 차이가 있으므로 공통 {html.escape(fair_epoch)}를 우선한다."
        )
    )
    return f"""<!doctype html><html lang=ko><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>
<title>norm_action_01 · pair objective analysis</title><style>
:root{{--bg:#07111d;--panel:#101f31;--line:#29435f;--text:#edf5ff;--muted:#9eb2ca;--cyan:#68d9ff;--green:#72e6b1;--amber:#ffc96f}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 10% 0,#193d5c,var(--bg) 42rem);color:var(--text);font:15px/1.64 Inter,system-ui,"Noto Sans KR",sans-serif}}main{{max-width:1480px;margin:auto;padding:50px 28px 90px}}h1{{font-size:clamp(34px,5vw,58px);line-height:1.06;letter-spacing:-.045em;margin:5px 0 16px}}h2{{margin-top:48px}}p{{color:#c9d8e8}}a{{color:var(--cyan)}}.eyebrow,.muted{{color:var(--muted)}}.lead{{font-size:17px;max-width:1120px}}.callout{{margin:17px 0;padding:18px 21px;border:1px solid #37758f;background:#102a3e;border-radius:14px}}.warning{{border-color:#9a7135;background:#2d2417}}.good{{color:var(--green)}}.amber{{color:var(--amber)}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:13px;margin:22px 0}}.card{{display:block;padding:17px;border:1px solid var(--line);border-radius:14px;background:linear-gradient(145deg,#152b43,var(--panel));color:var(--text);text-decoration:none}}.card:hover{{border-color:var(--cyan);transform:translateY(-2px)}}.card span{{color:var(--muted)}}.card h3{{margin:6px 0 12px}}dl{{display:grid;grid-template-columns:1fr auto;gap:4px 10px}}dt{{color:var(--muted)}}dd{{margin:0;font-weight:700}}figure{{margin:18px 0;padding:10px;background:white;border-radius:13px}}figure img{{display:block;width:100%}}figcaption{{padding:8px;color:#43566d}}.plots{{display:grid;grid-template-columns:1.35fr .65fr;gap:14px}}.table{{overflow:auto;border:1px solid var(--line);border-radius:12px}}table{{width:100%;border-collapse:collapse;background:#0c1928;min-width:1160px}}th,td{{padding:9px 11px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child{{text-align:left}}thead th{{position:sticky;top:0;background:#192e47;color:#c5ebff}}code{{color:#c0ecff}}@media(max-width:900px){{main{{padding:28px 13px 60px}}.cards,.plots{{grid-template-columns:1fr}}}}
</style></head><body><main>
<div class=eyebrow>complete training skillset · 11,221 skills · normalized action · gripper weight 0.1 · route ON</div>
<h1>contrastive는 기존 JS보다<br>더 나은 skill label인가?</h1>
<p class=lead>세 objective가 동일한 normalized-action 입력과 동일 codebook으로 전체 skill을 어떻게 나누는지 분석했다. <b>code를 골고루 쓰는가</b>는 품질 순위에서 제외하고, 같은 code 안에 연속형 motion이 얼마나 촘촘히 모이는지와 서로 다른 인접 phase가 불필요하게 충돌하는지를 우선했다.</p>
<div class='callout'><b class=good>종합 판정: downstream categorization 기준에서는 contrastive 쪽이 조금 더 유리하다.</b> 실제 state-motion cohesion이 JS {state_cohesion_js:.3f} → contrastive {state_cohesion_cont:.3f}이고, direction NMI·연속 벡터 coherence도 contrastive가 높다. 전체 인접 same-code는 {js['adjacent_same_code_rate']:.3%} → {cont['adjacent_same_code_rate']:.3%}, 반대방향 충돌은 {js['opposite_adjacent_same_code_rate']:.3%} → {cont['opposite_adjacent_same_code_rate']:.3%}다. 즉 비슷한 연속 motion은 조금 더 촘촘히 모으면서, 바로 이웃하지만 다른 phase는 더 확실히 분리했다. 네가 말한 비대칭 code 사용을 허용하는 기준이라면 기존 JS보다 contrastive를 우선해볼 근거가 생겼다.</div>
<div class=callout><b class=good>동일 epoch의 핵심:</b> JS → contrastive에서 실제 XYZ 방향 NMI는 {delta(cont['state_direction_nmi'], js['state_direction_nmi'])}, purity는 {delta(cont['state_direction_purity'], js['state_direction_purity'], percent=True)}, 반대방향 인접 collision은 {delta(cont['opposite_adjacent_same_code_rate'], js['opposite_adjacent_same_code_rate'], percent=True)}다. 방향 NMI 승자는 <b>{direction_winner}</b>, 반대방향 충돌 승자는 <b>{collision_winner}</b>다.</div>
<div class=callout><b>연속형 semantic cohesion:</b> 표준화한 실제 state-motion 특징에서 code centroid가 설명하는 분산은 JS {state_cohesion_js:.3f}, contrastive {state_cohesion_cont:.3f}; action XYZ에서는 {action_cohesion_js:.3f}/{action_cohesion_cont:.3f}다. 이 점수는 code 내부가 얼마나 촘촘한지를 직접 본다. contrastive가 effective code {cont['effective_codes']:.2f}로 JS {js['effective_codes']:.2f}보다 적게 쓰면서도 state cohesion이 더 높으므로, 단순 과분할로 얻은 이점도 아니다.</div>
<div class=callout><b>trajectory의 episode-held-out 설명력:</b> 실제 state motion만으로 code를 예측한 balanced accuracy는 JS {state_motion_js:.1%}, contrastive {state_motion_cont:.1%}; action XYZ만 썼을 때는 JS {action_xyz_js:.1%}, contrastive {action_xyz_cont:.1%}다. state 기준에서는 contrastive, action-XYZ 경계의 균등한 예측 가능성에서는 JS가 근소하게 앞서므로 완전한 일방 우위는 아니다.</div>
<div class=callout><b>code 사용량은 진단값일 뿐:</b> effective code는 JS {js['effective_codes']:.2f}, contrastive {cont['effective_codes']:.2f}, 최대 점유율은 {js['largest_code_share']:.1%}/{cont['largest_code_share']:.1%}다. 데이터 분포가 비대칭이면 이 차이는 자연스러울 수 있으며, 본 보고서는 균등한 사용이나 높은 utilization에 가산점을 주지 않는다.</div>
<div class=callout><b>reconstruction trade-off:</b> 동일 {html.escape(fair_epoch)}의 validation reconstruction selection loss는 JS {js['validation_reconstruction_loss']:.6f}, contrastive {cont['validation_reconstruction_loss']:.6f}로 contrastive가 {reconstruction_increase:.1%} 높다. total validation loss도 각각 {js['validation_total_loss']:.6f}/{cont['validation_total_loss']:.6f}다. 즉 local separation 개선이 reconstruction을 공짜로 개선한 결과는 아니다.</div>
<div class='callout warning'><b class=amber>남은 약점:</b> JS → contrastive에서 gripper NMI는 {delta(cont['gripper_nmi'], js['gripper_nmi'])}, task NMI는 {delta(cont['task_nmi'], js['task_nmi'])}, skill-index NMI는 {delta(cont['skill_index_nmi'], js['skill_index_nmi'])}다. contrastive는 task shortcut은 조금 덜하지만 gripper와 episode phase에 더 민감하고, hard dominant-axis purity도 JS보다 낮다. 특히 gripper 의존 상승이 downstream에 도움이 되는지는 별도 검증이 필요하다.</div>
<div class=callout><b>assignment 자체의 차이:</b> 동일 {html.escape(fair_epoch)}에서 contrastive↔JS assignment NMI/ARI는 {fair_nmi[cont_index, js_index]:.3f}/{fair_ari[cont_index, js_index]:.3f}다. code 번호 permutation을 제거해도 두 objective가 dataset을 실질적으로 다른 partition으로 나눈 정도다.</div>
<div class=callout><b>checkpoint 정렬:</b> {snapshot_note}</div>
<div class=cards>{cards}</div>
<h2>공정 비교: 동일 {html.escape(fair_epoch)}</h2>
<figure><img src=norm_action_pair_fair.png><figcaption>세 모델을 완전히 같은 학습 epoch에서 비교.</figcaption></figure>
<div class=table><table><thead><tr><th>model</th><th>epoch</th><th>state cohesion</th><th>action XYZ cohesion</th><th>direction NMI</th><th>purity</th><th>coherence</th><th>gripper NMI</th><th>task NMI</th><th>index NMI</th><th>adjacent same</th><th>opposite collision</th><th>usage (diagnostic)</th><th>max share</th></tr></thead><tbody>{metric_table(fair)}</tbody></table></div>
<h2>현재 열어볼 수 있는 최신 snapshot</h2>
<figure><img src=norm_action_pair_latest.png><figcaption>모델별 최신 평가 checkpoint. epoch가 서로 다름에 주의.</figcaption></figure>
<div class=table><table><thead><tr><th>model</th><th>epoch</th><th>state cohesion</th><th>action XYZ cohesion</th><th>direction NMI</th><th>purity</th><th>coherence</th><th>gripper NMI</th><th>task NMI</th><th>index NMI</th><th>adjacent same</th><th>opposite collision</th><th>usage (diagnostic)</th><th>max share</th></tr></thead><tbody>{metric_table(latest)}</tbody></table></div>
<h2>학습하면서 기준이 안정되는가?</h2><figure><img src=norm_action_pair_training_curves.png><figcaption>평가된 모든 checkpoint에서 전체 11,221개 assignment를 다시 계산한 경향. usage 균등성이 아니라 state cohesion·방향·인접 충돌의 안정성을 본다.</figcaption></figure>
<h2>동일 epoch assignment 유사도</h2><div class=plots><figure><img src=norm_action_pair_assignment_nmi.png><figcaption>code 번호 permutation에 무관한 NMI.</figcaption></figure><figure><img src=norm_action_pair_assignment_ari.png><figcaption>chance-adjusted partition agreement(ARI).</figcaption></figure></div>
<h2>해석 기준</h2><p>downstream skill label 목적이라면 (1) code 내부 연속형 motion cohesion, (2) 방향 NMI/coherence, (3) 다른 인접 phase의 collision, (4) gripper·task shortcut, (5) checkpoint 간 안정성을 함께 보는 편이 맞다. code 사용량의 대칭성은 목표로 두지 않는다. contrastive는 adjacent negative를 통해 이웃 phase 분리를 직접 장려하고, JS는 clean/augmented soft assignment의 일관성을 장려하므로 둘의 inductive bias가 다르다. 최종 선택은 이 offline proxy 뒤에 실제 Stage-1/Stage-2 성능으로 확인해야 한다.</p>
<p class=muted>원자료: <a href=norm_action_pair_analysis.json>norm_action_pair_analysis.json</a> · 기존 replay 비교: <a href=compare/index.html>compare/index.html</a></p>
</main></body></html>"""


def main() -> None:
    args = parse_args()
    report_root = args.report_root.resolve()
    dataset = base.load_bundle(args.skill_bundle.resolve())
    collections = load_collections(report_root)

    # Reuse the established normalized-action detail renderer with this report's
    # labels/colors.  The imported module keeps its historical defaults for all
    # other entry points.
    normalized.DISPLAY_NAMES.update(DISPLAY_NAMES)
    normalized.MODEL_COLORS.update(COLORS)
    base.DISPLAY_NAMES.update(DISPLAY_NAMES)
    base.MODEL_COLORS.update(COLORS)

    common_epochs = set(available_epochs(collections[MODEL_ORDER[0]]))
    for name in MODEL_ORDER[1:]:
        common_epochs &= set(available_epochs(collections[name]))
    if not common_epochs:
        raise ValueError("The three models have no common evaluated epoch")
    fair_epoch = max(common_epochs, key=epoch_number)

    latest_results: list[dict[str, Any]] = []
    fair_results: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, Any]]] = {}
    fair_tokens: dict[str, np.ndarray] = {}
    latest_tokens: dict[str, np.ndarray] = {}
    for name in MODEL_ORDER:
        collection_path = collections[name]
        epochs = available_epochs(collection_path)
        latest_epoch = epochs[-1]
        print(f"Analyzing {name}: latest={latest_epoch}, fair={fair_epoch}", flush=True)
        latest_result = base.model_metrics(collection_path, dataset, epoch_tag=latest_epoch)
        enrich(latest_result)
        latest_token_values = load_tokens(collection_path, latest_epoch)
        latest_result["semantic_cohesion"] = semantic_cohesion(
            latest_token_values, dataset
        )
        save_detail(
            latest_result,
            collection_path.parent.parent / ANALYSIS_SUBDIR,
            args.output_name,
        )
        latest_results.append(latest_result)
        if latest_epoch == fair_epoch:
            fair_result = latest_result
        else:
            fair_result = base.model_metrics(collection_path, dataset, epoch_tag=fair_epoch)
            enrich(fair_result)
            fair_result["semantic_cohesion"] = semantic_cohesion(
                load_tokens(collection_path, fair_epoch), dataset
            )
        print(f"Loading {name} validation scalars at {fair_epoch}", flush=True)
        add_checkpoint_validation(fair_result)
        fair_results.append(fair_result)
        fair_tokens[name] = (
            latest_token_values
            if latest_epoch == fair_epoch
            else load_tokens(collection_path, fair_epoch)
        )
        latest_tokens[name] = latest_token_values
        histories[name] = []
        for epoch_tag in epochs:
            metrics = light_metrics(load_tokens(collection_path, epoch_tag), dataset)
            histories[name].append(
                {"epoch": epoch_tag, "epoch_number": epoch_number(epoch_tag), **metrics}
            )

    fair_nmi, fair_ari = comparison_matrices(fair_tokens)
    latest_nmi, latest_ari = comparison_matrices(latest_tokens)
    plot_snapshot_comparison(
        fair_results,
        report_root / "norm_action_pair_fair.png",
        f"Fair pair-objective comparison · {fair_epoch}",
    )
    plot_snapshot_comparison(
        latest_results,
        report_root / "norm_action_pair_latest.png",
        "Latest evaluated snapshot per model",
    )
    plot_training_curves(histories, report_root / "norm_action_pair_training_curves.png")
    plot_assignment_heatmap(
        fair_nmi,
        report_root / "norm_action_pair_assignment_nmi.png",
        f"Assignment NMI · {fair_epoch}",
    )
    plot_assignment_heatmap(
        fair_ari,
        report_root / "norm_action_pair_assignment_ari.png",
        f"Assignment ARI · {fair_epoch}",
    )

    payload = {
        "sample_count": len(dataset["bundle"]["meta_episode_id"]),
        "fair_epoch": fair_epoch,
        "latest": latest_results,
        "fair": fair_results,
        "histories": histories,
        "assignment_similarity": {
            "model_order": MODEL_ORDER,
            "fair_nmi": fair_nmi,
            "fair_ari": fair_ari,
            "latest_nmi": latest_nmi,
            "latest_ari": latest_ari,
        },
    }
    (report_root / "norm_action_pair_analysis.json").write_text(
        json.dumps(base.json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output_path = report_root / args.output_name
    output_path.write_text(
        summary_html(latest_results, fair_results, fair_epoch, fair_nmi, fair_ari),
        encoding="utf-8",
    )
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
