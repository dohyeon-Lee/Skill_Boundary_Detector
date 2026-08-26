#!/usr/bin/env python3
"""Compare complete-skillset FSQ categorization reports on aligned samples."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


DEFAULT_MODELS = (
    "recon w0.1 contrastive",
    "term w0.1 contrastive",
    "term+recon w0.1 contrastive dino",
)

SHORT_LABELS = {
    "recon w0.1 contrastive": "Recon-only",
    "term w0.1 contrastive": "Termination-only",
    "term+recon w0.1 contrastive dino": "Term+Recon DINO",
}

COLORS = {
    "recon w0.1 contrastive": "#4cc9f0",
    "term w0.1 contrastive": "#ffb454",
    "term+recon w0.1 contrastive dino": "#80d99b",
}

FEATURE_LABELS = {
    "start_x": "start x",
    "start_y": "start y",
    "start_z": "start z",
    "mean_x": "mean x",
    "mean_y": "mean y",
    "mean_z": "mean z",
    "disp_x": "disp x",
    "disp_y": "disp y",
    "disp_z": "disp z",
    "net_xyz": "net distance",
    "path_xyz": "path length",
    "rot_net_angle": "net rotation",
    "rot_path_angle": "rotation path",
    "frames": "frames",
    "skill_index": "skill index",
    "skill_order": "skill order",
    "grip_range": "gripper range",
    "grip_path": "gripper path",
}

GROUP_LABELS = {
    "시작 XYZ만": "Start XYZ",
    "절대 XYZ 궤적": "Absolute XYZ",
    "상대 XYZ 모션": "Relative XYZ",
    "회전 특징": "Rotation",
    "시간·순서·gripper": "Time/order/gripper",
    "전체 proprio 특징": "All proprio",
}

AXIS_FEATURES = tuple(FEATURE_LABELS)
METADATA_KEYS = (
    "episode_id",
    "task_id",
    "skill_index",
    "frame_start",
    "frame_end",
    "length",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--epoch", default="epoch2000")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--analysis-subdir", default="code_categorization_analysis")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--robust-code-min", type=int, default=30)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def find_model_roots(report_root: Path, names: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    wanted = set(names)
    for path in report_root.glob("*/metrics/collection.json"):
        document = load_json(path)
        name = document.get("model_name")
        if name in wanted:
            roots[name] = path.parent.parent
    missing = wanted - roots.keys()
    if missing:
        raise FileNotFoundError(f"Missing report folder(s): {sorted(missing)}")
    return roots


def load_axis_associations(path: Path) -> dict[tuple[int, str], float]:
    values: dict[tuple[int, str], float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            raw_value = row["eta_squared"]
            values[(int(row["axis"]), row["feature"])] = (
                float(raw_value) if raw_value else 0.0
            )
    return values


def load_model(
    name: str,
    root: Path,
    epoch: str,
    analysis_subdir: str,
) -> dict[str, Any]:
    analysis_dir = root / analysis_subdir
    summary = load_json(analysis_dir / "summary.json")
    manifest = load_json(root / "checkpoints" / epoch / "metrics" / "manifest.json")
    latent_path = Path(manifest["signature"]["latents_path"])
    with np.load(latent_path, allow_pickle=False) as data:
        latent = {key: data[key].copy() for key in data.files}
    counts = np.asarray(manifest["train_codebook_counts"], dtype=np.int64)
    actual = np.bincount(latent["tokens"], minlength=len(counts))
    if not np.array_equal(counts, actual):
        raise ValueError(f"Code histogram mismatch for {name}")
    if int(summary["sample_count"]) != len(latent["tokens"]):
        raise ValueError(f"Standalone summary sample count mismatch for {name}")
    return {
        "name": name,
        "short": SHORT_LABELS.get(name, name),
        "color": COLORS.get(name, "#9aa8bd"),
        "root": root,
        "analysis_dir": analysis_dir,
        "summary": summary,
        "manifest": manifest,
        "latent": latent,
        "counts": counts,
        "axis": load_axis_associations(analysis_dir / "fsq_axis_associations.csv"),
    }


def check_alignment(models: list[dict[str, Any]]) -> None:
    reference = models[0]
    for model in models[1:]:
        for key in METADATA_KEYS:
            if not np.array_equal(reference["latent"][key], model["latent"][key]):
                raise ValueError(
                    f"Samples are not aligned: {reference['name']} vs {model['name']} ({key})"
                )


def usage_metrics(model: dict[str, Any], robust_min: int) -> dict[str, Any]:
    counts = model["counts"]
    total = int(counts.sum())
    probability = counts[counts > 0] / total
    entropy = float(-(probability * np.log(probability)).sum())
    sorted_counts = np.sort(counts)[::-1]
    largest_code = int(np.argmax(counts))
    return {
        "model_name": model["name"],
        "short_label": model["short"],
        "sample_count": total,
        "used_codes": int(np.count_nonzero(counts)),
        "robust_codes": int(np.count_nonzero(counts >= robust_min)),
        "rare_codes_lt_10": int(np.count_nonzero((counts > 0) & (counts < 10))),
        "largest_code": largest_code,
        "largest_count": int(counts[largest_code]),
        "largest_share": float(counts[largest_code] / total),
        "normalized_entropy": float(entropy / math.log(len(counts))),
        "effective_code_count": float(math.exp(entropy)),
        "top3_share": float(sorted_counts[:3].sum() / total),
        "hhi": float(np.square(counts / total).sum()),
    }


def contingency(left: np.ndarray, right: np.ndarray, n_codes: int) -> np.ndarray:
    matrix = np.zeros((n_codes, n_codes), dtype=np.int64)
    np.add.at(matrix, (left.astype(int), right.astype(int)), 1)
    return matrix


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = spearmanr(left, right).statistic
    return 0.0 if not np.isfinite(value) else float(value)


def pairwise_metrics(
    left: dict[str, Any], right: dict[str, Any]
) -> tuple[dict[str, Any], np.ndarray]:
    left_tokens = left["latent"]["tokens"].astype(int)
    right_tokens = right["latent"]["tokens"].astype(int)
    n_codes = max(len(left["counts"]), len(right["counts"]))
    matrix = contingency(left_tokens, right_tokens, n_codes)
    rows, cols = linear_sum_assignment(-matrix)
    left_coords = left["latent"]["latents"]
    right_coords = right["latent"]["latents"]
    return {
        "model_a": left["name"],
        "model_b": right["name"],
        "label_a": left["short"],
        "label_b": right["short"],
        "nmi": float(normalized_mutual_info_score(left_tokens, right_tokens)),
        "ari": float(adjusted_rand_score(left_tokens, right_tokens)),
        "hungarian_agreement": float(matrix[rows, cols].sum() / len(left_tokens)),
        "exact_token_agreement": float(np.mean(left_tokens == right_tokens)),
        "axis0_exact": float(np.mean(left_coords[:, 0] == right_coords[:, 0])),
        "axis1_exact": float(np.mean(left_coords[:, 1] == right_coords[:, 1])),
        "axis2_exact": float(np.mean(left_coords[:, 2] == right_coords[:, 2])),
        "axis0_spearman": safe_spearman(left_coords[:, 0], right_coords[:, 0]),
        "axis1_spearman": safe_spearman(left_coords[:, 1], right_coords[:, 1]),
        "axis2_spearman": safe_spearman(left_coords[:, 2], right_coords[:, 2]),
    }, matrix


def classification_rows(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        for source in model["summary"]["classification_ablation"]:
            rows.append({"model_name": model["name"], **source})
    return rows


def categorical_rows(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for model in models:
        rows.append({"model_name": model["name"], **model["summary"]["categorical_associations"]})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_code_usage(models: list[dict[str, Any]], output: Path) -> None:
    x = np.arange(max(len(model["counts"]) for model in models))
    width = 0.8 / len(models)
    fig, axis = plt.subplots(figsize=(15, 5.7))
    for index, model in enumerate(models):
        offset = (index - (len(models) - 1) / 2) * width
        axis.bar(
            x + offset,
            model["counts"],
            width=width,
            label=model["short"],
            color=model["color"],
            alpha=0.86,
        )
    axis.set_xticks(x)
    axis.set_xlabel("FSQ code")
    axis.set_ylabel("Complete training-skill count")
    axis.set_title("FSQ code usage on the same 11,221 skills")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_feature_groups(models: list[dict[str, Any]], output: Path) -> None:
    groups = [
        row["feature_group"]
        for row in models[0]["summary"]["classification_ablation"]
    ]
    x = np.arange(len(groups))
    width = 0.8 / len(models)
    fig, axis = plt.subplots(figsize=(13.5, 6.2))
    for index, model in enumerate(models):
        by_group = {
            row["feature_group"]: row
            for row in model["summary"]["classification_ablation"]
        }
        values = [by_group[group]["balanced_accuracy_mean"] for group in groups]
        errors = [by_group[group]["balanced_accuracy_std"] for group in groups]
        offset = (index - (len(models) - 1) / 2) * width
        axis.bar(
            x + offset,
            values,
            yerr=errors,
            width=width,
            capsize=3,
            label=model["short"],
            color=model["color"],
            alpha=0.88,
        )
    axis.set_xticks(x, [GROUP_LABELS.get(group, group) for group in groups], rotation=18, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("3-fold episode-grouped balanced accuracy")
    axis.set_title("FSQ-code predictability from proprio feature groups")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_pairwise_overlap(
    pairs: list[tuple[dict[str, Any], np.ndarray]], output: Path
) -> None:
    fig, axes = plt.subplots(1, len(pairs), figsize=(19, 6.4), constrained_layout=True)
    if len(pairs) == 1:
        axes = [axes]
    image_artist = None
    for axis, (metrics, matrix) in zip(axes, pairs, strict=True):
        denominator = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(
            matrix,
            denominator,
            out=np.zeros_like(matrix, dtype=float),
            where=denominator > 0,
        )
        image_artist = axis.imshow(normalized, vmin=0, vmax=1, cmap="magma", aspect="equal")
        axis.set_title(f"{metrics['label_a']} to {metrics['label_b']}")
        axis.set_xlabel(f"{metrics['label_b']} code")
        axis.set_ylabel(f"{metrics['label_a']} code")
        axis.set_xticks(range(matrix.shape[1]))
        axis.set_yticks(range(matrix.shape[0]))
        axis.tick_params(labelsize=6)
        for row, col in zip(*np.where(normalized >= 0.20), strict=True):
            axis.text(
                col,
                row,
                f"{normalized[row, col]:.0%}",
                ha="center",
                va="center",
                fontsize=5.5,
                color="white" if normalized[row, col] < 0.65 else "black",
            )
    fig.colorbar(image_artist, ax=axes, shrink=0.72, label="Share within source code")
    fig.suptitle("Cross-model code mapping for aligned skills (row-normalized)", fontsize=14)
    fig.savefig(output, dpi=175)
    plt.close(fig)


def plot_axis_roles(models: list[dict[str, Any]], output: Path) -> None:
    values = np.asarray([
        [model["axis"].get((axis, feature), 0.0) for feature in AXIS_FEATURES]
        for model in models
        for axis in range(3)
    ])
    labels = [
        f"{model['short']} · axis {axis}"
        for model in models
        for axis in range(3)
    ]
    fig, axis = plt.subplots(figsize=(16, 8.2))
    artist = axis.imshow(values, cmap="viridis", vmin=0, vmax=max(0.5, float(values.max())))
    axis.set_xticks(
        range(len(AXIS_FEATURES)),
        [FEATURE_LABELS[name] for name in AXIS_FEATURES],
        rotation=35,
        ha="right",
    )
    axis.set_yticks(range(len(labels)), labels)
    for row, col in zip(*np.where(values >= 0.08), strict=True):
        axis.text(col, row, f"{values[row, col]:.2f}", ha="center", va="center", fontsize=7)
    axis.set_title("Association between each FSQ axis and proprio features")
    fig.colorbar(artist, ax=axis, shrink=0.75, label="η²")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def build_html(
    models: list[dict[str, Any]],
    usage: list[dict[str, Any]],
    categorical: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
    pairs: list[tuple[dict[str, Any], np.ndarray]],
    output_dir: Path,
    robust_min: int,
) -> str:
    usage_by_name = {row["model_name"]: row for row in usage}
    categorical_by_name = {row["model_name"]: row for row in categorical}
    class_by_name = {
        (row["model_name"], row["feature_group"]): row for row in classifications
    }
    term = next((model for model in models if model["name"] == "term w0.1 contrastive"), None)
    conclusion = ""
    if term:
        term_usage = usage_by_name[term["name"]]
        term_all = class_by_name[(term["name"], "전체 proprio 특징")]
        conclusion = f"""
        <div class="callout warning">
          <strong>핵심 결론.</strong> Termination-only는 세 모델 중 가장 강한 code collapse를 보인다.
          27개 중 {term_usage['used_codes']}개만 사용하고, 표본 {robust_min}개 이상인 안정적인 code는
          {term_usage['robust_codes']}개뿐이다. 최대 code 하나가 {pct(term_usage['largest_share'])}를 차지한다.
          전체 proprio로 code를 예측한 balanced accuracy도 {fmt(term_all['balanced_accuracy_mean'])}로,
          높은 일반 accuracy {fmt(term_all['accuracy_mean'])}는 주로 불균형한 큰 code를 맞춘 효과다.
        </div>
        """

    model_cards = []
    for model in models:
        row = usage_by_name[model["name"]]
        assoc = categorical_by_name[model["name"]]
        model_cards.append(f"""
        <article class="card" style="border-top-color:{model['color']}">
          <div class="eyebrow">{html.escape(model['short'])}</div>
          <div class="big">{row['used_codes']} / 27 codes</div>
          <div>유효 code(≥{robust_min}): <b>{row['robust_codes']}</b> · 최대 share: <b>{pct(row['largest_share'])}</b></div>
          <div>effective codes: <b>{fmt(row['effective_code_count'], 1)}</b> · task NMI: <b>{fmt(assoc['task_id_nmi'])}</b></div>
        </article>
        """)

    usage_table = table(
        ["모델", "사용 code", f"≥{robust_min}", "1–9개 code", "최대 code", "최대 share", "유효 code 수", "Top-3 share", "entropy"],
        [[
            html.escape(row["short_label"]),
            str(row["used_codes"]),
            str(row["robust_codes"]),
            str(row["rare_codes_lt_10"]),
            f"{row['largest_code']} ({row['largest_count']:,})",
            pct(row["largest_share"]),
            fmt(row["effective_code_count"], 1),
            pct(row["top3_share"]),
            fmt(row["normalized_entropy"]),
        ] for row in usage],
    )

    categorical_table = table(
        ["모델", "scene family", "scene file", "task id", "skill index"],
        [[
            html.escape(SHORT_LABELS.get(row["model_name"], row["model_name"])),
            fmt(row["scene_family_nmi"]),
            fmt(row["scene_file_nmi"]),
            fmt(row["task_id_nmi"]),
            fmt(row["skill_index_nmi"]),
        ] for row in categorical],
    )

    groups = [
        source["feature_group"]
        for source in models[0]["summary"]["classification_ablation"]
    ]
    feature_table = table(
        ["특성군", *[model["short"] for model in models]],
        [[
            html.escape(group),
            *[
                fmt(class_by_name[(model["name"], group)]["balanced_accuracy_mean"])
                + " ± "
                + fmt(class_by_name[(model["name"], group)]["balanced_accuracy_std"])
                for model in models
            ],
        ] for group in groups],
    )

    pair_table = table(
        ["모델 쌍", "NMI", "ARI", "최적 재명명 일치", "code 번호 그대로", "axis 0", "axis 1", "axis 2"],
        [[
            f"{html.escape(row['label_a'])} ↔ {html.escape(row['label_b'])}",
            fmt(row["nmi"]),
            fmt(row["ari"]),
            pct(row["hungarian_agreement"]),
            pct(row["exact_token_agreement"]),
            pct(row["axis0_exact"]),
            pct(row["axis1_exact"]),
            pct(row["axis2_exact"]),
        ] for row, _ in pairs],
    )

    report_links = []
    for model in models:
        target = model["analysis_dir"] / "index.html"
        relative = os.path.relpath(target, output_dir)
        report_links.append(
            f"<a class='report-link' href='{html.escape(relative)}'>{html.escape(model['short'])} 단독 분석</a>"
        )

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FSQ code categorization · 세 모델 비교</title>
  <style>
    :root {{ --bg:#0b111b; --panel:#121b2a; --panel2:#172235; --text:#e8eef9; --muted:#9cabc2; --line:#2a3950; --accent:#79d6ff; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--text); background:radial-gradient(circle at 20% 0,#182844 0,var(--bg) 36%); font:15px/1.65 Inter,system-ui,-apple-system,"Noto Sans KR",sans-serif; }}
    main {{ max-width:1460px; margin:auto; padding:52px 34px 80px; }}
    h1 {{ margin:0 0 8px; font-size:clamp(29px,4vw,48px); letter-spacing:-.035em; }}
    h2 {{ margin:48px 0 12px; font-size:25px; }}
    h3 {{ margin:24px 0 8px; }}
    p {{ color:#c7d2e4; max-width:1050px; }}
    .subtitle,.note {{ color:var(--muted); }}
    .grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; margin:26px 0; }}
    .card {{ background:linear-gradient(150deg,var(--panel2),var(--panel)); border:1px solid var(--line); border-top:4px solid; border-radius:14px; padding:21px; }}
    .eyebrow {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.09em; }}
    .big {{ margin:3px 0 9px; font-size:25px; font-weight:760; }}
    .callout {{ border:1px solid #2a5970; background:#102536; border-radius:13px; padding:17px 20px; margin:24px 0; color:#d9efff; }}
    .warning {{ border-color:#7a5a2a; background:#302716; color:#ffe7bd; }}
    .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:12px; background:var(--panel); }}
    table {{ width:100%; border-collapse:collapse; min-width:800px; }}
    th,td {{ padding:10px 12px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap; }}
    th:first-child,td:first-child {{ text-align:left; position:sticky; left:0; background:#152033; }}
    th {{ color:#aee6ff; font-size:12px; background:#152033; }}
    figure {{ margin:18px 0 34px; padding:12px; background:white; border-radius:12px; }}
    figure img {{ width:100%; display:block; }}
    figcaption {{ color:#46546a; padding:8px 8px 1px; }}
    .links {{ display:flex; flex-wrap:wrap; gap:10px; margin:20px 0; }}
    a {{ color:#8adfff; }} .report-link {{ padding:8px 13px; border:1px solid #315571; border-radius:999px; text-decoration:none; }}
    code {{ color:#bfe9ff; }}
    @media(max-width:900px) {{ main {{ padding:30px 16px 60px; }} .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body><main>
  <div class="eyebrow">complete training skillset · {html.escape(models[0]['summary']['epoch'])}</div>
  <h1>FSQ code categorization: 학습 목표별 비교</h1>
  <p class="subtitle">동일하게 정렬된 전체 training skill {usage[0]['sample_count']:,}개에서 Recon-only, Termination-only, Term+Recon DINO의 code 배정 기준을 비교했다.</p>
  {conclusion}
  <div class="grid">{''.join(model_cards)}</div>
  <div class="links">{''.join(report_links)}</div>

  <h2>1. Codebook 사용 균형</h2>
  <p>code 개수만 보는 대신 분포 entropy, effective code 수, 상위 code 집중도를 함께 봤다. 표본 1–9개의 code는 개별 해석과 cross-validation 지표가 불안정하다.</p>
  {usage_table}
  <figure><img src="code_usage_comparison.png" alt="모델별 code 사용량"><figcaption>같은 code 번호는 FSQ 좌표가 같지만, 모델 간 의미가 반드시 같지는 않다.</figcaption></figure>

  <h2>2. 무엇을 기준으로 code가 갈리는가</h2>
  <h3>범주형 메타데이터와 code의 NMI</h3>
  {categorical_table}
  <p>Termination-only는 task/scene 연관이 확연히 약한 반면 skill index 연관은 상대적으로 유지된다. 즉 task 정체성보다는 episode 내 진행 단계와 종료 판단에 유용한 상태 구간으로 묶이는 경향이 더 강하다. 다만 이는 상관관계이며 학습 목적의 인과 효과를 단독으로 증명하지는 않는다.</p>

  <h3>proprio 특성군의 code 예측력</h3>
  <p>아래는 episode 단위로 분리한 3-fold balanced accuracy다. class 불균형의 영향을 줄이므로 Termination-only처럼 code collapse가 있는 모델 비교에 일반 accuracy보다 적합하다.</p>
  {feature_table}
  <figure><img src="feature_group_comparison.png" alt="특성군별 balanced accuracy"><figcaption>절대 XYZ가 세 모델 모두에서 가장 강하며, Termination-only는 모든 특성군에서 code 간 안정적인 분리도가 낮다.</figcaption></figure>
  <figure><img src="axis_role_comparison.png" alt="FSQ 축별 proprio 연관"><figcaption>η²는 각 FSQ 축의 {-1,0,1} level 사이에 각 특징 평균이 얼마나 다르게 나타나는지 보여준다.</figcaption></figure>

  <h2>3. 동일 skill의 code 배정이 얼마나 같은가</h2>
  <p>NMI/ARI는 code 번호 재명명에 무관한 partition 유사도다. “최적 재명명 일치”는 Hungarian matching으로 code label을 가장 잘 대응시킨 뒤의 일치율이고, axis 값은 FSQ 각 좌표 level이 그대로 같은 비율이다.</p>
  {pair_table}
  <figure><img src="pairwise_code_overlap.png" alt="모델 쌍별 code overlap"><figcaption>각 행은 source code 안의 skill이 상대 모델에서 어느 code로 이동했는지를 합계 100%로 정규화했다.</figcaption></figure>

  <div class="callout">
    <strong>해석상 주의.</strong> 이 비교는 objective만 완벽히 통제한 ablation이 아니다. Termination-only는 tuned ResNet 설정이고, Term+Recon DINO는 DINO vision backbone까지 함께 달라진다. Recon-only에는 활성 vision reconstruction target이 없다. 따라서 Termination-only 대 DINO 차이는 loss와 vision 표현의 복합 차이로 읽어야 한다.
  </div>
  <p class="note">원자료: <a href="model_metrics.csv">model_metrics.csv</a> · <a href="categorical_metrics.csv">categorical_metrics.csv</a> · <a href="feature_group_metrics.csv">feature_group_metrics.csv</a> · <a href="pairwise_metrics.csv">pairwise_metrics.csv</a> · <a href="comparison_summary.json">comparison_summary.json</a></p>
</main></body></html>
"""


def main() -> None:
    args = parse_args()
    report_root = args.report_root.resolve()
    output_dir = (args.output_dir or report_root / "compare" / "code_categorization_comparison").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    roots = find_model_roots(report_root, args.models)
    models = [
        load_model(name, roots[name], args.epoch, args.analysis_subdir)
        for name in args.models
    ]
    check_alignment(models)

    usage = [usage_metrics(model, args.robust_code_min) for model in models]
    categorical = categorical_rows(models)
    classifications = classification_rows(models)
    pairs = [pairwise_metrics(left, right) for left, right in combinations(models, 2)]

    write_csv(output_dir / "model_metrics.csv", usage)
    write_csv(output_dir / "categorical_metrics.csv", categorical)
    write_csv(output_dir / "feature_group_metrics.csv", classifications)
    write_csv(output_dir / "pairwise_metrics.csv", [row for row, _ in pairs])
    plot_code_usage(models, output_dir / "code_usage_comparison.png")
    plot_feature_groups(models, output_dir / "feature_group_comparison.png")
    plot_pairwise_overlap(pairs, output_dir / "pairwise_code_overlap.png")
    plot_axis_roles(models, output_dir / "axis_role_comparison.png")

    summary = {
        "epoch": args.epoch,
        "scope": "complete training skillset; aligned samples",
        "robust_code_min": args.robust_code_min,
        "models": usage,
        "categorical_associations": categorical,
        "classification_ablation": classifications,
        "pairwise": [row for row, _ in pairs],
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    (output_dir / "index.html").write_text(
        build_html(
            models,
            usage,
            categorical,
            classifications,
            pairs,
            output_dir,
            args.robust_code_min,
        )
    )
    print(f"Wrote comparison report: {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
