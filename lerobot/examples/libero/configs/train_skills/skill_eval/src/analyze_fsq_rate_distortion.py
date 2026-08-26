#!/usr/bin/env python3
"""Compare trained FSQ assignments with their centroids and unconstrained k-means."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans

from visualize_fsq_spline_io import (
    DIM_NAMES,
    cfg_value,
    load_bundle,
    load_fsq_model,
    load_stats,
    resolve_checkpoint,
    scalar_text,
    spline_encode,
)
from FSQ import episode_grouped_train_val_ids


GROUPS = {
    "all": np.arange(8),
    "XYZ": np.arange(3),
    "rotvec": np.arange(3, 6),
    "gripper": np.arange(6, 8),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-bundle", type=Path, required=True)
    parser.add_argument("--start-run", type=Path, required=True)
    parser.add_argument("--zero-run", type=Path, required=True)
    parser.add_argument("--start-checkpoint", default="latest")
    parser.add_argument("--zero-checkpoint", default="latest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kmeans-n-init", type=int, default=10)
    return parser.parse_args()


def normalized_targets(
    bundle: dict[str, np.ndarray],
    *,
    mode: str,
    minimum: np.ndarray,
    maximum: np.ndarray,
    n_control: int,
    degree: int,
) -> np.ndarray:
    lengths = bundle["states_len"].astype(np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths[:-1])))
    states_cat = bundle["states_cat"]
    targets = np.empty((len(lengths), n_control, len(DIM_NAMES)), dtype=np.float32)
    for index, (offset, length) in enumerate(zip(offsets, lengths, strict=True)):
        states = states_cat[int(offset) : int(offset + length)].astype(np.float32, copy=False)
        grid, _ = spline_encode(states, n_control, degree, input_mode=mode)
        targets[index] = 2.0 * (grid - minimum) / (maximum - minimum + 1e-8) - 1.0
        if (index + 1) % 2000 == 0:
            print(f"[{mode}] prepared {index + 1:,}/{len(lengths):,} targets", flush=True)
    return targets


def decoder_prototypes(model: Any, codebook_size: int) -> np.ndarray:
    if getattr(model.reconstructor, "state_dim", 0) != 0:
        raise ValueError("This analysis expects a code-only oneshot reconstructor.")
    with torch.inference_mode():
        codes = torch.arange(codebook_size, dtype=torch.long)
        z_norm = model.fsq.code_to_normalized(codes)
        predicted, _ = model.reconstructor(z_norm, start_state=None)
    return predicted.cpu().numpy().astype(np.float32)


def restore_bundle_assignment_order(
    saved_assignments: np.ndarray,
    bundle: dict[str, np.ndarray],
    cfg: Any,
    saved_fingerprint: str | None,
) -> np.ndarray:
    """Undo the validation-then-training order used by checkpoint diagnostics."""
    metadata = [
        {
            "task_id": int(task_id),
            "episode_id": int(episode_id),
            "skill_index": int(skill_index),
        }
        for task_id, episode_id, skill_index in zip(
            bundle["meta_task_id"],
            bundle["meta_episode_id"],
            bundle["meta_skill_index"],
            strict=True,
        )
    ]
    n_val = max(1, int(len(metadata) * float(cfg_value(cfg, "val_split"))))
    if str(cfg_value(cfg, "pair_loss")) == "contrastive":
        train_ids, val_ids = episode_grouped_train_val_ids(metadata, n_val)
    else:
        def identity_hash(index: int) -> int:
            item = metadata[index]
            identity = f"{item['episode_id']}_{item['skill_index']}"
            return int(hashlib.sha1(identity.encode()).hexdigest(), 16)

        order = sorted(range(len(metadata)), key=identity_hash)
        val_ids, train_ids = order[:n_val], order[n_val:]
    stored_order = [*val_ids, *train_ids]
    identity = ",".join(
        f"{metadata[index]['episode_id']}_{metadata[index]['skill_index']}"
        for index in stored_order
    )
    fingerprint = hashlib.sha1(identity.encode()).hexdigest()[:12]
    if saved_fingerprint is not None and fingerprint != saved_fingerprint:
        raise ValueError(
            "Checkpoint assignment fingerprint does not match reconstructed order: "
            f"checkpoint={saved_fingerprint}, reconstructed={fingerprint}"
        )
    restored = np.empty_like(saved_assignments)
    restored[np.asarray(stored_order, dtype=np.int64)] = saved_assignments
    return restored


def group_mse(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        name: float(np.mean(np.square(predicted[..., dims] - target[..., dims])))
        for name, dims in GROUPS.items()
    }


def occupancy_summary(assignments: np.ndarray, codebook_size: int) -> dict[str, float]:
    counts = np.bincount(assignments, minlength=codebook_size)
    probability = counts[counts > 0] / counts.sum()
    entropy = float(-(probability * np.log(probability)).sum())
    return {
        "active": int(np.count_nonzero(counts)),
        "top_count": int(counts.max()),
        "top_pct": float(100.0 * counts.max() / counts.sum()),
        "effective_codes": float(np.exp(entropy)),
    }


def analyze_mode(
    *,
    label: str,
    bundle: dict[str, np.ndarray],
    run_dir: Path,
    requested_checkpoint: str,
    n_init: int,
) -> dict[str, Any]:
    checkpoint = resolve_checkpoint(run_dir, requested_checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assignments_tensor = payload.get("code_assignments")
    if assignments_tensor is None:
        raise ValueError(f"{checkpoint} has no full-dataset code_assignments")
    saved_assignments = assignments_tensor.cpu().numpy().astype(np.int64)
    if len(saved_assignments) != len(bundle["states_len"]):
        raise ValueError(
            f"Assignment count {len(saved_assignments)} != skill count {len(bundle['states_len'])}"
        )

    model, cfg = load_fsq_model(checkpoint, device="cpu")
    model.eval()
    assignments = restore_bundle_assignment_order(
        saved_assignments,
        bundle,
        cfg,
        payload.get("code_assignments_fingerprint"),
    )
    stats = load_stats(run_dir)
    mode = scalar_text(stats["reconstructor_output_mode"])
    n_control = int(cfg_value(cfg, "n_control"))
    degree = int(cfg_value(cfg, "spline_degree"))
    codebook_size = int(model.fsq.codebook_size)
    targets = normalized_targets(
        bundle,
        mode=mode,
        minimum=stats["reconstructor_min"],
        maximum=stats["reconstructor_max"],
        n_control=n_control,
        degree=degree,
    )
    flat = targets.reshape(len(targets), -1)
    prototypes = decoder_prototypes(model, codebook_size)
    model_prediction = prototypes[assignments]

    counts = np.bincount(assignments, minlength=codebook_size)
    centroids = np.full_like(prototypes, np.nan)
    for code in np.flatnonzero(counts):
        centroids[code] = targets[assignments == code].mean(axis=0)
    centroid_prediction = centroids[assignments]

    print(f"[{label}] fitting unconstrained K={codebook_size} k-means", flush=True)
    kmeans = KMeans(
        n_clusters=codebook_size,
        n_init=n_init,
        max_iter=300,
        algorithm="lloyd",
        random_state=20260825,
        verbose=0,
    ).fit(flat)
    kmeans_assignments = kmeans.labels_.astype(np.int64)
    kmeans_prediction = kmeans.cluster_centers_[kmeans_assignments].reshape(targets.shape)

    rows: list[dict[str, Any]] = []
    for code in range(codebook_size):
        count = int(counts[code])
        if count == 0:
            rows.append({"code": code, "count": 0})
            continue
        members = targets[assignments == code]
        within_mse = float(np.mean(np.square(members - centroids[code])))
        model_mse = float(np.mean(np.square(members - prototypes[code])))
        prototype_centroid_mse = float(np.mean(np.square(prototypes[code] - centroids[code])))
        split_mse = within_mse
        if count >= 2:
            member_flat = members.reshape(count, -1)
            split = KMeans(
                n_clusters=2,
                n_init=min(n_init, 10),
                max_iter=200,
                algorithm="lloyd",
                random_state=20260825 + code,
            ).fit(member_flat)
            split_mse = float(split.inertia_ / member_flat.size)
        split_gain = within_mse - split_mse
        rows.append(
            {
                "code": code,
                "count": count,
                "pct": 100.0 * count / len(targets),
                "within_mse": within_mse,
                "model_mse": model_mse,
                "prototype_centroid_mse": prototype_centroid_mse,
                "split_mse": split_mse,
                "split_gain": split_gain,
                "global_split_gain": split_gain * count / len(targets),
            }
        )

    result = {
        "label": label,
        "run_dir": str(run_dir),
        "checkpoint": checkpoint.name,
        "assignment_epoch": int(payload.get("code_assignments_epoch", -1)),
        "mode": mode,
        "n_samples": len(targets),
        "codebook_size": codebook_size,
        "metrics": {
            "model": group_mse(model_prediction, targets),
            "centroid": group_mse(centroid_prediction, targets),
            "kmeans": group_mse(kmeans_prediction, targets),
        },
        "trained_occupancy": occupancy_summary(assignments, codebook_size),
        "kmeans_occupancy": occupancy_summary(kmeans_assignments, codebook_size),
        "trained_counts": counts.tolist(),
        "kmeans_counts": np.bincount(
            kmeans_assignments, minlength=codebook_size
        ).tolist(),
        "codes": rows,
        "kmeans_inertia": float(kmeans.inertia_),
        "kmeans_iterations": int(kmeans.n_iter_),
    }
    result["decoder_gap"] = {
        group: result["metrics"]["model"][group] - result["metrics"]["centroid"][group]
        for group in GROUPS
    }
    result["assignment_gap"] = {
        group: result["metrics"]["centroid"][group] - result["metrics"]["kmeans"][group]
        for group in GROUPS
    }
    return result


def plot_mode(result: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    methods = ("model", "centroid", "kmeans")
    labels = ("FSQ model", "trained assignment\ncentroid", "unconstrained\nk-means K=27")
    colors = ("#d62828", "#f4a261", "#2a9d8f")

    all_values = [result["metrics"][method]["all"] for method in methods]
    axes[0, 0].bar(labels, all_values, color=colors)
    axes[0, 0].set_title("Normalized reconstruction MSE (30×8)")
    axes[0, 0].set_ylabel("MSE (lower is better)")
    axes[0, 0].grid(axis="y", alpha=0.25)
    for index, value in enumerate(all_values):
        axes[0, 0].text(index, value, f"{value:.6f}", ha="center", va="bottom")

    x = np.arange(3)
    width = 0.25
    for offset, (method, color, label) in enumerate(zip(methods, colors, labels, strict=True)):
        values = [result["metrics"][method][group] for group in ("XYZ", "rotvec", "gripper")]
        axes[0, 1].bar(x + (offset - 1) * width, values, width, color=color, label=label)
    axes[0, 1].set_xticks(x, ("XYZ", "rotvec", "gripper"))
    axes[0, 1].set_title("MSE by dimension group")
    axes[0, 1].set_ylabel("normalized MSE")
    axes[0, 1].grid(axis="y", alpha=0.25)
    axes[0, 1].legend(fontsize=8)

    trained = np.sort(np.asarray(result["trained_counts"], dtype=np.float64))[::-1]
    baseline = np.sort(np.asarray(result["kmeans_counts"], dtype=np.float64))[::-1]
    trained = 100.0 * trained / result["n_samples"]
    baseline = 100.0 * baseline / result["n_samples"]
    ranks = np.arange(1, result["codebook_size"] + 1)
    axes[1, 0].plot(ranks, trained, marker="o", label="trained FSQ assignment")
    axes[1, 0].plot(ranks, baseline, marker="o", label="unconstrained k-means")
    axes[1, 0].set_title("Occupancy rank curve — no uniformity constraint")
    axes[1, 0].set_xlabel("occupancy rank")
    axes[1, 0].set_ylabel("dataset share (%)")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    active = [row for row in result["codes"] if row["count"] > 0]
    occupancy = np.asarray([row["pct"] for row in active])
    within = np.asarray([row["within_mse"] for row in active])
    split_gain = np.asarray([row["global_split_gain"] for row in active])
    sizes = 45 + 400 * split_gain / max(float(split_gain.max()), 1e-12)
    axes[1, 1].scatter(occupancy, within, s=sizes, c=split_gain, cmap="viridis", alpha=0.8)
    for row in active:
        axes[1, 1].annotate(
            str(row["code"]),
            (row["pct"], row["within_mse"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1, 1].set_title("trained code: occupancy vs within-code distortion\n(size/color = global 2-way split gain)")
    axes[1, 1].set_xlabel("dataset share (%)")
    axes[1, 1].set_ylabel("centroid MSE within code")
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle(
        f"{result['label']} · rate–distortion decomposition · {result['checkpoint']}",
        fontsize=16,
    )
    fig.savefig(path, dpi=175)
    plt.close(fig)


def metric_table(result: dict[str, Any]) -> str:
    rows = []
    names = {
        "model": "현재 FSQ model output",
        "centroid": "현재 assignment의 최적 centroid",
        "kmeans": "unconstrained k-means (K=27)",
    }
    for method in ("model", "centroid", "kmeans"):
        values = result["metrics"][method]
        rows.append(
            "<tr>"
            f"<th>{html.escape(names[method])}</th>"
            f"<td>{values['all']:.7f}</td><td>{values['XYZ']:.7f}</td>"
            f"<td>{values['rotvec']:.7f}</td><td>{values['gripper']:.7f}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>comparison</th><th>all 30×8</th><th>XYZ</th>"
        "<th>rotvec</th><th>gripper</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def code_table(result: dict[str, Any]) -> str:
    rows = []
    ordered = sorted(result["codes"], key=lambda row: row.get("global_split_gain", -1), reverse=True)
    for row in ordered:
        if row["count"] == 0:
            rows.append(
                f"<tr class='inactive'><td>{row['code']}</td><td>0</td>"
                "<td colspan='6'>unused</td></tr>"
            )
            continue
        rows.append(
            "<tr>"
            f"<td>{row['code']}</td><td>{row['count']:,}</td><td>{row['pct']:.2f}%</td>"
            f"<td>{row['model_mse']:.7f}</td><td>{row['within_mse']:.7f}</td>"
            f"<td>{row['prototype_centroid_mse']:.7f}</td>"
            f"<td>{row['split_gain']:.7f}</td><td>{row['global_split_gain']:.7f}</td>"
            "</tr>"
        )
    return (
        "<div class='scroll'><table><thead><tr><th>code</th><th>N</th><th>share</th>"
        "<th>model MSE</th><th>centroid MSE</th><th>prototype↔centroid</th>"
        "<th>local split gain</th><th>global split gain</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def conclusion(result: dict[str, Any]) -> str:
    model = result["metrics"]["model"]["all"]
    centroid = result["metrics"]["centroid"]["all"]
    kmeans = result["metrics"]["kmeans"]["all"]
    total_improvable = max(model - kmeans, 1e-12)
    decoder_fraction = 100.0 * max(model - centroid, 0.0) / total_improvable
    assignment_fraction = 100.0 * max(centroid - kmeans, 0.0) / total_improvable
    best_split = max(
        (row for row in result["codes"] if row["count"] > 0),
        key=lambda row: row["global_split_gain"],
    )
    return (
        f"K=27 k-means까지 줄일 수 있는 gap 중 decoder가 차지하는 비율은 "
        f"<b>{decoder_fraction:.1f}%</b>, 현재 assignment가 차지하는 비율은 "
        f"<b>{assignment_fraction:.1f}%</b>다. 가장 split 이득이 큰 code는 "
        f"<b>{best_split['code']}</b> (N={best_split['count']:,}, global MSE 감소 "
        f"{best_split['global_split_gain']:.7f})다."
    )


def build_html(results: list[dict[str, Any]]) -> str:
    sections = []
    for result in results:
        trained = result["trained_occupancy"]
        baseline = result["kmeans_occupancy"]
        sections.append(
            f"""
            <section class="panel">
              <div class="eyebrow">{html.escape(result['mode'])}</div>
              <h2>{html.escape(result['label'])}</h2>
              <p class="source">{html.escape(result['run_dir'])}<br>{html.escape(result['checkpoint'])} · assignment epoch {result['assignment_epoch']}</p>
              <div class="chips">
                <span>trained active {trained['active']}/{result['codebook_size']}</span>
                <span>trained top {trained['top_pct']:.2f}% ({trained['top_count']:,})</span>
                <span>trained effective codes {trained['effective_codes']:.2f}</span>
                <span>k-means top {baseline['top_pct']:.2f}%</span>
                <span>k-means effective codes {baseline['effective_codes']:.2f}</span>
              </div>
              <div class="callout">{conclusion(result)}</div>
              {metric_table(result)}
              <figure><img src="rate_distortion_{result['label']}.png" alt="rate distortion plots for {html.escape(result['label'])}"></figure>
              <details><summary>code별 distortion과 split gain</summary>{code_table(result)}</details>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FSQ rate–distortion: model vs centroid vs k-means</title>
<style>
:root{{--bg:#07111d;--panel:#111f31;--line:#2d425b;--text:#edf5ff;--muted:#9fb1c8;--cyan:#6bdcff;--amber:#ffc66d}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 15% 0,#173957,var(--bg) 38%);color:var(--text);font:15px/1.65 Inter,system-ui,-apple-system,"Noto Sans KR",sans-serif}}main{{max-width:1450px;margin:auto;padding:52px 28px 90px}}h1{{font-size:clamp(32px,5vw,58px);line-height:1.08;letter-spacing:-.045em;margin:5px 0 16px}}h2{{font-size:28px;margin:3px 0}}p{{color:#cad7e8}}.lead{{max-width:1050px;font-size:17px}}.eyebrow{{color:var(--cyan);text-transform:uppercase;letter-spacing:.12em;font-size:12px}}.panel{{margin:38px 0 70px;padding:24px;background:linear-gradient(150deg,#172941,var(--panel));border:1px solid var(--line);border-radius:18px}}.source{{font:12px/1.5 ui-monospace,monospace;color:var(--muted);overflow-wrap:anywhere}}.chips{{display:flex;gap:8px;flex-wrap:wrap;margin:17px 0}}.chips span{{padding:6px 11px;border-radius:999px;background:#223a55;color:#c6eaff}}.callout{{margin:18px 0;padding:17px 20px;border:1px solid #39718b;background:#0d293d;border-radius:13px;color:#e0f3ff}}figure{{margin:22px 0;padding:10px;background:white;border-radius:12px}}figure img{{display:block;width:100%}}table{{width:100%;border-collapse:collapse;background:#0c1928}}th,td{{padding:9px 11px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}thead th{{background:#1b3049;color:#bcecff;position:sticky;top:0}}.inactive{{color:#73869d}}details{{margin-top:20px;padding:13px;border:1px solid var(--line);border-radius:11px;background:#0c1725}}summary{{cursor:pointer;color:var(--amber);font-weight:700}}.scroll{{overflow:auto;margin-top:12px}}code{{color:#bcecff}}
@media(max-width:800px){{main{{padding:30px 13px 60px}}.panel{{padding:15px}}}}
</style></head><body><main>
<div class="eyebrow">full dataset · no occupancy constraint</div>
<h1>FSQ rate–distortion decomposition</h1>
<p class="lead">전체 11,221개 skill의 실제 normalized 30×8 reconstruction target을 사용했다. 현재 assignment를 유지한 최적 centroid와 occupancy를 전혀 제한하지 않은 K=27 k-means를 비교한다. 따라서 k-means의 개선은 “uniform code 사용” 효과가 아니라 같은 code 수에서 순수하게 reconstruction distortion을 더 잘 줄인 결과다.</p>
<div class="callout"><b>해석:</b> model−centroid는 decoder gap, centroid−k-means는 encoder/FSQ assignment gap, k-means distortion은 27개 category만 사용할 때 남는 rate limit이다. code별 2-way split gain은 그 code 하나를 둘로 나눴을 때 줄어드는 reconstruction error다.</div>
{''.join(sections)}
</main></body></html>"""


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(args.skill_bundle.resolve())
    results = [
        analyze_mode(
            label="start",
            bundle=bundle,
            run_dir=args.start_run.resolve(),
            requested_checkpoint=args.start_checkpoint,
            n_init=args.kmeans_n_init,
        ),
        analyze_mode(
            label="zero",
            bundle=bundle,
            run_dir=args.zero_run.resolve(),
            requested_checkpoint=args.zero_checkpoint,
            n_init=args.kmeans_n_init,
        ),
    ]
    for result in results:
        plot_mode(result, output_dir / f"rate_distortion_{result['label']}.png")
    (output_dir / "rate_distortion_metrics.json").write_text(
        json.dumps(json_ready(results), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = output_dir / "rate_distortion_comparison.html"
    report.write_text(build_html(results), encoding="utf-8")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
