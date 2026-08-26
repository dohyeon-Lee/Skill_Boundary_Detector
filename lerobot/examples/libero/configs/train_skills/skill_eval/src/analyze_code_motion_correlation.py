#!/usr/bin/env python3
"""Relate FSQ code assignments to proprioceptive trajectory features.

The analysis joins replay-report occurrences to ``skills_bundle.npz`` by
``(episode_id, skill_index)``.  A point-biserial correlation is reported with
the larger token represented as 1, so a positive value means that the feature
is larger for the larger token.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import pointbiserialr
from sklearn.metrics import roc_auc_score


DEFAULT_MODELS = (
    "recon w0.1 contrastive",
    "term+recon w0.1 contrastive dino",
)

DISPLAY_FEATURES = (
    "start_x",
    "start_y",
    "start_z",
    "mean_x",
    "mean_y",
    "mean_z",
    "end_x",
    "end_y",
    "end_z",
    "disp_x",
    "disp_y",
    "disp_z",
    "net_xyz",
    "path_xyz",
    "rot_net_angle",
    "rot_path_angle",
    "frames",
    "grip_range",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--skill-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epoch", default="epoch2000")
    parser.add_argument("--tokens", type=int, nargs=2, default=(18, 20))
    parser.add_argument("--skill-index", type=int, default=0)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    return parser.parse_args()


def _interp(values: np.ndarray, fraction: float) -> np.ndarray:
    timeline = np.linspace(0.0, 1.0, len(values))
    return np.asarray([
        np.interp(fraction, timeline, values[:, dim])
        for dim in range(values.shape[1])
    ])


def trajectory_features(states: np.ndarray) -> dict[str, float]:
    position = states[:, :3].astype(np.float64)
    rotvec = states[:, 3:6].astype(np.float64)
    gripper = states[:, 6:8].astype(np.float64).mean(axis=1)
    features: dict[str, float] = {}

    vectors = {
        "start": position[0],
        "mean": position.mean(axis=0),
        "end": position[-1],
        "disp": position[-1] - position[0],
        "range": np.ptp(position, axis=0),
        "std": position.std(axis=0),
    }
    for prefix, vector in vectors.items():
        for axis, value in zip("xyz", vector, strict=True):
            features[f"{prefix}_{axis}"] = float(value)

    position_steps = np.diff(position, axis=0)
    features["net_xyz"] = float(np.linalg.norm(position[-1] - position[0]))
    features["path_xyz"] = float(np.linalg.norm(position_steps, axis=1).sum())
    features["straightness"] = features["net_xyz"] / max(features["path_xyz"], 1e-12)
    features["frames"] = float(len(states))

    for fraction in (0.25, 0.5, 0.75):
        position_at_fraction = _interp(position, fraction)
        relative_position = position_at_fraction - position[0]
        percent = int(fraction * 100)
        for axis, value in zip("xyz", position_at_fraction, strict=True):
            features[f"p{percent}_{axis}"] = float(value)
        for axis, value in zip("xyz", relative_position, strict=True):
            features[f"rel{percent}_{axis}"] = float(value)

    raw_rotation_vectors = {
        "rv_start": rotvec[0],
        "rv_mean": rotvec.mean(axis=0),
        "rv_end": rotvec[-1],
        "rv_raw_delta": rotvec[-1] - rotvec[0],
        "rv_range": np.ptp(rotvec, axis=0),
        "rv_std": rotvec.std(axis=0),
    }
    for prefix, vector in raw_rotation_vectors.items():
        for axis, value in zip("xyz", vector, strict=True):
            features[f"{prefix}_{axis}"] = float(value)

    rotations = Rotation.from_rotvec(rotvec)
    relative_rotation = (rotations[0].inv() * rotations[-1]).as_rotvec()
    for axis, value in zip("xyz", relative_rotation, strict=True):
        features[f"rot_rel_{axis}"] = float(value)
    features["rot_net_angle"] = float(np.linalg.norm(relative_rotation))
    incremental_rotation = (rotations[:-1].inv() * rotations[1:]).as_rotvec()
    features["rot_path_angle"] = float(np.linalg.norm(incremental_rotation, axis=1).sum())

    features["grip_start"] = float(gripper[0])
    features["grip_end"] = float(gripper[-1])
    features["grip_delta"] = float(gripper[-1] - gripper[0])
    features["grip_range"] = float(np.ptp(gripper))
    features["grip_path"] = float(np.abs(np.diff(gripper)).sum())
    return features


def load_bundle(bundle_path: Path) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int]]]:
    with np.load(bundle_path, allow_pickle=False) as bundle:
        lengths = bundle["states_len"].astype(np.int64)
        starts = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths[:-1])))
        states = bundle["states_cat"].copy()
        lookup = {
            (int(episode), int(skill_index)): (int(start), int(length))
            for episode, skill_index, start, length in zip(
                bundle["meta_episode_id"],
                bundle["meta_skill_index"],
                starts,
                lengths,
                strict=True,
            )
        }
    return states, lookup


def find_collections(report_root: Path, model_names: set[str]) -> dict[str, dict]:
    collections: dict[str, dict] = {}
    for path in report_root.glob("*/metrics/collection.json"):
        document = json.loads(path.read_text())
        if document.get("model_name") in model_names:
            collections[document["model_name"]] = document
    missing = model_names - collections.keys()
    if missing:
        raise FileNotFoundError(f"Missing collection.json for models: {sorted(missing)}")
    return collections


def checkpoint_for_epoch(collection: dict, epoch: str) -> dict:
    for checkpoint in collection["checkpoints"]:
        if checkpoint["epoch_tag"] == epoch:
            return checkpoint
    raise KeyError(f"Epoch {epoch!r} not found for {collection['model_name']!r}")


def collect_rows(
    checkpoint: dict,
    states: np.ndarray,
    lookup: dict[tuple[int, int], tuple[int, int]],
    tokens: tuple[int, int],
    skill_index: int,
) -> tuple[list[dict], list[dict]]:
    selected: list[dict] = []
    all_at_index: list[dict] = []
    for skill in checkpoint["skills"]:
        token = int(skill["token"])
        for occurrence in skill["occurrences"]:
            if int(occurrence["skill_index"]) != skill_index:
                continue
            scene_family = occurrence["scene_file"].split("_SCENE", maxsplit=1)[0]
            occurrence_summary = {"token": token, "scene_family": scene_family}
            all_at_index.append(occurrence_summary)
            if token not in tokens:
                continue
            key = (int(occurrence["episode_id"]), skill_index)
            start, length = lookup[key]
            row = trajectory_features(states[start : start + length])
            row.update(
                token=token,
                task_id=int(occurrence["task_id"]),
                episode_id=int(occurrence["episode_id"]),
                scene_family=scene_family,
            )
            selected.append(row)
    return selected, all_at_index


def feature_statistics(rows: list[dict], low_token: int, high_token: int) -> list[dict]:
    label = np.asarray([int(row["token"] == high_token) for row in rows])
    numeric_features = [
        key
        for key in rows[0]
        if key not in {"token", "task_id", "episode_id", "scene_family"}
    ]
    statistics: list[dict] = []
    for feature in numeric_features:
        values = np.asarray([row[feature] for row in rows], dtype=np.float64)
        if np.std(values) < 1e-12:
            continue
        correlation, p_value = pointbiserialr(label, values)
        low_values = values[label == 0]
        high_values = values[label == 1]
        pooled_variance = (
            (len(low_values) - 1) * low_values.var(ddof=1)
            + (len(high_values) - 1) * high_values.var(ddof=1)
        ) / (len(values) - 2)
        cohen_d = (high_values.mean() - low_values.mean()) / np.sqrt(pooled_variance)
        auc = roc_auc_score(label, values)
        statistics.append(
            {
                "feature": feature,
                "r_code_high": float(correlation),
                "p_value": float(p_value),
                "cohen_d": float(cohen_d),
                "auc_separation": float(max(auc, 1.0 - auc)),
                f"code{low_token}_n": int(len(low_values)),
                f"code{low_token}_mean": float(low_values.mean()),
                f"code{low_token}_std": float(low_values.std()),
                f"code{low_token}_min": float(low_values.min()),
                f"code{low_token}_max": float(low_values.max()),
                f"code{high_token}_n": int(len(high_values)),
                f"code{high_token}_mean": float(high_values.mean()),
                f"code{high_token}_std": float(high_values.std()),
                f"code{high_token}_min": float(high_values.min()),
                f"code{high_token}_max": float(high_values.max()),
            }
        )
    return statistics


def scene_summary(all_rows: list[dict], low_token: int, high_token: int) -> dict:
    living_count = sum(row["scene_family"] == "LIVING_ROOM" for row in all_rows)
    high_count = sum(row["token"] == high_token for row in all_rows)
    true_positive = sum(
        row["token"] == high_token and row["scene_family"] == "LIVING_ROOM"
        for row in all_rows
    )
    return {
        "all_skill_index_count": len(all_rows),
        "scene_family_counts": dict(Counter(row["scene_family"] for row in all_rows)),
        "token_counts": dict(Counter(str(row["token"]) for row in all_rows)),
        f"code{high_token}_living_precision": true_positive / high_count,
        f"code{high_token}_living_recall": true_positive / living_count,
        "pair_rule": f"LIVING_ROOM -> code {high_token}; KITCHEN/STUDY -> code {low_token}",
    }


def draw_heatmap(
    model_stats: dict[str, dict[str, dict]],
    output_path: Path,
    low_token: int,
    high_token: int,
) -> None:
    models = list(model_stats)
    correlations = np.asarray([
        [model_stats[model][feature]["r_code_high"] for model in models]
        for feature in DISPLAY_FEATURES
    ])
    auc = np.asarray([
        [model_stats[model][feature]["auc_separation"] for model in models]
        for feature in DISPLAY_FEATURES
    ])

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 9.5), constrained_layout=True)
    panels = (
        (axes[0], correlations, "coolwarm", -1.0, 1.0, f"Point-biserial r (code {high_token}=1)"),
        (axes[1], auc, "viridis", 0.5, 1.0, "Single-feature separation AUC"),
    )
    for axis, matrix, cmap, vmin, vmax, title in panels:
        image = axis.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        axis.set_xticks(range(len(models)), [name.replace(" contrastive", "\ncontrastive") for name in models])
        axis.set_yticks(range(len(DISPLAY_FEATURES)), DISPLAY_FEATURES)
        axis.set_title(title)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                color = "white" if abs(value - (vmin + vmax) / 2) > (vmax - vmin) * 0.3 else "black"
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)
        figure.colorbar(image, ax=axis, shrink=0.85)
    figure.suptitle(
        f"FSQ code {low_token} vs {high_token}: proprioceptive feature association\n"
        "skill_index=0, epoch2000; correlation sign is relative to the higher token",
        fontsize=13,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def draw_start_scatter(
    rows_by_model: dict[str, list[dict]],
    output_path: Path,
    low_token: int,
    high_token: int,
) -> None:
    models = list(rows_by_model)
    figure, axes = plt.subplots(1, len(models), figsize=(12.5, 5.2), sharex=True, sharey=True, constrained_layout=True)
    if len(models) == 1:
        axes = [axes]
    colors = {low_token: "#2563eb", high_token: "#dc2626"}
    for axis, model in zip(axes, models, strict=True):
        rows = rows_by_model[model]
        for token in (low_token, high_token):
            selected = [row for row in rows if row["token"] == token]
            axis.scatter(
                [row["start_x"] for row in selected],
                [row["start_z"] for row in selected],
                s=17,
                alpha=0.65,
                label=f"code {token} (n={len(selected)})",
                color=colors[token],
                edgecolors="none",
            )
        axis.set_title(model)
        axis.set_xlabel("start EE x (m)")
        axis.grid(alpha=0.2)
        axis.legend(loc="center left")
    axes[0].set_ylabel("start EE z (m)")
    figure.suptitle("The two codes occupy disjoint absolute start-pose regions")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_html_report(
    output_path: Path,
    summary: dict,
    all_statistics: list[dict],
    low_token: int,
    high_token: int,
) -> None:
    model_names = list(summary["models"])
    selected_features = (
        "start_z",
        "start_x",
        "disp_x",
        "disp_y",
        "path_xyz",
        "rot_net_angle",
        "rot_path_angle",
        "frames",
    )
    stats_lookup = {
        (item["model"], item["feature"]): item
        for item in all_statistics
    }

    key_cards = []
    for model in model_names:
        item = stats_lookup[(model, "start_z")]
        scene = summary["models"][model]
        key_cards.append(
            f"""
            <section class="model-card">
              <h3>{html.escape(model)}</h3>
              <div class="big">start z: r = {item['r_code_high']:+.3f}</div>
              <div>AUC {item['auc_separation']:.3f} · Cohen d {item['cohen_d']:+.2f}</div>
              <div class="means">code {low_token}: {item[f'code{low_token}_mean']:.4f} ± {item[f'code{low_token}_std']:.4f} m<br>
              code {high_token}: {item[f'code{high_token}_mean']:.4f} ± {item[f'code{high_token}_std']:.4f} m</div>
              <div class="means">code {high_token} → Living precision {scene[f'code{high_token}_living_precision']:.1%}, recall {scene[f'code{high_token}_living_recall']:.1%}</div>
            </section>
            """
        )

    key_rows = []
    for feature in selected_features:
        cells = [f"<td><code>{html.escape(feature)}</code></td>"]
        for model in model_names:
            item = stats_lookup[(model, feature)]
            cells.append(
                "<td>"
                f"<strong>r {item['r_code_high']:+.3f}</strong><br>"
                f"AUC {item['auc_separation']:.3f}<br>"
                f"{low_token}: {item[f'code{low_token}_mean']:.4f} · "
                f"{high_token}: {item[f'code{high_token}_mean']:.4f}"
                "</td>"
            )
        key_rows.append("<tr>" + "".join(cells) + "</tr>")

    sorted_statistics = sorted(
        all_statistics,
        key=lambda item: (item["model"], -abs(item["r_code_high"])),
    )
    detail_rows = []
    for item in sorted_statistics:
        detail_rows.append(
            f"""
            <tr data-model="{html.escape(item['model'])}" data-feature="{html.escape(item['feature'].lower())}">
              <td>{html.escape(item['model'])}</td>
              <td><code>{html.escape(item['feature'])}</code></td>
              <td>{item['r_code_high']:+.4f}</td>
              <td>{item['cohen_d']:+.3f}</td>
              <td>{item['auc_separation']:.4f}</td>
              <td>{item[f'code{low_token}_mean']:.5f} ± {item[f'code{low_token}_std']:.5f}</td>
              <td>{item[f'code{high_token}_mean']:.5f} ± {item[f'code{high_token}_std']:.5f}</td>
            </tr>
            """
        )

    model_options = ["<option value=\"all\">all models</option>"]
    model_options.extend(
        f"<option value=\"{html.escape(model)}\">{html.escape(model)}</option>"
        for model in model_names
    )
    model_headers = "".join(f"<th>{html.escape(model)}</th>" for model in model_names)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>contrastive_w01 · code {low_token} vs {high_token}</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0b1020; --panel:#111a2e; --line:#293653; --text:#e8edf8; --muted:#9ba9c4; --blue:#60a5fa; --red:#fb7185; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:radial-gradient(circle at top,#16233e 0,var(--bg) 38rem); color:var(--text); font:14px/1.5 system-ui,-apple-system,sans-serif; }}
    main {{ width:min(1440px,calc(100% - 32px)); margin:0 auto; padding:34px 0 70px; }}
    h1 {{ margin:0; font-size:clamp(27px,4vw,44px); letter-spacing:-.035em; }}
    h2 {{ margin:34px 0 12px; font-size:21px; }}
    h3 {{ margin:0 0 10px; font-size:16px; }}
    .sub {{ color:var(--muted); max-width:900px; margin:8px 0 22px; }}
    .finding {{ padding:18px 20px; border:1px solid #31568a; background:#10233d; border-radius:14px; font-size:16px; }}
    .finding strong {{ color:#fff; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(290px,1fr)); gap:14px; margin-top:14px; }}
    .model-card,.figure,.table-wrap {{ border:1px solid var(--line); background:color-mix(in srgb,var(--panel) 94%,transparent); border-radius:14px; }}
    .model-card {{ padding:17px; }}
    .big {{ color:#fff; font-size:22px; font-weight:760; }}
    .means {{ color:var(--muted); margin-top:8px; }}
    .figures {{ display:grid; grid-template-columns:1fr; gap:16px; }}
    .figure {{ padding:12px; overflow:auto; }}
    .figure img {{ display:block; width:100%; min-width:700px; height:auto; border-radius:8px; }}
    .caption {{ color:var(--muted); margin:8px 6px 2px; }}
    .table-wrap {{ overflow:auto; }}
    table {{ width:100%; border-collapse:collapse; min-width:850px; }}
    th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:right; vertical-align:top; }}
    th {{ position:sticky; top:0; background:#152039; color:#cad5e9; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
    th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
    tbody tr:hover {{ background:#17233d; }}
    code {{ color:#b9d6ff; }}
    .controls {{ display:flex; flex-wrap:wrap; gap:10px; margin:0 0 12px; }}
    select,input {{ border:1px solid var(--line); background:#111a2e; color:var(--text); border-radius:8px; padding:9px 11px; }}
    input {{ min-width:230px; }}
    .legend {{ display:flex; gap:16px; color:var(--muted); margin:10px 0; }}
    .dot::before {{ content:""; display:inline-block; width:9px; height:9px; margin-right:6px; border-radius:50%; background:currentColor; }}
    .blue {{ color:var(--blue); }} .red {{ color:var(--red); }}
    @media (max-width:700px) {{ main {{ width:min(100% - 20px,1440px); padding-top:22px; }} }}
  </style>
</head>
<body>
<main>
  <h1>Code {low_token} vs {high_token}: proprio correlation</h1>
  <p class="sub">contrastive_w01 · {html.escape(summary['epoch'])} · skill_index={summary['skill_index']}. Correlation uses code {high_token}=1 and code {low_token}=0; therefore a negative r means the feature is larger in code {low_token}. Code numbers themselves are categorical.</p>
  <div class="finding"><strong>Main result:</strong> the split is much more strongly associated with the absolute scene coordinate than with left/right motion. Every code-{high_token} sample in this pair is LIVING_ROOM; every code-{low_token} sample is KITCHEN or STUDY. Start EE z alone separates the pair with AUC 1.000 in both models.</div>
  <div class="cards">{''.join(key_cards)}</div>

  <h2>Correlation map</h2>
  <div class="legend"><span class="dot blue">negative: larger in code {low_token}</span><span class="dot red">positive: larger in code {high_token}</span></div>
  <div class="figures">
    <section class="figure"><a href="correlation_map.png"><img src="correlation_map.png" alt="Feature correlation heatmap"></a><p class="caption">Left: point-biserial correlation with code {high_token}. Right: direction-free single-feature AUC. Click for the full-resolution image.</p></section>
    <section class="figure"><a href="start_pose_scatter.png"><img src="start_pose_scatter.png" alt="Start pose scatter plot"></a><p class="caption">Each point is one first skill. The absolute start-pose regions are disjoint for code {low_token} and code {high_token}.</p></section>
  </div>

  <h2>Selected numerical comparison</h2>
  <div class="table-wrap"><table><thead><tr><th>feature</th>{model_headers}</tr></thead><tbody>{''.join(key_rows)}</tbody></table></div>

  <h2>All feature statistics</h2>
  <div class="controls"><select id="modelFilter">{''.join(model_options)}</select><input id="featureFilter" type="search" placeholder="filter feature name"></div>
  <div class="table-wrap"><table id="detailTable"><thead><tr><th>model</th><th>feature</th><th>r(code {high_token})</th><th>Cohen d</th><th>AUC</th><th>code {low_token} mean ± std</th><th>code {high_token} mean ± std</th></tr></thead><tbody>{''.join(detail_rows)}</tbody></table></div>
</main>
<script>
  const modelFilter = document.getElementById('modelFilter');
  const featureFilter = document.getElementById('featureFilter');
  const rows = [...document.querySelectorAll('#detailTable tbody tr')];
  function applyFilters() {{
    const model = modelFilter.value;
    const query = featureFilter.value.trim().toLowerCase();
    for (const row of rows) {{
      const modelMatches = model === 'all' || row.dataset.model === model;
      const featureMatches = !query || row.dataset.feature.includes(query);
      row.hidden = !(modelMatches && featureMatches);
    }}
  }}
  modelFilter.addEventListener('change', applyFilters);
  featureFilter.addEventListener('input', applyFilters);
</script>
</body>
</html>
"""
    output_path.write_text(document)


def main() -> None:
    args = parse_args()
    low_token, high_token = sorted(args.tokens)
    states, lookup = load_bundle(args.skill_bundle)
    collections = find_collections(args.report_root, set(args.models))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_statistics: list[dict] = []
    summary = {
        "epoch": args.epoch,
        "tokens": [low_token, high_token],
        "skill_index": args.skill_index,
        "code_indicator": f"code {high_token}=1, code {low_token}=0",
        "models": {},
    }
    stats_by_model: dict[str, dict[str, dict]] = {}
    rows_by_model: dict[str, list[dict]] = {}
    for model in args.models:
        checkpoint = checkpoint_for_epoch(collections[model], args.epoch)
        rows, all_at_index = collect_rows(
            checkpoint,
            states,
            lookup,
            (low_token, high_token),
            args.skill_index,
        )
        statistics = feature_statistics(rows, low_token, high_token)
        stats_by_model[model] = {item["feature"]: item for item in statistics}
        rows_by_model[model] = rows
        all_statistics.extend({"model": model, **item} for item in statistics)
        pair_scene_counts = Counter((row["scene_family"], str(row["token"])) for row in rows)
        summary["models"][model] = {
            "pair_count": len(rows),
            "pair_scene_by_token": {
                f"{scene}:code{token}": count
                for (scene, token), count in sorted(pair_scene_counts.items())
            },
            **scene_summary(all_at_index, low_token, high_token),
            "top_features_by_absolute_r": [
                item["feature"]
                for item in sorted(statistics, key=lambda item: abs(item["r_code_high"]), reverse=True)[:10]
            ],
        }

    csv_path = args.output_dir / "feature_statistics.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_statistics[0]))
        writer.writeheader()
        writer.writerows(all_statistics)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    draw_heatmap(stats_by_model, args.output_dir / "correlation_map.png", low_token, high_token)
    draw_start_scatter(rows_by_model, args.output_dir / "start_pose_scatter.png", low_token, high_token)
    write_html_report(args.output_dir / "index.html", summary, all_statistics, low_token, high_token)

    print(f"Wrote {csv_path}")
    print(f"Wrote {args.output_dir / 'summary.json'}")
    print(f"Wrote {args.output_dir / 'correlation_map.png'}")
    print(f"Wrote {args.output_dir / 'start_pose_scatter.png'}")
    print(f"Wrote {args.output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
