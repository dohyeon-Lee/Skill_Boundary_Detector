#!/usr/bin/env python3
"""Render several DP skillsets and build one interactive comparison dashboard."""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np

LIBERO_EXAMPLES = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(LIBERO_EXAMPLES))

from codebook_visualizer import _load_episodes_meta  # noqa: E402
from dp_skillset_eval import (  # noqa: E402
    _task_instruction_map,
    index_skillset,
    load_episode_skills,
    select_episodes,
    suite_to_dataset_task_ids,
)


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._-")
    return value or "model"


def _manifest(skillset_dir: Path) -> dict:
    path = skillset_dir / "skillset_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Skillset manifest not found: {path}")
    data = json.loads(path.read_text())
    if not (skillset_dir / "skills").is_dir():
        raise FileNotFoundError(f"Skill directory not found: {skillset_dir / 'skills'}")
    return data


def _policy_metadata(manifest: dict) -> dict:
    policy_path = Path(str(manifest.get("policy_path", "")))
    config_path = policy_path / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    return {
        "policy": policy_path.parents[2].name if len(policy_path.parents) >= 3 else policy_path.name,
        "checkpoint": policy_path.parent.name if policy_path.name == "pretrained_model" else "unknown",
        "mode": str(manifest.get("mode", "custom")),
        "sequence": str(config.get("action_sequence_mode", "legacy")),
        "n_obs": config.get("n_obs_steps", "?"),
        "horizon": config.get("future_action_horizon") or config.get("horizon", "?"),
        "grounding": str(config.get("proprio_grounding", "none")),
    }


def _selection_summary(
    skillset_dir: Path,
    *,
    task_ids: list[int],
    task_id_space: str,
    target_task: str,
    n_episodes: int,
) -> tuple[list[dict], list[dict]]:
    ep_task, ep_files = index_skillset(skillset_dir / "skills")
    manifest = _manifest(skillset_dir)
    episodes_meta = _load_episodes_meta(Path(manifest["dataset_dir"]))
    instructions = _task_instruction_map(episodes_meta, ep_task)
    selected = list(task_ids)
    mappings = []
    if selected and task_id_space == "suite":
        translated = suite_to_dataset_task_ids(
            selected, suite_name=target_task, instructions=instructions
        )
        mappings = [
            {
                "suite_id": int(suite_id),
                "dataset_id": int(dataset_id),
                "instruction": instructions.get(int(dataset_id), ""),
            }
            for suite_id, dataset_id in zip(selected, translated, strict=True)
        ]
        selected = translated
    elif selected:
        mappings = [
            {
                "suite_id": None,
                "dataset_id": int(dataset_id),
                "instruction": instructions.get(int(dataset_id), ""),
            }
            for dataset_id in selected
        ]

    rows = []
    for dataset_task_id, episodes in select_episodes(ep_task, selected, n_episodes):
        for episode_id in episodes:
            skills, _gripper, _indices = load_episode_skills(ep_files[episode_id])
            rows.append(
                {
                    "dataset_task_id": dataset_task_id,
                    "episode_id": int(episode_id),
                    "count": len(skills),
                    "boundaries": [int(end) for _start, end, _label in skills[:-1]],
                }
            )
    return rows, mappings


def _render_dashboard(models: list[dict], output_path: Path, *, output_suffix: str) -> None:
    all_keys = sorted(
        {
            (row["dataset_task_id"], row["episode_id"])
            for model in models
            for row in model["rows"]
        },
        key=lambda item: ((-1 if item[0] is None else item[0]), item[1]),
    )
    row_maps = {
        model["label"]: {
            (row["dataset_task_id"], row["episode_id"]): row for row in model["rows"]
        }
        for model in models
    }

    cards = []
    baseline_rows = row_maps[models[0]["label"]] if models else {}
    for index, model in enumerate(models):
        counts = np.asarray([row["count"] for row in model["rows"]], dtype=np.float64)
        mean = float(counts.mean()) if len(counts) else float("nan")
        std = float(counts.std()) if len(counts) else float("nan")
        minimum = int(counts.min()) if len(counts) else 0
        maximum = int(counts.max()) if len(counts) else 0
        model["stats"] = {
            "mean": mean,
            "std": std,
            "min": minimum,
            "max": maximum,
            "episodes": int(len(counts)),
        }
        boundary_deltas = []
        if index > 0:
            for key, row in row_maps[model["label"]].items():
                baseline = baseline_rows.get(key)
                if baseline is None or len(baseline["boundaries"]) != len(row["boundaries"]):
                    continue
                boundary_deltas.extend(
                    abs(left - right)
                    for left, right in zip(
                        baseline["boundaries"], row["boundaries"], strict=True
                    )
                )
        boundary_mae = float(np.mean(boundary_deltas)) if boundary_deltas else None
        model["stats"]["boundary_mae_vs_baseline"] = boundary_mae
        meta = model["metadata"]
        comparison_text = (
            "comparison baseline"
            if index == 0
            else (
                f"boundary MAE vs baseline {boundary_mae:.1f} frames"
                if boundary_mae is not None
                else "boundary MAE unavailable (different skill counts)"
            )
        )
        cards.append(
            f"<button class='model-card' data-index='{index}'>"
            f"<span class='model-number'>{index + 1}</span>"
            f"<b>{html.escape(model['label'])}</b>"
            f"<span>{html.escape(meta['sequence'])} · obs {meta['n_obs']} · future {meta['horizon']}</span>"
            f"<span>{html.escape(meta['grounding'])} · ckpt {html.escape(meta['checkpoint'])}</span>"
            f"<strong>{mean:.2f} ± {std:.2f} skills/episode</strong>"
            f"<span>{html.escape(comparison_text)}</span>"
            "</button>"
        )

    head_cells = "".join(f"<th>{html.escape(model['label'])}</th>" for model in models)
    body_rows = []
    for task_id, episode_id in all_keys:
        cells = []
        counts = []
        for model in models:
            row = row_maps[model["label"]].get((task_id, episode_id))
            counts.append(None if row is None else row["count"])
        numeric = [value for value in counts if value is not None]
        count_spread = max(numeric) - min(numeric) if numeric else 0
        available_rows = [
            row_maps[model["label"]].get((task_id, episode_id)) for model in models
        ]
        boundary_lengths = {
            len(row["boundaries"]) for row in available_rows if row is not None
        }
        boundary_spread = 0
        if len(boundary_lengths) == 1:
            for position in range(next(iter(boundary_lengths), 0)):
                positions = [
                    row["boundaries"][position] for row in available_rows if row is not None
                ]
                if positions:
                    boundary_spread = max(boundary_spread, max(positions) - min(positions))
        if count_spread:
            severity = 3
            disagreement = f"skill count Δ{count_spread}"
        elif boundary_spread >= 15:
            severity = 2
            disagreement = f"boundary max Δ{boundary_spread}f"
        elif boundary_spread >= 5:
            severity = 1
            disagreement = f"boundary max Δ{boundary_spread}f"
        else:
            severity = 0
            disagreement = f"boundary max Δ{boundary_spread}f"
        for model, count in zip(models, counts, strict=True):
            row = row_maps[model["label"]].get((task_id, episode_id))
            if row is None:
                cells.append("<td class='missing'>—</td>")
                continue
            boundaries = ", ".join(str(value) for value in row["boundaries"]) or "none"
            cells.append(
                f"<td title='internal boundaries: {html.escape(boundaries)}'>"
                f"<b>{row['count']}</b><small>{html.escape(boundaries)}</small></td>"
            )
        task_text = "all" if task_id is None else str(task_id)
        body_rows.append(
            f"<tr class='spread-{severity}'><th>task {task_text} · ep {episode_id}</th>"
            + "".join(cells)
            + f"<td class='disagreement'>{html.escape(disagreement)}</td>"
            + "</tr>"
        )

    model_json = json.dumps(
        [{"label": model["label"], "file": model["file"]} for model in models],
        ensure_ascii=False,
    ).replace("</", "<\\/")
    mapping = models[0].get("mappings", []) if models else []
    mapping_text = ", ".join(
        (
            f"suite {item['suite_id']} → dataset {item['dataset_id']} · {item['instruction']}"
            if item["suite_id"] is not None
            else f"dataset {item['dataset_id']} · {item['instruction']}"
        )
        for item in mapping
    )
    html_text = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>DP boundary comparison · {html.escape(output_suffix)}</title>
<style>
:root{{--bg:#f4f6fa;--card:#fff;--line:#d7deea;--ink:#172033;--muted:#667085;--blue:#2563eb}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:20;background:rgba(255,255,255,.96);border-bottom:1px solid var(--line);padding:16px 22px;backdrop-filter:blur(8px)}}
h1{{font-size:21px;margin:0 0 5px}}.subtitle{{font-size:13px;color:var(--muted)}}main{{padding:18px 22px 40px}}
.models{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin-bottom:18px}}
.model-card{{position:relative;text-align:left;border:1px solid var(--line);border-radius:12px;background:var(--card);padding:13px 13px 13px 42px;cursor:pointer;color:var(--ink)}}
.model-card:hover,.model-card.active{{border-color:var(--blue);box-shadow:0 0 0 2px #2563eb22}}.model-card span,.model-card strong{{display:block;margin-top:4px;font-size:12px}}.model-card span{{color:var(--muted)}}
.model-number{{position:absolute;left:12px;top:12px!important;width:22px;height:22px;border-radius:50%;background:#eaf0ff;color:var(--blue)!important;text-align:center;padding-top:3px;font-weight:800}}
.panel{{background:var(--card);border:1px solid var(--line);border-radius:12px;margin:0 0 18px;overflow:hidden}}.panel h2{{font-size:15px;margin:0;padding:12px 14px;border-bottom:1px solid var(--line)}}
.controls{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:12px 14px}}select,button.mode{{padding:7px 9px;border:1px solid #c8d0dc;border-radius:7px;background:white}}button.mode{{cursor:pointer}}button.mode.active{{background:var(--blue);color:white;border-color:var(--blue)}}
.viewers{{display:grid;grid-template-columns:1fr;gap:8px;padding:0 8px 8px}}.viewers.compare{{grid-template-columns:1fr 1fr}}.viewer-head{{display:flex;justify-content:space-between;align-items:center;padding:7px 4px;font-size:12px;font-weight:750}}iframe{{width:100%;height:72vh;border:1px solid var(--line);border-radius:8px;background:white}}.secondary{{display:none}}.compare .secondary{{display:block}}
	.table-wrap{{overflow:auto;max-height:62vh}}table{{border-collapse:separate;border-spacing:0;width:100%;font-size:12px}}th,td{{padding:8px 10px;border-right:1px solid var(--line);border-bottom:1px solid var(--line);background:#fff;white-space:nowrap}}thead th{{position:sticky;top:0;background:#eef2f8;z-index:2}}tbody th{{position:sticky;left:0;background:#f8fafc;text-align:left;z-index:1}}td{{text-align:center}}td small{{display:block;color:var(--muted);font-size:10px;margin-top:3px}}td.disagreement{{font-weight:750}}tr.spread-1 td{{background:#fffbea}}tr.spread-2 td,tr.spread-3 td{{background:#fff1f0}}.hint{{padding:10px 14px;color:var(--muted);font-size:12px;border-top:1px solid var(--line)}}
@media(max-width:1000px){{.viewers.compare{{grid-template-columns:1fr}}iframe{{height:65vh}}}}
</style></head><body>
	<header><h1>DP skill-boundary comparison</h1><div class="subtitle">{html.escape(mapping_text)} · {len(models)} models · 각 셀의 작은 숫자는 내부 boundary frame입니다.</div></header>
<main><section class="models">{''.join(cards)}</section>
<section class="panel"><h2>Interactive viewer</h2><div class="controls">
<label>A <select id="selectA"></select></label><label>B <select id="selectB"></select></label>
<button class="mode active" id="single">Single</button><button class="mode" id="compare">Side by side</button></div>
<div class="viewers" id="viewers"><div><div class="viewer-head"><span id="labelA"></span><a id="openA" target="_blank">open standalone ↗</a></div><iframe id="frameA"></iframe></div>
<div class="secondary"><div class="viewer-head"><span id="labelB"></span><a id="openB" target="_blank">open standalone ↗</a></div><iframe id="frameB"></iframe></div></div></section>
	<section class="panel"><h2>Episode-level boundary overview</h2><div class="table-wrap"><table><thead><tr><th>Episode</th>{head_cells}<th>Difference</th></tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>
	<div class="hint">노랑/빨강 행은 모델 간 boundary 위치나 skill 개수가 크게 다른 episode입니다. 위의 Side by side로 바로 비교할 수 있습니다.</div></section></main>
<script>const MODELS={model_json};const a=document.getElementById('selectA'),b=document.getElementById('selectB');
for(const [i,m] of MODELS.entries()){{for(const s of [a,b]){{const o=document.createElement('option');o.value=i;o.textContent=`${{i+1}} · ${{m.label}}`;s.appendChild(o)}}}}b.value=MODELS.length>1?1:0;
function load(which){{const s=which==='A'?a:b,m=MODELS[Number(s.value)],f=document.getElementById('frame'+which),l=document.getElementById('label'+which),o=document.getElementById('open'+which);f.src=m.file;l.textContent=m.label;o.href=m.file;document.querySelectorAll('.model-card').forEach((c,i)=>c.classList.toggle('active',which==='A'&&i===Number(s.value)))}}a.onchange=()=>load('A');b.onchange=()=>load('B');
document.querySelectorAll('.model-card').forEach((c,i)=>c.onclick=()=>{{a.value=i;load('A')}});const viewers=document.getElementById('viewers'),single=document.getElementById('single'),compare=document.getElementById('compare');single.onclick=()=>{{viewers.classList.remove('compare');single.classList.add('active');compare.classList.remove('active')}};compare.onclick=()=>{{viewers.classList.add('compare');compare.classList.add('active');single.classList.remove('active');load('B')}};load('A');load('B');</script></body></html>"""
    output_path.write_text(html_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skillsets-json", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--output-suffix", default="comparison")
    parser.add_argument("--n-episodes", type=int, required=True)
    parser.add_argument("--task-ids", type=int, nargs="*", default=[])
    parser.add_argument("--task-id-space", choices=("dataset", "suite"), default="dataset")
    parser.add_argument("--target-task", default="")
    parser.add_argument("--thumb-size", type=int, default=160)
    parser.add_argument("--skill-video", action="store_true")
    parser.add_argument("--hide-start-end-frames", action="store_true")
    parser.add_argument("--hide-cos-graph", action="store_true")
    parser.add_argument("--hide-gain-graph", action="store_true")
    parser.add_argument("--hide-bic-graph", action="store_true")
    parser.add_argument("--hide-gripper-graph", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = json.loads(args.skillsets_json)
    if not isinstance(specs, list) or not specs:
        raise ValueError("--skillsets-json must contain a non-empty list")
    first_manifest = _manifest(Path(specs[0]["skillset_dir"]))
    dataset_name = str(first_manifest["dataset_name"])
    run_dir = args.out_root / dataset_name / _slug(args.output_suffix)
    run_dir.mkdir(parents=True, exist_ok=True)

    display_flags = []
    for enabled, flag in (
        (args.skill_video, "--skill_video"),
        (args.hide_start_end_frames, "--hide_start_end_frames"),
        (args.hide_cos_graph, "--hide_cos_graph"),
        (args.hide_gain_graph, "--hide_gain_graph"),
        (args.hide_bic_graph, "--hide_bic_graph"),
        (args.hide_gripper_graph, "--hide_gripper_graph"),
    ):
        if enabled:
            display_flags.append(flag)

    rendered = []
    for index, spec in enumerate(specs):
        label = str(spec["label"])
        skillset_dir = Path(spec["skillset_dir"])
        manifest = _manifest(skillset_dir)
        if str(manifest.get("dataset_name")) != dataset_name:
            raise ValueError("All comparison skillsets must use the same source dataset")
        output_file = f"{index + 1:02d}_{_slug(label)}.html"
        command = [
            sys.executable,
            str(LIBERO_EXAMPLES / "dp_skillset_eval.py"),
            "--skillset_dir",
            str(skillset_dir),
            "--out_dir",
            str(run_dir),
            "--out_html",
            output_file,
            "--title",
            f"DP boundary · {label}",
            "--n_episodes",
            str(args.n_episodes),
            "--task_id_space",
            args.task_id_space,
            "--target_task",
            args.target_task,
            "--thumb_size",
            str(args.thumb_size),
            *display_flags,
        ]
        if args.task_ids:
            command.extend(["--task_ids", *(str(value) for value in args.task_ids)])
        print(f"\n[{index + 1}/{len(specs)}] {label}", flush=True)
        subprocess.run(command, check=True)
        rows, mappings = _selection_summary(
            skillset_dir,
            task_ids=args.task_ids,
            task_id_space=args.task_id_space,
            target_task=args.target_task,
            n_episodes=args.n_episodes,
        )
        rendered.append(
            {
                "label": label,
                "file": quote(output_file),
                "rows": rows,
                "mappings": mappings,
                "metadata": _policy_metadata(manifest),
            }
        )

    index_path = run_dir / "index.html"
    _render_dashboard(rendered, index_path, output_suffix=args.output_suffix)
    (run_dir / "comparison_summary.json").write_text(
        json.dumps(rendered, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[dp_compare] done -> {index_path}")


if __name__ == "__main__":
    main()
