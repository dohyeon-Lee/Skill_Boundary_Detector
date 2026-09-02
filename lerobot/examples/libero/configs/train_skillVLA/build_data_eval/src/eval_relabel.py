#!/usr/bin/env python3
"""Compare original and predictor-relabeled SkillVLA skill labels."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = THIS_DIR.parent / "relabel_eval_config.yaml"


def _find_global(start: Path) -> Path:
    for directory in (start.resolve(), *start.resolve().parents):
        candidate = directory / "global_config.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find global_config.yaml above {start}")


def _load_config(path: Path) -> dict[str, Any]:
    local = yaml.safe_load(path.read_text()) or {}
    global_path = _find_global(path.parent)
    global_config = yaml.safe_load(global_path.read_text()) or {}
    return {**global_config, **local}


def _resolve_run(
    value: str,
    *,
    project_root: Path,
    dataset_root: Path,
    source_dataset: str,
) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if len(path.parts) > 1:
        return (project_root / path).resolve()
    return (dataset_root / "skillvla_dataset" / source_dataset / path).resolve()


def _decode_code(code: int, levels: list[int]) -> tuple[int, ...]:
    coordinates = []
    remainder = int(code)
    for level in levels:
        coordinates.append(remainder % level)
        remainder //= level
    return tuple(coordinates)


def _coord_text(code: int, levels: list[int]) -> str:
    return "(" + ", ".join(str(value) for value in _decode_code(code, levels)) + ")"


def _pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / max(denominator, 1)


def _load_tasks(dataset_dir: Path) -> dict[int, str]:
    path = dataset_dir / "meta" / "tasks.parquet"
    if not path.is_file():
        return {}
    payload = pq.read_table(path, columns=["task_index", "task"]).to_pydict()
    return {
        int(task_id): str(task)
        for task_id, task in zip(payload["task_index"], payload["task"], strict=True)
    }


def _paired_frame_scan(
    original_dataset: Path,
    relabeled_dataset: Path,
    expected: dict[tuple[int, int], tuple[int, int, int]],
) -> tuple[Counter, Counter, int]:
    """Validate both parquet copies and count frames for every skill occurrence."""
    original_files = {
        path.relative_to(original_dataset / "data"): path
        for path in (original_dataset / "data").rglob("*.parquet")
    }
    relabeled_files = {
        path.relative_to(relabeled_dataset / "data"): path
        for path in (relabeled_dataset / "data").rglob("*.parquet")
    }
    if original_files.keys() != relabeled_files.keys():
        raise ValueError("Original and relabeled parquet file sets differ.")

    columns = ["episode_index", "task_index", "frame_index", "skill_index", "skill_sequence"]
    frames_by_segment: Counter = Counter()
    frames_by_task: Counter = Counter()
    rows_without_real_skill = 0
    for relative in sorted(original_files):
        original_batches = pq.ParquetFile(original_files[relative]).iter_batches(
            batch_size=65536, columns=columns
        )
        relabeled_batches = pq.ParquetFile(relabeled_files[relative]).iter_batches(
            batch_size=65536, columns=columns
        )
        for original_batch, relabeled_batch in zip(
            original_batches, relabeled_batches, strict=True
        ):
            original = original_batch.to_pydict()
            relabeled = relabeled_batch.to_pydict()
            identity_columns = ("episode_index", "task_index", "frame_index", "skill_index")
            if any(original[key] != relabeled[key] for key in identity_columns):
                raise ValueError(f"Row identity changed while relabeling: {relative}")
            for episode, task, skill_index, old_sequence, new_sequence in zip(
                original["episode_index"],
                original["task_index"],
                original["skill_index"],
                original["skill_sequence"],
                relabeled["skill_sequence"],
                strict=True,
            ):
                key = (int(episode), int(skill_index))
                values = expected.get(key)
                if values is None:
                    rows_without_real_skill += 1
                    continue
                expected_task, expected_old, expected_new = values
                if int(task) != expected_task:
                    raise ValueError(f"Task mismatch for episode/skill {key}.")
                old_code = int(old_sequence[int(skill_index)])
                new_code = int(new_sequence[int(skill_index)])
                if old_code != expected_old or new_code != expected_new:
                    raise ValueError(
                        f"Stored label mismatch for episode/skill {key}: "
                        f"parquet={old_code}->{new_code}, artifact={expected_old}->{expected_new}."
                    )
                frames_by_segment[key] += 1
                frames_by_task[int(task)] += 1
    return frames_by_segment, frames_by_task, rows_without_real_skill


def _td(value: Any, *, cls: str = "", sort: Any | None = None) -> str:
    attributes = f' class="{cls}"' if cls else ""
    if sort is not None:
        attributes += f' data-sort="{html.escape(str(sort), quote=True)}"'
    return f"<td{attributes}>{value}</td>"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "episode_id",
        "task_id",
        "task",
        "skill_index",
        "frame_start",
        "original_code",
        "relabeled_code",
        "original_coordinate",
        "relabeled_coordinate",
        "code_manhattan_distance",
        "frames",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _render_report(metrics: dict[str, Any], changed_rows: list[dict[str, Any]]) -> str:
    summary = metrics["summary"]
    levels = metrics["levels"]
    segment_total = summary["segments_total"]
    segment_changed = summary["segments_changed"]
    frame_total = summary.get("frames_total", 0)
    frame_changed = summary.get("frames_changed", 0)

    task_rows = []
    for row in metrics["by_task"]:
        task_rows.append(
            "<tr>"
            + _td(row["task_id"], sort=row["task_id"])
            + _td(html.escape(row["task"]), cls="language")
            + _td(row["segments"], sort=row["segments"])
            + _td(row["unchanged"], sort=row["unchanged"])
            + _td(row["changed"], sort=row["changed"])
            + _td(f'{row["changed_pct"]:.2f}%', sort=row["changed_pct"])
            + _td(row.get("frames", "·"), sort=row.get("frames", -1))
            + _td(
                f'{row["changed_frames_pct"]:.2f}%'
                if "changed_frames_pct" in row
                else "·",
                sort=row.get("changed_frames_pct", -1),
            )
            + "</tr>"
        )

    pair_rows = []
    for row in metrics["changed_pairs"]:
        pair_rows.append(
            "<tr>"
            + _td(f'{row["original_code"]} {_coord_text(row["original_code"], levels)}')
            + _td("→", cls="arrow")
            + _td(f'{row["relabeled_code"]} {_coord_text(row["relabeled_code"], levels)}')
            + _td(row["count"], sort=row["count"])
            + _td(f'{row["share_of_all_changes_pct"]:.2f}%', sort=row["share_of_all_changes_pct"])
            + _td(f'{row["share_of_original_code_pct"]:.2f}%', sort=row["share_of_original_code_pct"])
            + _td(row["manhattan_distance"], sort=row["manhattan_distance"])
            + "</tr>"
        )

    code_rows = []
    for row in metrics["by_code"]:
        delta = row["after"] - row["before"]
        delta_class = "positive" if delta > 0 else "negative" if delta < 0 else ""
        code_rows.append(
            "<tr>"
            + _td(row["code"], sort=row["code"])
            + _td(_coord_text(row["code"], levels))
            + _td(row["before"], sort=row["before"])
            + _td(row["after"], sort=row["after"])
            + _td(f'{delta:+d}', cls=delta_class, sort=delta)
            + _td(row["unchanged"], sort=row["unchanged"])
            + _td(row["changed_out"], sort=row["changed_out"])
            + _td(row["changed_in"], sort=row["changed_in"])
            + _td(f'{row["retention_pct"]:.2f}%', sort=row["retention_pct"])
            + "</tr>"
        )

    matrix_header = "".join(f"<th>{code}</th>" for code in range(len(metrics["matrix"])))
    max_matrix = max((max(row) for row in metrics["matrix"]), default=1)
    matrix_rows = []
    for old_code, values in enumerate(metrics["matrix"]):
        cells = []
        for new_code, count in enumerate(values):
            alpha = 0.08 + 0.82 * math.sqrt(count / max(max_matrix, 1)) if count else 0.0
            color = "37, 99, 235" if old_code == new_code else "234, 88, 12"
            style = f' style="background:rgba({color},{alpha:.3f})"' if count else ""
            cells.append(f'<td{style} title="{old_code} → {new_code}">{count or "·"}</td>')
        matrix_rows.append(f"<tr><th>{old_code}</th>{''.join(cells)}</tr>")

    occurrence_rows = []
    for row in changed_rows:
        occurrence_rows.append(
            "<tr>"
            + _td(row["task_id"], sort=row["task_id"])
            + _td(html.escape(row["task"]), cls="language")
            + _td(row["episode_id"], sort=row["episode_id"])
            + _td(row["skill_index"], sort=row["skill_index"])
            + _td(row["frame_start"], sort=row["frame_start"])
            + _td(f'{row["original_code"]} {html.escape(row["original_coordinate"])}')
            + _td(f'{row["relabeled_code"]} {html.escape(row["relabeled_coordinate"])}')
            + _td(row["code_manhattan_distance"], sort=row["code_manhattan_distance"])
            + _td(row["frames"], sort=row["frames"])
            + "</tr>"
        )

    frame_card = (
        f"""
        <div class="card"><span>프레임 유지</span><strong>{frame_total-frame_changed:,}</strong><small>{_pct(frame_total-frame_changed, frame_total):.2f}%</small></div>
        <div class="card changed"><span>프레임 변경</span><strong>{frame_changed:,}</strong><small>{_pct(frame_changed, frame_total):.2f}%</small></div>
        """
        if frame_total
        else ""
    )
    frame_bar = (
        f'<div class="bar"><div class="kept" style="width:{_pct(frame_total-frame_changed, frame_total):.6f}%"></div>'
        f'<div class="moved" style="width:{_pct(frame_changed, frame_total):.6f}%"></div></div>'
        if frame_total
        else ""
    )
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Skill label relabel comparison</title>
<style>
:root{{--ink:#172033;--muted:#667085;--line:#d9e0ea;--bg:#f4f7fb;--card:#fff;--blue:#2563eb;--orange:#ea580c}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Noto Sans KR",sans-serif}}
main{{max-width:1500px;margin:auto;padding:28px}} h1{{font-size:25px;margin:0 0 5px}} h2{{font-size:18px;margin:28px 0 10px}} p{{margin:6px 0;color:var(--muted)}} code{{background:#e8edf5;padding:2px 5px;border-radius:4px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:20px 0 12px}} .card{{background:var(--card);border:1px solid var(--line);border-top:4px solid var(--blue);border-radius:10px;padding:13px 15px;box-shadow:0 2px 8px #1720330a}} .card.changed{{border-top-color:var(--orange)}} .card span,.card small{{display:block;color:var(--muted)}} .card strong{{display:block;font-size:24px;margin:3px 0}}
.bar{{height:15px;display:flex;overflow:hidden;border-radius:8px;background:#e5e7eb;margin:8px 0}} .kept{{background:var(--blue)}} .moved{{background:var(--orange)}}
.panel{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;overflow:auto}} table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}} th,td{{border-bottom:1px solid #e7ebf1;padding:7px 9px;text-align:right;white-space:nowrap}} th{{position:sticky;top:0;background:#f8fafc;z-index:1;cursor:pointer}} th:first-child,td:first-child{{text-align:left}} td.language{{white-space:normal;text-align:left;min-width:280px}} .arrow{{color:var(--muted);text-align:center}} .positive{{color:#047857}} .negative{{color:#b91c1c}} .matrix table{{width:auto}} .matrix th,.matrix td{{min-width:34px;padding:5px;text-align:center}} .matrix th:first-child{{position:sticky;left:0;z-index:2}} .tools{{display:flex;gap:8px;align-items:center;margin-bottom:9px}} input{{width:min(430px,100%);padding:8px 10px;border:1px solid #cfd6e1;border-radius:7px}} .note{{font-size:12px}} a{{color:var(--blue)}}
</style></head><body><main>
<h1>원본 vs predictor relabel skill label 비교</h1>
<p><code>{html.escape(metrics['original_run'])}</code> → <code>{html.escape(metrics['relabeled_run'])}</code></p>
<p>FSQ levels: {html.escape(str(levels))} · predictor: {html.escape(metrics['predictor_model'])} @ {html.escape(metrics['predictor_checkpoint'])}</p>
<div class="cards">
  <div class="card"><span>Canonical skill 유지</span><strong>{segment_total-segment_changed:,}</strong><small>{_pct(segment_total-segment_changed, segment_total):.2f}%</small></div>
  <div class="card changed"><span>Canonical skill 변경</span><strong>{segment_changed:,}</strong><small>{_pct(segment_changed, segment_total):.2f}%</small></div>
  <div class="card changed"><span>변경된 episode</span><strong>{summary['episodes_changed']:,}</strong><small>{_pct(summary['episodes_changed'], summary['episodes_total']):.2f}%</small></div>
  <div class="card"><span>변경 시 평균 code 거리</span><strong>{summary['mean_changed_manhattan_distance']:.2f}</strong><small>FSQ grid Manhattan</small></div>
  {frame_card}
</div>
<p class="note">파랑은 유지, 주황은 변경입니다. Canonical skill 비율은 transition 단위이고, 프레임 비율은 긴 skill에 더 큰 가중치가 들어간 결과입니다.</p>
<div class="bar"><div class="kept" style="width:{_pct(segment_total-segment_changed, segment_total):.6f}%"></div><div class="moved" style="width:{_pct(segment_changed, segment_total):.6f}%"></div></div>
{frame_bar}

<h2>Task별 변경률</h2><div class="panel"><table class="sortable"><thead><tr><th>Task</th><th>Language</th><th>Skills</th><th>유지</th><th>변경</th><th>변경률</th><th>Frames</th><th>Frame 변경률</th></tr></thead><tbody>{''.join(task_rows)}</tbody></table></div>

<h2>변경된 code pair</h2><div class="panel"><table class="sortable"><thead><tr><th>원본 code</th><th></th><th>Relabeled code</th><th>횟수</th><th>전체 변경 중</th><th>해당 원본 code 중</th><th>거리</th></tr></thead><tbody>{''.join(pair_rows)}</tbody></table></div>

<h2>Code별 사용량과 retention</h2><div class="panel"><table class="sortable"><thead><tr><th>Code</th><th>좌표</th><th>Before</th><th>After</th><th>Δ</th><th>유지</th><th>나감</th><th>들어옴</th><th>Retention</th></tr></thead><tbody>{''.join(code_rows)}</tbody></table></div>

<h2>원본 → relabeled confusion matrix</h2><p class="note">행이 원본, 열이 predictor relabel입니다. 대각선(파랑)은 유지, 비대각선(주황)은 변경입니다.</p><div class="panel matrix"><table><thead><tr><th>old \\ new</th>{matrix_header}</tr></thead><tbody>{''.join(matrix_rows)}</tbody></table></div>

<h2>변경된 canonical skill 목록</h2><div class="panel"><div class="tools"><input id="filter" placeholder="task, language, episode, code 검색"><a href="changed_segments.csv">CSV 다운로드</a></div><table id="occurrences" class="sortable"><thead><tr><th>Task</th><th>Language</th><th>Episode</th><th>Skill index</th><th>Start frame</th><th>원본 code</th><th>Relabeled code</th><th>거리</th><th>Frames</th></tr></thead><tbody>{''.join(occurrence_rows)}</tbody></table></div>
<script>
document.querySelectorAll('table.sortable th').forEach(th=>th.addEventListener('click',()=>{{const i=th.cellIndex,table=th.closest('table'),body=table.tBodies[0],rows=[...body.rows],asc=th.dataset.asc!=='1'; rows.sort((a,b)=>{{let x=a.cells[i]?.dataset.sort??a.cells[i]?.innerText??'',y=b.cells[i]?.dataset.sort??b.cells[i]?.innerText??''; const nx=Number(x),ny=Number(y); return (Number.isFinite(nx)&&Number.isFinite(ny)?nx-ny:x.localeCompare(y))* (asc?1:-1)}}); rows.forEach(r=>body.appendChild(r)); th.dataset.asc=asc?'1':'0'}}));
document.getElementById('filter').addEventListener('input',e=>{{const q=e.target.value.toLowerCase(); document.querySelectorAll('#occurrences tbody tr').forEach(r=>r.hidden=!r.innerText.toLowerCase().includes(q))}});
</script></main></body></html>"""


def evaluate(config_path: Path) -> Path:
    config = _load_config(config_path)
    project_root = Path(str(config["project_root"])).expanduser().resolve()
    dataset_root = Path(str(config.get("dataset_root", "dataset_filtered"))).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    dataset_config = config.get("dataset") or {}
    source_dataset = str(dataset_config["source"])
    relabeled_run = _resolve_run(
        str(dataset_config["relabeled_run"]),
        project_root=project_root,
        dataset_root=dataset_root,
        source_dataset=source_dataset,
    )
    provenance_path = relabeled_run / "relabel_provenance.json"
    artifact_path = relabeled_run / "skill_relabel.npz"
    if not provenance_path.is_file() or not artifact_path.is_file():
        raise FileNotFoundError(
            f"Relabel artifacts not found under {relabeled_run}; expected "
            "relabel_provenance.json and skill_relabel.npz."
        )
    provenance = json.loads(provenance_path.read_text())
    original_value = str(dataset_config.get("original_run", "") or "").strip()
    if original_value:
        original_run = _resolve_run(
            original_value,
            project_root=project_root,
            dataset_root=dataset_root,
            source_dataset=source_dataset,
        )
    else:
        original_run = Path(str(provenance["source_run_path"])).resolve()
    original_dataset = original_run / "skillvla"
    relabeled_dataset = relabeled_run / "skillvla"
    for path in (original_dataset, relabeled_dataset):
        if not (path / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"SkillVLA dataset not found: {path}")

    with np.load(artifact_path, allow_pickle=False) as archive:
        required = {
            "episode_id", "task_id", "skill_index", "frame_start",
            "original_code", "predicted_code",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"skill_relabel.npz lacks {missing}")
        arrays = {name: np.asarray(archive[name]).astype(np.int64) for name in required}
    lengths = {name: len(values) for name, values in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Relabel artifact arrays have different lengths: {lengths}")

    info = json.loads((relabeled_dataset / "meta" / "info.json").read_text())
    levels = [int(value) for value in info.get("skill_fsq_levels", [])]
    if not levels or any(level <= 1 for level in levels):
        raise ValueError(f"Invalid skill_fsq_levels: {levels}")
    code_count = math.prod(levels)
    old = arrays["original_code"]
    new = arrays["predicted_code"]
    if np.any((old < 0) | (old >= code_count) | (new < 0) | (new >= code_count)):
        raise ValueError("Relabel artifact contains a code outside the configured FSQ grid.")

    expected: dict[tuple[int, int], tuple[int, int, int]] = {}
    for episode, task, skill_index, old_code, new_code in zip(
        arrays["episode_id"], arrays["task_id"], arrays["skill_index"], old, new, strict=True
    ):
        key = (int(episode), int(skill_index))
        if key in expected:
            raise ValueError(f"Duplicate episode/skill in relabel artifact: {key}")
        expected[key] = (int(task), int(old_code), int(new_code))

    analysis = config.get("analysis") or {}
    frames_by_segment: Counter = Counter()
    frames_by_task: Counter = Counter()
    skipped_frames = 0
    if bool(analysis.get("frame_level", True)):
        frames_by_segment, frames_by_task, skipped_frames = _paired_frame_scan(
            original_dataset, relabeled_dataset, expected
        )

    tasks = _load_tasks(original_dataset)
    changed_mask = old != new
    segment_total = len(old)
    segment_changed = int(changed_mask.sum())
    pair_counts = Counter(zip(old.tolist(), new.tolist(), strict=True))
    before_counts = Counter(old.tolist())
    after_counts = Counter(new.tolist())
    unchanged_counts = Counter(old[~changed_mask].tolist())
    changed_out = Counter(old[changed_mask].tolist())
    changed_in = Counter(new[changed_mask].tolist())

    by_task_segments: dict[int, dict[str, int]] = defaultdict(
        lambda: {"segments": 0, "changed": 0, "changed_frames": 0}
    )
    changed_rows: list[dict[str, Any]] = []
    changed_distances: list[int] = []
    for index, (episode, task, skill_index, frame_start, old_code, new_code) in enumerate(
        zip(
            arrays["episode_id"], arrays["task_id"], arrays["skill_index"],
            arrays["frame_start"], old, new, strict=True,
        )
    ):
        episode, task, skill_index = int(episode), int(task), int(skill_index)
        old_code, new_code = int(old_code), int(new_code)
        key = (episode, skill_index)
        frame_count = int(frames_by_segment.get(key, 0))
        aggregate = by_task_segments[task]
        aggregate["segments"] += 1
        if old_code != new_code:
            aggregate["changed"] += 1
            aggregate["changed_frames"] += frame_count
            old_coord = _decode_code(old_code, levels)
            new_coord = _decode_code(new_code, levels)
            distance = sum(abs(a - b) for a, b in zip(old_coord, new_coord, strict=True))
            changed_distances.append(distance)
            changed_rows.append(
                {
                    "episode_id": episode,
                    "task_id": task,
                    "task": tasks.get(task, ""),
                    "skill_index": skill_index,
                    "frame_start": int(frame_start),
                    "original_code": old_code,
                    "relabeled_code": new_code,
                    "original_coordinate": _coord_text(old_code, levels),
                    "relabeled_coordinate": _coord_text(new_code, levels),
                    "code_manhattan_distance": distance,
                    "frames": frame_count,
                }
            )

    frame_total = int(sum(frames_by_segment.values()))
    frame_changed = int(sum(row["frames"] for row in changed_rows))
    changed_episodes = set(arrays["episode_id"][changed_mask].tolist())
    all_episodes = set(arrays["episode_id"].tolist())

    by_task = []
    for task in sorted(by_task_segments):
        row = by_task_segments[task]
        result = {
            "task_id": task,
            "task": tasks.get(task, ""),
            "segments": row["segments"],
            "unchanged": row["segments"] - row["changed"],
            "changed": row["changed"],
            "changed_pct": _pct(row["changed"], row["segments"]),
        }
        if frame_total:
            task_frames = int(frames_by_task[task])
            result.update(
                frames=task_frames,
                changed_frames=row["changed_frames"],
                changed_frames_pct=_pct(row["changed_frames"], task_frames),
            )
        by_task.append(result)

    changed_pairs = []
    for (old_code, new_code), count in pair_counts.most_common():
        if old_code == new_code:
            continue
        old_coord = _decode_code(old_code, levels)
        new_coord = _decode_code(new_code, levels)
        changed_pairs.append(
            {
                "original_code": old_code,
                "relabeled_code": new_code,
                "count": count,
                "share_of_all_changes_pct": _pct(count, segment_changed),
                "share_of_original_code_pct": _pct(count, before_counts[old_code]),
                "manhattan_distance": sum(
                    abs(a - b) for a, b in zip(old_coord, new_coord, strict=True)
                ),
            }
        )

    matrix = [[0 for _ in range(code_count)] for _ in range(code_count)]
    for (old_code, new_code), count in pair_counts.items():
        matrix[int(old_code)][int(new_code)] = int(count)
    by_code = [
        {
            "code": code,
            "before": int(before_counts[code]),
            "after": int(after_counts[code]),
            "unchanged": int(unchanged_counts[code]),
            "changed_out": int(changed_out[code]),
            "changed_in": int(changed_in[code]),
            "retention_pct": _pct(unchanged_counts[code], before_counts[code]),
        }
        for code in range(code_count)
    ]

    metrics = {
        "schema_version": 1,
        "source_dataset": source_dataset,
        "original_run": original_run.name,
        "relabeled_run": relabeled_run.name,
        "original_run_path": str(original_run),
        "relabeled_run_path": str(relabeled_run),
        "predictor_model": str(provenance.get("predictor_model", "")),
        "predictor_checkpoint": str(provenance.get("predictor_checkpoint", "")),
        "levels": levels,
        "summary": {
            "segments_total": segment_total,
            "segments_unchanged": segment_total - segment_changed,
            "segments_changed": segment_changed,
            "segment_agreement_pct": _pct(segment_total - segment_changed, segment_total),
            "segment_changed_pct": _pct(segment_changed, segment_total),
            "episodes_total": len(all_episodes),
            "episodes_changed": len(changed_episodes),
            "episode_changed_pct": _pct(len(changed_episodes), len(all_episodes)),
            "frames_total": frame_total,
            "frames_unchanged": frame_total - frame_changed,
            "frames_changed": frame_changed,
            "frame_agreement_pct": _pct(frame_total - frame_changed, frame_total),
            "frame_changed_pct": _pct(frame_changed, frame_total),
            "frames_without_real_skill": skipped_frames,
            "mean_changed_manhattan_distance": (
                float(np.mean(changed_distances)) if changed_distances else 0.0
            ),
        },
        "by_task": by_task,
        "by_code": by_code,
        "changed_pairs": changed_pairs,
        "matrix": matrix,
    }

    output_value = str(config.get("output_dir", "") or "").strip()
    output_dir = (
        Path(output_value).expanduser()
        if output_value
        else relabeled_run / "eval" / "relabel_comparison"
    )
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    max_changed_rows = int(analysis.get("max_changed_rows", 0) or 0)
    html_rows = changed_rows[:max_changed_rows] if max_changed_rows > 0 else changed_rows
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    )
    _write_csv(output_dir / "changed_segments.csv", changed_rows)
    report_path = output_dir / "index.html"
    report_path.write_text(_render_report(metrics, html_rows), encoding="utf-8")
    print(
        f"DONE -> {report_path}\n"
        f"segments: unchanged={segment_total-segment_changed}/{segment_total} "
        f"({_pct(segment_total-segment_changed, segment_total):.2f}%), "
        f"changed={segment_changed}/{segment_total} "
        f"({_pct(segment_changed, segment_total):.2f}%)"
    )
    if frame_total:
        print(
            f"frames: unchanged={frame_total-frame_changed}/{frame_total} "
            f"({_pct(frame_total-frame_changed, frame_total):.2f}%), "
            f"changed={frame_changed}/{frame_total} ({_pct(frame_changed, frame_total):.2f}%)"
        )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    evaluate(args.config.resolve())


if __name__ == "__main__":
    main()
