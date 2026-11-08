"""Group existing FT rollout results without copying or deleting videos.

Dry-run is the default. --apply moves directories on the same filesystem,
records old-name mappings for the evaluator, and creates video browsing indexes.
It does not touch outputs_predictor, Prev, ablation, YAMLs, or metrics values.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from collections import defaultdict
from itertools import chain
from pathlib import Path
from urllib.parse import quote


def destination(name: str) -> Path:
    step = re.search(r"_(\d+k)_", name)
    if not step:
        raise ValueError(f"Cannot identify checkpoint: {name}")
    condition = "gt_broadcast" if "GTskill_GTUV" in name else (
        "pred_preskill_vs_nopreskill" if "preskill_vs_nopreskill" in name else "pred_preskill"
    )
    if "pi05vlm" in name:
        base = Path("pi05vlm")
        training = "expert_frozen_vision_trainable"
        if name.startswith("ft_dual_bottleneck_"):
            if "_vanila_" in name:
                return base / "comparisons/dual_bottleneck_5k_vs_vanilla_30k" / condition
            checkpoints = re.search(r"_(\d+k_\d+k)_fullvlm", name)
            if not checkpoints:
                raise ValueError(f"Cannot identify checkpoint sweep: {name}")
            return base / "dual_bottleneck_9_9" / training / condition / checkpoints[1]
        if name.startswith("ft_vanila_last_pi05vlm_9_9_"):
            checkpoints = re.search(r"_frozenexpert_(\d+k_\d+k)_", name)
            if not checkpoints:
                raise ValueError(f"Cannot identify checkpoint sweep: {name}")
            return base / "vanilla_9_9" / training / condition / checkpoints[1]
        if name.startswith("ft_denoising_wovis_pi05vlm_"):
            if "_bridgehead2_" in name:
                training = "bridge_and_head_trainable"
            return base / "denoising_wovis" / training / condition / step[1]
        raise ValueError(f"Unrecognized pi05 VLM result family: {name}")
    if "_shared_init_range2_" in name:
        return Path("vanilla/expert_frozen_dino_trainable") / condition / "random_range2" / step[1]
    if name.startswith("ft_denoising_8models_visaux_onoff_"):
        return Path("comparisons/visaux_onoff/term_token3_both_proprio_ft20k") / condition / step[1]
    family = next((v for k, v in (
        ("ft_denoising_last_", "denoising_last"), ("ft_denoising_", "denoising"),
        ("ft_learnable_last_", "learnable_last"),
        ("ft_learnable_", "learnable"), ("ft_vanila_", "vanilla"),
    ) if name.startswith(k)), None)
    if family is None:
        raise ValueError(f"Unrecognized result family: {name}")
    if "_frozenexpert_dino_" in name:
        training = "expert_frozen_dino_trainable"
    elif "_full_dino_" in name:
        training = "expert_and_dino_trainable"
    elif "_skillcore_" in name:
        training = "skillcore_frozen_term_token3"
    else:
        training = "expert_frozen_dino_frozen/visaux_off" if "_novisaux_" in name else (
            "expert_frozen_dino_frozen/visaux_on"
        )
    return Path(family) / training / condition / step[1]


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".organizing.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def href(path: Path, base: Path) -> str:
    return quote(path.relative_to(base).as_posix(), safe="/")


STYLE = "body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:16px} " \
        "td,th{padding:8px;text-align:left} video{width:480px;max-width:100%} " \
        "details{padding:12px;border:1px solid #ddd;margin:12px 0} summary{cursor:pointer} " \
        ".videos{display:flex;flex-wrap:wrap;gap:12px} .video{max-width:500px}"


def page(title: str, body: str) -> str:
    return f'<!doctype html><meta charset="utf-8"><title>{html.escape(title)}</title><style>{STYLE}</style>{body}'


def summary(folder: Path) -> str:
    metrics = folder / "metrics/eval_info_merged.json"
    if not metrics.is_file():
        return "집계 결과 없음 (미완료/빈 폴더)"
    rows = []
    for label, result in json.loads(metrics.read_text()).items():
        if not isinstance(result, dict):
            continue
        overall = result.get("overall", {})
        rate = overall.get("pc_success")
        episodes = sum(len(t.get("metrics", {}).get("successes", [])) for t in result.get("per_task", []))
        if rate is not None:
            count = f" · {episodes} episodes" if episodes else ""
            rows.append(f"{html.escape(label)}: {rate:g}%{count}")
    return "<br>".join(rows) or "집계 결과 없음"


def run_index(folder: Path, original: str, root: Path) -> int:
    """No recomputation: link existing videos, plots and per-skill traces."""
    groups = defaultdict(list)
    videos = sorted(folder.glob("side_by_side/*/*.mp4")) + sorted(folder.glob("panels/*/videos/*/*.mp4"))
    for video in videos:
        label = "Side-by-side" if video.parts[-3] == "side_by_side" else video.parts[-4]
        groups[video.parent.name].append((label, video))
    body = f'<a href="{quote(os.path.relpath(root / "index.html", folder), safe="/")}">All FT results</a>'
    body += f"<h1>{html.escape(folder.relative_to(root).as_posix())}</h1><p>{html.escape(original)}</p>"
    body += "<p>기존 저장 영상만 표시합니다. 평가 에피소드 전체의 영상이 저장된 것은 아닙니다. "
    body += ("해당 모델·체크포인트의 결과만 표시합니다.</p>" if (folder / "source_run.json").exists()
             else "먼저 Side-by-side를 보면 모델들을 한 화면에서 비교할 수 있습니다.</p>")
    body += f"<p>{summary(folder)}</p>"
    analysis = folder.parent / "analysis/index.html"
    if (folder / "source_run.json").exists() and analysis.is_file():
        body += f'<p><a href="{quote(os.path.relpath(analysis, folder), safe="/")}">전체·태스크별 체크포인트 성공률 변화</a></p>'
    if (folder / "task_success_rates.png").is_file():
        body += '<p><a href="task_success_rates.png">태스크별 성공률 그래프</a></p>'
    if (folder / "checkpoint_success_rates.png").is_file():
        body += '<p><a href="checkpoint_success_rates.png">체크포인트별 성공률 그래프</a></p>'
    if not videos:
        body += "<p>저장된 MP4 없음 — 취소/미완료 여부 확인 필요.</p>"
    for task, entries in sorted(groups.items(), key=lambda item: (item[0].rsplit("_", 1)[0], int(item[0].rsplit("_", 1)[1]))):
        body += f"<details><summary>{html.escape(task)} · {len(entries)} videos</summary><div class=videos>"
        for label, video in entries:
            link = href(video, folder)
            body += f'<div class=video><p>{html.escape(label)} · {video.stem}</p><video controls preload="none" src="{link}"></video><p><a href="{link}">영상 열기</a></p></div>'
        body += "</div></details>"
    presentation = sorted(folder.glob("panels/*/videos/*/presentation/*.mp4"))
    if presentation:
        body += "<details><summary>발표용 rollout 영상</summary><ul>"
        body += "".join(f'<li><a href="{href(p, folder)}">{html.escape(p.parts[-5])} / {html.escape(p.parts[-3])} / {p.stem}</a></li>' for p in presentation)
        body += "</ul></details>"
    traces = sorted(folder.glob("panels/*/skill_html/*/skill_trace.html"))
    if traces:
        body += "<details><summary>스킬별 상세 trace</summary><ul>"
        body += "".join(f'<li><a href="{href(p, folder)}">{html.escape(p.parts[-4])} / {html.escape(p.parent.name)}</a></li>' for p in traces)
        body += "</ul></details>"
    target = folder / "index.html"
    if target.exists() and "All FT results" not in target.read_text():
        raise FileExistsError(f"Refusing to overwrite existing report: {target}")
    atomic_text(target, page(original, body))
    return len(videos)


def organize(root: Path, *, apply: bool = False) -> dict:
    root = root.resolve(strict=True)
    if (root / "model_results.json").exists():
        from split_model_outputs import split_outputs
        return split_outputs(root, apply=apply)
    manifest = root / "output_locations.json"
    saved = json.loads(manifest.read_text()) if manifest.exists() else {"version": 1, "locations": {}}
    locations = dict(saved["locations"])
    moves = []
    for source in sorted(root.iterdir()):
        if not source.name.startswith("ft_") or not source.is_dir():
            continue
        if source.is_symlink():
            raise ValueError(f"Refusing to move symlink: {source}")
        relative = destination(source.name).as_posix()
        target = root / relative
        if target.exists() or relative in locations.values():
            raise FileExistsError(f"Destination collision: {target}")
        moves.append((source, target))
        locations[source.name] = relative
    for source, target in moves:
        print(f"{source.name} -> {target.relative_to(root)}", flush=True)
    if not apply:
        return {"moves": len(moves), "locations": locations}
    # Persist mapping per move: a later interrupted run can find every moved directory.
    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        saved["locations"][source.name] = target.relative_to(root).as_posix()
        atomic_text(manifest, json.dumps(saved, indent=2) + "\n")
    # Mechanical path repair only; numeric success metrics and media stay unchanged.
    repaired, total_videos = 0, 0
    for original, relative in sorted(locations.items()):
        folder = root / relative
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        old, new = str(root / original), str(folder)
        for path in chain(folder.rglob("*.json"), folder.rglob("*.html")):
            before = path.read_text()
            after = before.replace(old + "/", new + "/").replace('"' + old + '"', '"' + new + '"')
            if after != before:
                if path.suffix == ".json":
                    json.loads(after)
                atomic_text(path, after)
                repaired += 1
        total_videos += run_index(folder, original, root)
    grouped = defaultdict(list)
    for original, relative in sorted(locations.items(), key=lambda pair: pair[1]):
        grouped[relative.split("/")[0]].append((original, relative))
    body = "<h1>FT 평가 결과</h1><p>모델 계열 → 학습 조건 → 평가 입력 → FT 체크포인트 순서입니다. "
    body += "각 결과를 열면 태스크별 Side-by-side 및 개별 모델 영상을 볼 수 있습니다.</p>"
    body += "<p>pi05vlm/은 pretrained VLM 계열, 그 외 모델 계열 폴더는 기존 DINO 계열입니다. "
    body += "체크포인트 비교는 원래 한 평가에서 생성된 영상·metrics를 분리하지 않고 묶어 보존했습니다. "
    body += "random_range2/는 초기 상태 범위 x2 평가이며 기본 범위 결과와 분리되어 있습니다.</p>"
    for family, entries in sorted(grouped.items()):
        body += f"<h2>{html.escape(family)}</h2><table><tr><th>평가 결과</th><th>저장된 성공률</th></tr>"
        body += "".join(f'<tr><td><a href="{quote(relative, safe="/")}/index.html">{html.escape(relative.split("/", 1)[1])}</a></td><td>{summary(root / relative)}</td></tr>' for _, relative in entries)
        body += "</table>"
    body += '<h2>기존 보관 결과</h2><p><a href="Prev/">Prev</a> · <a href="ablation/">ablation</a> (변경 없음)</p>'
    atomic_text(root / "index.html", page("FT evaluation results", body))
    return {"moves": len(moves), "runs": len(locations), "videos": total_videos, "repaired_json": repaired}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(organize(args.root, apply=args.apply), indent=2))
