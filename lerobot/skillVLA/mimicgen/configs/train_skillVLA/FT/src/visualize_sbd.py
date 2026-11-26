"""Show the saved SBD used by the FT selection, without DP inference or recutting."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "lerobot/src/lerobot").is_dir())
FT_DIR = Path(__file__).resolve().parents[1]
SHARED = ROOT / "lerobot/skillVLA/libero/configs/train_skills"
sys.path.insert(0, str(SHARED / "src"))
from train_skills_config import (  # noqa: E402
    _merge_global, _read_yaml, resolve_skillvla_dataset_run, resolve_training_episodes,
    training_episode_suffix,
)


def portable_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        return ROOT / path
    if path.exists():
        return path
    for index, part in enumerate(path.parts):
        if part.startswith("dataset_"):
            candidate = ROOT.joinpath(*path.parts[index:])
            if candidate.exists():
                return candidate
    raise FileNotFoundError(path)


def selection(config_path, *, all_episodes=False):
    # A preview only needs dataset settings. Do not require any warm-start models.
    raw = _read_yaml(config_path)
    cfg = _merge_global(config_path, {"dataset": raw["dataset"]})
    data = cfg["dataset"]
    if all_episodes:
        data = {**data, "n_episodes": "all", "episode_ids": []}
        cfg = {**cfg, "dataset": data}
    source = ROOT / cfg["dataset_root"] / data.get("skillvla_root", "skillvla") / data["source"]
    tag, _ = resolve_skillvla_dataset_run(source, data["run"], data.get("relabeled", ""))
    run = source / tag
    dataset = run / "skillvla"
    episodes = resolve_training_episodes(cfg, dataset)
    provenance = json.loads((run / "sbd_source.json").read_text())
    skillset = portable_path(provenance["skillset_dir"])
    manifest_path = skillset / "skillset_manifest.json"
    build = json.loads((run / "mimicgen_build.json").read_text())
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != build["skillset_manifest_sha256"]:
        raise ValueError("SBD manifest changed after this FT dataset was built; rebuild before previewing.")
    manifest = json.loads(manifest_path.read_text())
    with np.load(run / "skill_latents.npz", allow_pickle=False) as values:
        bounds = {k: values[k].copy() for k in ("episode_id", "task_id", "skill_index", "frame_start", "frame_end", "tokens")}
    if episodes is None:
        # Use the same sorted episode metadata as FT, including non-consecutive IDs.
        import pandas as pd

        episodes = sorted(int(ep) for p in (dataset / "meta/episodes").rglob("*.parquet")
                          for ep in pd.read_parquet(p, columns=["episode_index"])["episode_index"])
    for episode in episodes:
        if not np.any(bounds["episode_id"] == episode):
            raise ValueError(f"FT episode {episode} has no saved skills")
        if not (skillset / f"curves/ep{episode:07d}.npz").is_file():
            raise FileNotFoundError(f"Saved SBD curve missing for episode {episode}; rebuild SBD first.")
    suffix = (training_episode_suffix(cfg, episodes) if data.get("episode_ids")
              else "_epall" if str(data.get("n_episodes", "all")).lower() == "all"
              else f"_ep{len(episodes)}")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", f"sbd_{data['source']}_{run.name}{suffix}")
    return dict(config=str(config_path), source=data["source"], run=str(run),
                skillset=str(skillset), raw_dataset=str(portable_path(manifest["dataset_dir"])),
                episodes=episodes, all_episodes=all_episodes,
                output=str(FT_DIR / "outputs" / name)), bounds, manifest


def codebook_legend(levels):
    from skillset_boundary_viz import fsq_grid_color, _color_text

    if len(levels) != 3:
        return ''
    nx, ny, nz = map(int, levels)
    sections = []
    for z in range(nz):
        rows = []
        for y in reversed(range(ny)):
            cells = []
            for x in range(nx):
                code = x + nx * y + nx * ny * z
                color = fsq_grid_color(code, levels)
                cells.append(f"<td style='background:{color};color:{_color_text(color)};padding:9px;border:1px solid #555' "
                             f"title='FSQ {code} · grid ({x}, {y}, {z})'>FSQ {code}<br><small>({x}, {y}, {z})</small></td>")
            rows.append('<tr>' + ''.join(cells) + '</tr>')
        sections.append(f"<div><strong>z = {z}</strong><table style='border-collapse:collapse'>{''.join(rows)}</table></div>")
    return ("<details style='text-align:center;margin:12px'><summary>FSQ 코드북 색상 · 3D 격자</summary>"
            "<p>x=빨강, y=초록, z=파랑 성분 · 각 축 0 → 2 · 가까운 격자는 비슷한 색</p>"
            "<div style='display:flex;flex-wrap:wrap;justify-content:center;gap:16px'>"
            + ''.join(sections) + '</div></details>')


def render(plan, bounds, manifest):
    # These are the same video player, timeline, curve, and mask renderers as DP eval.
    sys.path.insert(0, str(SHARED / "skill_eval/src"))
    from codebook_visualizer import _episode_row, _load_episodes_meta
    from dp_skillset_eval import _episode_video_metadata
    from skillset_boundary_viz import load_boundary_curve, render_skillset_card, save_gallery

    output = Path(plan["output"])
    levels = json.loads((Path(plan['run']) / 'fsq_meta.json').read_text())['fsq_levels']
    assets = output / "src"
    assets.mkdir(parents=True, exist_ok=True)
    raw = Path(plan["raw_dataset"])
    meta = _load_episodes_meta(raw)
    info = json.loads((raw / "meta/info.json").read_text())
    fps = float(info["fps"])
    image_key = manifest.get("image_key", "observation.images.image")
    cards, records = [], []
    for count, episode in enumerate(plan["episodes"], 1):
        rows = np.flatnonzero(bounds["episode_id"] == episode)
        rows = rows[np.argsort(bounds["skill_index"][rows])]
        skills = [(int(bounds["frame_start"][r]), int(bounds["frame_end"][r]), int(bounds["tokens"][r])) for r in rows]
        curve = load_boundary_curve(Path(plan["skillset"]) / "curves", episode)
        edges = [a for a, _, _ in skills] + [skills[-1][1]]
        if curve is None or not np.array_equal(curve.get("boundaries", []), edges):
            raise ValueError(f"Episode {episode}: FT skill boundaries disagree with the saved SBD curve")
        curve["code_labels_only"] = True
        if len(levels) == 3:
            curve['fsq_levels'] = levels
        curve.pop("denoising_gain", None)
        curve.pop("delta_bic", None)
        video = _episode_video_metadata(raw, meta, image_key, episode, output, fps,
                                        clip_cache_dir=assets / "videos")
        video["code_labels_only"] = True
        if len(levels) == 3:
            video['fsq_levels'] = levels
        row = _episode_row(meta, episode)
        task = int(bounds["task_id"][rows[0]])
        label = f"task{task}: {' / '.join(map(str, row['tasks']))}"
        media = render_skillset_card(skills, None, curve, episode_video=video, show_frames=False)
        cards.append((label, f"Episode ID: {episode} · episode_ids: [{episode}] · {len(skills)} skills · {edges[-1]} frames", media))
        records.append(dict(episode_id=episode, task_id=task, boundaries=edges,
                            codes=[code for _, _, code in skills],
                            threshold=float(np.asarray(curve["mean_val"]).item())))
        print(f"[SBD preview] {count}/{len(plan['episodes'])} · episode {episode}: {len(skills)} skills", flush=True)
    save_gallery(output, f"MimicGen FT · {plan['source']} · {len(records)} selected episodes", cards)
    report = output / "index.html"
    content = report.read_text()
    content = content.replace("<body>", "<body data-server-no-reload>", 1)
    selection_yaml = "episode_ids: " + json.dumps(plan["episodes"])
    note = ("<p style='text-align:center'>Episode ID는 0부터 시작하며 common YAML의 episode_ids에 그대로 입력합니다. "
            "<a href='src/selection.json'>Episode selection and provenance</a></p>"
            "<details style='text-align:center'><summary>표시된 Episode ID 목록</summary>"
            f"<pre style='white-space:pre-wrap'>{html.escape(selection_yaml)}</pre></details>")
    note += codebook_legend(levels)
    content = content.replace("</head>", "<style>.cap{font-size:16px;font-weight:600;text-align:center}</style></head>", 1)
    content = content.replace("</h1>", "</h1>" + note, 1)
    report.write_text(content, encoding="utf-8")
    (assets / "selection.json").write_text(json.dumps({**plan, "results": records}, indent=2) + "\n")
    print(f"HTML: {report}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=FT_DIR / "newtask_ft_common_config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--all-episodes", action="store_true",
                        help="Browse every episode without changing the training selection in the common YAML.")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    plan, bounds, manifest = selection(args.config.resolve(), all_episodes=args.all_episodes)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.dry_run:
        render(plan, bounds, manifest)


if __name__ == "__main__":
    main()
