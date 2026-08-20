#!/usr/bin/env python3
"""Combine per-model terminator probes into one comparison page.

Every model scored the SAME GT skills, so the comparison axis is the skill: one
GT video per skill with each model's termination signal drawn under it. The
codebook grid is only a navigation device and belongs to one grouping model --
token 5 of one FSQ run is unrelated to token 5 of another.
"""

from __future__ import annotations

import argparse
import fcntl
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

MODEL_COLORS = (
    "#d32f2f", "#1976d2", "#2e7d32", "#f9a825", "#7b1fa2",
    "#00838f", "#ef6c00", "#5d4037", "#c2185b", "#455a64",
)


def load_manifests(collection_dir: Path) -> dict[str, dict]:
    """Every finished per-model manifest, keyed by label."""
    manifests: dict[str, dict] = {}
    for path in sorted(collection_dir.glob("models/*/metrics/manifest.json")):
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not manifest.get("completed", False):
            continue
        label = str(manifest.get("label") or "")
        if label != path.parents[1].name:
            raise ValueError(f"Manifest {path} declares label {label!r}.")
        manifests[label] = manifest
    return manifests


def _skill_key(record: dict) -> tuple[int, int, int]:
    return (int(record["task_id"]), int(record["episode_id"]), int(record["skill_index"]))


def align_skills(manifests: dict[str, dict], labels: list[str]) -> list[dict]:
    """Join the models on the skills they all scored.

    A skill missing from one model would silently compare different subsets, so
    only the intersection is reported and the drop is stated in the payload.
    """
    per_label = {
        label: {_skill_key(r): r for r in manifests[label]["records"]} for label in labels
    }
    shared = set.intersection(*(set(v) for v in per_label.values()))
    skills = []
    for key in sorted(shared):
        first = per_label[labels[0]][key]
        skills.append(
            {
                "task_id": key[0],
                "episode_id": key[1],
                "skill_index": key[2],
                "frame_start": int(first["frame_start"]),
                "frame_end": int(first["frame_end"]),
                "length": int(first["length"]),
                "gt_end": int(first["gt_end"]),
                "models": {
                    label: {
                        "token": int(per_label[label][key]["token"]),
                        "pred_end": int(per_label[label][key]["pred_end"]),
                        "timing": int(per_label[label][key]["timing"]),
                        "fired": bool(per_label[label][key]["fired"]),
                        "termination": per_label[label][key]["termination"],
                        "progress": per_label[label][key]["progress"],
                    }
                    for label in labels
                },
            }
        )
    return skills


def timing_histogram(records: list[dict], limit: int = 15) -> dict:
    """Signed timing error counts, clipped into a +/-limit window plus overflow."""
    timings = np.array([r["timing"] for r in records], dtype=np.int64)
    clipped = np.clip(timings, -limit, limit)
    bins = {int(v): int((clipped == v).sum()) for v in range(-limit, limit + 1)}
    return {
        "limit": limit,
        "bins": bins,
        "under": int((timings < -limit).sum()),
        "over": int((timings > limit).sum()),
    }


def select_display_skills(
    skills: list[dict], grouping_label: str, *, max_entries: int, max_samples: int, seed: int
) -> dict[int, list[int]]:
    """Skill indices to render, grouped by the grouping model's codebook entry."""
    by_token: dict[int, list[int]] = defaultdict(list)
    for index, skill in enumerate(skills):
        by_token[int(skill["models"][grouping_label]["token"])].append(index)
    tokens = sorted(by_token, key=lambda t: (-len(by_token[t]), t))
    if max_entries > 0:
        tokens = tokens[:max_entries]
    rng = np.random.default_rng(int(seed))
    chosen: dict[int, list[int]] = {}
    for token in sorted(tokens):
        pool = by_token[token]
        take = min(max_samples, len(pool)) if max_samples > 0 else 0
        if take <= 0:
            chosen[token] = []
            continue
        picked = rng.choice(len(pool), size=take, replace=False)
        chosen[token] = sorted(pool[int(i)] for i in picked)
    return chosen


class _FrameReader:
    """Decode chosen frames of an episode from the dataset's own videos.

    Deliberately not run_fsq_gt_replay's reader: importing that module pulls in
    lerobot.scripts.lerobot_skillvla_eval and skill_data, which costs minutes on
    this cluster's shared venv for the sake of thirty lines.
    """

    def __init__(self, dataset_dir: Path, video_key: str = "observation.images.image") -> None:
        import pandas as pd  # noqa: PLC0415

        self.dataset_dir = Path(dataset_dir)
        self.video_key = video_key
        info = json.loads((self.dataset_dir / "meta" / "info.json").read_text())
        self.fps = float(info["fps"])
        self.path_template = str(info["video_path"])
        files = sorted((self.dataset_dir / "meta" / "episodes").glob("**/*.parquet"))
        if not files:
            raise FileNotFoundError(f"No episode metadata under {self.dataset_dir}")
        columns = [
            "episode_index",
            "length",
            f"videos/{video_key}/chunk_index",
            f"videos/{video_key}/file_index",
            f"videos/{video_key}/from_timestamp",
        ]
        self.index = pd.concat(
            [pd.read_parquet(path, columns=columns) for path in files], ignore_index=True
        ).set_index("episode_index", drop=False)

    def episode_length(self, episode_id: int) -> int:
        return int(self.index.loc[int(episode_id), "length"])

    def frames(self, episode_id: int, frame_indices) -> dict:
        import torch  # noqa: PLC0415
        from lerobot.datasets.video_utils import decode_video_frames  # noqa: PLC0415

        row = self.index.loc[int(episode_id)]
        video_path = self.dataset_dir / self.path_template.format(
            video_key=self.video_key,
            chunk_index=int(row[f"videos/{self.video_key}/chunk_index"]),
            file_index=int(row[f"videos/{self.video_key}/file_index"]),
        )
        start = float(row[f"videos/{self.video_key}/from_timestamp"])
        ordered = sorted({int(index) for index in frame_indices})
        decoded = decode_video_frames(
            video_path,
            [start + index / self.fps for index in ordered],
            tolerance_s=0.5 / self.fps,
        )
        images = (
            (decoded.clamp(0.0, 1.0) * 255.0)
            .round()
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        return {index: images[position] for position, index in enumerate(ordered)}


def render_media(
    skills: list[dict],
    display: dict[int, list[int]],
    labels: list[str],
    *,
    dataset_dir: Path,
    output_dir: Path,
    fps: int,
    frame_stride: int,
    end_threshold: float,
) -> dict[int, dict]:
    """One GT video plus start/end stills per displayed skill, shared by all models.

    The frames are the same for every model -- only the overlaid signals differ --
    so they are decoded once here rather than once per model.
    """
    import imageio.v2 as imageio
    from PIL import Image

    reader = _FrameReader(dataset_dir)
    media: dict[int, dict] = {}
    wanted = sorted({index for indices in display.values() for index in indices})
    for position, index in enumerate(wanted):
        skill = skills[index]
        episode = int(skill["episode_id"])
        start = int(skill["frame_start"])
        length = int(skill["length"])
        episode_length = reader.episode_length(episode)
        frame_indices = [
            min(start + offset, episode_length - 1)
            for offset in range(0, length, max(1, frame_stride))
        ]
        images = reader.frames(episode, frame_indices)
        relative = Path("media") / f"task_{skill['task_id']:02d}" / (
            f"ep{episode:05d}_skill{skill['skill_index']:02d}"
        )
        (output_dir / relative).mkdir(parents=True, exist_ok=True)
        stills = {}
        for name, frame_index in (("start", frame_indices[0]), ("end", frame_indices[-1])):
            path = relative / f"{name}.jpg"
            Image.fromarray(np.asarray(images[frame_index], np.uint8)).save(
                output_dir / path, quality=85
            )
            stills[name] = path.as_posix()
        video_path = relative / "gt.mp4"
        frames = [
            _overlay_signals(
                np.asarray(images[frame_index], np.uint8),
                skill,
                labels,
                step=min(offset, length - 1),
                end_threshold=end_threshold,
            )
            for offset, frame_index in zip(
                range(0, length, max(1, frame_stride)), frame_indices, strict=True
            )
        ]
        imageio.mimsave(str(output_dir / video_path), frames, fps=max(1, fps), macro_block_size=1)
        media[index] = {"video": video_path.as_posix(), **stills}
        if (position + 1) % 20 == 0:
            print(f"[report] rendered {position + 1}/{len(wanted)} skills", flush=True)
    return media


def _overlay_signals(frame, skill, labels, *, step, end_threshold):
    """Draw one termination bar per model under the frame, filled to this step."""
    import cv2

    height, width = frame.shape[:2]
    row = 14
    panel = np.full((row * len(labels) + 6, width, 3), 24, np.uint8)
    for slot, label in enumerate(labels):
        signal = skill["models"][label]["termination"]
        value = float(signal[min(step, len(signal) - 1)])
        color = _bgr(MODEL_COLORS[slot % len(MODEL_COLORS)])
        top = 3 + slot * row
        cv2.rectangle(panel, (60, top), (width - 4, top + row - 4), (60, 60, 60), -1)
        filled = int((width - 64) * max(0.0, min(1.0, value)))
        cv2.rectangle(panel, (60, top), (60 + filled, top + row - 4), color, -1)
        marker = 60 + int((width - 64) * end_threshold)
        cv2.line(panel, (marker, top), (marker, top + row - 4), (200, 200, 200), 1)
        cv2.putText(
            panel, label[:8], (2, top + row - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1
        )
    # A white strip marks the GT end frame so early/late firing is visible.
    if step >= int(skill["gt_end"]):
        cv2.rectangle(panel, (0, 0), (width - 1, panel.shape[0] - 1), (255, 255, 255), 1)
    return np.vstack([frame, panel])


def _bgr(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def build_payload(
    manifests: dict[str, dict],
    labels: list[str],
    *,
    max_entries: int,
    max_samples: int,
    seed: int,
) -> dict:
    skills = align_skills(manifests, labels)
    if not skills:
        raise ValueError("The models share no scored skills; check task_ids and seeds.")
    dropped = {
        label: len(manifests[label]["records"]) - len(skills) for label in labels
    }
    grouping = labels[0]
    display = select_display_skills(
        skills, grouping, max_entries=max_entries, max_samples=max_samples, seed=seed
    )
    return {
        "format": "fsq_terminator_eval_compare_v1",
        "labels": labels,
        "grouping_label": grouping,
        "models": {
            label: {
                "label": label,
                "run_name": manifests[label].get("run_name", ""),
                "epoch_tag": manifests[label].get("epoch_tag", ""),
                "terminator_kind": manifests[label].get("terminator_kind", "?"),
                "termination_only": bool(manifests[label].get("termination_only", False)),
                "fsq_levels": manifests[label].get("fsq_levels", []),
                "codebook_size": int(manifests[label].get("codebook_size", 0)),
                "end_threshold": float(manifests[label].get("end_threshold", 0.5)),
                "summary": manifests[label].get("summary", {}),
                "histogram": timing_histogram(manifests[label]["records"]),
                "skills_scored": len(manifests[label]["records"]),
                "skills_dropped": dropped[label],
            }
            for label in labels
        },
        "skills": skills,
        "display": {str(token): indices for token, indices in display.items()},
    }


def write_html(output_dir: Path, payload: dict, media: dict[int, dict]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = payload["labels"]
    models = payload["models"]
    rows = []
    for slot, label in enumerate(labels):
        model = models[label]
        summary = model["summary"]
        rows.append(
            "<tr>"
            f"<td><span class='dot' style='background:{MODEL_COLORS[slot % len(MODEL_COLORS)]}'></span>"
            f"{html.escape(label)}</td>"
            f"<td>{html.escape(model['terminator_kind'])}</td>"
            f"<td>{'term-only' if model['termination_only'] else 'term+prog'}</td>"
            f"<td>{summary.get('skills', 0)}</td>"
            f"<td>{summary.get('timing_abs_mean', float('nan')):.2f}</td>"
            f"<td>{summary.get('timing_abs_median', float('nan')):.1f}</td>"
            f"<td>{100 * summary.get('within_3_rate', 0):.0f}%</td>"
            f"<td>{100 * summary.get('early_rate', 0):.0f}%</td>"
            f"<td>{100 * summary.get('late_rate', 0):.0f}%</td>"
            f"<td>{100 * summary.get('no_fire_rate', 0):.0f}%</td>"
            f"<td>{html.escape(model['epoch_tag'])}</td>"
            "</tr>"
        )
    data = {
        "labels": labels,
        "colors": [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(labels))],
        "grouping": payload["grouping_label"],
        "codebookSize": models[payload["grouping_label"]]["codebook_size"],
        "levels": models[payload["grouping_label"]]["fsq_levels"],
        "threshold": models[labels[0]]["end_threshold"],
        "display": payload["display"],
        "skills": [
            {
                "task_id": skill["task_id"],
                "episode_id": skill["episode_id"],
                "skill_index": skill["skill_index"],
                "length": skill["length"],
                "gt_end": skill["gt_end"],
                "media": media.get(index, {}),
                "models": {
                    label: {
                        "token": skill["models"][label]["token"],
                        "pred_end": skill["models"][label]["pred_end"],
                        "timing": skill["models"][label]["timing"],
                        "fired": skill["models"][label]["fired"],
                        "termination": skill["models"][label]["termination"],
                    }
                    for label in labels
                },
            }
            for index, skill in enumerate(payload["skills"])
        ],
        "histograms": {label: models[label]["histogram"] for label in labels},
    }
    title = f"FSQ terminator probe — {len(labels)} model(s)"
    page = _HTML_TEMPLATE.replace("__TITLE__", html.escape(title))
    page = page.replace("__ROWS__", "".join(rows))
    page = page.replace("__DATA__", json.dumps(data))
    path = output_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>__TITLE__</title><style>
body{font-family:system-ui,sans-serif;background:#f5f5f5;margin:0;padding:14px;font-size:13px;color:#222}
h1{font-size:18px;margin:0 0 10px}
h2{font-size:15px;margin:16px 0 6px}
table{border-collapse:collapse;font-size:12px;background:#fff;margin-bottom:12px}
th,td{border:1px solid #e0e0e0;padding:4px 8px;text-align:right}
th{background:#f0f0f0;text-align:center}td:first-child{text-align:left}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
.note{color:#666;font-size:12px;margin:4px 0 10px}
#grid{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px;display:inline-block}
#panel{margin-top:12px;background:#fff;border:1px solid #ddd;border-radius:6px;padding:10px;display:none}
.skill{display:flex;gap:12px;align-items:flex-start;border-top:1px solid #eee;padding:10px 0}
.skill video{width:260px;border-radius:4px;background:#000}
.stills img{width:84px;border:1px solid #ddd;border-radius:3px;margin-right:4px}
.cap{font-size:11px;color:#666;margin-bottom:4px}
canvas.cb{cursor:pointer;display:block}
</style></head><body>
<h1>__TITLE__</h1>
<table><tr><th>model</th><th>variant</th><th>heads</th><th>skills</th><th>|err| mean</th>
<th>|err| med</th><th>&le;3</th><th>early</th><th>late</th><th>no-fire</th><th>ckpt</th></tr>__ROWS__</table>
<p class="note">Every model was scored on the same GT skills, so the rows are directly comparable.
The codebook grid below belongs to the <b id="grpname"></b> run only — it is a way to find skills,
not a shared axis, because each run assigns its own codes.</p>
<div id="grid"><canvas class="cb" id="cb"></canvas></div>
<div id="panel"><h2 id="ptitle"></h2><div id="skills"></div></div>
<script>
const D = __DATA__;
document.getElementById('grpname').textContent = D.grouping;
const counts = {};
Object.entries(D.display).forEach(([tok, list]) => { counts[tok] = list.length; });
const tokens = Object.keys(D.display).map(Number).sort((a,b)=>a-b);
const cv = document.getElementById('cb'), ctx = cv.getContext('2d');
const COLS = Math.min(16, Math.max(1, tokens.length)), CELL = 34;
cv.width = COLS*CELL+2; cv.height = Math.ceil(tokens.length/COLS)*CELL+2;
let SEL = -1;
function draw(){
  ctx.clearRect(0,0,cv.width,cv.height);
  tokens.forEach((tok,i)=>{
    const x=(i%COLS)*CELL+1, y=Math.floor(i/COLS)*CELL+1;
    ctx.fillStyle = tok===SEL ? '#f44336' : (counts[tok] ? '#1976d2' : '#ccc');
    ctx.fillRect(x,y,CELL-3,CELL-3);
    ctx.fillStyle='#fff'; ctx.font='10px sans-serif'; ctx.textAlign='center';
    ctx.fillText(tok, x+(CELL-3)/2, y+(CELL-3)/2+3);
  });
}
cv.addEventListener('click', e=>{
  const r=cv.getBoundingClientRect();
  const col=Math.floor((e.clientX-r.left)*cv.width/r.width/CELL);
  const row=Math.floor((e.clientY-r.top)*cv.height/r.height/CELL);
  const i=row*COLS+col;
  if(i>=0 && i<tokens.length) select(tokens[i]);
});
function curve(sig, gtEnd, color, w, h){
  if(!sig.length) return '';
  const pts = sig.map((v,i)=>`${(i/(sig.length-1||1))*w},${h-v*h}`).join(' ');
  const gx = (gtEnd/(sig.length-1||1))*w;
  return `<polyline fill="none" stroke="${color}" stroke-width="1.6" points="${pts}"/>`
       + `<line x1="${gx}" y1="0" x2="${gx}" y2="${h}" stroke="#0d47a1" stroke-dasharray="3,2"/>`;
}
function select(tok){
  SEL=tok; draw();
  const list=D.display[String(tok)]||[];
  document.getElementById('ptitle').textContent =
    `${D.grouping} code ${tok} — ${list.length} skill(s) shown`;
  const W=260,H=54;
  document.getElementById('skills').innerHTML = list.map(idx=>{
    const s=D.skills[idx];
    const media=s.media||{};
    const vid = media.video ? `<video src="${media.video}" controls loop muted></video>` : '';
    const stills = (media.start&&media.end)
      ? `<div class="stills"><img src="${media.start}"><img src="${media.end}"></div>` : '';
    const rows = D.labels.map((lb,si)=>{
      const m=s.models[lb];
      const sign = m.timing>0?`+${m.timing}`:`${m.timing}`;
      return `<div class="cap"><span class="dot" style="background:${D.colors[si]}"></span>`
           + `${lb} · code ${m.token} · fire ${m.fired?m.pred_end:'—'} / GT ${s.gt_end} (${sign})</div>`
           + `<svg width="${W}" height="${H}" style="background:#fafafa;border:1px solid #eee">`
           + curve(m.termination, s.gt_end, D.colors[si], W, H)
           + `<line x1="0" y1="${H-D.threshold*H}" x2="${W}" y2="${H-D.threshold*H}" stroke="#888" stroke-dasharray="2,2"/>`
           + `</svg>`;
    }).join('');
    return `<div class="skill"><div><div class="cap">task ${s.task_id} · ep ${s.episode_id}`
         + ` · skill ${s.skill_index} · len ${s.length}</div>${vid}${stills}</div><div>${rows}</div></div>`;
  }).join('') || '<div class="cap">No skills rendered for this code.</div>';
  document.getElementById('panel').style.display='block';
}
draw();
if(tokens.length) select(tokens[0]);
</script></body></html>
"""


def maybe_build(collection_dir: Path, expected: list[str], **kwargs) -> Path | None:
    """Build the comparison once every expected model has finished.

    Safe to call from every model's job: a lock serializes builders and the page
    is only written when the last manifest is in place.
    """
    metrics = collection_dir / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    with (metrics / "compare.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        manifests = load_manifests(collection_dir)
        missing = [label for label in expected if label not in manifests]
        if missing:
            print(f"[report] waiting for {missing}", flush=True)
            return None
        return build(collection_dir, expected, manifests, **kwargs)


def build(
    collection_dir: Path,
    labels: list[str],
    manifests: dict[str, dict],
    *,
    max_entries: int,
    max_samples: int,
    seed: int,
    fps: int,
    frame_stride: int,
    render_video: bool,
) -> Path:
    payload = build_payload(
        manifests, labels, max_entries=max_entries, max_samples=max_samples, seed=seed
    )
    display = {int(token): indices for token, indices in payload["display"].items()}
    media: dict[int, dict] = {}
    if render_video:
        media = render_media(
            payload["skills"],
            display,
            labels,
            dataset_dir=Path(manifests[labels[0]]["dataset_dir"]),
            output_dir=collection_dir,
            fps=fps,
            frame_stride=frame_stride,
            end_threshold=float(manifests[labels[0]].get("end_threshold", 0.5)),
        )
    temporary = collection_dir / "metrics" / "compare.tmp.json"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(collection_dir / "metrics" / "compare.json")
    path = write_html(collection_dir, payload, media)
    print(f"[report] {path}", flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-labels",
        default="",
        help="Space-separated labels; the page is built only once all have finished.",
    )
    parser.add_argument("--max-entries", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    options = {
        "max_entries": args.max_entries,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "fps": args.fps,
        "frame_stride": args.frame_stride,
        "render_video": not args.no_video,
    }
    expected = args.expected_labels.split()
    if expected:
        maybe_build(args.collection_dir, expected, **options)
        return
    manifests = load_manifests(args.collection_dir)
    if not manifests:
        raise SystemExit(f"No finished manifests under {args.collection_dir}/models.")
    build(args.collection_dir, sorted(manifests), manifests, **options)


if __name__ == "__main__":
    main()
