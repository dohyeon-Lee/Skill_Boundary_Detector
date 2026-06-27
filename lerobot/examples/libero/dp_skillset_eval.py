"""DP skill-boundary eval for the train_skills (FSQ_dataset) pipeline.

Visualises how the DP segmented each demo into skills: per episode, the boxed
start/end frames of every skill plus the multimodality (VF cos-divergence) curve
the boundaries were cut from. Same rendering as the build_data_eval DP eval, but
this one works off the FSQ_dataset only (skillset npz + curves + raw videos) — i.e.
BEFORE any skillvla dataset exists.

Inputs:
  --skills_dir   {SKILLSET_DIR}/skills   (ep*_task*_skill*.npz: frame_start/end, episode_id, task_id)
  --curves_dir   {SKILLSET_DIR}/curves   (ep{ep:07d}.npz; optional → frames-only if absent)
  --dataset_dir  raw LeRobot dataset (videos + meta) for frames
Output:
  {out_dir}/index.html
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# examples/libero on path (codebook_visualizer + skillset_boundary_viz live here).
LIBERO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LIBERO_DIR))

from codebook_visualizer import (  # noqa: E402
    _episode_row,
    _load_episodes_meta,
    _read_episode_clip,
    _resolve_image_key,
    _video_path,
)
from skillset_boundary_viz import load_boundary_curve, render_skillset_card, save_gallery  # noqa: E402

_SKILL_RE = re.compile(r"ep(\d+)_task(\d+)_skill(\d+)\.npz$")


def index_skillset(skills_dir: Path) -> tuple[dict[int, int], dict[int, list[Path]]]:
    """Scan skills_dir filenames (no npz load) → (episode→task_id, episode→[skill npz paths])."""
    ep_task: dict[int, int] = {}
    ep_files: dict[int, list[Path]] = defaultdict(list)
    for path in skills_dir.rglob("ep*_task*_skill*.npz"):
        m = _SKILL_RE.search(path.name)
        if not m:
            continue
        ep, task = int(m.group(1)), int(m.group(2))
        ep_task[ep] = task
        ep_files[ep].append(path)
    return ep_task, ep_files


def load_episode_skills(paths: list[Path]) -> list[tuple[int, int, None]]:
    """Per-episode skill segments → [(frame_start, frame_end, None), ...] sorted by start.
    Label is None: the skillset has no FSQ token (DP boundary view, pre-FSQ)."""
    skills = []
    for p in paths:
        z = np.load(str(p))
        skills.append((int(z["frame_start"]), int(z["frame_end"]), None))
    skills.sort(key=lambda t: t[0])
    return skills


def select_episodes(ep_task: dict[int, int], task_ids, n_episodes: int) -> list[tuple]:
    """Ordered [(task_label, [ep, ...]), ...] — mirrors build_data_eval.select_episodes.
    Empty task_ids → first n_episodes episodes overall (task_label=None); else
    n_episodes episodes per listed task."""
    if not task_ids:
        return [(None, sorted(ep_task)[:n_episodes])]
    groups = []
    for t in task_ids:
        eps = sorted(ep for ep, ti in ep_task.items() if ti == int(t))
        groups.append((int(t), eps[:n_episodes]))
    return groups


def _cap(task_label, ep) -> str:
    return f"episode {ep}" if task_label is None else f"task{int(task_label):02d} · episode {ep}"


def make_frames_loader(dataset_dir: Path, image_key: str):
    episodes_meta = _load_episodes_meta(dataset_dir)
    key = _resolve_image_key(episodes_meta, image_key)

    def load(ep_id: int):
        try:
            row = _episode_row(episodes_meta, ep_id)
            from_ts = float(row[f"videos/{key}/from_timestamp"])
            to_ts = float(row[f"videos/{key}/to_timestamp"])
            return _read_episode_clip(_video_path(dataset_dir, episodes_meta, ep_id, key),
                                      from_ts, to_ts, int(row["length"]))
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] frames ep{ep_id}: {exc}")
            return None
    return load


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skills_dir", required=True, help="{SKILLSET_DIR}/skills")
    p.add_argument("--curves_dir", default=None,
                   help="{SKILLSET_DIR}/curves (per-episode multimodality curves; "
                        "absent → frames only, no graph)")
    p.add_argument("--dataset_dir", required=True, help="raw LeRobot dataset (videos + meta)")
    p.add_argument("--image_key", default="observation.images.image")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--out_html", default="index.html",
                   help="output HTML filename within out_dir; encode the DP (e.g. state_obs10_ck100000.html) "
                        "so different DPs don't overwrite each other in a shallow folder")
    p.add_argument("--n_episodes", type=int, default=12,
                   help="episodes shown; per task when --task_ids is given")
    p.add_argument("--task_ids", type=int, nargs="*", default=None,
                   help="restrict to these tasks (n_episodes each); empty = first n_episodes overall")
    p.add_argument("--thumb_size", type=int, default=160)
    return p.parse_args()


def main():
    args = parse_args()
    skills_dir = Path(args.skills_dir)
    if not skills_dir.is_dir():
        raise FileNotFoundError(f"skills_dir not found: {skills_dir}")

    ep_task, ep_files = index_skillset(skills_dir)
    if not ep_task:
        raise FileNotFoundError(f"No skill npz under {skills_dir}")
    frames_src = make_frames_loader(Path(args.dataset_dir), args.image_key)

    cards = []
    for task_label, eps in select_episodes(ep_task, args.task_ids, args.n_episodes):
        for ep in eps:
            skills = load_episode_skills(ep_files[ep])
            raw = frames_src(int(ep))
            curve = load_boundary_curve(args.curves_dir, ep)
            b64 = render_skillset_card(skills, raw, curve, thumb=args.thumb_size)
            cards.append((task_label, f"{_cap(task_label, ep)} — {len(skills)} skills", b64))
    save_gallery(Path(args.out_dir), "DP skill boundary split", cards, filename=args.out_html)
    print(f"[dp_eval] done → {Path(args.out_dir) / args.out_html}")


if __name__ == "__main__":
    main()
