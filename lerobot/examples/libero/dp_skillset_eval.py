"""DP skill-boundary eval for the train_skills (FSQ_dataset) pipeline.

Visualises how the DP segmented each demo into skills: per episode, the boxed
start/end frames of every skill plus the multimodality (VF cos-divergence) curve
the boundaries were cut from. Same rendering as the build_data_eval DP eval, but
this one works off the FSQ_dataset only (skillset npz + curves + raw videos) — i.e.
BEFORE any skillvla dataset exists.

Inputs:
  --skillset_dir  {SKILLSET_DIR} (uses skillset_manifest.json to infer other paths)
  --skills_dir   {SKILLSET_DIR}/skills   (ep*_task*_skill*.npz: frame_start/end, episode_id, task_id)
  --curves_dir   {SKILLSET_DIR}/curves   (ep{ep:07d}.npz; optional → frames-only if absent)
  --dataset_dir  raw LeRobot dataset (videos + meta) for frames
Output:
  {out_dir}/index.html
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

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


def _resolve_indices(indices, dim: int) -> tuple[int, ...]:
    resolved = []
    for raw in indices:
        index = int(raw)
        if index < 0:
            index += dim
        if index < 0 or index >= dim:
            raise IndexError(f"Action index {raw} is out of range for action_dim={dim}")
        if index not in resolved:
            resolved.append(index)
    return tuple(resolved)


def load_episode_skills(paths: list[Path], gripper_indices=()):
    """Load skill bounds and reconstruct raw action gripper signals on episode time."""
    records = []
    for p in paths:
        with np.load(str(p)) as z:
            records.append((
                int(z["frame_start"]),
                int(z["frame_end"]),
                np.asarray(z["actions"], dtype=np.float32),
            ))
    records.sort(key=lambda t: t[0])
    skills = [(start, end, None) for start, end, _actions in records]
    if not records or not gripper_indices:
        return skills, None, ()

    action_dim = int(records[0][2].shape[-1])
    resolved = _resolve_indices(gripper_indices, action_dim)
    n_frames = max(end for _start, end, _actions in records)
    signal = np.full((n_frames, len(resolved)), np.nan, dtype=np.float32)
    for start, end, actions in records:
        length = min(end - start, len(actions))
        signal[start:start + length] = actions[:length, list(resolved)]
    return skills, signal, resolved


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


def _skill_stats(lengths: list[int]) -> str:
    """'N skills · len min/max/mean' summary for a set of skill lengths (frames)."""
    if not lengths:
        return "0 skills"
    arr = np.asarray(lengths)
    return (
        f"{len(arr)} skills · len min {int(arr.min())} / max {int(arr.max())}"
        f" / mean {float(arr.mean()):.1f}"
    )


def make_frames_loader(dataset_dir: Path, image_key: str, episodes_meta=None):
    episodes_meta = _load_episodes_meta(dataset_dir) if episodes_meta is None else episodes_meta
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


def _task_instruction_map(episodes_meta, ep_task: dict[int, int]) -> dict[int, str]:
    instructions = {}
    if "tasks" not in episodes_meta.columns:
        return instructions
    for _, row in episodes_meta.iterrows():
        ep = int(row["episode_index"])
        task_id = ep_task.get(ep)
        if task_id is None or task_id in instructions:
            continue
        value = row["tasks"]
        if isinstance(value, str):
            text = value
        elif value is None:
            text = ""
        else:
            values = np.asarray(value, dtype=object).reshape(-1).tolist()
            text = str(values[0]) if values else ""
        if text:
            instructions[task_id] = text
    return instructions


def _gripper_labels(dataset_dir: Path, indices: tuple[int, ...]) -> list[str]:
    try:
        info = json.loads((dataset_dir / "meta" / "info.json").read_text())
        names = info["features"]["action"].get("names")
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
        names = None
    labels = []
    for index in indices:
        if names and index < len(names):
            labels.append(str(names[index]))
        elif len(indices) == 1:
            labels.append("gripper")
        else:
            labels.append(f"gripper action[{index}]")
    return labels


def _dataset_fps(dataset_dir: Path) -> float:
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    if fps <= 0:
        raise ValueError(f"Invalid dataset fps: {fps}")
    return fps


def _skill_video_metadata(
    dataset_dir: Path,
    episodes_meta,
    image_key: str,
    episode_id: int,
    skills,
    out_dir: Path,
    fps: float,
) -> list[dict]:
    key = _resolve_image_key(episodes_meta, image_key)
    row = _episode_row(episodes_meta, episode_id)
    video_path = _video_path(dataset_dir, episodes_meta, episode_id, key).resolve()
    relative_path = os.path.relpath(video_path, out_dir.expanduser().resolve())
    video_src = quote(Path(relative_path).as_posix(), safe="/.:@-_")
    episode_start = float(row[f"videos/{key}/from_timestamp"])
    episode_end = float(row[f"videos/{key}/to_timestamp"])
    result = []
    for frame_start, frame_end, _label in skills:
        start_sec = min(episode_end, episode_start + float(frame_start) / fps)
        end_sec = min(episode_end, episode_start + float(frame_end) / fps)
        result.append({
            "video_src": video_src,
            "start_sec": start_sec,
            "end_sec": max(start_sec, end_sec),
        })
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skillset_dir", default=None,
                   help="skillset root; reads skillset_manifest.json and infers skills/curves/dataset")
    p.add_argument("--skills_dir", default=None, help="{SKILLSET_DIR}/skills")
    p.add_argument("--curves_dir", default=None,
                   help="{SKILLSET_DIR}/curves (per-episode multimodality curves; "
                        "absent → frames only, no graph)")
    p.add_argument("--dataset_dir", default=None, help="raw LeRobot dataset (videos + meta)")
    p.add_argument("--image_key", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--out_html", default=None,
                   help="output HTML filename within out_dir; encode the DP (e.g. state_obs10_ck100000.html) "
                        "so different DPs don't overwrite each other in a shallow folder")
    p.add_argument("--n_episodes", type=int, default=12,
                   help="episodes shown; per task when --task_ids is given")
    p.add_argument("--task_ids", type=int, nargs="*", default=None,
                   help="restrict to these tasks (n_episodes each); empty = first n_episodes overall")
    p.add_argument("--thumb_size", type=int, default=160)
    p.add_argument(
        "--skill_video",
        action="store_true",
        help="show a lazy-loaded looping source-video slot between each skill's start/end frames",
    )
    p.add_argument("--hide_start_end_frames", action="store_true", help="omit start/end still images")
    p.add_argument("--hide_cos_graph", action="store_true", help="omit the VF cosine-divergence graph")
    p.add_argument("--hide_gripper_graph", action="store_true", help="omit all gripper signal graphs")
    return p.parse_args()


def _resolve_eval_inputs(args):
    if args.skillset_dir is None:
        missing = [name for name, value in (
            ("--skills_dir", args.skills_dir),
            ("--dataset_dir", args.dataset_dir),
            ("--out_dir", args.out_dir),
        ) if value is None]
        if missing:
            raise ValueError(f"Without --skillset_dir, these arguments are required: {', '.join(missing)}")
        skills_dir = Path(args.skills_dir)
        manifest_path = skills_dir.parent / "skillset_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        return (
            skills_dir,
            Path(args.curves_dir) if args.curves_dir else None,
            Path(args.dataset_dir),
            args.image_key or "observation.images.image",
            Path(args.out_dir),
            args.out_html or "index.html",
            manifest,
        )

    skillset_dir = Path(args.skillset_dir).expanduser().resolve()
    manifest_path = skillset_dir / "skillset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Skillset manifest not found: {manifest_path}. "
            "Rebuild the skillset with the current builder or provide explicit paths."
        )
    manifest = json.loads(manifest_path.read_text())
    skills_dir = Path(args.skills_dir) if args.skills_dir else skillset_dir / "skills"
    curves_dir = Path(args.curves_dir) if args.curves_dir else skillset_dir / "curves"
    if not curves_dir.is_dir():
        curves_dir = None
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path(manifest["dataset_dir"])
    image_key = args.image_key or manifest.get("image_key", "observation.images.image")
    out_dir = Path(args.out_dir) if args.out_dir else skillset_dir / "eval"
    out_html = args.out_html or "index.html"
    return skills_dir, curves_dir, dataset_dir, image_key, out_dir, out_html, manifest


def main():
    args = parse_args()
    skills_dir, curves_dir, dataset_dir, image_key, out_dir, out_html, manifest = _resolve_eval_inputs(args)
    if not skills_dir.is_dir():
        raise FileNotFoundError(f"skills_dir not found: {skills_dir}")

    ep_task, ep_files = index_skillset(skills_dir)
    if not ep_task:
        raise FileNotFoundError(f"No skill npz under {skills_dir}")
    episodes_meta = _load_episodes_meta(dataset_dir)
    frames_src = (
        make_frames_loader(dataset_dir, image_key, episodes_meta=episodes_meta)
        if not args.hide_start_end_frames
        else None
    )
    instructions = _task_instruction_map(episodes_meta, ep_task)
    configured_gripper_indices = (
        [] if args.hide_gripper_graph else manifest.get("action", {}).get("gripper_indices", [])
    )
    fps = _dataset_fps(dataset_dir)

    cards = []
    for task_label, eps in select_episodes(ep_task, args.task_ids, args.n_episodes):
        task_cards = []
        task_lengths = []
        for ep in eps:
            skills, gripper_signal, resolved_gripper_indices = load_episode_skills(
                ep_files[ep], configured_gripper_indices
            )
            raw = frames_src(int(ep)) if frames_src is not None else None
            curve = (
                None
                if args.hide_cos_graph
                else load_boundary_curve(str(curves_dir) if curves_dir else None, ep)
            )
            skill_videos = (
                _skill_video_metadata(
                    dataset_dir,
                    episodes_meta,
                    image_key,
                    int(ep),
                    skills,
                    out_dir,
                    fps,
                )
                if args.skill_video
                else None
            )
            media = render_skillset_card(
                skills,
                raw,
                curve,
                thumb=args.thumb_size,
                gripper_signal=gripper_signal,
                gripper_labels=_gripper_labels(dataset_dir, resolved_gripper_indices),
                skill_videos=skill_videos,
                show_frames=not args.hide_start_end_frames,
            )
            lengths = [end - start for start, end, _video in skills]
            task_lengths.extend(lengths)
            task_cards.append((f"{_cap(task_label, ep)} — {_skill_stats(lengths)}", media))
        # Section header (task id + instruction) gets the stats aggregated over the
        # episodes shown for that task; each episode caption carries its own stats.
        section_label = None
        if task_label is not None:
            section_label = (
                f"task{int(task_label):02d}: "
                f"{instructions.get(int(task_label), '(instruction unavailable)')}"
                f" — {_skill_stats(task_lengths)}"
            )
        cards.extend((section_label, caption, media) for caption, media in task_cards)
    save_gallery(out_dir, "DP skill boundary split", cards, filename=out_html)
    print(f"[dp_eval] done → {out_dir / out_html}")


if __name__ == "__main__":
    main()
