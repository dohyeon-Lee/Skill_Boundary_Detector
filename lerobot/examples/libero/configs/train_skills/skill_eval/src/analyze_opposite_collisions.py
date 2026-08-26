#!/usr/bin/env python3
"""Render concrete same-code examples among opposite adjacent skill pairs."""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--skill-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=40)
    parser.add_argument("--motion-threshold", type=float, default=0.01)
    return parser.parse_args()


def offsets(lengths: np.ndarray) -> np.ndarray:
    return np.concatenate(([0], np.cumsum(lengths[:-1], dtype=np.int64)))


def adjacent_pairs(episodes: np.ndarray, skill_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, episode in enumerate(episodes):
        grouped[int(episode)].append(index)
    pairs: list[tuple[int, int]] = []
    for members in grouped.values():
        members.sort(key=lambda i: (int(skill_indices[i]), i))
        pairs.extend(
            (left, right)
            for left, right in zip(members[:-1], members[1:], strict=True)
            if int(skill_indices[right]) == int(skill_indices[left]) + 1
        )
    return (
        np.asarray([pair[0] for pair in pairs], dtype=np.int64),
        np.asarray([pair[1] for pair in pairs], dtype=np.int64),
    )


def resampled(values: np.ndarray, steps: int = 100) -> tuple[np.ndarray, np.ndarray]:
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, steps)
    return target, np.column_stack(
        [np.interp(target, source, values[:, dim]) for dim in range(values.shape[1])]
    )


def plot_pair(
    first_state: np.ndarray,
    second_state: np.ndarray,
    first_action: np.ndarray,
    second_action: np.ndarray,
    title: str,
    path: Path,
) -> None:
    colors = ("#e76f51", "#277da1")
    labels = ("first skill", "next skill")
    states = (first_state, second_state)
    actions = (first_action, second_action)
    figure = plt.figure(figsize=(15.5, 8.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 3)
    axis_3d = figure.add_subplot(grid[:, 0], projection="3d")
    axes = [figure.add_subplot(grid[row, column]) for row in range(2) for column in (1, 2)]

    for state, color, label in zip(states, colors, labels, strict=True):
        xyz = state[:, :3] - state[0, :3]
        axis_3d.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, linewidth=2.4, label=label)
        axis_3d.scatter(*xyz[0], color=color, marker="o", s=45)
        axis_3d.scatter(*xyz[-1], color=color, marker="X", s=65)
    axis_3d.set(xlabel="relative x", ylabel="relative y", zlabel="relative z", title="Relative XYZ trajectory")
    axis_3d.legend()

    line_styles = ("-", "--", ":")
    axis_labels = ("x", "y", "z")
    for state, color, label in zip(states, colors, labels, strict=True):
        progress, xyz = resampled(state[:, :3] - state[0, :3])
        for dim in range(3):
            axes[0].plot(progress, xyz[:, dim], color=color, linestyle=line_styles[dim], label=f"{label} {axis_labels[dim]}")
        progress, rotation = resampled(state[:, 3:6] - state[0, 3:6])
        for dim in range(3):
            axes[1].plot(progress, rotation[:, dim], color=color, linestyle=line_styles[dim], label=f"{label} r{axis_labels[dim]}")
    axes[0].set(title="Relative position by normalized time", xlabel="skill progress", ylabel="relative position")
    axes[1].set(title="Relative rotation-state change", xlabel="skill progress", ylabel="state rotation delta")

    for action, color, label in zip(actions, colors, labels, strict=True):
        progress, xyz_action = resampled(action[:, :3])
        for dim in range(3):
            axes[2].plot(progress, xyz_action[:, dim], color=color, linestyle=line_styles[dim], label=f"{label} a{axis_labels[dim]}")
        progress, grip = resampled(action[:, 6:7])
        axes[3].plot(progress, grip[:, 0], color=color, linewidth=2.0, label=label)
    axes[2].set(title="Dataset action XYZ", xlabel="skill progress", ylabel="delta action")
    axes[3].set(title="Dataset gripper action", xlabel="skill progress", ylabel="gripper command", ylim=(-1.1, 1.1))
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, ncol=2)
    figure.suptitle(title, fontsize=15)
    figure.savefig(path, dpi=145)
    plt.close(figure)


def vector(value: np.ndarray) -> str:
    return "[" + ", ".join(f"{item:+.3f}" for item in value) + "]"


def copy_replay_images(
    occurrence: dict[str, Any] | None,
    model_root: Path,
    output_dir: Path,
    prefix: str,
) -> list[tuple[str, str]]:
    if occurrence is None:
        return []
    copied: list[tuple[str, str]] = []
    for label, key in (("start", "start_image_path"), ("end", "final_image_path")):
        source = model_root / occurrence[key]
        if not source.is_file():
            continue
        destination = output_dir / "images" / f"{prefix}_{label}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append((label, destination.relative_to(output_dir).as_posix()))
    return copied


def main() -> None:
    args = parse_args()
    collection_path = args.collection.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    collection = json.loads(collection_path.read_text())
    checkpoint = collection["checkpoints"][-1]
    model_root = collection_path.parent.parent
    manifest_path = model_root / "checkpoints" / checkpoint["epoch_tag"] / "metrics" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    with np.load(manifest["signature"]["latents_path"], allow_pickle=False) as latent:
        tokens = latent["tokens"].astype(np.int64)
    with np.load(args.skill_bundle.resolve(), allow_pickle=False) as source:
        bundle = {key: source[key].copy() for key in source.files}

    episodes = bundle["meta_episode_id"].astype(np.int64)
    tasks = bundle["meta_task_id"].astype(np.int64)
    skill_indices = bundle["meta_skill_index"].astype(np.int64)
    state_lengths = bundle["states_len"].astype(np.int64)
    action_lengths = bundle["actions_len"].astype(np.int64)
    state_offsets = offsets(state_lengths)
    action_offsets = offsets(action_lengths)
    states_cat = bundle["states_cat"].astype(np.float32)
    actions_cat = bundle["actions_cat"].astype(np.float32)
    displacement = np.asarray(
        [states_cat[start + length - 1, :3] - states_cat[start, :3] for start, length in zip(state_offsets, state_lengths, strict=True)]
    )
    norms = np.linalg.norm(displacement, axis=1)
    left, right = adjacent_pairs(episodes, skill_indices)
    active = (norms[left] >= args.motion_threshold) & (norms[right] >= args.motion_threshold)
    cosine = np.full(len(left), np.nan, dtype=np.float64)
    cosine[active] = np.sum(displacement[left[active]] * displacement[right[active]], axis=1) / (
        norms[left[active]] * norms[right[active]]
    )
    conflict = active & (cosine < 0.0)
    collision = conflict & (tokens[left] == tokens[right])
    collision_pair_indices = np.flatnonzero(collision)

    occurrence_lookup: dict[tuple[int, int], dict[str, Any]] = {}
    code_coordinates: dict[int, list[int]] = {}
    for skill in checkpoint["skills"]:
        code_coordinates[int(skill["token"])] = [int(x) for x in skill["coord"]]
        for occurrence in skill["occurrences"]:
            occurrence_lookup[(int(occurrence["episode_id"]), int(occurrence["skill_index"]))] = occurrence

    replay_pairs: list[int] = []
    other_by_code: dict[int, list[int]] = defaultdict(list)
    for pair_index in collision_pair_indices:
        first, second = int(left[pair_index]), int(right[pair_index])
        keys = ((int(episodes[first]), int(skill_indices[first])), (int(episodes[second]), int(skill_indices[second])))
        if all(key in occurrence_lookup for key in keys):
            replay_pairs.append(int(pair_index))
        else:
            other_by_code[int(tokens[first])].append(int(pair_index))
    replay_pairs.sort(key=lambda index: cosine[index])
    for values in other_by_code.values():
        values.sort(key=lambda index: cosine[index])
    selected = list(replay_pairs)
    while len(selected) < args.max_examples and any(other_by_code.values()):
        for code, values in sorted(other_by_code.items(), key=lambda item: (-len(item[1]), item[0])):
            if values and len(selected) < args.max_examples:
                selected.append(values.pop(0))

    records: list[dict[str, Any]] = []
    for number, pair_index in enumerate(selected, start=1):
        first, second = int(left[pair_index]), int(right[pair_index])
        first_state = states_cat[state_offsets[first] : state_offsets[first] + state_lengths[first]]
        second_state = states_cat[state_offsets[second] : state_offsets[second] + state_lengths[second]]
        first_action = actions_cat[action_offsets[first] : action_offsets[first] + action_lengths[first]]
        second_action = actions_cat[action_offsets[second] : action_offsets[second] + action_lengths[second]]
        angle = float(np.degrees(np.arccos(np.clip(cosine[pair_index], -1.0, 1.0))))
        code = int(tokens[first])
        plot_name = f"pair_{number:03d}_code_{code:02d}.png"
        plot_pair(
            first_state,
            second_state,
            first_action,
            second_action,
            f"code {code} | task {int(tasks[first])} | episode {int(episodes[first])} | "
            f"skill {int(skill_indices[first])}→{int(skill_indices[second])} | {angle:.1f}°",
            output_dir / "plots" / plot_name,
        )
        first_occurrence = occurrence_lookup.get((int(episodes[first]), int(skill_indices[first])))
        second_occurrence = occurrence_lookup.get((int(episodes[second]), int(skill_indices[second])))
        record = {
            "number": number,
            "pair_index": int(pair_index),
            "first_index": first,
            "second_index": second,
            "task_id": int(tasks[first]),
            "episode_id": int(episodes[first]),
            "first_skill": int(skill_indices[first]),
            "second_skill": int(skill_indices[second]),
            "code": code,
            "coord": code_coordinates.get(code),
            "angle_deg": angle,
            "cosine": float(cosine[pair_index]),
            "first_displacement": displacement[first].tolist(),
            "second_displacement": displacement[second].tolist(),
            "first_length": int(state_lengths[first]),
            "second_length": int(state_lengths[second]),
            "task_description": "" if first_occurrence is None else first_occurrence.get("task_description", ""),
            "has_replay_images": first_occurrence is not None and second_occurrence is not None,
            "plot": f"plots/{plot_name}",
            "first_images": copy_replay_images(first_occurrence, model_root, output_dir, f"pair_{number:03d}_first"),
            "second_images": copy_replay_images(second_occurrence, model_root, output_dir, f"pair_{number:03d}_second"),
        }
        records.append(record)

    code_counts = Counter(int(tokens[left[index]]) for index in collision_pair_indices)
    task_counts = Counter(int(tasks[left[index]]) for index in collision_pair_indices)
    code_rows = "".join(
        f"<tr><td>{code}</td><td>{html.escape(str(code_coordinates.get(code, '')))}</td>"
        f"<td>{count}</td><td>{count / len(collision_pair_indices):.1%}</td></tr>"
        for code, count in code_counts.most_common()
    )
    task_rows = "".join(
        f"<tr><td>{task}</td><td>{count}</td><td>{count / len(collision_pair_indices):.1%}</td></tr>"
        for task, count in task_counts.most_common()
    )
    cards: list[str] = []
    for record in records:
        camera = ""
        if record["has_replay_images"]:
            def image_group(title: str, images: list[tuple[str, str]]) -> str:
                return f"<div><b>{title}</b><div class='camera'>" + "".join(
                    f"<figure><img src='{html.escape(path)}'><figcaption>{label}</figcaption></figure>"
                    for label, path in images
                ) + "</div></div>"
            camera = (
                "<div class='camera-pair'>"
                + image_group("first skill", record["first_images"])
                + image_group("next skill", record["second_images"])
                + "</div>"
            )
        cards.append(
            f"<section class='sample' id='pair-{record['number']}'><div class='chips'>"
            f"<span>code {record['code']} · coord {html.escape(str(record['coord']))}</span>"
            f"<span>task {record['task_id']}</span><span>episode {record['episode_id']}</span>"
            f"<span>skill {record['first_skill']}→{record['second_skill']}</span>"
            f"<span>angle {record['angle_deg']:.1f}°</span>"
            + ("<span class='visual'>replay camera available</span>" if record["has_replay_images"] else "")
            + "</div>"
            + (f"<h3>{html.escape(record['task_description'])}</h3>" if record["task_description"] else "")
            + f"<p><code>{vector(np.asarray(record['first_displacement']))}</code> → "
            f"<code>{vector(np.asarray(record['second_displacement']))}</code> · "
            f"length {record['first_length']}/{record['second_length']}</p>"
            + camera
            + f"<figure class='motion'><img src='{record['plot']}'><figcaption>원: skill 시작, X: skill 종료. 모든 position은 각 skill 시작점을 0으로 두었다.</figcaption></figure></section>"
        )

    payload = {
        "model_name": collection["model_name"],
        "run_name": collection["run_name"],
        "epoch": checkpoint["epoch_tag"],
        "motion_threshold": args.motion_threshold,
        "adjacent_pairs": int(len(left)),
        "opposite_pairs": int(conflict.sum()),
        "same_code_opposite_pairs": int(collision.sum()),
        "opposite_collision_rate": float(collision.sum() / max(conflict.sum(), 1)),
        "replay_image_pairs": len(replay_pairs),
        "code_collision_counts": dict(code_counts),
        "task_collision_counts": dict(task_counts),
        "examples": records,
    }
    (output_dir / "opposite_collision_examples.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    document = f"""<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>zero JS 0.1 · opposite adjacent collision examples</title><style>
:root{{--bg:#07111d;--panel:#101e30;--line:#29425d;--text:#edf5ff;--muted:#9fb2c9;--cyan:#68d8ff;--amber:#ffc66d}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 12% 0,#183957,var(--bg) 42rem);color:var(--text);font:15px/1.62 Inter,system-ui,"Noto Sans KR",sans-serif}}main{{max-width:1500px;margin:auto;padding:48px 28px 90px}}h1{{font-size:clamp(32px,5vw,58px);line-height:1.06;margin:6px 0 15px}}h2{{margin-top:45px}}h3{{margin:12px 0 5px}}p{{color:#cad8e9}}code{{color:#bcecff}}.lead{{max-width:1100px;font-size:17px}}.summary{{display:grid;grid-template-columns:repeat(4,1fr);gap:11px}}.metric,.sample,.table{{background:var(--panel);border:1px solid var(--line);border-radius:14px}}.metric{{padding:16px}}.metric b{{display:block;font-size:28px;color:var(--amber)}}.tables{{display:grid;grid-template-columns:1.2fr 1fr;gap:14px}}.table{{overflow:auto}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px 11px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#192d45;color:#c0ecff}}.sample{{padding:20px;margin:26px 0 48px}}.chips{{display:flex;gap:7px;flex-wrap:wrap}}.chips span{{background:#203a56;padding:5px 10px;border-radius:999px}}.chips .visual{{background:#275b49;color:#caffea}}figure{{margin:10px 0;background:white;border-radius:11px;padding:7px}}figure img{{display:block;width:100%}}figcaption{{color:#46576d;padding:6px}}.camera-pair{{display:grid;grid-template-columns:1fr 1fr;gap:13px;margin-top:16px}}.camera{{display:grid;grid-template-columns:1fr 1fr;gap:7px}}.camera figure{{margin:4px 0}}.motion{{margin-top:14px}}.note{{padding:16px 19px;background:#102a3d;border:1px solid #39748d;border-radius:13px;margin:18px 0}}a{{color:var(--cyan)}}@media(max-width:900px){{.summary,.tables,.camera-pair{{grid-template-columns:1fr}}main{{padding:28px 13px}}}}
</style></head><body><main><div style='color:var(--muted)'>complete training skillset · {checkpoint['epoch_tag']} · {html.escape(collection['run_name'])}</div>
<h1>zero JS 0.1<br>opposite adjacent collision 실제 샘플</h1>
<p class='lead'>같은 episode의 연속 skill 중, 둘 다 XYZ 순변위가 1cm 이상이고 방향각이 90°보다 큰데 같은 FSQ code를 받은 pair를 전수 조사했다.</p>
<div class='summary'><div class='metric'><b>{len(left):,}</b>adjacent pairs</div><div class='metric'><b>{int(conflict.sum()):,}</b>opposite (&gt;90°)</div><div class='metric'><b>{int(collision.sum()):,}</b>same-code collisions</div><div class='metric'><b>{collision.sum()/max(conflict.sum(),1):.2%}</b>P(same code | opposite)</div></div>
<div class='note'><b>핵심:</b> 충돌의 {code_counts[26] / max(int(collision.sum()),1):.1%}가 code 26에, {task_counts[29] / max(int(collision.sum()),1):.1%}가 task 29에 모였다. 즉 3.56%가 27개 code에 균등하게 남은 것이 아니라 특정 code·task의 국소적 merge가 큰 비중을 차지한다.</div>
<div class='tables'><div><h2>code별 충돌</h2><div class='table'><table><thead><tr><th>code</th><th>coord</th><th>pairs</th><th>share</th></tr></thead><tbody>{code_rows}</tbody></table></div></div><div><h2>task별 충돌</h2><div class='table'><table><thead><tr><th>task</th><th>pairs</th><th>share</th></tr></thead><tbody>{task_rows}</tbody></table></div></div></div>
<h2>대표 pair {len(records)}개</h2><p>먼저 replay camera가 있는 {len(replay_pairs)}개를 모두 보여주고, 나머지는 code별로 방향각이 큰 pair를 고르도록 했다.</p>{''.join(cards)}
<p><a href='opposite_collision_examples.json'>raw JSON</a></p></main></body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")
    print(f"Wrote {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
