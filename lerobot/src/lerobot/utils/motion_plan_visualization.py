"""Read-only 4H or separately sampled canonical full-skill versus actual H.

XYZ curves integrate controller-scaled translational commands, NOT forward
simulation. They cannot predict contact, tracking error, or rotational effects.
The past-H segment is reconstructed backwards from the current EEF anchor.
Window diagnostics distinguish a last-step readout from a genuinely independent
base-only denoising rollout initialized with exactly the deployed noise.
"""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from lerobot.policies.skillVLA.chunk_patch_target import project_top_points


def decode_motion_plan(plan, contexts, postprocessor, env_postprocessor, *, anchors=None):
    """Apply isolated copies of the real action processors to diagnostic readouts."""
    import torch

    chunk = plan["chunk"]
    batch, h, dim = chunk.shape
    if len(contexts) != batch:
        raise ValueError("Motion-plan camera contexts must match the policy batch.")
    if plan.get("kind") == "canonical":
        anchors = {} if anchors is None else anchors
        records = []
        for index, full in enumerate(plan["full_skills"]):
            normalized = torch.cat([full["actions"][..., :dim], chunk[index]], dim=0)
            decoded = postprocessor(normalized.clone())
            decoded = env_postprocessor({"action": decoded})["action"].detach().float().cpu().numpy()
            length = int(full["gt_length"])
            if len(decoded) != length + h:
                raise ValueError("Canonical diagnostic length does not match its GT horizon.")
            key = (index, int(plan["skill_indices"][index]))
            anchor = anchors.setdefault(key, np.asarray(contexts[index]["eef_world"]).copy())
            records.append({
                "kind": "canonical", "batch_index": index,
                "episode_step": int(plan["episode_step"]), "skill_index": key[1],
                "token": int(plan["tokens"][index]), "n_action_steps": int(plan["n_action_steps"]),
                "base_actions": decoded[:length].copy(), "chunk_actions": decoded[length:].copy(),
                "context": contexts[index], "full_skill_start_world": anchor.copy(),
                "skill_start_step": int(full["skill_start_step"]), "gt_length": length,
                "flow_time": 0.0,  # Independently fully denoised, NOT a 4H last-step estimate.
            })
        return records
    normalized = torch.cat([plan["base_window"][..., :dim], chunk,
                            plan["final_window"][..., :dim]], dim=1)
    if "last_step_base_window" in plan:
        normalized = torch.cat([normalized, plan["last_step_base_window"][..., :dim]], dim=1)
    # Match the regular per-step [B,D] processor contract, not [B,T,D].
    decoded = postprocessor(normalized.reshape(-1, dim).clone())
    decoded = env_postprocessor({"action": decoded})["action"]
    decoded = decoded.reshape(batch, normalized.shape[1], dim).detach().float().cpu().numpy()
    records = [{
        "kind": plan.get("kind", "window"),
        "batch_index": index, "episode_step": int(plan["episode_step"]),
        "skill_index": int(plan["skill_indices"][index]), "token": int(plan["tokens"][index]),
        "flow_time": float(plan["flow_time"]), "n_action_steps": int(plan["n_action_steps"]),
        "base_actions": decoded[index, :4*h].copy(),
        "chunk_actions": decoded[index, 4*h:5*h].copy(),
        "final_window_actions": decoded[index, 5*h:9*h].copy(), "context": contexts[index],
    } for index in range(batch)]
    if plan.get("kind") == "independent_window":
        for index, record in enumerate(records):
            record["last_step_base_actions"] = decoded[index, 9*h:13*h].copy()
            record["last_step_flow_time"] = float(plan["last_step_flow_time"])
            record["initial_noise_normalized"] = plan["initial_noise_normalized"][index].detach().float().cpu().numpy().copy()
    return records


def command_path(actions, context, *, anchor_index=0):
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < 3:
        raise ValueError("Actions must be a [T,D>=3] array.")
    if not 0 <= anchor_index <= len(actions):
        raise ValueError("Anchor must be a trajectory point index.")
    low, high = np.asarray(context["input_min"]), np.asarray(context["input_max"])
    out_low, out_high = np.asarray(context["output_min"]), np.asarray(context["output_max"])
    if np.any(high <= low):
        raise ValueError("Invalid controller input limits.")
    delta = ((np.clip(actions[:, :3], low, high) - (high + low) / 2)
             * np.abs(out_high - out_low) / np.abs(high - low) + (out_high + out_low) / 2)
    points = np.concatenate([np.zeros((1, 3)), np.cumsum(delta, axis=0)])
    return points - points[anchor_index] + np.asarray(context["eef_world"])


def project_path(points, camera):
    height, width = camera["image"].shape[:2]
    uv, valid = project_top_points(points, camera["transform"], height=height, width=width)
    pixels = (uv + 1) * np.asarray([width - 1, height - 1]) / 2
    pixels[~valid] = np.nan  # Do not connect a curve across behind/out-of-view points.
    return pixels


def select_skill_samples(records, count):
    """Uniformly spaced real replan events, separately for each skill occurrence."""
    if count <= 0:
        raise ValueError("Sample count must be positive.")
    groups = defaultdict(list)
    for record in records:
        groups[int(record["skill_index"])].append(record)
    result = {}
    for skill, group in sorted(groups.items()):
        group.sort(key=lambda record: record["episode_step"])
        indices = np.unique(np.linspace(0, len(group) - 1, min(count, len(group))).round().astype(int))
        result[skill] = [group[i] for i in indices]
    return result


def draw_motion_plan(record, path):
    # Isolate pyplot/backend imports from the ordinary, disabled eval path.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    base, chunk = record["base_actions"], record["chunk_actions"]
    h = len(chunk)
    canonical = record.get("kind") == "canonical"
    independent = record.get("kind") == "independent_window"
    if not canonical and len(base) != 4 * h:
        raise ValueError("Expected a 4H base and an H action chunk.")
    context = record["context"]
    base_context = {**context, "eef_world": record["full_skill_start_world"]} if canonical else context
    anchor_index = 0 if canonical else h
    base_xyz = command_path(base, base_context, anchor_index=anchor_index)
    chunk_xyz = command_path(chunk, context)
    cameras = context["cameras"]
    fig = Figure(figsize=(7 * len(cameras), 8), constrained_layout=True)
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(2, len(cameras), height_ratios=[4, 1])
    segments = [(0, len(base) + 1, "#2878bd", "Full skill (sampled once)")] if canonical else [
                (0, h + 1, "#888888", "Base past H"),
                (h, 2 * h + 1, "#2878bd", "Base current H"),
                (2 * h, 4 * h + 1, "#00a9a0", "Base future 2H")]
    for col, (name, camera) in enumerate(cameras.items()):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(camera["image"])
        projected = project_path(base_xyz, camera)
        for start, end, color, label in segments:
            curve = projected[start:end]
            ax.plot(curve[:, 0], curve[:, 1], ".-", color=color, lw=2, ms=4, label=label)
        final = project_path(chunk_xyz, camera)
        prefix = min(int(record["n_action_steps"]), h)
        ax.plot(final[:, 0], final[:, 1], "--", color="#e44436", lw=2, label="Final action H")
        ax.plot(final[:prefix + 1, 0], final[:prefix + 1, 1], ".-", color="#e44436", lw=3,
                label=f"Execution prefix (up to {prefix})")
        ax.plot(final[0, 0], final[0, 1], "*", color="white", markeredgecolor="black", ms=13,
                label="Current EEF")
        if canonical:
            ax.plot(projected[0, 0], projected[0, 1], "s", color="#2878bd", ms=7, label="Skill-start EEF")
            ax.plot(projected[-1, 0], projected[-1, 1], "o", color="#2878bd", fillstyle="none", ms=8,
                    label="Full-skill end (GT horizon)")
        ax.set(xlim=(-.5, camera["image"].shape[1] - .5),
               ylim=(camera["image"].shape[0] - .5, -.5), title=f"Current {name} camera")
        ax.axis("off")
        ax.legend(loc="lower left", fontsize=8, framealpha=.85)
    gripper = fig.add_subplot(grid[1, :])
    age = record["episode_step"] - record["skill_start_step"] if canonical else 0
    if base.shape[1] >= 7:
        offsets = np.arange(len(base)) if canonical else np.arange(4 * h) - h
        gripper.plot(offsets, base[:, -1], ".-", color="#2878bd", label="Full-skill gripper" if canonical else "Base gripper")
        gripper.plot(np.arange(h) + age, chunk[:, -1], ".-", color="#e44436", label="Final gripper")
        gripper.legend(loc="upper right", fontsize=8)
    gripper.axvspan(age, age + h, color="#2878bd", alpha=.08)
    gripper.set(xlabel=("Steps from skill start (elapsed time only; NOT predicted phase)" if canonical else
                        "Action offset from current state (steps)"), ylabel="Gripper command")
    gripper.grid(alpha=.2)
    description = (f"Canonical full skill: GT length {len(base)}, separate sample at skill start; current age {age}" if canonical
                   else ("Independent base-only 4H vs deployed H: SAME initial noise, separate denoising states" if independent
                         else f"Base 4H clean estimate at last flow time {record['flow_time']:.3g}; final H from same rollout"))
    fig.suptitle(f"Skill {record['skill_index']} / code {record['token']} / episode step {record['episode_step']}\n"
                 f"{description}\n"
                 "Command-integrated XYZ approximation, NOT simulated/observed motion", fontsize=12)
    fig.savefig(path, dpi=140)
    return base_xyz, chunk_xyz


def write_motion_plan_gallery(records, output_dir, *, samples_per_skill=4, success=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sections, manifest = [], []
    for skill, samples in select_skill_samples(records, samples_per_skill).items():
        images = []
        for record in samples:
            stem = f"skill{skill:03d}_code{record['token']:03d}_step{record['episode_step']:04d}"
            image_path = output_dir / f"{stem}.png"
            base_xyz, chunk_xyz = draw_motion_plan(record, image_path)
            # Preserve raw commands/calibration so visualization can be changed
            # without repeating expensive policy evaluation.
            arrays = {key: record[key] for key in ("base_actions", "chunk_actions", "final_window_actions",
                       "full_skill_start_world", "skill_start_step", "gt_length", "last_step_base_actions",
                       "last_step_flow_time", "initial_noise_normalized") if key in record}
            arrays["kind"] = record.get("kind", "window")
            arrays.update(base_xyz=base_xyz, chunk_xyz=chunk_xyz,
                          flow_time=record["flow_time"], episode_step=record["episode_step"])
            arrays.update({key: record["context"][key] for key in
                           ("eef_world", "input_min", "input_max", "output_min", "output_max")})
            for camera_name, camera in record["context"]["cameras"].items():
                arrays[f"{camera_name}_image"] = camera["image"]
                arrays[f"{camera_name}_transform"] = camera["transform"]
            np.savez_compressed(output_dir / f"{stem}.npz", **arrays)
            manifest.append({"skill_index": skill, "token": record["token"],
                             "episode_step": record["episode_step"], "image": image_path.name,
                             "data": f"{stem}.npz", "flow_time": record["flow_time"],
                             "kind": record.get("kind", "window"), "gt_length": record.get("gt_length")})
            images.append(f'<a href="{image_path.name}" target="_blank"><img loading="lazy" '
                          f'src="{image_path.name}" alt="step {record["episode_step"]}"></a>')
        sections.append(f'<details open id="skill-{skill}"><summary>Skill {skill} / code {samples[0]["token"]}</summary>'
                        f'<div class="grid">{"".join(images)}</div></details>')
    status = "unknown" if success is None else ("SUCCESS" if success else "FAILURE")
    canonical = bool(records and records[0].get("kind") == "canonical")
    independent = bool(records and records[0].get("kind") == "independent_window")
    title = ("Full skill vs current action chunk" if canonical else
             "Independent base-only 4H vs deployed action chunk" if independent else
             "4H base vs final action chunk")
    description = (
        'Blue: a SEPARATELY sampled canonical full skill, anchored at the actual skill-start EEF. '
        'Red: the actual H action prediction, anchored at the current EEF. '
        'The full skill is NOT an intermediate plan consumed by the action path. '
        'GT length is used ONLY for diagnostic sampling, never for control or termination. '
        'One sample is reused for the entire skill occurrence; sampling preserves the real policy RNG. '
        'Gripper offsets use elapsed skill time for reference, NOT a learned/aligned phase. '
        if canonical else (
        'Blue/teal/grey: a base-only 4H rollout using NO image/Cond/terminal layer at any denoising step. '
        'Red: the actual deployed joint-rollout H. Both start from the EXACT SAME 4H Gaussian noise '
        'and state/skill/XYZ conditions but maintain SEPARATE intermediate action states. '
        'The independent base is diagnostic only; robot actions and their RNG stream are unchanged. '
        'The old last-step readout and initial normalized noise are preserved in each NPZ. '
        'Past H is predicted, not observed; out-of-skill/episode regions may be unsupervised. '
        if independent else
        'Blue/teal/grey: last-step base clean estimate; red: final H from the same denoising rollout. '
        'Past H is predicted, not observed. No extra sampling or teacher information. '
        'Regions outside the skill/episode were masked in the training loss and may be unsupervised. '
        )
    )
    document = (f'<!doctype html><meta charset="utf-8"><title>{title}</title>'
                '<style>body{font:16px system-ui;background:#f6f7fa;margin:24px;color:#17202a}'
                '.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}'
                'img{width:100%;background:white}details{margin:20px 0}summary{font-weight:bold;cursor:pointer}'
                '@media(max-width:800px){.grid{grid-template-columns:1fr}}</style>'
                f'<h1>{title} — {html.escape(status)}</h1>'
                '<p>Four uniformly spaced replans per skill (or the configured count). Click any image to enlarge.</p>'
                f'<p>{description}'
                'Curves integrate OSC-scaled commands: contacts/tracking/rotation are not simulated; '
                'a solid red prefix is the execution budget, not a guarantee that all steps executed.</p>'
                + ''.join(sections))
    (output_dir / "index.html").write_text(document, encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(output_dir / "index.html")
