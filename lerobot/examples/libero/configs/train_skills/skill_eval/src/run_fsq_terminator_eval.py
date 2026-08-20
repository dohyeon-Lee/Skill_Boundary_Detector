#!/usr/bin/env python3
"""Score one FSQ co-trained terminator against the GT skill boundaries.

No policy and no simulator: the skillset already fixes where every skill ends,
and the camera frames come from the dataset's own videos. The only question is
when this checkpoint's terminator fires relative to that known end.

The terminator variant is whatever the checkpoint built (state+image, image
only, wrist only, state MLP, or state RNN); it is dispatched from the module
type rather than configured.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from FSQ import (  # noqa: E402
    FSQImageOnlyQueryTerminator,
    FSQStateMLPTerminator,
    FSQStateRNNTerminator,
    FSQWristOnlyQueryTerminator,
)
from fsq_eval import batched_encode, load_model  # noqa: E402
from train_FSQ import attach_episode_offsets  # noqa: E402

log = logging.getLogger("fsq_terminator_eval")

THIRD_KEY = "observation.images.image"
WRIST_KEY = "observation.images.wrist_image"


def terminator_kind(module) -> str:
    """Name the terminator's input contract, from the module the checkpoint built."""
    if module is None:
        return "none"
    # Subclass order matters: wrist-only extends image-only extends state+image.
    if isinstance(module, FSQStateRNNTerminator):
        return "state_rnn"
    if isinstance(module, FSQStateMLPTerminator):
        return "state_mlp"
    if isinstance(module, FSQWristOnlyQueryTerminator):
        return "wrist_only"
    if isinstance(module, FSQImageOnlyQueryTerminator):
        return "image_only"
    return "state_image"


def needs_images(kind: str) -> bool:
    """Whether a variant reads camera frames, i.e. whether they must be decoded."""
    return kind in {"state_image", "image_only", "wrist_only"}


@torch.no_grad()
def _step_terminator(module, kind, z_norm, state, third, wrist, hidden):
    """One terminator step for a batch of skills; returns (progress, prob, hidden)."""
    if kind == "state_image":
        progress, logits = module(z_norm, state, third, wrist)
    elif kind == "image_only":
        progress, logits = module(z_norm, third, wrist)
    elif kind == "wrist_only":
        progress, logits = module(z_norm, wrist)
    elif kind == "state_mlp":
        progress, logits = module.forward_outputs(z_norm, state[..., : module.state_dim])
    elif kind == "state_rnn":
        progress, logits, hidden = module.step_outputs(
            z_norm, state[..., : module.state_dim], hidden
        )
    else:
        raise ValueError(f"This checkpoint has no usable terminator (kind={kind!r}).")
    return progress, torch.sigmoid(logits), hidden


@torch.no_grad()
def run_terminator(
    model,
    kind: str,
    latents: np.ndarray,
    states: list[np.ndarray],
    metadata: list[dict],
    lengths: list[int],
    raw_dataset,
    device,
    skills_per_batch: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Termination probability and progress for every frame of every skill.

    Skills are stepped in lockstep rather than flattened into frame batches: the
    recurrent variant carries hidden state across a skill's own frames, so the
    time axis has to stay ordered per skill.
    """
    module = model.terminator
    if module is None:
        raise ValueError("This checkpoint was trained without a terminator.")
    module.eval()
    wants_images = needs_images(kind)
    terms = [np.zeros(int(lengths[i]), np.float32) for i in range(len(lengths))]
    progs = [np.zeros(int(lengths[i]), np.float32) for i in range(len(lengths))]

    order = sorted(range(len(lengths)), key=lambda i: -int(lengths[i]))
    for start in range(0, len(order), skills_per_batch):
        group = order[start : start + skills_per_batch]
        z_norm_all = model.fsq.normalized(
            torch.from_numpy(latents[group].astype(np.float32)).to(device)
        )
        hidden = None
        horizon = max(int(lengths[i]) for i in group)
        for step in range(horizon):
            active = [slot for slot, i in enumerate(group) if step < int(lengths[i])]
            if not active:
                break
            if len(active) != len(group):
                # A finished skill must not keep stepping: it would advance the
                # recurrent state of the rows still running.
                group = [group[slot] for slot in active]
                z_norm_all = z_norm_all[active]
                if hidden is not None:
                    hidden = hidden[:, active] if hidden.ndim == 3 else hidden[active]
            state = torch.from_numpy(
                np.stack([states[i][step] for i in group]).astype(np.float32)
            ).to(device)
            third = wrist = None
            if wants_images:
                frames = [
                    raw_dataset[
                        int(metadata[i]["dataset_from_index"])
                        + int(metadata[i]["frame_start"])
                        + step
                    ]
                    for i in group
                ]
                third = torch.stack([frame[THIRD_KEY] for frame in frames]).to(device)
                wrist = torch.stack([frame[WRIST_KEY] for frame in frames]).to(device)
            progress, prob, hidden = _step_terminator(
                module, kind, z_norm_all, state, third, wrist, hidden
            )
            prob_np = prob.reshape(-1).float().cpu().numpy()
            progress_np = progress.reshape(-1).float().cpu().numpy()
            for slot, i in enumerate(group):
                terms[i][step] = prob_np[slot]
                progs[i][step] = progress_np[slot]
        log.info(
            "terminator: %d/%d skills done", min(start + skills_per_batch, len(order)), len(order)
        )
    return terms, progs


def skill_timing(term: np.ndarray, end_threshold: float) -> dict:
    """When this terminator fires, against the GT end at the last frame."""
    length = int(len(term))
    gt_end = length - 1
    hits = np.flatnonzero(term >= end_threshold)
    fired = bool(len(hits))
    pred_end = int(hits[0]) if fired else int(np.argmax(term))
    timing = pred_end - gt_end
    return {
        "length": length,
        "gt_end": gt_end,
        "pred_end": pred_end,
        "fired": fired,
        "timing": timing,
        "timing_abs": abs(timing),
        "peak": float(term.max()) if length else 0.0,
    }


def aggregate(per_skill: list[dict]) -> dict:
    if not per_skill:
        return {}
    timing = np.array([s["timing"] for s in per_skill], dtype=np.float32)
    return {
        "skills": len(per_skill),
        "timing_abs_mean": float(np.abs(timing).mean()),
        "timing_abs_median": float(np.median(np.abs(timing))),
        "timing_mean": float(timing.mean()),
        "early_rate": float((timing < 0).mean()),
        "late_rate": float((timing > 0).mean()),
        "exact_rate": float((timing == 0).mean()),
        "no_fire_rate": float(np.mean([not s["fired"] for s in per_skill])),
        "within_3_rate": float((np.abs(timing) <= 3).mean()),
        "within_5_rate": float((np.abs(timing) <= 5).mean()),
    }


SKILL_NAME = re.compile(r"^ep(\d+)_task(\d+)_skill(\d+)\.npz$")


def index_skill_files(skills_dir: Path) -> list[dict]:
    """Task/episode/skill of every skill file, from its NAME alone.

    build_skill_dataset.py writes task{tid}/ep{ep}_task{tid}_skill{si}.npz, so
    the selection keys are already in the path. Reading them here means only the
    handful of skills we actually score get opened -- a full skillset is >11k
    npz files, and opening all of them costs minutes per job on shared storage.
    """
    entries = []
    for path in sorted(skills_dir.rglob("*.npz")):
        match = SKILL_NAME.match(path.name)
        if match is None:
            continue
        episode, task, skill = (int(g) for g in match.groups())
        entries.append(
            {"path": path, "episode_id": episode, "task_id": task, "skill_index": skill}
        )
    if not entries:
        raise FileNotFoundError(f"No skill npz files under {skills_dir}")
    return entries


def load_selected_skills(entries: list[dict]) -> tuple[list, list, list[dict]]:
    """Open only the chosen skill files, matching train_FSQ.load_skill_files."""
    segments, states, metadata = [], [], []
    for entry in entries:
        data = np.load(str(entry["path"]))
        raw_states = data["states"].astype(np.float32)
        segments.append(raw_states)
        states.append(raw_states)
        metadata.append(
            {
                "file": entry["path"].name,
                "episode_id": int(data["episode_id"]),
                "task_id": int(data["task_id"]) if "task_id" in data else entry["task_id"],
                "skill_index": int(data["skill_index"]),
                "frame_start": int(data["frame_start"]),
                "frame_end": int(data["frame_end"]),
                "length": int(len(data["actions"])),
            }
        )
    return segments, states, metadata


def select_skills(metadata, *, task_ids, episodes_per_task, selection, seed, episode_ids):
    """Skill indices to score, chosen by task and episode exactly like the replay tools.

    Takes only the selection keys, so it runs on the cheap filename index before
    any skill file is opened.
    """
    by_task: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, item in enumerate(metadata):
        by_task[int(item["task_id"])][int(item["episode_id"])].append(index)
    if episode_ids:
        wanted = set(int(value) for value in episode_ids)
        chosen = [
            index
            for index, item in enumerate(metadata)
            if int(item["episode_id"]) in wanted
        ]
        if not chosen:
            raise RuntimeError(f"No skills for episode_ids {sorted(wanted)}.")
        return sorted(chosen)
    available = sorted(by_task) if task_ids is None else [t for t in task_ids if t in by_task]
    missing = [] if task_ids is None else [t for t in task_ids if t not in by_task]
    if missing:
        log.warning("Skipping task_ids absent from the skillset: %s", missing)
    if not available:
        raise RuntimeError("None of the requested task_ids exist in this skillset.")
    rng = np.random.default_rng(int(seed))
    chosen: list[int] = []
    for task in available:
        episodes = sorted(by_task[task])
        if len(episodes) < episodes_per_task:
            log.warning(
                "task %d has only %d episodes; using all of them", task, len(episodes)
            )
            picked = episodes
        elif selection == "random":
            picked = sorted(
                int(value)
                for value in rng.choice(episodes, size=episodes_per_task, replace=False)
            )
        else:
            picked = episodes[:episodes_per_task]
        for episode in picked:
            chosen.extend(by_task[task][episode])
    if not chosen:
        raise RuntimeError("No skills selected.")
    return sorted(chosen)


def suite_to_dataset_task_ids(
    dataset_dir: Path, suite_name: str, wanted: list[int]
) -> list[int]:
    """Translate LIBERO suite task ids into this dataset's own task_index.

    The skillset numbers tasks by the source dataset's meta/tasks.parquet
    (build_skill_dataset.py reads task_index there), while the episode-exact
    tools number them by the LIBERO suite. A filtered dataset drops tasks, so
    the two disagree -- libero_90_full_full has 73 tasks against the suite's
    order, and suite id 1 is dataset id 26. The task description is identical in
    both, so it bridges them without any episode-exact map.
    """
    import pandas as pd  # noqa: PLC0415
    from libero.libero import benchmark  # noqa: PLC0415

    table = pd.read_parquet(dataset_dir / "meta" / "tasks.parquet").reset_index()
    by_description = {str(row.task).strip(): int(row.task_index) for row in table.itertuples()}
    suite = benchmark.get_benchmark_dict()[suite_name]()
    translated, missing = [], []
    for suite_id in wanted:
        if not 0 <= suite_id < len(suite.tasks):
            missing.append(suite_id)
            continue
        dataset_id = by_description.get(str(suite.tasks[suite_id].language).strip())
        if dataset_id is None:
            missing.append(suite_id)
        else:
            translated.append(dataset_id)
    if missing:
        log.warning("suite task ids absent from this dataset: %s", missing)
    if not translated:
        raise RuntimeError(
            f"None of the suite task ids {wanted} exist in {dataset_dir.name}."
        )
    log.info(
        "task ids (suite -> dataset): %s",
        ", ".join(f"{a}->{b}" for a, b in zip(wanted, translated, strict=False)),
    )
    return translated


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-run", default="")
    parser.add_argument("--epoch-tag", default="")
    parser.add_argument("--skills-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-ids", required=True)
    parser.add_argument(
        "--task-id-space",
        choices=("dataset", "suite"),
        default="dataset",
        help="Numbering of --task-ids: the skillset's own task_index (default) or "
        "LIBERO suite ids, which the episode-exact tools use.",
    )
    parser.add_argument("--target-task", default="libero_90")
    parser.add_argument("--episode-ids", default="[]")
    parser.add_argument("--episodes-per-task", type=int, required=True)
    parser.add_argument("--episode-selection", choices=("first", "random"), default="first")
    parser.add_argument("--end-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    # Report options: the comparison is built in THIS process once every model
    # has finished, because a separate interpreter would pay the multi-minute
    # torch import all over again.
    parser.add_argument("--collection-dir", default="")
    parser.add_argument("--expected-labels", default="")
    parser.add_argument("--max-entries", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def maybe_report(args) -> None:
    """Build the comparison if this was the last model to finish."""
    if not args.collection_dir or not args.expected_labels:
        return
    import fsq_terminator_eval_report as report  # noqa: PLC0415

    report.maybe_build(
        Path(args.collection_dir),
        args.expected_labels.split(),
        max_entries=args.max_entries,
        max_samples=args.max_samples,
        seed=args.seed,
        fps=args.fps,
        frame_stride=args.frame_stride,
        render_video=not args.no_video,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True
    )
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "metrics" / "manifest.json"
    if manifest_path.is_file() and args.resume:
        existing = json.loads(manifest_path.read_text())
        if existing.get("completed"):
            log.info("Already complete, reusing: %s", manifest_path)
            maybe_report(args)
            return

    # Inline CUDA guard: importing torch costs minutes on this cluster's shared
    # venv, so the health check runs here instead of in a second interpreter.
    # The sbatch turns the marker into a requeue onto a different node.
    if args.device == "cuda" and not torch.cuda.is_available():
        marker = os.environ.get("LEROBOT_CUDA_GUARD_FAILURE_MARKER", "")
        if os.environ.get("LEROBOT_INLINE_CUDA_GUARD", "0") == "1" and marker:
            Path(marker).write_text("cuda unavailable\n", encoding="utf-8")
            raise SystemExit("CUDA is unavailable on this node; asking for a requeue.")
        log.warning("CUDA unavailable; falling back to CPU (this will be slow).")
    device = torch.device(
        args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model, cfg = load_model(args.model_path, str(device), "fsq", None)
    kind = terminator_kind(model.terminator)
    log.info("terminator variant: %s", kind)
    if kind == "none":
        raise SystemExit(
            f"{args.model_label}: this checkpoint has no terminator (reconstructor_only)."
        )

    # Index by filename first, select, then open only the chosen skills.
    entries = index_skill_files(Path(args.skills_dir))
    log.info("skillset holds %d skills", len(entries))
    raw_task_ids = args.task_ids.strip()
    task_ids = (
        None
        if raw_task_ids.lower() == "all"
        else [int(value) for value in json.loads(raw_task_ids)]
    )
    if task_ids is not None and args.task_id_space == "suite":
        task_ids = suite_to_dataset_task_ids(
            Path(args.dataset_dir), args.target_task, task_ids
        )
    picked = select_skills(
        entries,
        task_ids=task_ids,
        episodes_per_task=args.episodes_per_task,
        selection=args.episode_selection,
        seed=args.seed,
        episode_ids=[int(v) for v in json.loads(args.episode_ids)],
    )
    log.info("scoring %d skills of %d", len(picked), len(entries))
    segments, dec_states, metadata = load_selected_skills([entries[i] for i in picked])
    attach_episode_offsets(args.dataset_dir, metadata)
    lengths = [int(item["length"]) for item in metadata]

    raw_dataset = None
    if needs_images(kind):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

        raw_dataset = LeRobotDataset(
            repo_id=f"local/{Path(args.dataset_dir).name}",
            root=args.dataset_dir,
            video_keys_to_load=[THIRD_KEY, WRIST_KEY],
        )

    sub_segments, sub_states, sub_metadata, sub_lengths = (
        segments, dec_states, metadata, lengths
    )
    latents, tokens = batched_encode(model, sub_segments, sub_lengths, device, args.batch_size)

    terms, progs = run_terminator(
        model,
        kind,
        latents,
        sub_states,
        sub_metadata,
        sub_lengths,
        raw_dataset,
        device,
        skills_per_batch=max(1, args.batch_size // 8),
    )

    records = []
    for slot, item in enumerate(metadata):
        timing = skill_timing(terms[slot], args.end_threshold)
        records.append(
            {
                "uid": (
                    f"task{int(item['task_id']):02d}_ep{int(item['episode_id']):05d}"
                    f"_skill{int(item['skill_index']):02d}_token{int(tokens[slot]):04d}"
                    f"_f{int(item['frame_start']):04d}"
                ),
                "skillset_index": int(picked[slot]),
                "token": int(tokens[slot]),
                "task_id": int(item["task_id"]),
                "episode_id": int(item["episode_id"]),
                "skill_index": int(item["skill_index"]),
                "frame_start": int(item["frame_start"]),
                "frame_end": int(item["frame_end"]),
                "termination": [round(float(v), 5) for v in terms[slot]],
                "progress": [round(float(v), 5) for v in progs[slot]],
                **timing,
            }
        )

    manifest = {
        "format": "fsq_terminator_eval_v1",
        "completed": True,
        "label": args.model_label,
        "run_name": args.model_run,
        "epoch_tag": args.epoch_tag,
        "model_path": str(Path(args.model_path).resolve()),
        # The report renders GT frames once for every model, so it needs to know
        # where the shared skills and dataset live without re-reading the config.
        "skills_dir": str(Path(args.skills_dir).resolve()),
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "terminator_kind": kind,
        "termination_only": bool(getattr(model.terminator, "termination_only", False)),
        "fsq_levels": [int(level) for level in cfg.fsq_levels],
        "codebook_size": int(model.fsq.codebook_size),
        "end_threshold": float(args.end_threshold),
        "task_id_space": args.task_id_space,
        "seed": int(args.seed),
        "summary": aggregate(records),
        "records": records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    summary = manifest["summary"]
    log.info(
        "[%s] %s | skills=%d |err|=%.2f early=%.1f%% late=%.1f%% no-fire=%.1f%% <=3=%.1f%%",
        args.model_label,
        kind,
        summary["skills"],
        summary["timing_abs_mean"],
        100.0 * summary["early_rate"],
        100.0 * summary["late_rate"],
        100.0 * summary["no_fire_rate"],
        100.0 * summary["within_3_rate"],
    )
    log.info("manifest -> %s", manifest_path)
    maybe_report(args)


if __name__ == "__main__":
    main()
