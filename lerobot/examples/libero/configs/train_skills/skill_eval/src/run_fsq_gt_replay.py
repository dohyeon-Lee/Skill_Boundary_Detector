#!/usr/bin/env python3
"""Render GT start/end states of FSQ-labelled skill segments in exact LIBERO scenes.

No policy, no action replay, no terminator: every frame's full MuJoCo state is
stored in the original LIBERO HDF5, so each skill occurrence only needs its start
and end states restored and rendered. The report groups the image pairs by FSQ
code to show which skills the codebook collected.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs
from lerobot.scripts.lerobot_skillvla_eval import _libero_task_descriptions
from lerobot.utils.random_utils import set_seed

from FSQ import _checkpoint_config

_HERE = Path(__file__).resolve().parent
_EXACT_DATA_SRC = _HERE.parents[2] / "train_skillVLA" / "terminator_eval" / "src"
sys.path.insert(0, str(_EXACT_DATA_SRC))

from skill_data import SkillEvaluationDataset  # noqa: E402

from fsq_gt_replay_report import (  # noqa: E402
    maybe_merge_collection,
    maybe_merge_chunks,
    report_payload,
    write_html_report,
)

log = logging.getLogger("fsq_gt_replay")


def _restore_state(base_env, state: np.ndarray) -> None:
    base_env._env.reset()
    base_env._env.set_init_state(np.asarray(state, dtype=np.float64))


def _render(base_env) -> np.ndarray:
    return np.asarray(base_env.render(), dtype=np.uint8).copy()


def _fsq_levels(model_path: Path) -> list[int]:
    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    config = _checkpoint_config(checkpoint)
    del checkpoint
    gc.collect()
    return [int(value) for value in config.fsq_levels]


def _end_state(aligned, occurrence) -> np.ndarray:
    """State right after the occurrence's last GT action.

    That state is the start state of the next filtered frame; the final skill of
    an episode has no next frame, so fall back to the next original demo state
    (clamped to the last one the demo recorded).
    """
    if occurrence.frame_end < len(aligned.original_action_indices):
        return aligned.state_at(occurrence.frame_end)
    original_frame = min(
        aligned.original_frame_at(occurrence.frame_end - 1) + 1,
        len(aligned.original_states) - 1,
    )
    return np.asarray(aligned.original_states[original_frame], dtype=np.float64).copy()


def _capture_segment(*, base_env, aligned, occurrence) -> dict:
    if occurrence.frame_end <= occurrence.frame_start:
        raise RuntimeError(f"Skill occurrence has no GT frame: {occurrence.uid}")
    _restore_state(base_env, aligned.state_at(occurrence.frame_start))
    start_frame = _render(base_env)
    _restore_state(base_env, _end_state(aligned, occurrence))
    return {"start_frame": start_frame, "final_frame": _render(base_env)}


def _write_image(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(temporary, format="PNG")
    temporary.replace(path)


def _atomic_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--latents-path", type=Path, required=True)
    parser.add_argument("--skill-dataset-dir", type=Path, required=True)
    parser.add_argument("--eval-init-states-path", type=Path, required=True)
    parser.add_argument("--original-dataset-dir", type=Path, required=True)
    parser.add_argument("--target-task", required=True)
    parser.add_argument("--task-ids", required=True)
    parser.add_argument("--episode-ids", default="[]")
    parser.add_argument("--episodes-per-task", type=int, required=True)
    parser.add_argument("--episode-selection", choices=("first", "random"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--expected-epoch-tags", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epoch-tag", required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)
    task_ids = [int(value) for value in json.loads(args.task_ids)]
    episode_ids = [int(value) for value in json.loads(args.episode_ids)]
    expected_epoch_tags = [
        str(value) for value in json.loads(args.expected_epoch_tags)
    ]
    if not expected_epoch_tags or args.epoch_tag not in expected_epoch_tags:
        raise ValueError(
            f"Current checkpoint {args.epoch_tag!r} is not in expected epoch tags "
            f"{expected_epoch_tags}."
        )
    dataset = SkillEvaluationDataset(
        skill_dataset_dir=args.skill_dataset_dir,
        skill_latents_path=args.latents_path,
        eval_init_states_path=args.eval_init_states_path,
        original_dataset_dir=args.original_dataset_dir,
        suite_name=args.target_task,
    )
    selected = dataset.select_episodes(
        task_ids=task_ids,
        episodes_per_task=args.episodes_per_task,
        selection=args.episode_selection,
        seed=args.seed,
        explicit_episode_ids=episode_ids,
    )
    all_occurrences = dataset.occurrences(selected)
    if not all_occurrences:
        raise RuntimeError("No FSQ skill occurrences were found in the selected exact episodes.")
    if args.worker_count <= 0 or not 0 <= args.worker_index < args.worker_count:
        raise ValueError(f"Invalid worker {args.worker_index}/{args.worker_count}.")
    occurrences = all_occurrences[args.worker_index :: args.worker_count]
    if not occurrences:
        raise RuntimeError(
            f"Worker {args.worker_index}/{args.worker_count} received no occurrences."
        )

    output_dir = args.output_dir
    manifest_path = (
        output_dir / "metrics" / "manifest.json"
        if args.worker_count == 1
        else output_dir / "metrics" / "chunks" / f"chunk_{args.worker_index:03d}.json"
    )
    signature = {
        "format": "fsq_gt_replay_v2",
        "model_path": str(args.model_path.resolve()),
        "latents_path": str(args.latents_path.resolve()),
        "target_task": args.target_task,
        "selected_episodes": {str(key): value for key, value in selected.items()},
        "seed": args.seed,
    }
    if manifest_path.is_file():
        if not args.resume:
            raise FileExistsError(
                f"Replay output already exists: {manifest_path}. Set resume: true or change output_name."
            )
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("signature") != signature:
            raise ValueError("resume=true but the existing replay signature differs.")
    else:
        manifest = {
            "signature": signature,
            "run_name": args.run_name,
            "epoch_tag": args.epoch_tag,
            "chunk_index": args.worker_index,
            "chunk_count": args.worker_count,
            "completed": False,
            "records": {},
        }
        _atomic_manifest(manifest_path, manifest)

    levels = _fsq_levels(args.model_path)
    max_token = int(np.prod(levels))
    invalid_tokens = sorted(
        {occurrence.token for occurrence in occurrences if not 0 <= occurrence.token < max_token}
    )
    if invalid_tokens:
        raise ValueError(
            f"Latent artifact contains tokens outside FSQ{levels}: {invalid_tokens}."
        )
    manifest["levels"] = levels
    # The latents artifact encodes the full training skillset, so its distinct
    # token count is this checkpoint's codebook usage over all training data.
    train_tokens = np.asarray(np.load(str(args.latents_path))["tokens"], dtype=np.int64)
    manifest["train_codebook_used"] = int(np.unique(train_tokens).size)
    _atomic_manifest(manifest_path, manifest)

    worker_task_ids = sorted({occurrence.task_id for occurrence in occurrences})
    env_config = LiberoEnv(
        task=args.target_task,
        task_ids=worker_task_ids,
        fps=20,
        init_states=False,
        max_parallel_tasks=1,
    )
    envs = make_env(env_config, n_envs=1, use_async_envs=False)
    descriptions = _libero_task_descriptions(args.target_task)
    try:
        for index, occurrence in enumerate(occurrences, start=1):
            existing = manifest["records"].get(occurrence.uid)
            if existing is not None:
                artifact_keys = ("start_image_path", "final_image_path")
                if all(
                    key in existing and (output_dir / existing[key]).is_file()
                    for key in artifact_keys
                ):
                    continue
            aligned = dataset.load_aligned_episode(occurrence.episode_id)
            base_env = envs[args.target_task][occurrence.task_id].envs[0].unwrapped
            log.info(
                "[%d/%d] token=%d task=%d episode=%d skill=%d frames=[%d,%d)",
                index,
                len(occurrences),
                occurrence.token,
                occurrence.task_id,
                occurrence.episode_id,
                occurrence.skill_index,
                occurrence.frame_start,
                occurrence.frame_end,
            )
            capture = _capture_segment(
                base_env=base_env, aligned=aligned, occurrence=occurrence
            )
            image_dir = (
                Path("images")
                / f"task_{occurrence.task_id:02d}"
                / f"token_{occurrence.token:04d}"
            )
            relative_start_image = image_dir / f"{occurrence.uid}_start.png"
            relative_final_image = image_dir / f"{occurrence.uid}_final.png"
            _write_image(output_dir / relative_start_image, capture["start_frame"])
            _write_image(output_dir / relative_final_image, capture["final_frame"])
            manifest["records"][occurrence.uid] = {
                "uid": occurrence.uid,
                "token": occurrence.token,
                "task_id": occurrence.task_id,
                "task_description": descriptions.get(occurrence.task_id, ""),
                "episode_id": occurrence.episode_id,
                "skill_index": occurrence.skill_index,
                "frame_start": occurrence.frame_start,
                "frame_end": occurrence.frame_end,
                "length": occurrence.length,
                "scene_file": aligned.source.scene_file,
                "demo": aligned.source.demo,
                "start_image_path": relative_start_image.as_posix(),
                "final_image_path": relative_final_image.as_posix(),
            }
            _atomic_manifest(manifest_path, manifest)
        manifest["completed"] = True
        _atomic_manifest(manifest_path, manifest)
        if args.worker_count == 1:
            report_path = write_html_report(output_dir, report_payload(manifest))
        else:
            report_path = maybe_merge_chunks(
                output_dir, expected_chunks=args.worker_count
            )
        collection_path = None
        if report_path is not None:
            collection_path = maybe_merge_collection(
                args.collection_dir, expected_epoch_tags=expected_epoch_tags
            )
            log.info("FSQ checkpoint replay report: %s", report_path)
        if collection_path is not None:
            log.info("FSQ combined replay report: %s", collection_path)
        elif report_path is not None:
            log.info("Checkpoint complete; waiting for the remaining checkpoints.")
        else:
            log.info("Worker complete; waiting for the remaining chunks.")
    finally:
        close_envs(envs)
        gc.collect()


if __name__ == "__main__":
    main()
