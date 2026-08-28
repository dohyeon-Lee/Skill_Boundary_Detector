#!/usr/bin/env python3
"""Extract GT start/end frames of FSQ-labelled skill segments from dataset videos.

No simulator: the filtered LeRobot dataset already stores every camera frame the
demos produced, and skill occurrences index exactly those frames. Each occurrence
therefore costs two video-frame decodes. The report groups the image pairs by FSQ
code to show which skills the codebook collected.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from lerobot.datasets.video_utils import decode_video_frames
from lerobot.utils.random_utils import set_seed

_HERE = Path(__file__).resolve().parent
_EXACT_DATA_SRC = _HERE.parents[2] / "train_skillVLA" / "terminator_eval" / "src"
sys.path.insert(0, str(_EXACT_DATA_SRC))

from skill_data import SkillEvaluationDataset  # noqa: E402

from fsq_gt_replay_report import (  # noqa: E402
    maybe_build_compare,
    maybe_merge_collection,
    maybe_merge_chunks,
    report_payload,
    write_html_report,
)

log = logging.getLogger("fsq_gt_replay")


def _libero_task_descriptions(suite_name: str) -> dict[int, str]:
    """Task id -> language string for one LIBERO suite.

    Inlined rather than imported from lerobot.scripts.lerobot_skillvla_eval:
    that module pulls in lerobot.policies.factory (every policy, transformers,
    ...) at import time, which costs minutes on this cluster's shared venv --
    for a replay that never touches a policy.
    """
    try:
        from libero.libero import benchmark  # noqa: PLC0415

        suite = benchmark.get_benchmark_dict()[suite_name]()
        return {int(index): str(task.language) for index, task in enumerate(suite.tasks)}
    except Exception as error:  # noqa: BLE001 - descriptions are cosmetic
        log.warning("Could not load LIBERO task descriptions for %s: %s", suite_name, error)
        return {}


VIDEO_KEY = "observation.images.image"


def _fsq_levels(model_path: Path) -> list[int]:
    """Read the codebook shape from current FSQ or BSQ checkpoints.

    FSQ variants store `cfg.fsq_levels` directly. BSQ checkpoints carry
    quantizer="bsq" and an unused fsq_levels placeholder; their bit indexing is
    mathematically identical to FSQ levels [2]*code_dim (same strides), so we
    report exactly that — token validation and codebook stats stay correct.
    """
    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    config = checkpoint.get("cfg")

    def read(key: str):
        return config.get(key) if isinstance(config, dict) else getattr(config, key, None)

    if str(read("quantizer") or "fsq") == "bsq":
        levels = [2] * int(read("bsq_code_dim"))
    else:
        levels = read("fsq_levels")
    del checkpoint
    gc.collect()
    if not levels:
        raise ValueError(f"FSQ checkpoint carries no fsq_levels: {model_path}")
    return [int(value) for value in levels]


def _available_task_ids(dataset, *, episodes_per_task: int) -> list[int]:
    """Tasks the exact-episode dataset can actually serve, mirroring select_episodes."""
    counts: dict[int, int] = {}
    for episode_id, source in dataset.sources.items():
        if (
            episode_id in dataset._rows_by_episode
            and episode_id in dataset.episode_meta.index
        ):
            counts[source.task_id] = counts.get(source.task_id, 0) + 1
    return sorted(
        task_id for task_id, count in counts.items() if count >= episodes_per_task
    )


def _filter_task_ids(
    requested: list[int] | None, available: list[int]
) -> list[int]:
    """Keep only dataset-available tasks; None means every available task."""
    if requested is None:
        selected = list(available)
    else:
        available_set = set(available)
        selected = [task_id for task_id in requested if task_id in available_set]
        skipped = [task_id for task_id in requested if task_id not in available_set]
        if skipped:
            log.warning(
                "Skipping task_ids without enough episode-exact skill episodes: %s",
                skipped,
            )
    if not selected:
        raise RuntimeError(
            "No requested task has enough episode-exact skill episodes in the dataset."
        )
    return selected


def _frame_pair(occurrence, episode_length: int) -> tuple[int, int]:
    """Start frame and the frame right after the last GT action (clamped for the
    final skill of an episode, whose next frame was never recorded)."""
    if (
        occurrence.frame_end <= occurrence.frame_start
        or occurrence.frame_start >= episode_length
    ):
        raise RuntimeError(f"Skill occurrence has no GT frame: {occurrence.uid}")
    return occurrence.frame_start, min(occurrence.frame_end, episode_length - 1)


class _EpisodeFrameReader:
    """Decode requested frames once and reuse them across checkpoint replays."""

    def __init__(self, dataset_dir: Path, *, video_key: str = VIDEO_KEY) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.video_key = video_key
        info = json.loads((self.dataset_dir / "meta" / "info.json").read_text())
        self.fps = float(info["fps"])
        self.path_template = str(info["video_path"])
        self._frame_cache: dict[tuple[int, int], np.ndarray] = {}
        files = sorted((self.dataset_dir / "meta" / "episodes").glob("**/*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No episode metadata under {self.dataset_dir / 'meta/episodes'}"
            )
        columns = [
            "episode_index",
            "length",
            f"videos/{video_key}/chunk_index",
            f"videos/{video_key}/file_index",
            f"videos/{video_key}/from_timestamp",
        ]
        self.index = pd.concat(
            [pd.read_parquet(path, columns=columns) for path in files],
            ignore_index=True,
        ).set_index("episode_index", drop=False)

    def episode_length(self, episode_id: int) -> int:
        return int(self.index.loc[int(episode_id), "length"])

    def frames(self, episode_id: int, frame_indices) -> dict[int, np.ndarray]:
        episode_id = int(episode_id)
        ordered = sorted({int(index) for index in frame_indices})
        missing = [
            index for index in ordered if (episode_id, index) not in self._frame_cache
        ]
        if not missing:
            return {
                index: self._frame_cache[(episode_id, index)] for index in ordered
            }

        row = self.index.loc[int(episode_id)]
        video_path = self.dataset_dir / self.path_template.format(
            video_key=self.video_key,
            chunk_index=int(row[f"videos/{self.video_key}/chunk_index"]),
            file_index=int(row[f"videos/{self.video_key}/file_index"]),
        )
        start_timestamp = float(row[f"videos/{self.video_key}/from_timestamp"])
        timestamps = [start_timestamp + index / self.fps for index in missing]
        decoded = decode_video_frames(
            video_path, timestamps, tolerance_s=0.5 / self.fps
        )
        images = (
            (decoded.clamp(0.0, 1.0) * 255.0)
            .round()
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        for position, index in enumerate(missing):
            # Own each image independently: retaining a small view must not pin
            # the complete decoded batch after the remaining frames are unused.
            self._frame_cache[(episode_id, index)] = images[position].copy()
        return {index: self._frame_cache[(episode_id, index)] for index in ordered}

    @property
    def cached_frame_count(self) -> int:
        return len(self._frame_cache)

    @property
    def cached_bytes(self) -> int:
        return sum(frame.nbytes for frame in self._frame_cache.values())


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


def _record_is_complete(manifest: dict, occurrence, output_dir: Path) -> bool:
    existing = manifest["records"].get(occurrence.uid)
    if existing is None:
        return False
    return all(
        key in existing and (output_dir / existing[key]).is_file()
        for key in ("start_image_path", "final_image_path")
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--latents-path", type=Path, default=None)
    parser.add_argument("--skill-dataset-dir", type=Path, required=True)
    # Empty means episode_source=dataset: episodes are grouped by the dataset's
    # own task table instead of the rendered episode-exact map.
    parser.add_argument("--eval-init-states-path", default="")
    parser.add_argument("--original-dataset-dir", default="")
    parser.add_argument("--target-task", required=True)
    parser.add_argument("--task-ids", required=True)
    parser.add_argument("--episode-ids", default="[]")
    parser.add_argument("--episodes-per-task", type=int, required=True)
    parser.add_argument("--episode-selection", choices=("first", "random"), required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--compare-dir", default="")
    parser.add_argument("--compare-collection-dirs", default="")
    parser.add_argument("--expected-epoch-tags", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--report-title", default="FSQ GT skill replay")
    parser.add_argument("--epoch-tag", default="")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--plan-path",
        default="",
        help='JSON list of {"model_path","latents_path","output_dir","epoch_tag"}. '
        "Replaying several checkpoints in ONE process amortizes the import cost, "
        "which on this cluster is minutes against seconds of actual replay.",
    )
    return parser.parse_args()


def run_one(args, shared: dict) -> None:
    """Replay one checkpoint. ``shared`` caches what does not vary between them."""
    set_seed(args.seed)
    raw_task_ids = args.task_ids.strip()
    requested_task_ids = (
        None
        if raw_task_ids.lower() == "all"
        else [int(value) for value in json.loads(raw_task_ids)]
    )
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
        eval_init_states_path=(
            Path(args.eval_init_states_path) if args.eval_init_states_path else None
        ),
        original_dataset_dir=args.original_dataset_dir or None,
        suite_name=args.target_task,
    )
    log.info(
        "episode source: %s",
        "dataset task table" if not args.eval_init_states_path else "episode-exact map",
    )
    task_ids = _filter_task_ids(
        requested_task_ids,
        _available_task_ids(dataset, episodes_per_task=args.episodes_per_task),
    )
    log.info("evaluating %d task(s): %s", len(task_ids), task_ids)
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
        "format": "fsq_gt_replay_v3",
        "model_path": str(args.model_path.resolve()),
        "latents_path": str(args.latents_path.resolve()),
        "target_task": args.target_task,
        "selected_episodes": {str(key): value for key, value in selected.items()},
        "seed": args.seed,
    }
    request = {
        "format": "fsq_gt_replay_request_v1",
        "episode_source": "exact" if args.eval_init_states_path else "dataset",
        "target_task": args.target_task,
        "task_ids": requested_task_ids,
        "episode_ids": episode_ids,
        "episodes_per_task": args.episodes_per_task,
        "episode_selection": args.episode_selection,
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

    # Cosmetic report metadata stays outside the replay signature. This lets a
    # resume run rename the HTML without invalidating extracted image pairs.
    manifest["request"] = request
    manifest["model_name"] = args.model_name or args.run_name
    manifest["report_title"] = args.report_title

    levels = shared.get("levels")
    if levels is None:
        levels = shared["levels"] = _fsq_levels(args.model_path)
    max_token = int(np.prod(levels))
    invalid_tokens = sorted(
        {occurrence.token for occurrence in occurrences if not 0 <= occurrence.token < max_token}
    )
    if invalid_tokens:
        raise ValueError(
            f"Latent artifact contains tokens outside FSQ{levels}: {invalid_tokens}."
        )
    manifest["levels"] = levels
    # The latents artifact encodes the full training skillset, so its token
    # distribution is this checkpoint's codebook usage over all training data:
    # per-code counts for the full-data cube, the distinct count, and the
    # entropy-based effective count used to normalize the cohesion metric.
    train_tokens = np.asarray(np.load(str(args.latents_path))["tokens"], dtype=np.int64)
    token_counts = np.bincount(train_tokens, minlength=max_token)
    probabilities = token_counts[token_counts > 0] / train_tokens.size
    manifest["train_codebook_counts"] = [int(count) for count in token_counts]
    manifest["train_codebook_used"] = int((token_counts > 0).sum())
    manifest["train_codebook_effective"] = float(
        np.exp(-(probabilities * np.log(probabilities)).sum())
    )
    _atomic_manifest(manifest_path, manifest)

    reader = shared.get("reader")
    if reader is None:
        reader = shared["reader"] = _EpisodeFrameReader(Path(args.skill_dataset_dir))
    # Dataset-sourced episodes already carry their own task strings, which cover
    # every task even when the dataset spans several LIBERO suites.
    if "descriptions" not in shared:
        shared["descriptions"] = _libero_task_descriptions(args.target_task)
    descriptions = dataset.task_descriptions or shared["descriptions"]
    by_episode: dict[int, list] = defaultdict(list)
    for occurrence in occurrences:
        by_episode[occurrence.episode_id].append(occurrence)

    processed = 0
    for episode_id in sorted(by_episode):
        pending = [
            occurrence
            for occurrence in by_episode[episode_id]
            if not _record_is_complete(manifest, occurrence, output_dir)
        ]
        if not pending:
            continue
        episode_length = reader.episode_length(episode_id)
        frame_pairs = {
            occurrence.uid: _frame_pair(occurrence, episode_length)
            for occurrence in pending
        }
        needed_frames = sorted(
            {frame for pair in frame_pairs.values() for frame in pair}
        )
        images = reader.frames(episode_id, needed_frames)
        source = dataset.sources[episode_id]
        for occurrence in pending:
            processed += 1
            log.info(
                "[%d] token=%d task=%d episode=%d skill=%d frames=[%d,%d)",
                processed,
                occurrence.token,
                occurrence.task_id,
                occurrence.episode_id,
                occurrence.skill_index,
                occurrence.frame_start,
                occurrence.frame_end,
            )
            start_frame, final_frame = frame_pairs[occurrence.uid]
            image_dir = (
                Path("images")
                / f"task_{occurrence.task_id:02d}"
                / f"token_{occurrence.token:04d}"
            )
            relative_start_image = image_dir / f"{occurrence.uid}_start.png"
            relative_final_image = image_dir / f"{occurrence.uid}_final.png"
            _write_image(output_dir / relative_start_image, images[start_frame])
            _write_image(output_dir / relative_final_image, images[final_frame])
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
                "scene_file": source.scene_file,
                "demo": source.demo,
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
    if collection_path is not None and args.compare_dir:
        compare_path = maybe_build_compare(
            json.loads(args.compare_collection_dirs),
            output_dir=args.compare_dir,
        )
        if compare_path is not None:
            log.info("FSQ model comparison report: %s", compare_path)
        else:
            log.info("Run complete; waiting for the remaining runs to compare.")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    if args.plan_path:
        entries = json.loads(Path(args.plan_path).read_text())
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{args.plan_path} must hold a non-empty list of entries.")
    else:
        missing = [
            name
            for name, value in (
                ("--model-path", args.model_path),
                ("--latents-path", args.latents_path),
                ("--output-dir", args.output_dir),
                ("--epoch-tag", args.epoch_tag),
            )
            if not value
        ]
        if missing:
            raise SystemExit(f"Pass --plan-path, or all of {missing}.")
        entries = [
            {
                "model_path": str(args.model_path),
                "latents_path": str(args.latents_path),
                "output_dir": str(args.output_dir),
                "epoch_tag": args.epoch_tag,
            }
        ]
    # One process, many checkpoints: imports and the video frame reader are paid
    # once instead of once per checkpoint.
    shared: dict = {}
    for index, entry in enumerate(entries, start=1):
        log.info("[%d/%d] replaying %s", index, len(entries), entry["epoch_tag"])
        run_one(
            argparse.Namespace(
                **{
                    **vars(args),
                    "model_path": Path(entry["model_path"]),
                    "latents_path": Path(entry["latents_path"]),
                    "output_dir": Path(entry["output_dir"]),
                    "epoch_tag": entry["epoch_tag"],
                }
            ),
            shared,
        )
    reader = shared.get("reader")
    if reader is not None:
        log.info(
            "decoded-frame cache: %d frames, %.2f GiB",
            reader.cached_frame_count,
            reader.cached_bytes / (1024**3),
        )


if __name__ == "__main__":
    main()
