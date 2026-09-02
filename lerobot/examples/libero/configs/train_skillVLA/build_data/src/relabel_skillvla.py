#!/usr/bin/env python3
"""Relabel a completed SkillVLA dataset at its preserved GT skill boundaries.

The source dataset remains immutable.  A frozen auxiliary skill predictor is
queried once at every row with ``skill_ds == 0`` and its prediction replaces
the corresponding entry of the episode-wide ``skill_sequence``.  Boundary,
length, action, state, task, and image data are unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class Segment:
    dataset_index: int
    episode_index: int
    task_index: int
    skill_index: int
    frame_start: int
    original_code: int
    real_skill_count: int


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _link_or_copy(source: str, destination: str) -> str:
    """Hard-link immutable dataset payloads, falling back to an ordinary copy."""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def discover_segments(dataset_dir: Path, *, num_embeddings: int) -> list[Segment]:
    """Read only parquet metadata columns and return every canonical skill start."""
    columns = (
        "index",
        "episode_index",
        "task_index",
        "frame_index",
        "skill_index",
        "skill_sequence",
        "skill_sequence_len",
        "skill_initial_frame",
        "skill_ds",
    )
    segments: list[Segment] = []
    data_files = sorted((dataset_dir / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No SkillVLA parquet files found under {dataset_dir / 'data'}.")
    for path in data_files:
        table = pq.read_table(path, columns=list(columns))
        missing = sorted(set(columns).difference(table.column_names))
        if missing:
            raise ValueError(f"SkillVLA parquet lacks columns {missing}: {path}")
        rows = table.to_pydict()
        for row_index, distance in enumerate(rows["skill_ds"]):
            if int(distance) != 0:
                continue
            skill_index = int(rows["skill_index"][row_index])
            sequence_length = int(rows["skill_sequence_len"][row_index])
            real_skill_count = sequence_length - 1  # final valid token is EOS
            # Datasets may retain episodes excluded from training as zero-filled
            # rows. They have EOS only and intentionally contain no skill to
            # query or relabel.
            if real_skill_count == 0:
                continue
            if not 0 <= skill_index < real_skill_count:
                raise ValueError(
                    "Invalid skill start row: "
                    f"episode={rows['episode_index'][row_index]}, "
                    f"skill_index={skill_index}, sequence_len={sequence_length}."
                )
            initial_frames = rows["skill_initial_frame"][row_index]
            frame_index = int(rows["frame_index"][row_index])
            if frame_index != int(initial_frames[skill_index]):
                # ``skill_ds`` is also zero before a rare first skill whose
                # canonical start is later than frame zero. Query only the
                # recorded canonical boundary.
                continue
            sequence = np.asarray(rows["skill_sequence"][row_index], dtype=np.int64)
            code = int(sequence[skill_index])
            if not 0 <= code < num_embeddings:
                raise ValueError(
                    f"Episode {rows['episode_index'][row_index]} skill {skill_index} "
                    f"has non-FSQ code {code}."
                )
            segments.append(
                Segment(
                    dataset_index=int(rows["index"][row_index]),
                    episode_index=int(rows["episode_index"][row_index]),
                    task_index=int(rows["task_index"][row_index]),
                    skill_index=skill_index,
                    frame_start=frame_index,
                    original_code=code,
                    real_skill_count=real_skill_count,
                )
            )

    segments.sort(key=lambda item: (item.episode_index, item.skill_index))
    seen: set[tuple[int, int]] = set()
    grouped: dict[int, list[Segment]] = {}
    for segment in segments:
        key = (segment.episode_index, segment.skill_index)
        if key in seen:
            raise ValueError(f"Duplicate canonical skill start for episode/skill {key}.")
        seen.add(key)
        grouped.setdefault(segment.episode_index, []).append(segment)
    for episode_index, episode_segments in grouped.items():
        indices = [segment.skill_index for segment in episode_segments]
        expected_count = episode_segments[0].real_skill_count
        if indices != list(range(expected_count)):
            raise ValueError(
                f"Episode {episode_index} skill starts are incomplete: "
                f"found={indices}, expected={list(range(expected_count))}."
            )
        if any(segment.real_skill_count != expected_count for segment in episode_segments):
            raise ValueError(f"Episode {episode_index} has inconsistent skill_sequence_len values.")
    return segments


def normalized_latents_from_codes(codes: np.ndarray, levels: list[int]) -> np.ndarray:
    """Decode flat little-endian FSQ codes to their normalized grid points."""
    codes = np.asarray(codes, dtype=np.int64).reshape(-1, 1)
    levels_array = np.asarray(levels, dtype=np.int64)
    strides = np.ones_like(levels_array)
    if len(levels_array) > 1:
        strides[1:] = np.cumprod(levels_array[:-1])
    level_ids = (codes // strides[None, :]) % levels_array[None, :]
    half = (levels_array.astype(np.float32) - 1.0) / 2.0
    return ((level_ids.astype(np.float32) - half[None, :]) / half[None, :]).astype(
        np.float32
    )


def _prediction_map(
    segments: list[Segment], predictions: np.ndarray
) -> dict[tuple[int, int], int]:
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if len(predictions) != len(segments):
        raise ValueError(
            f"Prediction count {len(predictions)} does not match segment count {len(segments)}."
        )
    return {
        (segment.episode_index, segment.skill_index): int(code)
        for segment, code in zip(segments, predictions, strict=True)
    }


def rewrite_skill_sequences(
    dataset_dir: Path,
    *,
    segments: list[Segment],
    predictions: np.ndarray,
) -> None:
    """Replace only real skill entries; EOS/PAD and every boundary field stay fixed."""
    prediction_by_segment = _prediction_map(segments, predictions)
    sequence_by_episode: dict[int, list[int]] = {}
    for segment in segments:
        sequence_by_episode.setdefault(
            segment.episode_index, [0] * segment.real_skill_count
        )[segment.skill_index] = prediction_by_segment[
            (segment.episode_index, segment.skill_index)
        ]

    for path in sorted((dataset_dir / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        episode_ids = table["episode_index"].to_pylist()
        sequence_column = table["skill_sequence"]
        sequence_type = table.schema.field("skill_sequence").type
        updated = []
        for episode_index, original in zip(
            episode_ids, sequence_column.to_pylist(), strict=True
        ):
            sequence = list(original)
            replacement = sequence_by_episode.get(int(episode_index))
            if replacement is None:
                # Preserve zero-filled episodes with no canonical skills.
                updated.append(sequence)
                continue
            sequence[: len(replacement)] = replacement
            updated.append(sequence)
        table = table.set_column(
            table.schema.get_field_index("skill_sequence"),
            "skill_sequence",
            pa.array(updated, type=sequence_type),
        )
        temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
        temporary.replace(path)


def rewrite_skill_latents(
    source_path: Path,
    output_path: Path,
    *,
    segments: list[Segment],
    predictions: np.ndarray,
    levels: list[int],
) -> None:
    """Keep the analysis artifact aligned with the relabeled parquet codes."""
    with np.load(source_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    required = {"episode_id", "skill_index", "frame_start", "tokens", "latents"}
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"skill_latents.npz lacks {missing}: {source_path}")
    prediction_by_segment = _prediction_map(segments, predictions)
    segment_by_key = segments_by_key(segments)
    new_tokens = np.asarray(arrays["tokens"], dtype=np.int32).copy()
    for index, (episode, skill_index, frame_start) in enumerate(
        zip(
            arrays["episode_id"],
            arrays["skill_index"],
            arrays["frame_start"],
            strict=True,
        )
    ):
        key = (int(episode), int(skill_index))
        if key not in prediction_by_segment:
            raise ValueError(f"skill_latents.npz has unknown segment {key}.")
        source_segment = segment_by_key[key]
        if int(frame_start) != source_segment.frame_start:
            raise ValueError(
                f"skill_latents frame mismatch for {key}: "
                f"npz={int(frame_start)}, parquet={source_segment.frame_start}."
            )
        new_tokens[index] = prediction_by_segment[key]
    if len(new_tokens) != len(segments):
        raise ValueError(
            f"skill_latents has {len(new_tokens)} rows but parquet has {len(segments)} segments."
        )
    arrays["tokens"] = new_tokens
    arrays["latents"] = normalized_latents_from_codes(new_tokens, levels)
    temporary = output_path.with_name(f"{output_path.name}.tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output_path)


def segments_by_key(segments: Iterable[Segment]) -> dict[tuple[int, int], Segment]:
    return {(segment.episode_index, segment.skill_index): segment for segment in segments}


def _predict(
    *,
    dataset_dir: Path,
    predictor_path: Path,
    tokenizer_path: Path,
    segments: list[Segment],
    batch_size: int,
) -> np.ndarray:
    """Run the checkpoint-owned predictor and preprocessing contract on skill starts."""
    import torch
    from torch.utils.data._utils.collate import default_collate
    from tqdm import tqdm

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.skill_expert.modeling_skill_expert import (
        _load_complete_predictor_parameters,
    )
    from lerobot.policies.skill_expert.modeling_skill_predictor import (
        FrozenVLMSkillPredictor,
    )
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("SkillVLA relabeling requires a CUDA GPU.")
    device = torch.device("cuda")
    config = PreTrainedConfig.from_pretrained(str(predictor_path))
    if config.type not in {"skill_aux", "skill_expert"} or not bool(
        getattr(config, "train_skill_predictor", False)
    ):
        raise ValueError(f"Checkpoint is not a trained skill predictor: {predictor_path}")
    config.device = str(device)
    config.tokenizer_path = str(tokenizer_path)
    dtype = (
        torch.bfloat16
        if str(getattr(config, "dtype", "float32")) == "bfloat16"
        else torch.float32
    )
    predictor = FrozenVLMSkillPredictor(config).to(dtype=dtype)
    _load_complete_predictor_parameters(predictor, predictor_path)
    predictor.to(device=device).requires_grad_(False).eval()

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(predictor_path),
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "tokenizer_processor": {"tokenizer_name": str(tokenizer_path)},
        },
    )
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    dataset = LeRobotDataset(repo_id=str(info["repo_id"]), root=dataset_dir)
    image_keys = (
        "observation.images.image",
        "observation.images.wrist_image",
    )
    predictions: list[np.ndarray] = []
    for offset in tqdm(
        range(0, len(segments), batch_size), desc="predict skill starts"
    ):
        batch_segments = segments[offset : offset + batch_size]
        samples = []
        for segment in batch_segments:
            raw = dataset[segment.dataset_index]
            samples.append(
                {
                    image_keys[0]: raw[image_keys[0]],
                    image_keys[1]: raw[image_keys[1]],
                    "observation.state": raw["observation.state"],
                    "task": raw["task"],
                }
            )
        processed = preprocessor(default_collate(samples))
        with torch.inference_mode():
            codes = predictor.predict(
                [processed[key].to(device) for key in image_keys],
                processed[OBS_LANGUAGE_TOKENS].to(device),
                processed[OBS_LANGUAGE_ATTENTION_MASK].to(device),
            )
        predictions.append(codes.detach().cpu().numpy().astype(np.int64))
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.int64)


def _provenance_matches(path: Path, expected: dict) -> bool:
    provenance_path = path / "relabel_provenance.json"
    if not provenance_path.is_file():
        return False
    try:
        actual = json.loads(provenance_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    required_outputs = (
        path / "skillvla" / "meta" / "info.json",
        path / "skill_relabel.npz",
        path / "FSQ.pt",
    )
    return all(actual.get(key) == value for key, value in expected.items()) and all(
        output.is_file() for output in required_outputs
    )


def build_relabeled_dataset(args: argparse.Namespace) -> None:
    source_run = Path(args.source_run_dir).resolve()
    output_run = Path(args.output_run_dir).resolve()
    predictor_path = Path(args.predictor_path).resolve()
    tokenizer_path = Path(args.tokenizer_path).resolve()
    source_dataset = source_run / "skillvla"
    info_path = source_dataset / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Source SkillVLA info not found: {info_path}")
    source_info = json.loads(info_path.read_text())
    levels = [int(value) for value in source_info.get("skill_fsq_levels", [])]
    if not levels or any(value <= 1 for value in levels):
        raise ValueError(f"Invalid source skill_fsq_levels: {levels}")
    num_embeddings = math.prod(levels)
    source_code_space = str(
        source_info.get("skill_code_space_id", source_run.name) or source_run.name
    ).strip()
    if source_code_space != args.code_space_id:
        raise ValueError(
            "Requested code-space identity does not match the source dataset: "
            f"argument={args.code_space_id!r}, dataset={source_code_space!r}."
        )
    predictor_config_path = predictor_path / "config.json"
    predictor_weights_path = predictor_path / "model.safetensors"
    if not predictor_config_path.is_file() or not predictor_weights_path.is_file():
        raise FileNotFoundError(f"Incomplete predictor checkpoint: {predictor_path}")
    predictor_info = json.loads(predictor_config_path.read_text())
    predictor_levels = [
        int(value) for value in predictor_info.get("skill_fsq_levels", [])
    ]
    if predictor_levels != levels:
        raise ValueError(
            "Predictor FSQ geometry does not match the source dataset: "
            f"predictor={predictor_levels}, dataset={levels}."
        )
    predictor_code_space = str(
        predictor_info.get("skill_code_space_id", "") or ""
    ).strip()
    if not predictor_code_space:
        predictor_fsq_path = str(predictor_info.get("fsq_path", "") or "").strip()
        predictor_code_space = (
            Path(predictor_fsq_path).parent.name if predictor_fsq_path else ""
        )
    if predictor_code_space != source_code_space:
        raise ValueError(
            "Predictor and source dataset use different FSQ code spaces: "
            f"predictor={predictor_code_space!r}, dataset={source_code_space!r}."
        )
    expected_identity = {
        "schema_version": 1,
        "source_run": source_run.name,
        "predictor_model": args.predictor_model,
        "predictor_checkpoint": args.predictor_checkpoint,
        "skill_code_space_id": args.code_space_id,
    }
    if output_run.exists():
        if _provenance_matches(output_run, expected_identity):
            print(f"Relabeled dataset already complete -> {output_run}")
            return
        raise FileExistsError(
            f"Refusing to overwrite an existing non-matching output: {output_run}"
        )

    segments = discover_segments(source_dataset, num_embeddings=num_embeddings)
    print(f"Canonical skill starts: {len(segments)}")
    predictions = _predict(
        dataset_dir=source_dataset,
        predictor_path=predictor_path,
        tokenizer_path=tokenizer_path,
        segments=segments,
        batch_size=int(args.batch_size),
    )
    if np.any((predictions < 0) | (predictions >= num_embeddings)):
        raise ValueError(
            f"Predictor emitted a code outside [0, {num_embeddings}): "
            f"min={predictions.min()}, max={predictions.max()}."
        )

    temporary_run = output_run.with_name(
        f".{output_run.name}.tmp.{os.environ.get('SLURM_JOB_ID', 'local')}.{os.getpid()}"
    )
    if temporary_run.exists():
        shutil.rmtree(temporary_run)
    temporary_run.mkdir(parents=True)
    try:
        destination_dataset = temporary_run / "skillvla"
        shutil.copytree(
            source_dataset,
            destination_dataset,
            copy_function=_link_or_copy,
        )
        rewrite_skill_sequences(
            destination_dataset,
            segments=segments,
            predictions=predictions,
        )

        for artifact in ("FSQ.pt", "skill_initial_state.npz", "fsq_source.json"):
            source = source_run / artifact
            if source.is_file():
                _link_or_copy(str(source), str(temporary_run / artifact))
        source_latents = source_run / "skill_latents.npz"
        if source_latents.is_file():
            rewrite_skill_latents(
                source_latents,
                temporary_run / "skill_latents.npz",
                segments=segments,
                predictions=predictions,
                levels=levels,
            )

        original_codes = np.asarray(
            [segment.original_code for segment in segments], dtype=np.int32
        )
        predicted_codes = np.asarray(predictions, dtype=np.int32)
        np.savez_compressed(
            temporary_run / "skill_relabel.npz",
            episode_id=np.asarray(
                [segment.episode_index for segment in segments], dtype=np.int32
            ),
            task_id=np.asarray(
                [segment.task_index for segment in segments], dtype=np.int32
            ),
            skill_index=np.asarray(
                [segment.skill_index for segment in segments], dtype=np.int32
            ),
            frame_start=np.asarray(
                [segment.frame_start for segment in segments], dtype=np.int32
            ),
            original_code=original_codes,
            predicted_code=predicted_codes,
        )
        changed = int(np.count_nonzero(original_codes != predicted_codes))
        provenance = {
            **expected_identity,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source_run_path": str(source_run),
            "predictor_path": str(predictor_path),
            "boundary_policy": "preserve_gt_boundaries",
            "segment_count": len(segments),
            "unchanged_segments": len(segments) - changed,
            "changed_segments": changed,
            "changed_fraction": changed / max(len(segments), 1),
            "original_code_count": int(np.unique(original_codes).size),
            "predicted_code_count": int(np.unique(predicted_codes).size),
            "stats_recomputed": False,
            "transitions_artifact": (
                "omitted_to_avoid_stale_skill_codes; stage3 rebuilds it lazily "
                "from the relabeled parquet"
            ),
        }
        _atomic_json(temporary_run / "relabel_provenance.json", provenance)

        output_info_path = destination_dataset / "meta" / "info.json"
        output_info = json.loads(output_info_path.read_text())
        output_info["skill_code_space_id"] = args.code_space_id
        output_info["skill_dataset_variant"] = "predictor_relabeled"
        output_info["skill_label_source"] = "frozen_predictor_at_gt_boundaries"
        output_info["skill_relabeling"] = provenance
        output_info["skill_initial_state_path"] = str(
            output_run / "skill_initial_state.npz"
        )
        _atomic_json(output_info_path, output_info)

        temporary_run.replace(output_run)
    except BaseException:
        if temporary_run.exists():
            shutil.rmtree(temporary_run)
        raise

    print(
        f"DONE -> {output_run}  "
        f"(segments={len(segments)}, changed={changed}, "
        f"agreement={(len(segments) - changed) / max(len(segments), 1):.2%})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--output-run-dir", required=True)
    parser.add_argument("--predictor-path", required=True)
    parser.add_argument("--predictor-model", required=True)
    parser.add_argument("--predictor-checkpoint", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--code-space-id", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


if __name__ == "__main__":
    build_relabeled_dataset(parse_args())
