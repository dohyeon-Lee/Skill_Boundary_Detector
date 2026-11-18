"""Shared canonical skill-start and checkpoint-preprocessing helpers for evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
            # evaluate.
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


def _preprocessor_overrides(predictor_path: Path, tokenizer_path: Path, device: str) -> dict:
    """Relocate every saved tokenizer, including Joint's training-only tokenizers."""
    config = json.loads((predictor_path / "policy_preprocessor.json").read_text())
    overrides = {}
    for step in config.get("steps", []):
        key = step.get("registry_name") or step.get("class", "").rsplit(".", 1)[-1]
        if key in {"device_processor", "DeviceProcessorStep"}:
            overrides[key] = {"device": device}
        elif key in {
            "tokenizer_processor",
            "TokenizerProcessorStep",
            "skill_vla_terminator_predictor_tokenizer_processor_step",
            "SkillVLATerminatorPredictorTokenizerProcessorStep",
        }:
            overrides[key] = {"tokenizer_name": str(tokenizer_path)}
    return overrides


def _prepare_predictor_preprocessor(preprocessor):
    """Keep the checkpoint-owned normalization/prompt, not Terminator training prompts."""
    from lerobot.policies.skillVLA.processor_skillVLA import (
        SkillVLATerminatorPredictorTokenizerProcessorStep,
    )

    preprocessor.steps = [
        step
        for step in preprocessor.steps
        if not isinstance(step, SkillVLATerminatorPredictorTokenizerProcessorStep)
    ]
    return preprocessor

