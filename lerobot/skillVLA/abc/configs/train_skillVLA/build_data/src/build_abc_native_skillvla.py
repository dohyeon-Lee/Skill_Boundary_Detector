#!/usr/bin/env python3
"""Build storage-light SkillVLA annotations for an ABC native dataset.

The official state/action binaries and stacked camera videos stay in place.  A
SkillVLA dataset contains copied metadata, one compact annotation archive, an
ISS archive for transition jitter, and a native marker pointing back to the
same official episode files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets.abc_native_io import (
    abc_native_raw_root,
    load_abc_native_episodes,
    load_abc_native_marker,
    load_abc_native_state_action,
)


ANNOTATION_FORMAT = "sbd_abc_native_skillvla/v1"


def _shared_builder_module():
    path = next(
        parent
        / "libero/configs/train_skillVLA/build_data/src/add_skill_latents_to_dataset.py"
        for parent in Path(__file__).resolve().parents
        if (
            parent
            / "libero/configs/train_skillVLA/build_data/src/add_skill_latents_to_dataset.py"
        ).is_file()
    )
    spec = importlib.util.spec_from_file_location("sbd_skillvla_dataset_builder", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import shared SkillVLA builder: {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through sys.modules while the
    # module body is executing.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _skill_map(latents_path: Path) -> dict[int, list[tuple[int, int, int]]]:
    with np.load(latents_path, allow_pickle=False) as archive:
        required = {"episode_id", "frame_start", "frame_end", "tokens"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"{latents_path} is missing {missing}.")
        arrays = {key: np.asarray(archive[key]) for key in required}
    lengths = {key: len(value) for key, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Inconsistent skill_latents arrays: {lengths}")
    result: dict[int, list[tuple[int, int, int]]] = {}
    for episode, start, end, token in zip(
        arrays["episode_id"],
        arrays["frame_start"],
        arrays["frame_end"],
        arrays["tokens"],
        strict=True,
    ):
        result.setdefault(int(episode), []).append(
            (int(start), int(end), int(token))
        )
    if not result:
        raise ValueError(f"{latents_path} contains no encoded skills.")
    for episode, skills in result.items():
        skills.sort(key=lambda item: item[0])
        if not skills:
            raise ValueError(f"Episode {episode} has no encoded skill.")
        if any(start < 0 or end <= start for start, end, _ in skills):
            raise ValueError(f"Episode {episode} has an invalid skill interval.")
        if any(a[1] > b[0] for a, b in zip(skills, skills[1:], strict=False)):
            raise ValueError(f"Episode {episode} has overlapping skill intervals.")
    return result


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_native_view(source: Path, output: Path, *, raw_root: Path) -> dict:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    shutil.copytree(source / "meta", output / "meta")
    marker = load_abc_native_marker(source)
    marker["raw_root"] = os.path.relpath(raw_root, output)
    return marker


def build(args: argparse.Namespace) -> None:
    source = args.source_dataset_dir.resolve()
    output = args.output_dataset_dir.resolve()
    if output == source:
        raise ValueError("The ABC SkillVLA output must differ from the source dataset.")
    latents_path = args.skill_latents_path.resolve()
    iss_path = args.iss_path.resolve()
    annotation_path = output / "skill_annotations.npz"
    source_marker = load_abc_native_marker(source)
    raw_root = abc_native_raw_root(source, source_marker)
    episodes = load_abc_native_episodes(source).set_index("episode_index", drop=False)
    skills_by_episode = _skill_map(latents_path)
    selected_ids = sorted(skills_by_episode)
    missing = sorted(set(selected_ids) - set(int(value) for value in episodes.index))
    if missing:
        raise ValueError(f"Encoded skills reference unknown ABC episodes: {missing[:20]}")

    selected_rows = episodes.loc[selected_ids].sort_index()
    source_tasks = sorted(set(selected_rows["source_task"].astype(str)))
    requested_tasks = sorted(str(value) for value in args.source_tasks)
    if requested_tasks and source_tasks != requested_tasks:
        raise ValueError(
            "Encoded episode source tasks disagree with the build config: "
            f"latents={source_tasks}, config={requested_tasks}."
        )

    num_embeddings = math.prod(args.fsq_levels)
    eos_token = num_embeddings
    pad_token = num_embeddings + 1
    observed_max_order = max(len(skills) for skills in skills_by_episode.values())
    observed_max_length = max(
        end - start
        for skills in skills_by_episode.values()
        for start, end, _ in skills
    )
    max_order = args.max_order or observed_max_order
    max_length = args.max_length or observed_max_length
    if max_order < observed_max_order:
        raise ValueError(
            f"max_order={max_order} is below observed {observed_max_order}."
        )
    if max_length < observed_max_length:
        raise ValueError(
            f"max_length={max_length} is below observed {observed_max_length}."
        )

    shared = _shared_builder_module()
    episode_ids: list[int] = []
    episode_lengths: list[int] = []
    offsets = [0]
    frame_columns: dict[str, list[np.ndarray]] = {
        "skill_index": [],
        "skill_ds": [],
        "skill_de": [],
        "skill_boundary": [],
    }
    episode_columns: dict[str, list[np.ndarray | int]] = {
        "skill_sequence": [],
        "skill_length_sequence": [],
        "skill_initial_frame": [],
        "skill_sequence_mask": [],
        "skill_sequence_len": [],
    }
    states_by_episode: dict[int, np.ndarray] = {}
    actions_by_episode: dict[int, np.ndarray] = {}
    state_dim = int(source_marker["state_dim"])
    action_dim = int(source_marker["action_dim"])

    for position, (episode_id, row) in enumerate(selected_rows.iterrows(), start=1):
        episode_id = int(episode_id)
        states, actions = load_abc_native_state_action(
            source, row, marker=source_marker, raw_root=raw_root
        )
        length = len(states)
        if states.shape != (length, state_dim) or actions.shape != (length, action_dim):
            raise ValueError(
                f"Episode {episode_id} state/action dimensions are invalid: "
                f"{states.shape}, {actions.shape}."
            )
        ep_df = pd.DataFrame(
            {
                "frame_index": np.arange(length, dtype=np.int64),
                "observation.state": list(states),
                "action": list(actions),
            }
        )
        columns = shared.compute_skill_columns(
            ep_df,
            skills_by_episode[episode_id],
            max_order=max_order,
            max_length=max_length,
            eos_token_id=eos_token,
            pad_token_id=pad_token,
        )
        for key in frame_columns:
            frame_columns[key].append(np.asarray(columns[key]))
        for key in episode_columns:
            values = columns[key]
            # Episode-constant arrays are repeated by the LeRobot builder. Store
            # one row here and expand only at native-dataset read time.
            episode_columns[key].append(np.asarray(values[0]).copy())
        episode_ids.append(episode_id)
        episode_lengths.append(length)
        offsets.append(offsets[-1] + length)
        states_by_episode[episode_id] = states
        actions_by_episode[episode_id] = actions
        if position % 100 == 0 or position == len(selected_rows):
            print(
                f"ABC SkillVLA annotations: {position}/{len(selected_rows)} episodes, "
                f"{offsets[-1]} frames",
                flush=True,
            )

    marker = _copy_native_view(source, output, raw_root=raw_root)
    _atomic_npz(
        annotation_path,
        format=np.asarray(ANNOTATION_FORMAT),
        episode_id=np.asarray(episode_ids, dtype=np.int32),
        episode_length=np.asarray(episode_lengths, dtype=np.int32),
        offsets=np.asarray(offsets, dtype=np.int64),
        **{
            key: np.concatenate(values).astype(
                np.int8 if key == "skill_boundary" else np.int32, copy=False
            )
            for key, values in frame_columns.items()
        },
        **{
            key: np.stack(values).astype(
                np.int8 if key == "skill_sequence_mask" else np.int32,
                copy=False,
            )
            for key, values in episode_columns.items()
        },
        skill_max_order=np.asarray(max_order, dtype=np.int32),
        skill_max_length=np.asarray(max_length, dtype=np.int32),
    )

    shared.build_skill_initial_state_npz(
        skills_by_episode,
        states_by_episode,
        actions_by_episode,
        int(args.transition_jitter_pmax),
        state_dim,
        iss_path,
        str(args.transition_jitter_distribution),
        0.0,
        bool(args.precompute_gripper_validity),
        "none",
    )

    info_path = output / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info.update(
        {
            "repo_id": args.repo_id,
            "skill_annotation_format": ANNOTATION_FORMAT,
            "skill_annotation_path": str(annotation_path),
            "skill_num_embeddings": num_embeddings,
            "skill_fsq_levels": list(args.fsq_levels),
            "skill_eos_token_id": eos_token,
            "skill_pad_token_id": pad_token,
            "skill_vocab_size": num_embeddings + 2,
            "skill_output_vocab_size": num_embeddings + 1,
            "skill_max_order": max_order,
            "skill_observed_max_order": observed_max_order,
            "skill_max_length": max_length,
            "skill_observed_max_length": observed_max_length,
            "skill_sequence_size": max_order + 1,
            "skill_pmax": int(args.transition_jitter_pmax),
            "skill_jitter_max_percent": 0.0,
            "skill_jitter_gripper_validity": bool(args.precompute_gripper_validity),
            "skill_jitter_distribution": str(args.transition_jitter_distribution),
            "skill_initial_state_path": str(iss_path),
            "skill_initial_state_window": 2 * int(args.transition_jitter_pmax) + 1,
            "proprio_grounding": "none",
            "proprio_grounding_reference": "none",
            "training_total_episodes": len(episode_ids),
            "training_total_frames": int(offsets[-1]),
            "training_episode_ids": episode_ids,
            "training_source_tasks": source_tasks,
        }
    )
    sequence_size = max_order + 1
    info["features"].update(
        {
            "skill_index": {"dtype": "int32", "shape": [1]},
            "skill_sequence": {"dtype": "int32", "shape": [sequence_size]},
            "skill_length_sequence": {"dtype": "int32", "shape": [sequence_size]},
            "skill_initial_frame": {"dtype": "int32", "shape": [sequence_size]},
            "skill_sequence_mask": {"dtype": "int8", "shape": [sequence_size]},
            "skill_sequence_len": {"dtype": "int32", "shape": [1]},
            "skill_ds": {"dtype": "int32", "shape": [1]},
            "skill_de": {"dtype": "int32", "shape": [1]},
            "skill_boundary": {"dtype": "int8", "shape": [1]},
            "skill_max_order": {"dtype": "int32", "shape": [1]},
            "skill_max_length": {"dtype": "int32", "shape": [1]},
            "skill_decoder_state": {"dtype": "float32", "shape": [state_dim]},
        }
    )
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    marker.update(
        {
            "skill_annotation_format": ANNOTATION_FORMAT,
            "skill_annotations": "skill_annotations.npz",
            "default_episode_ids": episode_ids,
            "default_source_tasks": source_tasks,
            "source_dataset": str(source),
        }
    )
    (output / "native_dataset.json").write_text(
        json.dumps(marker, indent=2) + "\n", encoding="utf-8"
    )

    if args.stats_path:
        stats_source = args.stats_path.resolve()
        payload = json.loads(stats_source.read_text(encoding="utf-8"))
        selected_stats = payload.get("stats", payload)
        stats_path = output / "meta" / "stats.json"
        stats = json.loads(stats_path.read_text()) if stats_path.is_file() else {}
        for key in ("observation.state", "action"):
            if key in selected_stats:
                stats[key] = selected_stats[key]
        if "observation.state" in selected_stats:
            stats["skill_decoder_state"] = selected_stats["observation.state"]
        stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "format": ANNOTATION_FORMAT,
        "source_dataset": str(source),
        "source_tasks": source_tasks,
        "episode_count": len(episode_ids),
        "frame_count": int(offsets[-1]),
        "skill_count": sum(len(value) for value in skills_by_episode.values()),
        "fsq_levels": list(args.fsq_levels),
        "skill_latents_path": str(latents_path),
        "skill_latents_sha256": _sha256(latents_path),
        "annotation_path": str(annotation_path),
        "annotation_sha256": _sha256(annotation_path),
        "iss_path": str(iss_path),
        "iss_sha256": _sha256(iss_path),
        "transition_jitter_pmax": int(args.transition_jitter_pmax),
        "transition_jitter_distribution": str(args.transition_jitter_distribution),
        "transition_jitter_max_percent": 0.0,
        "transition_jitter_precompute_gripper_validity": bool(
            args.precompute_gripper_validity
        ),
        "proprio_grounding": "none",
        "action_space": "absolute_joint",
        "state_dim": state_dim,
        "action_dim": action_dim,
    }
    (output.parent / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"DONE: {output} ({len(episode_ids)} episodes, {offsets[-1]} frames, "
        f"{manifest['skill_count']} skills)",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dataset-dir", type=Path, required=True)
    parser.add_argument("--skill-latents-path", type=Path, required=True)
    parser.add_argument("--iss-path", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, default=None)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--source-tasks", nargs="+", required=True)
    parser.add_argument("--fsq-levels", nargs="+", type=int, required=True)
    parser.add_argument("--max-order", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--transition-jitter-pmax", type=int, default=30)
    parser.add_argument(
        "--transition-jitter-distribution",
        choices=("half_normal", "uniform"),
        default="half_normal",
    )
    parser.add_argument("--precompute-gripper-validity", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
