#!/usr/bin/env python3
"""Remap a skillset built from a compact ABC subset onto a larger ABC dataset.

The compact dataset and target dataset must contain byte-equivalent episodes in
the same source order.  Before writing anything, this script verifies every
episode length and requires a one-to-one task-id mapping.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _scalar(value: int, template: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=template.dtype)


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_episode_table(dataset_dir: Path, source_task: str) -> pd.DataFrame:
    parts = sorted((dataset_dir / "meta" / "episodes").rglob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No episode parquet files below {dataset_dir / 'meta/episodes'}")
    columns = ["episode_index", "length", "task_indices", "source_task"]
    table = pd.concat((pd.read_parquet(path, columns=columns) for path in parts), ignore_index=True)
    table = table.loc[table["source_task"] == source_task].sort_values("episode_index").reset_index(drop=True)
    if table.empty:
        raise ValueError(f"Target dataset has no source_task={source_task!r}")
    if table["episode_index"].duplicated().any():
        raise ValueError("Target episode_index values are not unique")
    return table


def _load_curve_index(source: Path) -> dict[int, tuple[int, int, Path]]:
    curves: dict[int, tuple[int, int, Path]] = {}
    for path in sorted((source / "curves").glob("ep*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            compact_episode = int(archive["episode_id"])
            compact_task = int(archive["task_id"])
            n_frames = int(archive["n_frames"])
        if compact_episode in curves:
            raise ValueError(f"Duplicate compact episode {compact_episode}")
        curves[compact_episode] = (compact_task, n_frames, path)
    expected = set(range(len(curves)))
    if set(curves) != expected:
        missing = sorted(expected - set(curves))[:10]
        raise ValueError(f"Compact episode ids must be contiguous from zero; missing={missing}")
    return curves


def _derive_mapping(
    source: Path, target_dataset_dir: Path, source_task: str
) -> tuple[dict[int, int], dict[int, int], pd.DataFrame]:
    curves = _load_curve_index(source)
    target = _load_episode_table(target_dataset_dir, source_task)
    if len(curves) != len(target):
        raise ValueError(f"Episode count mismatch: skillset={len(curves)}, target={len(target)}")

    episode_map: dict[int, int] = {}
    task_candidates: dict[int, set[int]] = {}
    for compact_episode, row in target.iterrows():
        compact_task, n_frames, _ = curves[compact_episode]
        if n_frames != int(row["length"]):
            raise ValueError(
                f"Episode ordering mismatch at compact episode {compact_episode}: "
                f"skillset n_frames={n_frames}, target length={int(row['length'])}"
            )
        task_indices = list(row["task_indices"])
        if len(task_indices) != 1:
            raise ValueError(
                f"Expected one task id for target episode {int(row['episode_index'])}, got {task_indices}"
            )
        global_task = int(task_indices[0])
        episode_map[compact_episode] = int(row["episode_index"])
        task_candidates.setdefault(compact_task, set()).add(global_task)

    ambiguous = {key: sorted(value) for key, value in task_candidates.items() if len(value) != 1}
    if ambiguous:
        raise ValueError(f"Compact task ids do not map one-to-one: {ambiguous}")
    task_map = {key: next(iter(value)) for key, value in task_candidates.items()}
    if len(set(task_map.values())) != len(task_map):
        raise ValueError(f"Multiple compact task ids map to the same target id: {task_map}")
    return episode_map, task_map, target


def _rewrite_npz_ids(source: Path, target: Path, episode: int, task: int) -> None:
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["episode_id"] = _scalar(episode, arrays["episode_id"])
    arrays["task_id"] = _scalar(task, arrays["task_id"])
    _write_npz(target, arrays)


def _rewrite_probe(source: Path, target: Path, target_dataset_name: str) -> None:
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if "metadata" in arrays:
        metadata = json.loads(str(arrays["metadata"].item()))
        metadata["dataset"] = target_dataset_name
        text = json.dumps(metadata, sort_keys=True)
        arrays["metadata"] = np.asarray(text, dtype=f"<U{max(1, len(text))}")
    _write_npz(target, arrays)


def _localize_repo_path(value: str, repo_root: Path) -> str:
    path = Path(value)
    for anchor in ("outputs_ABC", "dataset_ABC"):
        if anchor in path.parts:
            index = path.parts.index(anchor)
            return str(repo_root.joinpath(*path.parts[index:]).resolve())
    return value


def _rewrite_metadata(
    source: Path,
    staging: Path,
    output: Path,
    target_dataset_dir: Path,
    target_dataset_name: str,
    source_task: str,
    task_map: dict[int, int],
    repo_root: Path,
) -> None:
    shutil.copy2(source / "global_boundary_threshold.json", staging / "global_boundary_threshold.json")

    manifest = json.loads((source / "skillset_manifest.json").read_text())
    original_dataset = manifest.get("dataset_name")
    manifest["dataset_dir"] = str(target_dataset_dir.resolve())
    manifest["dataset_name"] = target_dataset_name
    manifest["policy_path"] = _localize_repo_path(str(manifest["policy_path"]), repo_root)
    manifest["detector"]["global_threshold_path"] = str(
        (output / "global_boundary_threshold.json").resolve()
    )
    manifest["remap"] = {
        "source_dataset": original_dataset,
        "source_task": source_task,
        "task_id_map": {str(key): value for key, value in sorted(task_map.items())},
        "remapped_at": datetime.now(timezone.utc).isoformat(),
    }
    (staging / "skillset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    complete: dict[str, str] = {}
    order: list[str] = []
    for line in (source / ".complete").read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        complete[key] = value
        order.append(key)
    complete["dataset"] = target_dataset_name
    complete["source_tasks"] = source_task
    complete["prompt_task_indices"] = ",".join(str(value) for value in sorted(task_map.values()))
    complete["global_threshold_path"] = str((output / "global_boundary_threshold.json").resolve())
    complete["remapped_from_dataset"] = str(original_dataset)
    complete["remapped_at"] = datetime.now(timezone.utc).isoformat()
    for key in ("remapped_from_dataset", "remapped_at"):
        if key not in order:
            order.append(key)
    (staging / ".complete").write_text("".join(f"{key}={complete[key]}\n" for key in order))


def _validate_output(
    output: Path,
    episode_map: dict[int, int],
    task_map: dict[int, int],
    expected_skills: int,
) -> None:
    curves = list((output / "curves").glob("*.npz"))
    skills = list((output / "skills").glob("task*/*skill*.npz"))
    done = list((output / "skills").glob("task*/*.done"))
    if len(curves) != len(episode_map) or len(done) != len(episode_map) or len(skills) != expected_skills:
        raise RuntimeError(
            f"Output count mismatch: curves={len(curves)}, done={len(done)}, skills={len(skills)}; "
            f"expected {len(episode_map)}, {len(episode_map)}, {expected_skills}"
        )
    valid_episodes = set(episode_map.values())
    valid_tasks = set(task_map.values())
    for path in curves + skills:
        with np.load(path, allow_pickle=False) as archive:
            episode = int(archive["episode_id"])
            task = int(archive["task_id"])
        if episode not in valid_episodes or task not in valid_tasks:
            raise RuntimeError(f"Invalid remapped ids in {path}: episode={episode}, task={task}")


def remap(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    output = args.output.resolve()
    target_dataset_dir = args.target_dataset_dir.resolve()
    repo_root = args.repo_root.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    episode_map, task_map, _ = _derive_mapping(source, target_dataset_dir, args.source_task)
    source_skills = list((source / "skills").glob("task*/*skill*.npz"))
    staging = output.with_name(f".{output.name}.remap-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"Staging path already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        _rewrite_probe(source / "action_probe_pca.npz", staging / "action_probe_pca.npz", args.target_dataset_name)
        for compact_episode, global_episode in episode_map.items():
            curve_source = source / "curves" / f"ep{compact_episode:07d}.npz"
            with np.load(curve_source, allow_pickle=False) as archive:
                compact_task = int(archive["task_id"])
            global_task = task_map[compact_task]
            _rewrite_npz_ids(
                curve_source,
                staging / "curves" / f"ep{global_episode:07d}.npz",
                global_episode,
                global_task,
            )

        for skill_source in source_skills:
            with np.load(skill_source, allow_pickle=False) as archive:
                compact_episode = int(archive["episode_id"])
                compact_task = int(archive["task_id"])
                skill_index = int(archive["skill_index"])
            global_episode = episode_map[compact_episode]
            global_task = task_map[compact_task]
            skill_target = (
                staging
                / "skills"
                / f"task{global_task:02d}"
                / f"ep{global_episode:05d}_task{global_task:02d}_skill{skill_index:02d}.npz"
            )
            _rewrite_npz_ids(skill_source, skill_target, global_episode, global_task)

        for compact_episode, global_episode in episode_map.items():
            curve_source = source / "curves" / f"ep{compact_episode:07d}.npz"
            with np.load(curve_source, allow_pickle=False) as archive:
                global_task = task_map[int(archive["task_id"])]
            marker = staging / "skills" / f"task{global_task:02d}" / f"ep{global_episode:05d}_task{global_task:02d}.done"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()

        _rewrite_metadata(
            source,
            staging,
            output,
            target_dataset_dir,
            args.target_dataset_name,
            args.source_task,
            task_map,
            repo_root,
        )
        os.replace(staging, output)
        _validate_output(output, episode_map, task_map, len(source_skills))
    except Exception:
        if output.exists() and not staging.exists():
            os.replace(output, staging)
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"Remapped {len(episode_map)} episodes and {len(source_skills)} skills")
    print(f"Task map: {task_map}")
    print(f"Output: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Compact skillset directory")
    parser.add_argument("--target-dataset-dir", type=Path, required=True)
    parser.add_argument("--target-dataset-name", required=True)
    parser.add_argument("--source-task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    remap(parse_args())
