#!/usr/bin/env python3
"""Recover exact per-episode LIBERO init states for a LangGap LeRobot dataset.

LangGap is distributed without the original HDF5 ``demo.attrs['init_state']``
or a scene/demo identifier.  The corresponding LIBERO suites still contain the
finite set of full MuJoCo init states used by each task.  This builder resets
every candidate state, records its settled robot state and agent-view image,
and matches each dataset episode's first observation to those candidates.

The output follows the same contract as ``build_init_states.py`` so Stage-1
evaluation can use ``oracle.episode_exact=true`` without a LangGap-specific
runtime path.  Ambiguous matches fail by default; an approximate mapping is
never silently labelled episode-exact.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# LangGap task_index 40..55 is the collected subset of the 59-task
# ``langgap_ext`` suite, in LangGap dataset order (not sorted suite order).
LANGGAP_EXT_TASK_IDS = (0, 1, 3, 5, 6, 8, 17, 19, 21, 29, 30, 32, 39, 45, 42, 41)

OFFICIAL_RANGES = (
    (range(0, 10), "libero_10"),
    (range(10, 20), "libero_goal"),
    (range(20, 30), "libero_object"),
    (range(30, 40), "libero_spatial"),
)

# Dataset state is EEF xyz + axis-angle + two gripper positions.  Normalizing
# prevents the ~pi orientation component from overwhelming millimetre-scale
# position/gripper differences when candidates are ranked.
STATE_SCALE = np.asarray((0.05, 0.05, 0.05, 0.20, 0.20, 0.20, 0.02, 0.02), dtype=np.float32)


@dataclass(frozen=True)
class TaskSpec:
    dataset_task_id: int
    language: str
    suite_name: str
    suite_task_id: int
    task_name: str


@dataclass(frozen=True)
class Match:
    init_index: int
    score: float
    state_score: float
    image_score: float
    wrist_score: float
    second_score: float
    margin: float
    confident: bool
    reason: str


def norm_language(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _official_suite(dataset_task_id: int) -> str | None:
    for indices, suite_name in OFFICIAL_RANGES:
        if dataset_task_id in indices:
            return suite_name
    return None


def resolve_task_specs(task_languages: dict[int, str], suites: dict[str, object]) -> dict[int, TaskSpec]:
    """Resolve every LangGap dataset task to one unambiguous benchmark task."""
    specs: dict[int, TaskSpec] = {}
    for dataset_task_id, language in sorted(task_languages.items()):
        suite_name = _official_suite(dataset_task_id)
        if suite_name is not None:
            candidate_ids = range(len(suites[suite_name].tasks))
        elif 40 <= dataset_task_id < 56:
            suite_name = "langgap_ext"
            candidate_ids = (LANGGAP_EXT_TASK_IDS[dataset_task_id - 40],)
        else:
            raise ValueError(
                f"LangGap exact mapping supports dataset task_index 0..55, got {dataset_task_id}."
            )

        matches = [
            task_id
            for task_id in candidate_ids
            if norm_language(suites[suite_name].tasks[task_id].language)
            == norm_language(language)
        ]
        if len(matches) != 1:
            raise ValueError(
                "LangGap task must map to exactly one suite task: "
                f"dataset_task={dataset_task_id}, language={language!r}, "
                f"suite={suite_name}, matches={matches}."
            )
        suite_task_id = matches[0]
        task = suites[suite_name].tasks[suite_task_id]
        specs[dataset_task_id] = TaskSpec(
            dataset_task_id=dataset_task_id,
            language=language,
            suite_name=suite_name,
            suite_task_id=suite_task_id,
            task_name=str(task.name),
        )
    return specs


def image_signature(image: np.ndarray | torch.Tensor, size: int) -> np.ndarray:
    """Return a deterministic uint8 HWC thumbnail for matching/caching."""
    tensor = torch.as_tensor(np.asarray(image) if not torch.is_tensor(image) else image)
    if tensor.ndim != 3:
        raise ValueError(f"Expected a 3-D image, got shape={tuple(tensor.shape)}.")
    if tensor.shape[0] in (1, 3, 4):
        tensor = tensor[:3].unsqueeze(0)
    elif tensor.shape[-1] in (1, 3, 4):
        tensor = tensor[..., :3].permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError(f"Cannot identify image channels in shape={tuple(tensor.shape)}.")
    tensor = tensor.float()
    if float(tensor.max()) > 1.5:
        tensor = tensor / 255.0
    tensor = F.interpolate(tensor, size=(size, size), mode="area")
    return (
        tensor.squeeze(0)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )


def rank_candidates(
    episode_state: np.ndarray,
    episode_image: np.ndarray | None,
    episode_wrist: np.ndarray | None,
    candidate_states: np.ndarray,
    candidate_images: np.ndarray | None,
    candidate_wrists: np.ndarray | None,
    *,
    state_weight: float,
    image_weight: float,
    wrist_weight: float,
    max_state_score: float,
    max_image_mae: float,
    min_score_margin: float,
) -> Match:
    """Rank task-local init candidates and report conservative confidence."""
    episode_state = np.asarray(episode_state, dtype=np.float32).reshape(-1)
    candidate_states = np.asarray(candidate_states, dtype=np.float32)
    if episode_state.shape != (8,) or candidate_states.ndim != 2 or candidate_states.shape[1] != 8:
        raise ValueError(
            "LangGap init matching requires 8-D episode/candidate states, got "
            f"episode={episode_state.shape}, candidates={candidate_states.shape}."
        )
    state_scores = np.abs((candidate_states - episode_state[None]) / STATE_SCALE[None]).mean(axis=1)

    def pixel_scores(observation, candidates) -> np.ndarray:
        if observation is None or candidates is None:
            return np.zeros(candidate_states.shape[0], dtype=np.float32)
        observation = np.asarray(observation, dtype=np.float32)
        candidates = np.asarray(candidates, dtype=np.float32)
        return np.abs(candidates - observation[None]).mean(axis=(1, 2, 3)) / 255.0

    image_scores = pixel_scores(episode_image, candidate_images)
    wrist_scores = pixel_scores(episode_wrist, candidate_wrists)
    scores = (
        state_weight * state_scores
        + image_weight * image_scores
        + wrist_weight * wrist_scores
    )
    order = np.argsort(scores, kind="stable")
    best = int(order[0])
    second_score = float(scores[order[1]]) if len(order) > 1 else float("inf")
    margin = second_score - float(scores[best])
    state_strong = float(state_scores[best]) <= max_state_score
    image_strong = episode_image is not None and float(image_scores[best]) <= max_image_mae
    confident = bool((state_strong or image_strong) and margin >= min_score_margin)
    reasons = []
    if not (state_strong or image_strong):
        reasons.append("neither state nor image passed its absolute threshold")
    if margin < min_score_margin:
        reasons.append("best/second candidate margin is too small")
    return Match(
        init_index=best,
        score=float(scores[best]),
        state_score=float(state_scores[best]),
        image_score=float(image_scores[best]),
        wrist_score=float(wrist_scores[best]),
        second_score=second_score,
        margin=float(margin),
        confident=confident,
        reason="; ".join(reasons),
    )


def _task_languages(dataset_dir: Path) -> dict[int, str]:
    tasks = pd.read_parquet(dataset_dir / "meta" / "tasks.parquet")
    return {int(row.task_index): str(language) for language, row in tasks.iterrows()}


def _episode_first_rows(dataset_dir: Path) -> pd.DataFrame:
    files = sorted((dataset_dir / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data parquet under {dataset_dir / 'data'}.")
    columns = ["episode_index", "frame_index", "task_index", "observation.state"]
    frame = pd.concat([pd.read_parquet(path, columns=columns) for path in files], ignore_index=True)
    return (
        frame.sort_values(["episode_index", "frame_index"])
        .drop_duplicates("episode_index", keep="first")
        .sort_values("episode_index")
        .reset_index(drop=True)
    )


def _episode_metadata(dataset_dir: Path) -> pd.DataFrame:
    files = sorted((dataset_dir / "meta" / "episodes").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata under {dataset_dir / 'meta/episodes'}.")
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    return frame.set_index("episode_index", drop=False)


def _decode_first_image(dataset_dir: Path, row: pd.Series, key: str, fps: float) -> np.ndarray:
    from lerobot.datasets.video_utils import decode_video_frames

    chunk = int(row[f"videos/{key}/chunk_index"])
    file_index = int(row[f"videos/{key}/file_index"])
    timestamp = float(row[f"videos/{key}/from_timestamp"])
    path = dataset_dir / "videos" / key / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
    return decode_video_frames(
        path,
        [timestamp],
        tolerance_s=max(1e-4, 0.1 / fps),
        backend="pyav",
        decoder_num_threads=1,
    )[0]


def _load_episode_signatures(
    dataset_dir: Path,
    first_rows: pd.DataFrame,
    *,
    signature_size: int,
    with_wrist: bool,
    cache_file: Path | None,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None, int]]:
    expected_episodes = first_rows["episode_index"].to_numpy(dtype=np.int32)
    if cache_file is not None and cache_file.is_file():
        cache = np.load(cache_file, allow_pickle=False)
        if (
            int(cache["signature_size"]) == signature_size
            and bool(cache["with_wrist"]) == with_wrist
            and np.array_equal(cache["episode_index"], expected_episodes)
        ):
            wrists = cache["wrist"] if with_wrist else None
            return {
                int(episode): (
                    cache["state"][index],
                    cache["image"][index],
                    None if wrists is None else wrists[index],
                    int(cache["task_index"][index]),
                )
                for index, episode in enumerate(cache["episode_index"])
            }

    metadata = _episode_metadata(dataset_dir)
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    states, images, wrists, task_indices = [], [], [], []
    for count, (_, row) in enumerate(first_rows.iterrows(), start=1):
        episode = int(row["episode_index"])
        meta = metadata.loc[episode]
        image = image_signature(
            _decode_first_image(dataset_dir, meta, "observation.images.image", fps),
            signature_size,
        )
        wrist = None
        if with_wrist:
            wrist = image_signature(
                _decode_first_image(
                    dataset_dir, meta, "observation.images.wrist_image", fps
                ),
                signature_size,
            )
        states.append(np.asarray(row["observation.state"], dtype=np.float32))
        images.append(image)
        if with_wrist:
            wrists.append(wrist)
        task_indices.append(int(row["task_index"]))
        if count % 100 == 0 or count == len(first_rows):
            print(f"  dataset signatures: {count}/{len(first_rows)}")

    arrays = {
        "episode_index": expected_episodes,
        "task_index": np.asarray(task_indices, dtype=np.int16),
        "state": np.stack(states),
        "image": np.stack(images),
        "signature_size": np.asarray(signature_size, dtype=np.int16),
        "with_wrist": np.asarray(with_wrist, dtype=np.bool_),
    }
    if with_wrist:
        arrays["wrist"] = np.stack(wrists)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, **arrays)
    wrist_array = arrays.get("wrist")
    return {
        int(episode): (
            arrays["state"][index],
            arrays["image"][index],
            None if wrist_array is None else wrist_array[index],
            int(arrays["task_index"][index]),
        )
        for index, episode in enumerate(expected_episodes)
    }


def _candidate_cache_path(cache_dir: Path, spec: TaskSpec, size: int, wait: int) -> Path:
    return cache_dir / f"{spec.suite_name}_t{spec.suite_task_id:02d}_s{size}_wait{wait}.npz"


def _render_candidates(
    spec: TaskSpec,
    suite: object,
    *,
    signature_size: int,
    num_steps_wait: int,
    with_wrist: bool,
    cache_dir: Path,
) -> dict[str, np.ndarray]:
    cache_path = _candidate_cache_path(
        cache_dir, spec, signature_size, num_steps_wait
    )
    if cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        if str(cached["task_name"]) == spec.task_name:
            return {key: cached[key] for key in cached.files}

    import robosuite.utils.transform_utils as transform
    from lerobot.envs.libero import LiberoEnv

    env = LiberoEnv(
        suite,
        spec.suite_task_id,
        spec.suite_name,
        obs_type="pixels_agent_pos",
        observation_width=256,
        observation_height=256,
        num_steps_wait=num_steps_wait,
    )
    init_states = np.asarray(env._init_states)
    settled_states, images, wrists = [], [], []
    try:
        for init_index in range(len(init_states)):
            env.init_state_id = init_index
            observation, _ = env.reset(seed=0, _advance=False)
            robot = observation["robot_state"]
            settled_states.append(
                np.concatenate(
                    (
                        np.asarray(robot["eef"]["pos"]),
                        transform.quat2axisangle(np.asarray(robot["eef"]["quat"])),
                        np.asarray(robot["gripper"]["qpos"]),
                    )
                ).astype(np.float32)
            )
            images.append(
                image_signature(observation["pixels"]["image"], signature_size)
            )
            if with_wrist:
                wrists.append(
                    image_signature(
                        observation["pixels"]["image2"], signature_size
                    )
                )
    finally:
        env.close()

    arrays = {
        "task_name": np.asarray(spec.task_name),
        "init_states": init_states,
        "settled_states": np.stack(settled_states),
        "images": np.stack(images),
    }
    if with_wrist:
        arrays["wrists"] = np.stack(wrists)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    return arrays


def _parse_task_indices(raw: str, available: set[int]) -> list[int]:
    if not raw.strip():
        return sorted(available)
    result = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    unknown = set(result) - available
    if unknown:
        raise ValueError(f"Unknown dataset task indices: {sorted(unknown)}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot_dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--task-indices", default="")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--signature-size", type=int, default=64)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--state-weight", type=float, default=1.0)
    parser.add_argument("--image-weight", type=float, default=4.0)
    parser.add_argument("--wrist-weight", type=float, default=0.0)
    parser.add_argument("--max-state-score", type=float, default=1.0)
    parser.add_argument("--max-image-mae", type=float, default=0.18)
    parser.add_argument("--min-score-margin", type=float, default=0.01)
    parser.add_argument("--accept-ambiguous", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.out}; pass --overwrite.")
    if args.signature_size <= 0 or args.num_steps_wait < 0:
        raise ValueError("signature-size must be positive and num-steps-wait non-negative.")

    from libero.libero import benchmark

    task_languages = _task_languages(args.lerobot_dataset)
    suite_names = {name for _, name in OFFICIAL_RANGES} | {"langgap_ext"}
    suites = {
        name: benchmark.get_benchmark_dict()[name]() for name in sorted(suite_names)
    }
    specs = resolve_task_specs(task_languages, suites)
    selected_tasks = _parse_task_indices(args.task_indices, set(specs))

    first_rows = _episode_first_rows(args.lerobot_dataset)
    first_rows = first_rows[first_rows["task_index"].isin(selected_tasks)]
    if args.max_episodes is not None:
        first_rows = first_rows.head(args.max_episodes)
    if first_rows.empty:
        raise ValueError("No episodes selected.")

    cache_dir = args.cache_dir or (
        args.out.parent / ".langgap_episode_exact_cache"
    )
    full_cache = (
        cache_dir / f"episode_signatures_s{args.signature_size}_w{int(args.wrist_weight > 0)}.npz"
        if not args.task_indices and args.max_episodes is None
        else None
    )
    print("LangGap episode-exact init matching")
    print(f"  dataset : {args.lerobot_dataset}")
    print(f"  output  : {args.out}")
    print(f"  tasks   : {selected_tasks}")
    print(f"  episodes: {len(first_rows)}")
    print(f"  cache   : {cache_dir}")
    signatures = _load_episode_signatures(
        args.lerobot_dataset,
        first_rows,
        signature_size=args.signature_size,
        with_wrist=args.wrist_weight > 0,
        cache_file=full_cache,
    )

    matched: list[dict] = []
    failures: list[dict] = []
    for task_number, task_id in enumerate(selected_tasks, start=1):
        spec = specs[task_id]
        rows = first_rows[first_rows["task_index"] == task_id]
        if rows.empty:
            continue
        print(
            f"  task {task_number}/{len(selected_tasks)}: dataset={task_id:02d} "
            f"-> {spec.suite_name}[{spec.suite_task_id}] ({len(rows)} episodes)"
        )
        candidates = _render_candidates(
            spec,
            suites[spec.suite_name],
            signature_size=args.signature_size,
            num_steps_wait=args.num_steps_wait,
            with_wrist=args.wrist_weight > 0,
            cache_dir=cache_dir,
        )
        candidate_wrists = candidates.get("wrists")
        for _, row in rows.iterrows():
            episode = int(row["episode_index"])
            state, image, wrist, signature_task = signatures[episode]
            if signature_task != task_id:
                raise RuntimeError(
                    f"Episode {episode} task changed while building signatures: "
                    f"{signature_task} != {task_id}."
                )
            match = rank_candidates(
                state,
                image,
                wrist,
                candidates["settled_states"],
                candidates["images"],
                candidate_wrists,
                state_weight=args.state_weight,
                image_weight=args.image_weight,
                wrist_weight=args.wrist_weight,
                max_state_score=args.max_state_score,
                max_image_mae=args.max_image_mae,
                min_score_margin=args.min_score_margin,
            )
            record = {
                "episode_index": episode,
                "dataset_task_id": task_id,
                "suite_name": spec.suite_name,
                "suite_task_id": spec.suite_task_id,
                "task_name": spec.task_name,
                "init_index": match.init_index,
                "score": match.score,
                "state_score": match.state_score,
                "image_score": match.image_score,
                "wrist_score": match.wrist_score,
                "second_score": match.second_score,
                "margin": match.margin,
                "confident": match.confident,
                "reason": match.reason,
            }
            if match.confident or args.accept_ambiguous:
                record["init_state"] = candidates["init_states"][match.init_index]
                matched.append(record)
            else:
                failures.append(record)

    diagnostics_path = args.out.with_suffix(".diagnostics.json")
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(
        json.dumps(
            {
                "dataset": str(args.lerobot_dataset),
                "matched": [{k: v for k, v in row.items() if k != "init_state"} for row in matched],
                "failed": failures,
            },
            indent=2,
        )
    )
    if failures and not args.accept_ambiguous:
        raise RuntimeError(
            f"Refusing to write episode-exact map: {len(failures)} ambiguous/unmatched "
            f"episodes. Inspect {diagnostics_path} and tune thresholds only after visual validation."
        )

    matched.sort(key=lambda row: row["episode_index"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    init_state_array = np.empty(len(matched), dtype=object)
    init_state_array[:] = [row["init_state"] for row in matched]
    np.savez(
        args.out,
        episode_index=np.asarray([row["episode_index"] for row in matched], dtype=np.int32),
        init_states=init_state_array,
        scene_file=np.asarray([f"{row['task_name']}_demo.hdf5" for row in matched]),
        demo=np.asarray([f"init_{row['init_index']:03d}" for row in matched]),
        match_err=np.asarray([row["score"] for row in matched], dtype=np.float32),
        match_method=np.asarray(["langgap_state_image" for _ in matched]),
        suite_name=np.asarray([row["suite_name"] for row in matched]),
        suite_task_id=np.asarray([row["suite_task_id"] for row in matched], dtype=np.int16),
        init_state_index=np.asarray([row["init_index"] for row in matched], dtype=np.int16),
        state_score=np.asarray([row["state_score"] for row in matched], dtype=np.float32),
        image_score=np.asarray([row["image_score"] for row in matched], dtype=np.float32),
        score_margin=np.asarray([row["margin"] for row in matched], dtype=np.float32),
    )
    print(f"Wrote {args.out}: {len(matched)} exact episode mappings")
    print(f"Diagnostics: {diagnostics_path}")


if __name__ == "__main__":
    main()
