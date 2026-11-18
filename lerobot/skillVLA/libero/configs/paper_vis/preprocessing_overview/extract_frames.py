#!/usr/bin/env python3
"""Extract paper-ready LIBERO frames aligned to SBD segments and FSQ codes."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[6]
OUTPUT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset_filtered" / "libero_90_full_full"
LATENTS_PATH = (
    PROJECT_ROOT
    / "outputs_filtered"
    / "FSQ"
    / "FSQ333_recon_term22__contrastiveON_routeON_loss_new_rand15pct"
    / "skill_latents_epoch2000.npz"
)
CAMERA_KEY = "observation.images.image"
FRAME_FRACTIONS = (("start", 0.15), ("middle", 0.50), ("end", 0.85))


# Each entry is one actual FSQ code observed at the same task phase in three
# different demonstration episodes.
MULTI_EPISODE_SKILLS = (
    {
        "folder": "skill_01_book_place_phase2_code_p1_p1_p1",
        "task_id": 35,
        "skill_index": 1,
        "code": (1, 1, 1),
        "episodes": (1750, 1753, 2073),
    },
    {
        "folder": "skill_02_bowl_to_cabinet_phase1_code_m1_p1_m1",
        "task_id": 4,
        "skill_index": 0,
        "code": (-1, 1, -1),
        "episodes": (2582, 2571, 183),
    },
    {
        "folder": "skill_03_bowl_to_cabinet_phase2_code_m1_m1_p1",
        "task_id": 4,
        "skill_index": 1,
        "code": (-1, -1, 1),
        "episodes": (2561, 214, 2599),
    },
    {
        "folder": "skill_04_open_drawer_phase1_code_m1_0_m1",
        "task_id": 61,
        "skill_index": 0,
        "code": (-1, 0, -1),
        "episodes": (3185, 3178, 3796),
    },
    {
        "folder": "skill_05_turn_on_stove_phase2_code_p1_m1_0",
        "task_id": 43,
        "skill_index": 1,
        "code": (1, -1, 0),
        "episodes": (2768, 2773, 2197),
    },
)

# This episode contains five detected segments for one continuous task.
SINGLE_EPISODE_ID = 318
SINGLE_EPISODE_FOLDER = "single_episode_ep00318_wine_bottle_to_rack"


def _load_segments() -> pd.DataFrame:
    payload = np.load(LATENTS_PATH)
    frame = pd.DataFrame(
        {
            key: payload[key]
            for key in (
                "episode_id",
                "task_id",
                "skill_index",
                "frame_start",
                "frame_end",
                "length",
                "tokens",
            )
        }
    )
    frame["code"] = [tuple(int(value) for value in row) for row in payload["latents"]]
    return frame


def _load_episode_metadata() -> pd.DataFrame:
    paths = sorted((DATASET_ROOT / "meta" / "episodes").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError("No LIBERO episode metadata was found.")
    episodes = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    return episodes.set_index("episode_index", drop=False)


def _task_text(value: object) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        return str(value[0])
    return str(value)


def _episode_video_and_time(
    episodes: pd.DataFrame, episode_id: int, frame_index: int
) -> tuple[Path, float, str]:
    episode = episodes.loc[int(episode_id)]
    prefix = f"videos/{CAMERA_KEY}"
    chunk = int(episode[f"{prefix}/chunk_index"])
    file_index = int(episode[f"{prefix}/file_index"])
    start_time = float(episode[f"{prefix}/from_timestamp"])
    length = int(episode["length"])
    duration = float(episode[f"{prefix}/to_timestamp"]) - start_time
    fps = length / duration
    timestamp = start_time + float(frame_index) / fps
    video = (
        DATASET_ROOT
        / "videos"
        / CAMERA_KEY
        / f"chunk-{chunk:03d}"
        / f"file-{file_index:03d}.mp4"
    )
    return video, timestamp, _task_text(episode["tasks"])


def _extract_frame(video: Path, timestamp: float, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        (
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{timestamp:.6f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(output),
        ),
        check=True,
    )


def _extract_segment(
    *,
    segment: pd.Series,
    episodes: pd.DataFrame,
    output_dir: Path,
    collection: str,
) -> list[dict[str, object]]:
    episode_id = int(segment.episode_id)
    skill_index = int(segment.skill_index)
    frame_start = int(segment.frame_start)
    frame_end = int(segment.frame_end)
    span = frame_end - frame_start
    code = tuple(int(value) for value in segment.code)
    records: list[dict[str, object]] = []

    for position, fraction in FRAME_FRACTIONS:
        frame_index = frame_start + int(round(fraction * max(span - 1, 0)))
        video, timestamp, task_text = _episode_video_and_time(
            episodes, episode_id, frame_index
        )
        filename = (
            f"ep{episode_id:05d}_skill{skill_index:02d}_{position}_"
            f"frame{frame_index:04d}.png"
        )
        output = output_dir / filename
        _extract_frame(video, timestamp, output)
        records.append(
            {
                "collection": collection,
                "relative_path": str(output.relative_to(OUTPUT_ROOT)),
                "episode_id": episode_id,
                "task_id": int(segment.task_id),
                "task": task_text,
                "skill_index": skill_index,
                "fsq_token": int(segment.tokens),
                "fsq_code": str(list(code)),
                "segment_frame_start": frame_start,
                "segment_frame_end": frame_end,
                "sample_position": position,
                "sample_frame_index": frame_index,
                "video_timestamp_seconds": timestamp,
                "source_video": str(video.relative_to(PROJECT_ROOT)),
            }
        )
    return records


def main() -> None:
    segments = _load_segments()
    episodes = _load_episode_metadata()
    records: list[dict[str, object]] = []

    multi_root = OUTPUT_ROOT / "multi_episode"
    for specification in MULTI_EPISODE_SKILLS:
        skill_dir = multi_root / str(specification["folder"])
        for episode_id in specification["episodes"]:
            matches = segments[
                (segments["episode_id"] == int(episode_id))
                & (segments["task_id"] == int(specification["task_id"]))
                & (segments["skill_index"] == int(specification["skill_index"]))
                & (segments["code"] == specification["code"])
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one segment for {specification['folder']} episode "
                    f"{episode_id}, found {len(matches)}."
                )
            records.extend(
                _extract_segment(
                    segment=matches.iloc[0],
                    episodes=episodes,
                    output_dir=skill_dir,
                    collection="multi_episode",
                )
            )

    episode_segments = segments[segments["episode_id"] == SINGLE_EPISODE_ID].sort_values(
        "skill_index"
    )
    if len(episode_segments) < 3:
        raise RuntimeError(
            f"Episode {SINGLE_EPISODE_ID} has only {len(episode_segments)} segments."
        )
    single_root = OUTPUT_ROOT / SINGLE_EPISODE_FOLDER
    for _, segment in episode_segments.iterrows():
        code_tag = "_".join("m1" if value == -1 else f"p{value}" if value > 0 else "0" for value in segment.code)
        skill_dir = single_root / f"skill_{int(segment.skill_index):02d}_code_{code_tag}"
        records.extend(
            _extract_segment(
                segment=segment,
                episodes=episodes,
                output_dir=skill_dir,
                collection="single_episode",
            )
        )

    pd.DataFrame.from_records(records).to_csv(OUTPUT_ROOT / "manifest.csv", index=False)
    print(f"Wrote {len(records)} frames and manifest to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
