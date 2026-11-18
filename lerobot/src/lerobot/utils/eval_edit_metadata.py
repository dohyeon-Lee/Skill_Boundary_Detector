"""Frame-accurate editing metadata for unannotated simulator recordings."""

from __future__ import annotations

import math


def write_simulator_video(path, frames, fps):
    """Encode native render pixels without overlays, resizing, or frame skipping."""
    import imageio

    imageio.mimsave(path, frames, fps=fps, macro_block_size=1, ffmpeg_params=["-threads", "1"])


def episode_edit_metadata(
    trace: list[dict], *, batch_index: int, episode_index: int,
    episode_steps: int, sim_fps: float, success: bool, seed: int | None,
    task_description: str | None, video_frame_stride: int, video_fps: float,
    raw_video: dict | None = None, annotated_video: str | None = None,
) -> dict:
    """Use executed skill starts, never predicted durations or rounded video frames.

    Frame 0 is the reset observation. Frame k precedes action k (zero based).
    The final post-action render is omitted because vector envs can auto-reset.
    Skill intervals and raw frame ranges are half-open [start, end).
    """
    if episode_steps < 1 or sim_fps <= 0 or video_fps <= 0 or video_frame_stride < 1:
        raise ValueError("Invalid episode length or video timing")
    records = sorted(
        (r for r in trace if int(r.get("batch_index", 0)) == batch_index
         and 0 <= int(r["episode_timestep"]) < episode_steps),
        key=lambda r: int(r["episode_timestep"]),
    )
    skills = []
    for i, record in enumerate(records):
        start = int(record["episode_timestep"])
        end = int(records[i + 1]["episode_timestep"]) if i + 1 < len(records) else episode_steps
        if end <= start:
            continue  # A proposal replaced before any action was executed.
        first_video_frame = math.ceil(start / video_frame_stride)
        end_video_frame = math.ceil(end / video_frame_stride)
        skills.append({
            "occurrence": len(skills), "skill_index": int(record["skill_index"]),
            "codebook_token": int(record["codebook_token"]),
            "start_step": start, "end_step_exclusive": end,
            "start_time_s": start / sim_fps, "end_time_s": end / sim_fps,
            "raw_start_frame": start, "raw_end_frame_exclusive": end,
            "annotated_start_frame": first_video_frame,
            "annotated_end_frame_exclusive": end_video_frame,
            "annotated_start_time_s": first_video_frame / video_fps,
            "annotated_end_time_s": end_video_frame / video_fps,
            "termination_samples": record.get("end_probs", []),
            "same_skill_transition_rejections": record.get("same_skill_transition_rejections", []),
        })
    transitions = [
        {"step": right["start_step"], "sim_time_s": right["start_time_s"],
         "raw_frame": right["raw_start_frame"],
         "annotated_frame": right["annotated_start_frame"],
         "annotated_time_s": right["annotated_start_time_s"],
         "from_skill_index": left["skill_index"], "to_skill_index": right["skill_index"],
         "from_token": left["codebook_token"], "to_token": right["codebook_token"]}
        for left, right in zip(skills, skills[1:])
    ]
    return {
        "schema_version": 1, "episode_index": episode_index, "seed": seed,
        "success": bool(success), "task_description": task_description,
        "episode_steps": episode_steps, "sim_fps": sim_fps,
        "sim_duration_s": episode_steps / sim_fps,
        "frame_convention": "frame k is the observation before action k; ranges [start,end); final post-action/autoreset frame omitted",
        "video_paths_relative_to": "task videos directory (parent of episode_metadata/)",
        "raw_video": raw_video,
        "annotated_video": None if annotated_video is None else {
            "path": annotated_video, "frame_stride": video_frame_stride, "fps": video_fps,
            "frame_count": math.ceil(episode_steps / video_frame_stride),
        },
        "skills": skills, "transitions": transitions,
    }
